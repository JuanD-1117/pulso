"""Entrenamiento, validación por red y versionado de modelos (Fase 6).

Validación: ``GroupKFold`` con ``network_id`` como grupo, así ninguna red aparece en
entrenamiento y prueba a la vez. Las métricas se calculan sobre predicciones fuera
de pliegue; el modelo guardado se reentrena con todos los datos.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from pulso.experiments import NODE_FEATURES, NODE_LABEL, SCENARIO_FEATURES
from pulso.ml import gnn
from pulso.ml.registry import save_model

log = logging.getLogger(__name__)


# ----------------------------------------------------------------- datos


def load_dataset(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    data_dir = Path(data_dir)
    for f in ("escenarios.parquet", "nodos.parquet", "networks.npz"):
        if not (data_dir / f).is_file():
            raise FileNotFoundError(
                f"Falta {data_dir / f}. Genere el dataset con "
                "`uv run pulso sweep configs/sweep.yaml`."
            )
    esc = pd.read_parquet(data_dir / "escenarios.parquet")
    nodos = pd.read_parquet(data_dir / "nodos.parquet")
    with np.load(data_dir / "networks.npz") as z:
        nets = {int(k.split("_")[1]): z[k] for k in z.files}
    return esc, nodos, nets


def _folds(groups: np.ndarray, n_splits: int):
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)


# ----------------------------------------------------------------- métricas


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
    }


def ranking_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """Spearman medio por red entre ranking predicho y simulado, más R² y MAE."""
    d = df[["network_id", NODE_LABEL]].assign(pred=pred)
    rhos = []
    for _, g in d.groupby("network_id"):
        if g[NODE_LABEL].nunique() > 1 and g["pred"].nunique() > 1:
            rhos.append(spearmanr(g[NODE_LABEL], g["pred"]).statistic)
    return {
        "spearman_mean": float(np.mean(rhos)) if rhos else float("nan"),
        "spearman_networks": len(rhos),
        "r2": float(r2_score(d[NODE_LABEL], d["pred"])),
        "mae": float(mean_absolute_error(d[NODE_LABEL], d["pred"])),
    }


def _ms_per_prediction(fn, n: int) -> float:
    fn()  # calentar
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000 * (time.perf_counter() - t) / n


# ----------------------------------------------------------------- modelos


def make_logistic(cfg: dict):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=cfg["logistic"]["C"],
            max_iter=cfg["logistic"]["max_iter"],
            class_weight="balanced",
        ),
    )


def make_xgb_classifier(cfg: dict, y: np.ndarray) -> XGBClassifier:
    pos = max(y.sum(), 1)
    return XGBClassifier(
        **cfg["xgboost"],
        scale_pos_weight=(len(y) - pos) / pos,
        random_state=cfg["seed"],
        eval_metric="logloss",
        tree_method="hist",
    )


def make_xgb_regressor(cfg: dict) -> XGBRegressor:
    return XGBRegressor(
        **cfg["xgboost_nodes"], random_state=cfg["seed"], tree_method="hist"
    )


def train_collapse_models(esc: pd.DataFrame, cfg: dict) -> dict[str, tuple]:
    """Logística y XGBoost para colapso global: ``{nombre: (modelo, métricas)}``."""
    X = esc[list(SCENARIO_FEATURES)].to_numpy(dtype=float)
    y = esc["collapse"].to_numpy()
    groups = esc["network_id"].to_numpy()
    out = {}
    for name, factory in (
        ("logistic_collapse", lambda: make_logistic(cfg)),
        ("xgb_collapse", lambda: make_xgb_classifier(cfg, y)),
    ):
        oof = np.zeros(len(y))
        for tr, te in _folds(groups, cfg["n_splits"]):
            m = factory().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        model = factory().fit(X, y)
        mets = classification_metrics(y, oof)
        mets["ms_per_prediction"] = _ms_per_prediction(
            lambda m=model: m.predict_proba(X[:1]), cfg["timing"]["n_predictions"]
        )
        out[name] = (model, mets)
        log.info("%s: %s", name, mets)
    return out


def train_node_xgb(nodos: pd.DataFrame, cfg: dict) -> tuple[XGBRegressor, dict]:
    X = gnn.transform_features(nodos, NODE_FEATURES)
    y = nodos[NODE_LABEL].to_numpy()
    groups = nodos["network_id"].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in _folds(groups, cfg["n_splits"]):
        oof[te] = make_xgb_regressor(cfg).fit(X[tr], y[tr]).predict(X[te])
    model = make_xgb_regressor(cfg).fit(X, y)
    mets = ranking_metrics(nodos, np.clip(oof, 0, 1))
    one_net = X[groups == groups[0]]
    mets["ms_per_network"] = _ms_per_prediction(
        lambda: model.predict(one_net), cfg["timing"]["n_predictions"]
    )
    mets["network_n"] = float(len(one_net))
    log.info("xgb_nodes: %s", mets)
    return model, mets


def train_node_gcn(
    nodos: pd.DataFrame,
    nets: dict[int, np.ndarray],
    cfg: dict,
) -> tuple[dict, dict]:
    cols = NODE_FEATURES
    X_all = gnn.transform_features(nodos, cols)
    net_ids = np.array(sorted(nodos["network_id"].unique()))
    oof = pd.Series(np.nan, index=nodos.index)
    for k, (tr_idx, te_idx) in enumerate(_folds(net_ids, cfg["n_splits"])):
        tr_nets, te_nets = set(net_ids[tr_idx]), set(net_ids[te_idx])
        tr_mask = nodos["network_id"].isin(tr_nets).to_numpy()
        mean, std = X_all[tr_mask].mean(0), X_all[tr_mask].std(0) + 1e-8
        tr_g = gnn.build_graphs(nodos[tr_mask], nets, cols, mean, std, NODE_LABEL)
        model = gnn.train_gcn(tr_g, cfg["gcn"], len(cols), seed=cfg["seed"] + k)
        te = nodos[nodos["network_id"].isin(te_nets)]
        for g in gnn.build_graphs(te, nets, cols, mean, std, NODE_LABEL):
            idx = te.index[te["network_id"] == g.network_id]
            order = te.loc[idx].sort_values("entity").index
            oof.loc[order] = gnn.predict_graph(model, g)
    mean, std = X_all.mean(0), X_all.std(0) + 1e-8
    graphs = gnn.build_graphs(nodos, nets, cols, mean, std, NODE_LABEL)
    model = gnn.train_gcn(graphs, cfg["gcn"], len(cols), seed=cfg["seed"])
    mets = ranking_metrics(nodos, oof.to_numpy())
    mets["ms_per_network"] = _ms_per_prediction(
        lambda: gnn.predict_graph(model, graphs[0]), cfg["timing"]["n_predictions"]
    )
    mets["network_n"] = float(graphs[0].num_nodes)
    artifact = {
        "state_dict": model.state_dict(),
        "in_dim": len(cols),
        "columns": list(cols),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "config": cfg["gcn"],
    }
    log.info("gcn_nodes: %s", mets)
    return artifact, mets


def simulation_seconds(
    nodos: pd.DataFrame,
    nets: dict[int, np.ndarray],
    n_nets: int = 3,
) -> float:
    """Segundos por escenario simulado (choque a una entidad) en redes del dataset."""
    from pulso.config import load_config
    from pulso.experiments import run_node_rows, sample_network_pool

    sweep = load_config("sweep.yaml")
    pool = sample_network_pool(sweep, int(sweep["seed"]))
    ids = sorted(nets)[:n_nets]
    total_t, total_n = 0.0, 0
    for i in ids:
        t = time.perf_counter()
        run_node_rows(nets[i], pool[i], i, sweep, seed=0)
        total_t += time.perf_counter() - t
        total_n += nets[i].shape[0]
    return total_t / total_n


def train_all(cfg: dict, data_dir: Path, models_dir: Path) -> dict[str, dict]:
    """Entrena todos los modelos, guarda artefactos y la tabla comparativa."""
    esc, nodos, nets = load_dataset(data_dir)
    version = str(cfg["version"])
    report: dict[str, dict] = {}
    meta_common = {"seed": cfg["seed"], "n_splits": cfg["n_splits"]}

    for name, (model, mets) in train_collapse_models(esc, cfg).items():
        save_model(
            models_dir,
            name,
            version,
            model,
            mets,
            {**meta_common, "features": list(SCENARIO_FEATURES), "label": "collapse"},
        )
        report[name] = mets

    xgb_nodes, mets = train_node_xgb(nodos, cfg)
    save_model(
        models_dir,
        "xgb_nodes",
        version,
        xgb_nodes,
        mets,
        {**meta_common, "features": list(NODE_FEATURES), "label": NODE_LABEL},
    )
    report["xgb_nodes"] = mets

    gcn_art, mets = train_node_gcn(nodos, nets, cfg)
    save_model(
        models_dir,
        "gcn_nodes",
        version,
        gcn_art,
        mets,
        {**meta_common, "features": list(NODE_FEATURES), "label": NODE_LABEL},
    )
    report["gcn_nodes"] = mets

    sim_s = simulation_seconds(nodos, nets)
    rows = []
    for name, m in report.items():
        # Un modelo de nodos responde por toda la red: equivale a n simulaciones.
        ms = m.get("ms_per_prediction", m.get("ms_per_network"))
        n_sims = m.get("network_n", 1.0)
        rows.append(
            {
                "modelo": name,
                **{k: v for k, v in m.items() if isinstance(v, float)},
                "speedup_vs_simulacion": (1000 * sim_s * n_sims) / ms if ms else np.nan,
            }
        )
    comp = pd.DataFrame(rows)
    comp["segundos_por_simulacion"] = sim_s
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    comp.to_csv(Path(models_dir) / "comparison.csv", index=False)
    return report
