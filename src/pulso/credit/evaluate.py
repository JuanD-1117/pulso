"""Comparación de los tres niveles de variables con validación cruzada por mundo.

Cada partición de ``GroupKFold`` deja mundos completos fuera: el modelo nunca ve en
entrenamiento clientes del mundo con el que se evalúa (sus vecinos comparten red).
Las diferencias entre niveles se miden **pareadas por partición**: mismas filas,
mismo modelo, solo cambian las variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from pulso.credit.features import LEVELS
from pulso.schema import assert_no_truth

MODELS = ("logistic", "xgboost")
METRICS = ("auc", "pr_auc", "ks", "brier", "ece")


def make_model(name: str, cfg: dict, seed: int) -> ClassifierMixin:
    if name == "logistic":
        c = cfg["logistic"]
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c["C"]), max_iter=int(c["max_iter"])),
        )
    if name == "xgboost":
        return XGBClassifier(
            **cfg["xgboost"],
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(f"Modelo desconocido {name!r}; use {MODELS}")


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov–Smirnov entre las puntuaciones de buenos y malos pagadores."""
    pos, neg = p[y == 1], p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(stats.ks_2samp(pos, neg).statistic)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE con bins de igual frecuencia: Σ_b (n_b/n) |media(y) - media(p)|."""
    order = np.argsort(p)
    err = 0.0
    for idx in np.array_split(order, bins):
        if idx.size:
            err += idx.size * abs(y[idx].mean() - p[idx].mean())
    return float(err / len(y))


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(y.mean()),
    }


def cross_validate(
    table: pd.DataFrame,
    model_cfg: dict,
    n_splits: int,
    seed: int,
    levels: dict[str, tuple[str, ...]] = LEVELS,
    models: tuple[str, ...] = MODELS,
) -> pd.DataFrame:
    """Métricas por (nivel, modelo, partición) y predicciones fuera de muestra."""
    if table["world_id"].nunique() < n_splits:
        raise ValueError(
            f"Se necesitan al menos {n_splits} mundos para {n_splits} particiones"
        )
    for cols in levels.values():
        assert_no_truth(cols)
    y = table["y"].to_numpy()
    groups = table["world_id"].to_numpy()
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits).split(table, y, groups)):
        if y[tr].sum() == 0 or y[te].sum() == 0:
            raise ValueError(f"Partición {fold} sin impagos: use más mundos")
        for level, cols in levels.items():
            X = table[list(cols)].to_numpy(dtype=float)
            for name in models:
                model = make_model(name, model_cfg, seed + fold)
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                rows.append(
                    {"level": level, "model": name, "fold": fold, **score(y[te], p)}
                )
    return pd.DataFrame(rows)


def summarize(cv: pd.DataFrame) -> pd.DataFrame:
    """Media y desviación por (nivel, modelo)."""
    agg = cv.groupby(["model", "level"])[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def paired_deltas(
    cv: pd.DataFrame,
    metric: str = "auc",
    base: str = "clasico",
    compare: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Mejora de cada nivel sobre ``base``, pareada por partición, con IC 95 % (t).

    ``compare`` limita los niveles comparados (por defecto, todos menos ``base``).
    """
    rows = []
    for (model, level), g in cv.groupby(["model", "level"]):
        if level == base or (compare is not None and level not in compare):
            continue
        ref = cv[(cv["model"] == model) & (cv["level"] == base)].set_index("fold")
        d = g.set_index("fold")[metric] - ref[metric]
        k = len(d)
        half = stats.t.ppf(0.975, k - 1) * d.std(ddof=1) / np.sqrt(k) if k > 1 else 0
        rows.append(
            {
                "model": model,
                "level": level,
                "base": base,
                "metric": metric,
                "delta_mean": float(d.mean()),
                "ci_low": float(d.mean() - half),
                "ci_high": float(d.mean() + half),
                "folds_positive": int((d > 0).sum()),
                "n_folds": k,
            }
        )
    return pd.DataFrame(rows)


COHERENCE_BINS = (-1.01, 0.25, 0.5, 0.75, 0.9, 1.01)


def default_by_coherence(table: pd.DataFrame) -> pd.DataFrame:
    """Tasa de impago por tramo de r̂, solo clientes con ingreso por la red.

    Quien no recibe nada por la red tiene r̂ = 1 por convención (no hay
    desincronización medible); incluirlo mezclaría informales sin pagadores con
    clientes realmente sincronizados.
    """
    t = table[table["n_payers"] > 0]
    bins = pd.cut(t["r_mean"], list(COHERENCE_BINS))
    out = (
        t.groupby(["regime", bins], observed=True)["y"]
        .agg(rate="mean", n="size")
        .reset_index()
    )
    out["tramo"] = out["r_mean"].astype(str)
    return out.drop(columns="r_mean")


def fit_final(
    table: pd.DataFrame, level: str, name: str, model_cfg: dict, seed: int
) -> ClassifierMixin:
    """Modelo final de un nivel, entrenado con todos los mundos."""
    cols = LEVELS[level]
    assert_no_truth(cols)
    model = make_model(name, model_cfg, seed)
    model.fit(table[list(cols)].to_numpy(dtype=float), table["y"].to_numpy())
    return model
