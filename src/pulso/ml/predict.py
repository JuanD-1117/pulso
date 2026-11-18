"""Predicción instantánea de riesgo, explicación SHAP y verificación por simulación.

El contexto (:class:`RiskContext`) reúne la red, los parámetros y el estado previo
al choque, y es lo único que necesitan los modelos: así la predicción usa
exactamente las mismas variables con que se entrenó (``experiments.node_features``
y ``experiments.scenario_features``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pulso import spectral
from pulso.experiments import (
    NODE_FEATURES,
    SCENARIO_FEATURES,
    ScenarioSpec,
    network_features,
    node_features,
    pre_shock_phases,
)
from pulso.integrate import Shock, simulate
from pulso.metrics import local_coherence, order_parameter
from pulso.ml import gnn
from pulso.ml.registry import load_model
from pulso.model import Params
from pulso.networks import NetworkConfig


@dataclass
class RiskContext:
    W: np.ndarray
    params: Params
    net_cfg: NetworkConfig
    theta_pre: np.ndarray
    buffer: float
    loss: float
    shock_size: float
    sim: dict

    @property
    def n(self) -> int:
        return self.W.shape[0]


def make_context(
    W: np.ndarray,
    params: Params,
    net_cfg: NetworkConfig,
    buffer: float,
    loss: float,
    shock_size: float,
    sim: dict,
    seed: int,
) -> RiskContext:
    """``sim`` necesita ``burn_in``, ``dt``, ``t0``, ``t_end`` y ``tau_cap``."""
    if params.thresholds is None:
        raise ValueError("El contexto de riesgo necesita umbrales de impago")
    theta_pre = pre_shock_phases(W, params, sim["burn_in"], sim["dt"], seed)
    return RiskContext(W, params, net_cfg, theta_pre, buffer, loss, shock_size, sim)


def node_table(ctx: RiskContext) -> pd.DataFrame:
    spec = ScenarioSpec(
        scenario_id=-1,
        network_id=0,
        seed=0,
        alpha=ctx.params.alpha,
        R=0.0,
        D0=ctx.params.D0,
        K=ctx.params.K,
        gamma=ctx.params.gamma,
        omega_width=0.0,
        buffer=ctx.buffer,
        loss=ctx.loss,
        operational=False,
        shock_size=ctx.shock_size,
        duration=0.0,
        target_u=0.0,
    )
    return node_features(ctx.W, ctx.params, ctx.theta_pre, spec, ctx.net_cfg)


def scenario_table(ctx: RiskContext, omega_width: float) -> pd.DataFrame:
    """Variables de escenario para un retiro de liquidez en cada entidad."""
    W = ctx.W
    base = network_features(W, ctx.params, ctx.net_cfg, ctx.sim["tau_cap"])
    rank = spectral.entity_ranking(W).set_index("entidad").sort_index()
    out = W.sum(axis=0)
    A = W > 0
    df = pd.DataFrame(
        {
            **{k: np.full(ctx.n, v) for k, v in base.items()},
            "r_init": float(order_parameter(ctx.theta_pre)),
            "r_local_mean": float(local_coherence(ctx.theta_pre, W).mean()),
            "alpha": ctx.params.alpha,
            "beta": ctx.params.beta,
            "R": spectral.amplification_index(W, ctx.params),
            "D0": ctx.params.D0,
            "K": ctx.params.K,
            "gamma": ctx.params.gamma,
            "omega_width": omega_width,
            "buffer": ctx.buffer,
            "loss": ctx.loss,
            "shock_operational": 0,
            "shock_size": ctx.shock_size,
            "shock_duration": 0.0,
            "target_importance": rank["importancia"].to_numpy() * ctx.n,
            "target_out_share": out / out.sum() if out.sum() > 0 else np.zeros(ctx.n),
            "target_in_degree": A.sum(axis=1),
            "target_out_degree": A.sum(axis=0),
        }
    )
    return df[list(SCENARIO_FEATURES)]


@lru_cache(maxsize=4)
def _models(models_dir: str):
    xgb_c, _, m_c = load_model(Path(models_dir), "xgb_collapse")
    xgb_n, _, m_n = load_model(Path(models_dir), "xgb_nodes")
    art, _, m_g = load_model(Path(models_dir), "gcn_nodes")
    cfg = art["config"]
    gcn = gnn.GCN(art["in_dim"], cfg["hidden"], cfg["layers"], cfg["dropout"])
    gcn.load_state_dict(art["state_dict"])
    gcn.eval()
    return {
        "xgb_collapse": xgb_c,
        "xgb_nodes": xgb_n,
        "gcn": gcn,
        "gcn_mean": np.array(art["mean"]),
        "gcn_std": np.array(art["std"]),
        "metrics": {"xgb_collapse": m_c, "xgb_nodes": m_n, "gcn_nodes": m_g},
    }


def models_available(models_dir: Path) -> bool:
    return all(
        (Path(models_dir) / n / "LATEST").is_file()
        for n in ("xgb_collapse", "xgb_nodes", "gcn_nodes")
    )


def predict_risk(
    ctx: RiskContext, models_dir: Path, omega_width: float
) -> pd.DataFrame:
    """Riesgo por entidad: cascada predicha (GCN y XGBoost) y P(colapso)."""
    M = _models(str(models_dir))
    nodes = node_table(ctx)
    X_nodes = gnn.transform_features(nodes, NODE_FEATURES)
    data = gnn.graph_from_W(ctx.W, (X_nodes - M["gcn_mean"]) / M["gcn_std"])
    cascade_gcn = gnn.predict_graph(M["gcn"], data)
    cascade_xgb = np.clip(M["xgb_nodes"].predict(X_nodes), 0, 1)
    X_sc = scenario_table(ctx, omega_width).to_numpy(dtype=float)
    p_collapse = M["xgb_collapse"].predict_proba(X_sc)[:, 1]
    return pd.DataFrame(
        {
            "entidad": np.arange(ctx.n),
            "cascada_gcn": cascade_gcn,
            "cascada_xgb": cascade_xgb,
            "p_colapso": p_collapse,
            "paga": ctx.W.sum(axis=0),
        }
    )


def explain(ctx: RiskContext, entity: int, models_dir: Path) -> pd.DataFrame:
    """Contribuciones SHAP de cada variable a la cascada predicha (XGBoost de nodos)."""
    import shap

    M = _models(str(models_dir))
    nodes = node_table(ctx)
    X = gnn.transform_features(nodes, NODE_FEATURES)
    explainer = shap.TreeExplainer(M["xgb_nodes"])
    sv = explainer.shap_values(X[entity : entity + 1])[0]
    df = pd.DataFrame(
        {
            "variable": NODE_FEATURES,
            "valor": nodes.loc[entity, list(NODE_FEATURES)].to_numpy(dtype=float),
            "shap": sv,
        }
    )
    df.attrs["base_value"] = float(np.ravel(explainer.expected_value)[0])
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def verify(ctx: RiskContext, entity: int) -> float:
    """Cascada simulada al chocar ``entity`` (misma definición que la etiqueta)."""
    shock = Shock("liquidity", (int(entity),), size=ctx.shock_size, t0=ctx.sim["t0"])
    traj = simulate(
        ctx.W,
        ctx.params,
        t_end=ctx.sim["t_end"],
        dt=ctx.sim["dt"],
        theta0=ctx.theta_pre,
        shock=shock,
        save_every=10**9,
    )
    others = traj.defaulted.copy()
    others[entity] = False
    return float(others.sum() / max(ctx.n - 1, 1))
