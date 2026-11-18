"""Fachada de la Tarea 6 para la app: funciones puras, sin Streamlit.

La app solo llama a este módulo (y a :mod:`pulso.service` para las tareas 1–5).
"""

from __future__ import annotations

import copy
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from pulso.config import data_dir, load_config, models_dir
from pulso.credit.engine import CreditParams
from pulso.credit.estimate import global_order
from pulso.credit.experiment import (
    MODEL_VERSION,
    load_final_model,
    load_results,
    predicted_pd,
)
from pulso.credit.features import LEVELS, global_synchrony
from pulso.credit.pipeline import process_world
from pulso.integrate import Shock
from pulso.model import Dynamics, Params
from pulso.world import World, generate_world, scaled_config

CREDIT_MODELS_DIR = models_dir() / "credit"
RESULTS_DIR = data_dir() / "credit"


def credit_config() -> dict:
    return load_config("credit.yaml")


def world_config() -> dict:
    return load_config("world.yaml")


def demo_world(n_persons: int, n_firms: int, informal_frac: float, seed: int) -> World:
    cfg = scaled_config(world_config(), n_persons, n_firms)
    cfg["persons"]["informal_frac"] = float(informal_frac)
    return generate_world(cfg, seed=seed)


def params_for(regime: str, overrides: dict | None = None) -> CreditParams:
    """Parámetros del régimen con cambios de la app (``K_N``, ``gamma``, ...)."""
    cfg = copy.deepcopy(credit_config())
    p = CreditParams.from_config(cfg, regime)
    if not overrides:
        return p
    o = dict(overrides)
    if "K_N" in o:
        p = p.with_(K={**p.K, "N": float(o.pop("K_N"))})
    if regime == "nulo":  # el control no tiene acople déficit-red: no se toca
        o = {k: v for k, v in o.items() if k not in ("gamma", "loss")}
    return p.with_(**o) if o else p


def layout_world(world: World, seed: int) -> np.ndarray:
    """Posiciones 2D: firmas al centro de sus empleados (spring layout)."""
    W = world.matrix().toarray()
    G = nx.from_numpy_array(0.5 * (W + W.T))
    pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=60)
    return np.array([pos[i] for i in range(world.n)])


def simulate_portfolio(
    world: World,
    regime: str,
    seed: int,
    overrides: dict | None = None,
    shock: Shock | None = None,
) -> dict:
    """Corre la cartera y devuelve lo que muestra la app."""
    cfg = credit_config()
    p = params_for(regime, overrides)
    ct = process_world(
        world,
        p,
        seed=seed,
        tx_cfg=cfg["transactions"],
        grid_step=float(cfg["experiment"]["phase_grid_step"]),
        shock=shock,
    )
    traj = ct.run.traj
    obs = traj.t < p.t_obs
    grid, r_hat = global_synchrony(ct.transactions, world.n, p.t_obs)
    alive = ~np.isfinite(traj.default_time)[None, :] | (
        traj.t[:, None] < traj.default_time[None, :]
    )
    theta_alive = np.where(alive, traj.theta, np.nan)
    dyn = Dynamics(world.matrix(), Params(alpha=1.0, beta=0.0))
    frac_def = np.mean(~alive, axis=1)
    return {
        "params": p,
        "run": ct.run,
        "table": ct.table,
        "transactions": ct.transactions,
        "R": ct.run.R,
        "t": traj.t,
        "r_true": global_order(theta_alive),
        "grid": grid,
        "r_hat": r_hat,
        "frac_defaulted": frac_def,
        "r_local_final_obs": dyn.local_coherence(traj.theta[obs][-1]),
    }


def results_available(out_dir: Path | str = RESULTS_DIR) -> bool:
    try:
        load_results(out_dir)
    except FileNotFoundError:
        return False
    return (CREDIT_MODELS_DIR / MODEL_VERSION / "metrics.json").is_file()


def experiment_results(out_dir: Path | str = RESULTS_DIR) -> dict[str, pd.DataFrame]:
    return load_results(out_dir)


def score_clients(table: pd.DataFrame, model: str = "xgboost") -> pd.DataFrame:
    """PD a 12 meses por cliente con los tres niveles (modelos finales)."""
    out = table[["entity_id", "y", "observed_default", "cause"]].copy()
    alive = table["observed_default"] == 0
    for level in LEVELS:
        m = load_final_model(level, model, CREDIT_MODELS_DIR)
        out[f"pd_{level}"] = np.where(alive, predicted_pd(m, table, level), np.nan)
    return out


def explain(table: pd.DataFrame, level: str = "red_dinamica") -> pd.DataFrame:
    """Contribuciones SHAP (log-odds) del XGBoost final, una fila por cliente."""
    import shap

    m = load_final_model(level, "xgboost", CREDIT_MODELS_DIR)
    X = table[list(LEVELS[level])].to_numpy(dtype=float)
    vals = shap.TreeExplainer(m).shap_values(X)
    return pd.DataFrame(vals, columns=list(LEVELS[level]), index=table.index)


def biggest_employers(world: World, k: int = 10) -> pd.DataFrame:
    e = world.entities
    f = e[e["kind"] == 0].nlargest(k, "n_employees")
    return f[["entity_id", "n_employees", "income", "debt_ratio", "sector"]]
