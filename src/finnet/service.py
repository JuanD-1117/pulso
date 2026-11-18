"""API interna que usa la app (capa de servicio).

Recibe parámetros planos (diccionarios de ``configs/``) y devuelve resultados listos
para mostrar: diccionarios, arreglos y DataFrames. Es la única puerta entre la
interfaz y el núcleo; una API HTTP se montaría sobre estas funciones sin tocar la
física.
"""

from __future__ import annotations

import copy
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from finnet import cascades, metrics, spectral
from finnet.config import load_config
from finnet.integrate import Shock
from finnet.integrate import simulate as _simulate
from finnet.model import Params, lorentz_frequencies, make_thresholds
from finnet.networks import NetworkConfig, core_mask, generate, summary

# ---------------------------------------------------------------- configuración


def default_config() -> dict[str, Any]:
    return load_config("default.yaml")


def merged(base: dict, overrides: dict | None) -> dict:
    """Copia profunda de ``base`` con ``overrides`` aplicados por sección."""
    out = copy.deepcopy(base)
    for key, val in (overrides or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **val}
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------- red


def build_network(net_cfg: dict, seed: int) -> dict[str, Any]:
    """Genera la red y sus descriptores."""
    cfg = NetworkConfig.from_dict(net_cfg, seed=seed)
    W = generate(cfg)
    return {
        "W": W,
        "network_config": cfg.to_dict(),
        "summary": summary(W),
        "core": core_mask(W, cfg),
    }


def layout(W: np.ndarray, seed: int) -> np.ndarray:
    """Posiciones 2D (n x 2) para dibujar la red."""
    G = nx.from_numpy_array(0.5 * (W + W.T))
    pos = nx.spring_layout(G, seed=seed, weight=None, iterations=80)
    return np.array([pos[i] for i in range(W.shape[0])])


def make_params(
    W: np.ndarray,
    model_cfg: dict,
    seed: int,
    with_thresholds: bool = True,
) -> Params:
    """Construye :class:`Params` desde la sección ``model`` de la configuración.

    Los ritmos ω son cuantiles de Lorentz(omega_mean, omega_width) asignados al azar;
    los umbrales, colchón ``buffer`` x pagos salientes con dispersión lognormal.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    omega = rng.permutation(
        lorentz_frequencies(n, model_cfg["omega_mean"], model_cfg["omega_width"])
    )
    thresholds = (
        make_thresholds(W, model_cfg["buffer"], model_cfg["buffer_dispersion"], rng)
        if with_thresholds
        else None
    )
    # Si la configuración fija R en lugar de β, β se deduce de ρ(W): así el mismo
    # YAML da la misma cercanía a la inestabilidad en cualquier red.
    beta = (
        beta_for_R(W, model_cfg["alpha"], model_cfg["R"])
        if model_cfg.get("R") is not None
        else model_cfg["beta"]
    )
    return Params(
        alpha=model_cfg["alpha"],
        beta=beta,
        D0=model_cfg["D0"],
        K=model_cfg["K"],
        gamma=model_cfg["gamma"],
        omega=omega,
        thresholds=thresholds,
        loss=model_cfg["loss"],
    )


def beta_for_R(W: np.ndarray, alpha: float, R: float) -> float:
    """β que produce un índice de amplificación R dado."""
    rho = spectral.perron_root(W)
    return R * alpha / rho if rho > 0 else 0.0


# ---------------------------------------------------------------- diagnóstico


def traffic_light(sigma: float, margin: float = 0.05) -> str:
    """Semáforo de estabilidad: verde (σ < -margen), amarillo (cerca de 0), rojo."""
    if sigma < -margin:
        return "verde"
    if sigma < margin:
        return "amarillo"
    return "rojo"


def diagnose(W: np.ndarray, params: Params, n_communities: int = 3) -> dict[str, Any]:
    """Índices espectrales de la red con los parámetros dados (Fases 1–2)."""
    sigma = spectral.stability(W, params)
    R = spectral.amplification_index(W, params)
    ranking = spectral.entity_ranking(W)
    return {
        "rho": spectral.perron_root(W),
        "R": R,
        "sigma": sigma,
        "recovery_time": -1.0 / sigma if sigma < 0 else float("inf"),
        "lambda2": spectral.algebraic_connectivity(W),
        "light": traffic_light(sigma),
        "light_R": "verde" if R < 1 else "rojo",
        "ranking": ranking,
        "communities": spectral.communities(W, n_communities=n_communities),
    }


# ---------------------------------------------------------------- simulación


def make_shock(shock_cfg: dict, targets: list[int] | tuple[int, ...]) -> Shock:
    return Shock(
        kind=shock_cfg["kind"],
        targets=tuple(targets),
        size=shock_cfg["size"],
        t0=shock_cfg["t0"],
        duration=shock_cfg["duration"],
    )


def simulate(
    W: np.ndarray,
    params: Params,
    shock: Shock | None,
    sim_cfg: dict,
    x0: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evolución de x(t) (y θ(t)) tras un choque; sin umbrales de impago."""
    p = params.with_(thresholds=None)
    traj = _simulate(
        W,
        p,
        t_end=sim_cfg["t_end"],
        dt=sim_cfg["dt"],
        x0=x0,
        theta0=theta0,
        shock=shock,
        save_every=sim_cfg["save_every"],
    )
    return {
        "trajectory": traj,
        "t": traj.t,
        "x": traj.x,
        "total_deficit": traj.x.sum(axis=1),
    }


def rhythm_run(
    W: np.ndarray,
    params: Params,
    sim_cfg: dict,
    theta0: np.ndarray,
    disconnect: int | None = None,
    t_disconnect: float | None = None,
) -> dict[str, Any]:
    """Ritmo de pagos: r(t), coherencia local y déficit, con desconexión opcional.

    Desconectar una entidad la retira de la red (fila y columna de W a cero) desde
    ``t_disconnect``; la corrida se integra en dos tramos continuos.
    """
    p = params.with_(thresholds=None)
    dt, t_end, se = sim_cfg["dt"], sim_cfg["t_end"], sim_cfg["save_every"]
    if disconnect is None:
        traj = _simulate(W, p, t_end, dt, theta0=theta0, save_every=se)
        t, x, th = traj.t, traj.x, traj.theta
        W_end = W
    else:
        t_cut = t_disconnect if t_disconnect is not None else t_end / 2
        a = _simulate(W, p, t_cut, dt, theta0=theta0, save_every=se)
        W_end = W.copy()
        W_end[disconnect, :] = 0.0
        W_end[:, disconnect] = 0.0
        b = _simulate(
            W_end,
            p,
            t_end - t_cut,
            dt,
            x0=a.x_final,
            theta0=a.theta_final,
            save_every=se,
        )
        t = np.concatenate([a.t, a.t[-1] + b.t[1:]])
        x = np.vstack([a.x, b.x[1:]])
        th = np.vstack([a.theta, b.theta[1:]])
    alive = np.ones(W.shape[0], dtype=bool)
    if disconnect is not None:
        alive[disconnect] = False
    return {
        "t": t,
        "x": x,
        "theta": th,
        "r": metrics.order_parameter(th[:, alive]),
        "r_local": metrics.local_coherence(th[-1], W_end),
        "deficit": x.sum(axis=1),
    }


def run_scenario(
    W: np.ndarray,
    params: Params,
    shock: Shock | None,
    sim_cfg: dict,
    collapse_fraction: float,
    theta0: np.ndarray | None = None,
) -> dict[str, Any]:
    """Escenario completo con impagos (Fase 4): trayectorias, eventos e indicadores."""
    if params.thresholds is None:
        raise ValueError("run_scenario necesita umbrales de impago (params.thresholds)")
    traj = _simulate(
        W,
        params,
        t_end=sim_cfg["t_end"],
        dt=sim_cfg["dt"],
        theta0=theta0,
        shock=shock,
        save_every=sim_cfg["save_every"],
    )
    summ = metrics.risk_summary(traj, W, collapse_fraction, shock)
    h0 = np.zeros(W.shape[0])
    if shock is not None:
        h0[list(shock.targets)] = 1.0
        summ["debtrank"] = cascades.debtrank(W, h0)["R"]
    else:
        summ["debtrank"] = 0.0
    # Estado por instante guardado:
    # 0 = normal, 1 = en estrés (x > umbral/2), 2 = impago.
    dtimes = traj.default_time
    status = np.where(traj.x > 0.5 * params.thresholds[None, :], 1, 0)
    status = np.where(
        np.isfinite(dtimes)[None, :] & (dtimes[None, :] <= traj.t[:, None] + 1e-12),
        2,
        status,
    )
    return {
        "trajectory": traj,
        "events": traj.events,
        "metrics": summ,
        "status": status,
    }


def dangerous_entities(
    W: np.ndarray, cfg: dict, seed: int, top: int = 10
) -> pd.DataFrame:
    """Ranking de entidades más peligrosas por importancia sistémica simulada."""
    m = cfg["metrics"]
    params = make_params(W, cfg["model"], seed=seed, with_thresholds=False)
    imp = metrics.systemic_importance(
        W,
        params,
        shock_size=cfg["shock"]["size"],
        n_seeds=m["importance_seeds"],
        dt=m["importance_dt"],
        t_end=m["importance_t_end"],
        buffer=cfg["model"]["buffer"],
        buffer_dispersion=cfg["model"]["buffer_dispersion"],
        seed=seed,
    )
    df = pd.DataFrame(
        {"entidad": np.arange(W.shape[0]), "cascada_media": imp, "paga": W.sum(axis=0)}
    )
    return df.sort_values("cascada_media", ascending=False, ignore_index=True).head(top)


def metrics_csv(summary_metrics: dict, events: pd.DataFrame) -> bytes:
    """CSV con indicadores (cabecera) y registro de eventos para exportar."""
    lines = ["indicador,valor"] + [f"{k},{v}" for k, v in summary_metrics.items()]
    lines += ["", events.to_csv(index=False)]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------- ejemplo


def example_scenario() -> dict[str, Any]:
    """Escenario "Bre-B estilizado" (parámetros ilustrativos, no calibrados)."""
    cfg = merged(default_config(), load_config("breb_example.yaml"))
    net = build_network(cfg["network"], seed=cfg["seed"])
    return {**net, "config": cfg}
