"""Índices que ve la app y que alimentan el dataset (Fases 3–4)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from finnet.integrate import Shock, Trajectory, simulate
from finnet.model import Dynamics, Params


def order_parameter(theta: np.ndarray) -> np.ndarray | float:
    """``r = |<e^{iθ}>|`` global. Acepta (N,) o (T, N)."""
    z = np.exp(1j * np.asarray(theta)).mean(axis=-1)
    return np.abs(z)


def local_coherence(theta: np.ndarray, W: np.ndarray) -> np.ndarray:
    """``r_i = Σ_j W_ij cos(θ_j - θ_i) / k_i``; 1 para quien no recibe pagos."""
    W = np.asarray(W, dtype=float)
    return Dynamics(W, Params(alpha=0.0, beta=0.0)).local_coherence(np.asarray(theta))


def fraction_defaulted(traj: Trajectory) -> float:
    return float(traj.defaulted.mean())


def unpaid_liquidity(traj: Trajectory) -> float:
    """Liquidez no pagada acumulada en el horizonte [pagos medios]."""
    return float(traj.unpaid[-1])


def time_to_first_default(traj: Trajectory, exclude: tuple[int, ...] = ()) -> float:
    """Primer impago (excluyendo entidades dadas, p. ej. la chocada). NaN si ninguno."""
    dt = traj.default_time.copy()
    if exclude:
        dt[list(exclude)] = np.nan
    return float(np.nanmin(dt)) if np.isfinite(dt).any() else float("nan")


def risk_summary(
    traj: Trajectory,
    W: np.ndarray,
    collapse_fraction: float,
    shock: Shock | None = None,
) -> dict[str, float]:
    """Indicadores de riesgo de una corrida."""
    out0 = np.asarray(W).sum(axis=0)
    frac = fraction_defaulted(traj)
    return {
        "fraction_defaulted": frac,
        "n_defaulted": int(traj.defaulted.sum()),
        "collapse": bool(frac >= collapse_fraction),
        "unpaid_liquidity": unpaid_liquidity(traj),
        "unpaid_flow_fraction": float(
            out0[traj.defaulted].sum() / max(out0.sum(), 1e-300)
        ),
        "time_to_first_default": time_to_first_default(traj),
        "time_to_first_contagion": time_to_first_default(
            traj, exclude=shock.targets if shock else ()
        ),
    }


def _cascade_size(W, params, target, size, dt, t_end, seed, buffer_cfg):
    from finnet.model import make_thresholds

    p = params
    if buffer_cfg is not None:
        rng = np.random.default_rng(seed)
        p = params.with_(thresholds=make_thresholds(W, *buffer_cfg, rng))
    shock = Shock(kind="liquidity", targets=(target,), size=size, t0=0.0)
    traj = simulate(W, p, t_end=t_end, dt=dt, shock=shock, save_every=10**9)
    others = traj.defaulted.copy()
    others[target] = False
    return others.mean()


def systemic_importance(
    W: np.ndarray,
    params: Params,
    shock_size: float,
    n_seeds: int,
    dt: float,
    t_end: float,
    buffer: float | None = None,
    buffer_dispersion: float = 0.0,
    seed: int = 0,
    n_jobs: int = -1,
) -> np.ndarray:
    """Fracción media de *otras* entidades que caen al chocar cada entidad.

    Si se dan ``buffer`` y ``buffer_dispersion`` se re-muestrean los umbrales con
    cada semilla; si no, se usan ``params.thresholds`` fijos (y basta 1 semilla).
    """
    if params.thresholds is None and buffer is None:
        raise ValueError(
            "systemic_importance necesita umbrales (params.thresholds o buffer)"
        )
    n = W.shape[0]
    buffer_cfg = (buffer, buffer_dispersion) if buffer is not None else None
    seeds = np.random.SeedSequence(seed).generate_state(n_seeds)
    jobs = [
        delayed(_cascade_size)(W, params, i, shock_size, dt, t_end, int(s), buffer_cfg)
        for i in range(n)
        for s in seeds
    ]
    res = (
        Parallel(n_jobs=n_jobs)(jobs)
        if n * n_seeds > 32
        else [j[0](*j[1], **j[2]) for j in jobs]
    )
    return np.asarray(res).reshape(n, n_seeds).mean(axis=1)
