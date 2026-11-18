"""Estimación desde el registro de transacciones: lo que el banco puede medir.

Nada aquí usa columnas ``truth_*``: las mismas funciones correrían sobre el registro
real de un banco.

Fase por eventos (Pikovsky, Rosenblum y Kurths, 2001, cap. 3): si ``t_k`` son los
instantes de los ciclos de pago de una entidad,

    θ̂(t) = 2π (k + (t - t_k) / (t_{k+1} - t_k)),   t_k <= t < t_{k+1}.

Antes del primer ciclo y después del último se extrapola con el período medio, pero
solo hasta 1.5 períodos: si una entidad deja de pagar, su fase deja de existir
(NaN) y no cuenta en la coherencia de sus receptores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pulso.world import BANK

TWO_PI = 2.0 * np.pi
_EXTRAPOLATE_PERIODS = 1.5


def active_span(tx: pd.DataFrame, t_obs: float, merge_days: float = 0.5) -> pd.Series:
    """Meses en que cada pagador estuvo activo: ``min(t_obs, último ciclo + período)``.

    Un pagador que dejó de pagar a mitad de la ventana no se promedia sobre toda la
    ventana: su ``Ŵ`` es el flujo mientras pagaba (lo que sus receptores perdieron).
    """
    n = int(tx["src"].max()) + 1 if len(tx) else 0
    spans = {}
    for src, c in enumerate(payment_cycles(tx, n, merge_days)):
        if c.size == 0:
            continue
        period = float(np.mean(np.diff(c))) if c.size >= 2 else t_obs
        spans[src] = min(t_obs, c[-1] + period)
    return pd.Series(spans, name="span", dtype=float)


def estimate_edges(tx: pd.DataFrame, t_obs: float) -> pd.DataFrame:
    """``Ŵ``: flujo medio mensual por (pagador, receptor, capa) mientras pagaba."""
    if t_obs <= 0:
        raise ValueError("t_obs debe ser positivo")
    net = tx[tx["dst"] != BANK]
    w = net.groupby(["src", "dst", "layer"], as_index=False)["amount"].sum()
    span = active_span(tx, t_obs).reindex(w["src"]).fillna(t_obs).to_numpy()
    w["weight"] = w.pop("amount").to_numpy() / span
    return w


def edges_matrix(edges: pd.DataFrame, n: int, layers=None) -> sp.csr_array:
    """``Ŵ[i, j]`` = lo que j le paga a i por mes (convención del repositorio)."""
    e = edges if layers is None else edges[edges["layer"].isin(list(layers))]
    return sp.csr_array(
        (e["weight"].to_numpy(), (e["dst"].to_numpy(), e["src"].to_numpy())),
        shape=(n, n),
    )


def payment_cycles(
    tx: pd.DataFrame, n: int, merge_days: float = 0.5
) -> list[np.ndarray]:
    """Instantes de los ciclos de pago de cada entidad.

    Pagos de una misma entidad separados por menos de ``merge_days`` se toman como
    un solo ciclo (un mismo día de pagos).
    """
    tol = merge_days / 30.0
    out: list[np.ndarray] = [np.empty(0) for _ in range(n)]
    for src, g in tx.groupby("src"):
        t = np.sort(g["t"].to_numpy())
        keep = np.concatenate([[True], np.diff(t) > tol])
        out[int(src)] = t[keep]
    return out


def event_phase(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """θ̂ en ``grid`` a partir de los instantes de ciclo (NaN donde no se sabe)."""
    theta = np.full(grid.shape, np.nan)
    if times.size < 2:
        return theta
    period = float(np.mean(np.diff(times)))
    k = np.searchsorted(times, grid, side="right") - 1
    inside = (k >= 0) & (k < times.size - 1)
    ki = k[inside]
    theta[inside] = TWO_PI * (
        ki + (grid[inside] - times[ki]) / (times[ki + 1] - times[ki])
    )
    before = (grid < times[0]) & (grid >= times[0] - _EXTRAPOLATE_PERIODS * period)
    theta[before] = TWO_PI * (grid[before] - times[0]) / period
    after = (grid >= times[-1]) & (grid <= times[-1] + _EXTRAPOLATE_PERIODS * period)
    theta[after] = TWO_PI * (times.size - 1 + (grid[after] - times[-1]) / period)
    return theta


def phase_matrix(cycles: list[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """θ̂ de todas las entidades en la rejilla, forma (G, N)."""
    return np.column_stack([event_phase(c, grid) for c in cycles])


def local_coherence_series(W: sp.csr_array, theta: np.ndarray) -> np.ndarray:
    """r̂_i(t) = Re Σ_j Ŵ_ij e^{i(θ_j - θ_i)} / Σ_j Ŵ_ij, sobre pagadores con fase.

    NaN si en ese instante ningún pagador (o la propia entidad) tiene fase.
    """
    valid = np.isfinite(theta)
    z = np.where(valid, np.exp(1j * np.nan_to_num(theta)), 0.0)  # (G, N)
    num = (W @ z.T).T  # Σ_j W_ij e^{iθ_j}
    den = (W @ valid.T.astype(float)).T  # ingreso de pagadores con fase
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.real(num * np.conj(z)) / den
    r[(den <= 0) | ~valid] = np.nan
    return r


def phase_lag_series(W: sp.csr_array, theta: np.ndarray) -> np.ndarray:
    """Desfase medio con los pagadores, arg Σ_j Ŵ_ij e^{i(θ_j - θ_i)} ∈ (-π, π]."""
    valid = np.isfinite(theta)
    z = np.where(valid, np.exp(1j * np.nan_to_num(theta)), 0.0)
    num = (W @ z.T).T * np.conj(z)
    lag = np.angle(num)
    lag[(np.abs(num) == 0) | ~valid] = np.nan
    return lag


def global_order(theta: np.ndarray) -> np.ndarray:
    """r(t) = |⟨e^{iθ}⟩| de la cartera (entidades con fase en ese instante)."""
    valid = np.isfinite(theta)
    z = np.where(valid, np.exp(1j * np.nan_to_num(theta)), 0.0)
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, np.abs(z.sum(axis=1)) / cnt, np.nan)


def cycle_stats(
    cycles: list[np.ndarray], t_obs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ritmo estimado ω̂ [rad/mes], CV de los períodos y si dejó de pagar.

    Una entidad "dejó de pagar" si su último ciclo quedó a más de 1.5 períodos del
    cierre de la ventana.
    """
    n = len(cycles)
    omega = np.full(n, np.nan)
    cv = np.full(n, np.nan)
    stopped = np.zeros(n, dtype=bool)
    for i, c in enumerate(cycles):
        if c.size < 2:
            # Con un solo ciclo no hay período: no se puede afirmar que dejó de pagar.
            continue
        d = np.diff(c)
        omega[i] = TWO_PI / d.mean()
        cv[i] = d.std() / d.mean() if d.size > 1 else 0.0
        stopped[i] = t_obs - c[-1] > _EXTRAPOLATE_PERIODS * d.mean()
    return omega, cv, stopped
