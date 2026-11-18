"""Modelos clásicos de contagio para comparar con la dinámica (Fase 4).

- Eisenberg y Noe (2001): vector de liquidación de pagos.
- DebtRank (Battiston et al. 2012): propagación de estrés sin reverberación.
- Watts (2002): cascadas de umbral fraccional.
"""

from __future__ import annotations

import numpy as np


def eisenberg_noe(
    p_bar: np.ndarray,
    Pi: np.ndarray,
    e: np.ndarray,
    tol: float = 1e-13,
    max_iter: int = 10_000,
) -> np.ndarray:
    r"""Vector de liquidación máximo ``p = min(p̄, e + Π^T p)``.

    Parameters
    ----------
    p_bar : obligaciones totales de cada entidad.
    Pi : pasivos relativos, ``Π[i, j]`` = fracción de lo que i debe que va a j
        (filas suman 1 o 0).
    e : activos externos (operativos) de cada entidad, ``e >= 0``.

    Iteración de punto fijo desde ``p = p̄``: el operador es monótono y la sucesión
    decrece hacia el mayor punto fijo (Eisenberg y Noe, 2001, Teorema 1).
    """
    p_bar = np.asarray(p_bar, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    e = np.asarray(e, dtype=float)
    if (e < 0).any() or (p_bar < 0).any():
        raise ValueError("e y p_bar deben ser no negativos")
    rows = Pi.sum(axis=1)
    if not np.all((np.abs(rows - 1) < 1e-9) | (rows == 0)):
        raise ValueError(
            "Cada fila de Pi debe sumar 1 (o 0 si la entidad no debe nada)"
        )
    p = p_bar.copy()
    for _ in range(max_iter):
        p_new = np.minimum(p_bar, e + Pi.T @ p)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    raise RuntimeError(f"Eisenberg–Noe no convergió en {max_iter} iteraciones")


def payment_obligations(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Traduce la red de pagos a (p̄, Π): j debe ``W[i, j]`` a i por ciclo."""
    W = np.asarray(W, dtype=float)
    p_bar = W.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pi = np.where(
            p_bar[:, None] > 0, W.T / np.where(p_bar > 0, p_bar, 1.0)[:, None], 0.0
        )
    return p_bar, Pi


def debtrank(
    W: np.ndarray,
    h0: np.ndarray,
    v: np.ndarray | None = None,
    max_iter: int = 1000,
) -> dict[str, np.ndarray | float]:
    """DebtRank sobre la red de pagos.

    Impacto de j sobre i: ``I[j, i] = W[i, j] / k_i`` (fracción del ingreso de i que
    aporta j). Estados: U (sin estrés), D (estresado, propaga una sola vez) e I
    (inactivo). ``h_i`` en [0, 1].

    Returns
    -------
    dict con ``h`` final, ``R`` (DebtRank: estrés inducido ponderado por ``v``,
    excluyendo el inicial) y ``R_total``.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    h0 = np.clip(np.asarray(h0, dtype=float), 0.0, 1.0)
    k = W.sum(axis=1)
    if v is None:
        v = k / k.sum() if k.sum() > 0 else np.full(n, 1.0 / n)
    v = np.asarray(v, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        impact = np.where(k[:, None] > 0, W / np.where(k > 0, k, 1.0)[:, None], 0.0).T
    h = h0.copy()
    state = np.where(h > 0, 1, 0)  # 0=U, 1=D, 2=I
    for _ in range(max_iter):
        distressed = state == 1
        if not distressed.any():
            break
        h_new = np.minimum(1.0, h + impact[distressed].T @ h[distressed])
        state[distressed] = 2
        newly = (h_new > 0) & (state == 0)
        state[newly] = 1
        h = h_new
    return {"h": h, "R": float(v @ h - v @ h0), "R_total": float(v @ h)}


def watts(
    W: np.ndarray,
    phi: np.ndarray | float,
    seeds: np.ndarray | list[int],
) -> np.ndarray:
    """Cascada de umbral de Watts (2002) dirigida.

    Una entidad cae si la fracción de sus pagadores caídos es ``>= phi_i``. Las
    entidades sin pagadores no caen salvo que sean semilla. Devuelve la máscara final.
    """
    A = np.asarray(W) > 0
    n = A.shape[0]
    phi = np.broadcast_to(np.asarray(phi, dtype=float), (n,))
    deg = A.sum(axis=1)
    active = np.zeros(n, dtype=bool)
    active[np.asarray(seeds, dtype=int)] = True
    while True:
        frac = np.where(
            deg > 0, (A & active[None, :]).sum(axis=1) / np.maximum(deg, 1), 0.0
        )
        new = ~active & (deg > 0) & (frac >= phi)
        if not new.any():
            return active
        active |= new
