r"""Espectros y estabilidad (Fases 1–2).

Fase 1: ``R = β ρ(W) / α``. El sistema lineal ``ẋ = (-αI + βW) x`` es estable si y
solo si ``R < 1`` (W no negativa: su valor propio de mayor parte real es ρ(W),
Perron–Frobenius).

Fase 2: ``M = -αI - D0 L + βW`` con ``L = D_out - W``;
``σ = max Re λ(M)`` y ``τ_rec = -1/σ`` si ``σ < 0``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from pulso.integrate import linear_operator
from pulso.model import Params


def perron_root(W: np.ndarray) -> float:
    """Valor propio dominante ρ(W) (real y no negativo para W >= 0)."""
    ev = np.linalg.eigvals(np.asarray(W, dtype=float))
    return float(np.max(ev.real)) if ev.size else 0.0


def amplification_index(W: np.ndarray, params: Params) -> float:
    """Índice de amplificación ``R = β ρ(W) / α``. ``inf`` si α = 0 y β ρ > 0."""
    num = params.beta * perron_root(W)
    if params.alpha == 0:
        return float("inf") if num > 0 else 0.0
    return num / params.alpha


def _perron_vector(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A)
    v = vecs[:, np.argmax(vals.real)].real
    v = np.abs(v)  # Perron: el vector dominante tiene un solo signo
    s = v.sum()
    return v / s if s > 0 else np.full(A.shape[0], 1.0 / A.shape[0])


def entity_ranking(W: np.ndarray) -> pd.DataFrame:
    """Ranking de entidades por vectores propios dominantes de W.

    - ``importancia`` (vector izquierdo, ``u^T W = ρ u^T``): cuánto déficit propaga
      la entidad en el modo dominante (es pagadora clave).
    - ``vulnerabilidad`` (vector derecho, ``W v = ρ v``): cuánto déficit acumula.
    """
    W = np.asarray(W, dtype=float)
    df = pd.DataFrame(
        {
            "entidad": np.arange(W.shape[0]),
            "importancia": _perron_vector(W.T),
            "vulnerabilidad": _perron_vector(W),
            "paga": W.sum(axis=0),
            "recibe": W.sum(axis=1),
        }
    )
    return df.sort_values("importancia", ascending=False, ignore_index=True)


def flow_laplacian(W: np.ndarray) -> np.ndarray:
    """``L = D_out - W``: columnas suman cero, así ``1^T L = 0`` (conservación)."""
    W = np.asarray(W, dtype=float)
    return np.diag(W.sum(axis=0)) - W


def spectrum(M: np.ndarray) -> np.ndarray:
    """Valores propios de M ordenados por parte real descendente."""
    ev = np.linalg.eigvals(np.asarray(M, dtype=float))
    return ev[np.argsort(-ev.real)]


def stability(W: np.ndarray, params: Params) -> float:
    """``σ = max Re λ(M)``; el sistema lineal es estable si σ < 0."""
    return float(spectrum(linear_operator(W, params))[0].real)


def recovery_time(W: np.ndarray, params: Params) -> float:
    """``τ_rec = -1/σ`` en ciclos; ``inf`` si σ >= 0 (no se recupera)."""
    sigma = stability(W, params)
    return -1.0 / sigma if sigma < 0 else float("inf")


def symmetric_laplacian(W: np.ndarray) -> np.ndarray:
    """Laplaciano de la parte simétrica ``(W + W^T)/2``."""
    W = np.asarray(W, dtype=float)
    S = 0.5 * (W + W.T)
    return np.diag(S.sum(axis=1)) - S


def algebraic_connectivity(W: np.ndarray) -> float:
    """λ2 del Laplaciano simétrico (conectividad algebraica de Fiedler)."""
    ev = np.linalg.eigvalsh(symmetric_laplacian(W))
    return float(ev[1]) if ev.size > 1 else 0.0


def fiedler_vector(W: np.ndarray) -> np.ndarray:
    _, vecs = np.linalg.eigh(symmetric_laplacian(W))
    return vecs[:, 1]


def normalized_fiedler_vector(W: np.ndarray) -> np.ndarray:
    """Fiedler de ``I - D^{-1/2} S D^{-1/2}`` devuelto como ``D^{-1/2} u``.

    Es la relajación del corte normalizado (Shi y Malik, 2000). El Laplaciano sin
    normalizar tiende a separar unos pocos nodos de periferia poco conectados; el
    normalizado pesa cada lado por su volumen y da comunidades balanceadas.
    """
    S = 0.5 * (np.asarray(W, dtype=float) + np.asarray(W, dtype=float).T)
    d = S.sum(axis=1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    Ln = np.eye(S.shape[0]) - inv[:, None] * S * inv[None, :]
    _, vecs = np.linalg.eigh(Ln)
    return inv * vecs[:, 1]


def _bisect(W: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = W[np.ix_(nodes, nodes)]
    f = normalized_fiedler_vector(sub)
    left = nodes[f < np.median(f)] if (f < 0).all() or (f >= 0).all() else nodes[f < 0]
    right = np.setdiff1d(nodes, left)
    return left, right


def communities(
    W: np.ndarray, n_communities: int = 2, method: str = "spectral"
) -> np.ndarray:
    """Etiqueta de comunidad por entidad.

    ``spectral``: bisección recursiva con el vector de Fiedler del Laplaciano
    normalizado de la parte simétrica, partiendo siempre la comunidad más grande.
    ``greedy``: modularidad de
    networkx (respaldo para redes desconectadas o muy heterogéneas).
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    labels = np.zeros(n, dtype=int)
    if method == "greedy":
        G = nx.from_numpy_array(0.5 * (W + W.T))
        for c, members in enumerate(
            nx.community.greedy_modularity_communities(G, weight="weight")
        ):
            labels[list(members)] = c
        return labels
    if method != "spectral":
        raise ValueError(f"method debe ser 'spectral' o 'greedy', no {method!r}")
    groups = [np.arange(n)]
    while len(groups) < n_communities:
        groups.sort(key=len, reverse=True)
        big = groups.pop(0)
        if len(big) < 2:
            groups.append(big)
            break
        a, b = _bisect(W, big)
        if len(a) == 0 or len(b) == 0:
            groups.append(big)
            break
        groups += [a, b]
    for c, g in enumerate(groups):
        labels[g] = c
    return labels


def null_vector(L: np.ndarray) -> np.ndarray:
    """Vector nulo derecho de L con suma 1 (estado conservado de la Fase 2)."""
    vals, vecs = np.linalg.eig(L)
    v = np.abs(vecs[:, np.argmin(np.abs(vals))].real)
    return v / v.sum()
