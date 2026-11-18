"""Generadores de redes de pagos dirigidas y con pesos (Fase 1).

Convención única del proyecto: ``W[i, j]`` es el valor medio de pagos **de j hacia
i** por ciclo. Las filas suman lo que recibe cada entidad (``k_i``) y las columnas
lo que paga (``out_j``). La única transposición hacia la convención de networkx
(arista j -> i) está en :func:`to_networkx`.

Los montos se normalizan para que el peso medio de las aristas sea 1: la unidad de
dinero del modelo es "un pago medio por ciclo", y así ``beta`` y ``K`` son
comparables entre redes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np

TOPOLOGIES = ("erdos_renyi", "barabasi_albert", "core_periphery")


@dataclass(frozen=True)
class NetworkConfig:
    """Parámetros de un generador de red. Los valores vienen de ``configs/``."""

    topology: str
    n: int
    mean_degree: float
    amount_sigma: float
    core_frac: float
    p_core: float
    periphery_share: float
    seed: int

    def __post_init__(self) -> None:
        if self.topology not in TOPOLOGIES:
            raise ValueError(
                f"topology debe ser una de {TOPOLOGIES}, no {self.topology!r}"
            )
        if self.n < 3:
            raise ValueError(f"n debe ser >= 3 (recibido {self.n})")
        if not 0 < self.mean_degree < self.n - 1:
            raise ValueError(
                f"mean_degree debe estar en (0, n-1); recibido {self.mean_degree}"
            )
        if self.amount_sigma < 0:
            raise ValueError("amount_sigma debe ser >= 0")
        if not 0 < self.core_frac < 1:
            raise ValueError("core_frac debe estar en (0, 1)")

    @classmethod
    def from_dict(cls, d: dict, seed: int | None = None) -> NetworkConfig:
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        if seed is not None:
            fields["seed"] = seed
        missing = set(cls.__dataclass_fields__) - set(fields)
        if missing:
            raise KeyError(f"Faltan parámetros de red: {sorted(missing)}")
        return cls(**fields)

    def to_dict(self) -> dict:
        return asdict(self)


def _adjacency_erdos_renyi(n: int, k: float, rng: np.random.Generator) -> np.ndarray:
    p = k / (n - 1)
    A = rng.random((n, n)) < p
    np.fill_diagonal(A, False)
    return A


def _adjacency_barabasi_albert(
    n: int, k: float, rng: np.random.Generator
) -> np.ndarray:
    # Cada arista no dirigida de BA se vuelve recíproca: las redes de pagos tienen
    # reciprocidad alta (quien paga a un banco también recibe de él).
    m = max(1, min(n - 1, round(k / 2)))
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(2**31 - 1)))
    A = nx.to_numpy_array(G, dtype=bool)
    return A


def _adjacency_core_periphery(
    n: int,
    k: float,
    core_frac: float,
    p_core: float,
    periphery_share: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_core = max(2, round(core_frac * n))
    n_per = n - n_core
    core = np.zeros(n, dtype=bool)
    core[:n_core] = True

    total_edges = k * n
    core_edges = p_core * n_core * (n_core - 1)
    rest = max(total_edges - core_edges, 0.0)
    # Aristas núcleo<->periferia en ambos sentidos y una fracción pequeña entre
    # nodos de la periferia.
    p_cp = min(1.0, (1 - periphery_share) * rest / max(2 * n_core * n_per, 1))
    p_pp = min(1.0, periphery_share * rest / max(n_per * (n_per - 1), 1))

    prob = np.where(
        core[:, None] & core[None, :],
        p_core,
        np.where(core[:, None] ^ core[None, :], p_cp, p_pp),
    )
    A = rng.random((n, n)) < prob
    np.fill_diagonal(A, False)
    return A


def generate(cfg: NetworkConfig) -> np.ndarray:
    """Genera la matriz de pagos ``W`` (n x n, ``W[i, j]`` = pago de j a i).

    Montos: ``W_ij = sqrt(s_i s_j) * eps_ij`` con tamaños de entidad ``s`` y ruido
    ``eps`` lognormales de parámetro ``amount_sigma``; luego se normaliza a peso
    medio 1 sobre las aristas existentes.
    """
    rng = np.random.default_rng(cfg.seed)
    match cfg.topology:
        case "erdos_renyi":
            A = _adjacency_erdos_renyi(cfg.n, cfg.mean_degree, rng)
        case "barabasi_albert":
            A = _adjacency_barabasi_albert(cfg.n, cfg.mean_degree, rng)
        case "core_periphery":
            A = _adjacency_core_periphery(
                cfg.n,
                cfg.mean_degree,
                cfg.core_frac,
                cfg.p_core,
                cfg.periphery_share,
                rng,
            )
    sizes = rng.lognormal(0.0, cfg.amount_sigma, cfg.n)
    noise = rng.lognormal(0.0, cfg.amount_sigma, (cfg.n, cfg.n))
    W = np.where(A, np.sqrt(np.outer(sizes, sizes)) * noise, 0.0)
    return normalize_weights(W)


def normalize_weights(W: np.ndarray) -> np.ndarray:
    """Escala ``W`` para que el peso medio de las aristas no nulas sea 1."""
    W = np.asarray(W, dtype=float)
    nz = W[W > 0]
    if nz.size == 0:
        return W.copy()
    return W / nz.mean()


def complete_graph(n: int) -> np.ndarray:
    """Red completa sin autoaristas, pesos 1 (pruebas de Kuramoto)."""
    W = np.ones((n, n))
    np.fill_diagonal(W, 0.0)
    return W


def received(W: np.ndarray) -> np.ndarray:
    """``k_i = sum_j W_ij``: lo que recibe cada entidad por ciclo."""
    return W.sum(axis=1)


def paid(W: np.ndarray) -> np.ndarray:
    """``out_j = sum_i W_ij``: lo que paga cada entidad por ciclo."""
    return W.sum(axis=0)


def core_mask(W: np.ndarray, cfg: NetworkConfig) -> np.ndarray:
    """Máscara de nodos del núcleo (solo tiene sentido en ``core_periphery``)."""
    mask = np.zeros(W.shape[0], dtype=bool)
    if cfg.topology == "core_periphery":
        mask[: max(2, round(cfg.core_frac * cfg.n))] = True
    return mask


def to_networkx(W: np.ndarray) -> nx.DiGraph:
    """Grafo dirigido con arista j -> i de peso ``W[i, j]``.

    Único punto del proyecto donde se transpone la convención de ``W``.
    """
    return nx.from_numpy_array(np.asarray(W).T, create_using=nx.DiGraph)


def summary(W: np.ndarray) -> dict[str, float]:
    """Descriptores estructurales que usan la app y el dataset."""
    n = W.shape[0]
    A = W > 0
    edges = int(A.sum())
    nz = W[A]
    return {
        "n": n,
        "edges": edges,
        "density": edges / (n * (n - 1)),
        "mean_degree": edges / n,
        "amount_cv": float(nz.std() / nz.mean()) if edges else 0.0,
    }
