from __future__ import annotations

import numpy as np
import pytest

from pulso.networks import NetworkConfig, generate


def _cfg(
    topology: str, n: int = 40, mean_degree: float = 5.0, seed: int = 3
) -> NetworkConfig:
    return NetworkConfig(
        topology=topology,
        n=n,
        mean_degree=mean_degree,
        amount_sigma=0.8,
        core_frac=0.1,
        p_core=0.8,
        periphery_share=0.1,
        seed=seed,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(params=["erdos_renyi", "barabasi_albert", "core_periphery"])
def W_any(request) -> np.ndarray:
    return generate(_cfg(request.param))


@pytest.fixture
def W_er() -> np.ndarray:
    return generate(_cfg("erdos_renyi"))


@pytest.fixture
def W_strong() -> np.ndarray:
    """Red ER densa: fuertemente conexa con probabilidad ~1 para esta semilla."""
    import networkx as nx

    from pulso.networks import to_networkx

    W = generate(_cfg("erdos_renyi", n=30, mean_degree=8.0, seed=7))
    assert nx.is_strongly_connected(to_networkx(W))
    return W


@pytest.fixture
def W_symmetric() -> np.ndarray:
    W = generate(_cfg("erdos_renyi", n=30, mean_degree=6.0, seed=11))
    return 0.5 * (W + W.T)
