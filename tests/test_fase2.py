"""Fase 2: redistribución conservativa -D0 L x y estabilidad espectral."""

from __future__ import annotations

import numpy as np
import pytest

from finnet.integrate import linear_operator, simulate
from finnet.model import Params
from finnet.spectral import (
    algebraic_connectivity,
    communities,
    fiedler_vector,
    flow_laplacian,
    null_vector,
    perron_root,
    recovery_time,
    stability,
    symmetric_laplacian,
)


def test_flow_laplacian_columns_sum_zero(W_any):
    np.testing.assert_allclose(flow_laplacian(W_any).sum(axis=0), 0, atol=1e-12)


def test_pure_redistribution_conserves_total_deficit(W_any):
    rng = np.random.default_rng(1)
    x0 = rng.random(W_any.shape[0])
    p = Params(alpha=0.0, beta=0.0, D0=0.8)
    traj = simulate(W_any, p, t_end=5.0, dt=0.01, x0=x0, save_every=50)
    sums = traj.x.sum(axis=1)
    assert np.max(np.abs(sums - x0.sum())) < 1e-10


def test_pure_redistribution_converges_to_scaled_null_vector(W_strong):
    rng = np.random.default_rng(2)
    x0 = rng.random(W_strong.shape[0])
    p = Params(alpha=0.0, beta=0.0, D0=1.0)
    traj = simulate(W_strong, p, t_end=60.0, dt=0.01, x0=x0, save_every=6000)
    v = null_vector(flow_laplacian(W_strong))
    np.testing.assert_allclose(traj.x_final, v * x0.sum(), rtol=1e-6, atol=1e-9)


def test_symmetric_network_decay_rate_alpha_plus_D0_lambda2(W_symmetric):
    alpha, D0 = 0.3, 0.5
    p = Params(alpha=alpha, beta=0.0, D0=D0)
    lam2 = algebraic_connectivity(W_symmetric)
    # Fiedler: modo más lento del subespacio ortogonal al uniforme.
    x0 = fiedler_vector(W_symmetric)
    traj = simulate(W_symmetric, p, t_end=4.0, dt=0.01, x0=x0, save_every=10)
    norms = np.linalg.norm(traj.x, axis=1)
    rate = -np.polyfit(traj.t, np.log(norms), 1)[0]
    assert rate == pytest.approx(alpha + D0 * lam2, rel=0.01)


def test_symmetric_laplacian_matches_flow_laplacian_when_symmetric(W_symmetric):
    np.testing.assert_allclose(
        symmetric_laplacian(W_symmetric), flow_laplacian(W_symmetric)
    )


def test_stability_reduces_to_R_without_redistribution(W_any):
    rho = perron_root(W_any)
    p = Params(alpha=1.0, beta=0.5 / rho)
    assert stability(W_any, p) == pytest.approx(-0.5, abs=1e-8)
    assert recovery_time(W_any, p) == pytest.approx(2.0, rel=1e-6)


def test_unstable_recovery_time_infinite(W_any):
    rho = perron_root(W_any)
    assert recovery_time(W_any, Params(alpha=1.0, beta=1.5 / rho)) == float("inf")


def test_operator_with_D0_zero_equals_phase1(W_any):
    p = Params(alpha=1.0, beta=0.1)
    M = linear_operator(W_any, p)
    np.testing.assert_allclose(M, -np.eye(W_any.shape[0]) + 0.1 * W_any)


def test_communities_split_two_disconnected_blocks():
    block = np.ones((5, 5)) - np.eye(5)
    W = np.zeros((10, 10))
    W[:5, :5] = block
    W[5:, 5:] = block
    labels = communities(W, n_communities=2)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_normalized_communities_balanced_on_core_periphery():
    from finnet.networks import NetworkConfig, generate

    W = generate(
        NetworkConfig("core_periphery", 200, 8.0, 1.0, 0.06, 0.9, 0.05, seed=1)
    )
    sizes = np.bincount(communities(W, n_communities=2))
    assert sizes.min() >= 0.15 * W.shape[0]
