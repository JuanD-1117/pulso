"""Fase 3: ritmo de pagos (Kuramoto en red) y acople γ(1 - r_i)."""

from __future__ import annotations

import numpy as np
import pytest

from pulso.integrate import simulate
from pulso.metrics import local_coherence, order_parameter
from pulso.model import Dynamics, Params, lorentz_frequencies
from pulso.networks import complete_graph
from pulso.spectral import perron_root

DELTA = 0.5
N_KURAMOTO = 2000


@pytest.fixture(scope="module")
def W_complete() -> np.ndarray:
    return complete_graph(N_KURAMOTO)


def _steady_r(W, K, seed=0, t_end=60.0, dt=0.05):
    rng = np.random.default_rng(seed)
    omega = lorentz_frequencies(W.shape[0], 0.0, DELTA)
    theta0 = rng.uniform(0, 2 * np.pi, W.shape[0])
    p = Params(alpha=1.0, beta=0.0, K=K, omega=omega)
    traj = simulate(W, p, t_end=t_end, dt=dt, theta0=theta0, save_every=20)
    r = order_parameter(traj.theta)
    return r[traj.t > t_end / 2].mean()


@pytest.mark.slow
@pytest.mark.parametrize("K", [2.0, 3.0, 4.0])
def test_complete_graph_lorentz_matches_kuramoto_theory(W_complete, K):
    r_theory = np.sqrt(1 - 2 * DELTA / K)
    assert _steady_r(W_complete, K) == pytest.approx(r_theory, abs=0.05)


@pytest.mark.slow
def test_below_critical_coupling_incoherent(W_complete):
    # K < Kc = 2Δ = 1: r ~ O(1/sqrt(N)) por tamaño finito.
    assert _steady_r(W_complete, 0.5) < 0.1


def test_zero_coupling_free_rotation_exact(W_er):
    rng = np.random.default_rng(0)
    n = W_er.shape[0]
    omega = rng.normal(6.0, 1.0, n)
    theta0 = rng.uniform(0, 2 * np.pi, n)
    traj = simulate(
        W_er,
        Params(alpha=1.0, beta=0.0, omega=omega),
        t_end=5.0,
        dt=0.01,
        theta0=theta0,
        save_every=50,
    )
    expected = theta0[None, :] + omega[None, :] * traj.t[:, None]
    np.testing.assert_allclose(traj.theta, expected, rtol=0, atol=1e-10)


def test_identical_frequencies_connected_network_synchronize(W_strong):
    rng = np.random.default_rng(3)
    n = W_strong.shape[0]
    p = Params(alpha=1.0, beta=0.0, K=float(n), omega=np.full(n, 2 * np.pi))
    theta0 = rng.uniform(-1.0, 1.0, n)
    traj = simulate(W_strong, p, t_end=30.0, dt=0.01, theta0=theta0, save_every=100)
    assert order_parameter(traj.theta_final) > 0.999


def test_local_coherence_vectorized_matches_definition(W_er):
    rng = np.random.default_rng(5)
    theta = rng.uniform(0, 2 * np.pi, W_er.shape[0])
    r = local_coherence(theta, W_er)
    k = W_er.sum(axis=1)
    direct = np.array(
        [
            (W_er[i] * np.cos(theta - theta[i])).sum() / k[i] if k[i] > 0 else 1.0
            for i in range(len(theta))
        ]
    )
    np.testing.assert_allclose(r, direct, atol=1e-12)


def test_kuramoto_rhs_matches_explicit_sum(W_er):
    rng = np.random.default_rng(6)
    n = W_er.shape[0]
    theta = rng.uniform(0, 2 * np.pi, n)
    omega = rng.normal(0, 1, n)
    K = 3.0
    _, dth = Dynamics(W_er, Params(alpha=1.0, beta=0.0, K=K, omega=omega)).rhs(
        np.zeros(n), theta
    )
    kbar = W_er.sum(axis=1).mean()
    direct = omega + K / kbar * (W_er * np.sin(theta[None, :] - theta[:, None])).sum(1)
    np.testing.assert_allclose(dth, direct, atol=1e-12)


def test_desynchronization_creates_deficit(W_er):
    rng = np.random.default_rng(8)
    n = W_er.shape[0]
    p = Params(alpha=1.0, beta=0.0, gamma=0.5, omega=np.zeros(n))
    traj = simulate(
        W_er,
        p,
        t_end=10.0,
        dt=0.01,
        theta0=rng.uniform(0, 2 * np.pi, n),
        save_every=100,
    )
    assert traj.x_final.mean() > 0.05
    synced = simulate(W_er, p, t_end=10.0, dt=0.01, save_every=100)
    np.testing.assert_allclose(synced.x_final, 0.0, atol=1e-14)


def test_K_gamma_zero_reproduce_linear_model(W_er):
    rho = perron_root(W_er)
    base = Params(alpha=1.0, beta=0.8 / rho, D0=0.3)
    x0 = np.linspace(0, 1, W_er.shape[0])
    a = simulate(W_er, base, t_end=3.0, dt=0.01, x0=x0)
    b = simulate(
        W_er, base.with_(omega=np.ones(W_er.shape[0])), t_end=3.0, dt=0.01, x0=x0
    )
    np.testing.assert_array_equal(a.x, b.x)
