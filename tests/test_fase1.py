"""Fase 1: red de pagos y modelo lineal ẋ = (-αI + βW) x."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from finnet.integrate import linear_operator, simulate, simulate_scipy
from finnet.model import Params
from finnet.networks import TOPOLOGIES, NetworkConfig, generate, paid, received
from finnet.spectral import amplification_index, entity_ranking, perron_root


def test_generate_all_topologies_valid_weights_mean_one():
    for topo in TOPOLOGIES:
        cfg = NetworkConfig(
            topology=topo,
            n=60,
            mean_degree=6.0,
            amount_sigma=1.0,
            core_frac=0.1,
            p_core=0.8,
            periphery_share=0.1,
            seed=0,
        )
        W = generate(cfg)
        assert W.shape == (60, 60)
        assert (W >= 0).all()
        assert np.allclose(np.diag(W), 0)
        np.testing.assert_allclose(W[W > 0].mean(), 1.0)
        assert received(W).sum() == pytest.approx(paid(W).sum())


def test_generate_same_seed_identical():
    cfg = NetworkConfig("core_periphery", 50, 5.0, 0.8, 0.1, 0.8, 0.1, seed=9)
    np.testing.assert_array_equal(generate(cfg), generate(cfg))


def test_rk4_matches_matrix_exponential_relative_error_below_1e6(W_any):
    rng = np.random.default_rng(0)
    rho = perron_root(W_any)
    p = Params(alpha=1.0, beta=0.9 / rho)
    x0 = rng.random(W_any.shape[0])
    t_end = 3.0
    traj = simulate(W_any, p, t_end=t_end, dt=0.01, x0=x0, save_every=300)
    exact = expm(linear_operator(W_any, p) * t_end) @ x0
    err = np.linalg.norm(traj.x_final - exact) / np.linalg.norm(exact)
    assert err < 1e-6


def test_rk4_agrees_with_solve_ivp(W_er):
    p = Params(alpha=1.0, beta=0.05)
    x0 = np.linspace(0, 1, W_er.shape[0])
    traj = simulate(W_er, p, t_end=2.0, dt=0.01, x0=x0, save_every=200)
    ref = simulate_scipy(W_er, p, x0, 2.0)
    np.testing.assert_allclose(traj.x_final, ref, rtol=1e-6)


def test_empty_network_decays_exactly_exponential():
    n, alpha, t_end = 10, 0.7, 4.0
    W = np.zeros((n, n))
    x0 = np.arange(1.0, n + 1)
    traj = simulate(W, Params(alpha=alpha, beta=0.5), t_end=t_end, dt=0.01, x0=x0)
    expected = x0[None, :] * np.exp(-alpha * traj.t)[:, None]
    np.testing.assert_allclose(traj.x, expected, rtol=1e-9)


@pytest.mark.parametrize(("R", "grows"), [(0.95, False), (1.05, True)])
def test_threshold_R_controls_growth(W_any, R, grows):
    rho = perron_root(W_any)
    p = Params(alpha=1.0, beta=R / rho)
    assert amplification_index(W_any, p) == pytest.approx(R)
    x0 = np.ones(W_any.shape[0])
    traj = simulate(W_any, p, t_end=60.0, dt=0.01, x0=x0, save_every=100)
    norms = np.linalg.norm(traj.x, axis=1)
    assert (norms[-1] > norms[0]) == grows


def test_nonnegative_initial_state_stays_nonnegative_metzler(W_any):
    rho = perron_root(W_any)
    p = Params(alpha=1.0, beta=1.2 / rho)
    x0 = np.zeros(W_any.shape[0])
    x0[0] = 1.0
    traj = simulate(W_any, p, t_end=10.0, dt=0.01, x0=x0)
    assert traj.x.min() >= -1e-12


def test_entity_ranking_sorted_and_normalized(W_er):
    df = entity_ranking(W_er)
    assert df["importancia"].is_monotonic_decreasing
    assert df["importancia"].sum() == pytest.approx(1.0)
    assert (df["vulnerabilidad"] >= 0).all()
