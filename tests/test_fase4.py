"""Fase 4: impagos, cascadas y modelos de comparación."""

from __future__ import annotations

import numpy as np
import pytest

from pulso.cascades import debtrank, eisenberg_noe, payment_obligations, watts
from pulso.integrate import Shock, simulate
from pulso.metrics import risk_summary, systemic_importance
from pulso.model import Params, make_thresholds
from pulso.spectral import perron_root


def test_eisenberg_noe_three_entities_hand_solved():
    # Ciclo A -> B -> C -> A, cada uno debe 10. e = (0, 5, 2).
    # p_A = min(10, p_C), p_B = min(10, 5 + p_A), p_C = min(10, 2 + p_B).
    # p = (10, 10, 10) lo cumple y es el mayor punto fijo: todos pagan completo.
    p_bar = np.array([10.0, 10.0, 10.0])
    Pi = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    e = np.array([0.0, 5.0, 2.0])
    p = eisenberg_noe(p_bar, Pi, e)
    np.testing.assert_allclose(p, [10, 10, 10])

    # Caso con impago: A debe 10 a B y nada más entra a A; B debe 8 a C.
    # e = (4, 0, 0): p_A = 4, p_B = min(8, 0 + 4) = 4, p_C = 0 (no debe nada).
    p_bar = np.array([10.0, 8.0, 0.0])
    Pi = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    e = np.array([4.0, 0.0, 0.0])
    p = eisenberg_noe(p_bar, Pi, e)
    np.testing.assert_allclose(p, [4.0, 4.0, 0.0], atol=1e-12)
    assert np.max(np.abs(p - np.minimum(p_bar, e + Pi.T @ p))) < 1e-10


def test_eisenberg_noe_fixed_point_on_random_network(W_er):
    p_bar, Pi = payment_obligations(W_er)
    e = 0.3 * p_bar
    p = eisenberg_noe(p_bar, Pi, e)
    assert np.max(np.abs(p - np.minimum(p_bar, e + Pi.T @ p))) < 1e-10


def test_infinite_thresholds_identical_to_phase3(W_er):
    n = W_er.shape[0]
    rng = np.random.default_rng(0)
    base = Params(
        alpha=1.0,
        beta=0.5 / perron_root(W_er),
        D0=0.2,
        K=2.0,
        gamma=0.3,
        omega=rng.normal(6, 0.5, n),
        loss=0.5,
    )
    theta0 = rng.uniform(0, 2 * np.pi, n)
    shock = Shock("liquidity", (0,), size=3.0, t0=0.5)
    a = simulate(W_er, base, 5.0, 0.01, theta0=theta0, shock=shock)
    b = simulate(
        W_er,
        base.with_(thresholds=np.full(n, np.inf)),
        5.0,
        0.01,
        theta0=theta0,
        shock=shock,
    )
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.theta, b.theta)
    assert not b.defaulted.any()


def test_no_shock_stable_no_defaults(W_any):
    n = W_any.shape[0]
    rng = np.random.default_rng(1)
    p = Params(
        alpha=1.0,
        beta=0.9 / perron_root(W_any),
        loss=1.0,
        thresholds=make_thresholds(W_any, 0.5, 0.2, rng),
    )
    traj = simulate(W_any, p, 20.0, 0.01)
    assert not traj.defaulted.any()
    assert traj.events.empty
    assert n == traj.x.shape[1]


def test_defaults_monotone_and_columns_zeroed(W_er):
    rng = np.random.default_rng(2)
    p = Params(
        alpha=0.5,
        beta=1.1 / perron_root(W_er) * 0.5,
        loss=1.0,
        thresholds=make_thresholds(W_er, 0.4, 0.3, rng),
    )
    shock = Shock("liquidity", (int(np.argmax(W_er.sum(0))),), size=5.0, t0=0.0)
    traj = simulate(W_er, p, 20.0, 0.01, shock=shock, save_every=1)
    assert traj.defaulted.sum() > 1
    # Cada entidad aparece a lo sumo una vez como impago y nunca "se recupera".
    imp = traj.events[traj.events.event == "impago"]
    assert imp.entity.is_unique
    assert set(imp.entity) == set(np.flatnonzero(traj.defaulted))
    assert (traj.W_final[:, traj.defaulted] == 0).all()
    assert (np.diff(traj.unpaid) >= -1e-15).all()


def test_operational_outage_recovers_if_below_threshold(W_er):
    n = W_er.shape[0]
    p = Params(alpha=1.0, beta=0.0, loss=0.1, thresholds=np.full(n, 1e6))
    shock = Shock("operational", (0,), t0=1.0, duration=2.0)
    traj = simulate(W_er, p, 6.0, 0.01, shock=shock)
    ev = traj.events
    assert list(ev.event) == ["caida", "recuperacion"]
    assert ev.t.iloc[1] - ev.t.iloc[0] == pytest.approx(2.0)
    np.testing.assert_allclose(traj.W_final, W_er)
    # La liquidez no pagada es el flujo de 0 por la duración de la caída.
    assert traj.unpaid[-1] == pytest.approx(W_er[:, 0].sum() * 2.0, rel=1e-6)


def test_risk_summary_fields(W_er):
    rng = np.random.default_rng(4)
    p = Params(
        alpha=1.0, beta=0.0, loss=1.0, thresholds=make_thresholds(W_er, 1.0, 0.1, rng)
    )
    shock = Shock("liquidity", (3,), size=5.0)
    traj = simulate(W_er, p, 5.0, 0.01, shock=shock)
    s = risk_summary(traj, W_er, collapse_fraction=0.3, shock=shock)
    assert s["n_defaulted"] >= 1 and s["time_to_first_default"] == 0.0
    assert 0 <= s["fraction_defaulted"] <= 1


def test_debtrank_star_network():
    # Entidad 0 paga todo el ingreso de 1..4: si 0 cae (h=1), 1..4 quedan con h=1.
    W = np.zeros((5, 5))
    W[1:, 0] = 1.0
    res = debtrank(W, h0=np.eye(5)[0])
    np.testing.assert_allclose(res["h"], 1.0)
    assert res["R"] == pytest.approx(1.0)  # 0 no recibe nada: v_0 = 0


def test_watts_chain_cascade():
    # Cadena 0 -> 1 -> 2 -> 3 con umbral 0.5: todo cae desde la semilla 0.
    W = np.zeros((4, 4))
    for i in range(3):
        W[i + 1, i] = 1.0
    assert watts(W, 0.5, [0]).all()
    assert watts(W, 1.1, [0]).sum() == 1


def test_systemic_importance_shape_and_range(W_er):
    rng = np.random.default_rng(5)
    small = W_er[:15, :15]
    p = Params(
        alpha=1.0, beta=0.0, loss=1.0, thresholds=make_thresholds(small, 0.5, 0.1, rng)
    )
    imp = systemic_importance(
        small, p, shock_size=3.0, n_seeds=1, dt=0.05, t_end=3.0, n_jobs=1
    )
    assert imp.shape == (15,)
    assert ((imp >= 0) & (imp <= 1)).all()


def test_defaults_before_shock_not_counted_as_contagion(W_er):
    n = W_er.shape[0]
    # Umbral nulo para la entidad 5: cae en el primer paso, antes del choque en t0=1.
    th = np.full(n, 1e6)
    th[5] = 0.0
    p = Params(
        alpha=1.0,
        beta=0.0,
        gamma=0.5,
        omega=np.linspace(0, 3, n),
        loss=0.0,
        thresholds=th,
    )
    shock = Shock("liquidity", (0,), size=1.0, t0=1.0)
    traj = simulate(W_er, p, 3.0, 0.01, theta0=np.linspace(0, 6, n), shock=shock)
    ev = traj.events.set_index("entity")
    assert ev.loc[5, "cause"] == "previo_al_choque"
    s = risk_summary(traj, W_er, collapse_fraction=0.3, shock=shock)
    assert s["n_defaulted_before_shock"] == 1
    assert np.isnan(s["time_to_first_contagion"])
