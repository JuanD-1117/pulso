"""Fase 12: estimación desde transacciones, variables por nivel y evaluación."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finnet.config import load_config
from finnet.credit import CreditParams
from finnet.credit.estimate import (
    cycle_stats,
    edges_matrix,
    estimate_edges,
    event_phase,
    global_order,
    local_coherence_series,
    payment_cycles,
    phase_matrix,
)
from finnet.credit.evaluate import (
    cross_validate,
    expected_calibration_error,
    ks_statistic,
    paired_deltas,
)
from finnet.credit.features import DYNAMIC, LEVELS, STATIC, client_features
from finnet.credit.pipeline import modeling_rows, process_world, world_seed
from finnet.model import Dynamics, Params
from finnet.world import (
    emit_transactions,
    free_phases,
    generate_world,
    scaled_config,
)


@pytest.fixture(scope="module")
def credit_cfg() -> dict:
    return load_config("credit.yaml")


@pytest.fixture(scope="module")
def world():
    return generate_world(scaled_config(load_config("world.yaml"), 500, 20), seed=21)


@pytest.fixture(scope="module")
def processed(world, credit_cfg):
    p = CreditParams.from_config(credit_cfg, "red")
    return process_world(world, p, seed=21, tx_cfg=credit_cfg["transactions"])


def test_event_phase_is_exact_for_regular_cycles():
    times = 0.25 + np.arange(6) * 0.5
    grid = np.linspace(0.3, 2.7, 50)
    th = event_phase(times, grid)
    np.testing.assert_allclose(th, 2 * np.pi * (grid - 0.25) / 0.5, atol=1e-12)


def test_event_phase_vanishes_after_payer_stops():
    times = np.array([0.0, 1.0, 2.0])
    th = event_phase(times, np.array([2.5, 3.4, 3.6, 5.0]))
    assert np.isfinite(th[:2]).all()
    assert np.isnan(th[2:]).all()
    assert np.isnan(event_phase(np.array([1.0]), np.array([1.0]))).all()


def test_local_coherence_in_phase_and_antiphase():
    import scipy.sparse as sp

    W = sp.csr_array(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 1.0, 0]]))
    theta = np.array([[0.0, 0.0, np.pi]])
    r = local_coherence_series(W, theta)[0]
    np.testing.assert_allclose(r, [1.0, 1.0, -1.0], atol=1e-12)
    assert global_order(np.array([[0.0, np.pi]]))[0] == pytest.approx(0.0, abs=1e-12)


def test_estimated_network_and_rhythms_match_truth(world):
    t_obs = 6.0
    t = np.linspace(0, t_obs, 1201)
    tx = emit_transactions(
        world, t, free_phases(world, t), np.random.default_rng(0), window=(0, t_obs)
    )
    est = estimate_edges(tx, t_obs).set_index(["src", "dst"])["weight"]
    true = world.edges.set_index(["src", "dst"])["weight"].reindex(est.index)
    assert (est / true).median() == pytest.approx(1.0, abs=0.05)
    omega_hat, _, stopped = cycle_stats(payment_cycles(tx, world.n), t_obs)
    rel = np.abs(omega_hat / world.entities["truth_omega"].to_numpy() - 1)
    assert np.nanmedian(rel) < 0.05
    assert not stopped.any()


def test_estimated_coherence_tracks_simulated_coherence(world, processed):
    """r̂ desde transacciones ≈ r del modelo (Dynamics.local_coherence)."""
    run, tx = processed.run, processed.transactions
    t_obs = run.params.t_obs
    traj = run.traj
    m = traj.t < t_obs
    dyn = Dynamics(world.matrix(), Params(alpha=1.0, beta=0.0))
    r_true = np.mean([dyn.local_coherence(th) for th in traj.theta[m][::5]], axis=0)
    W_hat = edges_matrix(estimate_edges(tx, t_obs), world.n)
    grid = np.arange(0.0, t_obs, 0.05)
    r_hat = local_coherence_series(
        W_hat, phase_matrix(payment_cycles(tx, world.n), grid)
    )
    alive = ~(np.isfinite(traj.default_time) & (traj.default_time < t_obs))
    ok = alive & np.isfinite(r_hat).any(axis=0) & (dyn.k > 0)
    r_hat_mean = np.nanmean(r_hat[:, ok], axis=0)
    assert np.corrcoef(r_hat_mean, r_true[ok])[0, 1] > 0.8


def test_client_features_are_finite_and_truth_free(processed):
    tab = processed.table
    cols = set(LEVELS["red_dinamica"])
    assert cols <= set(tab.columns)
    assert not any(c.startswith("truth_") for c in tab.columns)
    assert np.isfinite(tab[list(cols)].to_numpy(float)).all()
    assert len(set(STATIC) & set(DYNAMIC)) == 0


def test_client_features_reject_future_transactions(world, processed):
    late = processed.transactions.copy()
    late.loc[late.index[0], "t"] = 99.0
    with pytest.raises(ValueError, match="fuga"):
        client_features(world.entities, late, 6.0)


def test_income_lost_detects_defaulted_employer(world, credit_cfg):
    from finnet.integrate import Shock

    e = world.entities
    emp = int(e.loc[e["kind"] == 0, "n_employees"].idxmax())
    p = CreditParams.from_config(credit_cfg, "nulo")
    ct = process_world(
        world,
        p,
        seed=3,
        tx_cfg=credit_cfg["transactions"],
        shock=Shock("default", (emp,), t0=1.0),
    )
    tab = ct.table.set_index("entity_id")
    staff = e.index[e["employer"] == emp]
    assert tab.loc[staff, "income_lost_frac"].median() > 0.8
    assert tab["income_lost_frac"].drop(index=staff).median() < 0.05


def test_ks_and_calibration_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    p = np.full(1000, 0.3)
    yy = (np.random.default_rng(0).random(1000) < 0.3).astype(int)
    assert expected_calibration_error(yy, p) < 0.05
    assert expected_calibration_error(yy, np.full(1000, 0.9)) > 0.5


def test_world_seed_is_paired_across_regimes():
    assert world_seed(5, 0, 3) == world_seed(5, 1, 3)
    assert world_seed(5, 0, 3) != world_seed(5, 0, 4)


@pytest.mark.slow
def test_dynamic_features_help_only_when_network_drives_defaults(credit_cfg):
    """Control nulo: sin acople la dinámica no mejora; con acople sí."""
    wcfg = load_config("world.yaml")
    auc_gain = {}
    for regime in ("nulo", "red"):
        frames = []
        for wi in range(6):
            seed = world_seed(9, 0, wi)
            w = generate_world(wcfg, seed=seed)
            p = CreditParams.from_config(credit_cfg, regime)
            ct = process_world(w, p, seed=seed, tx_cfg=credit_cfg["transactions"])
            rows = modeling_rows(ct.table)
            rows.insert(0, "world_id", wi)
            frames.append(rows)
        table = pd.concat(frames, ignore_index=True)
        cv = cross_validate(
            table,
            credit_cfg["models"],
            n_splits=3,
            seed=0,
            levels={k: LEVELS[k] for k in ("clasico", "red_dinamica")},
            models=("logistic",),
        )
        d = paired_deltas(cv, "auc")
        auc_gain[regime] = float(d["delta_mean"].iloc[0])
    # Sin acople, sumar variables de red no mejora (a lo sumo sobreajusta).
    assert auc_gain["nulo"] < 0.005
    assert auc_gain["red"] > 0.01
    assert auc_gain["red"] > auc_gain["nulo"] + 0.01
