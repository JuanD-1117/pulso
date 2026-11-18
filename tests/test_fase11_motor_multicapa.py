"""Fase 11: motor multicapa (acoples por capa, impagos programados, ruido)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pulso.config import load_config
from pulso.credit import (
    CreditParams,
    amplification_index,
    build_coupling,
    idiosyncratic_schedule,
    run_credit,
    spectral_radius,
)
from pulso.integrate import Shock, simulate
from pulso.model import Coupling, Params
from pulso.world import generate_world, scaled_config


@pytest.fixture(scope="module")
def credit_cfg() -> dict:
    return load_config("credit.yaml")


@pytest.fixture(scope="module")
def small_world():
    return generate_world(scaled_config(load_config("world.yaml"), 300, 15), seed=4)


def test_sparse_and_dense_networks_give_same_v1_trajectory(W_er):
    n = W_er.shape[0]
    p = Params(alpha=1.0, beta=0.05, K=1.0, gamma=0.2, omega=np.linspace(0, 1, n))
    th0 = np.linspace(0, 3, n)
    a = simulate(W_er, p, 2.0, 0.01, x0=np.ones(n), theta0=th0)
    b = simulate(sp.csr_array(W_er), p, 2.0, 0.01, x0=np.ones(n), theta0=th0)
    np.testing.assert_allclose(a.x, b.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.theta, b.theta, rtol=1e-12, atol=1e-12)


def test_single_layer_coupling_reproduces_v1_model(W_er):
    """B = βW, C = (K/⟨k⟩)W y s = 1 es exactamente el modelo de las Fases 1–3."""
    n = W_er.shape[0]
    beta, K = 0.05, 1.2
    p = Params(alpha=1.0, beta=beta, K=K, gamma=0.3, omega=np.linspace(0, 1, n))
    k_mean = W_er.sum(axis=1).mean()
    cpl = Coupling(stress=beta * W_er, phase=(K / k_mean) * W_er, income=np.ones(n))
    th0 = np.linspace(0, 3, n)
    v1 = simulate(W_er, p, 2.0, 0.01, x0=np.ones(n), theta0=th0)
    v2 = simulate(
        W_er, p.with_(beta=0.0), 2.0, 0.01, x0=np.ones(n), theta0=th0, coupling=cpl
    )
    np.testing.assert_allclose(v2.x, v1.x, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2.theta, v1.theta, rtol=1e-10, atol=1e-12)


def test_coupling_rejects_redistribution_and_bad_income(W_er):
    n = W_er.shape[0]
    with pytest.raises(ValueError, match="s_i"):
        Coupling(W_er, W_er, np.zeros(n))
    cpl = Coupling(W_er, W_er, np.ones(n))
    with pytest.raises(ValueError, match="D0"):
        simulate(W_er, Params(alpha=1, beta=0, D0=0.1), 1.0, 0.1, coupling=cpl)


def test_forced_default_happens_on_schedule_and_stops_payments(W_er):
    n = W_er.shape[0]
    p = Params(alpha=1.0, beta=0.0, thresholds=np.full(n, 1e9), loss=0.5)
    traj = simulate(W_er, p, 2.0, 0.01, forced=(np.array([3]), np.array([0.5])))
    assert traj.default_time[3] == pytest.approx(0.5)
    ev = traj.events.set_index("entity")
    assert ev.loc[3, "cause"] == "idiosincratico"
    assert np.asarray(traj.W_final)[:, 3].sum() == 0
    with pytest.raises(ValueError, match="fuera de rango"):
        simulate(W_er, p, 1.0, 0.1, forced=(np.array([99]), np.array([0.5])))


def test_default_shock_hits_target_at_t0(W_er):
    n = W_er.shape[0]
    p = Params(alpha=1.0, beta=0.0, thresholds=np.full(n, 1e9))
    shock = Shock("default", (2,), t0=0.7)
    traj = simulate(W_er, p, 1.5, 0.01, shock=shock)
    assert traj.default_time[2] == pytest.approx(0.7)
    assert traj.events.set_index("entity").loc[2, "cause"] == "choque"


def test_noise_needs_rng_and_is_reproducible(W_er):
    p = Params(alpha=1.0, beta=0.0)
    with pytest.raises(ValueError, match="rng"):
        simulate(W_er, p, 1.0, 0.1, noise=0.3)
    a = simulate(W_er, p, 1.0, 0.1, noise=0.3, rng=np.random.default_rng(5))
    b = simulate(W_er, p, 1.0, 0.1, noise=0.3, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a.x, b.x)
    # Ornstein–Uhlenbeck: varianza estacionaria σ²/(2α).
    long = simulate(
        sp.csr_array((400, 400)),
        p,
        40.0,
        0.01,
        noise=0.3,
        rng=np.random.default_rng(6),
        save_every=50,
    )
    var = long.x[len(long.t) // 4 :].var()
    assert var == pytest.approx(0.3**2 / 2, rel=0.1)


def test_build_coupling_rows_for_formal_worker(small_world, credit_cfg):
    p = CreditParams.from_config(credit_cfg, "red")
    _, cpl = build_coupling(small_world, p)
    e = small_world.entities
    # Un formal sin transferencias entrantes: todo su ingreso de red es su salario.
    inflow_T = small_world.edges[small_world.edges["layer"] == 2]["dst"].unique()
    cand = e[(e["kind"] == 1) & (e["formal"] == 1) & ~e["entity_id"].isin(inflow_T)]
    i = int(cand["entity_id"].iloc[0])
    emp = int(cand["employer"].iloc[0])
    assert cpl.stress[i, emp] == pytest.approx(p.beta["N"])
    assert cpl.phase[i, emp] == pytest.approx(p.K["N"])
    assert sp.csr_array(cpl.stress)[[i], :].count_nonzero() == 1


def test_amplification_index_matches_dense_eigenvalues(small_world, credit_cfg):
    p = CreditParams.from_config(credit_cfg, "red")
    _, cpl = build_coupling(small_world, p)
    dense = np.abs(np.linalg.eigvals(cpl.stress.toarray())).max()
    assert amplification_index(cpl, p.alpha) == pytest.approx(dense / p.alpha)
    assert spectral_radius(sp.csr_array((5, 5))) == 0.0


def test_idiosyncratic_schedule_matches_annual_pd(small_world):
    rng = np.random.default_rng(0)
    pd_true = small_world.entities["truth_pd"].to_numpy()
    hits = [idiosyncratic_schedule(small_world, 12.0, rng)[0].size for _ in range(200)]
    assert np.mean(hits) == pytest.approx(pd_true.sum(), rel=0.05)


def test_credit_params_regimes_and_validation(credit_cfg):
    null = CreditParams.from_config(credit_cfg, "nulo")
    net = CreditParams.from_config(credit_cfg, "red")
    assert null.gamma == 0 and null.loss == 0 and sum(null.beta.values()) == 0
    assert net.gamma > 0 and null.K == net.K
    with pytest.raises(KeyError, match="Régimen"):
        CreditParams.from_config(credit_cfg, "otro")
    with pytest.raises(ValueError, match="capas"):
        net.with_(beta={"N": 1.0})


def test_null_and_network_share_random_draws(small_world, credit_cfg):
    a = run_credit(small_world, CreditParams.from_config(credit_cfg, "nulo"), seed=3)
    b = run_credit(small_world, CreditParams.from_config(credit_cfg, "red"), seed=3)
    np.testing.assert_array_equal(a.thresholds, b.thresholds)
    np.testing.assert_array_equal(a.forced[0], b.forced[0])
    assert a.R == 0.0 and b.R > 0.0


def test_employer_bankruptcy_spreads_only_with_network_coupling(credit_cfg):
    world = generate_world(scaled_config(load_config("world.yaml"), 600, 10), seed=8)
    e = world.entities
    emp = int(e.loc[e["kind"] == 0, "n_employees"].idxmax())
    staff = e.index[e["employer"] == emp].to_numpy()
    others = e.index[(e["kind"] == 1) & (e["employer"] != emp)].to_numpy()
    shock = Shock("default", (emp,), t0=1.0)

    def after_shock(dt, who):
        return np.mean(np.isfinite(dt[who]) & (dt[who] >= 1.0))

    rates = {}
    for reg in ("nulo", "red"):
        p = CreditParams.from_config(credit_cfg, reg)
        runs = [run_credit(world, p, seed=s, shock=shock, t_end=4.0) for s in range(3)]
        rates[reg] = np.mean([after_shock(r.traj.default_time, staff) for r in runs])
        rates[reg + "_otros"] = np.mean(
            [after_shock(r.traj.default_time, others) for r in runs]
        )
    # Con acople, el despido masivo dispara impagos de la nómina por encima del resto
    # de la cartera; sin acople, la nómina se comporta como cualquier otro cliente.
    assert rates["red"] > rates["red_otros"] + 0.05
    assert rates["red"] > rates["nulo"] + 0.05
    assert abs(rates["nulo"] - rates["nulo_otros"]) < 0.05


def test_labels_split_observation_and_prediction_windows(small_world, credit_cfg):
    run = run_credit(small_world, CreditParams.from_config(credit_cfg, "red"), seed=2)
    lab = run.labels()
    assert not ((lab["observed_default"] == 1) & (lab["y"] == 1)).any()
    dt = lab.loc[lab["y"] == 1, "default_time"]
    assert (dt >= run.params.t_obs).all()
    assert set(lab.loc[lab["y"] == 1, "cause"]) <= {"idiosincratico", "umbral"}
