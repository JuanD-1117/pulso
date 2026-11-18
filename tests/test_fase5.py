"""Fase 5: motor de experimentos y dataset."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from finnet.config import load_config
from finnet.experiments import (
    NODE_SCHEMA,
    SCENARIO_SCHEMA,
    critical_R,
    run_node_rows,
    run_scenario_row,
    run_sweep,
    sample_network_pool,
    sample_scenarios,
    scenario_features,
    validate_schema,
)
from finnet.networks import generate


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = load_config("sweep.yaml")
    c["pool"] = {**c["pool"], "n_networks": 4, "n": [20, 30]}
    c["scenarios"] = {**c["scenarios"], "n": 6}
    c["simulation"] = {**c["simulation"], "dt": 0.05, "t_end": 4.0, "burn_in": 1.0}
    return c


def test_same_seed_same_row_bit_for_bit(cfg):
    net_cfg = sample_network_pool(cfg, 1)[0]
    W = generate(net_cfg)
    spec = sample_scenarios(cfg, 1, 1)[0]
    a = pd.DataFrame([run_scenario_row(W, net_cfg, spec, cfg)])
    b = pd.DataFrame([run_scenario_row(W, net_cfg, spec, cfg)])
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_sweep_deterministic_regardless_of_jobs(cfg, tmp_path):
    p1 = run_sweep(cfg, out_dir=tmp_path / "a", n_jobs=1, skip_nodes=True)
    p2 = run_sweep(cfg, out_dir=tmp_path / "b", n_jobs=2, skip_nodes=True)
    a = pd.read_parquet(p1["escenarios"])
    b = pd.read_parquet(p2["escenarios"])
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_features_only_receive_pre_shock_state():
    params = inspect.signature(scenario_features).parameters
    assert list(params) == ["W", "params", "shock", "theta_pre", "net_cfg", "tau_cap"]
    for forbidden in ("traj", "trajectory", "events", "defaulted", "x"):
        assert forbidden not in params


def test_features_independent_of_outcome(cfg):
    """Cambiar solo el horizonte (afecta etiquetas) no cambia las variables."""
    net_cfg = sample_network_pool(cfg, 2)[0]
    W = generate(net_cfg)
    spec = sample_scenarios(cfg, 1, 2)[0]
    short = {**cfg, "simulation": {**cfg["simulation"], "t_end": 1.0}}
    a = run_scenario_row(W, net_cfg, spec, cfg)
    b = run_scenario_row(W, net_cfg, spec, short)
    labels = {
        "collapse",
        "fraction_defaulted",
        "unpaid_liquidity",
        "time_to_first_default",
    }
    for k in a:
        if k not in labels:
            assert a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])), k


def test_schema_validation_rejects_out_of_range():
    df = pd.DataFrame({c: [0.0] for c in NODE_SCHEMA})
    df["n"] = 10.0
    validate_schema(df, NODE_SCHEMA)
    df.loc[0, "cascade_size"] = 1.5
    with pytest.raises(ValueError, match="cascade_size"):
        validate_schema(df, NODE_SCHEMA)
    with pytest.raises(ValueError, match="Faltan"):
        validate_schema(df.drop(columns="pagerank"), NODE_SCHEMA)


def test_sweep_writes_valid_tables(cfg, tmp_path):
    paths = run_sweep(cfg, out_dir=tmp_path, n_jobs=1)
    esc = pd.read_parquet(paths["escenarios"])
    nodos = pd.read_parquet(paths["nodos"])
    validate_schema(esc, SCENARIO_SCHEMA)
    validate_schema(nodos, NODE_SCHEMA)
    assert len(esc) == 6
    assert nodos.groupby("network_id").size().sum() == len(nodos)
    nets = np.load(paths["networks"])
    assert len(nets.files) == 4


def test_node_rows_cascade_in_unit_interval(cfg):
    net_cfg = sample_network_pool(cfg, 3)[0]
    W = generate(net_cfg)
    df = run_node_rows(W, net_cfg, 0, cfg, seed=5)
    assert len(df) == W.shape[0]
    assert df["cascade_size"].between(0, 1).all()


def test_critical_sampling_concentrates_near_center():
    u = np.linspace(0, 1, 10001)
    R = critical_R(u, 1.0, 0.6, 2.0)
    assert R.min() == pytest.approx(0.4) and R.max() == pytest.approx(1.6)
    assert (np.abs(R - 1) < 0.15).mean() > 0.4  # uniforme daría 0.25
