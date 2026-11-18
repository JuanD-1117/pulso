"""Fase 6: modelos predictivos, versionado y predicción sobre una red fija."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pulso.config import load_config
from pulso.experiments import NODE_FEATURES, run_sweep, sample_network_pool
from pulso.ml.predict import explain, make_context, predict_risk, verify
from pulso.ml.registry import load_model
from pulso.ml.train import train_all
from pulso.model import Params, make_thresholds
from pulso.networks import generate


@pytest.fixture(scope="module")
def sweep_cfg() -> dict:
    c = load_config("sweep.yaml")
    c["pool"] = {**c["pool"], "n_networks": 6, "n": [15, 25]}
    c["scenarios"] = {**c["scenarios"], "n": 80, "buffer": [0.1, 1.0]}
    c["simulation"] = {**c["simulation"], "dt": 0.05, "t_end": 4.0, "burn_in": 1.0}
    return c


@pytest.fixture(scope="module")
def train_cfg() -> dict:
    c = load_config("train.yaml")
    c["n_splits"] = 2
    c["xgboost"] = {**c["xgboost"], "n_estimators": 20, "n_jobs": 1}
    c["xgboost_nodes"] = {**c["xgboost_nodes"], "n_estimators": 20, "n_jobs": 1}
    c["gcn"] = {**c["gcn"], "epochs": 3, "hidden": 8}
    c["timing"] = {"n_predictions": 2}
    return c


@pytest.fixture(scope="module")
def trained(tmp_path_factory, sweep_cfg, train_cfg):
    root = tmp_path_factory.mktemp("ml")
    run_sweep(sweep_cfg, out_dir=root / "data", n_jobs=1)
    report = train_all(train_cfg, data_dir=root / "data", models_dir=root / "models")
    return root, report


def _comparable(report: dict) -> dict:
    return {
        m: {k: v for k, v in d.items() if not k.startswith("ms_")}
        for m, d in report.items()
    }


@pytest.mark.slow
def test_same_seed_same_metrics(trained, train_cfg, tmp_path):
    root, report = trained
    again = train_all(train_cfg, data_dir=root / "data", models_dir=tmp_path / "models")
    assert _comparable(again) == _comparable(report)


@pytest.mark.slow
def test_models_versioned_with_metrics(trained, train_cfg):
    root, _ = trained
    for name in ("logistic_collapse", "xgb_collapse", "xgb_nodes", "gcn_nodes"):
        _, meta, metrics = load_model(root / "models", name)
        assert meta["version"] == train_cfg["version"]
        d = root / "models" / name / train_cfg["version"]
        assert json.loads((d / "metrics.json").read_text()) == metrics
    assert (root / "models" / "comparison.csv").is_file()


@pytest.mark.slow
def test_saved_models_predict_on_fixed_network(trained, sweep_cfg):
    root, _ = trained
    net_cfg = sample_network_pool(sweep_cfg, 99)[0]
    W = generate(net_cfg)
    rng = np.random.default_rng(0)
    n = W.shape[0]
    p = Params(
        alpha=1.0,
        beta=0.5,
        K=1.0,
        gamma=0.02,
        omega=rng.normal(6, 0.3, n),
        thresholds=make_thresholds(W, 0.5, 0.3, rng),
        loss=1.0,
    )
    sim = {**sweep_cfg["simulation"]}
    ctx = make_context(
        W, p, net_cfg, buffer=0.5, loss=1.0, shock_size=3.0, sim=sim, seed=1
    )
    risk = predict_risk(ctx, root / "models", omega_width=0.3)
    assert risk.shape[0] == n
    for col in ("cascada_gcn", "cascada_xgb", "p_colapso"):
        assert risk[col].between(0, 1).all(), col
    ex = explain(ctx, 0, root / "models")
    assert len(ex) == len(NODE_FEATURES) and np.isfinite(ex["shap"]).all()
    assert 0.0 <= verify(ctx, 0) <= 1.0


@pytest.mark.slow
def test_pinn_recovers_known_parameters_within_tolerance():
    from pulso.ml.pinn import run_from_config

    cfg = load_config("pinn.yaml")
    res = run_from_config(cfg)
    for name, err in res["relative_error"].items():
        assert err < cfg["tolerance"], (name, res["estimate"][name], res["true"][name])
