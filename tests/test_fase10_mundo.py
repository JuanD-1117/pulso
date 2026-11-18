"""Fase 10: mundo sintético multicapa (Tarea 6, subproyecto 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pulso.config import load_config
from pulso.schema import assert_no_truth
from pulso.world import (
    BANK,
    FIRM,
    LAYER_C,
    LAYER_N,
    LAYER_S,
    LAYER_T,
    PERSON,
    World,
    crossing_times,
    emit_transactions,
    free_phases,
    generate_world,
    scaled_config,
)
from pulso.world.entities import calibrate_intercept


@pytest.fixture(scope="module")
def world_cfg() -> dict:
    return load_config("world.yaml")


@pytest.fixture(scope="module")
def world(world_cfg) -> World:
    return generate_world(world_cfg)


def test_generate_world_same_seed_bitwise_identical(world_cfg):
    a, b = generate_world(world_cfg, seed=11), generate_world(world_cfg, seed=11)
    pd.testing.assert_frame_equal(a.entities, b.entities)
    pd.testing.assert_frame_equal(a.edges, b.edges)
    c = generate_world(world_cfg, seed=12)
    assert not a.entities["income"].equals(c.entities["income"])


def test_every_formal_person_has_exactly_one_employer(world):
    e, edges = world.entities, world.edges
    formal = e[(e["kind"] == PERSON) & (e["formal"] == 1)]["entity_id"]
    payroll = edges[edges["layer"] == LAYER_N]
    counts = payroll.groupby("dst").size()
    assert set(counts.index) == set(formal)
    assert (counts == 1).all()
    informal = e[(e["kind"] == PERSON) & (e["formal"] == 0)]
    assert (informal["employer"] == -1).all()


def test_layers_respect_entity_kinds_and_have_no_self_loops(world):
    kind = world.kind
    edges = world.edges
    assert (edges["src"] != edges["dst"]).all()
    for layer, src_kind, dst_kind in (
        (LAYER_N, FIRM, PERSON),
        (LAYER_S, FIRM, FIRM),
        (LAYER_T, PERSON, PERSON),
    ):
        e = edges[edges["layer"] == layer]
        assert len(e) > 0
        assert (kind[e["src"]] == src_kind).all()
        assert (kind[e["dst"]] == dst_kind).all()
    assert not edges.duplicated(["src", "dst", "layer"]).any()


def test_matrix_convention_column_pays_row(world):
    W = world.matrix("N")
    row = world.edges[world.edges["layer"] == LAYER_N].iloc[0]
    assert W[int(row["dst"]), int(row["src"])] == pytest.approx(row["weight"])
    total = world.matrix()
    assert total.sum() == pytest.approx(world.edges["weight"].sum())
    with pytest.raises(ValueError, match="Capas desconocidas"):
        world.matrix("X")


def test_payroll_equals_salary_and_firm_size_matches(world):
    e = world.entities.set_index("entity_id")
    payroll = world.edges[world.edges["layer"] == LAYER_N]
    np.testing.assert_allclose(
        payroll["weight"].to_numpy(), e.loc[payroll["dst"], "income"].to_numpy()
    )
    sizes = payroll.groupby("src").size()
    firms = e[e["kind"] == FIRM]
    np.testing.assert_array_equal(
        firms["n_employees"].to_numpy(), sizes.reindex(firms.index, fill_value=0)
    )


@pytest.mark.parametrize("kind,key", [(FIRM, "firm"), (PERSON, "person")])
def test_mean_truth_pd_matches_target(world, world_cfg, kind, key):
    pd_true = world.entities.loc[world.kind == kind, "truth_pd"]
    assert pd_true.mean() == pytest.approx(
        world_cfg["truth"]["target_pd"][key], rel=1e-6
    )


def test_calibrate_intercept_recovers_known_value():
    score = np.random.default_rng(0).normal(size=5000)
    b = calibrate_intercept(score, 0.2)
    assert np.mean(1 / (1 + np.exp(-(b + score)))) == pytest.approx(0.2, abs=1e-9)
    with pytest.raises(ValueError):
        calibrate_intercept(score, 1.5)


def test_past_due_reveals_latent_risk(world):
    e = world.entities
    rho = np.corrcoef(e["past_due_12m"], e["truth_latent"])[0, 1]
    assert rho > 0.2


def test_formal_persons_inherit_employer_rhythm(world):
    e = world.entities
    p = e[(e["kind"] == PERSON) & (e["formal"] == 1)]
    emp_omega = e.loc[p["employer"], "truth_omega"].to_numpy()
    rel = np.abs(p["truth_omega"].to_numpy() / emp_omega - 1)
    assert np.median(rel) < 0.1
    informal = e[(e["kind"] == PERSON) & (e["formal"] == 0)]
    spread_inf = informal["truth_omega"].std() / informal["truth_omega"].mean()
    assert spread_inf > 0.1


def test_crossing_times_of_uniform_rotation():
    t = np.linspace(0, 3, 3001)
    theta = np.column_stack([2 * np.pi * t, 4 * np.pi * t + 0.1])
    c = crossing_times(t, theta)
    np.testing.assert_allclose(c[0], [1.0, 2.0, 3.0], atol=1e-9)
    assert len(c[1]) == 6


def test_transaction_flows_match_edge_weights(world):
    months = 6.0
    t = np.linspace(0, months, 1201)
    tx = emit_transactions(
        world, t, free_phases(world, t), np.random.default_rng(1), window=(0, months)
    )
    net = tx[tx["dst"] != BANK]
    flow = net.groupby(["src", "dst"])["amount"].sum() / months
    w = world.edges.set_index(["src", "dst"])["weight"]
    ratio = flow / w.reindex(flow.index)
    # Ventana finita: un ciclo de más o de menos pesa ~1/6 en flujos mensuales.
    assert ratio.median() == pytest.approx(1.0, abs=0.05)
    assert (tx["dst"] == BANK).sum() > 0
    assert set(tx.loc[tx["dst"] == BANK, "layer"]) == {LAYER_C}


def test_number_of_payment_cycles_matches_rhythm(world):
    months = 12.0
    t = np.linspace(0, months, 2401)
    tx = emit_transactions(world, t, free_phases(world, t), np.random.default_rng(2))
    cycles = tx[tx["dst"] == BANK].groupby("src").size()
    expected = world.entities["truth_omega"] * months / (2 * np.pi)
    err = (cycles - expected.reindex(cycles.index)).abs()
    assert (err <= 1.0 + 1e-9).mean() > 0.97  # jitter mueve un ciclo en los bordes


def test_defaulted_entities_stop_paying(world):
    t = np.linspace(0, 6, 1201)
    dt = np.full(world.n, np.nan)
    dt[0] = 2.0
    tx = emit_transactions(
        world, t, free_phases(world, t), np.random.default_rng(3), default_time=dt
    )
    assert tx.loc[tx["src"] == 0, "t"].max() < 2.0


def test_truth_columns_are_vetoed_for_models(world):
    with pytest.raises(ValueError, match="verdad del terreno"):
        assert_no_truth(world.entities.columns)
    assert_no_truth(["income", "debt_ratio"])


def test_save_load_roundtrip(world, tmp_path):
    world.save(tmp_path / "w")
    back = World.load(tmp_path / "w")
    pd.testing.assert_frame_equal(back.entities, world.entities)
    pd.testing.assert_frame_equal(back.edges, world.edges)
    with pytest.raises(FileNotFoundError):
        World.load(tmp_path / "nada")


def test_invalid_config_fails_early(world_cfg):
    bad = scaled_config(world_cfg, 100, 10)
    bad["persons"]["informal_frac"] = 1.5
    with pytest.raises(ValueError, match="informal_frac"):
        generate_world(bad)
    bad = scaled_config(world_cfg, 100, 10)
    del bad["truth"]
    with pytest.raises(KeyError, match="truth"):
        generate_world(bad)


def test_small_world_for_app(world_cfg):
    w = generate_world(scaled_config(world_cfg, 150, 12), seed=3)
    assert w.n == 162
    assert w.is_firm.sum() == 12
