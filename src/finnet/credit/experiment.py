"""Experimento completo de la Tarea 6: mundos → variables → comparación → modelos.

Salidas:

- ``<out>/clientes.parquet``: una fila por cliente vivo al cierre de la observación,
  con mundo, régimen, variables de los tres niveles y etiqueta.
- ``<out>/cv_metricas.csv``, ``<out>/resumen.csv``, ``<out>/mejoras.csv``.
- ``<models>/<version>/``: modelos finales del régimen con red y ``metrics.json``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from finnet.config import load_config
from finnet.credit.engine import CreditParams
from finnet.credit.evaluate import (
    MODELS,
    cross_validate,
    fit_final,
    paired_deltas,
    summarize,
)
from finnet.credit.features import LEVELS
from finnet.credit.pipeline import modeling_rows, process_world, world_seed
from finnet.schema import assert_no_truth
from finnet.world import generate_world

log = logging.getLogger(__name__)

MODEL_VERSION = "v1"


def _one_world(
    world_cfg: dict, cfg: dict, regime: str, regime_index: int, world_index: int
) -> pd.DataFrame:
    seed = world_seed(int(cfg["seed"]), regime_index, world_index)
    world = generate_world(world_cfg, seed=seed)
    params = CreditParams.from_config(cfg, regime)
    ct = process_world(
        world,
        params,
        seed=seed,
        tx_cfg=cfg["transactions"],
        grid_step=float(cfg["experiment"]["phase_grid_step"]),
    )
    rows = modeling_rows(ct.table)
    rows.insert(0, "world_id", world_index)
    rows.insert(1, "regime", regime)
    rows["R"] = ct.run.R
    return rows


def build_dataset(
    cfg: dict, n_worlds: int | None = None, n_jobs: int | None = None
) -> pd.DataFrame:
    """Tabla de clientes de todos los mundos y regímenes (pareados por mundo)."""
    world_cfg = load_config(cfg["world_config"])
    exp = cfg["experiment"]
    n_worlds = int(exp["worlds_per_regime"] if n_worlds is None else n_worlds)
    n_jobs = int(exp["n_jobs"] if n_jobs is None else n_jobs)
    tasks = [
        (reg, ri, wi) for ri, reg in enumerate(cfg["regimes"]) for wi in range(n_worlds)
    ]
    t0 = time.time()
    frames = Parallel(n_jobs=n_jobs)(
        delayed(_one_world)(world_cfg, cfg, reg, ri, wi) for reg, ri, wi in tasks
    )
    table = pd.concat(frames, ignore_index=True)
    log.info(
        "%d mundos x %d regímenes -> %d clientes en %.1f s",
        n_worlds,
        len(cfg["regimes"]),
        len(table),
        time.time() - t0,
    )
    assert_no_truth(table.columns)
    return table


def evaluate_regimes(table: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    exp = cfg["experiment"]
    cvs, sums, deltas = [], [], []
    for regime, g in table.groupby("regime", sort=False):
        cv = cross_validate(
            g.reset_index(drop=True),
            cfg["models"],
            int(exp["n_splits"]),
            int(cfg["seed"]),
        )
        cv.insert(0, "regime", regime)
        s = summarize(cv.drop(columns="regime"))
        s.insert(0, "regime", regime)
        rows = []
        for metric in ("auc", "pr_auc", "brier"):
            for base, compare in (
                ("clasico", None),
                # La tesis: ¿la dinámica aporta sobre lo que ya da la red estática?
                ("red_estatica", ("red_dinamica",)),
            ):
                d = paired_deltas(
                    cv.drop(columns="regime"), metric=metric, base=base, compare=compare
                )
                d.insert(0, "regime", regime)
                rows.append(d)
        cvs.append(cv)
        sums.append(s)
        deltas.append(pd.concat(rows, ignore_index=True))
        log.info("Régimen %s evaluado", regime)
    return {
        "cv": pd.concat(cvs, ignore_index=True),
        "summary": pd.concat(sums, ignore_index=True),
        "deltas": pd.concat(deltas, ignore_index=True),
    }


def save_final_models(
    table: pd.DataFrame, cfg: dict, results: dict, models_dir: Path
) -> Path:
    """Entrena con todos los mundos del régimen con red y versiona los modelos."""
    d = Path(models_dir) / MODEL_VERSION
    d.mkdir(parents=True, exist_ok=True)
    net = table[table["regime"] == "red"].reset_index(drop=True)
    if net.empty:
        raise ValueError("No hay filas del régimen 'red' para entrenar modelos finales")
    for level in LEVELS:
        for name in MODELS:
            model = fit_final(net, level, name, cfg["models"], int(cfg["seed"]))
            joblib.dump(model, d / f"{level}__{name}.joblib")
    summary = results["summary"]
    meta = {
        "version": MODEL_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trained_on": "regime=red",
        "n_rows": len(net),
        "n_worlds": int(net["world_id"].nunique()),
        "base_rate": float(net["y"].mean()),
        "levels": {k: list(v) for k, v in LEVELS.items()},
        "models": list(MODELS),
        "cv_summary": json.loads(summary.to_json(orient="records")),
        "config": cfg,
    }
    (d / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def run_experiment(
    cfg: dict,
    out_dir: Path | str = Path("data/credit"),
    models_dir: Path | str = Path("models/credit"),
    n_worlds: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = build_dataset(cfg, n_worlds=n_worlds, n_jobs=n_jobs)
    paths = {"clientes": out / "clientes.parquet"}
    table.to_parquet(paths["clientes"])
    results = evaluate_regimes(table, cfg)
    for key, name in (
        ("cv", "cv_metricas"),
        ("summary", "resumen"),
        ("deltas", "mejoras"),
    ):
        paths[name] = out / f"{name}.csv"
        results[key].to_csv(paths[name], index=False)
    paths["modelos"] = save_final_models(table, cfg, results, Path(models_dir))
    best = results["deltas"]
    for _, r in best[(best["metric"] == "auc")].iterrows():
        log.info(
            "%-6s %-9s %-13s vs %-13s ΔAUC = %+.4f [%+.4f, %+.4f]",
            r["regime"],
            r["model"],
            r["level"],
            r["base"],
            r["delta_mean"],
            r["ci_low"],
            r["ci_high"],
        )
    return paths


def load_results(out_dir: Path | str = Path("data/credit")) -> dict[str, pd.DataFrame]:
    """Lee las tablas del experimento (para la app y los notebooks)."""
    d = Path(out_dir)
    files = {"summary": "resumen.csv", "deltas": "mejoras.csv", "cv": "cv_metricas.csv"}
    missing = [f for f in files.values() if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {missing} en {d}. "
            "Ejecute: uv run finnet credit configs/credit.yaml"
        )
    return {k: pd.read_csv(d / f) for k, f in files.items()}


def load_final_model(
    level: str, name: str, models_dir: Path | str = Path("models/credit")
):
    path = Path(models_dir) / MODEL_VERSION / f"{level}__{name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute: uv run finnet credit configs/credit.yaml"
        )
    return joblib.load(path)


def predicted_pd(model, table: pd.DataFrame, level: str) -> np.ndarray:
    return model.predict_proba(table[list(LEVELS[level])].to_numpy(dtype=float))[:, 1]
