"""De un mundo a su tabla de clientes: simulación, transacciones, variables, etiqueta.

Es la unidad de trabajo del experimento (una por mundo y régimen) y también lo que
usa la app para la cartera de demostración.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pulso.credit.engine import CreditParams, CreditRun, run_credit, seed_streams
from pulso.credit.features import client_features
from pulso.integrate import Shock
from pulso.world import World, emit_transactions


@dataclass
class ClientTable:
    """Resultado de procesar un mundo: corrida, transacciones y variables."""

    run: CreditRun
    transactions: pd.DataFrame  # ventana de observación
    table: pd.DataFrame  # variables + etiqueta, todos los clientes


def process_world(
    world: World,
    params: CreditParams,
    seed: int,
    tx_cfg: dict,
    grid_step: float = 0.05,
    shock: Shock | None = None,
) -> ClientTable:
    """Simula la cartera, registra pagos de ``[0, t_obs)`` y arma las variables.

    La etiqueta ``y`` es el impago en ``[t_obs, t_obs + horizon]``; quien cayó en la
    ventana de observación queda marcado con ``observed_default = 1``.
    """
    run = run_credit(world, params, seed=seed, shock=shock)
    rng = seed_streams(seed)["transactions"]
    traj = run.traj
    tx = emit_transactions(
        world,
        traj.t,
        traj.theta,
        rng,
        jitter_days=float(tx_cfg["jitter_days"]),
        amount_sigma=float(tx_cfg["amount_sigma"]),
        cuota_months=float(tx_cfg["cuota_months"]),
        default_time=traj.default_time,
        window=(0.0, params.t_obs),
    )
    feats = client_features(world.entities, tx, params.t_obs, grid_step=grid_step)
    labels = run.labels()
    table = feats.merge(labels, on="entity_id", validate="one_to_one")
    return ClientTable(run=run, transactions=tx, table=table)


def modeling_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Clientes que llegan vivos al final de la ventana de observación."""
    return table[table["observed_default"] == 0].reset_index(drop=True)


def world_seed(base: int, regime_index: int, world_index: int) -> int:
    """Semilla del mundo ``world_index``; el régimen NO cambia el mundo ni los sorteos.

    ``regime_index`` se ignora a propósito: el mundo nulo y el mundo con red usan el
    mismo mundo y los mismos sorteos, así la comparación entre regímenes es pareada.
    """
    del regime_index
    return int(np.random.SeedSequence([base, world_index]).generate_state(1)[0])
