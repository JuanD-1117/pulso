"""Las tres capas de relaciones de pago entre clientes.

- ``N`` nómina F → P: cada persona formal recibe su salario de su único empleador.
- ``S`` proveedores F → F: el comprador paga a sus proveedores; el sector del
  proveedor sale de una matriz insumo-producto aleatoria y, dentro del sector, el
  proveedor se elige ∝ tamaño (firmas grandes concentran clientes).
- ``T`` transferencias P → P: dispersas, con homofilia por lugar de trabajo y por
  comunidad (modelo de bloques estocástico).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finnet.world.schema import FIRM, LAYER_N, LAYER_S, LAYER_T, PERSON


def _payroll(ent: pd.DataFrame) -> pd.DataFrame:
    p = ent[(ent["kind"] == PERSON) & (ent["employer"] >= 0)]
    return pd.DataFrame(
        {
            "src": p["employer"].to_numpy(),
            "dst": p["entity_id"].to_numpy(),
            "layer": LAYER_N,
            "weight": p["income"].to_numpy(),
        }
    )


def _suppliers(ent: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    fc, sc = cfg["firms"], cfg["layers"]["S"]
    f = ent[ent["kind"] == FIRM]
    ids = f["entity_id"].to_numpy()
    sector = f["sector"].to_numpy()
    size = f["n_employees"].to_numpy() + 1.0
    n_sec = int(fc["n_sectors"])
    io = rng.dirichlet(np.full(n_sec, fc["io_concentration"]), size=n_sec)

    rows = []
    for b_pos, buyer in enumerate(ids):
        n_sup = rng.poisson(sc["mean_suppliers"])
        chosen: set[int] = set()
        for _ in range(n_sup):
            sec = rng.choice(n_sec, p=io[sector[b_pos]])
            cand = np.flatnonzero((sector == sec) & (ids != buyer))
            if cand.size == 0:
                continue
            w = size[cand] / size[cand].sum()
            chosen.add(int(ids[rng.choice(cand, p=w)]))
        revenue = f["income"].iat[b_pos]
        for sup in sorted(chosen):
            frac = rng.lognormal(
                np.log(sc["amount_frac"]["median"]), sc["amount_frac"]["sigma"]
            )
            rows.append((int(buyer), sup, LAYER_S, frac * revenue))
    return pd.DataFrame(rows, columns=["src", "dst", "layer", "weight"])


def _transfers(ent: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    tc = cfg["layers"]["T"]
    p = ent[ent["kind"] == PERSON]
    ids = p["entity_id"].to_numpy()
    emp = p["employer"].to_numpy()
    com = p["truth_community"].to_numpy()
    income = p["income"].to_numpy()
    by_emp = {e: ids[emp == e] for e in np.unique(emp[emp >= 0])}
    by_com = {c: ids[com == c] for c in np.unique(com)}

    rows = []
    seen: set[tuple[int, int]] = set()
    n_out = rng.poisson(tc["mean_out_degree"], len(ids))
    for pos, payer in enumerate(ids):
        for _ in range(n_out[pos]):
            u = rng.random()
            if u < tc["coworker_frac"] and emp[pos] >= 0:
                pool = by_emp[emp[pos]]
            elif u < tc["coworker_frac"] + tc["community_frac"]:
                pool = by_com[com[pos]]
            else:
                pool = ids
            if pool.size < 2:
                pool = ids
            dst = int(rng.choice(pool))
            if dst == payer or (payer, dst) in seen:
                continue
            seen.add((int(payer), dst))
            frac = rng.lognormal(
                np.log(tc["amount_frac"]["median"]), tc["amount_frac"]["sigma"]
            )
            rows.append((int(payer), dst, LAYER_T, frac * income[pos]))
    return pd.DataFrame(rows, columns=["src", "dst", "layer", "weight"])


def generate_edges(
    ent: pd.DataFrame, cfg: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Tabla de aristas (``src`` paga ``weight`` SMMLV/mes a ``dst`` en ``layer``)."""
    edges = pd.concat(
        [_payroll(ent), _suppliers(ent, cfg, rng), _transfers(ent, cfg, rng)],
        ignore_index=True,
    )
    return edges.sort_values(["layer", "src", "dst"], ignore_index=True)
