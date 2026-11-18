"""El mundo multicapa: generación, matrices por capa y persistencia."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from finnet.schema import validate_schema
from finnet.world.entities import generate_entities
from finnet.world.layers import generate_edges
from finnet.world.schema import (
    EDGE_SCHEMA,
    ENTITY_SCHEMA,
    FIRM,
    LAYERS,
    PERSON,
)

_REQUIRED = {
    "persons": ("n", "informal_frac", "income_formal", "income_informal"),
    "firms": ("n", "size_pareto_shape", "payroll_share", "n_sectors"),
    "calendar": ("biweekly_frac", "firm_omega_cv", "formal_omega_cv"),
    "layers": ("T", "S"),
    "truth": ("target_pd", "coef", "latent_sd"),
}


def validate_world_config(cfg: dict) -> None:
    """Revisa claves y rangos antes de generar nada (falla pronto y con contexto)."""
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"Falta la sección '{section}' en la config. del mundo")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"Faltan claves en '{section}': {missing}")
    if cfg["persons"]["n"] < 2 or cfg["firms"]["n"] < 2:
        raise ValueError("Se necesitan al menos 2 personas y 2 firmas")
    if not 0 <= cfg["persons"]["informal_frac"] < 1:
        raise ValueError("persons.informal_frac debe estar en [0, 1)")
    if not 0 <= cfg["calendar"]["biweekly_frac"] <= 1:
        raise ValueError("calendar.biweekly_frac debe estar en [0, 1]")
    T = cfg["layers"]["T"]
    if T["coworker_frac"] + T["community_frac"] > 1:
        raise ValueError("layers.T: coworker_frac + community_frac debe ser <= 1")


@dataclass
class World:
    """Clientes del banco y sus relaciones de pago."""

    entities: pd.DataFrame
    edges: pd.DataFrame
    seed: int
    config: dict = field(repr=False, default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.entities)

    @property
    def kind(self) -> np.ndarray:
        return self.entities["kind"].to_numpy()

    @property
    def is_firm(self) -> np.ndarray:
        return self.kind == FIRM

    @property
    def is_person(self) -> np.ndarray:
        return self.kind == PERSON

    @property
    def income(self) -> np.ndarray:
        """Ingreso total s_i por mes (incluye lo que llega de fuera de la red)."""
        return self.entities["income"].to_numpy(dtype=float)

    def matrix(self, layers: str | Iterable[str] | None = None) -> sp.csr_array:
        """``W[i, j]`` = lo que j le paga a i por mes, sumando las capas pedidas."""
        if layers is None:
            codes = list(LAYERS.values())
        else:
            names = [layers] if isinstance(layers, str) else list(layers)
            unknown = [x for x in names if x not in LAYERS]
            if unknown:
                raise ValueError(f"Capas desconocidas {unknown}; use {list(LAYERS)}")
            codes = [LAYERS[x] for x in names]
        e = self.edges[self.edges["layer"].isin(codes)]
        return sp.csr_array(
            (e["weight"].to_numpy(), (e["dst"].to_numpy(), e["src"].to_numpy())),
            shape=(self.n, self.n),
        )

    def save(self, directory: Path | str) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        validate_schema(self.entities, ENTITY_SCHEMA).to_parquet(d / "entities.parquet")
        validate_schema(self.edges, EDGE_SCHEMA).to_parquet(d / "edges.parquet")
        (d / "world.json").write_text(
            json.dumps({"seed": self.seed, "config": self.config}, indent=2),
            encoding="utf-8",
        )
        return d

    @classmethod
    def load(cls, directory: Path | str) -> World:
        d = Path(directory)
        for name in ("entities.parquet", "edges.parquet", "world.json"):
            if not (d / name).is_file():
                raise FileNotFoundError(f"Falta {d / name}: ¿se generó el mundo?")
        meta = json.loads((d / "world.json").read_text(encoding="utf-8"))
        return cls(
            entities=validate_schema(
                pd.read_parquet(d / "entities.parquet"), ENTITY_SCHEMA
            ),
            edges=validate_schema(pd.read_parquet(d / "edges.parquet"), EDGE_SCHEMA),
            seed=int(meta["seed"]),
            config=meta["config"],
        )


def generate_world(cfg: dict, seed: int | None = None) -> World:
    """Genera un mundo reproducible: misma configuración y semilla, mismo mundo."""
    validate_world_config(cfg)
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(seed)
    ent = validate_schema(generate_entities(cfg, rng), ENTITY_SCHEMA)
    edges = validate_schema(generate_edges(ent, cfg, rng), EDGE_SCHEMA)
    return World(entities=ent, edges=edges, seed=seed, config=cfg)


def scaled_config(cfg: dict, n_persons: int, n_firms: int) -> dict:
    """Copia de la configuración con otro tamaño (p. ej. mundos pequeños en la app)."""
    out = json.loads(json.dumps(cfg))
    out["persons"]["n"] = int(n_persons)
    out["firms"]["n"] = int(n_firms)
    return out
