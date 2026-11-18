"""Mundo sintético multicapa: clientes de un banco y sus relaciones de pago (Tarea 6).

Ver ``docs/diseno/2026-11-18-riesgo-crediticio-multicapa.md``.
"""

from finnet.world.core import (
    World,
    generate_world,
    scaled_config,
    validate_world_config,
)
from finnet.world.schema import (
    BANK,
    EDGE_SCHEMA,
    ENTITY_SCHEMA,
    FIRM,
    KIND_NAMES,
    LAYER_C,
    LAYER_N,
    LAYER_NAMES,
    LAYER_S,
    LAYER_T,
    LAYERS,
    OBSERVABLE_ENTITY_COLUMNS,
    PERSON,
    TX_SCHEMA,
)
from finnet.world.transactions import crossing_times, emit_transactions, free_phases

__all__ = [
    "BANK",
    "EDGE_SCHEMA",
    "ENTITY_SCHEMA",
    "FIRM",
    "KIND_NAMES",
    "LAYERS",
    "LAYER_C",
    "LAYER_N",
    "LAYER_NAMES",
    "LAYER_S",
    "LAYER_T",
    "OBSERVABLE_ENTITY_COLUMNS",
    "PERSON",
    "TX_SCHEMA",
    "World",
    "crossing_times",
    "emit_transactions",
    "free_phases",
    "generate_world",
    "scaled_config",
    "validate_world_config",
]
