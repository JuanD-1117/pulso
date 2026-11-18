"""Validación de tablas: columnas, tipos y rangos.

Un esquema es ``{columna: (dtype, mínimo, máximo)}``; ``None`` = sin cota. Las tablas
que se escriben a disco o entran a un modelo pasan siempre por aquí.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Schema = dict[str, tuple[str, float | None, float | None]]

# Prefijo de las columnas de verdad del terreno de los mundos sintéticos. Ningún
# modelo puede verlas: con datos reales no existirían.
TRUTH_PREFIX = "truth_"


def validate_schema(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Valida columnas, tipos y rangos; devuelve el DataFrame con tipos exactos.

    Raises
    ------
    ValueError si falta una columna, sobra una, o un valor cae fuera de rango.
    """
    missing = set(schema) - set(df.columns)
    extra = set(df.columns) - set(schema)
    if missing or extra:
        raise ValueError(
            f"Esquema inválido. Faltan: {sorted(missing)}; sobran: {sorted(extra)}"
        )
    out = df[list(schema)].copy()
    tol = 1e-9
    for col, (dtype, lo, hi) in schema.items():
        out[col] = out[col].astype(dtype)
        vals = out[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if lo is not None and (finite < lo - tol).any():
            raise ValueError(f"{col}: valores < {lo} (mín {finite.min()})")
        if hi is not None and (finite > hi + tol).any():
            raise ValueError(f"{col}: valores > {hi} (máx {finite.max()})")
        if np.isinf(vals).any():
            raise ValueError(f"{col}: contiene infinitos; use un tope explícito")
    return out


def assert_no_truth(columns) -> None:
    """Falla si alguna columna es verdad del terreno (fuga de información)."""
    leaked = [c for c in columns if str(c).startswith(TRUTH_PREFIX)]
    if leaked:
        raise ValueError(
            f"Columnas de verdad del terreno no pueden entrar a un modelo: {leaked}"
        )
