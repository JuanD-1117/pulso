"""Códigos y esquemas de las tablas del mundo multicapa.

Las mismas tablas servirían para datos reales de un banco: ``entities`` (variables
observables por cliente), ``edges`` (relaciones de pago agregadas por mes) y
``transactions`` (pagos individuales con fecha). Las columnas ``truth_*`` solo existen
en mundos sintéticos.
"""

from __future__ import annotations

from pulso.schema import Schema

FIRM, PERSON = 0, 1
KIND_NAMES = {FIRM: "firma", PERSON: "persona"}

# Capas de la red. ``C`` (cuota al banco) solo aparece en transacciones.
LAYER_N, LAYER_S, LAYER_T, LAYER_C = 0, 1, 2, 3
LAYERS = {"N": LAYER_N, "S": LAYER_S, "T": LAYER_T}
LAYER_NAMES = {
    LAYER_N: "nómina",
    LAYER_S: "proveedores",
    LAYER_T: "transferencias",
    LAYER_C: "cuota",
}
BANK = -1  # destino de las cuotas

ENTITY_SCHEMA: Schema = {
    "entity_id": ("int64", 0, None),
    "kind": ("int64", 0, 1),
    "employer": ("int64", -1, None),  # -1: sin empleador en la red
    "sector": ("int64", -1, None),  # -1: persona
    "income": ("float64", 0, None),  # SMMLV/mes (ventas, si es firma)
    "debt_ratio": ("float64", 0, None),
    "age_years": ("float64", 0, None),
    "tenure_months": ("float64", 0, None),
    "n_products": ("int64", 1, None),
    "past_due_12m": ("int64", 0, None),
    "formal": ("int64", 0, 1),
    "n_employees": ("int64", 0, None),
    # verdad del terreno (no observable)
    "truth_pd": ("float64", 0, 1),
    "truth_latent": ("float64", None, None),
    "truth_omega": ("float64", 0, None),
    "truth_theta0": ("float64", 0, 6.283185307179586),
    "truth_community": ("int64", -1, None),
}
OBSERVABLE_ENTITY_COLUMNS = tuple(
    c for c in ENTITY_SCHEMA if not c.startswith("truth_")
)

EDGE_SCHEMA: Schema = {
    "src": ("int64", 0, None),  # quien paga
    "dst": ("int64", 0, None),  # quien recibe
    "layer": ("int64", 0, 2),
    "weight": ("float64", 0, None),  # SMMLV/mes
}

TX_SCHEMA: Schema = {
    "t": ("float64", None, None),  # meses
    "src": ("int64", 0, None),
    "dst": ("int64", BANK, None),
    "layer": ("int64", 0, 3),
    "amount": ("float64", 0, None),
}
