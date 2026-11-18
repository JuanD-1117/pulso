"""Registro de transacciones a partir de una trayectoria de fases.

Cada vez que la fase de un pagador cruza un múltiplo de 2π, el pagador ejecuta su
ciclo de pagos: paga a cada receptor de sus aristas y paga su cuota al banco. El
registro es lo que vería el banco; las fases y los ritmos naturales no.

- Monto a ``i`` en un cruce de ``j``: ``W_ij · 2π/ω_j · ε`` con ε lognormal de media 1,
  de modo que el flujo medio mensual es ``W_ij``.
- Jitter: todos los pagos de un mismo cruce comparten un desfase N(0, jitter) días.
- Quien cae en impago deja de pagar desde su instante de impago.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finnet.schema import validate_schema
from finnet.world.core import World
from finnet.world.schema import BANK, LAYER_C, TX_SCHEMA

TWO_PI = 2.0 * np.pi


def crossing_times(t: np.ndarray, theta: np.ndarray) -> list[np.ndarray]:
    """Instantes en que cada θ_j(t) cruza un múltiplo de 2π (interpolación lineal).

    ``theta`` tiene forma (T, N). Se usa el máximo acumulado para que la fase sea
    monótona: un retroceso transitorio por el acople no crea cruces repetidos.
    """
    t = np.asarray(t, dtype=float)
    theta = np.maximum.accumulate(np.asarray(theta, dtype=float), axis=0)
    out = []
    for j in range(theta.shape[1]):
        th = theta[:, j]
        first = np.floor(th[0] / TWO_PI) + 1
        last = np.floor(th[-1] / TWO_PI)
        if last < first:
            out.append(np.empty(0))
            continue
        levels = TWO_PI * np.arange(first, last + 1)
        out.append(np.interp(levels, th, t))
    return out


def free_phases(world: World, t: np.ndarray) -> np.ndarray:
    """θ_j(t) = θ_j(0) + ω_j t: ritmos naturales sin acople (forma (T, N))."""
    e = world.entities
    return e["truth_theta0"].to_numpy()[None, :] + np.outer(
        np.asarray(t, float), e["truth_omega"].to_numpy()
    )


def emit_transactions(
    world: World,
    t: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    jitter_days: float = 1.0,
    amount_sigma: float = 0.15,
    cuota_months: float = 24.0,
    default_time: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Tabla de transacciones (esquema :data:`TX_SCHEMA`) ordenada por tiempo.

    Parameters
    ----------
    t, theta : trayectoria de fases, formas (T,) y (T, N).
    jitter_days : desviación del desfase de cada ciclo de pagos [días].
    amount_sigma : dispersión lognormal de los montos.
    cuota_months : la cuota mensual es ``debt_ratio · income / cuota_months``.
    default_time : instante de impago por entidad (NaN = no cae).
    window : solo se registran pagos con ``window[0] <= t < window[1]``.
    """
    n = world.n
    if theta.shape != (len(t), n):
        raise ValueError(f"theta debe tener forma ({len(t)}, {n}); tiene {theta.shape}")
    ent = world.entities
    omega = ent["truth_omega"].to_numpy()
    dt_default = (
        np.full(n, np.inf)
        if default_time is None
        else np.where(np.isfinite(default_time), default_time, np.inf)
    )

    cross = crossing_times(t, theta)
    payer = np.concatenate([np.full(len(c), j) for j, c in enumerate(cross)])
    when = np.concatenate(cross) if payer.size else np.empty(0)
    when = when + rng.normal(0.0, jitter_days / 30.0, when.size)
    keep = when < dt_default[payer]
    if window is not None:
        keep &= (when >= window[0]) & (when < window[1])
    cycles = pd.DataFrame({"src": payer[keep], "t": when[keep]})

    # Pagos a receptores de la red: un renglón por (ciclo, arista saliente).
    to_net = cycles.merge(world.edges, on="src", how="inner")
    period = TWO_PI / omega[to_net["src"].to_numpy()]
    to_net["amount"] = to_net["weight"].to_numpy() * period
    to_net = to_net.drop(columns="weight")

    # Cuota al banco en cada ciclo.
    cuota = (ent["debt_ratio"] * ent["income"] / cuota_months).to_numpy()
    to_bank = cycles.assign(
        dst=BANK,
        layer=LAYER_C,
        amount=cuota[cycles["src"].to_numpy()]
        * TWO_PI
        / omega[cycles["src"].to_numpy()],
    )

    tx = pd.concat([to_net, to_bank], ignore_index=True)
    if len(tx):
        s = amount_sigma
        tx["amount"] = tx["amount"].to_numpy() * rng.lognormal(-0.5 * s * s, s, len(tx))
    tx = tx.sort_values(["t", "src", "dst"], ignore_index=True)
    return validate_schema(tx, TX_SCHEMA)
