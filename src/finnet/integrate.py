"""Integrador RK4 de paso fijo con choques y detección de impagos (Fases 1–4, 6).

Orden de cada paso de tiempo ``t -> t + dt``:

1. Eventos programados en ``t``: inicio de choque, fin de caída operativa e impagos
   idiosincráticos programados (``forced``).
2. Un paso RK4 sobre el estado ``[x, θ]`` con forzamiento ``η`` constante en el paso;
   si ``noise > 0``, se suma el término de ruido de Euler–Maruyama ``σ √dt ξ``.
3. Detección vectorizada de impagos en ``t + dt``; cada impago anula la columna de
   la entidad en ``W`` y aplica el salto de pérdida a sus receptores. Los saltos
   pueden disparar impagos nuevos en el mismo instante, así que se itera hasta que
   no aparece ninguno (cascada instantánea).

Con ``coupling`` (motor multicapa) el déficit es adimensional: los saltos de
pérdida y los choques de liquidez se miden en fracciones del ingreso total ``s_i``.
``W`` puede ser densa o ``scipy.sparse``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from finnet.model import (
    Coupling,
    Dynamics,
    Matrix,
    Params,
    as_matrix,
    col_sums,
    mask_columns,
    row_sums,
)

ShockKind = Literal["liquidity", "operational", "default"]
_EPS_T = 1e-9


@dataclass(frozen=True)
class Shock:
    """Choque externo.

    Attributes
    ----------
    kind : ``"liquidity"`` (retiro de liquidez: pulso η, salto en x),
        ``"operational"`` (la entidad recibe pero no paga durante ``duration``) o
        ``"default"`` (quiebra: la entidad cae en impago en ``t0``).
    targets : entidades chocadas.
    size : tamaño del retiro, en ciclos de pagos salientes de cada entidad chocada
        (en fracciones del ingreso total con el motor multicapa).
    t0 : instante del choque [ciclos].
    duration : duración T de la caída operativa [ciclos].
    """

    kind: ShockKind
    targets: tuple[int, ...]
    size: float = 0.0
    t0: float = 0.0
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("liquidity", "operational", "default"):
            raise ValueError(
                "kind debe ser 'liquidity', 'operational' o 'default', "
                f"no {self.kind!r}"
            )
        if len(self.targets) == 0:
            raise ValueError("El choque necesita al menos una entidad objetivo")
        if self.size < 0 or self.duration < 0 or self.t0 < 0:
            raise ValueError("size, duration y t0 deben ser >= 0")
        object.__setattr__(self, "targets", tuple(int(t) for t in self.targets))


@dataclass
class Trajectory:
    """Resultado de una simulación."""

    t: np.ndarray  # (T,)
    x: np.ndarray  # (T, N)
    theta: np.ndarray  # (T, N)
    unpaid: np.ndarray  # (T,) liquidez no pagada acumulada [pagos medios]
    defaulted: np.ndarray  # (N,) bool, estado final s_i
    default_time: np.ndarray  # (N,) instante de impago (NaN si no cayó)
    events: pd.DataFrame = field(repr=False)
    W_final: Matrix = field(repr=False, default=None)

    @property
    def x_final(self) -> np.ndarray:
        return self.x[-1]

    @property
    def theta_final(self) -> np.ndarray:
        return self.theta[-1]


def rk4_step(dyn: Dynamics, x, theta, source, dt):
    k1x, k1t = dyn.rhs(x, theta, source)
    k2x, k2t = dyn.rhs(x + 0.5 * dt * k1x, theta + 0.5 * dt * k1t, source)
    k3x, k3t = dyn.rhs(x + 0.5 * dt * k2x, theta + 0.5 * dt * k2t, source)
    k4x, k4t = dyn.rhs(x + dt * k3x, theta + dt * k3t, source)
    x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    th_new = theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    return x_new, th_new


def _column_sum(F: Matrix, idx: np.ndarray) -> np.ndarray:
    """Σ_{j ∈ idx} F[:, j] (F en CSC si es dispersa: columnas baratas)."""
    if idx.size == 0:
        return np.zeros(F.shape[0])
    return np.asarray(F[:, idx].sum(axis=1), dtype=float).ravel()


def simulate(
    W: Matrix,
    params: Params,
    t_end: float,
    dt: float,
    x0: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
    shock: Shock | None = None,
    save_every: int = 1,
    coupling: Coupling | None = None,
    forced: tuple[np.ndarray, np.ndarray] | None = None,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """Integra el modelo completo desde ``t = 0`` hasta ``t_end``.

    Parameters
    ----------
    W : red original (n x n), densa o dispersa. No se modifica.
    params : parámetros; ``params.thresholds`` activa los impagos.
    t_end, dt : horizonte y paso [ciclos].
    x0, theta0 : estado inicial (por defecto ceros).
    shock : choque opcional.
    save_every : guarda el estado cada ``save_every`` pasos (más el inicial).
    coupling : acoples multicapa (déficit adimensional); ``None`` = modelo v1.
    forced : ``(entidades, instantes)`` de impagos idiosincráticos programados.
    noise : σ del ruido aditivo en ẋ [1/√ciclo]; requiere ``rng``.
    rng : generador para el ruido (semilla explícita = reproducible).
    """
    W0 = as_matrix(W)
    n = W0.shape[0]
    if dt <= 0 or t_end <= 0:
        raise ValueError("dt y t_end deben ser positivos")
    if save_every < 1:
        raise ValueError("save_every debe ser >= 1")
    if noise < 0:
        raise ValueError("noise debe ser >= 0")
    if noise > 0 and rng is None:
        raise ValueError("noise > 0 requiere un rng con semilla explícita")
    n_steps = round(t_end / dt)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    theta = np.zeros(n) if theta0 is None else np.array(theta0, dtype=float)
    if x.shape != (n,) or theta.shape != (n,):
        raise ValueError(f"x0 y theta0 deben tener forma ({n},)")
    if shock is not None and max(shock.targets) >= n:
        raise ValueError(f"Entidad chocada fuera de rango: {shock.targets} (n={n})")

    thresholds = params.thresholds
    if thresholds is not None:
        thresholds = np.asarray(thresholds, dtype=float)
        if thresholds.shape != (n,):
            raise ValueError(f"thresholds debe tener forma ({n},)")

    if forced is not None:
        f_ent = np.asarray(forced[0], dtype=int)
        f_t = np.asarray(forced[1], dtype=float)
        if f_ent.shape != f_t.shape:
            raise ValueError("forced: entidades e instantes deben tener igual largo")
        if f_ent.size and (f_ent.min() < 0 or f_ent.max() >= n):
            raise ValueError("forced: entidad fuera de rango")
        order = np.argsort(f_t, kind="stable")
        f_ent, f_t = f_ent[order], f_t[order]
    else:
        f_ent, f_t = np.empty(0, dtype=int), np.empty(0)
    f_next = 0

    out0 = col_sums(W0)
    k0 = row_sums(W0)
    if coupling is None:
        out_scale = np.where(
            out0 > 0, out0, out0[out0 > 0].mean() if (out0 > 0).any() else 1.0
        )
        # Fracción del ingreso de k que aporta j: F[k, j] = W0[k, j] / k0[k].
        norm = k0
    else:
        # Déficit adimensional: la escala de cada entidad es su ingreso total.
        out_scale = np.ones(n)
        norm = coupling.income
    inv = np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 0.0)
    if sp.issparse(W0):
        F = sp.csc_array(sp.diags_array(inv) @ W0)
    else:
        F = W0 * inv[:, None]
    loss_scale = params.loss * out_scale  # salto/flujo por perder el 100 % del ingreso

    weight_scale = float(k0.mean()) if k0.mean() > 0 else 1.0

    defaulted = np.zeros(n, dtype=bool)
    down = np.zeros(n, dtype=bool)
    default_time = np.full(n, np.nan)
    events: list[dict] = []
    targets = set(shock.targets) if shock is not None else set()
    # Impagos antes del choque vienen de la fragilidad previa (p. ej. desincronización),
    # no del choque: se registran aparte para no contarlos como contagio.
    t_shock = shock.t0 if shock is not None else np.inf

    def cause_of(i: int, t: float) -> str:
        if i in targets:
            return "choque"
        if shock is None:
            return "umbral"
        return "previo_al_choque" if t < t_shock - _EPS_T else "contagio"

    def build() -> Dynamics:
        paying = ~(defaulted | down)
        return Dynamics(
            mask_columns(W0, paying),
            params,
            weight_scale=weight_scale,
            coupling=None if coupling is None else coupling.masked(paying),
        )

    def outage_source() -> np.ndarray | float:
        if not down.any():
            return 0.0
        return loss_scale * _column_sum(F, np.flatnonzero(down))

    def apply_defaults(idx: np.ndarray, t: float, cause: str | None) -> None:
        defaulted[idx] = True
        default_time[idx] = t
        # Una entidad caída que estaba en caída operativa ya perdía su flujo;
        # el salto del impago reemplaza ese flujo continuo.
        down[idx] = False
        x[:] += loss_scale * _column_sum(F, idx)
        for i in idx:
            events.append(
                {
                    "t": t,
                    "entity": int(i),
                    "event": "impago",
                    "cause": cause if cause is not None else cause_of(int(i), t),
                }
            )

    def detect_defaults(t: float) -> bool:
        """Aplica impagos (y sus saltos) hasta agotar la cascada instantánea."""
        if thresholds is None:
            return False
        changed = False
        while True:
            new = ~defaulted & (x > thresholds)
            if not new.any():
                return changed
            apply_defaults(np.flatnonzero(new), t, None)
            changed = True

    dyn = build()
    source: np.ndarray | float = 0.0
    shock_applied = False
    outage_end = np.inf
    unpaid_acc = 0.0

    n_save = n_steps // save_every + 2
    ts = np.empty(n_save)
    xs = np.empty((n_save, n))
    ths = np.empty((n_save, n))
    unpaid_s = np.empty(n_save)
    ts[0], xs[0], ths[0], unpaid_s[0] = 0.0, x, theta, 0.0
    i_save = 1

    for step in range(n_steps):
        t = step * dt
        rebuild = False
        # Impagos idiosincráticos programados hasta este instante.
        if f_next < f_t.size and f_t[f_next] <= t + _EPS_T:
            stop = np.searchsorted(f_t, t + _EPS_T, side="right")
            due = np.unique(f_ent[f_next:stop])
            f_next = stop
            due = due[~defaulted[due]]
            if due.size:
                apply_defaults(due, t, "idiosincratico")
                detect_defaults(t)
                rebuild = True
        if shock is not None and not shock_applied and t >= shock.t0 - _EPS_T:
            shock_applied = True
            tg = np.array(shock.targets)
            if shock.kind == "liquidity":
                x[tg] += shock.size * out_scale[tg]
                rebuild |= detect_defaults(t)
            elif shock.kind == "default":
                alive = tg[~defaulted[tg]]
                if alive.size:
                    apply_defaults(alive, t, "choque")
                    detect_defaults(t)
                    rebuild = True
            else:
                alive = tg[~defaulted[tg]]
                down[alive] = True
                outage_end = t + shock.duration
                for i in alive:
                    events.append(
                        {"t": t, "entity": int(i), "event": "caida", "cause": "choque"}
                    )
                rebuild = True
        if down.any() and t >= outage_end - _EPS_T:
            for i in np.flatnonzero(down):
                events.append(
                    {
                        "t": t,
                        "entity": int(i),
                        "event": "recuperacion",
                        "cause": "fin_caida",
                    }
                )
            down[:] = False
            rebuild = True
        if rebuild:
            dyn = build()
            source = outage_source()

        x, theta = rk4_step(dyn, x, theta, source, dt)
        if noise > 0:
            x += noise * np.sqrt(dt) * rng.standard_normal(n)
        unpaid_acc += dt * out0[defaulted | down].sum()
        t_new = (step + 1) * dt
        if detect_defaults(t_new):
            dyn = build()
            source = outage_source()

        # El estado final se guarda siempre, caiga o no en la rejilla de guardado.
        if (step + 1) % save_every == 0 or step == n_steps - 1:
            ts[i_save], xs[i_save], ths[i_save], unpaid_s[i_save] = (
                t_new,
                x,
                theta,
                unpaid_acc,
            )
            i_save += 1

    ev = pd.DataFrame(events, columns=["t", "entity", "event", "cause"])
    return Trajectory(
        t=ts[:i_save],
        x=xs[:i_save],
        theta=ths[:i_save],
        unpaid=unpaid_s[:i_save],
        defaulted=defaulted.copy(),
        default_time=default_time,
        events=ev,
        W_final=dyn.W,
    )


def linear_operator(W: np.ndarray, params: Params) -> np.ndarray:
    """M = -αI - D0 L + βW: el operador lineal completo (Fases 1–2)."""
    W = np.asarray(W, dtype=float)
    L = np.diag(W.sum(axis=0)) - W
    return -params.alpha * np.eye(W.shape[0]) - params.D0 * L + params.beta * W


def simulate_scipy(
    W: np.ndarray,
    params: Params,
    x0: np.ndarray,
    t_end: float,
    rtol: float = 1e-10,
) -> np.ndarray:
    """Referencia con ``scipy.integrate.solve_ivp`` (solo parte lineal, sin eventos)."""
    from scipy.integrate import solve_ivp

    M = linear_operator(W, params)
    sol = solve_ivp(
        lambda _t, y: M @ y,
        (0.0, t_end),
        x0,
        method="DOP853",
        rtol=rtol,
        atol=rtol * 1e-2,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp falló: {sol.message}")
    return sol.y[:, -1]
