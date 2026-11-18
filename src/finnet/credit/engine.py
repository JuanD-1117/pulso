"""Motor multicapa sobre la cartera de un banco: del mundo a la trayectoria.

Construye los acoples por capa, los colchones a partir de la PD verdadera y el
calendario de impagos idiosincráticos, e integra con :func:`finnet.integrate.simulate`.

Cada mundo usa cuatro flujos aleatorios independientes derivados de su semilla
(colchones, impagos idiosincráticos, ruido, transacciones). Así el mundo nulo y el
mundo con red comparten exactamente los mismos sorteos y la comparación es pareada:
lo único que cambia es el acople.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigs

from finnet.integrate import Shock, Trajectory, simulate
from finnet.model import Coupling, Params, row_sums
from finnet.world import LAYERS, World

_LAYER_KEYS = tuple(LAYERS)


@dataclass(frozen=True)
class CreditParams:
    """Parámetros de la dinámica de crédito (ver ``configs/credit.yaml``)."""

    alpha: float
    beta: dict[str, float]
    K: dict[str, float]
    gamma: float
    loss: float
    noise: float
    buffer: float
    pd_ref: float
    kappa: float
    dispersion: float
    dt: float
    t_obs: float
    horizon: float
    save_every: int

    def __post_init__(self) -> None:
        for name in ("beta", "K"):
            d = getattr(self, name)
            if set(d) != set(_LAYER_KEYS):
                raise ValueError(f"{name} necesita exactamente las capas {_LAYER_KEYS}")
            if any(v < 0 for v in d.values()):
                raise ValueError(f"{name}: los acoples deben ser >= 0")
        for name in ("alpha", "gamma", "loss", "noise", "buffer", "pd_ref"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} debe ser >= 0")
        if self.t_obs <= 0 or self.horizon <= 0 or self.dt <= 0:
            raise ValueError("t_obs, horizon y dt deben ser positivos")

    @property
    def t_end(self) -> float:
        return self.t_obs + self.horizon

    @classmethod
    def from_config(cls, cfg: dict, regime: str | None = None) -> CreditParams:
        dyn = dict(cfg["dynamics"])
        if regime is not None:
            regimes = cfg.get("regimes", {})
            if regime not in regimes:
                raise KeyError(f"Régimen desconocido {regime!r}; hay {list(regimes)}")
            dyn.update(regimes[regime] or {})
        th, sim = cfg["thresholds"], cfg["simulation"]
        return cls(
            alpha=float(dyn["alpha"]),
            beta={k: float(v) for k, v in dyn["beta"].items()},
            K={k: float(v) for k, v in dyn["K"].items()},
            gamma=float(dyn["gamma"]),
            loss=float(dyn["loss"]),
            noise=float(dyn["noise"]),
            buffer=float(th["buffer"]),
            pd_ref=float(th["pd_ref"]),
            kappa=float(th["kappa"]),
            dispersion=float(th["dispersion"]),
            dt=float(sim["dt"]),
            t_obs=float(sim["t_obs"]),
            horizon=float(sim["horizon"]),
            save_every=int(sim["save_every"]),
        )

    def with_(self, **changes) -> CreditParams:
        return replace(self, **changes)


def _row_scale(M: sp.csr_array, v: np.ndarray) -> sp.csr_array:
    return sp.csr_array(sp.diags_array(v) @ M)


def build_coupling(world: World, p: CreditParams) -> tuple[sp.csr_array, Coupling]:
    """``W`` total (dinero) y acoples ``B = Σ β_ℓ W^ℓ/s``, ``C = Σ K_ℓ W^ℓ/k``."""
    s = world.income
    W = world.matrix()
    k = row_sums(W)
    inv_s = 1.0 / s
    inv_k = np.where(k > 0, 1.0 / np.where(k > 0, k, 1.0), 0.0)
    B = sp.csr_array((world.n, world.n))
    C = sp.csr_array((world.n, world.n))
    for name in _LAYER_KEYS:
        Wl = world.matrix(name)
        if p.beta[name] > 0:
            B = B + p.beta[name] * _row_scale(Wl, inv_s)
        if p.K[name] > 0:
            C = C + p.K[name] * _row_scale(Wl, inv_k)
    return W, Coupling(stress=B, phase=C, income=s)


def spectral_radius(M: sp.csr_array) -> float:
    """ρ(M) de una matriz no negativa (raíz de Perron)."""
    if M.count_nonzero() == 0:
        return 0.0
    if M.shape[0] <= 400:
        return float(np.abs(np.linalg.eigvals(M.toarray())).max())
    try:
        vals = eigs(M.astype(float), k=1, which="LM", return_eigenvectors=False)
        return float(np.abs(vals).max())
    except (ArpackNoConvergence, ArpackError):
        return float(np.abs(np.linalg.eigvals(M.toarray())).max())


def amplification_index(coupling: Coupling, alpha: float) -> float:
    """R = ρ(B)/α: con R < 1 la cartera amortigua el déficit; con R > 1 lo amplifica."""
    return spectral_radius(sp.csr_array(coupling.stress)) / alpha


def credit_thresholds(
    world: World, p: CreditParams, rng: np.random.Generator
) -> np.ndarray:
    """Colchón ``x_c = buffer · (pd_ref / PD)^κ · ε``: más PD, menos colchón."""
    pd_true = world.entities["truth_pd"].to_numpy()
    d = p.dispersion
    eps = rng.lognormal(-0.5 * d * d, d, world.n)
    return p.buffer * (p.pd_ref / pd_true) ** p.kappa * eps


def idiosyncratic_schedule(
    world: World, t_end: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Impagos propios: τ_i ~ Exp(λ_i) con λ_i = -ln(1 - PD_i)/12 por mes."""
    pd_true = world.entities["truth_pd"].to_numpy()
    rate = -np.log1p(-pd_true) / 12.0
    tau = rng.exponential(1.0 / rate)
    hit = np.flatnonzero(tau < t_end)
    return hit, tau[hit]


@dataclass
class CreditRun:
    """Una corrida de la cartera: trayectoria y lo necesario para auditarla."""

    world: World
    params: CreditParams
    seed: int
    traj: Trajectory
    thresholds: np.ndarray
    forced: tuple[np.ndarray, np.ndarray]
    R: float

    def labels(self) -> pd.DataFrame:
        """Etiqueta por cliente: impago en la ventana de predicción.

        ``observed_default`` marca a quien ya cayó en la ventana de observación: no
        tiene sentido predecir su impago y se excluye del modelo.
        """
        dt = self.traj.default_time
        t_obs, t_end = self.params.t_obs, self.params.t_end
        cause = pd.Series("", index=range(self.world.n), dtype=object)
        ev = self.traj.events
        imp = ev[ev["event"] == "impago"].drop_duplicates("entity")
        cause.loc[imp["entity"].to_numpy()] = imp["cause"].to_numpy()
        observed = np.isfinite(dt) & (dt < t_obs)
        y = np.isfinite(dt) & (dt >= t_obs) & (dt <= t_end + 1e-9)
        return pd.DataFrame(
            {
                "entity_id": np.arange(self.world.n),
                "observed_default": observed.astype(int),
                "y": y.astype(int),
                "default_time": dt,
                "cause": cause.to_numpy(),
            }
        )


def seed_streams(seed: int) -> dict[str, np.random.Generator]:
    names = ("thresholds", "idiosyncratic", "noise", "transactions")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(c) for n, c in zip(names, children, strict=True)}


def run_credit(
    world: World,
    p: CreditParams,
    seed: int,
    shock: Shock | None = None,
    t_end: float | None = None,
) -> CreditRun:
    """Integra la cartera de ``t = 0`` a ``t_end`` (por defecto obs + horizonte)."""
    rngs = seed_streams(seed)
    t_end = p.t_end if t_end is None else t_end
    W, coupling = build_coupling(world, p)
    th = credit_thresholds(world, p, rngs["thresholds"])
    forced = idiosyncratic_schedule(world, t_end, rngs["idiosyncratic"])
    ent = world.entities
    params = Params(
        alpha=p.alpha,
        beta=0.0,  # el acople de déficit va en coupling.stress
        gamma=p.gamma,
        omega=ent["truth_omega"].to_numpy(),
        thresholds=th,
        loss=p.loss,
    )
    traj = simulate(
        W,
        params,
        t_end=t_end,
        dt=p.dt,
        theta0=ent["truth_theta0"].to_numpy(),
        shock=shock,
        save_every=p.save_every,
        coupling=coupling,
        forced=forced,
        noise=p.noise,
        rng=rngs["noise"],
    )
    return CreditRun(
        world=world,
        params=p,
        seed=seed,
        traj=traj,
        thresholds=th,
        forced=forced,
        R=amplification_index(coupling, p.alpha),
    )
