r"""PINN para el problema inverso: estimar α, β, γ y K desde trayectorias (Fase 6).

Una red ``t -> (x(t), θ(t))`` se ajusta a las observaciones mientras el residuo de
las ecuaciones del modelo (sin impagos ni redistribución) se anula en puntos de
colocación (Raissi et al., 2019):

.. math::

    \mathcal{L} = \|\hat x - x\|^2 + \|\hat\theta - \theta\|^2
        + w_\text{phys}\,\big(\|\dot{\hat x} - f_x\|^2
        + \|\dot{\hat\theta} - f_\theta\|^2\big)

Los parámetros se aprenden en escala logarítmica para mantenerlos positivos. W y ω
se suponen conocidos (la red de pagos y los ritmos son observables).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PARAM_NAMES = ("alpha", "beta", "gamma", "K")


@dataclass(frozen=True)
class PINNConfig:
    hidden: int
    layers: int
    epochs: int
    lr: float
    physics_weight: float
    n_collocation: int
    seed: int


class _Field(nn.Module):
    """MLP t -> [x, θ] con entrada en [0, 1] y salidas en escala de los datos.

    ``x̂ = x̄ + s_x · NN_x`` y ``θ̂ = θ_0 + ω t + s_θ · NN_θ``: la red solo aprende la
    desviación respecto a la rotación libre, que es lo que el acople modifica. Sin
    esta forma la MLP gasta su capacidad en la deriva lineal ω t y ajusta mal x.
    """

    def __init__(
        self,
        n: int,
        hidden: int,
        layers: int,
        t_end: float,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        theta0: torch.Tensor,
        omega: torch.Tensor,
        th_scale: torch.Tensor,
    ):
        super().__init__()
        dims = [1] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in itertools.pairwise(dims):
            mods += [nn.Linear(a, b), nn.Tanh()]
        mods.append(nn.Linear(hidden, 2 * n))
        self.net = nn.Sequential(*mods)
        self.t_end = t_end
        self.n = n
        for name, val in (
            ("x_mean", x_mean),
            ("x_scale", x_scale),
            ("theta0", theta0),
            ("omega", omega),
            ("th_scale", th_scale),
        ):
            self.register_buffer(name, val)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(t / self.t_end)
        x = self.x_mean + self.x_scale * out[:, : self.n]
        th = self.theta0 + self.omega * t + self.th_scale * out[:, self.n :]
        return x, th


def _rhs(x, th, W, k, kbar, omega, alpha, beta, gamma, K):
    """Lado derecho del modelo (Fases 1 y 3) en torch, vectorizado sobre el tiempo."""
    c, s = torch.cos(th), torch.sin(th)
    Wx = x @ W.T
    zr, zi = c @ W.T, s @ W.T
    dx = -alpha * x + beta * Wx + gamma * (k - (c * zr + s * zi))
    dth = omega + (K / kbar) * (c * zi - s * zr)
    return dx, dth


@torch.enable_grad()
def fit_pinn(
    t: np.ndarray,
    x_obs: np.ndarray,
    theta_obs: np.ndarray,
    W: np.ndarray,
    omega: np.ndarray,
    cfg: PINNConfig,
    init: dict[str, float] | None = None,
) -> dict[str, float]:
    """Ajusta la PINN y devuelve los parámetros estimados (más la pérdida final).

    Parameters
    ----------
    t : instantes de observación (T,).
    x_obs, theta_obs : observaciones (T, N).
    W, omega : red y ritmos naturales conocidos.
    init : valores iniciales de los parámetros (por defecto 1.0 para todos).
    """
    torch.manual_seed(cfg.seed)
    dtype = torch.float64
    n = W.shape[0]
    Wt = torch.tensor(W, dtype=dtype)
    k = Wt.sum(dim=1)
    kbar = k.mean()
    om = torch.tensor(omega, dtype=dtype)
    T = torch.tensor(t, dtype=dtype).reshape(-1, 1)
    X = torch.tensor(x_obs, dtype=dtype)
    TH = torch.tensor(theta_obs, dtype=dtype)
    # Escalas para que x y θ pesen parecido en la pérdida.
    sx = X.std().clamp_min(1e-8)
    sth = TH.std().clamp_min(1e-8)

    free = TH - TH[0] - om * T  # desviación de la rotación libre
    field = _Field(
        n,
        cfg.hidden,
        cfg.layers,
        float(t[-1]),
        X.mean(0),
        X.std(0) + sx * 1e-3,
        TH[0],
        om,
        free.std(0) + 1e-3,
    ).to(dtype)
    init = init or {}
    log_p = nn.Parameter(
        torch.log(torch.tensor([init.get(p, 1.0) for p in PARAM_NAMES], dtype=dtype))
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    def data_loss() -> torch.Tensor:
        xh, thh = field(T)
        return ((xh - X) / sx).pow(2).mean() + ((thh - TH) / sth).pow(2).mean()

    def physics_loss() -> torch.Tensor:
        tc = torch.rand(cfg.n_collocation, 1, generator=gen, dtype=dtype) * float(t[-1])
        # Entrada escalar: un solo jvp da d/dt de todas las salidas a la vez.
        (xc, thc), (dxc, dthc) = torch.func.jvp(field, (tc,), (torch.ones_like(tc),))
        a, b, g, K = torch.exp(log_p)
        fx, fth = _rhs(xc, thc, Wt, k, kbar, om, a, b, g, K)
        return ((dxc - fx) / sx).pow(2).mean() + ((dthc - fth) / sth).pow(2).mean()

    # Etapa 1: el campo aprende los datos (sin física), para que el residuo de la
    # etapa 2 se evalúe sobre trayectorias ya razonables y no arrastre a los
    # parámetros hacia valores espurios.
    opt = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs // 2):
        opt.zero_grad()
        data_loss().backward()
        opt.step()
    # Etapa 2: ajuste conjunto de campo y parámetros.
    opt = torch.optim.Adam([*field.parameters(), log_p], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss = torch.tensor(np.nan)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        loss = data_loss() + cfg.physics_weight * physics_loss()
        loss.backward()
        opt.step()
        sched.step()
    est = dict(
        zip(PARAM_NAMES, torch.exp(log_p).detach().numpy().tolist(), strict=True)
    )
    est["loss"] = float(loss.detach())
    return est


def synthetic_problem(cfg: dict) -> dict:
    """Trayectoria sintética con parámetros conocidos según ``configs/pinn.yaml``."""
    from pulso.integrate import simulate
    from pulso.model import Params
    from pulso.networks import NetworkConfig, generate

    d = cfg["data"]
    W = generate(NetworkConfig(**d["network"]))
    n = W.shape[0]
    rng = np.random.default_rng(cfg["seed"])
    omega = rng.normal(d["omega_mean"], d["omega_std"], n)
    p = Params(**d["params_true"], omega=omega)
    traj = simulate(
        W,
        p,
        t_end=d["t_end"],
        dt=d["dt"],
        x0=d["x0_scale"] * rng.random(n),
        theta0=rng.uniform(0, 2 * np.pi, n),
        save_every=d["save_every"],
    )
    return {
        "W": W,
        "omega": omega,
        "t": traj.t,
        "x": traj.x,
        "theta": traj.theta,
        "true": dict(d["params_true"]),
    }


def run_from_config(cfg: dict) -> dict:
    """Genera el problema, ajusta la PINN y devuelve estimados y errores relativos."""
    prob = synthetic_problem(cfg)
    pc = PINNConfig(**cfg["pinn"], seed=cfg["seed"])
    est = fit_pinn(prob["t"], prob["x"], prob["theta"], prob["W"], prob["omega"], pc)
    rel = {k: abs(est[k] - v) / v for k, v in prob["true"].items()}
    return {
        "estimate": est,
        "true": prob["true"],
        "relative_error": rel,
        "problem": prob,
    }
