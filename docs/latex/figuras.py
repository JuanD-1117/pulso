"""Figuras de la guía y la presentación (Tarea 6 e ilustraciones conceptuales).

Uso (desde la raíz del repositorio):

    PYTHONPATH=src uv run python docs/latex/figuras.py

Escribe PNG en ``docs/latex/figuras/``. Lee los resultados del experimento de
``data/credit/`` (``uv run finnet credit configs/credit.yaml``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from finnet.config import load_config
from finnet.credit import CreditParams, run_credit
from finnet.credit.estimate import (
    edges_matrix,
    estimate_edges,
    event_phase,
    local_coherence_series,
    payment_cycles,
    phase_matrix,
)
from finnet.credit.pipeline import process_world
from finnet.integrate import Shock
from finnet.model import Dynamics, Params
from finnet.world import generate_world, scaled_config

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figuras"
OUT.mkdir(exist_ok=True)

C = {
    "gris": "#9aa5b1",
    "azul": "#2563a8",
    "rojo": "#c2362b",
    "verde": "#2f7d5b",
    "ambar": "#e0a100",
}
LEVELS = {
    "clasico": ("1 · Clásico", C["gris"]),
    "red_estatica": ("2 · + Red estática", C["azul"]),
    "red_dinamica": ("3 · + Red dinámica", C["rojo"]),
}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("->", OUT / name)


def red_multicapa() -> None:
    w = generate_world(scaled_config(load_config("world.yaml"), 160, 10), seed=3)
    W = w.matrix().toarray()
    G = nx.from_numpy_array(0.5 * (W + W.T))
    pos = nx.spring_layout(G, seed=2, weight="weight", iterations=80)
    P = np.array([pos[i] for i in range(w.n)])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    styles = {
        0: (C["azul"], 0.9, "nómina"),
        1: (C["rojo"], 1.6, "proveedores"),
        2: ("#888", 0.5, "transferencias"),
    }
    for layer, (col, lw, lab) in styles.items():
        e = w.edges[w.edges["layer"] == layer]
        for k, (s, d) in enumerate(zip(e["src"], e["dst"], strict=True)):
            ax.plot(
                *zip(P[s], P[d], strict=True),
                color=col,
                lw=lw,
                alpha=0.55,
                label=lab if k == 0 else None,
                zorder=1,
            )
    f, p = w.is_firm, w.is_person
    formal = w.entities["formal"].to_numpy() == 1
    ax.scatter(
        *P[p & formal].T,
        s=18,
        c=C["verde"],
        edgecolor="k",
        lw=0.3,
        label="persona formal",
        zorder=2,
    )
    ax.scatter(
        *P[p & ~formal].T,
        s=18,
        c=C["ambar"],
        edgecolor="k",
        lw=0.3,
        label="persona informal",
        zorder=2,
    )
    ax.scatter(
        *P[f].T,
        s=110,
        marker="s",
        c=C["azul"],
        edgecolor="k",
        lw=0.6,
        label="firma",
        zorder=3,
    )
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=False)
    ax.set_title("Cartera multicapa: 160 personas, 10 firmas")
    save(fig, "tarea6_red_multicapa.png")


def fase_por_eventos() -> None:
    cfg = load_config("credit.yaml")
    w = generate_world(scaled_config(load_config("world.yaml"), 300, 15), seed=5)
    p = CreditParams.from_config(cfg, "red")
    ct = process_world(w, p, seed=5, tx_cfg=cfg["transactions"])
    traj, tx = ct.run.traj, ct.transactions
    t_obs = p.t_obs
    grid = np.arange(0.0, t_obs, 0.02)
    cycles = payment_cycles(tx, w.n)
    e = w.entities
    # Un empleado formal y uno informal con pagadores.
    W_hat = edges_matrix(estimate_edges(tx, t_obs), w.n)
    r_hat = local_coherence_series(W_hat, phase_matrix(cycles, grid))
    m = traj.t < t_obs
    dyn = Dynamics(w.matrix(), Params(alpha=1.0, beta=0.0))
    r_true = np.array([dyn.local_coherence(th) for th in traj.theta[m]])
    alive = ~np.isfinite(traj.default_time)
    ok = alive & np.isfinite(r_hat).all(axis=0)
    formal = ok & (e["formal"].to_numpy() == 1) & w.is_person
    informal = ok & (e["formal"].to_numpy() == 0) & w.is_person & (dyn.k > 0)
    i_f = int(np.flatnonzero(formal)[np.argmax(r_true[:, formal].mean(0))])
    i_i = int(np.flatnonzero(informal)[np.argmin(r_true[:, informal].mean(0))])

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    for col, (i, name) in enumerate(
        ((i_f, "empleado formal"), (i_i, "persona informal"))
    ):
        ax = axes[0, col]
        th = np.mod(np.interp(grid, traj.t, traj.theta[:, i]), 2 * np.pi)
        th_hat = np.mod(event_phase(cycles[i], grid), 2 * np.pi)
        ax.plot(grid, th, color=C["gris"], lw=2.5, label="θ real (no observable)")
        ax.plot(grid, th_hat, color=C["rojo"], lw=1, ls="--", label="θ̂ por eventos")
        ax.vlines(
            cycles[i],
            0,
            2 * np.pi,
            color="k",
            lw=0.5,
            alpha=0.6,
            label="pagos observados",
        )
        ax.set_title(f"Cliente {i} · {name}")
        ax.set_ylabel("fase [rad] mod 2π")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax2 = axes[1, col]
        ax2.plot(traj.t[m], r_true[:, i], color=C["gris"], lw=2.5, label="r real")
        ax2.plot(grid, r_hat[:, i], color=C["rojo"], lw=1, label="r̂ estimado")
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_xlabel("t [meses]")
        ax2.set_ylabel("coherencia local r")
        if col == 0:
            ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle("Del registro de pagos a la fase y la coherencia local")
    fig.tight_layout()
    save(fig, "tarea6_fase_eventos.png")


def resultados() -> None:
    d = ROOT / "data" / "credit"
    s = pd.read_csv(d / "resumen.csv")
    deltas = pd.read_csv(d / "mejoras.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, model, title in zip(
        axes, ("logistic", "xgboost"), ("Regresión logística", "XGBoost"), strict=True
    ):
        g = s[s["model"] == model]
        x = np.arange(2)
        for k, (lv, (lab, col)) in enumerate(LEVELS.items()):
            v = g[g["level"] == lv].set_index("regime").loc[["nulo", "red"]]
            ax.bar(
                x + (k - 1) * 0.26,
                v["auc_mean"],
                0.26,
                yerr=v["auc_std"],
                color=col,
                label=lab,
                capsize=3,
            )
        ax.set_xticks(x, ["Mundo nulo (control)", "Mundo con red"])
        ax.set_ylim(0.70, 0.83)
        ax.set_title(title)
    axes[0].set_ylabel("AUC fuera de muestra")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "tarea6_auc_niveles.png")

    a = deltas[deltas["metric"] == "auc"].copy()
    names = {"logistic": "logística", "xgboost": "XGBoost"}
    a["label"] = [
        f"{'con red' if r == 'red' else 'nulo'} · {names[m]} · "
        f"{LEVELS[lv][0][4:]} vs {LEVELS[b][0][4:]}"
        for r, m, lv, b in zip(
            a["regime"], a["model"], a["level"], a["base"], strict=True
        )
    ]
    a = a.sort_values(["regime", "base", "model"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(a))
    cols = np.where(a["ci_low"] > 0, C["verde"], "#777")
    ax.errorbar(
        a["delta_mean"],
        y,
        xerr=[a["delta_mean"] - a["ci_low"], a["ci_high"] - a["delta_mean"]],
        fmt="none",
        ecolor="k",
        capsize=3,
    )
    ax.scatter(a["delta_mean"], y, c=cols, s=50, zorder=3)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_yticks(y, a["label"], fontsize=8)
    ax.set_xlabel("ΔAUC pareado por partición (IC 95 %)")
    ax.set_title("¿Cuánto mejora cada nivel? (verde: IC por encima de 0)")
    save(fig, "tarea6_mejoras.png")

    b = pd.read_csv(d / "impago_por_r.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    tramos = b["tramo"].unique()
    x = np.arange(len(tramos))
    for k, (reg, col, lab) in enumerate(
        (
            ("nulo", C["gris"], "mundo nulo (control)"),
            ("red", C["rojo"], "mundo con red"),
        )
    ):
        v = b[b["regime"] == reg].set_index("tramo").loc[tramos]
        ax.bar(x + (k - 0.5) * 0.38, 100 * v["rate"], 0.38, color=col, label=lab)
    ax.set_xticks(x, tramos)
    ax.set_xlabel("coherencia local estimada r̂ (tramos)")
    ax.set_ylabel("% impago a 12 meses")
    ax.legend(frameon=False)
    ax.set_title("Impago según sincronía (clientes con ingreso por la red, 20 mundos)")
    save(fig, "tarea6_impago_por_r.png")


def quiebra() -> None:
    cfg = load_config("credit.yaml")
    w = generate_world(load_config("world.yaml"), seed=11)
    e = w.entities
    emp = int(e.loc[e["kind"] == 0, "n_employees"].idxmax())
    staff = e.index[e["employer"] == emp].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    t_grid = np.linspace(0, 7, 141)
    for reg, col, lab in (("red", C["rojo"], "con red"), ("nulo", C["gris"], "nulo")):
        p = CreditParams.from_config(cfg, reg)
        for shocked, ls in ((True, "-"), (False, ":")):
            curves = []
            for s in range(5):
                sh = Shock("default", (emp,), t0=1.0) if shocked else None
                run = run_credit(w, p, seed=s, shock=sh, t_end=7.0)
                dt = run.traj.default_time[staff]
                curves.append([(np.isfinite(dt) & (dt <= t)).mean() for t in t_grid])
            ax.plot(
                t_grid,
                100 * np.mean(curves, 0),
                color=col,
                ls=ls,
                label=f"{lab} · {'quiebra en t = 1' if shocked else 'sin quiebra'}",
            )
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xlabel("t [meses]")
    ax.set_ylabel(f"% de sus {len(staff)} empleados en impago")
    ax.set_title(f"Quiebra del mayor empleador (firma {emp}), media de 5 semillas")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "tarea6_quiebra.png")


def intuicion() -> None:
    """Un cliente: déficit con ruido, colchón e impago; ritmos sincronizados o no."""
    t = np.linspace(0, 12, 1201)
    dt = t[1] - t[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for shift, col, lab in (
        (0.0, C["verde"], "sincronizado (γ(1−r) ≈ 0)"),
        (0.5, C["rojo"], "desincronizado (γ(1−r) = 0.5)"),
    ):
        x = np.zeros_like(t)
        r2 = np.random.default_rng(9)
        for k in range(1, t.size):
            kick = 0.3 * np.sqrt(dt) * r2.standard_normal()
            x[k] = x[k - 1] + dt * (-x[k - 1] + shift) + kick
        ax.plot(t, x, color=col, lw=1, label=lab)
    ax.axhline(1.1, color="k", ls="--", lw=1)
    ax.text(0.2, 1.15, "colchón $x^{(c)}$: si lo cruza, impago", fontsize=8)
    ax.set_xlabel("t [meses]")
    ax.set_ylabel("déficit x (fracción del ingreso)")
    ax.set_title("El mismo cliente, mismo ruido")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    om = 2 * np.pi
    for K, col, lab in (
        (1.2, C["verde"], "acople fuerte: se engancha"),
        (0.2, C["rojo"], "acople débil: deriva"),
    ):
        th_e, th_p = 0.0, 1.0
        dphi = []
        for _ in t:
            th_p += dt * (om * 1.08 + K * np.sin(th_e - th_p))
            th_e += dt * om
            dphi.append(np.cos(th_e - th_p))
        ax.plot(t, dphi, color=col, lw=1.2, label=lab)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("t [meses]")
    ax.set_ylabel("r = cos(θ_empleador − θ_empleado)")
    ax.set_title("Empleado con ritmo 8 % más rápido que su nómina")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "concepto_deficit_y_ritmo.png")


if __name__ == "__main__":
    red_multicapa()
    fase_por_eventos()
    resultados()
    quiebra()
    intuicion()
