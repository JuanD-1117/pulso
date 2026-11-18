"""Motor de experimentos y dataset (Fase 5).

Corre la dinámica completa miles de veces a partir de un YAML y guarda tres tablas
Parquet en ``data/``:

- ``redes``: una fila por red del conjunto (configuración y descriptores).
- ``escenarios``: una fila por corrida; variables medidas **antes** del choque y
  etiquetas medidas después.
- ``nodos``: una fila por entidad de cada red; etiqueta = tamaño de la cascada
  cuando el choque empieza en esa entidad.

Las matrices W se guardan en ``data/networks.npz`` (clave ``net_<id>``) para que
los modelos de grafos (GCN) puedan reconstruir la red.

Reproducibilidad: cada red, escenario y conjunto de nodos tiene su propia semilla
derivada de la semilla global con ``SeedSequence``; la misma semilla produce la
misma fila bit a bit, sin importar el orden ni el número de procesos.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from finnet import metrics, spectral
from finnet.integrate import Shock, simulate
from finnet.model import Params, lorentz_frequencies, make_thresholds
from finnet.networks import NetworkConfig, generate, summary

log = logging.getLogger(__name__)

TOPOLOGY_CODE = {"erdos_renyi": 0, "barabasi_albert": 1, "core_periphery": 2}

# ----------------------------------------------------------------- esquema

# columna -> (tipo, mínimo, máximo); None = sin cota.
SCENARIO_SCHEMA: dict[str, tuple[str, float | None, float | None]] = {
    "scenario_id": ("int64", 0, None),
    "network_id": ("int64", 0, None),
    "seed": ("int64", 0, None),
    "topology": ("int64", 0, 2),
    "n": ("int64", 3, None),
    "density": ("float64", 0, 1),
    "core_frac": ("float64", 0, 1),
    "amount_cv": ("float64", 0, None),
    "rho": ("float64", 0, None),
    "sigma": ("float64", None, None),
    "tau_rec": ("float64", 0, None),
    "lambda2": ("float64", None, None),
    "r_init": ("float64", 0, 1),
    "r_local_mean": ("float64", -1, 1),
    "alpha": ("float64", 0, None),
    "beta": ("float64", 0, None),
    "R": ("float64", 0, None),
    "D0": ("float64", 0, None),
    "K": ("float64", 0, None),
    "gamma": ("float64", 0, None),
    "omega_width": ("float64", 0, None),
    "buffer": ("float64", 0, None),
    "loss": ("float64", 0, None),
    "shock_operational": ("int64", 0, 1),
    "shock_size": ("float64", 0, None),
    "shock_duration": ("float64", 0, None),
    "target": ("int64", 0, None),
    "target_importance": ("float64", 0, None),
    "target_out_share": ("float64", 0, 1),
    "target_in_degree": ("int64", 0, None),
    "target_out_degree": ("int64", 0, None),
    # etiquetas
    "collapse": ("int64", 0, 1),
    "fraction_defaulted": ("float64", 0, 1),
    "unpaid_liquidity": ("float64", 0, None),
    "time_to_first_default": ("float64", 0, None),
}
SCENARIO_LABELS = (
    "collapse",
    "fraction_defaulted",
    "unpaid_liquidity",
    "time_to_first_default",
)
SCENARIO_FEATURES = tuple(
    c
    for c in SCENARIO_SCHEMA
    if c not in SCENARIO_LABELS
    and c not in ("scenario_id", "network_id", "seed", "target")
)

NODE_SCHEMA: dict[str, tuple[str, float | None, float | None]] = {
    "network_id": ("int64", 0, None),
    "entity": ("int64", 0, None),
    "in_degree": ("int64", 0, None),
    "out_degree": ("int64", 0, None),
    "in_amount": ("float64", 0, None),
    "out_amount": ("float64", 0, None),
    "importance": ("float64", 0, None),
    "vulnerability": ("float64", 0, None),
    "pagerank": ("float64", 0, 1),
    "r_local": ("float64", -1, 1),
    "threshold": ("float64", 0, None),
    "out_share": ("float64", 0, 1),
    "shock_to_buffer": ("float64", 0, None),
    "max_dependency": ("float64", 0, 1),
    "downstream_impact": ("float64", 0, None),
    "first_order_defaults": ("float64", 0, 1),
    "debtrank": ("float64", 0, 1),
    # contexto de red
    "n": ("int64", 3, None),
    "topology": ("int64", 0, 2),
    "R": ("float64", 0, None),
    "sigma": ("float64", None, None),
    "K": ("float64", 0, None),
    "gamma": ("float64", 0, None),
    "buffer": ("float64", 0, None),
    "loss": ("float64", 0, None),
    # etiqueta
    "cascade_size": ("float64", 0, 1),
}
NODE_LABEL = "cascade_size"
NODE_FEATURES = tuple(
    c for c in NODE_SCHEMA if c not in (NODE_LABEL, "network_id", "entity")
)


def validate_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
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


# ----------------------------------------------------------------- muestreo


def _scale(u: np.ndarray, rng: list | tuple) -> np.ndarray:
    lo, hi = rng
    return lo + (hi - lo) * u


def critical_R(
    u: np.ndarray, center: float, half_width: float, power: float
) -> np.ndarray:
    """Transforma u ∈ [0,1] en R concentrado alrededor de ``center`` (zona crítica)."""
    s = 2.0 * u - 1.0
    return center + half_width * np.sign(s) * np.abs(s) ** power


def sample_network_pool(cfg: dict, seed: int) -> list[NetworkConfig]:
    """Configuraciones del conjunto de redes por hipercubo latino."""
    pool = cfg["pool"]
    n_nets = pool["n_networks"]
    u = qmc.LatinHypercube(d=5, seed=np.random.default_rng(seed)).random(n_nets)
    seeds = np.random.SeedSequence(seed).generate_state(n_nets)
    topos = pool["topologies"]
    configs = []
    for i in range(n_nets):
        n = round(float(_scale(u[i, 1], pool["n"])))
        k = float(min(_scale(u[i, 2], pool["mean_degree"]), n - 2))
        configs.append(
            NetworkConfig(
                topology=topos[min(int(u[i, 0] * len(topos)), len(topos) - 1)],
                n=n,
                mean_degree=k,
                amount_sigma=float(_scale(u[i, 3], pool["amount_sigma"])),
                core_frac=float(_scale(u[i, 4], pool["core_frac"])),
                p_core=pool["p_core"],
                periphery_share=pool["periphery_share"],
                seed=int(seeds[i]),
            )
        )
    return configs


@dataclass(frozen=True)
class ScenarioSpec:
    """Todo lo necesario para reproducir una corrida."""

    scenario_id: int
    network_id: int
    seed: int
    alpha: float
    R: float
    D0: float
    K: float
    gamma: float
    omega_width: float
    buffer: float
    loss: float
    operational: bool
    shock_size: float
    duration: float
    target_u: float  # se convierte en entidad según el tamaño de la red


def sample_scenarios(cfg: dict, n_networks: int, seed: int) -> list[ScenarioSpec]:
    sc = cfg["scenarios"]
    n = sc["n"]
    u = qmc.LatinHypercube(d=13, seed=np.random.default_rng(seed + 1)).random(n)
    seeds = np.random.SeedSequence(seed + 1).generate_state(n)
    R = critical_R(u[:, 1], sc["R_center"], sc["R_half_width"], sc["R_power"])
    specs = []
    for i in range(n):
        specs.append(
            ScenarioSpec(
                scenario_id=i,
                network_id=min(int(u[i, 0] * n_networks), n_networks - 1),
                seed=int(seeds[i]),
                alpha=float(_scale(u[i, 2], sc["alpha"])),
                R=float(R[i]),
                D0=float(_scale(u[i, 3], sc["D0"])),
                K=float(_scale(u[i, 4], sc["K"])),
                gamma=float(_scale(u[i, 5], sc["gamma"])),
                omega_width=float(_scale(u[i, 6], sc["omega_width"])),
                buffer=float(_scale(u[i, 7], sc["buffer"])),
                loss=float(_scale(u[i, 8], sc["loss"])),
                operational=bool(u[i, 9] < sc["operational_fraction"]),
                shock_size=float(_scale(u[i, 10], sc["shock_size"])),
                duration=float(_scale(u[i, 11], sc["duration"])),
                target_u=float(u[i, 12]),
            )
        )
    return specs


# ----------------------------------------------------------------- física


def build_params(W: np.ndarray, spec: ScenarioSpec, cfg: dict) -> Params:
    """Params de un escenario; toda la aleatoriedad sale de ``spec.seed``."""
    rng = np.random.default_rng(spec.seed)
    n = W.shape[0]
    rho = spectral.perron_root(W)
    omega = rng.permutation(
        lorentz_frequencies(n, cfg["scenarios"]["omega_mean"], spec.omega_width)
    )
    thresholds = make_thresholds(
        W, spec.buffer, cfg["scenarios"]["buffer_dispersion"], rng
    )
    return Params(
        alpha=spec.alpha,
        beta=spec.R * spec.alpha / rho if rho > 0 else 0.0,
        D0=spec.D0,
        K=spec.K,
        gamma=spec.gamma,
        omega=omega,
        thresholds=thresholds,
        loss=spec.loss,
    )


def pre_shock_phases(
    W: np.ndarray, params: Params, burn_in: float, dt: float, seed: int
) -> np.ndarray:
    """Ritmo de pagos antes del choque: θ tras ``burn_in`` ciclos desde fases al azar.

    Como el acople es en un solo sentido (θ afecta a x, no al revés) y antes del
    choque no hay impagos, θ se integra solo, sin déficit ni umbrales.
    """
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0, 2 * np.pi, W.shape[0])
    if burn_in <= 0:
        return theta0
    p = Params(alpha=params.alpha, beta=0.0, K=params.K, omega=params.omega)
    traj = simulate(W, p, t_end=burn_in, dt=dt, theta0=theta0, save_every=10**9)
    return traj.theta_final


def network_features(
    W: np.ndarray, params: Params, net_cfg: NetworkConfig, tau_cap: float
) -> dict[str, float]:
    s = summary(W)
    sigma = spectral.stability(W, params)
    return {
        "topology": TOPOLOGY_CODE[net_cfg.topology],
        "n": s["n"],
        "density": s["density"],
        "core_frac": net_cfg.core_frac if net_cfg.topology == "core_periphery" else 0.0,
        "amount_cv": s["amount_cv"],
        "rho": spectral.perron_root(W),
        "sigma": sigma,
        "tau_rec": min(-1.0 / sigma, tau_cap) if sigma < 0 else tau_cap,
        "lambda2": spectral.algebraic_connectivity(W),
    }


def scenario_features(
    W: np.ndarray,
    params: Params,
    shock: Shock,
    theta_pre: np.ndarray,
    net_cfg: NetworkConfig,
    tau_cap: float,
) -> dict[str, float]:
    """Variables del escenario medidas **antes** del choque.

    Recibe solo la red, los parámetros, la descripción del choque y el estado
    previo (θ tras el periodo de calentamiento): nunca la trayectoria posterior.
    """
    target = shock.targets[0]
    importance = spectral.entity_ranking(W).set_index("entidad")["importancia"]
    out = W.sum(axis=0)
    A = W > 0
    return {
        **network_features(W, params, net_cfg, tau_cap),
        "r_init": float(metrics.order_parameter(theta_pre)),
        "r_local_mean": float(metrics.local_coherence(theta_pre, W).mean()),
        "alpha": params.alpha,
        "beta": params.beta,
        "R": spectral.amplification_index(W, params),
        "D0": params.D0,
        "K": params.K,
        "gamma": params.gamma,
        "shock_operational": int(shock.kind == "operational"),
        "shock_size": shock.size if shock.kind == "liquidity" else 0.0,
        "shock_duration": shock.duration if shock.kind == "operational" else 0.0,
        "target_importance": float(importance.loc[target] * W.shape[0]),
        "target_out_share": float(out[target] / out.sum()) if out.sum() > 0 else 0.0,
        "target_in_degree": int(A[target].sum()),
        "target_out_degree": int(A[:, target].sum()),
    }


def run_scenario_row(
    W: np.ndarray, net_cfg: NetworkConfig, spec: ScenarioSpec, cfg: dict
) -> dict[str, Any]:
    """Una fila de la tabla ``escenarios`` (función pura de sus argumentos)."""
    sim = cfg["simulation"]
    params = build_params(W, spec, cfg)
    n = W.shape[0]
    target = min(int(spec.target_u * n), n - 1)
    shock = Shock(
        kind="operational" if spec.operational else "liquidity",
        targets=(target,),
        size=spec.shock_size,
        t0=sim["t0"],
        duration=spec.duration,
    )
    theta_pre = pre_shock_phases(W, params, sim["burn_in"], sim["dt"], spec.seed)
    feats = scenario_features(W, params, shock, theta_pre, net_cfg, sim["tau_cap"])
    traj = simulate(
        W,
        params,
        t_end=sim["t_end"],
        dt=sim["dt"],
        theta0=theta_pre,
        shock=shock,
        save_every=10**9,
    )
    summ = metrics.risk_summary(traj, W, sim["collapse_fraction"], shock)
    total_flow = W.sum()
    ttfd = summ["time_to_first_default"]
    return {
        "scenario_id": spec.scenario_id,
        "network_id": spec.network_id,
        "seed": spec.seed,
        "target": target,
        "omega_width": spec.omega_width,
        "buffer": spec.buffer,
        "loss": spec.loss,
        **feats,
        "collapse": int(summ["collapse"]),
        "fraction_defaulted": summ["fraction_defaulted"],
        # en "ciclos de flujo total del sistema": comparable entre redes
        "unpaid_liquidity": summ["unpaid_liquidity"] / total_flow
        if total_flow > 0
        else 0.0,
        "time_to_first_default": ttfd - sim["t0"] if np.isfinite(ttfd) else np.nan,
    }


def node_params(W: np.ndarray, net_id: int, cfg: dict, seed: int) -> ScenarioSpec:
    """Parámetros fijos por red para la tabla de nodos (una muestra por red)."""
    sc = cfg["scenarios"]
    rng = np.random.default_rng(seed)
    u = rng.random(9)
    return ScenarioSpec(
        scenario_id=-1,
        network_id=net_id,
        seed=seed,
        alpha=float(_scale(u[0], sc["alpha"])),
        R=float(
            critical_R(u[1:2], sc["R_center"], sc["R_half_width"], sc["R_power"])[0]
        ),
        D0=float(_scale(u[2], sc["D0"])),
        K=float(_scale(u[3], sc["K"])),
        gamma=float(_scale(u[4], sc["gamma"])),
        omega_width=float(_scale(u[5], sc["omega_width"])),
        buffer=float(_scale(u[6], sc["buffer"])),
        loss=float(_scale(u[7], sc["loss"])),
        operational=False,
        shock_size=cfg["nodes"]["shock_size"],
        duration=0.0,
        target_u=0.0,
    )


def first_order_impact(
    W: np.ndarray, thresholds: np.ndarray, loss: float, shock_size: float
) -> dict[str, np.ndarray]:
    """Impacto mecánico inmediato de que cada entidad caiga (sin simular).

    Con ``F[k, i] = W[k, i] / k_k`` (fracción del ingreso de k que paga i) y el salto
    de pérdida del integrador ``λ out_k F[k, i]``:

    - ``shock_to_buffer``: tamaño del retiro sobre el colchón propio (> 1: la
      entidad chocada cae en el acto y dispara los saltos).
    - ``max_dependency``: mayor fracción de ingreso que un receptor obtiene de i.
    - ``downstream_impact``: Σ_k salto_k / colchón_k.
    - ``first_order_defaults``: fracción de entidades que caen solo por ese salto.
    - ``debtrank``: DebtRank con la entidad totalmente estresada.
    """
    from finnet.cascades import debtrank

    n = W.shape[0]
    out = W.sum(axis=0)
    k = W.sum(axis=1)
    out_scale = np.where(out > 0, out, out[out > 0].mean() if (out > 0).any() else 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(k[:, None] > 0, W / np.where(k > 0, k, 1.0)[:, None], 0.0)
    ratio = loss * out_scale[:, None] * F / thresholds[:, None]  # [k, i]
    eye = np.eye(n)
    return {
        "shock_to_buffer": shock_size * out_scale / thresholds,
        "max_dependency": F.max(axis=0),
        "downstream_impact": ratio.sum(axis=0),
        "first_order_defaults": (ratio > 1.0).sum(axis=0) / n,
        "debtrank": np.array([debtrank(W, eye[i])["R"] for i in range(n)]),
    }


def node_features(
    W: np.ndarray,
    params: Params,
    theta_pre: np.ndarray,
    spec: ScenarioSpec,
    net_cfg: NetworkConfig,
) -> pd.DataFrame:
    """Variables por entidad, medidas antes de cualquier choque."""
    import networkx as nx

    from finnet.networks import to_networkx

    n = W.shape[0]
    A = W > 0
    out = W.sum(axis=0)
    rank = spectral.entity_ranking(W).set_index("entidad").sort_index()
    pr = nx.pagerank(to_networkx(W), weight="weight")
    return pd.DataFrame(
        {
            "network_id": spec.network_id,
            "entity": np.arange(n),
            "in_degree": A.sum(axis=1),
            "out_degree": A.sum(axis=0),
            "in_amount": W.sum(axis=1),
            "out_amount": out,
            "importance": rank["importancia"].to_numpy() * n,
            "vulnerability": rank["vulnerabilidad"].to_numpy() * n,
            "pagerank": [pr[i] for i in range(n)],
            "r_local": metrics.local_coherence(theta_pre, W),
            "threshold": params.thresholds,
            "out_share": out / out.sum() if out.sum() > 0 else np.zeros(n),
            **first_order_impact(W, params.thresholds, spec.loss, spec.shock_size),
            "n": n,
            "topology": TOPOLOGY_CODE[net_cfg.topology],
            "R": spectral.amplification_index(W, params),
            "sigma": spectral.stability(W, params),
            "K": params.K,
            "gamma": params.gamma,
            "buffer": spec.buffer,
            "loss": spec.loss,
        }
    )


def run_node_rows(
    W: np.ndarray, net_cfg: NetworkConfig, net_id: int, cfg: dict, seed: int
) -> pd.DataFrame:
    """Tabla de nodos de una red: choca cada entidad con el mismo choque."""
    sim = cfg["simulation"]
    spec = node_params(W, net_id, cfg, seed)
    params = build_params(W, spec, cfg)
    theta_pre = pre_shock_phases(W, params, sim["burn_in"], sim["dt"], seed)
    df = node_features(W, params, theta_pre, spec, net_cfg)
    sizes = np.empty(W.shape[0])
    for i in range(W.shape[0]):
        shock = Shock("liquidity", (i,), size=spec.shock_size, t0=sim["t0"])
        traj = simulate(
            W,
            params,
            t_end=sim["t_end"],
            dt=sim["dt"],
            theta0=theta_pre,
            shock=shock,
            save_every=10**9,
        )
        others = traj.defaulted.copy()
        others[i] = False
        sizes[i] = others.sum() / max(W.shape[0] - 1, 1)
    df["cascade_size"] = sizes
    return df


# ----------------------------------------------------------------- orquestación


def run_sweep(
    cfg: dict,
    out_dir: Path | None = None,
    n_scenarios: int | None = None,
    n_networks: int | None = None,
    n_jobs: int | None = None,
    skip_nodes: bool = False,
    skip_scenarios: bool = False,
) -> dict[str, Path]:
    """Genera redes, escenarios y nodos y los escribe en Parquet.

    ``n_scenarios`` / ``n_networks`` permiten una corrida piloto sin editar el YAML.
    """
    cfg = {**cfg, "pool": {**cfg["pool"]}, "scenarios": {**cfg["scenarios"]}}
    if n_networks is not None:
        cfg["pool"]["n_networks"] = n_networks
    if n_scenarios is not None:
        cfg["scenarios"]["n"] = n_scenarios
    out_dir = Path(out_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = cfg.get("n_jobs", -1) if n_jobs is None else n_jobs
    seed = int(cfg["seed"])

    t0 = time.perf_counter()
    net_cfgs = sample_network_pool(cfg, seed)
    nets = [generate(c) for c in net_cfgs]
    redes = pd.DataFrame(
        [
            {"network_id": i, **c.to_dict(), **summary(W)}
            for i, (c, W) in enumerate(zip(net_cfgs, nets, strict=True))
        ]
    )
    paths = {"redes": out_dir / "redes.parquet", "networks": out_dir / "networks.npz"}
    redes.to_parquet(paths["redes"], index=False)
    np.savez_compressed(
        paths["networks"], **{f"net_{i}": W for i, W in enumerate(nets)}
    )
    log.info("Conjunto de %d redes listo (%.1fs)", len(nets), time.perf_counter() - t0)

    if not skip_scenarios:
        specs = sample_scenarios(cfg, len(nets), seed)
        t1 = time.perf_counter()
        rows = Parallel(n_jobs=jobs, batch_size=16, verbose=0)(
            delayed(run_scenario_row)(
                nets[s.network_id], net_cfgs[s.network_id], s, cfg
            )
            for s in specs
        )
        esc = validate_schema(pd.DataFrame(rows), SCENARIO_SCHEMA)
        paths["escenarios"] = out_dir / "escenarios.parquet"
        esc.to_parquet(paths["escenarios"], index=False)
        log.info(
            "%d escenarios en %.1fs (colapso: %.1f %%)",
            len(esc),
            time.perf_counter() - t1,
            100 * esc["collapse"].mean(),
        )

    if not skip_nodes:
        t2 = time.perf_counter()
        node_seeds = np.random.SeedSequence(seed + 2).generate_state(len(nets))
        frames = Parallel(n_jobs=jobs, verbose=0)(
            delayed(run_node_rows)(W, c, i, cfg, int(node_seeds[i]))
            for i, (c, W) in enumerate(zip(net_cfgs, nets, strict=True))
        )
        nodos = validate_schema(pd.concat(frames, ignore_index=True), NODE_SCHEMA)
        paths["nodos"] = out_dir / "nodos.parquet"
        nodos.to_parquet(paths["nodos"], index=False)
        log.info("%d nodos en %.1fs", len(nodos), time.perf_counter() - t2)
    return paths
