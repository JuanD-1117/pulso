"""Variables por cliente en tres niveles: clásico, red estática y red dinámica.

Todas salen de lo observable: variables del cliente y registro de transacciones de
la ventana de observación ``[0, t_obs)``.

- **clásico**: lo que usa cualquier score (ingreso, endeudamiento, mora, ...).
- **red estática**: agregados de la red estimada ``Ŵ`` (de quién y cuánto recibe,
  concentración, PageRank, salud del empleador y de los pagadores). Es lo que ya
  explora la literatura de crédito en redes.
- **red dinámica**: el aporte de la Tarea 6. Coherencia local r̂_i(t) con los
  pagadores, desfase, desajuste de ritmo, regularidad de los ciclos y pagadores
  que dejaron de pagar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from finnet.credit.estimate import (
    cycle_stats,
    edges_matrix,
    estimate_edges,
    local_coherence_series,
    payment_cycles,
    phase_lag_series,
    phase_matrix,
)
from finnet.model import col_sums, row_sums
from finnet.schema import assert_no_truth
from finnet.world import LAYER_N, LAYER_S, LAYER_T, OBSERVABLE_ENTITY_COLUMNS, PERSON

TWO_PI = 2.0 * np.pi

CLASSIC = (
    "is_person",
    "log_income",
    "debt_ratio",
    "age_years",
    "tenure_months",
    "n_products",
    "past_due_12m",
    "formal",
    "log_employees",
)
STATIC = (
    "net_income_share",
    "in_share_N",
    "in_share_S",
    "in_share_T",
    "out_share_S",
    "out_share_T",
    "n_payers",
    "n_payees",
    "income_hhi",
    "pagerank",
    "employer_log_size",
    "employer_debt_ratio",
    "employer_past_due",
    "payer_past_due",
    "payer_debt_ratio",
)
DYNAMIC = (
    "r_mean",
    "r_std",
    "r_low_frac",
    "phase_lag_abs",
    "freq_mismatch",
    "period_cv",
    "payer_period_cv",
    "income_lost_frac",
)
LEVELS: dict[str, tuple[str, ...]] = {
    "clasico": CLASSIC,
    "red_estatica": CLASSIC + STATIC,
    "red_dinamica": CLASSIC + STATIC + DYNAMIC,
}
LEVEL_LABELS = {
    "clasico": "1 · Clásico",
    "red_estatica": "2 · + Red estática",
    "red_dinamica": "3 · + Red dinámica",
}
FEATURE_LABELS = {
    "is_person": "es persona",
    "log_income": "log ingreso",
    "debt_ratio": "endeudamiento",
    "age_years": "edad",
    "tenure_months": "antigüedad en el banco",
    "n_products": "n.º de productos",
    "past_due_12m": "moras en 12 meses",
    "formal": "empleo formal",
    "log_employees": "log empleados",
    "net_income_share": "ingreso que llega por la red",
    "in_share_N": "ingreso por nómina",
    "in_share_S": "ingreso por proveedores",
    "in_share_T": "ingreso por transferencias",
    "out_share_S": "pagos a proveedores",
    "out_share_T": "transferencias enviadas",
    "n_payers": "n.º de pagadores",
    "n_payees": "n.º de receptores",
    "income_hhi": "concentración del ingreso",
    "pagerank": "PageRank",
    "employer_log_size": "tamaño del empleador",
    "employer_debt_ratio": "apalancamiento del empleador",
    "employer_past_due": "moras del empleador",
    "payer_past_due": "moras de los pagadores",
    "payer_debt_ratio": "endeudamiento de los pagadores",
    "r_mean": "coherencia local media r̂",
    "r_std": "variabilidad de r̂",
    "r_low_frac": "tiempo desincronizado (r̂ < 0.5)",
    "phase_lag_abs": "desfase con los pagadores",
    "freq_mismatch": "desajuste de ritmo",
    "period_cv": "irregularidad de sus ciclos",
    "payer_period_cv": "irregularidad de sus pagadores",
    "income_lost_frac": "ingreso de pagadores que dejaron de pagar",
}


def pagerank(W: sp.csr_array, damping: float = 0.85, tol: float = 1e-10) -> np.ndarray:
    """PageRank del flujo de dinero (j reparte su importancia según lo que paga)."""
    n = W.shape[0]
    out = col_sums(W)
    inv = np.where(out > 0, 1.0 / np.where(out > 0, out, 1.0), 0.0)
    P = sp.csr_array(W @ sp.diags_array(inv))  # columnas estocásticas
    dangling = out == 0
    pr = np.full(n, 1.0 / n)
    for _ in range(500):
        new = damping * (P @ pr + pr[dangling].sum() / n) + (1 - damping) / n
        if np.abs(new - pr).sum() < tol:
            return new
        pr = new
    return pr


def _col_nanmean(a: np.ndarray) -> np.ndarray:
    """Media por columna ignorando NaN; NaN si la columna no tiene datos."""
    ok = np.isfinite(a)
    cnt = ok.sum(axis=0)
    tot = np.where(ok, a, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _weighted_payer_mean(W: sp.csr_array, v: np.ndarray) -> np.ndarray:
    """Σ_j Ŵ_ij v_j / Σ_j Ŵ_ij sobre pagadores con v finito; 0 si no hay."""
    ok = np.isfinite(v)
    num = W @ np.where(ok, v, 0.0)
    den = W @ ok.astype(float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def classic_features(entities: pd.DataFrame) -> pd.DataFrame:
    e = entities
    return pd.DataFrame(
        {
            "is_person": (e["kind"] == PERSON).astype(int),
            "log_income": np.log(e["income"]),
            "debt_ratio": e["debt_ratio"],
            "age_years": e["age_years"],
            "tenure_months": e["tenure_months"],
            "n_products": e["n_products"],
            "past_due_12m": e["past_due_12m"],
            "formal": e["formal"],
            "log_employees": np.log1p(e["n_employees"]),
        },
        index=e.index,
    )


def client_features(
    entities: pd.DataFrame,
    tx: pd.DataFrame,
    t_obs: float,
    grid_step: float = 0.05,
) -> pd.DataFrame:
    """Tabla de variables (una fila por cliente) con los tres niveles.

    Parameters
    ----------
    entities : tabla de clientes; solo se leen columnas observables.
    tx : transacciones con ``0 <= t < t_obs``.
    t_obs : largo de la ventana de observación [meses].
    grid_step : paso de la rejilla para r̂_i(t) [meses].
    """
    ent = entities[list(OBSERVABLE_ENTITY_COLUMNS)].reset_index(drop=True)
    n = len(ent)
    if (tx["t"] >= t_obs).any():
        raise ValueError("tx contiene pagos posteriores a t_obs: fuga de información")
    income = ent["income"].to_numpy(dtype=float)
    edges = estimate_edges(tx, t_obs)
    W = edges_matrix(edges, n)
    k = row_sums(W)

    def in_share(layer: int) -> np.ndarray:
        return row_sums(edges_matrix(edges, n, [layer])) / income

    def out_share(layer: int) -> np.ndarray:
        return col_sums(edges_matrix(edges, n, [layer])) / income

    shares = sp.csr_array(sp.diags_array(1.0 / np.where(k > 0, k, 1.0)) @ W)
    hhi = row_sums(shares.power(2))

    emp = ent["employer"].to_numpy()
    has_emp = emp >= 0
    emp_idx = np.where(has_emp, emp, 0)

    def employer_value(col: str, transform=lambda v: v) -> np.ndarray:
        v = transform(ent[col].to_numpy(dtype=float))
        return np.where(has_emp, v[emp_idx], 0.0)

    static = pd.DataFrame(
        {
            "net_income_share": k / income,
            "in_share_N": in_share(LAYER_N),
            "in_share_S": in_share(LAYER_S),
            "in_share_T": in_share(LAYER_T),
            "out_share_S": out_share(LAYER_S),
            "out_share_T": out_share(LAYER_T),
            "n_payers": np.diff(W.indptr),
            "n_payees": np.diff(sp.csc_array(W).indptr),
            "income_hhi": hhi,
            "pagerank": pagerank(W) * n,  # 1 = importancia media
            "employer_log_size": employer_value("n_employees", np.log1p),
            "employer_debt_ratio": employer_value("debt_ratio"),
            "employer_past_due": employer_value("past_due_12m"),
            "payer_past_due": _weighted_payer_mean(
                W, ent["past_due_12m"].to_numpy(dtype=float)
            ),
            "payer_debt_ratio": _weighted_payer_mean(
                W, ent["debt_ratio"].to_numpy(dtype=float)
            ),
        }
    )

    # ---- dinámica: fases por eventos y coherencia con los pagadores
    cycles = payment_cycles(tx, n)
    grid = np.arange(0.0, t_obs, grid_step)
    theta = phase_matrix(cycles, grid)
    r = local_coherence_series(W, theta)
    lag = phase_lag_series(W, theta)
    omega, cv, stopped = cycle_stats(cycles, t_obs)
    no_net = k <= 0
    valid_r = np.isfinite(r)
    has_r = valid_r.any(axis=0)
    r_avg = _col_nanmean(r)
    # Sin ingreso de red (o sin fases) no hay desincronización medible: r = 1.
    r_mean = np.where(has_r, r_avg, 1.0)
    r_var = np.maximum(_col_nanmean(r * r) - r_avg**2, 0.0)
    r_std = np.where(has_r, np.sqrt(np.nan_to_num(r_var)), 0.0)
    r_low = np.where(
        has_r, _col_nanmean(np.where(valid_r, (r < 0.5).astype(float), np.nan)), 0.0
    )
    lag_abs = np.where(np.isfinite(lag).any(axis=0), _col_nanmean(np.abs(lag)), 0.0)
    payer_omega = _weighted_payer_mean(W, omega)
    mismatch = np.where(
        np.isfinite(omega) & ~no_net, np.abs(omega - payer_omega) / TWO_PI, 0.0
    )
    lost = W @ stopped.astype(float)

    dynamic = pd.DataFrame(
        {
            "r_mean": r_mean,
            "r_std": np.nan_to_num(r_std),
            "r_low_frac": np.nan_to_num(r_low),
            "phase_lag_abs": np.nan_to_num(lag_abs),
            "freq_mismatch": mismatch,
            "period_cv": np.nan_to_num(cv),
            "payer_period_cv": _weighted_payer_mean(W, cv),
            "income_lost_frac": np.where(k > 0, lost / np.where(k > 0, k, 1.0), 0.0),
        }
    )
    out = pd.concat(
        [ent[["entity_id"]], classic_features(ent), static, dynamic], axis=1
    )
    assert_no_truth(out.columns)
    bad = [c for c in out.columns if not np.isfinite(out[c].to_numpy(float)).all()]
    if bad:
        raise ValueError(f"Variables con NaN o infinitos: {bad}")
    return out


def global_synchrony(tx: pd.DataFrame, n: int, t_obs: float, step: float = 0.05):
    """r̂(t) de toda la cartera estimado desde transacciones (alerta temprana)."""
    from finnet.credit.estimate import global_order

    grid = np.arange(0.0, t_obs, step)
    return grid, global_order(phase_matrix(payment_cycles(tx, n), grid))
