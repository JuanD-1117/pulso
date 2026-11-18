"""Clientes del banco: firmas y personas, con variables clásicas y verdad del terreno.

Identificadores: firmas ``0 .. n_F-1``, personas ``n_F .. n_F+n_P-1``.

La PD anual idiosincrática verdadera es

    logit(PD_i) = b_kind + Σ_k c_k z_k(i) + σ_u u_i

con ``z_k`` las variables observables estandarizadas dentro de cada tipo y ``u_i`` un
componente latente N(0, 1). La mora en 12 meses ~ Poisson(m · e^{a u_i}) revela parte
de ``u_i``, como el historial crediticio real. ``b_kind`` se ajusta por bisección para
que la PD media sea la tasa objetivo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finnet.world.schema import FIRM, PERSON

TWO_PI = 2.0 * np.pi
DAYS_PER_MONTH = 30.0


def _lognormal(rng: np.random.Generator, spec: dict, size: int) -> np.ndarray:
    return rng.lognormal(np.log(spec["median"]), spec["sigma"], size)


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v, dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_intercept(score: np.ndarray, target: float) -> float:
    """b tal que mean(σ(b + score)) = target (la media es monótona en b)."""
    if not 0 < target < 1:
        raise ValueError(f"La tasa objetivo debe estar en (0, 1); recibida {target}")
    lo, hi = -30.0, 30.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _sigmoid(mid + score).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _phase_from_pay_day(omega: np.ndarray, pay_day: np.ndarray) -> np.ndarray:
    """θ(0) tal que la primera vez que θ cruza 2π sea el día de pago."""
    return np.mod(-omega * pay_day / DAYS_PER_MONTH, TWO_PI)


def _assign_employers(
    rng: np.random.Generator, n_formal: int, n_firms: int, shape: float
) -> np.ndarray:
    """Empleador de cada persona formal; cada firma recibe al menos uno si alcanza."""
    size = rng.pareto(shape, n_firms) + 1.0
    if n_formal >= n_firms:
        first = rng.permutation(n_firms)
        rest = rng.choice(n_firms, size=n_formal - n_firms, p=size / size.sum())
        emp = np.concatenate([first, rest])
    else:
        emp = rng.choice(n_firms, size=n_formal, p=size / size.sum())
    return rng.permutation(emp)


def generate_entities(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Tabla de entidades (columnas de :data:`finnet.world.schema.ENTITY_SCHEMA`)."""
    pc, fc, cal, tr = cfg["persons"], cfg["firms"], cfg["calendar"], cfg["truth"]
    n_f, n_p = int(fc["n"]), int(pc["n"])
    n_formal = round(n_p * (1.0 - pc["informal_frac"]))

    # ---- calendario de las firmas
    biweekly = rng.random(n_f) < cal["biweekly_frac"]
    omega_f = TWO_PI * np.where(biweekly, 2.0, 1.0)
    omega_f = omega_f * (1.0 + cal["firm_omega_cv"] * rng.standard_normal(n_f))
    pay_day_f = np.where(
        biweekly,
        rng.choice(cal["biweekly_first_days"], n_f),
        rng.choice(cal["monthly_pay_days"], n_f),
    ).astype(float)
    theta_f = _phase_from_pay_day(omega_f, pay_day_f)

    # ---- personas: empleo, ingreso, ritmo
    formal = np.zeros(n_p, dtype=int)
    formal[rng.permutation(n_p)[:n_formal]] = 1
    employer = np.full(n_p, -1)
    employer[formal == 1] = _assign_employers(
        rng, n_formal, n_f, fc["size_pareto_shape"]
    )
    income_p = np.where(
        formal == 1,
        _lognormal(rng, pc["income_formal"], n_p),
        _lognormal(rng, pc["income_informal"], n_p),
    )
    emp_idx = np.where(employer >= 0, employer, 0)
    omega_p = np.where(
        formal == 1,
        omega_f[emp_idx] * (1.0 + cal["formal_omega_cv"] * rng.standard_normal(n_p)),
        TWO_PI * (1.0 + cal["informal_omega_cv"] * rng.standard_normal(n_p)),
    )
    # Un ritmo natural negativo o casi nulo no tiene sentido económico.
    omega_p = np.maximum(omega_p, 0.3 * TWO_PI)
    theta_p = np.where(
        formal == 1,
        theta_f[emp_idx] + cal["formal_phase_sd"] * rng.standard_normal(n_p),
        rng.uniform(0.0, TWO_PI, n_p),
    )
    theta_p = np.mod(theta_p, TWO_PI)

    # ---- firmas: tamaño en el banco y ventas
    n_emp = np.bincount(employer[employer >= 0], minlength=n_f)
    payroll = np.bincount(
        employer[employer >= 0], weights=income_p[employer >= 0], minlength=n_f
    )
    share = rng.beta(fc["payroll_share"]["a"], fc["payroll_share"]["b"], n_f)
    # Firmas sin empleados en el banco: ventas de una firma pequeña típica.
    small = np.median(payroll[payroll > 0]) if (payroll > 0).any() else 1.0
    revenue = np.where(payroll > 0, payroll, small) / share

    def common(n: int, spec: dict) -> dict[str, np.ndarray]:
        lo, hi = spec["age"]
        return {
            "debt_ratio": spec["debt_ratio_scale"]
            * rng.beta(spec["debt_ratio"]["a"], spec["debt_ratio"]["b"], n),
            "age_years": rng.uniform(lo, hi, n),
            "tenure_months": rng.exponential(spec["tenure_mean_months"], n),
            "n_products": 1 + rng.poisson(max(spec["n_products_mean"] - 1.0, 0.0), n),
            "latent": rng.standard_normal(n),
        }

    cf, cp = common(n_f, fc), common(n_p, pc)
    # La antigüedad en el banco no puede superar la vida adulta / la vida de la firma.
    cp["tenure_months"] = np.minimum(cp["tenure_months"], (cp["age_years"] - 18) * 12)
    cf["tenure_months"] = np.minimum(cf["tenure_months"], cf["age_years"] * 12)

    firms = pd.DataFrame(
        {
            "kind": FIRM,
            "employer": -1,
            "sector": rng.integers(0, fc["n_sectors"], n_f),
            "income": revenue,
            "formal": 1,
            "n_employees": n_emp,
            "truth_omega": omega_f,
            "truth_theta0": theta_f,
            "truth_community": -1,
            **{k: v for k, v in cf.items() if k != "latent"},
            "truth_latent": cf["latent"],
        }
    )
    persons = pd.DataFrame(
        {
            "kind": PERSON,
            "employer": employer,  # índice de firma = entity_id de la firma
            "sector": -1,
            "income": income_p,
            "formal": formal,
            "n_employees": 0,
            "truth_omega": omega_p,
            "truth_theta0": theta_p,
            "truth_community": rng.integers(0, pc["n_communities"], n_p),
            **{k: v for k, v in cp.items() if k != "latent"},
            "truth_latent": cp["latent"],
        }
    )
    ent = pd.concat([firms, persons], ignore_index=True)
    ent.insert(0, "entity_id", np.arange(len(ent)))

    # ---- verdad del terreno: mora observada y PD
    u = ent["truth_latent"].to_numpy()
    ent["past_due_12m"] = rng.poisson(
        tr["past_due_base"]
        * np.exp(tr["past_due_link"] * u - 0.5 * tr["past_due_link"] ** 2)
    )
    pd_true = np.empty(len(ent))
    c = tr["coef"]
    for kind, key in ((FIRM, "firm"), (PERSON, "person")):
        m = (ent["kind"] == kind).to_numpy()
        e = ent.loc[m]
        score = (
            c["debt_ratio"] * _zscore(e["debt_ratio"].to_numpy())
            + c["log_income"] * _zscore(np.log(e["income"].to_numpy()))
            + c["tenure"] * _zscore(e["tenure_months"].to_numpy())
            + c["age"] * _zscore(e["age_years"].to_numpy())
            + c["formal"] * _zscore(e["formal"].to_numpy().astype(float))
            + tr["latent_sd"] * u[m]
        )
        b = calibrate_intercept(score, tr["target_pd"][key])
        pd_true[m] = _sigmoid(b + score)
    ent["truth_pd"] = pd_true
    return ent
