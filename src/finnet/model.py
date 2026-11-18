r"""El sistema dinámico único del proyecto, con términos activables por fase.

.. math::

    \dot x_i = -\alpha x_i + \beta \sum_j W_{ij} x_j - D_0 (L x)_i
               + \gamma k_i (1 - r_i) + \eta_i(t)

    \dot\theta_i = \omega_i
        + \frac{K}{\langle k \rangle} \sum_j W_{ij} \sin(\theta_j - \theta_i)

con ``L = D_out - W`` (``D_out`` = diagonal de pagos salientes), ``k_i = Σ_j W_ij``,
``⟨k⟩`` su media y ``r_i = Σ_j W_ij cos(θ_j-θ_i) / k_i`` la coherencia local. Con
``D0 = K = gamma = 0`` y sin umbrales es el modelo lineal de la Fase 1; cada fase
solo enciende un término.

Dos precisiones respecto a la guía, ambas por consistencia dimensional:

- El término de desincronización es ``γ k_i (1 - r_i) = γ Σ_j W_ij (1 - cos Δθ)``:
  el déficit que crea la desincronización es proporcional al ingreso que llega
  desfasado (dinero/ciclo), no un número fijo igual para entidades grandes y
  pequeñas. γ queda en 1/ciclo.
- El acople de fase se normaliza por ``⟨k⟩`` (ingreso medio) en lugar de ``N w̄``.
  En la red completa de pesos 1, ``⟨k⟩ = N - 1`` y se recupera Kuramoto clásico
  (``K_c = 2Δ``); en redes dispersas evita que ``K_c`` escale como ``N/⟨k⟩``, que
  es lo que hace a K comparable entre redes.

Unidades: tiempo en ciclos de pago (1 = un día hábil); dinero en "pagos medios por
ciclo" (ver :mod:`finnet.networks`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True, eq=False)
class Params:
    """Parámetros del modelo. Los términos nuevos valen 0 (apagados) por defecto.

    Attributes
    ----------
    alpha : tasa de autocorrección del déficit [1/ciclo] (Fase 1).
    beta : amplificación por déficit de quien paga [1/(ciclo·pago medio)] (Fase 1).
    D0 : velocidad de redistribución conservativa de presión (Fase 2).
    K : acople de Kuramoto: dependencia de fondos entrantes (Fase 3).
    gamma : tasa a la que el ingreso desfasado se vuelve déficit [1/ciclo] (Fase 3).
    omega : ritmos naturales ω_i [rad/ciclo]; ``None`` = fases estáticas (Fase 3).
    thresholds : umbrales de impago x_i^(c); ``None`` = sin impagos (Fase 4).
    loss : λ, déficit que crea en un receptor perder el 100 % de su ingreso,
        en unidades de sus propios pagos salientes por ciclo (Fase 4).
    """

    alpha: float
    beta: float
    D0: float = 0.0
    K: float = 0.0
    gamma: float = 0.0
    omega: np.ndarray | None = field(default=None, repr=False)
    thresholds: np.ndarray | None = field(default=None, repr=False)
    loss: float = 0.0

    def __post_init__(self) -> None:
        for name in ("alpha", "D0", "K", "gamma", "loss"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} debe ser >= 0 (recibido {getattr(self, name)})"
                )
        if self.beta < 0:
            raise ValueError(
                "beta debe ser >= 0: el déficit de un pagador no reduce el ajeno"
            )

    @property
    def has_phases(self) -> bool:
        return self.omega is not None or self.K > 0

    def with_(self, **changes) -> Params:
        return replace(self, **changes)


class Dynamics:
    """Lado derecho compilado para una red fija.

    Precalcula los coeficientes para que cada evaluación cueste un solo producto
    matriz-matriz ``W @ [x, cos θ, sin θ]``. La parte lineal se reescribe como

        -α x - D0 (out ∘ x - W x) + β W x = -(α + D0 out) ∘ x + (β + D0) W x.

    ``W`` es la red *efectiva* (columnas de entidades que no pagan ya anuladas).
    """

    def __init__(
        self, W: np.ndarray, params: Params, weight_scale: float | None = None
    ):
        W = np.asarray(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"W debe ser cuadrada; forma recibida {W.shape}")
        if (W < 0).any():
            raise ValueError("W no puede tener pagos negativos")
        self.W = W
        self.p = params
        self.n = W.shape[0]
        out = W.sum(axis=0)
        self.k = W.sum(axis=1)
        self.diag = params.alpha + params.D0 * out
        self.offdiag = params.beta + params.D0
        # La escala de acople (ingreso medio ⟨k⟩) se fija con la red original: si
        # una entidad cae, el acople de las demás no debe re-normalizarse.
        if weight_scale is None:
            weight_scale = float(self.k.mean()) if self.k.mean() > 0 else 1.0
        self.weight_scale = weight_scale
        self.kur = params.K / weight_scale
        self.omega = (
            np.zeros(self.n)
            if params.omega is None
            else np.asarray(params.omega, float)
        )
        if self.omega.shape != (self.n,):
            raise ValueError(f"omega debe tener forma ({self.n},)")
        self.need_phases = params.K > 0 or params.gamma > 0

    def local_coherence(
        self, theta: np.ndarray, z: np.ndarray | None = None
    ) -> np.ndarray:
        """r_i = Re(e^{-iθ_i} z_i)/k_i con z = W e^{iθ}; r_i = 1 si no recibe nada."""
        if z is None:
            z = self.W @ np.exp(1j * theta)
        num = np.cos(theta) * z.real + np.sin(theta) * z.imag
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(self.k > 0, num / np.where(self.k > 0, self.k, 1.0), 1.0)
        return r

    def rhs(
        self, x: np.ndarray, theta: np.ndarray, source: np.ndarray | float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve ``(dx/dt, dθ/dt)``. ``source`` es η(t) (forzamiento externo)."""
        if self.need_phases:
            c, s = np.cos(theta), np.sin(theta)
            prod = self.W @ np.column_stack((x, c, s))
            Wx, zr, zi = prod[:, 0], prod[:, 1], prod[:, 2]
            dx = -self.diag * x + self.offdiag * Wx + source
            if self.p.gamma > 0:
                # k_i (1 - r_i) = k_i - Σ_j W_ij cos(θ_j - θ_i)
                dx = dx + self.p.gamma * (self.k - (c * zr + s * zi))
            # Im(e^{-iθ} z) = cos θ · Im z - sin θ · Re z
            dth = self.omega + self.kur * (c * zi - s * zr)
        else:
            dx = -self.diag * x + self.offdiag * (self.W @ x) + source
            dth = self.omega
        return dx, dth


def rhs(
    x: np.ndarray,
    theta: np.ndarray,
    W: np.ndarray,
    p: Params,
    source: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interfaz funcional de :class:`Dynamics` (cómoda, no para lazos calientes)."""
    return Dynamics(W, p).rhs(x, theta, source)


def lorentz_frequencies(n: int, center: float, width: float) -> np.ndarray:
    """Cuantiles deterministas de una Lorentz(center, width).

    Usar cuantiles en lugar de muestras aleatorias elimina el ruido de muestreo de
    g(ω) y deja solo el efecto de tamaño finito al comparar con la teoría.
    """
    u = (np.arange(n) + 0.5) / n
    return center + width * np.tan(np.pi * (u - 0.5))


def make_thresholds(
    W: np.ndarray, buffer: float, dispersion: float, rng: np.random.Generator
) -> np.ndarray:
    """Colchón x_i^(c) = c · out_i · ε_i con ε_i lognormal de media 1.

    Entidades que no pagan nada reciben el colchón medio de la red para no tener
    umbral cero (caerían con cualquier déficit).
    """
    out = W.sum(axis=0)
    base = np.where(out > 0, out, out[out > 0].mean() if (out > 0).any() else 1.0)
    eps = rng.lognormal(-0.5 * dispersion**2, dispersion, W.shape[0])
    return buffer * base * eps
