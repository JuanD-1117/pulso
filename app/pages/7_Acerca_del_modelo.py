"""Acerca del modelo: ecuaciones, supuestos y límites."""

from __future__ import annotations

import common
import streamlit as st

st.set_page_config(page_title="Acerca del modelo · finnet", layout="wide")
st.title("Acerca del modelo")
common.sidebar_status()

st.markdown(
    "Un solo sistema dinámico; cada capa enciende un término. Con los términos "
    "nuevos apagados, el modelo vuelve exactamente al de la capa anterior."
)
st.latex(
    r"\frac{dx_i}{dt} = -\alpha x_i + \beta\sum_j W_{ij}x_j - D_0 (Lx)_i"
    r" + \gamma\,k_i(1-r_i) + \eta_i(t)"
)
st.latex(
    r"\frac{d\theta_i}{dt} = \omega_i + \frac{K}{\langle k\rangle}\sum_j W_{ij}"
    r"\sin(\theta_j-\theta_i),\qquad r_i = \frac{1}{k_i}\sum_j W_{ij}\cos(\theta_j-\theta_i)"
)
st.latex(
    r"x_i > x_i^{(c)} \;\Rightarrow\; s_i = \text{impago},\quad W_{\cdot i}\leftarrow 0,"
    r"\quad x_k \leftarrow x_k + \lambda\,\mathrm{out}_k \frac{W_{ki}}{k_k}"
)

st.markdown(r"""
| Símbolo | Significado |
|---|---|
| $W_{ij}$ | pago medio de $j$ hacia $i$ por ciclo (peso medio de aristas = 1) |
| $x_i$ | déficit de liquidez de $i$ (en pagos medios) |
| $\alpha$, $\beta$ | autocorrección y amplificación; $R = \beta\rho(W)/\alpha$ |
| $L = D_\text{out} - W$ | Laplaciano de flujo: redistribuye sin crear ni destruir déficit |
| $\sigma$, $\tau_\text{rec}$ | $\max\mathrm{Re}\,\lambda(-\alpha I - D_0L + \beta W)$ y $-1/\sigma$ |
| $\theta_i$, $\omega_i$, $K$ | fase del ciclo intradía, ritmo natural (Lorentz) y acople |
| $k_i$, $r_i$, $\gamma$ | ingreso por ciclo, coherencia local y peso del ingreso desfasado |
| $x_i^{(c)} = c\,\mathrm{out}_i\,\varepsilon_i$ | colchón: múltiplo de pagos salientes con dispersión lognormal |
| $\lambda$ | déficit que crea perder el 100 % del ingreso, en pagos salientes propios |

**Tiempo**: 1 unidad = 1 ciclo de pagos (un día hábil). Integrador RK4 de paso fijo.

**Choques**: retiro de liquidez (salto en $x_i$ de *tamaño* × pagos salientes) o caída
operativa (la entidad recibe pero no paga durante $T$ ciclos; sus receptores pierden
ingreso a tasa $\lambda\,\mathrm{out}_k W_{ki}/k_k$ y la entidad se recupera si no cruzó
su umbral).

### Supuestos y límites
- **Parámetros ilustrativos, no calibrados** con datos de Bre-B ni de ningún sistema real.
- Las redes son sintéticas (Erdős–Rényi, Barabási–Albert, núcleo-periferia); los montos son
  lognormales.
- El acople es en un solo sentido: el ritmo de pagos afecta el déficit, no al revés.
- El modelo es lineal hasta el umbral: no incluye mecanismos de ahorro de liquidez,
  colas, ni intervención del banco central.
- Los modelos predictivos solo son válidos dentro de los rangos del barrido
  (`configs/sweep.yaml`); fuera de ellos extrapolan.
- Diferencias deliberadas con la guía, por consistencia dimensional: el término de
  desincronización es $\gamma k_i(1-r_i)$ y el acople se normaliza por $\langle k\rangle$
  (en la red completa ambas coinciden con Kuramoto clásico).

### Referencias
Newman (2010); Barabási (2016); Chung (1997); Acebrón et al. (2005); Rodrigues et al.
(2016); Eisenberg y Noe (2001); Acemoglu et al. (2015); Gai y Kapadia (2010); Battiston
et al. (2012); Watts (2002); Bech y Garratt (2006); Merrouche y Schanz (2008); Ikeda et al.
(2011); Kipf y Welling (2017).
""")
