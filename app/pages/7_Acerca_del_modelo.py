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
- **Parámetros ilustrativos, no calibrados** con datos de ningún sistema real (Bre-B
  aparece solo como escenario de ejemplo).
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

### Tarea 6 · riesgo crediticio en la red multicapa de un banco
Nodos: personas y firmas clientes de un mismo banco. Capas: nómina (firma → empleado),
proveedores (firma → firma) y transferencias (persona → persona). Tiempo en **meses**;
déficit $x_i$ como **fracción del ingreso total** $s_i$ (incluye el externo a la red).
""")
st.latex(
    r"\dot x_i = -\alpha x_i + \sum_\ell \beta_\ell \sum_j \frac{W^\ell_{ij}}{s_i}x_j"
    r" + \gamma\frac{k_i}{s_i}(1-r_i) + \sigma\xi_i(t)"
)
st.latex(
    r"\dot\theta_i = \omega_i + \sum_\ell K_\ell \sum_j \frac{W^\ell_{ij}}{k_i}"
    r"\sin(\theta_j-\theta_i),\qquad x_i^{(c)} = c\left(\frac{\mathrm{PD}_\text{ref}}"
    r"{\mathrm{PD}_i}\right)^{\kappa}\varepsilon_i"
)
st.markdown(r"""
- **Acople de fase normalizado por fila**: cada cliente ajusta su ritmo según la
  participación de cada pagador en su ingreso.
- **Impagos idiosincráticos** programados con tasa $-\ln(1-\mathrm{PD}_i)/12$ por mes;
  perder un pagador que aporta el 100 % del ingreso suma $\lambda$ meses de ingreso al
  déficit.
- **Lo observable**: solo el registro de transacciones. La fase se estima por eventos
  (Pikovsky et al., 2001): $\hat\theta(t) = 2\pi\,(k + (t-t_k)/(t_{k+1}-t_k))$.
- **Control nulo**: $\beta_\ell = \gamma = \lambda = 0$ con los mismos mundos y sorteos.
  Si las variables de red "mejoraran" ahí, sería fuga de información.
- **Límite**: muestra que el mecanismo es detectable desde transacciones; no prueba que
  exista en una cartera real (requiere datos anonimizados de un banco).

### Referencias
Newman (2010); Barabási (2016); Chung (1997); Acebrón et al. (2005); Rodrigues et al.
(2016); Pikovsky et al. (2001); Eisenberg y Noe (2001); Shi y Malik (2000); Acemoglu et
al. (2015); Gai y Kapadia (2010); Battiston et al. (2012); Watts (2002); Merton (1974);
Boccaletti et al. (2014); Letizia y Lillo (2019); Kipf y Welling (2017). Lista completa
en `docs/bibliografia.md`.
""")
