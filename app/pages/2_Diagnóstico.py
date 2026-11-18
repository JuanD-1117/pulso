"""Diagnóstico: R, σ, tiempo de recuperación, comunidades y ranking."""

from __future__ import annotations

import common
import numpy as np
import streamlit as st

from finnet import service, viz

st.set_page_config(page_title="Diagnóstico · finnet", layout="wide")
st.title("Diagnóstico")
common.sidebar_status()

c = common.cfg()
m = c["model"]
W = common.network()["W"]

col1, col2, col3 = st.columns(3)
m["alpha"] = col1.slider(
    "α · autocorrección [1/ciclo]", 0.05, 3.0, float(m["alpha"]), 0.05
)
with col2:
    common.beta_slider(W)
m["D0"] = col3.slider(
    "D₀ · redistribución",
    0.0,
    2.0,
    float(m["D0"]),
    0.01,
    help="Traslada presión de liquidez a los receptores sin crearla ni destruirla.",
)
n_comm = col3.slider("Comunidades", 2, 8, 3)

p = common.params(with_thresholds=False)
d = service.diagnose(W, p, n_communities=n_comm)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Índice de amplificación R", f"{d['R']:.3f}", help="β ρ(W)/α. R < 1: amortigua."
)
c2.metric(
    f"{common.LIGHT_EMOJI[d['light']]} Estabilidad σ",
    f"{d['sigma']:.3f}",
    help="Mayor parte real de los valores propios de M = -αI - D₀L + βW.",
)
c3.metric(
    "Tiempo de recuperación",
    "∞" if not np.isfinite(d["recovery_time"]) else f"{d['recovery_time']:.2f} ciclos",
)
c4.metric("Conectividad algebraica λ₂", f"{d['lambda2']:.3f}")

if d["light"] == "rojo":
    st.error("σ > 0: la red amplifica cualquier choque; el déficit crece sin límite.")
elif d["light"] == "amarillo":
    st.warning(
        "σ ≈ 0: la red está cerca del umbral; los choques tardan mucho en disiparse."
    )
else:
    st.success("σ < 0: la red amortigua los choques.")

left, right = st.columns([3, 2])
pos = common.positions()
with left:
    palette = [
        "#2563a8",
        "#c2362b",
        "#2f7d5b",
        "#e0a100",
        "#7b4fa0",
        "#d06a2c",
        "#4a9aa8",
        "#777",
    ]
    colors = [palette[k % len(palette)] for k in d["communities"]]
    st.plotly_chart(
        viz.network_figure(
            W,
            pos,
            discrete_colors=colors,
            title="Comunidades (bisección espectral de Fiedler)",
        ),
        width="stretch",
    )
with right:
    st.markdown("**Ranking por vector propio dominante**")
    st.dataframe(
        d["ranking"]
        .head(15)
        .style.format(
            {
                "importancia": "{:.4f}",
                "vulnerabilidad": "{:.4f}",
                "paga": "{:.2f}",
                "recibe": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Importancia: vector izquierdo (cuánto propaga). "
        "Vulnerabilidad: vector derecho (cuánto acumula)."
    )

st.subheader("Choque de prueba (sin impagos)")
if not common.live_allowed():
    st.info("Red demasiado grande para simulación en vivo.")
    st.stop()
col1, col2 = st.columns(2)
default_target = int(d["ranking"]["entidad"].iloc[0])
target = col1.number_input("Entidad chocada", 0, W.shape[0] - 1, default_target)
size = col2.slider(
    "Tamaño (ciclos de pagos salientes)", 0.1, 10.0, float(c["shock"]["size"]), 0.1
)
shock = service.make_shock(
    {**c["shock"], "kind": "liquidity", "size": size}, [int(target)]
)
res = service.simulate(W, p, shock, c["simulation"])
st.plotly_chart(
    viz.trajectories_figure(res["t"], res["x"], highlight=[int(target)]),
    width="stretch",
)
st.plotly_chart(
    viz.line_figure(
        res["t"], {"Σ x": res["total_deficit"]}, "Déficit total del sistema", "Σ x"
    ),
    width="stretch",
)
