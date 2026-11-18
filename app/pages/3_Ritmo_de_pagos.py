"""Ritmo de pagos: sincronización de Kuramoto y desconexión de entidades."""

from __future__ import annotations

import common
import numpy as np
import streamlit as st

from pulso import service, viz

st.set_page_config(page_title="Ritmo de pagos · PULSO", layout="wide")
st.title("Ritmo de pagos")
common.sidebar_status()
if not common.live_allowed():
    st.info(
        "Red demasiado grande para simulación en vivo; reduzca el número de entidades."
    )
    st.stop()

c = common.cfg()
m = c["model"]
W = common.network()["W"]

col1, col2, col3 = st.columns(3)
m["K"] = col1.slider(
    "K · dependencia de fondos entrantes", 0.0, 10.0, float(m["K"]) or 2.0, 0.1
)
m["omega_width"] = col2.slider(
    "Δ · dispersión de ritmos naturales", 0.0, 2.0, float(m["omega_width"]), 0.05
)
m["gamma"] = col3.slider(
    "γ · peso de la desincronización en el déficit", 0.0, 0.5, float(m["gamma"]), 0.01
)
order = np.argsort(-W.sum(axis=0))
disc = st.selectbox(
    "Desconectar entidad a mitad de la corrida",
    ["(ninguna)", *[int(i) for i in order[:30]]],
    help="Las 30 entidades que más pagan, en orden.",
)

p = common.params(with_thresholds=False)
rng = np.random.default_rng(int(c["seed"]))
theta0 = rng.uniform(0, 2 * np.pi, W.shape[0])
sim = c["simulation"]
base = service.rhythm_run(W, p, sim, theta0)
res = (
    base
    if disc == "(ninguna)"
    else service.rhythm_run(W, p, sim, theta0, disconnect=int(disc))
)

c1, c2, c3 = st.columns(3)
c1.metric(
    "r final",
    f"{res['r'][-1]:.3f}",
    delta=None if res is base else f"{res['r'][-1] - base['r'][-1]:+.3f}",
)
c2.metric(
    "Déficit total final",
    f"{res['deficit'][-1]:.2f}",
    delta=(
        None
        if res is base or abs(res["deficit"][-1] - base["deficit"][-1]) < 5e-3
        else f"{res['deficit'][-1] - base['deficit'][-1]:+.2f}"
    ),
    delta_color="inverse",
)
c3.metric("Coherencia local media", f"{np.mean(res['r_local']):.3f}")
if m["gamma"] == 0:
    st.info(
        "Con γ = 0 el ritmo de pagos no crea déficit: las fases se sincronizan, pero "
        "el déficit no cambia. Sube γ para ver el acople ritmo → déficit."
    )

left, right = st.columns(2)
with left:
    st.plotly_chart(viz.phase_circle_figure(res["t"], res["theta"]), width="stretch")
with right:
    ys = {"r (con desconexión)" if res is not base else "r": res["r"]}
    if res is not base:
        ys["r (red completa)"] = base["r"]
    st.plotly_chart(
        viz.line_figure(res["t"], ys, "Parámetro de orden r(t)", "r"),
        width="stretch",
    )
    ys = {"Σ x": res["deficit"]}
    if res is not base:
        ys["Σ x (red completa)"] = base["deficit"]
    st.plotly_chart(
        viz.line_figure(res["t"], ys, "Déficit total", "Σ x"), width="stretch"
    )

st.plotly_chart(
    viz.network_figure(
        W,
        common.positions(),
        color=res["r_local"],
        colorscale="RdYlGn",
        color_title="r_i",
        cmin=-1,
        cmax=1,
        title="Coherencia local al final de la corrida",
    ),
    width="stretch",
)
