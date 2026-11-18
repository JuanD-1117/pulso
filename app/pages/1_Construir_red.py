"""Construir red: topología, tamaño y montos."""

from __future__ import annotations

import common
import streamlit as st

from pulso import viz
from pulso.networks import TOPOLOGIES

st.set_page_config(page_title="Construir red · PULSO", layout="wide")
st.title("Construir red")

c = common.cfg()
net_cfg = c["network"]
labels = {
    "erdos_renyi": "Erdős–Rényi",
    "barabasi_albert": "Barabási–Albert",
    "core_periphery": "Núcleo-periferia",
}

with st.form("network"):
    col1, col2, col3 = st.columns(3)
    topo = col1.selectbox(
        "Topología",
        TOPOLOGIES,
        index=TOPOLOGIES.index(net_cfg["topology"]),
        format_func=labels.get,
    )
    n = col1.slider("Número de entidades", 10, 600, int(net_cfg["n"]), step=5)
    k = col2.slider("Grado medio", 1.0, 20.0, float(net_cfg["mean_degree"]), step=0.5)
    sigma = col2.slider(
        "Dispersión de montos (σ lognormal)",
        0.0,
        2.0,
        float(net_cfg["amount_sigma"]),
        step=0.1,
    )
    core = col3.slider(
        "Fracción en el núcleo",
        0.02,
        0.4,
        float(net_cfg["core_frac"]),
        step=0.01,
        disabled=topo != "core_periphery",
    )
    seed = col3.number_input("Semilla", 0, 10**6, int(c["seed"]))
    submitted = st.form_submit_button("Construir", type="primary")

if submitted:
    if k >= n - 1:
        st.error("El grado medio debe ser menor que n - 1.")
        st.stop()
    net_cfg.update(
        topology=topo, n=n, mean_degree=k, amount_sigma=sigma, core_frac=core
    )
    c["seed"] = int(seed)
    st.session_state.pop("results", None)

common.sidebar_status()
net = common.network()
W = net["W"]
pos = common.positions()
s = net["summary"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entidades", s["n"])
c2.metric("Aristas dirigidas", s["edges"])
c3.metric("Densidad", f"{s['density']:.3f}")
c4.metric("CV de montos", f"{s['amount_cv']:.2f}")

paid = W.sum(axis=0)
hover = [
    f"Entidad {i}<br>paga {paid[i]:.2f} · recibe {W[i].sum():.2f}"
    for i in range(W.shape[0])
]
discrete = None
if net["core"].any():
    discrete = [
        viz.PALETTE["default"] if m else viz.PALETTE["accent"] for m in net["core"]
    ]
st.plotly_chart(
    viz.network_figure(
        W,
        pos,
        discrete_colors=discrete,
        hover=hover,
        title="Red de pagos (tamaño ∝ volumen; rojo = núcleo)",
    ),
    width="stretch",
)
st.caption(
    "Montos en unidades de “pago medio por ciclo”: el peso medio de las aristas es 1."
)
