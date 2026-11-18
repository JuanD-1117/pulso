"""Tarea 6 · Cartera multicapa: clientes de un banco, sus tres capas y su ritmo."""

from __future__ import annotations

import common
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pulso import viz
from pulso.world import KIND_NAMES, LAYER_NAMES

st.set_page_config(page_title="Cartera multicapa · PULSO", layout="wide")
st.title("Cartera multicapa")
st.markdown(
    "Clientes de un banco (**personas** ● y **firmas** ■) unidos por tres capas de "
    "pagos: **nómina** (firma → empleado), **proveedores** (firma → firma) y "
    "**transferencias** (persona → persona). Cada cliente tiene un ciclo de pagos "
    "(fase θ); si el ingreso le llega desfasado de su propio ciclo, acumula déficit. "
    "El banco no ve las fases: las **estima desde las transacciones**."
)

n_p0, n_f0, inf0, seed0 = common.world_key()
with st.form("cartera"):
    c1, c2, c3, c4 = st.columns(4)
    n_p = c1.slider("Personas", 100, 600, int(n_p0), step=50)
    n_f = c2.slider("Firmas", 5, 40, int(n_f0), step=5)
    informal = c3.slider("Informalidad", 0.0, 0.9, float(inf0), step=0.05)
    seed = c4.number_input("Semilla", 0, 10_000, int(seed0), step=1)
    c5, c6, c7 = st.columns(3)
    regime = c5.radio(
        "Mundo",
        ["red", "nulo"],
        format_func=lambda r: {
            "red": "con red (acoples encendidos)",
            "nulo": "nulo (control: la red no crea déficit)",
        }[r],
    )
    base = common.credit_service.params_for("red")
    k_n = c6.slider(
        "K nómina · ajuste del ritmo al empleador",
        0.0,
        3.0,
        float(base.K["N"]),
        step=0.1,
        help="Mayor K: el ciclo de pagos del empleado se ajusta al de su nómina.",
    )
    gamma = c7.slider(
        "γ · déficit por ingreso desfasado",
        0.0,
        2.0,
        float(base.gamma),
        step=0.1,
        help="Con γ = 0 la sincronía no afecta el riesgo.",
    )
    st.form_submit_button("Simular cartera", type="primary")

st.session_state.credit_world = (int(n_p), int(n_f), float(informal), int(seed))
key = common.world_key()
common.credit_sidebar_status()
overrides = (("K_N", float(k_n)), ("gamma", float(gamma)))
world = common.cached_world(key)
pos = common.cached_world_layout(key)
res = common.cached_portfolio(key, regime, overrides)
tab = res["table"]
p = res["params"]

alive_obs = tab["observed_default"] == 0
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Clientes", world.n)
m2.metric(
    "R = ρ(B)/α",
    f"{res['R']:.3f}",
    help="Índice de amplificación del déficit en la cartera (< 1 amortigua).",
)
m3.metric("Impago en observación", f"{100 * tab['observed_default'].mean():.1f} %")
m4.metric(
    "Impago a 12 meses",
    f"{100 * tab.loc[alive_obs, 'y'].mean():.1f} %",
    help="Entre quienes llegan vivos al final de la ventana de observación.",
)
m5.metric("Sincronía final r̂", f"{np.nanmean(res['r_hat'][-10:]):.3f}")

left, right = st.columns([3, 2])
with left:
    color_by = st.radio(
        "Color",
        ["r_mean", "y", "kind"],
        horizontal=True,
        format_func={
            "r_mean": "coherencia local r̂ (estimada)",
            "y": "impago a 12 meses",
            "kind": "tipo de cliente",
        }.get,
    )
    kind = world.kind
    hover = [
        f"Cliente {i} · {KIND_NAMES[k]} · r̂ = {r:.2f} · impago 12m = {'sí' if yy else 'no'}"
        for i, k, r, yy in zip(
            tab["entity_id"], kind, tab["r_mean"], tab["y"], strict=True
        )
    ]
    if color_by == "kind":
        fig = viz.multilayer_figure(world.edges, pos, kind, hover=hover)
    elif color_by == "y":
        fig = viz.multilayer_figure(
            world.edges,
            pos,
            kind,
            color=tab["y"].to_numpy(),
            color_title="impago",
            colorscale=[[0, "#2f7d5b"], [1, "#c2362b"]],
            cmin=0,
            cmax=1,
            hover=hover,
        )
    else:
        fig = viz.multilayer_figure(
            world.edges,
            pos,
            kind,
            color=tab["r_mean"].to_numpy(),
            color_title="r̂",
            cmin=-1,
            cmax=1,
            hover=hover,
        )
    st.plotly_chart(fig, width="stretch")
with right:
    counts = world.edges.groupby("layer").agg(
        relaciones=("weight", "size"), monto_mensual=("weight", "sum")
    )
    counts.index = [LAYER_NAMES[i] for i in counts.index]
    st.markdown("**Capas** (montos en SMMLV/mes)")
    st.dataframe(counts.round(1), width="stretch")
    fig = go.Figure(
        [
            go.Scatter(
                x=res["t"],
                y=res["r_true"],
                name="r(t) del modelo (no observable)",
                line={"color": "#9aa5b1"},
            ),
            go.Scatter(
                x=res["grid"],
                y=res["r_hat"],
                name="r̂(t) estimado desde transacciones",
                line={"color": "#c2362b"},
            ),
        ]
    )
    fig.add_vline(x=p.t_obs, line={"dash": "dot"}, annotation_text="fin observación")
    fig.update_layout(
        title="Sincronía global de la cartera",
        xaxis_title="t [meses]",
        yaxis_title="r",
        height=300,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("¿Quién está desincronizado?")
groups = np.where(
    kind == 0,
    "firmas",
    np.where(tab["is_person"] & (tab["formal"] == 1), "formales", "informales"),
)
df = pd.DataFrame(
    {
        "grupo": groups,
        "r̂": tab["r_mean"],
        "impago 12m": tab["y"],
        # Sin ingreso por la red no hay desincronización medible (r̂ = 1 por convención).
        "medible": alive_obs & (tab["n_payers"] > 0),
    }
)
df = df[df["medible"]]
g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(
        [go.Box(y=d["r̂"], name=name, boxmean=True) for name, d in df.groupby("grupo")]
    )
    fig.update_layout(
        title="Coherencia local r̂ por grupo (cartera activa)",
        height=340,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")
with g2:
    if common.credit_service.results_available():
        byr = common.credit_service.experiment_results()["by_coherence"]
        fig = go.Figure(
            [
                go.Bar(
                    x=g["tramo"],
                    y=100 * g["rate"],
                    name={"red": "con red", "nulo": "nulo (control)"}[reg],
                    marker_color={"red": "#c2362b", "nulo": "#9aa5b1"}[reg],
                    text=[f"n={n:,}".replace(",", ".") for n in g["n"]],
                )
                for reg, g in byr.groupby("regime", sort=False)
            ]
        )
        fig.update_layout(
            barmode="group",
            title="Impago a 12 meses según r̂ (20 mundos del experimento)",
            xaxis_title="r̂ (coherencia local estimada)",
            yaxis_title="% impago",
            height=340,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            legend={"orientation": "h", "y": -0.3},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            "Ejecute `uv run pulso credit configs/credit.yaml` para ver esta gráfica."
        )
st.caption(
    "Solo clientes con ingreso por la red. En el mundo **con red** el impago crece "
    "al bajar r̂: el ingreso desfasado se vuelve déficit. En el mundo **nulo** también "
    "hay un gradiente, pero menor y explicado por variables clásicas (los informales, "
    "más riesgosos, están menos sincronizados); por eso la prueba de la tesis es la "
    "comparación de modelos de la página *Riesgo crediticio híbrido*, no esta gráfica."
)

st.subheader("Escenario: quiebra de un empleador")
emp = common.credit_service.biggest_employers(world, k=8)
choice = st.selectbox(
    "Firma que quiebra en t = 1 mes",
    emp["entity_id"].tolist(),
    format_func=lambda i: (
        f"Firma {i} · {int(emp.set_index('entity_id').loc[i, 'n_employees'])} "
        "empleados en el banco"
    ),
)
if st.button("Simular quiebra"):
    st.session_state.credit_shock = int(choice)
shock_firm = st.session_state.get("credit_shock")
if shock_firm is not None and shock_firm in set(emp["entity_id"]):
    shocked = common.cached_portfolio(key, regime, overrides, employer_shock=shock_firm)
    staff = world.entities.index[world.entities["employer"] == shock_firm].to_numpy()
    others = np.setdiff1d(np.flatnonzero(world.is_person), staff)

    def rate_after(r, who):
        dt = r["run"].traj.default_time[who]
        return 100 * np.mean(np.isfinite(dt) & (dt >= 1.0) & (dt <= 7.0))

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Empleados que caen (6 meses)",
        f"{rate_after(shocked, staff):.1f} %",
        delta=f"{rate_after(shocked, staff) - rate_after(res, staff):+.1f} pp vs sin quiebra",
        delta_color="inverse",
    )
    s2.metric("Resto de personas", f"{rate_after(shocked, others):.1f} %")
    s3.metric("Empleados afectados", len(staff))
    fig = go.Figure(
        [
            go.Scatter(x=res["t"], y=100 * res["frac_defaulted"], name="sin quiebra"),
            go.Scatter(
                x=shocked["t"],
                y=100 * shocked["frac_defaulted"],
                name=f"quiebra de la firma {shock_firm}",
            ),
        ]
    )
    fig.update_layout(
        title="Clientes en impago acumulados",
        xaxis_title="t [meses]",
        yaxis_title="%",
        height=300,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")
    if regime == "nulo":
        st.info(
            "En el mundo nulo la quiebra no transmite déficit: los empleados solo caen "
            "por sus propias razones. Cambie a «con red» para ver el contagio por nómina."
        )

with st.expander("Registro de transacciones (lo que ve el banco)"):
    tx = res["transactions"].copy()
    tx["capa"] = tx["layer"].map(LAYER_NAMES)
    st.dataframe(tx.head(500), width="stretch")
    st.download_button(
        "Descargar transacciones (CSV)",
        tx.to_csv(index=False).encode("utf-8"),
        file_name="transacciones.csv",
        mime="text/csv",
    )
