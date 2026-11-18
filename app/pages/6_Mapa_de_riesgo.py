"""Mapa de riesgo: predicción instantánea, explicación SHAP y verificación."""

from __future__ import annotations

import time

import common
import plotly.express as px
import streamlit as st

from finnet import service, viz

st.set_page_config(page_title="Mapa de riesgo · finnet", layout="wide")
st.title("Mapa de riesgo")
common.sidebar_status()

if not service.models_available():
    st.warning(
        "No hay modelos entrenados. Ejecute `uv run finnet train configs/train.yaml`."
    )
    st.stop()

c = common.cfg()
W = common.network()["W"]


@st.cache_resource(show_spinner="Preparando contexto de predicción…")
def _context(W, cfg_key: str, _cfg: dict):
    return service.risk_context(W, _cfg)


ctx = _context(W, repr((c["network"], c["model"], c["seed"])), c)
t = time.perf_counter()
risk = service.predict_risk(ctx, c)
ms = 1000 * (time.perf_counter() - t)

model_col = st.radio(
    "Modelo",
    ["cascada_gcn", "cascada_xgb", "p_colapso"],
    horizontal=True,
    format_func={
        "cascada_gcn": "Cascada (GCN)",
        "cascada_xgb": "Cascada (XGBoost)",
        "p_colapso": "P(colapso global)",
    }.get,
)
st.caption(
    f"Predicción para las {len(risk)} entidades en {ms:.0f} ms. Cascada = fracción de "
    f"las demás entidades que caerían si esta sufre un retiro de "
    f"{ctx.shock_size:g} ciclos de pagos."
)

left, right = st.columns([3, 2])
hover = [
    f"Entidad {i}<br>cascada GCN {g:.3f}<br>cascada XGB {x:.3f}<br>P(colapso) {p:.3f}"
    for i, g, x, p in zip(
        risk["entidad"],
        risk["cascada_gcn"],
        risk["cascada_xgb"],
        risk["p_colapso"],
        strict=True,
    )
]
with left:
    st.plotly_chart(
        viz.network_figure(
            W,
            common.positions(),
            color=risk[model_col],
            colorscale="YlOrRd",
            color_title="riesgo",
            cmin=0,
            hover=hover,
            title="Riesgo predicho por entidad",
        ),
        width="stretch",
    )
with right:
    top = risk.sort_values(model_col, ascending=False).head(12)
    st.dataframe(
        top.style.format(
            {
                "cascada_gcn": "{:.3f}",
                "cascada_xgb": "{:.3f}",
                "p_colapso": "{:.3f}",
                "paga": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("¿Por qué es riesgosa esta entidad?")
entity = st.selectbox(
    "Entidad", risk.sort_values(model_col, ascending=False)["entidad"].tolist()
)
ex = service.explain(ctx, int(entity))
top_ex = (
    ex.head(10)
    .iloc[::-1]
    .assign(efecto=lambda d: d["shap"].gt(0).map({True: "aumenta", False: "reduce"}))
)
fig = px.bar(
    top_ex,
    x="shap",
    y="variable",
    orientation="h",
    hover_data=["valor"],
    color="efecto",
    color_discrete_map={
        "aumenta": viz.PALETTE["default"],
        "reduce": viz.PALETTE["accent"],
    },
    title=f"Contribuciones SHAP (XGBoost de nodos) · base {ex.attrs['base_value']:.3f}",
)
fig.update_layout(height=380, legend_title_text="riesgo")
st.plotly_chart(fig, width="stretch")

if not common.live_allowed():
    st.info("Red demasiado grande para verificar con simulación en vivo.")
elif st.button("Verificar con simulación", type="primary"):
    t = time.perf_counter()
    sim = service.verify(ctx, int(entity))
    secs = time.perf_counter() - t
    row = risk.set_index("entidad").loc[int(entity)]
    v1, v2, v3 = st.columns(3)
    v1.metric("Simulado", f"{sim:.3f}", help=f"{secs:.2f} s de simulación")
    v2.metric(
        "GCN",
        f"{row['cascada_gcn']:.3f}",
        delta=f"{row['cascada_gcn'] - sim:+.3f}",
        delta_color="off",
    )
    v3.metric(
        "XGBoost",
        f"{row['cascada_xgb']:.3f}",
        delta=f"{row['cascada_xgb'] - sim:+.3f}",
        delta_color="off",
    )

rep = service.model_report()
if rep is not None:
    with st.expander("Métricas de validación (GroupKFold por red)"):
        st.dataframe(rep, hide_index=True, width="stretch")
