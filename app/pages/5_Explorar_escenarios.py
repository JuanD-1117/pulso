"""Explorar escenarios: filtros sobre el dataset y ventana de contagio."""

from __future__ import annotations

import common
import plotly.express as px
import streamlit as st

from finnet import service

st.set_page_config(page_title="Explorar escenarios · finnet", layout="wide")
st.title("Explorar escenarios")
common.sidebar_status()

if not service.dataset_available():
    st.warning("No hay dataset. Genérelo con `uv run finnet sweep configs/sweep.yaml`.")
    st.stop()


@st.cache_data(show_spinner="Cargando escenarios…")
def _scenarios():
    return service.load_scenarios()


df = _scenarios()
st.caption(
    f"{len(df):,} corridas sobre {df['network_id'].nunique()} redes · "
    f"colapso global en {100 * df['collapse'].mean():.1f} %"
)

with st.expander("Filtros", expanded=True):
    c1, c2, c3 = st.columns(3)
    topos = c1.multiselect(
        "Topología",
        sorted(df["topologia"].unique()),
        default=sorted(df["topologia"].unique()),
    )
    kinds = c1.multiselect(
        "Tipo de choque",
        sorted(df["tipo_choque"].unique()),
        default=sorted(df["tipo_choque"].unique()),
    )
    r_lo, r_hi = c2.slider(
        "Índice de amplificación R",
        float(df["R"].min()),
        float(df["R"].max()),
        (float(df["R"].min()), float(df["R"].max())),
    )
    k_lo, k_hi = c2.slider(
        "Grado medio",
        float(df["mean_degree"].min()),
        float(df["mean_degree"].max()),
        (float(df["mean_degree"].min()), float(df["mean_degree"].max())),
    )
    b_lo, b_hi = c3.slider(
        "Colchón c",
        float(df["buffer"].min()),
        float(df["buffer"].max()),
        (float(df["buffer"].min()), float(df["buffer"].max())),
    )
    K_lo, K_hi = c3.slider(
        "Acople K",
        float(df["K"].min()),
        float(df["K"].max()),
        (float(df["K"].min()), float(df["K"].max())),
    )

f = df[
    df["topologia"].isin(topos)
    & df["tipo_choque"].isin(kinds)
    & df["R"].between(r_lo, r_hi)
    & df["mean_degree"].between(k_lo, k_hi)
    & df["buffer"].between(b_lo, b_hi)
    & df["K"].between(K_lo, K_hi)
]
if f.empty:
    st.info("Ningún escenario cumple los filtros.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Corridas filtradas", f"{len(f):,}")
m2.metric("P(colapso)", f"{100 * f['collapse'].mean():.1f} %")
m3.metric("Fracción media en impago", f"{100 * f['fraction_defaulted'].mean():.1f} %")
m4.metric(
    "Liquidez no pagada media",
    f"{f['unpaid_liquidity'].mean():.2f}",
    help="En ciclos de flujo total del sistema.",
)

win = service.contagion_window(f)
st.plotly_chart(
    px.line(
        win,
        x="grado_medio",
        y="fraccion_impago",
        color="topologia",
        markers=True,
        hover_data=["prob_colapso", "corridas"],
        title="Ventana de contagio: fracción en impago contra grado medio",
        labels={"grado_medio": "grado medio", "fraccion_impago": "fracción en impago"},
    ),
    width="stretch",
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.scatter(
            f.sample(min(len(f), 3000), random_state=0),
            x="R",
            y="fraction_defaulted",
            color="topologia",
            opacity=0.5,
            title="Fracción en impago contra R",
            labels={"fraction_defaulted": "fracción en impago"},
        ),
        width="stretch",
    )
with right:
    st.plotly_chart(
        px.histogram(
            f,
            x="fraction_defaulted",
            color="tipo_choque",
            nbins=40,
            barmode="overlay",
            title="Distribución del tamaño de cascada",
            labels={"fraction_defaulted": "fracción en impago"},
        ),
        width="stretch",
    )

st.subheader("Comparar escenarios lado a lado")
cols = [
    "scenario_id",
    "topologia",
    "n",
    "mean_degree",
    "R",
    "sigma",
    "K",
    "gamma",
    "buffer",
    "loss",
    "tipo_choque",
    "shock_size",
    "target_importance",
    "fraction_defaulted",
    "collapse",
    "unpaid_liquidity",
    "time_to_first_default",
]
ids = st.multiselect(
    "Escenarios (id)",
    f["scenario_id"].tolist(),
    default=f.nlargest(2, "fraction_defaulted")["scenario_id"].tolist(),
    max_selections=4,
)
if ids:
    st.dataframe(f.set_index("scenario_id").loc[ids, cols[1:]].T, width="stretch")
st.download_button(
    "Exportar filtro (CSV)",
    f[cols].to_csv(index=False).encode(),
    "escenarios_filtrados.csv",
    "text/csv",
)
