"""Tarea 6 · Riesgo crediticio híbrido: score clásico vs. red estática vs. dinámica."""

from __future__ import annotations

import common
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pulso import viz
from pulso.credit.features import FEATURE_LABELS, LEVEL_LABELS, LEVELS
from pulso.world import KIND_NAMES

st.set_page_config(page_title="Riesgo crediticio híbrido · PULSO", layout="wide")
st.title("Riesgo crediticio híbrido")
common.credit_sidebar_status()
st.markdown(
    """
**Pregunta:** ¿la *dinámica* de la red de pagos (sincronía de los ciclos de pago)
aporta información de impago que ni el score clásico ni las variables *estáticas* de
red capturan?

Tres niveles de variables, mismo modelo, mismas particiones:

1. **Clásico**: ingreso, endeudamiento, moras, antigüedad, empleo formal…
2. **+ Red estática**: de quién y cuánto recibe, concentración del ingreso, PageRank,
   salud del empleador y de los pagadores (lo que ya explora la literatura).
3. **+ Red dinámica**: coherencia local r̂ con los pagadores, desfase, desajuste de
   ritmo, irregularidad de los ciclos, pagadores que dejaron de pagar.

**Control nulo:** el mismo mundo y los mismos sorteos, pero la red no crea déficit.
Si la dinámica "mejorara" ahí, sería una fuga de información.
"""
)

if not common.credit_service.results_available():
    st.warning(
        "No hay resultados del experimento. Ejecute "
        "`uv run pulso credit configs/credit.yaml` (tarda menos de un minuto)."
    )
    st.stop()

res = common.credit_service.experiment_results()
summary, deltas = res["summary"], res["deltas"]
REGIMES = {"nulo": "Mundo nulo (control)", "red": "Mundo con red"}
MODEL_NAMES = {"xgboost": "XGBoost", "logistic": "Regresión logística"}

model = st.radio(
    "Modelo", list(MODEL_NAMES), horizontal=True, format_func=MODEL_NAMES.get
)

key = deltas[
    (deltas["regime"] == "red")
    & (deltas["model"] == model)
    & (deltas["metric"] == "auc")
    & (deltas["level"] == "red_dinamica")
].set_index("base")
null = deltas[
    (deltas["regime"] == "nulo")
    & (deltas["model"] == model)
    & (deltas["metric"] == "auc")
    & (deltas["level"] == "red_dinamica")
].set_index("base")
h1, h2, h3 = st.columns(3)
h1.metric(
    "Dinámica vs. red estática (ΔAUC)",
    f"{key.loc['red_estatica', 'delta_mean']:+.4f}",
    help="Mejora pareada por partición, mundo con red. IC 95 % abajo.",
)
h1.caption(
    f"IC 95 %: [{key.loc['red_estatica', 'ci_low']:+.4f}, "
    f"{key.loc['red_estatica', 'ci_high']:+.4f}]"
)
h2.metric(
    "Híbrido vs. clásico (ΔAUC)",
    f"{key.loc['clasico', 'delta_mean']:+.4f}",
)
h2.caption(
    f"IC 95 %: [{key.loc['clasico', 'ci_low']:+.4f}, "
    f"{key.loc['clasico', 'ci_high']:+.4f}]"
)
h3.metric(
    "Control nulo: dinámica vs. clásico",
    f"{null.loc['clasico', 'delta_mean']:+.4f}",
    help="Debe ser ≈ 0: sin acople la dinámica no tiene nada que aportar.",
)
h3.caption(
    f"IC 95 %: [{null.loc['clasico', 'ci_low']:+.4f}, "
    f"{null.loc['clasico', 'ci_high']:+.4f}]"
)

s = summary[summary["model"] == model]
b1, b2 = st.columns(2)
with b1:
    fig = viz.level_bars(s, "auc", "AUC fuera de muestra", LEVEL_LABELS, REGIMES)
    lo = float((s["auc_mean"] - s["auc_std"]).min())
    fig.update_yaxes(range=[max(0.5, lo - 0.02), None])
    st.plotly_chart(fig, width="stretch")
with b2:
    fig = viz.level_bars(
        s, "pr_auc", "PR-AUC (impagos son minoría)", LEVEL_LABELS, REGIMES
    )
    st.plotly_chart(fig, width="stretch")

d = deltas[(deltas["model"] == model) & (deltas["metric"] == "auc")].copy()
labels = [
    f"{REGIMES[r].split(' (')[0]} · {LEVEL_LABELS[lv]} vs {LEVEL_LABELS[b]}"
    for r, lv, b in zip(d["regime"], d["level"], d["base"], strict=True)
]
st.plotly_chart(
    viz.delta_forest(d, labels, "Mejoras pareadas por partición (IC 95 %)", "ΔAUC"),
    width="stretch",
)
with st.expander("Tabla completa de métricas"):
    show = s[
        [
            "regime",
            "level",
            "auc_mean",
            "auc_std",
            "pr_auc_mean",
            "ks_mean",
            "brier_mean",
            "ece_mean",
        ]
    ].copy()
    show["level"] = show["level"].map(LEVEL_LABELS)
    show["regime"] = show["regime"].map(REGIMES)
    st.dataframe(show.round(4), width="stretch", hide_index=True)
    st.caption(
        "Validación GroupKFold por mundo: el modelo nunca ve en entrenamiento clientes "
        "del mundo con el que se evalúa. KS: separación entre buenos y malos "
        "pagadores. Brier y ECE: calibración (menor es mejor)."
    )

st.divider()
st.subheader("Riesgo por cliente en la cartera activa")
st.caption(
    "Modelos finales entrenados con todos los mundos «con red». La cartera es la de "
    "la página *Cartera multicapa*."
)
wkey = common.world_key()
world = common.cached_world(wkey)
port = common.cached_portfolio(wkey, "red")
tab = port["table"]
scores = common.credit_service.score_clients(tab, model="xgboost")
live = scores["observed_default"] == 0
sc = scores[live].copy()
sc["kind"] = world.kind[sc["entity_id"]]
sc["salto"] = sc["pd_red_dinamica"] - sc["pd_clasico"]

c1, c2 = st.columns([3, 2])
with c1:
    fig = go.Figure(
        go.Scatter(
            x=100 * sc["pd_clasico"],
            y=100 * sc["pd_red_dinamica"],
            mode="markers",
            marker={
                "color": np.where(sc["y"] == 1, "#c2362b", "#2f7d5b"),
                "size": np.where(sc["kind"] == 0, 11, 6),
                "symbol": np.where(sc["kind"] == 0, "square", "circle"),
                "opacity": 0.75,
            },
            text=[
                f"Cliente {i} · {KIND_NAMES[k]} · impago real: {'sí' if yy else 'no'}"
                for i, k, yy in zip(sc["entity_id"], sc["kind"], sc["y"], strict=True)
            ],
            hoverinfo="text+x+y",
        )
    )
    top = float(100 * max(sc["pd_clasico"].max(), sc["pd_red_dinamica"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, top], y=[0, top], mode="lines", line={"dash": "dot", "color": "#555"}
        )
    )
    fig.update_layout(
        title="PD a 12 meses: score clásico vs. híbrido (rojo = cayó)",
        xaxis_title="PD clásica [%]",
        yaxis_title="PD híbrida (red dinámica) [%]",
        showlegend=False,
        height=460,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")
with c2:
    st.markdown("**Riesgo oculto**: clientes cuyo riesgo sube más al mirar la red")
    hidden = sc.nlargest(10, "salto")[
        ["entity_id", "kind", "pd_clasico", "pd_red_dinamica", "y"]
    ].copy()
    hidden["kind"] = hidden["kind"].map(KIND_NAMES)
    hidden[["pd_clasico", "pd_red_dinamica"]] *= 100
    hidden.columns = ["cliente", "tipo", "PD clásica %", "PD híbrida %", "cayó"]
    st.dataframe(hidden.round(1), width="stretch", hide_index=True)
    caught = sc.nlargest(max(1, len(sc) // 10), "pd_red_dinamica")["y"].mean()
    caught_c = sc.nlargest(max(1, len(sc) // 10), "pd_clasico")["y"].mean()
    st.metric(
        "Tasa de impago en el 10 % más riesgoso",
        f"{100 * caught:.1f} % (híbrido)",
        delta=f"{100 * (caught - caught_c):+.1f} pp vs clásico ({100 * caught_c:.1f} %)",
    )
    st.caption(
        f"Una sola cartera de {len(sc)} clientes: el 10 % son ~{max(1, len(sc) // 10)} "
        "personas y esta cifra varía mucho entre semillas. La comparación válida es la "
        "de arriba (20 mundos, validación cruzada)."
    )

st.subheader("¿Por qué? Contribuciones SHAP del modelo híbrido")
shap_df = common.credit_service.explain(tab[live.to_numpy()].reset_index(drop=True))
imp = shap_df.abs().mean().sort_values(ascending=True).tail(15)
fig = go.Figure(
    go.Bar(
        x=imp.values,
        y=[FEATURE_LABELS.get(c, c) for c in imp.index],
        orientation="h",
        marker_color=[
            "#c2362b"
            if c in set(LEVELS["red_dinamica"]) - set(LEVELS["red_estatica"])
            else "#2563a8"
            if c in set(LEVELS["red_estatica"]) - set(LEVELS["clasico"])
            else "#9aa5b1"
            for c in imp.index
        ],
    )
)
fig.update_layout(
    title="Importancia media |SHAP| (gris: clásica · azul: red estática · rojo: dinámica)",
    height=460,
    margin={"l": 10, "r": 10, "t": 50, "b": 10},
)
st.plotly_chart(fig, width="stretch")

client = st.selectbox(
    "Explicar un cliente",
    hidden["cliente"].tolist() + sorted(set(sc["entity_id"]) - set(hidden["cliente"])),
)
row = int(np.flatnonzero(sc["entity_id"].to_numpy() == client)[0])
contrib = shap_df.iloc[row].sort_values(key=np.abs, ascending=False).head(10)
fig = go.Figure(
    go.Bar(
        x=contrib.values[::-1],
        y=[FEATURE_LABELS.get(c, c) for c in contrib.index[::-1]],
        orientation="h",
        marker_color=np.where(contrib.values[::-1] > 0, "#c2362b", "#2f7d5b"),
    )
)
fig.update_layout(
    title=f"Cliente {client}: qué sube (rojo) y qué baja (verde) su riesgo [log-odds]",
    height=380,
    margin={"l": 10, "r": 10, "t": 50, "b": 10},
)
st.plotly_chart(fig, width="stretch")
st.download_button(
    "Descargar PD por cliente (CSV)",
    pd.DataFrame(sc).to_csv(index=False).encode("utf-8"),
    file_name="pd_clientes.csv",
    mime="text/csv",
)
