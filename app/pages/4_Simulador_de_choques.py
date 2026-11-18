"""Simulador de choques: cascada de impagos animada con indicadores."""

from __future__ import annotations

import common
import numpy as np
import streamlit as st

from finnet import service, viz

st.set_page_config(page_title="Simulador de choques · finnet", layout="wide")
st.title("Simulador de choques")
common.sidebar_status()
if not common.live_allowed():
    st.info("Red demasiado grande para simulación en vivo; use el Mapa de riesgo.")
    st.stop()

c = common.cfg()
m, sh = c["model"], c["shock"]
W = common.network()["W"]
order = np.argsort(-W.sum(axis=0))

with st.form("shock"):
    col1, col2, col3 = st.columns(3)
    target = col1.selectbox(
        "Entidad chocada",
        [int(i) for i in order],
        format_func=lambda i: f"{i} (paga {W[:, i].sum():.1f})",
    )
    kind = col1.radio(
        "Tipo de choque",
        ["liquidity", "operational"],
        format_func={
            "liquidity": "Retiro de liquidez",
            "operational": "Caída operativa",
        }.get,
        index=0 if sh["kind"] == "liquidity" else 1,
    )
    size = col2.slider(
        "Tamaño del retiro (ciclos de pagos salientes)",
        0.1,
        10.0,
        float(sh["size"]),
        0.1,
    )
    duration = col2.slider(
        "Duración de la caída T [ciclos]", 0.5, 15.0, float(sh["duration"]), 0.5
    )
    m["buffer"] = col3.slider(
        "Colchón c (× pagos salientes)", 0.05, 3.0, float(m["buffer"]), 0.05
    )
    m["loss"] = col3.slider(
        "λ · pérdida por ingreso no recibido", 0.0, 3.0, float(m["loss"]), 0.05
    )
    st.form_submit_button("Simular", type="primary")

sh.update(kind=kind, size=size, duration=duration)
p = common.params(with_thresholds=True)
shock = service.make_shock(sh, [target])
theta0 = np.zeros(W.shape[0])
res = service.run_scenario(
    W, p, shock, c["simulation"], c["metrics"]["collapse_fraction"], theta0=theta0
)
met = res["metrics"]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "En impago",
    f"{100 * met['fraction_defaulted']:.1f} %",
    help=f"{met['n_defaulted']} entidades",
)
c2.metric(
    "Liquidez no pagada",
    f"{met['unpaid_liquidity']:.1f}",
    help="Acumulada, en pagos medios.",
)
c3.metric("DebtRank", f"{met['debtrank']:.3f}")
ttc = met["time_to_first_contagion"]
c4.metric("Primer contagio", "—" if np.isnan(ttc) else f"t = {ttc:.2f}")
if met["collapse"]:
    st.error(
        f"Colapso global: cae ≥ {100 * c['metrics']['collapse_fraction']:.0f} %"
        " de la red."
    )
if met["n_defaulted_before_shock"]:
    st.info(
        f"{met['n_defaulted_before_shock']} entidades cayeron antes del choque por "
        "fragilidad previa (desincronización con γ > 0); no cuentan como contagio."
    )

traj = res["trajectory"]
st.plotly_chart(
    viz.cascade_animation(
        W, common.positions(), traj.t, res["status"], highlight=[target]
    ),
    width="stretch",
)
left, right = st.columns(2)
with left:
    if res["events"].empty:
        st.info("Sin eventos: el choque se absorbió.")
    else:
        st.plotly_chart(viz.events_timeline(res["events"]), width="stretch")
with right:
    st.plotly_chart(
        viz.line_figure(
            traj.t,
            {"liquidez no pagada": traj.unpaid},
            "Liquidez no pagada acumulada",
            "pagos medios",
        ),
        width="stretch",
    )

csv = service.metrics_csv(met, res["events"])
st.download_button(
    "Exportar indicadores y eventos (CSV)", csv, "escenario.csv", "text/csv"
)

st.subheader("Entidades más peligrosas")
st.caption(
    "Choca cada entidad con el tamaño elegido y mide la fracción media de las demás "
    "que caen. Se calcula una vez por configuración y queda en caché."
)


@st.cache_data(show_spinner="Chocando cada entidad…")
def _dangerous(W: np.ndarray, cfg: dict, seed: int):
    return service.dangerous_entities(W, cfg, seed)


if st.button("Calcular ranking"):
    st.dataframe(
        _dangerous(W, c, int(c["seed"])).style.format(
            {"cascada_media": "{:.3f}", "paga": "{:.2f}"}
        ),
        hide_index=True,
    )
