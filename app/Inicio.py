"""Página de inicio de la app finnet."""

from __future__ import annotations

import common
import streamlit as st

st.set_page_config(
    page_title="finnet · riesgo en redes de pagos", page_icon="🕸️", layout="wide"
)

st.title("finnet · riesgo sistémico en redes de pagos")
st.markdown(
    """
Herramienta para explorar cómo un problema de liquidez en una entidad se propaga por
una red de pagos inmediatos tipo **Bre-B**. Un solo modelo dinámico, construido en
capas:

| Capa | Qué modela | Página |
|---|---|---|
| Red y déficit | El déficit de quien paga se amplifica en quien recibe (índice **R**) | Construir red · Diagnóstico |
| Redistribución | La presión de liquidez se traslada sin crearse ni destruirse (**σ**, **τ_rec**) | Diagnóstico |
| Ritmo de pagos | Fases del ciclo intradía (Kuramoto); el ingreso desfasado crea déficit | Ritmo de pagos |
| Impagos | Al cruzar su colchón una entidad deja de pagar: cascada | Simulador de choques |
| Datos y predicción | Miles de corridas y modelos que responden en milisegundos | Explorar escenarios · Mapa de riesgo |

Los parámetros de ejemplo son **ilustrativos**, no calibrados con datos reales.
"""
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Cargar escenario de ejemplo “Bre-B estilizado”", type="primary"):
        common.load_example()
        st.success("Escenario cargado: núcleo-periferia con 245 entidades.")
with col2:
    if st.button("Volver a la red por defecto"):
        st.session_state.pop("cfg", None)
        st.session_state.pop("results", None)
        st.info("Configuración por defecto restaurada.")

common.sidebar_status()
net = common.network()
st.subheader("Red activa")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entidades", net["summary"]["n"])
c2.metric("Aristas", net["summary"]["edges"])
c3.metric("Grado medio", f"{net['summary']['mean_degree']:.1f}")
c4.metric("CV de montos", f"{net['summary']['amount_cv']:.2f}")
st.caption(
    "Recorre las páginas de la barra lateral en orden: construir → diagnosticar → "
    "ritmo → choque → explorar → mapa de riesgo."
)
