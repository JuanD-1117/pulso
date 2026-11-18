"""Página de inicio de la app PULSO."""

from __future__ import annotations

import common
import streamlit as st

st.set_page_config(
    page_title="PULSO · riesgo en redes de pagos digitales",
    page_icon="🕸️",
    layout="wide",
)

st.title("PULSO · redes complejas, sincronización y riesgo en pagos digitales")
st.caption("Pagos, Umbrales, Liquidez y Sincronización Operativa")
st.markdown(
    """
Herramienta para explorar cómo un problema de liquidez se propaga por una red de
pagos digitales y cómo la **sincronía de los ciclos de pago** se convierte en riesgo
de impago. Un solo modelo dinámico, construido en capas (tareas de la propuesta):

| Capa | Qué modela | Página |
|---|---|---|
| Red y déficit | El déficit de quien paga se amplifica en quien recibe (índice **R**) | Construir red · Diagnóstico |
| Redistribución | La presión de liquidez se traslada sin crearse ni destruirse (**σ**, **τ_rec**) | Diagnóstico |
| Ritmo de pagos | Fases del ciclo intradía (Kuramoto); el ingreso desfasado crea déficit | Ritmo de pagos |
| Impagos | Al cruzar su colchón una entidad deja de pagar: cascada | Simulador de choques |
| Datos y predicción | Miles de corridas y modelos que responden en milisegundos | Explorar escenarios · Mapa de riesgo |
| **Riesgo crediticio (Tarea 6)** | Clientes de un banco en tres capas (nómina, proveedores, transferencias): ¿la sincronía predice impagos que el score clásico no ve? | Cartera multicapa · Riesgo crediticio híbrido |

Los parámetros son **ilustrativos**, no calibrados con datos reales. Bre-B (pagos
inmediatos de Colombia) es solo un escenario de ejemplo.
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
    "ritmo → choque → explorar → mapa de riesgo → cartera multicapa → riesgo "
    "crediticio híbrido."
)
