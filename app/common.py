"""Estado compartido y cachés de la app. Solo llama a ``finnet.service``."""

from __future__ import annotations

import copy

import numpy as np
import streamlit as st

from finnet import service

LIGHT_EMOJI = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}


@st.cache_data(show_spinner=False)
def default_config() -> dict:
    return service.default_config()


@st.cache_data(show_spinner="Construyendo red…")
def cached_network(net_cfg: dict, seed: int) -> dict:
    return service.build_network(net_cfg, seed)


@st.cache_data(show_spinner="Calculando posiciones…")
def cached_layout(W: np.ndarray, seed: int) -> np.ndarray:
    return service.layout(W, seed)


@st.cache_data(show_spinner=False)
def cached_example() -> dict:
    return service.example_scenario()


def cfg() -> dict:
    """Configuración activa de la sesión (copia editable de ``configs/``)."""
    if "cfg" not in st.session_state:
        st.session_state.cfg = copy.deepcopy(default_config())
    return st.session_state.cfg


def load_example() -> None:
    ex = cached_example()
    st.session_state.cfg = copy.deepcopy(ex["config"])
    st.session_state.pop("results", None)


def network() -> dict:
    c = cfg()
    return cached_network(c["network"], int(c["seed"]))


def positions() -> np.ndarray:
    return cached_layout(network()["W"], int(cfg()["seed"]))


def params(with_thresholds: bool = True):
    c = cfg()
    return service.make_params(
        network()["W"], c["model"], seed=int(c["seed"]), with_thresholds=with_thresholds
    )


def live_allowed() -> bool:
    n = network()["W"].shape[0]
    return n <= cfg()["app"]["max_live_entities"]


def sidebar_status() -> None:
    c = cfg()
    net = network()
    with st.sidebar:
        st.caption("Red activa")
        st.markdown(
            f"**{c['network']['topology']}** · {net['summary']['n']} entidades · "
            f"{net['summary']['edges']} aristas"
        )
        if not live_allowed():
            st.warning(
                f"Más de {c['app']['max_live_entities']} entidades: "
                "solo predicción, sin simulación en vivo."
            )


def beta_slider(W: np.ndarray, key: str = "beta") -> float:
    """Slider de β con rango relativo a ρ(W) para que R recorra [0, 2]."""
    from finnet.spectral import perron_root

    c = cfg()
    rho = perron_root(W)
    alpha = c["model"]["alpha"]
    beta_max = 2.0 * alpha / rho if rho > 0 else 1.0
    current = (
        service.beta_for_R(W, alpha, c["model"]["R"])
        if c["model"].get("R") is not None
        else c["model"]["beta"]
    )
    beta = st.slider(
        "β · amplificación",
        0.0,
        float(beta_max),
        float(min(current, beta_max)),
        step=float(beta_max / 200),
        format="%.4f",
        key=key,
        help="Déficit nuevo que genera en i el déficit de quien le paga.",
    )
    c["model"]["beta"] = beta
    c["model"]["R"] = None
    return beta
