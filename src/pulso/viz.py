"""Figuras Plotly compartidas por la app y los notebooks (sin lógica de cálculo)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

PALETTE = {
    "ok": "#2f7d5b",
    "stress": "#e0a100",
    "default": "#c2362b",
    "edge": "rgba(120,120,120,0.25)",
    "accent": "#2563a8",
}
LIGHT_COLORS = {"verde": "#2f7d5b", "amarillo": "#e0a100", "rojo": "#c2362b"}
MAX_EDGES_DRAWN = 2500


def _edge_trace(W: np.ndarray, pos: np.ndarray) -> go.Scatter:
    rows, cols = np.nonzero(W)
    if rows.size > MAX_EDGES_DRAWN:  # dibujar solo las aristas más pesadas
        keep = np.argsort(-W[rows, cols])[:MAX_EDGES_DRAWN]
        rows, cols = rows[keep], cols[keep]
    xs = np.column_stack(
        [pos[cols, 0], pos[rows, 0], np.full(rows.size, np.nan)]
    ).ravel()
    ys = np.column_stack(
        [pos[cols, 1], pos[rows, 1], np.full(rows.size, np.nan)]
    ).ravel()
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"width": 0.6, "color": PALETTE["edge"]},
        hoverinfo="skip",
        showlegend=False,
    )


def node_sizes(W: np.ndarray, lo: float = 6.0, hi: float = 22.0) -> np.ndarray:
    vol = W.sum(axis=0) + W.sum(axis=1)
    if vol.max() <= 0:
        return np.full(W.shape[0], lo)
    s = np.sqrt(vol / vol.max())
    return lo + (hi - lo) * s


def network_figure(
    W: np.ndarray,
    pos: np.ndarray,
    color: Sequence | np.ndarray | None = None,
    colorscale: str | list = "Viridis",
    color_title: str = "",
    hover: Sequence[str] | None = None,
    discrete_colors: Sequence[str] | None = None,
    title: str = "",
    cmin=None,
    cmax=None,
    height: int = 520,
) -> go.Figure:
    """Red dibujada con nodos de tamaño ∝ √volumen y color por un indicador."""
    n = W.shape[0]
    hover = hover if hover is not None else [f"Entidad {i}" for i in range(n)]
    marker: dict = {"size": node_sizes(W), "line": {"width": 0.5, "color": "#333"}}
    if discrete_colors is not None:
        marker["color"] = list(discrete_colors)
    elif color is not None:
        marker.update(
            color=np.asarray(color),
            colorscale=colorscale,
            showscale=True,
            colorbar={"title": color_title, "thickness": 12},
            cmin=cmin,
            cmax=cmax,
        )
    else:
        marker["color"] = PALETTE["accent"]
    nodes = go.Scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        mode="markers",
        marker=marker,
        text=hover,
        hoverinfo="text",
        showlegend=False,
    )
    fig = go.Figure([_edge_trace(W, pos), nodes])
    fig.update_layout(
        title=title,
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def trajectories_figure(
    t: np.ndarray,
    x: np.ndarray,
    highlight: Sequence[int] = (),
    max_lines: int = 60,
    title: str = "Déficit de liquidez x(t)",
) -> go.Figure:
    fig = go.Figure()
    n = x.shape[1]
    idx = np.argsort(-np.abs(x).max(axis=0))[:max_lines]
    for i in idx:
        hl = i in highlight
        fig.add_trace(
            go.Scatter(
                x=t,
                y=x[:, i],
                mode="lines",
                name=f"Entidad {i}",
                line={
                    "width": 2.5 if hl else 1,
                    "color": PALETTE["default"] if hl else None,
                },
                opacity=1 if hl else 0.55,
                showlegend=hl,
            )
        )
    fig.update_layout(
        title=title
        + (f" (las {len(idx)} de {n} con mayor déficit)" if n > max_lines else ""),
        xaxis_title="t [ciclos]",
        yaxis_title="x [pagos medios]",
        height=420,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def line_figure(
    t,
    ys: dict[str, np.ndarray],
    title: str,
    ytitle: str,
    height: int = 320,
) -> go.Figure:
    fig = go.Figure([go.Scatter(x=t, y=y, mode="lines", name=k) for k, y in ys.items()])
    fig.update_layout(
        title=title,
        xaxis_title="t [ciclos]",
        yaxis_title=ytitle,
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def phase_circle_figure(
    t: np.ndarray,
    theta: np.ndarray,
    r_local_final: np.ndarray | None = None,
    n_frames: int = 40,
    max_points: int = 300,
) -> go.Figure:
    """Fases sobre el círculo unitario, animadas; flecha = parámetro de orden."""
    n = theta.shape[1]
    pick = (
        np.arange(n)
        if n <= max_points
        else np.linspace(0, n - 1, max_points).astype(int)
    )
    frames_idx = np.unique(np.linspace(0, len(t) - 1, n_frames).astype(int))

    def data_at(k):
        th = theta[k, pick]
        z = np.exp(1j * theta[k]).mean()
        return [
            go.Scatter(
                x=np.cos(np.linspace(0, 2 * np.pi, 120)),
                y=np.sin(np.linspace(0, 2 * np.pi, 120)),
                mode="lines",
                line={"color": "#bbb"},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=np.cos(th),
                y=np.sin(th),
                mode="markers",
                marker={"size": 7, "color": PALETTE["accent"], "opacity": 0.6},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=[0, z.real],
                y=[0, z.imag],
                mode="lines+markers",
                line={"color": PALETTE["default"], "width": 4},
                showlegend=False,
                hovertext=f"r = {abs(z):.3f}",
            ),
        ]

    fig = go.Figure(
        # Estado final visible de entrada; ▶ reproduce desde t = 0.
        data=data_at(frames_idx[-1]),
        frames=[go.Frame(data=data_at(k), name=f"{t[k]:.2f}") for k in frames_idx],
    )
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title="Fases del ciclo de pagos",
        xaxis={"range": [-1.15, 1.15], "visible": False},
        yaxis={"range": [-1.15, 1.15], "visible": False, "scaleanchor": "x"},
        updatemenus=[_play_button()],
        sliders=[
            _slider(
                [f"{t[k]:.2f}" for k in frames_idx], "t = ", active=len(frames_idx) - 1
            )
        ],
    )
    return fig


def _play_button(duration: int = 120) -> dict:
    return {
        "type": "buttons",
        "showactive": False,
        "x": 0.0,
        "y": -0.02,
        "xanchor": "left",
        "buttons": [
            {
                "label": "▶",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    },
                ],
            },
            {
                "label": "⏸",
                "method": "animate",
                "args": [
                    [None],
                    {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                ],
            },
        ],
    }


def _slider(labels: list[str], prefix: str, active: int = 0) -> dict:
    return {
        "active": active,
        "currentvalue": {"prefix": prefix},
        "pad": {"t": 30},
        "x": 0.12,
        "len": 0.88,
        "steps": [
            {
                "label": lb,
                "method": "animate",
                "args": [
                    [lb],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
            }
            for lb in labels
        ],
    }


def cascade_animation(
    W: np.ndarray,
    pos: np.ndarray,
    t: np.ndarray,
    status: np.ndarray,
    n_frames: int = 50,
    height: int = 560,
    highlight: Sequence[int] = (),
) -> go.Figure:
    """Cascada animada: verde = normal, ámbar = en estrés, rojo = impago.

    Las entidades de ``highlight`` (las chocadas) se dibujan encima y con contorno.
    """
    colors = np.array([PALETTE["ok"], PALETTE["stress"], PALETTE["default"]])
    frames_idx = np.unique(np.linspace(0, len(t) - 1, n_frames).astype(int))
    sizes = node_sizes(W)
    edges = _edge_trace(W, pos)

    hl = np.zeros(len(sizes), dtype=bool)
    hl[list(highlight)] = True
    order = np.concatenate(
        [np.flatnonzero(~hl), np.flatnonzero(hl)]
    )  # chocadas al final
    line_w = np.where(hl[order], 3.0, 0.5)
    line_c = np.where(hl[order], "#000", "#333")

    def nodes_at(k):
        st = status[k][order]
        tag = np.where(hl[order], " · CHOCADA", "")
        hover = [
            f"Entidad {i} · {['normal', 'en estrés', 'impago'][s]}{g}"
            for i, s, g in zip(order, st, tag, strict=True)
        ]
        return go.Scatter(
            x=pos[order, 0],
            y=pos[order, 1],
            mode="markers",
            text=hover,
            hoverinfo="text",
            marker={
                "size": sizes[order] * np.where(hl[order], 1.4, 1.0),
                "color": colors[st].tolist(),
                "line": {"width": line_w, "color": line_c.tolist()},
            },
            showlegend=False,
        )

    fig = go.Figure(
        # Se muestra el estado final; ▶ reproduce la cascada desde t = 0.
        data=[edges, nodes_at(frames_idx[-1])],
        frames=[
            go.Frame(data=[edges, nodes_at(k)], name=f"{t[k]:.2f}") for k in frames_idx
        ],
    )
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        updatemenus=[_play_button(150)],
        sliders=[
            _slider(
                [f"{t[k]:.2f}" for k in frames_idx], "t = ", active=len(frames_idx) - 1
            )
        ],
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def events_timeline(events, height: int = 300) -> go.Figure:
    color = {
        "impago": PALETTE["default"],
        "caida": PALETTE["stress"],
        "recuperacion": PALETTE["ok"],
    }
    fig = go.Figure()
    for ev, g in events.groupby("event"):
        fig.add_trace(
            go.Scatter(
                x=g["t"],
                y=g["entity"],
                mode="markers",
                name=ev,
                marker={"color": color.get(ev, "#555"), "size": 8},
                text=g["cause"],
                hovertemplate="t=%{x:.2f}<br>entidad %{y}<br>%{text}",
            )
        )
    fig.update_layout(
        title="Línea de tiempo de eventos",
        xaxis_title="t [ciclos]",
        yaxis_title="entidad",
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


# ------------------------------------------------------------- Tarea 6: crédito

LAYER_COLORS = {
    0: "rgba(37,99,168,0.35)",  # nómina
    1: "rgba(194,54,43,0.45)",  # proveedores
    2: "rgba(120,120,120,0.22)",  # transferencias
}
LEVEL_COLORS = {
    "clasico": "#9aa5b1",
    "red_estatica": "#2563a8",
    "red_dinamica": "#c2362b",
}


def multilayer_figure(
    edges,
    pos: np.ndarray,
    kind: np.ndarray,
    color: np.ndarray | None = None,
    color_title: str = "",
    hover: Sequence[str] | None = None,
    highlight: Sequence[int] = (),
    layer_names: dict[int, str] | None = None,
    colorscale: str | list = "RdYlGn",
    cmin=None,
    cmax=None,
    height: int = 560,
) -> go.Figure:
    """Red multicapa: aristas coloreadas por capa; firmas como cuadrados."""
    layer_names = layer_names or {0: "nómina", 1: "proveedores", 2: "transferencias"}
    traces = []
    for layer, g in edges.groupby("layer"):
        src, dst = g["src"].to_numpy(), g["dst"].to_numpy()
        if src.size > MAX_EDGES_DRAWN:
            keep = np.argsort(-g["weight"].to_numpy())[:MAX_EDGES_DRAWN]
            src, dst = src[keep], dst[keep]
        nan = np.full(src.size, np.nan)
        traces.append(
            go.Scatter(
                x=np.column_stack([pos[src, 0], pos[dst, 0], nan]).ravel(),
                y=np.column_stack([pos[src, 1], pos[dst, 1], nan]).ravel(),
                mode="lines",
                line={"width": 0.7, "color": LAYER_COLORS.get(int(layer), "#999")},
                hoverinfo="skip",
                name=layer_names.get(int(layer), str(layer)),
            )
        )
    n = len(kind)
    hover = list(hover) if hover is not None else [f"Cliente {i}" for i in range(n)]
    hl = np.zeros(n, dtype=bool)
    hl[list(highlight)] = True
    marker: dict = {
        "size": np.where(kind == 0, 13, 6) * np.where(hl, 1.6, 1.0),
        "symbol": np.where(kind == 0, "square", "circle"),
        "line": {"width": np.where(hl, 2.5, 0.4), "color": "#111"},
    }
    if color is not None:
        marker.update(
            color=np.asarray(color),
            colorscale=colorscale,
            showscale=True,
            cmin=cmin,
            cmax=cmax,
            colorbar={"title": color_title, "thickness": 12},
        )
    else:
        marker["color"] = np.where(kind == 0, PALETTE["accent"], PALETTE["ok"])
    traces.append(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers",
            marker=marker,
            text=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.04},
    )
    return fig


def level_bars(
    summary,
    metric: str,
    title: str,
    level_labels: dict[str, str],
    regime_labels: dict[str, str],
    height: int = 380,
) -> go.Figure:
    """Métrica media (± desviación entre particiones) por nivel y régimen."""
    fig = go.Figure()
    for level, label in level_labels.items():
        g = summary[summary["level"] == level]
        fig.add_trace(
            go.Bar(
                x=[regime_labels.get(r, r) for r in g["regime"]],
                y=g[f"{metric}_mean"],
                error_y={"type": "data", "array": g[f"{metric}_std"]},
                name=label,
                marker_color=LEVEL_COLORS.get(level),
            )
        )
    fig.update_layout(
        barmode="group",
        title=title,
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": -0.15},
    )
    return fig


def delta_forest(deltas, labels: Sequence[str], title: str, xtitle: str) -> go.Figure:
    """Mejoras pareadas con IC 95 %: un punto por comparación."""
    d = deltas.reset_index(drop=True)
    colors = np.where(d["ci_low"] > 0, PALETTE["ok"], "#777")
    fig = go.Figure(
        go.Scatter(
            x=d["delta_mean"],
            y=list(labels),
            mode="markers",
            marker={"size": 11, "color": colors},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": d["ci_high"] - d["delta_mean"],
                "arrayminus": d["delta_mean"] - d["ci_low"],
            },
        )
    )
    fig.add_vline(x=0, line={"color": "#444", "dash": "dot"})
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        height=60 + 42 * len(d),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig
