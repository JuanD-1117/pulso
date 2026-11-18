"""GCN sobre grafos de pagos: predice el tamaño de cascada que causa cada entidad.

Arquitectura (Kipf y Welling, 2017): ``layers`` capas ``GCNConv`` con ReLU y
dropout, seguidas de una cabeza lineal y sigmoide (la etiqueta está en [0, 1]).
El grafo se simetriza: la cascada que inicia una entidad depende tanto de a quién
paga (pierde ingreso) como de quién le paga (le transmite déficit).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

LOG_COLUMNS = (
    "in_degree",
    "out_degree",
    "in_amount",
    "out_amount",
    "importance",
    "vulnerability",
    "threshold",
    "n",
)


def transform_features(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """log1p a las columnas de cola pesada; el resto queda igual."""
    X = df[list(columns)].to_numpy(dtype=np.float64).copy()
    for j, c in enumerate(columns):
        if c in LOG_COLUMNS:
            X[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
    return X


class GCN(nn.Module):
    """GCN con conexión residual desde las variables propias del nodo.

    Varias capas ``GCNConv`` promedian vecindarios y, en grafos pequeños y densos,
    vuelven casi iguales a todos los nodos de una red (sobre-suavizado): el modelo
    acierta el nivel de riesgo de la red pero no el orden de sus entidades. La rama
    ``skip`` lleva las variables del nodo directo a la cabeza.
    """

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in itertools.pairwise(dims))
        self.skip = nn.Linear(in_dim, hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_weight):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index, edge_weight))
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        z = torch.cat([h, torch.relu(self.skip(x))], dim=-1)
        return torch.sigmoid(self.head(z)).squeeze(-1)


def graph_from_W(W: np.ndarray, X: np.ndarray, y: np.ndarray | None = None) -> Data:
    """Grafo PyG con aristas simétricas ponderadas (peso normalizado por el máximo)."""
    S = W + W.T
    rows, cols = np.nonzero(S)
    w = S[rows, cols] / S.max() if S.max() > 0 else S[rows, cols]
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(np.vstack([rows, cols]), dtype=torch.long),
        edge_weight=torch.tensor(w, dtype=torch.float32),
    )
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32)
    return data


def build_graphs(
    nodos: pd.DataFrame,
    networks: dict[int, np.ndarray],
    columns: tuple[str, ...],
    mean: np.ndarray,
    std: np.ndarray,
    label: str | None,
) -> list[Data]:
    graphs = []
    for net_id, g in nodos.groupby("network_id", sort=True):
        g = g.sort_values("entity")
        X = (transform_features(g, columns) - mean) / std
        y = g[label].to_numpy() if label else None
        data = graph_from_W(networks[int(net_id)], X, y)
        data.network_id = int(net_id)
        graphs.append(data)
    return graphs


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - PyG/DataLoader usan el RNG global
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_gcn(train_graphs: list[Data], cfg: dict, in_dim: int, seed: int) -> GCN:
    """Entrena con MSE por nodo; lotes de grafos completos."""
    set_seed(seed)
    model = GCN(in_dim, cfg["hidden"], cfg["layers"], cfg["dropout"])
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_graphs, batch_size=cfg["batch_graphs"], shuffle=True, generator=gen
    )
    model.train()
    for _ in range(cfg["epochs"]):
        for batch in loader:
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_weight)
            # MSE + MSE de las desviaciones respecto a la media de cada grafo: el
            # segundo término premia el orden de las entidades dentro de su red.
            pm = scatter(pred, batch.batch, reduce="mean")[batch.batch]
            ym = scatter(batch.y, batch.batch, reduce="mean")[batch.batch]
            loss = nn.functional.mse_loss(pred, batch.y) + cfg[
                "centered_weight"
            ] * nn.functional.mse_loss(pred - pm, batch.y - ym)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_graph(model: GCN, data: Data) -> np.ndarray:
    model.eval()
    return model(data.x, data.edge_index, data.edge_weight).numpy().astype(np.float64)
