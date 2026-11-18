"""Modelos predictivos sobre el dataset de la Fase 5 (Fase 6).

Conflicto de OpenMP en macOS: torch y xgboost traen cada uno su propia libomp.
Con las dos cargadas, la que se use en paralelo en segundo lugar termina en
segmentation fault (``XGBClassifier.fit`` si torch se importó primero; una capa
``Linear`` de torch si xgboost se importó primero). Solución: cargar xgboost primero
y dejar torch en un solo hilo. Nuestros modelos de torch (GCN sobre grafos de
≤ 600 nodos, PINN pequeña) no ganan nada con más hilos. En Linux ambos usan libgomp
y no hay conflicto.
"""

import sys

import xgboost  # noqa: F401  (debe cargarse antes que torch; ver arriba)

if sys.platform == "darwin":
    import torch

    torch.set_num_threads(1)
