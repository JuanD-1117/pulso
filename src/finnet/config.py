"""Carga de configuración YAML.

Todo parámetro numérico del proyecto vive en ``configs/``; este módulo es la única
puerta de entrada a esos archivos.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR = "FINNET_CONFIG_DIR"


def config_dir() -> Path:
    """Directorio de configuraciones.

    Prioridad: variable de entorno ``FINNET_CONFIG_DIR``; si no existe, ``configs/``
    en la raíz del repositorio (dos niveles sobre ``src/finnet``).
    """
    env = os.environ.get(_ENV_VAR)
    path = Path(env) if env else Path(__file__).resolve().parents[2] / "configs"
    if not path.is_dir():
        raise FileNotFoundError(
            f"No existe el directorio de configuración {path}. "
            f"Defina {_ENV_VAR} o ejecute desde la raíz del repositorio."
        )
    return path


def load_config(name_or_path: str | Path = "default.yaml") -> dict[str, Any]:
    """Lee un YAML por nombre (dentro de ``configs/``) o por ruta."""
    path = Path(name_or_path)
    if not path.is_file():
        path = config_dir() / path
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de configuración {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} debe contener un mapeo YAML en la raíz")
    return data


def _repo_path(env: str, default: str) -> Path:
    val = os.environ.get(env)
    return Path(val) if val else Path(__file__).resolve().parents[2] / default


def data_dir() -> Path:
    """Dataset de la Fase 5 (``FINNET_DATA_DIR`` o ``data/`` del repositorio)."""
    return _repo_path("FINNET_DATA_DIR", "data")


def models_dir() -> Path:
    """Modelos versionados (``FINNET_MODELS_DIR`` o ``models/`` del repositorio)."""
    return _repo_path("FINNET_MODELS_DIR", "models")
