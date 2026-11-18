"""Guardado y carga de modelos versionados en ``models/<nombre>/<versión>/``.

Cada versión guarda el artefacto (``model.joblib`` o ``model.pt``), ``metrics.json``
y ``meta.json`` (columnas de entrada, escalado, configuración y fecha).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def model_dir(models_dir: Path, name: str, version: str) -> Path:
    return Path(models_dir) / name / version


def save_model(
    models_dir: Path, name: str, version: str, artifact: Any, metrics: dict, meta: dict
) -> Path:
    d = model_dir(models_dir, name, version)
    d.mkdir(parents=True, exist_ok=True)
    if name.startswith("gcn"):
        import torch

        torch.save(artifact, d / "model.pt")
    else:
        joblib.dump(artifact, d / "model.joblib")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    meta = {
        **meta,
        "name": name,
        "version": version,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (Path(models_dir) / name / "LATEST").write_text(version)
    return d


def latest_version(models_dir: Path, name: str) -> str:
    f = Path(models_dir) / name / "LATEST"
    if not f.is_file():
        raise FileNotFoundError(
            f"No hay modelo '{name}' en {models_dir}. Ejecute `uv run pulso train "
            "configs/train.yaml` después de generar el dataset."
        )
    return f.read_text().strip()


def load_model(
    models_dir: Path,
    name: str,
    version: str | None = None,
) -> tuple[Any, dict, dict]:
    """Devuelve ``(artefacto, meta, metrics)``."""
    version = version or latest_version(models_dir, name)
    d = model_dir(models_dir, name, version)
    if name.startswith("gcn"):
        import torch

        artifact = torch.load(d / "model.pt", weights_only=False)
    else:
        artifact = joblib.load(d / "model.joblib")
    meta = json.loads((d / "meta.json").read_text())
    metrics = json.loads((d / "metrics.json").read_text())
    return artifact, meta, metrics
