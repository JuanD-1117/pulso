# Imagen de la app pulso (Streamlit). Torch se instala en su variante CPU (ver pyproject).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# libgomp: OpenMP que necesitan xgboost y torch en Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencias primero (capa en caché mientras no cambie el lock).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY app ./app
COPY .streamlit ./.streamlit
COPY data ./data
COPY models ./models
RUN uv sync --locked --no-dev

RUN useradd --create-home pulso && chown -R pulso /app
USER pulso

ENV PULSO_CONFIG_DIR=/app/configs \
    PULSO_DATA_DIR=/app/data \
    PULSO_MODELS_DIR=/app/models

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/Inicio.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
