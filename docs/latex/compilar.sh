#!/usr/bin/env bash
# Compila los documentos LaTeX de PULSO y copia los PDF a docs/Presentaciones/.
# Uso (desde la raíz del repositorio):  bash docs/latex/compilar.sh
# Requiere TeX Live con lualatex, latexmk y el tema metropolis de beamer.
set -euo pipefail

raiz="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
latex="$raiz/docs/latex"
destino="$raiz/docs/Presentaciones"

# fuente (carpeta/archivo .tex) -> PDF de destino
documentos=(
  "guia/guia.tex:v1/pulso_guia_completa.pdf"
  "presentacion/presentacion.tex:v1/pulso_presentacion.pdf"
  "presentacion_v2/presentacion_v2.tex:v2/pulso_presentacion_v2.pdf"
  "guion/guion.tex:v2/pulso_guion_v2.pdf"
  "manual_app/manual_app.tex:v2/pulso_manual_app.pdf"
  "presentacion_v3/presentacion_v3.tex:v3/pulso_presentacion_v3.pdf"
  "guion_v3/guion_v3.tex:v3/pulso_guion_v3.pdf"
)

compilar() {
  local fuente="$1" salida="$2"
  local carpeta archivo
  carpeta="$latex/$(dirname "$fuente")"
  archivo="$(basename "$fuente")"
  echo "Compilando $fuente"
  (cd "$carpeta" && latexmk -lualatex -interaction=nonstopmode -halt-on-error \
    "$archivo" > /dev/null)
  mkdir -p "$(dirname "$destino/$salida")"
  cp "$carpeta/${archivo%.tex}.pdf" "$destino/$salida"
  (cd "$carpeta" && latexmk -C "$archivo" > /dev/null)
  echo "  -> docs/Presentaciones/$salida"
}

for doc in "${documentos[@]}"; do
  compilar "${doc%%:*}" "${doc#*:}"
done
