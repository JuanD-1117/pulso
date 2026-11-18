# Documentación de PULSO

| Archivo | Contenido |
|---|---|
| [`pulso_guia_completa.pdf`](pulso_guia_completa.pdf) | Guía desde cero (53 pp.): la idea, la base de cada tarea con su código, los resultados y el montaje del software. Para quien no conoce el tema. |
| [`pulso_presentacion.pdf`](pulso_presentacion.pdf) | Presentación (56 diapositivas, 16:9): planteamiento, las seis tareas con formulación, método, verificación y resultados, implementación y conclusiones. |
| [`bibliografia.md`](bibliografia.md) | Referencias por tarea. |
| [`diseno/`](diseno/) | Especificación de diseño de la Tarea 6. |
| [`latex/`](latex/) | Fuentes LaTeX de los dos PDF y el generador de figuras. |

## Regenerar los PDF

Requiere TeX Live (lualatex, latexmk, beamer con el tema metropolis).

```bash
# desde la raíz del repositorio
uv run pulso credit configs/credit.yaml              # resultados de la Tarea 6
PYTHONPATH=src uv run python docs/latex/figuras.py    # figuras -> docs/latex/figuras/

cd docs/latex/guia && latexmk -lualatex guia.tex && cp guia.pdf ../../pulso_guia_completa.pdf && latexmk -C
cd ../presentacion && latexmk -lualatex presentacion.tex && cp presentacion.pdf ../../pulso_presentacion.pdf && latexmk -C
```

Las figuras de las tareas 1–5 se toman de `figures/` (generadas por los notebooks).
