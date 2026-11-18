# Documentación de PULSO

| Archivo | Contenido |
|---|---|
| [`Presentaciones/`](Presentaciones/) | PDF por versión. **v1**: guía completa desde cero (53 pp.) y presentación completa (56 diapositivas). **v2**: presentación breve (31 diapositivas), guion y manual de la aplicación. **v3**: presentación de 30 minutos (23 diapositivas y respaldo) con guion cronometrado. |
| [`bibliografia.md`](bibliografia.md) | Referencias por tarea. |
| [`diseno/`](diseno/) | Especificación de diseño de la Tarea 6. |
| [`latex/`](latex/) | Fuentes LaTeX de los PDF, el generador de figuras (`figuras.py`) y el script de compilación (`compilar.sh`). |

## Regenerar los PDF

Requiere TeX Live (lualatex, latexmk, beamer con el tema metropolis).

```bash
# desde la raíz del repositorio
uv run pulso credit configs/credit.yaml              # resultados de la Tarea 6
PYTHONPATH=src uv run python docs/latex/figuras.py    # figuras -> docs/latex/figuras/

bash docs/latex/compilar.sh                           # todos los PDF -> docs/Presentaciones/
```

Las figuras de las tareas 1–5 se toman de `figures/` (generadas por los notebooks). Las
capturas del manual (`latex/figuras/manual/`) se tomaron de la aplicación en ejecución.
