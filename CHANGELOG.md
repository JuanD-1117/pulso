# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versionado
semántico.

## [3.0.1] · 2026-11-18 · Presentación completa

### Cambiado
- Presentación ampliada a 56 diapositivas: cada tarea con formulación, método,
  verificación y resultados; la tarea 6 paso a paso (generación de la cartera, verdad
  del terreno, modelo multicapa, transacciones, estimación, variables y protocolo).

### Agregado
- Figuras: comunidades con Laplaciano normalizado, sincronización por grupo,
  calibración y SHAP del modelo híbrido, y sensibilidad de ΔAUC a γ
  (`docs/latex/figuras.py`; resultado en `docs/latex/figuras/sensibilidad_gamma.csv`).

## [3.0.0] · 2026-11-18 · PULSO

### Cambiado (incompatible)
- El proyecto pasa a llamarse **PULSO** (Pagos, Umbrales, Liquidez y Sincronización
  Operativa). Paquete `finnet` → `pulso` (`import pulso`), comando `uv run finnet` →
  `uv run pulso`, variables `FINNET_*` → `PULSO_*`, imagen Docker `pulso`.
- Presentación reescrita en registro formal; títulos de la guía revisados.
- Documentos: `docs/pulso_guia_completa.pdf` y `docs/pulso_presentacion.pdf`, con
  capturas de la app actualizadas.

## [2.0.1] · 2026-11-18 · Documentación

### Agregado
- `docs/finnet_guia_completa.pdf`: guía desde cero (idea, base, resultados, montaje).
- `docs/finnet_presentacion.pdf`: presentación.
- Fuentes LaTeX y generador de figuras en `docs/latex/`; índice en `docs/README.md`.

### Cambiado
- Especificación de diseño movida a `docs/diseno/`; limpieza de temporales y cachés.

## [2.0.0] · 2026-11-18 · Tarea 6: riesgo crediticio híbrido

### Agregado
- `finnet.world`: mundo sintético multicapa (personas, firmas; nómina, proveedores,
  transferencias), verdad del terreno separada (`truth_*`, vetada por esquema) y
  registro de transacciones generado por los cruces de fase.
- Motor multicapa: `Coupling` (acoples por capa de déficit y de fase, déficit
  adimensional), matrices `scipy.sparse`, impagos idiosincráticos programados, ruido
  de Euler–Maruyama con semilla y choque `kind="default"` (quiebra de un empleador).
- `finnet.credit`: estimación desde transacciones (Ŵ, fase por eventos, r̂_i),
  variables de tres niveles (clásico, red estática, red dinámica), evaluación con
  `GroupKFold` por mundo, diferencias pareadas con IC 95 % y control nulo.
- CLI `finnet world` y `finnet credit`; modelos versionados en `models/credit/v1`.
- Páginas *Cartera multicapa* y *Riesgo crediticio híbrido*.
- `docs/bibliografia.md` y especificación de diseño en `docs/diseno/`.

### Cambiado
- Encuadre: "redes de pagos digitales"; Bre-B queda como escenario de ejemplo.
- `validate_schema` pasa a `finnet.schema` (se reexporta desde `finnet.experiments`).
- Sin choque, los impagos por umbral se etiquetan `umbral` (antes
  `previo_al_choque`).
- `use_container_width` reemplazado por `width="stretch"` (Streamlit lo retira).

### Resultado
- Mundo con red: la dinámica mejora sobre la red estática ΔAUC +0.011 (logística) y
  +0.014 (XGBoost), IC 95 % por encima de 0; mundo nulo: ΔAUC ≈ 0.

## [1.0.1] · 2026-11-18

### Corregido
- Impagos ocurridos antes del choque (fragilidad previa con γ > 0) se marcan como
  `previo_al_choque` y ya no cuentan como contagio; `risk_summary` reporta
  `n_defaulted_before_shock` y `time_to_first_contagion` solo mira `t ≥ t0`.
- Comunidades espectrales con el Laplaciano normalizado (corte normalizado de
  Shi y Malik): particiones balanceadas en redes núcleo-periferia.
- Escenario Bre-B estilizado con `gamma: 0.0`: todo impago lo causa el choque.

### Cambiado
- Animación de cascada resalta las entidades chocadas (contorno y tamaño).
- Círculo de fases muestra el estado final al cargar; ▶ reproduce desde t = 0.
- *Ritmo de pagos*: aviso cuando γ = 0 (no hay acople ritmo → déficit) y sin
  delta espurio en el déficit.

## [1.0.0] · 2026-11-17

### Agregado
- Página *Acerca del modelo* con ecuaciones, supuestos y límites.
- Escenario de ejemplo "Bre-B estilizado" (núcleo-periferia, 245 entidades,
  parámetros ilustrativos).
- Pruebas de interfaz con `streamlit.testing.v1.AppTest`: cada página carga sin errores.
- `Dockerfile` (torch CPU) y construcción de la imagen en GitHub Actions.
- Exportación de indicadores, eventos y filtros a CSV.
- Límite de simulación en vivo (300 entidades); por encima, solo predicción.

## [0.6.0] · Mapa de riesgo
- `finnet.ml`: regresión logística y XGBoost para colapso global; XGBoost y GCN
  (PyTorch Geometric) para el tamaño de cascada por entidad.
- Validación `GroupKFold` por red; ROC-AUC, PR-AUC, Brier, Spearman por red y
  milisegundos por predicción contra segundos por simulación.
- Modelos versionados en `models/<nombre>/<versión>/` con `metrics.json`.
- Página *Mapa de riesgo*: predicción instantánea, explicación SHAP y botón
  "verificar con simulación".

## [0.5.0] · Explorar escenarios
- `finnet.experiments`: hipercubo latino con densidad extra cerca de $R = 1$,
  paralelismo con joblib y semillas por fila (determinismo bit a bit).
- CLI `finnet sweep` que genera `escenarios`, `nodos` y `redes` en Parquet con
  validación de esquema.
- Página *Explorar escenarios*: filtros, ventana de contagio y comparación lado a lado.

## [0.4.0] · Simulador de choques (producto mínimo)
- Umbrales de impago, cascadas instantáneas y registro de eventos.
- Choques de retiro de liquidez y caída operativa.
- `cascades.py`: Eisenberg–Noe, DebtRank y Watts.
- Página *Simulador de choques* con cascada animada y ranking de entidades peligrosas.

## [0.3.0] · Ritmo de pagos
- Fases de Kuramoto vectorizadas y término de desincronización $\gamma k_i(1-r_i)$.
- Página *Ritmo de pagos*: fases animadas, $r(t)$ y desconexión de entidades.

## [0.2.0] · Diagnóstico ampliado
- Redistribución conservativa $-D_0Lx$; estabilidad $\sigma$, tiempo de recuperación,
  $\lambda_2$ y comunidades de Fiedler.

## [0.1.0] · Construir red y diagnóstico
- Generadores Erdős–Rényi, Barabási–Albert y núcleo-periferia con montos lognormales.
- Integrador RK4, índice de amplificación $R$ y ranking por vector propio.
- App Streamlit con páginas *Construir red* y *Diagnóstico*.
