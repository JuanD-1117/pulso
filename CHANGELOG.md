# Changelog

Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.1.0/); versionado
semántico.

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
