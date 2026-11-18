# Ficha del dataset

Generado con `uv run finnet sweep configs/sweep.yaml` (semilla global `20261117`).
Todos los rangos y semillas están en [`configs/sweep.yaml`](../configs/sweep.yaml).
Es un dataset **sintético**: ninguna fila proviene de datos reales de pagos.

## Archivos

| Archivo | Filas | Contenido |
|---|---|---|
| `redes.parquet` | 300 | configuración de cada red (topología, n, grado medio, dispersión de montos, semilla) y descriptores |
| `networks.npz` | 300 matrices | `W` de cada red bajo la clave `net_<network_id>` (`W[i, j]` = pago de j a i) |
| `escenarios.parquet` | 20 000 | una corrida por fila: variables antes del choque y etiquetas después |
| `nodos.parquet` | 27 008 | una entidad por fila: variables de la entidad y de su red; etiqueta = cascada si el choque empieza en ella |

## Cómo se generó

1. **Redes**: hipercubo latino sobre topología (Erdős–Rényi, Barabási–Albert,
   núcleo-periferia: 100 de cada una), n ∈ [30, 150] (media 90), grado medio ∈ [2, 12],
   dispersión lognormal de montos ∈ [0.2, 1.5], fracción de núcleo ∈ [0.05, 0.2].
2. **Escenarios**: hipercubo latino de 13 dimensiones. $R = \beta\rho(W)/\alpha$ se
   muestrea en [0.4, 1.6] con densidad extra alrededor de 1 (zona crítica $\sigma \approx 0$):
   $R = 1 + 0.6\,\mathrm{sign}(u)|u|^2$. Demás rangos: α ∈ [0.5, 2], D₀ ∈ [0, 0.5],
   K ∈ [0, 6], γ ∈ [0, 0.08], Δω ∈ [0.1, 1], colchón c ∈ [0.2, 1.5], λ ∈ [0.3, 1.5].
   30 % de los choques son caídas operativas (duración ∈ [1, 6] ciclos); el resto,
   retiros de liquidez (tamaño ∈ [0.5, 6] ciclos de pagos salientes).
3. **Estado previo al choque**: 5 ciclos de ritmo de pagos desde fases al azar; las
   variables $r$ inicial y $r_i$ se miden ahí.
4. **Simulación**: RK4, dt = 0.02, 15 ciclos tras el choque, con impagos.
5. **Nodos**: por red se muestrea un juego de parámetros y se choca cada entidad con
   un retiro de 3 ciclos de pagos salientes.

Cada fila tiene su semilla (derivada con `SeedSequence`); la misma semilla produce la
misma fila bit a bit (`tests/test_fase5.py`).

## Etiquetas

| Columna | Definición |
|---|---|
| `collapse` | 1 si cae ≥ 30 % de las entidades en el horizonte |
| `fraction_defaulted` | fracción de entidades en impago al final |
| `unpaid_liquidity` | pagos no realizados acumulados, en ciclos de flujo total del sistema |
| `time_to_first_default` | ciclos desde el choque hasta el primer impago (NaN si no hubo: 13.1 %) |
| `cascade_size` (nodos) | fracción de las *otras* entidades que caen al chocar esta |

## Variables de la tabla `nodos`

Estructurales (grados, montos, importancia y vulnerabilidad espectral, PageRank,
participación en el flujo), de estado previo (coherencia local $r_i$, colchón
$x_i^{(c)}$), de impacto mecánico inmediato (`shock_to_buffer`, `max_dependency`,
`downstream_impact`, `first_order_defaults`, `debtrank`; ver
`experiments.first_order_impact`) y de contexto de red (n, topología, R, σ, K, γ, c, λ).

## Balance de clases

- Colapso global: **11.1 %** de los escenarios (ER 16.3 %, BA 8.7 %, núcleo-periferia 8.3 %).
- Fracción en impago: media 0.134, mediana 0.029 (distribución muy sesgada: la mayoría
  de choques se absorbe y unos pocos arrasan la red).
- Mediana de `time_to_first_default` = 0: en la mayoría de corridas la entidad chocada
  cae en el instante del choque.

## Sin fuga de información

Las variables se calculan con `experiments.scenario_features` y
`experiments.node_features`, que solo reciben la red, los parámetros, la descripción
del choque y el estado previo; nunca la trayectoria posterior. Para validar, los
modelos separan **redes completas** (`GroupKFold` por `network_id`).
