# Riesgo crediticio híbrido en una red multicapa de clientes (finnet v2.0)

Fecha: 2026-11-18 · Estado: aprobado por el usuario en conversación.

## Pregunta de investigación

> ¿La **dinámica** de la red de pagos de los clientes de un banco (sincronía de sus
> ciclos de pago) aporta información de riesgo de impago que ni el score clásico ni
> las variables **estáticas** de red capturan?

Las tareas 1–5 de la propuesta del semillero se conservan (motor físico, espectro,
Kuramoto, cascadas, predicción). Esta extensión es la **Tarea 6**: aplicar el motor a
la cartera de un banco y medir, con un control nulo, cuánto aporta la dinámica.

## Mundo sintético (subproyecto 1, `finnet.world`)

- Entidades: personas (P, 2000) y firmas (F, 100), clientes de un mismo banco.
- Columnas comunes observables: `income` (SMMLV/mes; ventas para firmas),
  `debt_ratio` (deuda/ingreso o apalancamiento), `age_years`, `tenure_months`,
  `n_products`, `past_due_12m`, `formal`, `sector`, `n_employees`.
- Informalidad: 55 % de las personas sin empleador (orden de magnitud DANE).
- Tamaño de firmas: Pareto; los empleados formales se asignan ∝ tamaño.
- Capas (convención del repo `W[i, j]` = j le paga a i, por mes):
  - `N` nómina F→P: peso = salario.
  - `S` proveedores F→F: estructura sectorial insumo-producto.
  - `T` transferencias P→P: SBM por comunidades + sesgo a compañeros de trabajo.
- Ritmos: calendario mensual (ω = 2π/mes) o quincenal (ω = 4π/mes). Formales heredan el
  calendario del empleador con dispersión pequeña; informales, dispersión grande.
  Fase inicial a partir del día de pago.
- Verdad del terreno: PD anual idiosincrática `truth_pd` = logística conocida sobre las
  variables clásicas + componente latente no observable. Intercepto ajustado a la
  tasa base (5 % P, 3 % F). Las columnas `truth_*` nunca llegan a un modelo: la
  validación de esquema lo impide.
- Fuera de alcance v2.0: plazos de factura y fecha de cuota (no afectan la dinámica;
  se agregarían como campos sin uso).

## Motor multicapa (subproyecto 2)

Generalización compatible del motor actual (`Dynamics`, `simulate`):

$$\dot x_i = -\alpha x_i + \sum_\ell \beta_\ell \sum_j \frac{W^\ell_{ij}}{s_i} x_j
  + \gamma \frac{k_i}{s_i}(1 - r_i) + \sigma\,\xi_i(t)$$

$$\dot\theta_i = \omega_i + \sum_\ell K_\ell \sum_j \frac{W^\ell_{ij}}{k_i}
  \sin(\theta_j - \theta_i)$$

- `x_i` adimensional: déficit como fracción del ingreso total `s_i` (incluye ingreso
  externo a la red). `k_i` = ingreso que llega por la red.
- Acople de fase normalizado por fila: cada cliente ajusta su ritmo según la
  participación de cada pagador en su ingreso.
- `Coupling(stress, phase, income)` reemplaza a `βW`, `(K/⟨k⟩)W` y la escala; con
  `Coupling=None` el motor es bit a bit el de v1.
- Matrices `scipy.sparse` (2100 nodos).
- Umbral `x_c,i = c · (pd_ref / truth_pd_i)^κ · ε_i`: mayor PD, menor colchón.
- Impagos idiosincráticos programados: τ_i ~ Exp(−ln(1−PD_i)/12); causa
  `idiosincratico`. Cada impago anula los pagos salientes y salta `λ·W_ij/s_i` en sus
  receptores (pérdida de ingreso).
- Choque nuevo `kind="default"`: quiebra de una entidad en `t0` (p. ej. un empleador).
- Ruido aditivo Euler–Maruyama con semilla explícita.
- Unidades: 1 ciclo = 1 mes. Observación 6 meses, predicción 12 meses.

## Registro de transacciones y estimación (subproyecto 3)

- `emit_transactions(world, traj)`: cada vez que la fase de un pagador cruza 0 mod 2π
  emite un pago a cada receptor (monto = `W_ij · 2π/ω_j` con ruido lognormal) y la
  cuota al banco (`dst = -1`, capa `C`). Jitter de ±días compartido por cruce. Los
  impagados dejan de emitir.
- `finnet.credit.estimate`: desde el registro de la ventana de observación estima
  `Ŵ^ℓ`, la fase por eventos θ̂_i(t) = 2π(k + (t − t_k)/(t_{k+1} − t_k)) (Pikovsky et
  al., 2001), `r̂_i(t)`, desajuste de frecuencia con los pagadores, regularidad de
  períodos e ingreso perdido por pagadores que dejaron de pagar.

## Riesgo crediticio híbrido (subproyecto 4)

Tres niveles de variables, mismo modelo y mismas particiones:

1. **Clásico**: variables del cliente.
2. **+ Red estática**: fuerza de entrada/salida por capa, concentración del ingreso,
   PageRank, tamaño y variables del empleador, riesgo medio de los pagadores.
3. **+ Red dinámica**: r̂ medio y su variabilidad, desajuste de frecuencia, regularidad
   de períodos propia y de pagadores, ingreso perdido.

Mundos: **nulo** (β = γ = λ = 0; solo impagos idiosincráticos y ruido) y **red**
(acoples encendidos). En el nulo, los niveles 2–3 no deben mejorar; si mejoran hay
fuga. Validación `GroupKFold` por mundo; regresión logística y XGBoost; AUC, PR-AUC,
KS, Brier, error de calibración; diferencias pareadas por partición.

## App y documentación (subproyecto 5)

- Páginas *Cartera multicapa* (mundo, capas, sincronía, quiebra de un empleador) y
  *Riesgo crediticio híbrido* (comparación de niveles y mundos, riesgo por cliente).
- Encuadre "redes de pagos digitales"; Bre-B queda como ejemplo.
- README, CHANGELOG 2.0.0, Tarea 6 y bibliografía (Eisenberg y Noe 2001, Shi y Malik
  2000, Letizia y Lillo 2019, Merton 1974, Boccaletti et al. 2014, Pikovsky et al.
  2001).

## Pruebas

Determinismo por semilla; tipos de capa y empleador único; flujo de transacciones ≈ W;
número de pagos ≈ ωT/2π; tasa base ≈ objetivo; `truth_*` vetadas; motor v1 intacto con
`Coupling=None`; recuperación de Ŵ y de fases en osciladores libres; en el mundo nulo
el nivel 3 no mejora al nivel 1 más allá del ruido; en el mundo red sí mejora.
