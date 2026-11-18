# finnet · riesgo sistémico en redes de pagos

Simulador y mapa de riesgo para redes de pagos inmediatos tipo **Bre-B**. Un solo
modelo dinámico, construido en capas, alimenta una app web en la que se construye
una red, se diagnostica su estabilidad, se aplica un choque y se ve la cascada de
impagos. Los modelos predictivos dan el mapa de riesgo en milisegundos.

> Parámetros **ilustrativos**, no calibrados con datos reales de Bre-B ni de ningún
> sistema de pagos.

![Simulador de choques](figures/app_simulador.png)

![Mapa de riesgo](figures/app_mapa_riesgo.png)

## El modelo

$$
\frac{dx_i}{dt} = -\alpha x_i + \beta\sum_j W_{ij}x_j - D_0(Lx)_i + \gamma\,k_i(1-r_i) + \eta_i(t)
$$

$$
\frac{d\theta_i}{dt} = \omega_i + \frac{K}{\langle k\rangle}\sum_j W_{ij}\sin(\theta_j-\theta_i),
\qquad x_i > x_i^{(c)} \Rightarrow \text{impago: } W_{\cdot i}\leftarrow 0
$$

| Fase | Término | Indicador | Versión de la app |
|---|---|---|---|
| 1 · Base | $-\alpha x + \beta W x$ | $R = \beta\rho(W)/\alpha$ | v0.1 Construir red, Diagnóstico |
| 2 · Redistribución | $-D_0 L x$, $L = D_\text{out}-W$ | $\sigma$, $\tau_\text{rec}$, $\lambda_2$ | v0.2 Diagnóstico ampliado |
| 3 · Ritmo de pagos | Kuramoto + $\gamma k_i(1-r_i)$ | $r$, $r_i$ | v0.3 Ritmo de pagos |
| 4 · Impagos | umbral $x_i^{(c)}$, cascadas | % en impago, DebtRank | v0.4 Simulador de choques |
| 5 · Experimentos | 20 000 escenarios + 300 redes | dataset Parquet | v0.5 Explorar escenarios |
| 6 · Predicción | logística, XGBoost, GCN, SHAP | ROC-AUC, Spearman | v0.6 Mapa de riesgo |
| 7 · Publicación | pruebas de interfaz, Docker | — | v1.0 |

Convenciones: `W[i, j]` es el pago **de j hacia i**; el tiempo se mide en ciclos de
pago (1 = un día hábil); el dinero en "pagos medios por ciclo" (peso medio de las
aristas = 1). Detalle y supuestos en la página *Acerca del modelo* de la app.

## Arquitectura: tres capas

```text
┌──────────────────────────────┐
│ app/  (Streamlit)            │  interfaz: solo llama al servicio
└──────────────┬───────────────┘
┌──────────────▼───────────────┐
│ finnet/service.py            │  servicio: dicts y DataFrames listos para mostrar
└──────────────┬───────────────┘
┌──────────────▼───────────────┐
│ finnet/ núcleo               │  networks · model · integrate · spectral ·
│                              │  cascades · metrics · experiments · ml/
└──────────────────────────────┘
```

```text
finnet/
├── configs/        default.yaml, breb_example.yaml, sweep.yaml, train.yaml
├── src/finnet/     núcleo y servicio (cli.py: `finnet sweep | train`)
├── app/            Inicio.py + pages/1..7
├── tests/          una prueba por resultado analítico (test_fase1..6, test_app)
├── notebooks/      01..05: solo importan finnet y grafican
├── data/           escenarios.parquet, nodos.parquet, redes.parquet, networks.npz
├── models/         modelos versionados + metrics.json + comparison.csv
└── figures/        figuras clave de cada fase
```

## Uso local

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # entorno e instalación
uv run pytest                             # todas las pruebas (-m "not slow" para las rápidas)
uv run streamlit run app/Inicio.py        # la app en http://localhost:8501

uv run finnet sweep configs/sweep.yaml    # regenera el dataset (~20 min en 8 núcleos)
uv run finnet train configs/train.yaml    # reentrena y versiona los modelos
```

Con Docker:

```bash
docker build -t finnet .
docker run -p 8501:8501 finnet
```

### Problemas conocidos en macOS

- `ModuleNotFoundError: No module named 'finnet'` dentro de `~/Documents`: macOS marca
  el archivo `.pth` de la instalación editable como oculto y Python ≥ 3.12 lo ignora.
  Solución: `chflags -R nohidden .venv` (las pruebas no dependen de esto:
  `pythonpath = ["src"]` en `pyproject.toml`).
- torch y xgboost traen cada uno su libomp; `finnet.ml` importa xgboost primero y deja
  torch en un hilo para evitar segmentation faults (ver `src/finnet/ml/__init__.py`).

## Resultados de validación

Las pruebas comparan contra resultados analíticos: RK4 contra $e^{Mt}x_0$ (error
relativo < 1e-6), conservación de $\sum x$ (< 1e-10), tasa $\alpha + D_0\lambda_2$ en
redes simétricas (1 %), Kuramoto en red completa con frecuencias de Lorentz
($r = \sqrt{1-2\Delta/K}$, simulado 0.7076 contra 0.7071 en $K = 2$), Eisenberg–Noe
resuelto a mano, y determinismo bit a bit del dataset.

Modelos (validación `GroupKFold` de 5 pliegues por red, predicciones fuera de pliegue;
detalle en [`models/comparison.csv`](models/comparison.csv) y ficha del dataset en
[`data/README.md`](data/README.md)):

| Modelo | Tarea | Métricas | Tiempo |
|---|---|---|---|
| Regresión logística | colapso global | ROC-AUC 0.957 · PR-AUC 0.741 · Brier 0.083 | 0.2 ms |
| XGBoost | colapso global | ROC-AUC **0.980** · PR-AUC **0.880** · Brier 0.038 | 0.2 ms |
| XGBoost | cascada por entidad | R² 0.50 · MAE 0.088 · Spearman intra-red 0.30 | 0.5 ms / red |
| GCN (3 capas + residual) | cascada por entidad | R² **0.60** · MAE 0.075 · Spearman intra-red 0.29 | 0.7 ms / red |

Una simulación cuesta ~0.09 s por escenario: responder por una red de 125 entidades
con un modelo de nodos es ~10⁴ veces más rápido que simularla.

Límite conocido: los modelos de nodos aciertan bien el **nivel** de riesgo de una red
(R²) pero ordenan solo moderadamente a las entidades dentro de ella (Spearman ≈ 0.3).
La varianza intra-red es solo el 14 % de la total y la mayoría de redes tienen muchos
empates (cascada 0 o colapso total). Por eso la página *Mapa de riesgo* trae el botón
"verificar con simulación".

PINN (opcional, notebook 06): recupera α, β, γ y K de trayectorias simuladas con error
relativo < 30 % (tolerancia en `configs/pinn.yaml`): γ y K < 1 %, α 7 %, β 23 %
(el peor identificado: su efecto se confunde con el de α cuando el déficit se relaja).

## Después de la v1.0

API con FastAPI sobre `service.py`, módulo de fraude con los datos sintéticos de IBM,
retroalimentación del déficit sobre el ritmo de pago y mecanismos de ahorro de
liquidez.

## Referencias

Newman (2010); Barabási (2016); Chung (1997); Acebrón et al. (2005); Rodrigues et al.
(2016); Eisenberg y Noe (2001); Acemoglu et al. (2015); Gai y Kapadia (2010);
Battiston et al. (2012); Watts (2002); Bech y Garratt (2006); Merrouche y Schanz
(2008); Ikeda et al. (2011); Kipf y Welling (2017).

Semillero de Modelamiento de Sistemas Naturales Multiescala · Universidad Distrital
Francisco José de Caldas.
