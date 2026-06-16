# Trabajo Práctico Final — Análisis de Series de Tiempo

**Universidad de Buenos Aires — Especialización en Inteligencia Artificial**  
**Materia:** Análisis de Series de Tiempo 1  
**Docente:** Camilo Argoty  

**Integrantes:**
- Gaspar Acevedo Zain (código a2101)
- Rodrigo Lauro (código a2120)

**Fecha:** 18 de junio de 2026

---

## Introducción: propósito, consignas y elección del dataset

### Propósito del trabajo

Este trabajo práctico aplica métodos clásicos de análisis de series de tiempo al pronóstico del **volumen horario de tráfico vehicular** en la autopista interestatal I-94 (Minneapolis–Saint Paul, Minnesota, EE.UU.). El objetivo central es **comparar modelos de pronóstico** sobre un mismo periodo de evaluación, cuantificar si las variables meteorológicas y de feriado mejoran un modelo SARIMA univariado, y documentar de forma reproducible todo el pipeline: limpieza, exploración, ajuste, diagnóstico y validación out-of-sample.

El enfoque prioriza la **estadística tradicional** (suavizado exponencial, ETS, Box–Jenkins) complementada con Prophet como baseline alternativo y un análisis ARCH/GARCH sobre residuos. El código está organizado en cuatro notebooks ejecutables (`01` a `04`) y este informe sintetiza sus resultados.

### Dataset elegido y motivación

Se utilizó el dataset **Metro Interstate Traffic Volume** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)), que registra el volumen de vehículos por hora en un punto de la I-94 oeste, junto con variables meteorológicas y feriados, entre 2012 y 2018.

**Por qué se eligió este dataset:**

| Criterio | Motivo |
|----------|--------|
| Serie horaria nativa | No requiere resampleo desde otra frecuencia |
| Un solo archivo comprimido | Facilita reproducibilidad |
| Sin valores faltantes en atributos (UCI) | Reduce limpieza extrema |
| Exógenas continuas y dummy de feriado | Permite SARIMAX |
| Estacionalidad diaria y semanal visible | Adecuada para SARIMA con $s=24$, ETS y Prophet |
| Fenómeno interpretable | Hora pico, fines de semana y clima tienen sentido físico |

<!-- $s = 24$ (periodo estacional) -->

**Subperiodo 2016–2017:** se trabajó con dos años completos (2016-01-01 a 2017-12-31) en lugar de los $48204$ registros de 2012–2018.

Esta decisión equilibra **estacionalidad anual** (dos ciclos completos) con **costo computacional**. Ajustar SARIMA/SARIMAX sobre $48204$ observaciones horarias en un entorno local resultó inviable (memoria y tiempo). Se reduce a $16551$ registros tras deduplicación, lo que permite mantener la estructura del fenómeno sin saturar el hardware.

**Partición train/test:** train se compone de $15879$ horas (registros), mientras que test de $672$. Estas $672$ corresponden a las últimas cuatro semanas del período, es decir, desde *2017-12-03 20:00* hasta *2017-12-31 23:00*.

## 1. Planteamiento de la pregunta de investigación

> **¿Qué modelo clásico de series de tiempo pronostica mejor el volumen horario de tráfico en la I-94 oeste?**
> **¿Las variables meteorológicas y de feriado mejoran el pronóstico frente a un SARIMA univariado?**

Esta pregunta separa dos ejes:

1. **Comparación de modelos clásicos** (Holt-Winters, ETS, SARIMA, SARIMAX) bajo las mismas métricas out-of-sample.
2. **Aporte de exógenas** (temperatura, lluvia, nieve, feriado) dentro del marco Box–Jenkins, contrastando SARIMAX vs SARIMA univariado.

Además, se incluye **Prophet** como baseline alternativo univariado y **ARCH/GARCH** para diagnosticar heteroscedasticidad en residuos de SARIMAX.

**Criterio de evaluación principal:**

- RMSE (raíz del error cuadrático medio) en test, en vehículos/hora.
- MAE como complemento interpretable.
- MAPE solo como referencia secundaria (se distorsiona cuando el tráfico nocturno es muy bajo).

---

## 2. Descripción de los datos

### Dataset y tipo de datos

- **Registros originales:**
  - $48204$ filas, $9$ columnas.
- **Subperiodo analizado:**
  - 2016-01-01 a 2017-12-31.
  - $16551$ observaciones horarias tras eliminar $3360$ timestamps duplicados.
- **Huecos horarios:**
  - $993$ horas faltantes respecto a una malla regular.

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `date_time` | datetime | Marca temporal horaria (índice de la serie). |
| `traffic_volume` | int | **Variable objetivo:** vehículos/hora. |
| `temp` | float | Temperatura en **Kelvin**. |
| `rain_1h` | float | Milímetros de lluvia acumulados en la hora. |
| `snow_1h` | float | Milímetros de nieve acumulados en la hora. |
| `clouds_all` | int | Porcentaje de nubosidad (0–100). |
| `weather_main` | categórica | Categoría meteorológica (Rain, Snow, Clear, etc.). |
| `weather_description` | categórica | Descripción textual del clima. |
| `holiday` | categórica / nula | Nombre del feriado si aplica; nulo si no es feriado. |

**Variables derivadas para modelado:**

| Variable | Construcción | Uso |
|----------|--------------|-----|
| `is_holiday` | $1$ si `holiday` no es nulo, $0$ en caso contrario | Exógena SARIMAX |
| `hour`, `weekday` | Extraídas de `date_time` | EDA |

Se **excluyeron** `weather_main` y `weather_description` del modelado por ser categóricas de alta cardinalidad. Se priorizan exógenas numéricas y una dummy de feriado.

### Estadísticas descriptivas del subperiodo

**Variable objetivo `traffic_volume`:**

| Estadístico | Valor |
|-------------|-------|
| Media | $3290$ veh/h. |
| Desvío estándar | $1966$ veh/h. |
| Mínimo | $0$ |
| Máximo | $7280$ |

**Patrones temporales:**

| Patrón | Media (veh/h) |
|--------|---------------|
| Hora pico (7–9 h y 16–18 h) | $4865$ |
| Valle nocturno (2–4 h) | $510$ |
| Días laborables | $3543$ |
| Fin de semana | $2662$ |
| Feriados | $910$ |
| No feriados | $3293$ |

- La relación hora pico / valle (~9,5×) confirma estacionalidad diaria fuerte.
- El tráfico en feriados es mucho menor que en días normales (910 vs 3.293 veh/h).
  - Un feriado se comporta más como un día de muy bajo volumen que como un laborable típico.

**Interpretación de las medias:**

- **Hora pico (4.865 veh/h):** en las ventanas 7–9 h y 16–18 h converge el commute matutino y vespertino; es el régimen que más tensiona los modelos (amplitud alta, variabilidad entre días).
- **Valle nocturno (510 veh/h):** entre 2–4 h el volumen cae casi un orden de magnitud; errores absolutos pequeños aquí pueden generar **MAPE enormes** si el denominador es cercano a cero.
- **Laborables vs fin de semana (3.543 vs 2.662):** ~25 % menos tráfico en fin de semana; estacionalidad **semanal** real que $s=24$ no separa por construcción.
- **Feriados (910 veh/h):** efecto fuerte pero pocas observaciones, lo que dificulta estimarlo en SARIMAX (`is_holiday` no significativo).

**Correlaciones lineales con `traffic_volume`:** todas débiles (|r| < 0,15). La mayor es con `temp` (r ≈ 0,13). Esto indica que, a nivel global, el clima **no explica** la mayor parte de la varianza del tráfico (dominada por hora del día y día de la semana). No descarta efecto **condicional** en SARIMAX: la lluvia puede importar cuando controlamos la dinámica propia de la serie.

**Exógenas (medias):** temp ≈ 282 K (~9 °C); rain_1h ≈ 0,64 mm/h (mayormente seco); snow_1h ≈ 0; is_holiday ≈ 0,001 (feriados raros en el subperiodo).

---

<!-- ## 3. Descripción de los modelos

En esta sección se describe **por qué se eligió cada familia de modelos**, **qué permite hacer cada una**, **cómo leer su notación** (letras, números y subíndices), y cómo interpretar gráficos y resultados numéricos. El código completo está en los notebooks `01_eda_preprocesamiento.ipynb` a `04_comparacion_conclusiones.ipynb`.

### 3.0 Por qué se estudiaron estos modelos y qué permite hacer cada uno

El TP exige al menos **tres tipos** de modelos de series de tiempo. Se eligieron **cinco familias** porque cubren enfoques complementarios: un baseline rápido (suavizado exponencial), modelos paramétricos con estacionalidad fija (ETS, SARIMA), extensión con variables externas (SARIMAX), un benchmark moderno flexible (Prophet) y diagnóstico de varianza del error (GARCH). Ninguno reemplaza a los demás; juntos permiten responder la pregunta de investigación con evidencia numérica y gráfica.

| Modelo | Por qué se incluyó | Qué permite hacer |
|--------|-------------------|-------------------|
| **Descomposición clásica** | Explorar tendencia, estacionalidad y residuo antes de modelar | Visualizar si el ciclo diario domina; comparar formulación aditiva vs multiplicativa; orientar la elección ETS |
| **Holt-Winters / ETS** | Baseline obligatorio en series con estacionalidad determinística; rápido y interpretable | Pronosticar $h$ pasos adelante combinando nivel, tendencia y patrón de 24 horas; establecer un **piso de desempeño** que SARIMA debe superar |
| **SARIMA** | Modelar **autocorrelación** (memoria temporal) además del ciclo diario | Capturar dependencia entre horas consecutivas y entre mismas horas de días distintos; seleccionar órdenes con AIC/BIC |
| **SARIMAX** | Responder si **clima y feriados** mejoran el pronóstico | Incorporar regresores externos (`temp`, lluvia, nieve, feriado) sobre la dinámica SARIMA |
| **Prophet** | Benchmark alternativo con estacionalidades múltiples sin fijar un solo $s$ | Estimar tendencia + ciclos diario/semanal/anual; contextualizar el techo de RMSE alcanzable |
| **ARCH / GARCH** | Diagnosticar si el error tiene varianza variable (hora pico, eventos extremos) | Modelar volatilidad persistente en residuos; no mejora la media pero informa incertidumbre |

**Orden del pipeline:** EDA y descomposición (NB1–NB2) informan baselines; baselines fijan umbral RMSE; ACF/PACF y ADF guían SARIMA; SARIMAX prueba exógenas; NB4 consolida con Prophet, ARCH/GARCH y gráficos comparativos. -->

<!-- ### 3.0.1 Guía de notación: letras, números y subíndices

#### Periodo estacional: $s = 24$ y el subíndice $_ {24}$

- **$s$** (o `seasonal_periods=24`): número de observaciones que componen **un ciclo estacional completo**. Con datos **horarios**, $s = 24$ significa que el patrón se repite cada **24 horas** (un día).
- El subíndice **$_{24}$** en SARIMA$(p,d,q)(P,D,Q)_{24}$ indica que la parte **estacional** del modelo usa periodo 24. No es un parámetro extra: **identifica qué estacionalidad se modela** (la diaria, no la semanal).

#### SARIMA$(p,d,q)(P,D,Q)_{24}$ — parte regular $(p,d,q)$

| Símbolo | Nombre | Qué hace | En este TP |
|---------|--------|----------|------------|
| **$p$** | Orden autorregresivo (AR) | Usa $p$ valores **pasados** de la serie para predecir el presente | Ganador: $p=2$ (depende de las 2 horas anteriores) |
| **$d$** | Orden de integración | Número de **diferencias** regulares ($y_t - y_{t-1}$) para estabilizar media | $d=1$: una diferencia horaria |
| **$q$** | Orden media móvil (MA) | Usa $q$ **shocks** (errores) pasados | Ganador: $q=2$ |

#### SARIMA — parte estacional $(P,D,Q)_{24}$

| Símbolo | Nombre | Qué hace | En este TP |
|---------|--------|----------|------------|
| **$P$** | AR estacional | Usa el valor de **hace 24 h** (misma hora, día anterior) | $P=1$ |
| **$D$** | Diferencia estacional | Resta $y_t - y_{t-24}$ para eliminar ciclo diario | $D=1$ |
| **$Q$** | MA estacional | Usa el shock de **hace 24 h** | $Q=1$ |

**Modelo final univariado:** SARIMA$(2,1,2)(1,1,1)_{24}$ — dos AR y dos MA regulares; un AR, una diferencia y un MA estacionales con periodo 24 h.

#### Notación ETS: ETS(A,Ad,A), ETS(M,Ad,M)

La notación **ETS(error, trend, seasonal)** describe tres componentes:

| Posición | Letras posibles | Significado |
|----------|-----------------|-------------|
| **Error** | **A** (aditivo), **M** (multiplicativo) | Cómo entra el ruido: sumado o proporcional al nivel |
| **Tendencia** | **N** (ninguna), **A** (aditiva), **Ad** (aditiva **amortiguada**) | Si hay drift; **Ad** frena la extrapolación en horizontes largos |
| **Estacional** | **A** (aditiva), **M** (multiplicativa) | Patrón de 24 h sumado o multiplicado al nivel |

- **Holt-Winters amortiguado (A,Ad,A):** error aditivo, tendencia aditiva amortiguada, estacionalidad aditiva de periodo 24. Mejor clásico en test.
- **Holt-Winters (A,A,A):** igual pero tendencia **sin** amortiguar; extrapola mal a 672 h.
- **ETS(M,Ad,M) sobre $y+1$:** componentes multiplicativos; requiere desplazamiento +1 por ceros en train; falló en test.

#### Holt-Winters: parámetros de suavizado

| Símbolo | Nombre | Qué controla |
|---------|--------|--------------|
| **$\alpha$** | Suavizado de nivel | Qué tan rápido se actualiza el nivel base ante un dato nuevo |
| **$\beta$** | Suavizado de tendencia | Qué tan rápido cambia la pendiente estimada |
| **$\gamma$** | Suavizado estacional | Qué tan rápido se actualiza el patrón de las 24 horas |
| **$\phi$** | Amortiguación (solo Ad) | Entre 0 y 1; valores menores frenan más la tendencia en el futuro |

#### SARIMAX

Misma notación SARIMA más coeficientes **$\beta$** (distintos de $\beta$ de tendencia HW) sobre regresores: `temp`, `rain_1h`, `snow_1h`, `is_holiday`. Cada $\beta$ mide cuánto cambia el tráfico esperado por unidad de la exógena **manteniendo fija** la dinámica ARMA.

#### Prophet

No usa $(p,d,q)$; activa componentes booleanos: `daily_seasonality`, `weekly_seasonality`, `yearly_seasonality`. Cada uno agrega suma de funciones sinusoidales (Fourier) para capturar ese ciclo sin fijar manualmente $s=168$ para la semana.

#### GARCH(1,1)

| Símbolo | Significado |
|---------|-------------|
| **GARCH** | Generalized Autoregressive Conditional Heteroskedasticity; modela varianza del error |
| **(1,1)** | Un rezago de $\varepsilon_{t-1}^2$ y un rezago de $\sigma_{t-1}^2$ |
| **$\omega$** | Constante de la varianza |
| **$\alpha_1$** | Peso del shock cuadrado reciente |
| **$\beta_1$** | Peso de la varianza pasada; $\alpha_1 + \beta_1$ cerca de 1 implica **persistencia alta** de volatilidad |

#### Descomposición clásica: $T_t$, $S_t$, $R_t$

| Símbolo | Significado |
|---------|-------------|
| **$T_t$** | Componente de **tendencia** (nivel lento de la serie) |
| **$S_t$** | Componente **estacional** (patrón que se repite cada $s$ pasos; aquí cada 24 h) |
| **$R_t$** | **Residuo** (lo no explicado por tendencia ni estacionalidad) |
| **$period=24$** | Argumento de `seasonal_decompose`: longitud del ciclo estacional en número de observaciones |

#### Métricas de error (test)

| Símbolo | Nombre | Qué mide |
|---------|--------|----------|
| **MAE** | Mean Absolute Error | Promedio de $|y - \hat{y}|$ en veh/h; error “típico” |
| **RMSE** | Root Mean Squared Error | Raíz del promedio de $(y-\hat{y})^2$; penaliza errores grandes |
| **MAPE** | Mean Absolute Percentage Error | Promedio de $|y-\hat{y}|/|y|$ en %; inestable cuando $y$ es cercano a cero |

#### Criterios de información y tests

| Símbolo | Significado | Uso |
|---------|-------------|-----|
| **AIC** | Criterio de Akaike | Menor es mejor in-sample; penaliza pocos parámetros |
| **BIC** | Criterio bayesiano | Penaliza más parámetros que AIC; usado para elegir SARIMA |
| **$p$-valor (ADF, Ljung-Box, ARCH-LM, coeficientes)** | Probabilidad bajo $H_0$ | Si $p < 0{,}05$, hay evidencia **estadística** contra la hipótesis nula (ver interpretaciones en cada sección) | -->

---
## 3. Preprocesamiento

**Decisiones y justificación:**

| Decisión | Por qué |
|----------|---------|
| Parseo datetime + orden cronológico | Prerequisito para toda serie temporal; evita rezagos incorrectos. |
| Subperiodo 2016–2017 | Dos ciclos anuales completos; $16551$ observaciones frente a $48204$ totales, lo que hace viable SARIMA en hardware local. |
| Deduplicación ($3360$ filas) | Múltiples registros por misma hora distorsionan ACF y estacionalidad. |
| No imputar $993$ huecos | Imputar (por ejemplo: interpolar) introduciría tráfico **sintético**. |
| Dummy `is_holiday` | Convierte feriados esporádicos en regresor numérico para SARIMAX. |
| Excluir categorías de clima | Evita explosión de dummies (`weather_description` tiene decenas de niveles). |

**Gráficos exploratorios:**

### Vehículos por hora - primer semana

![Zoom semanal](capturas/notebook_1/Vehículos%20por%20hora%20-%20primer%20semana.png)

> Este zoom semanal revela dos picos por día.
> NOTA: el 2016-01-01 fue viernes.

### Boxplot por hora

![Boxplot por hora](capturas/notebook_1/Distribución%20de%20tráfico%20por%20hora%20del%20día.png)

> Los boxplots por hora confirman medianas altas en 7–9 y 16–18 h y mínimos en 2–4 h.

### Boxplot por día de la semana

![Boxplot por día](capturas/notebook_1/Distribución%20del%20tráfico%20por%20día%20de%20la%20semana.png)

> Los boxplots por día de semana muestran menor tráfico sábado/domingo.

### Boxplot feriados vs no feriados

![Boxplot feriados vs no feriados](capturas/notebook_1/Tráfico%20feriados%20vs%20no%20feriados.png)

> Este gráfico confirman una caída fuerte para los días feriados.

## 4. Descomposición ETS y baselines

Antes de ajustar modelos predictivos se exploró la estructura de la serie de entrenamiento y se fijaron baselines de referencia. El objetivo es establecer un piso de desempeño out-of-sample que SARIMA y SARIMAX deban superar, y orientar las órdenes del modelo Box–Jenkins con pruebas de estacionaridad y autocorrelación.

### 4.1 Descomposición clásica con periodo diario

Se descompuso la serie de entrenamiento en tendencia, estacionalidad y residuo mediante `seasonal_decompose` con periodo $s = 24$, porque el análisis exploratorio mostró que el ciclo dominante es el diario. Se compararon formulaciones aditiva y multiplicativa para decidir qué familia ETS conviene probar después.

**Descomposición aditiva.** En el panel superior se ve el patrón de dos picos por día superpuesto a un nivel medio estable. La componente estacional tiene amplitud del orden de ±250 veh/h, mientras que el residuo alcanza ±3500 veh/h. El ciclo diario existe, pero gran parte de la variabilidad queda fuera de la estacionalidad de 24 horas (diferencias entre laborables y fin de semana, eventos puntuales, huecos en la malla horaria). Esto anticipa límites para modelos que solo fijan $s = 24$.

![Descomposición aditiva (periodo 24)](capturas/notebook_2/Celda%202_3%20-%20descomposicion%20aditiva.png)

**Descomposición multiplicativa.** La forma multiplicativa exige valores estrictamente positivos. Como en entrenamiento hay 2 observaciones con tráfico igual a cero, se aplicó el desplazamiento $y'_t = y_t + 1$ veh/h solo para esta visualización. La estacionalidad oscila en torno a 1 (±5–10 %), lo que encaja con la intuición de que en hora pico hay más volumen relativo al día. La separación entre tendencia y estacionalidad resulta más limpia que en la aditiva, por lo que se decidió probar también ETS(M,Ad,M) sobre $y + 1$.

![Descomposición multiplicativa (periodo 24, y+1)](capturas/notebook_2/Celda%202_3%20-%20descomposicion%20multiplicativa.png)

**Conclusión de la descomposición.** La exploración visual sugiere que la amplitud del ciclo podría escalar con el nivel (argumento a favor de componentes multiplicativos), pero la validación out-of-sample será la que decida. Los boxplots por hora del análisis exploratorio confirman medianas altas entre 7–9 h y 16–18 h y mínimos entre 2–4 h, coherente con $s = 24$ para Holt-Winters y ETS.

### 4.2 Holt-Winters, ETS y evaluación de baselines

Con la descomposición como guía se ajustaron cuatro baselines sobre las 15879 horas de entrenamiento, todos con estacionalidad de periodo 24. Se usó Holt-Winters (`ExponentialSmoothing`) y la familia ETS (`ETSModel`) para cubrir la misma lógica con distintas implementaciones y especificaciones aditivas y multiplicativas.

**Holt-Winters clásico ETS(A,A,A).** Tendencia aditiva sin amortiguar y estacionalidad aditiva. Funciona como contraste: en entrenamiento el AIC es ≈ 217265, casi igual al modelo amortiguado.

**Holt-Winters amortiguado ETS(A,Ad,A).** Igual formulación aditiva pero con `damped_trend=True`, de modo que la tendencia se frena al extrapolar muchas horas adelante. Obtiene el mejor AIC in-sample entre los aditivos (≈ 217192).

**ETS(A,Ad,A).** Misma idea de error, tendencia amortiguada y estacionalidad aditiva vía `ETSModel`. El AIC in-sample es ≈ 262413, peor que Holt-Winters, pero en test se comporta de forma similar al amortiguado.

**ETS(M,Ad,M) sobre $y + 1$.** Motivado por la descomposición multiplicativa. Requiere el desplazamiento por los ceros en entrenamiento. Converge con AIC ≈ 293248.

**Pronóstico out-of-sample.** Se generaron pronósticos de 672 pasos (las cuatro semanas de test) para cada baseline y se calcularon MAE, RMSE y MAPE.

| Modelo | AIC (train) | MAE (test) | RMSE (test) | MAPE (test) |
|--------|-------------|------------|-------------|-------------|
| Holt-Winters amortiguado (A,Ad,A) | ≈ 217192 | 1513 | **1804** | 170 % |
| ETS(A,Ad,A) | ≈ 262413 | 1613 | 1848 | 158 % |
| Holt-Winters (A,A,A) | ≈ 217265 | 8593 | 9896 | 732 % |
| ETS(M,Ad,M) y+1 | ≈ 293248 | 25082 | 29094 | 1993 % |

La diferencia entre Holt-Winters clásico y amortiguado es metodológica: el AIC in-sample casi no los separa, pero al pronosticar 672 horas la tendencia no amortiguada extrapola un drift lineal demasiado lejos y el RMSE se multiplica por más de cinco. La amortiguación estabiliza el nivel a escala mensual, coherente con una autopista en régimen estacionario. El modelo multiplicativo, pese a la descomposición exploratoria, no generaliza en test.

**Conclusión de baselines.** El mejor clásico por RMSE es Holt-Winters amortiguado (1804 veh/h). ETS(A,Ad,A) queda cerca (1848 veh/h). Estos valores fijan el umbral que SARIMA debe intentar superar. El error restante se concentra sobre todo en el valle nocturno, donde los modelos con $s = 24$ aditivo mantienen un piso elevado respecto al tráfico real (~500 veh/h).

### 4.3 Estacionaridad y autocorrelación

Para orientar el ajuste SARIMA se aplicó el test ADF sobre distintas transformaciones de la serie de entrenamiento y se analizaron ACF y PACF de la serie estacionarizada (una diferencia regular y una estacional de orden 24).

**Test ADF.** Se evaluaron niveles, log(1+y), diff(1) y diff estacional (24). Con aproximadamente 15800 observaciones el test tiene alto poder y rechaza la raíz unitaria incluso en niveles ($p \approx 10^{-29}$). Eso no implica que la serie esté lista para ARIMA(0,0,0), porque persisten estacionalidad y autocorrelación. Las diferencias regulares y estacionales arrojan $p \approx 0$, lo que respalda usar $d = 1$ y $D = 1$ como punto de partida.

| Transformación | Estadístico | $p$-valor |
|----------------|-------------|-----------|
| Niveles | −16,39 | ≈ $2{,}7 \times 10^{-29}$ |
| log(1+y) | −16,71 | ≈ $1{,}4 \times 10^{-29}$ |
| Diff(1) | −24,22 | ≈ 0 |
| Diff estacional (24) | −19,55 | ≈ 0 |

**ACF y PACF.** El gráfico muestra picos en los lags 24, 48 y 72 de la ACF: la serie de hoy se parece a la de ayer a la misma hora, señal de estacionalidad diaria. El PACF corta en el lag 1 (parte regular) y en el lag 24 (parte estacional), lo que sugiere una propuesta inicial SARIMA$(1,1,1)(1,1,1)_{24}$.

![ACF y PACF de serie estacionarizada](capturas/notebook_2/Celda%202_8%20-%20ACF%20y%20PACF%20de%20serie%20estacionarizada.png)

**Conclusión.** La evidencia conjunta de descomposición, ADF y ACF/PACF justifica un SARIMA estacional con $s = 24$, $d = 1$ y $D = 1$. Las órdenes $(p,q,P,Q)$ se refinarán en la etapa de modelado con un grid acotado y criterio BIC sobre el train completo.

---

## 5. Modelado SARIMA y SARIMAX

El objetivo de esta etapa es cuantificar si SARIMA supera los baselines ETS/Holt-Winters en el mismo periodo test y, en una segunda variante, si las exógenas meteorológicas y de feriado aportan señal adicional.

### 5.1 Identificación y selección del SARIMA univariado

Se partió de la propuesta SARIMA$(1,1,1)(1,1,1)_{24}$ obtenida del análisis ACF/PACF. Para refinar las órdenes se ejecutó un grid stepwise acotado sobre una submuestra de 3000 horas (con $d = 1$ y $D = 1$ fijos), porque el ajuste automático sobre las 15879 observaciones completas agotaba la memoria disponible. El candidato $(2,1,2)(1,1,1)_{24}$ se re-estimó luego sobre el train completo con `low_memory=True`, y se comparó con la propuesta manual mediante BIC.

| Candidato | AIC | BIC |
|-----------|-----|-----|
| Manual $(1,1,1)(1,1,1)_{24}$ | 249880 | 249918 |
| Grid $(2,1,2)(1,1,1)_{24}$ | 249174 | **249228** |

Se seleccionó SARIMA$(2,1,2)(1,1,1)_{24}$ por BIC inferior en unos 690 puntos. La parte estacional $(1,1,1)_{24}$ se mantiene y los términos AR(2) y MA(2) regulares absorben memoria de corto plazo que la especificación más parsimoniosa dejaba en residuos.

**Pronóstico y comparación con baselines.** Sobre las 672 horas de test:

| Métrica | SARIMA | HW amortiguado |
|---------|--------|----------------|
| MAE | 1489 veh/h | 1513 veh/h |
| RMSE | 1898 veh/h | **1804 veh/h** |
| MAPE | 102 % | 170 % |

SARIMA mejora levemente el MAE (−24 veh/h) pero empeora el RMSE (+94 veh/h). El modelo comete errores más grandes en algunos instantes (hora pico, días atípicos), aunque el error medio absoluto sea similar. **No supera** al mejor baseline clásico.

![Pronóstico SARIMA vs tráfico real (2 semanas test)](capturas/notebook_3/Celda%203_5%20-%20Pronóstico%20SARIMA%20vs%20tráfico%20real.png)

En el gráfico de dos semanas de test el pronóstico reproduce los dos picos diarios, pero aplana el valle nocturno respecto al tráfico real (comportamiento similar a Holt-Winters y ETS). Las bandas de confianza son amplias y en algunos picos el modelo llega tarde o subestima la magnitud.

**Diagnóstico de residuos.** El test Ljung-Box rechaza ruido blanco en los residuos (estadístico ≈ 886 en lag 24, $p \approx 0$), lo que indica estructura predecible no capturada (por ejemplo estacionalidad semanal o efecto de huecos en la serie).

![Diagnóstico residuos SARIMA](capturas/notebook_3/Celda%203_6%20-%20diagnóstico%20SARIMA.png)

El panel muestra clusters de error en ciertos periodos, colas pesadas en el histograma y desviaciones en el Q-Q plot. El ajuste es útil para pronosticar pero no agota toda la estructura de la serie.

### 5.2 SARIMAX con exógenas

Se ajustó SARIMAX con las mismas órdenes $(2,1,2)(1,1,1)_{24}$ e incorporando `temp`, `rain_1h`, `snow_1h` e `is_holiday` como regresores. En test se usaron valores **observados** de clima (evaluación optimista respecto a un escenario operativo).

**Coeficientes e interpretación del $p$-valor (umbral 5 %):**

| Parámetro | Coeficiente | $p$-valor | Lectura |
|-----------|-------------|-----------|---------|
| `rain_1h` | −0,26 | ≈ 0 | Significativo. Más lluvia se asocia a menos tráfico, con efecto modesto en veh/h. |
| `temp` | +63,5 | ≈ 0 | Significativo, pero difícil de interpretar causalmente porque correlaciona con estación y hora del día. |
| `snow_1h` | +2865 | 0,00027 | Significativo pero frágil: pocos eventos de nieve y coeficiente muy grande. |
| `is_holiday` | +27,0 | 0,888 | **No significativo**. Con $p = 0{,}888$ no hay evidencia de un efecto lineal del feriado condicionado al resto del modelo. |

**Comparación SARIMA vs SARIMAX:**

| Criterio | SARIMA | SARIMAX |
|----------|--------|---------|
| AIC | 249174 | **249085** |
| BIC | 249228 | **249170** |
| MAE test | 1489 | **1411** |
| RMSE test | 1898 | **1855** |
| MAPE test | 102 % | **80 %** |

SARIMAX mejora en todos los criterios respecto al univariado (ΔRMSE ≈ 44 veh/h, ΔMAE ≈ 78 veh/h). Las exógenas **sí ayudan** dentro de Box–Jenkins, pero el RMSE test (1855 veh/h) sigue siendo mayor que el de Holt-Winters amortiguado (1804 veh/h). El clima aporta señal incremental y no reemplaza la estacionalidad determinística del baseline ETS/HW.

El Ljung-Box sobre residuos de SARIMAX mejora (estadístico ≈ 712 en lag 24 frente a 886), aunque $p \approx 0$ persiste.

---

## 6. Prophet, comparación final y diagnóstico de volatilidad

Tras fijar baselines y modelos Box–Jenkins se incorporó Prophet como benchmark univariado alternativo y se consolidaron todas las métricas en una tabla comparativa. Se añadieron gráficos de overlay y de error por hora del día, y un diagnóstico ARCH/GARCH sobre residuos de SARIMAX.

### 6.1 Prophet

Prophet se configuró con estacionalidad diaria, semanal y anual activa, sin incorporar clima. Responde a otra pregunta que SARIMAX: cuál es el techo univariado cuando no se fija un único $s = 24$ y se permiten ciclos múltiples con amplitud flexible.

En test arrojó MAE 642 veh/h, RMSE 884 veh/h y MAPE 42,5 %. Dieciocho pronósticos negativos se recortaron a cero porque el tráfico no puede ser negativo. Sin recorte el RMSE sube levemente a 893 veh/h.

Frente al mejor clásico (Holt-Winters amortiguado, RMSE 1804 veh/h), Prophet reduce el error cuadrático medio a aproximadamente la mitad. La brecha es estructural: Prophet modela estacionalidad semanal y anual además del ciclo diario y ajusta mejor la amplitud del perfil horario, evitando el piso nocturno elevado de los modelos con $s = 24$ aditivo.

### 6.2 Tabla comparativa out-of-sample

Periodo test: 672 horas (cuatro semanas). Métricas en veh/h salvo MAPE (%).

| Rank (RMSE) | Modelo | MAE | RMSE | MAPE |
|-------------|--------|-----|------|------|
| 1 | **Prophet** | 642 | **884** | 42,5 % |
| 2 | Holt-Winters amortiguado (A,Ad,A) | 1513 | 1804 | 170 % |
| 3 | ETS(A,Ad,A) | 1613 | 1848 | 158 % |
| 4 | SARIMAX | **1411** | 1855 | 80 % |
| 5 | SARIMA$(2,1,2)(1,1,1)_{24}$ | 1489 | 1898 | 102 % |

Quedaron fuera de la tabla final Holt-Winters (A,A,A) con RMSE 9896 y ETS(M,Ad,M) sobre $y+1$ con RMSE 29094, por desempeño inaceptable en test.

**Lectura de la tabla.** Prophet es el mejor global por RMSE. Entre modelos clásicos, Holt-Winters amortiguado lidera por RMSE y SARIMAX por MAE (1411 veh/h), aunque SARIMAX queda cuarto en RMSE por un pico de error a las 7 h. SARIMA univariado no supera al baseline ETS/HW pese a mayor complejidad.

### 6.3 Gráficos comparativos

**Overlay (dos semanas de test).** Se superponen el tráfico real y los tres mejores modelos por RMSE: Prophet, Holt-Winters amortiguado y ETS(A,Ad,A).

![Overlay — Real vs Prophet, HW y ETS](capturas/notebook_4/Celda%204.6%20-%20Grafico%20overlay.png)

| Serie (color) | Lectura |
|---------------|---------|
| **Real (azul)** | Dos picos diarios (~6000–6500 veh/h en hora pico) y valles nocturnos (~500 veh/h). |
| **Prophet (naranja)** | Sigue fase y amplitud del real. Valles profundos, picos vespertinos ligeramente por debajo en algunos días. |
| **HW amortiguado (rojo)** | Ciclo diario correcto pero piso nocturno ~2500–3000 veh/h cuando el real está ~500 veh/h. |
| **ETS(A,Ad,A) (verde)** | Casi superpuesto a HW amortiguado: misma limitación de amplitud. |

Donde la línea azul desciende de noche, la roja apenas baja. Ese error sistemático en madrugada concentra gran parte del RMSE de los clásicos y explica la brecha visual con Prophet.

**Error absoluto medio por hora del día.** Promedia $|y - \hat{y}|$ en test por hora del reloj (0–23).

![MAE por hora — HW, SARIMAX y Prophet](capturas/notebook_4/Celda%204.7%20-%20error%20absoluto%20medio%20por%20hora%20-%20test.png)

| Franja | HW (azul) | SARIMAX (púrpura) | Prophet (naranja) |
|--------|-----------|-------------------|-------------------|
| 0–5 h | Alto (~2300–3100) | Bajo (~400–800) | Bajo (~300–600) |
| 6–8 h | Moderado (~2000) | **Pico ~3750 en h=7** | Moderado (~1000–1600) |
| 10–15 h | ~800–1200 | ~800–1000 | ~300–600 |
| 16–20 h | ~1000–1800 | Similar a HW | ~400–800 |

No hay un único mejor modelo en todas las horas. SARIMAX gana en madrugada gracias a exógenas y dinámica ARMA, Holt-Winters es más robusto en hora pico agregada y Prophet domina en casi todo el reloj. El pico de SARIMAX a las 7 h explica por qué tiene el mejor MAE clásico (1411) pero peor RMSE que HW (1855 frente a 1804).

Promedios por régimen en test:

| Régimen | HW amortiguado | SARIMAX | Prophet |
|---------|----------------|---------|---------|
| Hora pico (7–9, 16–18 h) | 1303 | 1894 | **915** |
| Resto del día | 1585 | **1248** | **551** |

### 6.4 Diagnóstico de heteroscedasticidad (ARCH-LM y GARCH)

Ljung-Box evaluó autocorrelación en los residuos de SARIMA y SARIMAX. ARCH-LM complementa ese diagnóstico al testear si la **varianza** del error cambia en el tiempo (heteroscedasticidad).

Sobre residuos de SARIMAX in-sample, ARCH-LM arrojó estadístico 5954,93 con $p \approx 0$, por lo que se rechaza homocedasticidad. La varianza del error no es constante y puede ser mayor en hora pico o ante eventos extremos.

Se ajustó GARCH(1,1) sobre esos residuos, con $\alpha_1 \approx 0{,}076$, $\beta_1 \approx 0{,}902$ y $\alpha + \beta \approx 0{,}978$. Un valor sumado muy cercano a 1 implica persistencia alta de la volatilidad: un shock grande infla la varianza prevista durante muchas horas siguientes. GARCH no re-pronostica el nivel de tráfico en este trabajo, pero informa intervalos de confianza dinámicos. No mejora MAE ni RMSE del pronóstico puntual de la media.

---

## 7. Conclusiones

### 7.1 Respuesta a la pregunta de investigación

**¿Qué modelo clásico pronostica mejor?**  
Entre Holt-Winters, ETS, SARIMA y SARIMAX, **Holt-Winters amortiguado (A,Ad,A)** obtiene el menor RMSE out-of-sample (1804 veh/h, MAE 1513), seguido de ETS(A,Ad,A) (1848) y SARIMAX (1855). SARIMA$(2,1,2)(1,1,1)_{24}$ queda último entre clásicos (1898) y no supera al baseline pese a mayor complejidad.

**¿Las exógenas mejoran frente a SARIMA univariado?**  
**Sí, de forma incremental.** SARIMAX reduce RMSE de 1898 a 1855 (~44 veh/h) y MAE de 1489 a 1411. La lluvia tiene efecto negativo significativo. Sin embargo SARIMAX no supera a Holt-Winters amortiguado: el clima aporta señal dentro de Box–Jenkins pero no es el driver dominante.

**Hallazgo complementario.** Prophet alcanza RMSE 884 veh/h, aproximadamente la mitad del mejor clásico, al modelar estacionalidad diaria, semanal y anual con amplitud flexible. No usa clima y responde otra pregunta metodológica.

### 7.2 Conclusiones sobre el fenómeno estudiado

1. El tráfico en I-94 presenta estacionalidad diaria marcada (media hora pico 4865 veh/h frente a valle 510) y diferencias entre laborables, fines de semana y feriados.
2. Los modelos con solo $s = 24$ reproducen la forma del ciclo diario pero subestiman el valle nocturno (piso ~2500–3000 veh/h pronosticado frente a ~500 real), lo que infla RMSE y MAPE.
3. El clima tiene efecto detectable (lluvia reduce tráfico) pero correlación lineal débil, con aporte predictivo modesto frente a la estructura temporal propia de la serie.
4. La hora pico concentra los errores más grandes en modelos clásicos y SARIMAX falla especialmente a las 7 h.

### 7.3 Conclusiones sobre los modelos

| Modelo | Fortaleza | Limitación principal |
|--------|-----------|----------------------|
| HW amortiguado | Mejor clásico out-of-sample, simple, robusto en hora pico | No modela semana ni clima, piso nocturno alto |
| ETS(A,Ad,A) | Similar a HW en test | Mismo piso nocturno, AIC peor in-sample |
| SARIMA | Captura autocorrelación, BIC guía órdenes | Peor RMSE que HW, residuos no blancos |
| SARIMAX | Mejor MAE clásico, lluvia significativa | No supera HW, pico en h=7, clima observado en test |
| Prophet | Mejor RMSE global, amplitud diaria | Sin clima, 18 pronósticos recortados a 0 |
| GARCH(1,1) | Modela volatilidad persistente en residuos | No mejora pronóstico de la media |

### 7.4 Lecciones aprendidas y dificultades superadas

1. **Memoria y tiempo de cómputo.** El ajuste automático de órdenes SARIMA sobre las 15879 observaciones de entrenamiento agotó la memoria RAM al explorar combinaciones de $(p,d,q)(P,D,Q)_{24}$. Se resolvió con un grid stepwise acotado sobre 3000 horas, $d = 1$ y $D = 1$ fijos, y refit del ganador sobre el train completo con `low_memory=True`. Conviene separar identificación (submuestra o restricciones) de estimación final (train completo).

2. **AIC in-sample versus RMSE out-of-sample.** Holt-Winters clásico y amortiguado tienen AIC casi idénticos en train (≈ 217265 vs ≈ 217192), pero en 672 horas de test el RMSE diverge (9896 frente a 1804). La tendencia no amortiguada extrapola drift lineal demasiado lejos. La validación out-of-sample es imprescindible en horizontes largos.

3. **ACF/PACF orientan, BIC decide.** La propuesta $(1,1,1)(1,1,1)_{24}$ del análisis de autocorrelación fue superada por $(2,1,2)(1,1,1)_{24}$ en train completo (~690 puntos de BIC de diferencia). Mejor BIC in-sample no garantizó mejor RMSE test.

4. **993 huecos horarios sin imputar.** No se interpolaron para no introducir tráfico sintético. Puede contribuir a residuos no blancos en Ljung-Box además de la estacionalidad semanal no modelada.

5. **MAPE poco fiable con tráfico nocturno bajo.** Holt-Winters amortiguado alcanza MAPE 170 % pese a RMSE razonable, porque en horas con ~200 veh/h reales un error de ~1500 veh/h produce ratios superiores al 100 %. Conviene priorizar MAE y RMSE en veh/h.

6. **Brecha Prophet versus clásicos.** Prophet (RMSE 884) supera ampliamente a HW (1804) porque modela estacionalidad semanal y anual y permite amplitud variable del ciclo diario, estructuras que un único $s = 24$ aditivo no captura.

### 7.5 Limitaciones y trabajo futuro

1. **Alcance geográfico y temporal.** Un solo punto de la I-94 oeste, subperiodo 2016–2017, 16551 horas válidas. No se extrapola a otras autopistas ni se modelan obras o accidentes.

2. **Clima observado en test.** SARIMAX se evaluó con valores reales de temperatura, lluvia y nieve en las 672 horas de pronóstico. En producción harían falta pronósticos meteorológicos, por lo que el RMSE 1855 es un techo optimista.

3. **Feriados no significativos en SARIMAX.** El EDA muestra media 910 veh/h en feriados frente a 3293 en no feriados, pero solo hay 21 horas con `is_holiday = 1` (~0,13 % de las observaciones). En SARIMAX el coeficiente es +27 veh/h con $p = 0{,}888$, por lo que no hay evidencia estadística de efecto lineal condicionado al resto del modelo. El contraste agregado no controla hora del día ni dinámica SARIMA y muchos feriados coinciden con fines de semana. “No significativo” no implica que los feriados no importen en la práctica, sino que este modelo no pudo estimarlos con tan pocos casos.

4. **Estacionalidad semanal no modelada en clásicos.** Con $s = 24$, lunes 8 h y sábado 8 h comparten el mismo perfil horario. Prophet mitiga esto con estacionalidad semanal explícita.

5. **Modelos neuronales omitidos.** Se priorizó el pipeline clásico completo. No hay comparación con LSTM u otras redes recurrentes.

6. **Trabajo futuro.** Imputación documentada de huecos, SARIMA/SARIMAX con $s = 168$ o regresores de fin de semana, evaluación de SARIMAX con clima previsto, calendario de feriados en Prophet o más años de datos, y benchmarks con modelos secuenciales sobre la misma partición de 672 horas.
