# Trabajo Práctico Final — Análisis de Series de Tiempo

**Universidad de Buenos Aires — Especialización en Inteligencia Artificial**  
**Materia:** Análisis de Series de Tiempo 1  
**Docente:** Camilo Argoty  

**Integrantes:**
- Gaspar Acevedo Zain (código a2101)
- Rodrigo Lauro (código aXXXX)

**Fecha:** 18 de junio de 2026

---

## Introducción: propósito, consignas y elección del dataset

### Propósito del trabajo

Este trabajo práctico aplica métodos clásicos de análisis de series de tiempo al pronóstico del **volumen horario de tráfico vehicular** en la autopista interestatal I-94 (Minneapolis–Saint Paul, Minnesota, EE.UU.). El objetivo central es **comparar modelos de pronóstico** sobre un mismo periodo de evaluación, cuantificar si las variables meteorológicas y de feriado mejoran un modelo SARIMA univariado, y documentar de forma reproducible todo el pipeline: limpieza, exploración, ajuste, diagnóstico y validación out-of-sample.

El enfoque prioriza la **estadística tradicional** (suavizado exponencial, ETS, Box–Jenkins) complementada con Prophet como baseline alternativo y un análisis ARCH/GARCH sobre residuos. El código está organizado en cuatro notebooks ejecutables (`01` a `04`) y este informe sintetiza sus resultados.

### Consignas del TP

Según el enunciado oficial, la entrega comprende:

1. **Código en Python reproducible** con: (a) limpieza y preparación de datos; (b) al menos **tres tipos de modelos** de series de tiempo; (c) pronósticos, evaluación y comparación.
2. **Informe de análisis** con: pregunta de investigación, descripción de datos, descripción de modelos (con gráficos), pruebas y validaciones, y conclusiones finales.

Este proyecto cumple ambos requisitos con cuatro notebooks y cinco familias de modelos (Holt-Winters/ETS, SARIMA, SARIMAX, Prophet, GARCH sobre residuos).

### Dataset elegido y motivación

Se utilizó el dataset **Metro Interstate Traffic Volume** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)), que registra el volumen de vehículos por hora en un punto de la I-94 oeste, junto con variables meteorológicas y feriados, entre 2012 y 2018.

**Por qué se eligió este dataset:**

| Criterio | Motivo |
|----------|--------|
| Serie horaria nativa | No requiere resampleo desde otra frecuencia |
| Un solo archivo comprimido | Facilita reproducibilidad |
| Sin valores faltantes en atributos (UCI) | Reduce limpieza extrema |
| Exógenas continuas y dummy de feriado | Permite SARIMAX (pregunta central del TP) |
| Estacionalidad diaria y semanal visible | Adecuada para SARIMA con $s=24$, ETS y Prophet |
| Fenómeno interpretable | Hora pico, fines de semana y clima tienen sentido físico |

**Subperiodo 2016–2017:** se trabajó con dos años completos (2016-01-01 a 2017-12-31) en lugar de los 48.204 registros de 2012–2018. Esta decisión equilibra **estacionalidad anual** (dos ciclos completos) con **costo computacional**: ajustar SARIMA/SARIMAX sobre ~48.000 observaciones horarias en un entorno local resultó inviable (memoria y tiempo); 16.551 registros tras deduplicación mantienen la estructura del fenómeno sin saturar el hardware.

**Partición train/test:** entrenamiento = 15.879 horas; test = **672 horas** (últimas cuatro semanas: desde 2017-12-03 20:00 hasta 2017-12-31 23:00). Horizonte fijo para comparación justa entre modelos.

### Qué significa $s = 24$ (periodo estacional)

En modelos como SARIMA, Holt-Winters y ETS, el parámetro **$s$** indica **cada cuántas observaciones se repite el patrón estacional**. Como la serie está en **frecuencia horaria**, $s = 24$ significa que el ciclo estacional tiene periodo **24 horas = 1 día calendario**:

- La observación de las **10:00** de un martes debería comportarse de forma similar a la de las **10:00** del miércoles (misma “hora del reloj”), no a la de las **09:00** del mismo martes.
- En notación de rezagos: $y_t$ se relaciona con $y_{t-24}$, $y_{t-48}$, $y_{t-72}$, etc. (misma hora en días anteriores).
- En Holt-Winters/ETS: el componente estacional tiene **24 “perfiles”** (uno por hora 0–23) que se repiten cada día.

**Por qué 24 y no otro valor:** el EDA (Notebook 1) muestra que la mayor parte de la variabilidad ocurre **dentro del día** (media hora pico 4.865 veh/h vs valle nocturno 510 veh/h). También hay diferencias entre días laborables y fin de semana (3.543 vs 2.662 veh/h), lo que sería una estacionalidad semanal ($s = 168$ horas), pero modelarla junto con $s = 24$ multiplica la complejidad. Por eso se fijó **$s = 24$ como estacionalidad principal** en los modelos clásicos, y Prophet —que no exige fijar un solo $s$— modela además ciclo semanal y anual por separado.

**Limitación explícita:** con $s = 24$ fijo, el “lunes 8:00” y el “sábado 8:00” comparten el mismo perfil estacional horario; la diferencia entre semana y fin de semana queda en tendencia, residuo o en modelos alternativos (Prophet).

---

## 1. Planteamiento de la pregunta de investigación

> **¿Qué modelo clásico de series de tiempo pronostica mejor el volumen horario de tráfico en la I-94 oeste, y las variables meteorológicas y de feriado mejoran el pronóstico frente a un SARIMA univariado?**

Esta pregunta separa dos ejes:

1. **Comparación de modelos clásicos** (Holt-Winters, ETS, SARIMA, SARIMAX) bajo las mismas métricas out-of-sample.
2. **Aporte de exógenas** (temperatura, lluvia, nieve, feriado) dentro del marco Box–Jenkins, contrastando SARIMAX vs SARIMA univariado.

Como extensión metodológica (Notebook 4), se incluye **Prophet** como baseline alternativo univariado y **ARCH/GARCH** para diagnosticar heteroscedasticidad en residuos de SARIMAX. Prophet no responde estrictamente a “modelo clásico”, pero contextualiza el techo de desempeño alcanzable con estacionalidades múltiples. La justificación detallada de cada familia y la guía de notación $(p,d,q)$, ETS(A,Ad,A), etc. están en §3.0 y §3.0.1.

**Criterio de evaluación principal:** RMSE (raíz del error cuadrático medio) en test, en vehículos/hora. MAE como complemento interpretable; MAPE solo como referencia secundaria (se distorsiona cuando el tráfico nocturno es muy bajo).

---

## 2. Descripción de los datos

### 2.1 Origen y alcance

- **Fuente:** sensores de volumen + API OpenWeatherMap (según documentación UCI).
- **Ubicación:** autopista I-94 oeste, Minneapolis–Saint Paul.
- **Archivo:** `Metro_Interstate_Traffic_Volume.csv.gz`
- **Registros originales:** 48.204 filas, 9 columnas.
- **Subperiodo analizado:** 2016-01-01 a 2017-12-31; **16.551** observaciones horarias tras eliminar **3.360** timestamps duplicados.
- **Huecos horarios:** **993** horas faltantes respecto a una malla regular (no se imputaron; se documentan como limitación).

### 2.2 Atributos de la tabla

| Atributo | Tipo | Descripción |
|----------|------|-------------|
| `date_time` | datetime | Marca temporal horaria (índice de la serie) |
| `traffic_volume` | int | **Variable objetivo:** vehículos/hora |
| `temp` | float | Temperatura en **Kelvin** |
| `rain_1h` | float | Milímetros de lluvia acumulados en la hora |
| `snow_1h` | float | Milímetros de nieve acumulados en la hora |
| `clouds_all` | int | Porcentaje de nubosidad (0–100) |
| `weather_main` | categórica | Categoría meteorológica (Rain, Snow, Clear, etc.) |
| `weather_description` | categórica | Descripción textual del clima |
| `holiday` | categórica / nula | Nombre del feriado si aplica; nulo si no es feriado |

**Variables derivadas para modelado:**

| Variable | Construcción | Uso |
|----------|--------------|-----|
| `is_holiday` | 1 si `holiday` no es nulo, 0 en caso contrario | Exógena SARIMAX |
| `hour`, `weekday` | Extraídas de `date_time` | EDA (Notebook 1) |

Se **excluyeron** `weather_main` y `weather_description` del modelado por ser categóricas de alta cardinalidad; el TP prioriza exógenas numéricas y una dummy de feriado.

### 2.3 Estadísticas descriptivas del subperiodo

**Variable objetivo `traffic_volume`:**

| Estadístico | Valor |
|-------------|-------|
| Media | 3.290 veh/h |
| Desvío estándar | 1.966 veh/h |
| Mínimo | 0 |
| Máximo | 7.280 |

**Patrones temporales (Notebook 1):**

| Patrón | Media (veh/h) |
|--------|---------------|
| Hora pico (7–9 h y 16–18 h) | 4.865 |
| Valle nocturno (2–4 h) | 510 |
| Días laborables | 3.543 |
| Fin de semana | 2.662 |
| Feriados | 910 |
| No feriados | 3.293 |

La relación hora pico / valle (~9,5×) confirma estacionalidad diaria fuerte. El tráfico en feriados es mucho menor que en días normales (910 vs 3.293 veh/h): un feriado se comporta más como un día de muy bajo volumen que como un laborable típico.

**Interpretación de las medias:**

- **Hora pico (4.865 veh/h):** en las ventanas 7–9 h y 16–18 h converge el commute matutino y vespertino; es el régimen que más tensiona los modelos (amplitud alta, variabilidad entre días).
- **Valle nocturno (510 veh/h):** entre 2–4 h el volumen cae casi un orden de magnitud; errores absolutos pequeños aquí pueden generar **MAPE enormes** si el denominador es cercano a cero.
- **Laborables vs fin de semana (3.543 vs 2.662):** ~25 % menos tráfico en fin de semana; estacionalidad **semanal** real que $s=24$ no separa por construcción.
- **Feriados (910 veh/h):** efecto fuerte pero pocas observaciones, lo que dificulta estimarlo en SARIMAX (`is_holiday` no significativo).

**Correlaciones lineales con `traffic_volume`:** todas débiles (|r| < 0,15). La mayor es con `temp` (r ≈ 0,13). Esto indica que, a nivel global, el clima **no explica** la mayor parte de la varianza del tráfico (dominada por hora del día y día de la semana). No descarta efecto **condicional** en SARIMAX: la lluvia puede importar cuando controlamos la dinámica propia de la serie.

**Exógenas (medias):** temp ≈ 282 K (~9 °C); rain_1h ≈ 0,64 mm/h (mayormente seco); snow_1h ≈ 0; is_holiday ≈ 0,001 (feriados raros en el subperiodo).

---

## 3. Descripción de los modelos

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

**Orden del pipeline:** EDA y descomposición (NB1–NB2) informan baselines; baselines fijan umbral RMSE; ACF/PACF y ADF guían SARIMA; SARIMAX prueba exógenas; NB4 consolida con Prophet, ARCH/GARCH y gráficos comparativos.

### 3.0.1 Guía de notación: letras, números y subíndices

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
| **$p$-valor (ADF, Ljung-Box, ARCH-LM, coeficientes)** | Probabilidad bajo $H_0$ | Si $p < 0{,}05$, hay evidencia **estadística** contra la hipótesis nula (ver interpretaciones en cada sección) |

---
### 3.1 Preprocesamiento (Notebook 1)

**Decisiones y justificación:**

| Decisión | Por qué |
|----------|---------|
| Parseo datetime + orden cronológico | Prerequisito para toda serie temporal; evita rezagos incorrectos |
| Subperiodo 2016–2017 | Dos ciclos anuales completos; ~16.551 obs. frente a 48.204 totales, lo que hace viable SARIMA en hardware local |
| Deduplicación (3.360 filas) | Múltiples registros por misma hora distorsionan ACF y estacionalidad |
| No imputar 993 huecos | Imputar (p. ej. interpolar) introduciría tráfico **sintético**; se documenta como limitación |
| Dummy `is_holiday` | Convierte feriados esporádicos en regresor numérico para SARIMAX |
| Excluir categorías de clima | Evita explosión de dummies (`weather_description` tiene decenas de niveles) |

**Gráficos exploratorios (Notebook 1, sin captura en carpeta):** la serie completa muestra ciclo diario superpuesto; el zoom semanal revela dos picos por día; boxplots por hora confirman medianas altas en 7–9 y 16–18 h y mínimos en 2–4 h; boxplots por día de semana muestran menor tráfico sábado/domingo; feriados vs no feriados confirman caída fuerte en feriados — todo coherente con $s=24$ y con la necesidad posterior de Prophet o estacionalidad semanal.

### 3.2 Descomposición ETS y baselines (Notebook 2)

#### Concepto: descomposición aditiva vs multiplicativa

Una serie puede descomponerse en **tendencia** ($T_t$), **estacionalidad** ($S_t$) y **residuo** ($R_t$):

| Tipo | Ecuación | Cuándo usarla |
|------|----------|---------------|
| **Aditiva** | $Y_t = T_t + S_t + R_t$ | La **amplitud** del ciclo (pico − valle) es aproximadamente **constante** en el tiempo |
| **Multiplicativa** | $Y_t = T_t \cdot S_t \cdot R_t$ | La amplitud **crece** cuando sube el nivel general (p. ej. +10 % del nivel en hora pico, sea invierno o verano) |

La descomposición clásica (`seasonal_decompose`, `period=24`) **no ajusta un modelo predictivo**: es una herramienta **exploratoria** para decidir qué familia ETS/Holt-Winters probar después.

![Descomposición aditiva (periodo 24)](capturas/notebook_2/Celda%202_3%20-%20descomposicion%20aditiva.png)

**Lectura del gráfico aditivo (cuatro paneles, de arriba hacia abajo):**

1. **Serie original (panel superior):** línea azul con oscilaciones diarias regulares sobre un nivel medio ~3.000–3.500 veh/h. Cada “diente de sierra” es un día: subida en commute, caída nocturna. No se aprecia tendencia fuerte a largo plazo, pero sí variación lenta del nivel entre estaciones.
2. **Tendencia (segundo panel):** curva suave que filtra las oscilaciones diarias. Captura el nivel medio y variaciones lentas; también absorbe parte de la estacionalidad **semanal** que el filtro con $period=24$ no puede separar (lunes y sábado comparten la misma hora en este descompuesto).
3. **Estacionalidad (tercer panel):** patrón que se repite cada 24 puntos horizontales (1 día). La amplitud es del orden de **±250 veh/h** en escala aditiva: el ciclo diario existe, pero es **pequeño** frente a la serie original.
4. **Residuo (panel inferior):** oscilaciones amplias (**±3.500 veh/h**). Aquí queda lo que el filtro aditivo con $s=24$ **no explicó**: diferencias entre laborables y fin de semana, eventos puntuales, huecos en la malla horaria y efectos meteorológicos.

**Conclusión del panel aditivo:** el ciclo diario existe, pero **no es el único factor**; gran parte de la variabilidad queda en el residuo. Eso anticipa que un modelo solo con $s=24$ tendrá límites.

![Descomposición multiplicativa (periodo 24, y+1)](capturas/notebook_2/Celda%202_3%20-%20descomposicion%20multiplicativa.png)

**Lectura del gráfico multiplicativo (misma estructura de cuatro paneles):**

- Requiere valores **estrictamente positivos**; en train hay **2 observaciones con tráfico = 0**, por lo que se usó $y'_t = y_t + 1$ solo para esta visualización.
- **Estacionalidad (tercer panel):** oscila en torno a **1,0** (±5–10 %): interpretación natural en tráfico urbano (“en hora pico hay ~8–10 % más volumen **relativo** al día”).
- **Residuo:** menor en escala relativa que en la aditiva; la separación tendencia / estacionalidad es **más limpia** visualmente.
- **Serie original vs reconstruida:** la suma (o producto) de componentes se parece más a la serie original que en la aditiva, señal de que la amplitud del ciclo **crece** cuando sube el nivel general.

**¿Cuál descomposición es “mejor”?**

| Criterio | Aditiva | Multiplicativa |
|----------|---------|----------------|
| Separación visual de componentes | Residuo grande vs estacionalidad pequeña | Estacionalidad % más clara |
| Coherencia con fenómeno (amplitud crece con nivel) | Menos natural si pico−valle escala con el nivel | Más natural teóricamente |
| **Pronóstico out-of-sample (§2.6)** | **HW/ETS aditivos ganan** (RMSE 1.803–1.848) | ETS(M,Ad,M) **falla** (RMSE 29.094) |

**Decisión del TP:** la descomposición multiplicativa orientó a **probar** ETS(M,Ad,M), pero en **test** el modelo multiplicativo colapsó (revertir el +1 y extrapolar 672 h amplifica errores). Los **baselines competitivos finales son aditivos** (HW amortiguado, ETS(A,Ad,A)). La lección metodológica: un gráfico exploratorio puede sugerir multiplicativo, pero la **validación out-of-sample** es la que decide.

**Evento en tendencia (~julio 2016):** pico local en la componente de tendencia, coherente con huecos horarios documentados en NB1 y con cambios de régimen semanal no modelados con $period=24$.

#### Concepto: Holt-Winters / suavizado exponencial triple

Holt-Winters extiende el suavizado exponencial con **nivel** ($\ell_t$), **tendencia** ($b_t$) y **estacionalidad** ($s_t$). Con estacionalidad aditiva y periodo 24:

$$\hat{y}_{t+h} = \ell_t + h \cdot b_t + s_{t+h-24}$$

**Parámetros de suavizado** ($\alpha$, $\beta$, $\gamma$) se estiman por máxima verosimilitud. La variante **amortiguada** (damped trend, `damped_trend=True`) introduce factor de amortiguación $\phi \in (0,1]$ sobre la tendencia.

**Por qué HW amortiguado ganó en test (RMSE 1.803,7):** al pronosticar **672 horas** (~28 días), la tendencia no amortiguada de ETS(A,A,A) **extrapola** el drift lineal demasiado lejos (RMSE 9.896). La amortiguación hace que el incremento horario previsto **decaiga**, de modo que el nivel de tráfico se estabiliza en horizonte largo, coherente con una autopista en régimen estacionario a escala mensual. El componente estacional aditivo de 24 horas captura el ciclo diario; el error restante viene sobre todo del **valle nocturno** (ver overlay §3.7).

**Modelos baseline ajustados:**

| Modelo | Especificación | Rol |
|--------|----------------|-----|
| Holt-Winters (A,A,A) | Aditivo, tendencia no amortiguada | Contraste negativo |
| Holt-Winters amortiguado (A,Ad,A) | Aditivo, tendencia amortiguada | **Mejor clásico OOS** |
| ETS(A,Ad,A) | `ETSModel`, error/tendencia/estacional aditivos | Baseline alternativo |
| ETS(M,Ad,M) sobre $y+1$ | Multiplicativo con desplazamiento | Descartado en test (RMSE 29.094) |

Los boxplots por hora del Notebook 1 muestran medianas crecientes de madrugada hacia 7–9 h y 16–18 h, confirmando que $s=24$ captura el ciclo relevante para Holt-Winters/ETS.

#### Concepto: estacionaridad y test ADF

Una serie es **estacionaria** si sus propiedades estadísticas (media, varianza, autocovarianza) no cambian en el tiempo. Los modelos ARIMA/SARIMA requieren estacionariedad en media, lograda mediante **diferenciación**.

El test **Augmented Dickey-Fuller (ADF)** evalúa $H_0$: la serie tiene raíz unitaria (no estacionaria). Si el **$p$-valor es menor que 0,05**, se rechaza $H_0$ y hay evidencia estadística de estacionariedad en el sentido del test (media estable tras la transformación aplicada).

**Cómo interpretar el $p$-valor del ADF:** un valor muy pequeño (por ejemplo, del orden de $10^{-29}$) indica que, bajo la hipótesis de raíz unitaria, observar un estadístico tan extremo es **prácticamente imposible**; por tanto, se concluye que la serie transformada no se comporta como un proceso con raíz unitaria. Esto **no** garantiza que la serie esté lista para un ARIMA(0,0,0): puede seguir teniendo estacionalidad y autocorrelación que el ADF no evalúa.

**Resultados ADF (train):**

| Transformación | Estadístico | p-valor |
|----------------|-------------|---------|
| Niveles | −16,39 | ≈ 2,7×10⁻²⁹ |
| log(1+y) | −16,71 | ≈ 1,4×10⁻²⁹ |
| Diff(1) | −24,22 | ≈ 0 |
| Diff estacional (24) | −19,55 | ≈ 0 |

Con $n \approx 15.800$, el ADF tiene **alto poder** y puede rechazar $H_0$ incluso en niveles. **Qué significa cada fila:**

| Transformación | Interpretación del $p$-valor ≈ 0 |
|----------------|----------------------------------|
| **Niveles** | Rechazo de raíz unitaria, pero la serie **sigue teniendo** estacionalidad y autocorrelación; esto **no** implica que esté “lista para ARIMA(0,0,0)” |
| **log(1+y)** | Estabiliza varianza cuando hay ceros; resultado similar a niveles |
| **Diff(1)** | Estacionariedad en media tras una diferencia regular; esto **justifica** usar **$d \geq 1$** |
| **Diff(24)** | Estacionariedad tras diferencia estacional; esto **justifica** usar **$D \geq 1$** con $s=24$ |

La evidencia decisiva para SARIMA proviene de **ACF/PACF** y descomposición, no solo del ADF. Se adoptó **$d=1$, $D=1$** como punto de partida.

#### Concepto: ACF y PACF

- **ACF** ($\rho_k$, Autocorrelation Function): correlación entre $y_t$ e $y_{t-k}$. Mide cuánto se parece la serie consigo misma desplazada $k$ horas.
- **PACF** ($\phi_{kk}$, Partial Autocorrelation Function): correlación entre $y_t$ e $y_{t-k}$ **eliminando** la influencia de los lags intermedios ($1, \ldots, k-1$). Ayuda a identificar el orden AR ($p$ o $P$).

Picos en lags 24, 48, 72 indican estacionalidad diaria y sugieren órdenes estacionales $(P,D,Q)_{24}$. Las **bandas punteadas** en el gráfico delimitan intervalos de confianza aproximados al 95 %: barras que las superan indican correlaciones significativas.

![ACF y PACF de serie estacionarizada](capturas/notebook_2/Celda%202_8%20-%20ACF%20y%20PACF%20de%20serie%20estacionarizada.png)

**Interpretación del gráfico ACF/PACF** (serie estacionarizada con diff(1) y diff estacional 24):

- **Picos en lags 24, 48, 72… (ACF):** la serie horaria de hoy se parece a la de ayer a la **misma hora**; esto es evidencia de **estacionalidad diaria** ($s=24$). La persistencia en múltiplos de 24 confirma que un SARIMA estacional es apropiado.
- **PACF con corte en lag 1 (parte regular) y lag 24 (parte estacional):** sugiere componentes AR de orden 1 en ambas partes; de ahí la propuesta inicial **$(1,1,1)(1,1,1)_{24}$** (un AR y un MA regular y estacional).
- **Decaimiento lento en ACF sin diff:** en niveles la autocorrelación persiste muchos lags; la serie **no es ruido blanco**, lo que justifica modelos dinámicos frente a un pronóstico ingenuo de media.

**Lectura visual del gráfico:** en el panel ACF, barras que sobresalen de las bandas punteadas (intervalos de confianza) indican correlaciones **estadísticamente distintas de cero**. Los picos periódicos cada 24 lags forman una “columna” de barras altas: es la firma gráfica del ciclo diario. En el PACF, un corte brusco después del lag 1 (parte regular) y del lag 24 (parte estacional) sugiere órdenes AR pequeños; si el PACF decayera lentamente sin cortes, habría que aumentar $d$ o $D$.

**Propuesta inicial (Notebook 2):** SARIMA$(1,1,1)(1,1,1)_{24}$ — punto de partida; NB3 la revisa con grid y BIC.

### 3.3 SARIMA univariado (Notebook 3)

#### Concepto: SARIMA

Un **SARIMA**$(p,d,q)(P,D,Q)_s$ modela la serie $y_t$ mediante componentes:

- **AR($p$):** regresión sobre $p$ valores pasados de la serie (autoregresión).
- **I($d$):** diferenciación regular $d$ veces para estacionariedad en media.
- **MA($q$):** media móvil sobre $q$ shocks pasados.
- **Componente estacional:** $(P,D,Q)_s$ con periodo $s=24$ (diferenciación estacional, AR y MA estacionales).

Forma general (operador de rezagos $L$):

$$\Phi_P(L^{24}) \phi_p(L) (1-L^{24})^D (1-L)^d y_t = \Theta_Q(L^{24}) \theta_q(L) \varepsilon_t$$

donde $\varepsilon_t$ es ruido blanco (idealmente).

**Identificación:**
1. **Manual (NB2):** $(1,1,1)(1,1,1)_{24}$ desde ACF/PACF.
2. **Grid stepwise (NB3):** sobre submuestra de 3.000 obs. (evitar `pmdarima` por memoria), $d=1$, $D=1$ fijos; resultado: $(2,1,2)(1,1,1)_{24}$.
3. **Selección final:** refit sobre train completo (15.879 obs.); criterio **BIC** (penaliza más parámetros que AIC).

| Candidato | AIC | BIC |
|-----------|-----|-----|
| Manual $(1,1,1)(1,1,1)_{24}$ | 249.879,9 | 249.918,2 |
| Grid $(2,1,2)(1,1,1)_{24}$ | 249.174,1 | **249.227,8** |

**Ganador univariado:** SARIMA$(2,1,2)(1,1,1)_{24}$ (BIC menor por **~690 puntos**). **Qué implica:**

- Más términos AR(2) y MA(2) **regulares** absorben memoria de corto plazo (hora anterior, hace 2 h) que $(1,1,1)$ dejaba en residuos.
- La parte estacional $(1,1,1)_{24}$ se mantiene: un shock estacional de hace 24 h y un MA estacional de orden 1 bastan una vez fijados $d=1$, $D=1$.
- **Compromiso entre ajuste y complejidad:** más parámetros mejoran el ajuste in-sample (AIC/BIC), pero aumentan el riesgo de sobreajuste; BIC penaliza más y aun así prefiere $(2,1,2)$.

#### Concepto: AIC y BIC

- **AIC** = $2k - 2\ln(L)$: balance ajuste vs complejidad ($k$ parámetros, $L$ verosimilitud).
- **BIC** = $k\ln(n) - 2\ln(L)$: penalización más fuerte que AIC; favorece modelos parsimoniosos.

Menor AIC/BIC indica mejor ajuste in-sample, pero **no garantiza** mejor pronóstico out-of-sample.

**Pronóstico SARIMA (672 h test):**

| Métrica | SARIMA | HW amortiguado (baseline) |
|---------|--------|---------------------------|
| MAE | 1.489,3 veh/h | 1.512,7 |
| RMSE | **1.898,5** | **1.803,7** |
| MAPE | 102,1% | 170,0% |

SARIMA mejora levemente **MAE** (−23 veh/h respecto a HW) pero **empeora RMSE en +94,8 veh/h**; el modelo Box–Jenkins comete **errores más grandes en algunos instantes** (hora pico, días atípicos), aunque el error medio absoluto sea similar.

**Lectura del gráfico de pronóstico SARIMA (2 semanas test):**

![Pronóstico SARIMA vs tráfico real (2 semanas test)](capturas/notebook_3/Celda%203_5%20-%20Pronóstico%20SARIMA%20vs%20tráfico%20real.png)

- **Línea del tráfico real:** muestra el patrón observado en test; sirve de referencia para evaluar fase y amplitud del pronóstico.
- **Línea de pronóstico SARIMA:** reproduce el patrón de **dos picos por día** (commute matutino y vespertino); la estructura $(1,1,1)_{24}$ en la parte estacional captura el ciclo de 24 h.
- **Bandas de confianza (intervalo 95 %):** son amplias; indican incertidumbre alta a 672 h de horizonte. Donde la banda es muy ancha, el modelo admite errores grandes.
- **Amplitud:** la línea de pronóstico **aplana el valle nocturno** respecto al real (similar a HW/ETS en el overlay de §3.7): predice más tráfico de madrugada del que ocurre.
- **Desfases puntuales:** en algunos picos el modelo llega tarde o subestima la magnitud; esos errores grandes elevan RMSE más que MAE.
- **Comparación con baseline:** RMSE 1.898,5 es mayor que 1.803,7 (HW); la mayor complejidad SARIMA **no se tradujo** en mejor generalización a 4 semanas en este dataset.

### 3.4 SARIMAX (Notebook 3)

#### Concepto: SARIMAX

**SARIMAX** extiende SARIMA incorporando **variables exógenas** $X_t$ que explican desvíos del tráfico no capturados por la dinámica propia de la serie:

$$y_t = \text{SARIMA}(p,d,q)(P,D,Q)_{24} + \beta_1 X_{1,t} + \cdots + \beta_k X_{k,t} + \varepsilon_t$$

En este TP: `temp`, `rain_1h`, `snow_1h`, `is_holiday`. Mismas órdenes SARIMA que el ganador univariado: $(2,1,2)(1,1,1)_{24}$.

**Decisión sobre exógenas en test:** se usaron valores **observados** de clima en el periodo test (backtest optimista). En producción habría que pronosticar el clima o usar pronósticos meteorológicos.

**Coeficientes SARIMAX (significancia 5%) — interpretación detallada:**

| Parámetro | Coeficiente | p-valor | Interpretación |
|-----------|-------------|---------|----------------|
| `rain_1h` | −0,26 | ≈ 0 | Por cada **mm adicional** de lluvia en la hora, el tráfico esperado baja ~0,26 veh/h **manteniendo fija** la dinámica SARIMA. Signo coherente con la intuición (más lluvia se asocia a menos viajes). Magnitud modesta frente al rango 0–7.280 veh/h. |
| `temp` | +63,5 | ≈ 0 | Estadísticamente significativo pero **difícil de interpretar causalmente**: la temperatura correlaciona con estación del año y hora del día; parte del efecto ya captura el SARIMA estacional. |
| `snow_1h` | +2.865 | 0,00027 | Significativo pero **sospechoso**: pocos eventos de nieve, coeficiente muy grande; posible artefacto de colinealidad o outliers meteorológicos. |
| `is_holiday` | +27,0 | 0,888 | **No significativo** ($p > 0{,}05$): pocos feriados en train; el modelo no puede estimar bien un efecto feriado estable. Contradice el EDA agregado (910 vs 3.293 veh/h) porque el feriado queda absorbido por la estacionalidad diaria/semanal. |

**Qué significa el $p$-valor de cada coeficiente SARIMAX:** bajo la hipótesis nula “este coeficiente es cero”, el $p$-valor mide qué tan probable es observar un estimador tan extremo si la variable **no aportara** nada al modelo. Con umbral 5 %:

| $p$-valor | Interpretación |
|-----------|----------------|
| **$p < 0{,}05$** (p. ej. `rain_1h`, `temp`) | Hay evidencia estadística de que el coeficiente es distinto de cero **condicionado** al resto del modelo. |
| **$p \geq 0{,}05$** (p. ej. `is_holiday`, 0,888) | **No** hay evidencia suficiente para afirmar un efecto lineal del feriado una vez incluidos los demás términos; el estimador +27 veh/h puede deberse al azar muestral. |

Un $p$-valor bajo **no** implica que la variable sea la más importante para pronosticar: `snow_1h` es significativo pero poco robusto (pocos eventos); `is_holiday` no lo es pese al contraste descriptivo del EDA.

**Comparación SARIMA vs SARIMAX:**

| Criterio | SARIMA | SARIMAX |
|----------|--------|---------|
| AIC | 249.174,1 | **249.085,2** |
| BIC | 249.227,8 | **249.169,5** |
| MAE test | 1.489,3 | **1.410,9** |
| RMSE test | 1.898,5 | **1.854,8** |
| MAPE test | 102,1% | **80,2%** |

SARIMAX mejora en todos los criterios respecto a SARIMA univariado (ΔRMSE ≈ **44 veh/h**, ΔMAE ≈ **78 veh/h**). **Traducción práctica:** incorporar clima reduce el error típico en ~78 veh/h y el error cuadrático medio en ~44 veh/h — mejora **real pero acotada** (~2–4 % del RMSE base).

Sin embargo RMSE test **1.854,8** es mayor que **1.803,7** (HW): las exógenas **ayudan dentro de Box–Jenkins**, pero **no superan** al mejor baseline ETS/HW que modela estacionalidad determinística con solo $s=24$.

### 3.5 Prophet (Notebook 4)

#### Prophet

Prophet (Meta) descompone la serie en tendencia piecewise, estacionalidades de Fourier (diaria, semanal, anual) y efectos de feriados opcionales. Se configuró:

- `daily_seasonality=True`: ciclo de 24 h mediante suma de senos/cosenos (no fija un único perfil aditivo como HW).
- `weekly_seasonality=True`: diferencia lunes–sábado sin imponer $s=168$ en un SARIMA.
- `yearly_seasonality=True`: captura invierno vs verano en Minnesota.

No incorpora clima; es un benchmark **univariado alternativo**, no comparable directamente con SARIMAX en la pregunta de exógenas.

**Post-procesamiento:** 18 pronósticos negativos en test se recortaron a 0 (tráfico no puede ser negativo). Sin recorte, RMSE pasaba de 884,4 a 893,2 (impacto leve).

**Métricas test:** MAE **642,3**, RMSE **884,4**, MAPE **42,5%**.

**Interpretación numérica vs HW (RMSE 884 vs 1.804):** Prophet reduce el error cuadrático medio a **~49 %** del mejor clásico. En términos absolutos, el error típico baja ~920 veh/h en RMSE — la brecha es **estructural**: Prophet modela estacionalidad semanal/anual además del ciclo diario, y ajusta amplitud con componentes flexibles de Fourier, evitando el “piso nocturno” de HW/ETS (ver overlay §3.7).

### 3.6 GARCH sobre residuos (Notebook 4)

#### Concepto: heteroscedasticidad, ARCH-LM y GARCH

- **Homocedasticidad:** varianza del error constante.
- **Heterocedasticidad:** varianza cambia en el tiempo (p. ej. mayor error en hora pico o con clima extremo).

**ARCH-LM** testea si los residuos al cuadrado tienen autocorrelación (efecto ARCH). $H_0$: no hay efecto ARCH hasta lag 24.

**Resultado sobre residuos SARIMAX in-sample:**
- Estadístico LM: **5.954,93**, $p \approx 0$; se **rechaza** la homocedasticidad.

**Interpretación del $p$-valor del ARCH-LM:** bajo $H_0$ (varianza constante), un $p$-valor prácticamente cero indica que la autocorrelación en $\varepsilon_t^2$ es demasiado fuerte para atribuirla al azar. En palabras: no basta con que la **media** del error esté bien modelada; la **varianza** del error cambia según la hora, el clima o el régimen de tráfico. Esto es coherente con mayor incertidumbre en hora pico o en eventos extremos.

**GARCH(1,1)** modela la varianza condicional:

$$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

Estimaciones: $\alpha_1 \approx 0{,}076$, $\beta_1 \approx 0{,}902$, **$\alpha+\beta \approx 0{,}978$**.

**Interpretación de $\alpha+\beta \approx 0{,}98$:** este valor está **muy cerca de 1**, lo que implica **persistencia alta** de la volatilidad. Un shock grande en el error (por ejemplo, un día de tormenta o un accidente) infla la varianza prevista durante **muchas horas siguientes**, no solo en $t+1$. Es útil para intervalos de confianza dinámicos; **no** implica que el pronóstico puntual de la media deba cambiar.

GARCH **no re-pronostica el nivel** de tráfico en este TP; complementa el diagnóstico de incertidumbre, no mejora MAE/RMSE de la media.

### 3.7 Gráficos comparativos finales — interpretación detallada

#### Overlay top 3 por RMSE (2 semanas test)

El gráfico muestra las **últimas 336 horas** del periodo test (dos semanas) superponiendo el tráfico observado y los tres mejores modelos según RMSE global: Prophet, Holt-Winters amortiguado y ETS(A,Ad,A). Permite comparar **forma** del error (picos, valles, fines de semana), no solo números agregados.

![Overlay — Real vs Prophet, HW y ETS](capturas/notebook_4/Celda%204.6%20-%20Grafico%20overlay.png)

| Serie (color en la captura) | Qué representa | Lectura detallada del gráfico |
|----------------------------|----------------|------------------------------|
| **Real (azul, línea gruesa)** | Volumen observado | Referencia. Se ven **dos picos diarios** (commute matutino y vespertino) con alturas ~6.000–6.500 veh/h; entre medianoche y 5 h el tráfico cae a ~500 veh/h. Los fines de semana dentro de la ventana muestran picos algo más bajos que los laborables. |
| **Prophet (naranja)** | Mejor RMSE global (884 veh/h) | Sigue **en fase** con el real: sube y baja a la misma hora. Los **valles nocturnos** se acercan al nivel real (cerca de 0 en varias horas; 18 valores negativos fueron recortados a 0). En algunos picos vespertinos queda **ligeramente por debajo** del real, pero la amplitud diaria es mucho más fiel que la de los clásicos. |
| **HW amortiguado (rojo)** | Mejor clásico por RMSE (1.804 veh/h) | Reproduce la **frecuencia** del ciclo diario (dos subidas por día), pero mantiene un **piso nocturno elevado** (~2.500–3.000 veh/h) cuando el real está ~500 veh/h. Visualmente, la línea roja “flota” sobre el valle azul: ahí se concentra gran parte del RMSE. En picos, a veces subestima la cima respecto al real. |
| **ETS(A,Ad,A) (verde)** | Tercer lugar (RMSE 1.848 veh/h) | Casi **superpuesto** a HW amortiguado (rojo): misma limitación de amplitud. Confirma que el problema no es la implementación statsmodels vs Holt-Winters, sino modelar solo $s=24$ con estacionalidad aditiva fija. |

**Lectura comparativa azul vs rojo:** donde la línea azul (real) desciende de noche, la roja (HW) apenas baja. Ese **error sistemático en madrugada** (sobreestimación de ~2.000 veh/h durante muchas horas) explica por qué HW, pese a ser el mejor clásico, queda lejos de Prophet. En hora pico, rojo y azul se acercan más, pero HW aún pierde precisión en las cimas.

**Por qué Prophet “gana” visualmente:** no impone un único perfil horario repetido con offset fijo; combina ciclo diario, semanal y anual, permitiendo **valles profundos** y **picos altos**. HW/ETS con $s=24$ estiman un patrón estacional aditivo que **no baja lo suficiente** de noche.

**RMSE en números vs gráfico:** el error de HW (~1.804 veh/h) concentra aproximadamente el 40–50 % en valle nocturno mal predicho (error ~2.000 veh/h multiplicado por muchas horas) más subestimación de picos (~1.000 veh/h en horas pico).

#### Error absoluto medio por hora del día

Este gráfico promedia el error absoluto $|y - \hat{y}|$ en test **por hora del reloj** (0–23), independientemente del día. Permite ver **en qué franja horaria** falla cada modelo, algo que RMSE global oculta.

![MAE por hora — HW, SARIMAX y Prophet](capturas/notebook_4/Celda%204.7%20-%20error%20absoluto%20medio%20por%20hora%20-%20test.png)

**Lectura hora por hora (colores según la captura):**

| Franja | HW amortiguado (azul) | SARIMAX (púrpura) | Prophet (naranja) |
|--------|----------------------|-------------------|-------------------|
| **0–5 h** | Error alto (~2.300–3.100 veh/h): coincide con el piso nocturno del overlay | Error **bajo** (~400–800): las exógenas y la dinámica ARMA ayudan en madrugada | Error bajo (~300–600): Prophet captura el valle |
| **6–8 h** | Error moderado (~2.000) | **Pico ~3.750 en h=7**: falla en el arranque del commute matutino | Error moderado (~1.000–1.600) |
| **10–15 h** | ~800–1.200 | ~800–1.000 | ~300–600 |
| **16–20 h** | ~1.000–1.800 | Similar a HW en hora pico vespertina | ~400–800 |
| **18 h** | Mínimo local HW (~370): en esa hora el perfil estacional aditivo coincide mejor | — | — |

**Descripción del gráfico:** el eje horizontal es la hora (0–23); el vertical es MAE en veh/h. La curva **naranja (Prophet)** queda casi siempre **debajo** de las demás: domina en casi todo el reloj. La **azul (HW)** presenta un “plateau” alto entre 0 h y 6 h (error >2.000 veh/h), coherente con el valle mal predicho del overlay. La **púrpura (SARIMAX)** es baja de madrugada pero **dispara** en h=7: un solo pico horario eleva mucho el RMSE aunque el MAE global sea el mejor entre clásicos.

**Conclusión del gráfico:** no hay un solo “mejor modelo” en todas las horas. SARIMAX gana en madrugada (exógenas más dinámica ARMA), HW es más robusto en hora pico agregada, Prophet domina **en casi todo el reloj**. El pico de SARIMAX a las **7 h** explica por qué tiene el **mejor MAE clásico** (1.411) pero **peor RMSE** que HW (1.855 frente a 1.804): un outlier horario penalizado fuertemente por RMSE.

---

## 4. Pruebas sobre los modelos

### 4.1 Evaluación out-of-sample (tabla comparativa final)

Periodo test: **672 horas** (cuatro semanas). Métricas en veh/h salvo MAPE (%).

| Rank (RMSE) | Modelo | MAE | RMSE | MAPE |
|-------------|--------|-----|------|------|
| 1 | **Prophet** | 642,3 | **884,4** | 42,5% |
| 2 | Holt-Winters amortiguado (A,Ad,A) | 1.512,7 | 1.803,7 | 170,0% |
| 3 | ETS(A,Ad,A) | 1.612,6 | 1.848,2 | 157,6% |
| 4 | SARIMAX | **1.410,9** | 1.854,8 | 80,2% |
| 5 | SARIMA$(2,1,2)(1,1,1)_{24}$ | 1.489,3 | 1.898,5 | 102,1% |

*Excluidos de la tabla final:* Holt-Winters (A,A,A) (RMSE 9.896) y ETS(M,Ad,M) (RMSE 29.094) — baselines de referencia con desempeño inaceptable en test.

**Interpretación de métricas — ejemplos numéricos del test:**

| Comparación | MAE | RMSE | Qué concluir |
|-------------|-----|------|--------------|
| Prophet vs HW | 642 vs 1.513 | 884 vs 1.804 | Prophet reduce error típico en **~870 veh/h** (RMSE) |
| SARIMAX vs SARIMA | 1.411 vs 1.489 | 1.855 vs 1.899 | Exógenas ayudan **~44 veh/h** RMSE dentro de Box–Jenkins |
| SARIMAX vs HW | 1.411 vs 1.513 | 1.855 vs 1.804 | Mejor MAE clásico, pero **peor RMSE** por h=7 |
| HW vs ETS | 1.513 vs 1.613 | 1.804 vs 1.848 | HW amortiguado gana por ~44 veh/h RMSE |

- **MAE** = promedio de $|y - \hat{y}|$: “error típico” en veh/h. SARIMAX 1.411 significa que, en promedio, el pronóstico se equivoca ~1.411 veh/h.
- **RMSE** = $\sqrt{\text{mean}((y-\hat{y})^2)}$: penaliza errores grandes. Si RMSE es mucho mayor que MAE (SARIMA: 1.899 frente a 1.489), hay **outliers** de error.
- **MAPE** = promedio de $|y-\hat{y}|/|y|$: HW 170 % porque en horas con $y \approx 200$ veh/h un error de 1.500 veh/h produce ratios >100 %.

**Mejor global:** Prophet (RMSE 884,4).  
**Mejor clásico (RMSE):** Holt-Winters amortiguado (1.803,7).  
**Mejor clásico (MAE):** SARIMAX (1.410,9), pero 4.º en RMSE por pico de error en h=7.

### 4.2 Baselines in-sample vs out-of-sample (Notebook 2)

| Modelo | AIC in-sample | RMSE test |
|--------|---------------|-----------|
| HW amortiguado | ≈ 217.192 | **1.803,7** |
| HW clásico (A,A,A) | ≈ 217.265 | 9.896,2 |
| ETS(A,Ad,A) | ≈ 262.413 | 1.848,2 |
| ETS(M,Ad,M) y+1 | ≈ 293.248 | 29.093,6 |

**Lección con números:**

| Modelo | AIC in-sample | RMSE test | Interpretación |
|--------|---------------|-----------|----------------|
| HW amortiguado | ≈ 217.192 | **1.803,7** | Mejor AIC **y** mejor test entre aditivos; especificación coherente |
| HW clásico | ≈ 217.265 | 9.896,2 | AIC casi igual, test **5× peor**; la tendencia no amortiguada **explota** a 672 h |
| ETS(A,Ad,A) | ≈ 262.413 | 1.848,2 | AIC peor, test competitivo; AIC no ordena bien baselines aquí |
| ETS(M,Ad,M) | ≈ 293.248 | 29.093,6 | Multiplicativo + desplazamiento **no generaliza** |

**Por qué importa la amortiguación:** sin ella, la tendencia extrapola linealmente 672 pasos y el nivel de tráfico **deriva** hacia valores irreales; con amortiguación ($\phi < 1$), la tendencia se estabiliza y el RMSE test resulta razonable.

### 4.3 Diagnóstico de residuos SARIMA (Notebook 3)

#### Concepto: Ljung-Box

El test **Ljung-Box** evalúa $H_0$: los residuos son ruido blanco (sin autocorrelación hasta lag $h$). Si el **$p$-valor es menor que 0,05**, se rechaza $H_0$ y queda estructura sin modelar en los residuos.

**SARIMA$(2,1,2)(1,1,1)_{24}$:**

| Lag | Estadístico | p-valor |
|-----|-------------|---------|
| 24 | ≈ 886 | ≈ 0 |
| 48 | ≈ 1.367 | ≈ 0 |

![Diagnóstico residuos SARIMA](capturas/notebook_3/Celda%203_6%20-%20diagnóstico%20SARIMA.png)

**Lectura del panel de diagnóstico (4 subgráficos típicos):**

1. **Residuos vs tiempo:** si hay patrones (ondas, tendencia), el modelo **no capturó** toda la estructura. Aquí se observan clusters de error en ciertos periodos; posible efecto fin de semana o clima.
2. **Histograma / densidad:** desviación de la normal (colas pesadas); errores extremos más frecuentes que en una Gaussiana, coherente con hora pico mal predicha.
3. **Q–Q plot:** puntos desviados de la diagonal en colas; residuos **no normales**; los intervalos de confianza clásicos ±1,96σ pueden ser optimistas.
4. **ACF de residuos:** si hubiera picos significativos en 24, 48… indicaría estacionalidad residual; Ljung-Box resume en un solo test la autocorrelación acumulada hasta lag $h$.

**Ljung-Box — qué significa stat ≈ 886 (lag 24), $p \approx 0$:**

- $H_0$: los primeros 24 autocorrelaciones de los residuos son cero (ruido blanco).
- **Rechazo contundente** ($p \approx 0$): bajo la hipótesis de ruido blanco, la probabilidad de observar un estadístico tan grande es despreciable; los residuos **siguen correlacionados** a escala diaria. El SARIMA no agotó toda la estructura predecible.
- Stat **712** (SARIMAX, lag 24) es menor; las exógenas **ayudan al diagnóstico**, pero $p \approx 0$ persiste; el modelo sigue incompleto (semana, huecos, no linealidades).

Los residuos **no pasan** Ljung-Box: el ajuste es **útil para pronosticar** (mejor que naïve en algunos casos) pero **no teóricamente “correcto”** como modelo generador de ruido blanco.

### 4.4 Análisis de error por hora del día (Notebook 4)

Promedios en test (veh/h):

| Regimen | HW amortiguado | SARIMAX | Prophet |
|---------|----------------|---------|---------|
| Hora pico (7–9, 16–18 h) | 1.303 | 1.894 | **915** |
| Resto del día | 1.585 | **1.248** | **551** |

**Picos puntuales:**
- SARIMAX: MAE ≈ **3.750** en h=7 (arranque matutino).
- HW: MAE ≈ **3.100** en h=4 (madrugada).
- Prophet: MAE < 1.600 en todas las horas.

**Interpretación:** HW es más robusto en hora pico agregada; SARIMAX gana en valle/madrugada (exógenas ayudan fuera de commute); Prophet domina ambos regímenes. El pico de SARIMAX en h=7 explica la paradoja MAE bajo vs RMSE alto (outliers que RMSE penaliza fuertemente).

### 4.5 Test ARCH-LM y GARCH (Notebook 4)

Ya detallado en §3.6. Conclusión: los residuos de SARIMAX presentan **volatilidad clusterizada**; un GARCH(1,1) con $\alpha+\beta \approx 0{,}98$ confirma persistencia, útil para intervalos de predicción dinámicos, no para mejorar el pronóstico puntual de la media en este trabajo.

---

## 5. Conclusiones

### 5.1 Respuesta a la pregunta de investigación

**¿Qué modelo clásico pronostica mejor?**  
Entre Holt-Winters, ETS, SARIMA y SARIMAX, el **Holt-Winters amortiguado (A,Ad,A)** obtiene el menor RMSE out-of-sample: **1.803,7 veh/h** (MAE 1.512,7), seguido de ETS(A,Ad,A) (RMSE 1.848,2) y SARIMAX (RMSE 1.854,8). El SARIMA$(2,1,2)(1,1,1)_{24}$ seleccionado por BIC queda último entre clásicos (RMSE 1.898,5) y **no supera** al baseline ETS/HW pese a mayor complejidad.

**¿Las exógenas mejoran frente a SARIMA univariado?**  
**Sí, de forma incremental:** SARIMAX reduce RMSE de 1.898,5 a 1.854,8 (~44 veh/h) y MAE de 1.489,3 a 1.410,9. La lluvia tiene efecto negativo significativo sobre el tráfico. Sin embargo, SARIMAX **no supera** a Holt-Winters amortiguado; el clima aporta señal dentro de Box–Jenkins pero no es el driver dominante del pronóstico.

**Hallazgo complementario (Prophet):** RMSE **884,4** — aproximadamente la mitad del mejor clásico. Prophet captura estacionalidad diaria, semanal y anual con flexibilidad, incluyendo la **amplitud** del ciclo (valle nocturno vs hora pico). No usa clima; responde otra pregunta (techo univariado con estacionalidades múltiples).

### 5.2 Conclusiones sobre el fenómeno estudiado

1. El tráfico en I-94 presenta **estacionalidad diaria marcada** (media hora pico 4.865 vs valle 510 veh/h) y diferencias entre laborables, fines de semana y feriados.
2. Los modelos con **solo $s=24$** (HW, ETS, SARIMA) reproducen la forma del ciclo diario pero **subestiman el valle nocturno** (piso ~2.500–3.000 veh/h pronosticado vs ~500 real), lo que infla RMSE y MAPE.
3. El clima tiene efecto detectable (lluvia reduce tráfico) pero correlación lineal débil; su aporte predictivo es **modesto** frente a la estructura temporal propia de la serie.
4. La **hora pico** concentra los errores más grandes en modelos clásicos; SARIMAX falla especialmente a las 7 h.

### 5.3 Conclusiones sobre los modelos

| Modelo | Fortaleza | Limitación principal |
|--------|-----------|----------------------|
| HW amortiguado | Mejor clásico OOS; simple; robusto en hora pico | No modela semana ni clima; piso nocturno alto |
| ETS(A,Ad,A) | Similar a HW; framework ETS formal | Mismo piso nocturno; AIC peor in-sample |
| SARIMA | Captura autocorrelación; BIC guía órdenes | Peor RMSE que HW; residuos no blancos |
| SARIMAX | Mejor MAE clásico; clima (lluvia) significativo | No supera HW; pico error h=7; exógenas observadas en test |
| Prophet | Mejor RMSE global; amplitud diaria | Sin clima; 18 pronósticos recortados a 0; no “clásico” |
| GARCH(1,1) | Modela volatilidad persistente en residuos | No mejora pronóstico de la media |

### 5.4 Lecciones aprendidas y dificultades superadas

1. **Memoria y tiempo de cómputo con `auto_arima` / `pmdarima`.**  
   El ajuste automático de órdenes SARIMA sobre el train completo (**15.879 observaciones horarias**) agotó la memoria RAM del entorno local al explorar combinaciones de $(p,d,q)(P,D,Q)_{24}$. Esto no es un fallo del dataset sino del **espacio de búsqueda**: cada candidato estima decenas de parámetros sobre una matriz de diseño muy grande.  
   **Solución adoptada:** (a) ejecutar un **grid stepwise acotado** sobre una **submuestra de 3.000 horas** para reducir el costo de la búsqueda; (b) fijar $d=1$ y $D=1$ con base en ADF y ACF/PACF, acotando aún más el espacio; (c) **refit** del modelo ganador sobre el train completo con `low_memory=True` en statsmodels.  
   **Lección:** en series horarias largas conviene separar **identificación** (submuestra o restricciones) de **estimación final** (train completo). Documentar esta decisión evita que el lector interprete la submuestra como recorte arbitrario de datos.

2. **AIC in-sample no ordena el desempeño out-of-sample en horizontes largos.**  
   Holt-Winters clásico (A,A,A) y Holt-Winters amortiguado (A,Ad,A) tienen AIC casi idénticos en train (≈ 217.265 vs ≈ 217.192), pero en test de **672 horas** el RMSE diverge fuertemente: **9.896 veh/h** frente a **1.804 veh/h** (más de cinco veces peor).  
   **Por qué ocurre:** la tendencia **no amortiguada** extrapola un drift lineal durante 28 días; el nivel pronosticado **deriva** hacia valores irreales aunque el ajuste in-sample sea bueno. AIC premia ajuste y parsimonia **dentro del periodo de entrenamiento**, no estabilidad a largo plazo.  
   **Lección:** para comparar modelos con horizonte fijo de 672 h, la **validación out-of-sample** (mismas últimas cuatro semanas para todos) es imprescindible; confiar solo en AIC habría descartado incorrectamente la amortiguación o elegido el HW clásico como “casi equivalente”.

3. **ACF/PACF orientan, pero BIC sobre train completo puede cambiar la especificación.**  
   La lectura gráfica de ACF/PACF (Notebook 2) sugirió SARIMA$(1,1,1)(1,1,1)_{24}$: corte en PACF en lags 1 y 24. Tras refit en train completo, el candidato $(2,1,2)(1,1,1)_{24}$ obtenido del grid redujo el **BIC en ~690 puntos** (249.918 vs 249.228) pese a más parámetros AR y MA regulares.  
   **Lección:** ACF/PACF son una **propuesta inicial**, no la palabra final; con $n \approx 15.800$ el criterio de información puede preferir más memoria de corto plazo (AR(2), MA(2)) que deja menos estructura en residuos. Aun así, ese SARIMA **no superó** a HW en RMSE test, recordando que mejor BIC in-sample no garantiza mejor pronóstico.

4. **Huecos horarios (993 h) sin imputar.**  
   Respecto a una malla horaria regular en 2016–2017 faltan **993 horas** (~6 % del calendario esperado). No se interpolaron ni rellenaron: cualquier imputación (media, forward-fill, interpolación) habría introducido **observaciones sintéticas** de tráfico que no fueron medidas, distorsionando ACF, estacionalidad y verosimilitud de los modelos.  
   **Consecuencia:** los rezagos $y_{t-24}$, $y_{t-48}$, etc. pueden saltar periodos irregulares; los residuos de SARIMA/SARIMAX **no pasan** Ljung-Box ($p \approx 0$), y parte de esa autocorrelación residual podría deberse a discontinuidades en la serie además de estacionalidad semanal no modelada.  
   **Lección:** documentar huecos como limitación es preferible a “arreglar” la serie sin criterio físico; en un trabajo futuro conviene evaluar imputación explícita y comparar diagnósticos.

5. **MAPE poco fiable cuando el tráfico nocturno es muy bajo.**  
   Holt-Winters amortiguado alcanza MAPE **170 %** en test pese a RMSE razonable (1.804 veh/h), porque MAPE divide por $|y|$: en horas con ~200 veh/h reales, un error absoluto de ~1.500 veh/h (típico del piso nocturno mal predicho) produce ratios **superiores al 100 %**.  
   SARIMAX, con mejor MAE (1.411), tiene MAPE **80 %**; Prophet **42,5 %**, porque acerca mejor los valles.  
   **Lección:** para este fenómeno conviene priorizar **MAE y RMSE en veh/h** (interpretables y estables) y usar MAPE solo como referencia secundaria, especialmente cuando la serie tiene muchos valores pequeños en madrugada.

6. **Brecha Prophet vs clásicos: estacionalidades que $s=24$ no captura.**  
   Prophet obtiene RMSE **884 veh/h** frente a **1.804 veh/h** del mejor clásico (HW amortiguado): roughly la mitad. Esa diferencia no se explica solo por “más parámetros”, sino por **estacionalidad semanal y anual explícita** (`weekly_seasonality`, `yearly_seasonality`) y componentes de Fourier que permiten **amplitud variable** del ciclo diario (valles profundos ~500 veh/h vs piso HW ~2.500–3.000).  
   **Lección:** si el objetivo es competir con Prophet, los modelos clásicos con un único $s=24$ necesitan extensiones (dummy fin de semana, $s=168$, descomposición múltiple); la comparación numérica obliga a discutir **qué estructura falta**, no solo qué modelo “ganó” la tabla.

### 5.5 Limitaciones y trabajo futuro

1. **Alcance geográfico, temporal y de medición.**  
   Los resultados corresponden a **un solo punto** de la I-94 oeste (Minneapolis–Saint Paul), en un **subperiodo de dos años** (2016–2017), con **16.551** horas válidas tras deduplicación. No se puede extrapolar a otras autopistas, ciudades o regímenes de tráfico sin re-estimar. Tampoco se modela el efecto de obras, accidentes o cambios de capacidad vial; el sensor agrega todo el volumen horario en un único contador. Cualquier conclusión sobre “el tráfico en general” queda acotada a este tramo y ventana temporal.

2. **Exógenas meteorológicas observadas en el periodo test (evaluación optimista).**  
   SARIMAX se entrenó con clima en train y se evaluó en test usando **valores reales** de `temp`, `rain_1h` y `snow_1h` en las 672 h de pronóstico. En un escenario operativo (pronosticar tráfico **mañana**), esas variables no estarían disponibles salvo que se usen **pronósticos meteorológicos**, con error propio.  
   Por tanto, el RMSE de SARIMAX (1.855 veh/h) es un **techo optimista** del desempeño con clima; en producción el error podría ser mayor. Esta limitación no invalida la comparación SARIMA vs SARIMAX (misma información en ambos), pero sí la comparación “SARIMAX con clima perfecto” vs HW univariado.

3. **Feriados: contraste descriptivo fuerte, pero `is_holiday` no significativo en SARIMAX.**  
   En el EDA agregado, la media en feriados es **910 veh/h** frente a **3.293 veh/h** en no feriados: un feriado se parece más a un día de volumen muy bajo. Sin embargo, en el subperiodo solo hay **21 horas marcadas como feriado** (`is_holiday = 1`), es decir **~0,13 %** de las observaciones (proporción media ≈ 0,001).  
   En el SARIMAX ajustado (Notebook 3, tabla exportada en `coeficientes_sarimax.csv`), el coeficiente de `is_holiday` es **+27,0 veh/h** con **$p$-valor = 0,888** (umbral de significancia 5 %). **Cómo interpretarlo:** bajo $H_0$ (“el feriado no aporta nada linealmente al tráfico condicionado al resto del modelo”), un $p$-valor tan alto significa que **no hay evidencia estadística** para rechazar $H_0$; el +27 veh/h es compatible con **ruido muestral** dado el escaso número de feriados.  
   **Por qué contradice el EDA:** el contraste 910 vs 3.293 **no controla** hora del día, día de la semana ni dinámica SARIMA; muchos feriados caen en fines de semana o en horas ya bajas, y el efecto queda **absorbido** por la estacionalidad diaria ($s=24$) y por otros regresores. Además, un coeficiente **positivo** (+27) no coincide con la intuición del boxplot (menos tráfico en feriado), señal de **identificación débil** con tan pocos casos.  
   **Conclusión metodológica:** “no significativo” no significa “los feriados no importan en la vida real”, sino que **este modelo lineal con dummy no pudo estimar un efecto estable** con los datos disponibles; harían falta más feriados, regresores de fin de semana separados o un modelo que combine calendario y feriados (como Prophet con holidays).

4. **Estacionalidad semanal no modelada explícitamente en SARIMA/SARIMAX/HW ($s=24$ fijo).**  
   Con $s=24$, el perfil de las **8:00 de un lunes** y el de las **8:00 de un sábado** comparten el mismo componente estacional horario; la diferencia laborable/fin de semana (medias 3.543 vs 2.662 veh/h) queda en tendencia, residuo o en errores sistemáticos. Eso aparece en el overlay (picos distintos entre días) y en Ljung-Box (autocorrelación residual a lags múltiplos de 24).  
   Prophet mitiga esto con `weekly_seasonality=True`; los clásicos del TP no. Es una limitación **de especificación**, no necesariamente de la familia SARIMA en abstracto (podría ampliarse con $s=168$, regresores dummy o SARIMAX con `weekday`).

5. **LSTM y modelos neuronales omitidos.**  
   El plan del TP contemplaba LSTM como **apéndice opcional** si sobraba tiempo; se priorizó completar el pipeline clásico (ETS, Box–Jenkins, Prophet, GARCH) con diagnóstico reproducible. No hay comparación con redes recurrentes ni con ventanas deslizantes; no se puede afirmar si un LSTM superaría o no a Prophet en este dataset.

6. **Líneas de trabajo futuro concretas.**  
   - **Imputación de los 993 huecos** con criterios documentados (p. ej. interpolación solo en gaps cortos) y repetir Ljung-Box para ver cuánta autocorrelación residual desaparece.  
   - **SARIMA/SARIMAX con estacionalidad semanal** ($s=168$ o modelos con dos estacionalidades) o **regresores de fin de semana** en SARIMAX, para acercar el comportamiento a Prophet sin abandonar Box–Jenkins.  
   - **Pronóstico de exógenas** en test (clima previsto) para una evaluación más realista de SARIMAX.  
   - **Calendario de feriados en Prophet** o más datos multianuales para re-estimar `is_holiday` con poder estadístico.  
   - **LSTM u otros modelos secuenciales** como benchmark adicional, manteniendo la misma partición train/test de 672 h.

---

## Referencias de artefactos

| Artefacto | Ubicación |
|-----------|-----------|
| Datos procesados | `dataset/processed/traffic_horario.csv`, `exogenas_horarias.csv` |
| Partición train/test | `dataset/processed/train_test_split.json` |
| Métricas baselines + SARIMA/SARIMAX | `dataset/results/metricas_modelos.csv` |
| Tabla comparativa final | `dataset/results/comparativa_final.csv` |
| Órdenes SARIMA finales | `dataset/results/sarima_orders_final.json` |
| Coeficientes SARIMAX | `dataset/results/coeficientes_sarimax.csv` |
| ADF | `dataset/results/adf_results.csv` |
| Pronósticos test | `dataset/results/forecasts/sarima_test.csv`, `sarimax_test.csv` |
| Notebooks | `01_eda_preprocesamiento.ipynb` … `04_comparacion_conclusiones.ipynb` |
| Capturas | `capturas/notebook_2/`, `notebook_3/`, `notebook_4/` |

---

*Informe generado a partir de la ejecución de los notebooks del TP-02 (CEIA-AST). Semilla aleatoria: 381047.*
