# Detección de fraude con concept drift — IEEE-CIS

Proyecto del curso *Planificación y Toma de Decisiones en IA*. Sistema de clasificación
de transacciones financieras como fraudulentas usando el dataset IEEE-CIS, asumiendo
de entrada que la distribución de los datos no es estacionaria.

El objetivo no era solo entrenar un buen modelo, sino mostrar el ciclo completo: EDA →
preprocesamiento con split temporal → tres modelos comparados → análisis de drift →
servicio de inferencia desplegado en Docker → script de reentrenamiento que cierra el
loop adaptativo.

---

## Resultados que vale la pena reportar

Sobre el set de test (último mes y medio del periodo, 92.427 transacciones):

| Modelo               | ROC-AUC | PR-AUC | F1     | Recall | Precision |
|----------------------|---------|--------|--------|--------|-----------|
| Logistic Regression  | 0.8223  | 0.1710 | 0.1872 | 0.7373 | 0.1072    |
| Random Forest        | 0.8810  | 0.4864 | 0.3597 | 0.6010 | 0.2566    |
| **XGBoost (GPU)**    | **0.8962** | **0.5469** | **0.5422** | 0.4812 | 0.6209 |

Tres cosas a notar:

1. **PR-AUC importa más que ROC-AUC** acá: el dataset es 3,5% positivo y el costo
   de un falso positivo (bloquear una transacción legítima) es alto. PR-AUC refleja
   directamente esa tensión. ROC-AUC por sí solo da una imagen demasiado optimista.
2. **Logistic Regression no es viable en producción** aunque su ROC-AUC parece OK:
   con precisión 0.107 marcaría 10 falsos positivos por cada fraude real.
3. **XGBoost gana por interacciones no lineales**, no por ajuste fino. Lo confirmamos
   probando que reducir `max_depth` o subir regularización degrada poco — el lift
   viene del tipo de modelo.

### El hallazgo central sobre concept drift

| Estrategia                  | ROC-AUC promedio | PR-AUC promedio |
|-----------------------------|------------------|-----------------|
| Estática (modelo fijo)      | 0.8967           | 0.5475          |
| Adaptativa (reentrena semanal) | 0.9256        | 0.6112          |
| **Mejora**                  | **+3.2%**        | **+11.6%**      |

Lo curioso: si solo miramos PSI (la métrica estándar para drift en banca), no hay
features con drift significativo (PSI máximo = 0.030, umbral de alerta = 0.20). Sin
embargo, el modelo adaptativo gana 11,6% en PR-AUC. Conclusión: hay **concept drift**
(la relación feature→target cambia) que las métricas de **distribution drift** no capturan.
Por eso el monitoreo en producción debería incluir performance metrics, no solo PSI.

Detalle completo en `notebooks/04_drift_analysis.py` y en `artifacts/drift_summary.json`.

---

## Estructura del repo

```
.
├── data/                       # CSVs del IEEE-CIS (no versionados, en .gitignore)
├── notebooks/                  # Notebooks marimo (.py)
│   ├── 01_eda.py               # Exploración + análisis temporal
│   ├── 02_preprocessing.py     # Split temporal + encoding + parquets
│   ├── 03_training.py          # 3 modelos + evaluación semanal
│   └── 04_drift_analysis.py    # KS, PSI, simulación adaptativa, triggers
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI: /predict, /predict/batch, /health, /version
│   │   ├── schemas.py          # Pydantic
│   │   └── example_request.py  # Cliente de ejemplo con datos del test set
│   └── retrain.py              # Pipeline de reentrenamiento end-to-end (sin marimo)
├── artifacts/                  # Modelos, encoders, métricas, CSVs de drift
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            # Dev (notebooks)
├── requirements-api.txt        # Producción (solo lo que la API necesita)
└── README.md
```

---

## Setup

Probado con Python 3.11 y `uv`. Para entrenar usamos GPU (CUDA 12.8 sobre RTX 4070);
si no hay GPU, se puede correr todo en CPU pasando `--device cpu` al script de retrain
(es ~10× más lento pero funciona).

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Para PyTorch/XGBoost con CUDA 12.8 específicamente:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Cómo ejecutar el proyecto

### 1. Conseguir el dataset

Descargar desde Kaggle (requiere cuenta + aceptar las reglas de la competencia):

```powershell
kaggle competitions download -c ieee-fraud-detection -p data/
```

Esperamos cuatro CSVs en `data/`. Solo se usan `train_transaction.csv` y
`train_identity.csv` — los `test_*` no traen labels porque eran para la competencia.

### 2. Correr los notebooks

Los notebooks están en formato marimo (son `.py`, no `.ipynb`). Hay que correrlos
en orden porque cada uno consume las salidas del anterior:

```powershell
marimo edit notebooks/01_eda.py
marimo edit notebooks/02_preprocessing.py
marimo edit notebooks/03_training.py
marimo edit notebooks/04_drift_analysis.py
```

Tiempos esperados con RTX 4070:

- `02_preprocessing.py`: ~2 min (lee 680 MB de CSVs, hace merge, encoding y guarda parquets)
- `03_training.py`: ~5 min (LogReg ~90s, RF ~40s, XGBoost ~75s + evaluación semanal)
- `04_drift_analysis.py`: ~10 min (incluye 4 reentrenamientos para la simulación adaptativa)

Cada notebook escribe sus salidas en `artifacts/`.

### 3. Levantar la API

Con el venv activo, desde la raíz:

```powershell
docker-compose up -d --build fraud-api
```

Swagger UI en `http://localhost:8000/docs`. Para probar con datos reales:

```powershell
python src/api/example_request.py
```

Ese script toma 5 transacciones del `test.parquet`, las manda a `/predict` y a
`/predict/batch`, y muestra las decisiones (`allow` / `review` / `block`).

### 4. Docker

```powershell
docker compose up -d --build
```

El `artifacts/` está montado como volumen read-only. Si reentreno el modelo (notebook 03
o `retrain.py`), basta con `docker compose restart fraud-api` para que cargue la nueva
versión sin rebuild de la imagen.

### 5. Reentrenamiento programado

El script `src/retrain.py` reproduce todo el pipeline (preprocesamiento + entrenamiento)
sin necesidad de marimo. Pensado para correr en un cron job o como respuesta automática
cuando un trigger de drift se dispara.

```powershell
python -m src.retrain --device cuda
```

Argumentos relevantes:

| Argumento              | Default          | Para qué                                       |
|------------------------|------------------|------------------------------------------------|
| `--data-dir`           | `data`           | Dónde están los CSVs                           |
| `--artifacts-dir`      | `artifacts`      | Dónde escribir `xgb_model.joblib` etc.         |
| `--reference-date`     | `2017-12-01`     | Fecha base para `TransactionDT`                |
| `--train-end`          | 80% del periodo  | Hasta cuándo usar como train (`YYYY-MM-DD`)    |
| `--val-days`           | `22`             | Días para validación post-train                |
| `--high-null-threshold`| `0.90`           | % de nulos para descartar columna              |
| `--n-estimators`       | `500`            | Árboles XGBoost                                |
| `--device`             | `cuda`           | `cuda` o `cpu`                                 |

Genera un `retrain_metadata.json` con timestamp, métricas y tiempo total — útil para
auditoría.

---

## Decisiones de diseño y por qué

### Split temporal en lugar de aleatorio

La tasa de fraude diaria oscila entre 1,1% y 7,0% (CV=28%) y muestra patrones que
varían con el tiempo. Un split aleatorio mezclaría datos de mayo en train con datos
de febrero en test y sobreestimaría el rendimiento. El corte cronológico (74% / 10% / 16%)
es más conservador y refleja cómo se comportaría el modelo con datos futuros.

Verificable: la tasa de fraude por split queda en 3,50% / 3,52% / 3,48% — confirma
que la variación temporal es de alta frecuencia (semanal/diaria), no de bloque.

### Por qué tirar columnas con >90% nulos y no menos

Solo 12 columnas superan ese umbral. Bajamos a 80% y la performance en validación
empeoró ~0,5pp en PR-AUC; subimos a 95% y no cambió nada. XGBoost maneja NaN
nativamente, así que no hay incentivo a limpiar más allá de lo razonable.

### Por qué Label Encoding y no One-Hot

XGBoost funciona bien con categóricas como enteros y nos ahorra ~3.500 columnas
extra que tendría one-hot (algunas categóricas tienen 100+ valores únicos). Probamos
target encoding también pero introducía data leakage si no se hacía con cuidado por
fold.

### Por qué `class_weight="balanced"` solo en LogReg/RF y `scale_pos_weight` en XGBoost

Es la API correspondiente en cada librería. En XGBoost calculamos
`scale_pos_weight = n_negative / n_positive ≈ 27.5` lo cual es equivalente
matemáticamente a `class_weight="balanced"` pero más explícito.

### Por qué thresholds 0,30 y 0,70 en la API

La API devuelve `allow` / `review` / `block` según la probabilidad:

- `proba < 0.30` → `allow`: en la curva PR del modelo, este punto da recall ~0,65 y precision ~0,40 (filtro inicial razonable).
- `0.30 ≤ proba < 0.70` → `review`: ambigüedad, lo decide un humano.
- `proba ≥ 0.70` → `block`: precisión >0,80, suficiente para automatizar el bloqueo.

Estos números son ajustables y en un escenario real deberían validarse con el área
de negocio (cuál es el costo de un falso positivo bloqueado vs un fraude que pasa).

### Por qué reentrenamiento programado + trigger en lugar de uno solo

La simulación muestra que el reentrenamiento semanal mejora PR-AUC en 11,6%, pero
el trigger por caída de performance se activa solo en una semana del periodo de test
(la del 14-mayo). Conclusión operativa: una cadencia fija (cada 2-4 semanas) es
suficiente la mayor parte del tiempo, pero conviene tener el trigger como red de
seguridad para reaccionar antes en caso de un evento brusco.

---

## Limitaciones conocidas

- **Periodo corto**: el dataset cubre solo 6 meses, no podemos validar frente a
  estacionalidades anuales (Black Friday, Navidad, fin de año fiscal). El modelo
  podría fallar cuando aparezcan esos picos.
- **Fecha base es una convención**: `2017-12-01` es lo que la comunidad de Kaggle
  asumió, no está documentada oficialmente. El análisis temporal es robusto a esto
  (los offsets relativos no cambian) pero los nombres de mes/día sí.
- **Un solo worker en la API**: para producción real haría falta un load balancer
  con múltiples réplicas detrás. Como está, da ~70 req/s en CPU.
- **Ventana expansiva de reentrenamiento**: para drift más fuerte convendría una
  ventana deslizante (descartar datos viejos). En este dataset el drift es moderado
  y la ventana expansiva es suficiente.
- **Sin autenticación en la API**: en un entorno real iría detrás de un API gateway
  con tokens. Para el alcance del proyecto es aceptable.

---

## Stack

- Python 3.11 — pandas, numpy, scikit-learn, xgboost
- marimo (notebooks reactivos guardados como `.py`, mucho mejor que `.ipynb` para git)
- FastAPI + uvicorn
- Docker + docker-compose
- GPU: CUDA 12.8, RTX 4070

---

## Mapeo a la rúbrica del curso

| Sección de la rúbrica                              | Dónde está                                          |
|----------------------------------------------------|-----------------------------------------------------|
| Análisis y preparación de datos                    | `notebooks/01_eda.py`, `02_preprocessing.py`        |
| Modelado y evaluación temporal                     | `notebooks/03_training.py`                          |
| Detección de drift                                 | `notebooks/04_drift_analysis.py` (KS, chi², PSI)    |
| Estrategia de adaptación                           | `notebooks/04_drift_analysis.py` + `src/retrain.py` |
| Despliegue del sistema                             | `src/api/`, `Dockerfile`, `docker-compose.yml`      |
| Reproducibilidad / calidad del producto            | Este README + `requirements-api.txt`                |
