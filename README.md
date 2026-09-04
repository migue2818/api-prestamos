# API de Predicción de Crédito

Microservicio que predice si un crédito será aprobado o no,
usando un modelo de Machine Learning desplegado con FastAPI y Docker.

## Estructura del proyecto

- `eda.py` — Análisis exploratorio de datos
- `comparar_modelos.py` — Comparación de modelos candidatos
- `train_model.py` — Pipeline de entrenamiento (Logistic Regression)
- `modelo.joblib` — Pipeline serializado
- `app.py` — API FastAPI
- `requirements.txt` — Dependencias con versiones fijadas
- `Dockerfile` — Contenerización
- `datos_credito.parquet` — Dataset original
- `eda_output/` — Gráficas del análisis exploratorio



## Proceso de desarrollo

### 1. Análisis Exploratorio (EDA)
Se analizó el dataset de 15,000 registros con 9 variables numéricas.
Se verificó que no hay nulos ni variables categóricas.
Se generaron gráficas de correlación, distribuciones y boxplots.

### 2. Comparación de modelos
Se evaluaron 3 modelos con cross-validation (5 folds):

| Modelo              | Accuracy | F1     | Recall | AUC-ROC |
|---------------------|----------|--------|--------|---------|
| Logistic Regression | 0.7357   | 0.8155 | 0.8843 | 0.7614  |
| Random Forest       | 0.7187   | 0.8013 | 0.8585 | 0.7380  |
| Gradient Boosting   | 0.7332   | 0.8138 | 0.8822 | 0.7631  |

Se seleccionó **Logistic Regression** por mejor F1-score e interpretabilidad.

### 3. Pipeline de entrenamiento
El modelo se entrena dentro de un `sklearn.Pipeline` que incluye:
- `SimpleImputer` (imputación de nulos por mediana)
- `StandardScaler` (escalado de variables)
- `LogisticRegression` (clasificación)

## Requisitos

- Docker instalado

## Uso

### 1. Compilar la imagen

```bash
docker build -t api-prestamos .
```

### 2. Ejecutar el contenedor

```bash
docker run -p 8000:8000 api-prestamos
```

### 3. Probar el endpoint

Abrir `http://localhost:8000/docs` o enviar un POST a `/predict`:

```json
{
  "edad": 35,
  "personas_a_cargo": 2,
  "ingreso_mensual": 3500000,
  "monto_solicitado": 10000000,
  "plazo_meses": 36,
  "cuota_estimada": 400000,
  "score_crediticio": 700,
  "dti_previo": 0.25,
  "dti_total_proyectado": 0.45
}
```

### Respuesta esperada

```json
{
  "decision": "Aprobado",
  "probabilidad_aprobacion": 0.8098
}
```

## Validaciones

- Edad: entre 18 y 100 años
- Score crediticio: entre 300 y 850

## Tecnologías

- Python 3.12
- FastAPI + Uvicorn
- scikit-learn (Logistic Regression + Pipeline)
- Docker

## Autores

- Elizabeth Galindez Obando
- Ivan Andres Torres Blanco
- Miguel Ángel Ordóñez Tandioy
