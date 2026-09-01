# API de Predicción de Crédito

Microservicio que predice si un crédito será aprobado o no,
usando un modelo Random Forest entrenado con datos de clientes.

## Estructura del proyecto

api-prestamos/
├── train_model.py # Entrenamiento y exportación del modelo
├── modelo.joblib # Modelo serializado
├── app.py # API FastAPI
├── requirements.txt # Dependencias
├── Dockerfile # Contenerización
└── datos_credito.parquet # Dataset original


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
  "probabilidad": 0.9
}
```

## Tecnologías

- Python 3.10
- FastAPI + Uvicorn
- scikit-learn (Random Forest)
- Docker

## Autores

Elizabeth Galiden Obando
Ivan Andres Torres Blanco
Miguel Ángel Ordóñez Tandioy

