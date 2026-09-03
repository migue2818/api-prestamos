from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

# Cargar modelo al iniciar
modelo = joblib.load("modelo.joblib")

app = FastAPI(title="API Predicción de Crédito")

# Esquema de entrada
class ClienteInput(BaseModel):
    edad: int = Field(ge=18, le=100)
    personas_a_cargo: int
    ingreso_mensual: float
    monto_solicitado: float
    plazo_meses: int
    cuota_estimada: float
    score_crediticio: int = Field(ge=300, le=850)
    dti_previo: float
    dti_total_proyectado: float

@app.post("/predict")
def predecir(cliente: ClienteInput):
    datos = pd.DataFrame([cliente.model_dump()])

    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0]

    return {
    "decision": "Aprobado" if prediccion == 1 else "No Aprobado",
    "probabilidad_aprobacion": round(float(probabilidad[1]), 4)
}