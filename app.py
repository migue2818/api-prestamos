from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo al iniciar
modelo = joblib.load("modelo.joblib")

app = FastAPI(title="API Predicción de Crédito")

# Esquema de entrada
class ClienteInput(BaseModel):
    edad: int
    personas_a_cargo: int
    ingreso_mensual: float
    monto_solicitado: float
    plazo_meses: int
    cuota_estimada: float
    score_crediticio: int
    dti_previo: float
    dti_total_proyectado: float

@app.post("/predict")
def predecir(cliente: ClienteInput):
    datos = np.array([[
        cliente.edad,
        cliente.personas_a_cargo,
        cliente.ingreso_mensual,
        cliente.monto_solicitado,
        cliente.plazo_meses,
        cliente.cuota_estimada,
        cliente.score_crediticio,
        cliente.dti_previo,
        cliente.dti_total_proyectado
    ]])

    prediccion = modelo.predict(datos)[0]
    probabilidad = modelo.predict_proba(datos)[0]

    return {
        "decision": "Aprobado" if prediccion == 1 else "No Aprobado",
        "probabilidad": round(float(probabilidad[prediccion]), 4)
    }