import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar datos
df = pd.read_parquet("datos_credito.parquet")

# 2. Separar features (X) y target (y)
X = df.drop(columns=["aprobado"])
y = df["aprobado"]

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 5. Evaluar
y_pred = modelo.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Aprobado", "Aprobado"]))

# 6. Exportar modelo
joblib.dump(modelo, "modelo.joblib")
print("Modelo guardado como modelo.joblib")