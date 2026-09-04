import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar datos
df = pd.read_parquet("datos_credito.parquet")

# 2. Separar features y target
X = df.drop(columns=["aprobado"])
y = df["aprobado"]

# 3. Identificar columnas 
cols_numericas = X.select_dtypes(include="number").columns.tolist()

# 4. Preprocesamiento
preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), cols_numericas)
])

# 5. Pipeline completo (Logistic Regression)
pipeline = Pipeline([
    ("preprocesador", preprocesador),
    ("modelo", LogisticRegression(max_iter=1000, random_state=42))
])

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Entrenar
pipeline.fit(X_train, y_train)

# 8. Evaluar
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Aprobado", "Aprobado"]))

# 9. Cross-validation con F1
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1")
print(f"F1 cross-val: {scores.mean():.4f} ± {scores.std():.4f}")

# 10. Exportar pipeline completo
joblib.dump(pipeline, "modelo.joblib")
print("Pipeline guardado como modelo.joblib")