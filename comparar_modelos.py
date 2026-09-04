import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# 1. Cargar datos
df = pd.read_parquet("datos_credito.parquet")
X = df.drop(columns=["aprobado"])
y = df["aprobado"]

# 2. Preprocesamiento
cols_numericas = X.select_dtypes(include="number").columns.tolist()

preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), cols_numericas)
])

# 3. Modelos candidatos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 4. Métricas a evaluar
metricas = ["accuracy", "f1", "recall", "roc_auc"]

# 5. Comparar modelos con cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print(f"{'Modelo':<25} {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'AUC-ROC':>10}")
print("=" * 70)

resultados = {}

for nombre, modelo in modelos.items():
    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", modelo)
    ])

    scores = {}
    for metrica in metricas:
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metrica)
        scores[metrica] = cv_scores.mean()

    resultados[nombre] = scores

    print(f"{nombre:<25} {scores['accuracy']:>10.4f} {scores['f1']:>10.4f} {scores['recall']:>10.4f} {scores['roc_auc']:>10.4f}")

# 6. Seleccionar el mejor por F1
mejor_nombre = max(resultados, key=lambda x: resultados[x]["f1"])
print("=" * 70)
print(f"\nMejor modelo por F1: {mejor_nombre}")
print(f"  Accuracy:  {resultados[mejor_nombre]['accuracy']:.4f}")
print(f"  F1:        {resultados[mejor_nombre]['f1']:.4f}")
print(f"  Recall:    {resultados[mejor_nombre]['recall']:.4f}")
print(f"  AUC-ROC:   {resultados[mejor_nombre]['roc_auc']:.4f}")