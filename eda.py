import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("datos_credito.parquet")

# 1. Dimensiones y tipos
print("=== DIMENSIONES ===")
print(df.shape)
print("\n=== TIPOS ===")
print(df.dtypes)

# 2. Nulos
print("\n=== NULOS ===")
print(df.isnull().sum())

# 3. Estadísticas descriptivas
print("\n=== ESTADÍSTICAS ===")
print(df.describe())

# 4. Distribución del target
print("\n=== DISTRIBUCIÓN TARGET ===")
print(df["aprobado"].value_counts(normalize=True))

# 5. Tipos de columnas
print("\n=== COLUMNAS NUMÉRICAS ===")
print(df.select_dtypes(include="number").columns.tolist())
print("\n=== COLUMNAS CATEGÓRICAS ===")
print(df.select_dtypes(include="object").columns.tolist())

# 6. Correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación")
plt.tight_layout()
plt.savefig("correlaciones.png")
print("\nGuardado: correlaciones.png")

# 7. Distribuciones
df.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.savefig("distribuciones.png")
print("Guardado: distribuciones.png")

# 8. Boxplots por clase
for col in df.select_dtypes(include="number").columns:
    if col != "aprobado":
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="aprobado", y=col, data=df)
        plt.title(f"{col} por clase")
        plt.tight_layout()
        plt.savefig(f"boxplot_{col}.png")
        plt.close()
        print(f"Guardado: boxplot_{col}.png")

print("\n=== EDA COMPLETO ===")