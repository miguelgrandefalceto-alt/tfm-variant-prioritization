import os
import time
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# Rutas
# =========================

DATA_PATH = "data/dataset_ml_ready.csv"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "results/predictions"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# =========================
# Inicio
# =========================

start_total = time.time()

print("=" * 60)
print("Entrenamiento modelo SVM lineal eficiente")
print("=" * 60)


# =========================
# Cargar dataset
# =========================

print("\n[1/8] Cargando dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset cargado correctamente.")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")


# =========================
# Separar X e y
# =========================

print("\n[2/8] Preparando variables predictoras y etiqueta...")

X = df.drop(columns=["label", "variant", "clnsig"], errors="ignore")
y = df["label"]

print(f"Variables iniciales: {X.shape[1]}")
print("Distribución de clases:")
print(y.value_counts())


# =========================
# Detectar variables numéricas y categóricas
# =========================

print("\n[3/8] Detectando variables numéricas y categóricas...")

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print(f"Variables numéricas: {len(numeric_features)}")
print(numeric_features)

print(f"Variables categóricas: {len(categorical_features)}")
print(categorical_features)


# =========================
# División train/test
# =========================

print("\n[4/8] Dividiendo en entrenamiento y prueba...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Muestras de entrenamiento: {X_train.shape[0]}")
print(f"Muestras de prueba: {X_test.shape[0]}")


# =========================
# Preprocesamiento sparse con imputación
# =========================

print("\n[5/8] Definiendo preprocesamiento sparse con imputación...")

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop",
    sparse_threshold=0.3
)

print("Preprocesamiento creado: imputación + escalado numérico + one-hot sparse categórico")

# =========================
# Modelo SVM lineal eficiente
# =========================

print("\n[6/8] Definiendo modelo...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        class_weight="balanced",
        random_state=42,
        max_iter=2000,
        tol=1e-3,
        n_jobs=-1
    ))
])

print("Modelo creado: SGDClassifier con loss='hinge' equivalente a SVM lineal")


# =========================
# Entrenar
# =========================

print("\n[7/8] Entrenando modelo...")

start_train = time.time()

model.fit(X_train, y_train)

train_time = time.time() - start_train

print(f"Entrenamiento completado en {train_time / 60:.2f} minutos")


# =========================
# Evaluar
# =========================

print("\n[8/8] Realizando predicciones y calculando métricas...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

total_time = time.time() - start_total


# =========================
# Mostrar resultados
# =========================

print("\n" + "=" * 60)
print("Resultados SVM lineal eficiente")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.5f}")

print("\nMatriz de confusión:")
print(cm)

print("\nClassification report:")
print(report)

print(f"Tiempo de entrenamiento: {train_time / 60:.2f} minutos")
print(f"Tiempo total del proceso: {total_time / 60:.2f} minutos")


# =========================
# Guardar resultados
# =========================

results_path = os.path.join(RESULTS_DIR, "results_svm.txt")

with open(results_path, "w", encoding="utf-8") as f:
    f.write("SVM lineal eficiente\n")
    f.write("Modelo: SGDClassifier(loss='hinge')\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy: {accuracy:.5f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm))
    f.write("\n\nClassification report:\n")
    f.write(report)
    f.write("\n")
    f.write(f"Tiempo de entrenamiento: {train_time / 60:.2f} minutos\n")
    f.write(f"Tiempo total del proceso: {total_time / 60:.2f} minutos\n")

print(f"\nResultados guardados en: {results_path}")


# =========================
# Guardar predicciones
# =========================

predictions = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})

predictions_path = os.path.join(PREDICTIONS_DIR, "predictions_svm.csv")

predictions.to_csv(predictions_path, index=False)

print(f"Predicciones guardadas en: {predictions_path}")

print("\nProceso finalizado correctamente.")
