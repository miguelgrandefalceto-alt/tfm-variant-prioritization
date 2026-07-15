import os
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


DATA_PATH = "data/dataset_ml_ready.csv"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "results/predictions"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

start_total = time.time()

print("=" * 60)
print("Entrenamiento SVM RBF con muestra reducida")
print("=" * 60)

print("\n[1/8] Cargando dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")

print("\n[2/8] Tomando muestra estratificada...")

df_sample, _ = train_test_split(
    df,
    train_size=15000,
    random_state=42,
    stratify=df["label"]
)

print(f"Muestra usada: {df_sample.shape[0]} filas")
print(df_sample["label"].value_counts())

X = df_sample.drop(columns=["label", "variant", "clnsig"], errors="ignore")
y = df_sample["label"]

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\n[3/8] Variables detectadas")
print(f"Numéricas: {numeric_features}")
print(f"Categóricas: {categorical_features}")

print("\n[4/8] Dividiendo train/test...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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

print("\n[5/8] Definiendo SVM RBF...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])

print("\n[6/8] Entrenando modelo...")
start_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_train

print(f"Entrenamiento completado en {train_time / 60:.2f} minutos")

print("\n[7/8] Evaluando...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

total_time = time.time() - start_total

print("\n" + "=" * 60)
print("Resultados SVM RBF muestra reducida")
print("=" * 60)
print(f"\nAccuracy: {accuracy:.5f}")
print("\nMatriz de confusión:")
print(cm)
print("\nClassification report:")
print(report)
print(f"Tiempo de entrenamiento: {train_time / 60:.2f} minutos")
print(f"Tiempo total: {total_time / 60:.2f} minutos")

results_path = os.path.join(RESULTS_DIR, "results_svm_rbf_sample.txt")

with open(results_path, "w", encoding="utf-8") as f:
    f.write("SVM RBF con muestra reducida\n")
    f.write("=" * 60 + "\n\n")
    f.write("Muestra: 15000 variantes\n")
    f.write(f"Accuracy: {accuracy:.5f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm))
    f.write("\n\nClassification report:\n")
    f.write(report)
    f.write("\n")
    f.write(f"Tiempo de entrenamiento: {train_time / 60:.2f} minutos\n")
    f.write(f"Tiempo total: {total_time / 60:.2f} minutos\n")

predictions = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": y_pred
})

predictions_path = os.path.join(PREDICTIONS_DIR, "predictions_svm_rbf_sample.csv")
predictions.to_csv(predictions_path, index=False)

print(f"\nResultados guardados en: {results_path}")
print(f"Predicciones guardadas en: {predictions_path}")
print("\nProceso finalizado correctamente.")
