import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "dataset_ml_ready.csv"

RESULTS_BASE_DIR = PROJECT_DIR / "results"
PREDICTIONS_BASE_DIR = PROJECT_DIR / "predictions"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_model_directories(model_name):
    """
    Crea las carpetas necesarias para guardar resultados y predicciones.
    """
    results_dir = os.path.join(RESULTS_BASE_DIR, model_name)
    predictions_dir = os.path.join(PREDICTIONS_BASE_DIR, model_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    return results_dir, predictions_dir


def evaluate_and_save_final_model(model_name, model, X_test, y_test, test_variants=None, training_time=None):
    """
    Evalúa un modelo final sobre el conjunto de test y guarda:
    - resultados finales en TXT
    - predicciones finales en CSV
    - métricas para la tabla comparativa final
    """

    results_dir, predictions_dir = create_model_directories(model_name)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    # ------------------------------------------------------------
    # Guardar resultados finales en TXT
    # ------------------------------------------------------------

    results_txt_path = os.path.join(results_dir, f"final_results_{model_name}.txt")

    with open(results_txt_path, "w") as f:
        f.write(f"Modelo final: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuración final del modelo:\n")
        f.write(str(model.named_steps["classifier"].get_params()))
        f.write("\n\n")

        if training_time is not None:
            f.write(f"Tiempo de entrenamiento: {training_time:.2f} minutos\n\n")

        f.write("Resultados en conjunto de test:\n")
        f.write(f"Accuracy: {acc:.5f}\n")
        f.write(f"Precision macro: {precision_macro:.5f}\n")
        f.write(f"Recall macro: {recall_macro:.5f}\n")
        f.write(f"F1 macro: {f1_macro:.5f}\n")
        f.write(f"Precision weighted: {precision_weighted:.5f}\n")
        f.write(f"Recall weighted: {recall_weighted:.5f}\n")
        f.write(f"F1 weighted: {f1_weighted:.5f}\n\n")

        f.write("Matriz de confusión:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    # ------------------------------------------------------------
    # Guardar predicciones finales
    # ------------------------------------------------------------

    predictions_path = os.path.join(predictions_dir, f"final_predictions_{model_name}.csv")

    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    if test_variants is not None:
        predictions_df.insert(0, "variant", test_variants.values)

    predictions_df.to_csv(predictions_path, index=False)

    # También guardamos una versión compatible con Excel en español
    predictions_excel_path = os.path.join(predictions_dir, f"final_predictions_{model_name}_excel.csv")
    predictions_df.to_csv(predictions_excel_path, index=False, sep=";")

    return {
        "model": model_name,
        "test_accuracy": round(acc, 5),
        "test_precision_macro": round(precision_macro, 5),
        "test_recall_macro": round(recall_macro, 5),
        "test_f1_macro": round(f1_macro, 5),
        "test_precision_weighted": round(precision_weighted, 5),
        "test_recall_weighted": round(recall_weighted, 5),
        "test_f1_weighted": round(f1_weighted, 5),
        "confusion_matrix": cm.tolist(),
        "training_time_minutes": round(training_time, 2) if training_time is not None else None
    }


# ============================================================
# CARGA DEL DATASET
# ============================================================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")

target = "label"

variants = df["variant"] if "variant" in df.columns else None

drop_cols = ["label", "variant", "clnsig"]
X = df.drop(columns=drop_cols)
y = df[target]

numeric_features = ["pos", "impact_num", "sift_score", "polyphen_score"]
categorical_features = ["chr", "ref", "alt", "clnvc", "consequence", "gene", "Consequence_group"]

print("\nDistribución de clases:")
print(y.value_counts())

print("\nVariables numéricas:")
print(numeric_features)

print("\nVariables categóricas:")
print(categorical_features)


# ============================================================
# DIVISIÓN TRAIN / TEST
# ============================================================

if variants is not None:
    X_train, X_test, y_train, y_test, variants_train, variants_test = train_test_split(
        X,
        y,
        variants,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    variants_test = None

print("\nDivisión de datos:")
print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# MODELOS FINALES CON HIPERPARÁMETROS OPTIMIZADOS
# ============================================================

final_models = {}


# ------------------------------------------------------------
# 1. Regresión logística
# Mejores parámetros encontrados:
# C = 10
# penalty = l2
# ------------------------------------------------------------

final_models["logistic_regression"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=10,
        penalty="l2",
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])


# ------------------------------------------------------------
# 2. Random Forest
# Mejores parámetros encontrados:
# max_depth = None
# max_features = sqrt
# min_samples_leaf = 1
# min_samples_split = 4
# n_estimators = 334
# ------------------------------------------------------------

final_models["random_forest"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=334,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])


# ------------------------------------------------------------
# 3. XGBoost
# Mejores parámetros encontrados:
# colsample_bytree = 0.8123620356542087
# learning_rate = 0.17254716573280354
# max_depth = 5
# n_estimators = 221
# reg_lambda = 1.5751320499779735
# subsample = 0.7468055921327309
# ------------------------------------------------------------

final_models["xgboost"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=221,
        max_depth=5,
        learning_rate=0.17254716573280354,
        subsample=0.7468055921327309,
        colsample_bytree=0.8123620356542087,
        reg_lambda=1.5751320499779735,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])


# ------------------------------------------------------------
# 4. SVM lineal eficiente
# Mejores parámetros encontrados:
# alpha = 0.0001
# penalty = l2
# learning_rate = optimal
# ------------------------------------------------------------

final_models["svm_linear"] = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SGDClassifier(
        loss="hinge",
        alpha=0.0001,
        penalty="l2",
        learning_rate="optimal",
        max_iter=3000,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN DE MODELOS FINALES
# ============================================================

summary_results = []

for model_name, model in final_models.items():

    print("\n" + "=" * 80)
    print(f"Entrenando modelo final: {model_name}")
    print("=" * 80)

    start_time = time.time()

    model.fit(X_train, y_train)

    elapsed_time = (time.time() - start_time) / 60

    print(f"Modelo finalizado: {model_name}")
    print(f"Tiempo de entrenamiento: {elapsed_time:.2f} minutos")

    result = evaluate_and_save_final_model(
        model_name=model_name,
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_variants=variants_test,
        training_time=elapsed_time
    )

    print("Resultados en test:")
    print(f"Accuracy: {result['test_accuracy']}")
    print(f"Precision macro: {result['test_precision_macro']}")
    print(f"Recall macro: {result['test_recall_macro']}")
    print(f"F1 macro: {result['test_f1_macro']}")
    print(f"Matriz de confusión: {result['confusion_matrix']}")

    summary_results.append(result)


# ============================================================
# TABLA FINAL COMPARATIVA
# ============================================================

summary_df = pd.DataFrame(summary_results)

final_summary_csv_path = os.path.join(RESULTS_BASE_DIR, "final_model_comparison_summary.csv")
final_summary_txt_path = os.path.join(RESULTS_BASE_DIR, "final_model_comparison_summary.txt")
final_summary_excel_path = os.path.join(RESULTS_BASE_DIR, "final_model_comparison_summary_excel.csv")

summary_df.to_csv(final_summary_csv_path, index=False)
summary_df.to_csv(final_summary_excel_path, index=False, sep=";")

with open(final_summary_txt_path, "w") as f:
    f.write("Comparación final de modelos entrenados con hiperparámetros optimizados\n")
    f.write("=" * 100 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 80)
print("Entrenamiento finalizado correctamente.")
print(f"Resumen final CSV guardado en: {final_summary_csv_path}")
print(f"Resumen final TXT guardado en: {final_summary_txt_path}")
print(f"Resumen final compatible con Excel guardado en: {final_summary_excel_path}")
print("=" * 80)
