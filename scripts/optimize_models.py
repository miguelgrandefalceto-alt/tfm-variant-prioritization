import os
import json
import time
import warnings
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
CV_FOLDS = 3

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

REFIT_METRIC = "f1_macro"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def create_model_directories(model_name):
    """
    Crea las carpetas de resultados y predicciones para cada modelo.
    """
    results_dir = os.path.join(RESULTS_BASE_DIR, model_name)
    predictions_dir = os.path.join(PREDICTIONS_BASE_DIR, model_name)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    return results_dir, predictions_dir


def make_json_serializable(params):
    """
    Convierte los mejores parámetros en un formato compatible con JSON.
    """
    clean_params = {}

    for key, value in params.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            clean_params[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            clean_params[key] = float(value)
        else:
            clean_params[key] = value

    return clean_params


def evaluate_and_save(model_name, search_model, X_test, y_test, test_variants=None):
    """
    Evalúa el mejor modelo sobre el conjunto de test y guarda:
    - resultados generales en TXT
    - resultados completos de validación cruzada en CSV
    - mejores hiperparámetros en JSON
    - predicciones en CSV
    """

    results_dir, predictions_dir = create_model_directories(model_name)

    best_model = search_model.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    best_params = make_json_serializable(search_model.best_params_)

    # ------------------------------------------------------------
    # Guardar resumen TXT
    # ------------------------------------------------------------

    results_txt_path = os.path.join(results_dir, f"results_{model_name}.txt")

    with open(results_txt_path, "w") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Mejores hiperparámetros:\n")
        f.write(json.dumps(best_params, indent=4))
        f.write("\n\n")

        f.write(f"Mejor score de validación cruzada ({REFIT_METRIC}): {search_model.best_score_:.5f}\n\n")

        f.write("Resultados en conjunto de test:\n")
        f.write(f"Accuracy: {acc:.5f}\n")
        f.write(f"Precision macro: {precision_macro:.5f}\n")
        f.write(f"Recall macro: {recall_macro:.5f}\n")
        f.write(f"F1 macro: {f1_macro:.5f}\n\n")

        f.write("Matriz de confusión:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    # ------------------------------------------------------------
    # Guardar resultados completos de CV
    # ------------------------------------------------------------

    cv_results_path = os.path.join(results_dir, f"cv_results_{model_name}.csv")
    cv_results_df = pd.DataFrame(search_model.cv_results_)
    cv_results_df.to_csv(cv_results_path, index=False)

    # ------------------------------------------------------------
    # Guardar mejores hiperparámetros
    # ------------------------------------------------------------

    best_params_path = os.path.join(results_dir, f"best_params_{model_name}.json")

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    # ------------------------------------------------------------
    # Guardar predicciones
    # ------------------------------------------------------------

    predictions_path = os.path.join(predictions_dir, f"predictions_{model_name}.csv")

    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    if test_variants is not None:
        predictions_df.insert(0, "variant", test_variants.values)

    predictions_df.to_csv(predictions_path, index=False)

    return {
        "model": model_name,
        "best_cv_f1_macro": round(search_model.best_score_, 5),
        "test_accuracy": round(acc, 5),
        "test_precision_macro": round(precision_macro, 5),
        "test_recall_macro": round(recall_macro, 5),
        "test_f1_macro": round(f1_macro, 5),
        "confusion_matrix": cm.tolist(),
        "best_params": best_params
    }


# ============================================================
# CARGA DEL DATASET
# ============================================================

print("Cargando dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")

target = "label"

# Guardamos variant aparte para poder identificar predicciones en test
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
# DEFINICIÓN DE MODELOS
# ============================================================

models = {}


# ------------------------------------------------------------
# 1. Regresión logística
# ------------------------------------------------------------

logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

logreg_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l2"]
}

models["logistic_regression"] = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid=logreg_params,
    scoring=SCORING,
    refit=REFIT_METRIC,
    cv=CV_FOLDS,
    n_jobs=-1,
    verbose=2
)


# ------------------------------------------------------------
# 2. Random Forest
# ------------------------------------------------------------

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

rf_params = {
    "classifier__n_estimators": randint(200, 600),
    "classifier__max_depth": [None, 10, 20, 30, 40],
    "classifier__min_samples_split": randint(2, 10),
    "classifier__min_samples_leaf": randint(1, 5),
    "classifier__max_features": ["sqrt", "log2"]
}

models["random_forest"] = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=10,
    scoring=SCORING,
    refit=REFIT_METRIC,
    cv=CV_FOLDS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2
)


# ------------------------------------------------------------
# 3. XGBoost
# ------------------------------------------------------------

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

xgb_params = {
    "classifier__n_estimators": randint(150, 500),
    "classifier__max_depth": randint(3, 8),
    "classifier__learning_rate": loguniform(0.01, 0.2),
    "classifier__subsample": uniform(0.7, 0.3),
    "classifier__colsample_bytree": uniform(0.7, 0.3),
    "classifier__reg_lambda": loguniform(0.1, 10)
}

models["xgboost"] = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_params,
    n_iter=10,
    scoring=SCORING,
    refit=REFIT_METRIC,
    cv=CV_FOLDS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2
)


# ------------------------------------------------------------
# 4. SVM lineal eficiente
# ------------------------------------------------------------

svm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SGDClassifier(
        loss="hinge",
        max_iter=3000,
        tol=1e-3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

svm_params = {
    "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
    "classifier__penalty": ["l2", "elasticnet"],
    "classifier__learning_rate": ["optimal"]
}

models["svm_linear"] = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_params,
    scoring=SCORING,
    refit=REFIT_METRIC,
    cv=CV_FOLDS,
    n_jobs=-1,
    verbose=2
)


# ============================================================
# ENTRENAMIENTO, OPTIMIZACIÓN Y EVALUACIÓN FINAL
# ============================================================

summary_results = []

for model_name, search_model in models.items():

    print("\n" + "=" * 80)
    print(f"Optimizando modelo: {model_name}")
    print("=" * 80)

    start_time = time.time()

    search_model.fit(X_train, y_train)

    elapsed_time = time.time() - start_time

    print(f"\nModelo finalizado: {model_name}")
    print(f"Tiempo total: {elapsed_time / 60:.2f} minutos")
    print(f"Mejor {REFIT_METRIC} en CV: {search_model.best_score_:.5f}")
    print(f"Mejores hiperparámetros:")
    print(search_model.best_params_)

    result = evaluate_and_save(
        model_name=model_name,
        search_model=search_model,
        X_test=X_test,
        y_test=y_test,
        test_variants=variants_test
    )

    result["training_time_minutes"] = round(elapsed_time / 60, 2)
    summary_results.append(result)


# ============================================================
# TABLA FINAL COMPARATIVA
# ============================================================

summary_df = pd.DataFrame(summary_results)

summary_csv_path = os.path.join(RESULTS_BASE_DIR, "model_comparison_summary.csv")
summary_txt_path = os.path.join(RESULTS_BASE_DIR, "model_comparison_summary.txt")

summary_df.to_csv(summary_csv_path, index=False)

with open(summary_txt_path, "w") as f:
    f.write("Comparación final de modelos optimizados\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 80)
print("Optimización finalizada correctamente.")
print(f"Resumen CSV guardado en: {summary_csv_path}")
print(f"Resumen TXT guardado en: {summary_txt_path}")
print("=" * 80)
