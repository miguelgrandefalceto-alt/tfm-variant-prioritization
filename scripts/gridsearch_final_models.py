#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_gridsearch_final_models.py

Optimización final de modelos mediante GridSearchCV para el TFM:
Priorización de variantes potencialmente patogénicas mediante Machine Learning.

Modelos evaluados:
- Regresión logística
- Random Forest
- XGBoost
- SVM lineal eficiente mediante SGDClassifier

Características principales:
- Uso de Pipeline y ColumnTransformer para evitar fuga de datos.
- Codificación one-hot sparse para variables categóricas.
- Imputación de valores ausentes.
- Escalado de variables numéricas.
- Optimización mediante GridSearchCV sobre el conjunto de entrenamiento.
- Validación cruzada estratificada.
- Métrica de optimización: F1-macro.
- Evaluación final sobre conjunto de test independiente.
- Guardado de resultados, matrices de confusión, mejores parámetros y predicciones.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "No se ha encontrado xgboost. Instálalo con: pip install xgboost"
    )


warnings.filterwarnings("ignore")


# 1. CONFIGURACIÓN GENERAL

RANDOM_STATE = 42

PROJECT_DIR = Path("/mnt/c/Users/Usuario/Desktop/UNIR/TFM")
DATA_PATH = PROJECT_DIR / "Data" / "dataset_ml_ready.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "gridsearch_final"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PARAMS_DIR = RESULTS_DIR / "best_params"
CV_RESULTS_DIR = RESULTS_DIR / "cv_results"
REPORTS_DIR = RESULTS_DIR / "reports"

for directory in [RESULTS_DIR, PREDICTIONS_DIR, PARAMS_DIR, CV_RESULTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# 2. FUNCIONES AUXILIARES

def create_one_hot_encoder():
    """
    Crea un OneHotEncoder compatible con versiones recientes y antiguas
    de scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def save_json(data, path):
    """
    Guarda un diccionario en formato JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def evaluate_model(model_name, best_estimator, X_test, y_test, output_prefix):
    """
    Evalúa el mejor modelo sobre el conjunto de test independiente.
    Guarda métricas, matriz de confusión, informe de clasificación y predicciones.
    """

    y_pred = best_estimator.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benign/Likely benign", "Pathogenic/Likely pathogenic"],
        digits=4,
        zero_division=0
    )

    # Guardar métricas
    metrics_path = REPORTS_DIR / f"{output_prefix}_final_metrics.json"
    save_json(metrics, metrics_path)

    # Guardar matriz de confusión
    cm_df = pd.DataFrame(
        cm,
        index=["True_Benign", "True_Pathogenic"],
        columns=["Pred_Benign", "Pred_Pathogenic"]
    )
    cm_path = REPORTS_DIR / f"{output_prefix}_confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    # Guardar classification report
    report_path = REPORTS_DIR / f"{output_prefix}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Classification report\n")
        f.write("=====================\n\n")
        f.write(report)
        f.write("\n\nConfusion matrix\n")
        f.write("================\n")
        f.write(str(cm))

    # Guardar predicciones
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    if hasattr(best_estimator, "predict_proba"):
        try:
            y_proba = best_estimator.predict_proba(X_test)[:, 1]
            predictions_df["y_proba_pathogenic"] = y_proba
        except Exception:
            pass

    pred_path = PREDICTIONS_DIR / f"{output_prefix}_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    return metrics, cm


def run_gridsearch(
    model_name,
    output_prefix,
    pipeline,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test,
    cv
):
    """
    Ejecuta GridSearchCV, guarda resultados y evalúa el mejor estimador.
    """

    print("\n" + "=" * 80)
    print(f"Optimizing model: {model_name}")
    print("=" * 80)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters for {model_name}:")
    print(grid_search.best_params_)
    print(f"Best CV F1-macro: {grid_search.best_score_:.5f}")

    # Guardar mejores parámetros
    best_params_data = {
        "model": model_name,
        "best_params": grid_search.best_params_,
        "best_cv_f1_macro": grid_search.best_score_
    }

    params_path = PARAMS_DIR / f"{output_prefix}_best_params.json"
    save_json(best_params_data, params_path)

    # Guardar resultados de CV
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_path = CV_RESULTS_DIR / f"{output_prefix}_cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)

    # Evaluación final en test
    metrics, cm = evaluate_model(
        model_name=model_name,
        best_estimator=grid_search.best_estimator_,
        X_test=X_test,
        y_test=y_test,
        output_prefix=output_prefix
    )

    metrics["best_cv_f1_macro"] = grid_search.best_score_

    print(f"\nFinal test results for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.5f}")
    print(f"F1-macro: {metrics['f1_macro']:.5f}")
    print("Confusion matrix:")
    print(cm)

    return metrics, grid_search.best_estimator_


# 3. CARGA DEL CONJUNTO DE DATOS

print("\nLoading dataset...")
print(f"Dataset path: {DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print("Columns:")
print(df.columns.tolist())

if "label" not in df.columns:
    raise ValueError("El dataset debe contener una columna llamada 'label'.")

print("\nClass distribution:")
print(df["label"].value_counts())


# 4. DEFINICIÓN DE VARIABLES PREDICTORAS Y VARIABLE OBJETIVO

# Columnas que no deben entrar como predictores
columns_to_exclude = ["label", "variant", "clnsig"]

existing_columns_to_exclude = [
    col for col in columns_to_exclude if col in df.columns
]

X = df.drop(columns=existing_columns_to_exclude)
y = df["label"].astype(int)

print("\nPredictor columns:")
print(X.columns.tolist())

print("\nExcluded columns:")
print(existing_columns_to_exclude)

# Identificación automática de columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# 5. DIVISIÓN TRAIN / TEST

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain/test split:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print("\nTrain class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())


# 6. PREPROCESAMIENTO

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", create_one_hot_encoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)


# 7. VALIDACIÓN CRUZADA

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)


# 8. DEFINICIÓN DE MODELOS Y REJILLAS GRIDSEARCH

models_and_grids = []


# 8.1 Regresión logística

logreg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

logreg_param_grid = {
    "model__C": [0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["saga"],
    "model__class_weight": [None, "balanced"]
}

models_and_grids.append(
    ("Logistic Regression", "logistic_regression", logreg_pipeline, logreg_param_grid)
)


# 8.2 Random Forest

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

rf_param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 4],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"],
    "model__class_weight": [None, "balanced"]
}

models_and_grids.append(
    ("Random Forest", "random_forest", rf_pipeline, rf_param_grid)
)


# 8.3 XGBoost

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"
        ))
    ]
)

xgb_param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    "model__reg_lambda": [1, 2]
}

models_and_grids.append(
    ("XGBoost", "xgboost", xgb_pipeline, xgb_param_grid)
)


# 8.4 SVM lineal eficiente

svm_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", SGDClassifier(
            random_state=RANDOM_STATE,
            max_iter=3000,
            tol=1e-3
        ))
    ]
)

svm_param_grid = {
    "model__loss": ["hinge", "squared_hinge"],
    "model__alpha": [1e-5, 1e-4, 1e-3],
    "model__penalty": ["l2"],
    "model__learning_rate": ["optimal"],
    "model__class_weight": [None, "balanced"]
}

models_and_grids.append(
    ("Linear SVM - SGDClassifier", "linear_svm_sgd", svm_pipeline, svm_param_grid)
)


# 9. EJECUCIÓN DE GRIDSEARCHCV

all_metrics = []
best_estimators = {}

for model_name, output_prefix, pipeline, param_grid in models_and_grids:
    metrics, best_estimator = run_gridsearch(
        model_name=model_name,
        output_prefix=output_prefix,
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv=cv
    )

    all_metrics.append(metrics)
    best_estimators[output_prefix] = best_estimator


# 10. RESUMEN FINAL COMPARATIVO

summary_df = pd.DataFrame(all_metrics)

ordered_columns = [
    "model",
    "best_cv_f1_macro",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted"
]

summary_df = summary_df[ordered_columns]
summary_df = summary_df.sort_values(by="f1_macro", ascending=False)

summary_path = RESULTS_DIR / "final_gridsearch_model_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)
print(summary_df)

print(f"\nResults saved in: {RESULTS_DIR}")
print(f"Summary saved in: {summary_path}")

print("\nGridSearch finalizado correctamente.")
