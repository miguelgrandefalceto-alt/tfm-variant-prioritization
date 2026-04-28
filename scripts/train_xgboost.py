import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    roc_curve
)

from xgboost import XGBClassifier


# =========================
# 1. Cargar dataset
# =========================

df = pd.read_csv("../data/dataset_ml_ready.csv")

print("Shape del dataset:", df.shape)
print(df.head())


# =========================
# 2. Separar X e y
# =========================

y = df["label"]

cols_to_drop = [
    "label",
    "variant",
    "chr",
    "pos",
    "ref",
    "alt",
    "clnsig"
]

X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])


# =========================
# 3. Codificar categóricas
# =========================

X = pd.get_dummies(X, drop_first=True)

print("Número de features finales:", X.shape[1])


# =========================
# 4. Train/test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 5. Modelo XGBoost
# =========================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)


# =========================
# 6. Predicciones
# =========================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# =========================
# 7. Métricas
# =========================

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(y_test, y_pred))


# =========================
# 8. Guardar métricas
# =========================

metrics_df = pd.DataFrame([{
    "model": "XGBoost",
    "accuracy": accuracy,
    "f1_score": f1,
    "roc_auc": roc_auc
}])

metrics_df.to_csv("../results/results/xgboost_metrics.csv", index=False)


# =========================
# 9. Matriz de confusión
# =========================

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - XGBoost")

plt.savefig("../results/results/xgboost_confusion_matrix.png")
plt.close()


# =========================
# 10. Curva ROC
# =========================

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()

plt.savefig("../results/results/xgboost_roc_curve.png")
plt.close()


# =========================
# 11. Importancia de variables
# =========================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance.to_csv("../results/results/xgboost_feature_importance.csv", index=False)

print("\nTop 10 features:")
print(feature_importance.head(10))


# =========================
# 12. Guardar predicciones
# =========================

results = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba_pathogenic": y_proba
})

results.to_csv("../results/predictions/xgboost_predictions.csv", index=False)


# =========================
# 13. Summary TXT
# =========================

with open("../results/results/xgboost_summary.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1-score: {f1}\n")
    f.write(f"ROC-AUC: {roc_auc}\n\n")
    f.write("Confusion matrix:\n")
    f.write(str(cm))


print("\nArchivos guardados correctamente en /results/")
