import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar datos
X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

# Modelo
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Entrenar
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nTop 10 features:")
for i in range(10):
    print(f"{X_train.columns[indices[i]]}: {importances[indices[i]]}")

# Guardar resultados
with open("../results/results_rf.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification report:\n")
    f.write(classification_report(y_test, y_pred))

# Guardar predicciones
results_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
results_df.to_csv("../results/predictions_rf.csv", index=False)
