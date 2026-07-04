import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
with open("results_logreg.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write("\nConfusion matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\n\nClassification report:\n")
    f.write(classification_report(y_test, y_pred))

results_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})

results_df.to_csv("predictions_logreg.csv", index=False)
