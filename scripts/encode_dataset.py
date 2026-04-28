import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv("/mnt/c/Users/Usuario/Desktop/UNIR/TFM/Dataset/dataset_ml_ready.csv")

# Target
y = df["label"]

# Features
X = df[
    [
        "chr",
        "pos",
        "ref",
        "alt",
        "clnvc",
        "consequence",
        "impact_num",
        "sift_score",
        "polyphen_score",
        "Consequence_group"
    ]
].copy()

# Columnas numéricas y categóricas
num_cols = ["pos", "impact_num", "sift_score", "polyphen_score"]
cat_cols = ["chr", "ref", "alt", "clnvc", "consequence", "Consequence_group"]

# Missing values numéricos
for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

# Missing values categóricos
for col in cat_cols:
    X[col] = X[col].astype(str).fillna("unknown")

# One-hot encoding
X_encoded = pd.get_dummies(X, columns=cat_cols)

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Guardar con rutas absolutas
X_train.to_csv("/mnt/c/Users/Usuario/Desktop/UNIR/TFM/Dataset/X_train.csv", index=False)
X_test.to_csv("/mnt/c/Users/Usuario/Desktop/UNIR/TFM/Dataset/X_test.csv", index=False)
y_train.to_csv("/mnt/c/Users/Usuario/Desktop/UNIR/TFM/Dataset/y_train.csv", index=False)
y_test.to_csv("/mnt/c/Users/Usuario/Desktop/UNIR/TFM/Dataset/y_test.csv", index=False)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Codificación completada")
