import pandas as pd

# Leer datasets
base = pd.read_csv("dataset_v1_balanced.csv")
vep = pd.read_csv("dataset_ml_features.csv")

# Crear clave variant con el mismo formato que VEP
base["variant"] = (
    base["chr"].astype(str) + "_" +
    base["pos"].astype(str) + "_" +
    base["ref"].astype(str) + "/" +
    base["alt"].astype(str)
)

# Asegurar que la label sea entera
base["label"] = base["label"].astype(int)

# Seleccionar columnas útiles del dataset base
base_small = base[[
    "variant", "chr", "pos", "ref", "alt",
    "clnsig", "label", "clnvc", "consequence"
]].copy()

# Merge
merged = base_small.merge(vep, on="variant", how="inner")

# Reordenar columnas
merged = merged[[
    "variant", "chr", "pos", "ref", "alt",
    "clnsig", "label", "clnvc", "consequence",
    "gene", "impact_num", "sift_score", "polyphen_score", "Consequence_group"
]]

# Guardar
merged.to_csv("dataset_ml_ready.csv", index=False)

# Resumen
print("Base shape:", base.shape)
print("VEP shape:", vep.shape)
print("Merged shape:", merged.shape)

print("\nPrimeras filas:")
print(merged.head())

print("\nValores perdidos:")
print(merged.isna().sum())

print("\nDistribución label:")
print(merged["label"].value_counts())

print("\nDistribución Consequence_group:")
print(merged["Consequence_group"].value_counts())
