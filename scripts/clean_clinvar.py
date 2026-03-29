import pandas as pd

cols = ["chr", "pos", "ref", "alt", "clnsig", "clnvc", "mc"]
df = pd.read_csv("clinvar_snps_all.tsv", sep="\t", header=None, names=cols)

# Etiquetas claras
pathogenic_labels = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic"
}

benign_labels = {
    "Benign",
    "Likely_benign",
    "Benign/Likely_benign"
}

valid_labels = pathogenic_labels.union(benign_labels)

# Filtrar solo variantes con etiquetas claras
df = df[df["clnsig"].isin(valid_labels)].copy()

# Crear label binaria
df["label"] = df["clnsig"].apply(lambda x: 1 if x in pathogenic_labels else 0)

# Sacar consequence desde el campo MC
df["consequence"] = df["mc"].astype(str).str.split("|").str[-1]

# Ordenar columnas
df = df[["chr", "pos", "ref", "alt", "clnsig", "label", "clnvc", "consequence"]]

# Guardar
df.to_csv("clinvar_clean.csv", index=False)

print("Shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

print("\nCLNSIG counts:")
print(df["clnsig"].value_counts())

print("\nConsequence counts (top 10):")
print(df["consequence"].value_counts().head(10))
