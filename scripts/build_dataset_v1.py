import pandas as pd

df = pd.read_csv("clinvar_clean.csv", low_memory=False)

# Eliminar duplicados exactos por variante
df = df.drop_duplicates(subset=["chr", "pos", "ref", "alt"]).copy()

# Consecuencias que sí queremos en la v1
allowed_consequences = {
    "missense_variant",
    "synonymous_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "nonsense",
    "stop_gained"
}

df = df[df["consequence"].isin(allowed_consequences)].copy()

print("Después de filtrar consecuencias:")
print(df.shape)
print(df["label"].value_counts())
print(df["consequence"].value_counts().head(20))

# Separar clases
df_benign = df[df["label"] == 0].copy()
df_path = df[df["label"] == 1].copy()

# Tamaño objetivo balanceado
n = min(len(df_benign), len(df_path), 35000)

df_benign_sample = df_benign.sample(n=n, random_state=42)
df_path_sample = df_path.sample(n=n, random_state=42)

df_final = pd.concat([df_benign_sample, df_path_sample], axis=0)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset final balanceado:")
print(df_final.shape)
print(df_final["label"].value_counts())
print(df_final["consequence"].value_counts())

df_final.to_csv("dataset_v1_balanced.csv", index=False)
# Limitar synonymous para evitar sesgo
syn_df = df[df["consequence"] == "synonymous_variant"]
other_df = df[df["consequence"] != "synonymous_variant"]

syn_df = syn_df.sample(n=min(len(syn_df), 10000), random_state=42)

df = pd.concat([syn_df, other_df])
