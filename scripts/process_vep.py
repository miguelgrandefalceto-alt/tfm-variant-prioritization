"""
process_vep.py

Procesa el output de VEP, extrae variables funcionales,
convierte anotaciones en features numéricas y colapsa
múltiples transcritos en una única fila por variante.
"""


import pandas as pd
import numpy as np
import re

# Archivo de entrada
vep_file = "dataset_v1_vep.txt"

# Leer cabecera real de VEP
header = None
with open(vep_file, "r") as f:
    for line in f:
        if line.startswith("#Uploaded_variation"):
            header = line.strip().lstrip("#").split("\t")
            break

if header is None:
    raise ValueError("No se encontró la cabecera de columnas de VEP en el archivo.")

# Leer archivo ignorando metadatos ##
df = pd.read_csv(
    vep_file,
    comment="#",
    sep="\t",
    names=header,
    dtype=str,
    low_memory=False
)

# Quedarnos con columnas clave
cols_needed = ["Uploaded_variation", "SYMBOL", "Consequence", "IMPACT", "SIFT", "PolyPhen"]
missing = [c for c in cols_needed if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas esperadas: {missing}")

df = df[cols_needed].copy()

# Reemplazar "-" por NaN
df = df.replace("-", np.nan)

# Función para extraer score numérico de SIFT y PolyPhen
def extract_score(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\(([\d\.]+)\)", str(value))
    if match:
        return float(match.group(1))
    return np.nan

df["SIFT_score"] = df["SIFT"].apply(extract_score)
df["PolyPhen_score"] = df["PolyPhen"].apply(extract_score)

# Codificar impacto
impact_map = {
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
    "MODIFIER": 0
}
df["IMPACT_num"] = df["IMPACT"].map(impact_map)

# Agrupar consecuencias
def consequence_group(cons):
    if pd.isna(cons):
        return "Other"
    cons = str(cons)

    lof_terms = [
        "stop_gained", "frameshift_variant",
        "splice_donor_variant", "splice_acceptor_variant",
        "start_lost", "stop_lost"
    ]
    if any(term in cons for term in lof_terms):
        return "LoF"
    elif "missense_variant" in cons:
        return "Missense"
    elif "synonymous_variant" in cons:
        return "Synonymous"
    elif any(term in cons for term in [
        "intron_variant", "upstream_gene_variant",
        "downstream_gene_variant", "non_coding_transcript_variant",
        "non_coding_transcript_exon_variant", "UTR"
    ]):
        return "Non_coding"
    else:
        return "Other"

df["Consequence_group"] = df["Consequence"].apply(consequence_group)

# Prioridad para colapsar la consecuencia
cons_priority = {
    "LoF": 4,
    "Missense": 3,
    "Synonymous": 2,
    "Non_coding": 1,
    "Other": 0
}
df["Consequence_priority"] = df["Consequence_group"].map(cons_priority)

# Rellenar símbolo vacío por NaN real
df["SYMBOL"] = df["SYMBOL"].replace("", np.nan)

# Función para coger el primer gen no vacío
def first_non_null(series):
    series = series.dropna()
    return series.iloc[0] if len(series) > 0 else np.nan

# Colapsar por variante
collapsed = df.groupby("Uploaded_variation", as_index=False).agg({
    "SYMBOL": first_non_null,
    "IMPACT_num": "max",
    "SIFT_score": "min",
    "PolyPhen_score": "max",
    "Consequence_priority": "max"
})

# Recuperar nombre del grupo de consecuencia desde prioridad
priority_to_cons = {v: k for k, v in cons_priority.items()}
collapsed["Consequence_group"] = collapsed["Consequence_priority"].map(priority_to_cons)

# Renombrar columnas finales
collapsed = collapsed.rename(columns={
    "Uploaded_variation": "variant",
    "SYMBOL": "gene",
    "IMPACT_num": "impact_num",
    "SIFT_score": "sift_score",
    "PolyPhen_score": "polyphen_score"
})

collapsed = collapsed[[
    "variant", "gene", "impact_num",
    "sift_score", "polyphen_score", "Consequence_group"
]]

# Guardar dataset final
collapsed.to_csv("dataset_ml_features.csv", index=False)

# Resumen
print(collapsed.head())
print("\nShape final:", collapsed.shape)
print("\nValores perdidos:")
print(collapsed.isna().sum())
print("\nDistribución de consecuencias:")
print(collapsed["Consequence_group"].value_counts())
