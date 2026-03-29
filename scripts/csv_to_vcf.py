import pandas as pd

df = pd.read_csv("dataset_v1_balanced.csv")

with open("dataset_v1.vcf", "w") as f:
    # Cabecera mínima VCF
    f.write("##fileformat=VCFv4.2\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

    for _, row in df.iterrows():
        f.write(f"{row['chr']}\t{row['pos']}\t.\t{row['ref']}\t{row['alt']}\t.\t.\t.\n")
