# TFM - Variant Prioritization using Machine Learning

This project implements a bioinformatics pipeline for the prioritization of pathogenic variants using machine learning techniques.

## Pipeline

1. ClinVar data processing
2. Variant filtering and balancing
3. VCF generation
4. Functional annotation with VEP
5. Feature engineering
6. Dataset construction
7. Model training

## Scripts

- clean_clinvar.py
- build_dataset_v1.py
- csv_to_vcf.py
- process_vep.py
- merge_ml_dataset.py
- train_baseline_models.py

## Requirements

- Python 3
- pandas
- scikit-learn
- VEP (Ensembl)

## Notes

The dataset is not included due to size limitations but can be reproduced using the provided scripts.
