#  TFM – Priorización de variantes patogénicas mediante Machine Learning

Este proyecto implementa un pipeline bioinformático completo para la construcción de un dataset orientado al entrenamiento de modelos de aprendizaje automático para la priorización de variantes genéticas en el contexto de enfermedades raras.

El flujo de trabajo integra datos clínicos públicos (ClinVar) con anotación funcional (Ensembl VEP) y procesos de ingeniería de características.

---

##  Descripción del proyecto

El objetivo de este trabajo es:

* Construir un dataset de alta calidad de variantes genéticas
* Integrar información clínica y funcional
* Preparar variables para modelos de Machine Learning
* Desarrollar un pipeline reproducible de bioinformática

---

##  Pipeline

El pipeline consta de las siguientes etapas:

1. **Extracción de datos de ClinVar**

   * Descarga y procesamiento de archivos VCF
   * Selección de variantes tipo SNP

2. **Filtrado por significancia clínica**

   * Inclusión:

     * Pathogenic / Likely pathogenic
     * Benign / Likely benign
   * Exclusión de variantes con significado incierto

3. **Construcción de la variable objetivo**

   * Clasificación binaria:

     * `1` → patogénicas
     * `0` → benignas

4. **Balanceo del dataset**

   * 35.000 variantes por clase
   * Tamaño final: **70.000 variantes**

5. **Generación de archivo VCF**

   * Conversión del dataset a formato compatible con herramientas bioinformáticas

6. **Anotación funcional (VEP)**

   * Herramienta: Ensembl Variant Effect Predictor (VEP v115)
   * Ensamblado: GRCh38
   * Modo offline con caché local

7. **Procesamiento de anotaciones**

   * Extracción de variables relevantes:

     * Gen (SYMBOL)
     * Consecuencia
     * Impacto
     * SIFT
     * PolyPhen
   * Colapso de múltiples transcritos por variante
   * Selección de la anotación más relevante

8. **Ingeniería de características**

   * Codificación del impacto funcional
   * Agrupación de consecuencias:

     * Synonymous
     * Missense
     * LoF (loss of function)
     * Otros

9. **Integración final**

   * Fusión de datos clínicos y funcionales
   * Dataset final: `dataset_ml_ready.csv`

---

##  Dataset final

* **Tamaño**: 70.000 variantes
* **Clases balanceadas**: 35.000 benignas / 35.000 patogénicas
* **Variables**:

  * Coordenadas genómicas
  * Significancia clínica (ClinVar)
  * Anotaciones funcionales (VEP)
  * Variables derivadas

---

##  Estructura del repositorio

```text
├── scripts/
│   ├── clean_clinvar.py
│   ├── build_dataset_v1.py
│   ├── csv_to_vcf.py
│   ├── process_vep.py
│   ├── merge_ml_dataset.py
│   ├── train_baseline_models.py
├── data/
│   └── dataset_sample.csv
├── results/
│   └── baseline_model_results.csv
├── README.md
├── requirements.txt
```

---

##  Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Contenido:

```
pandas
numpy
scikit-learn
```

---

##  Ejecución del pipeline

```bash
python clean_clinvar.py
python build_dataset_v1.py
python csv_to_vcf.py
python process_vep.py
python merge_ml_dataset.py
python train_baseline_models.py
```

---

##  Notas

* El dataset puede reproducirse ejecutando los scripts

---

##  Reproducibilidad

Desarrollado en entorno Linux (WSL) con:

* Python 3
* VEP v115 (modo offline)
* GRCh38

---

##  Trabajo futuro

* Incorporación de nuevas variables (CADD, REVEL, gnomAD)
* Optimización de modelos
* Interpretabilidad (SHAP)

---

##  Autor

Miguel Grande Falceto
Máster en Bioinformática (UNIR)
