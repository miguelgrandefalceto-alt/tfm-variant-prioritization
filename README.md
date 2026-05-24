# TFM – Priorización de variantes patogénicas mediante Machine Learning

Este proyecto implementa un pipeline bioinformático reproducible para la construcción de un dataset de variantes genéticas y el entrenamiento de modelos de aprendizaje automático orientados a la priorización de variantes potencialmente patogénicas en el contexto de la genómica clínica y las enfermedades raras.

El flujo de trabajo integra datos clínicos públicos procedentes de ClinVar, anotación funcional mediante Ensembl Variant Effect Predictor (VEP) y modelos de clasificación supervisada para diferenciar variantes benignas y patogénicas.

---

## Objetivo

El objetivo principal del proyecto es desarrollar un pipeline de análisis reproducible que permita construir un dataset anotado de variantes genéticas y evaluar distintos modelos de machine learning para la clasificación binaria de variantes benignas y patogénicas.

Los objetivos específicos son:

* Construir un dataset de variantes genéticas a partir de ClinVar.
* Filtrar variantes con significancia clínica claramente definida.
* Integrar información clínica, genómica y funcional.
* Anotar funcionalmente las variantes mediante VEP.
* Preparar un dataset final apto para modelos de machine learning.
* Entrenar, optimizar y comparar distintos clasificadores.
* Evaluar el rendimiento de los modelos sobre un conjunto de prueba independiente.
* Mantener un flujo de trabajo reproducible y documentado.

---

## Pipeline

El pipeline consta de las siguientes etapas:

### 1. Extracción de datos desde ClinVar

Se parte de datos públicos de ClinVar en formato VCF. A partir de estos archivos se extraen variantes genéticas junto con información clínica relevante.

Tareas principales:

* Procesamiento inicial de archivos VCF.
* Selección de variantes tipo SNP.
* Extracción de campos clínicos y genómicos relevantes.

### 2. Filtrado por significancia clínica

Se seleccionaron únicamente variantes con clasificación clínica clara:

* Pathogenic / Likely pathogenic.
* Benign / Likely benign.

Se excluyeron variantes con significado incierto, como las clasificadas como VUS, para reducir la ambigüedad de la variable objetivo durante el entrenamiento supervisado.

### 3. Construcción de la variable objetivo

Se definió una clasificación binaria:

* `1` → variante patogénica o probablemente patogénica.
* `0` → variante benigna o probablemente benigna.

### 4. Balanceo del dataset

Para evitar un desequilibrio entre clases, se construyó un dataset balanceado:

* 35.000 variantes clase 1.
* 35.000 variantes clase 0.
* Tamaño total: 70.000 variantes.

Además, se comprobó que no existían variantes duplicadas en el dataset final.

### 5. Anotación funcional con VEP

Las variantes seleccionadas se anotaron mediante Ensembl Variant Effect Predictor.

Configuración general:

* Herramienta: Ensembl Variant Effect Predictor.
* Versión: VEP v115.
* Ensamblado: GRCh38.
* Ejecución en modo offline con caché local.

Variables funcionales extraídas:

* Gen (`SYMBOL`).
* Consecuencia funcional.
* Impacto.
* SIFT.
* PolyPhen.

### 6. Ingeniería de características

Se procesaron las variables clínicas y funcionales para construir un dataset apto para machine learning.

Variables predictoras utilizadas:

* Variables numéricas:
  * `pos`
  * `impact_num`
  * `sift_score`
  * `polyphen_score`

* Variables categóricas:
  * `chr`
  * `ref`
  * `alt`
  * `clnvc`
  * `consequence`
  * `gene`
  * `Consequence_group`

Variables excluidas del entrenamiento:

* `label`
* `variant`
* `clnsig`

La variable `label` se utilizó como variable objetivo, mientras que `variant` y `clnsig` se excluyeron para evitar fuga de información durante el entrenamiento.

### 7. Dataset final

El dataset final utilizado para el entrenamiento se encuentra en:

```text
Data/dataset_ml_ready.csv
```

Características principales:

* 70.000 variantes.
* 14 columnas.
* 35.000 variantes benignas o probablemente benignas.
* 35.000 variantes patogénicas o probablemente patogénicas.
* Sin variantes duplicadas.

---

## Modelos de Machine Learning

Se entrenaron y compararon cuatro modelos de clasificación supervisada:

* Regresión logística.
* Random Forest.
* XGBoost.
* SVM lineal eficiente.

La regresión logística se utilizó como modelo base interpretable. Random Forest y XGBoost se incorporaron como modelos basados en árboles, capaces de capturar relaciones no lineales entre variables. Finalmente, se añadió una SVM lineal eficiente para ampliar la comparación con un modelo de margen máximo.

La SVM lineal se implementó mediante `SGDClassifier(loss="hinge")`, ya que una implementación tradicional con matriz densa resultaba computacionalmente costosa tras la codificación one-hot de las variables categóricas.

También se realizó una prueba exploratoria con SVM RBF sobre una muestra estratificada reducida. Sin embargo, el modelo principal considerado en la comparación final fue la SVM lineal eficiente, debido a su menor coste computacional y a su rendimiento competitivo.

---

## Optimización de hiperparámetros

Además del entrenamiento inicial de los modelos, se realizó una optimización moderada de hiperparámetros.

La estrategia seguida fue:

* Mantener una única división estratificada entrenamiento/prueba común para todos los modelos.
* Optimizar hiperparámetros únicamente sobre el conjunto de entrenamiento.
* Utilizar validación cruzada con 3 folds.
* Seleccionar la mejor configuración según `f1_macro`.
* Evaluar los modelos finales una única vez sobre el conjunto de prueba.

Métodos utilizados:

* `GridSearchCV` para:
  * Regresión logística.
  * SVM lineal.

* `RandomizedSearchCV` para:
  * Random Forest.
  * XGBoost.

El uso de `Pipeline` y `ColumnTransformer` permitió encapsular el preprocesamiento y evitar fuga de información durante la validación cruzada.

El preprocesamiento incluyó:

* Imputación de valores ausentes en variables numéricas mediante la mediana.
* Escalado de variables numéricas mediante `StandardScaler`.
* Imputación de valores ausentes en variables categóricas con la categoría `"missing"`.
* Codificación one-hot de variables categóricas mediante `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`.

---

## Mejores hiperparámetros seleccionados

Tras la optimización, se seleccionaron los siguientes hiperparámetros para los modelos finales:

### Regresión logística

```text
C = 10
penalty = l2
solver = saga
```

### Random Forest

```text
n_estimators = 334
max_depth = None
max_features = sqrt
min_samples_leaf = 1
min_samples_split = 4
class_weight = balanced
```

### XGBoost

```text
n_estimators = 221
max_depth = 5
learning_rate = 0.17254716573280354
subsample = 0.7468055921327309
colsample_bytree = 0.8123620356542087
reg_lambda = 1.5751320499779735
```

### SVM lineal eficiente

```text
loss = hinge
alpha = 0.0001
penalty = l2
learning_rate = optimal
class_weight = balanced
```

---

## Resultados finales

Tras la optimización de hiperparámetros, los modelos finales se entrenaron sobre el conjunto completo de entrenamiento y se evaluaron sobre el conjunto de prueba.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| Regresión logística | 0.9721 | 0.9722 | 0.9721 | 0.9721 |
| Random Forest | 0.9701 | 0.9702 | 0.9701 | 0.9701 |
| XGBoost | 0.9701 | 0.9702 | 0.9701 | 0.9701 |
| SVM lineal | 0.9701 | 0.9705 | 0.9701 | 0.9701 |

La regresión logística obtuvo el mejor rendimiento global en el conjunto de prueba, aunque las diferencias entre modelos fueron reducidas. Esto sugiere que las variables funcionales y clínicas seleccionadas contienen una señal discriminativa suficiente para separar eficazmente variantes benignas y patogénicas.

---

## Matrices de confusión finales

### Regresión logística

```text
[[6751, 249],
 [142, 6858]]
```

### Random Forest

```text
[[6740, 260],
 [158, 6842]]
```

### XGBoost

```text
[[6752, 248],
 [170, 6830]]
```

### SVM lineal eficiente

```text
[[6682, 318],
 [101, 6899]]
```

---

## Interpretación de resultados

Todos los modelos evaluados alcanzaron métricas elevadas y muy similares. La regresión logística obtuvo el mejor rendimiento global, con una accuracy y un F1-score macro de 0,9721.

Este resultado indica que las variables utilizadas, especialmente aquellas relacionadas con la consecuencia funcional, el impacto de la variante y los predictores in silico, permiten una separación robusta entre variantes benignas y patogénicas.

La SVM lineal mostró el menor número de falsos negativos para la clase patogénica. Este comportamiento puede ser relevante en un contexto de priorización de variantes, donde puede ser preferible minimizar la pérdida de variantes potencialmente patogénicas, aunque aumente el número de falsos positivos.

Los resultados completos se almacenan en:

```text
results/final_model_comparison_summary.csv
results/final_model_comparison_summary.txt
results/final_model_comparison_summary_excel.csv
```

Las predicciones finales se almacenan en:

```text
predictions/
```
---

## Estructura del repositorio

```text
TFM/
├── Data/
│   └── dataset_ml_ready.csv
│
├── Scripts/
│   ├── build_dataset_v1.py
│   ├── clean_clinvar.py
│   ├── csv_to_vcf.py
│   ├── encode_dataset.py
│   ├── merge_ml_dataset.py
│   ├── process_vep.py
│   ├── train_logreg.py
│   ├── train_rf.py
│   ├── train_xgboost.py
│   ├── train_svm.py
│   ├── train_svm_rbf_sample.py
│   ├── optimize_models.py
│   └── train_final_models.py
│
├── results/
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── svm_linear/
│   ├── svm_rbf_sample/
│   ├── xgboost/
│   ├── final_model_comparison_summary.csv
│   ├── final_model_comparison_summary_excel.csv
│   └── final_model_comparison_summary.txt
│
├── predictions/
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── svm_linear/
│   └── xgboost/
│
├── README.md
└── requirements.txt
```
---

## Scripts principales

### Construcción y procesamiento del dataset

* `build_dataset_v1.py`: construye el dataset inicial.
* `clean_clinvar.py`: limpia y filtra variantes procedentes de ClinVar.
* `csv_to_vcf.py`: convierte variantes seleccionadas a formato VCF.
* `process_vep.py`: procesa la salida de VEP.
* `merge_ml_dataset.py`: fusiona información clínica, genómica y funcional.
* `encode_dataset.py`: prepara el dataset final para machine learning.

### Entrenamiento inicial de modelos

* `train_logreg.py`: entrenamiento inicial de regresión logística.
* `train_rf.py`: entrenamiento inicial de Random Forest.
* `train_xgboost.py`: entrenamiento inicial de XGBoost.
* `train_svm.py`: entrenamiento de SVM lineal eficiente.
* `train_svm_rbf_sample.py`: prueba exploratoria de SVM RBF sobre muestra reducida.

### Optimización y evaluación final

* `optimize_models.py`: optimización moderada de hiperparámetros mediante validación cruzada.
* `train_final_models.py`: entrenamiento final con los mejores hiperparámetros y evaluación en test.

---

## Ejecución

Desde la raíz del proyecto, primero se puede ejecutar la optimización moderada de hiperparámetros:

```bash
python3 Scripts/optimize_models.py
```

Después, para entrenar los modelos finales con los mejores hiperparámetros seleccionados:

```bash
python3 Scripts/train_final_models.py
```

Los resultados finales se guardan en:

```text
results/
```

Las predicciones finales se guardan en:

```text
predictions/
```

---

## Requisitos

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
```

---

## Reproducibilidad

El proyecto fue desarrollado en entorno Linux mediante WSL/Ubuntu.

Configuración principal:

* Python 3.
* scikit-learn.
* XGBoost.
* Ensembl VEP v115.
* Ensamblado GRCh38.
* VEP ejecutado en modo offline con caché local.

Para garantizar la reproducibilidad:

* Se utilizó una división estratificada común de entrenamiento y prueba.
* Se fijó `random_state=42`.
* El preprocesamiento se integró dentro de objetos `Pipeline`.
* La optimización de hiperparámetros se realizó únicamente sobre el conjunto de entrenamiento.
* La evaluación final se realizó sobre un conjunto de prueba independiente.

---

## Limitaciones

Aunque los modelos alcanzaron un rendimiento elevado, deben considerarse varias limitaciones:

* Posible sesgo inherente a ClinVar.
* Dataset balanceado artificialmente.
* Ausencia de validación externa independiente.
* Posible sesgo por genes y regiones clínicas sobrerrepresentadas.
* Diferencias entre variantes recogidas en bases de datos clínicas y variantes observadas en cohortes reales.
* La buena capacidad predictiva en este dataset no garantiza necesariamente el mismo rendimiento sobre variantes nuevas o genes menos estudiados.

---

## Autor

Miguel Grande Falceto  
Máster Universitario en Bioinformática – UNIR
