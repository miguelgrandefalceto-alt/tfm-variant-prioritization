# TFM – Priorización de variantes patogénicas mediante Machine Learning

Este proyecto implementa un pipeline bioinformático reproducible para la construcción de un conjunto de datos de variantes genéticas y el entrenamiento de modelos de aprendizaje automático orientados a la priorización de variantes potencialmente patogénicas en el contexto de la genómica clínica y las enfermedades raras.

El flujo de trabajo integra datos clínicos públicos procedentes de ClinVar, anotación funcional mediante Ensembl Variant Effect Predictor (VEP) y modelos de clasificación supervisada para diferenciar variantes benignas/probablemente benignas y patogénicas/probablemente patogénicas.

---

## Objetivo

El objetivo principal del proyecto es desarrollar un pipeline de análisis reproducible que permita construir un conjunto de datos anotado de variantes genéticas y evaluar distintos modelos de machine learning para la clasificación binaria de variantes benignas y patogénicas.

Los objetivos específicos son:

* Construir un conjunto de datos de variantes genéticas a partir de ClinVar.
* Filtrar variantes con clasificación clínica claramente definida.
* Integrar información clínica, genómica y funcional.
* Anotar funcionalmente las variantes mediante VEP.
* Preparar un conjunto de datos final apto para modelos de machine learning.
* Entrenar, optimizar y comparar distintos clasificadores supervisados.
* Evaluar el rendimiento de los modelos sobre un conjunto de prueba independiente.
* Mantener un flujo de trabajo reproducible y documentado.

---

## Pipeline

El pipeline consta de las siguientes etapas:

### 1. Obtención de datos desde ClinVar

Se partió de datos públicos procedentes de ClinVar. El procesamiento inicial se realizó a partir de un archivo tabular derivado de ClinVar, denominado `clinvar_snps_all.tsv`, que contenía información básica de las variantes, incluyendo cromosoma, posición, alelo de referencia, alelo alternativo, clasificación clínica, tipo de variante y anotación molecular inicial.

Tareas principales:

* Procesamiento inicial de registros procedentes de ClinVar.
* Selección de variantes de un solo nucleótido.
* Extracción de campos clínicos y genómicos relevantes.
* Preparación de archivos intermedios para el filtrado, balanceo y anotación funcional.

### 2. Filtrado por clasificación clínica

Se seleccionaron únicamente variantes con clasificación clínica compatible con una definición binaria clara.

Etiquetas incluidas en la clase patogénica:

* `Pathogenic`
* `Likely_pathogenic`
* `Pathogenic/Likely_pathogenic`

Etiquetas incluidas en la clase benigna:

* `Benign`
* `Likely_benign`
* `Benign/Likely_benign`

Se excluyeron variantes de significado incierto, clasificaciones conflictivas, ambiguas o no adecuadas para una etiqueta binaria robusta.

### 3. Construcción de la variable objetivo

Se definió una clasificación binaria mediante la variable `label`:

* `1` → variante patogénica o probablemente patogénica.
* `0` → variante benigna o probablemente benigna.

La variable original `clnsig` se conservó como información de referencia durante la construcción del conjunto de datos, pero se excluyó del entrenamiento para evitar fuga de información.

### 4. Limpieza, filtrado funcional y control de duplicados

Durante la construcción del conjunto de datos se eliminaron duplicados exactos utilizando como criterio la combinación de las columnas:

* `chr`
* `pos`
* `ref`
* `alt`

También se filtraron variantes según su consecuencia molecular inicial, conservando las siguientes categorías:

* `missense_variant`
* `synonymous_variant`
* `splice_donor_variant`
* `splice_acceptor_variant`
* `nonsense`
* `stop_gained`

El conjunto final no presentó duplicados por identificador de variante ni por combinación de coordenada genómica y alelos.

Además, se verificó que todas las variantes correspondían a variantes de un solo nucleótido. Esta comprobación se realizó mediante la variable `clnvc`, en la que todas las observaciones aparecían anotadas como `single_nucleotide_variant`, y mediante la comprobación de que los alelos de referencia y alternativo presentaban longitud de una base en todos los casos.

### 5. Balanceo del conjunto de datos

Para evitar un desequilibrio entre clases, se construyó un conjunto balanceado mediante muestreo aleatorio reproducible con `random_state=42`.

Distribución final:

* 35.000 variantes benignas/probablemente benignas.
* 35.000 variantes patogénicas/probablemente patogénicas.
* Tamaño total: 70.000 variantes.

El conjunto balanceado inicial se guardó como:

```text
dataset_v1_balanced.csv
```

Posteriormente, este archivo se convirtió a formato VCF para su anotación funcional mediante VEP.

### 6. Anotación funcional con VEP

Las variantes seleccionadas se anotaron mediante Ensembl Variant Effect Predictor (VEP).

Configuración general:

* Herramienta: Ensembl Variant Effect Predictor.
* Versión: VEP v115.
* Ensamblado de referencia: GRCh38.
* Especie: `homo_sapiens`.
* Ejecución en modo offline con caché local.
* Procesamiento paralelo mediante 4 procesos.

El comando utilizado para la anotación fue:

```bash
~/ensembl-vep/vep \
  -i dataset_v1_sorted.vcf \
  -o dataset_v1_vep.txt \
  --cache \
  --offline \
  --assembly GRCh38 \
  --species homo_sapiens \
  --tab \
  --symbol \
  --sift b \
  --polyphen b \
  --fork 4 \
  --no_stats \
  --force_overwrite
```

Variables funcionales extraídas:

* Gen (`SYMBOL`).
* Consecuencia funcional.
* Impacto funcional.
* SIFT.
* PolyPhen.

La salida generada por VEP se procesó posteriormente y se combinó con la información clínica inicial. A partir de este proceso se construyó el conjunto de datos final utilizado para el entrenamiento y evaluación de los modelos de aprendizaje automático:

```text
data/dataset_ml_ready.csv
```

### 7. Preparación de variables para machine learning

Se procesaron las variables clínicas, genómicas y funcionales para construir un conjunto de datos apto para modelos de clasificación supervisada.

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

---

## Conjunto de datos final

El conjunto de datos final utilizado para el entrenamiento y evaluación de los modelos se encuentra en:

```text
Data/dataset_ml_ready.csv
```

Características principales:

* 70.000 variantes.
* 14 columnas.
* 35.000 variantes benignas o probablemente benignas.
* 35.000 variantes patogénicas o probablemente patogénicas.
* Todas las variantes corresponden a `single_nucleotide_variant`.
* Sin variantes duplicadas.
* Variable objetivo binaria: `label`.

Columnas del conjunto final:

```text
variant
chr
pos
ref
alt
clnsig
label
clnvc
consequence
gene
impact_num
sift_score
polyphen_score
Consequence_group
```

---

## Modelos de Machine Learning

Se entrenaron y compararon cuatro modelos de clasificación supervisada:

* Regresión logística.
* Random Forest.
* XGBoost.
* SVM lineal eficiente.

La regresión logística se utilizó como modelo lineal interpretable. Random Forest y XGBoost se incorporaron como modelos basados en árboles, capaces de capturar relaciones no lineales entre variables. Finalmente, se añadió una SVM lineal eficiente para ampliar la comparación con un modelo basado en margen máximo.

La SVM lineal se implementó mediante `SGDClassifier`, ya que una implementación tradicional con matriz densa resultaba computacionalmente costosa tras la codificación one-hot de las variables categóricas.

También se realizó una prueba exploratoria con SVM RBF sobre una muestra estratificada reducida. Sin embargo, el modelo principal considerado en la comparación final fue la SVM lineal eficiente, debido a su menor coste computacional y a su compatibilidad con datos de alta dimensionalidad.

---

## Optimización de hiperparámetros

La optimización final de hiperparámetros se realizó mediante `GridSearchCV` para los cuatro modelos evaluados.

La estrategia seguida fue:

* Mantener una única división estratificada entrenamiento/prueba común para todos los modelos.
* Dividir el conjunto de datos en 56.000 variantes para entrenamiento y 14.000 variantes para prueba.
* Optimizar hiperparámetros únicamente sobre el conjunto de entrenamiento.
* Utilizar validación cruzada estratificada de 3 particiones.
* Seleccionar la mejor configuración según `f1_macro`.
* Evaluar los modelos finales sobre un conjunto de prueba independiente no utilizado durante la optimización.

El uso de `Pipeline` y `ColumnTransformer` permitió encapsular el preprocesamiento y evitar fuga de información durante la validación cruzada.

El preprocesamiento incluyó:

* Imputación de valores ausentes en variables numéricas mediante la mediana.
* Escalado de variables numéricas mediante `StandardScaler`.
* Imputación de valores ausentes en variables categóricas.
* Codificación one-hot de variables categóricas mediante `OneHotEncoder(handle_unknown="ignore")`.

Los resultados de la optimización final se almacenan en:

```text
results/gridsearch_final/
```

---

## Mejores hiperparámetros seleccionados

Tras la optimización mediante `GridSearchCV`, se seleccionaron los siguientes hiperparámetros para los modelos finales:

### Regresión logística

```text
C = 10
class_weight = None
penalty = l2
solver = saga
```

### Random Forest

```text
class_weight = None
max_depth = None
max_features = sqrt
min_samples_leaf = 1
min_samples_split = 4
n_estimators = 200
```

### XGBoost

```text
colsample_bytree = 0.8
learning_rate = 0.2
max_depth = 5
n_estimators = 250
reg_lambda = 1
subsample = 0.8
```

### SVM lineal eficiente

```text
alpha = 1e-05
class_weight = balanced
learning_rate = optimal
loss = hinge
penalty = l2
```

---

## Resultados finales

Tras la optimización de hiperparámetros, los mejores modelos se evaluaron sobre el conjunto de prueba independiente.

| Modelo               | CV F1-macro | Accuracy test | Precision macro | Recall macro | F1-macro test |
| -------------------- | ----------: | ------------: | --------------: | -----------: | ------------: |
| Regresión logística  |      0.9701 |        0.9721 |          0.9722 |       0.9721 |        0.9721 |
| XGBoost              |      0.9702 |        0.9699 |          0.9699 |       0.9699 |        0.9699 |
| Random Forest        |      0.9698 |        0.9697 |          0.9698 |       0.9697 |        0.9697 |
| SVM lineal eficiente |      0.9698 |        0.9666 |          0.9675 |       0.9666 |        0.9666 |

La regresión logística obtuvo el mejor rendimiento global en el conjunto de prueba, con una accuracy y un F1-macro de aproximadamente 0,9721. No obstante, las diferencias entre regresión logística, XGBoost y Random Forest fueron reducidas, por lo que no debe interpretarse como una superioridad clara o absoluta de un único modelo.

---

## Matrices de confusión finales

Las matrices de confusión se presentan en el formato:

```text
[[Verdaderos benignos, Falsos positivos],
 [Falsos negativos, Verdaderos patogénicos]]
```

### Regresión logística

```text
[[6751, 249],
 [142, 6858]]
```

### Random Forest

```text
[[6735, 265],
 [159, 6841]]
```

### XGBoost

```text
[[6757, 243],
 [179, 6821]]
```

### SVM lineal eficiente

```text
[[6613, 387],
 [80, 6920]]
```

---

## Interpretación de resultados

Todos los modelos evaluados alcanzaron métricas elevadas, con valores de F1-macro superiores a 0,966 en el conjunto de prueba. La regresión logística obtuvo el mejor rendimiento global, aunque las diferencias con XGBoost y Random Forest fueron pequeñas.

XGBoost obtuvo el valor medio de F1-macro más alto durante la validación cruzada, pero no alcanzó el mejor rendimiento en el conjunto de prueba independiente. Esto refuerza la importancia de evaluar los modelos optimizados sobre datos no utilizados durante el ajuste de hiperparámetros.

La SVM lineal eficiente obtuvo el menor rendimiento global, pero presentó el menor número de falsos negativos para variantes patogénicas: 80, frente a 142 en regresión logística, 159 en Random Forest y 179 en XGBoost. Este comportamiento puede ser relevante en un contexto de priorización de variantes, donde puede ser preferible reducir la pérdida de variantes potencialmente patogénicas, aunque ello implique aumentar el número de falsos positivos.

En conjunto, los resultados indican que las variables genómicas, funcionales y de anotación utilizadas contienen una señal discriminativa suficiente para diferenciar variantes benignas y patogénicas en el conjunto de datos construido. Sin embargo, estos resultados deben interpretarse teniendo en cuenta las limitaciones del conjunto de datos, especialmente el uso de variantes procedentes de ClinVar, el balanceo artificial y la ausencia de validación externa independiente.

---

## Archivos de resultados

Los resultados finales de la optimización mediante `GridSearchCV` se almacenan en:

```text
results/gridsearch_final/
```

Esta carpeta incluye:

```text
final_gridsearch_model_comparison_summary.csv
best_params/
cv_results/
reports/
predictions/
```

Archivo comparativo principal:

```text
results/gridsearch_final/final_gridsearch_model_comparison_summary.csv
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
│   ├── gridsearch_final_models.py
│   ├── merge_ml_dataset.py
│   ├── process_vep.py
│   ├── train_final_models.py
│   ├── train_logreg.py
│   ├── train_rf.py
│   ├── train_svm.py
│   ├── train_svm_rbf_sample.py
│   └── train_xgboost.py
│
├── results/
│   ├── gridsearch_final/
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── svm_linear/
│   ├── svm_rbf_sample/
│   └── xgboost/
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

### Construcción y procesamiento del conjunto de datos

* `clean_clinvar.py`: limpia y filtra variantes procedentes de ClinVar.
* `build_dataset_v1.py`: construye el conjunto inicial balanceado.
* `csv_to_vcf.py`: convierte variantes seleccionadas a formato VCF.
* `process_vep.py`: procesa la salida de VEP.
* `merge_ml_dataset.py`: fusiona información clínica, genómica y funcional.
* `encode_dataset.py`: prepara versiones codificadas del conjunto de datos.

### Entrenamiento inicial y pruebas exploratorias

* `train_logreg.py`: entrenamiento inicial de regresión logística.
* `train_rf.py`: entrenamiento inicial de Random Forest.
* `train_xgboost.py`: entrenamiento inicial de XGBoost.
* `train_svm.py`: entrenamiento de SVM lineal eficiente.
* `train_svm_rbf_sample.py`: prueba exploratoria de SVM RBF sobre muestra reducida.

### Optimización y evaluación final

* `gridsearch_final_models.py`: optimización final mediante `GridSearchCV` para los cuatro modelos, evaluación sobre test independiente y guardado de resultados.
* `train_final_models.py`: entrenamiento final previo con mejores hiperparámetros y evaluación en test.

---

## Ejecución

Desde la raíz del proyecto, para ejecutar la optimización final mediante `GridSearchCV`:

```bash
python3 Scripts/gridsearch_final_models.py
```

Los resultados finales se guardan en:

```text
results/gridsearch_final/
```

Para ejecutar scripts previos de entrenamiento o procesamiento, consultar la carpeta `Scripts/`.

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
* Se fijó `random_state=42` en los procesos de muestreo y modelado.
* El preprocesamiento se integró dentro de objetos `Pipeline`.
* La transformación de variables se realizó mediante `ColumnTransformer`.
* La optimización de hiperparámetros se realizó únicamente sobre el conjunto de entrenamiento.
* La evaluación final se realizó sobre un conjunto de prueba independiente.

---

## Limitaciones

Aunque los modelos alcanzaron un rendimiento elevado, deben considerarse varias limitaciones:

* Posible sesgo inherente a ClinVar.
* Conjunto de datos balanceado artificialmente.
* Ausencia de validación externa independiente.
* Posible sesgo por genes y regiones clínicas sobrerrepresentadas.
* Diferencias entre variantes recogidas en bases de datos clínicas y variantes observadas en cohortes reales.
* La buena capacidad predictiva en este conjunto de datos no garantiza necesariamente el mismo rendimiento sobre variantes nuevas, genes menos estudiados o cohortes clínicas independientes.
* Las predicciones generadas por los modelos deben interpretarse como apoyo a la priorización computacional y no como una clasificación clínica definitiva.

---

## Autor

Miguel Grande Falceto
Máster Universitario en Bioinformática – UNIR
