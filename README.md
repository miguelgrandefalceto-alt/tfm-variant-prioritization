TFM – Priorización de variantes patogénicas mediante Machine Learning

Este proyecto implementa un pipeline bioinformático completo para la construcción de un dataset y el entrenamiento de modelos de aprendizaje automático orientados a la priorización de variantes genéticas en el contexto de enfermedades raras.

El flujo de trabajo integra datos clínicos públicos (ClinVar), anotación funcional mediante Ensembl Variant Effect Predictor (VEP) y técnicas de machine learning.

Objetivo
Construir un dataset de alta calidad de variantes genéticas
Integrar información clínica y funcional
Preparar variables para modelos de aprendizaje automático
Entrenar y evaluar modelos de clasificación
Desarrollar un pipeline reproducible de bioinformática
Pipeline

El pipeline consta de las siguientes etapas:

1. Extracción de datos (ClinVar)
Procesamiento de archivos VCF
Selección de variantes tipo SNP
2. Filtrado por significancia clínica
Inclusión:
Pathogenic / Likely pathogenic
Benign / Likely benign
Exclusión de variantes con significado incierto
3. Construcción de la variable objetivo
Clasificación binaria:
1 → patogénicas
0 → benignas
4. Balanceo del dataset
35.000 variantes por clase
Tamaño total: 70.000 variantes
5. Anotación funcional
Herramienta: Ensembl Variant Effect Predictor (VEP)
Ensamblado: GRCh38
Modo offline con caché local

Se extraen variables como:

Gen (SYMBOL)
Consecuencia
Impacto
SIFT
PolyPhen
6. Ingeniería de características
Codificación del impacto funcional
Agrupación de consecuencias:
Synonymous
Missense
Loss of Function (LoF)
Otros
7. Integración final
Fusión de datos clínicos y funcionales
Dataset final: dataset_ml_ready.csv
Modelos de Machine Learning

Se entrenaron los siguientes modelos:

Regresión logística (baseline)
Random Forest
XGBoost
Resultados
Modelo	Accuracy	F1-score	ROC-AUC
Random Forest	~0.965	~0.96	~0.99
XGBoost	0.969	0.970	0.994
Ejemplo de resultados
### Matriz de confusión (XGBoost)

![Confusion Matrix](results/results/xgboost_confusion_matrix.png)

### Curva ROC

![ROC Curve](results/results/xgboost_roc_curve.png)

Interpretación

El modelo XGBoost muestra un alto rendimiento en ambas clases, con un equilibrio adecuado entre precisión y sensibilidad. El número de falsos negativos es reducido, lo cual es relevante en un contexto de priorización de variantes, donde es preferible minimizar la pérdida de variantes potencialmente patogénicas.

Las variables relacionadas con la consecuencia funcional de la variante presentan una alta importancia en la predicción.

Estructura del repositorio
scripts/
data/
results/
Limitaciones
Posible sesgo en las anotaciones de ClinVar
Dataset balanceado artificialmente
Ausencia de validación externa independiente
Requisitos

Instalar dependencias:

pip install -r requirements.txt

Contenido de requirements.txt:

pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
Ejecución
python scripts/modeling/train_xgboost.py
Reproducibilidad

Desarrollado en entorno Linux (WSL) con:

Python 3
VEP v115 (modo offline)
GRCh38
Autor

Miguel Grande Falceto
Máster en Bioinformática (UNIR)
