# Practica 1 - Modelización de datos (Predicción de impago)

## Descripción

En esta práctica se ha desarrollado un pipeline completo de Machine Learning para predecir si un cliente va a devolver su préstamo o no (detección de impago).

El objetivo es construir un pipeline alternativo al trabajado en clase, introduciendo decisiones propias en:
- preprocesamiento,
- generación de variables,
- filtrado,
- modelado,
- evaluación.

---

## Estructura del proyecto

- `src/preprocessing/practica1_preprocessing.py`  
  Clase de preprocesamiento personalizada siguiendo el patrón `fit/transform`.

- `src/filtering/practica1_filtering.py`  
  Clase de filtrado de variables.

- `practica1_notebook.ipynb`  
  Notebook principal con todo el pipeline:
  - preprocesamiento
  - filtrado
  - entrenamiento de modelos
  - evaluación y comparación

- `data/`  
  Contiene los datasets de entrenamiento y test.

---

## Pipeline implementado

### 1. Preprocesamiento

Se ha implementado una clase `Practica1Preprocess` que realiza:

- Selección de variables usando `variables_withExperts.xlsx`
- Eliminación de variables con **data leakage**
- Imputación:
  - numéricas → valor constante (-1)
  - categóricas → `"missing"`
- Codificación:
  - `OrdinalEncoder` para variables ordinales (`grade`, `sub_grade`)
  - `CountFrequencyEncoder` para el resto
- Escalado con `RobustScaler`
- Creación de nuevas variables:
  - `fico_mean`
  - ratios financieros
  - antigüedad del historial crediticio
  - binning de ingresos

---

### 2. Filtrado

Se ha implementado `Practica1Filtering` con:

- limpieza de valores problemáticos (`NaN`, `inf`)
- imputación residual
- `VarianceThreshold`
- `SelectKBest` con `mutual_info_classif`

---

### 3. Modelos

Se han entrenado tres modelos:

- **Random Forest** (ensemble de árboles)
- **SVM (RBF)**
- **MLP (red neuronal)**

---

### 4. Evaluación

Se han utilizado las métricas:

- Accuracy
- Precision (clase impago)
- Recall (clase impago)
- PR-AUC

Se comparan con el modelo base (FICO score).

---

## Resultados principales

- El modelo **MLP** obtiene el mejor rendimiento global.
- El **Random Forest** destaca en recall (detección de impagos).
- La **SVM** no resulta adecuada en este problema.

---

## Data Leakage

Durante el desarrollo se detectó la presencia de **data leakage**, debido a variables que contenían información posterior a la concesión del préstamo (pagos, recoveries, etc.).

Estas variables fueron eliminadas para asegurar que el modelo solo utiliza información disponible en el momento de predicción.

---

## Conclusión

El trabajo demuestra la importancia de:
- un buen preprocesamiento,
- la selección adecuada de variables,
- y la detección de fugas de información.

Pequeñas decisiones en el pipeline pueden tener un gran impacto en el rendimiento y validez del modelo.

---

## Ejecución

El notebook `practica1_notebook.ipynb` contiene todo el pipeline ejecutado.

Para reproducir los resultados:
1. instalar dependencias (`scikit-learn`, `feature-engine`, `pandas`, `numpy`)
2. ejecutar el notebook completo