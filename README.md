# Aprendizaje Supervisado y No Supervisado con Python

Este repositorio contiene notebooks que implementan diferentes modelos de aprendizaje automático (Machine Learning), tanto supervisado como no supervisado. El código está basado en el tutorial: ["Machine Learning for Everybody – Full Course"](https://youtu.be/i_LwzRVP7bg?si=7lWl84-NSREh2aiJ).

## Contenido del Repositorio

El repositorio se divide en tres secciones principales, cada una correspondiente a un notebook específico:

1.  **Regresión (Aprendizaje Supervisado)**
2.  **Clasificación (Aprendizaje Supervisado)**
3.  **Clustering (Aprendizaje No Supervisado)**

---

### 1. Regresión: Predicción de Alquiler de Bicicletas en Seúl

**Notebook:** `Supervised_learning_(regression_bikes).ipynb`

**Objetivo:** Predecir la cantidad de bicicletas alquiladas en una hora específica basándose en condiciones meteorológicas y temporales.

**Dataset:** Seoul Bike Sharing Demand Data Set (UCI Machine Learning Repository).

**Pasos Realizados:**

* **Análisis Exploratorio de Datos (EDA):**
    * Visualización de la relación entre la cantidad de bicicletas (`bike_count`) y otras variables como temperatura, humedad, viento, visibilidad, radiación solar, lluvia y nieve.
    * Se observó una correlación lineal clara con la temperatura.
* **Preprocesamiento:**
    * Selección de datos para una hora específica (12:00 PM) para simplificar el análisis inicial.
    * Eliminación de columnas no numéricas o irrelevantes para este modelo simple (`Date`, `Holiday`, `Seasons`).
* **Modelado (Regresión Lineal Simple):**
    * Se entrenó un modelo de regresión lineal usando solo la temperatura como variable predictora.
    * Visualización de la línea de regresión ajustada a los datos.
* **Modelado (Regresión Lineal Múltiple):**
    * Se entrenó un modelo utilizando todas las variables numéricas disponibles.
    * **Evaluación:** Se utilizó el Coeficiente de Determinación ($R^2$) y el Error Medio Absoluto (MAE) para medir el rendimiento.
        * *Resultado:* El modelo con todas las variables obtuvo un $R^2$ significativamente mejor (aprox. 0.57) que el modelo de una sola variable (aprox. 0.38).
* **Red Neuronal para Regresión:**
    * Se implementó una red neuronal simple utilizando `TensorFlow`/`Keras` con una capa de normalización y una capa densa.
    * Se comparó su rendimiento (pérdida durante el entrenamiento) con el modelo de regresión lineal.

---

### 2. Clasificación: Detección de Rayos Gamma (MAGIC Gamma Telescope)

**Notebook:** `Supervised_learning_(classification_MAGIC).ipynb`

**Objetivo:** Clasificar eventos capturados por un telescopio Cherenkov como "señal" (rayos gamma, clase `g`) o "ruido" (hadrones, clase `h`).

**Dataset:** MAGIC Gamma Telescope Data Set (UCI Machine Learning Repository).

**Pasos Realizados:**

* **Preprocesamiento:**
    * Codificación de la variable objetivo (`class`): `g` (gamma) = 1, `h` (hadrón) = 0.
    * Visualización de las distribuciones de las características para cada clase.
    * **Oversampling:** Se utilizó `RandomOverSampler` para equilibrar las clases, ya que había más eventos gamma que hadrones.
* **Modelos Entrenados y Evaluados:**
    * Se dividió el dataset en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%).
    * **k-Nearest Neighbors (k-NN):** Clasificación basada en la cercanía a los vecinos más próximos. Se evaluó la precisión, recall y f1-score.
    * **Naive Bayes (GaussianNB):** Modelo probabilístico basado en el teorema de Bayes asumiendo independencia entre características.
    * **Regresión Logística:** Modelo lineal para clasificación binaria.
    * **Support Vector Machine (SVM):** Busca el hiperplano óptimo que separa las clases.
    * **Red Neuronal (TensorFlow):** Se construyó una red neuronal con capas densas, `Dropout` para evitar sobreajuste y función de activación `Sigmoid` en la salida.
        * Se entrenó minimizando la `binary_crossentropy`.
        * Se evaluó la precisión y la pérdida en los conjuntos de entrenamiento y validación.

---

### 3. Clustering: Agrupamiento de Semillas de Trigo

**Notebook:** `Unsupervised_learning_(seeds).ipynb`

**Objetivo:** Agrupar semillas de trigo en diferentes variedades basándose en sus características geométricas, sin usar las etiquetas de clase durante el entrenamiento.

**Dataset:** Seeds Data Set (UCI Machine Learning Repository).

**Pasos Realizados:**

* **Análisis Exploratorio:**
    * Visualización de pares de características (`scatterplot`) para identificar agrupamientos naturales.
* **Modelado (K-Means Clustering):**
    * Se utilizó el algoritmo K-Means para agrupar los datos en 3 clusters (correspondientes a las 3 variedades de trigo).
    * **Transformación de Datos:** Se aplicó Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de los datos a 2 componentes y poder visualizar los clusters resultantes en un gráfico 2D.
    * **Resultados:** Se visualizó cómo el algoritmo K-Means separó las muestras y se comparó cualitativamente con las etiquetas reales (aunque en un escenario puramente no supervisado, no se tendrían estas etiquetas).

---

## Librerías Utilizadas

* `numpy`: Cálculo numérico.
* `pandas`: Manipulación y análisis de datos.
* `matplotlib` y `seaborn`: Visualización de datos.
* `scikit-learn`: Algoritmos de Machine Learning (Regresión, k-NN, SVM, Naive Bayes, K-Means, PCA, métricas, preprocesamiento).
* `imblearn`: Manejo de datasets desbalanceados (Oversampling).
* `tensorflow`: Construcción y entrenamiento de redes neuronales.

## Cómo usar este repositorio

1.  Clona el repositorio.
2.  Asegúrate de tener instaladas las dependencias (`pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow`).
3.  Ejecuta los notebooks en Jupyter o Google Colab para ver el código y los resultados paso a paso.

---
*Este repositorio fue creado con fines educativos siguiendo el tutorial mencionado anteriormente.*


