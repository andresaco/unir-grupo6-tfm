# Explicabilidad de Modelos de Predicción Financiera con SHAP

Este documento detalla la aproximación metodológica y técnica utilizada para la **explicabilidad** de los modelos de aprendizaje automático integrados en el pipeline de entrenamiento financiero. 

En la predicción de mercados financieros, entender *por qué* un modelo toma una decisión de inversión (como predecir una dirección alcista o bajista) es tan crucial como su precisión, tanto para la confianza del operador como para el cumplimiento normativo.

---

## ¿Qué es SHAP (SHapley Additive exPlanations)?

**SHAP** es un marco de trabajo basado en la teoría de juegos cooperativos para explicar la salida de cualquier modelo de aprendizaje automático. 

Didácticamente, imagina que cada característica de los datos (el precio de cierre, el VIX, el sentimiento de las noticias) es un "jugador" en un equipo, y la predicción final del modelo es el "premio" del juego. SHAP calcula la **contribución marginal** de cada jugador a la puntuación del equipo comparándolo con todas las coaliciones posibles de jugadores. 

A diferencia de otros métodos de importancia de variables, SHAP cumple con tres propiedades esenciales:
1. **Consistencia:** Si un modelo cambia de modo que una característica tiene más impacto en la predicción, su valor SHAP no disminuirá.
2. **Eficiencia Local (Aditividad):** La suma de las atribuciones SHAP de todas las variables coincide exactamente con la diferencia entre la predicción local y el valor esperado (predicción promedio) del conjunto de entrenamiento.
3. **Inutilidad (Missingness):** Si una variable no tiene ningún impacto, su atribución SHAP es cero.

---

## 1. Random Forest (Bosque Aleatorio)

### Metodología de Explicabilidad
* **Algoritmo Base:** Bagging de árboles de decisión independientes.
* **Explainer de SHAP:** `TreeExplainer` (invocado de forma nativa mediante `mlflow.models.evaluate`).
* **Dimensión de Entrada:** Tabular 2D `(muestras, características)`.

### Explicación Didáctica
El Bosque Aleatorio divide recursivamente el espacio de características seleccionando subconjuntos aleatorios de variables en cada nodo de sus árboles. El `TreeExplainer` aprovecha esta estructura de árbol para calcular valores SHAP exactos en tiempo polinómico. 

Para cada árbol del bosque, el algoritmo evalúa cómo varía la probabilidad de la predicción en función de si una característica está o no disponible en el camino de decisión. Al final, promedia las contribuciones individuales de todos los árboles para obtener la contribución total de la variable. 

* **Integración con MLflow:** Se realiza de forma directa pasando el modelo de scikit-learn y los datos de prueba sin procesar. MLflow almacena gráficos globales de importancia de variables (*SHAP Summary Plots*) y guarda el objeto explainer para su reutilización.

---

## 2. XGBoost (Extreme Gradient Boosting)

### Metodología de Explicabilidad
* **Algoritmo Base:** Boosting de árboles secuenciales, donde cada nuevo árbol corrige los errores de los anteriores.
* **Explainer de SHAP:** `TreeExplainer` de optimización rápida (invocado de forma nativa mediante `mlflow.models.evaluate`).
* **Dimensión de Entrada:** Tabular 2D `(muestras, características)` con variables previamente normalizadas.

### Explicación Didáctica
A diferencia de Random Forest, donde los árboles se promedian, en XGBoost los árboles se suman secuencialmente y están altamente correlacionados entre sí. SHAP calcula el impacto marginal de cada característica teniendo en cuenta que el impacto de una variable en un árbol tardío depende del error remanente de los árboles anteriores.

Un aspecto didáctico crítico aquí es la **escala y balance de datos**:
* El modelo se entrena sobre datos normalizados (`StandardScaler`) y balanceados artificialmente mediante `SMOTE`.
* Por lo tanto, para que los valores SHAP y el explainer sean representativos, el conjunto de prueba también debe suministrarse en el mismo espacio normalizado. 
* Reconstruimos un DataFrame a partir de los datos normalizados para que el explainer asocie correctamente cada columna numérica con su nombre de característica semántico original.

---

## 3. Red LSTM (Long Short-Term Memory)

### Metodología de Explicabilidad
* **Algoritmo Base:** Red Neuronal Recurrente (RNN) profunda con compuertas de memoria para aprender dependencias temporales en series de tiempo.
* **Explainer de SHAP:** `DeepExplainer` (con backend de PyTorch implementado a medida).
* **Dimensión de Entrada:** Tensorial 3D `(muestras, pasos_de_tiempo, características)`. En nuestro pipeline, `(N, 5, F)` donde 5 representa los 5 días de secuencia histórica evaluados para predecir el día siguiente.

### Explicación Didáctica y Desafío Tridimensional
Los algoritmos neuronales y recursivos no tienen estructuras lógicas rígidas como los árboles, y manejan tensores tridimensionales que los evaluadores tabulares tradicionales de MLflow no pueden interpretar de forma nativa.

Para solucionar esto, aplicamos una estrategia de **explicabilidad profunda**:
1. **Definición de Fondo (Background Distribution):** `DeepExplainer` requiere una distribución de referencia que represente la "ausencia de información". Tomamos un subconjunto representativo del conjunto de entrenamiento de series de tiempo para actuar como línea base.
2. **Propagación del Gradiente (Deep SHAP):** El algoritmo propaga las atribuciones SHAP hacia atrás desde el nodo de salida (Sigmoide) hasta las entradas físicas mediante una aproximación basada en gradientes (*Integrated Gradients* adaptado a SHAP).
3. **Colapso Temporal:** La salida del explainer entrega una matriz SHAP de 3 dimensiones de forma `(N, 5, F)`. Esto significa que obtenemos la contribución de la característica $F$ en el día $t-4, t-3, \dots, t$. Para producir un resumen legible:
   * Promediamos las contribuciones absolutas a lo largo del eje temporal (dimensión 1). Esto colapsa el tensor a una matriz de forma `(N, F)`.
   * Esto nos permite responder a la pregunta: *"De media, ¿cuál es el peso de esta característica a lo largo de la ventana de predicción de 5 días?"*.
   * Generamos un gráfico *summary_plot* 2D que mapea la contribución agregada de cada característica y lo registramos como una imagen PNG de forma directa en los artefactos de la ejecución de MLflow.

---

## Resumen Comparativo de Aproximaciones

| Modelo | Tipo de Explainer | Tipo de Datos | Desafío de Explicabilidad | Método de Registro en MLflow |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | `TreeExplainer` | Tabular (2D) | Modelado básico sin dependencias jerárquicas. | `mlflow.models.evaluate` (Nativo) |
| **XGBoost** | `TreeExplainer` | Tabular (2D) | Sensible a la normalización de la escala de entrada. | `mlflow.models.evaluate` con DataFrame reconstruido a escala. |
| **LSTM** | `DeepExplainer` | Secuencias (3D) | Dimensiones temporales adicionales y comportamiento no lineal profundo. | Cálculo y colapso temporal manual; registro como artefacto PNG. |
