# 📊 Comparativa de Ejecuciones de Modelos (MLflow Runs)

Este informe presenta la evaluación y comparación de las últimas 6 días/ejecuciones registradas en MLflow para los modelos de predicción financiera de **Apple (AAPL)** en el período `2021-01-02` a `2024-12-31`.

---

## 📈 Tabla Comparativa de Métricas

A continuación se detallan las métricas clave obtenidas en la evaluación de cada modelo:

| Run ID | Nombre de Ejecución | Tipo de Modelo | Sentimiento (GDELT) | Accuracy 🎯 | Precision 🔍 | F1-Score ⚖️ | Estado |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `7fc6f85b820c` | `LSTM_NeuralNetwork_Traditional` | Red Neuronal LSTM | ❌ No | **0.5241** | 0.6709 | **0.5436** | Finalizado |
| `48e9c2addec7` | `XGBoost_Classifier` | XGBoost Classifier |  Sí | 0.4688 | **0.7222** | 0.3377 | Finalizado |
| `296af1bca0e0` | `LSTM_NeuralNetwork` | Red Neuronal LSTM |  Sí | 0.4545 | 0.6400 | 0.3855 | Finalizado |
| `bcf03c42f8ea` | `XGBoost_Classifier_Traditional` | XGBoost Classifier | ❌ No | 0.4375 | 0.7083 | 0.2394 | Finalizado |
| `87e5bc6ba8f5` | `RandomForest_Ensemble` | Random Forest |  Sí | 0.4167 | 0.6667 | 0.1765 | Finalizado |
| `762ca15d7327` | `RandomForest_Ensemble_Traditional` | Random Forest | ❌ No | 0.3802 | 0.4286 | 0.0480 | Finalizado |

---

## 🔍 Análisis y Conclusiones

### 1. Impacto del Sentimiento (GDELT)
El impacto de incorporar los datos de sentimiento de noticias (capa NLP) varía según la arquitectura del modelo:
* **Mejora en XGBoost y Random Forest:** 
  Tanto para **XGBoost** (Accuracy aumenta de `0.4375` a `0.4688`, Precision de `0.7083` a `0.7222`) como para **Random Forest** (Accuracy de `0.3802` a `0.4167`, Precision de `0.4286` a `0.6667`), la inclusión del análisis de sentimiento de GDELT generó mejoras significativas en todas las métricas. El caso de Random Forest es el más notable, rescatando al modelo tradicional del colapso (F1-score de solo `0.048` sube a `0.1765`).
* **Degradación en LSTM:**
  La red neuronal **LSTM** sufrió una reducción en el desempeño al añadir el sentimiento (`0.5241` vs `0.4545` de Accuracy). Esto sugiere que la secuenciación temporal de las características de sentimiento de GDELT podría estar introduciendo ruido o que la estructura de la red requiere mayor regularización al incrementar la dimensionalidad.

### 2. Ganador General
* **Mejor Modelo por Tasa de Acierto (Accuracy) y Balance (F1):** **`LSTM_NeuralNetwork_Traditional`** (Accuracy = **52.41%**, F1 = **0.5436**). A pesar de descartar datos de noticias, la capacidad de la LSTM de capturar patrones de series de tiempo puramente financieros resulta ser la más efectiva en general.
* **Mejor Modelo por Fiabilidad de Señal de Compra (Precision):** **`XGBoost_Classifier` con sentimiento** (Precision = **72.22%**). Si se busca una estrategia conservadora donde las alertas de "Compra" tengan la mínima tasa de falsos positivos, este modelo de XGBoost enriquecido con sentimientos es la opción óptima.
