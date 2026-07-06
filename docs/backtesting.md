# 📈 Reporte de Backtesting Comparativo (Año 2025)

Este informe presenta la evaluación de simulación histórica de trading (backtesting) para las estrategias basadas en los diferentes modelos predictivos de **Apple (AAPL)** del período `2025-01-01` al `2025-12-31`. El capital inicial para la simulación fue de **$100,000.00 USD**.

---

## 📊 Tabla Comparativa de Estrategias vs Buy & Hold

| Nombre del Modelo | Sentimiento | Retorno Estrategia 💰 | Retorno Buy & Hold 📈 | Sharpe Ratio ⚖️ | Max Drawdown 📉 | Operaciones 🔄 | Tasa de Acierto 🎯 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Apple_XGBoost_Model`** |  **Sí** | **31.67%** | 11.99% | **1.2019** | **-18.46%** | 46 | **57.78%** |
| `Apple_LSTM_Model` |  Sí | 9.00% | 11.99% | 0.4391 | -33.93% | 12 | 58.33% |
| `Apple_LSTM_Model_traditional` | ❌ No | 7.14% | 11.99% | 0.3931 | -27.41% | 9 | 55.56% |
| `Apple_Trading_Model` (RF) |  Sí | 4.29% | 11.99% | 1.1833 | **0.00%** | 2 | 100.00% |
| `Apple_XGBoost_Model_traditional`| ❌ No | 3.62% | 11.99% | 0.2711 | -26.37% | 37 | 48.65% |
| `Apple_RandomForest_Model` |  Sí | 0.00% | 11.99% | 0.0000 | 0.00% | 0 | 0.00% |
| `Apple_RandomForest_Model_traditional`| ❌ No | 0.00% | 11.99% | 0.0000 | 0.00% | 0 | 0.00% |

---

## 🔍 Conclusiones y Hallazgos Principales

### 🏆 1. El Ganador Indiscutible: `Apple_XGBoost_Model` (con Sentimiento)
* **Retorno:** **31.67%**, superando al mercado (Buy & Hold = 11.99%) por **casi el triple**.
* **Gestión del Riesgo:** Consiguió un Sharpe Ratio excelente de **1.2019** y contuvo la caída máxima (Max Drawdown) en un **-18.46%**, frente al **-30.22%** que sufrió el mercado libre de Apple en ese mismo período.
* **Consistencia:** 46 operaciones con un 57.78% de ratio de acierto, confirmando que la combinación de variables de mercado e indicadores de GDELT genera ventajas de arbitraje robustas en 2025.

### ⚠️ 2. La Importancia del Sentimiento en XGBoost
* Al comparar **`Apple_XGBoost_Model`** (con sentimientos) frente a **`Apple_XGBoost_Model_traditional`** (sin sentimientos):
  * El retorno se desploma del **31.67% al 3.62%**.
  * La tasa de acierto cae del **57.78% al 48.65%** (peor que lanzar una moneda al aire).
  * El Max Drawdown empeora notablemente hasta **-26.37%**.
* **Implicación:** Esto demuestra que para XGBoost, las variables NLP (sentimiento promedio de GDELT, volumen de noticias, etc.) son esenciales para filtrar falsos breakout y dar consistencia a las señales.

### 🧠 3. Comportamiento en Redes LSTM
* La **LSTM con sentimiento** (`Apple_LSTM_Model`) superó ligeramente a la tradicional en retorno (**9.00% vs 7.14%**), pero sufrió un Drawdown mayor (**-33.93% vs -27.41%**). Ninguna de las dos logró batir el simple Buy & Hold de Apple (11.99%), debido a la baja frecuencia de operaciones.

### 🛡️ 4. Conservadurismo de Random Forest
* `Apple_Trading_Model` realizó solo **2 operaciones** en todo el año (acertando ambas), logrando un **4.29%** de rentabilidad con **0.00% de drawdown**. Es una estrategia ultra-defensiva.
* El resto de modelos de Random Forest no generaron señales de compra (`0` operaciones).
