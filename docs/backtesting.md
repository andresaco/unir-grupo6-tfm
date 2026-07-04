# **📊 Manual del Sistema de Backtesting de Estrategias**

Este documento describe la arquitectura, la lógica matemática, la forma de ejecución y los lugares donde se pueden consultar los resultados del sistema de backtesting comparativo para los modelos de Machine Learning.

---

## **1. Arquitectura y Lógica del Pipeline**

El pipeline de backtesting se implementa mediante un job de Dagster llamado **`run_backtest_pipeline`**. Este job ejecuta el asset:

* **`run_backtest`:**
  * Depende directamente del pipeline de ETL (`engineered_features`).
  * Carga el dataset de características históricas (`data/03_features/features/{ticker}/features.csv`).
  * Filtra los datos para el período temporal indicado (`initial_date` a `end_date`).
  * Para **todos los modelos** configurados (por defecto: `Apple_Trading_Model` (RandomForest), `Apple_XGBoost_Model` y `Apple_LSTM_Model`):
    1. Carga el modelo desde el **Model Registry de MLflow** (usando la versión `Production` o la `latest` de respaldo).
    2. Ejecuta predicciones de señales en el período. Si el modelo es **LSTM (PyTorch)**, escala los datos de forma local y genera secuencias 3D del tamaño correcto. Si el modelo es **XGBoost**, aplica escalado. Si es **RandomForest**, ejecuta las predicciones en crudo.
    3. Simula la ejecución de la estrategia y calcula las métricas financieras clave.
    4. Guarda los resultados y curvas de equidad detallados de cada modelo en un CSV separado.
    5. Inyecta los resultados del backtest directamente en la ejecución (run) que entrenó ese modelo en MLflow.
    6. Registra la ejecución en el experimento centralizado de comparación `Backtest_Comparison` en MLflow.
    7. Genera una tabla comparativa en Markdown para la UI de Dagster y actualiza el histórico en `runtime/backtest_comparison.csv`.

---

## **2. Lógica Matemática del Backtesting**

* **Retorno del Activo ($R_t$):** Cambios porcentuales en el precio de cierre:
  $$R_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$$
* **Retorno de la Estrategia ($R_s$):** Dado que la predicción se realiza al final del día $t$, la orden de compra o venta se ejecuta al precio del día $t+1$:
  $$R_{s, t+1} = R_{t+1} \times \text{predicted\_signal}_t$$
* **Curvas de Equity (Equidad):**
  * Buy & Hold: $\text{Capital}_0 \times \prod (1 + R_t)$
  * Estrategia: $\text{Capital}_0 \times \prod (1 + R_{s, t})$
* **Sharpe Ratio (Anualizado):**
  $$\text{Sharpe} = \sqrt{252} \times \frac{\text{Media}(R_s)}{\text{Desviación Estándar}(R_s)}$$
* **Sortino Ratio (Anualizado):** Considera únicamente la desviación estándar de los retornos negativos (riesgo a la baja).
* **Maximum Drawdown (Máxima Caída):** El peor descenso desde el pico anterior en la curva de equidad.
* **Calmar Ratio:**
  $$\text{Calmar} = \frac{\text{Retorno Total (\%)}}{\text{Max Drawdown (\%)}}$$

---

## **3. ¿Cómo Ejecutar el Backtesting?**

### **A. Desde la Interfaz Gráfica de Dagster (Recomendado)**

1. Levanta el orquestador si no lo has hecho:
   ```bash
   uv run dagster-dev
   ```
2. Accede a la interfaz web (normalmente en [http://127.0.0.1:3000](http://127.0.0.1:3000)).
3. Navega a **Jobs** en la barra lateral izquierda y selecciona **`run_backtest_pipeline`**.
4. Haz clic en la pestaña **Launchpad**.
5. Define la configuración de fechas, capital inicial y la lista de modelos a evaluar en el YAML:
   ```yaml
   ops:
     run_backtest:
       config:
         ticker: "AAPL"
         initial_date: "2023-01-01"
         end_date: "2023-12-31"
         capital_inicial: 100000.0
         model_names:
           - "Apple_Trading_Model"
           - "Apple_XGBoost_Model"
           - "Apple_LSTM_Model"
   ```
6. Haz clic en **Launch Run** (en la esquina inferior derecha).

---

## **4. ¿Dónde Observar los Resultados?**

El sistema centraliza las métricas comparativas en cuatro ubicaciones:

### **1. Dagster UI (Tabla comparativa visual)**
Al terminar la ejecución del job, haz clic en el asset `run_backtest` dentro del run log o la vista de grafo. En el panel lateral derecho verás la pestaña **Metadata** con la tabla Markdown comparando todos los modelos evaluados en la misma ventana de tiempo:

| Modelo | Retorno Estrategia | Retorno B&H | Sharpe Ratio (Estr) | Max Drawdown (Estr) | Calmar Ratio | N° Trades | Win Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Apple_Trading_Model** | 15.40% | 12.10% | 1.15 | -8.50% | 1.81 | 14 | 55.0% |
| **Apple_XGBoost_Model** | 22.10% | 12.10% | 1.85 | -5.20% | 4.25 | 18 | 65.0% |
| **Apple_LSTM_Model** | -2.30% | 12.10% | -0.10 | -15.40% | -0.15 | 8 | 40.0% |

### **2. Archivo de Comparación Local (`runtime/backtest_comparison.csv`)**
Cada ejecución del backtesting añade una fila con el resumen a [runtime/backtest_comparison.csv](runtime/backtest_comparison.csv).
Este archivo te permite mantener un histórico persistente de todas las pruebas realizadas para todos los modelos y períodos.

### **3. MLflow UI (Gráficos comparativos)**
Levanta la UI de MLflow:
```bash
uv run mlflow-ui
```
Navega a [http://127.0.0.1:5000](http://127.0.0.1:5000). Tendrás los resultados en:
* **El experimento "Backtest_Comparison":** Contiene corridas exclusivas para cada backtest ejecutado. Puedes seleccionar varias de estas corridas en MLflow y hacer clic en **Compare** para visualizar gráficos de barras que comparan el Retorno, Sharpe Ratio o Max Drawdown de los modelos XGBoost, RandomForest y LSTM lado a lado.
* **Corridas originales:** Si buscas la corrida de entrenamiento original del modelo (en los experimentos `Apple`, etc.), verás que se le han inyectado nuevas métricas prefijadas con `backtest_` (ej. `backtest_return_strategy`, `backtest_sharpe_strategy`).

### **4. Detalle Diario del Backtest (`data/05_backtests/`)**
Si deseas graficar las curvas de equidad detalladas de cada modelo o analizar sus señales diarias, el pipeline genera un archivo CSV completo para cada modelo en:
`data/05_backtests/{ticker}_{model_name}_backtest.csv` (ej. [AAPL_Apple_XGBoost_Model_backtest.csv](data/05_backtests/AAPL_Apple_XGBoost_Model_backtest.csv)).
