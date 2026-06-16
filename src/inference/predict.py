import pandas as pd
import os
import mlflow
import mlflow.sklearn
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from datetime import datetime

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"


@asset(
    group_name="inference",
    description="Carga el último modelo validado de MLflow y genera señales de trading diarias.",
)
def daily_trading_signals(context: AssetExecutionContext) -> MaterializeResult:
    """
    Asset que toma los últimos datos calculados en el día, los pasa por el modelo
    registrado y genera un archivo de señales para la ejecución de trading.
    """
    # 1. Obtener los últimos datos de mercado (calculados por el pipeline ETL diario)
    # Estos datos no deben contener la columna 'target_direction'
    latest_features_path = "data/03_features/latest_market_features.csv"

    if not os.path.exists(latest_features_path):
        raise FileNotFoundError(
            f"Datos de inferencia no encontrados en {latest_features_path}."
        )

    df_latest = pd.read_csv(latest_features_path)

    # Guardamos las fechas para asociarlas a la señal
    fechas = df_latest["Date"] if "Date" in df_latest.columns else None

    if "Date" in df_latest.columns:
        X_latest = df_latest.set_index("Date")
    else:
        X_latest = df_latest.copy()

    # 2. Cargar el modelo desde MLflow
    # En un entorno real, descargaríamos la versión con la etiqueta "Production"
    model_name = "Apple_Trading_Model"
    context.log.info(f"Cargando el modelo {model_name} desde MLflow...")

    try:
        # Cargar la última versión del modelo registrado
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    except Exception as e:
        raise RuntimeError(
            f"Error al cargar el modelo de MLflow. ¿Se ha entrenado el modelo? Error: {e}"
        )

    # 3. Generar Inferencia (Señal de Trading)
    context.log.info("Generando predicciones de trading...")
    predictions = model.predict(X_latest)
    probabilities = model.predict_proba(X_latest)[
        :, 1
    ]  # Probabilidad de que suba (clase 1)

    # 4. Formatear y guardar resultados
    results_df = pd.DataFrame(
        {
            "signal": predictions,
            "confidence": probabilities,
            "execution_timestamp": datetime.now().isoformat(),
        }
    )

    if fechas is not None:
        results_df["target_date"] = fechas.values

    # Crear directorio si no existe
    os.makedirs("data/04_predictions", exist_ok=True)

    output_path = (
        f"data/04_predictions/trading_signals_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    results_df.to_csv(output_path, index=False)

    context.log.info(f"Señales generadas y guardadas en {output_path}")

    # Reportar metadata a Dagster
    num_signals = len(results_df)
    buy_signals = int((results_df["signal"] == 1).sum())

    return MaterializeResult(
        metadata={
            "output_path": MetadataValue.path(output_path),
            "total_signals_generated": MetadataValue.int(num_signals),
            "buy_signals": MetadataValue.int(buy_signals),
            "sell_signals": MetadataValue.int(num_signals - buy_signals),
            "mean_confidence": MetadataValue.float(
                float(results_df["confidence"].mean())
            ),
        }
    )
