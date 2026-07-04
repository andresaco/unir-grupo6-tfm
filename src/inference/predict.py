import os
import pandas as pd
import mlflow
import mlflow.sklearn
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)
from ..schemas import validate_df, EngineeredFeaturesRow

# Configuración del backend store de MLflow
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class InferenceConfig(Config):
    """
    Configuración parametrizable para la generación de señales de trading (inferencia).
    """

    company_name: str = "Apple"
    ticker: str = "AAPL"
    dataset_path: str = "data/03_features/AAPL/features/features.csv"


@asset(
    deps=[
        "engineered_features"
    ],  # Se ejecuta automáticamente cuando se recalculan las features
    group_name="inference",
    description="Carga el modelo óptimo desde el Registry de MLflow y genera predicciones operativas para los últimos 60 días.",
)
def generate_trading_signals(
    context: AssetExecutionContext, config: InferenceConfig
) -> MaterializeResult:
    """
    Asset que toma los últimos datos técnicos preparados, aplica un slicing de los últimos
    60 días, carga el modelo de producción de MLflow y genera las señales operativas finales.
    """
    features_path = config.dataset_path

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No se encontró el dataset de características en {features_path}. Ejecuta la ETL primero."
        )

    context.log.info(f"Cargando dataset de características desde {features_path}...")
    df = pd.read_csv(features_path)

    # Validar features leídas
    validate_df(
        df, EngineeredFeaturesRow, stage="generate_trading_signals (read features)"
    )

    # 1. Sincronizar fechas y ordenar de forma cronológica
    if "date" in df.columns:
        df_inference = df.set_index("date")
    else:
        df_inference = df.copy()

    df_inference.index = pd.to_datetime(df_inference.index)
    df_inference = df_inference.sort_index()

    # 2. Aplicar slicing para quedarse únicamente con los últimos 60 días hábiles
    # Esto asegura que los indicadores técnicos (SMA_50, etc.) estén correctamente calculados
    # pero el modelo solo haga inferencia sobre la ventana operativa relevante de 60 días.
    context.log.info(
        "Aplicando slicing temporal para evaluar los últimos 60 días del mercado..."
    )
    # df_inference = df_inference.tail(60)

    # 3. Separar las columnas predictoras excluyendo el target real si estuviera presente
    X_inference = df_inference.drop(columns=["target_direction"], errors="ignore")

    # Asegurar que solo pasamos las columnas numéricas que usó el modelo
    X_inference = X_inference.select_dtypes(include=["number"])

    # 4. Intentar cargar el modelo productivo desde el Model Registry de MLflow
    model_name = f"{config.company_name}_Trading_Model"
    model_uri_production = f"models:/{model_name}/Production"
    model_uri_latest = f"models:/{model_name}/latest"

    model = None
    used_stage = "Production"

    try:
        context.log.info(
            f"Intentando cargar modelo promocionado en Production: {model_uri_production}"
        )
        model = mlflow.sklearn.load_model(model_uri_production)
    except Exception as e:
        context.log.warning(
            f"No se encontró una versión en Production o falló la carga: {e}. "
            f"Se procederá a cargar la versión más reciente ('latest') de respaldo."
        )
        try:
            model = mlflow.sklearn.load_model(model_uri_latest)
            used_stage = "Latest (Respaldo)"
        except Exception as err:
            raise RuntimeError(
                f"Error crítico: No se pudo recuperar ninguna versión del modelo '{model_name}' desde MLflow. "
                f"Detalle: {err}"
            )

    # 5. Inferencia sobre la ventana reducida (60 días)
    context.log.info(
        f"Generando predicciones sobre las {len(X_inference)} observaciones de la ventana..."
    )
    predictions = model.predict(X_inference)
    probabilities = model.predict_proba(X_inference)[
        :, 1
    ]  # Probabilidad de dirección alcista (clase 1)

    # 6. Estructurar el DataFrame de Salida
    results_df = pd.DataFrame(index=df_inference.index)
    results_df["price_close"] = (
        df_inference["close"] if "close" in df_inference.columns else None
    )
    results_df["predicted_signal"] = predictions  # 1 = Compra, 0 = Mantener/Venta
    results_df["confidence"] = probabilities

    df_to_save = results_df.reset_index()
    # Validar señales operativas antes de guardar
    # validate_df(
    #    df_to_save,
    #    TradingSignalsRow,
    #    stage="generate_trading_signals (write predictions)",
    # )

    # 7. Almacenamiento local del artefacto
    output_dir = "data/04_predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config.ticker}_trading_signals.csv")
    df_to_save.to_csv(output_path, index=False)

    context.log.info(f"Señales operativas guardadas exitosamente en {output_path}")

    # 8. Estadísticas finales para el metadato de Dagster
    total_days = len(results_df)
    buy_signals = int((results_df["predicted_signal"] == 1).sum())
    last_signal = int(results_df["predicted_signal"].iloc[-1])
    last_confidence = float(results_df["confidence"].iloc[-1])
    last_date = str(results_df.index[-1].strftime("%Y-%m-%d"))

    return MaterializeResult(
        metadata={
            "output_path": MetadataValue.path(output_path),
            "mlflow_source_stage": MetadataValue.text(used_stage),
            "total_predictions": MetadataValue.int(total_days),
            "buy_signals_count": MetadataValue.int(buy_signals),
            "sell_signals_count": MetadataValue.int(total_days - buy_signals),
            "last_available_date": MetadataValue.text(last_date),
            "last_signal_prediction": MetadataValue.int(
                last_signal
            ),  # 1 = Sube, 0 = Baja
            "last_signal_confidence": MetadataValue.float(last_confidence),
        }
    )
