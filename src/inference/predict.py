import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)
from ..schemas import validate_df, EngineeredFeaturesRow, TradingSignalsRow
from ..etl.core.config import StockDownloadConfig
from .backtesting import load_model_helper

# Configuración del backend store de MLflow
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class InferenceConfig(Config):
    """
    Configuración parametrizable para la generación de señales de trading (inferencia).
    """

    company_name: str = "Apple"
    ticker: str = "AAPL"
    dataset_path: str = "data/03_features/features/AAPL/features.csv"
    model_name: str = "Apple_Trading_Model"


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
    model_name = config.model_name
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
    output_path = os.path.join(
        output_dir, f"{config.ticker}_{model_name}_trading_signals.csv"
    )
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


@asset(
    deps=["incremental_market_data", "aggregated_daily_social_sentiment"],
    group_name="inference",
    description="Descarga e ingesta diaria, selecciona el mejor modelo de MLflow y genera la predicción para un día concreto.",
)
def daily_prediction(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset de producción diaria. Toma los datos incrementales de mercado y las agregaciones de redes sociales,
    realiza la preparación de features de forma idéntica a la fase de entrenamiento, consulta el modelo con
    el mejor accuracy en MLflow, y genera las señales predictivas finales.
    """
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    stock_path = f"data/03_features/stock/{ticker}.csv"
    vix_path = "data/03_features/stock/VIX.csv"
    sentiment_path = f"data/03_features/daily/{ticker}_sentiment.csv"

    if not os.path.exists(stock_path) or not os.path.exists(vix_path):
        raise FileNotFoundError(
            "No se encontraron datos de stock/VIX en data/03_features/stock/. Ejecuta la ETL de mercado primero."
        )

    context.log.info("Cargando datos de stock y VIX para inferencia...")
    df_stock = pd.read_csv(stock_path)
    df_vix = pd.read_csv(vix_path)

    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date
    df_vix["date"] = pd.to_datetime(df_vix["date"]).dt.date

    df_vix = df_vix[["date", "close"]].rename(columns={"close": "VIX"})
    df_features = pd.merge(df_stock, df_vix, on="date", how="left")

    # Cargar sentimiento de Bluesky si existe
    if os.path.exists(sentiment_path):
        context.log.info(f"Incorporando sentimiento de Bluesky desde {sentiment_path}")
        df_sent = pd.read_csv(sentiment_path)
        df_sent["date"] = pd.to_datetime(df_sent["fecha_limpia"]).dt.date

        # Mapear columnas de Bluesky a las esperadas por el esquema (que hereda de GDELT)
        df_sent["sentimiento_promedio"] = df_sent["sentimiento_medio"]
        df_sent["volumen_noticias"] = df_sent["volumen_posts"]
        df_sent["volatilidad_sentimiento"] = df_sent["sentimiento_std"]
        df_sent["sentiment_score"] = df_sent["sentimiento_medio"]

        # Rellenar otras columnas de sentimiento con 0.0
        for col in [
            "puntuacion_positiva",
            "puntuacion_negativa",
            "polaridad_promedio",
            "uso_primera_persona",
        ]:
            df_sent[col] = 0.0

        df_features = pd.merge(
            df_features,
            df_sent[
                [
                    "date",
                    "sentimiento_promedio",
                    "volumen_noticias",
                    "volatilidad_sentimiento",
                    "sentiment_score",
                    "puntuacion_positiva",
                    "puntuacion_negativa",
                    "polaridad_promedio",
                    "uso_primera_persona",
                ]
            ],
            on="date",
            how="left",
        )
    else:
        context.log.warning(
            f"No se encontró archivo de sentimiento de Bluesky en {sentiment_path}. Se usarán valores por defecto."
        )
        for col in [
            "sentimiento_promedio",
            "volumen_noticias",
            "volatilidad_sentimiento",
            "sentiment_score",
            "puntuacion_positiva",
            "puntuacion_negativa",
            "polaridad_promedio",
            "uso_primera_persona",
        ]:
            df_features[col] = 0.0

    # Rellenar cualquier NaN en columnas de sentimiento
    sentiment_cols = [
        "sentimiento_promedio",
        "volumen_noticias",
        "volatilidad_sentimiento",
        "sentiment_score",
        "puntuacion_positiva",
        "puntuacion_negativa",
        "polaridad_promedio",
        "uso_primera_persona",
    ]
    df_features[sentiment_cols] = df_features[sentiment_cols].fillna(0.0)

    # 1. Convertir la columna 'date' a datetime
    df_features["date"] = pd.to_datetime(df_features["date"])

    # Añadir características temporales
    df_features["day_of_week"] = df_features["date"].dt.dayofweek
    df_features["month"] = df_features["date"].dt.month
    df_features["quarter"] = df_features["date"].dt.quarter
    df_features["year"] = df_features["date"].dt.year
    df_features["day_of_month"] = df_features["date"].dt.day
    df_features["week_of_year"] = df_features["date"].dt.isocalendar().week.astype(int)

    # Añadir características cíclicas
    df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
    df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
    df_features["day_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
    df_features["day_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

    # Crear interacciones
    df_features["sentiment_volume_interaction"] = (
        df_features["sentimiento_promedio"] * df_features["volume"]
    )
    df_features["vix_rsi_interaction"] = df_features["VIX"] * df_features["RSI"] / 100

    # Ordenar y asegurar que no haya NaNs en indicadores
    df_features.sort_values("date", inplace=True)
    df_features.reset_index(drop=True, inplace=True)
    df_features = df_features.bfill().ffill()

    # Añadir target temporal para pasar validación de esquema
    if "target_direction" not in df_features.columns:
        df_features["target_direction"] = 0
    else:
        df_features["target_direction"] = (
            df_features["target_direction"].fillna(0).astype(int)
        )

    # Validar el df con EngineeredFeaturesRow
    validate_df(
        df_features,
        EngineeredFeaturesRow,
        stage="daily_prediction (engineered features)",
    )

    # Filtrar por el rango de fechas solicitado para inferencia
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df_inference = df_features[
        (df_features["date"] >= start_dt) & (df_features["date"] <= end_dt)
    ].copy()

    if df_inference.empty:
        context.log.warning(
            f"No hay registros en el rango de fechas {start_date} a {end_date}. Usando el último registro disponible."
        )
        df_inference = df_features.tail(1).copy()

    # Conservar el índice temporal para estructurar el DataFrame final
    df_inference_indexed = df_inference.set_index("date")

    # Separar predictores numéricos en el mismo orden exacto del entrenamiento
    numeric_cols = [
        "close",
        "high",
        "low",
        "open",
        "volume",
        "RSI",
        "daily_return",
        "SMA_10",
        "SMA_50",
        "volatilidad_10d",
        "VIX",
        "volumen_noticias",
        "sentimiento_promedio",
        "puntuacion_positiva",
        "puntuacion_negativa",
        "polaridad_promedio",
        "volatilidad_sentimiento",
        "uso_primera_persona",
        "day_of_week",
        "month",
        "quarter",
        "year",
        "day_of_month",
        "week_of_year",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "sentiment_volume_interaction",
        "vix_rsi_interaction",
    ]

    X_inference = df_inference_indexed[numeric_cols].copy()

    # 1. Configuración del backend store de MLflow
    mlflow.set_tracking_uri("sqlite:///runtime/mlflow/mlflow.db")
    client = mlflow.tracking.MlflowClient()

    model_names = [
        f"{config.name}_Trading_Model",
        f"{config.name}_XGBoost_Model",
        f"{config.name}_LSTM_Model",
    ]

    best_model_name = None
    best_version = None
    best_accuracy = 0.0

    context.log.info("Buscando el modelo con el alias 'champion' en MLflow...")
    for name in model_names:
        try:
            model_version = client.get_model_version_by_alias(name, "champion")
            if model_version:
                best_model_name = name
                best_version = model_version.version
                # Obtener la precisión (accuracy) del run original
                run = client.get_run(model_version.run_id)
                best_accuracy = float(run.data.metrics.get("accuracy", 0.0))
                break
        except Exception:
            continue

    if not best_model_name:
        context.log.warning(
            "No se encontró ningún modelo con alias 'champion' en MLflow. Usando RandomForest de respaldo..."
        )
        best_model_name = f"{config.name}_Trading_Model"
        best_version = "Latest"
        best_accuracy = 0.0
        # Cargar de respaldo sin alias usando load_model_helper
        model, selected_uri = load_model_helper(best_model_name, context)
    else:
        context.log.info(
            f"Modelo seleccionado para inferencia diaria (alias 'champion'): {best_model_name} (Versión: {best_version}, Accuracy: {best_accuracy:.4f})"
        )
        try:
            model, selected_uri = load_model_helper(
                best_model_name, context, alias="champion"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error cargando el modelo '{best_model_name}' con alias 'champion': {e}"
            )

    # Inferencia adaptada según el sabor del modelo
    if "LSTM" in best_model_name:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_inference)

        seq_length = 10
        xs = []
        for i in range(len(X_scaled)):
            if i < seq_length - 1:
                pad_count = seq_length - 1 - i
                padding = np.repeat(X_scaled[0:1], pad_count, axis=0)
                x = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
            else:
                x = X_scaled[i - seq_length + 1 : i + 1]
            xs.append(x)
        xs = np.array(xs)

        import torch

        if hasattr(model, "eval"):
            model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(xs, dtype=torch.float32)
                outputs = model(tensor_x).numpy().flatten()
        else:
            outputs = model.predict(xs).flatten()

        predictions = (outputs > 0.5).astype(int)
        probabilities = outputs

    elif "XGBoost" in best_model_name:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_input = scaler.fit_transform(X_inference)

        predictions = model.predict(X_input)
        if predictions.dtype.kind in ("f", "c"):
            predictions = (predictions > 0.5).astype(int)
        else:
            predictions = predictions.astype(int)

        probabilities = model.predict_proba(X_input)[:, 1]

    else:
        X_input = X_inference.values if hasattr(X_inference, "values") else X_inference
        predictions = model.predict(X_input)
        if predictions.dtype.kind in ("f", "c"):
            predictions = (predictions > 0.5).astype(int)
        else:
            predictions = predictions.astype(int)

        probabilities = model.predict_proba(X_input)[:, 1]

    # Estructurar DataFrame de salida
    results_df = pd.DataFrame(index=df_inference_indexed.index)
    results_df["price_close"] = (
        df_inference_indexed["close"]
        if "close" in df_inference_indexed.columns
        else None
    )
    results_df["predicted_signal"] = predictions
    results_df["confidence"] = probabilities

    df_to_save = results_df.reset_index()
    if "index" in df_to_save.columns:
        df_to_save = df_to_save.rename(columns={"index": "date"})
    elif "date" not in df_to_save.columns:
        df_to_save = df_to_save.rename(columns={df_to_save.columns[0]: "date"})

    df_to_save["date"] = df_to_save["date"].astype(str)

    # Validar señales operativas antes de guardar
    validate_df(
        df_to_save,
        TradingSignalsRow,
        stage="daily_prediction (write predictions)",
    )

    # Guardar predicción de producción haciendo append y deduplicando en el archivo unificado
    output_dir = "data/04_predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config.ticker}_trading_signals.csv")

    if os.path.exists(output_path):
        try:
            context.log.info(
                f"Leyendo predicciones existentes desde {output_path} para realizar append..."
            )
            df_existing = pd.read_csv(output_path)
            df_combined = pd.concat([df_existing, df_to_save], ignore_index=True)
        except Exception as e:
            context.log.warning(
                f"Error leyendo archivo existente {output_path}: {e}. Se sobrescribirá."
            )
            df_combined = df_to_save
    else:
        df_combined = df_to_save

    # Asegurar tipo str en date para ordenación y deduplicación consistente
    df_combined["date"] = df_combined["date"].astype(str)
    df_combined = df_combined.drop_duplicates(subset=["date"], keep="last")
    df_combined = df_combined.sort_values(by="date").reset_index(drop=True)

    # Validar señales operativas combinadas completas antes de guardar
    validate_df(
        df_combined,
        TradingSignalsRow,
        stage="daily_prediction (write unified predictions)",
    )

    df_combined.to_csv(output_path, index=False)
    context.log.info(
        f"Predicciones unificadas de producción actualizadas en: {output_path}"
    )

    total_days = len(results_df)
    buy_signals = int((results_df["predicted_signal"] == 1).sum())
    last_signal = int(results_df["predicted_signal"].iloc[-1]) if total_days > 0 else 0
    last_confidence = (
        float(results_df["confidence"].iloc[-1]) if total_days > 0 else 0.0
    )
    last_date = str(df_to_save["date"].iloc[-1]) if total_days > 0 else start_date

    return MaterializeResult(
        metadata={
            "output_path": MetadataValue.path(output_path),
            "selected_best_model": MetadataValue.text(best_model_name),
            "model_accuracy": MetadataValue.float(best_accuracy),
            "total_predictions": MetadataValue.int(len(df_combined)),
            "buy_signals_count": MetadataValue.int(buy_signals),
            "sell_signals_count": MetadataValue.int(total_days - buy_signals),
            "last_prediction_date": MetadataValue.text(last_date),
            "last_signal_prediction": MetadataValue.int(last_signal),
            "last_signal_confidence": MetadataValue.float(last_confidence),
        }
    )


@asset(
    deps=["daily_prediction"],
    group_name="inference",
    description="Genera el order book operativo a partir de las predicciones diarias de la señal de trading, calculando métricas de equidad y señales de compra/venta/mantener del backtest.",
)
def order_book(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker = config.ticker
    predictions_path = f"data/04_predictions/{ticker}_trading_signals.csv"

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de predicciones en {predictions_path}. Ejecuta la predicción diaria primero."
        )

    context.log.info(f"Cargando predicciones desde {predictions_path}...")
    df = pd.read_csv(predictions_path)

    if df.empty:
        context.log.warning("El dataset de predicciones está vacío.")
        return MaterializeResult()

    # 1. Asegurar orden cronológico por fecha
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # 2. Calcular retornos diarios de la cotización
    df["Returns"] = df["price_close"].pct_change().fillna(0.0)

    # 3. Calcular Buy & Hold Equity
    capital_inicial = 100000.0
    df["Buy_Hold_Equity"] = capital_inicial * (1 + df["Returns"]).cumprod()

    # 4. Calcular retornos de la estrategia y equidad (desplazamiento de 1 día)
    df["Strategy_Returns"] = df["Returns"] * df["predicted_signal"].shift(1).fillna(0.0)
    df["Strategy_Equity"] = capital_inicial * (1 + df["Strategy_Returns"]).cumprod()

    # 5. Generar señales operativas (transición de posiciones)
    df["Position"] = df["predicted_signal"].diff()
    # Para la primera fila, diff() es NaN. Si el primer día el modelo predice 1, compramos (Position = 1).
    # Si predice 0, nos quedamos en liquidez (Position = 0).
    df.loc[df.index[0], "Position"] = df.loc[df.index[0], "predicted_signal"]

    df["Buy_Signal"] = df["Position"] == 1
    df["Sell_Signal"] = df["Position"] == -1

    # Mapeo descriptivo de la acción recomendada
    def map_trading_action(row):
        pos = row["Position"]
        sig = row["predicted_signal"]
        if pos == 1:
            return "COMPRA"
        elif pos == -1:
            return "VENTA"
        else:
            return "MANTENER_POSICION" if sig == 1 else "MANTENER_LIQUIDEZ"

    df["Trading_Action"] = df.apply(map_trading_action, axis=1)

    # Convertir fecha de nuevo a string para consistencia de guardado
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Guardar únicamente en el fichero oficial por ticker
    output_path = f"data/04_predictions/{ticker}_order_book.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    context.log.info(f"Order book guardado en: {output_path}")

    return MaterializeResult(
        metadata={
            "ticker": MetadataValue.text(ticker),
            "total_days": MetadataValue.int(len(df)),
            "capital_inicial": MetadataValue.float(capital_inicial),
            "final_bh_equity": MetadataValue.float(
                float(df["Buy_Hold_Equity"].iloc[-1])
            ),
            "final_strategy_equity": MetadataValue.float(
                float(df["Strategy_Equity"].iloc[-1])
            ),
            "buy_signals_count": MetadataValue.int(int(df["Buy_Signal"].sum())),
            "sell_signals_count": MetadataValue.int(int(df["Sell_Signal"].sum())),
        }
    )
