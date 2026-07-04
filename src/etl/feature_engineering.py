import os
import pandas as pd
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from .core.config import StockDownloadConfig
from ..schemas import (
    validate_df,
    StockProcessedRow,
    GdeltSentimentRow,
    EngineeredFeaturesRow,
)
import numpy as np


OUTPUT_DIR = "features"


@asset(
    deps=[
        "raw_stock_data",
        "raw_vix_data",
        "processed_gdelt_sentiment_data",
    ],
    group_name="feature_engineering",
    description="Une datos de cotización, VIX y sentimiento, y calcula indicadores técnicos.",
)
def engineered_features(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    stock_path = f"data/03_features/stock/{ticker}.csv"
    vix_path = "data/03_features/stock/VIX.csv"
    safe_name = str(config.name).lower().replace(" ", "_")
    dynamic_sentiment_path = f"data/02_processed/sentiment/gdelt_{safe_name}_avanzado_{start_date}_a_{end_date}.csv"
    fallback_sentiment_path = (
        "data/01_raw/sentiment/gdelt_apple_avanzado_2020-01-01_a_2026-12-31.csv"
    )
    sentiment_path = (
        dynamic_sentiment_path
        if os.path.exists(dynamic_sentiment_path)
        else fallback_sentiment_path
    )

    context.log.info("Cargando datos crudos para Feature Engineering...")
    df_stock = pd.read_csv(stock_path)
    df_vix = pd.read_csv(vix_path)

    # Validar al leer raw stock y vix
    validate_df(
        df_stock, StockProcessedRow, stage="engineered_features (read stock raw)"
    )
    validate_df(df_vix, StockProcessedRow, stage="engineered_features (read vix raw)")

    df_stock["date"] = pd.to_datetime(df_stock["date"]).dt.date
    df_vix["date"] = pd.to_datetime(df_vix["date"]).dt.date

    df_vix = df_vix[["date", "close"]].rename(columns={"close": "VIX"})
    df_features = pd.merge(df_stock, df_vix, on="date", how="left")

    if os.path.exists(sentiment_path):
        context.log.info(f"Incorporando datos de sentimiento desde {sentiment_path}")
        df_sent = pd.read_csv(sentiment_path)

        # Validar al leer sentimiento GDELT
        validate_df(
            df_sent, GdeltSentimentRow, stage="engineered_features (read GDELT raw)"
        )

        df_sent["date"] = pd.to_datetime(df_sent["fecha"]).dt.date
        df_features = pd.merge(df_features, df_sent, on="date", how="left")

        if "sentiment_score" in df_features.columns:
            df_features["sentiment_score"] = df_features["sentiment_score"].fillna(0)
    else:
        context.log.warning(
            f"No se encontró archivo de sentimiento en {sentiment_path}. Se omitirá."
        )

    # 1. Convertir la columna 'date' a datetime
    df_features["date"] = pd.to_datetime(df_features["date"])
    # Añadir características temporales
    df_features["day_of_week"] = df_features["date"].dt.dayofweek
    df_features["month"] = df_features["date"].dt.month
    df_features["quarter"] = df_features["date"].dt.quarter
    df_features["year"] = df_features["date"].dt.year
    df_features["day_of_month"] = df_features["date"].dt.day
    df_features["week_of_year"] = df_features["date"].dt.isocalendar().week

    # Añadir características cíclicas
    df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
    df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)
    df_features["day_sin"] = np.sin(2 * np.pi * df_features["day_of_week"] / 7)
    df_features["day_cos"] = np.cos(2 * np.pi * df_features["day_of_week"] / 7)

    # Crear interacciones entre variables clave
    # 'Return_5d' no está en el DataFrame, por lo que se eliminará por ahora.
    # Necesitaríamos recalcular los retornos si se quieren usar.
    df_features["sentiment_volume_interaction"] = (
        df_features["sentimiento_promedio"] * df_features["volume"]
    )
    df_features["vix_rsi_interaction"] = df_features["VIX"] * df_features["RSI"] / 100

    print(df_features.head(5))
    context.log.info("Calculando indicadores técnicos...")
    df_features.sort_values("date", inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    filas_antes = len(df_features)
    df_features.dropna(inplace=True)
    context.log.info(f"Filas limpiadas (NaNs): {filas_antes - len(df_features)}")

    print(df_features.head(5))
    df_features.to_csv("test.csv", index=False)

    # Validar el dataset final de features antes de guardar
    validate_df(
        df_features, EngineeredFeaturesRow, stage="engineered_features (write features)"
    )

    output_dir = f"data/03_features/{OUTPUT_DIR}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features.csv")

    df_features.to_csv(output_path, index=False)

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(output_path),
            "total_rows": MetadataValue.int(len(df_features)),
            "class_balance_up": MetadataValue.float(
                float(df_features["target_direction"].mean())
            ),
        }
    )
