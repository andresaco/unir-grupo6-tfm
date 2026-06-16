import os
import pandas as pd
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from .stock import StockDownloadConfig  # Importar config compartida


@asset(
    deps=["raw_stock_data", "raw_vix_data"],  # Asegura el orden de ejecución
    group_name="feature_engineering",
    description="Une datos de cotización, VIX y sentimiento, y calcula indicadores técnicos.",
)
def engineered_features(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    stock_path = f"data/01_raw/stock/{ticker}_{start_date}_{end_date}.csv"
    vix_path = f"data/01_raw/stock/VIX_{start_date}_{end_date}.csv"
    sentiment_path = (
        "data/01_raw/sentiment/gdelt_apple_avanzado_2020-01-01_a_2026-12-31.csv"
    )

    context.log.info("Cargando datos crudos para Feature Engineering...")
    df_stock = pd.read_csv(stock_path)
    df_vix = pd.read_csv(vix_path)

    df_stock["Date"] = pd.to_datetime(df_stock["Date"]).dt.date
    df_vix["Date"] = pd.to_datetime(df_vix["Date"]).dt.date

    df_vix = df_vix[["Date", "Close"]].rename(columns={"Close": "VIX_Close"})
    df_features = pd.merge(df_stock, df_vix, on="Date", how="left")

    if os.path.exists(sentiment_path):
        context.log.info(f"Incorporando datos de sentimiento desde {sentiment_path}")
        df_sent = pd.read_csv(sentiment_path)
        df_sent["Date"] = pd.to_datetime(df_sent["fecha"]).dt.date
        df_features = pd.merge(df_features, df_sent, on="Date", how="left")

        if "sentiment_score" in df_features.columns:
            df_features["sentiment_score"] = df_features["sentiment_score"].fillna(0)
    else:
        context.log.warning(
            f"No se encontró archivo de sentimiento en {sentiment_path}. Se omitirá."
        )

    context.log.info("Calculando indicadores técnicos...")
    df_features.sort_values("Date", inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    df_features["Daily_Return"] = df_features["Close"].pct_change()
    df_features["SMA_10"] = df_features["Close"].rolling(window=10).mean()
    df_features["SMA_50"] = df_features["Close"].rolling(window=50).mean()
    df_features["Volatilidad_10d"] = (
        df_features["Daily_Return"].rolling(window=10).std()
    )

    df_features["target_direction"] = (
        df_features["Close"].shift(-1) > df_features["Close"]
    ).astype(int)

    filas_antes = len(df_features)
    df_features.dropna(inplace=True)
    context.log.info(f"Filas limpiadas (NaNs): {filas_antes - len(df_features)}")

    output_dir = "data/03_features"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "historical_features.csv")

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
