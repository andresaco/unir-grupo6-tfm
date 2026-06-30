import os
import pandas as pd
import yfinance as yf
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
)

from .core.config import StockDownloadConfig
from ..schemas import validate_df, StockRawRow, StockProcessedRow


OUTPUT_DIR = "stock"


@asset(
    group_name="ingestion",
    description="Descarga datos históricos de cotización de un Ticker desde Yahoo Finance.",
)
def raw_stock_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    context.log.info(
        f"Iniciando descarga de datos para {ticker} desde {start_date} hasta {end_date}..."
    )

    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(
            f"No se pudieron descargar datos para el ticker {ticker} en el rango {start_date} a {end_date}."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)

    # Validar antes de guardar
    validate_df(df, StockRawRow, stage="raw_stock_data (write)")

    output_dir = f"data/01_raw/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{ticker}_{start_date}_{end_date}.csv")
    df.to_csv(filepath, index=False)

    context.log.info(f"Datos guardados en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "ticker": MetadataValue.text(ticker),
            "total_rows": MetadataValue.int(len(df)),
        }
    )


@asset(
    group_name="ingestion",
    description="Descarga datos históricos del índice de volatilidad VIX desde Yahoo Finance.",
)
def raw_vix_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker_vix = "^VIX"
    start_date = config.initial_date
    end_date = config.end_date

    context.log.info(
        f"Iniciando descarga de datos para {ticker_vix} desde {start_date} hasta {end_date}..."
    )

    df_vix = yf.download(ticker_vix, start=start_date, end=end_date, progress=False)

    if df_vix.empty:
        raise ValueError(
            f"No se pudieron descargar datos para el VIX en el rango {start_date} a {end_date}."
        )

    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(1)
    df_vix.reset_index(inplace=True)

    # Validar antes de guardar
    validate_df(df_vix, StockRawRow, stage="raw_vix_data (write)")

    output_dir = f"data/01_raw/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"VIX_{start_date}_{end_date}.csv")
    df_vix.to_csv(filepath, index=False)

    context.log.info(f"Datos guardados en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "total_rows": MetadataValue.int(len(df_vix)),
        }
    )


@asset(
    deps=[raw_stock_data],
    group_name="processing",
    description="Estandariza los nombres de las columnas de stock a minúsculas en la capa Silver.",
)
def processed_stock_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset de procesamiento (Silver). Toma el archivo de cotización descargado en Raw,
    limpia y renombra las columnas a minúsculas, y lo almacena en la capa Processed.
    """
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    raw_path = f"data/01_raw/{OUTPUT_DIR}/{ticker}_{start_date}_{end_date}.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo raw de stock en: {raw_path}"
        )

    context.log.info(f"Cargando archivo raw de cotización desde {raw_path}...")
    df = pd.read_csv(raw_path)

    # Validar al leer raw
    validate_df(df, StockRawRow, stage="processed_stock_data (read raw)")

    # Transformación: renombrar columnas a minúsculas
    df.columns = [col.lower() for col in df.columns]

    # Validar procesado antes de guardar
    validate_df(df, StockProcessedRow, stage="processed_stock_data (write processed)")

    # Guardar en data/02_processed/stock/ con el mismo nombre que el archivo raw original
    output_dir = f"data/02_processed/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{ticker}_{start_date}_{end_date}.csv")
    df.to_csv(filepath, index=False)

    context.log.info(f"Datos de stock estandarizados (Silver) guardados en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath_silver": MetadataValue.path(filepath),
            "total_rows": MetadataValue.int(len(df)),
            "columns_processed": MetadataValue.text(str(list(df.columns))),
        }
    )


@asset(
    deps=[raw_vix_data],
    group_name="processing",
    description="Estandariza los nombres de las columnas del VIX a minúsculas en la capa Silver.",
)
def processed_vix_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset de procesamiento (Silver). Toma el archivo VIX descargado en Raw,
    limpia y renombra las columnas a minúsculas, y lo almacena en la capa Processed.
    """
    start_date = config.initial_date
    end_date = config.end_date

    raw_path = f"data/01_raw/{OUTPUT_DIR}/VIX_{start_date}_{end_date}.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo raw del VIX en: {raw_path}")

    context.log.info(f"Cargando archivo raw de VIX desde {raw_path}...")
    df_vix = pd.read_csv(raw_path)

    # Validar al leer raw
    validate_df(df_vix, StockRawRow, stage="processed_vix_data (read raw)")

    # Transformación: renombrar columnas a minúsculas
    df_vix.columns = [col.lower() for col in df_vix.columns]

    # Validar procesado antes de guardar
    validate_df(df_vix, StockProcessedRow, stage="processed_vix_data (write processed)")

    # Guardar en data/02_processed/stock/ con el mismo nombre que el archivo raw original
    output_dir = "data/02_processed/stock"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"VIX_{start_date}_{end_date}.csv")
    df_vix.to_csv(filepath, index=False)

    context.log.info(f"Datos de VIX estandarizados (Silver) guardados en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath_silver": MetadataValue.path(filepath),
            "total_rows": MetadataValue.int(len(df_vix)),
            "columns_processed": MetadataValue.text(str(list(df_vix.columns))),
        }
    )


@asset(
    deps=[processed_stock_data, processed_vix_data],
    group_name="feature_engineering",
    description="Consolida de manera incremental los datos de stock y VIX en data/03_features, evitando duplicados por fecha.",
)
def incremental_market_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    # Rutas de origen
    processed_stock_path = (
        f"data/02_processed/{OUTPUT_DIR}/{ticker}_{start_date}_{end_date}.csv"
    )
    processed_vix_path = (
        f"data/02_processed/{OUTPUT_DIR}/VIX_{start_date}_{end_date}.csv"
    )

    # Rutas de destino en 03_features
    output_dir = "data/03_features"
    os.makedirs(output_dir, exist_ok=True)

    stock_output_dir = f"{output_dir}/{OUTPUT_DIR}"
    stock_output_path = os.path.join(stock_output_dir, f"{ticker}.csv")
    vix_output_dir = f"{output_dir}/{OUTPUT_DIR}"
    vix_output_path = os.path.join(vix_output_dir, "VIX.csv")

    context.log.info(
        f"Cargando nuevos datos procesados desde {processed_stock_path} y {processed_vix_path}..."
    )
    df_new_stock = pd.read_csv(processed_stock_path)
    df_new_vix = pd.read_csv(processed_vix_path)

    # Validar tipos
    validate_df(
        df_new_stock,
        StockProcessedRow,
        stage="incremental_market_data (read processed stock)",
    )
    validate_df(
        df_new_vix,
        StockProcessedRow,
        stage="incremental_market_data (read processed vix)",
    )

    # Procesar Stock
    if os.path.exists(stock_output_path):
        context.log.info(
            f"El archivo existente de stock {stock_output_path} ya existe. Haciendo append..."
        )
        df_old_stock = pd.read_csv(stock_output_path)
        df_stock_combined = pd.concat([df_old_stock, df_new_stock], ignore_index=True)
    else:
        context.log.info(f"Creando nuevo archivo de stock en {stock_output_path}...")
        df_stock_combined = df_new_stock

    # Convertir fecha a string para consistencia, eliminar duplicados y ordenar
    df_stock_combined["date"] = df_stock_combined["date"].astype(str)
    df_stock_combined = df_stock_combined.drop_duplicates(subset=["date"], keep="last")
    df_stock_combined = df_stock_combined.sort_values(by="date").reset_index(drop=True)
    df_stock_combined.to_csv(stock_output_path, index=False)

    # Procesar VIX
    if os.path.exists(vix_output_path):
        context.log.info(
            f"El archivo existente de VIX {vix_output_path} ya existe. Haciendo append..."
        )
        df_old_vix = pd.read_csv(vix_output_path)
        df_vix_combined = pd.concat([df_old_vix, df_new_vix], ignore_index=True)
    else:
        context.log.info(f"Creando nuevo archivo de VIX en {vix_output_path}...")
        df_vix_combined = df_new_vix

    df_vix_combined["date"] = df_vix_combined["date"].astype(str)
    df_vix_combined = df_vix_combined.drop_duplicates(subset=["date"], keep="last")
    df_vix_combined = df_vix_combined.sort_values(by="date").reset_index(drop=True)
    df_vix_combined.to_csv(vix_output_path, index=False)

    context.log.info("Datos de stock y VIX consolidados e incrementados exitosamente.")

    return MaterializeResult(
        metadata={
            "stock_filepath": MetadataValue.path(stock_output_path),
            "stock_total_rows": MetadataValue.int(len(df_stock_combined)),
            "vix_filepath": MetadataValue.path(vix_output_path),
            "vix_total_rows": MetadataValue.int(len(df_vix_combined)),
        }
    )
