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

    output_dir = "data/01_raw/stock"
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

    output_dir = "data/01_raw/stock"
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

    raw_path = f"data/01_raw/stock/{ticker}_{start_date}_{end_date}.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo raw de stock en: {raw_path}"
        )

    context.log.info(f"Cargando archivo raw de cotización desde {raw_path}...")
    df = pd.read_csv(raw_path)

    # Transformación: renombrar columnas a minúsculas
    df.columns = [col.lower() for col in df.columns]

    # Guardar en data/02_processed/stock/ con el mismo nombre que el archivo raw original
    output_dir = "data/02_processed/stock"
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

    raw_path = f"data/01_raw/stock/VIX_{start_date}_{end_date}.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo raw del VIX en: {raw_path}")

    context.log.info(f"Cargando archivo raw de VIX desde {raw_path}...")
    df_vix = pd.read_csv(raw_path)

    # Transformación: renombrar columnas a minúsculas
    df_vix.columns = [col.lower() for col in df_vix.columns]

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
