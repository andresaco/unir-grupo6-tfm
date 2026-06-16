import os
import pandas as pd
import yfinance as yf
from dagster import (
    asset,
    Config,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
)


# 1. Definimos la configuración parametrizable
class StockDownloadConfig(Config):
    ticker: str = "AAPL"
    initial_date: str = "2020-01-01"  # Formato YYYY-MM-DD
    end_date: str = "2023-12-31"  # Formato YYYY-MM-DD


@asset(
    group_name="ingestion",
    description="Descarga datos históricos de cotización de un Ticker desde Yahoo Finance.",
)
def raw_stock_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset que descarga datos de mercado basados en parámetros de configuración
    y los persiste en formato CSV en la capa raw.
    """
    ticker = config.ticker
    start_date = config.initial_date
    end_date = config.end_date

    context.log.info(
        f"Iniciando descarga de datos para {ticker} desde {start_date} hasta {end_date}..."
    )

    # 2. Descarga de datos usando yfinance
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(
            f"No se pudieron descargar datos para el ticker {ticker} en el rango {start_date} a {end_date}."
        )

    # Aplanar el MultiIndex que a veces devuelve yfinance y convertir el índice 'Date' en columna
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)

    # 3. Creación dinámica del directorio si no existe
    output_dir = "data/01_raw/stock"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Formateo de la ruta de salida solicitada
    filename = f"{ticker}_{start_date}_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)

    # 5. Persistencia del CSV
    df.to_csv(filepath, index=False)

    context.log.info(f"Datos descargados exitosamente y guardados en: {filepath}")

    # 6. Devolver el resultado materializado con metadatos para la UI de Dagster
    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "ticker": MetadataValue.text(ticker),
            "start_date": MetadataValue.text(start_date),
            "end_date": MetadataValue.text(end_date),
            "total_rows": MetadataValue.int(len(df)),
            "columns": MetadataValue.text(str(list(df.columns))),
        }
    )


@asset(
    group_name="ingestion",
    description="Descarga datos históricos del índice de volatilidad VIX desde Yahoo Finance.",
)
def raw_vix_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset que descarga los datos del VIX (^VIX) utilizando las mismas fechas
    configuradas para la descarga de cotizaciones.
    """
    ticker_vix = "^VIX"
    start_date = config.initial_date
    end_date = config.end_date

    context.log.info(
        f"Iniciando descarga de datos para {ticker_vix} desde {start_date} hasta {end_date}..."
    )

    # Descarga de datos usando yfinance
    df_vix = yf.download(ticker_vix, start=start_date, end=end_date, progress=False)

    if df_vix.empty:
        raise ValueError(
            f"No se pudieron descargar datos para el VIX en el rango {start_date} a {end_date}."
        )

    # Aplanar el MultiIndex y convertir el índice 'Date' en columna (igual que el notebook)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(1)
    df_vix.reset_index(inplace=True)

    # Creación dinámica del directorio si no existe
    output_dir = "data/01_raw/stock"
    os.makedirs(output_dir, exist_ok=True)

    # Formateo de la ruta de salida (Reemplazamos el ^ para que el nombre de archivo sea limpio)
    filename = f"VIX_{start_date}_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)

    # Persistencia del CSV
    df_vix.to_csv(filepath, index=False)

    context.log.info(
        f"Datos del VIX descargados exitosamente y guardados en: {filepath}"
    )

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "ticker": MetadataValue.text(ticker_vix),
            "start_date": MetadataValue.text(start_date),
            "end_date": MetadataValue.text(end_date),
            "total_rows": MetadataValue.int(len(df_vix)),
        }
    )
