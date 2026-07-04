from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    AssetSelection,
)

# Importamos nuestros módulos que contienen los assets
from .etl import social, stock, feature_engineering, social_daily, history
from .training import rf_train, xgboost_train, lstm_train
from .inference import predict


# Cargamos los assets de cada módulo
ingestion_assets = load_assets_from_modules([stock])
feateng_assets = load_assets_from_modules([feature_engineering])
social_assets = load_assets_from_modules([social])
social_daily_assets = load_assets_from_modules([social_daily])
train_assets = load_assets_from_modules([rf_train, xgboost_train, lstm_train])
inference_assets = load_assets_from_modules([predict])
history_assets = load_assets_from_modules([history])

# Combinamos todos los assets
all_assets = (
    ingestion_assets
    + feateng_assets
    + social_assets
    + social_daily_assets
    + train_assets
    + inference_assets
    + history_assets
)

# 1. Job para descargar datos de mercado coordinadamente (Stock + VIX)
market_download_job = define_asset_job(
    name="download_stock_and_vix_job",
    selection=AssetSelection.assets(
        "raw_stock_data",
        "raw_vix_data",
        "processed_stock_data",
        "processed_vix_data",
        "incremental_market_data",
    ),
    description="Descarga de forma coordinada y paralela los datos de cotización histórica de acciones y del índice de volatilidad VIX de forma incremental.",
)

# 1. Creamos el Job específico para el pipeline social
# Ejecutará en estricto orden: raw_social_data -> processed_social_data -> social_sentiment_analysis
social_pipeline_job = define_asset_job(
    name="run_social_sentiment_pipeline",
    selection=AssetSelection.assets(
        "raw_social_data", "processed_social_data", "social_sentiment_analysis"
    ),
    description="Ejecuta de manera secuencial y coordinada la ingesta, limpieza estructural y análisis NLP de redes sociales.",
)

# 2. Creamos el Job específico para el pipeline social de top diario
social_daily_top_pipeline_job = define_asset_job(
    name="run_social_daily_top_pipeline",
    selection=AssetSelection.assets(
        "raw_daily_social_data",
        "processed_daily_social_data",
        "daily_social_sentiment_analysis",
        "aggregated_daily_social_sentiment",
    ),
    description="Extrae los 300 posts más importantes diarios, realiza NLP y agrega los datos por día de forma tolerante a fallos.",
)

# 3. Job específico para descargar y procesar los datos históricos de GDELT
gdelt_download_job = define_asset_job(
    name="download_gdelt",
    selection=AssetSelection.assets(
        "raw_gdelt_sentiment_data",
        "processed_gdelt_sentiment_data",
    ),
    description="Ejecuta de manera secuencial la descarga y el procesamiento de los datos de sentimiento histórico de GDELT.",
)


# Definimos el repositorio global
defs = Definitions(
    assets=all_assets,
    jobs=[
        social_pipeline_job,
        market_download_job,
        social_daily_top_pipeline_job,
        gdelt_download_job,
    ],
)
