from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    AssetSelection,
)

# Importamos nuestros módulos que contienen los assets
from .etl import social, stock, feature_engineering
from .training import train
from .inference import predict


# Cargamos los assets de cada módulo
ingestion_assets = load_assets_from_modules([stock])
feateng_assets = load_assets_from_modules([feature_engineering])
social_assets = load_assets_from_modules([social])
train_assets = load_assets_from_modules([train])
inference_assets = load_assets_from_modules([predict])

# Combinamos todos los assets
all_assets = (
    ingestion_assets + feateng_assets + social_assets + train_assets + inference_assets
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

# Definimos el repositorio global
defs = Definitions(assets=all_assets, jobs=[social_pipeline_job])
