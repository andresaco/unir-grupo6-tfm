from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    AssetSelection,
    schedule,
    RunConfig,
)

# Importamos nuestros módulos que contienen los assets
from .etl import social, stock, feature_engineering, social_daily, history
from .training import (
    rf_train,
    xgboost_train,
    lstm_train,
    rf_traditional_train,
    xgboost_traditional_train,
    lstm_traditional_train,
)
from .inference import predict, backtesting


# Cargamos los assets de cada módulo
ingestion_assets = load_assets_from_modules([stock])
feateng_assets = load_assets_from_modules([feature_engineering])
social_assets = load_assets_from_modules([social])
social_daily_assets = load_assets_from_modules([social_daily])
train_assets = load_assets_from_modules(
    [
        rf_train,
        xgboost_train,
        lstm_train,
        rf_traditional_train,
        xgboost_traditional_train,
        lstm_traditional_train,
    ]
)
inference_assets = load_assets_from_modules([predict, backtesting])
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
pre_market_download_job = define_asset_job(
    name="pre_market_download_job",
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
pre_social_pipeline_job = define_asset_job(
    name="pre_social_pipeline_job",
    selection=AssetSelection.assets(
        "raw_social_data", "processed_social_data", "social_sentiment_analysis"
    ),
    description="Ejecuta de manera secuencial y coordinada la ingesta, limpieza estructural y análisis NLP de redes sociales.",
)

# 2. Creamos el Job específico para el pipeline social de top diario
pre_social_daily_top_pipeline_job = define_asset_job(
    name="pre_social_daily_top_pipeline_job",
    selection=AssetSelection.assets(
        "raw_daily_social_data",
        "processed_daily_social_data",
        "daily_social_sentiment_analysis",
        "aggregated_daily_social_sentiment",
    ),
    description="Extrae los 300 posts más importantes diarios, realiza NLP y agrega los datos por día de forma tolerante a fallos.",
)

# 3. Job específico para descargar y procesar los datos históricos de GDELT
pre_gdelt_download_job = define_asset_job(
    name="pre_gdelt_download_job",
    selection=AssetSelection.assets(
        "raw_gdelt_sentiment_data",
        "processed_gdelt_sentiment_data",
    ),
    description="Ejecuta de manera secuencial la descarga y el procesamiento de los datos de sentimiento histórico de GDELT.",
)


# Job específico para ejecutar el backtesting de todos los modelos
pre_backtesting_job = define_asset_job(
    name="pre_backtesting_job",
    selection=AssetSelection.assets("run_backtest"),
    description="Ejecuta la evaluación histórica comparativa sobre todos los modelos de Machine Learning registrados en MLflow.",
)


# Job para el pipeline de producción diario que integra mercado, redes sociales y predicción
prod_daily_order_job = define_asset_job(
    name="prod_daily_order_job",
    selection=AssetSelection.assets(
        "raw_stock_data",
        "raw_vix_data",
        "processed_stock_data",
        "processed_vix_data",
        "incremental_market_data",
        "raw_daily_social_data",
        "processed_daily_social_data",
        "daily_social_sentiment_analysis",
        "aggregated_daily_social_sentiment",
        "daily_prediction",
    ),
    tags={"type": "producción"},
    description="Ejecuta el pipeline de producción diario: descarga datos de mercado, descarga y analiza sentimientos de Bluesky, selecciona el mejor modelo de MLflow y genera predicciones.",
)


# Job para entrenar los modelos tradicionales (sin sentimientos)
test_training_no_sentiment_job = define_asset_job(
    name="test_training_no_sentiment_job",
    selection=AssetSelection.assets(
        "rf_traditional_training",
        "xgboost_traditional_training",
        "lstm_traditional_training",
    ),
    tags={"type": "test"},
    description="Entrena los modelos RandomForest, XGBoost y LSTM tradicionales que descartan datos de redes sociales.",
)


@schedule(
    cron_schedule="0 23 * * 1-5",  # De lunes a viernes a las 23:00 (cierre de mercado)
    job=prod_daily_order_job,
    name="prod_daily_order_schedule",
    description="Ejecuta diariamente el pipeline de producción para AAPL con la fecha actual.",
)
def programacion_prod_daily_order_job(context):
    """
    Planificación diaria para el pipeline de producción. Genera la predicción
    para el día de hoy pasando la fecha actual como parámetro de configuración.
    """
    import datetime

    # Obtenemos la fecha actual en formato YYYY-MM-DD
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    config_dict = {
        "ticker": "AAPL",
        "name": "Apple",
        "initial_date": today_str,
        "end_date": today_str,
    }

    return RunConfig(
        ops={
            "raw_stock_data": {"config": config_dict},
            "raw_vix_data": {"config": config_dict},
            "processed_stock_data": {"config": config_dict},
            "processed_vix_data": {"config": config_dict},
            "incremental_market_data": {"config": config_dict},
            "raw_daily_social_data": {"config": config_dict},
            "aggregated_daily_social_sentiment": {"config": config_dict},
            "daily_prediction": {"config": config_dict},
        }
    )


# Definimos el repositorio global
defs = Definitions(
    assets=all_assets,
    jobs=[
        pre_social_pipeline_job,
        pre_market_download_job,
        pre_social_daily_top_pipeline_job,
        pre_gdelt_download_job,
        pre_backtesting_job,
        test_training_no_sentiment_job,
        prod_daily_order_job,
    ],
    schedules=[
        programacion_prod_daily_order_job,
    ],
)
