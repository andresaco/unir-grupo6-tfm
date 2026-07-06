from dagster import (
    Definitions,
    load_assets_from_modules,
    define_asset_job,
    AssetSelection,
    schedule,
    RunConfig,
    config_mapping,
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

# Conjunto de assets que aceptan configuración del tipo StockDownloadConfig
CONFIGURABLE_ASSETS = {
    "raw_stock_data",
    "raw_vix_data",
    "processed_stock_data",
    "processed_vix_data",
    "incremental_market_data",
    "raw_gdelt_sentiment_data",
    "processed_gdelt_sentiment_data",
    "raw_social_data",
    "raw_daily_social_data",
    "aggregated_daily_social_sentiment",
    "daily_prediction",
    "engineered_features",
    "order_book",
}


def make_shared_config(asset_names):
    """
    Crea dinámicamente un mapeador de configuración a nivel de Job
    para todos los assets que admiten StockDownloadConfig.
    """
    configurable_assets_in_job = [
        name for name in asset_names if name in CONFIGURABLE_ASSETS
    ]

    @config_mapping(
        config_schema={
            "ticker": str,
            "name": str,
            "initial_date": str,
            "end_date": str,
        }
    )
    def config_fn(val):
        config_dict = {
            "ticker": val["ticker"],
            "name": val["name"],
            "initial_date": val["initial_date"],
            "end_date": val["end_date"],
        }
        return {
            "ops": {
                asset_name: {"config": config_dict}
                for asset_name in configurable_assets_in_job
            }
        }

    return config_fn


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
    config=make_shared_config(
        [
            "raw_stock_data",
            "raw_vix_data",
            "processed_stock_data",
            "processed_vix_data",
            "incremental_market_data",
        ]
    ),
    description="Descarga de forma coordinada y paralela los datos de cotización histórica de acciones y del índice de volatilidad VIX de forma incremental.",
)

# 2. Creamos el Job específico para el pipeline social
# Ejecutará en estricto orden: raw_social_data -> processed_social_data -> social_sentiment_analysis
pre_social_pipeline_job = define_asset_job(
    name="pre_social_pipeline_job",
    selection=AssetSelection.assets(
        "raw_social_data", "processed_social_data", "social_sentiment_analysis"
    ),
    config=make_shared_config(
        [
            "raw_social_data",
            "processed_social_data",
            "social_sentiment_analysis",
        ]
    ),
    description="Ejecuta de manera secuencial y coordinada la ingesta, limpieza estructural y análisis NLP de redes sociales.",
)

# 3. Creamos el Job específico para el pipeline social de top diario
pre_social_daily_top_pipeline_job = define_asset_job(
    name="pre_social_daily_top_pipeline_job",
    selection=AssetSelection.assets(
        "raw_daily_social_data",
        "processed_daily_social_data",
        "daily_social_sentiment_analysis",
        "aggregated_daily_social_sentiment",
    ),
    config=make_shared_config(
        [
            "raw_daily_social_data",
            "processed_daily_social_data",
            "daily_social_sentiment_analysis",
            "aggregated_daily_social_sentiment",
        ]
    ),
    description="Extrae los 300 posts más importantes diarios, realiza NLP y agrega los datos por día de forma tolerante a fallos.",
)

# 4. Job específico para descargar y procesar los datos históricos de GDELT
pre_gdelt_download_job = define_asset_job(
    name="pre_gdelt_download_job",
    selection=AssetSelection.assets(
        "raw_gdelt_sentiment_data",
        "processed_gdelt_sentiment_data",
    ),
    config=make_shared_config(
        [
            "raw_gdelt_sentiment_data",
            "processed_gdelt_sentiment_data",
        ]
    ),
    description="Ejecuta de manera secuencial la descarga y el procesamiento de los datos de sentimiento histórico de GDELT.",
)


# Job específico para ejecutar el backtesting de todos los modelos
pre_backtesting_job = define_asset_job(
    name="pre_backtesting_job",
    selection=AssetSelection.assets("run_backtest"),
    description="Ejecuta la evaluación histórica comparativa sobre todos los modelos de Machine Learning registrados en MLflow.",
)


# Job para descargar y procesar datos de GDELT, stock y VIX históricos de forma unificada
pre_ticker_data = define_asset_job(
    name="pre_ticker_data",
    selection=AssetSelection.assets(
        "raw_gdelt_sentiment_data",
        "processed_gdelt_sentiment_data",
        "raw_stock_data",
        "raw_vix_data",
        "processed_stock_data",
        "processed_vix_data",
        "incremental_market_data",
    ),
    config=make_shared_config(
        [
            "raw_gdelt_sentiment_data",
            "processed_gdelt_sentiment_data",
            "raw_stock_data",
            "raw_vix_data",
            "processed_stock_data",
            "processed_vix_data",
            "incremental_market_data",
        ]
    ),
    tags={"circuit": "desarrollo"},
    description="Job unificado de desarrollo para descargar y procesar datos históricos de GDELT, stock y VIX.",
)


# Job de desarrollo para ejecutar la ingeniería de características (Feature Engineering)
pre_feature_engineering = define_asset_job(
    name="pre_feature_engineering",
    selection=AssetSelection.assets("engineered_features"),
    config=make_shared_config(["engineered_features"]),
    tags={"circuit": "desarrollo"},
    description="Job de desarrollo para calcular las características de ingeniería (features).",
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
        "order_book",
    ),
    config=make_shared_config(
        [
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
            "order_book",
        ]
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

    return RunConfig(
        config={
            "ticker": "AAPL",
            "name": "Apple",
            "initial_date": today_str,
            "end_date": today_str,
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
        pre_ticker_data,
        pre_feature_engineering,
    ],
    schedules=[
        programacion_prod_daily_order_job,
    ],
)
