from dagster import Definitions, load_assets_from_modules

# Importamos nuestros módulos que contienen los assets
from .etl import social, stock, feature_engineering
from .training import train


# Cargamos los assets de cada módulo
ingestion_assets = load_assets_from_modules([stock])
feateng_assets = load_assets_from_modules([feature_engineering])
social_assets = load_assets_from_modules([social])
train_assets = load_assets_from_modules([train])

# Combinamos todos los assets
all_assets = ingestion_assets + feateng_assets + social_assets + train_assets

# Definimos el repositorio global
defs = Definitions(
    assets=all_assets,
    # Aquí en el futuro puedes añadir jobs, schedules (cron), sensors, etc.
)
