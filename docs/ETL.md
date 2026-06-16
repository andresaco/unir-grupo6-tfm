# Documentación del Pipeline ETL

## Objetivo

Este pipeline ejecuta:
1. Extracción de tweets desde X.com (antes Twitter) mediante Playwright.
2. Análisis de sentimientos sobre el texto de los tweets usando `transformers`.
3. Persistencia del dataset enriquecido con sentimiento en un CSV dentro de `datos/`.

## Archivo principal

- `src/etl.py`

Contiene un flujo de Prefect con tres tareas principales:
- `extract_tweets`: extrae tweets desde X.com.
- `analyze_sentiment`: clasifica cada texto con un modelo de Hugging Face.
- `save_results`: guarda el DataFrame en un archivo CSV con timestamp.

## Dependencias

El proyecto usa estas librerías principales:
- `prefect`
- `playwright`
- `transformers`
- `torch`
- `pandas`
- `python-dotenv`

Estas dependencias se declaran en `pyproject.toml`.

## Ejecución

Desde la raíz del proyecto, activa el entorno virtual y ejecuta:

```bash
.venv/bin/python src/etl.py --query "AAPL;Apple" --count 100 --output-dir datos --headless
```

### Argumentos disponibles

- `--query`: consulta de búsqueda en X.com. Ejemplo: `"AAPL;Apple"`.
- `--count`: número máximo de tweets a capturar.
- `--profile-dir`: directorio de perfil persistente de Playwright.
- `--model`: modelo de `transformers` para análisis de sentimiento.
- `--output-dir`: carpeta donde se guardará el CSV de salida.
- `--headless`: ejecutar Playwright en modo headless.

## Variables de entorno opcionales

El pipeline puede usar variables de entorno definidas en un archivo `.env`:

- `SOCIAL_QUERY`: consulta por defecto para extracción de tweets.
- `SOCIAL_TWEET_COUNT`: número de tweets a extraer por defecto.
- `PLAYWRIGHT_USER_DATA_DIR`: directorio del perfil de Playwright.
- `SENTIMENT_MODEL`: modelo por defecto de análisis de sentimiento.
- `ETL_OUTPUT_DIR`: directorio de salida por defecto.
- `PLAYWRIGHT_HEADLESS`: `True` o `False` para el modo headless.

## Resultado

El pipeline genera un archivo CSV en `datos/` con estas columnas mínimas:
- `ID_Tweet`
- `Fecha_UTC`
- `Contenido_Texto`
- `Retweets`
- `Favoritos`
- `Sentimiento`
- `Puntuacion_Sentimiento`

El nombre del archivo incluye un timestamp para evitar sobrescribir datos.

## Notas de uso

- En la primera ejecución, Playwright puede requerir autenticación manual en X.com.
- Si el login se muestra, el flujo esperará hasta que aparezcan tweets.
- El modelo de sentimiento usado por defecto es `cardiffnlp/twitter-roberta-base-sentiment-latest`.

# Gestión de credenciales

## Bluesky

- Crea cuenta en Bluesky (https://bsky.app)
- Accede a [https://bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)
- Crea una password nueva
- En el fichero .env almacena las credenciales BLUESKY_HANDLE con tu nombre de usuario y BLUESKY_PASSWORD con la que acabas de generar.