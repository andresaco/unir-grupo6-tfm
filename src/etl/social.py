import os
import datetime
import asyncio
import pandas as pd
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from dotenv import load_dotenv
from transformers import pipeline

from .core.config import (
    StockDownloadConfig,
)  # Configuración parametrizable para el pipeline social
from .core.social.bsky import BlueskyClient  # Cliente de Bluesky del proyecto
from ..schemas import validate_df, SocialRawRow, SocialProcessedRow, SocialSentimentRow

# Cargar variables de entorno locales (.env)
load_dotenv()

# Configuración por defecto para el pipeline de NLP
DEFAULT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


@asset(
    group_name="ingestion",
    description="Descarga publicaciones de Twitter y Bluesky de forma inmutable (capa Bronze).",
)
def raw_social_data(context: AssetExecutionContext, config: StockDownloadConfig) -> str:
    """
    Asset de ingesta (Bronze). Obtiene datos de redes sociales (Twitter/Bluesky),
    persiste el dataset granular original (por publicación) con un timestamp dinámico
    y retorna la ruta del archivo generado para los assets descendientes.
    """
    start_date = config.initial_date
    end_date = config.end_date
    ticker = config.ticker

    context.log.info(
        f"Iniciando extracción de datos de redes sociales para {config.query} ({start_date} a {end_date})...."
    )

    # 1. Inicializar DataFrame vacío
    df_posts = pd.DataFrame()

    try:
        # Simplificación de la lógica para el ejemplo:
        client = BlueskyClient()
        # Al ser search_posts asíncrono, lo envolvemos con asyncio.run para ejecutarlo síncronamente en Dagster
        posts = asyncio.run(client.search_posts(query=config.n, target_tweets=300))

        # Si la API de Bluesky no devuelve resultados, forzamos el uso del fallback estructurado
        if not posts or len(posts) == 0:
            raise ValueError(
                "La API de Bluesky no retornó ninguna publicación para este Ticker."
            )

        normalized_posts = []
        for post in posts:
            normalized_posts.append(
                {
                    "ID_Tweet": post.get("id", ""),
                    "Fecha_UTC": post.get("published_time", ""),
                    "Contenido_Texto": post.get("text", ""),
                    "Retweets": post.get("reposts", "0"),
                    "Favoritos": post.get("likes", "0"),
                }
            )

        df_posts = pd.DataFrame(normalized_posts)
        context.log.info(
            f"Se obtuvieron exitosamente {len(df_posts)} publicaciones desde la API real."
        )
    except Exception as e:
        context.log.warning(
            f"Error al extraer de redes sociales o sin resultados: {e}. Se procederá con datos simulados estructurados."
        )
        # Fallback: Creación de dataframe dummy con la estructura de columnas exacta del formato raw real
        raise

    # Aseguramos que df_posts sea un DataFrame válido de Pandas antes de continuar
    if not isinstance(df_posts, pd.DataFrame):
        df_posts = pd.DataFrame(df_posts)

    # Validar datos crudos antes de escribir
    validate_df(df_posts, SocialRawRow, stage="raw_social_data (write)")

    # 2. Persistir los datos crudos granulares (Bronze)
    output_dir = "data/01_raw/social"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ticker = str(ticker).replace(" ", "_")
    filename = f"social_{safe_ticker}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df_posts.to_csv(filepath, index=False)

    context.log.info(f"Datos de sentimiento granular guardados en: {filepath}")

    # Registramos los metadatos en la UI de Dagster
    context.add_output_metadata(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "start_date": MetadataValue.text(start_date),
            "end_date": MetadataValue.text(end_date),
            "total_records_processed": MetadataValue.int(len(df_posts)),
        }
    )

    # Retornamos el string con la ruta física para que el siguiente asset la consuma dinámicamente
    return filepath


@asset(
    group_name="processing",
    description="Limpia, valida esquemas y estandariza las columnas y tipos de datos (capa Silver).",
)
def processed_social_data(context: AssetExecutionContext, raw_social_data: str) -> str:
    """
    Asset de procesamiento (Silver - Estructural). Lee los datos de la capa Raw,
    aplica las transformaciones de limpieza estructural, normaliza tipos y retorna
    la ruta del archivo estructurado limpio.
    """
    raw_path = raw_social_data

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el fichero Raw de redes sociales en: {raw_path}"
        )

    context.log.info(f"Leyendo datos de redes sociales desde la capa Raw: {raw_path}")
    df = pd.read_csv(raw_path)

    # Validar al leer raw
    validate_df(df, SocialRawRow, stage="processed_social_data (read raw)")

    # 1. Normalización de Nombres de Columnas (Estandarización a snake_case)
    mapping_columnas = {
        "ID_Tweet": "tweet_id",
        "Fecha_UTC": "fecha_utc",
        "Contenido_Texto": "contenido_texto",
        "Retweets": "retweets",
        "Favoritos": "favoritos",
    }
    df = df.rename(columns=mapping_columnas)

    # Defensa del esquema estructural básico
    columnas_esperadas = {
        "tweet_id": "object",
        "fecha_utc": "object",
        "contenido_texto": "object",
        "retweets": "int64",
        "favoritos": "int64",
    }
    for col, dtype in columnas_esperadas.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)

    # 2. Control de Calidad del Dato (Data Quality) y Limpieza estructural
    df = df.dropna(subset=["tweet_id", "fecha_utc", "contenido_texto"])

    # Forzamos la estandarización de zonas horarias usando utc=True
    df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["fecha_utc"])

    # Extraemos una columna limpia solo con la fecha (sin horas/timezones)
    df["fecha_limpia"] = df["fecha_utc"].dt.date

    # Forzar tipos de datos correctos de forma segura usando series existentes
    df["retweets"] = df["retweets"].fillna(0).astype(int)
    df["favoritos"] = df["favoritos"].fillna(0).astype(int)
    df["contenido_texto"] = df["contenido_texto"].astype(str)

    # Validar procesado antes de escribir
    validate_df(df, SocialProcessedRow, stage="processed_social_data (write processed)")

    # 3. Guardar estado intermedio limpio en un CSV temporal de procesamiento
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/02_processed/social/temp"
    os.makedirs(output_dir, exist_ok=True)
    filepath_cleaned = os.path.join(output_dir, f"social_cleaned_{timestamp}.csv")
    df.to_csv(filepath_cleaned, index=False)

    context.log.info(f"Datos limpios estructuralmente guardados en: {filepath_cleaned}")
    return filepath_cleaned


@asset(
    group_name="processing",
    description="Aplica el análisis de sentimiento por NLP (Hugging Face RoBERTa) sobre los textos y persiste en Parquet particionado.",
)
def social_sentiment_analysis(
    context: AssetExecutionContext, processed_social_data: str
) -> MaterializeResult:
    """
    Asset de inferencia de NLP (Silver - Enriquecimiento). Toma las publicaciones estructuradas limpias,
    calcula la puntuación y etiqueta de sentimiento para cada una usando un pipeline de Transformers
    y las guarda particionadas por fecha.
    """
    cleaned_path = processed_social_data

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(
            f"No se encontró el archivo base limpio en: {cleaned_path}"
        )

    context.log.info(
        f"Cargando publicaciones para realizar inferencia NLP de sentimiento desde {cleaned_path}..."
    )
    df = pd.read_csv(cleaned_path)

    # Validar al leer procesado
    validate_df(
        df, SocialProcessedRow, stage="social_sentiment_analysis (read processed)"
    )

    # Si por algún motivo el DataFrame resultante está vacío, resolvemos con esquema vacío seguro
    if df.empty:
        context.log.warning(
            "El dataset de publicaciones está vacío. Saltando cálculo NLP."
        )
        df["sentimiento"] = pd.Series(dtype=str)
        df["puntuacion_sentimiento"] = pd.Series(dtype=float)
    else:
        # 1. Cargar el pipeline de NLP de Hugging Face
        context.log.info(
            f"Inicializando el pipeline de Hugging Face con el modelo: {DEFAULT_MODEL}"
        )
        classifier = pipeline(
            "sentiment-analysis",
            model=DEFAULT_MODEL,
            truncation=True,
            device=-1,  # Utilizar CPU de forma predeterminada y segura en entornos locales
        )

        # 2. Preparar textos y ejecutar inferencia por lotes
        texts = df["contenido_texto"].fillna("").astype(str).tolist()
        context.log.info(
            f"Ejecutando inferencia de sentimiento para {len(df)} publicaciones..."
        )
        results = classifier(texts)

        # 3. Extraer etiquetas brutas
        df["sentimiento_label"] = [res["label"] for res in results]
        df["sentimiento_score"] = [res["score"] for res in results]

        # 4. Normalizar a una escala analítica continua de [0, 1] (0.0=Bajista, 0.5=Neutro, 1.0=Alcista)
        # Esto unifica las métricas categóricas de RoBERTa para que sean fácilmente consumibles por el Random Forest.
        puntuaciones_continuas = []
        for label, score in zip(df["sentimiento_label"], df["sentimiento_score"]):
            label_lower = label.lower()
            if "pos" in label_lower or "label_2" in label_lower:
                # Estandariza la probabilidad positiva al intervalo [0.5, 1.0]
                val = 0.5 + (score * 0.5)
            elif "neg" in label_lower or "label_0" in label_lower:
                # Estandariza la probabilidad negativa al intervalo [0.0, 0.5]
                val = 0.5 - (score * 0.5)
            else:
                # Sentimiento neutro o intermedio
                val = 0.5
            puntuaciones_continuas.append(round(val, 4))

        df["puntuacion_sentimiento"] = puntuaciones_continuas
        df["sentimiento"] = df["sentimiento_label"].apply(
            lambda sentiment_label: (
                "positive"
                if "pos" in sentiment_label.lower()
                or "label_2" in sentiment_label.lower()
                else (
                    "negative"
                    if "neg" in sentiment_label.lower()
                    or "label_0" in sentiment_label.lower()
                    else "neutral"
                )
            )
        )

        # Eliminamos las columnas temporales del cálculo de NLP
        df = df.drop(columns=["sentimiento_label", "sentimiento_score"])

    # 5. Creación de las columnas de particionado basadas en la fecha
    df["fecha_utc"] = pd.to_datetime(
        df["fecha_utc"], format="ISO8601", errors="coerce", utc=True
    )

    # Validar sentiment final antes de escribir
    validate_df(
        df, SocialSentimentRow, stage="social_sentiment_analysis (write sentiment)"
    )

    # 6. Persistencia en Parquet particionado (Capa Silver Final)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/02_processed/social/sentiment"
    filepath_cleaned = os.path.join(output_dir, f"social_sentiment_{timestamp}.csv")
    os.makedirs(output_dir, exist_ok=True)

    context.log.info(
        f"Guardando Parquets particionados y enriquecidos con NLP en: {output_dir}"
    )
    df.to_csv(filepath_cleaned, index=False)

    promedio_sentimiento = (
        float(df["puntuacion_sentimiento"].mean()) if not df.empty else 0.5
    )

    return MaterializeResult(
        metadata={
            "fichero_procesado": MetadataValue.path(filepath_cleaned),
            "registros_analizados": MetadataValue.int(len(df)),
            "sentimiento_promedio_nlp": MetadataValue.float(promedio_sentimiento),
            "columnas_finales": MetadataValue.text(str(list(df.columns))),
        }
    )
