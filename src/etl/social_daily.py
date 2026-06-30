import os
import datetime
import asyncio
import glob
import pandas as pd
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from transformers import pipeline

from .core.config import StockDownloadConfig
from .core.social.bsky import BlueskyClient
from ..schemas import (
    validate_df,
    SocialRawRow,
    SocialProcessedRow,
    SocialSentimentRow,
    SocialAggregatedRow,
)

DEFAULT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)


@asset(
    group_name="ingestion_daily_top",
    description="Descarga los 300 posts más importantes POR DÍA de Bluesky aplicando el esquema estructural original.",
)
def raw_daily_social_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> str:
    """
    Ingesta que itera día a día en el rango de configuración buscando los top 300 posts.
    Implementa guardado diario (checkpointing) para tolerancia a fallos.
    """
    ticker_name = config.name
    start_date = datetime.datetime.strptime(config.initial_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config.end_date, "%Y-%m-%d")

    context.log.info(
        f"Iniciando extracción DIARIA de top posts para {ticker_name} desde {start_date.date()} hasta {end_date.date()}..."
    )

    client = BlueskyClient()

    output_dir = "data/01_raw/social_daily_top"
    safe_ticker = str(config.ticker).replace(" ", "_")
    chunks_dir = os.path.join(output_dir, f"{safe_ticker}_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + datetime.timedelta(days=1)

        chunk_filename = f"{current_date.strftime('%Y-%m-%d')}.csv"
        chunk_filepath = os.path.join(chunks_dir, chunk_filename)

        if os.path.exists(chunk_filepath):
            context.log.info(
                f"-> [Checkpoint] Día {current_date.date()} ya extraído. Saltando..."
            )
            current_date = next_date
            continue

        str_since = current_date.strftime("%Y-%m-%dT00:00:00Z")
        str_until = next_date.strftime("%Y-%m-%dT00:00:00Z")
        day_query = f"{ticker_name} since:{str_since} until:{str_until}"

        context.log.info(
            f"-> Extrayendo Top 300 posts para el día {current_date.date()}..."
        )

        try:
            posts = asyncio.run(
                client.search_posts(query=day_query, target_tweets=300, sort_type="top")
            )

            if posts:
                daily_normalized_posts = []
                for post in posts:
                    daily_normalized_posts.append(
                        {
                            "ID_Tweet": post.get("id", ""),
                            "Fecha_UTC": post.get("published_time", ""),
                            "Contenido_Texto": post.get("text", ""),
                            "Retweets": post.get("reposts", "0"),
                            "Favoritos": post.get("likes", "0"),
                        }
                    )

                df_daily = pd.DataFrame(daily_normalized_posts)
                # Validar chunk antes de guardar
                validate_df(
                    df_daily, SocialRawRow, stage="raw_daily_social_data (write chunk)"
                )
                df_daily.to_csv(chunk_filepath, index=False)
                context.log.info(
                    f"   Obtenidos y guardados {len(posts)} posts para {current_date.date()}."
                )
            else:
                context.log.warning(
                    f"   Sin resultados para el día {current_date.date()}."
                )
                df_empty = pd.DataFrame(
                    columns=[
                        "ID_Tweet",
                        "Fecha_UTC",
                        "Contenido_Texto",
                        "Retweets",
                        "Favoritos",
                    ]
                )
                validate_df(
                    df_empty,
                    SocialRawRow,
                    stage="raw_daily_social_data (write empty chunk)",
                )
                df_empty.to_csv(chunk_filepath, index=False)

        except Exception as e:
            context.log.error(
                f"   Error en la llamada de Bluesky para el día {current_date.date()}: {e}"
            )

        current_date = next_date

    context.log.info(
        "Consolidando todos los archivos diarios extraídos en un único dataset..."
    )
    all_chunks = glob.glob(f"{chunks_dir}/*.csv")

    if not all_chunks:
        raise ValueError(
            f"No se pudieron extraer datos para ningún día en el rango {config.initial_date} a {config.end_date}."
        )

    df_final = pd.concat([pd.read_csv(f) for f in all_chunks], ignore_index=True)
    df_final = df_final.dropna(subset=["ID_Tweet"])

    # Validar dataset consolidado antes de guardar
    validate_df(
        df_final, SocialRawRow, stage="raw_daily_social_data (write consolidated)"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"daily_top_{safe_ticker}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    df_final.to_csv(filepath, index=False)
    context.log.info(f"Datos consolidados guardados exitosamente en: {filepath}")

    context.add_output_metadata(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "total_records_extracted": MetadataValue.int(len(df_final)),
            "days_processed": MetadataValue.int(len(all_chunks)),
        }
    )

    return filepath


@asset(
    group_name="processing_daily_top",
    description="Estandariza los top posts diarios de manera robusta y aislada.",
)
def processed_daily_social_data(
    context: AssetExecutionContext, raw_daily_social_data: str
) -> str:
    """Limpieza estructural implementada de forma nativa para evitar colisiones de Dagster."""
    raw_path = raw_daily_social_data

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el fichero Raw de redes sociales en: {raw_path}"
        )

    context.log.info(f"Leyendo datos de redes sociales desde la capa Raw: {raw_path}")
    df = pd.read_csv(raw_path)

    # Validar al leer raw consolidado
    validate_df(df, SocialRawRow, stage="processed_daily_social_data (read raw)")

    # 1. Normalización
    mapping_columnas = {
        "ID_Tweet": "tweet_id",
        "Fecha_UTC": "fecha_utc",
        "Contenido_Texto": "contenido_texto",
        "Retweets": "retweets",
        "Favoritos": "favoritos",
    }
    df = df.rename(columns=mapping_columnas)

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

    # 2. Data Quality
    df = df.dropna(subset=["tweet_id", "fecha_utc", "contenido_texto"])
    df["fecha_utc"] = pd.to_datetime(df["fecha_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["fecha_utc"])
    df["fecha_limpia"] = df["fecha_utc"].dt.date

    df["retweets"] = df["retweets"].fillna(0).astype(int)
    df["favoritos"] = df["favoritos"].fillna(0).astype(int)
    df["contenido_texto"] = df["contenido_texto"].astype(str)

    # Validar procesado antes de guardar
    validate_df(
        df, SocialProcessedRow, stage="processed_daily_social_data (write processed)"
    )

    # 3. Guardado
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/02_processed/social_daily_top/temp"
    os.makedirs(output_dir, exist_ok=True)
    filepath_cleaned = os.path.join(output_dir, f"social_cleaned_{timestamp}.csv")
    df.to_csv(filepath_cleaned, index=False)

    context.log.info(f"Datos limpios estructuralmente guardados en: {filepath_cleaned}")
    return filepath_cleaned


@asset(
    group_name="processing_daily_top",
    description="Aplica NLP sobre los posts top diarios de manera robusta y aislada.",
)
def daily_social_sentiment_analysis(
    context: AssetExecutionContext, processed_daily_social_data: str
) -> MaterializeResult:
    """Análisis de sentimiento nativo para evitar dependencias cruzadas en el grafo."""
    cleaned_path = processed_daily_social_data

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(
            f"No se encontró el archivo base limpio en: {cleaned_path}"
        )

    context.log.info(
        f"Cargando publicaciones para realizar inferencia NLP desde {cleaned_path}..."
    )
    df = pd.read_csv(cleaned_path)

    # Validar al leer procesado
    validate_df(
        df, SocialProcessedRow, stage="daily_social_sentiment_analysis (read processed)"
    )

    if df.empty:
        context.log.warning("Dataset vacío. Saltando cálculo NLP.")
        df["sentimiento"] = pd.Series(dtype=str)
        df["puntuacion_sentimiento"] = pd.Series(dtype=float)
    else:
        context.log.info(f"Inicializando pipeline NLP: {DEFAULT_MODEL}")
        classifier = pipeline(
            "sentiment-analysis", model=DEFAULT_MODEL, truncation=True, device=-1
        )

        texts = df["contenido_texto"].fillna("").astype(str).tolist()
        results = classifier(texts, truncation=True, max_length=512)

        df["sentimiento_label"] = [res["label"] for res in results]
        df["sentimiento_score"] = [res["score"] for res in results]

        puntuaciones_continuas = []
        for label, score in zip(df["sentimiento_label"], df["sentimiento_score"]):
            label_lower = label.lower()
            if "pos" in label_lower or "label_2" in label_lower:
                val = 0.5 + (score * 0.5)
            elif "neg" in label_lower or "label_0" in label_lower:
                val = 0.5 - (score * 0.5)
            else:
                val = 0.5
            puntuaciones_continuas.append(round(val, 4))

        df["puntuacion_sentimiento"] = puntuaciones_continuas
        df["sentimiento"] = df["sentimiento_label"].apply(
            lambda labelled: (
                "positive"
                if "pos" in labelled.lower() or "label_2" in labelled.lower()
                else (
                    "negative"
                    if "neg" in labelled.lower() or "label_0" in labelled.lower()
                    else "neutral"
                )
            )
        )

        df = df.drop(columns=["sentimiento_label", "sentimiento_score"])

    df["fecha_utc"] = pd.to_datetime(
        df["fecha_utc"], format="ISO8601", errors="coerce", utc=True
    )

    # Validar antes de guardar
    validate_df(
        df,
        SocialSentimentRow,
        stage="daily_social_sentiment_analysis (write sentiment)",
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data/02_processed/social_daily_top/sentiment"
    filepath_cleaned = os.path.join(output_dir, f"social_sentiment_{timestamp}.csv")
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(filepath_cleaned, index=False)

    promedio_sentimiento = (
        float(df["puntuacion_sentimiento"].mean()) if not df.empty else 0.5
    )

    return MaterializeResult(
        metadata={
            "fichero_procesado": MetadataValue.path(filepath_cleaned),
            "registros_analizados": MetadataValue.int(len(df)),
            "sentimiento_promedio_nlp": MetadataValue.float(promedio_sentimiento),
        }
    )


@asset(
    deps=[daily_social_sentiment_analysis],
    group_name="feature_engineering",
    description="Agrega el sentimiento diario (volumen, media, desviación estándar) para cruzar con datos financieros.",
)
def aggregated_daily_social_sentiment(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """
    Asset de agregación. Lee el resultado granular del análisis de sentimiento
    y calcula métricas resumidas por día, preparándolas para el Random Forest.
    """
    input_dir = "data/02_processed/social_daily_top/sentiment"

    list_of_files = glob.glob(f"{input_dir}/*.csv")
    if not list_of_files:
        raise FileNotFoundError(
            f"No se encontraron archivos de sentimiento en {input_dir}"
        )

    latest_file = max(list_of_files, key=os.path.getctime)
    context.log.info(f"Cargando dataset de sentimiento granular desde: {latest_file}")

    df = pd.read_csv(latest_file)

    # Validar al leer sentiment
    validate_df(
        df,
        SocialSentimentRow,
        stage="aggregated_daily_social_sentiment (read sentiment)",
    )

    if "fecha_limpia" not in df.columns or "puntuacion_sentimiento" not in df.columns:
        raise ValueError(
            "El dataset de sentimiento no contiene las columnas necesarias ('fecha_limpia', 'puntuacion_sentimiento')."
        )

    context.log.info(
        "Calculando agregaciones diarias: Volumen, Media y Desviación Estándar..."
    )

    agg_df = (
        df.groupby("fecha_limpia")
        .agg(
            volumen_posts=("puntuacion_sentimiento", "count"),
            sentimiento_medio=("puntuacion_sentimiento", "mean"),
            sentimiento_std=("puntuacion_sentimiento", "std"),
        )
        .reset_index()
    )

    agg_df["sentimiento_std"] = agg_df["sentimiento_std"].fillna(0.0)

    # Validar agregación antes de guardar
    validate_df(
        agg_df,
        SocialAggregatedRow,
        stage="aggregated_daily_social_sentiment (write aggregated)",
    )

    output_dir = "data/03_features/social_daily_top"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "daily_sentiment_aggregated.csv")

    agg_df.to_csv(output_path, index=False)
    context.log.info(f"Features sociales agregadas y guardadas en: {output_path}")

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(output_path),
            "total_dias_generados": MetadataValue.int(len(agg_df)),
            "media_volumen_diario": MetadataValue.float(
                float(agg_df["volumen_posts"].mean())
            ),
            "sentimiento_medio_historico": MetadataValue.float(
                float(agg_df["sentimiento_medio"].mean())
            ),
        }
    )
