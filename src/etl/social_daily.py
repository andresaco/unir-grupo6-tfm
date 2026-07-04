import os
import datetime
import asyncio
import pandas as pd
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from dotenv import load_dotenv
from .core.config import StockDownloadConfig
from .core.social.bsky import BlueskyClient
from ..schemas import (
    validate_df,
    SocialRawRow,
    SocialProcessedRow,
    SocialSentimentRow,
    SocialAggregatedRow,
)

load_dotenv()

OUTPUT_DIR = "daily"

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

    output_dir = f"data/01_raw/{OUTPUT_DIR}"
    safe_ticker = str(config.ticker).replace(" ", "_")
    start_date_str = config.initial_date.replace("-", "")
    end_date_str = config.end_date.replace("-", "")
    filename = f"daily_top_{safe_ticker}_{start_date_str}_{end_date_str}.csv"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        context.log.info(
            f"-> [Checkpoint] El archivo consolidado raw ya existe: {filepath}. Saltando extracción..."
        )
        try:
            df_existing = pd.read_csv(filepath)
            total_records = len(df_existing)
        except Exception:
            total_records = 0
        context.add_output_metadata(
            metadata={
                "filepath": MetadataValue.path(filepath),
                "total_records_extracted": MetadataValue.int(total_records),
                "checkpoint_hit": MetadataValue.text("True"),
            }
        )
        return filepath

    chunks_dir = os.path.join(output_dir, f"{safe_ticker}")
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
        "Consolidando archivos diarios en el rango solicitado en un único dataset..."
    )
    run_chunks = []
    tmp_date = start_date
    while tmp_date <= end_date:
        chunk_filename = f"{tmp_date.strftime('%Y-%m-%d')}.csv"
        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
        if os.path.exists(chunk_filepath):
            run_chunks.append(chunk_filepath)
        tmp_date = tmp_date + datetime.timedelta(days=1)

    if not run_chunks:
        raise ValueError(
            f"No se pudieron extraer datos para ningún día en el rango {config.initial_date} a {config.end_date}."
        )

    df_final = pd.concat([pd.read_csv(f) for f in run_chunks], ignore_index=True)
    df_final = df_final.dropna(subset=["ID_Tweet"])

    # Validar dataset consolidado antes de guardar
    validate_df(
        df_final, SocialRawRow, stage="raw_daily_social_data (write consolidated)"
    )

    df_final.to_csv(filepath, index=False)
    context.log.info(f"Datos consolidados guardados exitosamente en: {filepath}")

    context.add_output_metadata(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "total_records_extracted": MetadataValue.int(len(df_final)),
            "days_processed": MetadataValue.int(len(run_chunks)),
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

    # Analizar el nombre base para extraer parámetros
    base_name = os.path.basename(raw_path)
    parts = base_name.replace(".csv", "").split("_")
    if len(parts) < 5:
        raise ValueError(
            f"El nombre del archivo raw '{base_name}' no cumple con el formato esperado."
        )

    start_date_str = parts[-2]
    end_date_str = parts[-1]
    safe_ticker = "_".join(parts[2:-2])

    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")

    # Directorios de entrada y salida de chunks
    raw_chunks_dir = os.path.join(f"data/01_raw/{OUTPUT_DIR}", f"{safe_ticker}")
    processed_chunks_dir = os.path.join(
        f"data/02_processed/{OUTPUT_DIR}/temp", f"{safe_ticker}"
    )
    os.makedirs(processed_chunks_dir, exist_ok=True)

    processed_chunk_paths = []

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        raw_chunk_path = os.path.join(raw_chunks_dir, f"{date_str}.csv")
        processed_chunk_path = os.path.join(processed_chunks_dir, f"{date_str}.csv")

        if os.path.exists(raw_chunk_path):
            if os.path.exists(processed_chunk_path):
                context.log.info(
                    f"-> [Checkpoint] Día {date_str} ya procesado estructuralmente. Saltando..."
                )
            else:
                context.log.info(f"-> Procesando estructuralmente día {date_str}...")
                df_chunk = pd.read_csv(raw_chunk_path)
                validate_df(
                    df_chunk,
                    SocialRawRow,
                    stage=f"processed_daily_social_data (read chunk {date_str})",
                )

                # 1. Normalización
                mapping_columnas = {
                    "ID_Tweet": "tweet_id",
                    "Fecha_UTC": "fecha_utc",
                    "Contenido_Texto": "contenido_texto",
                    "Retweets": "retweets",
                    "Favoritos": "favoritos",
                }
                df_chunk = df_chunk.rename(columns=mapping_columnas)

                columnas_esperadas = {
                    "tweet_id": "object",
                    "fecha_utc": "object",
                    "contenido_texto": "object",
                    "retweets": "int64",
                    "favoritos": "int64",
                }
                for col, dtype in columnas_esperadas.items():
                    if col not in df_chunk.columns:
                        df_chunk[col] = pd.Series(dtype=dtype)

                # 2. Data Quality
                df_chunk = df_chunk.dropna(
                    subset=["tweet_id", "fecha_utc", "contenido_texto"]
                )
                df_chunk["fecha_utc"] = pd.to_datetime(
                    df_chunk["fecha_utc"], errors="coerce", utc=True
                )
                df_chunk = df_chunk.dropna(subset=["fecha_utc"])
                df_chunk["fecha_limpia"] = df_chunk["fecha_utc"].dt.date

                df_chunk["retweets"] = df_chunk["retweets"].fillna(0).astype(int)
                df_chunk["favoritos"] = df_chunk["favoritos"].fillna(0).astype(int)
                df_chunk["contenido_texto"] = df_chunk["contenido_texto"].astype(str)

                # Validar procesado antes de guardar
                validate_df(
                    df_chunk,
                    SocialProcessedRow,
                    stage=f"processed_daily_social_data (write chunk {date_str})",
                )
                df_chunk.to_csv(processed_chunk_path, index=False)

            processed_chunk_paths.append(processed_chunk_path)
        else:
            context.log.warning(
                f"No se encontró el chunk raw para el día {date_str} en: {raw_chunk_path}"
            )

        current_date = current_date + datetime.timedelta(days=1)

    if not processed_chunk_paths:
        raise ValueError(
            f"No se procesaron chunks de datos para ningún día en el rango {start_date_str} a {end_date_str}."
        )

    # Consolidar todos los chunks procesados en un único archivo procesado final
    context.log.info("Consolidando todos los chunks procesados...")
    df_final = pd.concat(
        [pd.read_csv(f) for f in processed_chunk_paths], ignore_index=True
    )

    # Validar consolidado final
    validate_df(
        df_final,
        SocialProcessedRow,
        stage="processed_daily_social_data (write consolidated processed)",
    )

    cleaned_filename = base_name.replace("daily_top_", "social_cleaned_")
    output_dir = f"data/02_processed/{OUTPUT_DIR}/temp"
    os.makedirs(output_dir, exist_ok=True)
    filepath_cleaned = os.path.join(output_dir, cleaned_filename)
    df_final.to_csv(filepath_cleaned, index=False)

    context.log.info(f"Datos consolidados limpios guardados en: {filepath_cleaned}")
    return filepath_cleaned


@asset(
    group_name="processing_daily_top",
    description="Aplica NLP sobre los posts top diarios de manera robusta y aislada.",
)
def daily_social_sentiment_analysis(
    context: AssetExecutionContext, processed_daily_social_data: str
) -> pd.DataFrame:
    """Análisis de sentimiento nativo para evitar dependencias cruzadas en el grafo."""
    cleaned_path = processed_daily_social_data

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(
            f"No se encontró el archivo base limpio en: {cleaned_path}"
        )

    # Analizar el nombre base para extraer parámetros
    base_name = os.path.basename(cleaned_path)
    parts = base_name.replace(".csv", "").split("_")
    if len(parts) < 5:
        raise ValueError(
            f"El nombre del archivo procesado '{base_name}' no cumple con el formato esperado."
        )

    start_date_str = parts[-2]
    end_date_str = parts[-1]
    safe_ticker = "_".join(parts[2:-2])

    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")

    sentiment_dir = f"data/02_processed/{OUTPUT_DIR}/sentiment"
    os.makedirs(sentiment_dir, exist_ok=True)
    consolidated_sentiment_filename = (
        f"social_sentiment_{safe_ticker}_{start_date_str}_{end_date_str}.csv"
    )
    consolidated_sentiment_path = os.path.join(
        sentiment_dir, consolidated_sentiment_filename
    )

    if os.path.exists(consolidated_sentiment_path):
        context.log.info(
            f"-> [Checkpoint] El archivo consolidado de sentimiento ya existe: {consolidated_sentiment_path}. Saltando análisis NLP..."
        )
        df_final = pd.read_csv(consolidated_sentiment_path)
        df_final["fecha_utc"] = pd.to_datetime(
            df_final["fecha_utc"], errors="coerce", utc=True
        )
        promedio_sentimiento = (
            float(df_final["puntuacion_sentimiento"].mean())
            if not df_final.empty
            else 0.5
        )
        context.add_output_metadata(
            metadata={
                "consolidated_sentiment_path": MetadataValue.path(
                    consolidated_sentiment_path
                ),
                "registros_analizados": MetadataValue.int(len(df_final)),
                "sentimiento_promedio_nlp": MetadataValue.float(promedio_sentimiento),
                "checkpoint_hit": MetadataValue.text("True"),
            }
        )
        return df_final

    # Directorios de entrada y salida de chunks
    processed_chunks_dir = os.path.join(
        f"data/02_processed/{OUTPUT_DIR}/temp", f"{safe_ticker}"
    )
    sentiment_chunks_dir = os.path.join(
        f"data/02_processed/{OUTPUT_DIR}/sentiment", f"{safe_ticker}"
    )
    os.makedirs(sentiment_chunks_dir, exist_ok=True)

    sentiment_chunk_paths = []
    classifier = None

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        processed_chunk_path = os.path.join(processed_chunks_dir, f"{date_str}.csv")
        sentiment_chunk_path = os.path.join(sentiment_chunks_dir, f"{date_str}.csv")

        if os.path.exists(processed_chunk_path):
            if os.path.exists(sentiment_chunk_path):
                context.log.info(
                    f"-> [Checkpoint] Sentimiento para el día {date_str} ya calculado. Saltando..."
                )
            else:
                context.log.info(
                    f"-> Calculando sentimiento NLP para día {date_str}..."
                )
                df_chunk = pd.read_csv(processed_chunk_path)
                validate_df(
                    df_chunk,
                    SocialProcessedRow,
                    stage=f"daily_social_sentiment_analysis (read chunk {date_str})",
                )

                if df_chunk.empty:
                    context.log.warning(f"Día {date_str} vacío. Saltando cálculo NLP.")
                    df_chunk["sentimiento"] = pd.Series(dtype=str)
                    df_chunk["puntuacion_sentimiento"] = pd.Series(dtype=float)
                else:
                    if classifier is None:
                        context.log.info(f"Inicializando pipeline NLP: {DEFAULT_MODEL}")
                        import torch

                        device = (
                            0
                            if torch.cuda.is_available()
                            else ("mps" if torch.backends.mps.is_available() else -1)
                        )
                        if device == -1:
                            torch.set_num_threads(1)

                        from transformers import pipeline

                        classifier = pipeline(
                            "sentiment-analysis",
                            model=DEFAULT_MODEL,
                            truncation=True,
                            device=device,
                        )

                    texts = df_chunk["contenido_texto"].fillna("").astype(str).tolist()
                    results = classifier(
                        texts, truncation=True, max_length=512, batch_size=16
                    )

                    df_chunk["sentimiento_label"] = [res["label"] for res in results]
                    df_chunk["sentimiento_score"] = [res["score"] for res in results]

                    puntuaciones_continuas = []
                    for label, score in zip(
                        df_chunk["sentimiento_label"], df_chunk["sentimiento_score"]
                    ):
                        label_lower = label.lower()
                        if "pos" in label_lower or "label_2" in label_lower:
                            val = 0.5 + (score * 0.5)
                        elif "neg" in label_lower or "label_0" in label_lower:
                            val = 0.5 - (score * 0.5)
                        else:
                            val = 0.5
                        puntuaciones_continuas.append(round(val, 4))

                    df_chunk["puntuacion_sentimiento"] = puntuaciones_continuas
                    df_chunk["sentimiento"] = df_chunk["sentimiento_label"].apply(
                        lambda labelled: (
                            "positive"
                            if "pos" in labelled.lower()
                            or "label_2" in labelled.lower()
                            else (
                                "negative"
                                if "neg" in labelled.lower()
                                or "label_0" in labelled.lower()
                                else "neutral"
                            )
                        )
                    )

                    df_chunk = df_chunk.drop(
                        columns=["sentimiento_label", "sentimiento_score"]
                    )

                df_chunk["fecha_utc"] = pd.to_datetime(
                    df_chunk["fecha_utc"], format="ISO8601", errors="coerce", utc=True
                )

                # Validar chunk de sentimiento antes de guardar
                validate_df(
                    df_chunk,
                    SocialSentimentRow,
                    stage=f"daily_social_sentiment_analysis (write chunk {date_str})",
                )
                df_chunk.to_csv(sentiment_chunk_path, index=False)

            sentiment_chunk_paths.append(sentiment_chunk_path)
        else:
            context.log.warning(
                f"No se encontró el chunk procesado para el día {date_str} en: {processed_chunk_path}"
            )

        current_date = current_date + datetime.timedelta(days=1)

    if not sentiment_chunk_paths:
        raise ValueError(
            f"No se analizaron sentimientos para ningún día en el rango {start_date_str} a {end_date_str}."
        )

    # Consolidar todos los chunks de sentimiento en un DataFrame único
    context.log.info("Consolidando todos los chunks de sentimiento...")
    df_final = pd.concat(
        [pd.read_csv(f) for f in sentiment_chunk_paths], ignore_index=True
    )

    validate_df(
        df_final,
        SocialSentimentRow,
        stage="daily_social_sentiment_analysis (consolidate sentiment)",
    )

    promedio_sentimiento = (
        float(df_final["puntuacion_sentimiento"].mean()) if not df_final.empty else 0.5
    )

    df_final.to_csv(consolidated_sentiment_path, index=False)
    context.log.info(
        f"Datos consolidados de sentimiento guardados en: {consolidated_sentiment_path}"
    )

    context.add_output_metadata(
        metadata={
            "directorio_chunks": MetadataValue.path(sentiment_chunks_dir),
            "consolidated_sentiment_path": MetadataValue.path(
                consolidated_sentiment_path
            ),
            "registros_analizados": MetadataValue.int(len(df_final)),
            "sentimiento_promedio_nlp": MetadataValue.float(promedio_sentimiento),
            "checkpoint_hit": MetadataValue.text("False"),
        }
    )

    return df_final


@asset(
    group_name="feature_engineering",
    description="Agrega el sentimiento diario (volumen, media, desviación estándar) para cruzar con datos financieros.",
)
def aggregated_daily_social_sentiment(
    context: AssetExecutionContext,
    config: StockDownloadConfig,
    daily_social_sentiment_analysis: pd.DataFrame,
) -> MaterializeResult:
    """
    Asset de agregación. Lee el resultado granular del análisis de sentimiento
    y calcula métricas resumidas por día, preparándolas para el Random Forest.
    """
    df = daily_social_sentiment_analysis

    context.log.info(
        "Cargando dataset de sentimiento granular desde el DataFrame en memoria..."
    )

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

    output_dir = f"data/03_features/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config.ticker}_sentiment.csv")

    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # Combinar ambos datasets
            combined_df = pd.concat([existing_df, agg_df], ignore_index=True)
            # Asegurar tipo cadena para ordenación y deduplicación consistente
            combined_df["fecha_limpia"] = combined_df["fecha_limpia"].astype(str)
            # Eliminar duplicados manteniendo la versión más reciente (keep="last")
            combined_df = combined_df.drop_duplicates(
                subset=["fecha_limpia"], keep="last"
            )
            # Ordenar por fecha
            combined_df = combined_df.sort_values(by="fecha_limpia").reset_index(
                drop=True
            )
            agg_df_final = combined_df
            context.log.info(
                f"Se combinaron {len(agg_df)} registros nuevos con {len(existing_df)} registros existentes."
            )
        except Exception as e:
            context.log.warning(
                f"Error al combinar con el archivo existente {output_path}: {e}. Se sobrescribirá."
            )
            agg_df_final = agg_df
    else:
        agg_df["fecha_limpia"] = agg_df["fecha_limpia"].astype(str)
        agg_df_final = agg_df.sort_values(by="fecha_limpia").reset_index(drop=True)

    # Validar agregación final antes de guardar
    validate_df(
        agg_df_final,
        SocialAggregatedRow,
        stage="aggregated_daily_social_sentiment (write aggregated)",
    )

    agg_df_final.to_csv(output_path, index=False)
    context.log.info(f"Features sociales agregadas e incrementadas en: {output_path}")

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(output_path),
            "total_dias_generados": MetadataValue.int(len(agg_df_final)),
            "media_volumen_diario": MetadataValue.float(
                float(agg_df_final["volumen_posts"].mean())
            ),
            "sentimiento_medio_historico": MetadataValue.float(
                float(agg_df_final["sentimiento_medio"].mean())
            ),
        }
    )
