import os
import pandas as pd
from google.cloud import bigquery
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
)

from .core.config import StockDownloadConfig
from ..schemas import validate_df, GdeltSentimentRow

OUTPUT_DIR = "sentiment"


@asset(
    group_name="ingestion",
    description="Descarga datos de sentimiento de GDELT desde Google BigQuery (capa Bronze).",
)
def raw_gdelt_sentiment_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset de ingesta (Bronze). Consulta la tabla particionada de GDELT en BigQuery
    para un rango específico de fechas y organización, y guarda el resultado.
    """
    palabra_clave = config.name
    fecha_inicio = config.initial_date
    fecha_fin = config.end_date

    context.log.info(
        f"Iniciando consulta a GDELT en BigQuery para '{palabra_clave}' desde {fecha_inicio} hasta {fecha_fin}..."
    )

    # Obtener el project ID del entorno para evitar el error OSError
    project_id = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
    )
    if not project_id:
        context.log.warning(
            "No se ha detectado la variable de entorno GOOGLE_CLOUD_PROJECT. "
            "Si el cliente de BigQuery falla, asegúrate de definir GOOGLE_CLOUD_PROJECT en tu archivo .env o en el entorno."
        )

    # Inicializar el cliente de BigQuery indicando el proyecto si está disponible
    client = bigquery.Client(project=project_id)

    query = """
        SELECT 
            DATE(_PARTITIONTIME) AS fecha,
            COUNT(1) AS volumen_noticias,
            
            -- 1. Sentimiento Promedio (Tono general: positivo - negativo)
            ROUND(AVG(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)), 3) AS sentimiento_promedio,
            
            -- 2. Puntuación Positiva (Porcentaje de palabras positivas en el texto)
            ROUND(AVG(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(1)] AS FLOAT64)), 3) AS puntuacion_positiva,
            
            -- 3. Puntuación Negativa (Porcentaje de palabras negativas en el texto)
            ROUND(AVG(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(2)] AS FLOAT64)), 3) AS puntuacion_negativa,
            
            -- 4. Polaridad Promedio (Carga emocional total / subjetividad de los textos)
            ROUND(AVG(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(3)] AS FLOAT64)), 3) AS polaridad_promedio,
            
            -- 5. Volatilidad / Desviación Estándar (Mide si hay mucha división de opiniones en el mismo día)
            ROUND(STDDEV_SAMP(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(0)] AS FLOAT64)), 3) AS volatilidad_sentimiento,
            
            -- 6. Autorreferencias (Mide el porcentaje de pronombres en primera persona)
            ROUND(AVG(SAFE_CAST(SPLIT(v2tone, ',')[SAFE_OFFSET(4)] AS FLOAT64)), 3) AS uso_primera_persona
            
        FROM 
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE 
            -- Filtro crítico de partición para optimizar costes de consulta
            _PARTITIONTIME BETWEEN TIMESTAMP(@start_date) AND TIMESTAMP(@end_date)
            
            -- Filtro de búsqueda de la organización
            AND (LOWER(v2organizations) LIKE @keyword)
            
            -- Filtrar para evitar duplicados por traducciones automáticas
            AND translationinfo IS NULL
        GROUP BY 
            fecha
        ORDER BY 
            fecha ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", fecha_inicio),
            bigquery.ScalarQueryParameter("end_date", "STRING", fecha_fin),
            bigquery.ScalarQueryParameter(
                "keyword", "STRING", f"%{palabra_clave.lower()}%"
            ),
        ]
    )

    query_job = client.query(query, job_config=job_config)

    # Monitoreo de bytes procesados
    bytes_scanned = query_job.total_bytes_processed
    if bytes_scanned is not None:
        gb_scanned = bytes_scanned / (1024**3)
        context.log.info(f"Datos escaneados para esta consulta: {gb_scanned:.2f} GB")

    results = query_job.result()
    df = results.to_dataframe()

    if df.empty:
        raise ValueError(
            f"No se encontraron registros en GDELT para '{palabra_clave}' en el rango {fecha_inicio} a {fecha_fin}."
        )

    # Asegurar formato de fecha en Pandas y ordenar
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Validar el esquema antes de guardar
    validate_df(df, GdeltSentimentRow, stage="raw_gdelt_sentiment_data (write raw)")

    # Almacenar en la capa raw de sentiment
    output_dir = f"data/01_raw/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    safe_name = str(palabra_clave).lower().replace(" ", "_")
    filename = f"gdelt_{safe_name}_avanzado_{fecha_inicio}_a_{fecha_fin}.csv"
    filepath = os.path.join(output_dir, filename)

    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    context.log.info(f"Datos de GDELT guardados correctamente en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath": MetadataValue.path(filepath),
            "keyword": MetadataValue.text(palabra_clave),
            "total_rows": MetadataValue.int(len(df)),
            "start_date": MetadataValue.text(fecha_inicio),
            "end_date": MetadataValue.text(fecha_fin),
        }
    )


@asset(
    deps=[raw_gdelt_sentiment_data],
    group_name="processing",
    description="Estandariza y limpia los datos de sentimiento de GDELT en la capa Silver.",
)
def processed_gdelt_sentiment_data(
    context: AssetExecutionContext, config: StockDownloadConfig
) -> MaterializeResult:
    """
    Asset de procesamiento (Silver). Toma el archivo GDELT descargado en Raw,
    asegura que las columnas estén estandarizadas, normaliza tipos y lo almacena en Processed.
    """
    palabra_clave = config.name
    fecha_inicio = config.initial_date
    fecha_fin = config.end_date

    safe_name = str(palabra_clave).lower().replace(" ", "_")
    raw_path = f"data/01_raw/{OUTPUT_DIR}/gdelt_{safe_name}_avanzado_{fecha_inicio}_a_{fecha_fin}.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No se encontró el archivo raw de GDELT en: {raw_path}"
        )

    context.log.info(f"Cargando archivo raw de GDELT desde {raw_path}...")
    df = pd.read_csv(raw_path)

    # Validar al leer raw
    validate_df(
        df, GdeltSentimentRow, stage="processed_gdelt_sentiment_data (read raw)"
    )

    # Estandarizar nombres de columnas a minúsculas y snake_case si no lo estuvieran
    df.columns = [col.strip().lower() for col in df.columns]

    # Convertir la columna fecha al formato estándar de fecha (sin hora)
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date

    # Validar procesado antes de guardar
    validate_df(
        df, GdeltSentimentRow, stage="processed_gdelt_sentiment_data (write processed)"
    )

    # Guardar en data/02_processed/sentiment/
    output_dir = f"data/02_processed/{OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(
        output_dir, f"gdelt_{safe_name}_avanzado_{fecha_inicio}_a_{fecha_fin}.csv"
    )
    df.to_csv(filepath, index=False)

    context.log.info(f"Datos de GDELT procesados (Silver) guardados en: {filepath}")

    return MaterializeResult(
        metadata={
            "filepath_silver": MetadataValue.path(filepath),
            "total_rows": MetadataValue.int(len(df)),
            "columns": MetadataValue.text(str(list(df.columns))),
        }
    )
