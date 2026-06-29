import os
import sys
from google.cloud import bigquery
import pandas as pd

# Intento de importar validación y esquema
try:
    from ..schemas import validate_df, GdeltSentimentRow
except ImportError, ValueError:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from src.schemas import validate_df, GdeltSentimentRow

# 1. CONFIGURACIÓN DE CREDENCIALES
# Coloca la ruta física a tu archivo JSON de credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


def extraer_sentimiento_avanzado_rango(
    fecha_inicio: str, fecha_fin: str, palabra_clave: str = "Apple"
):
    """
    Realiza una única consulta optimizada a GDELT utilizando la tabla particionada
    para un rango específico de fechas, extrayendo métricas lingüísticas y de sentimiento avanzadas.

    Parámetros:
    - fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'
    - fecha_fin (str): Fecha de fin en formato 'YYYY-MM-DD'
    - palabra_clave (str): Organización a buscar (por defecto 'Apple')
    """

    # Inicializar el cliente de BigQuery
    client = bigquery.Client()

    print(
        f"Preparando consulta avanzada para '{palabra_clave}' desde {fecha_inicio} hasta {fecha_fin}..."
    )

    # CONSULTA OPTIMIZADA MULTI-MÉTRICA:
    # Usamos SAFE_OFFSET y SAFE_CAST para proteger la consulta de nulos o cadenas malformadas.
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

    # Configuración de parámetros seguros para evitar inyecciones de código SQL
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "STRING", fecha_inicio),
            bigquery.ScalarQueryParameter("end_date", "STRING", fecha_fin),
            bigquery.ScalarQueryParameter(
                "keyword", "STRING", f"%{palabra_clave.lower()}%"
            ),
        ]
    )

    try:
        # Ejecutar la consulta
        print("Enviando petición a Google BigQuery...")
        query_job = client.query(query, job_config=job_config)

        # Consumo de datos de la consulta
        bytes_scanned = query_job.total_bytes_processed
        if bytes_scanned is not None:
            gb_scanned = bytes_scanned / (1024**3)
            print(f"-> Datos escaneados para esta consulta: {gb_scanned:.2f} GB")

        # Esperar resultados
        results = query_job.result()
        df = results.to_dataframe()

        if not df.empty:
            # Asegurar formato de fecha en Pandas
            df["fecha"] = pd.to_datetime(df["fecha"])

            # Validar antes de guardar
            validate_df(
                df, GdeltSentimentRow, stage="gdelt_extraction (write GDELT raw)"
            )

            # Guardar el resultado enriquecido
            nombre_archivo = f"gdelt_apple_avanzado_{fecha_inicio}_a_{fecha_fin}.csv"
            df.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

            print("\n¡Éxito total! Datos enriquecidos descargados correctamente.")
            print(f"Archivo guardado: {nombre_archivo}")
            print(f"Total de días analizados: {len(df)}")
            print("\nPrimeras filas del DataFrame con métricas expandidas:")
            # Mostrar columnas clave en la previsualización
            print(
                df[
                    [
                        "fecha",
                        "volumen_noticias",
                        "sentimiento_promedio",
                        "puntuacion_positiva",
                        "puntuacion_negativa",
                        "volatilidad_sentimiento",
                    ]
                ].head(5)
            )
        else:
            print("No se encontraron registros para los filtros seleccionados.")

    except Exception as e:
        print(f"\nOcurrió un error al procesar la consulta en BigQuery: {e}")
        print(
            "Consejo: Verifica las fechas (YYYY-MM-DD) y los permisos de tu cuenta de Google Cloud."
        )


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # Rango de prueba (4 meses en 2025)
    FECHA_DESDE = "2020-01-01"
    FECHA_HASTA = "2026-12-31"
    ORGANIZACION = "Apple"

    extraer_sentimiento_avanzado_rango(FECHA_DESDE, FECHA_HASTA, ORGANIZACION)
