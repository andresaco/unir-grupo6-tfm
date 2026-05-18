#!/usr/bin/env python3
"""
Script optimizado para análisis de sentimientos local usando modelos de Hugging Face.

Este script mejora la versión base implementando procesamiento por lotes (batching),
detección automática de GPU, truncamiento seguro para textos largos y evitando la 
doble inferencia por cada fila para un rendimiento sustancialmente mayor.

Uso:
    python analisis_sentimientos_optimizado.py --csv_path ruta/al/archivo.csv --text_column nombre_columna [--output_path ruta/salida.csv]

Ejemplo:
    python analisis_sentimientos_optimizado.py --csv_path datos.csv --text_column Contenido_Texto --output_path resultados.csv

Dependencias:
    - pandas
    - transformers
    - torch
"""

import argparse
import sys

import pandas as pd
import torch
from transformers import pipeline


def parse_arguments() -> argparse.Namespace:
    """Analiza y retorna los argumentos de línea de comandos.

    Returns:
        argparse.Namespace: Objeto con los argumentos analizados listos para usar.
    """
    parser = argparse.ArgumentParser(description='Análisis de sentimientos optimizado con Hugging Face')
    parser.add_argument('--csv_path', required=True, help='Ruta al archivo CSV de entrada')
    parser.add_argument('--text_column', required=True, help='Nombre de la columna que contiene el texto a analizar')
    parser.add_argument('--output_path', default='resultados_sentimiento.csv', help='Ruta para guardar el CSV de salida (opcional)')
    parser.add_argument('--model', default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                        help='Modelo de Hugging Face a usar (opcional)')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Tamaño de lote (batch size) para la inferencia. Ajustar según la RAM/VRAM disponible.')

    return parser.parse_args()


def load_dataframe(csv_path: str, text_column: str) -> pd.DataFrame:
    """Carga un archivo CSV en un DataFrame y verifica la existencia de la columna de texto.

    Args:
        csv_path (str): Ruta al archivo CSV de entrada.
        text_column (str): Nombre de la columna que contiene el texto.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo CSV no existe en la ruta especificada.
        ValueError: Si la columna de texto no se encuentra en el DataFrame.
        Exception: Si ocurre un error inesperado al leer el CSV.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Archivo CSV cargado exitosamente: {csv_path}")
        print(f"Número de filas: {len(df)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: No se encontró el archivo {csv_path}")
    except Exception as e:
        raise Exception(f"Error al leer el CSV: {e}")

    if text_column not in df.columns:
        raise ValueError(f"Error: La columna '{text_column}' no existe en el CSV. Columnas disponibles: {list(df.columns)}")

    return df


def initialize_model(model_name: str):
    """Inicializa el pipeline de análisis de sentimientos de Hugging Face.

    Detecta automáticamente si hay una GPU disponible y la utiliza si es posible,
    de lo contrario utiliza la CPU.

    Args:
        model_name (str): Nombre del modelo en el Hugging Face Hub.

    Returns:
        transformers.pipelines.Pipeline: Pipeline configurado para el análisis de sentimientos.

    Raises:
        RuntimeError: Si ocurre un error al cargar el modelo o sus dependencias.
    """
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    
    print(f"Inicializando el modelo de análisis de sentimientos: {model_name}")
    print(f"Dispositivo de inferencia detectado: {device_name}")
    
    try:
        sentiment_analyzer = pipeline('sentiment-analysis', model=model_name, device=device)
        print("Modelo cargado exitosamente.")
        return sentiment_analyzer
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo: {e}\nAsegúrate de tener instaladas las dependencias: pip install transformers torch")


def analyze_sentiments(df: pd.DataFrame, text_column: str, sentiment_analyzer, batch_size: int) -> pd.DataFrame:
    """Aplica el modelo de análisis de sentimientos a una columna de texto.

    Utiliza procesamiento por lotes (batching) para optimizar la inferencia y 
    maneja valores nulos o cadenas vacías adecuadamente.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        text_column (str): Nombre de la columna con el texto a analizar.
        sentiment_analyzer (transformers.pipelines.Pipeline): Pipeline de Hugging Face inicializado.
        batch_size (int): Tamaño del lote para la inferencia.

    Returns:
        pd.DataFrame: DataFrame modificado con las nuevas columnas 'Sentimiento' y 'Puntuacion_Sentimiento'.
        
    Raises:
        RuntimeError: Si ocurre un error durante el proceso de inferencia.
    """
    print(f"Aplicando análisis de sentimientos en lotes de {batch_size}...")
    try:
        # Preprocesar textos: manejar nulos y textos vacíos
        textos_crudos = df[text_column].fillna("").astype(str).tolist()
        
        # Hugging Face pipelines pueden fallar con strings completamente vacíos, los reemplazamos temporalmente
        textos_seguros = [t if t.strip() else "Neutral" for t in textos_crudos]
        
        # Pasamos la lista completa al pipeline para aprovechar el batching (mucho más rápido)
        # truncation=True evita que el modelo falle si un texto supera los tokens máximos (ej. 512)
        resultados = sentiment_analyzer(textos_seguros, batch_size=batch_size, truncation=True)
        
        # Extraemos etiquetas y scores en una sola pasada
        df['Sentimiento'] = [res['label'] if orig.strip() else 'Neutral' for res, orig in zip(resultados, textos_crudos)]
        df['Puntuacion_Sentimiento'] = [res['score'] if orig.strip() else 0.0 for res, orig in zip(resultados, textos_crudos)]
        
        print("Análisis completado.")
        return df
    except Exception as e:
        raise RuntimeError(f"Error durante el análisis: {e}")


def save_results(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el DataFrame procesado en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame con los resultados.
        output_path (str): Ruta donde se guardará el archivo CSV.

    Raises:
        IOError: Si ocurre un error al intentar escribir el archivo en disco.
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"\nResultados guardados en: {output_path}")
    except Exception as e:
        raise IOError(f"Error al guardar el archivo: {e}")


def main():
    """Función principal que orquesta el flujo completo de análisis de sentimientos."""
    args = parse_arguments()

    try:
        # 1 y 2. Cargar datos y verificar columnas
        df = load_dataframe(args.csv_path, args.text_column)
        
        # 3 y 4. Inicializar modelo detectando el hardware
        sentiment_analyzer = initialize_model(args.model)
        
        # 5. Analizar sentimientos
        df = analyze_sentiments(df, args.text_column, sentiment_analyzer, args.batch_size)
        
        # 6. Mostrar resumen
        print("\nResumen del análisis:")
        print(df['Sentimiento'].value_counts())
        
        # 7. Guardar el resultado final
        save_results(df, args.output_path)
        
        print("Proceso completado exitosamente.")
        
    except (FileNotFoundError, ValueError, RuntimeError, IOError, Exception) as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()