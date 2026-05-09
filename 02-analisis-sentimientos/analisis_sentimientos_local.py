#!/usr/bin/env python3
"""
Script para análisis de sentimientos local usando modelos de Hugging Face.

Este script lee un archivo CSV local, aplica análisis de sentimientos a una columna de texto
especificada utilizando un modelo de Hugging Face, y guarda los resultados en un nuevo CSV.

Uso:
    python analisis_sentimientos_local.py --csv_path ruta/al/archivo.csv --text_column nombre_columna [--output_path ruta/salida.csv]

Ejemplo:
    python analisis_sentimientos_local.py --csv_path datos.csv --text_column Contenido_Texto --output_path resultados.csv

Dependencias:
    - pandas
    - transformers
    - torch

Instala con: pip install pandas transformers torch
"""

import argparse
import sys

import pandas as pd
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description='Análisis de sentimientos local con Hugging Face')
    parser.add_argument('--csv_path', required=True, help='Ruta al archivo CSV de entrada')
    parser.add_argument('--text_column', required=True, help='Nombre de la columna que contiene el texto a analizar')
    parser.add_argument('--output_path', default='resultados_sentimiento.csv', help='Ruta para guardar el CSV de salida (opcional)')
    parser.add_argument('--model', default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                        help='Modelo de Hugging Face a usar (opcional)')

    args = parser.parse_args()

    # Verificar que el archivo CSV existe
    try:
        df = pd.read_csv(args.csv_path)
        print(f"Archivo CSV cargado exitosamente: {args.csv_path}")
        print(f"Número de filas: {len(df)}")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {args.csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el CSV: {e}")
        sys.exit(1)

    # Verificar que la columna existe
    if args.text_column not in df.columns:
        print(f"Error: La columna '{args.text_column}' no existe en el CSV. Columnas disponibles: {list(df.columns)}")
        sys.exit(1)

    # Inicializar el pipeline de análisis de sentimientos
    print(f"Inicializando el modelo de análisis de sentimientos: {args.model}")
    try:
        sentiment_analyzer = pipeline('sentiment-analysis', model=args.model)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Asegúrate de tener instaladas las dependencias: pip install transformers torch")
        sys.exit(1)

    # Aplicar análisis de sentimientos
    print("Aplicando análisis de sentimientos...")
    try:
        df['Sentimiento'] = df[args.text_column].astype(str).apply(
            lambda x: sentiment_analyzer(x)[0]['label'] if x.strip() else 'Neutral'
        )
        df['Puntuacion_Sentimiento'] = df[args.text_column].astype(str).apply(
            lambda x: sentiment_analyzer(x)[0]['score'] if x.strip() else 0.0
        )
        print("Análisis completado.")
    except Exception as e:
        print(f"Error durante el análisis: {e}")
        sys.exit(1)

    # Mostrar resumen
    print("\nResumen del análisis:")
    print(df['Sentimiento'].value_counts())

    # Guardar el resultado
    try:
        df.to_csv(args.output_path, index=False)
        print(f"\nResultados guardados en: {args.output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
        sys.exit(1)

    print("Proceso completado exitosamente.")

if __name__ == "__main__":
    main()