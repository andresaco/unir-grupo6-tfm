import argparse
import os
import sys
from datetime import datetime

import pandas as pd


def partition_tweets_by_day(input_file, output_dir):
    """
    Lee un archivo CSV y lo divide en múltiples archivos basándose en la fecha.
    Cada fecha única generará un archivo independiente.
    """
    try:
        print(f"Leyendo el archivo: {input_file}...")
        # Forzamos ID_Tweet como string para evitar que Python lo convierta a notación científica
        df = pd.read_csv(input_file, dtype={'ID_Tweet': str})

        # 'coerce' convertirá fechas mal formadas en NaT (Not a Time)
        df['Fecha_UTC'] = pd.to_datetime(df['Fecha_UTC'], errors='coerce')

        initial_count = len(df)
        df = df.dropna(subset=['Fecha_UTC'])
        if len(df) < initial_count:
            print(f"Aviso: Se omitieron {initial_count - len(df)} filas con fechas inválidas.")

        # Esto permite que tweets de distintas horas del mismo día terminen en el mismo archivo
        df['Solo_Fecha'] = df['Fecha_UTC'].dt.date

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Carpeta de salida creada: {output_dir}")

        # groupby crea sub-datasets basados en el valor de 'Solo_Fecha'
        grouped = df.groupby('Solo_Fecha')
        for date, group in grouped:
            filename = f"tweets_{date}.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Eliminamos la columna auxiliar 'Solo_Fecha' antes de guardar
            export_df = group.drop(columns=['Solo_Fecha'])
            export_df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"Generado: {filepath} ({len(export_df)} registros)")

        print(f"\nProceso completado con éxito. Archivos en: {output_dir}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particiona un CSV de tweets por día.")
    parser.add_argument("fichero", help="Ruta al fichero CSV de entrada")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    input_path = args.fichero

    if os.path.exists(input_path):
        # Localización del fichero para crear la carpeta en el mismo sitio
        directorio_base = os.path.dirname(os.path.abspath(input_path))
        nombre_base = os.path.splitext(os.path.basename(input_path))[0]
        
        # Generamos el timestamp para la carpeta de salida
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ejemplo: datos_partition_20240509_163000
        nombre_carpeta_salida = f"{nombre_base}_partition_{timestamp}"
        ruta_salida_final = os.path.join(directorio_base, nombre_carpeta_salida)

        partition_tweets_by_day(input_path, ruta_salida_final)
    else:
        print(f"Error: El fichero '{input_path}' no existe.")