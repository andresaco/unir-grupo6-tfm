import os
from datetime import datetime

import ollama
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Ajusta estas rutas según la estructura de tu proyecto
DIRECTORIO_ENTRADA = "datos/AAPL_partition_20260509_162143"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DIRECTORIO_SALIDA = f"datos/analisis_ollama_{timestamp}"
COLUMNA_TEXTO = "Contenido_Texto"  # Cambia esto por el nombre de la columna que contiene los textos en tus CSV
MODELO_OLLAMA = "phi3.5:3.8b"  # Puedes cambiarlo a "mistral", "phi3", etc.


def analizar_sentimiento(texto: str) -> str:
    """
    Envía un texto al modelo local de Ollama para analizar su sentimiento.
    """
    if pd.isna(texto) or not str(texto).strip():
        return "Vacio"

    # Prompt instruyendo al modelo a ser conciso
    prompt = f"""
    Analiza el sentimiento del texto. Responde ÚNICAMENTE con una de estas palabras: Positivo, Negativo o Neutro.

    Texto: "Me encanta este producto, es maravilloso"
    Sentimiento: Positivo

    Texto: "Es el peor servicio que he recibido, estoy muy decepcionado"
    Sentimiento: Negativo

    Texto: "La tienda abre a las 9 de la mañana"
    Sentimiento: Neutro

    Texto: "{texto}"
    Sentimiento:
    """

    try:
        response = ollama.chat(
            model=MODELO_OLLAMA,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un sistema de clasificación estricto. Solo puedes responder con una única palabra. Nada de explicaciones.",
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 10,  # Límite físico: evita que genere textos largos ahorrando tiempo de computo
            },
        )
        # Extraemos y limpiamos la respuesta
        respuesta_bruta = response["message"]["content"].strip().lower()

        # Blindaje: Buscamos nuestra palabra clave e ignoramos el resto del texto
        for estado in ["positivo", "negativo", "neutro"]:
            if estado in respuesta_bruta:
                return estado.capitalize()

        return "Indeterminado"  # Valor por defecto si el modelo responde algo ininteligible
    except Exception as e:
        print(f"Error procesando texto: {e}")
        return "Error"


def procesar_directorio(input_dir: str, output_dir: str):
    """
    Recorre los archivos CSV de un directorio, analiza el sentimiento y guarda el resultado.
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    archivos_csv = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    if not archivos_csv:
        print(f"No se encontraron archivos CSV en el directorio: {input_dir}")
        return

    print(f"Se encontraron {len(archivos_csv)} archivos para procesar.")

    for archivo in archivos_csv:
        ruta_entrada = os.path.join(input_dir, archivo)
        ruta_salida = os.path.join(output_dir, f"analisis_{archivo}")

        print(f"\nProcesando: {archivo} ...")

        try:
            # Leer el CSV
            df = pd.read_csv(ruta_entrada)

            # Verificar si la columna existe
            if COLUMNA_TEXTO not in df.columns:
                print(
                    f"Advertencia: La columna '{COLUMNA_TEXTO}' no se encuentra en {archivo}. Se omitirá."
                )
                continue

            # Configurar la barra de progreso de tqdm para Pandas
            tqdm.pandas(desc=f"Progreso {archivo}")

            # Aplicar la función de análisis a la columna correspondiente
            df["sentimiento_ollama"] = df[COLUMNA_TEXTO].progress_apply(
                lambda x: analizar_sentimiento(str(x))
            )

            # Guardar el resultado en el directorio de salida
            df.to_csv(ruta_salida, index=False, encoding="utf-8")
            print(f"Éxito. Archivo guardado en: {ruta_salida}")

        except Exception as e:
            print(f"Error procesando el archivo {archivo}: {e}")


if __name__ == "__main__":
    print(f"Iniciando análisis de sentimientos con el modelo local: {MODELO_OLLAMA}")
    procesar_directorio(DIRECTORIO_ENTRADA, DIRECTORIO_SALIDA)
    print("\nProceso finalizado.")
