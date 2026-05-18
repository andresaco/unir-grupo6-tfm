import os
import sys
import time
from datetime import datetime
import argparse
import warnings

# Desactivar advertencias innecesarias de Hugging Face y PyTorch para mantener limpia la consola
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pandas as pd
import torch
from tqdm import tqdm
import psutil
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Imprime la información de bienvenida y detalles técnicos en consola."""
    print(f"{Colors.BLUE}{Colors.BOLD}" + "="*60)
    print("  ANALIZADOR DE SENTIMIENTOS LOCAL - OPTIMIZADO PARA CPU")
    print("="*60 + f"{Colors.ENDC}")
    print(f"{Colors.BOLD}Modelo recomendado:{Colors.ENDC} lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    print(f"{Colors.BOLD}Hardware Detectado:{Colors.ENDC}")
    print(f"  - CPU Cores: {psutil.cpu_count(logical=True)} (Lógicos) / {psutil.cpu_count(logical=False)} (Físicos)")
    ram = psutil.virtual_memory()
    print(f"  - Memoria RAM Total: {ram.total / (1024**3):.2f} GB (Disponible: {ram.available / (1024**3):.2f} GB)")
    print(f"  - Dispositivo de cálculo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU (Recomendado para tu setup)'}")
    print(f"{Colors.BLUE}" + "-"*60 + f"{Colors.ENDC}\n")

def get_system_usage():
    """Obtiene el uso actual de CPU y RAM para monitoreo."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    return f"CPU: {cpu}% | RAM: {ram}%"

def validate_inputs(file_path, column_name):
    """Valida que el archivo exista, sea un CSV y contenga la columna especificada."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {file_path}")
    
    if not file_path.lower().endswith('.csv'):
        raise ValueError("El archivo debe tener extensión .csv")
    
    # Intentar leer solo las primeras filas para verificar la columna (ahorra RAM)
    try:
        df_head = pd.read_csv(file_path, nrows=5)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo CSV. Detalles: {e}")
        
    if column_name not in df_head.columns:
        cols_available = ", ".join([f"'{c}'" for c in df_head.columns])
        raise KeyError(f"La columna '{column_name}' no existe en el CSV. Columnas disponibles: {cols_available}")
    
    return True

def load_sentiment_pipeline():
    """Carga de manera segura el modelo de NLP."""
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    print(f"{Colors.WARNING}Cargando el modelo '{model_name}' en memoria...{Colors.ENDC}")
    print("*(Si es la primera vez que lo ejecutas, se descargarán aprox. 270 MB de internet)*")
    
    start_time = time.time()
    try:
        # Forzar CPU para evitar problemas de compatibilidad en hardware sin GPU dedicada
        device = 0 if torch.cuda.is_available() else -1
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        nlp_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            truncation=True
        )
        
        elapsed = time.time() - start_time
        print(f"{Colors.GREEN}✔ Modelo cargado con éxito en {elapsed:.2f} segundos.{Colors.ENDC}\n")
        return nlp_pipeline
    except Exception as e:
        print(f"{Colors.FAIL}Error crítico cargando el modelo: {e}{Colors.ENDC}")
        sys.exit(1)

def run_sentiment_analysis(file_path, column_name, batch_size=32):
    """Lee el CSV, ejecuta el análisis por lotes (batches) e instrumenta el proceso."""
    print_banner()
    
    # Validar entradas
    try:
        validate_inputs(file_path, column_name)
    except Exception as e:
        print(f"{Colors.FAIL}Error de validación: {e}{Colors.ENDC}")
        return

    # Cargar datos
    print(f"{Colors.BLUE}Leyendo archivo CSV...{Colors.ENDC}")
    df = pd.read_csv(file_path)
    total_rows = len(df)
    print(f"Total de registros a procesar: {Colors.BOLD}{total_rows}{Colors.ENDC}")
    
    # Cargar el pipeline
    classifier = load_sentiment_pipeline()
    
    # Preparar listas para los resultados
    sentiments = []
    scores = []
    
    print(f"{Colors.BLUE}Iniciando análisis de sentimientos...{Colors.ENDC}")
    start_time = time.time()
    
    # Iterar por lotes (batching) para optimizar el rendimiento de la CPU
    # y evitar picos excesivos de RAM
    with tqdm(total=total_rows, desc="Procesando", unit="filas", colour="green") as pbar:
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Limpiar textos (reemplazar valores nulos o no-string por espacios)
            texts = batch_df[column_name].fillna("").astype(str).tolist()
            
            try:
                # Ejecutar inferencia en el lote actual
                results = classifier(texts)
                
                # Extraer etiquetas y confidencias
                for res in results:
                    sentiments.append(res['label'])
                    scores.append(round(res['score'], 4))
                    
            except Exception as e:
                # Fallback por si una fila causa error crítico de procesamiento
                print(f"\n{Colors.WARNING}Advertencia: Error en lote {i}-{i+batch_size}. Procesando uno a uno...{Colors.ENDC}")
                for text in texts:
                    try:
                        res = classifier(text)[0]
                        sentiments.append(res['label'])
                        scores.append(round(res['score'], 4))
                    except:
                        sentiments.append("ERROR")
                        scores.append(0.0)
            
            # Actualizar barra de progreso e inyectar info del sistema en la descripción
            pbar.update(len(batch_df))
            pbar.set_postfix_str(get_system_usage())

    # Agregar las nuevas columnas al dataframe
    df['pred_sentiment'] = sentiments
    df['pred_score'] = scores
    
    # Generar ruta de salida con Timestamp
    dir_name = os.path.dirname(os.path.abspath(file_path))
    base_name = os.path.basename(file_path)
    file_no_ext, ext = os.path.splitext(base_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{file_no_ext}_analizado_{timestamp}{ext}"
    output_path = os.path.join(dir_name, output_filename)
    
    # Guardar resultados
    print(f"\n{Colors.BLUE}Guardando resultados...{Colors.ENDC}")
    df.to_csv(output_path, index=False)
    
    total_time = time.time() - start_time
    avg_speed = total_rows / total_time if total_time > 0 else 0
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}¡Proceso Completado con éxito!{Colors.ENDC}")
    print("="*60)
    print(f"  - Archivo de entrada: {file_path}")
    print(f"  - Archivo de salida:  {Colors.BOLD}{output_path}{Colors.ENDC}")
    print(f"  - Tiempo total:       {total_time:.2f} segundos")
    print(f"  - Velocidad promedio: {avg_speed:.2f} textos/segundo")
    print("="*60)
    
    # Mostrar distribución de sentimientos rápida
    print(f"\n{Colors.BOLD}Distribución de Sentimientos detectada:{Colors.ENDC}")
    dist = df['pred_sentiment'].value_counts()
    for category, count in dist.items():
        percentage = (count / total_rows) * 100
        print(f"  - {category.upper()}: {count} ({percentage:.1f}%)")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script optimizado de análisis de sentimientos local para CPU.")
    parser.add_argument("--file", type=str, help="Ruta del archivo CSV de entrada.")
    parser.add_argument("--col", type=str, help="Nombre de la columna de texto a analizar.")
    parser.add_argument("--batch", type=int, default=16, help="Tamaño de lote (batch size). Por defecto es 16 (ideal para CPU).")
    
    args = parser.parse_args()
    
    # Si no se proveen argumentos desde consola, solicitarlos interactivamente de forma amigable
    if not args.file or not args.col:
        print_banner()
        print(f"{Colors.WARNING}No se detectaron argumentos de consola. Entrando en modo interactivo...{Colors.ENDC}\n")
        
        file_input = input("1. Introduce la ruta completa de tu archivo CSV: ").strip()
        # Limpiar posibles comillas que el SO añade al arrastrar un archivo a la consola
        file_input = file_input.strip("'\"")
        
        if not os.path.exists(file_input):
            print(f"{Colors.FAIL}El archivo indicado no existe. Saliendo...{Colors.ENDC}")
            sys.exit(1)
            
        try:
            sample_df = pd.read_csv(file_input, nrows=3)
            print(f"\nColumnas detectadas en tu archivo: {Colors.BOLD}{list(sample_df.columns)}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error al leer el CSV: {e}{Colors.ENDC}")
            sys.exit(1)
            
        col_input = input("2. Escribe el nombre exacto de la columna a analizar: ").strip()
        batch_input_str = input("3. Tamaño de lote (Presiona Enter para usar 16 por defecto): ").strip()
        
        batch_input = 16
        if batch_input_str.isdigit():
            batch_input = int(batch_input_str)
            
        run_sentiment_analysis(file_input, col_input, batch_input)
    else:
        run_sentiment_analysis(args.file, args.col, args.batch)