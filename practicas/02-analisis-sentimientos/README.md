# Análisis de Sentimientos Local (Optimizado para CPU)

Este directorio contiene las herramientas para analizar el sentimiento (Positivo, Negativo, Neutral) de textos recopilados, ejecutando modelos de Inteligencia Artificial de forma local.

El script principal es `analisis_sentimientos.py`, diseñado específicamente para maximizar el rendimiento en equipos que utilizan la **CPU** en lugar de una gráfica dedicada, procesando los datos por lotes (batching) y utilizando un modelo DistilBERT ultra-rápido multilingüe (`lxyuan/distilbert-base-multilingual-cased-sentiments-student`).

## Requisitos e Instalación

Asegúrate de instalar todas las dependencias requeridas antes de ejecutar cualquiera de los scripts de este directorio.

```bash
pip install -r requirements.txt
```

*Nota: La primera vez que se ejecute el script de análisis, se descargarán los pesos del modelo de Hugging Face automáticamente (aprox. 270 MB).*

## Uso de `analisis_sentimientos.py`

El script ofrece dos modos de uso para tu comodidad:

### 1. Modo Interactivo (Recomendado)
Si simplemente ejecutas el script sin pasarle ninguna opción, se activará el modo guiado. El sistema detectará automáticamente tu hardware y te pedirá la ruta del CSV y la columna a analizar de forma amigable en la consola:

```bash
python analisis_sentimientos.py
```

### 2. Modo Línea de Comandos (CLI)
Si prefieres usarlo en una sola línea o desde otros scripts automatizados, puedes especificar los argumentos:

```bash
python analisis_sentimientos.py --file ruta/al/archivo.csv --col "Contenido_Texto" --batch 16
```

**Argumentos disponibles:**
- `--file`: Ruta absoluta o relativa al archivo CSV que contiene los textos.
- `--col`: Nombre exacto de la columna en el CSV que se enviará al modelo (por ejemplo, `Contenido_Texto`).
- `--batch`: *(Opcional)* Cantidad de textos que el procesador evalúa en paralelo. El valor por defecto es **16** (Recomendado para CPU).

## Resultados

Al finalizar el proceso, el script creará automáticamente un **nuevo archivo CSV** en la misma carpeta que el original, añadiendo un *timestamp* (fecha y hora) a su nombre para evitar sobreescribir datos.

Este nuevo CSV contendrá dos columnas adicionales:
- `pred_sentiment`: La predicción obtenida (positive, negative, neutral, o ERROR).
- `pred_score`: Nivel de confianza del modelo sobre esa predicción (valor entre 0.0000 y 1.0000).

Además, la consola te mostrará un resumen claro con el tiempo de ejecución, la velocidad promedio (textos por segundo) y la distribución final de sentimientos detectada.