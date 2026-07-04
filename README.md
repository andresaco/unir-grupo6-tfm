# PROYECTO FIN DE MASTER

- Grupo 6

## 🏛️ End-to-End ML Pipeline Architecture

Este repositorio contiene la arquitectura de referencia para el desarrollo local de un ciclo completo de Machine Learning (MloPs). El diseño modular separa estrictamente el flujo de datos (**ETL**), el ciclo de experimentación (**Training**) y el despliegue operativo (**Inferencia**), utilizando **Dagster** como orquestador central basado en assets.

## 🏗️ Estructura del Directorio

```text
mi-proyecto-ml/
├── .gitignore               # Exclusiones de Git (datos, modelos y entornos locales)
├── README.md                # Documentación principal del proyecto
├── requirements.txt         # Dependencias del entorno de Python
├── config/                  # Archivos de configuración centralizada
│   ├── dagster.yaml         # Configuración de la instancia local de Dagster
│   └── model_config.yaml    # Parámetros, hiperparámetros y metadatos del modelo
├── data/                    # Almacenamiento local de datos (¡Excluido de Git / Tracks de DVC!)
│   ├── 01_raw/              # Datos puros e inmutables extraídos por la ETL
│   ├── 02_processed/        # Datos limpios, normalizados y tipados
│   └── 03_features/         # Tablas de características de ingeniería listas para el modelo
├── src/                     # Código fuente modularizado del proyecto
│   ├── __init__.py
│   ├── etl/                 # Módulo de Ingesta y Transformación (Dagster)
│   │   ├── __init__.py
│   │   ├── assets.py        # Definición de Software-Defined Assets (SDAs)
│   │   ├── jobs.py          # Definición de pipelines y ejecuciones programadas
│   │   └── resources.py     # Recursos e infraestructura (Conexiones a APIs, DBs)
│   ├── training/            # Módulo de Modelado y Trazabilidad (MLflow)
│   │   ├── __init__.py
│   │   ├── train.py         # Script principal de ejecución del entrenamiento
│   │   ├── pipeline.py      # Pipelines de ingeniería de variables técnicos
│   │   └── evaluate.py      # Cálculo de métricas, validación y testing del modelo
│   ├── inference/           # Módulo de Operacionalización (Batch / API)
│   │   ├── __init__.py
│   │   ├── predict.py       # Lógica interna de carga de artefactos y scoring
│   │   └── app.py           # Endpoint HTTP para servicio online en tiempo real (FastAPI)
│   └── utils/               # Funciones compartidas transversales
│       └── helpers.py       # Utilidades de logging, I/O y formateo de datos
├── notebooks/               # Análisis exploratorio inicial (EDA) y prototipado rápido
│   └── 01_exploratory_analysis.ipynb
└── models/                  # Artefactos locales serializados (¡Excluido de Git!)
    ├── model_v1.pkl         # Binario o pesos del modelo entrenado
    └── encoder.pkl          # Transformadores de variables categóricas o escaladores
```

## **🚀 Guía de Operación: Ejecución y Parametrización de Pipelines**

Este documento describe cómo lanzar los diferentes flujos de trabajo (pipelines) de nuestro proyecto MLOps utilizando Dagster, así como las instrucciones para parametrizar la extracción de datos desde la interfaz y desde código.

### **1. Levantar el Orquestador (Dagster UI)**

El punto de entrada centralizado de nuestro proyecto es el archivo src/definitions.py. Para visualizar y ejecutar los pipelines, primero debemos levantar el servidor local de Dagster.

Asegúrate de estar en la raíz de tu proyecto y ejecuta el siguiente comando (que configura automáticamente la persistencia de datos en `runtime/dagster`):

```bash
uv run dagster-dev
```

Esto levantará una interfaz web accesible desde tu navegador en **http://127.0.0.1:3000**.

## **2. Ejecución de Pipelines**

Dentro de la interfaz web (pestaña **Assets**), verás todos tus procesos conectados. Dagster te permite ejecutarlos de varias formas:

### **2.1 Ejecutar un solo Asset (Ej: Solo Ingesta)**

1. Haz clic en el nodo raw_stock_data o raw_vix_data.  
2. Haz clic en el botón azul superior derecho **Materialize selected**.  
3. Esto únicamente descargará los datos en data/01_raw/stock/.

### **2.2 Ejecutar un Flujo Completo (End-to-End)**

Si deseas ejecutar todo el proceso de una vez (desde la descarga de Yahoo Finance, pasando por la creación de variables hasta el entrenamiento del modelo):

1. Haz clic en el último nodo del grafo (financial_model_training).  
2. En lugar de darle a "Materialize selected", despliega la flecha junto al botón y selecciona **Materialize selected and all upstream**.  
3. Dagster orquestará automáticamente la ejecución en orden: Ingesta ![][image1] Feature Engineering ![][image1] Training.

## **3. Parametrización de Pipelines**

Nuestros assets de ingesta y *Feature Engineering* están configurados para recibir parámetros (Ticker, Fecha de inicio, Fecha de fin) a través de la clase StockDownloadConfig.

### **Opción A: Parametrizar desde la Interfaz Web (Launchpad)**

Ideal para experimentación puntual.

1. Selecciona los nodos que deseas ejecutar.  
2. Haz clic en la flecha a la derecha del botón azul de Materializar y selecciona **Launchpad**.  
3. Se abrirá un editor YAML. Modifica los parámetros bajo el nodo config del asset correspondiente:

```yaml
ops:  
  raw_stock_data:  
    config:  
      ticker: "MSFT" # Cambiado de AAPL a MSFT  
      initial_date: "2023-01-01"  
      end_date: "2024-01-01"
```

4. Haz clic en el botón inferior derecho **Launch Run**.

*Nota: Dado que varios assets comparten esta configuración (Stock y VIX), Dagster te permite definir estos parámetros globalmente en el Launchpad o para cada operación de forma independiente.*

### **Opción B: Parametrizar mediante Código (Scripts de Automatización)**

Ideal para crear *CRON Jobs* o ejecutar el pipeline como parte de otro proceso de backend.

Crea un archivo temporal (ej. run_custom_pipeline.py) en la raíz del proyecto y utiliza la función materialize de Dagster, inyectando un RunConfig:

```python
from dagster import materialize, RunConfig  
from src.etl.ingestion_assets import raw_stock_data

# Lanzamos el pipeline forzando nuevos parámetros  
resultado = materialize(  
    [raw_stock_data],  
    run_config=RunConfig(  
        ops={  
            "raw_stock_data": {  
                "config": {  
                    "ticker": "TSLA",  
                    "initial_date": "2022-06-01",  
                    "end_date": "2023-06-01"  
                }  
            }  
        }  
    )  
)


if resultado.success:  
    print("Pipeline de descarga ejecutado con éxito para TSLA.")
```

Para ejecutar este script sin levantar la interfaz gráfica:

```bash
uv run run_custom_pipeline.py
```

## **4. Visualizar Modelos Entrenados (MLflow)**

Una vez que el pipeline finaliza con éxito y el asset financial_model_training reporta el estado Success, el modelo y sus métricas quedan guardados.

Para consultar el registro de experimentos, levanta la interfaz de MLflow (que cargará automáticamente los experimentos y modelos guardados en `runtime/mlflow`):

```bash
uv run mlflow-ui
```

Entra en **http://127.0.0.1:5000**, busca el experimento Apple_Stock_Prediction y podrás ver todas las métricas de Accuracy, Precision y descargar el artefacto .pkl validado.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAABC0lEQVR4AezSsQqCQBgH8DQcrKWpllDUB4iSXiJoCBqiXqVH6BGCpmpsCHqGlqAWm0xcHJ2iRdH+OhwYV3AfBQ3Kd9x34v3u7jvlyg+eEv1+Uf+nppZlNT+dj7TTKIomtm0r72ASCqwdhmEXPTdIqCRJqzRNB4ZhtHgqCfV9/wrsnCTJ2DTNPvKCkw+w4lDX9alIw07rwEaA55jXQc4iR9lIIJFluYbPq0BPyBPkLHLU87w9jrQRaFvsNEOXcRwvMP/CRCQ5il4oNE3rYYcN/Fq7IAger5NJKG5/pijKmgdmC5BQ1PDouu4tA3iNhKqqegBWuByMWZBQx3HuTOAkJJTjFF6VaKEcXxk8AQAA//90n8b2AAAABklEQVQDAFsyeDE7uSH3AAAAAElFTkSuQmCC>