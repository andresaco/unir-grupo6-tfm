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
