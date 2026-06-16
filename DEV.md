# **🔄 Flujo de Trabajo y Guía de Inicio Rápido**

Sigue estos pasos para configurar tu entorno local y ejecutar el pipeline completo utilizando **uv**.

## **1. Instalación de uv**

uv es un instalador y gestor de paquetes de Python extremadamente rápido que sustituye a pip, pip-tools y virtualenv. Instálalo en tu sistema según tu sistema operativo:

* **En macOS/Linux:**  

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

* **En Windows (PowerShell):**  

```bash
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

Una vez instalado, reinicia tu terminal o ejecuta source $HOME/.local/bin/env (en Linux/macOS) para asegurar que el comando esté disponible.

## **2. Inicialización del Entorno e Instalación de Dependencias**

uv gestionará automáticamente el entorno virtual por ti. Para instalar todas las dependencias definidas en el proyecto (incluyendo Dagster, FastAPI y las librerías de ML), ejecuta en la raíz del proyecto:

```bash
uv pip install -r requirements.txt
```

*Nota: Si estás utilizando un archivo de configuración como pyproject.toml, puedes usar simplemente el comando uv sync para sincronizar tu entorno de manera exacta.*

## **3. Ejecución del Job de ETL (Pipeline de Datos)**

Con las dependencias instaladas, puedes lanzar el pipeline de orquestación de datos. uv permite ejecutar scripts de Python asegurando que se use el entorno virtual correcto de forma aislada mediante el comando uv run.

Para ejecutar el job de la ETL de Dagster de forma directa, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
uv run src/etl/jobs.py
```

Tras la ejecución del job, verás cómo se pueblan tus directorios locales de datos (data/01_raw, data/02_processed y data/03_features), dejando el set de datos listo para el módulo de entrenamiento (src/training/train.py).