import subprocess
import sys
import os


def launch_dagster():
    """Lanza el servidor de Dagster dev en la raíz del proyecto usando un directorio persistente para runtime."""
    dagster_home = os.path.abspath("runtime/dagster")
    os.makedirs(dagster_home, exist_ok=True)
    os.environ["DAGSTER_HOME"] = dagster_home

    cmd = ["dagster", "dev", "-m", "src.definitions"]
    print(f"Ejecutando con DAGSTER_HOME={dagster_home}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServidor de Dagster dev detenido.")
    except Exception as e:
        print(f"Error al iniciar Dagster: {e}")
        sys.exit(1)


def launch_mlflow():
    """Lanza la interfaz web de MLflow apuntando a la base de datos y artefactos en el directorio runtime."""
    mlflow_dir = os.path.abspath("runtime/mlflow")
    os.makedirs(mlflow_dir, exist_ok=True)

    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///runtime/mlflow/mlflow.db",
        "--default-artifact-root",
        "runtime/mlflow/mlruns",
    ]
    print(f"Ejecutando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServidor de MLflow detenido.")
    except Exception as e:
        print(f"Error al iniciar MLflow: {e}")
        sys.exit(1)


def launch_jupyter():
    """Lanza JupyterLab en el directorio actual (raíz del proyecto)."""
    cmd = ["jupyter", "lab"]
    print(f"Ejecutando: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nJupyterLab detenido.")
    except Exception as e:
        print(f"Error al iniciar JupyterLab: {e}")
        sys.exit(1)
