import pandas as pd
import os
import mlflow
import mlflow.sklearn
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from ..schemas import validate_df, EngineeredFeaturesRow

# Configuramos la ruta de MLflow para que guarde los experimentos en local
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class TrainingConfig(Config):
    """
    Clase para parametrizar el entrenamiento del modelo.
    Contiene el nombre de la ejecución/experimento, fechas para la criba de datos
    e hiperparámetros de RandomForestClassifier para tuning.
    """

    name: str = "Apple"
    ticker: str = "AAPL"
    initial_date: str = "2023-12-01"  # Fecha de inicio para criba de datos
    end_date: str = "2023-12-31"  # Fecha de fin para criba de datos

    # Parámetros para hiperparameter tuning del RandomForestClassifier
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42
    class_weight: str = "balanced"


@asset(
    group_name="training",
    description="Entrena el modelo RandomForest tradicional (sin sentimientos) y registra los artefactos y métricas en MLflow.",
)
def rf_traditional_training(
    context: AssetExecutionContext, config: TrainingConfig
) -> MaterializeResult:
    """
    Asset que lee las características preparadas, filtra por rango de fechas (criba),
    descarta columnas de sentimientos/redes sociales, entrena un modelo predictivo
    con RandomForest, y lo registra en el Model Registry de MLflow.
    """
    ticker = config.ticker
    # 1. Cargar las features (Asumimos que un pipeline ETL previo las guardó aquí)
    features_path = f"data/03_features/features/{ticker}/features.csv"

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No se encontraron datos de entrenamiento en {features_path}. Ejecuta la ETL primero."
        )

    df = pd.read_csv(features_path)

    # Validar features cargadas
    validate_df(
        df, EngineeredFeaturesRow, stage="rf_traditional_training (read features)"
    )

    # Criba de datos basada en las fechas especificadas en la configuración
    if "date" in df.columns:
        df = df[(df["date"] >= config.initial_date) & (df["date"] <= config.end_date)]
        df = df.set_index("date")
    else:
        df = df.loc[config.initial_date : config.end_date]

    if df.empty:
        raise ValueError(
            f"El conjunto de datos filtrado entre {config.initial_date} y {config.end_date} está vacío."
        )

    y = df["target_direction"]
    X = df.drop(columns=["target_direction"])

    # Descarta datos con origen redes sociales
    social_cols = [
        "fecha",
        "volumen_noticias",
        "sentimiento_promedio",
        "puntuacion_positiva",
        "puntuacion_negativa",
        "polaridad_promedio",
        "volatilidad_sentimiento",
        "uso_primera_persona",
        "sentiment_volume_interaction",
    ]
    X = X.drop(
        columns=[col for col in social_cols if col in X.columns], errors="ignore"
    )

    # Filtramos X para quedarnos SOLO con columnas numéricas
    X = X.select_dtypes(include=["number"])

    # División temporal (no aleatoria, vital en series financieras)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 2. Configurar MLflow
    experiment_name = f"{config.name}"
    try:
        # Se crea con artifact_location explícito para que no se cree en la raíz
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=os.path.abspath("runtime/mlflow/mlruns"),
        )
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_Ensemble_Traditional") as run:
        # Definir hiperparámetros dinámicamente desde el config
        params = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "random_state": config.random_state,
            "class_weight": config.class_weight,
        }
        mlflow.log_params(params)

        # 3. Entrenamiento del Modelo
        context.log.info(
            "Iniciando el entrenamiento del modelo Ensemble Tradicional..."
        )
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Evaluación (Backtesting básico en el conjunto de test)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        context.log.info(
            f"Métricas obtenidas - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}"
        )

        # 5. Registrar el modelo en MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name=f"{config.name}_RandomForest_Model_traditional",
        )

        model_uri = model_info.model_uri
        context.log.info(f"Modelo registrado exitosamente con URI: {model_uri}")

        # 6. Evaluar el modelo con MLflow y calcular SHAP
        eval_data = X_test.copy()
        eval_data["target_direction"] = y_test

        context.log.info(
            "Iniciando la evaluación de Random Forest tradicional con MLflow y cálculo de SHAP..."
        )
        mlflow.models.evaluate(
            model=model_uri,
            data=eval_data,
            targets="target_direction",
            model_type="classifier",
            evaluator_config={
                "log_explainer": True,
            },
        )

    # Devolver metadata a la interfaz de Dagster para trazabilidad
    return MaterializeResult(
        metadata={
            "mlflow_run_id": MetadataValue.text(run.info.run_id),
            "model_uri": MetadataValue.text(model_uri),
            "accuracy": MetadataValue.float(accuracy),
            "precision": MetadataValue.float(precision),
            "training_samples": MetadataValue.int(len(X_train)),
            "test_samples": MetadataValue.int(len(X_test)),
        }
    )
