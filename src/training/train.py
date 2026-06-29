import pandas as pd
import os
import mlflow
import mlflow.sklearn
from dagster import asset, AssetExecutionContext, MaterializeResult, MetadataValue
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from ..schemas import validate_df, EngineeredFeaturesRow

# Configuramos la ruta de MLflow para que guarde los experimentos en local
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"


@asset(
    group_name="training",
    description="Entrena el modelo financiero y registra los artefactos y métricas en MLflow.",
)
def financial_model_training(context: AssetExecutionContext) -> MaterializeResult:
    """
    Asset que lee las características preparadas, entrena un modelo predictivo
    para la dirección del precio y lo registra en el Model Registry de MLflow.
    """
    # 1. Cargar las features (Asumimos que un pipeline ETL previo las guardó aquí)
    features_path = "data/03_features/historical_features.csv"

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No se encontraron datos de entrenamiento en {features_path}. Ejecuta la ETL primero."
        )

    df = pd.read_csv(features_path)

    # Validar features cargadas
    validate_df(
        df, EngineeredFeaturesRow, stage="financial_model_training (read features)"
    )

    # Asumimos que la variable objetivo se llama 'target_direction' (1 = Sube, 0 = Baja)
    # y que la fecha es el índice o no debe usarse para predecir
    if "Date" in df.columns:
        df = df.set_index("Date")

    y = df["target_direction"]
    X = df.drop(columns=["target_direction"])

    # 2. LA CORRECCIÓN: Filtramos X para quedarnos SOLO con columnas numéricas
    X = X.select_dtypes(include=["number"])

    # División temporal (no aleatoria, vital en series financieras)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 2. Configurar MLflow
    mlflow.set_experiment("Apple_Stock_Prediction")

    with mlflow.start_run(run_name="RandomForest_Ensemble") as run:
        # Definir hiperparámetros
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "class_weight": "balanced",
        }
        mlflow.log_params(params)

        # 3. Entrenamiento del Modelo
        context.log.info("Iniciando el entrenamiento del modelo Ensemble...")
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
        # Esto guarda el .pkl internamente y versiona el modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name="Apple_Trading_Model",
        )

        model_uri = model_info.model_uri
        context.log.info(f"Modelo registrado exitosamente con URI: {model_uri}")

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
