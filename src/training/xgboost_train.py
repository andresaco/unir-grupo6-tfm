import pandas as pd
import os
import mlflow
import mlflow.xgboost
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from ..schemas import validate_df, EngineeredFeaturesRow

# Configuramos la ruta de MLflow para que guarde los experimentos en local
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class XGBoostTrainingConfig(Config):
    """
    Clase para parametrizar el entrenamiento del modelo XGBoost.
    """

    name: str = "Apple"
    ticker: str = "AAPL"
    initial_date: str = "2023-12-01"
    end_date: str = "2023-12-31"

    # Hiperparámetros de XGBoost
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.01
    random_state: int = 42


@asset(
    group_name="training",
    description="Entrena el modelo XGBoost y registra los artefactos y métricas en MLflow.",
)
def xgboost_model_training(
    context: AssetExecutionContext, config: XGBoostTrainingConfig
) -> MaterializeResult:
    """
    Asset que lee las características preparadas, filtra por rango de fechas (criba),
    aplica normalización y SMOTE, entrena un modelo XGBoost, y lo registra en MLflow.
    """
    ticker = config.ticker
    features_path = f"data/03_features/features/{ticker}/features.csv"

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No se encontraron datos de entrenamiento en {features_path}. Ejecuta la ETL primero."
        )

    df = pd.read_csv(features_path)

    # Validar features cargadas
    validate_df(
        df, EngineeredFeaturesRow, stage="xgboost_model_training (read features)"
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

    # Filtramos X para quedarnos SOLO con columnas numéricas
    X = X.select_dtypes(include=["number"])

    # División temporal (no aleatoria, vital en series financieras)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Normalización de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Aplicar SMOTE para balancear clases
    smote = SMOTE(random_state=config.random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Configurar MLflow
    experiment_name = config.name
    try:
        # Se crea con artifact_location explícito para que no se cree en la raíz
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=os.path.abspath("runtime/mlflow/mlruns"),
        )
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="XGBoost_Classifier") as run:
        # Definir hiperparámetros
        params = {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "random_state": config.random_state,
            "eval_metric": "logloss",
        }
        mlflow.log_params(params)

        # Entrenamiento del Modelo
        context.log.info("Iniciando el entrenamiento del modelo XGBoost...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced)

        # Evaluación
        predictions = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        context.log.info(
            f"Métricas obtenidas - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}"
        )

        # Registrar el modelo en MLflow
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost_model",
            registered_model_name=f"{config.name}_XGBoost_Model",
        )

        model_uri = model_info.model_uri
        context.log.info(f"Modelo registrado exitosamente con URI: {model_uri}")

        # Evaluar el modelo con MLflow y calcular SHAP
        eval_data = pd.DataFrame(X_test_scaled, columns=X.columns)
        eval_data["target_direction"] = y_test.values

        context.log.info(
            "Iniciando la evaluación de XGBoost con MLflow y cálculo de SHAP..."
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

    return MaterializeResult(
        metadata={
            "mlflow_run_id": MetadataValue.text(run.info.run_id),
            "model_uri": MetadataValue.text(model_uri),
            "accuracy": MetadataValue.float(accuracy),
            "precision": MetadataValue.float(precision),
            "f1_score": MetadataValue.float(f1),
            "training_samples": MetadataValue.int(len(X_train_balanced)),
            "test_samples": MetadataValue.int(len(X_test)),
        }
    )
