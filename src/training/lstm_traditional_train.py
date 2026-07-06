import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.pytorch
import shap
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score
from ..schemas import validate_df, EngineeredFeaturesRow

# Configuramos la ruta de MLflow para que guarde los experimentos en local
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class LSTMConfig(Config):
    """
    Clase para parametrizar el entrenamiento del modelo LSTM.
    """

    name: str = "Apple"
    ticker: str = "AAPL"
    initial_date: str = "2023-12-01"
    end_date: str = "2023-12-31"

    # Parámetros del modelo y entrenamiento
    sequence_length: int = 5
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 15


class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, seq_len):
        super(PyTorchLSTM, self).__init__()
        # First LSTM layer: input_size -> 128
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(128)

        # Second LSTM layer: 128 -> 64
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(64)

        # Third LSTM layer: 64 -> 32
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout3 = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm1d(32)

        # Dense layer: 32 -> 16
        self.fc1 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # Output layer: 16 -> 1
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out = out.transpose(1, 2)
        out = self.bn1(out)
        out = out.transpose(1, 2)

        # LSTM 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out.transpose(1, 2)
        out = self.bn2(out)
        out = out.transpose(1, 2)

        # LSTM 3
        out, _ = self.lstm3(out)
        out = out[:, -1, :]  # Take the last time step output
        out = self.dropout3(out)
        out = self.bn3(out)

        # Dense 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)

        # Output
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


@asset(
    group_name="training",
    description="Entrena la red neuronal LSTM tradicional (sin sentimientos) y registra los artefactos y métricas en MLflow.",
)
def lstm_traditional_training(
    context: AssetExecutionContext, config: LSTMConfig
) -> MaterializeResult:
    """
    Asset que lee las características preparadas, filtra por rango de fechas (criba),
    descarta columnas de sentimientos/redes sociales, prepara secuencias temporales para LSTM,
    entrena una red LSTM usando PyTorch, y la registra en MLflow con el sufijo _traditional.
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
        df, EngineeredFeaturesRow, stage="lstm_traditional_training (read features)"
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

    # Normalización de datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Preparar datos para LSTM (secuencias de longitud config.sequence_length)
    sequence_length = config.sequence_length

    if len(X_train_scaled) <= sequence_length or len(X_test_scaled) <= sequence_length:
        raise ValueError(
            f"La cantidad de datos es demasiado pequeña para la longitud de secuencia {sequence_length}."
        )

    X_train_seq, y_train_seq = [], []
    for i in range(sequence_length, len(X_train_scaled)):
        X_train_seq.append(X_train_scaled[i - sequence_length : i])
        y_train_seq.append(y_train.iloc[i])

    X_test_seq, y_test_seq = [], []
    for i in range(sequence_length, len(X_test_scaled)):
        X_test_seq.append(X_test_scaled[i - sequence_length : i])
        y_test_seq.append(y_test.iloc[i])

    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)
    y_train_seq = np.array(y_train_seq)
    y_test_seq = np.array(y_test_seq)

    # Separar un 20% para validación temporal dentro de train
    val_size = int(len(X_train_seq) * 0.2)
    train_size_seq = len(X_train_seq) - val_size

    X_tr, X_val = X_train_seq[:train_size_seq], X_train_seq[train_size_seq:]
    y_tr, y_val = y_train_seq[:train_size_seq], y_train_seq[train_size_seq:]

    # Crear DataLoaders de PyTorch
    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Configurar MLflow
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

    # Instanciar el modelo
    model = PyTorchLSTM(input_size=X_train.shape[1], seq_len=sequence_length)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCELoss()

    # Early stopping
    patience = config.patience
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # Reduce learning rate callback equivalent
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=0.00001
    )

    context.log.info(
        "Iniciando el entrenamiento del modelo LSTM Tradicional con PyTorch..."
    )

    with mlflow.start_run(run_name="LSTM_NeuralNetwork_Traditional") as run:
        # Registrar parámetros
        params = {
            "sequence_length": config.sequence_length,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "patience": config.patience,
        }
        mlflow.log_params(params)

        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)

            # Validación
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_x = torch.tensor(X_val, dtype=torch.float32)
                val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
                val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                context.log.info(f"Early stopping en la época {epoch}")
                break

        # Cargar los mejores pesos
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluación en test
        model.eval()
        with torch.no_grad():
            test_x = torch.tensor(X_test_seq, dtype=torch.float32)
            test_outputs = model(test_x).numpy()
            predictions = (test_outputs > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test_seq, predictions)
        precision = precision_score(y_test_seq, predictions)
        f1 = f1_score(y_test_seq, predictions)

        # Registrar métricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1_score", f1)

        context.log.info(
            f"Métricas obtenidas - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, F1-Score: {f1:.4f}"
        )

        # Registrar el modelo en MLflow
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="lstm_model",
            registered_model_name=f"{config.name}_LSTM_Model_traditional",
        )

        model_uri = model_info.model_uri
        context.log.info(f"Modelo registrado exitosamente con URI: {model_uri}")

        # Calcular explicabilidad SHAP para el modelo LSTM
        try:
            context.log.info(
                "Calculando explicabilidad SHAP para el modelo LSTM tradicional..."
            )
            num_background = min(100, len(X_tr))
            num_test = min(50, len(X_test_seq))
            background = torch.tensor(X_tr[:num_background], dtype=torch.float32)
            test_samples = torch.tensor(X_test_seq[:num_test], dtype=torch.float32)

            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(test_samples)

            # Procesar shap_values para generar un summary plot en 2D promediando sobre los time steps
            if isinstance(shap_values, list):
                shap_values_array = shap_values[0]
            else:
                shap_values_array = shap_values

            shap_values_2d = shap_values_array.mean(axis=1)
            X_test_2d = test_samples.mean(dim=1).numpy()

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values_2d, X_test_2d, feature_names=X.columns.tolist(), show=False
            )
            plt.tight_layout()

            plot_path = "lstm_traditional_shap_summary.png"
            plt.savefig(plot_path)
            plt.close()

            mlflow.log_artifact(plot_path)
            context.log.info(
                f"Gráfico SHAP guardado y registrado en MLflow en {plot_path}"
            )

            if os.path.exists(plot_path):
                os.remove(plot_path)
        except Exception as e:
            context.log.warning(
                f"No se pudo calcular la explicabilidad SHAP para LSTM tradicional: {e}"
            )

    return MaterializeResult(
        metadata={
            "mlflow_run_id": MetadataValue.text(run.info.run_id),
            "model_uri": MetadataValue.text(model_uri),
            "accuracy": MetadataValue.float(accuracy),
            "precision": MetadataValue.float(precision),
            "f1_score": MetadataValue.float(f1),
            "training_samples": MetadataValue.int(len(X_tr)),
            "test_samples": MetadataValue.int(len(X_test_seq)),
        }
    )
