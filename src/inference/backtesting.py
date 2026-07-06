import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from dagster import (
    asset,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    Config,
)

# Configuración del backend store de MLflow
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///runtime/mlflow/mlflow.db"


class BacktestConfig(Config):
    """
    Configuración para ejecutar el backtesting de múltiples modelos.
    """

    ticker: str = "AAPL"
    initial_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    capital_inicial: float = 100000.0
    model_names: list[str] = [
        "Apple_Trading_Model",
        "Apple_XGBoost_Model",
        "Apple_LSTM_Model",
    ]


def load_model_helper(model_name: str, context: AssetExecutionContext):
    """
    Carga un modelo desde el Model Registry de MLflow intentando usar su stage 'Production'
    o la versión 'latest' como respaldo. Intenta usar el cargador específico de su sabor.
    """
    model_uri_production = f"models:/{model_name}/Production"
    model_uri_latest = f"models:/{model_name}/latest"

    selected_uri = None
    client = MlflowClient()

    # Intentar buscar si existe versión en Production
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            selected_uri = model_uri_production
            context.log.info(
                f"Cargando modelo '{model_name}' desde Production stage..."
            )
        else:
            selected_uri = model_uri_latest
            context.log.info(
                f"Cargando modelo '{model_name}' desde la versión Latest..."
            )
    except Exception:
        selected_uri = model_uri_latest
        context.log.info(
            f"Cargando modelo '{model_name}' desde la versión Latest (fallback)..."
        )

    # Carga sabor-específica con fallbacks
    if "XGBoost" in model_name:
        try:
            import mlflow.xgboost

            return mlflow.xgboost.load_model(selected_uri), selected_uri
        except Exception as e:
            context.log.warning(
                f"Fallo al cargar XGBoost con cargador específico: {e}. Intentando pyfunc..."
            )
    elif "LSTM" in model_name:
        try:
            import mlflow.pytorch

            return mlflow.pytorch.load_model(selected_uri), selected_uri
        except Exception as e:
            context.log.warning(
                f"Fallo al cargar PyTorch con cargador específico: {e}. Intentando pyfunc..."
            )

    try:
        return mlflow.pyfunc.load_model(selected_uri), selected_uri
    except Exception:
        try:
            import mlflow.sklearn

            return mlflow.sklearn.load_model(selected_uri), selected_uri
        except Exception as e:
            raise RuntimeError(f"Error crítico al cargar el modelo '{model_name}': {e}")


def predict_helper(
    model, model_name: str, X_inference: pd.DataFrame, context: AssetExecutionContext
) -> np.ndarray:
    """
    Ejecuta predicciones adaptando el formato de entrada y escalado según el tipo de modelo.
    """
    if "traditional" in model_name.lower():
        context.log.info(
            f"El modelo '{model_name}' es tradicional. Eliminando columnas de sentimiento."
        )
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
        X_inference = X_inference.drop(
            columns=[col for col in social_cols if col in X_inference.columns],
            errors="ignore",
        )

    # 1. Caso del modelo LSTM (PyTorch) - Requiere 3D inputs [batch, seq_len, features] y escalado
    if "LSTM" in model_name:
        context.log.info(
            "Preparando secuencias 3D y escalado local para el modelo LSTM..."
        )
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_inference)

        seq_length = 10
        xs = []
        # Crear secuencias con padding para los primeros días para mantener el tamaño de salida idéntico
        for i in range(len(X_scaled)):
            if i < seq_length - 1:
                pad_count = seq_length - 1 - i
                padding = np.repeat(X_scaled[0:1], pad_count, axis=0)
                x = np.concatenate([padding, X_scaled[0 : i + 1]], axis=0)
            else:
                x = X_scaled[i - seq_length + 1 : i + 1]
            xs.append(x)

        xs = np.array(xs)  # Shape: (batch_size, 10, num_features)

        import torch

        # Asegurar que el modelo es de PyTorch y llamar a eval()
        if hasattr(model, "eval"):
            model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(xs, dtype=torch.float32)
                outputs = model(tensor_x).numpy()
                predictions = (outputs > 0.5).astype(int).flatten()
            return predictions
        else:
            # Si se cargó mediante pyfunc
            context.log.info(
                "Modelo LSTM cargado como PyFunc. Ejecutando predicción genérica..."
            )
            # mlflow pyfunc predict para pytorch usualmente espera numpy arrays
            outputs = model.predict(xs)
            return (outputs > 0.5).astype(int).flatten()

    # 2. Caso de XGBoost - Requiere escalado local
    elif "XGBoost" in model_name:
        context.log.info("Aplicando escalado local para el modelo XGBoost...")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_input = scaler.fit_transform(X_inference)

        # Ejecutar predicción
        if hasattr(model, "predict"):
            preds = model.predict(X_input)
            if preds.dtype.kind in ("f", "c"):
                return (preds > 0.5).astype(int)
            return preds.astype(int)
        else:
            raise AttributeError(
                f"El modelo XGBoost '{model_name}' no posee la función predict."
            )

    # 3. Caso de RandomForest - Requiere datos tal cual sin escalado
    else:
        context.log.info("Ejecutando predicción directa para el modelo RandomForest...")
        X_input = X_inference.values if hasattr(X_inference, "values") else X_inference
        preds = model.predict(X_input)
        if preds.dtype.kind in ("f", "c"):
            return (preds > 0.5).astype(int)
        return preds.astype(int)


@asset(
    deps=["engineered_features"],
    group_name="inference",
    description="Ejecuta el backtest comparativo sobre todos los modelos de Machine Learning registrados en MLflow.",
)
def run_backtest(
    context: AssetExecutionContext, config: BacktestConfig
) -> MaterializeResult:
    """
    Asset de backtesting modular. Carga el dataset de características,
    evalúa cada modelo dentro del rango de fechas indicado, guarda los
    reportes individuales, y registra las métricas comparativas.
    """
    # 1. Cargar el dataset de características desde el pipeline de ETL
    features_path = f"data/03_features/features/{config.ticker}/features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"No se encontró el dataset de características en {features_path}. Ejecuta el pipeline ETL primero."
        )

    context.log.info(f"Cargando dataset de características desde {features_path}...")
    df = pd.read_csv(features_path)

    # Convertir índice a fecha y ordenar
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # 2. Filtrar por rango de fechas del backtest
    start_date = pd.to_datetime(config.initial_date)
    end_date = pd.to_datetime(config.end_date)
    df_filtered = df.loc[start_date:end_date].copy()

    if df_filtered.empty:
        raise ValueError(
            f"El DataFrame está vacío para el rango de fechas {config.initial_date} a {config.end_date}."
        )

    # Separar predictores numéricos
    X_inference = df_filtered.drop(columns=["target_direction"], errors="ignore")
    X_inference = X_inference.select_dtypes(include=["number"])

    # Obtener el precio de cierre y retornos base para el cálculo del Buy & Hold
    df_filtered["Returns"] = df_filtered["close"].pct_change().fillna(0)
    df_filtered["Buy_Hold_Equity"] = (
        config.capital_inicial * (1 + df_filtered["Returns"]).cumprod()
    )

    # Resultados globales para el markdown y archivo de comparación
    comparison_results = []
    comparison_file = "runtime/backtest_comparison.csv"

    # 3. Iterar y evaluar cada modelo
    for model_name in config.model_names:
        context.log.info(f"=== INICIANDO BACKTEST PARA MODELO: {model_name} ===")

        try:
            # Cargar el modelo
            model, selected_uri = load_model_helper(model_name, context)

            # Obtener predicciones de señales
            signals = predict_helper(model, model_name, X_inference, context)

            # Crear DataFrame local para los cálculos de este modelo
            model_df = df_filtered.copy()
            model_df["predicted_signal"] = signals

            # Calcular retornos de la estrategia (desplazamiento de 1 día para evitar lookahead bias)
            model_df["Strategy_Returns"] = model_df["Returns"] * model_df[
                "predicted_signal"
            ].shift(1).fillna(0)
            model_df["Strategy_Equity"] = (
                config.capital_inicial * (1 + model_df["Strategy_Returns"]).cumprod()
            )

            # Identificar señales para trades y win rate
            model_df["Position"] = model_df["predicted_signal"].diff()
            model_df["Buy_Signal"] = model_df["Position"] == 1
            model_df["Sell_Signal"] = model_df["Position"] == -1

            # Métricas básicas
            capital_inicial = config.capital_inicial
            retorno_total_bh = (
                model_df["Buy_Hold_Equity"].iloc[-1] / capital_inicial - 1
            ) * 100
            retorno_total_est = (
                model_df["Strategy_Equity"].iloc[-1] / capital_inicial - 1
            ) * 100

            # Sharpe Ratio
            std_bh = model_df["Returns"].std()
            std_est = model_df["Strategy_Returns"].std()
            sharpe_bh = (
                np.sqrt(252) * model_df["Returns"].mean() / std_bh if std_bh != 0 else 0
            )
            sharpe_est = (
                np.sqrt(252) * model_df["Strategy_Returns"].mean() / std_est
                if std_est != 0
                else 0
            )

            # Sortino Ratio
            downside_returns_bh = model_df["Returns"][model_df["Returns"] < 0]
            downside_returns_est = model_df["Strategy_Returns"][
                model_df["Strategy_Returns"] < 0
            ]
            std_down_bh = downside_returns_bh.std()
            std_down_est = downside_returns_est.std()
            sortino_bh = (
                np.sqrt(252) * model_df["Returns"].mean() / std_down_bh
                if std_down_bh != 0
                else 0
            )
            sortino_est = (
                np.sqrt(252) * model_df["Strategy_Returns"].mean() / std_down_est
                if std_down_est != 0
                else 0
            )

            # Max Drawdown
            rolling_max_bh = model_df["Buy_Hold_Equity"].expanding().max()
            drawdown_bh = (
                model_df["Buy_Hold_Equity"] - rolling_max_bh
            ) / rolling_max_bh
            max_drawdown_bh = drawdown_bh.min() * 100

            rolling_max_est = model_df["Strategy_Equity"].expanding().max()
            drawdown_est = (
                model_df["Strategy_Equity"] - rolling_max_est
            ) / rolling_max_est
            max_drawdown_est = drawdown_est.min() * 100

            # Calmar Ratio
            calmar_bh = (
                retorno_total_bh / abs(max_drawdown_bh) if max_drawdown_bh != 0 else 0
            )
            calmar_est = (
                retorno_total_est / abs(max_drawdown_est)
                if max_drawdown_est != 0
                else 0
            )

            # Trades y Win Rate
            num_trades = int(model_df["Buy_Signal"].sum())
            trades_df = model_df[
                model_df["Buy_Signal"] | model_df["Sell_Signal"]
            ].copy()

            if len(trades_df) > 1:
                trade_returns = []
                for i in range(0, len(trades_df) - 1, 2):
                    if i + 1 < len(trades_df):
                        buy_price = trades_df.iloc[i]["close"]
                        sell_price = trades_df.iloc[i + 1]["close"]
                        trade_return = (sell_price - buy_price) / buy_price
                        trade_returns.append(trade_return)

                win_rate = (
                    sum(1 for r in trade_returns if r > 0) / len(trade_returns) * 100
                    if trade_returns
                    else 0
                )
            else:
                win_rate = 0.0

            # Guardar reporte de equidad diario individual
            backtests_dir = "data/05_backtests"
            os.makedirs(backtests_dir, exist_ok=True)
            report_path = os.path.join(
                backtests_dir, f"{config.ticker}_{model_name}_backtest.csv"
            )
            model_df.reset_index().to_csv(report_path, index=False)

            # Coleccionar resumen
            summary = {
                "model_name": model_name,
                "strategy_return": float(retorno_total_est),
                "bh_return": float(retorno_total_bh),
                "strategy_sharpe": float(sharpe_est),
                "bh_sharpe": float(sharpe_bh),
                "strategy_sortino": float(sortino_est),
                "bh_sortino": float(sortino_bh),
                "strategy_max_dd": float(max_drawdown_est),
                "bh_max_dd": float(max_drawdown_bh),
                "strategy_calmar": float(calmar_est),
                "bh_calmar": float(calmar_bh),
                "num_trades": int(num_trades),
                "win_rate": float(win_rate),
                "report_path": report_path,
            }

            comparison_results.append(summary)

            # 4. Actualizar registro en MLflow
            # Inyectar métricas en la ejecución de origen (si existe)
            client = MlflowClient()
            run_id = None
            try:
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if not versions:
                    versions = client.get_latest_versions(model_name, stages=["None"])
                if versions:
                    run_id = versions[0].run_id
            except Exception as e:
                context.log.warning(
                    f"No se pudo recuperar la versión del modelo '{model_name}' en MLflow: {e}"
                )

            if run_id:
                context.log.info(
                    f"Registrando métricas de backtest en la run de origen {run_id} en MLflow..."
                )
                try:
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_metric(
                            "backtest_return_strategy", float(retorno_total_est)
                        )
                        mlflow.log_metric("backtest_return_bh", float(retorno_total_bh))
                        mlflow.log_metric("backtest_sharpe_strategy", float(sharpe_est))
                        mlflow.log_metric("backtest_sharpe_bh", float(sharpe_bh))
                        mlflow.log_metric(
                            "backtest_max_dd_strategy", float(max_drawdown_est)
                        )
                        mlflow.log_metric("backtest_max_dd_bh", float(max_drawdown_bh))
                        mlflow.log_metric("backtest_win_rate", float(win_rate))
                        mlflow.log_metric("backtest_num_trades", float(num_trades))
                except Exception as e:
                    context.log.warning(
                        f"Error al registrar métricas en la run original de MLflow: {e}"
                    )

            # Registrar también en el experimento de comparación Backtests
            try:
                mlflow.set_experiment("Backtest_Comparison")
                with mlflow.start_run(
                    run_name=f"Backtest_{model_name}_{config.initial_date}_{config.end_date}"
                ):
                    mlflow.log_params(
                        {
                            "model_name": model_name,
                            "ticker": config.ticker,
                            "start_date": config.initial_date,
                            "end_date": config.end_date,
                            "capital_inicial": config.capital_inicial,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "strategy_return_pct": float(retorno_total_est),
                            "bh_return_pct": float(retorno_total_bh),
                            "strategy_sharpe": float(sharpe_est),
                            "bh_sharpe": float(sharpe_bh),
                            "strategy_max_dd_pct": float(max_drawdown_est),
                            "bh_max_dd_pct": float(max_drawdown_bh),
                            "strategy_calmar": float(calmar_est),
                            "num_trades": float(num_trades),
                            "win_rate_pct": float(win_rate),
                        }
                    )
            except Exception as e:
                context.log.warning(
                    f"Error al registrar ejecución en el experimento Backtest_Comparison: {e}"
                )

        except Exception as e:
            context.log.error(f"Error evaluando el modelo '{model_name}': {e}")
            continue

    # 5. Generar comparación global y guardar en CSV acumulado
    if not comparison_results:
        raise RuntimeError(
            "No se pudo evaluar con éxito ninguno de los modelos provistos."
        )

    new_data_df = pd.DataFrame(
        [
            {
                "model_name": r["model_name"],
                "ticker": config.ticker,
                "start_date": config.initial_date,
                "end_date": config.end_date,
                "strategy_return_pct": r["strategy_return"],
                "bh_return_pct": r["bh_return"],
                "strategy_sharpe": r["strategy_sharpe"],
                "bh_sharpe": r["bh_sharpe"],
                "strategy_max_dd_pct": r["strategy_max_dd"],
                "bh_max_dd_pct": r["bh_max_dd"],
                "strategy_calmar": r["strategy_calmar"],
                "num_trades": r["num_trades"],
                "win_rate_pct": r["win_rate"],
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            for r in comparison_results
        ]
    )

    if os.path.exists(comparison_file):
        existing_df = pd.read_csv(comparison_file)
        # Eliminar filas duplicadas para los mismos modelos y fechas en esta ejecución
        model_names_set = set(config.model_names)
        mask = (
            (existing_df["model_name"].isin(model_names_set))
            & (existing_df["start_date"] == config.initial_date)
            & (existing_df["end_date"] == config.end_date)
        )
        existing_df = existing_df[~mask]
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        combined_df = new_data_df
    combined_df.to_csv(comparison_file, index=False)

    # 6. Crear reporte Markdown de comparación para mostrar en la interfaz de Dagster
    md_lines = [
        f"# Comparativa de Backtesting para {config.ticker} ({config.initial_date} a {config.end_date})",
        "",
        "| Modelo | Retorno Estrategia | Retorno B&H | Sharpe Ratio (Estr) | Max Drawdown (Estr) | Calmar Ratio | N° Trades | Win Rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in comparison_results:
        md_lines.append(
            f"| **{r['model_name']}** | {r['strategy_return']:.2f}% | {r['bh_return']:.2f}% | "
            f"{r['strategy_sharpe']:.2f} | {r['strategy_max_dd']:.2f}% | "
            f"{r['strategy_calmar']:.2f} | {r['num_trades']} | {r['win_rate']:.2f}% |"
        )
    comparison_md = "\n".join(md_lines)

    # 7. Retornar los metadatos agregados
    metadata_payload = {
        "comparativa_modelos": MetadataValue.md(comparison_md),
        "comparison_report_path": MetadataValue.path(comparison_file),
    }
    # Añadir enlaces individuales de los reportes generados a los metadatos
    for r in comparison_results:
        metadata_payload[f"path_{r['model_name']}"] = MetadataValue.path(
            r["report_path"]
        )
        metadata_payload[f"return_{r['model_name']}"] = MetadataValue.float(
            r["strategy_return"]
        )
        metadata_payload[f"sharpe_{r['model_name']}"] = MetadataValue.float(
            r["strategy_sharpe"]
        )

    return MaterializeResult(metadata=metadata_payload)
