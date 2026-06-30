from dagster import Config


# Definimos la configuración parametrizable compartida
class StockDownloadConfig(Config):
    name: str = "Apple"
    ticker: str = "AAPL"
    initial_date: str = "2023-12-01"  # Formato YYYY-MM-DD
    end_date: str = "2023-12-31"  # Formato YYYY-MM-DD
