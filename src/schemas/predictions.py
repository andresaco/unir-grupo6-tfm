from pydantic import BaseModel
from datetime import date as dt_date
from typing import Union


class TradingSignalsRow(BaseModel):
    Date: Union[str, dt_date]
    price_close: float
    predicted_signal: int
    confidence: float
    execution_date: str
