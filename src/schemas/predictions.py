from pydantic import BaseModel
from datetime import date as dt_date
from typing import Union


class TradingSignalsRow(BaseModel):
    date: Union[str, dt_date]
    price_close: float
    predicted_signal: int
    confidence: float
