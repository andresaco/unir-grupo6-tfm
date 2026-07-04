from pydantic import BaseModel
from datetime import date as dt_date
from typing import Union


class StockRawRow(BaseModel):
    Date: Union[str, dt_date]
    Close: float
    High: float
    Low: float
    Open: float
    Volume: int


class StockProcessedRow(BaseModel):
    date: Union[str, dt_date]
    close: float
    high: float
    low: float
    open: float
    volume: int
