from pydantic import BaseModel
from datetime import date as dt_date
from typing import Optional, Union


class EngineeredFeaturesRow(BaseModel):
    date: Union[str, dt_date]
    close: float
    high: float
    low: float
    open: float
    volume: int
    VIX: float
    daily_return: float
    SMA_10: float
    SMA_50: float
    volatilidad_10d: float
    target_direction: int

    # GDELT columns (can be optional / None if not present or NaN after left merge)
    volumen_noticias: Optional[float] = None
    sentimiento_promedio: Optional[float] = None
    puntuacion_positiva: Optional[float] = None
    puntuacion_negativa: Optional[float] = None
    polaridad_promedio: Optional[float] = None
    volatilidad_sentimiento: Optional[float] = None
    uso_primera_persona: Optional[float] = None
    sentiment_score: Optional[float] = None
