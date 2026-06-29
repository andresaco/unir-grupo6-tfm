from pydantic import BaseModel
from datetime import date as dt_date
from typing import Optional, Union


class EngineeredFeaturesRow(BaseModel):
    Date: Union[str, dt_date]
    Close: float
    High: float
    Low: float
    Open: float
    Volume: int
    VIX_Close: float
    Daily_Return: float
    SMA_10: float
    SMA_50: float
    Volatilidad_10d: float
    target_direction: int

    # GDELT columns (can be optional / None if not present or NaN after left merge)
    fecha: Optional[Union[str, dt_date]] = None
    volumen_noticias: Optional[float] = None
    sentimiento_promedio: Optional[float] = None
    puntuacion_positiva: Optional[float] = None
    puntuacion_negativa: Optional[float] = None
    polaridad_promedio: Optional[float] = None
    volatilidad_sentimiento: Optional[float] = None
    uso_primera_persona: Optional[float] = None
    sentiment_score: Optional[float] = None
