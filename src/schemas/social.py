from pydantic import BaseModel
from datetime import date as dt_date, datetime as dt_datetime
from typing import Union


class GdeltSentimentRow(BaseModel):
    fecha: Union[str, dt_date]
    volumen_noticias: int
    sentimiento_promedio: float
    puntuacion_positiva: float
    puntuacion_negativa: float
    polaridad_promedio: float
    volatilidad_sentimiento: float
    uso_primera_persona: float


class SocialRawRow(BaseModel):
    ID_Tweet: str
    Fecha_UTC: Union[str, dt_datetime]
    Contenido_Texto: str
    Retweets: Union[int, str]
    Favoritos: Union[int, str]


class SocialProcessedRow(BaseModel):
    tweet_id: str
    fecha_utc: Union[str, dt_datetime]
    contenido_texto: str
    retweets: int
    favoritos: int
    fecha_limpia: Union[str, dt_date]


class SocialSentimentRow(SocialProcessedRow):
    puntuacion_sentimiento: float
    sentimiento: str


class SocialAggregatedRow(BaseModel):
    fecha_limpia: Union[str, dt_date]
    volumen_posts: int
    sentimiento_medio: float
    sentimiento_std: float
