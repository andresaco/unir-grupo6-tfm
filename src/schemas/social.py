from pydantic import BaseModel
from datetime import date as dt_date, datetime as dt_datetime
from typing import Union


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


class SocialSentimentRow(BaseModel):
    tweet_id: str
    fecha_utc: Union[str, dt_datetime]
    contenido_texto: str
    retweets: int
    favoritos: int
    fecha_limpia: Union[str, dt_date]
    puntuacion_sentimiento: float
    sentimiento: str


class SocialAggregatedRow(BaseModel):
    fecha_limpia: Union[str, dt_date]
    volumen_posts: int
    sentimiento_medio: float
    sentimiento_std: float
