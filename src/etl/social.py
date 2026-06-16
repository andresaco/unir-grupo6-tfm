from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dagster import Out, job, op
from dotenv import load_dotenv
from transformers import pipeline

from .core.social import SocialSearchClient
from .core.social.bsky import BlueskyClient

load_dotenv()

DEFAULT_QUERY = os.environ.get("SOCIAL_QUERY", "AAPL;Apple")
DEFAULT_TWEET_COUNT = int(os.environ.get("SOCIAL_TWEET_COUNT", "100"))
DEFAULT_PROFILE_DIR = os.environ.get(
    "PLAYWRIGHT_USER_DATA_DIR", ".playwright_x_profile"
)
DEFAULT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
DEFAULT_OUTPUT_DIR = os.environ.get("ETL_OUTPUT_DIR", "data/01_raw/twitter")
DEFAULT_HEADLESS = os.environ.get("PLAYWRIGHT_HEADLESS", "False").lower() in (
    "true",
    "1",
    "yes",
)


async def _async_extract_tweets(
    query: str,
    target_tweets: int,
    profile_dir: str,
    headless: bool,
) -> List[Dict[str, str]]:
    client: SocialSearchClient = BlueskyClient()
    posts = await client.search_posts(query, target_tweets)

    normalized: List[Dict[str, str]] = []
    for post in posts:
        normalized.append(
            {
                "ID_Tweet": post.get("id", ""),
                "Fecha_UTC": post.get("published_time", ""),
                "Contenido_Texto": post.get("text", ""),
                "Retweets": post.get("reposts", "0"),
                "Favoritos": post.get("likes", "0"),
            }
        )

    return normalized


@op(out=Out(list[Dict[str, str]]))
def extract_tweets_op(
    query: str = DEFAULT_QUERY,
    target_tweets: int = DEFAULT_TWEET_COUNT,
    profile_dir: str = DEFAULT_PROFILE_DIR,
    headless: bool = DEFAULT_HEADLESS,
) -> List[Dict[str, str]]:
    return asyncio.run(
        _async_extract_tweets(query, target_tweets, profile_dir, headless)
    )


@op(out=Out(pd.DataFrame))
def analyze_sentiment_op(
    tweets: List[Dict[str, str]],
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    df = pd.DataFrame(tweets)
    if df.empty:
        return df

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        device=-1,
    )
    texts = df["Contenido_Texto"].fillna("").astype(str).tolist()
    results = classifier(texts)

    df["Sentimiento"] = [res["label"] for res in results]
    df["Puntuacion_Sentimiento"] = [round(res["score"], 4) for res in results]
    return df


@op(out=Out(str))
def save_results_op(
    df: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    base_name: str = "tweets_sentiment",
) -> str:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_name = f"{base_name}_{timestamp}.csv"
    file_path = output_path / file_name

    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return str(file_path)


@job(name="twitter_sentiment_etl")
def twitter_sentiment_etl() -> str:
    tweets = extract_tweets_op()
    analyzed = analyze_sentiment_op(tweets)
    save_results_op(analyzed)


def main() -> None:
    twitter_sentiment_etl.execute_in_process()


if __name__ == "__main__":
    main()
