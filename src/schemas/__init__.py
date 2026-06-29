import pandas as pd
from pydantic import BaseModel, TypeAdapter
from typing import Type, List

from .stock import StockRawRow, StockProcessedRow
from .social import (
    SocialRawRow,
    SocialProcessedRow,
    SocialSentimentRow,
    SocialAggregatedRow,
    GdeltSentimentRow,
)
from .features import EngineeredFeaturesRow
from .predictions import TradingSignalsRow


def validate_df(
    df: pd.DataFrame, model_cls: Type[BaseModel], stage: str = "data"
) -> pd.DataFrame:
    """
    Validates a Pandas DataFrame against a Pydantic model.
    Converts DataFrame rows to dictionaries and validates them using Pydantic.
    """
    if df is None:
        raise ValueError(f"[{stage}] DataFrame is None and cannot be validated.")

    # 1. Structural check: Identify missing required fields
    required_fields = {
        name for name, field in model_cls.model_fields.items() if field.is_required()
    }
    actual_columns = set(df.columns)
    missing_fields = required_fields - actual_columns
    if missing_fields:
        raise ValueError(
            f"[{stage}] Schema validation failed: Missing required columns: {missing_fields}"
        )

    # If the dataframe is empty, we consider it valid (or just structurally verified)
    if df.empty:
        return df

    # 2. Row level validation
    # To handle timestamp issues, we convert timestamps or date objects to string/native types if needed.
    records = []
    for r in df.to_dict(orient="records"):
        # clean any nan to None so Pydantic understands it as optional/null
        cleaned_record = {}
        for k, v in r.items():
            if pd.isna(v):
                cleaned_record[k] = None
            elif isinstance(v, (pd.Timestamp, pd.Period)):
                # Convert pandas timestamp columns to python datetime/date
                try:
                    cleaned_record[k] = v.to_pydatetime()
                except ValueError:
                    # In case of date-only conversions
                    cleaned_record[k] = v.to_pycalendar()
            else:
                cleaned_record[k] = v
        records.append(cleaned_record)

    try:
        adapter = TypeAdapter(List[model_cls])
        adapter.validate_python(records)
    except Exception as e:
        raise ValueError(
            f"[{stage}] Schema validation failed for {model_cls.__name__} in dataset:\n{e}"
        ) from e

    return df


__all__ = [
    "StockRawRow",
    "StockProcessedRow",
    "SocialRawRow",
    "SocialProcessedRow",
    "SocialSentimentRow",
    "SocialAggregatedRow",
    "GdeltSentimentRow",
    "EngineeredFeaturesRow",
    "TradingSignalsRow",
    "validate_df",
]
