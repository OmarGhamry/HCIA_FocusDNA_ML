"""Sensible default input values derived from the saved dataset."""

from __future__ import annotations

import pandas as pd


def dataset_defaults(df: pd.DataFrame, input_features: list[str]) -> dict:
    """Median for numeric columns; mode for categorical columns."""
    defaults: dict = {}
    subset = df[input_features]
    for col in input_features:
        series = subset[col]
        if pd.api.types.is_numeric_dtype(series):
            median = series.median()
            defaults[col] = float(median) if pd.api.types.is_float_dtype(series) else int(median)
        else:
            value = series.mode().iloc[0]
            defaults[col] = value.item() if hasattr(value, "item") else value
            if not isinstance(defaults[col], str):
                defaults[col] = str(defaults[col])
    return defaults
