"""Input normalization helpers shared by profiling and drift checks."""

from __future__ import annotations

import warnings

import pandas as pd


NULL_LIKE_VALUES = ["", " ", "null", "NULL", "NaN", "nan", "N/A", "na"]
NORMALIZED_NULL_TOKENS = {"", "null", "nan", "n/a", "na"}


def normalize_dataframe_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip string cells and convert null-like tokens to real missing values."""
    def normalize_cell(value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in NORMALIZED_NULL_TOKENS:
                return pd.NA
            return stripped
        return value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return df.copy().applymap(normalize_cell)
