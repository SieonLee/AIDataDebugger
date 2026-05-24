"""Dataset profiling helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight metadata used by the Streamlit UI and reports."""
    row_count, column_count = df.shape

    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1_000_000, 3),
    }


def dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact per-column summary table."""
    if df.empty:
        return pd.DataFrame(columns=["column", "dtype", "missing", "missing_rate", "unique"])

    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[column].dtype) for column in df.columns],
            "missing": [int(df[column].isna().sum()) for column in df.columns],
            "missing_rate": [round(float(df[column].isna().mean()), 4) for column in df.columns],
            "unique": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )
