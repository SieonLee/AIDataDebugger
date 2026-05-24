"""CSV connector implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_debugger.connectors.base import DatasetConnector
from data_debugger.utils import NULL_LIKE_VALUES, normalize_dataframe_values


class CSVConnector(DatasetConnector):
    name = "csv"

    def load(self, source: Any) -> pd.DataFrame:
        df = pd.read_csv(source, na_values=NULL_LIKE_VALUES, keep_default_na=True)
        return normalize_dataframe_values(df)
