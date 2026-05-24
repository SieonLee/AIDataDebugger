"""Connector interfaces for future dataset sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class DatasetConnector(ABC):
    """Base class for dataset connectors."""

    name = "base"

    @abstractmethod
    def load(self, source: Any) -> pd.DataFrame:
        """Load a dataset into a DataFrame."""
