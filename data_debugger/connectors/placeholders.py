"""Future connector placeholders."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_debugger.connectors.base import DatasetConnector


class S3Connector(DatasetConnector):
    name = "s3"

    def load(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError("S3 connector is planned for a future release.")


class PostgresConnector(DatasetConnector):
    name = "postgres"

    def load(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError("Postgres connector is planned for a future release.")


class BigQueryConnector(DatasetConnector):
    name = "bigquery"

    def load(self, source: Any) -> pd.DataFrame:
        raise NotImplementedError("BigQuery connector is planned for a future release.")
