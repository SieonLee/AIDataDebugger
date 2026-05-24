"""Dataset connector registry."""

from data_debugger.connectors.base import DatasetConnector
from data_debugger.connectors.csv_connector import CSVConnector
from data_debugger.connectors.placeholders import BigQueryConnector, PostgresConnector, S3Connector

__all__ = ["BigQueryConnector", "CSVConnector", "DatasetConnector", "PostgresConnector", "S3Connector"]
