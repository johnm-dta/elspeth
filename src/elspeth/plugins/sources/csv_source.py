# src/elspeth/plugins/sources/csv_source.py
"""CSV source plugin for ELSPETH.

Loads rows from CSV files using pandas for robust parsing.
"""

from collections.abc import Iterator
from typing import Any

import pandas as pd

from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class CSVOutputSchema(PluginSchema):
    """Dynamic schema - CSV columns are determined at runtime."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class CSVSourceConfig(PathConfig):
    """Configuration for CSV source plugin."""

    delimiter: str = ","
    encoding: str = "utf-8"
    skip_rows: int = 0


class CSVSource(BaseSource):
    """Load rows from a CSV file.

    Config options:
        path: Path to CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
        skip_rows: Number of header rows to skip (default: 0)
    """

    name = "csv"
    output_schema = CSVOutputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSourceConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._skip_rows = cfg.skip_rows
        self._dataframe: pd.DataFrame | None = None

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from CSV file.

        Yields:
            Dict for each row with column names as keys.

        Raises:
            FileNotFoundError: If CSV file does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        self._dataframe = pd.read_csv(
            self._path,
            delimiter=self._delimiter,
            encoding=self._encoding,
            skiprows=self._skip_rows,
            dtype=str,  # Keep all values as strings for consistent handling
            keep_default_na=False,  # Don't convert empty strings to NaN
        )

        # DataFrame columns are strings from CSV headers
        for record in self._dataframe.to_dict(orient="records"):
            yield {str(k): v for k, v in record.items()}

    def close(self) -> None:
        """Release resources."""
        self._dataframe = None
