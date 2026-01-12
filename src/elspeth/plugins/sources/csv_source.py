# src/elspeth/plugins/sources/csv_source.py
"""CSV source plugin for ELSPETH.

Loads rows from CSV files using pandas for robust parsing.
"""

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class CSVOutputSchema(PluginSchema):
    """Dynamic schema - CSV columns are determined at runtime."""

    model_config = {"extra": "allow"}


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
        self._path = Path(config["path"])
        self._delimiter = config.get("delimiter", ",")
        self._encoding = config.get("encoding", "utf-8")
        self._skip_rows = config.get("skip_rows", 0)
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

        for _, row in self._dataframe.iterrows():
            yield row.to_dict()

    def close(self) -> None:
        """Release resources."""
        self._dataframe = None
