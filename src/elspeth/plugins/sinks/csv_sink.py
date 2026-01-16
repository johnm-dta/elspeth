# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files.
"""

import csv
from collections.abc import Sequence
from typing import IO, Any

from elspeth.contracts import PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext


class CSVInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class CSVSinkConfig(PathConfig):
    """Configuration for CSV sink plugin."""

    delimiter: str = ","
    encoding: str = "utf-8"


class CSVSink(BaseSink):
    """Write rows to a CSV file.

    Config options:
        path: Path to output CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
    """

    name = "csv"
    input_schema = CSVInputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Write a row to the CSV file.

        On first call, creates file and writes header row.
        """
        if self._file is None:
            # Lazy initialization - discover fieldnames from first row
            self._fieldnames = list(row.keys())
            self._file = open(self._path, "w", encoding=self._encoding, newline="")  # noqa: SIM115 - lifecycle managed by class
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                delimiter=self._delimiter,
            )
            self._writer.writeheader()

        self._writer.writerow(row)  # type: ignore[union-attr]

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
