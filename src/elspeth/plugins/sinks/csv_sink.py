# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.
"""

import csv
import hashlib
from collections.abc import Sequence
from typing import IO, Any

from elspeth.contracts import ArtifactDescriptor, PluginSchema
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

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
    """

    name = "csv"
    input_schema = CSVInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the CSV file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        # Lazy initialization - discover fieldnames from first row
        if self._file is None:
            self._fieldnames = list(rows[0].keys())
            self._file = open(  # noqa: SIM115 - lifecycle managed by class
                self._path, "w", encoding=self._encoding, newline=""
            )
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                delimiter=self._delimiter,
            )
            self._writer.writeheader()

        # Write all rows in batch
        for row in rows:
            self._writer.writerow(row)  # type: ignore[union-attr]

        # Flush to ensure content is on disk for hashing
        self._file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of the file contents."""
        sha256 = hashlib.sha256()
        with open(self._path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

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
