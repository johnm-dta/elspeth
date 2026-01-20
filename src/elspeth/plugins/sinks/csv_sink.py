# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import csv
import hashlib
from collections.abc import Sequence
from typing import IO, Any

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class CSVSinkConfig(PathConfig):
    """Configuration for CSV sink plugin.

    Inherits from PathConfig, which requires schema configuration.
    """

    delimiter: str = ","
    encoding: str = "utf-8"
    validate_input: bool = False  # Optional runtime validation of incoming rows
    mode: str = "write"  # "write" (truncate) or "append"


class CSVSink(BaseSink):
    """Write rows to a CSV file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output CSV file (required)
        schema: Schema configuration (required, via PathConfig)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
        validate_input: Validate incoming rows against schema (default: False)
        mode: "write" (truncate, default) or "append" (add to existing file)

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]}
        - Free: {"mode": "free", "fields": ["id: int"]} - at least these fields

    Append mode behavior:
        - If file exists: reads headers from it and appends rows without header
        - If file doesn't exist or is empty: creates file with header (like write mode)
    """

    name = "csv"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding
        self._validate_input = cfg.validate_input
        self._mode = cfg.mode

        # Store schema config for audit trail
        # PathConfig (via DataPluginConfig) ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "CSVRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write a batch of rows to the CSV file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes

        Raises:
            ValidationError: If validate_input=True and a row fails validation.
                This indicates a bug in an upstream transform.
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        # Optional input validation - crash on failure (upstream bug!)
        if self._validate_input and not self._schema_config.is_dynamic:
            for row in rows:
                # Raises ValidationError on failure - this is intentional
                self._schema_class.model_validate(row)

        # Lazy initialization - open file on first write
        if self._file is None:
            self._open_file(rows)

        # Write all rows in batch
        # Invariant: _file and _writer are always set together (by _open_file above)
        file = self._file
        writer = self._writer
        if file is None or writer is None:
            raise RuntimeError("CSVSink writer not initialized - this is a bug")
        for row in rows:
            writer.writerow(row)

        # Flush to ensure content is on disk for hashing
        file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _open_file(self, rows: list[dict[str, Any]]) -> None:
        """Open file for writing, handling append mode.

        In append mode:
        - If file exists with headers: read headers from it, open in append mode
        - If file doesn't exist or is empty: create with headers (like write mode)

        In write mode:
        - Always truncate and write headers

        Args:
            rows: First batch of rows (used to determine fieldnames if needed)
        """
        if self._mode == "append" and self._path.exists():
            # Try to read existing headers from file
            with open(self._path, encoding=self._encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=self._delimiter)
                existing_fieldnames = reader.fieldnames

            if existing_fieldnames:
                # Use existing headers, append mode (no header write)
                self._fieldnames = list(existing_fieldnames)
                self._file = open(  # noqa: SIM115 - handle kept open for streaming writes, closed in close()
                    self._path, "a", encoding=self._encoding, newline=""
                )
                self._writer = csv.DictWriter(
                    self._file,
                    fieldnames=self._fieldnames,
                    delimiter=self._delimiter,
                )
                # No header write - already exists
                return

        # Write mode OR append to non-existent/empty file
        # In both cases: determine fieldnames from first row, write header
        self._fieldnames = list(rows[0].keys())
        self._file = open(  # noqa: SIM115 - handle kept open for streaming writes, closed in close()
            self._path, "w", encoding=self._encoding, newline=""
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            delimiter=self._delimiter,
        )
        self._writer.writeheader()

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

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
