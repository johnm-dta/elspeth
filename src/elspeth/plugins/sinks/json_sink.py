# src/elspeth/plugins/sinks/json_sink.py
"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.
"""

import hashlib
import json
from typing import IO, Any, Literal

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext


class JSONInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class JSONSinkConfig(PathConfig):
    """Configuration for JSON sink plugin."""

    format: Literal["json", "jsonl"] | None = None
    indent: int | None = None
    encoding: str = "utf-8"


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output JSON file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        indent: Indentation for pretty-printing (default: None for compact)
        encoding: File encoding (default: "utf-8")
    """

    name = "json"
    input_schema = JSONInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._encoding = cfg.encoding
        self._indent = cfg.indent

        # Auto-detect format from extension if not specified
        fmt = cfg.format
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

        self._file: IO[str] | None = None
        self._rows: list[dict[str, Any]] = []  # Buffer for json array format

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the JSON file.

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

        if self._format == "jsonl":
            self._write_jsonl_batch(rows)
        else:
            # Buffer rows for JSON array format
            self._rows.extend(rows)
            # Write immediately (file is rewritten on each write for JSON format)
            self._write_json_array()

        # Flush to ensure content is on disk for hashing
        if self._file is not None:
            self._file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _write_jsonl_batch(self, rows: list[dict[str, Any]]) -> None:
        """Write rows as JSONL (append mode)."""
        if self._file is None:
            self._file = open(self._path, "w", encoding=self._encoding)  # noqa: SIM115 - lifecycle managed by class

        for row in rows:
            json.dump(row, self._file)
            self._file.write("\n")

    def _write_json_array(self) -> None:
        """Write buffered rows as JSON array (rewrite mode)."""
        if self._file is None:
            self._file = open(self._path, "w", encoding=self._encoding)  # noqa: SIM115 - lifecycle managed by class
        self._file.seek(0)
        self._file.truncate()
        json.dump(self._rows, self._file, indent=self._indent)

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
            self._rows = []

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
