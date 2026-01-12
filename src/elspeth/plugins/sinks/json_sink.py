# src/elspeth/plugins/sinks/json_sink.py
"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.
"""

import json
from pathlib import Path
from typing import IO, Any

from elspeth.plugins.base import BaseSink
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class JSONInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Config options:
        path: Path to output JSON file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        indent: Indentation for pretty-printing (default: None for compact)
        encoding: File encoding (default: "utf-8")
    """

    name = "json"
    input_schema = JSONInputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._encoding = config.get("encoding", "utf-8")
        self._indent = config.get("indent")

        # Auto-detect format from extension if not specified
        fmt = config.get("format")
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

        self._file: IO[str] | None = None
        self._rows: list[dict[str, Any]] = []  # Buffer for json array format

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Write a row to the JSON file."""
        if self._format == "jsonl":
            self._write_jsonl(row)
        else:
            # Buffer for JSON array format (written on close)
            self._rows.append(row)

    def _write_jsonl(self, row: dict[str, Any]) -> None:
        """Write a single row as JSONL."""
        if self._file is None:
            self._file = open(self._path, "w", encoding=self._encoding)

        json.dump(row, self._file)
        self._file.write("\n")

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._format == "json" and self._rows:
            # Write buffered rows as JSON array
            if self._file is None:
                self._file = open(self._path, "w", encoding=self._encoding)
            self._file.seek(0)
            self._file.truncate()
            json.dump(self._rows, self._file, indent=self._indent)

        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._format == "json" and self._rows and self._file is None:
            # Ensure rows are written if flush wasn't called
            self.flush()

        if self._file is not None:
            self._file.close()
            self._file = None
            self._rows = []
