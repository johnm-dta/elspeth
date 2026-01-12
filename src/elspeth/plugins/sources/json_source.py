# src/elspeth/plugins/sources/json_source.py
"""JSON source plugin for ELSPETH.

Loads rows from JSON files. Supports JSON array and JSONL formats.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class JSONOutputSchema(PluginSchema):
    """Dynamic schema - JSON fields determined at runtime."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class JSONSource(BaseSource):
    """Load rows from a JSON file.

    Config options:
        path: Path to JSON file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension if not set.
        data_key: Key to extract array from JSON object (e.g., "results")
        encoding: File encoding (default: "utf-8")
    """

    name = "json"
    output_schema = JSONOutputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._encoding = config.get("encoding", "utf-8")
        self._data_key = config.get("data_key")

        # Auto-detect format from extension if not specified
        fmt = config.get("format")
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from JSON file.

        Yields:
            Dict for each row.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is invalid or not an array.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"JSON file not found: {self._path}")

        if self._format == "jsonl":
            yield from self._load_jsonl()
        else:
            yield from self._load_json_array()

    def _load_jsonl(self) -> Iterator[dict[str, Any]]:
        """Load from JSONL format (one JSON object per line)."""
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield json.loads(line)

    def _load_json_array(self) -> Iterator[dict[str, Any]]:
        """Load from JSON array format."""
        with open(self._path, encoding=self._encoding) as f:
            data = json.load(f)

        # Extract from nested key if specified
        if self._data_key:
            data = data[self._data_key]

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        yield from data

    def close(self) -> None:
        """Release resources (no-op for JSON source)."""
        pass
