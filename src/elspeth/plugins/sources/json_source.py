# src/elspeth/plugins/sources/json_source.py
"""JSON source plugin for ELSPETH.

Loads rows from JSON files. Supports JSON array and JSONL formats.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.
"""

import json
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import ValidationError

from elspeth.contracts import PluginSchema
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import SourceDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class JSONSourceConfig(SourceDataConfig):
    """Configuration for JSON source plugin.

    Inherits from SourceDataConfig, which requires schema and on_validation_failure.
    """

    format: Literal["json", "jsonl"] | None = None
    data_key: str | None = None
    encoding: str = "utf-8"


class JSONSource(BaseSource):
    """Load rows from a JSON file.

    Config options:
        path: Path to JSON file (required)
        schema: Schema configuration (required, via SourceDataConfig)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension if not set.
        data_key: Key to extract array from JSON object (e.g., "results")
        encoding: File encoding (default: "utf-8")

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]}
        - Free: {"mode": "free", "fields": ["id: int"]} - at least these fields
    """

    name = "json"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONSourceConfig.from_dict(config)

        self._path = cfg.resolved_path()
        self._encoding = cfg.encoding
        self._data_key = cfg.data_key

        # Auto-detect format from extension if not specified
        fmt = cfg.format
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

        # Store schema config for audit trail
        # SourceDataConfig (via DataPluginConfig) ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Store quarantine routing destination
        self._on_validation_failure = cfg.on_validation_failure

        # CRITICAL: allow_coercion=True for sources (external data boundary)
        # Sources are the ONLY place where type coercion is allowed
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "JSONRowSchema",
            allow_coercion=True,
        )

        # Set output_schema for protocol compliance
        self.output_schema = self._schema_class

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from JSON file.

        Each row is validated against the configured schema:
        - Valid rows are yielded for processing
        - Invalid rows are quarantined (recorded but not yielded)

        Yields:
            Dict for each valid row.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is invalid or not an array.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"JSON file not found: {self._path}")

        if self._format == "jsonl":
            yield from self._load_jsonl(ctx)
        else:
            yield from self._load_json_array(ctx)

    def _load_jsonl(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load from JSONL format (one JSON object per line)."""
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    row = json.loads(line)
                    yield from self._validate_and_yield(row, ctx)

    def _load_json_array(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load from JSON array format."""
        with open(self._path, encoding=self._encoding) as f:
            data = json.load(f)

        # Extract from nested key if specified
        if self._data_key:
            data = data[self._data_key]

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        for row in data:
            yield from self._validate_and_yield(row, ctx)

    def _validate_and_yield(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> Iterator[dict[str, Any]]:
        """Validate a row and yield if valid, otherwise quarantine.

        Args:
            row: Row data to validate
            ctx: Plugin context for recording validation errors

        Yields:
            Validated row dict if valid, nothing if quarantined
        """
        try:
            # Validate and potentially coerce row data
            validated = self._schema_class.model_validate(row)
            yield validated.to_row()
        except ValidationError as e:
            # Record validation failure - row quarantined
            # This is a trust boundary: external data may be invalid
            ctx.record_validation_error(
                row=row,
                error=str(e),
                schema_mode=self._schema_config.mode or "dynamic",
                destination=self._on_validation_failure,
            )
            # Skip invalid row - don't yield

    def close(self) -> None:
        """Release resources (no-op for JSON source)."""
        pass
