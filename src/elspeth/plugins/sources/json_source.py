# src/elspeth/plugins/sources/json_source.py
"""JSON source plugin for ELSPETH.

Loads rows from JSON files. Supports JSON array and JSONL formats.

IMPORTANT: Sources use allow_coercion=True to normalize external data.
This is the ONLY place in the pipeline where coercion is allowed.

NOTE: Non-standard JSON constants (NaN, Infinity, -Infinity) are rejected
at parse time per canonical JSON policy. Use null for missing values.
"""

import json
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import ValidationError

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import SourceDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


def _reject_nonfinite_constant(value: str) -> None:
    """Reject non-standard JSON constants (NaN, Infinity, -Infinity).

    Python's json module accepts these by default, but they violate:
    1. RFC 8259 (JSON standard) - only allows null, true, false
    2. Canonical JSON policy - non-finite floats crash hashing

    This function is passed to json.loads/json.load via parse_constant
    parameter to reject these values at parse time.

    Args:
        value: The constant string (NaN, Infinity, or -Infinity)

    Raises:
        ValueError: Always - these constants are not allowed
    """
    raise ValueError(f"Non-standard JSON constant '{value}' not allowed. Use null for missing values, not NaN/Infinity.")


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
    plugin_version = "1.0.0"
    # Override parent type - SourceDataConfig requires this to be set
    _on_validation_failure: str

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

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from JSON file.

        Each row is validated against the configured schema:
        - Valid rows are yielded as SourceRow.valid()
        - Invalid rows are yielded as SourceRow.quarantined()

        Yields:
            SourceRow for each row (valid or quarantined).

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

    def _load_jsonl(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load from JSONL format (one JSON object per line).

        Per Three-Tier Trust Model (CLAUDE.md), external data (Tier 3) that
        fails to parse is quarantined, not crash the pipeline. This allows
        subsequent valid lines to still be processed.
        """
        with open(self._path, encoding=self._encoding) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Catch JSON parse errors at the trust boundary
                # parse_constant rejects NaN/Infinity at parse time (canonical JSON policy)
                try:
                    row = json.loads(line, parse_constant=_reject_nonfinite_constant)
                except (json.JSONDecodeError, ValueError) as e:
                    # External data parse failure - quarantine, don't crash
                    # Store raw line + metadata for audit traceability
                    raw_row = {"__raw_line__": line, "__line_number__": line_num}
                    error_msg = f"JSON parse error at line {line_num}: {e}"

                    ctx.record_validation_error(
                        row=raw_row,
                        error=error_msg,
                        schema_mode="parse",  # Distinct from schema validation
                        destination=self._on_validation_failure,
                    )

                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=raw_row,
                            error=error_msg,
                            destination=self._on_validation_failure,
                        )
                    continue

                yield from self._validate_and_yield(row, ctx)

    def _load_json_array(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load from JSON array format."""
        with open(self._path, encoding=self._encoding) as f:
            # parse_constant rejects NaN/Infinity at parse time (canonical JSON policy)
            try:
                data = json.load(f, parse_constant=_reject_nonfinite_constant)
            except (json.JSONDecodeError, ValueError) as e:
                # File-level parse error - treat as Tier 3 boundary
                # External data can be malformed; don't crash the pipeline
                if isinstance(e, json.JSONDecodeError):
                    error_msg = f"JSON parse error at line {e.lineno} col {e.colno}: {e.msg}"
                else:
                    # ValueError from _reject_nonfinite_constant (NaN/Infinity)
                    error_msg = f"JSON parse error: {e}"
                ctx.record_validation_error(
                    row={"file_path": str(self._path), "error": error_msg},
                    error=error_msg,
                    schema_mode="parse",  # File-level parse, not schema validation
                    destination=self._on_validation_failure,
                )

                # Yield quarantined row if not discarding
                # This allows audit trail to show the file-level failure
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row={"file_path": str(self._path), "parse_error": str(e)},
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                return  # Stop processing this file

        # Extract from nested key if specified
        # Per Three-Tier Trust Model (CLAUDE.md), structural mismatches in external
        # data are quarantined, not exceptions. This handles:
        # 1. data_key configured but JSON root is a list (not dict)
        # 2. data_key configured but key doesn't exist in JSON object
        # 3. data_key extraction results in non-list
        if self._data_key:
            # Check 1: Root must be a dict to use data_key
            if not isinstance(data, dict):
                error_msg = f"Cannot extract data_key '{self._data_key}': expected JSON object, got {type(data).__name__}"
                ctx.record_validation_error(
                    row={"file_path": str(self._path), "data_key": self._data_key},
                    error=error_msg,
                    schema_mode="structure",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row={"file_path": str(self._path), "structure_error": error_msg},
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                return

            # Check 2: Key must exist in the dict
            if self._data_key not in data:
                error_msg = f"data_key '{self._data_key}' not found in JSON object. Available keys: {list(data.keys())}"
                ctx.record_validation_error(
                    row={"file_path": str(self._path), "data_key": self._data_key},
                    error=error_msg,
                    schema_mode="structure",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row={"file_path": str(self._path), "structure_error": error_msg},
                        error=error_msg,
                        destination=self._on_validation_failure,
                    )
                return

            data = data[self._data_key]

        # Check 3: Data must be a list (either root or extracted via data_key)
        if not isinstance(data, list):
            error_msg = f"Expected JSON array, got {type(data).__name__}"
            ctx.record_validation_error(
                row={"file_path": str(self._path)},
                error=error_msg,
                schema_mode="structure",
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row={"file_path": str(self._path), "structure_error": error_msg},
                    error=error_msg,
                    destination=self._on_validation_failure,
                )
            return

        for row in data:
            yield from self._validate_and_yield(row, ctx)

    def _validate_and_yield(self, row: dict[str, Any], ctx: PluginContext) -> Iterator[SourceRow]:
        """Validate a row and yield if valid, otherwise quarantine.

        Args:
            row: Row data to validate
            ctx: Plugin context for recording validation errors

        Yields:
            SourceRow.valid() if valid, SourceRow.quarantined() if invalid
        """
        try:
            # Validate and potentially coerce row data
            validated = self._schema_class.model_validate(row)
            yield SourceRow.valid(validated.to_row())
        except ValidationError as e:
            # Record validation failure in audit trail
            # This is a trust boundary: external data may be invalid
            ctx.record_validation_error(
                row=row,
                error=str(e),
                schema_mode=self._schema_config.mode or "dynamic",
                destination=self._on_validation_failure,
            )

            # Yield quarantined row for routing to configured sink
            # If "discard", don't yield - row is intentionally dropped
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(e),
                    destination=self._on_validation_failure,
                )

    def close(self) -> None:
        """Release resources (no-op for JSON source)."""
        pass
