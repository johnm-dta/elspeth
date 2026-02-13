# src/elspeth/core/landscape/formatters.py
"""Export formatters for Landscape data.

Formatters transform audit records for different output formats.
Also provides serialization utilities for converting dataclasses and datetime
objects to JSON-serializable structures.
"""

import json
import math
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from elspeth.core.landscape.lineage import LineageResult


def serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings for JSON serialization.

    Recursively processes dicts and lists to convert all datetime values.
    Rejects NaN and Infinity per CLAUDE.md audit integrity requirements.

    Args:
        obj: Any value - datetime, dict, list, or other

    Returns:
        The same structure with datetime objects replaced by ISO strings

    Raises:
        ValueError: If NaN or Infinity values are encountered
    """
    # Reject NaN and Infinity - audit trail must be pristine
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("NaN values are not allowed in audit data (violates audit integrity)")
        if math.isinf(obj):
            raise ValueError("Infinity values are not allowed in audit data (violates audit integrity)")

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass (or list of dataclasses) to JSON-serializable dict.

    Handles:
    - Nested dataclasses (recursive conversion)
    - Lists of dataclasses
    - Enum values (converted to .value)
    - Datetime values (converted to ISO strings)
    - None (returns empty dict)
    - Plain values (pass through)

    Uses stdlib is_dataclass() and isinstance(Enum) for explicit type checking
    rather than hasattr() checks (clearer intent, better for maintenance).

    Args:
        obj: Dataclass instance, list, or primitive value

    Returns:
        dict for dataclasses, list for lists, or the original value
    """
    if obj is None:
        return {}
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        # is_dataclass returns True for both instances and classes
        # We only want instances, not the class itself
        result: dict[str, Any] = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if is_dataclass(value) and not isinstance(value, type):
                result[field_name] = dataclass_to_dict(value)
            elif isinstance(value, list):
                result[field_name] = [dataclass_to_dict(item) for item in value]
            elif isinstance(value, Enum):
                # Explicit Enum check instead of hasattr(value, "value")
                result[field_name] = value.value
            else:
                result[field_name] = serialize_datetime(value)
        return result
    return obj


class ExportFormatter(Protocol):
    """Protocol for export formatters."""

    def format(self, record: dict[str, Any]) -> str | dict[str, Any]:
        """Format a record for output."""
        ...


class JSONFormatter:
    """Format records as JSON lines."""

    def format(self, record: dict[str, Any]) -> str:
        """Format as JSON line."""
        normalized = serialize_datetime(record)
        return json.dumps(normalized, allow_nan=False)


class LineageTextFormatter:
    """Format LineageResult as human-readable text for CLI output."""

    def format(self, result: "LineageResult | None") -> str:
        """Format lineage result as text.

        Args:
            result: LineageResult to format, or None if not found

        Returns:
            Human-readable text representation
        """
        if result is None:
            return "No lineage found. Token or row may not exist, or processing is incomplete."

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("LINEAGE REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Token info
        lines.append(f"Token: {result.token.token_id}")
        lines.append(f"Row: {result.token.row_id}")
        if result.token.branch_name:
            lines.append(f"Branch: {result.token.branch_name}")
        lines.append("")

        # Source row
        lines.append("--- Source ---")
        lines.append(f"Row Index: {result.source_row.row_index}")
        lines.append(f"Source Data Hash: {result.source_row.source_data_hash}")
        lines.append(f"Payload Available: {result.source_row.payload_available}")
        if result.source_row.source_data:
            lines.append(f"Source Data: {result.source_row.source_data}")
        lines.append("")

        # Outcome
        if result.outcome:
            lines.append("--- Outcome ---")
            # Direct access to .name - Tier 1 trust (our audit data)
            # If outcome.outcome is not an Enum, that's a bug we want to crash on
            # Using .name gives uppercase (COMPLETED) which is more readable in CLI output
            lines.append(f"Outcome: {result.outcome.outcome.name}")
            if result.outcome.sink_name:
                lines.append(f"Sink: {result.outcome.sink_name}")
            lines.append(f"Terminal: {result.outcome.is_terminal}")
            lines.append("")

        # Node states
        if result.node_states:
            lines.append("--- Node States ---")
            for state in result.node_states:
                # Direct access to .value - Tier 1 trust (our audit data)
                # No defensive hasattr - if status isn't an Enum, crash
                lines.append(f"  [{state.step_index}] {state.node_id}: {state.status.value}")
            lines.append("")

        # Calls
        if result.calls:
            lines.append("--- External Calls ---")
            for call in result.calls:
                # Direct access to .value - Tier 1 trust (our audit data)
                if call.latency_ms is None:
                    latency_display = "N/A"
                else:
                    latency_display = f"{call.latency_ms:.1f}ms"
                lines.append(f"  {call.call_type.value}: {call.status.value} ({latency_display})")
            lines.append("")

        # Errors
        if result.validation_errors:
            lines.append("--- Validation Errors ---")
            for val_err in result.validation_errors:
                lines.append(f"  [{val_err.schema_mode}] {val_err.error}")
            lines.append("")

        if result.transform_errors:
            lines.append("--- Transform Errors ---")
            for tx_err in result.transform_errors:
                lines.append(f"  [{tx_err.transform_id}] {tx_err.destination}")
            lines.append("")

        # Parent tokens (for forks/joins)
        if result.parent_tokens:
            lines.append("--- Parent Tokens ---")
            for parent in result.parent_tokens:
                lines.append(f"  {parent.token_id}")
            lines.append("")

        return "\n".join(lines)


class CSVFormatter:
    """Format records for CSV output.

    Flattens nested structures using dot notation.
    """

    def flatten(self, record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict to dot-notation keys."""
        result: dict[str, Any] = {}

        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self.flatten(value, full_key))
            elif isinstance(value, list):
                # Convert lists to JSON strings for CSV
                # Use serialize_datetime to validate (rejects NaN/Infinity) and convert datetimes
                result[full_key] = json.dumps(serialize_datetime(value))
            else:
                # Validate scalar values and normalize datetimes to ISO strings.
                result[full_key] = serialize_datetime(value)

        return result

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format as flat dict for CSV."""
        return self.flatten(record)
