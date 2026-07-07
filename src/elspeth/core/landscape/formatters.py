"""Export formatters for Landscape data.

Formatters transform audit records for different output formats.
Also provides serialization utilities for converting dataclasses and datetime
objects to JSON-serializable structures.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, overload

from elspeth.contracts.errors import AuditIntegrityError

if TYPE_CHECKING:
    from elspeth.core.landscape.lineage import LineageResult


@overload
def serialize_datetime(obj: dict[str, object]) -> dict[str, object]: ...


@overload
def serialize_datetime(obj: Mapping[str, object]) -> dict[str, object]: ...


@overload
def serialize_datetime(obj: list[object]) -> list[object]: ...


@overload
def serialize_datetime(obj: float) -> float: ...


@overload
def serialize_datetime(obj: datetime) -> str: ...


@overload
def serialize_datetime(obj: Any) -> Any: ...


def serialize_datetime(obj: Any) -> Any:
    """Convert audit values to JSON-serializable structures.

    Recursively processes mappings and sequences to convert all datetime values.
    Frozen mapping containers are thawed to plain dicts for JSON output.
    Rejects NaN and Infinity per CLAUDE.md audit integrity requirements.

    Args:
        obj: Any value - datetime, mapping, list, or other

    Returns:
        The same structure with datetime objects replaced by ISO strings

    Raises:
        AuditIntegrityError: If NaN or Infinity values are encountered
    """
    # Reject NaN and Infinity - audit trail must be pristine
    if isinstance(obj, float):
        if math.isnan(obj):
            raise AuditIntegrityError("NaN values are not allowed in audit data (violates audit integrity)")
        if math.isinf(obj):
            raise AuditIntegrityError("Infinity values are not allowed in audit data (violates audit integrity)")

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_datetime(item) for item in obj]
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass (or list of dataclasses) to JSON-serializable dict.

    Handles:
    - Nested dataclasses (recursive conversion)
    - Mappings and lists of dataclasses
    - Enum values (converted to .value)
    - Datetime values (converted to ISO strings)
    - None (returns None — preserves absence semantics)
    - Plain values (pass through)

    Uses stdlib is_dataclass() and isinstance(Enum) for explicit type checking
    rather than hasattr() checks (clearer intent, better for maintenance).

    Args:
        obj: Dataclass instance, list, or primitive value

    Returns:
        dict for dataclasses, list for lists, or the original value
    """
    if obj is None:
        return None
    if isinstance(obj, Enum):
        # Explicit Enum check instead of hasattr(value, "value")
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        # is_dataclass returns True for both instances and classes
        # We only want instances, not the class itself
        result: dict[str, Any] = {}
        for dataclass_field in fields(obj):
            field_name = dataclass_field.name
            value = getattr(obj, field_name)
            result[field_name] = dataclass_to_dict(value)
        return result
    if isinstance(obj, Mapping):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    return serialize_datetime(obj)


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
        if result.source_row.source_data is not None:
            lines.append(f"Source Data: {serialize_datetime(result.source_row.source_data)}")
        lines.append("")

        # Outcome
        if result.outcome:
            lines.append("--- Outcome ---")
            # Direct access to .name - Tier 1 trust (our audit data)
            # If outcome.outcome is not an Enum, that's a bug we want to crash on
            # Using .name gives uppercase values that are more readable in CLI output.
            outcome_name = result.outcome.outcome.name if result.outcome.outcome else "NULL"
            lines.append(f"Outcome: {outcome_name}")
            lines.append(f"Path: {result.outcome.path.name}")
            if result.outcome.sink_name:
                lines.append(f"Sink: {result.outcome.sink_name}")
            lines.append(f"Completed: {result.outcome.completed}")
            lines.append("")

        # Node states
        if result.node_states:
            lines.append("--- Node States ---")
            for state in result.node_states:
                # Direct access to .value - Tier 1 trust (our audit data)
                # No defensive hasattr - if status isn't an Enum, crash
                lines.append(f"  [{state.step_index}] {state.node_id}: {state.status.value}")
            lines.append("")

        # Routing events
        if result.routing_events:
            lines.append("--- Routing Events ---")
            for event in result.routing_events:
                # Direct access to .value - Tier 1 trust (our audit data)
                parts = [
                    f"[{event.ordinal}]",
                    event.mode.value,
                    f"edge={event.edge_id}",
                    f"group={event.routing_group_id}",
                ]
                if event.reason_hash is not None:
                    parts.append(f"reason_hash={event.reason_hash}")
                lines.append(f"  {' '.join(parts)}")
            lines.append("")

        # Scheduler events
        if result.scheduler_events:
            lines.append("--- Scheduler Events ---")
            for scheduler_event in result.scheduler_events:
                parts = [
                    scheduler_event.event_type.value,
                    f"work_item={scheduler_event.work_item_id}",
                ]
                if scheduler_event.node_id is not None:
                    parts.append(f"node={scheduler_event.node_id}")
                if scheduler_event.from_status is not None:
                    parts.append(f"from={scheduler_event.from_status.value}")
                parts.append(f"to={scheduler_event.to_status.value}")
                parts.append(f"attempt={scheduler_event.to_attempt}")
                if scheduler_event.to_lease_owner is not None:
                    parts.append(f"owner={scheduler_event.to_lease_owner}")
                if scheduler_event.caller_owner is not None:
                    parts.append(f"caller={scheduler_event.caller_owner}")
                lines.append(f"  {' '.join(parts)}")
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

    def flatten(self, record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested mappings to dot-notation keys.

        Raises:
            ValueError: If flattened keys collide (e.g., dotted source keys
                vs nested mapping keys produce identical paths).
        """
        result: dict[str, Any] = {}

        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, Mapping):
                if not value:
                    # Preserve empty mappings as JSON "{}" — an empty object is a
                    # distinct datum from absence. Auditors must be able to tell
                    # "config was explicitly empty" from "config was not present".
                    if full_key in result:
                        raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                    result[full_key] = "{}"
                    continue
                nested = self.flatten(dict(value.items()), full_key)
                for nested_key, nested_val in nested.items():
                    if nested_key in result:
                        raise ValueError(
                            f"CSV flatten key collision: '{nested_key}' produced by both a nested mapping and a prior key. "
                            "Audit export would lose data."
                        )
                    result[nested_key] = nested_val
            elif isinstance(value, list):
                if full_key in result:
                    raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                # Convert lists to JSON strings for CSV
                # Use serialize_datetime to validate (rejects NaN/Infinity) and convert datetimes
                result[full_key] = json.dumps(serialize_datetime(value))
            else:
                if full_key in result:
                    raise ValueError(f"CSV flatten key collision: '{full_key}' already exists. Audit export would lose data.")
                # Validate scalar values and normalize datetimes to ISO strings.
                result[full_key] = serialize_datetime(value)

        return result

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        """Format as flat dict for CSV."""
        return self.flatten(record)
