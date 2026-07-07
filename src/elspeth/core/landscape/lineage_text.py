"""Human-readable lineage rendering for CLI/explain surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.core.landscape.serialization import serialize_datetime

if TYPE_CHECKING:
    from elspeth.core.landscape.lineage import LineageResult


class LineageTextFormatter:
    """Format LineageResult as human-readable text for CLI output."""

    def format(self, result: LineageResult | None) -> str:
        """Format lineage result as text."""
        if result is None:
            return "No lineage found. Token or row may not exist, or processing is incomplete."

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("LINEAGE REPORT")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Token: {result.token.token_id}")
        lines.append(f"Row: {result.token.row_id}")
        if result.token.branch_name:
            lines.append(f"Branch: {result.token.branch_name}")
        lines.append("")

        lines.append("--- Source ---")
        lines.append(f"Row Index: {result.source_row.row_index}")
        lines.append(f"Source Data Hash: {result.source_row.source_data_hash}")
        lines.append(f"Payload Available: {result.source_row.payload_available}")
        if result.source_row.source_data is not None:
            lines.append(f"Source Data: {serialize_datetime(result.source_row.source_data)}")
        lines.append("")

        if result.outcome:
            lines.append("--- Outcome ---")
            outcome_name = result.outcome.outcome.name if result.outcome.outcome else "NULL"
            lines.append(f"Outcome: {outcome_name}")
            lines.append(f"Path: {result.outcome.path.name}")
            if result.outcome.sink_name:
                lines.append(f"Sink: {result.outcome.sink_name}")
            lines.append(f"Completed: {result.outcome.completed}")
            lines.append("")

        if result.node_states:
            lines.append("--- Node States ---")
            for state in result.node_states:
                lines.append(f"  [{state.step_index}] {state.node_id}: {state.status.value}")
            lines.append("")

        if result.routing_events:
            lines.append("--- Routing Events ---")
            for event in result.routing_events:
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

        if result.calls:
            lines.append("--- External Calls ---")
            for call in result.calls:
                latency_display = "N/A" if call.latency_ms is None else f"{call.latency_ms:.1f}ms"
                lines.append(f"  {call.call_type.value}: {call.status.value} ({latency_display})")
            lines.append("")

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

        if result.parent_tokens:
            lines.append("--- Parent Tokens ---")
            for parent in result.parent_tokens:
                lines.append(f"  {parent.token_id}")
            lines.append("")

        return "\n".join(lines)
