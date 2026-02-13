# src/elspeth/telemetry/filtering.py
"""Event filtering based on telemetry granularity.

Telemetry events are filtered based on configured granularity level:
- LIFECYCLE: Only run start/complete/failed and phase transitions
- ROWS: Lifecycle + row-level events (creation, transforms, gates, completion)
- FULL: Rows + external call events (LLM, HTTP, SQL)

This module provides the single source of truth for granularity filtering,
used by TelemetryManager to decide which events to emit.
"""

from elspeth.contracts.enums import TelemetryGranularity
from elspeth.contracts.events import (
    ExternalCallCompleted,
    FieldResolutionApplied,
    GateEvaluated,
    PhaseChanged,
    RowCreated,
    RunFinished,
    RunStarted,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)


def should_emit(event: TelemetryEvent, granularity: TelemetryGranularity) -> bool:
    """Determine whether an event should be emitted based on granularity.

    Filter logic:
    - Lifecycle events (RunStarted, RunFinished, PhaseChanged): Always emit
    - Row events (
      RowCreated, TransformCompleted, GateEvaluated, TokenCompleted, FieldResolutionApplied
      ): Emit at ROWS or FULL granularity
    - External call events (ExternalCallCompleted): Emit only at FULL granularity
    - Unknown event types: Always emit (fail-open for forward compatibility)

    Args:
        event: The telemetry event to check
        granularity: The configured granularity level

    Returns:
        True if the event should be emitted, False otherwise

    Example:
        >>> from datetime import datetime, UTC
        >>> from elspeth.contracts import TelemetryGranularity
        >>> event = RunStarted(
        ...     timestamp=datetime.now(tz=UTC),
        ...     run_id="run-123",
        ...     config_hash="abc",
        ...     source_plugin="csv"
        ... )
        >>> should_emit(event, TelemetryGranularity.LIFECYCLE)
        True
    """
    match event:
        # Lifecycle events: always emit at any granularity
        case RunStarted() | RunFinished() | PhaseChanged():
            return True

        # Row-level events: emit at ROWS or FULL
        case RowCreated() | TransformCompleted() | GateEvaluated() | TokenCompleted() | FieldResolutionApplied():
            return granularity in (TelemetryGranularity.ROWS, TelemetryGranularity.FULL)

        # External call events: emit only at FULL
        case ExternalCallCompleted():
            return granularity == TelemetryGranularity.FULL

        # Unknown event types: pass through (fail-open for forward compatibility).
        # This ensures newly introduced telemetry events are visible immediately
        # even before this filter is explicitly updated.
        case _:
            return True
