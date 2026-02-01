"""Observability events for pipeline execution.

These domain events provide visibility into pipeline phases, progress,
and completion status. Events are emitted by the orchestrator and consumed
by CLI formatters for human-readable or structured output.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from elspeth.contracts.enums import (
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
)


class PipelinePhase(str, Enum):
    """Pipeline lifecycle phases for observability events.

    Uses (str, Enum) pattern for consistency with existing codebase
    (see contracts/enums.py RunStatus).
    """

    CONFIG = "config"
    GRAPH = "graph"
    PLUGINS = "plugins"
    AGGREGATIONS = "aggregations"
    DATABASE = "database"
    SCHEMA_VALIDATION = "schema_validation"
    SOURCE = "source"
    PROCESS = "process"
    EXPORT = "export"


class PhaseAction(str, Enum):
    """Actions within a pipeline phase."""

    LOADING = "loading"
    VALIDATING = "validating"
    BUILDING = "building"
    CONNECTING = "connecting"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    EXPORTING = "exporting"


class RunCompletionStatus(str, Enum):
    """Final status for RunSummary events."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class PhaseStarted:
    """Emitted when a pipeline phase begins.

    Phases represent major lifecycle stages:
    - config: Loading and validating settings
    - graph: Building and validating execution graph
    - plugins: Instantiating source, transforms, and sinks
    - aggregations: Instantiating aggregation plugins
    - database: Connecting to Landscape database
    - schema_validation: Validating plugin schemas
    - source: Loading source data
    - process: Processing rows through transforms
    - export: Exporting results (when enabled)

    Attributes:
        phase: The lifecycle phase starting
        action: What's happening (e.g., "loading", "validating")
        target: Optional target (e.g., file path, plugin name)
    """

    phase: PipelinePhase
    action: PhaseAction
    target: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseCompleted:
    """Emitted when a pipeline phase completes successfully."""

    phase: PipelinePhase
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PhaseError:
    """Emitted when a pipeline phase fails.

    Stores the full exception object to preserve traceback, exception type,
    and chained causes for debugging and audit trail integrity.
    """

    phase: PipelinePhase
    error: BaseException
    target: str | None = None  # What failed (plugin name, file path, etc.)

    @property
    def error_message(self) -> str:
        """Human-readable error message for formatting."""
        return str(self.error)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Summary emitted when pipeline run finishes (success or failure).

    Provides final metrics for CI integration: exit codes, row counts,
    routing breakdown.

    Routing breakdown:
    - routed: Total rows routed to non-default sinks (gates or error routing)
    - routed_destinations: Count per destination sink {sink_name: count}
    """

    run_id: str
    status: RunCompletionStatus
    total_rows: int
    succeeded: int
    failed: int
    quarantined: int
    duration_seconds: float
    exit_code: int  # 0=success, 1=partial failure, 2=total failure
    routed: int = 0  # Rows routed to non-default sinks
    routed_destinations: tuple[tuple[str, int], ...] = ()  # (sink_name, count) pairs


# =============================================================================
# Telemetry Events (Row-Level Observability)
# =============================================================================
# These events are emitted by the engine and consumed by telemetry exporters.
# They provide operational visibility alongside the Landscape audit trail.


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Base class for all telemetry events.

    All events include:
    - timestamp: When the event occurred (UTC)
    - run_id: Pipeline run this event belongs to

    Events are immutable (frozen) for thread-safety and to prevent
    accidental modification during export.
    """

    timestamp: datetime
    run_id: str


@dataclass(frozen=True, slots=True)
class TransformCompleted(TelemetryEvent):
    """Emitted when a transform finishes processing a row.

    Note: input_hash and output_hash are optional because:
    - Failed transforms may not have produced output (output_hash=None)
    - Edge cases during error handling may not have computed input hash
    """

    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    status: NodeStateStatus
    duration_ms: float
    input_hash: str | None
    output_hash: str | None


@dataclass(frozen=True, slots=True)
class GateEvaluated(TelemetryEvent):
    """Emitted when a gate makes a routing decision."""

    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    routing_mode: RoutingMode
    destinations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state."""

    row_id: str
    token_id: str
    outcome: RowOutcome
    sink_name: str | None
