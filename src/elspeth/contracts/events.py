"""Observability events for pipeline execution.

These domain events provide visibility into pipeline phases, progress,
and completion status. Events are emitted by the orchestrator and consumed
by CLI formatters for human-readable or structured output.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from elspeth.contracts.enums import (
    CallStatus,
    CallType,
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
    RunStatus,
)


class PipelinePhase(StrEnum):
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


class PhaseAction(StrEnum):
    """Actions within a pipeline phase."""

    LOADING = "loading"
    VALIDATING = "validating"
    BUILDING = "building"
    CONNECTING = "connecting"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    EXPORTING = "exporting"


class RunCompletionStatus(StrEnum):
    """Final status for RunSummary events."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    INTERRUPTED = "interrupted"


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


# =============================================================================
# Lifecycle Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunStarted(TelemetryEvent):
    """Emitted when a pipeline run begins.

    Attributes:
        config_hash: Hash of the pipeline configuration for change detection
        source_plugin: Name of the source plugin being used
    """

    config_hash: str
    source_plugin: str


@dataclass(frozen=True, slots=True)
class RunFinished(TelemetryEvent):
    """Emitted when a pipeline run finishes (success or failure).

    Pairs with RunStarted for telemetry lifecycle tracking.

    Attributes:
        status: Final run status (completed, failed)
        row_count: Total rows processed
        duration_ms: Total run duration in milliseconds
    """

    status: RunStatus
    row_count: int
    duration_ms: float


@dataclass(frozen=True, slots=True)
class PhaseChanged(TelemetryEvent):
    """Emitted when pipeline transitions between phases.

    Phases represent major lifecycle stages (config, graph, plugins,
    database, source, process, export). This event fires on phase
    entry with the action being performed.

    Attributes:
        phase: The pipeline phase being entered
        action: What's happening in this phase (loading, validating, etc.)
    """

    phase: PipelinePhase
    action: PhaseAction


@dataclass(frozen=True, slots=True)
class FieldResolutionApplied(TelemetryEvent):
    """Emitted when source field normalization is applied.

    Captures the mapping from original external headers to normalized
    field names. Useful for debugging field name issues and monitoring
    normalization patterns across runs.

    Attributes:
        source_plugin: Name of the source plugin
        field_count: Number of fields in the mapping
        normalization_version: Algorithm version used (None if no normalization)
        resolution_mapping: Complete original->normalized mapping
    """

    source_plugin: str
    field_count: int
    normalization_version: str | None
    resolution_mapping: dict[str, str]


# =============================================================================
# Row-Level Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class RowCreated(TelemetryEvent):
    """Emitted when a new row enters the pipeline from the source.

    Attributes:
        row_id: Stable source row identity
        token_id: Token instance for this row in the DAG
        content_hash: Hash of the row content for deduplication
    """

    row_id: str
    token_id: str
    content_hash: str


# =============================================================================
# External Call Events
# =============================================================================


@dataclass(frozen=True, slots=True)
class ExternalCallCompleted(TelemetryEvent):
    """Emitted when an external call (LLM, HTTP, SQL) completes.

    Calls can originate from two contexts:
    - Transform context: Call made during transform processing (has state_id)
    - Operation context: Call made during source load or sink write (has operation_id)

    Exactly one of state_id or operation_id should be set.

    Attributes:
        state_id: Node state that made the call (for transform context)
        operation_id: Operation that made the call (for source/sink context)
        token_id: Token associated with the transform context, if available
        call_type: Type of external call (llm, http, sql, filesystem)
        provider: Service provider (e.g., "azure-openai", "anthropic")
        status: Call result (success, error)
        latency_ms: Call duration in milliseconds
        request_hash: Hash of request payload for debugging (optional)
        response_hash: Hash of response payload for debugging (optional)
        request_payload: Full request data for observability (optional).
            For LLM calls: contains 'messages' (prompt), 'model', 'temperature', etc.
            For HTTP calls: contains 'method', 'url', 'json', 'headers', etc.
        response_payload: Full response data for observability (optional).
            For LLM calls: contains 'content' (completion), 'model', 'usage', etc.
            For HTTP calls: contains 'status_code', 'headers', 'body', etc.
        token_usage: LLM token counts if applicable (optional)
    """

    call_type: CallType
    provider: str
    status: CallStatus
    latency_ms: float
    state_id: str | None = None
    operation_id: str | None = None
    token_id: str | None = None
    request_hash: str | None = None
    response_hash: str | None = None
    request_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] | None = None
    token_usage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Validate XOR constraint: exactly one of state_id or operation_id must be set."""
        has_state = self.state_id is not None
        has_operation = self.operation_id is not None
        if has_state == has_operation:  # Both True or both False
            raise ValueError(
                f"ExternalCallCompleted requires exactly one of state_id or operation_id. "
                f"Got state_id={self.state_id!r}, operation_id={self.operation_id!r}"
            )
