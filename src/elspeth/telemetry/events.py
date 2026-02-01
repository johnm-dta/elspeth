# src/elspeth/telemetry/events.py
"""Telemetry event definitions for pipeline observability.

These events are emitted during pipeline execution and exported to
external observability platforms. They complement (not replace) the
Landscape audit trail:

- Landscape: Legal record, complete lineage, persisted forever
- Telemetry: Operational visibility, real-time streaming, ephemeral

Event categories:
- Lifecycle: Run start/complete, phase transitions
- Row-level: Token creation, transform completion, gate routing, token completion
- External calls: LLM/HTTP/SQL call completion with timing and status
"""

from dataclasses import dataclass
from typing import Any

from elspeth.contracts.enums import (
    CallStatus,
    CallType,
    RunStatus,
)
from elspeth.contracts.events import PhaseAction, PipelinePhase, TelemetryEvent

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


# =============================================================================
# Row-Level Events
# =============================================================================
# NOTE: TransformCompleted, GateEvaluated, TokenCompleted moved to contracts/events.py
# as they cross the engine<->telemetry boundary.


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
