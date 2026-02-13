"""All status codes, modes, and kinds used across subsystem boundaries.

CRITICAL: Every plugin MUST declare a Determinism value at registration.
There is no "unknown" - undeclared determinism crashes at registration time.
This is per ELSPETH's principle: "I don't know what happened" is never acceptable.
"""

from enum import StrEnum


class RunStatus(StrEnum):
    """Status of a pipeline run.

    Stored in the database (runs.status).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class NodeStateStatus(StrEnum):
    """Status of a node processing a token.

    Stored in database (node_states.status).
    """

    OPEN = "open"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatus(StrEnum):
    """Status of run export operation.

    Stored in the database.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(StrEnum):
    """Status of an aggregation batch.

    Stored in database (batches.status).
    """

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerType(StrEnum):
    """Type of trigger that caused an aggregation batch to execute.

    Stored in database (batches.trigger_type).

    Values:
        COUNT: Batch reached configured row count threshold
        TIMEOUT: Batch reached configured time limit
        CONDITION: Custom condition expression evaluated to true
        END_OF_SOURCE: Source exhausted, flush remaining rows
        MANUAL: Explicitly triggered via API/CLI
    """

    COUNT = "count"
    TIMEOUT = "timeout"
    CONDITION = "condition"
    END_OF_SOURCE = "end_of_source"
    MANUAL = "manual"


class NodeType(StrEnum):
    """Type of node in the execution graph.

    Stored in database (nodes.node_type).
    """

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class Determinism(StrEnum):
    """Plugin determinism classification for reproducibility.

    Every plugin MUST declare one of these at registration. No default.
    Undeclared determinism = crash at registration time.

    Each value tells you what to do for replay/verify:
    - DETERMINISTIC: Just re-run, expect identical output
    - SEEDED: Capture seed, replay with same seed
    - IO_READ: Capture what was read (time, files, env)
    - IO_WRITE: Be careful - has side effects on replay
    - EXTERNAL_CALL: Record request/response for replay
    - NON_DETERMINISTIC: Must record output, cannot reproduce

    Stored in database (nodes.determinism).
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    EXTERNAL_CALL = "external_call"
    NON_DETERMINISTIC = "non_deterministic"


class RoutingKind(StrEnum):
    """Kind of routing action from a gate.

    Stored in routing_events.
    """

    CONTINUE = "continue"
    ROUTE = "route"
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(StrEnum):
    """Mode for routing edges.

    MOVE: Token exits current path, goes to destination only
    COPY: Token clones to destination AND continues on current path
    DIVERT: Token is diverted from normal flow to error/quarantine sink.
            Like MOVE, but semantically distinct: represents failure handling,
            not intentional routing. Used for source quarantine and transform
            on_error edges. These are structural markers in the DAG — rows
            reach these sinks via exception handling, not by traversing the edge.

    Stored in the database.
    """

    MOVE = "move"
    COPY = "copy"
    DIVERT = "divert"


class RowOutcome(StrEnum):
    """Outcome for a token in the pipeline.

    These outcomes are explicitly recorded in the `token_outcomes` table
    (AUD-001) at determination time. The (StrEnum) base allows direct
    database storage via .value.

    Most outcomes are TERMINAL - the token's journey is complete:
    - COMPLETED: Reached output sink successfully
    - ROUTED: Sent to named sink by gate
    - FORKED: Split into multiple parallel paths (parent token)
    - FAILED: Processing failed, not recoverable
    - QUARANTINED: Failed validation, stored for investigation
    - CONSUMED_IN_BATCH: Absorbed into aggregate (single/transform mode)
    - COALESCED: Merged in join from parallel paths
    - EXPANDED: Deaggregated into child tokens (parent token)

    One outcome is NON-TERMINAL - the token will reappear:
    - BUFFERED: Held for batch processing in passthrough mode
    """

    # Terminal outcomes
    COMPLETED = "completed"
    ROUTED = "routed"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    COALESCED = "coalesced"
    EXPANDED = "expanded"

    # Non-terminal outcomes
    BUFFERED = "buffered"

    @property
    def is_terminal(self) -> bool:
        """Check if this outcome represents a final state for the token.

        Terminal outcomes mean the token's journey is complete - it won't
        appear again in results. Non-terminal outcomes (BUFFERED) mean
        the token is temporarily held and will reappear with a final outcome.
        """
        return self != RowOutcome.BUFFERED


class CallType(StrEnum):
    """Type of external call (Phase 6).

    Stored in database (calls.call_type).
    """

    LLM = "llm"
    HTTP = "http"
    HTTP_REDIRECT = "http_redirect"
    SQL = "sql"
    FILESYSTEM = "filesystem"


class CallStatus(StrEnum):
    """Status of an external call (Phase 6).

    Stored in database (calls.status).
    """

    SUCCESS = "success"
    ERROR = "error"


class RunMode(StrEnum):
    """Pipeline execution mode for live/replay/verify behavior.

    Stored in database (runs.run_mode).

    Values:
        LIVE: Make real API calls, record everything
        REPLAY: Use recorded responses, skip live calls
        VERIFY: Make real calls, compare to recorded
    """

    LIVE = "live"
    REPLAY = "replay"
    VERIFY = "verify"


class TelemetryGranularity(StrEnum):
    """Granularity of telemetry events emitted by the TelemetryManager.

    Values:
        LIFECYCLE: Only run start/complete/failed events (minimal overhead)
        ROWS: Lifecycle + row-level events (row_started, row_completed, etc.)
        FULL: Rows + external call events (LLM requests, HTTP calls, etc.)
    """

    LIFECYCLE = "lifecycle"
    ROWS = "rows"
    FULL = "full"


class BackpressureMode(StrEnum):
    """How to handle backpressure when telemetry exporters can't keep up.

    Values:
        BLOCK: Block the pipeline until exporters catch up (safest, may slow pipeline)
        DROP: Drop events when buffer is full (lossy, no pipeline impact)
        SLOW: Adaptive rate limiting (not yet implemented)
    """

    BLOCK = "block"
    DROP = "drop"
    SLOW = "slow"


# Backpressure modes that are currently implemented.
# Used by RuntimeTelemetryConfig.from_settings() to fail fast on unimplemented modes.
_IMPLEMENTED_BACKPRESSURE_MODES = frozenset({BackpressureMode.BLOCK, BackpressureMode.DROP})


class OutputMode(StrEnum):
    """Output mode for aggregation batches.

    Stored in database.

    Values:
        PASSTHROUGH: Emit buffered rows unchanged after flush
        TRANSFORM: Emit transformed output from aggregation plugin
    """

    PASSTHROUGH = "passthrough"
    TRANSFORM = "transform"


def error_edge_label(transform_id: str) -> str:
    """Canonical label for a transform error DIVERT edge.

    Shared between DAG construction (dag.py) and error-routing audit recording
    (executors.py, processor.py) to prevent label drift.

    Args:
        transform_id: Stable transform name for error-route labels.
    """
    return f"__error_{transform_id}__"
