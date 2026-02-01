"""All status codes, modes, and kinds used across subsystem boundaries.

CRITICAL: Every plugin MUST declare a Determinism value at registration.
There is no "unknown" - undeclared determinism crashes at registration time.
This is per ELSPETH's principle: "I don't know what happened" is never acceptable.
"""

from enum import Enum


class RunStatus(str, Enum):
    """Status of a pipeline run.

    Uses (str, Enum) because this IS stored in the database (runs.status).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStateStatus(str, Enum):
    """Status of a node processing a token.

    Uses (str, Enum) for database serialization to node_states.status.
    """

    OPEN = "open"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatus(str, Enum):
    """Status of run export operation.

    Uses (str, Enum) for database serialization.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    """Status of an aggregation batch.

    Uses (str, Enum) for database serialization to batches.status.
    """

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerType(str, Enum):
    """Type of trigger that caused an aggregation batch to execute.

    Uses (str, Enum) for database serialization to batches.trigger_type.

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


class NodeType(str, Enum):
    """Type of node in the execution graph.

    Uses (str, Enum) for database serialization to nodes.node_type.
    """

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class Determinism(str, Enum):
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

    Uses (str, Enum) for database serialization to nodes.determinism.
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    EXTERNAL_CALL = "external_call"
    NON_DETERMINISTIC = "non_deterministic"


class RoutingKind(str, Enum):
    """Kind of routing action from a gate.

    Uses (str, Enum) for serialization in routing_events.
    """

    CONTINUE = "continue"
    ROUTE = "route"
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(str, Enum):
    """Mode for routing edges.

    MOVE: Token exits current path, goes to destination only
    COPY: Token clones to destination AND continues on current path

    Uses (str, Enum) for database serialization.
    """

    MOVE = "move"
    COPY = "copy"


class RowOutcome(str, Enum):
    """Outcome for a token in the pipeline.

    These outcomes are explicitly recorded in the `token_outcomes` table
    (AUD-001) at determination time. The (str, Enum) base allows direct
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


class CallType(str, Enum):
    """Type of external call (Phase 6).

    Uses (str, Enum) for database serialization to calls.call_type.
    """

    LLM = "llm"
    HTTP = "http"
    SQL = "sql"
    FILESYSTEM = "filesystem"


class CallStatus(str, Enum):
    """Status of an external call (Phase 6).

    Uses (str, Enum) for database serialization to calls.status.
    """

    SUCCESS = "success"
    ERROR = "error"


class RunMode(str, Enum):
    """Pipeline execution mode for live/replay/verify behavior.

    Uses (str, Enum) for database serialization to runs.run_mode.

    Values:
        LIVE: Make real API calls, record everything
        REPLAY: Use recorded responses, skip live calls
        VERIFY: Make real calls, compare to recorded
    """

    LIVE = "live"
    REPLAY = "replay"
    VERIFY = "verify"


class TelemetryGranularity(str, Enum):
    """Granularity of telemetry events emitted by the TelemetryManager.

    Uses (str, Enum) for YAML/settings parsing and serialization.

    Values:
        LIFECYCLE: Only run start/complete/failed events (minimal overhead)
        ROWS: Lifecycle + row-level events (row_started, row_completed, etc.)
        FULL: Rows + external call events (LLM requests, HTTP calls, etc.)
    """

    LIFECYCLE = "lifecycle"
    ROWS = "rows"
    FULL = "full"


class BackpressureMode(str, Enum):
    """How to handle backpressure when telemetry exporters can't keep up.

    Uses (str, Enum) for YAML/settings parsing and serialization.

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


class OutputMode(str, Enum):
    """Output mode for aggregation batches.

    Uses (str, Enum) for YAML config compatibility and database serialization.

    Values:
        PASSTHROUGH: Emit buffered rows unchanged after flush
        TRANSFORM: Emit transformed output from aggregation plugin
    """

    PASSTHROUGH = "passthrough"
    TRANSFORM = "transform"
