"""Audit trail contracts for Landscape tables.

These are strict contracts - all enum fields use proper enum types.
Repository layer handles string→enum conversion for DB reads.

Per Data Manifesto: The audit database is OUR data. If we read
garbage from it, something catastrophic happened - crash immediately.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

if TYPE_CHECKING:
    pass  # Placeholder for future type-only imports

from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RowOutcome,
    RunStatus,
)


def _validate_enum(value: object, enum_type: type, field_name: str) -> None:
    """Validate that value is an instance of the expected enum type.

    Tier 1 audit data must crash on invalid types - no coercion, no defaults.
    Per Data Manifesto: If we read garbage from our own database,
    something catastrophic happened - crash immediately.
    """
    if value is not None and not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}: {value!r}")


@dataclass
class Run:
    """A single execution of a pipeline.

    Strict contract - status must be RunStatus enum.
    """

    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: RunStatus  # Strict: enum only
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None
    export_status: ExportStatus | None = None  # Strict: enum only
    export_error: str | None = None
    exported_at: datetime | None = None
    export_format: str | None = None
    export_sink: str | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.status, RunStatus, "status")
        _validate_enum(self.export_status, ExportStatus, "export_status")


@dataclass
class Node:
    """A node (plugin instance) in the execution graph.

    Strict contract - node_type and determinism must be enums.
    """

    node_id: str
    run_id: str
    plugin_name: str
    node_type: NodeType  # Strict: enum only
    plugin_version: str
    determinism: Determinism  # Strict: enum only
    config_hash: str
    config_json: str
    registered_at: datetime
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None
    # Schema configuration for audit trail (WP-11.99)
    schema_mode: str | None = None  # "dynamic", "strict", "free", "parse"
    schema_fields: list[dict[str, object]] | None = None  # Field definitions if explicit

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.node_type, NodeType, "node_type")
        _validate_enum(self.determinism, Determinism, "determinism")


@dataclass
class Edge:
    """An edge in the execution graph.

    Strict contract - default_mode must be RoutingMode enum.
    """

    edge_id: str
    run_id: str
    from_node_id: str
    to_node_id: str
    label: str
    default_mode: RoutingMode  # Strict: enum only
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.default_mode, RoutingMode, "default_mode")


@dataclass
class Row:
    """A source row loaded into the system."""

    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str
    created_at: datetime
    source_data_ref: str | None = None


@dataclass
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None  # For deaggregation grouping
    branch_name: str | None = None
    step_in_pipeline: int | None = None  # Step where token was created (fork/coalesce/expand)


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int


@dataclass(frozen=True)
class NodeStateOpen:
    """A node state currently being processed.

    Invariants:
    - No output_hash (not produced yet)
    - No completed_at (not completed)
    - No duration_ms (not finished timing)
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.OPEN]
    input_hash: str
    started_at: datetime
    context_before_json: str | None = None


@dataclass(frozen=True)
class NodeStatePending:
    """A node state where processing completed but output is pending.

    Used for async operations like batch submission where the operation
    completed successfully but the result won't be available until later.

    Invariants:
    - No output_hash (result not available yet)
    - Has completed_at (operation finished)
    - Has duration_ms (timing complete)
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.PENDING]
    input_hash: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    context_before_json: str | None = None
    context_after_json: str | None = None


@dataclass(frozen=True)
class NodeStateCompleted:
    """A node state that completed successfully.

    Invariants:
    - Has output_hash (produced output)
    - Has completed_at (finished)
    - Has duration_ms (timing complete)
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.COMPLETED]
    input_hash: str
    started_at: datetime
    output_hash: str
    completed_at: datetime
    duration_ms: float
    context_before_json: str | None = None
    context_after_json: str | None = None
    success_reason_json: str | None = None


@dataclass(frozen=True)
class NodeStateFailed:
    """A node state that failed during processing.

    Invariants:
    - Has completed_at (finished, with failure)
    - Has duration_ms (timing complete)
    - May have error_json
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.FAILED]
    input_hash: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    error_json: str | None = None
    output_hash: str | None = None
    context_before_json: str | None = None
    context_after_json: str | None = None


# Discriminated union type
NodeState = NodeStateOpen | NodeStatePending | NodeStateCompleted | NodeStateFailed


@dataclass
class Call:
    """An external call made during node processing or operation.

    Strict contract - call_type and status must be enums.

    Calls can be parented by either:
    - node_state (transform processing): state_id is set, operation_id is None
    - operation (source/sink I/O): operation_id is set, state_id is None

    The XOR constraint is enforced at the database level.
    """

    call_id: str
    call_index: int
    call_type: CallType  # Strict: enum only
    status: CallStatus  # Strict: enum only
    request_hash: str
    created_at: datetime
    # Parent context - exactly one must be set (XOR)
    state_id: str | None = None  # For transform calls
    operation_id: str | None = None  # For source/sink calls
    request_ref: str | None = None
    response_hash: str | None = None
    response_ref: str | None = None
    error_json: str | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.call_type, CallType, "call_type")
        _validate_enum(self.status, CallStatus, "status")


@dataclass
class Artifact:
    """An artifact produced by a sink."""

    artifact_id: str
    run_id: str
    produced_by_state_id: str
    sink_node_id: str
    artifact_type: str  # Not enum - user-defined (csv, json, webhook, etc.)
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime
    idempotency_key: str | None = None  # For retry deduplication


@dataclass
class RoutingEvent:
    """A routing decision at a gate node.

    Strict contract - mode must be RoutingMode enum.
    """

    event_id: str
    state_id: str
    edge_id: str
    routing_group_id: str
    ordinal: int
    mode: RoutingMode  # Strict: enum only
    created_at: datetime
    reason_hash: str | None = None
    reason_ref: str | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.mode, RoutingMode, "mode")


@dataclass
class Batch:
    """An aggregation batch collecting tokens.

    Strict contract - status must be BatchStatus enum.
    """

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: BatchStatus  # Strict: enum only
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_type: str | None = None  # TriggerType enum value (count, time, end_of_source, manual)
    trigger_reason: str | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        _validate_enum(self.status, BatchStatus, "status")


@dataclass
class BatchMember:
    """A token belonging to a batch."""

    batch_id: str
    token_id: str
    ordinal: int


@dataclass
class BatchOutput:
    """An output produced by a batch."""

    batch_id: str
    output_type: str  # token, artifact
    output_id: str


@dataclass
class Checkpoint:
    """Checkpoint for crash recovery.

    Captures run progress at row/transform boundaries.

    Format Versions:
        Version 1: Pre-deterministic node IDs (legacy, incompatible)
        Version 2: Deterministic node IDs (2026-01-24+, current)
    """

    # Current checkpoint format version (ClassVar excludes from dataclass fields)
    CURRENT_FORMAT_VERSION: ClassVar[int] = 2

    checkpoint_id: str
    run_id: str
    token_id: str
    node_id: str
    sequence_number: int
    created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)
    # Topology validation fields - REQUIRED for checkpoint compatibility checking
    # Schema enforces NOT NULL - these are audit-critical for resume validation
    upstream_topology_hash: str  # Hash of ALL nodes + edges in DAG (full topology)
    checkpoint_node_config_hash: str  # Hash of checkpoint node config only
    # Optional fields (with defaults) MUST come after required fields in dataclass
    aggregation_state_json: str | None = None
    # Format version for compatibility checking
    format_version: int | None = None

    def __post_init__(self) -> None:
        """Validate required fields - Tier 1 crash on invalid data.

        Per Data Manifesto: Audit data is OUR data. If we receive None
        for required hash fields, that's a bug in our code - crash immediately.
        """
        if not self.upstream_topology_hash:
            raise ValueError("upstream_topology_hash is required and cannot be empty")
        if not self.checkpoint_node_config_hash:
            raise ValueError("checkpoint_node_config_hash is required and cannot be empty")


@dataclass
class RowLineage:
    """Source row with resolved payload for explain output.

    Combines Row DB record fields with resolved payload data.
    Used by LineageResult.source_row for complete explain output.

    Supports graceful payload degradation - hash always preserved,
    actual data may be unavailable after retention purge.
    """

    # From Row (DB record fields)
    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str  # Consistent naming with Row
    created_at: datetime

    # Resolved payload (from PayloadStore)
    source_data: dict[str, object] | None  # None if purged
    payload_available: bool


class ExportStatusUpdate(TypedDict, total=False):
    """Schema for export status updates in recorder.

    Used by recorder methods that update export-related fields on Run records.
    Uses total=False to allow partial updates.
    """

    export_status: ExportStatus
    exported_at: datetime
    export_error: str
    export_format: str
    export_sink: str


class BatchStatusUpdate(TypedDict, total=False):
    """Schema for batch status updates in recorder.

    Used by recorder methods that update batch-related fields.
    Uses total=False to allow partial updates.
    """

    status: BatchStatus
    completed_at: datetime
    trigger_reason: str
    aggregation_state_id: str


@dataclass
class ValidationErrorRecord:
    """A validation error recorded in the audit trail.

    Created when a source row fails schema validation.
    These are operational errors (bad user data), not system bugs.
    """

    error_id: str
    run_id: str
    node_id: str | None
    row_hash: str
    error: str
    schema_mode: str  # "strict", "free", "dynamic", "parse"
    destination: str
    created_at: datetime
    row_data_json: str | None = None


@dataclass(frozen=True)
class NonCanonicalMetadata:
    """Metadata for non-canonical data stored in the audit trail.

    When data cannot be canonically serialized (contains NaN, Infinity,
    non-dict types, etc.), this metadata captures what we saw for forensic
    analysis.

    This is part of the Tier-3 (external data) trust boundary handling.
    Non-canonical data is quarantined and recorded with this metadata
    instead of crashing the pipeline.

    Invariants:
    - repr_value is never empty (captures what we saw)
    - type_name must be a valid Python type name
    - canonical_error explains why canonical serialization failed

    Fields:
        repr_value: Result of repr(data)
        type_name: type(data).__name__
        canonical_error: Why canonicalization failed
    """

    repr_value: str
    type_name: str
    canonical_error: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for JSON serialization.

        Returns dict with keys matching current inline dict structure
        for backwards compatibility with existing audit data.

        Returns:
            Dict with __repr__, __type__, __canonical_error__ keys
        """
        return {
            "__repr__": self.repr_value,
            "__type__": self.type_name,
            "__canonical_error__": self.canonical_error,
        }

    @classmethod
    def from_error(cls, data: Any, error: Exception) -> "NonCanonicalMetadata":
        """Create metadata from data that failed canonicalization.

        Factory method for convenient creation from exception context.

        Args:
            data: The non-canonical data
            error: The canonicalization exception (ValueError or TypeError)

        Returns:
            NonCanonicalMetadata instance

        Example:
            >>> try:
            ...     canonical_json({"value": float("nan")})
            ... except ValueError as e:
            ...     meta = NonCanonicalMetadata.from_error({"value": float("nan")}, e)
        """
        return cls(
            repr_value=repr(data),
            type_name=type(data).__name__,
            canonical_error=str(error),
        )


@dataclass
class TransformErrorRecord:
    """A transform processing error recorded in the audit trail.

    Created when a transform returns TransformResult.error().
    These are operational errors (bad data values), not transform bugs.
    """

    error_id: str
    run_id: str
    token_id: str
    transform_id: str
    row_hash: str
    destination: str
    created_at: datetime
    row_data_json: str | None = None
    error_details_json: str | None = None


@dataclass(frozen=True)
class TokenOutcome:
    """Recorded terminal state for a token.

    Captures the moment a token reached its terminal (or buffered) state.
    Part of AUD-001 audit integrity - explicit rather than derived.
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: RowOutcome  # Direct type, not forward reference
    is_terminal: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on outcome type)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORKED/EXPANDED

    def __post_init__(self) -> None:
        """Validate enum and bool fields - Tier 1 crash on invalid types."""
        _validate_enum(self.outcome, RowOutcome, "outcome")
        # is_terminal must be bool, not int or other truthy/falsy value
        if not isinstance(self.is_terminal, bool):
            raise TypeError(f"is_terminal must be bool, got {type(self.is_terminal).__name__}: {self.is_terminal!r}")


@dataclass(frozen=True, slots=True)
class Operation:
    """Represents a source/sink I/O operation in the audit trail.

    Operations are the equivalent of node_states for sources and sinks.
    They provide a parent context for external calls made during
    source.load() or sink.write().

    Unlike node_states (which require a token_id because they process
    existing data flow), operations exist at the run/node level because
    sources CREATE tokens rather than processing them.

    Lifecycle:
        1. begin_operation() creates with status='open'
        2. External calls recorded via record_operation_call()
        3. complete_operation() sets status to 'completed' or 'failed'

    The operation_id follows format "op_{uuid4().hex}" to stay within
    the 64-char column limit while remaining globally unique.
    """

    operation_id: str
    run_id: str
    node_id: str
    operation_type: Literal["source_load", "sink_write"]
    started_at: datetime
    status: Literal["open", "completed", "failed", "pending"]
    completed_at: datetime | None = None
    input_data_ref: str | None = None
    output_data_ref: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for database insertion.

        Returns dict with keys matching operations table columns.
        """
        return {
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "operation_type": self.operation_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "input_data_ref": self.input_data_ref,
            "output_data_ref": self.output_data_ref,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }
