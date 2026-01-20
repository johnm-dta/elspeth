"""Audit trail contracts for Landscape tables.

These are strict contracts - all enum fields use proper enum types.
Repository layer handles string→enum conversion for DB reads.

Per Data Manifesto: The audit database is OUR data. If we read
garbage from it, something catastrophic happened - crash immediately.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict

from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)


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
    schema_mode: str | None = None  # "dynamic", "strict", "free"
    schema_fields: list[dict[str, object]] | None = None  # Field definitions if explicit


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
NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed


@dataclass
class Call:
    """An external call made during node processing.

    Strict contract - call_type and status must be enums.
    """

    call_id: str
    state_id: str
    call_index: int
    call_type: CallType  # Strict: enum only
    status: CallStatus  # Strict: enum only
    request_hash: str
    created_at: datetime
    request_ref: str | None = None
    response_hash: str | None = None
    response_ref: str | None = None
    error_json: str | None = None
    latency_ms: float | None = None


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
    """

    checkpoint_id: str
    run_id: str
    token_id: str
    node_id: str
    sequence_number: int
    created_at: datetime | None
    aggregation_state_json: str | None = None


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
    schema_mode: str
    destination: str
    created_at: datetime
    row_data_json: str | None = None


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
