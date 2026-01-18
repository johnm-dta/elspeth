# src/elspeth/core/landscape/models.py
"""Dataclass models for Landscape audit tables.

These models define the schema for tracking:
- Runs and their configuration
- Nodes (plugin instances) in the execution graph
- Rows loaded from sources
- Tokens (row instances flowing through DAG paths)
- Node states (what happened at each node for each token)
- External calls
- Artifacts produced by sinks
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from elspeth.contracts import (
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)


@dataclass
class Run:
    """A single execution of a pipeline."""

    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: RunStatus
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None
    # Export tracking - separate from run status
    export_status: ExportStatus | None = None
    export_error: str | None = None
    exported_at: datetime | None = None
    export_format: str | None = None  # csv, json
    export_sink: str | None = None


@dataclass
class Node:
    """A node (plugin instance) in the execution graph."""

    node_id: str
    run_id: str
    plugin_name: str
    node_type: NodeType
    plugin_version: str
    determinism: Determinism
    config_hash: str
    config_json: str
    registered_at: datetime
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None


@dataclass
class Edge:
    """An edge in the execution graph."""

    edge_id: str
    run_id: str
    from_node_id: str
    to_node_id: str
    label: str  # "continue", route name, etc.
    default_mode: RoutingMode
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
    source_data_ref: str | None = None  # Payload store reference


@dataclass
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    fork_group_id: str | None = None
    join_group_id: str | None = None
    branch_name: str | None = None
    step_in_pipeline: int | None = (
        None  # Step where this token was created (fork/coalesce)
    )


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int


@dataclass(frozen=True)
class NodeStateOpen:
    """A node state that is currently being processed.

    This is the initial state created by begin_node_state().
    Processing has started but not yet completed.

    Invariants:
    - No output_hash (output not produced yet)
    - No completed_at (not completed yet)
    - No duration_ms (not finished timing)
    - No error_json (no error yet)
    - No context_after_json (processing not done)
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

    Created when complete_node_state() is called with status="completed".

    Invariants:
    - Has output_hash (processing produced output)
    - Has completed_at (finished processing)
    - Has duration_ms (timing complete)
    - No error_json (no error occurred)
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

    Created when complete_node_state() is called with status="failed".

    Invariants:
    - Has completed_at (finished, albeit with failure)
    - Has duration_ms (timing complete)
    - May have error_json (error details if captured)
    - May have output_hash (partial output in some cases)
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


# Discriminated union type - use status field to discriminate
NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed
"""Union type for all node states.

Use isinstance() or check the status field to discriminate:
    if state.status == NodeStateStatus.OPEN:
        # state is NodeStateOpen
    elif state.status == NodeStateStatus.COMPLETED:
        # state is NodeStateCompleted
    elif state.status == NodeStateStatus.FAILED:
        # state is NodeStateFailed
"""


@dataclass
class Call:
    """An external call made during node processing."""

    call_id: str
    state_id: str
    call_index: int
    call_type: str  # llm, http, sql, filesystem
    status: str  # success, error
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
    artifact_type: str
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime
    idempotency_key: str | None = None  # For retry deduplication


@dataclass
class RoutingEvent:
    """A routing decision at a gate node."""

    event_id: str
    state_id: str
    edge_id: str
    routing_group_id: str
    ordinal: int
    mode: str  # move, copy
    created_at: datetime
    reason_hash: str | None = None
    reason_ref: str | None = None


@dataclass
class Batch:
    """An aggregation batch collecting tokens."""

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: str  # draft, executing, completed, failed
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_reason: str | None = None
    trigger_type: str | None = None  # TriggerType enum value
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
    sequence_number is monotonically increasing within a run.
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
    """Lineage information for a row with graceful payload degradation.

    Used by explain_row() to report row lineage even when payloads
    have been purged. The hash is always preserved, but the actual
    data may be unavailable.
    """

    row_id: str
    """Unique identifier for the row."""

    run_id: str
    """Run this row belongs to."""

    source_hash: str
    """Hash of the original source data (always preserved)."""

    source_data: dict[str, object] | None
    """Original source data, or None if payload was purged."""

    payload_available: bool
    """True if source_data is available, False if purged or unavailable."""
