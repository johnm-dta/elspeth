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


@dataclass
class Run:
    """A single execution of a pipeline."""

    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: str  # running, completed, failed
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None
    # Export tracking - separate from run status
    export_status: str | None = None  # pending, completed, failed
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
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_version: str
    determinism: str  # From Determinism enum: deterministic, seeded, nondeterministic
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
    default_mode: str  # "move" or "copy"
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
    step_in_pipeline: int | None = None  # Step where this token was created (fork/coalesce)


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int


@dataclass
class NodeState:
    """What happened when a token visited a node."""

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: str  # open, completed, failed
    input_hash: str
    started_at: datetime
    output_hash: str | None = None
    context_before_json: str | None = None
    context_after_json: str | None = None
    duration_ms: float | None = None
    error_json: str | None = None
    completed_at: datetime | None = None


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
