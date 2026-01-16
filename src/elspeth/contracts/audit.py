"""Audit trail contracts for Landscape tables.

These are strict contracts - all enum fields use proper enum types.
Repository layer handles string→enum conversion for DB reads.

Per Data Manifesto: The audit database is OUR data. If we read
garbage from it, something catastrophic happened - crash immediately.
"""

from dataclasses import dataclass
from datetime import datetime

from elspeth.contracts.enums import (
    Determinism,
    ExportStatus,
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
    branch_name: str | None = None
    step_in_pipeline: int | None = None


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int
