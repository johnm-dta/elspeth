# src/elspeth/core/landscape/__init__.py
"""Landscape: The audit backbone for complete traceability.

This module provides the audit infrastructure for ELSPETH's SDA pipelines.
Every decision is traceable to its source through the LandscapeRecorder API.

Primary API:
    LandscapeRecorder - High-level API for recording audit trails
    LandscapeDB - Database connection management

Model Classes:
    Run, Node, Edge - Pipeline structure
    Row, Token, TokenParent - Data flow tracking
    NodeState, Call - Execution recording
    RoutingEvent - Gate decisions
    Batch, BatchMember, BatchOutput - Aggregation tracking
    Artifact - Output artifacts
"""

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.formatters import CSVFormatter, ExportFormatter, JSONFormatter
from elspeth.core.landscape.lineage import LineageResult, explain
from elspeth.core.landscape.models import (
    Artifact,
    Batch,
    BatchMember,
    BatchOutput,
    Call,
    Edge,
    Node,
    NodeState,
    RoutingEvent,
    Row,
    RowLineage,
    Run,
    Token,
    TokenParent,
)
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.reproducibility import (
    ReproducibilityGrade,
    compute_grade,
    set_run_grade,
    update_grade_after_purge,
)
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batch_outputs_table,
    batches_table,
    calls_table,
    edges_table,
    metadata,
    node_states_table,
    nodes_table,
    routing_events_table,
    rows_table,
    runs_table,
    token_parents_table,
    tokens_table,
)

__all__ = [
    "Artifact",
    "Batch",
    "BatchMember",
    "BatchOutput",
    "CSVFormatter",
    "Call",
    "Edge",
    "ExportFormatter",
    "JSONFormatter",
    "LandscapeDB",
    "LandscapeExporter",
    "LandscapeRecorder",
    "LineageResult",
    "Node",
    "NodeState",
    "ReproducibilityGrade",
    "RoutingEvent",
    "Row",
    "RowLineage",
    "Run",
    "Token",
    "TokenParent",
    "artifacts_table",
    "batch_members_table",
    "batch_outputs_table",
    "batches_table",
    "calls_table",
    "compute_grade",
    "edges_table",
    "explain",
    "metadata",
    "node_states_table",
    "nodes_table",
    "routing_events_table",
    "rows_table",
    "runs_table",
    "set_run_grade",
    "token_parents_table",
    "tokens_table",
    "update_grade_after_purge",
]
