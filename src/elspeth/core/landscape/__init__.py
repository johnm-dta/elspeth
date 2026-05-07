"""Landscape: The audit backbone for complete traceability.

This module provides the audit infrastructure for ELSPETH's SDA pipelines.
Every decision is traceable to its source through the RecorderFactory API.

Primary API:
    RecorderFactory - Constructs domain repositories from LandscapeDB
    LandscapeDB - Database connection management

Model Classes:
    Run, Node, Edge - Pipeline structure
    Row, Token, TokenParent - Data flow tracking
    NodeState, Call - Execution recording
    RoutingEvent - Gate decisions
    Batch, BatchMember, BatchOutput - Aggregation tracking
    Artifact - Output artifacts

Schema Contract Audit Records (Phase 5):
    ContractAuditRecord - Serialized schema contract for audit storage
    FieldAuditRecord - Individual field contract information
    ValidationErrorWithContract - Schema validation error with contract context
"""

from elspeth.contracts import (
    Artifact,
    Batch,
    BatchMember,
    BatchOutput,
    Call,
    CallStatus,
    CallType,
    Checkpoint,
    # Schema contract audit records (Phase 5: Audit Trail Integration)
    ContractAuditRecord,
    Edge,
    FieldAuditRecord,
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStateStatus,
    ReproducibilityGrade,
    RoutingEvent,
    RoutingSpec,
    Row,
    RowLineage,
    Run,
    RunStatus,
    Token,
    TokenParent,
    ValidationErrorWithContract,
)
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.formatters import (
    CSVFormatter,
    JSONFormatter,
    LineageTextFormatter,
    dataclass_to_dict,
    serialize_datetime,
)
from elspeth.core.landscape.lineage import LineageResult, explain
from elspeth.core.landscape.query_repository import QueryRepository
from elspeth.core.landscape.reproducibility import (
    compute_grade,
    update_grade_after_purge,
)
from elspeth.core.landscape.row_data import CallDataResult, CallDataState, RowDataResult, RowDataState
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
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
    "CallDataResult",
    "CallDataState",
    "CallStatus",
    "CallType",
    "Checkpoint",
    "ContractAuditRecord",
    "DataFlowRepository",
    "Edge",
    "ExecutionRepository",
    "FieldAuditRecord",
    "JSONFormatter",
    "LandscapeDB",
    "LandscapeExporter",
    "LineageResult",
    "LineageTextFormatter",
    "Node",
    "NodeState",
    "NodeStateCompleted",
    "NodeStateFailed",
    "NodeStateOpen",
    "NodeStateStatus",
    "QueryRepository",
    "RecorderFactory",
    "ReproducibilityGrade",
    "RoutingEvent",
    "RoutingSpec",
    "Row",
    "RowDataResult",
    "RowDataState",
    "RowLineage",
    "Run",
    "RunLifecycleRepository",
    "RunStatus",
    "SchemaCompatibilityError",
    "Token",
    "TokenParent",
    "ValidationErrorWithContract",
    "artifacts_table",
    "batch_members_table",
    "batch_outputs_table",
    "batches_table",
    "calls_table",
    "compute_grade",
    "dataclass_to_dict",
    "edges_table",
    "explain",
    "metadata",
    "node_states_table",
    "nodes_table",
    "routing_events_table",
    "rows_table",
    "runs_table",
    "serialize_datetime",
    "token_parents_table",
    "tokens_table",
    "update_grade_after_purge",
]
