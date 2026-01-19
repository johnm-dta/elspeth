"""Shared contracts for cross-boundary data types.

All dataclasses, enums, TypedDicts, and NamedTuples that cross subsystem
boundaries MUST be defined here. Internal types are whitelisted in
.contracts-whitelist.yaml.

Import pattern:
    from elspeth.contracts import NodeType, TransformResult, Run
"""

# isort: skip_file
# Import order is load-bearing: config imports MUST come last to avoid circular
# import through core.checkpoint -> core.landscape -> contracts.

from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingKind,
    RoutingMode,
    RowOutcome,
    RunStatus,
    TriggerType,
)
from elspeth.contracts.errors import (
    ExecutionError,
    RoutingReason,
    TransformReason,
)
from elspeth.contracts.audit import (
    Artifact,
    Batch,
    BatchMember,
    BatchOutput,
    BatchStatusUpdate,
    Call,
    Checkpoint,
    Edge,
    ExportStatusUpdate,
    Node,
    NodeState,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    RoutingEvent,
    Row,
    RowLineage,
    Run,
    Token,
    TokenParent,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.results import (
    AcceptResult,
    ArtifactDescriptor,
    FailureInfo,
    GateResult,
    RowResult,
    SourceRow,
    TransformResult,
)
from elspeth.contracts.routing import EdgeInfo, RoutingAction, RoutingSpec
from elspeth.contracts.data import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)
from elspeth.contracts.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
)
from elspeth.contracts.engine import RetryPolicy
from elspeth.contracts.cli import ExecutionResult

__all__ = [
    # audit
    "Artifact",
    # errors
    "ExecutionError",
    "RoutingReason",
    "TransformReason",
    "Batch",
    "BatchMember",
    "BatchOutput",
    "BatchStatusUpdate",
    "Call",
    "Checkpoint",
    "Edge",
    "ExportStatusUpdate",
    "Node",
    "NodeState",
    "NodeStateCompleted",
    "NodeStateFailed",
    "NodeStateOpen",
    "RoutingEvent",
    "Row",
    "RowLineage",
    "Run",
    "Token",
    "TokenParent",
    "TransformErrorRecord",
    "ValidationErrorRecord",
    # config
    "CheckpointSettings",
    "ConcurrencySettings",
    "DatabaseSettings",
    "DatasourceSettings",
    "ElspethSettings",
    "LandscapeExportSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RateLimitSettings",
    "RetrySettings",
    "RowPluginSettings",
    "SinkSettings",
    # enums
    "BatchStatus",
    "CallStatus",
    "CallType",
    "Determinism",
    "ExportStatus",
    "NodeStateStatus",
    "NodeType",
    "RoutingKind",
    "RoutingMode",
    "RowOutcome",
    "RunStatus",
    "TriggerType",
    # identity
    "TokenInfo",
    # results
    "AcceptResult",
    "ArtifactDescriptor",
    "FailureInfo",
    "GateResult",
    "RowResult",
    "SourceRow",
    "TransformResult",
    # routing
    "EdgeInfo",
    "RoutingAction",
    "RoutingSpec",
    # data
    "CompatibilityResult",
    "PluginSchema",
    "SchemaValidationError",
    "check_compatibility",
    "validate_row",
    # engine
    "RetryPolicy",
    # cli
    "ExecutionResult",
]
