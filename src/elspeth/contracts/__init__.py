"""Shared contracts for cross-boundary data types.

All dataclasses, enums, TypedDicts, and NamedTuples that cross subsystem
boundaries MUST be defined here. Internal types are whitelisted in
config/cicd/contracts-whitelist.yaml.

This package is a LEAF MODULE with no outbound dependencies to core/engine.
To maintain this property, Settings classes (RetrySettings, ElspethSettings, etc.)
are NOT re-exported here - import them from elspeth.core.config.

Import patterns:
    # Contracts (lightweight, no heavy dependencies)
    from elspeth.contracts import NodeType, TransformResult, Run

    # Settings classes (from core, pulls in heavy deps)
    from elspeth.core.config import RetrySettings, ElspethSettings
"""

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
    NodeStatePending,
    NonCanonicalMetadata,
    Operation,
    RoutingEvent,
    Row,
    RowLineage,
    Run,
    Token,
    TokenOutcome,
    TokenParent,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.checkpoint import ResumeCheck, ResumePoint
from elspeth.contracts.cli import ExecutionResult, ProgressEvent

# =============================================================================
# Settings classes are NOT re-exported from contracts
# =============================================================================
# To maintain contracts as a leaf module (no core dependencies), Settings classes
# must be imported directly from elspeth.core.config:
#     from elspeth.core.config import RetrySettings, ElspethSettings
#
# FIX: P2-2026-01-20-contracts-config-reexport-breaks-leaf-boundary
# =============================================================================
from elspeth.contracts.config import (
    # Alignment documentation
    EXEMPT_SETTINGS,
    FIELD_MAPPINGS,
    # Default registries
    INTERNAL_DEFAULTS,
    POLICY_DEFAULTS,
    SETTINGS_TO_RUNTIME,
    # Runtime config dataclasses
    ExporterConfig,
    # Runtime protocols (what engine components expect)
    RuntimeCheckpointProtocol,
    RuntimeConcurrencyProtocol,
    RuntimeRateLimitProtocol,
    RuntimeRetryProtocol,
    RuntimeTelemetryConfig,
    RuntimeTelemetryProtocol,
)
from elspeth.contracts.data import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)
from elspeth.contracts.engine import PendingOutcome, RetryPolicy
from elspeth.contracts.enums import (
    BackpressureMode,
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
    RunMode,
    RunStatus,
    TelemetryGranularity,
    TriggerType,
)
from elspeth.contracts.errors import (
    BatchPendingError,
    CoalesceFailureReason,
    ConfigGateReason,
    ErrorDetail,
    ExecutionError,
    FrameworkBugError,
    PluginContractViolation,
    PluginGateReason,
    QueryFailureDetail,
    RoutingReason,
    RowErrorEntry,
    TemplateErrorEntry,
    TransformActionCategory,
    TransformErrorCategory,
    TransformErrorReason,
    TransformSuccessReason,
    UsageStats,
)
from elspeth.contracts.events import (
    GateEvaluated,
    PhaseAction,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunSummary,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.payload_store import IntegrityError, PayloadStore
from elspeth.contracts.results import (
    ArtifactDescriptor,
    ExceptionResult,
    FailureInfo,
    GateResult,
    RowResult,
    SourceRow,
    TransformResult,
)
from elspeth.contracts.routing import EdgeInfo, RoutingAction, RoutingSpec
from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.contracts.url import (
    SENSITIVE_PARAMS,
    SanitizedDatabaseUrl,
    SanitizedWebhookUrl,
)

__all__ = [  # Grouped by category for readability
    # audit
    "Artifact",
    "Operation",
    # errors
    "BatchPendingError",
    "FrameworkBugError",
    "PluginContractViolation",
    "CoalesceFailureReason",
    "ConfigGateReason",
    "ErrorDetail",
    "ExecutionError",
    "PluginGateReason",
    "QueryFailureDetail",
    "RoutingReason",
    "RowErrorEntry",
    "TemplateErrorEntry",
    "TransformActionCategory",
    "TransformErrorCategory",
    "TransformErrorReason",
    "TransformSuccessReason",
    "UsageStats",
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
    "NodeStatePending",
    "NonCanonicalMetadata",
    "RoutingEvent",
    "Row",
    "RowLineage",
    "Run",
    "Token",
    "TokenOutcome",
    "TokenParent",
    "TransformErrorRecord",
    "ValidationErrorRecord",
    # config - Runtime protocols (contracts, not core)
    "RuntimeCheckpointProtocol",
    "RuntimeConcurrencyProtocol",
    "RuntimeRateLimitProtocol",
    "RuntimeRetryProtocol",
    "RuntimeTelemetryProtocol",
    # config - Runtime config dataclasses
    "ExporterConfig",
    "RuntimeTelemetryConfig",
    # config - Default registries
    "INTERNAL_DEFAULTS",
    "POLICY_DEFAULTS",
    # config - Alignment documentation
    "EXEMPT_SETTINGS",
    "FIELD_MAPPINGS",
    "SETTINGS_TO_RUNTIME",
    # NOTE: Settings classes (RetrySettings, ElspethSettings, etc.) are NOT here
    # Import them from elspeth.core.config to avoid breaking the leaf boundary
    # enums
    "BackpressureMode",
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
    "RunMode",
    "RunStatus",
    "TelemetryGranularity",
    "TriggerType",
    # identity
    "TokenInfo",
    # checkpoint
    "ResumeCheck",
    "ResumePoint",
    # types
    "AggregationName",
    "BranchName",
    "CoalesceName",
    "GateName",
    "NodeID",
    "SinkName",
    # results (NOTE: AcceptResult deleted in aggregation structural cleanup)
    "ArtifactDescriptor",
    "ExceptionResult",
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
    "PendingOutcome",
    "RetryPolicy",
    # payload_store
    "IntegrityError",
    "PayloadStore",
    # events
    "GateEvaluated",
    "PhaseAction",
    "PhaseCompleted",
    "PhaseError",
    "PhaseStarted",
    "PipelinePhase",
    "RunCompletionStatus",
    "RunSummary",
    "TelemetryEvent",
    "TokenCompleted",
    "TransformCompleted",
    # cli
    "ExecutionResult",
    "ProgressEvent",
    # url
    "SENSITIVE_PARAMS",
    "SanitizedDatabaseUrl",
    "SanitizedWebhookUrl",
    # sink
    "OutputValidationResult",
]
