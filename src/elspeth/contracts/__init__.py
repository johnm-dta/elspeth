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
    SecretResolution,
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

# Schema contracts (Phase 2: Source Integration)
from elspeth.contracts.contract_builder import ContractBuilder

# Schema contracts (Phase 3: Pipeline Integration)
from elspeth.contracts.contract_propagation import (
    merge_contract_with_output,
    narrow_contract_to_output,
    propagate_contract,
)

# Schema contracts (Phase 4: Audit Trail Integration)
from elspeth.contracts.contract_records import (
    ContractAuditRecord,
    FieldAuditRecord,
    ValidationErrorWithContract,
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
    error_edge_label,
)
from elspeth.contracts.errors import (
    BatchPendingError,
    CoalesceFailureReason,
    ConfigGateReason,
    # Schema contract violations
    ContractMergeError,
    ContractViolation,
    ErrorDetail,
    ExecutionError,
    ExtraFieldViolation,
    FrameworkBugError,
    GracefulShutdownError,
    MissingFieldViolation,
    PluginContractViolation,
    QueryFailureDetail,
    RoutingReason,
    RowErrorEntry,
    SourceQuarantineReason,
    TemplateErrorEntry,
    TransformActionCategory,
    TransformErrorCategory,
    TransformErrorReason,
    TransformSuccessReason,
    TypeMismatchViolation,
    UsageStats,
    violations_to_error_reason,
)
from elspeth.contracts.events import (
    ExternalCallCompleted,
    FieldResolutionApplied,
    GateEvaluated,
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RowCreated,
    RunCompletionStatus,
    RunFinished,
    RunStarted,
    RunSummary,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.contracts.header_modes import (
    HeaderMode,
    parse_header_mode,
    resolve_headers,
)
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.payload_store import IntegrityError, PayloadStore
from elspeth.contracts.plugin_context import (
    PluginContext,
    TransformErrorToken,
    ValidationErrorToken,
)
from elspeth.contracts.results import (
    ArtifactDescriptor,
    ExceptionResult,
    FailureInfo,
    GateResult,
    RowResult,
    SourceRow,
    TransformResult,
)
from elspeth.contracts.routing import (
    EdgeInfo,
    RouteDestination,
    RouteDestinationKind,
    RoutingAction,
    RoutingSpec,
)

# Schema contracts (Phase 1: Core Contracts)
from elspeth.contracts.schema_contract import (
    FieldContract,
    PipelineRow,
    SchemaContract,
)
from elspeth.contracts.schema_contract_factory import (
    create_contract_from_config,
    map_schema_mode,
)
from elspeth.contracts.sink import OutputValidationResult
from elspeth.contracts.transform_contract import (
    create_output_contract_from_schema,
    validate_output_against_contract,
)
from elspeth.contracts.type_normalization import normalize_type_for_contract
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
    StepResolver,
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
    "GracefulShutdownError",
    "PluginContractViolation",
    "CoalesceFailureReason",
    "ConfigGateReason",
    "ErrorDetail",
    "ExecutionError",
    "QueryFailureDetail",
    "RoutingReason",
    "RowErrorEntry",
    "SourceQuarantineReason",
    "TemplateErrorEntry",
    "TransformActionCategory",
    "TransformErrorCategory",
    "TransformErrorReason",
    "TransformSuccessReason",
    "UsageStats",
    # schema contract violations
    "ContractMergeError",
    "ContractViolation",
    "ExtraFieldViolation",
    "MissingFieldViolation",
    "TypeMismatchViolation",
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
    "SecretResolution",
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
    "error_edge_label",
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
    "StepResolver",
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
    "RouteDestination",
    "RouteDestinationKind",
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
    # plugin_context
    "PluginContext",
    "TransformErrorToken",
    "ValidationErrorToken",
    # events
    "ExternalCallCompleted",
    "FieldResolutionApplied",
    "GateEvaluated",
    "PhaseAction",
    "PhaseChanged",
    "PhaseCompleted",
    "PhaseError",
    "PhaseStarted",
    "PipelinePhase",
    "RowCreated",
    "RunCompletionStatus",
    "RunFinished",
    "RunStarted",
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
    # schema contracts (Phase 1: Core Contracts)
    "ContractBuilder",
    "create_contract_from_config",
    "FieldContract",
    "map_schema_mode",
    "normalize_type_for_contract",
    "PipelineRow",
    "SchemaContract",
    # schema contracts (Phase 4: Audit Trail Integration)
    "ContractAuditRecord",
    "FieldAuditRecord",
    "ValidationErrorWithContract",
    # schema contracts (Phase 3: Pipeline Integration)
    "create_output_contract_from_schema",
    "HeaderMode",
    "merge_contract_with_output",
    "narrow_contract_to_output",
    "parse_header_mode",
    "propagate_contract",
    "resolve_headers",
    "validate_output_against_contract",
    "violations_to_error_reason",
]
