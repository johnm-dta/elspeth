"""Audit trail contracts for Landscape tables.

These are strict contracts - all enum fields use proper enum types.
Model loader layer handles string→enum conversion for DB reads.

Per Data Manifesto: The audit database is OUR data. If we read
garbage from it, something catastrophic happened - crash immediately.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

from elspeth.contracts.freeze import freeze_fields, require_int

if TYPE_CHECKING:
    pass  # Placeholder for future type-only imports

from elspeth.contracts.enums import (
    _LEGAL_TERMINAL_PAIRS,
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    ReproducibilityGrade,
    RoutingMode,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    AuditExportSigningMode,
    SinkEffectAttemptAction,
    SinkEffectAttemptState,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectReconcileKind,
    SinkEffectRole,
    SinkEffectState,
)


@dataclass(frozen=True, slots=True)
class TokenRef:
    """Bundled reference to a token within a specific run.

    A token_id is meaningless without its run_id — they always travel
    together semantically. This type enforces that coupling at the
    type level, preventing parameter-order bugs and mismatched pairs.

    Construction: Construct directly — all call sites (processor,
    sink executor, plugin context) pair token_id with run_id at the
    point of use. Tier 1 data is trusted at construction time.
    """

    token_id: str
    run_id: str


_SOURCE_FILE_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{16}")
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")

type OperationType = Literal["source_load", "sink_write", "runtime_preflight"]
type ArtifactProducerKind = Literal["node_state", "sink_effect"]
type ArtifactPublicationEvidenceKind = Literal["returned", "reconciled", "inherited", "virtual", "legacy_returned"]

OPERATION_TYPE_VALUES: tuple[OperationType, ...] = ("source_load", "sink_write", "runtime_preflight")


def validate_resolved_prompt_template_hash(call_type: CallType, resolved_prompt_template_hash: str | None) -> None:
    """Validate the cross-DB prompt-hash anchor invariant (Tier 1).

    ``resolved_prompt_template_hash`` is defined only for LLM calls and, when
    present, must be a 64-character lowercase hex digest. Raises ``ValueError``
    on violation; ``None`` is always valid.

    This is the single source of truth for the invariant. ``Call.__post_init__``
    calls it, and the ExecutionRepository write path MUST call it BEFORE the
    audit row is inserted — otherwise a bad hash persists to ``calls`` and only
    surfaces as a post-commit ``ValueError`` when the ``Call`` is constructed,
    violating the Tier-1 guarantee that the audit trail is always pristine.
    """
    if resolved_prompt_template_hash is None:
        return
    if call_type is not CallType.LLM:
        raise ValueError(f"Call.resolved_prompt_template_hash is defined only for CallType.LLM calls, got call_type={call_type!r}")
    if not isinstance(resolved_prompt_template_hash, str) or not _SHA256_HEX_PATTERN.fullmatch(resolved_prompt_template_hash):
        raise ValueError("Call.resolved_prompt_template_hash must be a 64-character lowercase hex digest")


def _validate_enum(value: object, enum_type: type, field_name: str, *, optional: bool = False) -> None:
    """Validate that value is an instance of the expected enum type.

    Tier 1 audit data must crash on invalid types - no coercion, no defaults.
    Per Data Manifesto: If we read garbage from our own database,
    something catastrophic happened - crash immediately.
    """
    if value is None and optional:
        return
    if not isinstance(value, enum_type):
        raise TypeError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}: {value!r}")


def _validate_hash(value: object, field_name: str, *, optional: bool = False) -> None:
    """Validate a lowercase SHA-256 digest without coercion."""
    if value is None and optional:
        return
    if not isinstance(value, str) or _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character lowercase hexadecimal digest")


def _validate_nonempty_string(value: object, field_name: str, *, optional: bool = False) -> None:
    """Validate required audit identifiers without silently trimming them."""
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
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
    reproducibility_grade: ReproducibilityGrade | None = None
    export_status: ExportStatus | None = None  # Strict: enum only
    export_error: str | None = None
    exported_at: datetime | None = None
    export_format: str | None = None
    export_sink: str | None = None
    llm_call_count: int | None = None
    seeded_from_cache: bool = False
    cache_key: str | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        require_int(self.llm_call_count, "llm_call_count", optional=True, min_value=0)
        _validate_enum(self.status, RunStatus, "status")
        _validate_enum(self.reproducibility_grade, ReproducibilityGrade, "reproducibility_grade", optional=True)
        _validate_enum(self.export_status, ExportStatus, "export_status", optional=True)
        if type(self.seeded_from_cache) is not bool:
            raise TypeError(f"seeded_from_cache must be bool, got {type(self.seeded_from_cache).__name__}: {self.seeded_from_cache!r}")
        if self.seeded_from_cache and self.cache_key is None:
            raise ValueError(f"seeded_from_cache=True requires cache_key for run {self.run_id!r}")
        if self.cache_key is not None and not _SHA256_HEX_PATTERN.fullmatch(self.cache_key):
            raise ValueError(f"cache_key must be 64 lowercase hex chars or None, got {self.cache_key!r} for run {self.run_id!r}")


@dataclass(frozen=True, slots=True)
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
    source_file_hash: str | None = None
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None
    # Schema configuration for audit trail (WP-11.99)
    schema_mode: str | None = None  # "observed", "fixed", "flexible", "parse"
    schema_fields: Sequence[Mapping[str, object]] | None = None  # Field definitions if explicit

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        require_int(self.sequence_in_pipeline, "sequence_in_pipeline", optional=True, min_value=0)
        _validate_enum(self.node_type, NodeType, "node_type")
        _validate_enum(self.determinism, Determinism, "determinism")
        if self.source_file_hash is not None and not _SOURCE_FILE_HASH_PATTERN.fullmatch(self.source_file_hash):
            raise ValueError(
                f"Tier 1: source_file_hash must match 'sha256:<16-hex>' or be None, got {self.source_file_hash!r} for node {self.node_id!r}"
            )
        freeze_fields(self, "schema_fields")


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class Row:
    """A source row loaded into the system."""

    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str
    created_at: datetime
    # Mandatory source-scoped identity (G22 fabrication ban): the rows table
    # declares both NOT NULL and they anchor the (run_id, source_node_id,
    # source_row_index) / (run_id, ingest_sequence) unique constraints. Typed
    # `int` (not `int | None`) to match the require_int guard and NOT NULL column;
    # ordered before the genuinely-optional source_data_ref.
    source_row_index: int
    ingest_sequence: int
    source_data_ref: str | None = None  # None when payload stored inline

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.row_index, "row_index", min_value=0)
        require_int(self.source_row_index, "source_row_index", min_value=0)
        require_int(self.ingest_sequence, "ingest_sequence", min_value=0)


@dataclass(frozen=True, slots=True)
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    run_id: str
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None  # For deaggregation grouping
    branch_name: str | None = None
    step_in_pipeline: int | None = None  # Step where token was created (fork/coalesce/expand)
    token_data_ref: str | None = None  # Content-addressable ref for per-token payload (expand/coalesce writers)

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.step_in_pipeline, "step_in_pipeline", optional=True, min_value=0)


@dataclass(frozen=True, slots=True)
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.ordinal, "ordinal", min_value=0)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.step_index, "step_index", min_value=0)
        require_int(self.attempt, "attempt", min_value=0)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.step_index, "step_index", min_value=0)
        require_int(self.attempt, "attempt", min_value=0)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.step_index, "step_index", min_value=0)
        require_int(self.attempt, "attempt", min_value=0)


@dataclass(frozen=True, slots=True)
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

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.step_index, "step_index", min_value=0)
        require_int(self.attempt, "attempt", min_value=0)


# Discriminated union type
NodeState = NodeStateOpen | NodeStatePending | NodeStateCompleted | NodeStateFailed


@dataclass(frozen=True, slots=True)
class NodeStateFieldConstraints:
    """Column-level constraints for one node-state lifecycle status."""

    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


NODE_STATE_LIFECYCLE_FIELD_CONSTRAINTS: Mapping[NodeStateStatus, NodeStateFieldConstraints] = {
    NodeStateStatus.OPEN: NodeStateFieldConstraints(
        forbidden=(
            "output_hash",
            "completed_at",
            "duration_ms",
            "context_after_json",
            "error_json",
            "success_reason_json",
        ),
    ),
    NodeStateStatus.PENDING: NodeStateFieldConstraints(
        required=("completed_at", "duration_ms"),
        forbidden=("output_hash", "error_json", "success_reason_json"),
    ),
    NodeStateStatus.COMPLETED: NodeStateFieldConstraints(
        required=("output_hash", "completed_at", "duration_ms"),
        forbidden=("error_json",),
    ),
    NodeStateStatus.FAILED: NodeStateFieldConstraints(
        required=("completed_at", "duration_ms", "error_json"),
        forbidden=("success_reason_json",),
    ),
}


def validate_node_state_persisted_fields(
    state_id: str,
    status: NodeStateStatus,
    *,
    output_hash: object | None,
    completed_at: object | None,
    duration_ms: object | None,
    context_after_json: object | None,
    error_json: object | None,
    success_reason_json: object | None,
) -> None:
    """Validate status-dependent persisted node-state fields."""
    constraints = NODE_STATE_LIFECYCLE_FIELD_CONSTRAINTS[status]
    field_values = {
        "output_hash": output_hash,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "context_after_json": context_after_json,
        "error_json": error_json,
        "success_reason_json": success_reason_json,
    }
    status_name = status.name

    for field_name in constraints.required:
        if field_values[field_name] is None:
            raise ValueError(f"{status_name} state {state_id} has NULL {field_name} - audit integrity violation")
    for field_name in constraints.forbidden:
        if field_values[field_name] is not None:
            raise ValueError(f"{status_name} state {state_id} has non-NULL {field_name} - audit integrity violation")


def validate_node_state_completion_fields(
    status: NodeStateStatus,
    *,
    output_data_present: bool,
    duration_ms: float | None,
    error_present: bool,
    success_reason_present: bool,
) -> None:
    """Validate node-state completion inputs before persisting a lifecycle transition."""
    if status == NodeStateStatus.OPEN:
        raise ValueError("Cannot complete a node state with status OPEN")

    constraints = NODE_STATE_LIFECYCLE_FIELD_CONSTRAINTS[status]
    completion_values = {
        "output_hash": output_data_present,
        "completed_at": True,
        "duration_ms": duration_ms is not None,
        "context_after_json": False,
        "error_json": error_present,
        "success_reason_json": success_reason_present,
    }

    for field_name in constraints.required:
        if completion_values[field_name]:
            continue
        if field_name == "duration_ms":
            raise ValueError("duration_ms is required when completing a node state")
        if field_name == "output_hash":
            raise ValueError("COMPLETED node state requires output_data (output_hash would be NULL)")
        if field_name == "error_json":
            raise ValueError("FAILED node state requires error details")

    for field_name in constraints.forbidden:
        if not completion_values[field_name]:
            continue
        if field_name == "output_hash":
            raise ValueError(f"{status.name} node state must not have output_data")
        if field_name == "error_json":
            message = (
                "COMPLETED node state must not have error (contradicts success)"
                if status == NodeStateStatus.COMPLETED
                else f"{status.name} node state must not have error"
            )
            raise ValueError(message)
        if field_name == "success_reason_json":
            message = (
                "FAILED node state must not have success_reason (contradicts failure)"
                if status == NodeStateStatus.FAILED
                else f"{status.name} node state must not have success_reason"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
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
    # Cross-DB hash anchor for LLM transforms downstream of an interpretation
    # event (Phase 5b Task 9 / Option A). Populated at execution time by the
    # LLM plugin when it reads ``options.resolved_prompt_template_hash`` from
    # the node config (written there by ``resolve_interpretation_event`` at
    # compose time). ``None`` for non-LLM calls and for LLM transforms that
    # never went through an interpretation surface. Must equal the matching
    # ``interpretation_events.resolved_prompt_template_hash`` in the session
    # audit DB when non-None; inequality = Tier-1 audit anomaly.
    resolved_prompt_template_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate enum fields and structural invariants — Tier 1 crash on invalid types."""
        require_int(self.call_index, "call_index", min_value=0)
        _validate_enum(self.call_type, CallType, "call_type")
        _validate_enum(self.status, CallStatus, "status")
        validate_resolved_prompt_template_hash(self.call_type, self.resolved_prompt_template_hash)
        # XOR: exactly one of state_id or operation_id must be set
        has_state = self.state_id is not None
        has_operation = self.operation_id is not None
        if has_state == has_operation:
            raise ValueError(
                f"Call requires exactly one of state_id or operation_id. Got state_id={self.state_id!r}, operation_id={self.operation_id!r}"
            )


@dataclass(frozen=True, slots=True)
class SinkEffectStream:
    """Durable total-order authority for one replacing sink target."""

    stream_id: str
    run_id: str
    sink_node_id: str
    role: SinkEffectRole
    requested_target_hash: str
    resolved_target: str | None
    next_sequence: int
    tail_effect_id: str | None
    head_effect_id: str | None
    head_descriptor_hash: str | None

    def __post_init__(self) -> None:
        _validate_hash(self.stream_id, "stream_id")
        _validate_enum(self.role, SinkEffectRole, "role")
        _validate_hash(self.requested_target_hash, "requested_target_hash")
        require_int(self.next_sequence, "next_sequence", min_value=0)
        _validate_hash(self.tail_effect_id, "tail_effect_id", optional=True)
        _validate_hash(self.head_effect_id, "head_effect_id", optional=True)
        _validate_hash(self.head_descriptor_hash, "head_descriptor_hash", optional=True)
        if self.head_effect_id is None and self.head_descriptor_hash is not None:
            raise ValueError("head_descriptor_hash requires head_effect_id")


@dataclass(frozen=True, slots=True)
class SinkEffect:
    """Immutable view of one recoverable external publication effect."""

    effect_id: str
    run_id: str
    sink_node_id: str
    role: SinkEffectRole
    state: SinkEffectState
    protocol_version: str
    input_kind: SinkEffectInputKind
    required_member_ordinal: int | None
    required_snapshot_slot: int | None
    config_hash: str
    membership_or_manifest_hash: str
    group_payload_hash: str
    artifact_id: str
    artifact_idempotency_key: str
    target_json: str
    inspection_mode: SinkEffectInspectionMode | None
    inspection_attempt_id: str | None
    plan_json: str | None
    plan_hash: str | None
    descriptor_mode: SinkEffectDescriptorMode | None
    expected_descriptor_hash: str | None
    precondition_hash: str | None
    prepared_at: datetime | None
    lease_owner: str | None
    generation: int
    lease_expires_at: datetime | None
    lease_heartbeat_at: datetime | None
    reconcile_kind: SinkEffectReconcileKind | None
    reconcile_evidence_hash: str | None
    result_descriptor_hash: str | None
    publication_performed: bool | None
    publication_evidence_kind: str | None
    primary_effect_id: str | None
    stream_id: str | None
    stream_sequence: int | None
    predecessor_effect_id: str | None
    created_at: datetime
    updated_at: datetime
    finalized_at: datetime | None

    def __post_init__(self) -> None:
        for field_name in (
            "effect_id",
            "config_hash",
            "membership_or_manifest_hash",
            "group_payload_hash",
            "artifact_id",
        ):
            _validate_hash(getattr(self, field_name), field_name)
        for field_name in (
            "plan_hash",
            "expected_descriptor_hash",
            "precondition_hash",
            "reconcile_evidence_hash",
            "result_descriptor_hash",
            "primary_effect_id",
            "stream_id",
            "predecessor_effect_id",
        ):
            _validate_hash(getattr(self, field_name), field_name, optional=True)
        _validate_nonempty_string(self.protocol_version, "protocol_version")
        _validate_nonempty_string(self.artifact_idempotency_key, "artifact_idempotency_key")
        _validate_nonempty_string(self.target_json, "target_json")
        _validate_enum(self.role, SinkEffectRole, "role")
        _validate_enum(self.state, SinkEffectState, "state")
        _validate_enum(self.input_kind, SinkEffectInputKind, "input_kind")
        _validate_enum(self.inspection_mode, SinkEffectInspectionMode, "inspection_mode", optional=True)
        _validate_enum(self.descriptor_mode, SinkEffectDescriptorMode, "descriptor_mode", optional=True)
        _validate_enum(self.reconcile_kind, SinkEffectReconcileKind, "reconcile_kind", optional=True)
        require_int(self.required_member_ordinal, "required_member_ordinal", optional=True, min_value=0)
        require_int(self.required_snapshot_slot, "required_snapshot_slot", optional=True, min_value=0)
        require_int(self.generation, "generation", min_value=0)
        require_int(self.stream_sequence, "stream_sequence", optional=True, min_value=0)
        if self.publication_performed is not None and type(self.publication_performed) is not bool:
            raise TypeError("publication_performed must be bool or None")

        if self.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS:
            if self.required_member_ordinal != 0 or self.required_snapshot_slot is not None:
                raise ValueError("pipeline member effects require member ordinal zero and no snapshot slot")
        elif self.required_member_ordinal is not None or self.required_snapshot_slot != 0:
            raise ValueError("audit export effects require snapshot slot zero and no member ordinal")

        prepared_fields = (
            self.plan_json,
            self.plan_hash,
            self.inspection_mode,
            self.inspection_attempt_id,
            self.descriptor_mode,
            self.precondition_hash,
            self.prepared_at,
        )
        lease_fields = (self.lease_owner, self.lease_expires_at, self.lease_heartbeat_at)
        result_fields = (
            self.reconcile_kind,
            self.reconcile_evidence_hash,
            self.result_descriptor_hash,
            self.publication_performed,
            self.publication_evidence_kind,
        )
        if self.state is SinkEffectState.RESERVED:
            if any(value is not None for value in (*prepared_fields, *result_fields, self.finalized_at)):
                raise ValueError("reserved effect contains prepared, result, or finalized fields")
            if self.generation == 0:
                if any(value is not None for value in lease_fields):
                    raise ValueError("unclaimed reserved effect contains lease fields")
            elif any(value is None for value in lease_fields):
                raise ValueError("claimed reserved effect requires a complete preparation-claim lease")
        elif self.state is SinkEffectState.PREPARED:
            if any(value is None for value in prepared_fields) or any(
                value is not None for value in (*lease_fields, *result_fields, self.finalized_at)
            ):
                raise ValueError("prepared effect lifecycle fields are incomplete")
        elif self.state is SinkEffectState.IN_FLIGHT:
            if (
                any(value is None for value in (*prepared_fields, *lease_fields))
                or self.generation < 1
                or any(value is not None for value in (*result_fields, self.finalized_at))
            ):
                raise ValueError("in-flight effect lifecycle fields are incomplete")
        elif (
            any(value is None for value in prepared_fields)
            or self.result_descriptor_hash is None
            or self.publication_performed is None
            or self.publication_evidence_kind is None
            or self.finalized_at is None
            or any(value is not None for value in lease_fields)
        ):
            raise ValueError("finalized effect lifecycle fields are incomplete")
        elif self.publication_performed:
            if self.publication_evidence_kind == "returned":
                if self.reconcile_kind is not None or self.reconcile_evidence_hash is not None:
                    raise ValueError("returned effect cannot carry reconciliation evidence")
            elif self.publication_evidence_kind == "reconciled":
                if self.reconcile_kind is not SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR or self.reconcile_evidence_hash is None:
                    raise ValueError("reconciled effect requires exact reconciliation evidence")
            else:
                raise ValueError("published effect requires returned or reconciled evidence")
        elif (
            self.publication_evidence_kind not in {"inherited", "virtual"}
            or self.reconcile_kind is not None
            or self.reconcile_evidence_hash is not None
        ):
            raise ValueError("no-publication effect requires inherited or virtual evidence")

        if self.lease_expires_at is not None and self.lease_heartbeat_at is not None and self.lease_expires_at < self.lease_heartbeat_at:
            raise ValueError("lease_expires_at cannot precede lease_heartbeat_at")
        if self.descriptor_mode is SinkEffectDescriptorMode.PRECOMPUTED and self.expected_descriptor_hash is None:
            raise ValueError("precomputed descriptor mode requires expected_descriptor_hash")
        if self.descriptor_mode is SinkEffectDescriptorMode.RESULT_DERIVED and self.expected_descriptor_hash is not None:
            raise ValueError("result-derived descriptor mode forbids expected_descriptor_hash")
        if self.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION and self.expected_descriptor_hash is None:
            raise ValueError("no-publication descriptor mode requires expected_descriptor_hash")
        if self.stream_id is None:
            if self.stream_sequence is not None or self.predecessor_effect_id is not None:
                raise ValueError("unbound effects cannot carry stream ordering fields")
        elif self.stream_sequence == 0:
            if self.predecessor_effect_id is not None:
                raise ValueError("stream sequence zero cannot have a predecessor")
        elif self.stream_sequence is None or self.stream_sequence < 1 or self.predecessor_effect_id is None:
            raise ValueError("later stream effects require a sequence and predecessor")


@dataclass(frozen=True, slots=True)
class SinkEffectMemberRecord:
    """Canonical member identity and durable per-member publication state."""

    effect_id: str
    input_kind: SinkEffectInputKind
    ordinal: int
    run_id: str
    sink_node_id: str
    role: SinkEffectRole
    token_id: str
    row_id: str
    ingest_sequence: int
    lineage_json: str
    lineage_hash: str
    payload_hash: str
    prepared_disposition: Literal["accepted", "diverted"] | None
    reason_hash: str | None
    member_effect_id: str | None
    member_state: SinkEffectState | None
    descriptor_hash: str | None
    evidence_hash: str | None

    def __post_init__(self) -> None:
        _validate_hash(self.effect_id, "effect_id")
        _validate_enum(self.input_kind, SinkEffectInputKind, "input_kind")
        if self.input_kind is not SinkEffectInputKind.PIPELINE_MEMBERS:
            raise ValueError("sink effect members require pipeline_members input kind")
        _validate_enum(self.role, SinkEffectRole, "role")
        _validate_enum(self.member_state, SinkEffectState, "member_state", optional=True)
        require_int(self.ordinal, "ordinal", min_value=0)
        require_int(self.ingest_sequence, "ingest_sequence", min_value=0)
        _validate_nonempty_string(self.lineage_json, "lineage_json")
        for field_name in ("lineage_hash", "payload_hash"):
            _validate_hash(getattr(self, field_name), field_name)
        for field_name in ("reason_hash", "member_effect_id", "descriptor_hash", "evidence_hash"):
            _validate_hash(getattr(self, field_name), field_name, optional=True)
        if self.prepared_disposition not in (None, "accepted", "diverted"):
            raise ValueError("prepared_disposition must be accepted, diverted, or None")


@dataclass(frozen=True, slots=True)
class AuditExportSnapshotChunk:
    """One immutable, cumulatively sealed audit-export data chunk."""

    snapshot_id: str
    ordinal: int
    content_ref: str
    content_hash: str
    size_bytes: int
    record_count: int
    predecessor_seal_hash: str | None
    cumulative_records: int
    cumulative_bytes: int
    chunk_seal_hash: str

    def __post_init__(self) -> None:
        _validate_hash(self.snapshot_id, "snapshot_id")
        _validate_hash(self.content_hash, "content_hash")
        _validate_hash(self.predecessor_seal_hash, "predecessor_seal_hash", optional=True)
        _validate_hash(self.chunk_seal_hash, "chunk_seal_hash")
        if self.content_ref != f"sha256:{self.content_hash}":
            raise ValueError("content_ref must exactly match content_hash")
        require_int(self.ordinal, "ordinal", min_value=0)
        require_int(self.size_bytes, "size_bytes", min_value=1)
        require_int(self.record_count, "record_count", min_value=1)
        require_int(self.cumulative_records, "cumulative_records", min_value=1)
        require_int(self.cumulative_bytes, "cumulative_bytes", min_value=1)
        if self.ordinal == 0:
            if self.predecessor_seal_hash is not None:
                raise ValueError("chunk zero must not have a predecessor seal")
            if self.cumulative_records != self.record_count or self.cumulative_bytes != self.size_bytes:
                raise ValueError("chunk zero cumulative totals must equal its own totals")
        elif self.predecessor_seal_hash is None:
            raise ValueError("nonzero chunks require a predecessor seal")


@dataclass(frozen=True, slots=True)
class AuditExportSnapshot:
    """Strict immutable registry record for a durable audit-export winner.

    This record is not a content capability. Public loading must bind and
    verify it with its stored content resolver before exposing chunk bytes.
    """

    snapshot_id: str
    source_run_id: str
    source_status: RunStatus
    source_completed_at: datetime
    exported_at: datetime
    registry_key_hash: str
    exporter_version: str
    serialization_version: str
    export_format: AuditExportFormat
    signing_mode: AuditExportSigningMode
    signer_key_id: str
    derivation_version: str
    public_export_config_hash: str
    chunking_algorithm_version: str
    per_chunk_record_limit: int
    per_chunk_byte_limit: int
    record_count: int
    total_bytes: int
    chunk_count: int
    terminal_chunk_ordinal: int
    content_store_id: str
    manifest_hash: str
    last_chunk_seal_hash: str
    snapshot_hash: str
    snapshot_seal_hash: str
    signature_hex: str | None
    record_chain_algorithm: str
    final_hash: str
    signed_manifest_schema: str
    signed_manifest_hash: str
    signed_manifest_ref: str
    signed_manifest_size_bytes: int

    def __post_init__(self) -> None:
        _validate_enum(self.source_status, RunStatus, "source_status")
        if self.source_status not in {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY}:
            raise ValueError("audit export snapshot requires an immutable export-terminal run")
        _validate_enum(self.export_format, AuditExportFormat, "export_format")
        _validate_enum(self.signing_mode, AuditExportSigningMode, "signing_mode")
        for field_name in (
            "snapshot_id",
            "registry_key_hash",
            "public_export_config_hash",
            "manifest_hash",
            "last_chunk_seal_hash",
            "snapshot_hash",
            "snapshot_seal_hash",
            "final_hash",
            "signed_manifest_hash",
        ):
            _validate_hash(getattr(self, field_name), field_name)
        _validate_hash(self.signature_hex, "signature_hex", optional=True)
        for field_name in (
            "exporter_version",
            "serialization_version",
            "signer_key_id",
            "derivation_version",
            "chunking_algorithm_version",
            "content_store_id",
            "record_chain_algorithm",
        ):
            _validate_nonempty_string(getattr(self, field_name), field_name)
        for field_name in (
            "per_chunk_record_limit",
            "per_chunk_byte_limit",
            "record_count",
            "total_bytes",
            "chunk_count",
            "signed_manifest_size_bytes",
        ):
            require_int(getattr(self, field_name), field_name, min_value=1)
        require_int(self.terminal_chunk_ordinal, "terminal_chunk_ordinal", min_value=0)
        if self.terminal_chunk_ordinal != self.chunk_count - 1:
            raise ValueError("terminal_chunk_ordinal must equal chunk_count - 1")
        if self.signed_manifest_schema != "elspeth.audit-export-manifest.v2":
            raise ValueError("signed_manifest_schema is not the supported v2 schema")
        if self.derivation_version != "audit-export-derivation-v1":
            raise ValueError("derivation_version is not supported")
        if self.signed_manifest_ref != f"sha256:{self.signed_manifest_hash}":
            raise ValueError("signed_manifest_ref must exactly match signed_manifest_hash")
        if self.signed_manifest_size_bytes > 64 * 1024:
            raise ValueError("signed manifest exceeds the code-owned size limit")
        if self.signing_mode is AuditExportSigningMode.UNSIGNED:
            if self.signer_key_id != "UNSIGNED" or self.signature_hex is not None:
                raise ValueError("unsigned snapshots require UNSIGNED signer and no signature")
            if self.record_chain_algorithm != "sha256_concat_record_sha256_v1":
                raise ValueError("unsigned snapshot has the wrong record-chain algorithm")
        else:
            if self.signer_key_id == "UNSIGNED" or not self.signer_key_id.strip() or self.signature_hex is None:
                raise ValueError("HMAC snapshots require an identified signer and signature")
            if self.record_chain_algorithm != "sha256_concat_hmac_sha256_signatures_v1":
                raise ValueError("HMAC snapshot has the wrong record-chain algorithm")


@dataclass(frozen=True, slots=True)
class SinkEffectExportSnapshotAssociation:
    """The single immutable snapshot input selected by an export effect."""

    effect_id: str
    input_kind: SinkEffectInputKind
    slot: int
    snapshot_id: str

    def __post_init__(self) -> None:
        _validate_hash(self.effect_id, "effect_id")
        _validate_hash(self.snapshot_id, "snapshot_id")
        _validate_enum(self.input_kind, SinkEffectInputKind, "input_kind")
        require_int(self.slot, "slot", min_value=0)
        if self.input_kind is not SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT or self.slot != 0:
            raise ValueError("export snapshot associations require audit_export_snapshot input kind and slot zero")


@dataclass(frozen=True, slots=True)
class SinkEffectAttempt:
    """Durable intent/result row for one effect-side external call."""

    attempt_id: str
    effect_id: str
    member_ordinal: int | None
    generation: int
    action: SinkEffectAttemptAction
    call_kind: str
    request_hash: str
    state: SinkEffectAttemptState
    evidence_json: str | None
    evidence_hash: str | None
    started_at: datetime
    completed_at: datetime | None
    latency_ms: float | None

    def __post_init__(self) -> None:
        _validate_hash(self.attempt_id, "attempt_id")
        _validate_hash(self.effect_id, "effect_id")
        _validate_hash(self.request_hash, "request_hash")
        _validate_hash(self.evidence_hash, "evidence_hash", optional=True)
        _validate_enum(self.action, SinkEffectAttemptAction, "action")
        _validate_enum(self.state, SinkEffectAttemptState, "state")
        _validate_nonempty_string(self.call_kind, "call_kind")
        require_int(self.member_ordinal, "member_ordinal", optional=True, min_value=0)
        require_int(self.generation, "generation", min_value=0)
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.state is SinkEffectAttemptState.INTENT:
            if (
                self.completed_at is not None
                or self.latency_ms is not None
                or self.evidence_json is not None
                or self.evidence_hash is not None
            ):
                raise ValueError("intent attempts cannot contain completion evidence")
        elif self.completed_at is None or self.latency_ms is None:
            raise ValueError("closed attempts require completed_at and latency_ms")
        if (self.evidence_json is None) != (self.evidence_hash is None):
            raise ValueError("attempt evidence JSON and hash must be both present or both absent")


@dataclass(frozen=True, slots=True)
class Artifact:
    """An artifact produced by a sink."""

    artifact_id: str
    run_id: str
    sink_node_id: str
    artifact_type: str  # Not enum - user-defined (csv, json, webhook, etc.)
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime
    produced_by_state_id: str | None = None
    sink_effect_id: str | None = None
    idempotency_key: str | None = None  # For retry deduplication
    publication_performed: bool = True
    publication_evidence_kind: ArtifactPublicationEvidenceKind = "returned"

    def __post_init__(self) -> None:
        """Validate exclusive producer linkage and publication evidence."""
        if (self.produced_by_state_id is None) == (self.sink_effect_id is None):
            raise ValueError("Artifact requires exactly one producer link")
        require_int(self.size_bytes, "size_bytes", min_value=0)
        if type(self.publication_performed) is not bool:
            raise TypeError("publication_performed must be bool")

        if self.produced_by_state_id is not None:
            if not self.publication_performed or self.publication_evidence_kind != "legacy_returned":
                raise ValueError("legacy artifact publication requires performed legacy_returned evidence")
            return

        evidence_performed = {
            "returned": True,
            "reconciled": True,
            "inherited": False,
            "virtual": False,
        }
        expected = evidence_performed.get(self.publication_evidence_kind)
        if expected is None or self.publication_performed is not expected:
            raise ValueError("effect artifact publication evidence is invalid or contradicts publication_performed")

    @property
    def producer_kind(self) -> ArtifactProducerKind:
        """Return the explicit producer discriminator for serialization."""
        return "node_state" if self.produced_by_state_id is not None else "sink_effect"


@dataclass(frozen=True, slots=True)
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
        require_int(self.ordinal, "ordinal", min_value=0)
        _validate_enum(self.mode, RoutingMode, "mode")


@dataclass(frozen=True, slots=True)
class Batch:
    """An aggregation batch collecting tokens.

    Strict contract - status and trigger_type must be enums.
    ``retry_of_batch_id`` links a retry batch back to the failed batch whose
    member set it is replaying.
    """

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: BatchStatus  # Strict: enum only
    created_at: datetime
    aggregation_state_id: str | None = None
    retry_of_batch_id: str | None = None
    trigger_type: TriggerType | None = None  # Strict: enum only
    trigger_reason: str | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate enum fields - Tier 1 crash on invalid types."""
        require_int(self.attempt, "attempt", min_value=0)
        _validate_enum(self.status, BatchStatus, "status")
        _validate_enum(self.trigger_type, TriggerType, "trigger_type", optional=True)


@dataclass(frozen=True, slots=True)
class BatchMember:
    """A token belonging to a batch."""

    batch_id: str
    run_id: str
    token_id: str
    ordinal: int

    def __post_init__(self) -> None:
        """Validate int fields - Tier 1 crash on invalid types."""
        require_int(self.ordinal, "ordinal", min_value=0)


@dataclass(frozen=True, slots=True)
class BatchOutput:
    """An output produced by a batch."""

    batch_id: str
    output_type: str  # token, artifact
    output_id: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Checkpoint for crash recovery.

    Captures run progress at sink-durability boundaries. Post-F1 durability
    unification, buffered tokens live in token_work_items (journal BLOCKED
    rows); the checkpoint carries only scalar barrier metadata
    (``barrier_scalars_json``). The real resume work-set is token_outcomes
    completeness + journal BLOCKED rows + sequence_number; compatibility is
    enforced by the full-topology hash (which embeds every node's config
    hash), so the checkpoint carries no per-node anchor.

    Format Versions:
        Version 1: Pre-deterministic node IDs (legacy, incompatible)
        Version 2: Deterministic node IDs (2026-01-24+)
        Version 3: Phase 2 traversal refactor checkpoint break
        Version 4: Pending coalesce state persisted in checkpoints
        Version 5: F1 durability unification — buffered tokens live in
            token_work_items; checkpoint carries only scalar barrier
            metadata (current)
    """

    # Current checkpoint format version (ClassVar excludes from dataclass fields)
    CURRENT_FORMAT_VERSION: ClassVar[int] = 5

    checkpoint_id: str
    run_id: str
    sequence_number: int
    created_at: datetime  # Required - schema enforces NOT NULL (Tier 1 audit data)
    # Topology validation field - REQUIRED for checkpoint compatibility checking
    # Schema enforces NOT NULL - audit-critical for resume validation
    upstream_topology_hash: str  # Hash of ALL nodes + edges in DAG (full topology)
    # Optional fields (with defaults) MUST come after required fields in dataclass
    # Serialized BarrierScalars (elspeth.contracts.barrier_scalars), or None
    # when no barriers were in flight at checkpoint time.
    barrier_scalars_json: str | None = None
    # Format version for compatibility checking
    format_version: int | None = None

    def __post_init__(self) -> None:
        """Validate required fields - Tier 1 crash on invalid data.

        Per Data Manifesto: Audit data is OUR data. If we receive None
        for required hash fields, that's a bug in our code - crash immediately.
        """
        require_int(self.sequence_number, "sequence_number", min_value=0)
        require_int(self.format_version, "format_version", optional=True, min_value=0)
        if not self.upstream_topology_hash:
            raise ValueError("upstream_topology_hash is required and cannot be empty")

    @property
    def full_topology_hash(self) -> str:
        """Full-DAG topology hash used for checkpoint compatibility."""
        return self.upstream_topology_hash


@dataclass(frozen=True, slots=True)
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
    source_data: Mapping[str, object] | None  # None if purged
    payload_available: bool
    # Mandatory source-scoped identity (G22 fabrication ban): the rows table
    # declares both NOT NULL and they anchor the (run_id, source_node_id,
    # source_row_index) / (run_id, ingest_sequence) unique constraints, so they
    # are never None on a DB read. Typed `int` (not `int | None`) so the type
    # matches the require_int guard below and the NOT NULL column.
    source_row_index: int
    ingest_sequence: int

    def __post_init__(self) -> None:
        require_int(self.row_index, "row_index", min_value=0)
        require_int(self.source_row_index, "source_row_index", min_value=0)
        require_int(self.ingest_sequence, "ingest_sequence", min_value=0)
        freeze_fields(self, "source_data")


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


@dataclass(frozen=True, slots=True)
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
    schema_mode: str  # "fixed", "flexible", "observed", "parse"
    destination: str
    created_at: datetime
    row_id: str | None = None
    row_data_json: str | None = None
    violation_type: str | None = None  # "type_mismatch", "missing_field", "extra_field"
    original_field_name: str | None = None  # Display name e.g. "'Amount USD'"
    normalized_field_name: str | None = None  # Code name e.g. "amount_usd"
    expected_type: str | None = None  # e.g. "int", "str"
    actual_type: str | None = None  # Type of actual value


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
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


DISCARD_SINK_NAME = "__discard__"


@dataclass(frozen=True, slots=True)
class TokenOutcome:
    """Recorded terminal state for a token (ADR-019 two-axis model).

    ``outcome`` is the lifecycle answer when ``completed=True`` and ``None``
    when ``completed=False``. ``path`` is the producer-declared provenance axis
    and is always populated.
    """

    outcome_id: str
    run_id: str
    token_id: str
    outcome: TerminalOutcome | None
    path: TerminalPath
    completed: bool
    recorded_at: datetime

    # Outcome-specific fields (nullable based on (outcome, path) pair)
    sink_name: str | None = None
    batch_id: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    error_hash: str | None = None
    context_json: str | None = None
    expected_branches_json: str | None = None  # Branch contract for FORK_PARENT/EXPAND_PARENT

    def __post_init__(self) -> None:
        """Validate two-axis invariants — Tier 1 crash on invalid combinations."""
        if not isinstance(self.completed, bool):
            raise TypeError(f"completed must be bool, got {type(self.completed).__name__}: {self.completed!r}")
        if self.completed and self.outcome is None:
            raise ValueError(
                f"TokenOutcome {self.outcome_id}: completed=True requires non-NULL outcome "
                "(ADR-019 invariant: completed XOR (outcome IS NULL))"
            )
        if not self.completed and self.outcome is not None:
            raise ValueError(f"TokenOutcome {self.outcome_id}: completed=False requires outcome=None (got outcome={self.outcome!r})")

        if self.outcome is not None:
            _validate_enum(self.outcome, TerminalOutcome, "outcome")
        _validate_enum(self.path, TerminalPath, "path")

        if self.completed:
            assert self.outcome is not None
            if (self.outcome, self.path) not in _LEGAL_TERMINAL_PAIRS:
                raise ValueError(
                    f"TokenOutcome {self.outcome_id}: ({self.outcome!r}, {self.path!r}) "
                    "is not in _LEGAL_TERMINAL_PAIRS — see ADR-019 mapping table."
                )
        elif self.path != TerminalPath.BUFFERED:
            raise ValueError(f"TokenOutcome {self.outcome_id}: completed=False requires path=BUFFERED (got path={self.path!r})")


@dataclass(frozen=True, slots=True)
class TerminalPairFieldConstraints:
    """Column-level constraints for one ADR-019 (outcome, path) pair."""

    required: tuple[str, ...] = ()
    exact: Mapping[str, object] = field(default_factory=dict)
    forbidden: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # ``required`` and ``forbidden`` are tuple[str, ...] — strings are
        # immutable, the tuples themselves are immutable; they need no
        # guard. ``exact`` is Mapping[str, object] and the constructor
        # default is a fresh ``dict()`` — the dict is mutable through
        # the attribute reference and would lie about ``frozen=True``
        # without deep_freeze. Producers populate ``exact`` with
        # scalar/string values for ADR-019 (outcome, path) discriminator
        # matches, but the field type does not statically guarantee it,
        # so deep-freeze is the correct guard rather than a shallow wrap.
        freeze_fields(self, "exact")


_DISCRIMINATOR_FIELDS = (
    "sink_name",
    "batch_id",
    "fork_group_id",
    "join_group_id",
    "expand_group_id",
    "error_hash",
)


def _forbid_except(*allowed: str) -> tuple[str, ...]:
    return tuple(field_name for field_name in _DISCRIMINATOR_FIELDS if field_name not in allowed)


_TERMINAL_PAIR_FIELD_CONSTRAINTS: dict[
    tuple[TerminalOutcome | None, TerminalPath],
    TerminalPairFieldConstraints,
] = {
    (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW): TerminalPairFieldConstraints(
        required=("sink_name",),
        forbidden=_forbid_except("sink_name"),
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED): TerminalPairFieldConstraints(
        required=("sink_name",),
        forbidden=_forbid_except("sink_name"),
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.GATE_DISCARDED): TerminalPairFieldConstraints(
        forbidden=_DISCRIMINATOR_FIELDS,
    ),
    (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED): TerminalPairFieldConstraints(
        forbidden=_DISCRIMINATOR_FIELDS,
    ),
    (TerminalOutcome.SUCCESS, TerminalPath.COALESCED): TerminalPairFieldConstraints(
        required=("join_group_id",),
        forbidden=_forbid_except("sink_name", "join_group_id"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.UNROUTED): TerminalPairFieldConstraints(
        required=("error_hash",),
        forbidden=_forbid_except("error_hash"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE): TerminalPairFieldConstraints(
        required=("error_hash",),
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED): TerminalPairFieldConstraints(
        required=("sink_name", "error_hash"),
        exact={"sink_name": DISCARD_SINK_NAME},
        forbidden=_forbid_except("sink_name", "error_hash"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT): TerminalPairFieldConstraints(
        required=("fork_group_id",),
        forbidden=_forbid_except("fork_group_id"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT): TerminalPairFieldConstraints(
        required=("expand_group_id",),
        forbidden=_forbid_except("expand_group_id"),
    ),
    (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED): TerminalPairFieldConstraints(
        required=("batch_id",),
        forbidden=_forbid_except("batch_id"),
    ),
    (None, TerminalPath.BUFFERED): TerminalPairFieldConstraints(
        required=("batch_id",),
        forbidden=_forbid_except("batch_id"),
    ),
}


def validate_token_outcome_persisted_fields(
    outcome_id: str,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    completed: bool,
    *,
    sink_name: str | None,
    batch_id: str | None,
    fork_group_id: str | None,
    join_group_id: str | None,
    expand_group_id: str | None,
    error_hash: str | None,
) -> None:
    """Validate ADR-019 persisted discriminator fields for a token outcome."""
    if completed != (outcome is not None):
        raise ValueError(
            f"TokenOutcome {outcome_id}: completed={completed} but outcome={outcome!r} — "
            "completed must be true iff outcome is non-NULL (ADR-019 invariant)"
        )

    if completed:
        assert outcome is not None
        if (outcome, path) not in _LEGAL_TERMINAL_PAIRS:
            raise ValueError(f"TokenOutcome {outcome_id}: ({outcome!r}, {path!r}) not in _LEGAL_TERMINAL_PAIRS — audit integrity violation")
    elif path != TerminalPath.BUFFERED:
        raise ValueError(f"TokenOutcome {outcome_id}: completed=False requires path=BUFFERED, got {path!r} — audit integrity violation")

    pair: tuple[TerminalOutcome | None, TerminalPath] = (outcome, path)
    constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
    field_values = {
        "sink_name": sink_name,
        "batch_id": batch_id,
        "fork_group_id": fork_group_id,
        "join_group_id": join_group_id,
        "expand_group_id": expand_group_id,
        "error_hash": error_hash,
    }
    for field_name in constraints.required:
        if field_values[field_name] is None:
            raise ValueError(
                f"TokenOutcome {outcome_id}: ({outcome!r}, {path!r}) requires {field_name} but DB has NULL — audit integrity violation"
            )
    for field_name, expected in constraints.exact.items():
        if field_values[field_name] != expected:
            raise ValueError(
                f"TokenOutcome {outcome_id}: ({outcome!r}, {path!r}) requires "
                f"{field_name}={expected!r}, got {field_values[field_name]!r} — "
                "audit integrity violation"
            )
    for field_name in constraints.forbidden:
        if field_values[field_name] is not None:
            raise ValueError(
                f"TokenOutcome {outcome_id}: ({outcome!r}, {path!r}) forbids {field_name}, "
                f"got {field_values[field_name]!r} — audit integrity violation"
            )


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
    operation_type: OperationType
    started_at: datetime
    status: Literal["open", "completed", "failed", "pending"]
    sink_effect_id: str | None = None
    completed_at: datetime | None = None
    input_data_ref: str | None = None
    input_data_hash: str | None = None
    output_data_ref: str | None = None
    output_data_hash: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None

    _ALLOWED_OPERATION_TYPES: ClassVar[frozenset[str]] = frozenset(OPERATION_TYPE_VALUES)
    _ALLOWED_STATUSES: ClassVar[frozenset[str]] = frozenset({"open", "completed", "failed", "pending"})

    def __post_init__(self) -> None:
        """Validate constrained literal fields and lifecycle invariants for Tier 1 audit integrity.

        Status-dependent invariants:
        - open: completed_at, duration_ms, error_message must all be None
        - completed: completed_at and duration_ms must be present, error_message must be None
        - failed: completed_at and duration_ms must be present, error_message must be present
        - pending: completed_at and duration_ms must be present
        """
        if self.operation_type not in self._ALLOWED_OPERATION_TYPES:
            raise ValueError(f"operation_type must be one of {sorted(self._ALLOWED_OPERATION_TYPES)}, got {self.operation_type!r}")

        if self.status not in self._ALLOWED_STATUSES:
            raise ValueError(f"status must be one of {sorted(self._ALLOWED_STATUSES)}, got {self.status!r}")

        if self.sink_effect_id is not None and self.operation_type != "sink_write":
            raise ValueError("sink_effect_id is valid only for sink_write operations")

        # Lifecycle invariant validation — Tier 1 crash on impossible state combinations
        if self.status == "open":
            if self.completed_at is not None:
                raise ValueError(f"Operation {self.operation_id!r}: status='open' but completed_at is set")
            if self.duration_ms is not None:
                raise ValueError(f"Operation {self.operation_id!r}: status='open' but duration_ms is set")
            if self.error_message is not None:
                raise ValueError(f"Operation {self.operation_id!r}: status='open' but error_message is set")
        elif self.status in {"completed", "failed", "pending"}:
            if self.completed_at is None:
                raise ValueError(f"Operation {self.operation_id!r}: status={self.status!r} but completed_at is None")
            if self.duration_ms is None:
                raise ValueError(f"Operation {self.operation_id!r}: status={self.status!r} but duration_ms is None")
            if self.status == "failed" and self.error_message is None:
                raise ValueError(f"Operation {self.operation_id!r}: status='failed' but error_message is None")
            if self.status == "failed" and self.error_message == "":
                raise ValueError(f"Operation {self.operation_id!r}: status='failed' but error_message must not be empty")
            if self.status == "completed" and self.error_message is not None:
                raise ValueError(f"Operation {self.operation_id!r}: status='completed' but error_message is set")

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
            "sink_effect_id": self.sink_effect_id,
            "input_data_ref": self.input_data_ref,
            "input_data_hash": self.input_data_hash,
            "output_data_ref": self.output_data_ref,
            "output_data_hash": self.output_data_hash,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True, slots=True)
class SecretResolution:
    """Record of a secret loaded from Key Vault for audit trail.

    These records enable auditors to answer: "Which Key Vault and which
    secret was used for this pipeline run?" without exposing actual secret
    values (only HMAC fingerprints are stored).

    Secret resolutions are recorded at run start, before processing begins.
    They capture the provenance of secrets loaded from external vaults.

    Attributes:
        resolution_id: Unique identifier for this resolution event
        run_id: Run that used this secret
        timestamp: When the secret was loaded (epoch seconds, may be before run start)
        env_var_name: Environment variable the secret was injected into
        source: Source type ('keyvault', 'env', or 'user')
        vault_url: Key Vault URL (None if source != keyvault)
        secret_name: Secret name in the vault
        fingerprint: HMAC-SHA256 fingerprint of the secret value (not the value itself)
        resolution_latency_ms: Time to fetch from vault (None if not measured)
    """

    _ALLOWED_SOURCES: ClassVar[frozenset[str]] = frozenset({"keyvault", "env", "user"})

    resolution_id: str
    run_id: str
    timestamp: float  # Epoch seconds - may be before run start
    env_var_name: str
    source: str  # 'keyvault', 'env', or 'user'
    fingerprint: str  # HMAC fingerprint, NOT the secret value
    vault_url: str | None = None
    secret_name: str | None = None
    resolution_latency_ms: float | None = None

    def __post_init__(self) -> None:
        """Validate Tier 1 invariants for secret provenance records.

        Per Data Manifesto: The audit database is OUR data. If we read
        garbage from it, something catastrophic happened - crash immediately.

        Invariants:
        - resolution_id, run_id, env_var_name, source, fingerprint must be non-empty strings
        - source must be a known value ('keyvault')
        - fingerprint must be 64-char lowercase hex (HMAC-SHA256)
        - timestamp must be finite
        - resolution_latency_ms must be non-negative when present
        - keyvault source requires non-empty vault_url and secret_name
        """
        import math

        if not self.resolution_id:
            raise ValueError("SecretResolution: resolution_id is required and cannot be empty")
        if not self.run_id:
            raise ValueError("SecretResolution: run_id is required and cannot be empty")
        if not self.env_var_name:
            raise ValueError("SecretResolution: env_var_name is required and cannot be empty")
        if not self.source:
            raise ValueError("SecretResolution: source is required and cannot be empty")
        if self.source not in self._ALLOWED_SOURCES:
            raise ValueError(f"SecretResolution: source must be one of {sorted(self._ALLOWED_SOURCES)}, got {self.source!r}")
        if not self.fingerprint:
            raise ValueError("SecretResolution: fingerprint is required and cannot be empty")
        if len(self.fingerprint) != 64 or not all(c in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError(
                f"SecretResolution: fingerprint must be 64-char lowercase hex (HMAC-SHA256), "
                f"got {self.fingerprint!r} (length={len(self.fingerprint)})"
            )
        if not isinstance(self.timestamp, (int, float)) or math.isinf(self.timestamp) or math.isnan(self.timestamp):
            raise ValueError(f"SecretResolution: timestamp must be a finite number, got {self.timestamp!r}")
        if self.resolution_latency_ms is not None and self.resolution_latency_ms < 0:
            raise ValueError(f"SecretResolution: resolution_latency_ms must be non-negative, got {self.resolution_latency_ms!r}")
        if self.source == "keyvault":
            if not self.vault_url:
                raise ValueError("SecretResolution: vault_url is required when source='keyvault'")
            if not self.secret_name:
                raise ValueError("SecretResolution: secret_name is required when source='keyvault'")


@dataclass(frozen=True, slots=True)
class SecretResolutionInput:
    """Write-side DTO for secret resolution records.

    Used at the Tier 1 boundary when recording secret resolutions into the
    audit trail. Replaces the previous dict[str, Any] pattern with compile-time
    key validation. The resolution_id and run_id are assigned at record time,
    not at creation time.

    Follows the TokenUsage precedent (commit dffe74a6) for typed audit inputs.
    """

    _ALLOWED_SOURCES: ClassVar[frozenset[str]] = frozenset({"keyvault", "env", "user"})

    env_var_name: str
    source: str
    vault_url: str | None
    secret_name: str | None
    timestamp: float
    resolution_latency_ms: float
    fingerprint: str

    def __post_init__(self) -> None:
        """Validate write-side invariants before audit trail insertion.

        Lightweight checks for security-critical invariants. The full
        set of business rule validations lives on the read-side
        SecretResolution. These checks prevent:
        - Plaintext secrets being written as fingerprints (security)
        - Invalid source values persisting undetected (Tier 1 integrity)
        - Key Vault rows that cannot round-trip through the read-side contract
        - Non-finite timestamps persisting into Tier 1 audit data
        - Non-negative latency invariant (data quality)
        """
        import math

        if not self.env_var_name:
            raise ValueError("SecretResolutionInput: env_var_name is required and cannot be empty")
        if not self.source or self.source not in self._ALLOWED_SOURCES:
            raise ValueError(f"SecretResolutionInput: source must be one of {sorted(self._ALLOWED_SOURCES)}, got {self.source!r}")
        if len(self.fingerprint) != 64 or not all(c in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError(
                f"SecretResolutionInput: fingerprint must be 64-char lowercase hex (HMAC-SHA256), "
                f"got {self.fingerprint!r} (length={len(self.fingerprint)})"
            )
        if not isinstance(self.timestamp, (int, float)) or math.isinf(self.timestamp) or math.isnan(self.timestamp):
            raise ValueError(f"SecretResolutionInput: timestamp must be a finite number, got {self.timestamp!r}")
        if self.resolution_latency_ms < 0:
            raise ValueError(f"SecretResolutionInput: resolution_latency_ms must be non-negative, got {self.resolution_latency_ms!r}")
        if self.source == "keyvault":
            if not self.vault_url:
                raise ValueError("SecretResolutionInput: vault_url is required when source='keyvault'")
            if not self.secret_name:
                raise ValueError("SecretResolutionInput: secret_name is required when source='keyvault'")
