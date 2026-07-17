"""SessionService protocol and record dataclasses.

Record types are frozen dataclasses representing database rows.
CompositionStateData is the input DTO for saving new state versions.
"""

# ID Convention: Record dataclasses use UUID for type safety. The database
# stores IDs as String (TEXT). SessionServiceImpl converts between UUID
# and str at the query/record boundary. Callers work with UUID
# exclusively; the storage representation is an implementation detail.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, runtime_checkable
from uuid import UUID

from elspeth.contracts.auth import AuthProviderType
from elspeth.contracts.blobs_inline import ResolvedBlobContent
from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields, require_int

if TYPE_CHECKING:
    from elspeth.web.composer.pipeline_commit import PipelineDispatchAuditBinding
    from elspeth.web.composer.pipeline_planner import PipelinePlanResult
    from elspeth.web.composer.pipeline_proposal import PipelineProposal

ChatMessageRole = Literal["user", "assistant", "system", "tool", "audit"]
ComposerTrustMode = Literal["explicit_approve", "auto_commit"]
ComposerDensityDefault = Literal["high", "medium", "low"]
ProposalLifecycleStatus = Literal["pending", "committed", "rejected"]
PipelineProposalRejectionReason = Literal[
    "operator_rejected",
    "candidate_executor_mismatch",
    "validation_failed",
    "policy_changed",
    "base_conflict",
    "request_cancelled",
    "superseded",
]
PipelineProposalSurface = Literal["freeform", "guided_full", "guided_staged", "tutorial_profile"]
ProposalEventType = Literal[
    "proposal.created",
    "proposal.accepted",
    "proposal.rejected",
    "trust_mode.changed",
]
# ``audit`` is an internal-only role for breadcrumb rows that have no real
# OpenAI tool-response or assistant parent (LLM-call audit envelopes,
# pre-flight redaction failures, etc.). They MUST be filtered out of any
# user-facing chat response and any composer prompt-history rebuild —
# enforced at ``_is_composer_audit_tool_message`` /
# ``_composer_conversation_messages`` and the public messages route.
# Phase 2.2 (elspeth-0de989c56d): four-value terminal taxonomy.
# `completed_with_failures` and `empty` join the previous three so an
# operator scanning `/api/runs/{rid}` can distinguish "ran cleanly" from
# "ran but no row succeeded" without opening output files.  Mirrors the
# L0 RunStatus enum widening; row-count biconditional enforced in
# RunRecord.__post_init__.
SessionRunStatus = Literal["pending", "running", "completed", "completed_with_failures", "failed", "empty", "cancelled"]
TerminalSessionRunStatus = Literal["completed", "completed_with_failures", "failed", "empty", "cancelled"]
OperatorCompletionSessionRunStatus = Literal["completed", "completed_with_failures", "empty"]
SessionRunEventType = Literal["progress", "error", "completed", "cancelled", "failed"]

LANDSCAPE_RECONCILIATION_PENDING_SUFFIX = "[landscape-reconciliation:pending]"
LANDSCAPE_RECONCILIATION_COMPLETE_SUFFIX = "[landscape-reconciliation:complete]"
LANDSCAPE_RECONCILIATION_ABSENT_SUFFIX = "[landscape-reconciliation:absent]"

# Closed enum mirroring the ``ck_chat_messages_writer_principal`` CHECK
# constraint in ``web/sessions/models.py``. The Python Literal and the SQL
# CHECK are paired contracts: extending one without the other lets the
# dataclass validator pass while the DB rejects the row (or vice versa).
# The order here mirrors the CHECK declaration (models.py L116) for visual
# diff clarity. Adding a value is a governance action — see the
# closed-list-of-permitted-writers comment block at the
# ``audit_access_log_table`` definition for the same posture.
ChatMessageWriterPrincipal = Literal[
    "compose_loop",
    "route_user_message",
    "route_system_message",
    "admin_tool",
    "session_fork",
]

# Closed enum mirroring the ``ck_composition_states_provenance`` CHECK
# constraint in ``web/sessions/models.py``. Same paired-contract posture as
# ``ChatMessageWriterPrincipal``: extending one without the other lets the
# Python writer pass while the DB rejects the row (or vice versa). Order
# mirrors the CHECK declaration (models.py L257) for visual diff clarity.
# Adding a value is a governance action — see the dormant-value friction
# block at the ``composition_states_table`` definition for the activation
# contract (spec amendment + integration test + Filigree ticket).
CompositionStateProvenance = Literal[
    "tool_call",
    "convergence_persist",
    "plugin_crash_persist",
    "preflight_persist",
    # DORMANT (no live writer). Formerly written by the first-run tutorial's
    # pre-execution template normalizer, removed for tutorial-vs-regular
    # backend parity (the composer already emits ``row.field`` templates).
    # Retained in the closed list + CHECK constraint so historical audit rows
    # written under the old behavior remain representable; re-activating it is
    # a governance action (see the NO SILENT EXTENSION block in ``models.py``).
    "tutorial_normalization",
    "post_compose",
    "session_seed",
    "session_fork",
    "interpretation_resolve",
]

AUDIT_GRADE_VIEW_WRITER_PRINCIPAL = "audit_grade_view"
AUDIT_GRADE_VIEW_QUERY_ARG_ALLOWLIST: frozenset[str] = frozenset(
    {
        "include_tool_rows",
        "include_llm_audit",
        "include_raw_content",
        "limit",
        "offset",
    }
)

CHAT_MESSAGE_ROLE_VALUES: frozenset[str] = frozenset(get_args(ChatMessageRole))
COMPOSER_TRUST_MODE_VALUES: frozenset[str] = frozenset(get_args(ComposerTrustMode))
COMPOSER_DENSITY_DEFAULT_VALUES: frozenset[str] = frozenset(get_args(ComposerDensityDefault))
PROPOSAL_LIFECYCLE_STATUS_VALUES: frozenset[str] = frozenset(get_args(ProposalLifecycleStatus))
PROPOSAL_EVENT_TYPE_VALUES: frozenset[str] = frozenset(get_args(ProposalEventType))
CHAT_MESSAGE_WRITER_PRINCIPAL_VALUES: frozenset[str] = frozenset(get_args(ChatMessageWriterPrincipal))
COMPOSITION_STATE_PROVENANCE_VALUES: frozenset[str] = frozenset(get_args(CompositionStateProvenance))
SESSION_RUN_STATUS_VALUES: frozenset[str] = frozenset(get_args(SessionRunStatus))
SESSION_TERMINAL_RUN_STATUS_VALUES: frozenset[str] = frozenset(get_args(TerminalSessionRunStatus))
OPERATOR_COMPLETION_RUN_STATUS_VALUES: frozenset[str] = frozenset(get_args(OperatorCompletionSessionRunStatus))
SESSION_RUN_EVENT_TYPE_VALUES: frozenset[str] = frozenset(get_args(SessionRunEventType))
_RUN_COUNTER_FIELDS: tuple[str, ...] = (
    "rows_processed",
    "rows_succeeded",
    "rows_failed",
    "rows_routed_success",
    "rows_routed_failure",
    "rows_quarantined",
)

# Legal run status transitions. Implementations MUST reject any
# transition not in this table.
#
# Wrapped in MappingProxyType so importers cannot mutate the module-level
# table at runtime: ``LEGAL_RUN_TRANSITIONS["completed"] = frozenset({"running"})``
# raises TypeError rather than silently redefining terminal-state policy
# for every downstream consumer.  The inline dict has no retained alias,
# so the proxy is the only reference — there is no mutable back-door.
#
# Phase 2.2: pending → empty is legal (a run that begins and immediately
# finds an empty source skips the running state); running → every terminal
# value is legal (the row-count predicate decides which terminal value).
LEGAL_RUN_TRANSITIONS: Mapping[SessionRunStatus, frozenset[SessionRunStatus]] = MappingProxyType(
    {
        "pending": frozenset({"running", "completed_with_failures", "failed", "empty", "cancelled"}),
        "running": frozenset({"completed", "completed_with_failures", "failed", "empty", "cancelled"}),
        "completed": frozenset(),  # terminal
        "completed_with_failures": frozenset(),  # terminal
        "failed": frozenset(),  # terminal
        "empty": frozenset(),  # terminal
        "cancelled": frozenset(),  # terminal
    }
)

if frozenset(LEGAL_RUN_TRANSITIONS.keys()) != SESSION_RUN_STATUS_VALUES:
    raise AssertionError(
        f"LEGAL_RUN_TRANSITIONS keys {frozenset(LEGAL_RUN_TRANSITIONS.keys())} "
        f"must match SessionRunStatus values {SESSION_RUN_STATUS_VALUES}"
    )
if any(not allowed.issubset(SESSION_RUN_STATUS_VALUES) for allowed in LEGAL_RUN_TRANSITIONS.values()):
    raise AssertionError("LEGAL_RUN_TRANSITIONS contains a target not present in SessionRunStatus")
# elspeth-879f6de6bd: enforce that the empty-frozenset entries in
# LEGAL_RUN_TRANSITIONS exactly match the TerminalSessionRunStatus Literal.
# A drift here would silently re-introduce the recovery defect: a status
# could be terminal in the state machine (no legal outgoing transition)
# but absent from SESSION_TERMINAL_RUN_STATUS_VALUES (so the recovery
# guard would miss it and attempt an illegal transition), or vice versa.
_legal_transitions_terminal = frozenset(s for s, allowed in LEGAL_RUN_TRANSITIONS.items() if not allowed)
if _legal_transitions_terminal != SESSION_TERMINAL_RUN_STATUS_VALUES:
    raise AssertionError(
        f"LEGAL_RUN_TRANSITIONS terminal entries {sorted(_legal_transitions_terminal)} "
        f"must match TerminalSessionRunStatus {sorted(SESSION_TERMINAL_RUN_STATUS_VALUES)}"
    )


@dataclass(frozen=True, slots=True)
class RunEventRecord:
    """Represents a row from the run_events table."""

    id: UUID
    run_id: UUID
    sequence: int
    timestamp: datetime
    event_type: SessionRunEventType
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Represents a row from the sessions table.

    All fields are scalars or datetime -- no freeze guard needed.
    """

    id: UUID
    user_id: str
    auth_provider_type: AuthProviderType
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None
    forked_from_session_id: UUID | None = None
    forked_from_message_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ComposerSessionPreferencesRecord:
    """Represents composer preferences stored on the sessions row."""

    session_id: UUID
    trust_mode: ComposerTrustMode
    density_default: ComposerDensityDefault
    interpretation_review_disabled: bool
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.trust_mode not in COMPOSER_TRUST_MODE_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: sessions.trust_mode is {self.trust_mode!r}, expected one of {sorted(COMPOSER_TRUST_MODE_VALUES)}"
            )
        if self.density_default not in COMPOSER_DENSITY_DEFAULT_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: sessions.density_default is {self.density_default!r}, expected one of {sorted(COMPOSER_DENSITY_DEFAULT_VALUES)}"
            )


@dataclass(frozen=True, slots=True)
class ComposerSessionPreferencesTransition:
    """Result of a per-session composer-preferences PATCH.

    Carries both the value the PATCH overwrote (``prior``) and the value
    the PATCH wrote (``current``). Both are loaded inside the same write
    transaction as the audit + state update, so there is no TOCTOU
    window between read and write — see Phase 8 plan §"Service signature
    precondition (B2 — load-bearing)" for the rationale and the
    explicitly-rejected route-handler read-before-write alternative.

    The Phase 8b telemetry consumer reads ``prior.trust_mode`` to
    compute the ``from_mode`` attribute on
    ``composer.session.switched_total``; the B1 audit-payload extension
    records ``prior.trust_mode`` into ``proposal_events_table.payload``
    so the telemetry counter remains a strict subset of audit-recorded
    reality (audit-primacy superset rule, CLAUDE.md
    §"Telemetry and Logging").

    Both fields hold immutable frozen dataclass instances; no container
    fields here, so no ``__post_init__`` deep-freeze guard is required
    (per CLAUDE.md §"Frozen Dataclass Immutability"; scalar/frozen-
    dataclass wrappers do not need guards).
    """

    prior: ComposerSessionPreferencesRecord
    current: ComposerSessionPreferencesRecord


@dataclass(frozen=True, slots=True)
class PipelineProposalPublicMetadata:
    """Safe reload metadata for canonical pipeline proposals."""

    surface: PipelineProposalSurface
    draft_hash: str
    base: Mapping[str, Any]
    reviewed_anchor_hash: str
    repair_count: int
    skill_hash: str
    audit_payload_hash: str
    custody_result: Literal["not_required", "ready"]

    def __post_init__(self) -> None:
        freeze_fields(self, "base")


@dataclass(frozen=True, slots=True)
class CompositionProposalRecord:
    """Represents a durable pending/committed/rejected composer proposal."""

    id: UUID
    session_id: UUID
    tool_call_id: str
    user_message_id: UUID | None
    composer_model_identifier: str | None
    composer_model_version: str | None
    composer_provider: str | None
    composer_skill_hash: str | None
    tool_arguments_hash: str | None
    tool_name: str
    status: ProposalLifecycleStatus
    summary: str
    rationale: str
    affects: Sequence[str]
    arguments_json: Mapping[str, Any]
    arguments_redacted_json: Mapping[str, Any]
    base_state_id: UUID | None
    committed_state_id: UUID | None
    audit_event_id: UUID | None
    created_at: datetime
    updated_at: datetime
    pipeline_metadata: PipelineProposalPublicMetadata | None = None

    def __post_init__(self) -> None:
        if self.status not in PROPOSAL_LIFECYCLE_STATUS_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: composition_proposals.status is {self.status!r}, expected one of {sorted(PROPOSAL_LIFECYCLE_STATUS_VALUES)}"
            )
        composer_provenance = (
            self.composer_model_identifier,
            self.composer_model_version,
            self.composer_provider,
            self.composer_skill_hash,
            self.tool_arguments_hash,
        )
        if any(value is None for value in composer_provenance) and any(value is not None for value in composer_provenance):
            raise AuditIntegrityError("Tier 1: composition_proposals composer provenance fields must be all populated or all NULL")
        freeze_fields(self, "affects", "arguments_json", "arguments_redacted_json")


@dataclass(frozen=True, slots=True)
class ProposalEventRecord:
    """Represents an immutable composer proposal lifecycle event."""

    id: UUID
    session_id: UUID
    proposal_id: UUID | None
    event_type: ProposalEventType
    # Actor format is originator:role:id for request-scoped actors
    # (composer-web:user:{user_id}, user:{user_id}); system actors use
    # system:{component}.
    actor: str
    payload: Mapping[str, Any]
    created_at: datetime

    def __post_init__(self) -> None:
        if self.event_type not in PROPOSAL_EVENT_TYPE_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: proposal_events.event_type is {self.event_type!r}, expected one of {sorted(PROPOSAL_EVENT_TYPE_VALUES)}"
            )
        freeze_fields(self, "payload")


@dataclass(frozen=True, slots=True)
class AuthoritativePipelineProposal:
    """Verified pipeline authority reconstructed from one row + creation event."""

    row: CompositionProposalRecord
    proposal: PipelineProposal
    creation_event_id: UUID
    custody_result: Literal["not_required", "ready"]
    supersedes_proposal_id: UUID | None


@dataclass(frozen=True, slots=True)
class AuthoritativeCompositionProposal:
    """One exact creation-event classification used by generic routes."""

    row: CompositionProposalRecord
    pipeline: AuthoritativePipelineProposal | None


@dataclass(frozen=True, slots=True)
class PipelineProposalSettlementResult:
    """Atomic accepted proposal + committed immutable state."""

    proposal: CompositionProposalRecord
    state: CompositionStateRecord


@dataclass(frozen=True, slots=True)
class PipelineDispatchRecovery:
    """One durable successful dispatch available to resume settlement."""

    binding: PipelineDispatchAuditBinding
    executor_content_hash: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatMessageRecord:
    """Represents a row from the chat_messages table.

    tool_calls uses the stored LiteLLM array format and may contain nested
    mutable lists/dicts -- requires freeze guard when not None.

    raw_content stores the model's pre-synthesis prose when the visible
    content was augmented (operator-facing suffix appended) or replaced
    (false-completion-claim path) by ``_finalize_no_tool_response``. It
    is persisted for audit provenance and is returned in
    ChatMessageResponse only when the caller opts in via
    ``?include_raw_content=true``; otherwise the response field is
    ``null`` (the field is always present in the response shape).

    Producer contract: when raw_content is set, ``content`` MUST start
    with raw_content (all composer synthesis shapes are augmentations
    post-elspeth-9cfbad6901). Mechanically enforced at producer
    construction by
    ``web.composer.service._enforce_augmentation_prefix_invariant``.
    Consumers (notably ``routes._composer_history_content``) rely on
    the contract to detect synthesis structurally without a field-level
    discriminator; the field-level decoupling is tracked at
    ``elspeth-7ae1732ab2``.
    """

    id: UUID
    session_id: UUID
    role: ChatMessageRole
    content: str
    created_at: datetime
    writer_principal: ChatMessageWriterPrincipal
    sequence_no: int | None = None
    raw_content: str | None = None
    tool_calls: Sequence[Mapping[str, Any]] | None = None
    composition_state_id: UUID | None = None
    tool_call_id: str | None = None
    parent_assistant_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.role not in CHAT_MESSAGE_ROLE_VALUES:
            raise AuditIntegrityError(f"Tier 1: chat_messages.role is {self.role!r}, expected one of {sorted(CHAT_MESSAGE_ROLE_VALUES)}")
        # Tier-1 read guard: ``writer_principal`` mirrors the
        # ``ck_chat_messages_writer_principal`` CHECK constraint. Reading a
        # value outside the closed enum from our own session DB means
        # something catastrophic happened (constraint disabled, direct SQL
        # write, schema drift). Crash with a Tier-1 audit-integrity error
        # rather than letting a Literal-typed field carry a wider str at
        # runtime — same posture as the role guard above.
        if self.writer_principal not in CHAT_MESSAGE_WRITER_PRINCIPAL_VALUES:
            raise AuditIntegrityError(
                f"Tier 1: chat_messages.writer_principal is {self.writer_principal!r}, "
                f"expected one of {sorted(CHAT_MESSAGE_WRITER_PRINCIPAL_VALUES)}"
            )
        # tool_call_id / parent_assistant_id are scalar fields and need no
        # freeze guard (CLAUDE.md "Scalar-Only Fields Need No Guard"). Only
        # ``tool_calls`` carries mutable contents.
        if self.tool_calls is not None:
            freeze_fields(self, "tool_calls")


@dataclass(frozen=True, slots=True)
class CompositionStateData:
    """Input DTO for saving a new composition state version.

    Contains mutable container fields -- requires freeze guard.
    """

    source: InitVar[Mapping[str, Any] | None] = None
    sources: Mapping[str, Mapping[str, Any]] | None = None
    nodes: Sequence[Mapping[str, Any]] | None = None
    edges: Sequence[Mapping[str, Any]] | None = None
    outputs: Sequence[Mapping[str, Any]] | None = None
    metadata_: Mapping[str, Any] | None = None
    is_valid: bool = False
    validation_errors: Sequence[str] | None = None
    # Operational/audit meta describing how this state was reached. Distinct
    # from ``metadata_`` which carries user-facing PipelineMetadata. ``None``
    # is honest for revert/fork paths and for non-compose write paths.
    composer_meta: Mapping[str, Any] | None = None

    def __post_init__(self, source: Mapping[str, Any] | None) -> None:
        if source is not None:
            if self.sources is not None:
                raise AuditIntegrityError("CompositionStateData accepts either source or sources, not both")
            object.__setattr__(self, "sources", {"source": source})
        non_none = []
        if self.sources is not None:
            non_none.append("sources")
        if self.nodes is not None:
            non_none.append("nodes")
        if self.edges is not None:
            non_none.append("edges")
        if self.outputs is not None:
            non_none.append("outputs")
        if self.metadata_ is not None:
            non_none.append("metadata_")
        if self.validation_errors is not None:
            non_none.append("validation_errors")
        if self.composer_meta is not None:
            non_none.append("composer_meta")
        if non_none:
            freeze_fields(self, *non_none)


@dataclass(frozen=True, slots=True)
class CompositionStateRecord:
    """Represents a row from the composition_states table.

    Contains mutable container fields -- requires freeze guard.
    """

    id: UUID
    session_id: UUID
    version: int
    nodes: Sequence[Mapping[str, Any]] | None
    edges: Sequence[Mapping[str, Any]] | None
    outputs: Sequence[Mapping[str, Any]] | None
    metadata_: Mapping[str, Any] | None
    is_valid: bool
    validation_errors: Sequence[str] | None
    created_at: datetime
    derived_from_state_id: UUID | None
    # Operational/audit meta describing how this state was reached. Distinct
    # from ``metadata_`` which carries user-facing PipelineMetadata. ``None``
    # is honest for revert/fork paths and for non-compose write paths.
    composer_meta: Mapping[str, Any] | None = None
    sources: Mapping[str, Mapping[str, Any]] | None = None
    source: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        non_none = []
        if self.source is not None:
            non_none.append("source")
        if self.sources is not None:
            non_none.append("sources")
        if self.nodes is not None:
            non_none.append("nodes")
        if self.edges is not None:
            non_none.append("edges")
        if self.outputs is not None:
            non_none.append("outputs")
        if self.metadata_ is not None:
            non_none.append("metadata_")
        if self.validation_errors is not None:
            non_none.append("validation_errors")
        if self.composer_meta is not None:
            non_none.append("composer_meta")
        if non_none:
            freeze_fields(self, *non_none)


@dataclass(frozen=True, slots=True)
class AuditAccessLogRecord:
    """Represents a row from the audit_access_log table.

    ``query_args`` is a privacy-gated, closed allowlist mapping captured
    at the audit-grade messages route boundary. It may contain mutable
    JSON structures after SQLAlchemy deserialisation, so freeze it.
    """

    id: str
    timestamp: datetime
    session_id: str
    requesting_principal: str
    request_path: str
    query_args: Mapping[str, str]
    ip_address: str | None
    writer_principal: str

    def __post_init__(self) -> None:
        freeze_fields(self, "query_args")


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Represents a row from the runs table.

    All fields are scalars, datetime, or None -- no freeze guard needed.
    """

    id: UUID
    session_id: UUID
    state_id: UUID
    status: SessionRunStatus
    started_at: datetime
    finished_at: datetime | None
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed_success: int
    rows_routed_failure: int
    rows_quarantined: int
    error: str | None
    landscape_run_id: str | None
    pipeline_yaml: str | None

    def __post_init__(self) -> None:
        self._validate_counters()
        if self.status not in SESSION_RUN_STATUS_VALUES:
            raise AuditIntegrityError(f"Tier 1: runs.status is {self.status!r}, expected one of {sorted(SESSION_RUN_STATUS_VALUES)}")
        if self.status in SESSION_TERMINAL_RUN_STATUS_VALUES and self.finished_at is None:
            raise AuditIntegrityError(f"Tier 1: terminal runs.finished_at is NULL for status={self.status!r}")
        # Phase 2.2 (elspeth-0de989c56d): the four operator-completion terminal
        # values (completed / completed_with_failures / empty) all imply the
        # run reached the engine-completion path and produced a Landscape
        # audit record.  `failed` may or may not have a Landscape ID — the
        # engine takes the failed path on exceptions before any Landscape
        # write, so requiring a Landscape ID would crash legitimate
        # exception-bounded shapes.  `cancelled` is signal-bounded; same
        # rationale.
        if self.status in {"completed", "completed_with_failures", "empty"} and not self.landscape_run_id:
            raise AuditIntegrityError(f"Tier 1: status={self.status!r} run is missing landscape_run_id")
        if self.status == "failed" and not self.error:
            raise AuditIntegrityError("Tier 1: failed run is missing error")

    def _validate_counters(self) -> None:
        for field_name in _RUN_COUNTER_FIELDS:
            try:
                require_int(getattr(self, field_name), f"runs.{field_name}", min_value=0)
            except (TypeError, ValueError) as exc:
                raise AuditIntegrityError(f"Tier 1: {exc}") from exc

        if self.rows_routed_success > self.rows_succeeded:
            raise AuditIntegrityError(
                "Tier 1: rows_routed_success must be a subset of rows_succeeded "
                f"(got rows_routed_success={self.rows_routed_success}, rows_succeeded={self.rows_succeeded})"
            )
        if self.rows_routed_failure > self.rows_failed:
            raise AuditIntegrityError(
                "Tier 1: rows_routed_failure must be a subset of rows_failed "
                f"(got rows_routed_failure={self.rows_routed_failure}, rows_failed={self.rows_failed})"
            )
        if self.rows_quarantined > self.rows_failed:
            raise AuditIntegrityError(
                "Tier 1: rows_quarantined must be a subset of rows_failed "
                f"(got rows_quarantined={self.rows_quarantined}, rows_failed={self.rows_failed})"
            )


class InvalidForkTargetError(Exception):
    """Raised when attempting to fork from a non-user message.

    Route handlers catching this error should return 422.
    """

    def __init__(self, message_id: str, role: str) -> None:
        self.message_id = message_id
        self.role = role
        super().__init__(f"Can only fork from user messages, got role '{role}' for message {message_id}")


class SessionNotFoundError(ValueError):
    """Raised when a session id has no matching sessions row.

    Subclasses ``ValueError`` so older callers that still catch
    ``ValueError`` retain compatibility. New IDOR-sensitive route helpers
    catch this narrower type so unrelated value-construction failures do
    not collapse into not-found responses.
    """

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class IllegalRunTransitionError(ValueError):
    """Raised when ``update_run_status`` receives a transition forbidden by
    ``LEGAL_RUN_TRANSITIONS``.

    Subclasses ``ValueError`` for backwards-compatible reraise behaviour, but
    callers performing cancelled-race recovery (``ExecutionService._run_pipeline``)
    catch *this* class only — never the bare ``ValueError`` — so that the four
    other Tier-1 invariant breaches raised by ``update_run_status``
    (run-not-found, landscape_run_id overwrite, completed-without-landscape,
    failed-without-error) propagate without traversing a get_run round-trip
    that could mask the original fault.

    Why a subclass of ValueError (not Exception): existing test fixtures and
    one production call site at the run-lifecycle repository assert
    ``pytest.raises(ValueError)`` on illegal transitions; subclassing keeps
    those green while letting recovery code narrow on identity.
    """

    def __init__(self, current_status: str, target_status: str, allowed: frozenset[str]) -> None:
        self.current_status = current_status
        self.target_status = target_status
        self.allowed = allowed
        super().__init__(f"Illegal run transition: {current_status!r} → {target_status!r}. Allowed: {sorted(allowed)}")


class RunAlreadyActiveError(Exception):
    """Raised when attempting to create a run while one is already active.

    Seam contract D: HTTP handlers catching this error MUST return 409 with
    {"detail": str(exc), "error_type": "run_already_active"} -- not a bare
    HTTPException. See seam-contracts.md for the canonical error shape.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} already has an active run")


class StaleComposeStateError(RuntimeError):
    """Compose result was based on a no-longer-current composition state.

    Raised by ``SessionServiceProtocol.persist_compose_turn_async`` (and
    its concrete implementation ``SessionServiceImpl.persist_compose_turn``)
    when the session's current composition state changed between the LLM
    call and the persist attempt. Defined here on the protocol module so
    Phase 3 callers can catch the error without importing the concrete
    service class — the symbol is part of the public contract, not an
    implementation detail.

    Mirrors :class:`elspeth.contracts.errors.AuditIntegrityError`'s
    placement on the contracts layer: protocol-level error shapes belong
    on the abstraction, not on the concrete service module.
    """


class AuditAccessLogWriteError(RuntimeError):
    """Audit-grade transcript access could not be recorded.

    ``include_tool_rows=true`` exposes audit-grade transcript rows. If
    that access cannot be written to ``audit_access_log`` first, callers
    must fail closed and return no transcript rows.
    """


class ToolCallIDMismatchError(RuntimeError):
    """Assistant ``tool_calls`` and persisted tool rows disagreed on
    the set of tool-call IDs for one compose turn.

    Carries the four mutually-exclusive failure axes (missing, extra,
    duplicate-in-assistant, duplicate-in-rows) so the diagnostic
    string identifies WHICH violation fired without forcing the
    caller to re-derive it.

    Defined on the protocol module alongside
    :class:`StaleComposeStateError` because both are pre-DB exceptions
    referenced by ``SessionServiceProtocol.persist_compose_turn_async``.
    Phase 3 callers can catch the error without importing the concrete
    service class — the symbol is part of the public contract.
    """

    def __init__(
        self,
        *,
        missing: frozenset[str],
        extra: frozenset[str],
        duplicates_in_assistant: frozenset[str],
        duplicates_in_rows: frozenset[str],
    ) -> None:
        self.missing = missing
        self.extra = extra
        self.duplicates_in_assistant = duplicates_in_assistant
        self.duplicates_in_rows = duplicates_in_rows
        super().__init__(
            "persist_compose_turn: assistant tool_calls and tool rows "
            "disagree on the tool-call ID set "
            f"(missing={sorted(missing)!r}, extra={sorted(extra)!r}, "
            f"duplicates_in_assistant={sorted(duplicates_in_assistant)!r}, "
            f"duplicates_in_rows={sorted(duplicates_in_rows)!r}). "
            "Refusing to persist a turn that would leave the audit "
            "trail with an asymmetric assistant/tool transcript."
        )


@runtime_checkable
class SessionServiceProtocol(Protocol):
    """Protocol for session persistence operations."""

    async def create_session(
        self,
        user_id: str,
        title: str,
        auth_provider_type: AuthProviderType,
        forked_from_session_id: UUID | None = None,
        forked_from_message_id: UUID | None = None,
    ) -> SessionRecord: ...

    async def get_session(self, session_id: UUID) -> SessionRecord: ...

    async def update_session_title(self, session_id: UUID, title: str) -> SessionRecord: ...

    async def list_sessions(
        self,
        user_id: str,
        auth_provider_type: AuthProviderType,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> list[SessionRecord]: ...

    async def archive_session(self, session_id: UUID) -> None: ...

    async def get_composer_preferences(
        self,
        session_id: UUID,
    ) -> ComposerSessionPreferencesRecord: ...

    async def update_composer_preferences(
        self,
        session_id: UUID,
        *,
        trust_mode: ComposerTrustMode,
        density_default: ComposerDensityDefault,
        actor: str,
    ) -> ComposerSessionPreferencesTransition: ...

    async def create_composition_proposal(
        self,
        *,
        session_id: UUID,
        tool_call_id: str,
        tool_name: str,
        summary: str,
        rationale: str,
        affects: Sequence[str],
        arguments_json: Mapping[str, Any],
        arguments_redacted_json: Mapping[str, Any],
        base_state_id: UUID | None,
        actor: str,
        user_message_id: UUID | None = None,
        composer_model_identifier: str | None = None,
        composer_model_version: str | None = None,
        composer_provider: str | None = None,
        composer_skill_hash: str | None = None,
        tool_arguments_hash: str | None = None,
    ) -> CompositionProposalRecord: ...

    async def create_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        plan: PipelinePlanResult,
        summary: str,
        rationale: str,
        affects: Sequence[str],
        arguments_redacted_json: Mapping[str, Any],
        actor: str,
        composer_model_identifier: str,
        composer_model_version: str,
        composer_provider: str,
        user_message_id: UUID | None = None,
        supersedes_proposal_id: UUID | None = None,
    ) -> CompositionProposalRecord: ...

    async def get_authoritative_pipeline_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        reviewed_facts: Mapping[str, Any],
    ) -> AuthoritativePipelineProposal: ...

    async def get_authoritative_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        reviewed_facts: Mapping[str, Any] | None,
    ) -> AuthoritativeCompositionProposal: ...

    async def settle_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any],
        state: CompositionStateData,
        candidate_content_hash: str,
        executor_content_hash: str,
        final_composer_metadata: Mapping[str, Any] | None,
        dispatch: PipelineDispatchAuditBinding,
        actor: str,
    ) -> PipelineProposalSettlementResult: ...

    async def get_pipeline_dispatch_recovery(
        self,
        *,
        authority: AuthoritativePipelineProposal,
    ) -> PipelineDispatchRecovery | None: ...

    async def reject_pipeline_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        draft_hash: str,
        reviewed_facts: Mapping[str, Any] | None,
        reason: PipelineProposalRejectionReason,
        dispatch: PipelineDispatchAuditBinding | None,
        actor: str,
    ) -> CompositionProposalRecord: ...

    async def list_composition_proposals(
        self,
        session_id: UUID,
        *,
        status: ProposalLifecycleStatus | None = None,
    ) -> list[CompositionProposalRecord]: ...

    async def reject_composition_proposal(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord: ...

    async def mark_composition_proposal_committed(
        self,
        *,
        session_id: UUID,
        proposal_id: UUID,
        committed_state_id: UUID,
        actor: str,
    ) -> CompositionProposalRecord: ...

    async def list_proposal_events(
        self,
        session_id: UUID,
    ) -> list[ProposalEventRecord]: ...

    async def create_pending_interpretation_event(
        self,
        *,
        session_id: UUID,
        composition_state_id: UUID,
        affected_node_id: str,
        tool_call_id: str,
        user_term: str,
        kind: InterpretationKind,
        llm_draft: str,
        model_identifier: str,
        model_version: str,
        provider: str,
        composer_skill_hash: str,
        created_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Insert a PENDING interpretation event.

        ``kind`` must be supplied explicitly by the caller. Implementations
        MUST validate the affected component in the parent composition state
        before INSERT (writer-boundary check per CLAUDE.md offensive
        programming): ``invented_source`` targets the synthetic ``source``
        component and requires persisted source-authoring metadata;
        ``pipeline_decision`` targets the node that implements the reviewed
        shape decision; prompt/vague transform kinds target real LLM nodes in
        ``composition_states.nodes``. Raises
        ``ValueError`` on a missing state, malformed target, unknown node, or
        non-``InterpretationKind`` kind.
        """
        ...

    async def resolve_interpretation_event(
        self,
        *,
        session_id: UUID,
        event_id: UUID,
        choice: InterpretationChoice,
        amended_value: str | None,
        actor: str,
        resolved_at: datetime | None = None,
        runtime_model_identifier: str | None = None,
        runtime_model_version: str | None = None,
    ) -> tuple[InterpretationEventRecord, CompositionStateRecord]:
        """Commit a resolution and update the affected interpretation surface.

        F-14: ``accepted_value`` is computed internally — implementations
        read ``llm_draft`` from the pending row when ``choice`` is
        ``ACCEPTED_AS_DRAFTED``, and use ``amended_value`` when ``choice``
        is ``AMENDED``.

        Single transaction. Raises ``ValueError`` for no pending event
        (TOCTOU / IDOR), missing composition state or affected node,
        or any prompt-template patch failure.
        """
        ...

    async def list_interpretation_events(
        self,
        session_id: UUID,
        *,
        status: Literal["pending", "all"] = "all",
        composition_state_id: UUID | None = None,
        sources: Sequence[InterpretationSource] | None = None,
    ) -> list[InterpretationEventRecord]:
        """Read-back of interpretation events for the session.

        Returns rows ordered by ``created_at, id``. ``status='pending'``
        filters to ``choice='pending'`` rows; ``status='all'`` returns
        every row.

        ``sources``: when set, filters to rows whose
        ``interpretation_source`` is in the supplied sequence. Used by
        the opt-out audit-summary surface
        (``GET /interpretations/opt_out_summary``) to retrieve only
        ``auto_interpreted_opt_out`` and ``auto_interpreted_no_surfaces``
        rows. ``None`` (default) imposes no source filter.
        """
        ...

    async def record_session_interpretation_opt_out(
        self,
        *,
        session_id: UUID,
        actor: str,
        opted_out_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Mark the session as 'don't surface interpretations any more'.

        F-29: idempotent. If an opted-out row already exists for this
        session, returns the existing record without inserting a duplicate;
        the sessions boolean stays true. Atomic single transaction
        (interpretation_events INSERT + sessions UPDATE inside one
        write lock).
        """
        ...

    async def upsert_skill_markdown_history(
        self,
        *,
        skill_hash: str,
        filename: str,
        content: str,
        first_seen_at: datetime | None = None,
    ) -> bool:
        """Best-effort INSERT-OR-IGNORE into ``skill_markdown_history`` (F-5c).

        Captures the exact composer-skill markdown text the LLM was
        prompted with so a forensic auditor can reconstruct it from the
        ``composer_skill_hash`` recorded on later interpretation event
        rows. Hash is the primary key; subsequent calls with the same
        hash are no-ops.

        Returns ``True`` when a row was inserted, ``False`` when it
        already existed. Best-effort only — NOT transactional with the
        interpretation-event row write.
        """
        ...

    async def record_auto_interpreted_no_surfaces_event(
        self,
        *,
        session_id: UUID,
        actor: str,
        kind: InterpretationKind,
        model_identifier: str,
        model_version: str,
        provider: str,
        composer_skill_hash: str,
        created_at: datetime | None = None,
    ) -> InterpretationEventRecord:
        """Write an AUTO_INTERPRETED_NO_SURFACES row (Phase 5b Task 5, F-6).

        Called by the compose loop when the per-term or per-day rate cap
        is hit for ``request_interpretation_review``: the LLM is expected
        to fall back to baking the interpretation directly into the prompt
        template without surfacing it for review. This writer records the
        fact in the audit trail so an auditor can distinguish "user opted
        out" from "rate cap exhausted" via ``interpretation_source``.

        Row shape (see ``ck_interpretation_events_no_surfaces_shape``):
        * ``interpretation_source = 'auto_interpreted_no_surfaces'``
        * ``choice = 'opted_out'`` (semantics: resolved-at-write — there
          is no pending surface to acknowledge)
        * Interpretation-surface fields are NULL: ``composition_state_id``,
          ``affected_node_id``, ``tool_call_id``, ``user_term``,
          ``llm_draft`` (the rejected request never produced a surface).
        * ``kind`` and LLM provenance fields MUST be populated — the composer LLM
          that triggered the rate cap is fully identifiable from the
          compose-loop snapshot.
        * ``arguments_hash`` is NULL because no user-visible surface was
          created to resolve.
        * ``resolved_at`` equals ``created_at`` (the rate-cap event is
          itself a resolution).
        """
        ...

    async def add_message(
        self,
        session_id: UUID,
        role: ChatMessageRole,
        content: str,
        *,
        writer_principal: ChatMessageWriterPrincipal,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        composition_state_id: UUID | None = None,
        raw_content: str | None = None,
        tool_call_id: str | None = None,
        parent_assistant_id: UUID | None = None,
    ) -> ChatMessageRecord: ...

    async def get_messages(
        self,
        session_id: UUID,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[ChatMessageRecord]: ...

    async def count_tool_responses_for_assistant_async(
        self,
        *,
        session_id: str,
        assistant_message_id: str | None,
    ) -> int:
        """Count persisted tool rows linked to an assistant message."""
        ...

    async def record_audit_grade_view_async(
        self,
        *,
        session_id: str,
        requesting_principal: str,
        request_path: str,
        query_args: Mapping[str, str],
        ip_address: str | None,
    ) -> None:
        """Append one audit_access_log row before exposing tool rows."""
        ...

    async def save_composition_state(
        self,
        session_id: UUID,
        state: CompositionStateData,
        *,
        provenance: CompositionStateProvenance,
    ) -> CompositionStateRecord:
        """Save a new immutable composition state snapshot.

        ``provenance`` MUST be one of the values enumerated by the
        ``ck_composition_states_provenance`` CHECK constraint and the
        :data:`CompositionStateProvenance` Literal. It records WHY this row
        was written and is the load-bearing discriminator for the
        backward-direction INV-AUDIT-AHEAD invariant (§4.1.2). Implementations
        MUST persist the value verbatim — no defaulting, no coercion: a
        confident wrong attribution is evidence-tampering-class harm under
        the auditability standard.
        """
        ...

    async def get_current_state(
        self,
        session_id: UUID,
    ) -> CompositionStateRecord | None: ...

    async def get_state(self, state_id: UUID) -> CompositionStateRecord: ...

    async def get_state_in_session(
        self,
        state_id: UUID,
        session_id: UUID,
    ) -> CompositionStateRecord:
        """Fetch a composition state with a session-scope invariant check.

        Migration 007 added a composite FK ``(state_id, session_id)`` on
        tables that reference ``composition_states``, which prevents
        *future* cross-session state references at the schema layer. This
        method is the runtime defence-in-depth for pre-007 rows repaired
        with Variant-A (delete orphans) — and for any future code path
        that acquires a ``state_id`` indirectly (e.g. via a
        ``RunRecord.state_id`` carried through the fork lineage) and then
        resolves it inside a session-scoped handler.

        Implementations MUST raise ``AuditIntegrityError`` when the
        resolved state's ``session_id`` does not match the caller-supplied
        ``session_id``. That is a Tier 1 audit anomaly: the state was
        reachable from a run but does not belong to the session hosting
        the run. Silent coercion or a soft 404 would produce a confident
        wrong answer — exactly the pattern CLAUDE.md forbids for our own
        data. Raises ``ValueError`` when the state does not exist at all,
        consistent with ``get_state``.
        """
        ...

    async def get_state_versions(
        self,
        session_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CompositionStateRecord]: ...

    async def set_active_state(
        self,
        session_id: UUID,
        state_id: UUID,
    ) -> CompositionStateRecord:
        """Set the active composition state for a session.

        Creates a new state version derived from the specified state_id.
        Sets derived_from_state_id on the new version to record lineage.
        """
        ...

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord: ...

    async def get_run(self, run_id: UUID) -> RunRecord: ...

    async def list_runs_for_session(self, session_id: UUID) -> list[RunRecord]: ...

    async def update_run_status(
        self,
        run_id: UUID,
        status: SessionRunStatus,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_succeeded: int | None = None,
        rows_failed: int | None = None,
        rows_routed_success: int | None = None,
        rows_routed_failure: int | None = None,
        rows_quarantined: int | None = None,
    ) -> None:
        """Update a run's status and metadata.

        Transitions MUST comply with LEGAL_RUN_TRANSITIONS.

        landscape_run_id is write-once: once set to a non-None value,
        subsequent calls MUST NOT overwrite it. Implementations MUST
        raise ValueError if landscape_run_id is provided but the run
        already has one set.
        """
        ...

    async def append_run_event(
        self,
        *,
        run_id: UUID,
        timestamp: datetime,
        event_type: SessionRunEventType,
        data: Mapping[str, Any],
    ) -> RunEventRecord:
        """Append a structured execution event for replay/audit."""
        ...

    async def list_run_events(self, run_id: UUID) -> list[RunEventRecord]:
        """Return persisted execution events for a run in event order."""
        ...

    async def record_blob_inline_resolutions(
        self,
        *,
        run_id: UUID,
        resolutions: Sequence[ResolvedBlobContent],
        attempt: int = 1,
    ) -> None:
        """Write audit rows for inline-content blob refs before plugin construction."""
        ...

    async def get_active_run(
        self,
        session_id: UUID,
    ) -> RunRecord | None: ...

    async def prune_state_versions(
        self,
        session_id: UUID,
        keep_latest: int = 50,
    ) -> int:
        """Delete old composition state versions beyond keep_latest.

        Preserves the most recent `keep_latest` versions and any versions
        referenced by a run (via runs.state_id). Returns the count of
        deleted versions.
        """
        ...

    async def fork_session(
        self,
        source_session_id: UUID,
        fork_message_id: UUID,
        new_message_content: str,
        user_id: str,
        auth_provider_type: AuthProviderType,
    ) -> tuple[SessionRecord, list[ChatMessageRecord], CompositionStateRecord | None]:
        """Fork a session from a specific user message.

        Creates a new session with inherited history and state up to the
        fork point. The original session is never mutated.
        """
        ...

    async def update_message_composition_state(
        self,
        message_id: UUID,
        composition_state_id: UUID,
    ) -> None:
        """Re-point a message's composition_state_id to a different state.

        Used after fork blob-remapping creates a replacement state so
        the user message's provenance tracks the rewritten (self-contained)
        state rather than the original copy.
        """
        ...

    async def cancel_orphaned_runs(
        self,
        session_id: UUID,
        max_age_seconds: int = 3600,
    ) -> list[RunRecord]:
        """Force-cancel runs stuck in 'running' status beyond max_age_seconds.

        Returns the list of cancelled RunRecords. Called by the execution
        service on startup and periodically to prevent orphaned runs from
        permanently blocking sessions.
        """
        ...

    async def cancel_all_orphaned_runs(
        self,
        max_age_seconds: int | None = None,
        exclude_run_ids: frozenset[str] = frozenset(),
        reason: str | None = None,
    ) -> int:
        """Force-cancel orphaned runs across all sessions.

        Called on startup to recover sessions blocked by runs orphaned
        during a previous server crash. Returns the count of cancelled runs.

        Args:
            max_age_seconds: Only cancel runs older than this. None cancels
                all non-terminal runs (correct for single-process servers
                where every non-terminal run is orphaned after restart).
            exclude_run_ids: Run IDs known to have active executor threads.
                These are skipped even if they exceed max_age_seconds.
            reason: Written to the error column so operators can distinguish
                orphan-cleanup cancellations from user cancellations.
        """
        ...

    async def cancel_all_orphaned_run_records(
        self,
        max_age_seconds: int | None = None,
        exclude_run_ids: frozenset[str] = frozenset(),
        reason: str | None = None,
    ) -> list[RunRecord]:
        """Force-cancel orphaned runs and return the cancelled records.

        Used by app-level startup reconciliation to terminalize matching
        Landscape audit rows via each record's ``landscape_run_id``.
        """
        ...

    async def list_pending_landscape_reconciliations(self) -> list[RunRecord]:
        """Return cancelled runs whose error ends in the exact pending marker."""
        ...

    async def mark_landscape_reconciliation_outcomes(
        self,
        *,
        complete_run_ids: frozenset[UUID],
        absent_run_ids: frozenset[UUID],
    ) -> None:
        """Atomically replace exact pending suffixes with closed outcomes."""
        ...

    async def persist_compose_turn_async(
        self,
        *,
        session_id: str,
        assistant_content: str,
        raw_content: str | None = None,
        redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...],
        redacted_tool_rows: tuple[Any, ...],
        parent_composition_state_id: str | None,
        expected_current_state_id: str | None,
        writer_principal: ChatMessageWriterPrincipal,
        plugin_crash_pending: bool,
    ) -> Any:
        """Persist one compose turn (assistant + tool rows + per-tool
        composition states) atomically.

        Spec §5.2.2. The async dispatcher; the underlying sync work runs
        in a worker thread under ``asyncio.shield`` (commit-wins
        cancellation contract — see ``SessionServiceImpl
        .persist_compose_turn_async``).

        Raises :class:`StaleComposeStateError` when the session's current
        composition state changed between the LLM call and the persist
        attempt. Raises :class:`ToolCallIDMismatchError` when the
        assistant ``tool_calls`` IDs and the tool rows'
        ``tool_call_id`` values are not the same unique set.
        """
        ...
