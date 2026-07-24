"""Pydantic response models for execution endpoints.

All models in this module serialize **system-owned data** (Tier 1 in the
Data Manifesto).  They use strict validation and forbid extra fields so
that internal type drift crashes loudly instead of silently coercing
values or dropping unknown fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, ClassVar, Final, Literal, Self, TypeAliasType, get_args

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from elspeth.contracts import OPERATION_TYPE_VALUES, NodeStateStatus, Operation, OperationType, TerminalOutcome
from elspeth.web.sessions.protocol import (
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SessionRunStatus,
    TerminalSessionRunStatus,
)

# Closed enum mirroring the ``ck_run_events_type`` CHECK constraint in
# ``web/sessions/models.py`` (L429). The Python Literal and the SQL CHECK
# are paired contracts: extending one without the other lets the
# pydantic writer pass while the DB rejects the row, or vice versa.
# Order mirrors the CHECK declaration for visual diff clarity. Adding a
# value is a governance action — see the closed-list-of-permitted-writers
# comment block at ``audit_access_log_table`` for the same posture.
RunEventType = Literal["progress", "error", "completed", "cancelled", "failed"]
RUN_EVENT_TYPE_VALUES: frozenset[str] = frozenset(get_args(RunEventType))

ValidationCheckName = Literal[
    "plugin_enablement",
    "operator_profile_options",
    "required_control_availability",
    "required_control_coverage",
    "path_allowlist",
    "web_scrape_network_policy",
    "secret_refs",
    "semantic_contracts",
    "batch_transform_options",
    "interpretation_review",
    "blob_inline_refs",
    "managed_identity_policy",
    "llm_retry_budget_policy",
    "llm_base_url_policy",
    "llm_tracing_policy",
    "aws_s3_endpoint_url_policy",
    "settings_load",
    "plugin_instantiation",
    "value_source_compliance",
    "graph_structure",
    "route_target_resolution",
    "schema_compatibility",
    "identity_node_advisory",
    "state_exists",
    "advisor_signoff",
    "proof_diagnostics",
]
VALIDATION_CHECK_NAME_VALUES: frozenset[str] = frozenset(get_args(ValidationCheckName))

CHECK_PLUGIN_ENABLEMENT: Final[ValidationCheckName] = "plugin_enablement"
CHECK_OPERATOR_PROFILE_OPTIONS: Final[ValidationCheckName] = "operator_profile_options"
CHECK_REQUIRED_CONTROL_AVAILABILITY: Final[ValidationCheckName] = "required_control_availability"
CHECK_REQUIRED_CONTROL_COVERAGE: Final[ValidationCheckName] = "required_control_coverage"
CHECK_PATH_ALLOWLIST: Final[ValidationCheckName] = "path_allowlist"
CHECK_WEB_SCRAPE_NETWORK_POLICY: Final[ValidationCheckName] = "web_scrape_network_policy"
CHECK_SECRET_REFS: Final[ValidationCheckName] = "secret_refs"
CHECK_SEMANTIC_CONTRACTS: Final[ValidationCheckName] = "semantic_contracts"
CHECK_BATCH_TRANSFORM_OPTIONS: Final[ValidationCheckName] = "batch_transform_options"
CHECK_INTERPRETATION_REVIEW: Final[ValidationCheckName] = "interpretation_review"
CHECK_BLOB_INLINE_REFS: Final[ValidationCheckName] = "blob_inline_refs"
CHECK_MANAGED_IDENTITY_POLICY: Final[ValidationCheckName] = "managed_identity_policy"
CHECK_LLM_RETRY_BUDGET_POLICY: Final[ValidationCheckName] = "llm_retry_budget_policy"
CHECK_LLM_BASE_URL_POLICY: Final[ValidationCheckName] = "llm_base_url_policy"
CHECK_LLM_TRACING_POLICY: Final[ValidationCheckName] = "llm_tracing_policy"
CHECK_AWS_S3_ENDPOINT_URL_POLICY: Final[ValidationCheckName] = "aws_s3_endpoint_url_policy"
CHECK_SETTINGS: Final[ValidationCheckName] = "settings_load"
RUNTIME_CHECK_PLUGIN_INSTANTIATION: Final[ValidationCheckName] = "plugin_instantiation"
CHECK_VALUE_SOURCE_COMPLIANCE: Final[ValidationCheckName] = "value_source_compliance"
RUNTIME_CHECK_GRAPH_STRUCTURE: Final[ValidationCheckName] = "graph_structure"
CHECK_ROUTE_TARGETS: Final[ValidationCheckName] = "route_target_resolution"
RUNTIME_CHECK_SCHEMA_COMPATIBILITY: Final[ValidationCheckName] = "schema_compatibility"
CHECK_IDENTITY_NODE_ADVISORY: Final[ValidationCheckName] = "identity_node_advisory"
CHECK_STATE_EXISTS: Final[ValidationCheckName] = "state_exists"
CHECK_ADVISOR_SIGNOFF: Final[ValidationCheckName] = "advisor_signoff"
CHECK_PROOF_DIAGNOSTICS: Final[ValidationCheckName] = "proof_diagnostics"

VALIDATION_BLOCKING_CHECK_NAMES: tuple[ValidationCheckName, ...] = (
    CHECK_PLUGIN_ENABLEMENT,
    CHECK_OPERATOR_PROFILE_OPTIONS,
    CHECK_REQUIRED_CONTROL_AVAILABILITY,
    CHECK_REQUIRED_CONTROL_COVERAGE,
    CHECK_PATH_ALLOWLIST,
    CHECK_WEB_SCRAPE_NETWORK_POLICY,
    CHECK_SECRET_REFS,
    CHECK_SEMANTIC_CONTRACTS,
    CHECK_BATCH_TRANSFORM_OPTIONS,
    CHECK_INTERPRETATION_REVIEW,
    CHECK_BLOB_INLINE_REFS,
    CHECK_MANAGED_IDENTITY_POLICY,
    CHECK_LLM_RETRY_BUDGET_POLICY,
    CHECK_LLM_BASE_URL_POLICY,
    CHECK_LLM_TRACING_POLICY,
    CHECK_AWS_S3_ENDPOINT_URL_POLICY,
    CHECK_SETTINGS,
    RUNTIME_CHECK_PLUGIN_INSTANTIATION,
    CHECK_VALUE_SOURCE_COMPLIANCE,
    RUNTIME_CHECK_GRAPH_STRUCTURE,
    CHECK_ROUTE_TARGETS,
    RUNTIME_CHECK_SCHEMA_COMPATIBILITY,
    CHECK_STATE_EXISTS,
    CHECK_ADVISOR_SIGNOFF,
    CHECK_PROOF_DIAGNOSTICS,
)
VALIDATION_CHECK_NAMES: tuple[ValidationCheckName, ...] = (
    *VALIDATION_BLOCKING_CHECK_NAMES,
    CHECK_IDENTITY_NODE_ADVISORY,
)
if frozenset(VALIDATION_CHECK_NAMES) != VALIDATION_CHECK_NAME_VALUES:
    raise AssertionError("VALIDATION_CHECK_NAMES must match ValidationCheckName Literal values")

ValidationCheckOutcomeCode = Literal[
    "secret_refs.no_refs",
    "secret_refs.resolved",
    "secret_refs.unresolved",
    "secret_refs.skipped_no_service",
    "validation.skipped_after_failure",
]
VALIDATION_CHECK_OUTCOME_CODE_VALUES: frozenset[str] = frozenset(get_args(ValidationCheckOutcomeCode))

CHECK_OUTCOME_SECRET_REFS_NO_REFS: Final[ValidationCheckOutcomeCode] = "secret_refs.no_refs"
CHECK_OUTCOME_SECRET_REFS_RESOLVED: Final[ValidationCheckOutcomeCode] = "secret_refs.resolved"
CHECK_OUTCOME_SECRET_REFS_UNRESOLVED: Final[ValidationCheckOutcomeCode] = "secret_refs.unresolved"
CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE: Final[ValidationCheckOutcomeCode] = "secret_refs.skipped_no_service"
CHECK_OUTCOME_SKIPPED_AFTER_FAILURE: Final[ValidationCheckOutcomeCode] = "validation.skipped_after_failure"


class _StrictResponse(BaseModel):
    """Base model for execution response schemas — Tier 1 trust rules.

    strict=True:   No coercion.  ``"7"`` into an ``int`` field crashes
                   instead of silently becoming ``7``.
    extra="forbid": Unexpected fields crash instead of being silently
                    dropped.

    All execution schemas inherit this.  ``RunEvent.timestamp`` uses
    ``Field(strict=False)`` for the WebSocket reconnect JSON round-trip
    (ISO string → datetime), paired with a ``field_validator`` that
    rejects Unix epoch integers.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class ValidationCheck(_StrictResponse):
    """Individual check result from dry-run validation."""

    _SECRET_REFS_CHECK_NAME: ClassVar[ValidationCheckName] = CHECK_SECRET_REFS
    _SECRET_REFS_OUTCOME_CODES: ClassVar[frozenset[str]] = frozenset(
        {
            CHECK_OUTCOME_SECRET_REFS_NO_REFS,
            CHECK_OUTCOME_SECRET_REFS_RESOLVED,
            CHECK_OUTCOME_SECRET_REFS_UNRESOLVED,
            CHECK_OUTCOME_SECRET_REFS_SKIPPED_NO_SERVICE,
            CHECK_OUTCOME_SKIPPED_AFTER_FAILURE,
        }
    )

    name: ValidationCheckName
    passed: bool
    detail: str
    # Structured field: node ids affected by this check (e.g. identity-node
    # advisories). Populated by the producer (validation.py) in the same
    # commit that adds this field — no compat-shim default per CLAUDE.md
    # No-Legacy policy.
    affected_nodes: tuple[str, ...]
    # Machine-readable producer signal for checks whose detail is display prose.
    # Required-but-nullable: every construction site must either record a
    # structured outcome or explicitly say this check has none.
    outcome_code: ValidationCheckOutcomeCode | None

    @model_validator(mode="after")
    def _check_secret_refs_outcome_code(self) -> Self:
        if self.name == self._SECRET_REFS_CHECK_NAME and self.outcome_code not in self._SECRET_REFS_OUTCOME_CODES:
            raise ValueError("secret_refs checks must carry a secret_refs.* outcome_code or validation.skipped_after_failure")
        return self


class ValidationError(_StrictResponse):
    """Error with per-component attribution."""

    component_id: str | None
    component_type: str | None
    message: str
    suggestion: str | None
    # Structured discriminant for semantic error routing (e.g.
    # "missing_secret_ref", "fabricated_secret"). Populated at every
    # construction site — no compat-shim default per CLAUDE.md No-Legacy
    # policy. Sites that have no semantic code pass None explicitly.
    error_code: str | None


class ValidationWarning(_StrictResponse):
    """Non-blocking validation warning with per-component attribution."""

    component_id: str | None
    component_type: str | None
    message: str
    suggestion: str | None
    warning_code: str


class SemanticEdgeContractResponse(_StrictResponse):
    """Per-edge semantic-contract result for HTTP serialization.

    Field set mirrors composer_mcp/server.py::_SemanticEdgeContractPayload
    so MCP and HTTP clients receive identical shapes.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    outcome: str  # SemanticOutcome value: "satisfied" | "conflict" | "unknown"
    requirement_code: str


class ValidationReadinessBlocker(_StrictResponse):
    """Machine-readable readiness blocker.

    ``code`` is the routing discriminant consumed by the frontend and composer
    finalizer. ``detail`` is display-safe context, not raw prompt text.
    """

    code: str
    component_id: str | None
    component_type: str | None
    detail: str


class ValidationReadiness(_StrictResponse):
    """Backend-owned readiness classification for a composition state."""

    authoring_valid: bool
    execution_ready: bool
    completion_ready: bool
    blockers: list[ValidationReadinessBlocker]


class ValidationResult(_StrictResponse):
    """Result of dry-run validation against real engine code."""

    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[ValidationError]
    warnings: list[ValidationWarning] = []
    # Required: every producer must state which contract failed or passed.
    # This prevents callers from reconstructing readiness heuristically from
    # prose errors or parser side effects.
    readiness: ValidationReadiness
    semantic_contracts: list[SemanticEdgeContractResponse] = []


class ExecuteRequest(_StrictResponse):
    """Optional execution-launch acknowledgement payload."""

    fanout_ack_token: str | None = Field(default=None, min_length=1)


class WebSocketTicketResponse(_StrictResponse):
    """Short-lived one-use ticket for the execution progress WebSocket."""

    ticket: str
    expires_at: datetime


# ── Typed event payload models ──────────────────────────────────────────
#
# Each event_type has a dedicated payload model so that the server-side
# schema catches producer drift between service.py, routes.py, and the
# frontend TypeScript types.


class ProgressData(_StrictResponse):
    """Payload for ``progress`` events with explicit units.

    All six counter fields are REQUIRED with no defaults.  The engine's
    ``ProgressEvent`` (contracts/cli.py) always carries real counter values
    at emission time; defaulting any of them to ``0`` on the wire would
    fabricate "we don't know" as "definitely zero" — violating the CLAUDE.md
    fabrication test.  Mid-run, an operator must be able to distinguish
    "no rows have succeeded yet" from "the field was never populated".

    The old ``rows_*`` names mixed source rows and materialized token outcomes.
    Progress events are still best-known live counters, not terminal
    accounting, but the field names now make the unit boundary explicit.
    """

    source_rows_processed: int = Field(ge=0)
    tokens_succeeded: int = Field(ge=0)
    tokens_failed: int = Field(ge=0)
    tokens_quarantined: int = Field(ge=0)
    tokens_routed_success: int = Field(ge=0)
    tokens_routed_failure: int = Field(ge=0)


class ErrorData(_StrictResponse):
    """Payload for ``error`` events (non-terminal, per-row).

    Reserved for per-row error reporting during a run: the frontend has a
    handler for this event type (so dropping the schema would break the
    contract with clients), and the producer side is defined here so
    backend emission can be added without a second schema round-trip.
    No routine producer path emits it yet; adding one is the intended
    future use of this class.
    """

    message: str = Field(min_length=1)
    node_id: str | None
    row_id: str | None


class RunAccountingSource(_StrictResponse):
    """Source-ingestion counts for a run."""

    rows_processed: int = Field(ge=0)


class RunAccountingTokens(_StrictResponse):
    """Pipeline-token accounting for emitted materialized work."""

    emitted: int = Field(ge=0)
    terminal: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    structural: int = Field(ge=0)
    pending: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_token_balance(self) -> Self:
        if self.terminal != self.succeeded + self.failed + self.structural:
            raise ValueError(
                "tokens.terminal must equal tokens.succeeded + tokens.failed + tokens.structural "
                f"(got terminal={self.terminal}, succeeded={self.succeeded}, "
                f"failed={self.failed}, structural={self.structural})"
            )
        if self.emitted != self.terminal + self.pending:
            raise ValueError(
                "tokens.emitted must equal tokens.terminal + tokens.pending "
                f"(got emitted={self.emitted}, terminal={self.terminal}, pending={self.pending})"
            )
        return self


class RunAccountingRouting(_StrictResponse):
    """Routing/disposition subset counts for terminal tokens."""

    routed_success: int = Field(ge=0)
    routed_failure: int = Field(ge=0)
    quarantined: int = Field(ge=0)
    discarded: int = Field(ge=0)


class RunAccountingIntegrity(_StrictResponse):
    """Closure integrity of the Landscape token ledger."""

    closure: Literal["closed", "open", "unknown"]
    missing_terminal_outcomes: int = Field(ge=0)
    duplicate_terminal_outcomes: int = Field(ge=0)


class RunAccounting(_StrictResponse):
    """Explicit run accounting split by unit of account."""

    source: RunAccountingSource
    sources: dict[str, RunAccountingSource] = Field(default_factory=dict)
    tokens: RunAccountingTokens
    routing: RunAccountingRouting
    integrity: RunAccountingIntegrity

    @model_validator(mode="after")
    def _check_integrity_contract(self) -> Self:
        if not self.sources:
            object.__setattr__(self, "sources", {"source": self.source})
        else:
            source_total = sum(source.rows_processed for source in self.sources.values())
            if source_total != self.source.rows_processed:
                raise ValueError(
                    "source.rows_processed must equal the sum of per-source rows "
                    f"(got source.rows_processed={self.source.rows_processed}, per_source_total={source_total})"
                )
        if self.routing.routed_success > self.tokens.succeeded:
            raise ValueError(
                "routing.routed_success must be a subset of tokens.succeeded "
                f"(got routed_success={self.routing.routed_success}, succeeded={self.tokens.succeeded})"
            )
        if self.routing.routed_failure > self.tokens.failed:
            raise ValueError(
                "routing.routed_failure must be a subset of tokens.failed "
                f"(got routed_failure={self.routing.routed_failure}, failed={self.tokens.failed})"
            )
        if self.routing.quarantined > self.tokens.failed:
            raise ValueError(
                "routing.quarantined must be a subset of tokens.failed "
                f"(got quarantined={self.routing.quarantined}, failed={self.tokens.failed})"
            )
        if self.routing.discarded > self.tokens.failed:
            raise ValueError(
                f"routing.discarded must be a subset of tokens.failed (got discarded={self.routing.discarded}, failed={self.tokens.failed})"
            )
        if self.integrity.closure == "closed":
            if self.tokens.pending != 0:
                raise ValueError(f"closed accounting requires pending == 0, got {self.tokens.pending}")
            if self.integrity.missing_terminal_outcomes != 0:
                raise ValueError(
                    f"closed accounting requires missing_terminal_outcomes == 0, got {self.integrity.missing_terminal_outcomes}"
                )
            if self.integrity.duplicate_terminal_outcomes != 0:
                raise ValueError(
                    f"closed accounting requires duplicate_terminal_outcomes == 0, got {self.integrity.duplicate_terminal_outcomes}"
                )
        return self


def _require_terminal_run_fields(
    status: str,
    *,
    finished_at: datetime | None = None,
    finished_at_required: bool = False,
    error: str | None = None,
    landscape_run_id: str | None = None,
) -> None:
    """Enforce status-specific terminal invariants for Tier 1 run data."""
    if finished_at_required and status in RUN_STATUS_TERMINAL_VALUES and finished_at is None:
        raise ValueError(f"status={status!r} requires finished_at")
    # Phase 2.2 (elspeth-0de989c56d): operator-completion statuses
    # (the run reached engine completion and produced a Landscape audit
    # record) require a landscape_run_id.  `failed` and `cancelled` may not
    # have one — the engine can take the failed path on exceptions before
    # the Landscape audit row is created, and cancelled runs are signal-
    # bounded with similar timing.
    if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and not landscape_run_id:
        raise ValueError(f"status={status!r} requires landscape_run_id")
    if status == "failed" and not error:
        raise ValueError("status='failed' requires error")


def _check_status_accounting_invariant(status: str, accounting: RunAccounting | None) -> None:
    """Validate status taxonomy against explicit token accounting."""
    if status in {"running", "pending", "cancelled"}:
        return

    if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and accounting is None:
        raise ValueError(f"status={status!r} requires Landscape-derived accounting")

    if status == "completed":
        assert accounting is not None
        if accounting.integrity.closure != "closed":
            raise ValueError("status='completed' requires closed token accounting")
        if accounting.tokens.succeeded <= 0:
            raise ValueError("status='completed' requires tokens.succeeded > 0")
        if accounting.tokens.failed != 0:
            raise ValueError("status='completed' requires tokens.failed == 0")
        return

    if status == "completed_with_failures":
        assert accounting is not None
        if accounting.integrity.closure != "closed":
            raise ValueError("status='completed_with_failures' requires closed token accounting")
        if accounting.tokens.succeeded <= 0:
            raise ValueError("status='completed_with_failures' requires tokens.succeeded > 0")
        if accounting.tokens.failed <= 0:
            raise ValueError("status='completed_with_failures' requires tokens.failed > 0")
        return

    if status == "failed":
        return  # FAILED may come from token accounting or exception origin.

    if status == "empty":
        assert accounting is not None
        if accounting.integrity.closure != "closed":
            raise ValueError("status='empty' requires closed token accounting")
        if accounting.source.rows_processed != 0:
            raise ValueError(f"status='empty' requires accounting.source.rows_processed == 0, got {accounting.source.rows_processed}")
        if accounting.tokens.emitted != 0:
            raise ValueError(f"status='empty' requires accounting.tokens.emitted == 0, got {accounting.tokens.emitted}")
        return

    raise ValueError(f"Unknown status {status!r}")


class CompletedData(_StrictResponse):
    """Payload for ``completed`` events (terminal).

    The ``status`` field is the backend's authoritative classification of the
    run outcome — frontends MUST NOT re-derive it from accounting counters
    (that would duplicate the backend taxonomy invariant and create
    dual-source-of-truth drift).
    """

    status: Literal["completed", "completed_with_failures", "empty"]
    accounting: RunAccounting
    landscape_run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_status_consistency(self) -> Self:
        # Reuse the canonical accounting predicate at module scope so completed
        # events cannot drift from the run-status taxonomy.
        _check_status_accounting_invariant(self.status, self.accounting)
        return self


class CancelledData(_StrictResponse):
    """Payload for ``cancelled`` events with best-known progress counters.

    All six counter fields are REQUIRED with no defaults — same fabrication
    rationale as ``ProgressData``.  At cancellation time the engine carries
    every counter via the ``GracefulShutdownError`` (or ``RunResult``) that
    drove the cancellation; emission paths that pre-date pipeline start
    (early-cancel before any row has been seen) populate every field with
    a literal ``0``, which is then a documented producer assertion rather
    than a default-shimmed silence.  Cancelled is terminal; transient
    inconsistency is not a concern here.
    """

    status: Literal["cancelled"] = "cancelled"
    source_rows_processed: int = Field(ge=0)
    tokens_succeeded: int = Field(ge=0)
    tokens_failed: int = Field(ge=0)
    tokens_quarantined: int = Field(ge=0)
    tokens_routed_success: int = Field(ge=0)
    tokens_routed_failure: int = Field(ge=0)


class FailedData(_StrictResponse):
    """Payload for ``failed`` events (terminal)."""

    status: Literal["failed"] = "failed"
    detail: str = Field(min_length=1)
    node_id: str | None


class RunEvent(_StrictResponse):
    """WebSocket event payload for live progress streaming.

    ``data`` is a typed union keyed by ``event_type``.  The model_validator
    enforces the mapping — constructing a RunEvent with mismatched
    event_type/data types crashes immediately (offensive programming).
    """

    _event_sequence: int | None = PrivateAttr(default=None)

    run_id: str
    timestamp: datetime = Field(strict=False)
    # NOTE: Fast pipelines may produce identical timestamps.
    # Event ordering is guaranteed by the asyncio.Queue FIFO, not by timestamp.
    # Frontend must NOT sort by timestamp — use arrival order instead.
    event_type: RunEventType
    data: ProgressData | ErrorData | CompletedData | CancelledData | FailedData

    @property
    def event_sequence(self) -> int | None:
        """Internal durable replay cursor; never part of websocket JSON/schema."""
        return self._event_sequence

    def with_event_sequence(self, sequence: int) -> Self:
        if sequence < 1:
            raise ValueError(f"event sequence must be >= 1, got {sequence}")
        clone = self.model_copy()
        clone._event_sequence = sequence
        return clone

    @field_validator("timestamp", mode="before")
    @classmethod
    def _reject_epoch_timestamp(cls, v: object) -> object:
        """Reject Unix epoch integers while allowing ISO strings.

        ``Field(strict=False)`` lets Pydantic parse ISO strings back to
        datetime (needed for the WebSocket reconnect JSON round-trip).
        But lax mode also accepts ``int`` (Unix epoch) — which would
        hide a Tier 1 type error.  This before-validator fires first
        and rejects anything that isn't a ``datetime`` or ``str``.
        """
        if isinstance(v, (datetime, str)):
            return v
        raise ValueError(f"timestamp must be a datetime or ISO string, got {type(v).__name__}")

    _EVENT_TYPE_TO_DATA_TYPE: ClassVar[dict[str, type[_StrictResponse]]] = {
        "progress": ProgressData,
        "error": ErrorData,
        "completed": CompletedData,
        "cancelled": CancelledData,
        "failed": FailedData,
    }

    @model_validator(mode="before")
    @classmethod
    def _resolve_data_from_event_type(cls, values: Any) -> Any:
        """Pre-resolve the data union member during JSON deserialization.

        When deserializing from a dict (JSON round-trip), Pydantic's
        smart-union matching sees identical shapes for ProgressData and
        CancelledData and picks the first match. This pre-validator
        uses event_type to construct the correct model before union
        matching runs.
        """
        if isinstance(values, dict):
            event_type = values.get("event_type")
            data = values.get("data")
            if isinstance(data, dict) and event_type in cls._EVENT_TYPE_TO_DATA_TYPE:
                values = {**values, "data": cls._EVENT_TYPE_TO_DATA_TYPE[event_type](**data)}
        return values

    @model_validator(mode="after")
    def _enforce_data_type(self) -> Self:
        """Crash on event_type / data type mismatch.

        Belt-and-suspenders: the before-validator handles JSON round-trips;
        this after-validator catches programmer error when constructing
        RunEvent directly in Python with mismatched event_type/data.
        """
        expected = self._EVENT_TYPE_TO_DATA_TYPE[self.event_type]
        if not isinstance(self.data, expected):
            raise ValueError(f"event_type={self.event_type!r} requires {expected.__name__}, got {type(self.data).__name__}")
        return self


# Import-time sync guard: the mapping keys MUST match the event_type Literal.
# If a developer adds a new event type to the Literal but forgets the mapping
# (or vice versa), this assertion fires at module load — not at runtime when
# a user hits the mismatch.
_event_type_literal = get_args(RunEvent.model_fields["event_type"].annotation)
_mapping_keys = frozenset(RunEvent._EVENT_TYPE_TO_DATA_TYPE.keys())
_literal_values = frozenset(_event_type_literal)
if _mapping_keys != _literal_values:
    raise AssertionError(f"_EVENT_TYPE_TO_DATA_TYPE keys {_mapping_keys} != event_type Literal values {_literal_values}")
del _event_type_literal, _mapping_keys, _literal_values


class DiscardStageSummary(_StrictResponse):
    """Per-stage contribution to a virtual discard sink summary."""

    stage: Literal["source_validation", "transform_validation", "sink_discard"]
    node_id: str | None
    count: int = Field(ge=1)


class DiscardSummary(_StrictResponse):
    """Counts routed to the virtual ``discard`` sink.

    The backing records live in three audit surfaces:
    ``validation_errors.destination='discard'``,
    ``transform_errors.destination='discard'``, and terminal
    ``token_outcomes.sink_name='__discard__'`` rows for sink-write
    diversions.  ``total`` is stored explicitly in the response so clients
    can render the visible virtual sink without duplicating the arithmetic.
    ``stages`` carries the node/stage attribution needed for operator-facing
    copy; it is optional at construction time so older route-level tests that
    inject prebuilt summaries remain focused on their endpoint contracts.
    """

    total: int = Field(ge=0)
    validation_errors: int = Field(ge=0)
    transform_errors: int = Field(ge=0)
    sink_discards: int = Field(ge=0)
    stages: tuple[DiscardStageSummary, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _check_total(self) -> Self:
        expected = self.validation_errors + self.transform_errors + self.sink_discards
        if self.total != expected:
            raise ValueError(
                f"Discard summary total mismatch: total={self.total} "
                f"!= validation_errors({self.validation_errors}) "
                f"+ transform_errors({self.transform_errors}) "
                f"+ sink_discards({self.sink_discards}) = {expected}"
            )
        if self.stages:
            stage_totals = {
                "source_validation": 0,
                "transform_validation": 0,
                "sink_discard": 0,
            }
            for stage in self.stages:
                stage_totals[stage.stage] += stage.count
            if stage_totals["source_validation"] != self.validation_errors:
                raise ValueError(
                    "Discard source_validation stage count mismatch: "
                    f"{stage_totals['source_validation']} != validation_errors({self.validation_errors})"
                )
            if stage_totals["transform_validation"] != self.transform_errors:
                raise ValueError(
                    "Discard transform_validation stage count mismatch: "
                    f"{stage_totals['transform_validation']} != transform_errors({self.transform_errors})"
                )
            if stage_totals["sink_discard"] != self.sink_discards:
                raise ValueError(
                    f"Discard sink_discard stage count mismatch: {stage_totals['sink_discard']} != sink_discards({self.sink_discards})"
                )
        return self


type RunDiagnosticNodeStateStatus = Literal["open", "pending", "completed", "failed"]
type RunDiagnosticTerminalOutcome = Literal["success", "failure", "transient"]
RunDiagnosticOperationType = OperationType
type RunDiagnosticOperationStatus = Literal["open", "completed", "failed", "pending"]
type RunDiagnosticDurationMs = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type RunDiagnosticCount = Annotated[int, Field(ge=0)]


def _type_alias_literal_values(alias: TypeAliasType) -> frozenset[str]:
    return frozenset(get_args(alias.__value__))


RUN_DIAGNOSTIC_NODE_STATE_STATUS_VALUES: frozenset[str] = _type_alias_literal_values(RunDiagnosticNodeStateStatus)
RUN_DIAGNOSTIC_TERMINAL_OUTCOME_VALUES: frozenset[str] = _type_alias_literal_values(RunDiagnosticTerminalOutcome)
RUN_DIAGNOSTIC_OPERATION_TYPE_VALUES: frozenset[str] = frozenset(OPERATION_TYPE_VALUES)
RUN_DIAGNOSTIC_OPERATION_STATUS_VALUES: frozenset[str] = _type_alias_literal_values(RunDiagnosticOperationStatus)

if frozenset(status.value for status in NodeStateStatus) != RUN_DIAGNOSTIC_NODE_STATE_STATUS_VALUES:
    raise AssertionError(
        f"RunDiagnosticNodeState.status Literal values must mirror NodeStateStatus values; got {RUN_DIAGNOSTIC_NODE_STATE_STATUS_VALUES}"
    )
if frozenset(outcome.value for outcome in TerminalOutcome) != RUN_DIAGNOSTIC_TERMINAL_OUTCOME_VALUES:
    raise AssertionError(
        "RunDiagnosticToken.terminal_outcome Literal values must mirror "
        f"TerminalOutcome values; got {RUN_DIAGNOSTIC_TERMINAL_OUTCOME_VALUES}"
    )
if frozenset(OPERATION_TYPE_VALUES) != RUN_DIAGNOSTIC_OPERATION_TYPE_VALUES:
    raise AssertionError(
        "RunDiagnosticOperation.operation_type Literal values must mirror "
        f"OPERATION_TYPE_VALUES; got {RUN_DIAGNOSTIC_OPERATION_TYPE_VALUES}"
    )
if RUN_DIAGNOSTIC_OPERATION_STATUS_VALUES != Operation._ALLOWED_STATUSES:
    raise AssertionError(
        "RunDiagnosticOperation.status Literal values must mirror "
        f"Operation._ALLOWED_STATUSES; got {RUN_DIAGNOSTIC_OPERATION_STATUS_VALUES}"
    )


class RunDiagnosticNodeState(_StrictResponse):
    """Bounded node-state projection for run diagnostics.

    Deliberately excludes context_before/context_after payloads. The web
    diagnostics surface is for progress visibility, not raw row payload export.
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int = Field(ge=0)
    attempt: int = Field(ge=0)
    status: RunDiagnosticNodeStateStatus
    duration_ms: RunDiagnosticDurationMs | None
    started_at: datetime
    completed_at: datetime | None
    error: Any | None = None
    success_reason: Any | None = None


class RunDiagnosticToken(_StrictResponse):
    """One token in the bounded diagnostics preview."""

    token_id: str
    row_id: str
    row_index: int | None = Field(ge=0)
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None
    step_in_pipeline: int | None = Field(ge=0)
    created_at: datetime
    terminal_outcome: RunDiagnosticTerminalOutcome | None
    states: list[RunDiagnosticNodeState]


class RunDiagnosticOperation(_StrictResponse):
    """Source/sink operation projection for run diagnostics."""

    operation_id: str
    node_id: str
    operation_type: RunDiagnosticOperationType
    sink_effect_id: str | None
    status: RunDiagnosticOperationStatus
    duration_ms: RunDiagnosticDurationMs | None
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None


class RunDiagnosticArtifact(_StrictResponse):
    """Saved artifact projection for run diagnostics."""

    artifact_id: str
    sink_node_id: str
    producer_kind: Literal["node_state", "sink_effect"]
    produced_by_state_id: str | None
    sink_effect_id: str | None
    artifact_type: str
    path_or_uri: str
    size_bytes: int = Field(ge=0)
    publication_performed: bool
    publication_evidence_kind: Literal["returned", "reconciled", "inherited", "virtual", "legacy_returned"]
    created_at: datetime


RunOutputArtifactStorageKind = Literal["blob", "payload", "sink_file", "unknown"]


class RunOutputArtifact(_StrictResponse):
    """One sink-write artefact in the run's full outputs manifest.

    Distinct from ``RunDiagnosticArtifact`` (UI preview, capped at 20):
    this is the audit-evidence retrieval shape — full per-run list, with
    ``content_hash`` and ``exists_now`` fields suited to backfill /
    re-fetch flows. Returned by ``/api/runs/{rid}/outputs``.

    ``downloadable`` is a server-computed convenience: true iff the
    ``/outputs/{artifact_id}/content`` endpoint would actually serve
    bytes for this row. False when the artifact is not file-backed,
    when the file is gone, or when the path resolves outside the
    sink-allowlist. Lets the UI suppress download buttons that would
    otherwise 4xx on click.

    ``storage_kind`` is a structured discriminator classifying
    ``path_or_uri`` against elspeth's real filesystem storage layouts
    (see ``_classify_storage_kind`` in ``web/execution/outputs.py``):

    * ``"blob"``      — the composer blob store, ``{data_dir}/blobs/...``
    * ``"payload"``   — the content-addressed payload store,
                        ``{data_dir}/payloads/...``
    * ``"sink_file"`` — the canonical sink output directory,
                        ``{data_dir}/outputs/...``
    * ``"unknown"``   — anything else (a user-configured path outside
                        those layouts, an object-store URI, or a legacy
                        caller that omitted ``data_dir``)

    "blob" and "payload" are elspeth's own opaque internal storage —
    the UI must never render their basename or full path as the row
    title; it renders a friendly label instead. Independent of
    ``downloadable``/``exists_now``: classification does not require the
    file to exist on disk (a purged blob-store path still reports
    ``storage_kind="blob"``).
    """

    artifact_id: str
    sink_node_id: str
    producer_kind: Literal["node_state", "sink_effect"]
    produced_by_state_id: str | None
    sink_effect_id: str | None
    artifact_type: str
    path_or_uri: str
    content_hash: str
    size_bytes: int = Field(ge=0)
    publication_performed: bool
    publication_evidence_kind: Literal["returned", "reconciled", "inherited", "virtual", "legacy_returned"]
    created_at: datetime
    exists_now: bool
    downloadable: bool
    storage_kind: RunOutputArtifactStorageKind


PreviewContentType = Literal["text", "csv", "jsonl", "json", "binary"]


class RunOutputArtifactPreview(_StrictResponse):
    """Bounded preview of one sink-write artefact for in-UI inspection.

    Returned by ``GET /api/runs/{rid}/outputs/{aid}/preview``. Intended
    as a head-of-file render so an operator can decide whether to pull
    the full file via the ``/content`` endpoint. Bounded to the lesser
    of 256 KiB or 100 rows; ``truncated`` indicates the cap was hit.

    ``content_type`` is the renderer hint:
    * ``csv`` / ``jsonl`` — UI may render as a parsed table.
    * ``json`` — UI may pretty-print.
    * ``text`` — UI renders as monospace pre-formatted block.
    * ``binary`` — bytes are not text (or extension is unknown);
      ``preview_text`` is empty and the UI suggests downloading.
    """

    artifact_id: str
    content_type: PreviewContentType
    preview_text: str
    truncated: bool
    total_size_bytes: int = Field(ge=0)
    row_count_preview: int | None = Field(default=None, ge=0)


class RunOutputsResponse(_StrictResponse):
    """REST response for ``GET /api/runs/{rid}/outputs`` — full audit-evidence
    manifest of every sink-write artefact this run produced.

    Unlike the diagnostics ``artifacts`` field (capped for operator-UI
    pacing), this list is unbounded — sized to the actual write count.
    """

    run_id: str
    landscape_run_id: str
    artifacts: list[RunOutputArtifact]


class RunDiagnosticSummary(_StrictResponse):
    """Aggregate counts for a run diagnostics snapshot."""

    token_count: int = Field(ge=0)
    preview_limit: int = Field(ge=1, le=100)
    preview_truncated: bool
    state_counts: dict[RunDiagnosticNodeStateStatus, RunDiagnosticCount]
    operation_counts: dict[RunDiagnosticOperationType, RunDiagnosticCount]
    latest_activity_at: datetime | None


class RunDiagnosticFailureDetail(_StrictResponse):
    """Focused pointer to the operation that caused a run to fail.

    A run with hundreds of successful operations and one failure can hide the
    cause in the (paged, limited) operations list. This model surfaces the
    *latest* failed operation directly so the UI can render "what went wrong"
    without scanning. None on the response when no failed operation exists.

    ``error_message`` is the chain text persisted to ``operations.error_message``
    in Landscape — it carries the wrapper error plus its cause(s) including any
    truncated HTTP response body the provider returned. The full response body,
    if relevant, lives in the audit DB under ``calls.response_ref``.
    """

    operation_id: str
    node_id: str
    operation_type: RunDiagnosticOperationType
    error_message: str = Field(min_length=1)
    failed_at: datetime


class RunDiagnosticsResponse(_StrictResponse):
    """REST response for run diagnostics."""

    run_id: str
    landscape_run_id: str
    run_status: SessionRunStatus
    cancel_requested: bool = False
    summary: RunDiagnosticSummary
    tokens: list[RunDiagnosticToken]
    operations: list[RunDiagnosticOperation]
    artifacts: list[RunDiagnosticArtifact]
    failure_detail: RunDiagnosticFailureDetail | None = None


class RunDiagnosticsWorkingView(_StrictResponse):
    """Operator-facing read of visible run evidence.

    This is not model chain-of-thought. It is a bounded, UI-safe summary
    of what the diagnostics snapshot visibly shows and what that likely
    means for the running pipeline.
    """

    headline: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    meaning: str = Field(min_length=1)
    next_steps: list[str] = Field(default_factory=list)


class RunDiagnosticsEvaluationResponse(_StrictResponse):
    """LLM-generated explanation of a diagnostics snapshot."""

    run_id: str
    generated_at: datetime
    explanation: str = Field(min_length=1)
    working_view: RunDiagnosticsWorkingView


class RunStatusResponse(_StrictResponse):
    """REST response for run status queries."""

    run_id: str
    status: SessionRunStatus
    started_at: datetime | None
    finished_at: datetime | None
    accounting: RunAccounting | None = None
    error: str | None
    landscape_run_id: str | None
    discard_summary: DiscardSummary | None = None
    cancel_requested: bool = False

    @model_validator(mode="after")
    def _check_status_contract(self) -> Self:
        """Enforce terminal field requirements and token-accounting taxonomy."""
        if self.cancel_requested and self.status in RUN_STATUS_TERMINAL_VALUES:
            raise ValueError(f"status={self.status!r} cannot also be cancel_requested")
        if self.status in RUN_STATUS_TERMINAL_VALUES:
            _require_terminal_run_fields(
                self.status,
                finished_at=self.finished_at,
                finished_at_required=True,
                error=self.error,
                landscape_run_id=self.landscape_run_id,
            )
        _check_status_accounting_invariant(self.status, self.accounting)
        return self


class RunResultsResponse(_StrictResponse):
    """REST response for terminal run results."""

    run_id: str
    status: TerminalSessionRunStatus
    accounting: RunAccounting | None = None
    landscape_run_id: str | None
    error: str | None
    discard_summary: DiscardSummary | None = None

    @model_validator(mode="after")
    def _check_status_contract(self) -> Self:
        _require_terminal_run_fields(
            self.status,
            error=self.error,
            landscape_run_id=self.landscape_run_id,
        )
        _check_status_accounting_invariant(self.status, self.accounting)
        return self


# ── Status set derivation (Literal → frozenset) ────────────────────────
#
# The REST /results endpoint must reject non-terminal runs with 409.  The
# set of non-terminal statuses is derived from the difference between
# RunStatusResponse.status (all) and RunResultsResponse.status (terminal)
# so that adding a new non-terminal Literal value (e.g., "paused") to
# RunStatusResponse automatically propagates to the route guard.
#
# Import-time assertion catches the reverse drift (a Literal value in
# RunResultsResponse.status that is NOT in RunStatusResponse.status) —
# without this, a typo in one Literal would produce a runtime set diff
# with surprising contents.

RUN_STATUS_ALL_VALUES: frozenset[str] = frozenset(get_args(RunStatusResponse.model_fields["status"].annotation))
RUN_STATUS_TERMINAL_VALUES: frozenset[str] = frozenset(get_args(RunResultsResponse.model_fields["status"].annotation))
RUN_STATUS_NON_TERMINAL_VALUES: frozenset[str] = RUN_STATUS_ALL_VALUES - RUN_STATUS_TERMINAL_VALUES

if not RUN_STATUS_TERMINAL_VALUES.issubset(RUN_STATUS_ALL_VALUES):
    raise AssertionError(
        f"RunResultsResponse.status terminal values {RUN_STATUS_TERMINAL_VALUES} "
        f"must be a subset of RunStatusResponse.status values {RUN_STATUS_ALL_VALUES}"
    )
if not RUN_STATUS_NON_TERMINAL_VALUES:
    raise AssertionError(
        "RunStatusResponse must declare at least one non-terminal status "
        "not present in RunResultsResponse — otherwise the /results 409 "
        "guard is dead code"
    )
