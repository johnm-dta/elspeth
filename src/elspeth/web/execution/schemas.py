"""Pydantic response models for execution endpoints.

All models in this module serialize **system-owned data** (Tier 1 in the
Data Manifesto).  They use strict validation and forbid extra fields so
that internal type drift crashes loudly instead of silently coercing
values or dropping unknown fields.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from elspeth.web.sessions.protocol import (
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SessionRunStatus,
    TerminalSessionRunStatus,
)


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

    name: str
    passed: bool
    detail: str


class ValidationError(_StrictResponse):
    """Error with per-component attribution."""

    component_id: str | None
    component_type: str | None
    message: str
    suggestion: str | None


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


class ValidationResult(_StrictResponse):
    """Result of dry-run validation against real engine code."""

    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[ValidationError]
    semantic_contracts: list[SemanticEdgeContractResponse] = []


# ── Typed event payload models ──────────────────────────────────────────
#
# Each event_type has a dedicated payload model so that the server-side
# schema catches producer drift between service.py, routes.py, and the
# frontend TypeScript types.


class ProgressData(_StrictResponse):
    """Payload for ``progress`` events (non-terminal, streaming).

    elspeth-5069612f3c — ``rows_routed`` is split into MOVE (intentional gate
    ``route_to_sink``) and DIVERT (transform ``on_error`` reroute). Both
    fields are present on the wire so the streaming progress payload mirrors
    ``CompletedData`` / ``CancelledData`` / TS ``RunEventProgress`` shape
    exactly.

    All six counter fields are REQUIRED with no defaults.  The engine's
    ``ProgressEvent`` (contracts/cli.py) always carries real counter values
    at emission time; defaulting any of them to ``0`` on the wire would
    fabricate "we don't know" as "definitely zero" — violating the CLAUDE.md
    fabrication test.  Mid-run, an operator must be able to distinguish
    "no rows have succeeded yet" from "the field was never populated".

    The sum-invariant ``rows_succeeded + rows_failed <= rows_processed`` is
    intentionally NOT enforced here.  Non-terminal counts are allowed transient
    inconsistency while the orchestrator is mid-flight: a row may be marked
    processed before being categorised into one of the lifecycle terminal-state
    buckets.  Naming the unenforced sum here keeps the relaxation discoverable
    rather than implicit.  See
    ``RunStatusResponse._check_row_decomposition`` for the matching
    rationale on non-terminal status responses.
    """

    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_quarantined: int = Field(ge=0)
    rows_routed_success: int = Field(ge=0)
    rows_routed_failure: int = Field(ge=0)


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


def _validate_row_decomposition(
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
) -> None:
    """Enforce rows_processed >= succeeded + failed.

    ADR-019 makes rows_routed_* and rows_quarantined reporting subsets of the
    exhaustive lifecycle counters. They must not be added into this sum.

    NARROW INVARIANT (elspeth-31d53c7493 carry-forward). The original equality
    formulation does not hold for any DAG with aggregation, fork, expansion,
    or coalesce — source rows reach terminal states the formula does not
    account for. The relaxed inequality is preserved here.

    The architecturally-correct formula (full DAG-aware balance) is tracked
    in elspeth-cf84eb1b52. When that lands, this inequality is replaced by
    the full balance.
    """
    sum_terminal = rows_succeeded + rows_failed
    if rows_processed < sum_terminal:
        raise ValueError(
            f"Row count decomposition mismatch (over-counting): rows_processed={rows_processed} "
            f"< rows_succeeded({rows_succeeded}) + rows_failed({rows_failed}) "
            f"= {sum_terminal}. "
            f"Tier 1 anomaly: orchestrator emitted more terminal-state counts than input rows. "
            f"See elspeth-cf84eb1b52 for the full DAG-aware balance equation."
        )


def _validate_response_counter_subsets(
    *,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """ADR-019: response structural counters are subsets of base counters."""
    if rows_routed_success > rows_succeeded:
        raise ValueError(
            f"rows_routed_success must be a subset of rows_succeeded "
            f"(got rows_routed_success={rows_routed_success}, rows_succeeded={rows_succeeded})"
        )
    if rows_routed_failure > rows_failed:
        raise ValueError(
            f"rows_routed_failure must be a subset of rows_failed "
            f"(got rows_routed_failure={rows_routed_failure}, rows_failed={rows_failed})"
        )
    if rows_quarantined > rows_failed:
        raise ValueError(
            f"rows_quarantined must be a subset of rows_failed (got rows_quarantined={rows_quarantined}, rows_failed={rows_failed})"
        )


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


def _check_status_row_count_invariant(
    status: str,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """Pydantic mirror of the ADR-019 L0 status predicate.

    success_indicator = rows_succeeded > 0
    failure_indicator = rows_failed > 0

    The Pydantic mirror does NOT see rows_coalesce_failed (the API schema
    does not surface it) — see the original docstring for the rationale;
    every coalesce failure also increments rows_failed at the engine layer,
    so the failure indicator is preserved.

    Non-terminal (running / pending) and signal-bounded (cancelled) statuses
    bypass the predicate.
    """
    success_indicator = rows_succeeded > 0
    failure_indicator = rows_failed > 0

    if status in {"running", "pending", "cancelled"}:
        return

    if status == "completed":
        if not success_indicator:
            raise ValueError(
                f"status='completed' requires a success indicator "
                f"(rows_succeeded > 0); "
                f"got rows_succeeded={rows_succeeded} "
                f"(use status='empty' for ingested-zero-rows runs, "
                f"'failed' when rows were processed but none reached a success path)"
            )
        if failure_indicator:
            raise ValueError(f"status='completed' requires no failures (rows_failed={rows_failed}); use status='completed_with_failures'")
        return

    if status == "completed_with_failures":
        if not success_indicator:
            raise ValueError(
                f"status='completed_with_failures' requires a success indicator "
                f"(rows_succeeded > 0); "
                f"got rows_succeeded={rows_succeeded} "
                f"(use status='failed' when no row reached a success path)"
            )
        if not failure_indicator:
            raise ValueError(
                f"status='completed_with_failures' requires at least one failure indicator "
                f"(rows_failed > 0); got rows_failed={rows_failed} "
                f"(use status='completed' for clean runs)"
            )
        return

    if status == "failed":
        return  # FAILED tolerates any shape (predicate-or-exception origin)

    if status == "empty":
        if rows_processed != 0:
            raise ValueError(f"status='empty' requires rows_processed == 0, got {rows_processed}")
        if success_indicator:
            raise ValueError(f"status='empty' requires no success indicator (rows_succeeded={rows_succeeded})")
        if failure_indicator:
            raise ValueError(
                f"status='empty' requires no failures (rows_failed={rows_failed}); "
                f"use status='failed' when the run encountered failures with no successful rows"
            )
        return

    raise ValueError(f"Unknown status {status!r}")


class CompletedData(_StrictResponse):
    """Payload for ``completed`` events (terminal).

    The ``status`` field is the backend's authoritative classification of the
    run outcome — frontends MUST NOT re-derive it from row counts (that would
    duplicate the ``failure_indicator``/``success_indicator`` invariant in
    ``_validate_row_decomposition`` and create dual-source-of-truth drift).
    """

    status: Literal["completed", "completed_with_failures", "empty"]
    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed_success: int = Field(default=0, ge=0)
    rows_routed_failure: int = Field(default=0, ge=0)
    rows_quarantined: int = Field(ge=0)
    landscape_run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_row_decomposition(self) -> Self:
        _validate_row_decomposition(
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
        )
        _validate_response_counter_subsets(
            rows_succeeded=self.rows_succeeded,
            rows_failed=self.rows_failed,
            rows_routed_success=self.rows_routed_success,
            rows_routed_failure=self.rows_routed_failure,
            rows_quarantined=self.rows_quarantined,
        )
        return self

    @model_validator(mode="after")
    def _check_status_consistency(self) -> Self:
        # Mirror the L0 success_indicator/failure_indicator biconditional on
        # the wire so the SSE event cannot drift from the run-status taxonomy.
        # Reuses the canonical predicate at module scope — one source of truth
        # for the classification rule.
        _check_status_row_count_invariant(
            self.status,
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
            self.rows_routed_success,
            self.rows_routed_failure,
            self.rows_quarantined,
        )
        return self


class CancelledData(_StrictResponse):
    """Payload for ``cancelled`` events (terminal).

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
    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_quarantined: int = Field(ge=0)
    rows_routed_success: int = Field(ge=0)
    rows_routed_failure: int = Field(ge=0)


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

    run_id: str
    timestamp: datetime = Field(strict=False)
    # NOTE: Fast pipelines may produce identical timestamps.
    # Event ordering is guaranteed by the asyncio.Queue FIFO, not by timestamp.
    # Frontend must NOT sort by timestamp — use arrival order instead.
    event_type: Literal["progress", "error", "completed", "cancelled", "failed"]
    data: ProgressData | ErrorData | CompletedData | CancelledData | FailedData

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


class DiscardSummary(_StrictResponse):
    """Counts routed to the virtual ``discard`` sink.

    The backing records live in three audit surfaces:
    ``validation_errors.destination='discard'``,
    ``transform_errors.destination='discard'``, and terminal
    ``token_outcomes.sink_name='__discard__'`` rows for sink-write
    diversions.  ``total`` is stored explicitly in the response so clients
    can render the visible virtual sink without duplicating the arithmetic.
    """

    total: int = Field(ge=0)
    validation_errors: int = Field(ge=0)
    transform_errors: int = Field(ge=0)
    sink_discards: int = Field(ge=0)

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
        return self


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
    status: str
    duration_ms: float | None
    started_at: datetime
    completed_at: datetime | None
    error: Any | None = None
    success_reason: Any | None = None


class RunDiagnosticToken(_StrictResponse):
    """One token in the bounded diagnostics preview."""

    token_id: str
    row_id: str
    row_index: int | None
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None
    step_in_pipeline: int | None
    created_at: datetime
    terminal_outcome: str | None
    states: list[RunDiagnosticNodeState]


class RunDiagnosticOperation(_StrictResponse):
    """Source/sink operation projection for run diagnostics."""

    operation_id: str
    node_id: str
    operation_type: str
    status: str
    duration_ms: float | None
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None


class RunDiagnosticArtifact(_StrictResponse):
    """Saved artifact projection for run diagnostics."""

    artifact_id: str
    sink_node_id: str
    artifact_type: str
    path_or_uri: str
    size_bytes: int = Field(ge=0)
    created_at: datetime


class RunDiagnosticSummary(_StrictResponse):
    """Aggregate counts for a run diagnostics snapshot."""

    token_count: int = Field(ge=0)
    preview_limit: int = Field(ge=1, le=100)
    preview_truncated: bool
    state_counts: dict[str, int]
    operation_counts: dict[str, int]
    latest_activity_at: datetime | None


class RunDiagnosticsResponse(_StrictResponse):
    """REST response for run diagnostics."""

    run_id: str
    landscape_run_id: str
    run_status: SessionRunStatus
    summary: RunDiagnosticSummary
    tokens: list[RunDiagnosticToken]
    operations: list[RunDiagnosticOperation]
    artifacts: list[RunDiagnosticArtifact]


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
    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed_success: int = Field(default=0, ge=0)
    rows_routed_failure: int = Field(default=0, ge=0)
    rows_quarantined: int = Field(ge=0)
    error: str | None
    landscape_run_id: str | None
    discard_summary: DiscardSummary | None = None

    @model_validator(mode="after")
    def _check_row_decomposition(self) -> Self:
        """Enforce decomposition invariant ONLY on terminal states.

        Non-terminal states (pending/running) may have transiently
        inconsistent counts while the orchestrator is mid-flight
        (a row may be marked processed before being categorised).
        Terminal states must satisfy the narrow invariant in
        ``_validate_row_decomposition`` (rows_processed >= sum of
        terminal-state counts) — over-counting is a Tier 1 anomaly.
        Full DAG-aware balance equation is tracked in elspeth-cf84eb1b52
        (P0, blocks RC 5.0).

        Phase 2.2 (elspeth-0de989c56d) also gates the four-value status
        taxonomy via ``_check_status_row_count_invariant`` so the API
        cannot serialize a status string that doesn't match the
        row-count shape.  The two invariants are orthogonal — one is
        about count consistency, the other about status accuracy.
        """
        if self.status in RUN_STATUS_TERMINAL_VALUES:
            _validate_row_decomposition(
                self.rows_processed,
                self.rows_succeeded,
                self.rows_failed,
            )
            _validate_response_counter_subsets(
                rows_succeeded=self.rows_succeeded,
                rows_failed=self.rows_failed,
                rows_routed_success=self.rows_routed_success,
                rows_routed_failure=self.rows_routed_failure,
                rows_quarantined=self.rows_quarantined,
            )
            _require_terminal_run_fields(
                self.status,
                finished_at=self.finished_at,
                finished_at_required=True,
                error=self.error,
                landscape_run_id=self.landscape_run_id,
            )
        _check_status_row_count_invariant(
            self.status,
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
            self.rows_routed_success,
            self.rows_routed_failure,
            self.rows_quarantined,
        )
        return self


class RunResultsResponse(_StrictResponse):
    """REST response for terminal run results."""

    run_id: str
    status: TerminalSessionRunStatus
    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed_success: int = Field(default=0, ge=0)
    rows_routed_failure: int = Field(default=0, ge=0)
    rows_quarantined: int = Field(ge=0)
    landscape_run_id: str | None
    error: str | None
    discard_summary: DiscardSummary | None = None

    @model_validator(mode="after")
    def _check_row_decomposition(self) -> Self:
        _validate_row_decomposition(
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
        )
        _validate_response_counter_subsets(
            rows_succeeded=self.rows_succeeded,
            rows_failed=self.rows_failed,
            rows_routed_success=self.rows_routed_success,
            rows_routed_failure=self.rows_routed_failure,
            rows_quarantined=self.rows_quarantined,
        )
        _require_terminal_run_fields(
            self.status,
            error=self.error,
            landscape_run_id=self.landscape_run_id,
        )
        # Phase 2.2 (elspeth-0de989c56d): status / row-count biconditional.
        _check_status_row_count_invariant(
            self.status,
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
            self.rows_routed_success,
            self.rows_routed_failure,
            self.rows_quarantined,
        )
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
