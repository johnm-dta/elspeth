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

from elspeth.web.sessions.protocol import SessionRunStatus, TerminalSessionRunStatus


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
    """Payload for ``progress`` events (non-terminal, streaming)."""

    rows_processed: int = Field(ge=0)
    rows_failed: int = Field(ge=0)


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
    rows_routed: int,
    rows_quarantined: int,
) -> None:
    """Enforce rows_processed >= succeeded + failed + routed + quarantined.

    NARROW INVARIANT (elspeth-31d53c7493). The original equality
    formulation (``processed == succeeded + failed + routed + quarantined``)
    does not hold for any DAG with aggregation, fork, expansion, or
    coalesce — source rows reach terminal states the formula does not
    account for. A ``batch_stats`` aggregation pipeline absorbs source
    rows into ``CONSUMED_IN_BATCH`` (no counter incremented in
    ``engine/orchestrator/outcomes.py``), so a legitimate run reports
    ``rows_processed > sum``.

    The relaxed inequality preserves the half of the invariant we can
    still defend at this layer: the orchestrator must never emit MORE
    terminal-state counts than input rows ingested.  Over-counting is a
    bookkeeping bug (e.g., a token's outcome recorded twice) and remains
    a Tier 1 anomaly worth crashing on.

    The architecturally-correct formula — extending the schema to carry
    rows_consumed_in_batch / rows_forked / rows_coalesced /
    rows_expanded counters and re-deriving the orchestrator-side balance
    equation across parent/child token relationships — is tracked in
    elspeth-cf84eb1b52 (P0, blocks RC 5.0).  When that lands, this
    inequality is replaced by the full balance.

    Note: even this inequality does NOT hold for fork/coalesce pipelines
    (a single input can produce multiple terminal child counts).  Those
    shapes are not currently exposed via the composer, so the practical
    surface for this narrow patch is aggregation pipelines.
    """
    sum_terminal = rows_succeeded + rows_failed + rows_routed + rows_quarantined
    if rows_processed < sum_terminal:
        raise ValueError(
            f"Row count decomposition mismatch (over-counting): rows_processed={rows_processed} "
            f"< rows_succeeded({rows_succeeded}) + rows_failed({rows_failed}) "
            f"+ rows_routed({rows_routed}) + rows_quarantined({rows_quarantined}) = {sum_terminal}. "
            f"Tier 1 anomaly: orchestrator emitted more terminal-state counts than input rows. "
            f"See elspeth-cf84eb1b52 for the full DAG-aware balance equation."
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
    if status == "completed" and not landscape_run_id:
        raise ValueError("status='completed' requires landscape_run_id")
    if status == "failed" and not error:
        raise ValueError("status='failed' requires error")


class CompletedData(_StrictResponse):
    """Payload for ``completed`` events (terminal)."""

    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed: int = Field(default=0, ge=0)
    rows_quarantined: int = Field(ge=0)
    landscape_run_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_row_decomposition(self) -> Self:
        _validate_row_decomposition(
            self.rows_processed,
            self.rows_succeeded,
            self.rows_failed,
            self.rows_routed,
            self.rows_quarantined,
        )
        return self


class CancelledData(_StrictResponse):
    """Payload for ``cancelled`` events (terminal)."""

    rows_processed: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed: int = Field(default=0, ge=0)


class FailedData(_StrictResponse):
    """Payload for ``failed`` events (terminal)."""

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
    rows_routed: int = Field(default=0, ge=0)
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
        """
        if self.status in RUN_STATUS_TERMINAL_VALUES:
            _validate_row_decomposition(
                self.rows_processed,
                self.rows_succeeded,
                self.rows_failed,
                self.rows_routed,
                self.rows_quarantined,
            )
            _require_terminal_run_fields(
                self.status,
                finished_at=self.finished_at,
                finished_at_required=True,
                error=self.error,
                landscape_run_id=self.landscape_run_id,
            )
        return self


class RunResultsResponse(_StrictResponse):
    """REST response for terminal run results."""

    run_id: str
    status: TerminalSessionRunStatus
    rows_processed: int = Field(ge=0)
    rows_succeeded: int = Field(ge=0)
    rows_failed: int = Field(ge=0)
    rows_routed: int = Field(default=0, ge=0)
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
            self.rows_routed,
            self.rows_quarantined,
        )
        _require_terminal_run_fields(
            self.status,
            error=self.error,
            landscape_run_id=self.landscape_run_id,
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
