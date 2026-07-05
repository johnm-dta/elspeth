"""Pydantic request/response models for all session API endpoints.

Response models in this module serialize **system-owned data** (Tier 1 in
the Data Manifesto).  They inherit from ``_StrictResponse`` so that
coercion and unknown fields crash rather than silently passing through —
the Landscape record and the HTTP response must agree exactly.

Request models keep normal ``BaseModel`` coercion semantics: client input
is Tier 3 and the boundary-layer coercion rules (documented in
``tier-model-deep-dive``) apply.  They still reject unknown keys
mechanically so stale or typoed client payloads fail closed at the HTTP
boundary instead of being silently reinterpreted by the route layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import pydantic
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from elspeth.contracts.composer_interpretation import InterpretationChoice, InterpretationKind, InterpretationSource
from elspeth.web.execution.schemas import DiscardSummary, RunAccounting
from elspeth.web.sessions.protocol import (
    ComposerDensityDefault,
    ComposerTrustMode,
    ProposalEventType,
    ProposalLifecycleStatus,
    SessionRunStatus,
)
from elspeth.web.validation import (
    _validate_accepted_value_content,
    has_visible_content,
)


class _StrictResponse(BaseModel):
    """Base model for session response schemas — Tier 1 trust rules.

    ``strict=True`` rejects silent coercion (``"7"`` into an ``int`` field
    crashes instead of becoming ``7``).  ``extra="forbid"`` rejects
    unknown fields instead of dropping them.  Both are required for the
    audit-trail integrity contract: the HTTP response must not contain
    values the backend never emitted, and must not silently hide values
    the backend did emit.
    """

    model_config = ConfigDict(strict=True, extra="forbid")


class _RequestModel(BaseModel):
    """Tier 3 request base: allow coercion, reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


def _require_visible_content(value: str, *, field_label: str) -> str:
    """Reject strings that contain no visible characters."""
    if not has_visible_content(value):
        raise ValueError(f"{field_label} must contain at least one visible character")
    return value


class CreateSessionRequest(_RequestModel):
    """Request body for POST /api/sessions.

    ``title`` is optional: when omitted (or explicitly null) the route
    mints the app-wide default ("Session — 2 Jul 2026", auto-disambiguated
    per user — see ``elspeth.web.sessions.titles``). The default is minted
    server-side so every client shares one naming convention
    (elspeth-ef8c18a6cb killed the frontend's competing "New session" /
    "Untitled" defaults).
    """

    title: str | None = pydantic.Field(default=None, min_length=1)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_visible_content(value, field_label="Session title")


class UpdateSessionRequest(_RequestModel):
    """Request body for PATCH /api/sessions/{id}."""

    title: str = pydantic.Field(min_length=1)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        return _require_visible_content(value, field_label="Session title")


class SessionResponse(_StrictResponse):
    """Response for session CRUD operations."""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    archived: bool = False
    forked_from_session_id: str | None = None
    forked_from_message_id: str | None = None


class SendMessageRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/messages."""

    # max_length=65536 (64 KiB) caps message content at the schema boundary.
    # Phase 5b.0.5 (F-3): defense against unbounded payload allocation
    # before interpretation-event code paths can be exercised.  64 KiB
    # accommodates multi-paragraph user messages and long paste content
    # while preventing trivial-cost large-string attacks.  Mirrors the
    # _InlineBlobModel.content 256 KiB cap (web/composer/redaction.py).
    content: str = pydantic.Field(min_length=1, max_length=65536)
    state_id: UUID | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        return _require_visible_content(value, field_label="Message content")


type ToolCallObject = dict[str, JsonValue]
type ToolCallList = list[ToolCallObject]


class ChatMessageResponse(_StrictResponse):
    """Response for a single chat message.

    ``raw_content`` is the model's pre-synthesis prose for assistant turns
    where ``service._finalize_no_tool_response`` augmented the visible
    ``content`` with an operator-facing suffix or replaced it with a
    synthetic blocker message. It is ``null`` in the response by default
    (Tier 1 audit data; the conversation channel does not need it) and
    carries the original prose only when the caller passes
    ``include_raw_content=true`` on the GET endpoint. The field is always
    present in the response shape — Pydantic v2 with the default
    ``model_config`` does not enable ``exclude_none`` — so clients should
    test ``raw_content is not None`` rather than ``"raw_content" in body``.
    Eval/diagnosis tooling uses it to determine whether the model
    converged on useful output that the synthesizer hid.
    """

    id: str
    session_id: str
    role: str
    content: str
    raw_content: str | None = None
    tool_calls: ToolCallList | None = None
    created_at: datetime
    composition_state_id: str | None = None
    tool_call_id: str | None = None
    parent_assistant_id: str | None = None
    sequence_no: int | None = None


class MessageWithStateResponse(_StrictResponse):
    """Response for POST /api/sessions/{id}/messages.

    State is null when the composition version is unchanged; populated
    with the updated CompositionState when composition changes occur.
    """

    message: ChatMessageResponse
    state: CompositionStateResponse | None = None
    proposals: list[CompositionProposalResponse]


class ValidationEntryResponse(_StrictResponse):
    """Structured validation entry preserving component attribution.

    Mirrors ``ValidationEntry.to_dict()`` from the composer state module.
    """

    component: str
    message: str
    severity: str


type CompositionObject = dict[str, JsonValue]
type CompositionObjectList = list[CompositionObject]


class ComposerPreferencesResponse(_StrictResponse):
    session_id: str
    trust_mode: ComposerTrustMode
    density_default: ComposerDensityDefault
    interpretation_review_disabled: bool
    updated_at: datetime


class UpdateComposerPreferencesRequest(_RequestModel):
    trust_mode: ComposerTrustMode
    density_default: ComposerDensityDefault


class CompositionProposalResponse(_StrictResponse):
    id: str
    session_id: str
    tool_call_id: str
    tool_name: str
    status: ProposalLifecycleStatus
    summary: str
    rationale: str
    affects: list[str]
    arguments_redacted_json: CompositionObject
    base_state_id: str | None = None
    committed_state_id: str | None = None
    audit_event_id: str | None = None
    created_at: datetime
    updated_at: datetime


class RejectProposalRequest(_RequestModel):
    reason: str | None = None


class ProposalEventResponse(_StrictResponse):
    id: str
    session_id: str
    proposal_id: str | None = None
    event_type: ProposalEventType
    actor: str
    payload: CompositionObject
    created_at: datetime


class CompositionStateResponse(_StrictResponse):
    """Response for composition state endpoints."""

    id: str
    session_id: str
    version: int
    sources: dict[str, CompositionObject] | None = None
    nodes: CompositionObjectList | None = None
    edges: CompositionObjectList | None = None
    outputs: CompositionObjectList | None = None
    metadata: CompositionObject | None = None
    is_valid: bool
    validation_errors: list[str] | None = None
    validation_warnings: list[ValidationEntryResponse] | None = None
    validation_suggestions: list[ValidationEntryResponse] | None = None
    derived_from_state_id: str | None = None
    created_at: datetime
    # Operational/audit metadata produced by the composer pipeline.
    # Known keys include ``repair_turns_used``, ``guided_session``, and
    # ``implicit_decisions``. ``None`` is honest for revert/fork paths and
    # for historical states written before this surface existed.
    composer_meta: CompositionObject | None = None


class ForkSessionRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/fork."""

    from_message_id: UUID
    new_message_content: str = pydantic.Field(min_length=1)

    @field_validator("new_message_content")
    @classmethod
    def _validate_new_message_content(cls, value: str) -> str:
        return _require_visible_content(value, field_label="Fork message content")


class ForkSessionResponse(_StrictResponse):
    """Response for POST /api/sessions/{id}/fork."""

    session: SessionResponse
    messages: list[ChatMessageResponse]
    composition_state: CompositionStateResponse | None = None


class RevertStateRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/state/revert."""

    state_id: UUID


class StartGuidedRequest(_RequestModel):
    """Request body for POST /api/sessions/{session_id}/guided/start.

    ``profile`` is a raw boundary value whose valid form is a closed-enum
    discriminator (``WorkflowProfileKind``). The route validates that the
    value is a short string, maps it to the SERVER-owned WorkflowProfile
    constant, and rejects anything else with a generic 400. Typing it as
    ``object`` (not the enum) keeps a stale/hostile client's unknown or
    object-shaped value out of a Pydantic 422 and away from the response —
    the handler never echoes the raw value, so an attempted profile object
    carrying injected fields cannot leak back through the error.
    """

    profile: object = "live"


class RunResponse(_StrictResponse):
    """Response for GET /api/sessions/{id}/runs."""

    id: str
    session_id: str
    status: SessionRunStatus
    accounting: RunAccounting | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    composition_version: int
    discard_summary: DiscardSummary | None = None


class TurnRecordResponse(_StrictResponse):
    """Wire representation of a single TurnRecord in the guided-session history."""

    step: str
    turn_type: str
    payload_hash: str
    response_hash: str | None
    summary: str | None
    emitter: str


class TerminalStateResponse(_StrictResponse):
    """Wire representation of a TerminalState."""

    kind: str
    reason: str | None
    pipeline_yaml: str | None


class ChatTurnResponse(_StrictResponse):
    """Wire representation of one entry in :attr:`GuidedSessionResponse.chat_history`.

    Mirrors :class:`elspeth.web.composer.guided.protocol.ChatTurn`.  Field
    values are server-emitted (Tier 1) — ``role`` is one of ``"user"`` or
    ``"assistant"``, ``step`` is a :class:`GuidedStep` value, ``ts_iso``
    is the ISO 8601 timestamp the turn was appended to ``chat_history``.

    ``assistant_message_kind`` / ``synthetic_failure_reason`` (fp-review C-2
    persisted-history closure) mirror the same-named fields on
    :class:`elspeth.web.composer.guided.protocol.ChatTurn` — ``None`` on
    every ``USER`` turn, on any turn persisted before this field existed, and
    on commit-seam rejections whose redaction-safe classifier is carried by
    the chat-turn audit row instead of this closed user-facing reason set.
    """

    role: str
    content: str
    seq: int
    step: str
    ts_iso: str
    assistant_message_kind: Literal["assistant", "synthetic_failure"] | None
    synthetic_failure_reason: Literal["quality_guard", "unavailable"] | None


class WorkflowProfileResponse(_StrictResponse):
    """Wire-visible subset of a server-owned WorkflowProfile.

    Mirrors :class:`elspeth.web.composer.guided.profile.WorkflowProfile`
    (the four behavior flags). ``None`` at the parent
    ``GuidedSessionResponse.profile`` level means the empty/live-guided profile.
    """

    coaching: bool
    bookends: bool
    recipe_match: bool
    advisor_checkpoints: bool


class GuidedSessionResponse(_StrictResponse):
    """Wire representation of the GuidedSession attached to a CompositionState."""

    step: str
    history: list[TurnRecordResponse]
    terminal: TerminalStateResponse | None
    # Phase A slice 5 — per-step chat history persisted on the GuidedSession.
    # Required (no Pydantic default) so every route surfacing a
    # GuidedSessionResponse must explicitly pass the live values.  A default
    # of ``[]`` / ``0`` here would hide drift: a route that forgot to thread
    # ``guided.chat_history`` through would silently return an empty wire
    # field while the server held real history.  Per CLAUDE.md auditability
    # standard, that is evidence tampering.
    chat_history: list[ChatTurnResponse]
    chat_turn_seq: int
    # Server-owned WorkflowProfile (wire-visible subset). ``None`` for the
    # empty/live-guided profile. Defaulted to ``None`` because most
    # GuidedSessionResponse construction sites carry the empty profile; the
    # start/GET path overrides it explicitly.
    profile: WorkflowProfileResponse | None = None


class TurnPayloadResponse(_StrictResponse):
    """Opaque turn payload — type discriminated by ``type`` at the parent level.

    ``payload`` carries the raw TypedDict contents for the turn (e.g.
    ``SingleSelectPayload``, ``InspectAndConfirmPayload``).  Pydantic strict
    mode does not apply to ``Any``-typed fields; the route handler guarantees
    the payload is a plain dict at construction time.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    type: str
    step_index: int
    payload: JsonValue


class GetGuidedResponse(_StrictResponse):
    """Response for GET /api/sessions/{id}/guided."""

    guided_session: GuidedSessionResponse
    next_turn: TurnPayloadResponse | None
    terminal: TerminalStateResponse | None
    composition_state: CompositionStateResponse | None


class TutorialSampleResponse(_StrictResponse):
    """Response for GET /api/sessions/{id}/guided/tutorial-sample.

    Runtime-derived inputs for the tutorial's prefilled worked example: the 3
    synthetic sample-page URLs (appended to the locked STEP_1 prompt the learner
    sends verbatim) for the active tutorial session's resolved origin. The URLs
    are computed from the resolved base at request time (they cannot ride the
    frozen profile constants). The tutorial's ``web_scrape`` node relies on the
    plugin default ``allowed_hosts="public_only"`` — the pages are publicly
    hosted, so the server injects no SSRF allowlist.
    """

    sample_urls: list[str]


class GuidedRespondRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/guided/respond.

    Carries the user's typed response to the current guided turn.  All
    fields from ``TurnResponse`` are optional at the HTTP boundary (Pydantic
    coerces absent keys to ``None``); the route handler validates that the
    combination is consistent with the current turn type.

    ``control_signal`` is a plain string so that stale clients sending an
    unknown signal value fail gracefully rather than crashing at the HTTP
    boundary.  The route handler validates the value against the closed
    ``ControlSignal`` enum.
    """

    chosen: list[str] | None = None
    edited_values: dict[str, Any] | None = None
    custom_inputs: list[str] | None = None
    accepted_step_index: int | None = None
    edit_step_index: int | None = None
    control_signal: str | None = None
    # Optimistic-concurrency token (D16): the client's expected current step.
    # When present, the route 409s if it does not match the session's live
    # ``guided.step``. A plain ``str`` (not the enum) makes unknown values fail
    # with a route-handler 400 rather than a Pydantic 422.
    step_index: str | None = None


class GuidedRespondResponse(_StrictResponse):
    """Response for POST /api/sessions/{id}/guided/respond.

    Mirrors GET /guided response shape so the frontend can replace its
    cached guided_session in a single pass.
    """

    guided_session: GuidedSessionResponse
    next_turn: TurnPayloadResponse | None
    terminal: TerminalStateResponse | None
    composition_state: CompositionStateResponse | None


class GuidedChatRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/guided/chat.

    Carries a free-text chat message scoped to the user's current wizard
    step. **Not** a turn-answer — chat does not advance step state. The
    backend invokes the per-step chat solver with a step-scoped skill
    briefing and returns the LLM's advisory reply.

    ``step_index`` is a plain string (not the ``GuidedStep`` enum) so a
    stale client that sends an unknown step value fails with a 400 from
    the route handler carrying a clear message, rather than a Pydantic
    422. This mirrors the ``control_signal`` convention in
    ``GuidedRespondRequest``.

    Length cap of 4096 is enforced at the boundary; ``solve_step_chat``
    contains a redundant inner empty-check as a defense-in-depth guard
    against route handler misuse, but length is checked here only.
    """

    message: str = pydantic.Field(min_length=1, max_length=4096)
    step_index: str

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _require_visible_content(value, field_label="Chat message")


class GuidedChatResponse(_StrictResponse):
    """Response for POST /api/sessions/{id}/guided/chat.

    ``assistant_message`` is the LLM's reply, or a fallback message on
    failure. ``assistant_message_kind`` is the wire discriminator that used
    to be missing (fp-review C-2): ``"assistant"`` for a real LLM reply,
    ``"synthetic_failure"`` for a fallback message the server generated
    instead of forwarding a model response. Synthetic-failure causes
    (scaffold-leak guard rejection, commit-seam rejection, or provider /
    solver unavailability) render distinct message copy; all surface the same
    recovery affordance, so this discriminator deliberately stops at kind and
    does not also carry a reason.

    ``guided_session`` always carries the updated chat history. Most chat
    remains advisory, but Step 1 schema-form chat may resolve a complete
    source/data request; in that case ``next_turn`` and
    ``composition_state`` mirror ``/respond`` so the frontend can advance
    atomically.
    """

    assistant_message: str
    assistant_message_kind: Literal["assistant", "synthetic_failure"]
    guided_session: GuidedSessionResponse
    next_turn: TurnPayloadResponse | None
    terminal: TerminalStateResponse | None
    composition_state: CompositionStateResponse | None


# ---------------------------------------------------------------------------
# Phase 5b — interpretation-event wire schemas (Task 3)
# ---------------------------------------------------------------------------
#
# These models mirror the Phase 5b interpretation-event contract types
# (``elspeth.contracts.composer_interpretation``).  The contract dataclass
# is the read-side type used inside the service layer; these pydantic
# models are the wire-side types used by HTTP routes.  Per project
# convention, both sides exist so that:
#
# - The DB row → record → HTTP response path passes through three
#   independent gates (DB CHECK constraint, dataclass validator,
#   pydantic strict-mode validator), each rejecting bad data with a
#   crash rather than silently coercing it.
# - The wire model can carry max-length caps without polluting the
#   contract dataclass (which models the DB row exactly).
#
# All field caps are documented in spec lines 1428-1459.


class InterpretationEventResponse(BaseModel):
    """Wire mirror of :class:`InterpretationEventRecord`.

    Read-side wire schema for a single row of the
    ``interpretation_events_table``.  Used by:

    - GET /api/sessions/{id}/interpretations  (list — wrapped in
      :class:`ListInterpretationEventsResponse`).
    - POST /api/sessions/{id}/interpretations/{event_id}/resolve  (the
      resolved row is returned alongside the new composition state in
      :class:`InterpretationResolveResponse`).

    The three structural row shapes (``user_approved``,
    ``auto_interpreted_opt_out``, ``auto_interpreted_no_surfaces``) are
    distinguished by ``interpretation_source``.  The nullable fields
    carry NULL for opted-out rows; ``llm_draft`` and ``user_term``
    additionally carry NULL for ``auto_interpreted_no_surfaces`` rows
    (rate-cap exhaustion — no draft was produced).  See the
    :class:`InterpretationEventRecord` docstring for the full per-shape
    field nullability table.

    ``frozen=True`` matches the read-side contract: clients receive an
    immutable view of an audit-table row.  ``extra="forbid"`` keeps the
    HTTP response surface aligned with the DB schema — adding a field
    requires a coordinated schema/contract/wire update.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    session_id: UUID
    # ``None`` for session-level ``auto_interpreted_opt_out`` marker rows
    # (no surfacing occurred). Surface-specific opt-out rows carry kind,
    # surface/provenance fields, accepted_value, arguments_hash, and
    # hash_domain_version='v2'.
    composition_state_id: UUID | None = None
    affected_node_id: str | None = Field(default=None, max_length=256)
    tool_call_id: str | None = Field(default=None, max_length=256)
    user_term: str | None = Field(default=None, max_length=8192)
    kind: InterpretationKind | None = None
    llm_draft: str | None = Field(default=None, max_length=8192)
    accepted_value: str | None = Field(default=None, max_length=8192)
    choice: InterpretationChoice
    created_at: datetime
    resolved_at: datetime | None = None
    # Actor format mirrors ``ProposalEventRecord.actor``: originator:role:id
    # for request-scoped actors, system:{component} for system writers.
    actor: str = Field(min_length=1, max_length=256)
    interpretation_source: InterpretationSource
    # Audit-provenance fields — bound to which LLM produced the draft.
    # Exposed on the wire so the audit-readiness panel and any future
    # reviewer surface can render "drafted by claude-opus-4-7 v… on
    # 2026-05-18" without a second DB round-trip.  NULL for
    # ``auto_interpreted_opt_out`` rows (no LLM was consulted).
    model_identifier: str | None = Field(default=None, max_length=256)
    model_version: str | None = Field(default=None, max_length=128)
    provider: str | None = Field(default=None, max_length=64)
    # hex SHA-256 of pipeline_composer.md content at draft time.
    composer_skill_hash: str | None = Field(default=None, max_length=64)
    # hex rfc8785-canonical hash over INTERPRETATION_HASH_DOMAIN_V2;
    # populated at resolve time, NULL until then and for opt-out rows.
    arguments_hash: str | None = Field(default=None, max_length=64)
    # ``v2`` once resolved (F-12); NULL until then and for marker opt-out rows.
    hash_domain_version: str | None = Field(default=None, max_length=16)
    # F-19: runtime model snapshot at resolve time (may differ from the
    # composer model that produced the draft if a model swap happened
    # between surfacing and resolution).
    runtime_model_identifier_at_resolve: str | None = Field(default=None, max_length=256)
    runtime_model_version_at_resolve: str | None = Field(default=None, max_length=128)
    # Cross-DB hash anchor (Option A): hex SHA-256 of the resolved
    # prompt-template string.  NULL until resolved; NULL for opted-out
    # rows (no prompt template is patched).  Exposed on the wire so
    # audit-tooling consumers can verify hash equality without a second
    # DB round-trip.
    resolved_prompt_template_hash: str | None = Field(default=None, max_length=64)


class InterpretationResolveRequest(BaseModel):
    """Request body for POST /api/sessions/{id}/interpretations/{event_id}/resolve.

    Carries the user's resolution of a previously-surfaced interpretation
    event.  ``opted_out`` and ``abandoned`` are NOT valid resolve
    choices — opt-out goes through a separate route, and abandoned is
    written by the session-end cleanup job, not by user action.
    ``pending`` is the pre-resolve state and is also not a valid input.

    The ``amended_value`` field carries the user's edited interpretation
    when ``choice == "amended"``.  Schema-layer content checks
    (metacharacters, control chars, length caps, credential-shape
    prefilter) live in :func:`_validate_accepted_value_content` so the
    same regex set guards both the schema boundary (this validator) and
    the tool boundary (request_interpretation_review — Task 5).  The
    duplicated validation is intentional defense-in-depth for F-2
    (prompt-injection bypass on the accepted_as_drafted path).
    """

    model_config = ConfigDict(extra="forbid")

    choice: Literal["accepted_as_drafted", "amended"]
    # Optional from the wire perspective; the model_validator enforces
    # that ``amended`` requires it and ``accepted_as_drafted`` forbids
    # it.  ``max_length`` is the outer cap; the per-line cap (1024) is
    # enforced inside ``_validate_accepted_value_content`` to give a
    # specific error message rather than the generic ``max_length``
    # rejection.
    amended_value: str | None = Field(default=None, max_length=8192)

    @model_validator(mode="after")
    def _amended_value_consistency(self) -> InterpretationResolveRequest:
        # An empty string ("") is treated the same as None for the
        # required-when-amended check: a zero-length amendment cannot be
        # the user's intended interpretation — accepted_as_drafted is
        # the correct path for "no change".
        if self.choice == "amended" and not self.amended_value:
            raise ValueError("amended_value is required when choice == 'amended'")
        if self.choice == "accepted_as_drafted" and self.amended_value is not None:
            raise ValueError("amended_value must be omitted when choice == 'accepted_as_drafted'")
        return self

    @field_validator("amended_value", mode="after")
    @classmethod
    def _validate_amended_value_content(cls, v: str | None) -> str | None:
        """Reject template metacharacters, control chars, credential-shaped
        content, and overlength single-line strings.

        Delegates to :func:`_validate_accepted_value_content` so the
        schema layer and the tool boundary (Task 5) share one regex set
        for content checks.  Empty string and None bypass the content
        checks — the model_validator handles their consistency with the
        ``choice`` field.
        """
        if not v:
            return v
        _validate_accepted_value_content(v)
        return v


class InterpretationResolveResponse(BaseModel):
    """Response for POST /api/sessions/{id}/interpretations/{event_id}/resolve.

    The resolved event row PLUS the new composition state produced by
    patching the affected LLM transform (provenance:
    ``interpretation_resolve``).  Returning both in one envelope lets
    the frontend update its event-list view and its composition-state
    view atomically; a two-call shape would create a window where the
    UI shows a resolved event whose downstream pipeline patch is not
    yet visible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: InterpretationEventResponse
    new_state: CompositionStateResponse


class InterpretationOptOutResponse(BaseModel):
    """Response for POST /api/sessions/{id}/interpretations/opt_out.

    The opt-out route has no body fields beyond the implicit actor
    (carried by the auth middleware).  The response surfaces:

    - ``session_id`` — echoed so the caller can confirm the right
      session was modified.
    - ``interpretation_review_disabled`` — always ``True`` on success;
      surfaced explicitly so the caller can re-render the toggle
      without a follow-up GET.
    - ``opted_out_at`` — the persisted timestamp.  Tied to the
      ``interpretation_events_table`` opt-out row written in the same
      transaction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: UUID
    interpretation_review_disabled: bool
    opted_out_at: datetime


class ListInterpretationEventsResponse(BaseModel):
    """Response for GET /api/sessions/{id}/interpretations.

    Wraps the list in an envelope object rather than returning a bare
    JSON array — consistent with every other list route on the session
    surface (e.g. ``GuidedSessionResponse.history``) — so future
    pagination metadata can be added without a breaking wire change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    events: list[InterpretationEventResponse]


class OptOutSummaryResponse(BaseModel):
    """Response for GET /api/sessions/{id}/interpretations/opt_out_summary.

    Per F-22 of the Phase 5b backend spec: after a session has opted out
    of interpretation review, the composer-LLM continues to auto-bake
    interpretations (now flagged as ``auto_interpreted_opt_out``) and may
    also write ``auto_interpreted_no_surfaces`` rows when the rate cap is
    exhausted. This route lets a user retroactively review every
    auto-baked interpretation produced during the opted-out portion of
    the session, closing the audit gap of "click opt-out once, dozens of
    auto-interpretations accumulate invisibly."

    Returns rows of both ``auto_interpreted_opt_out`` and
    ``auto_interpreted_no_surfaces`` interpretation_source — the two
    structural row shapes that represent auto-baked interpretations —
    ordered by ``created_at``. ``user_approved`` rows are excluded; the
    standard ``GET /interpretations`` route is the right surface for
    those.

    Envelope shape matches :class:`ListInterpretationEventsResponse` so
    the two list routes have consistent wire ergonomics on the session
    surface.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    events: list[InterpretationEventResponse]


# Forward reference resolution
MessageWithStateResponse.model_rebuild()
ForkSessionResponse.model_rebuild()
InterpretationResolveResponse.model_rebuild()
