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
from typing import Any
from uuid import UUID

import pydantic
from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

from elspeth.web.execution.schemas import DiscardSummary, RunAccounting
from elspeth.web.sessions.protocol import SessionRunStatus
from elspeth.web.validation import has_visible_content


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
    """Request body for POST /api/sessions."""

    title: str = pydantic.Field(default="New session", min_length=1)

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
    forked_from_session_id: str | None = None
    forked_from_message_id: str | None = None


class SendMessageRequest(_RequestModel):
    """Request body for POST /api/sessions/{id}/messages."""

    content: str = pydantic.Field(min_length=1)
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


class MessageWithStateResponse(_StrictResponse):
    """Response for POST /api/sessions/{id}/messages.

    State is null when the composition version is unchanged; populated
    with the updated CompositionState when composition changes occur.
    """

    message: ChatMessageResponse
    state: CompositionStateResponse | None = None


class ValidationEntryResponse(_StrictResponse):
    """Structured validation entry preserving component attribution.

    Mirrors ``ValidationEntry.to_dict()`` from the composer state module.
    """

    component: str
    message: str
    severity: str


type CompositionObject = dict[str, JsonValue]
type CompositionObjectList = list[CompositionObject]


class CompositionStateResponse(_StrictResponse):
    """Response for composition state endpoints."""

    id: str
    session_id: str
    version: int
    source: CompositionObject | None = None
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
    # ``repair_turns_used`` (number of forced repair turns the proof step
    # injected when finalising this state) is currently the only field;
    # ``None`` is honest for revert/fork paths and for historical states
    # written before this surface existed.
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
    """

    role: str
    content: str
    seq: int
    step: str
    ts_iso: str


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

    ``assistant_message`` is the LLM's reply (or, on transient LLM
    failure, a synthetic "I'm unavailable" message — Phase A does not yet
    distinguish the two on the wire; slice 5's ``ComposerChatTurn`` audit
    record adds that discriminator).

    ``guided_session`` is echoed verbatim in Phase A — chat does not
    mutate session state, but the frontend store keeps a single object
    so we return it to keep the client/server contract symmetric with
    ``/respond``. Slice 5 makes ``chat_history`` an additive field that
    will carry incremental turns.
    """

    assistant_message: str
    guided_session: GuidedSessionResponse


# Forward reference resolution
MessageWithStateResponse.model_rebuild()
ForkSessionResponse.model_rebuild()
