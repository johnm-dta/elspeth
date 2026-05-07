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
    """Response for a single chat message."""

    id: str
    session_id: str
    role: str
    content: str
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


# Forward reference resolution
MessageWithStateResponse.model_rebuild()
ForkSessionResponse.model_rebuild()
