"""Session-scoped composer progress snapshots.

The progress surface is a UI status channel, not a reasoning transcript.
Snapshots summarize visible composer lifecycle boundaries and tool categories;
they must never carry raw tool arguments, tool results, secrets, or provider
chain-of-thought.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

COMPOSER_PROGRESS_MAX_EVIDENCE = 4
_MAX_PROGRESS_TEXT_CHARS = 180

type ComposerProgressPhase = Literal[
    "idle",
    "starting",
    "calling_model",
    "using_tools",
    "validating",
    "saving",
    "complete",
    "failed",
    "cancelled",
]

# Phases that indicate the composer is actively working on a request. The
# in-flight enumeration endpoint filters on this set so an operator polling
# /_active sees the live composer requests for their own sessions even if
# the SPA tab that posted them is no longer connected.
NON_TERMINAL_PROGRESS_PHASES: frozenset[ComposerProgressPhase] = frozenset(
    {
        "starting",
        "calling_model",
        "using_tools",
        "validating",
        "saving",
    }
)

# Stable machine-readable reason codes for composer progress events.
#
# Public taxonomy distinct from ComposerConvergenceError.budget_exhausted —
# the exception models which budget tripped (a private engine concept), this
# Literal is the public-facing UX/observability discriminator. They map but
# they are not the same enum: the convergence error contributes three of
# these codes; the others come from sibling exception classes or from
# success/idle sentinels.
#
# Required when phase == "failed" (enforced by the model_validator on
# ComposerProgressEvent below) so a new failure site cannot ship without
# carrying a stable code. The frontend, structured logs, and the 422
# response body all branch on this value.
type ComposerProgressReason = Literal[
    # Convergence sub-causes — split out from the single
    # ComposerConvergenceError class via its budget_exhausted discriminator.
    "convergence_composition_budget",
    "convergence_discovery_budget",
    "convergence_wall_clock_timeout",
    # Provider-side failures — LiteLLM exception families.
    "provider_auth_failed",
    "provider_unavailable",
    # Server-side plugin bug escaping execute_tool.
    "plugin_crash",
    # Runtime preflight failure (cached path-1 or post-compose path-2 —
    # users cannot act on the path distinction, so a single code).
    "runtime_preflight_failed",
    # Generic ComposerServiceError — prompt prep / availability / catch-all.
    "service_setup_failed",
    # Client closed the HTTP connection or operator cancelled the request
    # before the composer returned. Distinct from convergence_wall_clock_timeout
    # (server budget exceeded) so dashboards and audit can tell apart "the
    # client gave up" from "the server gave up". Required when phase ==
    # "cancelled" by the same model_validator that requires it on "failed".
    "client_cancelled",
    # Non-failure sentinels — every snapshot carries a code so observability
    # and the SPA never have to special-case None.
    "composer_idle",
    "composer_complete",
]
type ComposerProgressSink = Callable[["ComposerProgressEvent"], Awaitable[None]]


class _StrictProgressModel(BaseModel):
    """Strict model for system-owned progress snapshots."""

    model_config = ConfigDict(strict=True, extra="forbid")


class ComposerProgressEvent(_StrictProgressModel):
    """Provider-safe progress event emitted by the composer path."""

    phase: ComposerProgressPhase
    headline: str
    evidence: tuple[str, ...] = ()
    likely_next: str | None = None
    reason: ComposerProgressReason | None = None

    @field_validator("headline")
    @classmethod
    def _validate_headline(cls, value: str) -> str:
        return _clean_required_text(value, field_name="headline")

    @field_validator("likely_next")
    @classmethod
    def _validate_likely_next(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_required_text(value, field_name="likely_next")

    @field_validator("evidence")
    @classmethod
    def _bound_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bounded: list[str] = []
        for item in value:
            cleaned = _clean_required_text(item, field_name="evidence")
            bounded.append(cleaned)
            if len(bounded) == COMPOSER_PROGRESS_MAX_EVIDENCE:
                break
        return tuple(bounded)

    @model_validator(mode="after")
    def _require_reason_when_terminal_non_success(self) -> Self:
        # Mechanically forbids the drift the original bug exhibited: a
        # phase="failed" event was emitted with text-only differentiation
        # at three distinct sites and three sub-causes collapsed into one
        # generic message because nothing in the contract required a
        # discriminator. With this validator, a new failure site cannot
        # be added without choosing a code from ComposerProgressReason.
        # The same rule applies to phase == "cancelled" (added with the
        # in-flight observability work) — operator dashboards branch on
        # reason to distinguish client_cancelled from a future operator-
        # initiated cancel without parsing the headline.
        # Other phases keep reason optional — they're status pings, not
        # routing decisions.
        if self.phase in ("failed", "cancelled") and self.reason is None:
            raise ValueError(
                "ComposerProgressEvent.reason is required when phase is 'failed' or "
                "'cancelled' so the frontend, audit logs, and HTTP response body can "
                "branch on a stable taxonomy instead of free-text headline parsing."
            )
        return self


class ComposerProgressSnapshot(ComposerProgressEvent):
    """Latest composer progress snapshot for one session."""

    session_id: str
    request_id: str | None
    updated_at: datetime


class ComposerProgressRegistry:
    """In-memory latest-progress registry keyed by session id.

    The registry intentionally stores one bounded snapshot per session, not an
    append-only log. The immutable session/chat tables remain the source of
    truth for persisted conversation history.

    The registry also maintains a parallel session_id -> user_id index used
    only by ``list_active`` to scope cross-session enumeration to the
    authenticated user. user_id is intentionally NOT a field on
    ComposerProgressSnapshot — the per-session GET endpoint already
    authenticates the caller against session ownership, so leaking the
    user_id back through the snapshot body would be redundant and would
    couple the public progress contract to auth identity.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, ComposerProgressSnapshot] = {}
        self._user_index: dict[str, str] = {}
        self._lock = threading.Lock()

    async def publish(
        self,
        *,
        session_id: str,
        request_id: str | None,
        user_id: str,
        event: ComposerProgressEvent,
    ) -> ComposerProgressSnapshot:
        """Store and return the latest progress snapshot for a session.

        ``user_id`` is recorded in the registry's internal user index so
        :meth:`list_active` can scope cross-session enumeration to one
        authenticated principal. It is NOT written into the snapshot body
        returned to the SPA.
        """
        with self._lock:
            updated_at = self._next_timestamp(session_id)
            snapshot = ComposerProgressSnapshot(
                session_id=session_id,
                request_id=request_id,
                phase=event.phase,
                headline=event.headline,
                evidence=event.evidence,
                likely_next=event.likely_next,
                reason=event.reason,
                updated_at=updated_at,
            )
            self._snapshots[session_id] = snapshot
            self._user_index[session_id] = user_id
            return snapshot

    async def get_latest(self, session_id: str) -> ComposerProgressSnapshot:
        """Return latest progress or a neutral idle snapshot."""
        with self._lock:
            if session_id in self._snapshots:
                return self._snapshots[session_id]
            return _idle_snapshot(session_id)

    async def list_active(self, *, user_id: str) -> tuple[ComposerProgressSnapshot, ...]:
        """Return non-terminal snapshots for one user's sessions.

        "Non-terminal" means the composer is still working (starting,
        calling_model, using_tools, validating, saving). Snapshots whose
        phase is idle/complete/failed/cancelled are excluded — those
        sessions are no longer in flight.

        Filtered by ``user_id`` against the internal user index so a
        caller cannot enumerate other users' sessions even if they hold
        the same registry reference. The ordering is by updated_at
        (oldest first) so an operator's view shows the most-stuck
        request at the top, which is the typical triage starting point.

        Indexes _user_index directly (not via ``.get``) — publish() and
        clear() maintain the invariant that every session_id in
        _snapshots also has an entry in _user_index. A KeyError here
        would be a registry bug, not user input, and crashing surfaces
        it instead of silently returning empty results.
        """
        with self._lock:
            owned = (
                snap
                for sid, snap in self._snapshots.items()
                if self._user_index[sid] == user_id and snap.phase in NON_TERMINAL_PROGRESS_PHASES
            )
            return tuple(sorted(owned, key=lambda snap: snap.updated_at))

    async def clear(self, session_id: str) -> None:
        """Remove a session snapshot and its user-index entry.

        Idempotent — clear() is called from session archival regardless of
        whether the registry ever held a snapshot for that session, so the
        ``in`` guard is the offensive-programming-compliant way to express
        "remove if present" without using ``dict.pop(default)``.
        """
        with self._lock:
            if session_id in self._snapshots:
                del self._snapshots[session_id]
            if session_id in self._user_index:
                del self._user_index[session_id]

    def _next_timestamp(self, session_id: str) -> datetime:
        now = datetime.now(UTC)
        if session_id in self._snapshots:
            previous = self._snapshots[session_id].updated_at
            if now <= previous:
                return previous + timedelta(microseconds=1)
        return now


def convergence_progress_event(
    *,
    budget_exhausted: Literal["composition", "discovery", "timeout"],
) -> ComposerProgressEvent:
    """Map a convergence budget discriminator to a discriminated progress event.

    Three sub-causes (composition turn budget, discovery turn budget, wall-clock
    timeout) used to collapse into one generic ``phase: failed`` event because
    only ``ComposerConvergenceError.budget_exhausted`` carried the discriminator
    and the emit sites discarded it. This helper is the single dispatch point
    — both the service-level catch (compose() outer try/except) and the
    route-handler catches in web/sessions/routes.py route through it so the
    per-cause headline / evidence / likely_next / reason copy is defined
    exactly once.

    Lives in this module rather than service.py because:

    - the failure-mode taxonomy is a property of the progress contract, not
      the service implementation;
    - taking a string discriminator (not the exception) avoids importing
      ``ComposerConvergenceError`` from ``composer.protocol``, which already
      imports ``ComposerProgressSink`` from this module — keeping the helper
      here would otherwise create a circular import.

    Recovery copy is what the user can act on:

    - composition budget → split the request into smaller turns
    - discovery budget   → narrow the schema/catalog exploration
    - wall-clock timeout → retry, or ask an operator to raise the server budget
    """
    if budget_exhausted == "timeout":
        return ComposerProgressEvent(
            phase="failed",
            headline="The composer timed out before producing a final answer.",
            evidence=("The composer wall-clock budget elapsed during this request.",),
            likely_next=("Retry once the provider responds faster, or ask an operator to raise the composer wall-clock budget."),
            reason="convergence_wall_clock_timeout",
        )
    if budget_exhausted == "discovery":
        return ComposerProgressEvent(
            phase="failed",
            headline="The composer used its discovery turn budget without finishing.",
            evidence=("The discovery-turn budget was exhausted before a final answer.",),
            likely_next=("Narrow the schema or catalog exploration, or ask an operator to raise the discovery-turn budget."),
            reason="convergence_discovery_budget",
        )
    return ComposerProgressEvent(
        phase="failed",
        headline="The composer used its mutation turn budget without finishing.",
        evidence=("The mutation-turn budget was exhausted before a final answer.",),
        likely_next=("Split the request into smaller turns, or ask an operator to raise the mutation-turn budget."),
        reason="convergence_composition_budget",
    )


def client_cancelled_progress_event() -> ComposerProgressEvent:
    """Build the progress event for a client-disconnect cancellation.

    Centralised here for the same reason ``convergence_progress_event`` is —
    the failure-mode taxonomy is a property of the progress contract. Both
    composer entry points (send_message and recompose) emit this exact text
    on ``asyncio.CancelledError`` so an operator inspecting the snapshot
    cannot tell which route was cancelled from the headline alone (the
    request_id on the snapshot disambiguates if needed). Recovery copy is
    written for the user, not the operator: the user sees this through
    /composer-progress polling if they ever reconnect to the session.
    """
    return ComposerProgressEvent(
        phase="cancelled",
        headline="The request was cancelled before the composer finished.",
        evidence=("The client closed the connection before a response was returned.",),
        likely_next="Resubmit the message when ready, or try a smaller request.",
        reason="client_cancelled",
    )


def _idle_snapshot(session_id: str) -> ComposerProgressSnapshot:
    return ComposerProgressSnapshot(
        session_id=session_id,
        request_id=None,
        phase="idle",
        headline="No active composer work.",
        evidence=(),
        likely_next=None,
        reason="composer_idle",
        updated_at=datetime.now(UTC),
    )


def _clean_required_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"composer progress {field_name} must contain visible text")
    if len(cleaned) > _MAX_PROGRESS_TEXT_CHARS:
        return cleaned[: _MAX_PROGRESS_TEXT_CHARS - 1].rstrip() + "."
    return cleaned
