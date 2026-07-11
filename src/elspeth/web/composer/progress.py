"""Session-scoped composer progress snapshots.

The progress surface is a UI status channel, not a reasoning transcript.
Snapshots summarize visible composer lifecycle boundaries and tool categories;
they must never carry raw tool arguments, tool results, secrets, or provider
chain-of-thought.

The L0-suitable progress contracts (``ComposerProgressEvent``,
``ComposerProgressPhase``, ``ComposerProgressReason``,
``ComposerProgressSink``, ``COMPOSER_PROGRESS_MAX_EVIDENCE``,
``NON_TERMINAL_PROGRESS_PHASES``) live in
``elspeth.contracts.composer_progress``.  This module owns only the
L3-dependent residue: the in-memory ``ComposerProgressRegistry`` (which
uses threading), the snapshot subclass that joins event with session
identity, and the per-phase event factory functions whose copy
references tool names from ``elspeth.web.composer.tools``.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Literal

from elspeth.contracts.composer_progress import (
    NON_TERMINAL_PROGRESS_PHASES,
    ComposerProgressEvent,
    ComposerProgressSink,
)
from elspeth.web.composer.tools import is_discovery_tool

__all__ = [
    "ComposerProgressRegistry",
    "ComposerProgressSnapshot",
    "advisor_checkpoint_progress_event",
    "client_cancelled_progress_event",
    "convergence_progress_event",
    "emit_progress",
    "model_call_progress_event",
    "tool_batch_progress_event",
    "tool_completed_progress_event",
    "tool_started_progress_event",
]


class ComposerProgressSnapshot(ComposerProgressEvent):
    """Latest composer progress snapshot for one session."""

    session_id: str
    request_id: str | None
    updated_at: datetime
    # Live count of compose requests (send_message / recompose) currently
    # inside the route for this session — including time spent queued on
    # the per-session compose lock, before any progress is published.
    # Enriched at read time by the registry (see get_latest); the SPA's
    # post-abort reconciliation treats zero as its only settlement signal
    # (elspeth-06a23adfcc) because the phase alone cannot distinguish an
    # aborted-but-still-running request from full quiescence.
    inflight_requests: int = 0


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
        self._inflight: dict[str, int] = {}
        self._lock = threading.Lock()

    def begin_request(self, session_id: str) -> None:
        """Count one compose request as in flight for ``session_id``.

        Called at route entry (before the compose-lock wait) by the
        ``_track_compose_inflight`` dependency; paired with
        :meth:`end_request` at request teardown.
        """
        with self._lock:
            self._inflight[session_id] = self._inflight.get(session_id, 0) + 1

    def end_request(self, session_id: str) -> None:
        """Release one in-flight compose request for ``session_id``."""
        with self._lock:
            remaining = self._inflight.get(session_id, 0) - 1
            if remaining > 0:
                self._inflight[session_id] = remaining
            else:
                self._inflight.pop(session_id, None)

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
        """Return latest progress or a neutral idle snapshot.

        The snapshot is enriched with the CURRENT in-flight request count —
        not the count at publish time — so a poller always observes the
        live quiescence state alongside the last narrative phase.
        """
        with self._lock:
            snapshot = self._snapshots.get(session_id) or _idle_snapshot(session_id)
            return self._with_live_inflight(session_id, snapshot)

    def _with_live_inflight(self, session_id: str, snapshot: ComposerProgressSnapshot) -> ComposerProgressSnapshot:
        """Overlay the live in-flight count onto a stored snapshot.

        Stored snapshots carry the field's default (0) from publish();
        every read surface must overlay the CURRENT count or it reports
        an actively composing session as quiescent. Caller must hold
        ``self._lock``.
        """
        inflight = self._inflight.get(session_id, 0)
        if snapshot.inflight_requests == inflight:
            return snapshot
        return snapshot.model_copy(update={"inflight_requests": inflight})

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
                self._with_live_inflight(sid, snap)
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
      imports ``ComposerProgressSink`` from contracts — keeping the helper
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


# ---------------------------------------------------------------------------
# Progress event factories — co-located with ComposerProgressEvent so the
# per-phase headline / evidence / likely_next copy lives in one place.
# ---------------------------------------------------------------------------


async def emit_progress(
    progress: ComposerProgressSink | None,
    event: ComposerProgressEvent,
) -> None:
    """Emit provider-safe progress when a sink is available."""
    if progress is None:
        return
    await progress(event)


def model_call_progress_event(message: str) -> ComposerProgressEvent:
    headline = "I'm asking the model to choose the next safe pipeline update."
    normalized = message.lower()
    if "html" in normalized and "json" in normalized:
        headline = "I'm asking the model to choose an HTML input and JSON output."
    return ComposerProgressEvent(
        phase="calling_model",
        headline=headline,
        evidence=("The composer is using the prepared prompt and visible pipeline state.",),
        likely_next="The model may answer directly or request safe pipeline tools.",
    )


def advisor_checkpoint_progress_event(checkpoint: str) -> ComposerProgressEvent:
    """Progress for a deterministic advisor (model-distinct reviewer) checkpoint.

    Emitted so the advisor call is visible like every other model call —
    otherwise the snapshot stays frozen on its previous phase while the
    (slower, frontier) advisor model runs, which is indistinguishable from a
    stall to a poller or a watching user. ``checkpoint`` is "early" (plan
    review) or "end" (sign-off).
    """
    if checkpoint == "early":
        headline = "I'm asking the advisor model to review the plan."
        likely_next = "The advisor may suggest changes before the composer continues."
    else:
        headline = "I'm asking the advisor model to sign off on the pipeline."
        likely_next = "The advisor may approve the pipeline or flag changes before finalizing."
    return ComposerProgressEvent(
        phase="calling_model",
        headline=headline,
        evidence=("A second, model-distinct advisor is reviewing the pipeline.",),
        likely_next=likely_next,
    )


def tool_batch_progress_event(tool_names: tuple[str, ...]) -> ComposerProgressEvent:
    if any(_is_schema_or_catalog_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model requested plugin schemas.",
            evidence=("Checking available source, transform, and sink tools.",),
            likely_next="ELSPETH will use visible schemas to guide the pipeline shape.",
        )
    if any(name in {"get_pipeline_state", "preview_pipeline", "diff_pipeline"} for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is checking the current pipeline.",
            evidence=("Reading the visible pipeline graph and validation summary.",),
            likely_next="ELSPETH will compare the request with the current setup.",
        )
    if any(_is_secret_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is checking available secret references.",
            evidence=("Checking available secret references without reading secret values.",),
            likely_next="ELSPETH will keep any credential references deferred.",
        )
    if any(not is_discovery_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is updating the pipeline graph.",
            evidence=("A pipeline-editing tool was requested.",),
            likely_next="ELSPETH will validate the result before saving it.",
        )
    return ComposerProgressEvent(
        phase="using_tools",
        headline="The model requested composer tool information.",
        evidence=("Checking visible composer tool results.",),
        likely_next="ELSPETH will continue from the tool response.",
    )


def tool_started_progress_event(tool_name: str) -> ComposerProgressEvent:
    if _is_schema_or_catalog_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking available source, transform, and sink tools.",
            evidence=("Reading plugin names and schemas only.",),
            likely_next="ELSPETH will choose compatible pipeline components.",
        )
    if _is_secret_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking available secret references.",
            evidence=("Secret names can be checked; secret values stay hidden.",),
            likely_next="ELSPETH will wire only deferred secret references if needed.",
        )
    if is_discovery_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking the current pipeline and tool context.",
            evidence=("Reading visible composer state.",),
            likely_next="ELSPETH will use the result to decide the next action.",
        )
    return ComposerProgressEvent(
        phase="using_tools",
        headline="I'm updating the pipeline graph.",
        evidence=("A pipeline-editing tool is running.",),
        likely_next="ELSPETH will validate the updated pipeline.",
    )


def tool_completed_progress_event(tool_name: str, success: bool) -> ComposerProgressEvent:
    if not success:
        return ComposerProgressEvent(
            phase="using_tools",
            headline="A composer tool reported a visible blocker.",
            evidence=("The tool result was returned without exposing raw request values.",),
            likely_next="ELSPETH will ask the model to adjust the pipeline request.",
        )
    if is_discovery_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The requested tool information is ready.",
            evidence=(_safe_tool_evidence(tool_name),),
            likely_next="ELSPETH will continue with the visible result.",
        )
    return ComposerProgressEvent(
        phase="validating",
        headline="The composer has updated the pipeline and is validating the result.",
        evidence=("A pipeline-editing tool completed successfully.",),
        likely_next="ELSPETH will save the updated pipeline if it is accepted.",
    )


def _is_schema_or_catalog_tool(tool_name: str) -> bool:
    return tool_name in {
        "list_sources",
        "list_transforms",
        "list_sinks",
        "get_plugin_schema",
        "list_models",
    }


def _is_secret_tool(tool_name: str) -> bool:
    return tool_name in {"list_secret_refs", "validate_secret_ref", "wire_secret_ref"}


def _safe_tool_evidence(tool_name: str) -> str:
    if _is_schema_or_catalog_tool(tool_name):
        return "Checking available source, transform, and sink tools."
    if _is_secret_tool(tool_name):
        return "Checking available secret references without reading secret values."
    if tool_name in {"get_pipeline_state", "preview_pipeline", "diff_pipeline"}:
        return "Reading the visible pipeline graph and validation summary."
    return "Using visible composer tool output."
