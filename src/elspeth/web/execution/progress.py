"""ProgressBroadcaster — thread-safe bridge from sync pipeline to async WebSocket.

B1 Fix: All cross-thread pushes use loop.call_soon_threadsafe() to schedule
work on the event loop thread. Direct queue.put_nowait() from a background
thread would corrupt the event loop's internal state.

Backpressure (coalesced-drain): broadcast() does NOT schedule one callback
per event. Each event is appended to a per-subscriber bounded ``pending``
deque under ``self._lock``; a single ``_drain_pending`` callback is
scheduled only when the subscriber's ``drain_scheduled`` flag flips
False → True. That callback runs on the event loop thread, drains the
pending deque to the subscriber's ``asyncio.Queue``, and re-schedules
itself via ``loop.call_soon`` while pending is non-empty (to yield to
other loop work between batches). This bounds the number of pending
``Handle`` objects in the loop's internal ``_ready`` deque to one per
subscriber, rather than one per (event x subscriber) — the earlier
design let the loop's callback queue grow unboundedly even though the
async queue stayed at its maxsize, defeating the OOM guard under slow
WebSocket clients.

Construction timing: Created inside the FastAPI lifespan async context manager,
NOT in the synchronous create_app() factory. The loop reference is captured via
asyncio.get_running_loop() inside the lifespan, which guarantees a running event
loop exists and is Python 3.12+ compatible (asyncio.get_event_loop() emits a
deprecation warning when no running loop exists). The ExecutionServiceImpl is
also constructed inside the lifespan (after ProgressBroadcaster), not in
create_app(), because it depends on the broadcaster instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import deque
from dataclasses import dataclass, field

import structlog

from elspeth.web.execution.schemas import RunEvent

slog = structlog.get_logger()

# Backpressure: maximum events buffered per WebSocket subscriber.
# A stalled client that cannot drain its queue will have oldest events
# dropped rather than accumulating unboundedly toward OOM.
#
# Applied at TWO layers:
#   1. The per-subscriber ``pending`` deque (producer-side, pre-loop).
#   2. The subscriber's asyncio.Queue (consumer-side, post-drain).
# The first layer bounds memory when the event loop is slow or stalled;
# the second layer bounds memory when the WebSocket client is slow.
_SUBSCRIBER_QUEUE_MAXSIZE = 1000

_TERMINAL_EVENT_TYPES = frozenset({"completed", "failed", "cancelled"})


@dataclass(slots=True)
class _SubState:
    """Internal per-subscriber state for coalesced cross-thread delivery.

    Mutable by design: ``pending`` is the producer-side buffer, ``queue``
    is the consumer-side async buffer, and ``drain_scheduled`` coalesces
    scheduled ``_drain_pending`` callbacks to at most one per subscriber
    in flight. All fields are accessed only under ``ProgressBroadcaster._lock``.
    """

    queue: asyncio.Queue[RunEvent]
    pending: deque[RunEvent] = field(default_factory=deque)
    drain_scheduled: bool = False


class ProgressBroadcaster:
    """In-process event broadcaster for pipeline progress.

    Bridges the synchronous background thread (where the Orchestrator runs)
    to the async WebSocket handlers (where clients receive events).

    Thread safety contract:
    - subscribe() and unsubscribe() may be called from the asyncio event loop thread.
    - broadcast() is called from the background pipeline thread.
    - broadcast() appends to a per-subscriber pending deque under self._lock
      and schedules self._drain_pending via self._loop.call_soon_threadsafe()
      only when the subscriber's drain_scheduled flag flips False → True.
    - All mutations to and iterations over self._subscribers and any
      _SubState field are protected by self._lock (a threading.Lock). This
      ensures correctness regardless of GIL presence (safe under PEP 703 /
      free-threaded Python).
    """

    _TERMINAL_EVENT_TYPES = _TERMINAL_EVENT_TYPES

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._subscribers: dict[str, dict[asyncio.Queue[RunEvent], _SubState]] = {}
        self._terminalized: set[str] = set()

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent]:
        """Create and register a subscriber queue for a run.

        Called from the WebSocket handler when a client connects.
        Returns the queue that the handler awaits on.
        """
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAXSIZE)
        state = _SubState(queue=queue)
        with self._lock:
            self._subscribers.setdefault(run_id, {})[queue] = state
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent]) -> None:
        """Remove a subscriber queue. Called on WebSocket disconnect.

        Any drain callback already in flight for this subscriber's state
        continues to run; it still holds a direct reference to the state
        via its closure args. The state is garbage collected once the
        drain completes and the callback's Handle is released.

        Always called from a finally block to ensure cleanup.
        """
        with self._lock:
            if run_id in self._subscribers:
                self._subscribers[run_id].pop(queue, None)

    def broadcast(self, run_id: str, event: RunEvent) -> None:
        """Thread-safe broadcast — callable from background threads.

        Appends the event to each subscriber's bounded pending deque and
        schedules a drain callback per subscriber whose drain_scheduled
        flag was not already set. At most one drain callback is scheduled
        or running per subscriber at any time — this is what bounds the
        event loop's internal _ready callback queue.
        """
        with self._lock:
            if run_id in self._terminalized:
                if event.event_type in self._TERMINAL_EVENT_TYPES:
                    slog.error(
                        "duplicate_terminal_broadcast_suppressed",
                        run_id=run_id,
                        event_type=event.event_type,
                    )
                else:
                    slog.warning(
                        "post_terminal_broadcast_suppressed",
                        run_id=run_id,
                        event_type=event.event_type,
                    )
                return
            if event.event_type in self._TERMINAL_EVENT_TYPES:
                self._terminalized.add(run_id)
            sub_map = self._subscribers.get(run_id)
            if not sub_map:
                return
            states_to_schedule: list[_SubState] = []
            for state in sub_map.values():
                self._append_pending_locked(state, event)
                if not state.drain_scheduled:
                    state.drain_scheduled = True
                    states_to_schedule.append(state)
        for state in states_to_schedule:
            self._loop.call_soon_threadsafe(self._drain_pending, run_id, state)

    @staticmethod
    def _append_pending_locked(state: _SubState, event: RunEvent) -> None:
        """Append an event to the pending deque under the broadcaster lock.

        Drop-head policy mirrors ``_safe_put``:
          * Terminal event on full pending → clear pending, append terminal.
            (Terminals always drain the downstream async queue anyway, so
            losing non-terminals in pending is consistent with the final
            client-visible state.)
          * Non-terminal event on full pending → drop oldest, append new.
        """
        is_terminal = event.event_type in _TERMINAL_EVENT_TYPES
        if len(state.pending) >= _SUBSCRIBER_QUEUE_MAXSIZE:
            if is_terminal:
                state.pending.clear()
            else:
                state.pending.popleft()
        state.pending.append(event)

    def _drain_pending(self, run_id: str, state: _SubState) -> None:
        """Drain one batch of pending events for a subscriber. Runs on loop thread.

        Re-schedules itself via ``loop.call_soon`` while pending is
        non-empty, which yields to other loop callbacks between batches.
        The drain_scheduled flag stays True across re-schedules and is
        cleared only when pending is fully drained — this preserves the
        "at most one scheduled drain per subscriber" invariant that
        bounds the loop's _ready deque.
        """
        with self._lock:
            if not state.pending:
                state.drain_scheduled = False
                return
            batch = list(state.pending)
            state.pending.clear()
        for event in batch:
            self._safe_put(state.queue, event, run_id)
        # Re-enter via call_soon so other loop work (WS writes, timers)
        # can interleave between batches. drain_scheduled stays True.
        self._loop.call_soon(self._drain_pending, run_id, state)

    @staticmethod
    def _safe_put(queue: asyncio.Queue[RunEvent], event: RunEvent, run_id: str) -> None:
        """Put an event on the async queue, dropping oldest on backpressure.

        Runs on the event loop thread (invoked from _drain_pending, which
        is itself scheduled via call_soon_threadsafe / call_soon).

        Terminal events (completed, failed, cancelled) must never be dropped —
        a lost terminal event leaves WebSocket clients hanging forever. For
        terminal events, drain the entire queue if necessary to make room.
        """
        is_terminal = event.event_type in _TERMINAL_EVENT_TYPES

        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            if is_terminal:
                # Terminal events MUST be delivered. Drain queue to make room.
                drained = 0
                while not queue.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                        drained += 1
                if drained > 0:
                    slog.info("subscriber_queue_drained_for_terminal", run_id=run_id, drained=drained)
                queue.put_nowait(event)  # Queue is empty — this cannot fail
            else:
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # Drop oldest
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    slog.warning("subscriber_queue_drop", run_id=run_id)

    def cleanup_run(self, run_id: str) -> None:
        """Remove subscriber mapping for a completed/failed/cancelled run.

        Called from _run_pipeline's finally block after the terminal event
        has been scheduled via broadcast(). Does NOT destroy existing state
        objects — events already queued will still be drained by connected
        WS handlers. Only prevents new subscribers from being added for
        this run_id.

        Duplicate-terminal suppression is only needed while the run is
        active. Once cleanup runs, future listeners seed terminal state
        from the persisted Run record instead of this in-memory guard, so
        retaining run_id in ``_terminalized`` only leaks memory.
        """
        with self._lock:
            self._subscribers.pop(run_id, None)
            self._terminalized.discard(run_id)
