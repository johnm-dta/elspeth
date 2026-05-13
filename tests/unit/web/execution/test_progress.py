"""Tests for ProgressBroadcaster — thread-safe async event delivery.

These tests verify the B1 fix: broadcast() uses loop.call_soon_threadsafe()
to safely push events from a background thread into asyncio queues.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import (
    CancelledData,
    CompletedData,
    ErrorData,
    FailedData,
    ProgressData,
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunEvent,
    RunEventType,
)

_EventData = ProgressData | ErrorData | CompletedData | CancelledData | FailedData


def _accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=10),
        tokens=RunAccountingTokens(
            emitted=10,
            terminal=10,
            succeeded=10,
            failed=0,
            structural=0,
            pending=0,
        ),
        routing=RunAccountingRouting(
            routed_success=0,
            routed_failure=0,
            quarantined=0,
            discarded=0,
        ),
        integrity=RunAccountingIntegrity(
            closure="closed",
            missing_terminal_outcomes=0,
            duplicate_terminal_outcomes=0,
        ),
    )


def _progress_data(value: int = 10) -> ProgressData:
    return ProgressData(
        source_rows_processed=value,
        tokens_succeeded=value,
        tokens_failed=0,
        tokens_quarantined=0,
        tokens_routed_success=0,
        tokens_routed_failure=0,
    )


def _cancelled_data() -> CancelledData:
    return CancelledData(
        source_rows_processed=10,
        tokens_succeeded=10,
        tokens_failed=0,
        tokens_quarantined=0,
        tokens_routed_success=0,
        tokens_routed_failure=0,
    )


# Typed as the concrete union so mypy resolves RunEvent(data=...) rather than
# widening to the _StrictResponse base; also ensures the lambdas stay in
# lockstep with RunEvent's discriminated union.
_EVENT_DATA: dict[str, Callable[[], _EventData]] = {
    "progress": lambda: _progress_data(),
    "error": lambda: ErrorData(message="test error", node_id=None, row_id=None),
    "completed": lambda: CompletedData(
        status="completed",
        accounting=_accounting(),
        landscape_run_id="lscape-1",
    ),
    "cancelled": lambda: _cancelled_data(),
    "failed": lambda: FailedData(detail="test failure", node_id=None),
}


def _make_event(
    run_id: str = "run-1",
    event_type: RunEventType = "progress",
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        timestamp=datetime.now(tz=UTC),
        event_type=event_type,
        data=_EVENT_DATA[event_type](),
    )


class TestProgressBroadcasterSubscription:
    def test_subscribe_returns_queue(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue = broadcaster.subscribe("run-1")
            assert isinstance(queue, asyncio.Queue)
        finally:
            loop.close()

    def test_unsubscribe_removes_queue(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue = broadcaster.subscribe("run-1")
            broadcaster.unsubscribe("run-1", queue)
            # Verify subscriber set is empty after unsubscribe
            assert len(broadcaster._subscribers.get("run-1", set())) == 0
        finally:
            loop.close()

    def test_unsubscribe_unknown_run_is_noop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue: asyncio.Queue[RunEvent] = asyncio.Queue()
            # Should not raise
            broadcaster.unsubscribe("nonexistent", queue)
        finally:
            loop.close()

    def test_multiple_subscribers_for_same_run(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            q1 = broadcaster.subscribe("run-1")
            q2 = broadcaster.subscribe("run-1")
            assert q1 is not q2
            assert len(broadcaster._subscribers["run-1"]) == 2
        finally:
            loop.close()


class TestProgressBroadcasterThreadSafety:
    """Verify B1 fix: broadcast() uses loop.call_soon_threadsafe()."""

    def test_broadcast_uses_call_soon_threadsafe(self) -> None:
        """broadcast() MUST use loop.call_soon_threadsafe, not direct put_nowait.

        After the coalesced-drain fix, the scheduled callable is
        ``_drain_pending(run_id, state)`` rather than
        ``_safe_put(queue, event, run_id)``. The thread-safety contract
        (no direct put_nowait from producer thread) is preserved —
        _drain_pending runs on the loop thread and is responsible for
        invoking _safe_put.
        """
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        assert mock_loop.call_soon_threadsafe.call_count == 1
        call_args = mock_loop.call_soon_threadsafe.call_args
        # The scheduled callable must be _drain_pending bound to this
        # broadcaster instance — anything else would either bypass the
        # coalescing bound (direct _safe_put) or corrupt loop internals.
        assert call_args.args[0] == broadcaster._drain_pending
        # Args carry the run_id and the subscriber state, not the event
        # itself — the event lives in state.pending.
        assert call_args.args[1] == "run-1"

    def test_broadcast_to_multiple_subscribers(self) -> None:
        """Each subscriber queue gets its own call_soon_threadsafe call."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")
        broadcaster.subscribe("run-1")
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        assert mock_loop.call_soon_threadsafe.call_count == 2

    def test_broadcast_no_subscribers_is_noop(self) -> None:
        """Broadcasting with no subscribers does not raise."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        event = _make_event()

        # Should not raise
        broadcaster.broadcast("run-1", event)
        mock_loop.call_soon_threadsafe.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_from_background_thread_delivers_events(self) -> None:
        """End-to-end: background thread broadcasts, async consumer receives.

        This is the actual B1 scenario — _run_pipeline() calls broadcast()
        from a ThreadPoolExecutor worker thread, and the WebSocket handler
        awaits queue.get() on the event loop thread.
        """
        loop = asyncio.get_running_loop()
        broadcaster = ProgressBroadcaster(loop)
        queue = broadcaster.subscribe("run-1")
        event = _make_event()

        def background_broadcast() -> None:
            """Simulate _run_pipeline() calling broadcast from worker thread."""
            broadcaster.broadcast("run-1", event)

        # Run broadcast in a real background thread
        thread = threading.Thread(target=background_broadcast)
        thread.start()
        thread.join(timeout=5.0)

        # The event should be in the queue via call_soon_threadsafe
        received = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert received.run_id == "run-1"
        assert received.event_type == "progress"
        assert isinstance(received.data, ProgressData)
        assert received.data.source_rows_processed == 10

    @pytest.mark.asyncio
    async def test_multiple_broadcasts_from_thread_arrive_in_order(self) -> None:
        """Events broadcast from a single thread arrive in FIFO order."""
        loop = asyncio.get_running_loop()
        broadcaster = ProgressBroadcaster(loop)
        queue = broadcaster.subscribe("run-1")

        events = [
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="progress",
                data=_progress_data(i),
            )
            for i in range(5)
        ]

        def background_broadcast() -> None:
            for event in events:
                broadcaster.broadcast("run-1", event)

        thread = threading.Thread(target=background_broadcast)
        thread.start()
        thread.join(timeout=5.0)

        received = []
        for _ in range(5):
            item = await asyncio.wait_for(queue.get(), timeout=5.0)
            assert isinstance(item.data, ProgressData)
            received.append(item.data.source_rows_processed)

        assert received == [0, 1, 2, 3, 4]

    def test_broadcast_after_unsubscribe_does_not_deliver(self) -> None:
        """After unsubscribe, broadcast does not target the removed queue."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        queue = broadcaster.subscribe("run-1")
        broadcaster.unsubscribe("run-1", queue)
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        mock_loop.call_soon_threadsafe.assert_not_called()


class TestProgressBroadcasterBackpressure:
    """Verify backpressure: bounded queue + drop-head on QueueFull."""

    def test_subscribe_creates_bounded_queue(self) -> None:
        """Subscriber queues must have maxsize to prevent OOM."""
        from elspeth.web.execution.progress import _SUBSCRIBER_QUEUE_MAXSIZE

        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue = broadcaster.subscribe("run-1")
            assert queue.maxsize == _SUBSCRIBER_QUEUE_MAXSIZE
        finally:
            loop.close()

    def test_safe_put_succeeds_when_queue_has_room(self) -> None:
        """Normal case: event is placed on the queue."""
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=10)
        event = _make_event()

        ProgressBroadcaster._safe_put(queue, event, "run-1")

        assert queue.qsize() == 1

    def test_safe_put_drops_oldest_when_full_progress(self) -> None:
        """When queue is full, oldest progress event is evicted and new one inserted."""
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=3)
        events = [_make_event(event_type="progress") for _ in range(3)]
        for e in events:
            queue.put_nowait(e)
        assert queue.full()

        new_event = _make_event(event_type="progress")
        ProgressBroadcaster._safe_put(queue, new_event, "run-1")

        # Queue still at maxsize, oldest was evicted
        assert queue.qsize() == 3
        # The first item should be events[1] (events[0] was evicted)
        first = queue.get_nowait()
        assert first is events[1]
        # Last item should be our new event
        _ = queue.get_nowait()  # events[2]
        last = queue.get_nowait()
        assert last is new_event

    def test_terminal_event_drains_full_queue(self) -> None:
        """Terminal events drain the queue to guarantee delivery.

        Bug: elspeth-ca156c71b7 — terminal events (completed, failed, cancelled)
        must never be silently dropped, or WS clients hang forever.
        """
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=3)
        events = [_make_event(event_type="progress") for _ in range(3)]
        for e in events:
            queue.put_nowait(e)
        assert queue.full()

        terminal = _make_event(event_type="completed")
        ProgressBroadcaster._safe_put(queue, terminal, "run-1")

        # Queue was drained then terminal inserted — only terminal event remains
        assert queue.qsize() == 1
        assert queue.get_nowait() is terminal

    def test_failed_event_drains_full_queue(self) -> None:
        """Failed events also drain the queue (terminal)."""
        queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=3)
        for _ in range(3):
            queue.put_nowait(_make_event(event_type="progress"))

        failed = _make_event(event_type="failed")
        ProgressBroadcaster._safe_put(queue, failed, "run-1")

        assert queue.qsize() == 1
        assert queue.get_nowait() is failed


class TestProgressBroadcasterCallbackBacklogBound:
    """Regression: coalesced cross-thread scheduling.

    The advertised OOM guard (``_SUBSCRIBER_QUEUE_MAXSIZE``) bounds the
    async queue, but without coalescing, each ``broadcast()`` would enqueue
    a fresh ``call_soon_threadsafe()`` callback into the event loop's
    internal ``_ready`` deque — one per event per subscriber. A slow or
    stalled loop turns that deque into an unbounded buffer, pinning every
    ``RunEvent`` in memory long before the "bounded" queue drops anything.

    Fix contract: broadcast() must cap scheduled callbacks at one per
    subscriber at any moment. The per-subscriber pending buffer carries the
    backlog and applies the drop-head policy at broadcast time, not after
    the cross-thread handoff.
    """

    def test_callback_backlog_is_bounded_when_loop_is_stalled(self) -> None:
        """10k broadcasts with a stalled loop schedule at most one drain per subscriber.

        This is the core regression guard for the ticket. Against the old
        implementation, ``call_soon_threadsafe`` was invoked once per event
        (10000 times); the fix coalesces to one scheduled drain per
        subscriber until that drain actually runs on the loop thread.
        """
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")

        event_count = 10_000
        for i in range(event_count):
            broadcaster.broadcast(
                "run-1",
                RunEvent(
                    run_id="run-1",
                    timestamp=datetime.now(tz=UTC),
                    event_type="progress",
                    data=_progress_data(i),
                ),
            )

        # One subscriber → at most one scheduled drain callback pending.
        assert mock_loop.call_soon_threadsafe.call_count == 1, (
            f"Expected 1 scheduled drain callback for the stalled loop; got "
            f"{mock_loop.call_soon_threadsafe.call_count}. The loop's _ready "
            f"deque is not bounded by _SUBSCRIBER_QUEUE_MAXSIZE, so one "
            f"scheduled callback per event defeats the OOM guard."
        )

    def test_pending_buffer_is_bounded_at_subscriber_maxsize(self) -> None:
        """The pending pre-loop buffer must obey _SUBSCRIBER_QUEUE_MAXSIZE.

        If pending grew without bound, the fix would merely shift the
        unbounded-buffer problem from the loop's _ready deque into our own
        bookkeeping. Drop-head policy must apply at broadcast time.
        """
        from elspeth.web.execution.progress import _SUBSCRIBER_QUEUE_MAXSIZE

        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")

        # Emit 3x the cap; pending must never exceed the cap.
        for i in range(_SUBSCRIBER_QUEUE_MAXSIZE * 3):
            broadcaster.broadcast(
                "run-1",
                RunEvent(
                    run_id="run-1",
                    timestamp=datetime.now(tz=UTC),
                    event_type="progress",
                    data=_progress_data(i),
                ),
            )

        # Look up the one subscriber state. Whitebox read justified: the
        # invariant ("pending stays bounded") is the exact thing being tested.
        sub_map = broadcaster._subscribers["run-1"]
        state = next(iter(sub_map.values()))
        assert len(state.pending) <= _SUBSCRIBER_QUEUE_MAXSIZE, (
            f"Pending pre-loop buffer grew to {len(state.pending)} — "
            f"exceeds _SUBSCRIBER_QUEUE_MAXSIZE={_SUBSCRIBER_QUEUE_MAXSIZE}. "
            f"Drop-head policy must run at broadcast time."
        )

    def test_coalesced_drain_delivers_all_buffered_events_in_order(self) -> None:
        """Running the scheduled drain callback must deliver the full pending batch.

        Coalescing scheduled callbacks is only safe if the single drain
        callback consumes the entire pending buffer. This test runs the
        scheduled drain and verifies all (unthrottled, post-drop-head)
        events reach the subscriber queue in FIFO order.
        """
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        queue = broadcaster.subscribe("run-1")

        events = [
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="progress",
                data=_progress_data(i),
            )
            for i in range(5)
        ]
        for event in events:
            broadcaster.broadcast("run-1", event)

        # The first and only call should schedule _drain_pending with
        # (run_id, state) arguments. Invoke it synchronously to simulate
        # the loop servicing the callback.
        assert mock_loop.call_soon_threadsafe.call_count == 1
        call_args = mock_loop.call_soon_threadsafe.call_args
        drain_fn = call_args.args[0]
        drain_args = call_args.args[1:]
        drain_fn(*drain_args)

        # All 5 events delivered in order via _safe_put → asyncio.Queue.
        received = [queue.get_nowait() for _ in range(5)]
        assert [r.data.source_rows_processed for r in received if isinstance(r.data, ProgressData)] == [0, 1, 2, 3, 4]

    def test_drain_clears_schedule_flag_so_next_broadcast_reschedules(self) -> None:
        """After the drain callback runs, the next broadcast must schedule again.

        The coalescing flag must flip back to False once pending is drained;
        otherwise a second burst of events would sit in pending with no
        drain ever scheduled — silent loss under any real load pattern.
        """
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")
        event = _make_event()

        broadcaster.broadcast("run-1", event)
        assert mock_loop.call_soon_threadsafe.call_count == 1

        # Drive the scheduled drain to completion. The drain may re-schedule
        # itself via loop.call_soon; we drive that too, since in real
        # operation the loop would keep servicing until pending is empty.
        while mock_loop.call_soon_threadsafe.called or mock_loop.call_soon.called:
            pending_calls = []
            if mock_loop.call_soon_threadsafe.called:
                pending_calls.extend(list(mock_loop.call_soon_threadsafe.call_args_list))
                mock_loop.call_soon_threadsafe.reset_mock()
            if mock_loop.call_soon.called:
                pending_calls.extend(list(mock_loop.call_soon.call_args_list))
                mock_loop.call_soon.reset_mock()
            for call in pending_calls:
                call.args[0](*call.args[1:])

        # Second broadcast after the drain has fully drained must schedule.
        broadcaster.broadcast("run-1", event)
        assert mock_loop.call_soon_threadsafe.call_count == 1, (
            "After drain fully drained pending, the next broadcast must "
            "schedule a new drain — otherwise future events would never reach "
            "the async queue."
        )


class TestProgressBroadcasterCleanup:
    def test_cleanup_run_removes_all_subscribers(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            broadcaster.subscribe("run-1")
            broadcaster.subscribe("run-1")
            broadcaster.cleanup_run("run-1")
            assert "run-1" not in broadcaster._subscribers
        finally:
            loop.close()

    def test_cleanup_unknown_run_is_noop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            broadcaster.cleanup_run("nonexistent")  # Should not raise
        finally:
            loop.close()


class TestBroadcasterSingleTerminalGuard:
    """Defense-in-depth: once a terminal event has been broadcast for a run,
    subsequent terminal broadcasts for that run are dropped with a warning.

    Bug: elspeth-25df1be367 — contradictory terminal events (completed then
    failed). The broadcaster must enforce "at most one terminal per run" as
    a policy, not rely on callers to get ordering right.
    """

    def test_duplicate_terminal_is_suppressed(self) -> None:
        """Second terminal broadcast for the same run must not reach subscribers."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(loop)
        broadcaster.subscribe("run-1")

        completed = _make_event(run_id="run-1", event_type="completed")
        failed = _make_event(run_id="run-1", event_type="failed")

        broadcaster.broadcast("run-1", completed)
        calls_after_first = loop.call_soon_threadsafe.call_count

        broadcaster.broadcast("run-1", failed)
        assert loop.call_soon_threadsafe.call_count == calls_after_first, (
            "Duplicate terminal event must be suppressed — only the first terminal per run should reach call_soon_threadsafe."
        )

    def test_non_terminal_after_terminal_is_suppressed(self) -> None:
        """Once a terminal event is accepted, later events for that run are dropped.

        This is the defensive boundary for bad producer ordering. If a
        later progress event is still accepted, it can evict an already
        queued terminal under backpressure and leave the WebSocket client
        hanging forever.
        """
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(loop)
        queue = broadcaster.subscribe("run-1")
        queue._maxsize = 1

        completed = _make_event(run_id="run-1", event_type="completed")
        progress = _make_event(run_id="run-1", event_type="progress")

        broadcaster.broadcast("run-1", completed)
        broadcaster.broadcast("run-1", progress)

        # Drive any scheduled drain callbacks manually (MagicMock loop
        # does not service them). Run until no further callbacks pend.
        def _drive(mock_loop: MagicMock) -> int:
            pending_calls = []
            if mock_loop.call_soon_threadsafe.called:
                pending_calls.extend(list(mock_loop.call_soon_threadsafe.call_args_list))
                mock_loop.call_soon_threadsafe.reset_mock()
            if mock_loop.call_soon.called:
                pending_calls.extend(list(mock_loop.call_soon.call_args_list))
                mock_loop.call_soon.reset_mock()
            for call in pending_calls:
                call.args[0](*call.args[1:])
            return len(pending_calls)

        while _drive(loop) > 0:
            pass

        delivered = []
        while not queue.empty():
            delivered.append(queue.get_nowait())
        assert [e.event_type for e in delivered] == ["completed"], (
            "Post-terminal events must be suppressed so a queued terminal cannot be evicted under backpressure."
        )

    def test_cleanup_run_discards_terminal_tracking(self) -> None:
        """cleanup_run must remove terminal tracking for completed runs."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(loop)
        broadcaster.subscribe("run-1")

        completed = _make_event(run_id="run-1", event_type="completed")
        broadcaster.broadcast("run-1", completed)

        assert "run-1" in broadcaster._terminalized, "Terminal tracking state must be present after terminal broadcast."

        broadcaster.cleanup_run("run-1")
        assert "run-1" not in broadcaster._terminalized, (
            "cleanup_run must discard terminal tracking after subscribers are removed so the set does not grow for process lifetime."
        )

    def test_different_runs_have_independent_terminal_tracking(self) -> None:
        """Terminal guard must be per-run, not global."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(loop)
        broadcaster.subscribe("run-1")
        broadcaster.subscribe("run-2")

        c1 = _make_event(run_id="run-1", event_type="completed")
        c2 = _make_event(run_id="run-2", event_type="completed")

        broadcaster.broadcast("run-1", c1)
        broadcaster.broadcast("run-2", c2)
        assert loop.call_soon_threadsafe.call_count == 2, "Different runs must have independent terminal guards."
