"""Unit tests for IdleTimeoutPump (elspeth-735df9576d).

Deterministic threaded tests: every cross-thread rendezvous is a
``threading.Event`` handshake with a generous bounded wait — never a bare
sleep race. Timing-parity, serialization (join-equivalence), error-precedence,
and thread-churn behaviour are each pinned structurally: the assertions hold
by the pump's handshake guarantees, not by winning a race.
"""

from __future__ import annotations

import threading

import pytest

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.orchestrator.idle_timeout_pump import IdleTimeoutPump

# Generous bounded wait for cross-thread handshakes; tests fail loudly on
# timeout instead of hanging.
_WAIT = 5.0
# Poll interval for tests that need flushes to fire quickly.
_FAST = 0.002
# Poll interval for tests that must observe ZERO flushes: any flush would
# require this long an idle window, which the test never provides.
_SLOW = 30.0


def _alive_pump_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "idle-timeout-pump" and t.is_alive()]


class TestIdleFlushing:
    def test_flushes_fire_repeatedly_while_fetch_is_blocked(self) -> None:
        """The worker flushes every poll interval while the fetch callable blocks."""
        count_lock = threading.Lock()
        flush_count = 0
        two_flushes = threading.Event()

        def flush() -> None:
            nonlocal flush_count
            with count_lock:
                flush_count += 1
                if flush_count >= 2:
                    two_flushes.set()

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def blocked_fetch() -> str:
                assert two_flushes.wait(_WAIT), "expected at least two idle flushes while fetch blocked"
                return "row"

            assert pump.fetch(blocked_fetch) == "row"
        finally:
            pump.stop()
        assert flush_count >= 2

    def test_fetch_faster_than_poll_interval_sees_zero_flushes(self) -> None:
        """One full interval elapses before the first flush — a fast fetch flushes nothing."""
        flush_calls: list[int] = []
        pump = IdleTimeoutPump(flush=lambda: flush_calls.append(1), poll_interval=_SLOW)
        pump.start()
        try:
            assert pump.fetch(lambda: "row") == "row"
        finally:
            pump.stop()
        assert flush_calls == []

    def test_no_flush_runs_after_fetch_returns(self) -> None:
        """The park handshake guarantees a returned fetch leaves no flushing behind."""
        flush_count = 0
        first_flush = threading.Event()

        def flush() -> None:
            nonlocal flush_count
            flush_count += 1
            first_flush.set()

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def blocked_fetch() -> str:
                assert first_flush.wait(_WAIT)
                return "row"

            assert pump.fetch(blocked_fetch) == "row"
            # fetch() only returns once the worker has parked, so this read is
            # the final count: a parked worker cannot flush again without a
            # new fetch, and none follows.
            count_after_fetch = flush_count
        finally:
            pump.stop()
        assert flush_count == count_after_fetch

    def test_sequential_fetches_reuse_one_worker_thread(self) -> None:
        """No per-fetch thread churn: both fetches are served by the same worker."""
        idents_lock = threading.Lock()
        flush_idents: list[int] = []
        flush_seen = threading.Event()

        def flush() -> None:
            with idents_lock:
                flush_idents.append(threading.get_ident())
            flush_seen.set()

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def blocked_fetch() -> str:
                assert flush_seen.wait(_WAIT)
                return "row"

            assert pump.fetch(blocked_fetch) == "row"
            flush_seen.clear()  # worker is parked here; no flush can race this
            assert pump.fetch(blocked_fetch) == "row"
        finally:
            pump.stop()
        assert len(flush_idents) >= 2
        assert len(set(flush_idents)) == 1, f"expected one persistent worker, saw idents {set(flush_idents)}"
        assert threading.get_ident() not in set(flush_idents)


class TestSerializationHandshake:
    def test_fetch_does_not_return_while_flush_is_in_flight(self) -> None:
        """Join-equivalence: fetch() blocks until an in-flight flush completes."""
        order: list[str] = []
        flush_entered = threading.Event()
        flush_release = threading.Event()
        flush_ran_once = threading.Event()
        fetch_done = threading.Event()
        results: list[str] = []

        def flush() -> None:
            if flush_ran_once.is_set():
                return
            flush_ran_once.set()
            flush_entered.set()
            assert flush_release.wait(_WAIT)
            order.append("flush_finished")

        def fetch_fn() -> str:
            assert flush_entered.wait(_WAIT)
            return "row"

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()

        def run_fetch() -> None:
            results.append(pump.fetch(fetch_fn))
            order.append("fetch_returned")
            fetch_done.set()

        fetcher = threading.Thread(target=run_fetch, daemon=True)
        try:
            fetcher.start()
            assert flush_entered.wait(_WAIT)
            # The fetch callable has returned (or is about to), but fetch()
            # itself CANNOT complete while the flush is mid-flight: its park
            # handshake structurally requires the flush to finish first. If
            # this assertion ever fails, the serialization edge is broken.
            assert not fetch_done.is_set()
            flush_release.set()
            assert fetch_done.wait(_WAIT)
        finally:
            pump.stop()
            fetcher.join(_WAIT)
        assert results == ["row"]
        assert order.index("flush_finished") < order.index("fetch_returned")


class TestErrorPropagation:
    def test_flush_error_surfaces_from_fetch_even_when_fn_returns_row(self) -> None:
        flush_attempts = 0
        flush_raised = threading.Event()

        def flush() -> None:
            nonlocal flush_attempts
            flush_attempts += 1
            flush_raised.set()
            raise RuntimeError("idle-flush-boom")

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def blocked_fetch() -> str:
                assert flush_raised.wait(_WAIT)
                return "row"

            with pytest.raises(RuntimeError, match="idle-flush-boom"):
                pump.fetch(blocked_fetch)
        finally:
            pump.stop()
        # First error wins structurally: recording it parks the worker until
        # stop(), so no second flush attempt can ever run.
        assert flush_attempts == 1

    def test_flush_error_beats_fetch_callable_exception(self) -> None:
        flush_raised = threading.Event()

        def flush() -> None:
            flush_raised.set()
            raise RuntimeError("flush-error-wins")

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def failing_fetch() -> str:
                assert flush_raised.wait(_WAIT)
                raise ValueError("source-error-loses")

            with pytest.raises(RuntimeError, match="flush-error-wins"):
                pump.fetch(failing_fetch)
        finally:
            pump.stop()

    def test_flush_error_beats_stop_iteration(self) -> None:
        flush_raised = threading.Event()

        def flush() -> None:
            flush_raised.set()
            raise RuntimeError("flush-error-beats-exhaustion")

        pump = IdleTimeoutPump(flush=flush, poll_interval=_FAST)
        pump.start()
        try:

            def exhausted_fetch() -> str:
                assert flush_raised.wait(_WAIT)
                raise StopIteration

            with pytest.raises(RuntimeError, match="flush-error-beats-exhaustion"):
                pump.fetch(exhausted_fetch)
        finally:
            pump.stop()

    def test_fetch_callable_exception_is_reraised(self) -> None:
        flush_calls: list[int] = []
        pump = IdleTimeoutPump(flush=lambda: flush_calls.append(1), poll_interval=_SLOW)
        pump.start()
        try:

            def failing_fetch() -> str:
                raise ValueError("source-boom")

            with pytest.raises(ValueError, match="source-boom"):
                pump.fetch(failing_fetch)
        finally:
            pump.stop()
        assert flush_calls == []

    def test_stop_iteration_from_fetch_callable_is_reraised(self) -> None:
        flush_calls: list[int] = []
        pump = IdleTimeoutPump(flush=lambda: flush_calls.append(1), poll_interval=_SLOW)
        pump.start()
        try:

            def exhausted_fetch() -> str:
                raise StopIteration

            with pytest.raises(StopIteration):
                pump.fetch(exhausted_fetch)
        finally:
            pump.stop()
        assert flush_calls == []


class TestLifecycleInvariants:
    def test_stop_is_idempotent_and_joins_worker(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_FAST)
        pump.start()
        assert len(_alive_pump_threads()) == 1
        pump.stop()
        assert _alive_pump_threads() == []
        pump.stop()
        assert _alive_pump_threads() == []

    def test_stop_without_start_is_safe(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_FAST)
        pump.stop()
        assert _alive_pump_threads() == []

    def test_start_twice_is_an_invariant_violation(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_FAST)
        pump.start()
        try:
            with pytest.raises(OrchestrationInvariantError, match="more than once"):
                pump.start()
        finally:
            pump.stop()

    def test_fetch_before_start_is_an_invariant_violation(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_FAST)
        with pytest.raises(OrchestrationInvariantError, match="before start"):
            pump.fetch(lambda: "row")

    def test_fetch_after_stop_is_an_invariant_violation(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_FAST)
        pump.start()
        pump.stop()
        with pytest.raises(OrchestrationInvariantError, match="after stop"):
            pump.fetch(lambda: "row")

    def test_reentrant_fetch_is_an_invariant_violation(self) -> None:
        pump = IdleTimeoutPump(flush=lambda: None, poll_interval=_SLOW)
        pump.start()
        try:

            def reentering_fetch() -> str:
                return pump.fetch(lambda: "inner")

            with pytest.raises(OrchestrationInvariantError, match="re-entered"):
                pump.fetch(reentering_fetch)
        finally:
            pump.stop()
