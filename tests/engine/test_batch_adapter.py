# tests/engine/test_batch_adapter.py
"""Tests for SharedBatchAdapter.

These tests verify the core multiplexing behavior:
- Single row waits correctly
- Multiple concurrent rows get correct results
- Out-of-order completion works
- Timeout behavior works
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.engine.batch_adapter import SharedBatchAdapter


@dataclass
class MockTokenInfo:
    """Minimal TokenInfo for testing emit()."""

    token_id: str
    row_id: int = 0
    row_data: dict[str, Any] = field(default_factory=dict)
    branch_name: str | None = None


class TestSharedBatchAdapter:
    """Tests for SharedBatchAdapter multiplexing behavior."""

    def test_single_row_wait(self) -> None:
        """Test waiting for a single row's result."""
        adapter = SharedBatchAdapter()

        # Register waiter with (token_id, state_id) key for retry safety
        waiter = adapter.register("token-1", "state-1")

        # Use Event for deterministic synchronization instead of sleep
        registration_complete = threading.Event()
        emit_allowed = threading.Event()

        def emit_when_signaled() -> None:
            registration_complete.wait()  # Wait for test setup to complete
            emit_allowed.wait()  # Wait for explicit signal
            token = MockTokenInfo(token_id="token-1", row_id=1)
            result = TransformResult.success({"output": "done"}, success_reason={"action": "test"})
            adapter.emit(token, result, "state-1")  # type: ignore[arg-type]

        thread = threading.Thread(target=emit_when_signaled)
        thread.start()

        # Signal that registration is complete
        registration_complete.set()

        # Signal emit is allowed (test is ready to receive)
        emit_allowed.set()

        # Wait for result
        result = waiter.wait(timeout=5.0)

        assert result.status == "success"
        assert result.row == {"output": "done"}

        thread.join()

    def test_multiple_concurrent_rows(self) -> None:
        """Test multiple rows waiting concurrently."""
        adapter = SharedBatchAdapter()

        # Register 3 waiters with unique state_ids
        waiter1 = adapter.register("token-1", "state-1")
        waiter2 = adapter.register("token-2", "state-2")
        waiter3 = adapter.register("token-3", "state-3")

        # Use Events for deterministic out-of-order completion
        setup_complete = threading.Event()
        emit_events = {
            "token-2": threading.Event(),  # Will be signaled first
            "token-1": threading.Event(),  # Will be signaled second
            "token-3": threading.Event(),  # Will be signaled third
        }

        def emit_results() -> None:
            setup_complete.wait()  # Wait for test setup

            # Emit in controlled order: token-2, token-1, token-3
            emit_events["token-2"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-2", row_id=2),  # type: ignore[arg-type]
                TransformResult.success({"value": 2}, success_reason={"action": "test"}),
                "state-2",
            )

            emit_events["token-1"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-1", row_id=1),  # type: ignore[arg-type]
                TransformResult.success({"value": 1}, success_reason={"action": "test"}),
                "state-1",
            )

            emit_events["token-3"].wait()
            adapter.emit(
                MockTokenInfo(token_id="token-3", row_id=3),  # type: ignore[arg-type]
                TransformResult.success({"value": 3}, success_reason={"action": "test"}),
                "state-3",
            )

        thread = threading.Thread(target=emit_results)
        thread.start()

        # Setup complete
        setup_complete.set()

        # Signal emits in out-of-order sequence
        emit_events["token-2"].set()  # First
        emit_events["token-1"].set()  # Second
        emit_events["token-3"].set()  # Third

        # Wait for results (each waiter gets correct result regardless of emit order)
        result1 = waiter1.wait(timeout=5.0)
        result2 = waiter2.wait(timeout=5.0)
        result3 = waiter3.wait(timeout=5.0)

        assert result1.row == {"value": 1}
        assert result2.row == {"value": 2}
        assert result3.row == {"value": 3}

        thread.join()

    def test_emit_before_wait(self) -> None:
        """Test that emit() before wait() still works correctly.

        This can happen if the worker processes very quickly.
        """
        adapter = SharedBatchAdapter()

        # Register waiter
        waiter = adapter.register("token-fast", "state-fast")

        # Emit result IMMEDIATELY (before wait is called)
        token = MockTokenInfo(token_id="token-fast", row_id=1)
        result = TransformResult.success({"fast": True}, success_reason={"action": "test"})
        adapter.emit(token, result, "state-fast")  # type: ignore[arg-type]

        # Now wait - should return immediately since event is already set
        start = time.perf_counter()
        got_result = waiter.wait(timeout=5.0)
        elapsed = time.perf_counter() - start

        assert got_result.status == "success"
        assert got_result.row == {"fast": True}
        assert elapsed < 0.1  # Should be nearly instant

    def test_timeout(self) -> None:
        """Test that wait() times out if result never arrives."""
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-never", "state-never")

        with pytest.raises(TimeoutError, match="No result received"):
            waiter.wait(timeout=0.1)

    def test_timeout_cleans_up_waiter_entry(self) -> None:
        """Test that timeout cleans up waiter entry to prevent memory leak.

        When wait() times out, the waiter entry must be removed from _waiters.
        Otherwise, late results arriving via emit() would still find the entry
        and store results in _results that no one will ever retrieve.
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-timeout", "state-timeout")

        # Waiter entry should exist before timeout
        assert ("token-timeout", "state-timeout") in adapter._waiters

        # Timeout
        with pytest.raises(TimeoutError):
            waiter.wait(timeout=0.05)

        # After timeout, waiter entry must be cleaned up
        assert ("token-timeout", "state-timeout") not in adapter._waiters

    def test_late_result_after_timeout_not_stored(self) -> None:
        """Test that late results after timeout are discarded, not stored.

        This is the core memory leak bug: if a waiter times out but the
        worker eventually completes and calls emit(), the result should
        NOT be stored in _results (since no one will retrieve it).

        Scenario:
        1. Register waiter, wait times out
        2. Late result arrives via emit()
        3. Result should NOT be stored (no leak)
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-late", "state-late")

        # Timeout
        with pytest.raises(TimeoutError):
            waiter.wait(timeout=0.05)

        # Late result arrives after timeout
        adapter.emit(
            MockTokenInfo(token_id="token-late", row_id=1),  # type: ignore[arg-type]
            TransformResult.success({"late": "result"}, success_reason={"action": "test"}),
            "state-late",
        )

        # Result must NOT be stored (would be a memory leak)
        assert ("token-late", "state-late") not in adapter._results
        assert len(adapter._results) == 0

    def test_error_result_propagated(self) -> None:
        """Test that error results are correctly propagated."""
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-error", "state-error")

        # Emit error result
        token = MockTokenInfo(token_id="token-error", row_id=1)
        result = TransformResult.error({"reason": "llm_call_failed", "error": "API down"})
        adapter.emit(token, result, "state-error")  # type: ignore[arg-type]

        got_result = waiter.wait(timeout=1.0)

        assert got_result.status == "error"
        assert got_result.reason == {"reason": "llm_call_failed", "error": "API down"}

    def test_concurrent_waiters_in_parallel_threads(self) -> None:
        """Test multiple threads waiting concurrently."""
        adapter = SharedBatchAdapter()
        results: dict[str, TransformResult] = {}
        errors: list[Exception] = []
        all_registered = threading.Barrier(6)  # 5 waiters + 1 emitter thread

        def wait_for_token(token_id: str, state_id: str) -> None:
            try:
                waiter = adapter.register(token_id, state_id)
                all_registered.wait()  # Synchronize: all threads registered
                result = waiter.wait(timeout=5.0)
                results[token_id] = result
            except Exception as e:
                errors.append(e)

        # Start 5 waiter threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=wait_for_token, args=(f"token-{i}", f"state-{i}"))
            threads.append(t)
            t.start()

        def emit_all() -> None:
            all_registered.wait()  # Wait for all waiters to register
            # Emit all results
            for i in range(5):
                adapter.emit(
                    MockTokenInfo(token_id=f"token-{i}", row_id=i),  # type: ignore[arg-type]
                    TransformResult.success({"index": i}, success_reason={"action": "test"}),
                    f"state-{i}",
                )

        emit_thread = threading.Thread(target=emit_all)
        emit_thread.start()

        # Wait for all threads
        for t in threads:
            t.join(timeout=5.0)
        emit_thread.join(timeout=5.0)

        # Verify all succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        for i in range(5):
            assert results[f"token-{i}"].row == {"index": i}

    def test_clear(self) -> None:
        """Test that clear() removes all state."""
        adapter = SharedBatchAdapter()

        # Register some waiters
        adapter.register("token-1", "state-1")
        adapter.register("token-2", "state-2")

        # Emit a result that won't be consumed (orphaned due to no matching waiter key)
        adapter.emit(
            MockTokenInfo(token_id="token-orphan", row_id=99),  # type: ignore[arg-type]
            TransformResult.success({"orphan": True}, success_reason={"action": "test"}),
            "state-orphan",
        )

        # Clear
        adapter.clear()

        # Verify internal state is empty
        assert len(adapter._waiters) == 0
        assert len(adapter._results) == 0

    def test_stale_result_not_delivered_to_retry(self) -> None:
        """Test that results from timed-out attempts don't interfere with retries.

        This is the key retry safety test. When:
        1. First attempt registers with state_id="attempt-1"
        2. First attempt times out
        3. Retry registers with state_id="attempt-2"
        4. First worker finally finishes and emits with state_id="attempt-1"
        5. Retry worker finishes and emits with state_id="attempt-2"

        The retry's waiter should get the retry's result, not the stale one.
        """
        adapter = SharedBatchAdapter()

        # Simulate first attempt timing out - waiter times out, but worker hasn't finished
        waiter1 = adapter.register("token-42", "attempt-1")
        # waiter1.wait() would timeout here in real scenario

        # Retry registers with new state_id
        waiter2 = adapter.register("token-42", "attempt-2")

        # First worker finishes late (after timeout), emits with old state_id
        adapter.emit(
            MockTokenInfo(token_id="token-42", row_id=42),  # type: ignore[arg-type]
            TransformResult.success({"result": "stale"}, success_reason={"action": "test"}),
            "attempt-1",
        )

        # Retry worker finishes, emits with retry state_id
        adapter.emit(
            MockTokenInfo(token_id="token-42", row_id=42),  # type: ignore[arg-type]
            TransformResult.success({"result": "fresh"}, success_reason={"action": "test"}),
            "attempt-2",
        )

        # First waiter gets the stale result (it's still registered)
        result1 = waiter1.wait(timeout=1.0)
        assert result1.row == {"result": "stale"}

        # Retry waiter gets the fresh result (correct behavior!)
        result2 = waiter2.wait(timeout=1.0)
        assert result2.row == {"result": "fresh"}

    def test_timeout_race_cleans_up_late_result(self) -> None:
        """Test that timeout path cleans up results stored during race window.

        This tests the TOCTOU race between wait() timeout and emit():

        Timeline where the race leaks memory:
            Thread A (wait)                  Thread B (emit)
            -----------------                -----------------
            1. event.wait(timeout) → False
                                             2. Acquires _lock
                                             3. key in _waiters → True
                                             4. _results[key] = result
                                             5. _waiters[key].set()
                                             6. del _waiters[key]
                                             7. Releases _lock
            8. Acquires _lock
            9. _waiters.pop() → None (gone!)
            10. Releases _lock
            11. Raises TimeoutError

            Result: _results[key] is never cleaned up → memory leak

        The fix: timeout path must also clean up _results[key].

        Since timing races are non-deterministic, we test the fix directly:
        simulate the post-race state and verify cleanup happens.
        """
        adapter = SharedBatchAdapter()
        key = ("token-race", "state-race")

        # Register waiter
        waiter = adapter.register(*key)

        # Simulate the race: emit() wins, stores result and removes waiter
        # This is what happens when emit() executes between event.wait() timeout
        # and wait()'s lock acquisition
        with adapter._lock:
            adapter._results[key] = TransformResult.success({"race": "leaked"}, success_reason={"action": "test"})
            adapter._waiters[key].set()  # Signal (even though we're about to timeout)
            del adapter._waiters[key]  # emit() removes waiter entry

        # Now the event IS set (by our simulated emit), but we want to test
        # timeout behavior. Re-create the waiter without event being set:
        adapter._waiters[key] = threading.Event()  # Fresh unset event
        waiter._event = adapter._waiters[key]  # Update waiter's event reference

        # Now wait() will timeout because event is not set
        # The timeout path should clean up both _waiters AND _results
        with pytest.raises(TimeoutError):
            waiter.wait(timeout=0.05)

        # Verify BOTH are cleaned up - no memory leak
        assert key not in adapter._waiters, "Waiter entry should be removed"
        assert key not in adapter._results, (
            "Result entry should be removed to prevent memory leak. "
            "The race between timeout and emit can store a result that "
            "no one will ever retrieve."
        )
