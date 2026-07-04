# tests/unit/engine/test_batch_adapter.py
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

import pytest

from elspeth.contracts import TokenInfo, TransformResult
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.batch_adapter import SharedBatchAdapter, _WaiterEntry
from elspeth.testing import make_pipeline_row


def _make_token(token_id: str, row_id: str = "row-1") -> TokenInfo:
    """Create a TokenInfo with proper PipelineRow for testing."""
    return TokenInfo(
        token_id=token_id,
        row_id=row_id,
        row_data=make_pipeline_row({}),
    )


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
            token = _make_token("token-1", "row-1")
            result = TransformResult.success(make_pipeline_row({"output": "done"}), success_reason={"action": "test"})
            adapter.emit(token, result, "state-1")

        thread = threading.Thread(target=emit_when_signaled)
        thread.start()

        # Signal that registration is complete
        registration_complete.set()

        # Signal emit is allowed (test is ready to receive)
        emit_allowed.set()

        # Wait for result
        result = waiter.wait(timeout=5.0)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"output": "done"}

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
                _make_token("token-2", "row-2"),
                TransformResult.success(make_pipeline_row({"value": 2}), success_reason={"action": "test"}),
                "state-2",
            )

            emit_events["token-1"].wait()
            adapter.emit(
                _make_token("token-1", "row-1"),
                TransformResult.success(make_pipeline_row({"value": 1}), success_reason={"action": "test"}),
                "state-1",
            )

            emit_events["token-3"].wait()
            adapter.emit(
                _make_token("token-3", "row-3"),
                TransformResult.success(make_pipeline_row({"value": 3}), success_reason={"action": "test"}),
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

        assert result1.row is not None
        assert result2.row is not None
        assert result3.row is not None
        assert result1.row.to_dict() == {"value": 1}
        assert result2.row.to_dict() == {"value": 2}
        assert result3.row.to_dict() == {"value": 3}

        thread.join()

    def test_emit_before_wait(self) -> None:
        """Test that emit() before wait() still works correctly.

        This can happen if the worker processes very quickly.
        """
        adapter = SharedBatchAdapter()

        # Register waiter
        waiter = adapter.register("token-fast", "state-fast")

        # Emit result IMMEDIATELY (before wait is called)
        token = _make_token("token-fast", "row-1")
        result = TransformResult.success(make_pipeline_row({"fast": True}), success_reason={"action": "test"})
        adapter.emit(token, result, "state-fast")

        # Now wait - should return immediately since event is already set
        start = time.perf_counter()
        got_result = waiter.wait(timeout=5.0)
        elapsed = time.perf_counter() - start

        assert got_result.status == "success"
        assert got_result.row is not None
        assert got_result.row.to_dict() == {"fast": True}
        assert elapsed < 0.1  # Should be nearly instant

    def test_timeout(self) -> None:
        """Test that wait() times out if result never arrives."""
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-never", "state-never")

        with pytest.raises(TimeoutError, match="No result received"):
            waiter.wait(timeout=0.1)

    def test_timeout_cleans_up_entry(self) -> None:
        """Test that timeout cleans up entry to prevent memory leak.

        When wait() times out, the entry must be removed from _entries.
        Otherwise, late results arriving via emit() would still find the entry
        and store results that no one will ever retrieve.
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-timeout", "state-timeout")

        # Entry should exist before timeout
        assert ("token-timeout", "state-timeout") in adapter._entries

        # Timeout
        with pytest.raises(TimeoutError):
            waiter.wait(timeout=0.05)

        # After timeout, entry must be cleaned up
        assert ("token-timeout", "state-timeout") not in adapter._entries

    def test_shutdown_event_raises_interrupted_error(self) -> None:
        """Run cancellation wakes batch waiters with InterruptedError, not a domain result.

        The waiter is a delivery primitive: it signals cancellation with the
        same typed exception the sync path uses (retry.py), and the PROCESSOR
        translates that into the shutdown_requested error result
        (_convert_retryable_to_error_result) — shutdown policy lives in one
        layer (filigree elspeth-14571961a6). Entry cleanup still happens so a
        late emit() cannot leak a result nobody will retrieve.
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-cancel", "state-cancel")
        shutdown_event = threading.Event()
        shutdown_event.set()

        with pytest.raises(InterruptedError, match="Shutdown requested while waiting for token token-cancel"):
            waiter.wait(timeout=60.0, shutdown_event=shutdown_event)

        assert ("token-cancel", "state-cancel") not in adapter._entries

    def test_late_result_after_shutdown_not_stored(self) -> None:
        """A result emitted after a shutdown-cancelled wait is discarded, not stored.

        Mirrors the timeout leak test: the shutdown arm removed the entry, so
        the (token_id, state_id) retry-safety keying must drop the late result.
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-cancel-late", "state-cancel-late")
        shutdown_event = threading.Event()
        shutdown_event.set()

        with pytest.raises(InterruptedError):
            waiter.wait(timeout=60.0, shutdown_event=shutdown_event)

        adapter.emit(
            _make_token("token-cancel-late", "row-1"),
            TransformResult.success(make_pipeline_row({"late": "result"}), success_reason={"action": "test"}),
            "state-cancel-late",
        )

        assert ("token-cancel-late", "state-cancel-late") not in adapter._entries
        assert len(adapter._entries) == 0

    def test_late_result_after_timeout_not_stored(self) -> None:
        """Test that late results after timeout are discarded, not stored.

        This is the core memory leak bug: if a waiter times out but the
        worker eventually completes and calls emit(), the result should
        NOT be stored (since no one will retrieve it).

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
            _make_token("token-late", "row-1"),
            TransformResult.success(make_pipeline_row({"late": "result"}), success_reason={"action": "test"}),
            "state-late",
        )

        # Entry must NOT exist (would be a memory leak)
        assert ("token-late", "state-late") not in adapter._entries
        assert len(adapter._entries) == 0

    def test_error_result_propagated(self) -> None:
        """Test that error results are correctly propagated."""
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-error", "state-error")

        # Emit error result
        token = _make_token("token-error", "row-1")
        result = TransformResult.error({"reason": "llm_call_failed", "error": "API down"})
        adapter.emit(token, result, "state-error")

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
                    _make_token(f"token-{i}", f"row-{i}"),
                    TransformResult.success(make_pipeline_row({"index": i}), success_reason={"action": "test"}),
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
            row = results[f"token-{i}"].row
            assert row is not None
            assert row.to_dict() == {"index": i}

    def test_clear(self) -> None:
        """Test that clear() removes all state."""
        adapter = SharedBatchAdapter()

        # Register some waiters
        adapter.register("token-1", "state-1")
        adapter.register("token-2", "state-2")

        # Emit a result that won't be consumed (orphaned due to no matching waiter key)
        adapter.emit(
            _make_token("token-orphan", "row-99"),
            TransformResult.success(make_pipeline_row({"orphan": True}), success_reason={"action": "test"}),
            "state-orphan",
        )

        # Clear
        adapter.clear()

        # Verify internal state is empty
        assert len(adapter._entries) == 0

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
            _make_token("token-42", "row-42"),
            TransformResult.success(make_pipeline_row({"result": "stale"}), success_reason={"action": "test"}),
            "attempt-1",
        )

        # Retry worker finishes, emits with retry state_id
        adapter.emit(
            _make_token("token-42", "row-42"),
            TransformResult.success(make_pipeline_row({"result": "fresh"}), success_reason={"action": "test"}),
            "attempt-2",
        )

        # First waiter gets the stale result (it's still registered)
        result1 = waiter1.wait(timeout=1.0)
        assert result1.row is not None
        assert result1.row.to_dict() == {"result": "stale"}

        # Retry waiter gets the fresh result (correct behavior!)
        result2 = waiter2.wait(timeout=1.0)
        assert result2.row is not None
        assert result2.row.to_dict() == {"result": "fresh"}

    def test_timeout_race_cleans_up_late_result(self) -> None:
        """Test that timeout path cleans up entries stored during race window.

        This tests the TOCTOU race between wait() timeout and emit():

        Timeline where the race leaks memory:
            Thread A (wait)                  Thread B (emit)
            -----------------                -----------------
            1. event.wait(timeout) -> False
                                             2. Acquires _lock
                                             3. key in _entries -> True
                                             4. entry.result = result
                                             5. entry.event.set()
                                             6. Releases _lock
            7. Acquires _lock
            8. _entries.pop() -> removes entry (including late result)
            9. Releases _lock
            10. Raises TimeoutError

            Result: entry is cleaned up, no memory leak.

        Since timing races are non-deterministic, we test the fix directly:
        simulate the post-race state and verify cleanup happens.
        """
        adapter = SharedBatchAdapter()
        key = ("token-race", "state-race")

        # Register waiter
        waiter = adapter.register(*key)

        # Simulate the race: emit() wins — sets result and signals event
        with adapter._lock:
            entry = adapter._entries[key]
            entry.result = TransformResult.success(make_pipeline_row({"race": "leaked"}), success_reason={"action": "test"})
            entry.event.set()

        # Re-create with fresh unset event to simulate timeout path
        adapter._entries[key] = _WaiterEntry()
        waiter._event = adapter._entries[key].event

        # Now wait() will timeout because event is not set
        # The timeout path should clean up the entry entirely
        with pytest.raises(TimeoutError):
            waiter.wait(timeout=0.05)

        # Verify entry is cleaned up - no memory leak
        assert key not in adapter._entries, (
            "Entry should be removed to prevent memory leak. "
            "The race between timeout and emit can store a result that "
            "no one will ever retrieve."
        )

    def test_duplicate_emit_preserves_first_result(self) -> None:
        """Duplicate emit() for the same key must not overwrite the first result.

        Regression: When _WaiterEntry stays in _entries until wait() pops it,
        a second emit() for the same (token_id, state_id) could overwrite
        entry.result. This test verifies first-result-wins semantics: once
        an entry is signaled, subsequent emits for the same key are no-ops.
        """
        adapter = SharedBatchAdapter()
        waiter = adapter.register("token-dup", "state-dup")

        token = _make_token("token-dup", "row-1")

        # First emit — should be delivered
        first_result = TransformResult.success(
            make_pipeline_row({"delivery": "first"}),
            success_reason={"action": "test"},
        )
        adapter.emit(token, first_result, "state-dup")

        # Second emit — should be discarded (first-result-wins)
        second_result = TransformResult.success(
            make_pipeline_row({"delivery": "second"}),
            success_reason={"action": "test"},
        )
        adapter.emit(token, second_result, "state-dup")

        # Waiter must get the first result, not the second
        got = waiter.wait(timeout=1.0)
        assert got.row is not None
        assert got.row.to_dict() == {"delivery": "first"}

    def test_emit_with_none_state_id_raises_invariant_error(self) -> None:
        """emit() with state_id=None raises OrchestrationInvariantError.

        Regression: C1 — state_id=None means the executor failed to set up
        node state via begin_node_state() before calling the batch transform.
        Previously this was a silent drop, masking the real bug behind a
        3600s timeout.
        """
        adapter = SharedBatchAdapter()
        token = _make_token("token-1", "row-1")
        result = TransformResult.success(make_pipeline_row({"v": 1}), success_reason={"action": "test"})

        with pytest.raises(OrchestrationInvariantError, match="state_id=None"):
            adapter.emit(token, result, state_id=None)

    def test_emit_with_none_state_id_signals_waiter_instead_of_timeout(self) -> None:
        """emit() with state_id=None delivers error to registered waiter.

        Regression: When state_id=None and a waiter IS registered for the
        token, the waiter must receive an OrchestrationInvariantError
        immediately instead of hanging until batch_wait_timeout (often 3600s).
        The release loop's catch-and-retry cannot fix state_id=None, so
        emit() must deliver the error directly to the waiter by scanning
        for token_id matches.
        """
        adapter = SharedBatchAdapter()
        token = _make_token("token-1", "row-1")
        result = TransformResult.success(make_pipeline_row({"v": 1}), success_reason={"action": "test"})

        # Register a waiter (as orchestrator thread would)
        waiter = adapter.register("token-1", "state-abc")

        # emit with state_id=None (executor bug)
        adapter.emit(token, result, state_id=None)

        # Waiter should be immediately signaled with the error (not timeout)
        with pytest.raises(OrchestrationInvariantError, match="state_id=None"):
            waiter.wait(timeout=1.0)
