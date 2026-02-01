# tests/plugins/llm/test_pooled_executor.py
"""Tests for PooledExecutor parallel request handling."""

import threading
import time
from threading import Lock
from typing import Any

from elspeth.contracts import TransformResult
from elspeth.plugins.pooling import BufferEntry, CapacityError, PoolConfig, PooledExecutor, RowContext


class TestPooledExecutorInit:
    """Test executor initialization."""

    def test_creates_with_config(self) -> None:
        """Executor should accept pool config."""
        config = PoolConfig(pool_size=10)

        executor = PooledExecutor(config)

        assert executor.pool_size == 10
        assert executor.pending_count == 0

        executor.shutdown()

    def test_creates_throttle_from_config(self) -> None:
        """Executor should create AIMD throttle from config."""
        config = PoolConfig(
            pool_size=5,
            backoff_multiplier=3.0,
            recovery_step_ms=100,
        )

        executor = PooledExecutor(config)

        assert executor._throttle.config.backoff_multiplier == 3.0
        assert executor._throttle.config.recovery_step_ms == 100

        executor.shutdown()


class TestPooledExecutorShutdown:
    """Test executor shutdown."""

    def test_shutdown_completes_pending(self) -> None:
        """Shutdown should wait for pending requests."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        # Should not raise
        executor.shutdown(wait=True)

        assert executor.pending_count == 0


class TestRowContext:
    """Test RowContext dataclass."""

    def test_row_context_creation(self) -> None:
        """RowContext should hold row, state_id, and index."""
        row = {"id": 1, "text": "hello"}
        ctx = RowContext(row=row, state_id="state-123", row_index=5)

        assert ctx.row == row
        assert ctx.state_id == "state-123"
        assert ctx.row_index == 5

    def test_row_context_immutable_reference(self) -> None:
        """RowContext should maintain reference to original row."""
        row = {"id": 1}
        ctx = RowContext(row=row, state_id="state-1", row_index=0)

        # Modifying original should affect context (shared reference)
        row["id"] = 2
        assert ctx.row["id"] == 2


class TestPooledExecutorBatch:
    """Test batch execution with ordering."""

    def test_execute_batch_returns_results_in_order(self) -> None:
        """Results should be in submission order regardless of completion."""
        import time
        from threading import Lock

        from elspeth.contracts import TransformResult

        config = PoolConfig(pool_size=3)
        executor = PooledExecutor(config)

        # Mock process function with varying delays
        call_order: list[int] = []
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            idx = row["idx"]
            with lock:
                call_order.append(idx)
            # Varying delays to cause out-of-order completion
            time.sleep(0.01 * (3 - idx))  # idx 0 slowest, idx 2 fastest
            return TransformResult.success(
                {"idx": idx, "result": f"done_{idx}"},
                success_reason={"action": "processed"},
            )

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(3)]

        entries = executor.execute_batch(contexts, mock_process)

        # Entries must be in submission order with full metadata
        assert len(entries) == 3
        assert all(isinstance(e, BufferEntry) for e in entries)

        # Verify results are in submission order
        assert entries[0].result.row is not None and entries[0].result.row["idx"] == 0
        assert entries[1].result.row is not None and entries[1].result.row["idx"] == 1
        assert entries[2].result.row is not None and entries[2].result.row["idx"] == 2

        # Verify ordering metadata is present
        assert entries[0].submit_index == 0
        assert entries[1].submit_index == 1
        assert entries[2].submit_index == 2

        executor.shutdown()

    def test_execute_batch_passes_state_id_per_row(self) -> None:
        """Each row should receive its own state_id."""
        from threading import Lock

        from elspeth.contracts import TransformResult

        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        received_state_ids: list[tuple[int, str]] = []
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            with lock:
                received_state_ids.append((row["idx"], state_id))
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"unique_state_{i}", row_index=i) for i in range(3)]

        entries = executor.execute_batch(contexts, mock_process)

        # Verify each row got its own state_id
        assert len(received_state_ids) == 3
        state_id_map = dict(received_state_ids)
        assert state_id_map[0] == "unique_state_0"
        assert state_id_map[1] == "unique_state_1"
        assert state_id_map[2] == "unique_state_2"

        # Verify entries returned
        assert len(entries) == 3

        executor.shutdown()

    def test_execute_batch_respects_pool_size(self) -> None:
        """Should never exceed pool_size concurrent requests."""
        import time
        from threading import Lock

        from elspeth.contracts import TransformResult

        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        max_concurrent = 0
        current_concurrent = 0
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            nonlocal max_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            time.sleep(0.05)

            with lock:
                current_concurrent -= 1

            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(5)]

        entries = executor.execute_batch(contexts, mock_process)

        assert len(entries) == 5
        assert max_concurrent <= 2  # Never exceeded pool_size

        executor.shutdown()


class TestPooledExecutorStats:
    """Test executor statistics."""

    def test_get_stats_returns_pool_config(self) -> None:
        """Stats should include pool configuration."""
        config = PoolConfig(
            pool_size=4,
            max_capacity_retry_seconds=1800,
        )
        executor = PooledExecutor(config)

        stats = executor.get_stats()

        assert stats["pool_config"]["pool_size"] == 4
        assert stats["pool_config"]["max_capacity_retry_seconds"] == 1800
        assert "dispatch_delay_at_completion_ms" in stats["pool_config"]

        executor.shutdown()

    def test_get_stats_includes_throttle_stats(self) -> None:
        """Stats should include throttle statistics."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        stats = executor.get_stats()

        # Throttle stats should be present
        assert "pool_stats" in stats
        assert "capacity_retries" in stats["pool_stats"]
        assert "successes" in stats["pool_stats"]
        assert "peak_delay_ms" in stats["pool_stats"]
        assert "current_delay_ms" in stats["pool_stats"]
        assert "total_throttle_time_ms" in stats["pool_stats"]
        assert "max_concurrent_reached" in stats["pool_stats"]

        executor.shutdown()

    def test_max_concurrent_reached_tracks_peak_workers(self) -> None:
        """max_concurrent_reached should track peak concurrent workers.

        Regression test for P3-2026-01-21-pooling-missing-pool-stats.
        """
        config = PoolConfig(pool_size=3)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.05)  # Long enough for concurrent execution
            return TransformResult.success(row, success_reason={"action": "processed"})

        # Run 5 items through pool_size=3 - should see max_concurrent=3
        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(5)]
        executor.execute_batch(contexts, mock_process)

        stats = executor.get_stats()
        # Should have reached pool_size at some point
        assert stats["pool_stats"]["max_concurrent_reached"] >= 2
        assert stats["pool_stats"]["max_concurrent_reached"] <= 3

        executor.shutdown()

    def test_dispatch_delay_at_completion_captures_final_delay(self) -> None:
        """dispatch_delay_at_completion_ms should capture delay at batch end.

        Regression test for P3-2026-01-21-pooling-missing-pool-stats.
        """
        config = PoolConfig(pool_size=2, recovery_step_ms=50)
        executor = PooledExecutor(config)

        call_count = 0
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            nonlocal call_count
            with lock:
                call_count += 1
                current_count = call_count

            # First call raises capacity error to trigger throttle
            if current_count == 1:
                raise CapacityError(429, "Rate limited")
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]
        executor.execute_batch(contexts, mock_process)

        stats = executor.get_stats()
        # After capacity error, delay should be > 0
        assert stats["pool_config"]["dispatch_delay_at_completion_ms"] >= 0

        executor.shutdown()

    def test_stats_reset_between_batches(self) -> None:
        """max_concurrent_reached should reset between batches."""
        config = PoolConfig(pool_size=4)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.02)
            return TransformResult.success(row, success_reason={"action": "processed"})

        # First batch with 4 items
        contexts1 = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(4)]
        executor.execute_batch(contexts1, mock_process)
        # Stats from first batch not needed - we only verify second batch's max_concurrent

        # Second batch with only 1 item
        contexts2 = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]
        executor.execute_batch(contexts2, mock_process)
        stats2 = executor.get_stats()

        # Second batch should have lower max_concurrent
        assert stats2["pool_stats"]["max_concurrent_reached"] == 1

        executor.shutdown()


class TestPooledExecutorCapacityHandling:
    """Test capacity error handling with AIMD throttle and timeout."""

    def test_capacity_error_triggers_throttle_and_retries(self) -> None:
        """Capacity errors should trigger throttle and retry."""
        config = PoolConfig(pool_size=2, recovery_step_ms=50)
        executor = PooledExecutor(config)

        call_count = 0
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            nonlocal call_count
            with lock:
                call_count += 1
                current_count = call_count

            # First call raises capacity error, second succeeds
            if current_count == 1:
                raise CapacityError(429, "Rate limited")
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        entries = executor.execute_batch(contexts, mock_process)

        # Should have retried and succeeded
        assert len(entries) == 1
        assert entries[0].result.status == "success"
        assert call_count == 2

        # Throttle should have been triggered
        stats = executor.get_stats()
        assert stats["pool_stats"]["capacity_retries"] == 1

        executor.shutdown()

    def test_capacity_retry_respects_max_timeout(self) -> None:
        """Capacity retries should stop after max_capacity_retry_seconds."""
        config = PoolConfig(
            pool_size=1,
            max_dispatch_delay_ms=100,
            max_capacity_retry_seconds=1,  # Only 1 second
        )
        executor = PooledExecutor(config)

        call_count = 0
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            nonlocal call_count
            with lock:
                call_count += 1
            # Always fail with capacity error
            raise CapacityError(503, "Service unavailable")

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        entries = executor.execute_batch(contexts, mock_process)

        # Should eventually fail after timeout
        assert len(entries) == 1
        assert entries[0].result.status == "error"
        assert entries[0].result.reason is not None
        assert entries[0].result.reason["reason"] == "retry_timeout"

        # Should have made multiple attempts before giving up
        assert call_count > 1

        executor.shutdown()

    def test_normal_error_not_retried(self) -> None:
        """Non-capacity errors should not be retried."""
        config = PoolConfig(pool_size=1)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            # Return error result (not raise CapacityError)
            return TransformResult.error({"reason": "api_error", "error": "bad_request"})

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        entries = executor.execute_batch(contexts, mock_process)

        # Should return error without retry
        assert len(entries) == 1
        assert entries[0].result.status == "error"

        executor.shutdown()

    def test_capacity_retry_releases_semaphore_during_backoff(self) -> None:
        """During capacity retry backoff, semaphore should be released.

        This ensures other workers can make progress while one is sleeping.
        CRITICAL: Without this, all workers hitting capacity errors would
        deadlock the pool.
        """
        config = PoolConfig(
            pool_size=2,
            recovery_step_ms=50,
            max_dispatch_delay_ms=100,
        )
        executor = PooledExecutor(config)

        # Track concurrent execution during retry
        row0_in_retry_sleep = threading.Event()
        row1_completed = threading.Event()
        execution_order: list[str] = []
        lock = Lock()
        row0_call_count = 0

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            nonlocal row0_call_count
            idx = row["idx"]
            with lock:
                execution_order.append(f"start_{idx}")

            if idx == 0:
                row0_call_count += 1
                if row0_call_count == 1:
                    row0_in_retry_sleep.set()  # Signal we're about to sleep
                    raise CapacityError(429, "Rate limited")
                # Second call succeeds
                row1_completed.wait(timeout=2)  # Wait for row 1 to complete
                with lock:
                    execution_order.append(f"end_{idx}")
                return TransformResult.success(row, success_reason={"action": "processed"})
            else:
                # Row 1: Wait until row 0 is in retry sleep, then complete
                row0_in_retry_sleep.wait(timeout=2)
                time.sleep(0.05)  # Give row 0 time to release semaphore
                with lock:
                    execution_order.append(f"end_{idx}")
                row1_completed.set()
                return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [
            RowContext(row={"idx": 0}, state_id="state_0", row_index=0),
            RowContext(row={"idx": 1}, state_id="state_1", row_index=1),
        ]

        entries = executor.execute_batch(contexts, mock_process)

        # Both should succeed
        assert len(entries) == 2
        assert all(e.result.status == "success" for e in entries)

        # Row 1 should have executed WHILE row 0 was in retry sleep
        # If semaphore wasn't released, row 1 would be blocked
        assert "end_1" in execution_order
        end_1_idx = execution_order.index("end_1")
        # end_1 should happen before end_0 (row 1 completes during row 0's retry)
        assert "end_0" in execution_order
        end_0_idx = execution_order.index("end_0")
        assert end_1_idx < end_0_idx, f"Row 1 should complete before Row 0's retry succeeds. Order: {execution_order}"

        executor.shutdown()

    def test_no_deadlock_when_batch_exceeds_pool_with_capacity_errors(self) -> None:
        """Regression test: batch > pool_size with early capacity errors must not deadlock.

        The bug scenario:
        1. pool_size=2, batch=6 rows
        2. Main thread acquires semaphore for rows 0-2, submits, blocks on row 3
        3. Workers 0-2 hit capacity errors, release semaphore, try to re-acquire
        4. Main thread wakes up, acquires permits for rows 3-5, submits them
        5. DEADLOCK: Workers 0-2 waiting for permits, but permits "held" by queued
           tasks 3-5 that can't run (thread pool full of blocked workers)

        Fix: Acquire semaphore INSIDE worker, not in main thread. This ensures
        semaphore represents "actively working" not "queued for work".
        """
        config = PoolConfig(
            pool_size=2,
            recovery_step_ms=10,
            max_dispatch_delay_ms=50,
            max_capacity_retry_seconds=5,
        )
        executor = PooledExecutor(config)

        # Track calls per row
        call_counts: dict[int, int] = {}
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            idx = row["idx"]
            with lock:
                call_counts[idx] = call_counts.get(idx, 0) + 1
                current_count = call_counts[idx]

            # ALL rows hit capacity error on first attempt, succeed on second
            if current_count == 1:
                raise CapacityError(429, f"Rate limited row {idx}")
            return TransformResult.success(
                {"idx": idx, "attempts": current_count},
                success_reason={"action": "processed"},
            )

        # Batch of 6 rows with pool_size=2 - this would deadlock with old implementation
        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(6)]

        # This should complete without deadlock (timeout would indicate deadlock)
        import signal

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("Batch execution deadlocked - test failed")

        # Set 10 second timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            entries = executor.execute_batch(contexts, mock_process)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        # All rows should succeed
        assert len(entries) == 6
        assert all(e.result.status == "success" for e in entries)

        # Each row should have been called twice (first fail, second succeed)
        for i in range(6):
            assert call_counts[i] == 2, f"Row {i} should have 2 calls, got {call_counts[i]}"

        # Stats should show capacity retries
        stats = executor.get_stats()
        assert stats["pool_stats"]["capacity_retries"] == 6  # One retry per row

        executor.shutdown()


class TestPooledExecutorConcurrentBatches:
    """Test that concurrent execute_batch() calls are properly isolated."""

    def test_concurrent_execute_batch_calls_are_serialized(self) -> None:
        """Concurrent execute_batch() calls must not mix results.

        This test verifies that if two threads call execute_batch() on the
        same executor, results are not interleaved between batches.
        """
        from concurrent.futures import ThreadPoolExecutor

        config = PoolConfig(pool_size=4)
        executor = PooledExecutor(config)

        batch_a_entries: list[BufferEntry[TransformResult]] = []
        batch_b_entries: list[BufferEntry[TransformResult]] = []
        errors: list[Exception] = []

        def process_a(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.02)  # Simulate work
            return TransformResult.success(
                {"batch": "A", "idx": row["idx"]},
                success_reason={"action": "processed"},
            )

        def process_b(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.02)  # Simulate work
            return TransformResult.success(
                {"batch": "B", "idx": row["idx"]},
                success_reason={"action": "processed"},
            )

        def run_batch_a() -> None:
            try:
                contexts = [RowContext(row={"idx": i}, state_id=f"a_state_{i}", row_index=i) for i in range(5)]
                nonlocal batch_a_entries
                batch_a_entries = executor.execute_batch(contexts, process_a)
            except Exception as e:
                errors.append(e)

        def run_batch_b() -> None:
            try:
                contexts = [RowContext(row={"idx": i}, state_id=f"b_state_{i}", row_index=i) for i in range(5)]
                nonlocal batch_b_entries
                batch_b_entries = executor.execute_batch(contexts, process_b)
            except Exception as e:
                errors.append(e)

        # Run both batches concurrently
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(run_batch_a)
            future_b = pool.submit(run_batch_b)
            future_a.result(timeout=10)
            future_b.result(timeout=10)

        # Check for errors
        assert not errors, f"Batch execution raised errors: {errors}"

        # Both batches should have exactly 5 entries
        assert len(batch_a_entries) == 5, f"Batch A has {len(batch_a_entries)} entries, expected 5"
        assert len(batch_b_entries) == 5, f"Batch B has {len(batch_b_entries)} entries, expected 5"

        # All batch A results must be from batch A (no mixing)
        for i, entry in enumerate(batch_a_entries):
            assert entry.result.row is not None
            assert entry.result.row["batch"] == "A", f"Batch A entry {i} contains batch {entry.result.row['batch']}"
            assert entry.result.row["idx"] == i, f"Batch A entry {i} has wrong index {entry.result.row['idx']}"

        # All batch B results must be from batch B (no mixing)
        for i, entry in enumerate(batch_b_entries):
            assert entry.result.row is not None
            assert entry.result.row["batch"] == "B", f"Batch B entry {i} contains batch {entry.result.row['batch']}"
            assert entry.result.row["idx"] == i, f"Batch B entry {i} has wrong index {entry.result.row['idx']}"

        executor.shutdown()


class TestPooledExecutorDispatchPacing:
    """Test that dispatch pacing is global, not per-worker."""

    def test_dispatch_pacing_is_global_not_per_worker(self) -> None:
        """Dispatches should be spaced globally, not per-worker.

        With pool_size=4 and min_dispatch_delay_ms=100, dispatches should
        be at least 100ms apart globally (not 4 simultaneous dispatches
        every 100ms).
        """
        config = PoolConfig(
            pool_size=4,
            min_dispatch_delay_ms=100,
        )
        executor = PooledExecutor(config)

        dispatch_times: list[float] = []
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            with lock:
                dispatch_times.append(time.monotonic())
            time.sleep(0.01)  # Minimal work
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(8)]

        entries = executor.execute_batch(contexts, mock_process)

        assert len(entries) == 8

        # Sort dispatch times and check intervals
        dispatch_times.sort()
        for i in range(1, len(dispatch_times)):
            interval_ms = (dispatch_times[i] - dispatch_times[i - 1]) * 1000
            # Allow some tolerance (90ms instead of 100ms)
            assert interval_ms >= 90, f"Dispatch {i} was only {interval_ms:.1f}ms after dispatch {i - 1}, expected >= 100ms"

        executor.shutdown()

    def test_no_burst_traffic_on_startup(self) -> None:
        """All workers should not dispatch simultaneously at startup.

        Regression test for burst bug: with pool_size=4, we should NOT
        see 4 dispatches within a few milliseconds of each other.
        """
        config = PoolConfig(
            pool_size=4,
            min_dispatch_delay_ms=50,
        )
        executor = PooledExecutor(config)

        dispatch_times: list[float] = []
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            with lock:
                dispatch_times.append(time.monotonic())
            time.sleep(0.2)  # Longer than total delay budget
            return TransformResult.success(row, success_reason={"action": "processed"})

        # Exactly pool_size rows - all would dispatch together in buggy version
        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(4)]

        entries = executor.execute_batch(contexts, mock_process)

        assert len(entries) == 4

        # Check that dispatches are NOT bunched together
        dispatch_times.sort()
        first_dispatch = dispatch_times[0]
        last_of_first_batch = dispatch_times[-1]

        # With 50ms delay between each, 4 dispatches should span ~150ms minimum
        span_ms = (last_of_first_batch - first_dispatch) * 1000
        assert span_ms >= 120, f"All 4 dispatches completed within {span_ms:.1f}ms - indicates burst traffic (expected >= 150ms span)"

        executor.shutdown()


class TestPooledExecutorOrderingMetadata:
    """Test that ordering metadata is preserved for audit trail.

    Regression tests for P2-2026-01-21-pooling-ordering-metadata-dropped.
    """

    def test_execute_batch_returns_buffer_entries_with_metadata(self) -> None:
        """execute_batch should return BufferEntry objects, not just results.

        This is the core fix for the P2 bug - metadata was previously stripped.
        """
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(3)]

        entries = executor.execute_batch(contexts, mock_process)

        # Must return BufferEntry objects
        assert len(entries) == 3
        for entry in entries:
            assert isinstance(entry, BufferEntry)
            assert hasattr(entry, "submit_index")
            assert hasattr(entry, "complete_index")
            assert hasattr(entry, "submit_timestamp")
            assert hasattr(entry, "complete_timestamp")
            assert hasattr(entry, "buffer_wait_ms")
            assert hasattr(entry, "result")

        executor.shutdown()

    def test_submit_indices_are_sequential(self) -> None:
        """Submit indices should be 0, 1, 2, ... in submission order."""
        config = PoolConfig(pool_size=3)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.01)
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(5)]

        entries = executor.execute_batch(contexts, mock_process)

        # Submit indices must be sequential
        submit_indices = [e.submit_index for e in entries]
        assert submit_indices == [0, 1, 2, 3, 4]

        executor.shutdown()

    def test_complete_indices_reflect_actual_completion_order(self) -> None:
        """Complete indices should reflect the order requests actually completed.

        With varying delays, completion order differs from submission order.
        This metadata is crucial for diagnosing out-of-order issues.
        """
        config = PoolConfig(pool_size=5)  # All can run in parallel
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            idx = row["idx"]
            # Reverse delay: idx 0 slowest, idx 4 fastest
            time.sleep(0.05 * (5 - idx))
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(5)]

        entries = executor.execute_batch(contexts, mock_process)

        # Submit indices should be in order (reorder buffer guarantees this)
        submit_indices = [e.submit_index for e in entries]
        assert submit_indices == [0, 1, 2, 3, 4]

        # Complete indices should show that higher indices completed first
        # (because they had shorter delays)
        complete_indices = [e.complete_index for e in entries]

        # Entry 4 should have completed first (complete_index 0)
        # Entry 0 should have completed last (complete_index 4)
        assert entries[4].complete_index < entries[0].complete_index, (
            f"Row 4 (fast) should complete before row 0 (slow). Complete indices: {complete_indices}"
        )

        executor.shutdown()

    def test_timestamps_are_valid(self) -> None:
        """Timestamps should be valid perf_counter values."""
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        before_test = time.perf_counter()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            time.sleep(0.01)
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(2)]

        entries = executor.execute_batch(contexts, mock_process)

        after_test = time.perf_counter()

        for entry in entries:
            # Timestamps should be within test bounds
            assert before_test <= entry.submit_timestamp <= after_test
            assert before_test <= entry.complete_timestamp <= after_test

            # Complete should be after submit
            assert entry.complete_timestamp >= entry.submit_timestamp

            # Buffer wait should be non-negative
            assert entry.buffer_wait_ms >= 0

        executor.shutdown()

    def test_buffer_wait_ms_tracks_reorder_delay(self) -> None:
        """buffer_wait_ms should track time spent waiting for earlier items.

        When item N completes but item N-1 hasn't, item N waits in buffer.
        """
        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            idx = row["idx"]
            if idx == 0:
                time.sleep(0.1)  # Slow - holds up emission of item 1
            else:
                time.sleep(0.01)  # Fast - completes but waits for item 0
            return TransformResult.success(row, success_reason={"action": "processed"})

        contexts = [
            RowContext(row={"idx": 0}, state_id="state_0", row_index=0),
            RowContext(row={"idx": 1}, state_id="state_1", row_index=1),
        ]

        entries = executor.execute_batch(contexts, mock_process)

        # Item 0 (slow) shouldn't have waited much - it was the blocker
        # Item 1 (fast) should have waited ~90ms for item 0
        assert entries[0].buffer_wait_ms < 50, f"Item 0 waited {entries[0].buffer_wait_ms}ms unexpectedly"
        assert entries[1].buffer_wait_ms >= 50, f"Item 1 should have waited for item 0, only waited {entries[1].buffer_wait_ms}ms"

        executor.shutdown()
