# tests/plugins/llm/test_pooled_executor.py
"""Tests for PooledExecutor parallel request handling."""

import threading
import time
from threading import Lock
from typing import Any

from elspeth.contracts import TransformResult
from elspeth.plugins.llm.base import PoolConfig
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.pooled_executor import PooledExecutor, RowContext


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
            return TransformResult.success({"idx": idx, "result": f"done_{idx}"})

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(3)]

        results = executor.execute_batch(contexts, mock_process)

        # Results must be in submission order
        assert len(results) == 3
        assert results[0].row is not None and results[0].row["idx"] == 0
        assert results[1].row is not None and results[1].row["idx"] == 1
        assert results[2].row is not None and results[2].row["idx"] == 2

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
            return TransformResult.success(row)

        contexts = [RowContext(row={"idx": i}, state_id=f"unique_state_{i}", row_index=i) for i in range(3)]

        executor.execute_batch(contexts, mock_process)

        # Verify each row got its own state_id
        assert len(received_state_ids) == 3
        state_id_map = dict(received_state_ids)
        assert state_id_map[0] == "unique_state_0"
        assert state_id_map[1] == "unique_state_1"
        assert state_id_map[2] == "unique_state_2"

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

            return TransformResult.success(row)

        contexts = [RowContext(row={"idx": i}, state_id=f"state_{i}", row_index=i) for i in range(5)]

        results = executor.execute_batch(contexts, mock_process)

        assert len(results) == 5
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
            return TransformResult.success(row)

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        results = executor.execute_batch(contexts, mock_process)

        # Should have retried and succeeded
        assert len(results) == 1
        assert results[0].status == "success"
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

        results = executor.execute_batch(contexts, mock_process)

        # Should eventually fail after timeout
        assert len(results) == 1
        assert results[0].status == "error"
        assert results[0].reason is not None
        assert "capacity_retry_timeout" in results[0].reason["reason"]

        # Should have made multiple attempts before giving up
        assert call_count > 1

        executor.shutdown()

    def test_normal_error_not_retried(self) -> None:
        """Non-capacity errors should not be retried."""
        config = PoolConfig(pool_size=1)
        executor = PooledExecutor(config)

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            # Return error result (not raise CapacityError)
            return TransformResult.error({"reason": "bad_request"})

        contexts = [RowContext(row={"idx": 0}, state_id="state_0", row_index=0)]

        results = executor.execute_batch(contexts, mock_process)

        # Should return error without retry
        assert len(results) == 1
        assert results[0].status == "error"

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
                return TransformResult.success(row)
            else:
                # Row 1: Wait until row 0 is in retry sleep, then complete
                row0_in_retry_sleep.wait(timeout=2)
                time.sleep(0.05)  # Give row 0 time to release semaphore
                with lock:
                    execution_order.append(f"end_{idx}")
                row1_completed.set()
                return TransformResult.success(row)

        contexts = [
            RowContext(row={"idx": 0}, state_id="state_0", row_index=0),
            RowContext(row={"idx": 1}, state_id="state_1", row_index=1),
        ]

        results = executor.execute_batch(contexts, mock_process)

        # Both should succeed
        assert len(results) == 2
        assert all(r.status == "success" for r in results)

        # Row 1 should have executed WHILE row 0 was in retry sleep
        # If semaphore wasn't released, row 1 would be blocked
        assert "end_1" in execution_order
        end_1_idx = execution_order.index("end_1")
        # end_1 should happen before end_0 (row 1 completes during row 0's retry)
        assert "end_0" in execution_order
        end_0_idx = execution_order.index("end_0")
        assert end_1_idx < end_0_idx, f"Row 1 should complete before Row 0's retry succeeds. Order: {execution_order}"

        executor.shutdown()
