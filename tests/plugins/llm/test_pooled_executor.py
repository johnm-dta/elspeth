# tests/plugins/llm/test_pooled_executor.py
"""Tests for PooledExecutor parallel request handling."""

from elspeth.plugins.llm.base import PoolConfig
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

        def mock_process(row: dict, state_id: str) -> TransformResult:
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
        assert results[0].row["idx"] == 0
        assert results[1].row["idx"] == 1
        assert results[2].row["idx"] == 2

        executor.shutdown()

    def test_execute_batch_passes_state_id_per_row(self) -> None:
        """Each row should receive its own state_id."""
        from threading import Lock

        from elspeth.contracts import TransformResult

        config = PoolConfig(pool_size=2)
        executor = PooledExecutor(config)

        received_state_ids: list[tuple[int, str]] = []
        lock = Lock()

        def mock_process(row: dict, state_id: str) -> TransformResult:
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

        def mock_process(row: dict, state_id: str) -> TransformResult:
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
