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
