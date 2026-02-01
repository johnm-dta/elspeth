# tests/plugins/pooling/test_executor_retryable_errors.py
"""Integration tests for PooledExecutor retry behavior with various error types."""

from __future__ import annotations

import time
from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.plugins.clients.llm import (
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.pooling.config import PoolConfig
from elspeth.plugins.pooling.errors import CapacityError
from elspeth.plugins.pooling.executor import PooledExecutor, RowContext


class TestRetryableErrorHandling:
    """Test that PooledExecutor retries retryable errors."""

    @pytest.fixture
    def pool_config(self) -> PoolConfig:
        """Create pool config with short retry timeout for testing."""
        return PoolConfig(
            pool_size=10,
            max_capacity_retry_seconds=1,  # Short timeout for tests
            min_dispatch_delay_ms=10,  # Fast retries for testing
        )

    def test_network_error_triggers_retry_until_success(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """Network errors should retry until success."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            if call_count[0] <= 2:
                raise NetworkError("Connection timeout")
            # Third attempt succeeds
            return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "success"
        assert call_count[0] == 3  # Failed twice, succeeded third time

    def test_server_error_503_triggers_retry(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """503 Service Unavailable should retry."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            if call_count[0] == 1:
                raise ServerError("503 Service Unavailable")
            return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "success"
        assert call_count[0] == 2  # Failed once, succeeded second time

    def test_rate_limit_error_triggers_retry(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """RateLimitError should retry (existing behavior verified)."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            if call_count[0] <= 3:
                raise RateLimitError("429 Rate limit exceeded")
            return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "success"
        assert call_count[0] == 4  # Failed 3x, succeeded 4th time

    def test_content_policy_error_fails_immediately_no_retry(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """ContentPolicyError should fail immediately without retry."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            raise ContentPolicyError("Your request was rejected by our safety system")

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert call_count[0] == 1  # Only called once, no retry
        assert results[0].result.reason is not None
        assert results[0].result.reason["reason"] == "permanent_error"
        assert "ContentPolicyError" in results[0].result.reason["error_type"]

    def test_context_length_error_fails_immediately_no_retry(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """ContextLengthError should fail immediately without retry."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            raise ContextLengthError("Maximum context length exceeded")

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert call_count[0] == 1  # Only called once, no retry
        assert results[0].result.reason is not None
        assert results[0].result.reason["reason"] == "permanent_error"

    def test_client_error_401_fails_immediately_no_retry(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """401 Unauthorized should fail immediately without retry."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            raise LLMClientError("401 Unauthorized: Invalid API key", retryable=False)

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert call_count[0] == 1  # Only called once, no retry
        assert results[0].result.reason is not None
        assert "401" in results[0].result.reason["error"]

    def test_retryable_error_timeout_after_max_retry_seconds(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """Retryable errors should timeout after max_capacity_retry_seconds."""

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            # Always fail with retryable error
            raise NetworkError("Connection timeout")

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        start = time.monotonic()
        results = executor.execute_batch(contexts, process_fn)
        elapsed = time.monotonic() - start

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert results[0].result.reason is not None
        assert results[0].result.reason["reason"] == "retry_timeout"
        assert results[0].result.reason["error_type"] == "NetworkError"
        # Should timeout around 1 second (max_capacity_retry_seconds)
        # Allow some variance for system load
        assert 0.9 <= elapsed <= 2.0

    def test_mixed_retryable_and_permanent_errors(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """Test batch with mix of retryable and permanent errors."""
        call_counts = {0: [0], 1: [0], 2: [0]}

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            idx = row["id"]
            call_counts[idx][0] += 1

            if idx == 0:
                # Retryable - succeeds on second attempt
                if call_counts[idx][0] == 1:
                    raise ServerError("503 Service Unavailable")
                return TransformResult.success({"result": "success"}, success_reason={"action": "test"})
            elif idx == 1:
                # Permanent - fails immediately
                raise ContentPolicyError("Safety system rejected request")
            else:
                # Immediate success
                return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(pool_config)
        contexts = [
            RowContext(row={"id": 0}, state_id="test-0", row_index=0),
            RowContext(row={"id": 1}, state_id="test-1", row_index=1),
            RowContext(row={"id": 2}, state_id="test-2", row_index=2),
        ]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 3

        # Row 0: Retried and succeeded
        assert results[0].result.status == "success"
        assert call_counts[0][0] == 2

        # Row 1: Failed immediately (no retry)
        assert results[1].result.status == "error"
        assert call_counts[1][0] == 1
        assert results[1].result.reason is not None
        assert results[1].result.reason["reason"] == "permanent_error"

        # Row 2: Immediate success
        assert results[2].result.status == "success"
        assert call_counts[2][0] == 1

    def test_capacity_error_still_works(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """CapacityError should still trigger retry (existing behavior)."""
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            call_count[0] += 1
            if call_count[0] == 1:
                raise CapacityError(status_code=429, message="Rate limit")
            return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "success"
        assert call_count[0] == 2  # Failed once, succeeded second time

    def test_capacity_error_timeout_includes_status_code(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """CapacityError timeout should include status_code in error details."""

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            raise CapacityError(status_code=429, message="Rate limit")

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert results[0].result.reason is not None
        assert results[0].result.reason["reason"] == "retry_timeout"
        assert results[0].result.reason["status_code"] == 429  # Should include status_code

    def test_non_capacity_error_timeout_no_status_code(
        self,
        pool_config: PoolConfig,
    ) -> None:
        """Non-CapacityError timeout should not include status_code."""

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            raise NetworkError("Connection timeout")

        executor = PooledExecutor(pool_config)
        contexts = [RowContext(row={"id": 1}, state_id="test-1", row_index=0)]

        results = executor.execute_batch(contexts, process_fn)

        assert len(results) == 1
        assert results[0].result.status == "error"
        assert results[0].result.reason is not None
        assert results[0].result.reason["reason"] == "retry_timeout"
        assert "status_code" not in results[0].result.reason  # Should NOT include status_code
        assert results[0].result.reason["error_type"] == "NetworkError"


class TestDispatchGateAfterRetry:
    """Regression tests for dispatch gate behavior after capacity error retries.

    Bug context: Previously, workers that hit capacity errors would skip the
    dispatch gate check after retry (via `just_retried` flag), which could
    violate min_dispatch_delay_ms pacing when other workers dispatched during
    the retry backoff sleep.
    """

    def test_retry_respects_dispatch_gate_timing(self) -> None:
        """Verify retries still wait for dispatch gate, ensuring global pacing.

        This test verifies the fix for the bug where retries could bypass
        the dispatch gate, causing bursts that violate min_dispatch_delay_ms.
        """
        # Use significant min_dispatch_delay to make violations detectable
        config = PoolConfig(
            pool_size=2,
            max_capacity_retry_seconds=5,
            min_dispatch_delay_ms=50,  # 50ms minimum between ALL dispatches
        )

        dispatch_times: list[float] = []
        dispatch_lock = __import__("threading").Lock()
        call_count = [0]

        def process_fn(row: dict[str, Any], state_id: str) -> TransformResult:
            with dispatch_lock:
                call_count[0] += 1
                dispatch_times.append(time.monotonic())

            # First call on row 0 triggers capacity error (retry)
            if row["id"] == 0 and call_count[0] == 1:
                # Sleep briefly to let row 1 dispatch while we're "backing off"
                time.sleep(0.02)  # 20ms
                raise CapacityError(status_code=429, message="Rate limit")

            return TransformResult.success({"result": "success"}, success_reason={"action": "test"})

        executor = PooledExecutor(config)
        contexts = [
            RowContext(row={"id": 0}, state_id="test-0", row_index=0),
            RowContext(row={"id": 1}, state_id="test-1", row_index=1),
        ]

        executor.execute_batch(contexts, process_fn)

        # Verify we have 3 dispatches: row 0 (fail), row 1 (success), row 0 retry (success)
        assert len(dispatch_times) == 3

        # Check that ALL consecutive dispatches respect min_dispatch_delay_ms
        # This is the critical check - the bug would cause a violation here
        min_delay_s = config.min_dispatch_delay_ms / 1000
        for i in range(1, len(dispatch_times)):
            gap = dispatch_times[i] - dispatch_times[i - 1]
            # Allow 10% tolerance for timing jitter
            assert gap >= min_delay_s * 0.9, (
                f"Dispatch gap {i - 1}->{i} was {gap * 1000:.1f}ms, "
                f"expected >= {config.min_dispatch_delay_ms}ms. "
                f"This indicates the retry bypassed the dispatch gate."
            )
