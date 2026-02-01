"""Tests for Azure Multi-Query LLM transform retry behavior and concurrent processing.

Tests the FIXED implementation that uses PooledExecutor with AIMD retry.
Updated to use row-level pipelining API (BatchTransformMixin).
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import TransformResult
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

from .conftest import (
    chaosllm_azure_openai_responses,
    chaosllm_azure_openai_sequence,
    make_azure_multi_query_config,
    make_plugin_context,
    make_token,
)


def make_config(**overrides: Any) -> dict[str, Any]:
    """Create retry-specific config with extra timeout field."""
    # Set default retry timeout, but allow overrides
    defaults = {"max_capacity_retry_seconds": 10}
    defaults.update(overrides)
    return make_azure_multi_query_config(**defaults)


@contextmanager
def mock_azure_openai_with_counter(
    chaosllm_server,
    success_response: dict[str, Any],
    failure_condition: Any | None = None,
) -> Generator[tuple[Mock, list[int]], None, None]:
    """Mock Azure OpenAI with thread-safe call counter.

    Args:
        success_response: Default response data for successful calls
        failure_condition: Optional callable(call_count) -> Exception or None
                          If returns Exception, raise it; if None, succeed

    Yields:
        Tuple of (mock_client, call_count_list) where call_count_list[0] is the count
    """

    def response_factory(call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
        if failure_condition is not None:
            exc = failure_condition(call_index)
            if exc is not None:
                raise exc
        return success_response

    with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
        mock_client,
        call_count,
        _mock_azure_class,
    ):
        yield mock_client, call_count


class TestRetryBehavior:
    """Tests for capacity error retry with AIMD backoff."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_capacity_error_triggers_retry(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Capacity errors trigger automatic retry until success."""
        from openai import RateLimitError as OpenAIRateLimitError

        def failure_condition(count: int) -> OpenAIRateLimitError | None:
            if count <= 2:
                return OpenAIRateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(status_code=429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success after retry"},
            failure_condition,
        ) as (_mock_client, _call_count):
            transform = AzureMultiQueryLLMTransform(make_config())
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-retry-1")
                ctx = make_plugin_context(state_id="state-retry-1", token=token)

                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Should succeed after retries
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            # Verify all 4 queries succeeded (each tried up to 3 times)
            assert "cs1_diagnosis_score" in result.row
            assert "cs1_treatment_score" in result.row
            assert "cs2_diagnosis_score" in result.row
            assert "cs2_treatment_score" in result.row

    def test_capacity_retry_timeout(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Row fails after max_capacity_retry_seconds exceeded."""
        from openai import RateLimitError as OpenAIRateLimitError

        def always_fail(count: int) -> OpenAIRateLimitError:
            return OpenAIRateLimitError(
                message="Rate limit exceeded - never succeeds",
                response=Mock(status_code=429),
                body=None,
            )

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Never returned"},
            always_fail,
        ):
            # Short timeout for test speed
            transform = AzureMultiQueryLLMTransform(
                make_config(max_capacity_retry_seconds=1)  # 1 second timeout
            )
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-timeout-1")
                ctx = make_plugin_context(state_id="state-timeout-1", token=token)

                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Should fail after timeout
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            # Check that it's a query failure
            assert "query_failed" in result.reason["reason"]

    def test_mixed_success_and_retry(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Some queries succeed immediately, others succeed after retry."""
        from openai import RateLimitError as OpenAIRateLimitError

        def intermittent_failure(count: int) -> OpenAIRateLimitError | None:
            # Queries 1, 3, 5, 7... fail on first attempt (odd-numbered calls)
            # Only fail once per query slot
            if count % 2 == 1 and count < 10:
                return OpenAIRateLimitError(
                    message="Rate limit",
                    response=Mock(status_code=429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success"},
            intermittent_failure,
        ):
            transform = AzureMultiQueryLLMTransform(make_config())
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-mixed-1")
                ctx = make_plugin_context(state_id="state-mixed-1", token=token)

                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # All queries eventually succeed
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert "cs1_diagnosis_score" in result.row


class TestConcurrentRowProcessing:
    """Tests for concurrent row processing.

    Note: Row-level pipelining and query-level pooling are independent.
    When testing multi-row pipelining, use sequential query mode (no pool_size)
    to avoid buffer contention in the shared PooledExecutor.
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_via_pipelining(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Multiple rows processed via pipelining with sequential query execution.

        Uses sequential query mode (no pool_size) to focus on row-level pipelining
        without interference from query-level pooling.
        """
        # Use consistent response for all queries
        responses = [{"score": 85, "rationale": "R"}]

        # Sequential query execution - focus on row pipelining
        config = make_config()
        del config["pool_size"]

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # 10 rows x 4 queries = 40 total queries
                for i in range(10):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"batch-100-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 10

            # All rows succeeded
            for token, result, _state_id in collector.results:
                assert isinstance(result, TransformResult)
                assert result.status == "success", f"Row {token.row_id} failed: {result.reason}"
                assert result.row is not None
                assert "_error" not in result.row
                assert "cs1_diagnosis_score" in result.row
                assert "row_id" in result.row

            # Total calls: 40 queries
            assert mock_client.chat.completions.create.call_count == 40

    def test_atomicity_with_failures_in_sequential_mode(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Atomicity maintained when processing rows with failures.

        Uses sequential query mode to focus on row-level atomicity without
        interference from query-level pooling.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def every_7th_fails(count: int) -> OpenAIRateLimitError | None:
            if count % 7 == 0:
                return OpenAIRateLimitError(
                    message="Rate limit",
                    response=Mock(status_code=429),
                    body=None,
                )
            return None

        # Sequential query execution to focus on row atomicity
        config = make_config()
        del config["pool_size"]

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "OK"},
            every_7th_fails,
        ):
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # 20 rows x 4 queries = 80 queries
                # Every 7th query fails: 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77
                for i in range(20):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"concurrent-atomicity-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 20

            # Verify atomicity: each row has 0 or 4 output fields
            for token, result, _state_id in collector.results:
                assert isinstance(result, TransformResult)
                row = result.row if result.row is not None else {}
                output_field_count = sum(
                    [
                        "cs1_diagnosis_score" in row,
                        "cs1_treatment_score" in row,
                        "cs2_diagnosis_score" in row,
                        "cs2_treatment_score" in row,
                    ]
                )

                if result.status == "error":
                    assert output_field_count == 0, f"Row {token.row_id} has error + {output_field_count} outputs"
                else:
                    assert output_field_count == 4, f"Row {token.row_id} has {output_field_count} outputs (expected 4)"

    def test_query_level_pool_utilization(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Verify pool is utilized for query-level parallelism within a single row.

        Pool size is for query-level parallelism (multiple queries per row),
        not row-level parallelism (multiple rows simultaneously).
        """
        import time

        max_concurrent = [0]
        current_concurrent = [0]
        lock = threading.Lock()

        def response_factory(_call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
            """Track concurrent execution."""
            with lock:
                current_concurrent[0] += 1
                if current_concurrent[0] > max_concurrent[0]:
                    max_concurrent[0] = current_concurrent[0]

            # Simulate some work
            time.sleep(0.05)  # Longer delay to allow concurrency to be observed

            with lock:
                current_concurrent[0] -= 1

            return {"score": 85, "rationale": "OK"}

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            # pool_size=4, 4 queries/row -> all queries can run in parallel
            transform = AzureMultiQueryLLMTransform(make_config(pool_size=4))
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                # Single row with 4 queries
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-0")
                ctx = make_plugin_context(state_id="pool-util-0", token=token)
                transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert call_count[0] == 4

            # Max concurrent should be close to pool_size (4) or at least > 1
            # This verifies query-level parallelism is working
            assert max_concurrent[0] >= 2, f"Expected parallel query execution, got max {max_concurrent[0]} concurrent"

            print("\nQuery-level pool utilization test:")
            print("  Pool size: 4")
            print("  Queries per row: 4")
            print(f"  Max concurrent observed: {max_concurrent[0]}")
            print(f"  Pool utilized at {max_concurrent[0] / 4 * 100:.0f}%")


class TestSequentialFallback:
    """Tests for sequential processing when no executor configured."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_sequential_mode_no_retry(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Sequential mode fails query immediately on capacity error (no retry).

        Note: Uses sequential mode (no pool_size) to test FIFO ordering without
        interference from concurrent thread execution.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def first_call_fails(count: int) -> OpenAIRateLimitError | None:
            if count == 1:
                return OpenAIRateLimitError(
                    message="Rate limit",
                    response=Mock(status_code=429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success"},
            first_call_fails,
        ) as (_mock_client, call_count):
            # No pool_size = sequential mode
            config = make_config()
            del config["pool_size"]
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-seq-1")
                ctx = make_plugin_context(state_id="state-seq-1", token=token)

                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Row fails because first query failed (all-or-nothing)
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert "query_failed" in result.reason["reason"]

            # All 4 queries attempted (no retry, but all queries run once)
            assert call_count[0] == 4

            # Verify it's an immediate failure, not a retry timeout
            # error is now a string extracted from the TransformErrorReason["error"] field
            assert "Rate limit" in result.reason["failed_queries"][0]["error"]


class TestMemoryLeakPrevention:
    """Test that per-query LLM clients are properly cleaned up (memory leak prevention)."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_per_query_clients_cleaned_up_after_batch(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Per-query LLM clients should be cleaned up after batch processing.

        Regression test for P2 memory leak:
        - Each query gets unique state_id: f"{ctx.state_id}_r{row_idx}_q{query_idx}"
        - _get_llm_client caches a client per state_id
        - Without cleanup, _llm_clients grows without bound

        Uses sequential query mode (no pool_size) to focus on client cleanup
        without interference from query-level pooling race conditions.
        """
        # Use consistent response for all queries
        responses = [{"score": 90, "rationale": "Good"}]

        # Sequential query execution - focus on client cleanup behavior
        config = make_config(max_capacity_retry_seconds=10)
        del config["pool_size"]

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # Process 10 rows x 4 queries = 40 per-query clients created
                for i in range(10):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"batch-memory-leak-test-{i}", token=token)
                    transform.accept(row, ctx)

                # BEFORE FIX: _llm_clients would contain 40 cached clients
                # AFTER FIX: _llm_clients should be cleaned up to only batch client

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 10
                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    assert result.status == "success"

                # Verify all queries were executed
                assert mock_client.chat.completions.create.call_count == 40  # 10 rows x 4 queries

                # CRITICAL: Verify per-query clients were cleaned up
                # Only the batch-level client (ctx.state_id) should remain during cleanup
                # After process_batch finishes, even that gets cleaned up
                # So _llm_clients should be empty

                # Access internal state for verification
                assert len(transform._llm_clients) == 0, (
                    f"Memory leak: {len(transform._llm_clients)} clients still cached. Per-query clients should be cleaned up after batch."
                )
            finally:
                transform.close()

    def test_per_query_clients_cleaned_up_even_on_failure(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Per-query clients should be cleaned up even if batch processing fails."""
        from openai import RateLimitError as OpenAIRateLimitError

        def always_fail(count: int) -> OpenAIRateLimitError:
            return OpenAIRateLimitError(
                message="Rate limit",
                response=Mock(status_code=429),
                body=None,
            )

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 90, "rationale": "Never returned"},
            always_fail,
        ):
            config = make_config(pool_size=10, max_capacity_retry_seconds=1)
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=50)

            try:
                for i in range(5):
                    row = {
                        "cs1_bg": "data",
                        "cs1_sym": "data",
                        "cs1_hist": "data",
                        "cs2_bg": "data",
                        "cs2_sym": "data",
                        "cs2_hist": "data",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"batch-failure-cleanup-{i}", token=token)
                    transform.accept(row, ctx)

                # Process batch - all rows will fail with retry timeout
                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 5
                # All rows should have failed (either as error result or exception)
                for _, result, _state_id in collector.results:
                    if isinstance(result, TransformResult):
                        assert result.status == "error"
                    else:
                        # ExceptionResult - exception was propagated (also a failure)
                        assert isinstance(result, ExceptionResult)

                # CRITICAL: Even with failures, per-query clients should be cleaned up
                assert len(transform._llm_clients) == 0, (
                    f"Memory leak on failure: {len(transform._llm_clients)} clients still cached. "
                    "Per-query clients should be cleaned up even when queries fail."
                )
            finally:
                transform.close()


class TestLLMErrorRetry:
    """Test that retryable LLM errors are actually retried by the pool."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_network_error_is_retried(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """NetworkError should be retried by the pool (P2 bug fix).

        BEFORE FIX: LLMClientError was caught and converted to TransformResult.error
        without retryable flag, so pool never retried transient network errors.

        AFTER FIX: Retryable LLMClientErrors (NetworkError, ServerError) are re-raised,
        allowing the pool to apply AIMD retry logic.
        """
        from openai import APITimeoutError

        def first_two_fail(count: int) -> APITimeoutError | None:
            if count <= 2:
                # Timeout error (transient network issue) - matches "timeout" pattern
                # in _is_retryable_error()
                return APITimeoutError(request=Mock())
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Good"},
            first_two_fail,
        ) as (_mock_client, call_count):
            config = make_config(pool_size=4, max_capacity_retry_seconds=10)
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-network-error-1")
                ctx = make_plugin_context(state_id="network-error-retry", token=token)
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert "_error" not in result.row, "Row should succeed after retry"

            # Verify that all 4 queries were tried, some retried due to network errors
            # Each query hits the mock, some fail twice before succeeding
            assert call_count[0] > 4, f"Expected retries to happen, got {call_count[0]} calls"

    def test_content_policy_error_not_retried(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """ContentPolicyError should NOT be retried (non-retryable error).

        P2 FIX VERIFICATION: Non-retryable errors should return immediately
        with TransformResult.error(retryable=False) instead of being retried.
        """
        from openai import BadRequestError

        def always_fail_content_policy(count: int) -> BadRequestError:
            # Content policy violation is a 400 error
            return BadRequestError(
                message="Content violates safety policy",
                response=Mock(status_code=400),
                body={"error": {"code": "content_filter", "message": "Content violates safety policy"}},
            )

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Never returned"},
            always_fail_content_policy,
        ) as (_mock_client, call_count):
            config = make_config(pool_size=4, max_capacity_retry_seconds=10)
            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = PluginContext(run_id="test", config={}, landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-content-policy-1")
                ctx = make_plugin_context(state_id="content-policy-no-retry", token=token)
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error", "Row should have error"

            # Verify ContentPolicyError caused immediate failure (no retries)
            # 4 queries, each called exactly once (no retries)
            assert call_count[0] == 4, f"Expected 4 calls (no retries), got {call_count[0]}"
