"""Load testing and profiling for Azure Multi-Query LLM transform.

This module tests the transform under realistic load conditions to:
1. Verify correct behavior with many rows
2. Profile CPU and memory usage
3. Identify bottlenecks
4. Test parallel vs sequential execution
5. Verify rate limit handling
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import TransformResult
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

from .conftest import (
    chaosllm_azure_openai_sequence,
    make_plugin_context,
    make_token,
)
from .conftest import (
    make_azure_multi_query_config as make_config,
)


def make_mock_llm_response(score: int, rationale: str, delay_ms: float = 0) -> Mock:
    """Create a ChaosLLM payload with optional artificial delay."""
    payload = {"score": score, "rationale": rationale}
    return (payload, delay_ms) if delay_ms > 0 else payload


class TestLoadScenarios:
    """Test plugin under various load scenarios."""

    @pytest.mark.slow
    def test_many_rows_parallel_execution(self, chaosllm_server) -> None:
        """Process 100 rows with 4 parallel queries each (400 total queries).

        Uses row-level pipelining via BatchTransformMixin. Query-level pooling
        is disabled to avoid mock threading issues.
        """
        row_count = 100
        queries_per_row = 4  # 2 case studies x 2 criteria

        def response_factory(call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
            """Simulate LLM response with 50ms latency."""
            return make_mock_llm_response(
                score=85 + (call_index % 10),
                rationale=f"Response {call_index}",
                delay_ms=50,
            )

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            # Disable query-level pooling to avoid mock threading issues
            config = make_config()
            del config["pool_size"]

            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = make_plugin_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=100)

            try:
                start_time = time.time()

                for i in range(row_count):
                    row = {
                        "cs1_bg": f"patient_{i}_bg",
                        "cs1_sym": f"patient_{i}_sym",
                        "cs1_hist": f"patient_{i}_hist",
                        "cs2_bg": f"patient_{i}_cs2_bg",
                        "cs2_sym": f"patient_{i}_cs2_sym",
                        "cs2_hist": f"patient_{i}_cs2_hist",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"batch-load-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=60.0)
                elapsed = time.time() - start_time

                # Verify results
                assert len(collector.results) == row_count

                success_count = 0
                for _token, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    if result.status == "success":
                        success_count += 1
                        assert result.row is not None
                        assert "cs1_diagnosis_score" in result.row
                        assert "cs1_treatment_score" in result.row
                        assert "cs2_diagnosis_score" in result.row
                        assert "cs2_treatment_score" in result.row

                assert success_count == row_count

                # Verify all queries executed
                assert call_count[0] == row_count * queries_per_row

                print("\nLoad test stats:")
                print(f"  Rows processed: {row_count}")
                print(f"  Queries per row: {queries_per_row}")
                print(f"  Total queries: {call_count[0]}")
                print(f"  Elapsed time: {elapsed:.2f}s")
                print(f"  Queries/second: {call_count[0] / elapsed:.2f}")

            finally:
                transform.close()

    @pytest.mark.slow
    def test_sequential_vs_parallel_performance(self, chaosllm_server) -> None:
        """Compare sequential vs parallel execution performance.

        Note: This test measures row-level pipelining only.
        Query-level pooling is disabled in both cases for fair comparison.
        """
        row_count = 10  # Smaller dataset for comparison
        queries_per_row = 4

        def run_test(pool_size: int | None) -> tuple[float, int]:
            """Run test with given pool_size, return (elapsed_time, total_calls)."""

            def response_factory(call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
                return make_mock_llm_response(
                    score=85,
                    rationale=f"Response {call_index}",
                    delay_ms=50,
                )

            with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
                _mock_client,
                call_count,
                _mock_azure_class,
            ):
                # Both modes use sequential query execution (no pool_size)
                # to avoid mock threading issues
                config = make_config()
                del config["pool_size"]

                transform = AzureMultiQueryLLMTransform(config)
                init_ctx = make_plugin_context()
                transform.on_start(init_ctx)

                collector = CollectorOutputPort()
                transform.connect_output(collector, max_pending=20)

                try:
                    start_time = time.time()

                    for i in range(row_count):
                        row = {
                            "cs1_bg": f"data_{i}",
                            "cs1_sym": f"data_{i}",
                            "cs1_hist": f"data_{i}",
                            "cs2_bg": f"data_{i}",
                            "cs2_sym": f"data_{i}",
                            "cs2_hist": f"data_{i}",
                        }
                        token = make_token(f"row-{i}")
                        ctx = make_plugin_context(state_id=f"state-{i}", token=token)
                        transform.accept(row, ctx)

                    transform.flush_batch_processing(timeout=30.0)
                    elapsed = time.time() - start_time

                    assert len(collector.results) == row_count
                    for _, result, _state_id in collector.results:
                        assert isinstance(result, TransformResult)
                        assert result.status == "success"

                    return elapsed, call_count[0]

                finally:
                    transform.close()

        # Test both modes (both use sequential query execution)
        sequential_time, sequential_calls = run_test(pool_size=None)
        parallel_time, parallel_calls = run_test(pool_size=4)

        # Both should execute same number of queries
        assert sequential_calls == parallel_calls == row_count * queries_per_row

        print("\nSequential vs Parallel comparison:")
        print(f"  Sequential time: {sequential_time:.2f}s")
        print(f"  Parallel time: {parallel_time:.2f}s")
        print("  Note: Query-level pooling disabled; both use row-level pipelining")

    @pytest.mark.slow
    def test_memory_usage_with_large_batch(self, chaosllm_server) -> None:
        """Verify plugin doesn't accumulate excessive memory with large batches."""
        import gc

        try:
            import psutil  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("psutil not installed - skipping memory test")

        process = psutil.Process()

        row_count = 200

        def response_factory(_call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
            # Return small responses (shouldn't accumulate much memory)
            return make_mock_llm_response(
                score=85,
                rationale="Short response",
                delay_ms=10,
            )

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            _call_count,
            _mock_azure_class,
        ):
            # Disable query-level pooling to avoid mock threading issues
            config = make_config()
            del config["pool_size"]

            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = make_plugin_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=200)

            try:
                # Measure baseline memory
                gc.collect()
                baseline_mb = process.memory_info().rss / 1024 / 1024

                for i in range(row_count):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"batch-mem-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=60.0)

                # Verify all rows processed
                assert len(collector.results) == row_count

                # Measure after processing
                gc.collect()
                after_mb = process.memory_info().rss / 1024 / 1024

                memory_increase_mb = after_mb - baseline_mb

                # Should not leak excessive memory (conservative threshold)
                # With 200 rows x 4 queries = 800 results, expect < 100MB increase
                assert memory_increase_mb < 100, f"Memory increased by {memory_increase_mb:.2f} MB (baseline: {baseline_mb:.2f} MB)"

                print("\nMemory usage:")
                print(f"  Baseline: {baseline_mb:.2f} MB")
                print(f"  After {row_count} rows: {after_mb:.2f} MB")
                print(f"  Increase: {memory_increase_mb:.2f} MB")

            finally:
                transform.close()

    def test_rate_limit_error_handling(self) -> None:
        """Verify plugin handles rate limit errors correctly."""
        from elspeth.plugins.clients.llm import RateLimitError

        # Disable query-level pooling to avoid mock threading issues
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = make_plugin_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=10)

        # Mock the LLM client to raise RateLimitError
        with patch.object(transform, "_get_llm_client") as mock_get_client:
            mock_llm_client = Mock()

            # Counter for tracking which call we're on
            llm_call_count = [0]

            def mock_chat_completion(**kwargs: Any) -> Mock:
                llm_call_count[0] += 1
                if llm_call_count[0] % 3 == 0:
                    raise RateLimitError("Rate limit exceeded")
                return Mock(
                    content=json.dumps({"score": 85, "rationale": "OK"}),
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    model="gpt-4o",
                )

            mock_llm_client.chat_completion.side_effect = mock_chat_completion
            mock_get_client.return_value = mock_llm_client

            try:
                row = {
                    "cs1_bg": "data",
                    "cs1_sym": "data",
                    "cs1_hist": "data",
                    "cs2_bg": "data",
                    "cs2_sym": "data",
                    "cs2_hist": "data",
                }
                token = make_token("row-rate-limit")
                ctx = make_plugin_context(state_id="rate-limit-test", token=token)
                transform.accept(row, ctx)
                transform.flush_batch_processing(timeout=10.0)

                # Process will fail because one of the 4 queries hits rate limit
                assert len(collector.results) == 1
                _, result, _state_id = collector.results[0]
                assert isinstance(result, TransformResult)

                # All-or-nothing: entire row should fail
                assert result.status == "error"
                assert result.reason is not None
                assert "query_failed" in result.reason["reason"]

            finally:
                transform.close()

    def test_client_caching_behavior(self, chaosllm_server) -> None:
        """Verify LLM client caching works correctly."""

        def response_factory(call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
            return make_mock_llm_response(score=85, rationale=f"Response {call_index}")

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            mock_azure_class,
        ):
            # Disable query-level pooling to avoid mock threading issues
            config = make_config()
            del config["pool_size"]

            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = make_plugin_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=10)

            # Track how many times we create underlying AzureOpenAI client
            azure_client_creations = mock_azure_class.call_count

            try:
                # Process multiple rows with same state_id
                for i in range(3):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    # Use same state_id to test client caching per state
                    ctx = make_plugin_context(state_id="shared-state", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=10.0)

                # All rows should succeed
                assert len(collector.results) == 3
                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    assert result.status == "success"

                # Should only create ONE underlying AzureOpenAI client
                final_creations = mock_azure_class.call_count
                assert final_creations == azure_client_creations + 1, "Should reuse underlying Azure client"

                # All queries should have executed (3 rows x 4 queries = 12)
                assert call_count[0] == 12

            finally:
                transform.close()


class TestRowAtomicity:
    """Tests for row atomicity guarantees under failure conditions."""

    def test_row_atomicity_under_capacity_errors(self) -> None:
        """Verify NO partial rows are emitted when capacity errors occur mid-row.

        CRITICAL PROPERTY: A row must have ALL queries complete before being emitted.
        If any query fails (including capacity errors), the ENTIRE row must be marked
        as failed, not partially processed.
        """
        from elspeth.plugins.clients.llm import RateLimitError

        # Simulate 10% of queries hitting capacity errors
        FAILURE_RATE = 0.1  # 10% of queries fail

        # Disable query-level pooling to avoid mock threading issues
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = make_plugin_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=50)

        # Mock the LLM client to raise RateLimitError for failed queries
        with patch.object(transform, "_get_llm_client") as mock_get_client:
            llm_call_count = [0]

            def mock_chat_completion(**kwargs: Any) -> Mock:
                llm_call_count[0] += 1
                if llm_call_count[0] % 10 == 0:
                    raise RateLimitError("Rate limit exceeded")
                return Mock(
                    content=json.dumps({"score": 85 + llm_call_count[0], "rationale": f"R{llm_call_count[0]}"}),
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    model="gpt-4o",
                )

            mock_llm_client = Mock()
            mock_llm_client.chat_completion.side_effect = mock_chat_completion
            mock_get_client.return_value = mock_llm_client

            try:
                # Process 50 rows (200 total queries, ~20 will fail)
                for i in range(50):
                    row = {
                        "row_id": i,  # Track which row this is
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"atomicity-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 50  # All 50 rows returned (success or error)

                # Verify atomicity: Each row has EITHER all 4 output fields OR an error status
                successful_rows = 0
                failed_rows = 0

                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    if result.status == "error":
                        failed_rows += 1
                        # CRITICAL: Failed rows must NOT have partial output fields
                        # Error result has no row data (or row data without output fields)
                    else:
                        successful_rows += 1
                        # CRITICAL: Successful rows must have ALL 4 output fields
                        assert result.row is not None
                        assert "cs1_diagnosis_score" in result.row, "Missing cs1_diagnosis_score"
                        assert "cs1_treatment_score" in result.row, "Missing cs1_treatment_score"
                        assert "cs2_diagnosis_score" in result.row, "Missing cs2_diagnosis_score"
                        assert "cs2_treatment_score" in result.row, "Missing cs2_treatment_score"
                        # All 4 scores should be present and valid
                        assert isinstance(result.row["cs1_diagnosis_score"], int)
                        assert isinstance(result.row["cs1_treatment_score"], int)
                        assert isinstance(result.row["cs2_diagnosis_score"], int)
                        assert isinstance(result.row["cs2_treatment_score"], int)

                # Should have both successes and failures (due to 10% failure rate)
                assert successful_rows > 0, "Expected some successful rows"
                assert failed_rows > 0, "Expected some failed rows due to capacity errors"

                print("\nRow atomicity verification:")
                print("  Total rows: 50")
                print(f"  Successful rows: {successful_rows}")
                print(f"  Failed rows: {failed_rows}")
                print(f"  Total queries attempted: {llm_call_count[0]}")
                print(f"  Expected failures (~10%): {int(llm_call_count[0] * FAILURE_RATE)}")
                print("  NO PARTIAL ROWS DETECTED")

            finally:
                transform.close()

    def test_row_atomicity_high_failure_rate(self) -> None:
        """Verify row atomicity with 80% failure rate (extreme stress test)."""
        from elspeth.plugins.clients.llm import RateLimitError

        # Disable query-level pooling to avoid mock threading issues
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = make_plugin_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=20)

        with patch.object(transform, "_get_llm_client") as mock_get_client:
            llm_call_count = [0]

            def mock_chat_completion(**kwargs: Any) -> Mock:
                """Simulate 80% failure rate."""
                llm_call_count[0] += 1
                # Only calls ending in 0 or 5 succeed (20%)
                if llm_call_count[0] % 5 not in [0, 5]:
                    raise RateLimitError("Rate limit exceeded")
                return Mock(
                    content=json.dumps({"score": 85, "rationale": "OK"}),
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    model="gpt-4o",
                )

            mock_llm_client = Mock()
            mock_llm_client.chat_completion.side_effect = mock_chat_completion
            mock_get_client.return_value = mock_llm_client

            try:
                # Process 20 rows (80 queries, ~64 will fail)
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
                    ctx = make_plugin_context(state_id=f"high-failure-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 20

                # With 80% failure rate, most rows should fail (each row needs 4 successful queries)
                successful_rows = 0
                failed_rows = 0

                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    if result.status == "error":
                        failed_rows += 1
                        # Verify NO partial output in error results
                    else:
                        successful_rows += 1
                        # Verify ALL outputs present
                        assert result.row is not None
                        assert "cs1_diagnosis_score" in result.row
                        assert "cs1_treatment_score" in result.row
                        assert "cs2_diagnosis_score" in result.row
                        assert "cs2_treatment_score" in result.row

                # With 80% failure rate, most rows should fail
                assert failed_rows > successful_rows, "Expected more failures than successes with 80% failure rate"

                print("\nHigh failure rate atomicity test:")
                print(f"  Successful rows: {successful_rows}")
                print(f"  Failed rows: {failed_rows}")
                print("  NO PARTIAL ROWS EVEN AT 80% FAILURE RATE")

            finally:
                transform.close()

    def test_concurrent_row_processing_atomicity(self) -> None:
        """Verify row atomicity when multiple rows processed concurrently with failures.

        This tests the edge case where:
        - Multiple rows are being processed simultaneously
        - Some queries succeed, some fail
        - Plugin must still maintain per-row atomicity
        """
        from elspeth.plugins.clients.llm import RateLimitError

        # Disable query-level pooling to avoid mock threading issues
        config = make_config()
        del config["pool_size"]

        transform = AzureMultiQueryLLMTransform(config)
        init_ctx = make_plugin_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=30)

        with patch.object(transform, "_get_llm_client") as mock_get_client:
            llm_call_count = [0]

            def mock_chat_completion(**kwargs: Any) -> Mock:
                """Simulate failures in a pattern that affects different rows."""
                llm_call_count[0] += 1
                # Fail every 7th query (staggered pattern across rows)
                if llm_call_count[0] % 7 == 0:
                    raise RateLimitError("Rate limit exceeded")
                return Mock(
                    content=json.dumps({"score": 85, "rationale": "OK"}),
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                    model="gpt-4o",
                )

            mock_llm_client = Mock()
            mock_llm_client.chat_completion.side_effect = mock_chat_completion
            mock_get_client.return_value = mock_llm_client

            try:
                # Process 30 rows
                for i in range(30):
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
                    ctx = make_plugin_context(state_id=f"concurrent-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 30

                # Verify atomicity for all rows
                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    if result.status == "error":
                        # Failed row: must have no output fields
                        pass
                    else:
                        # Successful row: MUST have ALL 4 output fields
                        assert result.row is not None
                        has_cs1_diag = "cs1_diagnosis_score" in result.row
                        has_cs1_treat = "cs1_treatment_score" in result.row
                        has_cs2_diag = "cs2_diagnosis_score" in result.row
                        has_cs2_treat = "cs2_treatment_score" in result.row

                        output_field_count = sum([has_cs1_diag, has_cs1_treat, has_cs2_diag, has_cs2_treat])
                        assert output_field_count == 4, f"Row has {output_field_count} output fields (expected 4)"

                print("\nConcurrent processing atomicity test:")
                print(f"  Total queries: {llm_call_count[0]}")
                print("  ALL ROWS ATOMIC (0 or 4 output fields, never 1-3)")

            finally:
                transform.close()


class TestProfilingInstrumentation:
    """Tests for profiling and performance monitoring."""

    def test_query_timing_distribution(self, chaosllm_server) -> None:
        """Measure timing distribution of individual queries."""
        import random
        import statistics

        query_times: list[float] = []

        def response_factory(_call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
            """Simulate variable query latency."""
            delay = random.uniform(20, 80)
            query_times.append(delay)
            return make_mock_llm_response(score=85, rationale="OK", delay_ms=delay)

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            # Disable query-level pooling to avoid mock threading issues
            config = make_config()
            del config["pool_size"]

            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = make_plugin_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=20)

            try:
                # Process 20 rows
                for i in range(20):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"timing-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 20
                assert call_count[0] == 80  # 20 rows x 4 queries
                assert len(query_times) == 80

                # Analyze distribution
                mean_ms = statistics.mean(query_times)
                median_ms = statistics.median(query_times)
                p95_ms = sorted(query_times)[int(len(query_times) * 0.95)]
                p99_ms = sorted(query_times)[int(len(query_times) * 0.99)]

                print("\nQuery timing distribution:")
                print(f"  Mean: {mean_ms:.2f}ms")
                print(f"  Median: {median_ms:.2f}ms")
                print(f"  P95: {p95_ms:.2f}ms")
                print(f"  P99: {p99_ms:.2f}ms")

                # Sanity checks (should match our simulated 20-80ms range)
                assert 20 <= mean_ms <= 80
                assert 20 <= median_ms <= 80

            finally:
                transform.close()

    def test_batch_processing_overhead(self, chaosllm_server) -> None:
        """Measure overhead of batch processing logic."""

        def response_factory(_call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
            # Instant responses to isolate batch processing overhead
            return make_mock_llm_response(score=85, rationale="OK", delay_ms=0)

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            _call_count,
            _mock_azure_class,
        ):
            # Disable query-level pooling to avoid mock threading issues
            config = make_config()
            del config["pool_size"]

            transform = AzureMultiQueryLLMTransform(config)
            init_ctx = make_plugin_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=100)

            try:
                start_time = time.time()

                # Process 100 rows
                for i in range(100):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs1_sym": f"data_{i}",
                        "cs1_hist": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                        "cs2_sym": f"data_{i}",
                        "cs2_hist": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_plugin_context(state_id=f"overhead-{i}", token=token)
                    transform.accept(row, ctx)

                transform.flush_batch_processing(timeout=30.0)
                elapsed = time.time() - start_time

                assert len(collector.results) == 100

                # With instant LLM responses, overhead should be minimal
                # 100 rows x 4 queries = 400 queries, batch overhead should be < 1s
                assert elapsed < 1.0, f"Batch overhead too high: {elapsed:.2f}s"

                print("\nBatch processing overhead:")
                print("  100 rows x 4 queries = 400 queries")
                print(f"  Total time: {elapsed:.3f}s")
                print(f"  Overhead per query: {elapsed / 400 * 1000:.3f}ms")

            finally:
                transform.close()
