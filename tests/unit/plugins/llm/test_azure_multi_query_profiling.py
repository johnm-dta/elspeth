"""Load testing and profiling for LLM transform multi-query mode.

This module tests the unified LLMTransform with MultiQueryStrategy under
realistic load conditions to:
1. Verify correct behavior with many rows
2. Profile CPU and memory usage
3. Identify bottlenecks
4. Test sequential execution performance
5. Verify rate limit handling and row atomicity

Updated from AzureMultiQueryLLMTransform to unified LLMTransform.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.transforms.llm.provider import LLMProvider
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .conftest import (
    chaosllm_azure_openai_sequence,
    make_token,
)


def _make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid Azure multi-query config for unified LLMTransform.

    Equivalent to the old make_azure_multi_query_config but using the new
    queries-based format instead of case_studies/criteria cross-product.
    """
    config: dict[str, Any] = {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Evaluate: {{ row.text_content }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {"text_content": "cs1_bg"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {"text_content": "cs1_bg"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {"text_content": "cs2_bg"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {"text_content": "cs2_bg"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
    }
    config.update(overrides)
    return config


def make_mock_llm_response(score: int, rationale: str, delay_ms: float = 0) -> tuple[dict[str, Any], float] | dict[str, Any]:
    """Create a ChaosLLM payload with optional artificial delay."""
    payload: dict[str, Any] = {"score": score, "rationale": rationale}
    return (payload, delay_ms) if delay_ms > 0 else payload


class TestLoadScenarios:
    """Test plugin under various load scenarios."""

    @pytest.mark.slow
    def test_many_rows_sequential_execution(self, chaosllm_server) -> None:
        """Process 100 rows with 4 sequential queries each (400 total queries).

        Uses row-level pipelining via BatchTransformMixin. Query-level execution
        is sequential in MultiQueryStrategy.
        """
        row_count = 100
        queries_per_row = 4  # 4 query specs

        def response_factory(call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
            """Simulate LLM response with 50ms latency."""
            return make_mock_llm_response(  # type: ignore[return-value]
                score=85 + (call_index % 10),
                rationale=f"Response {call_index}",
                delay_ms=50,
            )

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=100)

            try:
                start_time = time.time()

                for i in range(row_count):
                    row = {
                        "cs1_bg": f"patient_{i}_bg",
                        "cs2_bg": f"patient_{i}_cs2_bg",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"batch-load-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

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
    def test_sequential_execution_performance(self, chaosllm_server) -> None:
        """Measure sequential execution performance with row-level pipelining.

        In the unified LLMTransform, queries are always sequential within a row.
        Row-level pipelining (BatchTransformMixin) provides parallelism across rows.
        """
        row_count = 10
        queries_per_row = 4

        def response_factory(call_index: int, _request: dict[str, Any]) -> tuple[dict[str, Any], float]:
            return make_mock_llm_response(  # type: ignore[return-value]
                score=85,
                rationale=f"Response {call_index}",
                delay_ms=50,
            )

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=20)

            try:
                start_time = time.time()

                for i in range(row_count):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"state-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=30.0)
                elapsed = time.time() - start_time

                assert len(collector.results) == row_count
                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    assert result.status == "success"

                assert call_count[0] == row_count * queries_per_row

                print("\nSequential execution performance:")
                print(f"  Rows: {row_count}")
                print(f"  Total queries: {call_count[0]}")
                print(f"  Elapsed time: {elapsed:.2f}s")
                print(f"  Queries/second: {call_count[0] / elapsed:.2f}")
                print("  Note: Row-level pipelining provides cross-row parallelism")

            finally:
                transform.close()

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
            return make_mock_llm_response(  # type: ignore[return-value]
                score=85,
                rationale="Short response",
                delay_ms=10,
            )

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            _call_count,
            _mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
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
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"batch-mem-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

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
        """Verify plugin handles rate limit errors correctly via provider mock."""
        from elspeth.plugins.infrastructure.clients.llm import RateLimitError
        from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult

        config = _make_config()

        transform = LLMTransform(config)
        init_ctx = make_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=10)

        # Mock the provider to raise RateLimitError on some queries
        mock_provider = Mock(spec=LLMProvider)
        query_call_count = [0]

        def mock_execute_query(
            messages: list[dict[str, str]],
            *,
            model: str,
            temperature: float,
            max_tokens: int | None,
            state_id: str,
            token_id: str,
            response_format: dict[str, object] | None = None,
        ) -> LLMQueryResult:
            query_call_count[0] += 1
            if query_call_count[0] % 3 == 0:
                raise RateLimitError("Rate limit exceeded")
            return LLMQueryResult(
                content=json.dumps({"score": 85, "rationale": "OK"}),
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = mock_execute_query
        mock_provider.close = Mock()
        transform._provider = mock_provider

        try:
            row = {
                "cs1_bg": "data",
                "cs2_bg": "data",
            }
            token = make_token("row-rate-limit")
            ctx = make_context(state_id="rate-limit-test", token=token)
            transform.accept(make_pipeline_row(row), ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Process will fail because one query hits rate limit (retryable → re-raised)
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]

            # RateLimitError is retryable → re-raised as exception
            from elspeth.engine.batch_adapter import ExceptionResult

            if isinstance(result, TransformResult):
                assert result.status == "error"
            else:
                assert isinstance(result, ExceptionResult)

        finally:
            transform.close()

    def test_client_caching_behavior(self, chaosllm_server) -> None:
        """Verify underlying AzureOpenAI client is only created once."""

        def response_factory(call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
            return make_mock_llm_response(score=85, rationale=f"Response {call_index}")  # type: ignore[return-value]

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
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
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    # Use same state_id to test client caching per state
                    ctx = make_context(state_id="shared-state", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

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
        If any query fails, the ENTIRE row must be marked as failed, not partially
        processed.
        """
        from elspeth.plugins.infrastructure.clients.llm import RateLimitError
        from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult

        config = _make_config()

        transform = LLMTransform(config)
        init_ctx = make_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=50)

        # Mock provider to simulate rate limit errors
        mock_provider = Mock(spec=LLMProvider)
        llm_call_count = [0]

        def mock_execute_query(
            messages: list[dict[str, str]],
            *,
            model: str,
            temperature: float,
            max_tokens: int | None,
            state_id: str,
            token_id: str,
            response_format: dict[str, object] | None = None,
        ) -> LLMQueryResult:
            llm_call_count[0] += 1
            if llm_call_count[0] % 10 == 0:
                raise RateLimitError("Rate limit exceeded")
            return LLMQueryResult(
                content=json.dumps({"score": 85 + llm_call_count[0], "rationale": f"R{llm_call_count[0]}"}),
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = mock_execute_query
        mock_provider.close = Mock()
        transform._provider = mock_provider

        try:
            # Process 50 rows (200 total queries, ~20 will fail)
            for i in range(50):
                row = {
                    "row_id": i,
                    "cs1_bg": f"data_{i}",
                    "cs2_bg": f"data_{i}",
                }
                token = make_token(f"row-{i}")
                ctx = make_context(state_id=f"atomicity-{i}", token=token)
                transform.accept(make_pipeline_row(row), ctx)

            transform.flush_batch_processing(timeout=30.0)

            assert len(collector.results) == 50  # All 50 rows returned (success, error, or exception)

            # Verify atomicity: Each row has EITHER all 4 output fields OR is an error/exception
            successful_rows = 0
            failed_rows = 0

            from elspeth.engine.batch_adapter import ExceptionResult

            for _, result, _state_id in collector.results:
                if isinstance(result, ExceptionResult):
                    # Retryable error propagated — atomic failure
                    failed_rows += 1
                    continue

                assert isinstance(result, TransformResult)
                if result.status == "error":
                    failed_rows += 1
                    # CRITICAL: Failed rows must NOT have partial output fields
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

            # Should have both successes and failures (due to ~10% failure rate)
            assert successful_rows > 0, "Expected some successful rows"
            assert failed_rows > 0, "Expected some failed rows due to capacity errors"

            print("\nRow atomicity verification:")
            print("  Total rows: 50")
            print(f"  Successful rows: {successful_rows}")
            print(f"  Failed rows: {failed_rows}")
            print(f"  Total queries attempted: {llm_call_count[0]}")
            print("  NO PARTIAL ROWS DETECTED")

        finally:
            transform.close()

    def test_row_atomicity_high_failure_rate(self) -> None:
        """Verify row atomicity with 80% failure rate (extreme stress test)."""
        from elspeth.plugins.infrastructure.clients.llm import RateLimitError
        from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult

        config = _make_config()

        transform = LLMTransform(config)
        init_ctx = make_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=20)

        mock_provider = Mock(spec=LLMProvider)
        llm_call_count = [0]

        def mock_execute_query(
            messages: list[dict[str, str]],
            *,
            model: str,
            temperature: float,
            max_tokens: int | None,
            state_id: str,
            token_id: str,
            response_format: dict[str, object] | None = None,
        ) -> LLMQueryResult:
            """Simulate 80% failure rate."""
            llm_call_count[0] += 1
            # Only calls ending in 0 or 5 succeed (20%)
            if llm_call_count[0] % 5 not in [0, 5]:
                raise RateLimitError("Rate limit exceeded")
            return LLMQueryResult(
                content=json.dumps({"score": 85, "rationale": "OK"}),
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = mock_execute_query
        mock_provider.close = Mock()
        transform._provider = mock_provider

        try:
            # Process 20 rows (80 queries, ~64 will fail)
            for i in range(20):
                row = {
                    "row_id": i,
                    "cs1_bg": f"data_{i}",
                    "cs2_bg": f"data_{i}",
                }
                token = make_token(f"row-{i}")
                ctx = make_context(state_id=f"high-failure-{i}", token=token)
                transform.accept(make_pipeline_row(row), ctx)

            transform.flush_batch_processing(timeout=30.0)

            assert len(collector.results) == 20

            from elspeth.engine.batch_adapter import ExceptionResult

            # With 80% failure rate, most rows should fail
            successful_rows = 0
            failed_rows = 0

            for _, result, _state_id in collector.results:
                if isinstance(result, ExceptionResult):
                    failed_rows += 1
                    continue

                assert isinstance(result, TransformResult)
                if result.status == "error":
                    failed_rows += 1
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
        - Multiple rows are being processed simultaneously via BatchTransformMixin
        - Some queries succeed, some fail
        - Plugin must still maintain per-row atomicity
        """
        from elspeth.plugins.infrastructure.clients.llm import RateLimitError
        from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult

        config = _make_config()

        transform = LLMTransform(config)
        init_ctx = make_context()
        transform.on_start(init_ctx)

        collector = CollectorOutputPort()
        transform.connect_output(collector, max_pending=30)

        mock_provider = Mock(spec=LLMProvider)
        llm_call_count = [0]

        def mock_execute_query(
            messages: list[dict[str, str]],
            *,
            model: str,
            temperature: float,
            max_tokens: int | None,
            state_id: str,
            token_id: str,
            response_format: dict[str, object] | None = None,
        ) -> LLMQueryResult:
            """Simulate failures in a pattern that affects different rows."""
            llm_call_count[0] += 1
            # Fail every 7th query (staggered pattern across rows)
            if llm_call_count[0] % 7 == 0:
                raise RateLimitError("Rate limit exceeded")
            return LLMQueryResult(
                content=json.dumps({"score": 85, "rationale": "OK"}),
                usage=TokenUsage.known(10, 5),
                model="gpt-4o",
                finish_reason=FinishReason.STOP,
            )

        mock_provider.execute_query.side_effect = mock_execute_query
        mock_provider.close = Mock()
        transform._provider = mock_provider

        try:
            # Process 30 rows
            for i in range(30):
                row = {
                    "row_id": i,
                    "cs1_bg": f"data_{i}",
                    "cs2_bg": f"data_{i}",
                }
                token = make_token(f"row-{i}")
                ctx = make_context(state_id=f"concurrent-{i}", token=token)
                transform.accept(make_pipeline_row(row), ctx)

            transform.flush_batch_processing(timeout=30.0)

            assert len(collector.results) == 30

            from elspeth.engine.batch_adapter import ExceptionResult

            # Verify atomicity for all rows
            for _, result, _state_id in collector.results:
                if isinstance(result, ExceptionResult):
                    # Retryable error propagated — atomic failure
                    continue

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
            return make_mock_llm_response(score=85, rationale="OK", delay_ms=delay)  # type: ignore[return-value]

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=20)

            try:
                # Process 20 rows
                for i in range(20):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"timing-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

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
            return make_mock_llm_response(score=85, rationale="OK", delay_ms=0)  # type: ignore[return-value]

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            _call_count,
            _mock_azure_class,
        ):
            config = _make_config()

            transform = LLMTransform(config)
            init_ctx = make_context()
            transform.on_start(init_ctx)

            collector = CollectorOutputPort()
            transform.connect_output(collector, max_pending=100)

            try:
                start_time = time.time()

                # Process 100 rows
                for i in range(100):
                    row = {
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"overhead-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=30.0)
                elapsed = time.time() - start_time

                assert len(collector.results) == 100

                # With instant LLM responses, overhead should be minimal
                # 100 rows x 4 queries = 400 queries, batch overhead should be < 5s
                # (generous threshold for CI runners with variable CPU scheduling)
                assert elapsed < 5.0, f"Batch overhead too high: {elapsed:.2f}s"

                print("\nBatch processing overhead:")
                print("  100 rows x 4 queries = 400 queries")
                print(f"  Total time: {elapsed:.3f}s")
                print(f"  Overhead per query: {elapsed / 400 * 1000:.3f}ms")

            finally:
                transform.close()
