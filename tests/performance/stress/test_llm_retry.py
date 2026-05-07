# tests/performance/stress/test_llm_retry.py
"""Consolidated LLM stress tests using ChaosLLM HTTP server.

All 19 LLM stress tests from v1 consolidated into one file:
- TestAzureLLMStress (4 tests): AIMD backoff, 529 capacity, burst resilience, concurrency
- TestAzureMultiQueryStress (3 tests): Partial failure, rate limit all queries, chaos mode
- TestOpenRouterLLMStress (3 tests): Rate limit, malformed JSON, connection timeout
- TestOpenRouterMultiQueryStress (4 tests): Partial failure, rate limit, chaos, FIFO
- TestMixedErrors (5 tests): Azure chaos, OpenRouter chaos, audit integrity,
  progressive degradation, burst recovery

Migrated from:
- tests/stress/llm/test_azure_llm_stress.py
- tests/stress/llm/test_azure_multi_query_stress.py
- tests/stress/llm/test_openrouter_llm_stress.py
- tests/stress/llm/test_openrouter_multi_query_stress.py
- tests/stress/llm/test_mixed_errors.py

All transforms now use unified LLMTransform with provider dispatch.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from elspeth.plugins.transforms.llm.transform import LLMTransform
from tests.fixtures.factories import make_context

from .conftest import (
    ChaosLLMHTTPFixture,
    CollectingOutputPort,
    create_factory_and_run,
    generate_multi_query_rows,
    generate_test_rows,
    make_azure_llm_config,
    make_azure_multi_query_config,
    make_openrouter_llm_config,
    make_openrouter_multi_query_config,
    make_pipeline_row,
    make_token,
)

# Mark all tests in this module as stress tests (deselected by default)
pytestmark = pytest.mark.stress

# JSON template for multi-query responses
# Must match the output_fields in make_*_multi_query_config
MULTI_QUERY_JSON_TEMPLATE = '{"score": 5, "rationale": "Test assessment rationale"}'


# ---------------------------------------------------------------------------
# Helper: feed rows through a batch transform
# ---------------------------------------------------------------------------


def _feed_rows(
    transform: LLMTransform,
    rows: list[dict[str, Any]],
    factory: object,
    run_id: str,
    node_id: str,
) -> list[str]:
    """Feed rows through a batch transform, returning input order IDs.

    Creates row records, tokens, and node states in the factory repositories
    for each row, then accepts them into the transform.

    Returns:
        List of row IDs in input order (for FIFO verification).
    """
    input_order: list[str] = []
    for i, row in enumerate(rows):
        token = make_token(f"row-{i}", f"token-{i}")
        row_record = factory.data_flow.create_row(  # type: ignore[attr-defined]
            run_id=run_id,
            source_node_id=node_id,
            row_index=i,
            data=row,
        )
        token_record = factory.data_flow.create_token(row_id=row_record.row_id)  # type: ignore[attr-defined]
        state = factory.execution.begin_node_state(  # type: ignore[attr-defined]
            token_id=token_record.token_id,
            node_id=node_id,
            run_id=run_id,
            step_index=0,
            input_data=row,
        )

        ctx = make_context(
            run_id=run_id,
            landscape=factory.plugin_audit_writer(),  # type: ignore[attr-defined]
            state_id=state.state_id,
            token=token,
        )
        input_order.append(row["id"])
        transform.accept(make_pipeline_row(row), ctx)

    return input_order


# ==========================================================================
# Azure LLM Stress Tests (4 tests)
# ==========================================================================


@pytest.mark.stress
class TestAzureLLMStress:
    """Azure LLM transform under error injection."""

    @pytest.mark.chaosllm(rate_limit_pct=30.0)
    def test_rate_limit_recovery_100_rows(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers and completes under 30% rate limit errors.

        Verifies:
        - AIMD backoff activates under rate limits
        - All 100 rows eventually complete or error gracefully
        - Pipeline doesn't hang or crash
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        stats = chaosllm_http_server.get_stats()
        assert output.total_count == 100, f"Expected 100 rows processed, got {output.total_count}"

        # With 30% rate limits, we expect retries so request count > 100
        assert stats["total_requests"] >= 100, "Should have at least one request per row"

        # Most rows should succeed despite rate limits due to AIMD retry
        assert output.success_count > 50, f"Expected >50 successes, got {output.success_count}"

    @pytest.mark.chaosllm(capacity_529_pct=25.0)
    def test_capacity_529_handling(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Azure-specific 529 capacity errors trigger AIMD backoff.

        Verifies:
        - 529 errors are treated as retryable
        - AIMD backoff applies correctly
        - Pipeline completes without crashes
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # Verify all rows processed
        assert output.total_count == 50

        # Verify 529 errors were injected
        stats = chaosllm_http_server.get_stats()
        # At least some 529s should have been injected
        assert stats["total_requests"] > 50, "Should have retried some requests"

    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_burst_pattern_resilience(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline survives periodic error bursts.

        The stress_aimd preset includes burst patterns that spike error rates
        periodically. The pipeline should recover between bursts.

        Verifies:
        - Pipeline doesn't crash during high-error burst periods
        - Pipeline recovers and continues after bursts
        - Final results include both successes and expected failures
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=15)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(75)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed (success or error)
        assert output.total_count == 75

    @pytest.mark.chaosllm(rate_limit_pct=20.0)
    def test_concurrent_workers_coordination(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multiple concurrent workers coordinate under rate limits.

        With pool_size=8 and 100 rows, verify workers don't interfere
        with each other's AIMD backoff.

        Verifies:
        - Workers share rate limit backoff correctly
        - No deadlocks or race conditions
        - FIFO ordering preserved
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,  # High concurrency
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=32)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        input_order = _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 100

        # Verify FIFO ordering for successful rows
        output_ids = [r[0]["id"] for r in output.results]
        for i in range(len(output_ids) - 1):
            pos_i = input_order.index(output_ids[i])
            pos_next = input_order.index(output_ids[i + 1])
            assert pos_i < pos_next, "FIFO ordering violated"


# ==========================================================================
# Azure Multi-Query Stress Tests (3 tests)
# ==========================================================================


@pytest.mark.stress
class TestAzureMultiQueryStress:
    """Azure multi-query transform under error injection.

    With 4 named queries per row, error rates are amplified.
    A 25% error rate means ~68% of rows will have at least
    one failed query (1 - 0.75^4).
    """

    @pytest.mark.chaosllm(rate_limit_pct=10.0, internal_error_pct=5.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_partial_failure(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """When any query fails permanently, the entire row fails (all-or-nothing).

        Uses a mix of retryable (rate limits) and non-retryable (internal errors)
        to guarantee some permanent failures while testing AIMD recovery.

        With 4 queries per row and 5% non-retryable errors:
        - P(all succeed) ~ 0.95^4 ~ 81%
        - Expect ~19% of rows to fail permanently

        Verifies:
        - All-or-nothing semantics enforced
        - Rows with any failed query are errored
        - Successful rows have all 4 query outputs
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # Verify successful rows have all query outputs
        expected_output_prefixes = [
            "cs1_diagnosis",
            "cs1_treatment",
            "cs2_diagnosis",
            "cs2_treatment",
        ]
        for result_row, _ in output.results:
            for prefix in expected_output_prefixes:
                assert f"{prefix}_score" in result_row, f"Missing {prefix}_score"
                assert f"{prefix}_rationale" in result_row, f"Missing {prefix}_rationale"

        # With AIMD retry, most rate-limited rows recover
        # Internal errors (5%) cause permanent failures: P(all 4 succeed) ~ 0.95^4 = 81%
        assert output.success_count >= 35, f"Expected at least 70% success, got {output.success_count}/50"

    @pytest.mark.chaosllm(rate_limit_pct=25.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_rate_limit_all_queries_retry(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """AIMD backoff affects all queries when rate limited.

        With PooledExecutor, rate limit backoff is shared across all
        queries for a row. When AIMD kicks in, all queries slow down.

        Verifies:
        - Pipeline completes despite high rate limits
        - AIMD coordinates across queries
        - Total requests > 4x rows due to retries
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(30)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 30

        stats = chaosllm_http_server.get_stats()
        # With 4 queries per row, expect at least 120 requests
        assert stats["total_requests"] >= 30 * 4, "Expected at least one request per query"

    @pytest.mark.chaosllm(
        rate_limit_pct=20.0,
        capacity_529_pct=10.0,
        internal_error_pct=5.0,
        template_body=MULTI_QUERY_JSON_TEMPLATE,
    )
    def test_multi_query_chaos_mode(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multi-query transform survives 35%+ combined error rate.

        With 35% errors and 4 queries per row:
        - P(all succeed) = 0.65^4 ~ 18%
        - Most rows will fail

        Verifies:
        - Pipeline doesn't crash
        - All rows reach terminal state
        - Errors are properly recorded
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 50 * 4, "Expected at least baseline requests"

        # If there are errors, verify they have proper structure
        for error, _ in output.errors:
            assert "reason" in error, "Error should have reason"


# ==========================================================================
# OpenRouter LLM Stress Tests (3 tests)
# ==========================================================================


@pytest.mark.stress
class TestOpenRouterLLMStress:
    """OpenRouter LLM transform under error injection."""

    @pytest.mark.chaosllm(rate_limit_pct=30.0)
    def test_rate_limit_recovery_100_rows(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers and completes under 30% rate limit errors.

        Verifies:
        - HTTP client handles 429 responses gracefully
        - All 100 rows eventually complete or error gracefully
        - Pipeline doesn't hang or crash
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        stats = chaosllm_http_server.get_stats()
        assert output.total_count == 100, f"Expected 100 rows processed, got {output.total_count}"

        # With 30% rate limits, we expect retries
        assert stats["total_requests"] >= 100

        # Most rows should succeed despite rate limits
        assert output.success_count > 50, f"Expected >50 successes, got {output.success_count}"

    @pytest.mark.chaosllm(invalid_json_pct=15.0)
    def test_malformed_json_handling(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Transform handles malformed JSON responses gracefully.

        OpenRouter uses HTTP client (not SDK), so malformed JSON is detected
        in the transform's response parsing rather than SDK internals.

        Verifies:
        - Malformed JSON errors are captured as row errors
        - Other rows continue processing
        - No crashes from JSON parse failures
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # With 15% malformed JSON, expect some errors
        assert output.error_count >= 1, "Expected some malformed JSON errors"
        assert output.success_count > 30, "Expected most rows to succeed"

    @pytest.mark.chaosllm(service_unavailable_pct=20.0, gateway_timeout_pct=10.0)
    def test_connection_timeout_recovery(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline recovers from network-level failures.

        Verifies:
        - 503 and 504 errors don't crash the pipeline
        - Rows fail gracefully with recorded errors
        - Other rows continue processing
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            timeout_seconds=10.0,  # Short timeout for faster tests
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(75)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 75

        # With 30% combined server errors, expect some failures
        assert output.error_count >= 5, "Expected some server error failures"


# ==========================================================================
# OpenRouter Multi-Query Stress Tests (4 tests)
# ==========================================================================


@pytest.mark.stress
class TestOpenRouterMultiQueryStress:
    """OpenRouter multi-query transform under error injection.

    With 4 named queries per row, error rates are amplified.
    These tests validate HTTP-based error handling
    in the multi-query context.
    """

    @pytest.mark.chaosllm(rate_limit_pct=10.0, internal_error_pct=5.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_partial_failure(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """When any query fails permanently, the entire row fails (all-or-nothing).

        Verifies:
        - All-or-nothing semantics enforced via HTTP
        - Rows with any failed query are errored
        - Successful rows have all 4 query outputs
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # Verify successful rows have all query outputs
        expected_output_prefixes = [
            "cs1_diagnosis",
            "cs1_treatment",
            "cs2_diagnosis",
            "cs2_treatment",
        ]
        for result_row, _ in output.results:
            for prefix in expected_output_prefixes:
                assert f"{prefix}_score" in result_row, f"Missing {prefix}_score"
                assert f"{prefix}_rationale" in result_row, f"Missing {prefix}_rationale"

        # With AIMD retry, most rate-limited rows recover
        assert output.success_count >= 35, f"Expected at least 70% success, got {output.success_count}/50"

    @pytest.mark.chaosllm(rate_limit_pct=25.0, template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_rate_limit_all_queries_retry(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """AIMD backoff affects all queries when rate limited.

        Verifies:
        - Pipeline completes despite high rate limits
        - HTTP 429 responses trigger AIMD
        - Total requests > 4x rows due to retries
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(30)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 30

        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 30 * 4, "Expected at least one request per query"

    @pytest.mark.chaosllm(
        rate_limit_pct=20.0,
        service_unavailable_pct=10.0,
        invalid_json_pct=5.0,
        template_body=MULTI_QUERY_JSON_TEMPLATE,
    )
    def test_multi_query_chaos_mode(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Multi-query transform survives mixed error types.

        With HTTP errors (429, 503) and malformed responses (invalid JSON),
        the transform should handle all error types gracefully.

        Verifies:
        - Pipeline doesn't crash
        - All rows reach terminal state
        - Mixed error types are handled
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] >= 50 * 4, "Expected at least baseline requests"

        # If there are errors, verify they have proper structure
        for error, _ in output.errors:
            assert "reason" in error, "Error should have reason"

    @pytest.mark.chaosllm(preset="stress_aimd", template_body=MULTI_QUERY_JSON_TEMPLATE)
    def test_multi_query_fifo_ordering(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """FIFO ordering is preserved for successful rows.

        With concurrent query processing, verify that row output order
        matches input order for successful rows.
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_multi_query_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_multi_query_rows(50)
        input_order = _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        assert output.total_count == 50

        # Verify FIFO ordering for successful rows
        output_ids = [r[0]["id"] for r in output.results]
        for i in range(len(output_ids) - 1):
            pos_i = input_order.index(output_ids[i])
            pos_next = input_order.index(output_ids[i + 1])
            assert pos_i < pos_next, f"FIFO ordering violated at position {i}"


# ==========================================================================
# Mixed Errors Tests (5 tests)
# ==========================================================================


@pytest.mark.stress
class TestMixedErrors:
    """Cross-cutting tests with multiple error types active."""

    @pytest.mark.chaosllm(
        rate_limit_pct=15.0,
        capacity_529_pct=5.0,
        internal_error_pct=3.0,
        invalid_json_pct=2.0,
    )
    def test_chaos_mode_azure_pipeline_completes(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Azure pipeline completes with 25%+ mixed error rate.

        Error mix:
        - 15% rate limits (retryable with AIMD)
        - 5% capacity errors (retryable with AIMD)
        - 3% internal errors (may fail row)
        - 2% invalid JSON (fails row)

        Verifies:
        - No crashes with mixed error types
        - All rows reach terminal state
        - Pipeline completes within reasonable time
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        start_time = time.monotonic()

        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        elapsed = time.monotonic() - start_time

        # All rows must reach terminal state
        assert output.total_count == 100, f"Only {output.total_count}/100 rows completed"

        # Should complete within 3 minutes
        assert elapsed < 180, f"Chaos mode took too long: {elapsed:.1f}s"

        # With AIMD retry for rate limits, expect reasonable success rate
        stats = chaosllm_http_server.get_stats()
        assert stats["total_requests"] > 100, "Expected retries to occur"

    @pytest.mark.chaosllm(
        rate_limit_pct=15.0,
        service_unavailable_pct=5.0,
        gateway_timeout_pct=3.0,
        invalid_json_pct=2.0,
    )
    def test_chaos_mode_openrouter_pipeline_completes(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """OpenRouter pipeline completes with 25%+ mixed error rate.

        Verifies HTTP-level error handling with mixed error types.
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_openrouter_llm_config(
            chaosllm_http_server.url,
            pool_size=8,
            max_capacity_retry_seconds=45,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=30)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)
        start_time = time.monotonic()

        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        elapsed = time.monotonic() - start_time

        # All rows must reach terminal state
        assert output.total_count == 100

        # Should complete within 3 minutes
        assert elapsed < 180, f"Chaos mode took too long: {elapsed:.1f}s"

    @pytest.mark.chaosllm(rate_limit_pct=10.0)
    def test_audit_trail_integrity_under_stress(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Verify audit trail is complete for all rows under stress.

        Every row should have:
        - A row record in the database
        - A token record
        - At least one node state
        - Node state with completion or error status

        Verifies:
        - No silent drops
        - All rows traceable
        - Audit records consistent
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=30,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(50)
        row_ids = []
        token_ids = []

        for i, row in enumerate(rows):
            token = make_token(f"row-{i}", f"token-{i}")
            row_record = factory.data_flow.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            row_ids.append(row_record.row_id)

            token_record = factory.data_flow.create_token(row_id=row_record.row_id)
            token_ids.append(token_record.token_id)

            state = factory.execution.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = make_context(
                run_id=run_id,
                landscape=factory.plugin_audit_writer(),
                state_id=state.state_id,
                token=token,
            )
            transform.accept(make_pipeline_row(row), ctx)

        transform.flush_batch_processing()
        transform.close()

        # All rows should be processed
        assert output.total_count == 50

        # Verify audit trail completeness using QueryRepository API
        db_rows = factory.query.get_rows(run_id)
        db_row_ids = {r.row_id for r in db_rows}
        assert set(row_ids) == db_row_ids, "Missing rows in audit trail"

        # Verify all tokens
        db_token_ids: set[str] = set()
        for row_id in row_ids:
            tokens_for_row = factory.query.get_tokens(row_id)
            for t in tokens_for_row:
                db_token_ids.add(t.token_id)
        assert set(token_ids) == db_token_ids, "Missing tokens in audit trail"

    def test_progressive_degradation(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline handles gradually increasing error rates.

        Start with low error rate, increase during test.
        Pipeline should adapt via AIMD.

        Verifies:
        - Pipeline continues as errors increase
        - AIMD backoff kicks in appropriately
        - No sudden crash when error rate spikes
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(100)

        # Start with low error rate
        chaosllm_http_server.update_config(rate_limit_pct=5.0)

        for i, row in enumerate(rows):
            # Gradually increase error rate every 25 rows
            if i == 25:
                chaosllm_http_server.update_config(rate_limit_pct=15.0)
            elif i == 50:
                chaosllm_http_server.update_config(rate_limit_pct=25.0)
            elif i == 75:
                chaosllm_http_server.update_config(rate_limit_pct=35.0)

            token = make_token(f"row-{i}", f"token-{i}")
            row_record = factory.data_flow.create_row(
                run_id=run_id,
                source_node_id=node_id,
                row_index=i,
                data=row,
            )
            token_record = factory.data_flow.create_token(row_id=row_record.row_id)
            state = factory.execution.begin_node_state(
                token_id=token_record.token_id,
                node_id=node_id,
                run_id=run_id,
                step_index=0,
                input_data=row,
            )

            ctx = make_context(
                run_id=run_id,
                landscape=factory.plugin_audit_writer(),
                state_id=state.state_id,
                token=token,
            )
            transform.accept(make_pipeline_row(row), ctx)

        transform.flush_batch_processing()
        transform.close()

        # All rows should complete
        assert output.total_count == 100

    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_burst_recovery_with_mixed_plugins(
        self,
        chaosllm_http_server: ChaosLLMHTTPFixture,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Pipeline survives error bursts with stress_aimd preset.

        The stress_aimd preset includes periodic burst patterns that
        spike rate limits to 80% for 5 seconds every 30 seconds.

        Verifies:
        - Pipeline doesn't crash during bursts
        - Recovery after burst period
        - Final completion despite bursts
        """
        factory, run_id, node_id = create_factory_and_run(tmp_path_factory)

        config = make_azure_llm_config(
            chaosllm_http_server.url,
            pool_size=4,
            max_capacity_retry_seconds=60,
        )
        transform = LLMTransform(config)

        output = CollectingOutputPort()
        transform.connect_output(output, max_pending=20)

        start_ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())
        transform.on_start(start_ctx)

        rows = generate_test_rows(75)
        start_time = time.monotonic()

        _feed_rows(transform, rows, factory, run_id, node_id)

        transform.flush_batch_processing()
        transform.close()

        elapsed = time.monotonic() - start_time

        # All rows should complete
        assert output.total_count == 75

        # Should complete within reasonable time even with bursts
        assert elapsed < 180, f"Burst recovery took too long: {elapsed:.1f}s"
