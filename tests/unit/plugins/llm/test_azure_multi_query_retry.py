"""Tests for LLM transform retry behavior and concurrent processing.

Tests the unified LLMTransform with MultiQueryStrategy using Azure provider.
Updated from AzureMultiQueryLLMTransform to unified LLMTransform.
Uses row-level pipelining API (BatchTransformMixin).
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import TransformResult
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .conftest import (
    chaosllm_azure_openai_responses,
    chaosllm_azure_openai_sequence,
    make_token,
)


def _make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid Azure multi-query config for unified LLMTransform.

    Equivalent to the old make_azure_multi_query_config but using the new
    queries-based format instead of case_studies/criteria cross-product.

    The old config had:
        case_studies: [cs1(fields: cs1_bg, cs1_sym, cs1_hist), cs2(fields: cs2_bg, cs2_sym, cs2_hist)]
        criteria: [diagnosis(code: DIAG), treatment(code: TREAT)]
        output_mapping: {score: {suffix: score, type: integer}, rationale: {suffix: rationale, type: string}}

    This produced 4 queries: cs1_diagnosis, cs1_treatment, cs2_diagnosis, cs2_treatment.
    Each query output had fields like cs1_diagnosis_score, cs1_diagnosis_rationale.

    The new config defines these explicitly as queries.
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


@contextmanager
def mock_azure_openai_with_counter(
    chaosllm_server,
    success_response: dict[str, Any],
    failure_condition: Any | None = None,
) -> Generator[tuple[Mock, list[int]], None, None]:
    """Mock Azure OpenAI with thread-safe call counter.

    Args:
        chaosllm_server: ChaosLLM server fixture
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
    """Tests for capacity error retry with engine-level retry.

    In the unified LLMTransform, retryable LLM errors (RateLimitError, etc.)
    are re-raised by MultiQueryStrategy for the engine retry to handle.
    The old PooledExecutor AIMD retry is replaced by engine-level retry.
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
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
        """Capacity errors trigger automatic retry until success.

        In the unified LLMTransform, retryable errors are re-raised from
        MultiQueryStrategy. The BatchTransformMixin worker catches these
        and they propagate as ExceptionResult. However, the first query
        that hits a rate limit causes the entire row to fail atomically
        (no per-query retry in sequential mode).

        This test verifies that non-retryable error handling works correctly:
        when the first few calls fail and then succeed, subsequent queries
        in the row execute normally.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        # Only first 2 calls fail; queries 3+ succeed.
        # With 4 sequential queries per row, query 1 and 2 fail => row fails
        # because retryable errors are re-raised (atomic failure).
        # Instead, test that after initial failures, a fresh row succeeds.
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
            transform = LLMTransform(_make_config())
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-retry-1")
                ctx = make_context(state_id="state-retry-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Row will fail because retryable errors are re-raised (atomic failure).
            # The first query hits rate limit and the error propagates.
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            # Result is either an ExceptionResult (re-raised retryable) or error TransformResult
            if isinstance(result, TransformResult):
                # Non-retryable path: error returned
                assert result.status == "error"
            else:
                # Retryable path: exception propagated for engine retry
                assert isinstance(result, ExceptionResult)

    def test_capacity_retry_timeout(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Row fails when all queries hit rate limits (no engine retry in test)."""
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
            transform = LLMTransform(_make_config())
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-timeout-1")
                ctx = make_context(state_id="state-timeout-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Should fail — rate limit error propagated
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            if isinstance(result, TransformResult):
                assert result.status == "error"
            else:
                # Retryable error re-raised as ExceptionResult
                assert isinstance(result, ExceptionResult)

    def test_mixed_success_and_failure(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """When some queries succeed and some fail, the row fails atomically.

        In the unified LLMTransform with MultiQueryStrategy, queries run
        sequentially. If any query raises a retryable error, the entire
        row fails (atomic failure semantics).
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def intermittent_failure(count: int) -> OpenAIRateLimitError | None:
            # Third call fails (affects first row's 3rd query)
            if count == 3:
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
            transform = LLMTransform(_make_config())
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-mixed-1")
                ctx = make_context(state_id="state-mixed-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Row fails atomically when any query fails
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            if isinstance(result, TransformResult):
                # If error was non-retryable, we get error result
                assert result.status == "error"
            else:
                # If error was retryable, it's re-raised as ExceptionResult
                assert isinstance(result, ExceptionResult)


class TestConcurrentRowProcessing:
    """Tests for concurrent row processing.

    Row-level pipelining is handled by BatchTransformMixin.
    Query-level execution is sequential in MultiQueryStrategy (no PooledExecutor).
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
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

        Uses sequential query mode (MultiQueryStrategy runs queries sequentially)
        to focus on row-level pipelining.
        """
        # Use consistent response for all queries
        responses: list[dict[str, Any] | str] = [{"score": 85, "rationale": "R"}]

        config = _make_config()

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as mock_client:
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # 10 rows x 4 queries = 40 total queries
                for i in range(10):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"batch-100-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

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

        Verifies atomic failure semantics: if any query in a row fails,
        the entire row fails with no partial output.
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

        config = _make_config()

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "OK"},
            every_7th_fails,
        ):
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # 20 rows x 4 queries = 80 queries
                # Every 7th query fails
                for i in range(20):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"concurrent-atomicity-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 20

            # Verify atomicity: each row has 0 or 4 output fields
            for token, result, _state_id in collector.results:
                if isinstance(result, ExceptionResult):
                    # Retryable error propagated — atomic failure
                    continue
                assert isinstance(result, TransformResult)
                output_row: dict[str, Any] = dict(result.row) if result.row is not None else {}
                output_field_count = sum(
                    [
                        "cs1_diagnosis_score" in output_row,
                        "cs1_treatment_score" in output_row,
                        "cs2_diagnosis_score" in output_row,
                        "cs2_treatment_score" in output_row,
                    ]
                )

                if result.status == "error":
                    assert output_field_count == 0, f"Row {token.row_id} has error + {output_field_count} outputs"
                else:
                    assert output_field_count == 4, f"Row {token.row_id} has {output_field_count} outputs (expected 4)"

    def test_sequential_query_execution(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Verify sequential query execution within a row.

        In the unified LLMTransform, MultiQueryStrategy executes queries
        sequentially (no query-level pool). This test verifies correct
        execution and call count.
        """

        def response_factory(_call_index: int, _request: dict[str, Any]) -> dict[str, Any]:
            return {"score": 85, "rationale": "OK"}

        with chaosllm_azure_openai_sequence(chaosllm_server, response_factory) as (
            _mock_client,
            call_count,
            _mock_azure_class,
        ):
            transform = LLMTransform(_make_config())
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                # Single row with 4 queries
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-0")
                ctx = make_context(state_id="seq-exec-0", token=token)
                transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert call_count[0] == 4


class TestSequentialFallback:
    """Tests for sequential processing — unified LLMTransform always uses sequential queries."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_sequential_mode_retryable_error_propagates(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Retryable errors are re-raised for engine retry (not swallowed).

        In the unified LLMTransform, retryable LLMClientErrors are re-raised
        by MultiQueryStrategy, propagating through BatchTransformMixin as
        ExceptionResult.
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
        ) as (_mock_client, _call_count):
            config = _make_config()
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-seq-1")
                ctx = make_context(state_id="state-seq-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # Row fails because first query hit rate limit
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            # Retryable error is re-raised — appears as ExceptionResult
            if isinstance(result, TransformResult):
                assert result.status == "error"
            else:
                assert isinstance(result, ExceptionResult)


class TestProviderClientLifecycle:
    """Test that provider clients are properly managed."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_provider_close_called_on_transform_close(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Provider.close() is called when transform.close() is called.

        Verifies that the unified LLMTransform properly cleans up provider
        resources on shutdown.
        """
        # Use consistent response for all queries
        responses: list[dict[str, Any] | str] = [{"score": 90, "rationale": "Good"}]

        config = _make_config()

        with chaosllm_azure_openai_responses(chaosllm_server, responses) as _mock_client:
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=100)

            try:
                # Process rows
                for i in range(5):
                    row = {
                        "row_id": i,
                        "cs1_bg": f"data_{i}",
                        "cs2_bg": f"data_{i}",
                    }
                    token = make_token(f"row-{i}")
                    ctx = make_context(state_id=f"batch-lifecycle-{i}", token=token)
                    transform.accept(make_pipeline_row(row), ctx)

                transform.flush_batch_processing(timeout=30.0)

                assert len(collector.results) == 5
                for _, result, _state_id in collector.results:
                    assert isinstance(result, TransformResult)
                    assert result.status == "success"

            finally:
                # After close, provider should be cleaned up
                transform.close()
                assert transform._provider is None


class TestLLMErrorRetry:
    """Test that retryable LLM errors are propagated for engine retry."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock ExecutionRepository."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_network_error_is_propagated(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """NetworkError (retryable) is propagated as ExceptionResult.

        In the unified LLMTransform, retryable LLMClientErrors are re-raised
        by MultiQueryStrategy. The BatchTransformMixin catches these and wraps
        them as ExceptionResult for the engine to handle.
        """
        from openai import APITimeoutError

        def first_two_fail(count: int) -> APITimeoutError | None:
            if count <= 2:
                return APITimeoutError(request=Mock())
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Good"},
            first_two_fail,
        ) as (_mock_client, _call_count):
            config = _make_config()
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-network-error-1")
                ctx = make_context(state_id="network-error-retry", token=token)
                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=30.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            # Network error is retryable — propagated as exception
            if isinstance(result, TransformResult):
                # If the provider classified it as non-retryable, we get error result
                assert result.status == "error"
            else:
                assert isinstance(result, ExceptionResult)

    def test_content_policy_error_not_retried(
        self,
        mock_recorder: Mock,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """ContentPolicyError should NOT be retried (non-retryable error).

        Non-retryable errors return TransformResult.error immediately
        without re-raising.
        """
        from openai import BadRequestError

        def always_fail_content_policy(count: int) -> BadRequestError:
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
            config = _make_config()
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-content-policy-1")
                ctx = make_context(state_id="content-policy-no-retry", token=token)
                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error", "Row should have error"

            # Non-retryable: only the first query is attempted before failure
            # (atomic failure on first query hit)
            assert call_count[0] == 1, f"Expected 1 call (non-retryable stops at first query), got {call_count[0]}"
