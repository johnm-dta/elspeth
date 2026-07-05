"""Tests for LLM transform retry behavior and concurrent processing.

Tests the unified LLMTransform with MultiQueryStrategy using Azure provider.
Updated from AzureMultiQueryLLMTransform to unified LLMTransform.
Uses row-level pipelining API (BatchTransformMixin).
"""

from __future__ import annotations

import itertools
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from elspeth.contracts import Call, CallStatus, CallType, TransformResult
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


class _CallRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _ExecutionRepositoryDouble:
    def __init__(self) -> None:
        self._call_counter = itertools.count()
        self._operation_call_counter = itertools.count()
        self.recorded_calls: list[dict[str, Any]] = []

    def allocate_call_index(self, state_id: str) -> int:
        return next(self._call_counter)

    def allocate_operation_call_index(self, operation_id: str) -> int:
        return next(self._operation_call_counter)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        call_kwargs = {
            "state_id": state_id,
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        actual_call_index = call_index if call_index is not None else self.allocate_operation_call_index(operation_id)
        call_kwargs = {
            "operation_id": operation_id,
            "call_index": actual_call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def _recorded_call(self, call_kwargs: dict[str, Any]) -> Call:
        return Call(
            call_id=f"call_{len(self.recorded_calls)}",
            call_index=call_kwargs["call_index"],
            call_type=call_kwargs["call_type"],
            status=call_kwargs["status"],
            request_hash="req_hash_123",
            response_hash="resp_hash_456" if call_kwargs["response_data"] is not None else None,
            created_at=datetime.now(UTC),
            state_id=call_kwargs.get("state_id"),
            operation_id=call_kwargs.get("operation_id"),
            latency_ms=call_kwargs["latency_ms"],
            resolved_prompt_template_hash=call_kwargs["resolved_prompt_template_hash"],
        )


def _openai_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://test.openai.azure.com/openai/deployments/gpt-4o/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "https://test.openai.azure.com/openai/deployments/gpt-4o/chat/completions")


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
        "prompt_template": "Evaluate: {{ row.text_content }}",
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
) -> Generator[tuple[Any, list[int]], None, None]:
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
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_capacity_error_triggers_retry(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Transient rate-limit on first two calls then success -> row succeeds.

        B3.7 bounded local retry: when the first few calls fail with a retryable
        error then succeed, the bounded local retry in _execute_sequential absorbs
        the transient failures and the row completes successfully.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        # First 2 calls fail with rate-limit; calls 3+ succeed.
        # The bounded local retry absorbs these and the row succeeds.
        def failure_condition(count: int) -> OpenAIRateLimitError | None:
            if count <= 2:
                return OpenAIRateLimitError(
                    message="Rate limit exceeded",
                    response=_openai_response(429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success after retry"},
            failure_condition,
        ) as (_mock_client, call_count):
            # Large budget so retries complete quickly
            transform = LLMTransform(_make_config(max_capacity_retry_seconds=30))
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

            # Bounded local retry absorbs the transient failures - row succeeds
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success", (
                f"Expected success after bounded retry, got status={result.status!r}, reason={result.reason!r}"
            )
            # More calls than 4 queries proves retries happened
            assert call_count[0] > 4

    def test_capacity_retry_timeout(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Row diverts with reason='retry_timeout' when budget exhausted.

        B3.7 bounded local retry: when the provider never recovers, the bounded
        retry times out and diverts the row as TransformResult.error with
        reason='retry_timeout'. The result is non-retryable (terminal divert).
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def always_fail(count: int) -> OpenAIRateLimitError:
            return OpenAIRateLimitError(
                message="Rate limit exceeded - never succeeds",
                response=_openai_response(429),
                body=None,
            )

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Never returned"},
            always_fail,
        ):
            # Small budget so the test completes quickly
            transform = LLMTransform(_make_config(max_capacity_retry_seconds=1))
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

            # Bounded retry exhausted - TransformResult.error with reason='retry_timeout'
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "retry_timeout"
            assert result.retryable is False

    def test_mixed_success_and_failure(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Single transient failure on the 3rd call is absorbed by bounded local retry.

        B3.7 bounded local retry: when the 3rd query call fails once then succeeds,
        the bounded local retry absorbs it. Earlier queries are not re-executed.
        The row completes successfully with all 4 query outputs.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def intermittent_failure(count: int) -> OpenAIRateLimitError | None:
            # Third call fails ONCE (affects first row's 3rd query on first attempt)
            if count == 3:
                return OpenAIRateLimitError(
                    message="Rate limit",
                    response=_openai_response(429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success"},
            intermittent_failure,
        ) as (_mock_client, call_count):
            # Large budget so retry completes quickly
            transform = LLMTransform(_make_config(max_capacity_retry_seconds=30))
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

            # Bounded local retry absorbs the single transient failure - row succeeds
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success", f"Expected success after single transient retry, got {result.status!r}: {result.reason!r}"
            # 5 calls: 4 queries + 1 retry for the 3rd query
            assert call_count[0] == 5, f"Expected 5 calls (4 queries + 1 retry), got {call_count[0]}"


class TestConcurrentRowProcessing:
    """Tests for concurrent row processing.

    Row-level pipelining is handled by BatchTransformMixin.
    Query-level execution is sequential in MultiQueryStrategy (no PooledExecutor).
    """

    @pytest.fixture
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_multiple_rows_processed_via_pipelining(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
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
        mock_recorder: _ExecutionRepositoryDouble,
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
                    response=_openai_response(429),
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
        mock_recorder: _ExecutionRepositoryDouble,
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
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_sequential_mode_retryable_error_absorbed_by_bounded_retry(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Single transient error on first call is absorbed by bounded local retry.

        B3.7 bounded local retry: a single rate-limit error on the first call
        is retried locally. On the next attempt the call succeeds, and all 4
        queries complete - the row succeeds rather than being diverted.
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def first_call_fails(count: int) -> OpenAIRateLimitError | None:
            if count == 1:
                return OpenAIRateLimitError(
                    message="Rate limit",
                    response=_openai_response(429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success"},
            first_call_fails,
        ) as (_mock_client, call_count):
            # Large budget so the single retry completes immediately
            config = _make_config(max_capacity_retry_seconds=30)
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

            # Bounded local retry absorbed the single transient error - row succeeds
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success", f"Expected success after bounded retry, got {result.status!r}: {result.reason!r}"
            # 5 calls: 4 queries + 1 retry for the first query
            assert call_count[0] == 5, f"Expected 5 calls (4 queries + 1 retry), got {call_count[0]}"


class TestProviderClientLifecycle:
    """Test that provider clients are properly managed."""

    @pytest.fixture
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_provider_close_called_on_transform_close(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
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
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_network_error_absorbed_by_bounded_retry(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """NetworkError (retryable) on first two calls absorbed by bounded local retry.

        B3.7 bounded local retry: transient network timeouts on the first two calls
        are retried locally. Once the provider recovers, the row completes successfully.
        The error is NOT propagated as ExceptionResult.
        """
        from openai import APITimeoutError

        def first_two_fail(count: int) -> APITimeoutError | None:
            if count <= 2:
                return APITimeoutError(request=_openai_request())
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Good"},
            first_two_fail,
        ) as (_mock_client, call_count):
            # Large budget so retries complete quickly
            config = _make_config(max_capacity_retry_seconds=30)
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

            # Bounded local retry absorbed the transient network errors - row succeeds
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success", f"Expected success after bounded retry, got {result.status!r}: {result.reason!r}"
            # More than 4 calls proves retries happened
            assert call_count[0] > 4

    def test_content_policy_error_not_retried(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
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
                response=_openai_response(400),
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


class TestSequentialBoundedLocalRetry:
    """Tests for B3.7 bounded local retry in sequential multi-query mode.

    A transient retryable error (429/5xx/network) on a sequential multi-query
    row must be retried locally within a bounded budget rather than immediately
    diverting the row. Earlier queries that already succeeded must NOT be
    re-executed.
    """

    @pytest.fixture
    def mock_recorder(self) -> _ExecutionRepositoryDouble:
        """Create ExecutionRepository double."""
        return _ExecutionRepositoryDouble()

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        """Create output collector for capturing results."""
        return CollectorOutputPort()

    def test_sequential_transient_then_success_does_not_divert(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Transient error on first attempt then success -> row succeeds, not diverted.

        B3.7 fix: a bounded local retry must be applied for retryable LLMClientErrors
        in sequential mode. A single transient 429/RateLimitError that then succeeds
        must produce a successful row result, NOT an error result or ExceptionResult.

        AUDITED-OUTCOME assertions:
          (a) result is TransformResult.success (status=='success')
          (b) success_reason['action']=='multi_query_enriched' and
              queries_completed==len(query_specs) (4 queries)
          (c) the provider was called >4 times (proving retry happened) and
              earlier-succeeded queries are NOT re-executed (call_count > 4 but
              the row output has all 4 query fields)
        """
        from openai import RateLimitError as OpenAIRateLimitError

        # Per-query call tracking: the 2nd query (call index 2) fails once then succeeds.
        # Call index 1 -> query cs1_diagnosis succeeds
        # Call index 2 -> query cs1_treatment FAILS (rate limit)
        # Call index 3 -> retry of cs1_treatment succeeds (bounded local retry)
        # Call index 4 -> query cs2_diagnosis succeeds
        # Call index 5 -> query cs2_treatment succeeds
        # Total: 5 calls (4 queries + 1 retry)
        def second_call_fails_once(count: int) -> OpenAIRateLimitError | None:
            if count == 2:
                return OpenAIRateLimitError(
                    message="Rate limit exceeded",
                    response=_openai_response(429),
                    body=None,
                )
            return None

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Success after retry"},
            second_call_fails_once,
        ) as (_mock_client, call_count):
            # Use a large retry budget so the retry completes quickly in tests
            config = _make_config(max_capacity_retry_seconds=30)
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-transient-retry-1")
                ctx = make_context(state_id="state-transient-retry-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # (a) Row SUCCEEDS - not diverted
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), f"Expected TransformResult, got {type(result).__name__}: {result}"
            assert result.status == "success", f"Expected status='success', got status={result.status!r}, reason={result.reason!r}"

            # (b) All queries completed - success_reason reflects full enrichment
            assert result.success_reason is not None, "success_reason must be set"
            assert result.success_reason["action"] == "multi_query_enriched", f"Unexpected action: {result.success_reason['action']!r}"
            assert result.success_reason["queries_completed"] == 4, (
                f"Expected queries_completed=4, got {result.success_reason['queries_completed']}"
            )

            # (c) Provider was called >4 times (retry happened); row has all 4 output fields
            assert call_count[0] > 4, f"Expected >4 provider calls (retry proof), got {call_count[0]}"
            assert result.row is not None
            row_dict = dict(result.row)
            assert "cs1_diagnosis_score" in row_dict, "cs1_diagnosis must have been produced"
            assert "cs1_treatment_score" in row_dict, "cs1_treatment must have been produced (after retry)"
            assert "cs2_diagnosis_score" in row_dict, "cs2_diagnosis must have been produced"
            assert "cs2_treatment_score" in row_dict, "cs2_treatment must have been produced"

    def test_sequential_bounded_retry_timeout_diverts_with_audited_result(
        self,
        mock_recorder: _ExecutionRepositoryDouble,
        collector: CollectorOutputPort,
        chaosllm_server,
    ) -> None:
        """Always-failing retryable error + small budget -> bounded divert with audit record.

        B3.7 fix: when retry exhausts the max_capacity_retry_seconds budget,
        the row must divert to TransformResult.error with reason='retry_timeout',
        mirroring the Azure _capacity_retry_timeout_result shape. The result must
        NOT be an ExceptionResult (engine must not be asked to retry a timed-out row).

        AUDITED-OUTCOME assertions:
          (a) result is TransformResult.error (status=='error'), NOT ExceptionResult
          (b) result.reason['reason']=='retry_timeout'
          (c) result.reason contains 'elapsed_seconds' and 'max_seconds'
          (d) result.retryable is False (bounded timeout is terminal, not retriable)
        """
        from openai import RateLimitError as OpenAIRateLimitError

        def always_fail(count: int) -> OpenAIRateLimitError:
            return OpenAIRateLimitError(
                message="Rate limit exceeded - never recovers",
                response=_openai_response(429),
                body=None,
            )

        with mock_azure_openai_with_counter(
            chaosllm_server,
            {"score": 85, "rationale": "Never returned"},
            always_fail,
        ) as (_mock_client, call_count):
            # Very small budget - will exhaust almost immediately
            config = _make_config(max_capacity_retry_seconds=1)
            transform = LLMTransform(config)
            init_ctx = make_context(landscape=mock_recorder)
            transform.on_start(init_ctx)
            transform.connect_output(collector, max_pending=10)

            try:
                row = {
                    "cs1_bg": "data",
                    "cs2_bg": "data",
                }
                token = make_token("row-timeout-budget-1")
                ctx = make_context(state_id="state-timeout-budget-1", token=token)

                transform.accept(make_pipeline_row(row), ctx)
                transform.flush_batch_processing(timeout=10.0)
            finally:
                transform.close()

            # (a) Result is a diverted TransformResult.error, not ExceptionResult
            assert len(collector.results) == 1
            _, result, _state_id = collector.results[0]
            assert isinstance(result, TransformResult), (
                f"Expected TransformResult.error (bounded divert), got {type(result).__name__}: {result}"
            )
            assert result.status == "error", f"Expected status='error', got {result.status!r}"

            # (b) Reason is retry_timeout - proves bounded retry exhausted
            assert result.reason is not None
            assert result.reason["reason"] == "retry_timeout", f"Expected reason='retry_timeout', got {result.reason['reason']!r}"

            # (c) Audit fields present
            assert "elapsed_seconds" in result.reason, f"Missing elapsed_seconds in reason: {result.reason}"
            assert "max_seconds" in result.reason, f"Missing max_seconds in reason: {result.reason}"

            # (d) Non-retryable (terminal divert - engine must not retry)
            assert result.retryable is False, f"retry_timeout result must not be retryable, got retryable={result.retryable!r}"

            # Proof that retry happened (called more than once per query)
            assert call_count[0] > 1, f"Expected >1 call (retry proof), got {call_count[0]}"
