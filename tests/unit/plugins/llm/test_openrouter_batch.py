# tests/plugins/llm/test_openrouter_batch.py
"""Tests for OpenRouter batch LLM transform.

Tests cover:
- Configuration validation
- Single row processing (fallback mode)
- Batch processing with parallel execution
- Error handling (template, HTTP, API errors)
- AuditedHTTPClient caching (one per state_id)
- Audit trail recording via AuditedHTTPClient
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.llm.openrouter_batch import (
    OpenRouterBatchConfig,
    OpenRouterBatchLLMTransform,
)
from elspeth.testing import make_pipeline_row

from .conftest import chaosllm_openrouter_http_responses, chaosllm_openrouter_httpx_response

# Common schema config for dynamic field handling
DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_valid_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a valid config dict with optional overrides."""
    config = {
        "api_key": "sk-or-test-key",
        "model": "openai/gpt-4o-mini",
        "template": "Analyze: {{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out
    }
    if overrides:
        config.update(overrides)
    return config


def _create_mock_response(
    chaosllm_server,
    content: str = "Analysis result",
    model: str = "openai/gpt-4o-mini",
    usage: dict[str, int] | None = None,
    status_code: int = 200,
    raw_body: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Create an httpx.Response using ChaosLLM response generation."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "test prompt"}],
        "temperature": 0.0,
    }
    return chaosllm_openrouter_httpx_response(
        chaosllm_server,
        request,
        status_code=status_code,
        headers=headers or {"content-type": "application/json"},
        template_override=content,
        raw_body=raw_body,
        usage_override=usage,
    )


def _create_mock_context(
    run_id: str = "run-test",
    state_id: str = "state-test",
) -> Mock:
    """Create a mock PluginContext for testing.

    Sets up both ctx-level and recorder-level mocks so that:
    - ctx.record_call works (for template errors, which bypass AuditedHTTPClient)
    - ctx.landscape.record_call works (for AuditedHTTPClient recording)
    - ctx.landscape.allocate_call_index works (for AuditedHTTPClient call indexing)
    """
    ctx = Mock(spec=PluginContext)
    ctx.run_id = run_id
    ctx.state_id = state_id

    # Recorder mock with call index allocation
    recorder = Mock()
    recorder.allocate_call_index = Mock(side_effect=lambda _state_id: 0)
    recorder.record_call = Mock()
    ctx.landscape = recorder

    # ctx.record_call still needed for pre-HTTP errors (template rendering)
    ctx.record_call = Mock()

    # Telemetry emit (no-op for tests)
    ctx.telemetry_emit = Mock()

    # Rate limit registry (disabled for tests)
    ctx.rate_limit_registry = None

    # Batch token identity (None = single-row mode, set by AggregationExecutor for batches)
    ctx.batch_token_ids = None

    return ctx


def _create_transform_with_context(
    config_overrides: dict[str, Any] | None = None,
    ctx: Mock | None = None,
) -> tuple[OpenRouterBatchLLMTransform, Mock]:
    """Create a transform and call on_start() to initialize AuditedHTTPClient support.

    Returns:
        Tuple of (transform, ctx) ready for process() calls.
    """
    transform = OpenRouterBatchLLMTransform(_make_valid_config(config_overrides))
    if ctx is None:
        ctx = _create_mock_context()
    transform.on_start(ctx)
    return transform, ctx


@contextmanager
def mock_httpx_client(
    chaosllm_server,
    responses: list[httpx.Response] | httpx.Response | None = None,
    side_effect: Exception | None = None,
) -> Iterator[Mock]:
    """Context manager to mock httpx.Client using ChaosLLM responses."""
    if responses is None and side_effect is None:
        responses = _create_mock_response(chaosllm_server)
    # Build list of responses, ensuring no None values
    if responses is None:
        # side_effect is provided, responses list not used
        normalized: list[httpx.Response] = []
    elif isinstance(responses, list):
        normalized = responses
    else:
        normalized = [responses]
    # Cast to the expected type for chaosllm_openrouter_http_responses
    typed_responses = cast(list[dict[str, Any] | str | httpx.Response], normalized)
    with chaosllm_openrouter_http_responses(
        chaosllm_server,
        typed_responses,
        side_effect=side_effect,
    ) as mock_client:
        yield mock_client


class TestOpenRouterBatchConfig:
    """Tests for OpenRouterBatchConfig validation."""

    def test_config_requires_api_key(self) -> None:
        """OpenRouterBatchConfig requires API key."""
        with pytest.raises(PluginConfigError):
            OpenRouterBatchConfig.from_dict(
                {
                    "model": "openai/gpt-4o-mini",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],
                }
            )

    def test_config_requires_model(self) -> None:
        """OpenRouterBatchConfig requires model."""
        with pytest.raises(PluginConfigError):
            OpenRouterBatchConfig.from_dict(
                {
                    "api_key": "sk-test",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],
                }
            )

    def test_config_requires_template(self) -> None:
        """OpenRouterBatchConfig requires template."""
        with pytest.raises(PluginConfigError):
            OpenRouterBatchConfig.from_dict(
                {
                    "api_key": "sk-test",
                    "model": "openai/gpt-4o-mini",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],
                }
            )

    def test_config_requires_schema(self) -> None:
        """OpenRouterBatchConfig requires schema."""
        with pytest.raises(PluginConfigError, match="schema"):
            OpenRouterBatchConfig.from_dict(
                {
                    "api_key": "sk-test",
                    "model": "openai/gpt-4o-mini",
                    "template": "{{ row.text }}",
                }
            )

    def test_valid_minimal_config(self) -> None:
        """Valid minimal config passes validation."""
        config = OpenRouterBatchConfig.from_dict(_make_valid_config())

        assert config.api_key == "sk-or-test-key"
        assert config.model == "openai/gpt-4o-mini"
        assert config.template == "Analyze: {{ row.text }}"

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = OpenRouterBatchConfig.from_dict(_make_valid_config())

        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.timeout_seconds == 60.0
        assert config.pool_size == 1
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.response_field == "llm_response"

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = OpenRouterBatchConfig.from_dict(
            _make_valid_config(
                {
                    "base_url": "https://custom.api.com",
                    "timeout_seconds": 120.0,
                    "pool_size": 10,
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "response_field": "analysis",
                    "system_prompt": "You are helpful.",
                }
            )
        )

        assert config.base_url == "https://custom.api.com"
        assert config.timeout_seconds == 120.0
        assert config.pool_size == 10
        assert config.temperature == 0.7
        assert config.max_tokens == 500
        assert config.response_field == "analysis"
        assert config.system_prompt == "You are helpful."

    def test_timeout_must_be_positive(self) -> None:
        """Timeout must be greater than 0."""
        with pytest.raises(PluginConfigError):
            OpenRouterBatchConfig.from_dict(_make_valid_config({"timeout_seconds": 0}))

        with pytest.raises(PluginConfigError):
            OpenRouterBatchConfig.from_dict(_make_valid_config({"timeout_seconds": -1}))


class TestOpenRouterBatchLLMTransformInit:
    """Tests for OpenRouterBatchLLMTransform initialization."""

    def test_transform_name(self) -> None:
        """Transform has correct name."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        assert transform.name == "openrouter_batch_llm"

    def test_is_batch_aware(self) -> None:
        """Transform is marked as batch-aware."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        assert transform.is_batch_aware is True

    def test_determinism(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC


class TestOpenRouterBatchEmptyBatch:
    """Tests for empty batch handling.

    Empty batches should never reach batch-aware transforms - the engine
    guards against flushing empty buffers. If an empty batch does reach
    the transform, it indicates a bug in the engine and should crash
    immediately rather than return garbage data.
    """

    def test_empty_batch_raises_runtime_error(self) -> None:
        """Empty batch raises RuntimeError - engine invariant violated."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()

        with pytest.raises(RuntimeError, match="Empty batch passed to batch-aware transform"):
            transform.process([], ctx)


class TestOpenRouterBatchSingleRow:
    """Tests for single row processing (fallback mode)."""

    def test_single_row_success(self, chaosllm_server) -> None:
        """Single row processed successfully."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Hello world"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server, content="Analyzed!")):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Analyzed!"
        assert result.row["text"] == "Hello world"

    def test_single_row_template_error(self, chaosllm_server) -> None:
        """Single row with template error returns error result."""
        transform, ctx = _create_transform_with_context({"template": "{{ row.missing_field }}"})
        row = {"text": "Hello"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(make_pipeline_row(row), ctx)

        # Template error returns error in the row, not TransformResult.error()
        # because batch processing continues with other rows
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert "template_rendering_failed" in str(result.row.get("llm_response_error", {}))


class TestOpenRouterBatchProcessing:
    """Tests for batch processing with parallel execution."""

    def test_batch_success(self, chaosllm_server) -> None:
        """Batch of rows processed successfully."""
        transform, ctx = _create_transform_with_context({"pool_size": 2})
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
            {"text": "Row 3"},
        ]

        # Create responses for each row
        responses = [_create_mock_response(chaosllm_server, content=f"Result {i}") for i in range(3)]

        with mock_httpx_client(chaosllm_server, responses):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Results should be in original order
        for i, output_row in enumerate(result.rows):
            assert output_row["text"] == f"Row {i + 1}"
            assert "llm_response" in output_row

    def test_batch_records_calls_to_audit_trail(self, chaosllm_server) -> None:
        """Batch processing records HTTP calls via AuditedHTTPClient.

        After the refactor, HTTP calls are recorded by AuditedHTTPClient
        through recorder.record_call(), not through ctx.record_call().
        """
        transform, ctx = _create_transform_with_context()
        rows = [{"text": "Row 1"}, {"text": "Row 2"}]

        responses = [_create_mock_response(chaosllm_server) for _ in range(2)]

        with mock_httpx_client(chaosllm_server, responses):
            transform.process([make_pipeline_row(r) for r in rows], ctx)

        # AuditedHTTPClient records calls through the recorder (ctx.landscape)
        recorder = ctx.landscape
        assert recorder.record_call.call_count == 2
        for call in recorder.record_call.call_args_list:
            assert call.kwargs["call_type"] == CallType.HTTP
            assert call.kwargs["status"] == CallStatus.SUCCESS

    def test_batch_partial_failure(self, chaosllm_server) -> None:
        """Batch with some rows failing continues processing others."""
        transform, ctx = _create_transform_with_context({"pool_size": 2})
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
        ]

        # First succeeds, second fails
        success_response = _create_mock_response(chaosllm_server, content="Success")
        error_response = _create_mock_response(chaosllm_server, status_code=500)

        with mock_httpx_client(chaosllm_server, [success_response, error_response]):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2

        # One should have response, one should have error
        has_success = any(r.get("llm_response") is not None for r in result.rows)
        has_error = any(r.get("llm_response_error") is not None for r in result.rows)
        assert has_success
        assert has_error


class TestOpenRouterBatchOutputContract:
    """Tests for output schema contract completeness."""

    def test_contract_includes_error_fields_when_first_row_succeeds(self, chaosllm_server) -> None:
        """Contract must include error fields even when first row succeeds.

        Regression: contract was inferred from first row only, so error-specific
        fields (e.g. llm_response_error) were missing when first row succeeded.
        """
        transform, ctx = _create_transform_with_context({"pool_size": 2})
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
        ]

        success_response = _create_mock_response(chaosllm_server, content="OK")
        error_response = _create_mock_response(chaosllm_server, status_code=500)

        with mock_httpx_client(chaosllm_server, [success_response, error_response]):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.rows is not None
        assert len(result.rows) == 2

        # The error row has llm_response_error — contract must know about it
        error_row = next(r for r in result.rows if r.get("llm_response_error") is not None)
        contract = error_row.contract

        field_names = {f.normalized_name for f in contract.fields}
        assert "llm_response_error" in field_names, "Contract must include error fields from all rows, not just the first"


class TestOpenRouterBatchErrorHandling:
    """Tests for error handling in batch processing."""

    def test_http_status_error_captured(self, chaosllm_server) -> None:
        """HTTP status errors are captured per-row, not raised."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 429
        error = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=error_response,
        )

        with mock_httpx_client(chaosllm_server, side_effect=error):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"  # Single row fallback
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "api_call_failed"
        assert result.row["llm_response_error"]["status_code"] == 429

    def test_network_error_captured(self, chaosllm_server) -> None:
        """Network errors are captured per-row, not raised."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Connection refused")):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "api_call_failed"

    def test_invalid_json_response(self, chaosllm_server) -> None:
        """Invalid JSON response is captured per-row."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body="not json",
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "invalid_json_response"

    def test_malformed_response_structure(self, chaosllm_server) -> None:
        """Response missing expected fields is captured."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps({"unexpected": "structure"}),
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "malformed_response"

    def test_empty_choices_captured(self, chaosllm_server) -> None:
        """Empty choices array is captured per-row."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps({"choices": []}),
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "empty_choices"

    def test_missing_state_id_raises_runtime_error(self, chaosllm_server) -> None:
        """Missing state_id on context is a framework bug and must crash."""
        transform, ctx = _create_transform_with_context()
        ctx.state_id = None  # No state_id
        row = {"text": "Test"}

        with (
            mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)),
            pytest.raises(RuntimeError, match="state_id"),
        ):
            transform.process(make_pipeline_row(row), ctx)


class TestOpenRouterBatchAuditFields:
    """Tests for audit trail field generation."""

    def test_audit_fields_present(self, chaosllm_server) -> None:
        """Successful response includes all audit fields."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.row is not None
        output = result.row

        # Required audit fields
        assert "llm_response" in output
        assert "llm_response_usage" in output
        assert "llm_response_model" in output
        assert "llm_response_template_hash" in output
        assert "llm_response_variables_hash" in output
        assert "llm_response_template_source" in output
        assert "llm_response_lookup_hash" in output
        assert "llm_response_lookup_source" in output
        assert "llm_response_system_prompt_source" in output

    def test_custom_response_field(self, chaosllm_server) -> None:
        """Custom response_field is used for all output fields."""
        transform, ctx = _create_transform_with_context({"response_field": "analysis"})
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.row is not None
        output = result.row

        # Custom field name used
        assert "analysis" in output
        assert "analysis_usage" in output
        assert "analysis_model" in output

        # Default field name not used
        assert "llm_response" not in output


class TestOpenRouterBatchClientCaching:
    """Tests verifying AuditedHTTPClient caching behavior."""

    def test_client_cached_per_state_id(self, chaosllm_server) -> None:
        """AuditedHTTPClient is cached per state_id, reused across rows."""
        transform, _ctx = _create_transform_with_context({"pool_size": 1})

        # With pool_size=1, all rows use the same thread, same state_id
        # so _get_http_client should return the same client each time
        client1 = transform._get_http_client("state-a")
        client2 = transform._get_http_client("state-a")
        client3 = transform._get_http_client("state-b")

        assert client1 is client2, "Same state_id should return cached client"
        assert client1 is not client3, "Different state_id should create new client"

        # Clean up
        transform.close()

    def test_close_cleans_up_clients(self, chaosllm_server) -> None:
        """close() clears all cached HTTP clients."""
        transform, _ctx = _create_transform_with_context()

        # Create a cached client
        transform._get_http_client("state-test")
        assert len(transform._http_clients) == 1

        # Close should clear cache
        transform.close()
        assert len(transform._http_clients) == 0


class TestOpenRouterBatchAllRowsFail:
    """Tests for when all rows in a batch fail."""

    def test_all_rows_fail_returns_success_with_error_markers(self, chaosllm_server) -> None:
        """When all rows fail, success_multi returns rows with error markers.

        Batch processing never returns TransformResult.error() - every row gets
        processed and included in output, even if all fail. Failed rows have
        their error details in the {response_field}_error field.
        """
        transform, ctx = _create_transform_with_context()
        rows = [{"text": "Row 1"}, {"text": "Row 2"}]

        # All requests fail with 500
        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Server Error",
            request=Mock(),
            response=error_response,
        )

        with mock_httpx_client(chaosllm_server, side_effect=error):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        # All rows processed - success_multi with error markers on each row
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        assert all(r.get("llm_response_error") is not None for r in result.rows)


class TestOpenRouterBatchTemplateErrorAuditTrail:
    """Tests for template error audit recording (Bug 56b).

    Template rendering failures must be recorded to the audit trail via
    ctx.record_call() so that explain() queries can show why rows failed.
    Template errors happen BEFORE any HTTP call, so they bypass AuditedHTTPClient.
    """

    def test_template_error_records_to_audit_trail(self, chaosllm_server) -> None:
        """Template rendering error is recorded to audit trail.

        Per CLAUDE.md auditability: 'Every decision must be traceable.'
        A template error is a decision to skip processing that row.
        """
        transform, ctx = _create_transform_with_context({"template": "{{ row.nonexistent_field }}"})
        rows = [{"text": "Test row"}]  # Template references nonexistent_field

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            transform.process([make_pipeline_row(r) for r in rows], ctx)

        # Template error should be recorded to audit trail via ctx.record_call()
        # (not through AuditedHTTPClient, since no HTTP call was made)
        assert ctx.record_call.call_count >= 1

        # Find the template error call
        template_error_calls = [
            call for call in ctx.record_call.call_args_list if call.kwargs.get("error", {}).get("reason") == "template_rendering_failed"
        ]
        assert len(template_error_calls) == 1, "Template error should be recorded to audit trail"

        call = template_error_calls[0]
        assert call.kwargs["call_type"] == CallType.LLM
        assert call.kwargs["status"] == CallStatus.ERROR
        assert "row_index" in call.kwargs["request_data"]


class TestOpenRouterBatchSingleRowFallback:
    """Tests for _process_single fallback behavior (Bug 9g7).

    The _process_single method converts batch results back to single-row.
    If _process_batch returns an unexpected state, it should crash rather
    than silently passing through the original row unprocessed.
    """

    def test_process_single_crashes_on_unexpected_batch_result(self) -> None:
        """TransformResult __post_init__ rejects success without output data.

        Defense-in-depth: status="success" with no row/rows is rejected at
        construction time by the dataclass invariant — the invalid state
        can never exist.
        """
        with pytest.raises(ValueError, match="MUST have output data"):
            TransformResult(
                status="success",
                row=None,  # Single-row not set
                rows=None,  # Multi-row not set either - INVALID for success
                reason=None,
                success_reason={"action": "processed"},  # Required for success status
            )


class TestOpenRouterBatchTelemetryAttribution:
    """Tests for per-row token_id telemetry attribution in batch mode.

    Regression: Before the fix, all rows in a batch had their telemetry
    tagged with the flush-triggering token's ID, breaking per-token
    correlation in observability workflows.
    """

    def test_batch_uses_per_row_token_ids_for_telemetry(self, chaosllm_server) -> None:
        """Each row's HTTP call emits telemetry with its own token_id."""
        from elspeth.contracts.events import ExternalCallCompleted

        emitted_events: list[ExternalCallCompleted] = []
        original_emit = None

        # Create transform and context
        transform, ctx = _create_transform_with_context({"pool_size": 1})

        # Set batch_token_ids as AggregationExecutor would
        ctx.batch_token_ids = ["token-aaa", "token-bbb", "token-ccc"]

        # Intercept telemetry emissions from the AuditedHTTPClient.
        # The transform's on_start() stores self._telemetry_emit from ctx,
        # so we need to capture events at that level.
        original_emit = transform._telemetry_emit

        def capture_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)
            if original_emit is not None:
                original_emit(event)

        transform._telemetry_emit = capture_emit

        rows = [{"text": "Row A"}, {"text": "Row B"}, {"text": "Row C"}]
        responses = [_create_mock_response(chaosllm_server, content=f"Result {i}") for i in range(3)]

        with mock_httpx_client(chaosllm_server, responses):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Verify telemetry events have distinct, correct token_ids
        assert len(emitted_events) == 3
        emitted_token_ids = {e.token_id for e in emitted_events}
        assert emitted_token_ids == {"token-aaa", "token-bbb", "token-ccc"}

    def test_batch_falls_back_to_ctx_token_without_batch_token_ids(self, chaosllm_server) -> None:
        """Without batch_token_ids, falls back to ctx.token (single-row mode)."""
        transform, ctx = _create_transform_with_context()

        # Simulate single-row mode: batch_token_ids is None, ctx.token set
        ctx.batch_token_ids = None
        token_mock = Mock()
        token_mock.token_id = "token-single"
        ctx.token = token_mock

        row = {"text": "Solo row"}
        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
