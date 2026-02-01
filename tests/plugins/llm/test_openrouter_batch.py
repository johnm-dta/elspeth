# tests/plugins/llm/test_openrouter_batch.py
"""Tests for OpenRouter batch LLM transform.

Tests cover:
- Configuration validation
- Single row processing (fallback mode)
- Batch processing with parallel execution
- Error handling (template, HTTP, API errors)
- Thread safety with shared httpx.Client
- Audit trail recording
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.openrouter_batch import (
    OpenRouterBatchConfig,
    OpenRouterBatchLLMTransform,
)

from .conftest import chaosllm_openrouter_http_responses, chaosllm_openrouter_httpx_response

# Common schema config for dynamic field handling
DYNAMIC_SCHEMA = {"fields": "dynamic"}


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
) -> Mock:
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
    """Create a mock PluginContext for testing."""
    ctx = Mock(spec=PluginContext)
    ctx.run_id = run_id
    ctx.state_id = state_id
    ctx.landscape = Mock()
    ctx.record_call = Mock()
    return ctx


def mock_httpx_client(
    chaosllm_server,
    responses: list[httpx.Response] | httpx.Response | None = None,
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Context manager to mock httpx.Client using ChaosLLM responses."""
    if responses is None and side_effect is None:
        responses = _create_mock_response(chaosllm_server)
    normalized = responses if isinstance(responses, list) else [responses]
    return chaosllm_openrouter_http_responses(
        chaosllm_server,
        normalized,
        side_effect=side_effect,
    )


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
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Hello world"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server, content="Analyzed!")):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Analyzed!"
        assert result.row["text"] == "Hello world"

    def test_single_row_template_error(self, chaosllm_server) -> None:
        """Single row with template error returns error result."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"template": "{{ row.missing_field }}"}))
        ctx = _create_mock_context()
        row = {"text": "Hello"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(row, ctx)

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
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"pool_size": 2}))
        ctx = _create_mock_context()
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
            {"text": "Row 3"},
        ]

        # Create responses for each row
        responses = [_create_mock_response(chaosllm_server, content=f"Result {i}") for i in range(3)]

        with mock_httpx_client(chaosllm_server, responses):
            result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Results should be in original order
        for i, output_row in enumerate(result.rows):
            assert output_row["text"] == f"Row {i + 1}"
            assert "llm_response" in output_row

    def test_batch_records_calls_to_audit_trail(self, chaosllm_server) -> None:
        """Batch processing records all calls to audit trail."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        rows = [{"text": "Row 1"}, {"text": "Row 2"}]

        responses = [_create_mock_response(chaosllm_server) for _ in range(2)]

        with mock_httpx_client(chaosllm_server, responses):
            transform.process(rows, ctx)

        # Should have recorded calls for each row
        assert ctx.record_call.call_count == 2
        for call in ctx.record_call.call_args_list:
            assert call.kwargs["call_type"] == CallType.LLM
            assert call.kwargs["status"] == CallStatus.SUCCESS

    def test_batch_partial_failure(self, chaosllm_server) -> None:
        """Batch with some rows failing continues processing others."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"pool_size": 2}))
        ctx = _create_mock_context()
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
        ]

        # First succeeds, second fails
        success_response = _create_mock_response(chaosllm_server, content="Success")
        error_response = _create_mock_response(chaosllm_server, status_code=500)

        with mock_httpx_client(chaosllm_server, [success_response, error_response]):
            result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2

        # One should have response, one should have error
        has_success = any(r.get("llm_response") is not None for r in result.rows)
        has_error = any(r.get("llm_response_error") is not None for r in result.rows)
        assert has_success
        assert has_error


class TestOpenRouterBatchErrorHandling:
    """Tests for error handling in batch processing."""

    def test_http_status_error_captured(self, chaosllm_server) -> None:
        """HTTP status errors are captured per-row, not raised."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 429
        error = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=error_response,
        )

        with mock_httpx_client(chaosllm_server, side_effect=error):
            result = transform.process(row, ctx)

        assert result.status == "success"  # Single row fallback
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "api_call_failed"
        assert result.row["llm_response_error"]["status_code"] == 429

    def test_network_error_captured(self, chaosllm_server) -> None:
        """Network errors are captured per-row, not raised."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, side_effect=httpx.ConnectError("Connection refused")):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "api_call_failed"

    def test_invalid_json_response(self, chaosllm_server) -> None:
        """Invalid JSON response is captured per-row."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body="not json",
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "invalid_json_response"

    def test_malformed_response_structure(self, chaosllm_server) -> None:
        """Response missing expected fields is captured."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps({"unexpected": "structure"}),
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "malformed_response"

    def test_empty_choices_captured(self, chaosllm_server) -> None:
        """Empty choices array is captured per-row."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        response = _create_mock_response(
            chaosllm_server,
            raw_body=json.dumps({"choices": []}),
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "empty_choices"

    def test_missing_state_id_returns_error(self, chaosllm_server) -> None:
        """Missing state_id on context returns row-level error."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        ctx.state_id = None  # No state_id
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "missing_state_id"


class TestOpenRouterBatchAuditFields:
    """Tests for audit trail field generation."""

    def test_audit_fields_present(self, chaosllm_server) -> None:
        """Successful response includes all audit fields."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(row, ctx)

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
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"response_field": "analysis"}))
        ctx = _create_mock_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(row, ctx)

        assert result.row is not None
        output = result.row

        # Custom field name used
        assert "analysis" in output
        assert "analysis_usage" in output
        assert "analysis_model" in output

        # Default field name not used
        assert "llm_response" not in output


class TestOpenRouterBatchSharedClient:
    """Tests verifying shared httpx.Client behavior."""

    def test_client_created_once_per_batch(self, chaosllm_server) -> None:
        """httpx.Client is created once for the entire batch, not per row."""
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"pool_size": 3}))
        ctx = _create_mock_context()
        rows = [{"text": f"Row {i}"} for i in range(5)]

        with patch("elspeth.plugins.llm.openrouter_batch.httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.post.return_value = _create_mock_response(chaosllm_server)
            mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = Mock(return_value=None)

            transform.process(rows, ctx)

            # Client constructor called exactly once (not 5 times)
            assert mock_client_class.call_count == 1

            # But post() called 5 times (once per row)
            assert mock_client.post.call_count == 5


class TestOpenRouterBatchAllRowsFail:
    """Tests for when all rows in a batch fail."""

    def test_all_rows_fail_returns_success_with_error_markers(self, chaosllm_server) -> None:
        """When all rows fail, success_multi returns rows with error markers.

        Batch processing never returns TransformResult.error() - every row gets
        processed and included in output, even if all fail. Failed rows have
        their error details in the {response_field}_error field.
        """
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
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
            result = transform.process(rows, ctx)

        # All rows processed - success_multi with error markers on each row
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        assert all(r.get("llm_response_error") is not None for r in result.rows)


class TestOpenRouterBatchTemplateErrorAuditTrail:
    """Tests for template error audit recording (Bug 56b).

    Template rendering failures must be recorded to the audit trail via
    ctx.record_call() so that explain() queries can show why rows failed.
    """

    def test_template_error_records_to_audit_trail(self, chaosllm_server) -> None:
        """Template rendering error is recorded to audit trail.

        Per CLAUDE.md auditability: 'Every decision must be traceable.'
        A template error is a decision to skip processing that row.
        """
        transform = OpenRouterBatchLLMTransform(_make_valid_config({"template": "{{ row.nonexistent_field }}"}))
        ctx = _create_mock_context()
        rows = [{"text": "Test row"}]  # Template references nonexistent_field

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            transform.process(rows, ctx)

        # Template error should be recorded to audit trail
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
        """_process_single raises RuntimeError on unexpected _process_batch result.

        Defense-in-depth: if _process_batch somehow returns success without
        rows (should never happen), we crash rather than silently pass through.
        """
        transform = OpenRouterBatchLLMTransform(_make_valid_config())
        ctx = _create_mock_context()
        row = {"text": "Test"}

        # Mock _process_batch to return an invalid state:
        # status="success" but rows=None (violates contract)
        # Note: We must provide success_reason to pass __post_init__ validation,
        # but the result is still invalid because neither row nor rows is set.
        invalid_result = TransformResult(
            status="success",
            row=None,  # Single-row not set
            rows=None,  # Multi-row not set either - INVALID for success
            reason=None,
            success_reason={"action": "processed"},  # Required for success status
        )

        with (
            patch.object(transform, "_process_batch", return_value=invalid_result),
            pytest.raises(RuntimeError, match="Unexpected result from _process_batch"),
        ):
            transform.process(row, ctx)
