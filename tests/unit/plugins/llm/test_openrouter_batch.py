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
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.llm.openrouter_batch import (
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

    def test_batch_partial_failure_reports_quarantined_indices(self, chaosllm_server) -> None:
        """Mixed-success batches must surface failed row indices to the engine."""
        transform, ctx = _create_transform_with_context({"pool_size": 1})
        rows = [
            {"text": "Row 1"},
            {"text": "Row 2"},
        ]

        success_response = _create_mock_response(chaosllm_server, content="Success")
        error_response = _create_mock_response(chaosllm_server, status_code=500)

        with mock_httpx_client(chaosllm_server, [success_response, error_response]):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.success_reason is not None
        metadata = result.success_reason["metadata"]
        assert metadata["quarantined_indices"] == [1]
        assert metadata["quarantined_count"] == 1
        assert len(metadata["row_errors"]) == 1
        assert metadata["row_errors"][0]["row_index"] == 1
        assert metadata["row_errors"][0]["reason"] == "api_call_failed"
        assert metadata["row_errors"][0]["status_code"] == 500
        assert "500" in metadata["row_errors"][0]["error"]


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

    def test_tool_calls_finish_reason_returns_failure(self, chaosllm_server) -> None:
        """tool_calls finish_reason must produce per-row failure, not success."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "tool_calls"}],
                "model": "openai/gpt-4o-mini",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        response = _create_mock_response(
            chaosllm_server,
            raw_body=body,
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "unsupported_finish_reason"
        assert result.row["llm_response_error"]["finish_reason"] == "tool_calls"

    def test_length_finish_reason_returns_failure(self, chaosllm_server) -> None:
        """finish_reason=length (truncated) must produce per-row failure, not success."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "partial answer"}, "finish_reason": "length"}],
                "model": "openai/gpt-4o-mini",
                "usage": {"prompt_tokens": 10, "completion_tokens": 100},
            }
        )
        response = _create_mock_response(
            chaosllm_server,
            raw_body=body,
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] is None
        assert result.row["llm_response_error"]["reason"] == "non_stop_finish_reason"
        assert result.row["llm_response_error"]["finish_reason"] == "length"

    def test_stop_finish_reason_succeeds(self, chaosllm_server) -> None:
        """finish_reason=stop is the normal success case — must produce row with content."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "Complete answer"}, "finish_reason": "stop"}],
                "model": "openai/gpt-4o-mini",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        response = _create_mock_response(
            chaosllm_server,
            raw_body=body,
            headers={"content-type": "application/json"},
        )

        with mock_httpx_client(chaosllm_server, response):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Complete answer"
        assert result.row.get("llm_response_error") is None

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

    def test_operational_fields_present_in_row(self, chaosllm_server) -> None:
        """Successful response includes operational fields in row (not audit provenance)."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.row is not None
        output = result.row

        # Operational fields in row (contract-stable)
        assert "llm_response" in output
        assert "llm_response_usage" in output
        assert "llm_response_model" in output

        # Audit provenance fields must NOT be in row — they go to success_reason["metadata"]
        assert "llm_response_template_hash" not in output
        assert "llm_response_variables_hash" not in output
        assert "llm_response_template_source" not in output
        assert "llm_response_lookup_hash" not in output
        assert "llm_response_lookup_source" not in output
        assert "llm_response_system_prompt_source" not in output

    def test_batch_audit_metadata_in_success_reason(self, chaosllm_server) -> None:
        """Batch-level template provenance appears in success_reason["metadata"]."""
        transform, ctx = _create_transform_with_context()
        rows = [{"text": "Row 1"}, {"text": "Row 2"}]

        responses = [_create_mock_response(chaosllm_server) for _ in range(2)]

        with mock_httpx_client(chaosllm_server, responses):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.success_reason is not None
        assert result.success_reason["action"] == "enriched"
        metadata = result.success_reason["metadata"]
        assert metadata["batch_size"] == 2
        assert "llm_response_template_hash" in metadata
        assert "llm_response_variables_hash" in metadata
        assert metadata["llm_response_variables_hash"] is None  # Per-row hashes in calls table
        assert "llm_response_template_source" in metadata

    def test_per_row_variables_hash_is_recorded_in_call_audit(self, chaosllm_server) -> None:
        """Per-row prompt lineage must persist variables_hash in the call audit record."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Test"}
        row_input = make_pipeline_row(row)
        expected_hash = transform._template.render_with_metadata(row_input, contract=row_input.contract).variables_hash

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server)):
            result = transform.process(row_input, ctx)

        assert result.status == "success"
        recorder = ctx.landscape
        assert recorder.record_call.call_count == 1
        request_data = recorder.record_call.call_args.kwargs["request_data"].to_dict()
        assert request_data["audit_metadata"]["variables_hash"] == expected_hash

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
        ctx.batch_token_ids = ("token-aaa", "token-bbb", "token-ccc")

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


class TestOpenRouterBatchDeclaredOutputFields:
    """Tests for declared_output_fields — centralized collision detection support.

    Field collision detection is enforced centrally by TransformExecutor
    (see TestTransformExecutor in test_executors.py). These tests verify
    that OpenRouterBatchLLMTransform correctly declares its output fields so the
    executor can perform pre-execution collision checks.
    """

    def test_declared_output_fields_contains_response_field(self) -> None:
        """declared_output_fields includes the main response field."""
        transform, _ctx = _create_transform_with_context()
        assert "llm_response" in transform.declared_output_fields

    def test_declared_output_fields_contains_operational_fields(self) -> None:
        """declared_output_fields includes guaranteed operational metadata fields."""
        transform, _ctx = _create_transform_with_context()
        assert "llm_response_usage" in transform.declared_output_fields
        assert "llm_response_model" in transform.declared_output_fields
        # Audit provenance fields (template_hash etc.) are NOT in declared_output_fields —
        # they go to success_reason["metadata"], not the row
        assert "llm_response_template_hash" not in transform.declared_output_fields


class TestOpenRouterBatchErrorKeyCollision:
    """Regression tests for ydk4: source rows with an 'error' field.

    Before the fix, _process_batch used key-presence detection
    ("error" in result) to distinguish success from error dicts. If a
    source row had a field named "error", the success output dict
    contained that key, triggering the error branch and silently
    dropping the LLM response.

    The fix replaces raw dict sentinels with a typed per-row process result
    dataclass that branches on an explicit .ok flag.
    """

    def test_success_row_with_error_field_not_misclassified(self, chaosllm_server) -> None:
        """Source row with 'error' field is processed as SUCCESS, not misclassified."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Hello world", "error": "previous_pipeline_error_value"}

        with mock_httpx_client(chaosllm_server, _create_mock_response(chaosllm_server, content="Analyzed!")):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        # LLM response must be populated, NOT None
        assert result.row["llm_response"] == "Analyzed!"
        # Original 'error' field must be preserved
        assert result.row["error"] == "previous_pipeline_error_value"
        # No error marker should be present
        assert result.row.get("llm_response_error") is None

    def test_batch_with_error_field_rows_all_succeed(self, chaosllm_server) -> None:
        """Batch where every row has an 'error' field — all should succeed."""
        transform, ctx = _create_transform_with_context({"pool_size": 2})
        rows = [
            {"text": "Row 1", "error": "err_a"},
            {"text": "Row 2", "error": "err_b"},
            {"text": "Row 3", "error": "err_c"},
        ]

        responses = [_create_mock_response(chaosllm_server, content=f"Result {i}") for i in range(3)]

        with mock_httpx_client(chaosllm_server, responses):
            result = transform.process([make_pipeline_row(r) for r in rows], ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        for i, output_row in enumerate(result.rows):
            # LLM response populated
            assert output_row["llm_response"] is not None
            # Original error field preserved
            assert output_row["error"] == f"err_{'abc'[i]}"
            # No error marker
            assert output_row.get("llm_response_error") is None

    def test_error_row_with_error_field_still_captured_correctly(self, chaosllm_server) -> None:
        """Source row with 'error' field + failed LLM call captures the error correctly."""
        transform, ctx = _create_transform_with_context()
        row = {"text": "Hello", "error": "pre-existing"}

        error_response = Mock(spec=httpx.Response)
        error_response.status_code = 500
        error = httpx.HTTPStatusError(
            "Server Error",
            request=Mock(),
            response=error_response,
        )

        with mock_httpx_client(chaosllm_server, side_effect=error):
            result = transform.process(make_pipeline_row(row), ctx)

        assert result.status == "success"
        assert result.row is not None
        # LLM response should be None (error)
        assert result.row["llm_response"] is None
        # Error should be from the API failure, not from the source row's "error" field
        assert result.row["llm_response_error"]["reason"] == "api_call_failed"
        assert result.row["llm_response_error"]["status_code"] == 500


class TestBug4_8_FieldCollisionBeforeAPICall:
    """Bug 4.8: Field collision detected before making API call.

    Originally, the plugin-level collision check happened after the API call,
    wasting expensive HTTP requests. That was fixed to check before the call.

    Now, collision detection is fully centralized in TransformExecutor
    (see TestTransformExecutor in test_executors.py). The pre-execution check
    prevents the transform from running at all when collisions exist.

    This test verifies declared_output_fields is populated correctly so the
    executor can perform its pre-execution collision check.
    """

    def test_declared_output_fields_enables_executor_level_check(self) -> None:
        """declared_output_fields is populated, enabling executor collision detection."""
        transform, _ctx = _create_transform_with_context()

        # The field that would collide must be in declared_output_fields
        assert "llm_response" in transform.declared_output_fields
        assert isinstance(transform.declared_output_fields, frozenset)
        assert len(transform.declared_output_fields) > 0


class TestRowProcessResultPostFreezeMutationGuards:
    """Exercise the ``freeze_fields`` post-init guards on ``_RowSuccess`` /
    ``_RowFailure``.

    ``enforce_freeze_guards.py`` is a static scanner — a typo like
    ``freeze_fields(self)`` (no args) or ``freeze_fields(self, "wrong_name")``
    still parses as a valid call expression and silently defeats the scan
    without tripping the regex.  These tests pin the runtime contract by
    mutating each container field after construction and asserting the
    post-freeze TypeError surfaces.  If a future refactor drops a field
    from the ``freeze_fields`` call, the corresponding mutation succeeds
    and the test fails loudly.
    """

    def test_row_success_row_is_immutable_after_construction(self) -> None:
        """Mutating ``_RowSuccess.row`` after construction must raise TypeError.

        A mutation that succeeds would leave a frozen dataclass with a
        live shared reference — the "frozen=True is a lie" pattern
        CLAUDE.md's deep-freeze contract was written to prevent.
        """
        from elspeth.plugins.transforms.llm.openrouter_batch import _RowSuccess

        success = _RowSuccess(row={"k": "v"}, finish_reason="stop")

        with pytest.raises(TypeError):
            # MappingProxyType raises TypeError on item assignment.
            success.row["k"] = "mutated"  # type: ignore[index]

        with pytest.raises(TypeError):
            success.row["new_key"] = "x"  # type: ignore[index]

    def test_row_failure_error_is_immutable_after_construction(self) -> None:
        """Mutating ``_RowFailure.error`` after construction must raise TypeError."""
        from elspeth.plugins.transforms.llm.openrouter_batch import _RowFailure

        failure = _RowFailure(error={"code": "rate_limited", "detail": "429"})

        with pytest.raises(TypeError):
            failure.error["code"] = "timeout"  # type: ignore[index]

        with pytest.raises(TypeError):
            failure.error["extra"] = "x"  # type: ignore[index]

    def test_row_success_source_detachment(self) -> None:
        """Mutating the source dict passed to _RowSuccess must NOT affect the stored view.

        Belt-and-suspenders alongside the post-freeze tests: confirms the
        deep-freeze took a detached snapshot, not a live reference.
        """
        from elspeth.plugins.transforms.llm.openrouter_batch import _RowSuccess

        source = {"k": "v"}
        success = _RowSuccess(row=source, finish_reason="stop")
        source["k"] = "mutated"
        source["new_key"] = "x"

        assert success.row["k"] == "v"
        assert "new_key" not in success.row
