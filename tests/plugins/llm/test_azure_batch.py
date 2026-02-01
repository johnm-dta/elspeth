# tests/plugins/llm/test_azure_batch.py
"""Tests for Azure OpenAI Batch API LLM transform."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from elspeth.contracts import BatchPendingError, Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_batch import AzureBatchConfig, AzureBatchLLMTransform

# Common schema config for dynamic field handling
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestBatchPendingError:
    """Tests for BatchPendingError exception."""

    def test_basic_initialization(self) -> None:
        """BatchPendingError stores batch_id and status."""
        error = BatchPendingError("batch-123", "submitted")

        assert error.batch_id == "batch-123"
        assert error.status == "submitted"
        assert error.check_after_seconds == 300  # Default

    def test_custom_check_after_seconds(self) -> None:
        """BatchPendingError accepts custom check interval."""
        error = BatchPendingError("batch-456", "in_progress", check_after_seconds=600)

        assert error.batch_id == "batch-456"
        assert error.status == "in_progress"
        assert error.check_after_seconds == 600

    def test_error_message_format(self) -> None:
        """BatchPendingError has descriptive message."""
        error = BatchPendingError("batch-789", "submitted", check_after_seconds=120)

        assert "batch-789" in str(error)
        assert "submitted" in str(error)
        assert "120" in str(error)


class TestAzureBatchConfig:
    """Tests for AzureBatchConfig validation."""

    def test_config_requires_deployment_name(self) -> None:
        """AzureBatchConfig requires deployment_name."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )

    def test_config_requires_endpoint(self) -> None:
        """AzureBatchConfig requires endpoint."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "api_key": "azure-api-key",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )

    def test_config_requires_api_key(self) -> None:
        """AzureBatchConfig requires api_key."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                }
            )

    def test_config_requires_template(self) -> None:
        """AzureBatchConfig requires template."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_config_requires_schema(self) -> None:
        """AzureBatchConfig requires schema."""
        with pytest.raises(PluginConfigError, match="schema"):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "template": "Analyze: {{ row.text }}",
                }
            )

    def test_valid_minimal_config(self) -> None:
        """Valid minimal config passes validation."""
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert config.deployment_name == "my-gpt4o-batch"
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.api_key == "azure-api-key"
        assert config.template == "Analyze: {{ row.text }}"

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert config.api_version == "2024-10-21"
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"
        assert config.poll_interval_seconds == 300
        assert config.max_wait_hours == 24
        assert config.on_error is None

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "api_version": "2024-06-01",
                "temperature": 0.7,
                "max_tokens": 500,
                "system_prompt": "You are helpful.",
                "response_field": "analysis",
                "poll_interval_seconds": 600,
                "max_wait_hours": 12,
                "on_error": "error_sink",
            }
        )

        assert config.api_version == "2024-06-01"
        assert config.temperature == 0.7
        assert config.max_tokens == 500
        assert config.system_prompt == "You are helpful."
        assert config.response_field == "analysis"
        assert config.poll_interval_seconds == 600
        assert config.max_wait_hours == 12
        assert config.on_error == "error_sink"

    def test_temperature_bounds(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        # Lower bound OK
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "test",
                "endpoint": "https://test.azure.com",
                "api_key": "key",
                "template": "{{ row.x }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "temperature": 0.0,
            }
        )
        assert config.temperature == 0.0

        # Upper bound OK
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "test",
                "endpoint": "https://test.azure.com",
                "api_key": "key",
                "template": "{{ row.x }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "temperature": 2.0,
            }
        )
        assert config.temperature == 2.0

        # Below lower bound
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "test",
                    "endpoint": "https://test.azure.com",
                    "api_key": "key",
                    "template": "{{ row.x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "temperature": -0.1,
                }
            )

        # Above upper bound
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "test",
                    "endpoint": "https://test.azure.com",
                    "api_key": "key",
                    "template": "{{ row.x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "temperature": 2.1,
                }
            )


class TestAzureBatchLLMTransformInit:
    """Tests for AzureBatchLLMTransform initialization."""

    def test_transform_name(self) -> None:
        """Transform has correct name."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert transform.name == "azure_batch_llm"

    def test_is_batch_aware(self) -> None:
        """Transform is batch-aware."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert transform.is_batch_aware is True

    def test_determinism_is_non_deterministic(self) -> None:
        """Transform is marked as non-deterministic."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_stores_config_values(self) -> None:
        """Transform stores all config values."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com/",  # trailing slash
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "api_version": "2024-06-01",
                "temperature": 0.5,
                "max_tokens": 100,
                "system_prompt": "Be helpful",
                "response_field": "result",
                "poll_interval_seconds": 120,
                "max_wait_hours": 6,
            }
        )

        assert transform._deployment_name == "my-gpt4o-batch"
        assert transform._endpoint == "https://my-resource.openai.azure.com"  # stripped
        assert transform._api_key == "azure-api-key"
        assert transform._api_version == "2024-06-01"
        assert transform._temperature == 0.5
        assert transform._max_tokens == 100
        assert transform._system_prompt == "Be helpful"
        assert transform._response_field == "result"
        assert transform._poll_interval == 120
        assert transform._max_wait_hours == 6

    def test_azure_config_property(self) -> None:
        """azure_config property returns correct values."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "api_version": "2024-06-01",
            }
        )

        config = transform.azure_config
        assert config["endpoint"] == "https://my-resource.openai.azure.com"
        assert config["api_key"] == "azure-api-key"
        assert config["api_version"] == "2024-06-01"
        assert config["provider"] == "azure_batch"

    def test_deployment_name_property(self) -> None:
        """deployment_name property returns correct value."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert transform.deployment_name == "my-gpt4o-batch"


class TestAzureBatchLLMTransformEmptyBatch:
    """Tests for empty batch handling.

    Empty batches should never reach batch-aware transforms - the engine
    guards against flushing empty buffers. If an empty batch does reach
    the transform, it indicates a bug in the engine and should crash
    immediately rather than return garbage data.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context for testing.

        PluginContext now has native checkpoint support.
        """
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

    def test_empty_batch_raises_runtime_error(self, ctx: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Empty batch raises RuntimeError - engine invariant violated."""
        with pytest.raises(RuntimeError, match="Empty batch passed to batch-aware transform"):
            transform.process([], ctx)


class TestAzureBatchLLMTransformSubmit:
    """Tests for batch submission (Phase 1)."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context for testing.

        PluginContext now has native checkpoint support via
        get_checkpoint/update_checkpoint/clear_checkpoint methods.
        """
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

    def test_fresh_batch_submits_and_raises_pending(self, ctx: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Fresh batch submits to Azure and raises BatchPendingError."""
        # Mock Azure client
        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}, {"text": "world"}]

        with pytest.raises(BatchPendingError) as exc_info:
            transform.process(rows, ctx)

        error = exc_info.value
        assert error.batch_id == "batch-456"
        assert error.status == "submitted"
        assert error.check_after_seconds == 300

    def test_checkpoint_saved_after_submit(self, ctx: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Checkpoint is saved immediately after batch submission."""
        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx)

        # Verify checkpoint was saved
        checkpoint = ctx._checkpoint  # type: ignore[attr-defined]
        assert checkpoint["batch_id"] == "batch-456"
        assert checkpoint["input_file_id"] == "file-123"
        assert "row_mapping" in checkpoint
        assert checkpoint["row_count"] == 1
        assert "submitted_at" in checkpoint

    def test_system_prompt_included_in_batch_requests(self, ctx: PluginContext) -> None:
        """System prompt is included in batch requests when configured."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "system_prompt": "You are helpful.",
            }
        )

        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx)

        # Check JSONL content
        call_args = mock_client.files.create.call_args
        file_tuple = call_args.kwargs["file"]
        file_content = file_tuple[1].read().decode("utf-8")
        request = json.loads(file_content)

        messages = request["body"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    def test_max_tokens_included_when_set(self, ctx: PluginContext) -> None:
        """max_tokens is included in requests when configured."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "max_tokens": 100,
            }
        )

        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx)

        # Check JSONL content
        call_args = mock_client.files.create.call_args
        file_tuple = call_args.kwargs["file"]
        file_content = file_tuple[1].read().decode("utf-8")
        request = json.loads(file_content)

        assert request["body"]["max_tokens"] == 100

    def test_checkpoint_includes_original_requests(self, ctx: PluginContext) -> None:
        """Checkpoint should include original LLM request data for audit recording.

        This is essential for BUG-AZURE-01 fix: the audit trail must record the actual
        LLM request bodies (including model, messages, temperature) that were sent,
        so that when results come back they can be properly attributed.
        """
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "gpt-4o-batch",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_file.status = "uploaded"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "validating"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "Hello"}, {"text": "World"}]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx)

        # Verify checkpoint includes requests
        checkpoint = ctx._checkpoint  # type: ignore[attr-defined]
        assert checkpoint is not None
        assert "requests" in checkpoint
        assert len(checkpoint["requests"]) == 2

        # Each request should have the full LLM request body
        for _custom_id, request_body in checkpoint["requests"].items():
            assert "messages" in request_body
            assert "model" in request_body
            assert request_body["model"] == "gpt-4o-batch"


class TestAzureBatchLLMTransformTemplateErrors:
    """Tests for template rendering error handling."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context for testing.

        PluginContext now has native checkpoint support.
        """
        return PluginContext(run_id="test-run", config={})

    def test_all_templates_fail_returns_error(self, ctx: PluginContext) -> None:
        """When all templates fail, return error immediately."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.required_field }}",  # Requires specific field
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        # All rows missing required_field
        rows = [{"other": "value1"}, {"other": "value2"}]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "all_templates_failed"
        assert len(result.reason["template_errors"]) == 2

    def test_partial_template_failures_continue(self, ctx: PluginContext) -> None:
        """When some templates fail, successful ones are submitted."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        # Mix of valid and invalid rows
        rows = [
            {"text": "valid1"},
            {"other": "missing_text"},  # Will fail template
            {"text": "valid2"},
        ]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx)

        # Checkpoint should have template_errors
        checkpoint = ctx._checkpoint  # type: ignore[attr-defined]
        assert len(checkpoint["template_errors"]) == 1
        assert checkpoint["template_errors"][0][0] == 1  # Index of failed row


class TestAzureBatchLLMTransformResume:
    """Tests for batch status checking (Phase 2)."""

    @pytest.fixture
    def ctx_with_checkpoint(self) -> PluginContext:
        """Create plugin context with existing checkpoint for resume tests."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        # Pre-populate checkpoint for resume scenario (recent timestamp to avoid timeout)
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint.update(
            {
                "batch_id": "batch-456",
                "input_file_id": "file-123",
                "row_mapping": {"row-0-abc12345": {"index": 0, "variables_hash": "hash0"}},
                "template_errors": [],
                "submitted_at": recent_timestamp,
                "row_count": 1,
                "requests": {
                    "row-0-abc12345": {
                        "messages": [{"role": "user", "content": "test"}],
                        "model": "my-gpt4o-batch",
                    },
                },
            }
        )
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

    def test_resume_with_checkpoint_checks_status(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Resume with checkpoint checks batch status."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        with pytest.raises(BatchPendingError) as exc_info:
            transform.process(rows, ctx_with_checkpoint)

        error = exc_info.value
        assert error.batch_id == "batch-456"
        assert error.status == "in_progress"

    def test_completed_batch_downloads_results(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Completed batch downloads and returns results."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-789"
        mock_client.batches.retrieve.return_value = mock_batch

        # Mock output file content
        output_content = Mock()
        output_content.text = json.dumps(
            {
                "custom_id": "row-0-abc12345",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "Analysis result"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                    }
                },
            }
        )
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [{"text": "hello"}]

        result = transform.process(rows, ctx_with_checkpoint)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 1
        assert result.rows[0]["llm_response"] == "Analysis result"
        assert result.rows[0]["llm_response_usage"]["prompt_tokens"] == 10

    def test_failed_batch_returns_error(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Failed batch returns TransformResult.error()."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "failed"
        mock_batch.errors = Mock()
        mock_batch.errors.data = [Mock(code="rate_limit", message="Too many requests")]
        mock_client.batches.retrieve.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        result = transform.process(rows, ctx_with_checkpoint)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "batch_failed"
        assert result.reason["batch_id"] == "batch-456"

    def test_cancelled_batch_returns_error(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Cancelled batch returns TransformResult.error()."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "cancelled"
        mock_client.batches.retrieve.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        result = transform.process(rows, ctx_with_checkpoint)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "batch_cancelled"

    def test_expired_batch_returns_error(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Expired batch returns TransformResult.error()."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "expired"
        mock_client.batches.retrieve.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        result = transform.process(rows, ctx_with_checkpoint)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "batch_expired"

    def test_checkpoint_cleared_on_completion(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Checkpoint is cleared after successful completion."""
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-789"
        mock_client.batches.retrieve.return_value = mock_batch

        output_content = Mock()
        output_content.text = json.dumps(
            {
                "custom_id": "row-0-abc12345",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "Result"}}],
                        "usage": {},
                    }
                },
            }
        )
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [{"text": "hello"}]

        transform.process(rows, ctx_with_checkpoint)

        # Checkpoint should be cleared
        assert ctx_with_checkpoint._checkpoint == {}  # type: ignore[attr-defined]


class TestAzureBatchLLMTransformTimeout:
    """Tests for batch timeout handling."""

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create transform with short max_wait."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
                "max_wait_hours": 1,  # Short timeout for testing
            }
        )

    def test_batch_timeout_returns_error(self, transform: AzureBatchLLMTransform) -> None:
        """Batch exceeding max_wait_hours returns error."""
        ctx = PluginContext(run_id="test-run", config={})
        # Pre-populate checkpoint from old timestamp for timeout test
        ctx._checkpoint.update(
            {
                "batch_id": "batch-456",
                "input_file_id": "file-123",
                "row_mapping": {},
                "template_errors": [],
                "submitted_at": "2020-01-01T10:00:00+00:00",  # Old timestamp
                "row_count": 1,
            }
        )

        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "in_progress"
        mock_client.batches.retrieve.return_value = mock_batch

        transform._client = mock_client

        rows = [{"text": "hello"}]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "batch_timeout"
        assert result.reason["batch_id"] == "batch-456"


class TestAzureBatchLLMTransformSingleRow:
    """Tests for single row fallback."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context for testing.

        PluginContext now has native checkpoint support.
        """
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

    def test_single_row_input_raises_pending(self, ctx: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Single row input is processed through batch API."""
        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        # Single row (dict, not list)
        row = {"text": "hello"}

        with pytest.raises(BatchPendingError) as exc_info:
            transform.process(row, ctx)

        assert exc_info.value.batch_id == "batch-456"


class TestAzureBatchLLMTransformResultAssembly:
    """Tests for result assembly with multiple rows."""

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

    def test_results_assembled_in_original_order(self, transform: AzureBatchLLMTransform) -> None:
        """Results are assembled in original row order."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        # Pre-populate checkpoint for resume test
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint.update(
            {
                "batch_id": "batch-456",
                "input_file_id": "file-123",
                "row_mapping": {
                    "row-0-aaa": {"index": 0, "variables_hash": "hash0"},
                    "row-1-bbb": {"index": 1, "variables_hash": "hash1"},
                    "row-2-ccc": {"index": 2, "variables_hash": "hash2"},
                },
                "template_errors": [],
                "submitted_at": recent_timestamp,
                "row_count": 3,
                "requests": {
                    "row-0-aaa": {"messages": [{"role": "user", "content": "a"}], "model": "gpt-4o-batch"},
                    "row-1-bbb": {"messages": [{"role": "user", "content": "b"}], "model": "gpt-4o-batch"},
                    "row-2-ccc": {"messages": [{"role": "user", "content": "c"}], "model": "gpt-4o-batch"},
                },
            }
        )

        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-789"
        mock_client.batches.retrieve.return_value = mock_batch

        # Results in random order
        output_lines = [
            json.dumps(
                {
                    "custom_id": "row-2-ccc",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "C"}}],
                            "usage": {},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-0-aaa",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "A"}}],
                            "usage": {},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-1-bbb",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "B"}}],
                            "usage": {},
                        }
                    },
                }
            ),
        ]
        output_content = Mock()
        output_content.text = "\n".join(output_lines)
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3
        # Check order matches input
        assert result.rows[0]["llm_response"] == "A"
        assert result.rows[1]["llm_response"] == "B"
        assert result.rows[2]["llm_response"] == "C"

    def test_api_error_per_row_handled(self, transform: AzureBatchLLMTransform) -> None:
        """API errors for individual rows are handled gracefully."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        # Pre-populate checkpoint for resume test
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint.update(
            {
                "batch_id": "batch-456",
                "input_file_id": "file-123",
                "row_mapping": {"row-0-aaa": {"index": 0, "variables_hash": "hash0"}, "row-1-bbb": {"index": 1, "variables_hash": "hash1"}},
                "template_errors": [],
                "submitted_at": recent_timestamp,
                "row_count": 2,
                "requests": {
                    "row-0-aaa": {"messages": [{"role": "user", "content": "a"}], "model": "gpt-4o-batch"},
                    "row-1-bbb": {"messages": [{"role": "user", "content": "b"}], "model": "gpt-4o-batch"},
                },
            }
        )

        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-456"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-789"
        mock_client.batches.retrieve.return_value = mock_batch

        output_lines = [
            json.dumps(
                {
                    "custom_id": "row-0-aaa",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Success"}}],
                            "usage": {},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-1-bbb",
                    "error": {"code": "content_filter", "message": "Content blocked"},
                }
            ),
        ]
        output_content = Mock()
        output_content.text = "\n".join(output_lines)
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [{"text": "good"}, {"text": "bad"}]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        assert result.rows[0]["llm_response"] == "Success"
        assert result.rows[1]["llm_response"] is None
        assert result.rows[1]["llm_response_error"]["reason"] == "api_error"


class TestAzureBatchLLMTransformAuditRecording:
    """Tests for LLM call audit recording (BUG-AZURE-01)."""

    def test_download_results_records_llm_calls(self) -> None:
        """Processing results should record LLM calls per row against batch state."""
        config = {
            "deployment_name": "gpt-4o-batch",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "Analyze: {{ row.text }}",
            "schema": {"fields": "dynamic"},
            "required_input_fields": [],  # Explicit opt-out for this test
        }
        transform = AzureBatchLLMTransform(config)

        # Track recorded calls
        recorded_calls: list[dict[str, Any]] = []

        def capture_call(**kwargs: Any) -> MagicMock:
            recorded_calls.append(kwargs)
            return MagicMock()

        # Create mock context with state_id (simulates batch's aggregation state)
        ctx = MagicMock()
        ctx.run_id = "test-run"
        ctx.state_id = "batch-state-123"  # The batch's node_state
        ctx.record_call = capture_call
        ctx.get_checkpoint.return_value = {
            "batch_id": "azure-batch-789",
            "row_mapping": {
                "row-0-abc": {"index": 0, "variables_hash": "hash0"},
                "row-1-def": {"index": 1, "variables_hash": "hash1"},
            },
            "template_errors": [],
            "requests": {
                "row-0-abc": {
                    "messages": [{"role": "user", "content": "Analyze: Hello"}],
                    "model": "gpt-4o-batch",
                },
                "row-1-def": {
                    "messages": [{"role": "user", "content": "Analyze: World"}],
                    "model": "gpt-4o-batch",
                },
            },
        }
        ctx.clear_checkpoint = MagicMock()

        # Mock Azure batch completion
        mock_batch = MagicMock()
        mock_batch.id = "azure-batch-789"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-999"

        # Mock output file content (JSONL)
        output_jsonl = """{"custom_id": "row-0-abc", "response": {"body": {"choices": [{"message": {"content": "Analysis: Greeting"}}], "usage": {"total_tokens": 10}}}}
{"custom_id": "row-1-def", "response": {"body": {"choices": [{"message": {"content": "Analysis: Planet"}}], "usage": {"total_tokens": 12}}}}"""

        mock_output_content = MagicMock()
        mock_output_content.text = output_jsonl

        rows = [{"text": "Hello"}, {"text": "World"}]

        with patch.object(transform, "_get_client") as mock_client:
            mock_client.return_value.files.content.return_value = mock_output_content

            result = transform._download_results(mock_batch, ctx.get_checkpoint(), rows, ctx)

        # Verify result is successful
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2

        # Verify LLM calls were recorded
        from elspeth.contracts import CallStatus, CallType

        llm_calls = [c for c in recorded_calls if c.get("call_type") == CallType.LLM]
        assert len(llm_calls) == 2, f"Expected 2 LLM calls, got {len(llm_calls)}"

        # Verify first LLM call has correct data
        call1 = llm_calls[0]
        assert "custom_id" in call1["request_data"]
        assert call1["request_data"]["messages"][0]["content"] == "Analyze: Hello"
        assert call1["status"] == CallStatus.SUCCESS

        # Verify second LLM call
        call2 = llm_calls[1]
        assert call2["request_data"]["messages"][0]["content"] == "Analyze: World"

    def test_download_results_records_failed_llm_calls_correctly(self) -> None:
        """LLM calls that failed should be recorded with ERROR status."""
        config = {
            "deployment_name": "gpt-4o-batch",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "Analyze: {{ row.text }}",
            "schema": {"fields": "dynamic"},
        }
        transform = AzureBatchLLMTransform(config)

        recorded_calls: list[dict[str, Any]] = []

        def capture_call(**kwargs: Any) -> MagicMock:
            recorded_calls.append(kwargs)
            return MagicMock()

        ctx = MagicMock()
        ctx.run_id = "test-run"
        ctx.state_id = "batch-state-123"
        ctx.record_call = capture_call
        ctx.get_checkpoint.return_value = {
            "batch_id": "azure-batch-789",
            "row_mapping": {
                "row-0-abc": {"index": 0, "variables_hash": "hash0"},
                "row-1-def": {"index": 1, "variables_hash": "hash1"},
            },
            "template_errors": [],
            "requests": {
                "row-0-abc": {"messages": [{"role": "user", "content": "Good"}], "model": "gpt-4o-batch"},
                "row-1-def": {"messages": [{"role": "user", "content": "Bad"}], "model": "gpt-4o-batch"},
            },
        }
        ctx.clear_checkpoint = MagicMock()

        mock_batch = MagicMock()
        mock_batch.output_file_id = "output-file-999"

        # One success, one error
        output_jsonl = """{"custom_id": "row-0-abc", "response": {"body": {"choices": [{"message": {"content": "OK"}}]}}}
{"custom_id": "row-1-def", "error": {"code": "content_filter", "message": "Content filtered"}}"""

        mock_output_content = MagicMock()
        mock_output_content.text = output_jsonl

        rows = [{"text": "Good"}, {"text": "Bad"}]

        with patch.object(transform, "_get_client") as mock_client:
            mock_client.return_value.files.content.return_value = mock_output_content
            # Result unused - we're testing side effects (recorded calls)
            transform._download_results(mock_batch, ctx.get_checkpoint(), rows, ctx)

        from elspeth.contracts import CallStatus, CallType

        llm_calls = [c for c in recorded_calls if c.get("call_type") == CallType.LLM]
        assert len(llm_calls) == 2

        # Find success and error calls by content
        success_call = next(c for c in llm_calls if "Good" in str(c["request_data"]))
        error_call = next(c for c in llm_calls if "Bad" in str(c["request_data"]))

        # Success call assertions
        assert success_call["status"] == CallStatus.SUCCESS
        assert success_call["response_data"] is not None
        assert "custom_id" in success_call["request_data"]  # QA requirement

        # Error call assertions
        assert error_call["status"] == CallStatus.ERROR
        assert error_call["error"] is not None
        assert "content_filter" in str(error_call["error"])
        assert "custom_id" in error_call["request_data"]  # QA requirement


class TestAzureBatchLLMTransformClose:
    """Tests for resource cleanup."""

    def test_close_clears_client(self) -> None:
        """close() clears the client reference."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        # Simulate client being set
        transform._client = Mock()

        transform.close()

        assert transform._client is None

    def test_close_is_safe_when_no_client(self) -> None:
        """close() is safe when no client exists."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        # No client set
        assert transform._client is None

        # Should not raise
        transform.close()


class TestAzureBatchLLMTransformMissingResults:
    """Tests for handling missing results from Azure batch output.

    Regression tests for P2-2026-01-31-azure-batch-missing-call-records:
    When a row's result is missing from Azure batch output, we must still
    record an LLM Call to maintain audit trail completeness.
    """

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "required_input_fields": [],
            }
        )

    def test_missing_result_records_error_call(self, transform: AzureBatchLLMTransform) -> None:
        """Missing result from batch output records an ERROR Call for audit trail.

        Per CLAUDE.md: "External calls - Full request AND response recorded"
        A missing response is still recordable - request was made, no response.
        """
        # Create context with mocked landscape
        mock_landscape = MagicMock()
        # Use incrementing call_index (multiple calls: HTTP for retrieve, HTTP for download, then LLM calls)
        call_index_counter = iter(range(100))
        mock_landscape.allocate_call_index.side_effect = lambda _: next(call_index_counter)

        ctx = PluginContext(run_id="test-run", config={})
        ctx.landscape = mock_landscape
        ctx.state_id = "test-state-123"  # Required for record_call

        # Set up checkpoint as if batch was submitted with 2 rows
        checkpoint = {
            "batch_id": "batch-999",
            "input_file_id": "file-input",
            "row_mapping": {
                "row-0-abc": {"index": 0, "variables_hash": "hash0"},
                "row-1-def": {"index": 1, "variables_hash": "hash1"},
            },
            "template_errors": [],
            "submitted_at": "2026-01-31T12:00:00Z",
            "row_count": 2,
            "requests": {
                "row-0-abc": {
                    "model": "my-gpt4o-batch",
                    "messages": [{"role": "user", "content": "Analyze: Hello"}],
                    "temperature": 0.0,
                },
                "row-1-def": {
                    "model": "my-gpt4o-batch",
                    "messages": [{"role": "user", "content": "Analyze: World"}],
                    "temperature": 0.0,
                },
            },
        }
        ctx._checkpoint = checkpoint

        # Mock Azure client - batch is completed
        mock_batch = Mock()
        mock_batch.id = "batch-999"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output"

        mock_file_content = Mock()
        # Only return result for row-0, NOT row-1 (simulating missing result)
        mock_file_content.text = json.dumps(
            {
                "custom_id": "row-0-abc",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "Analysis result"}}],
                        "usage": {"total_tokens": 10},
                        "model": "gpt-4o-batch",
                    },
                },
            }
        )

        mock_client = Mock()
        mock_client.batches.retrieve.return_value = mock_batch
        mock_client.files.content.return_value = mock_file_content

        transform._client = mock_client

        # Process with rows
        rows = [{"text": "Hello"}, {"text": "World"}]
        result = transform.process(rows, ctx)

        # Result should succeed (partial results are allowed)
        assert result.status == "success"
        assert len(result.rows) == 2

        # First row should have response
        assert result.rows[0]["llm_response"] == "Analysis result"

        # Second row should have error
        assert result.rows[1]["llm_response"] is None
        assert result.rows[1]["llm_response_error"]["reason"] == "result_not_found"

        # CRITICAL: Verify record_call was called for BOTH rows
        # The fix adds a record_call for the missing result
        call_args_list = mock_landscape.record_call.call_args_list

        # Should have 4 calls: 2 HTTP (retrieve + download) + 2 LLM (missing + success)
        assert len(call_args_list) == 4, f"Expected 4 record_call invocations, got {len(call_args_list)}"

        # Find the LLM calls only (filter out HTTP calls)
        from elspeth.contracts import CallStatus, CallType

        llm_calls = [call for call in call_args_list if call.kwargs.get("call_type") == CallType.LLM]
        assert len(llm_calls) == 2, f"Expected 2 LLM calls, got {len(llm_calls)}"

        # Find the error call (for missing result)
        error_calls = [call for call in llm_calls if call.kwargs.get("status") == CallStatus.ERROR]
        assert len(error_calls) == 1, "Expected exactly 1 ERROR LLM call for missing result"

        error_call = error_calls[0]
        assert error_call.kwargs["call_type"] == CallType.LLM
        assert error_call.kwargs["request_data"]["custom_id"] == "row-1-def"
        assert error_call.kwargs["request_data"]["row_index"] == 1
        assert error_call.kwargs["error"]["reason"] == "result_not_found"

    def test_all_results_present_no_error_calls(self, transform: AzureBatchLLMTransform) -> None:
        """When all results are present, only SUCCESS calls are recorded."""
        mock_landscape = MagicMock()
        # Use incrementing call_index (multiple calls: HTTP for retrieve, HTTP for download, then LLM calls)
        call_index_counter = iter(range(100))
        mock_landscape.allocate_call_index.side_effect = lambda _: next(call_index_counter)

        ctx = PluginContext(run_id="test-run", config={})
        ctx.landscape = mock_landscape
        ctx.state_id = "test-state-456"

        checkpoint = {
            "batch_id": "batch-888",
            "input_file_id": "file-input",
            "row_mapping": {
                "row-0-abc": {"index": 0, "variables_hash": "hash0"},
            },
            "template_errors": [],
            "submitted_at": "2026-01-31T12:00:00Z",
            "row_count": 1,
            "requests": {
                "row-0-abc": {
                    "model": "my-gpt4o-batch",
                    "messages": [{"role": "user", "content": "Analyze: Hello"}],
                    "temperature": 0.0,
                },
            },
        }
        ctx._checkpoint = checkpoint

        mock_batch = Mock()
        mock_batch.id = "batch-888"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output"

        mock_file_content = Mock()
        mock_file_content.text = json.dumps(
            {
                "custom_id": "row-0-abc",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "Analysis result"}}],
                        "usage": {"total_tokens": 10},
                        "model": "gpt-4o-batch",
                    },
                },
            }
        )

        mock_client = Mock()
        mock_client.batches.retrieve.return_value = mock_batch
        mock_client.files.content.return_value = mock_file_content

        transform._client = mock_client

        rows = [{"text": "Hello"}]
        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows[0]["llm_response"] == "Analysis result"

        # Only SUCCESS calls for LLM, no ERROR LLM calls
        from elspeth.contracts import CallStatus, CallType

        call_args_list = mock_landscape.record_call.call_args_list

        # Find LLM calls only
        llm_calls = [call for call in call_args_list if call.kwargs.get("call_type") == CallType.LLM]
        assert len(llm_calls) == 1, f"Expected 1 LLM call, got {len(llm_calls)}"

        # Should be SUCCESS, not ERROR
        assert llm_calls[0].kwargs["status"] == CallStatus.SUCCESS

        # No ERROR LLM calls when all results present
        error_llm_calls = [call for call in llm_calls if call.kwargs.get("status") == CallStatus.ERROR]
        assert len(error_llm_calls) == 0, "Should have no ERROR LLM calls when all results present"
