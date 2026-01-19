# tests/plugins/llm/test_azure_batch.py
"""Tests for Azure OpenAI Batch API LLM transform."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_batch import AzureBatchConfig, AzureBatchLLMTransform
from elspeth.plugins.llm.batch_errors import BatchPendingError

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
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_config_requires_endpoint(self) -> None:
        """AzureBatchConfig requires endpoint."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "api_key": "azure-api-key",
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_config_requires_api_key(self) -> None:
        """AzureBatchConfig requires api_key."""
        with pytest.raises(PluginConfigError):
            AzureBatchConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-batch",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
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
                    "template": "Analyze: {{ text }}",
                }
            )

    def test_valid_minimal_config(self) -> None:
        """Valid minimal config passes validation."""
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        assert config.deployment_name == "my-gpt4o-batch"
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.api_key == "azure-api-key"
        assert config.template == "Analyze: {{ text }}"

    def test_default_values(self) -> None:
        """Config has sensible defaults."""
        config = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ x }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ x }}",
                "schema": DYNAMIC_SCHEMA,
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
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
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
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        assert transform.deployment_name == "my-gpt4o-batch"


class TestAzureBatchLLMTransformEmptyBatch:
    """Tests for empty batch handling."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context with checkpoint support."""
        ctx = PluginContext(run_id="test-run", config={})
        ctx._checkpoint = {}  # type: ignore[attr-defined]
        ctx.get_checkpoint = lambda: ctx._checkpoint or None  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_empty_batch_returns_success_with_metadata(
        self, ctx: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
        """Empty batch returns success with batch_empty metadata."""
        result = transform.process([], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["batch_empty"] is True
        assert result.row["row_count"] == 0


class TestAzureBatchLLMTransformCheckpointRequired:
    """Tests for checkpoint API requirement."""

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_raises_without_checkpoint_api(
        self, transform: AzureBatchLLMTransform
    ) -> None:
        """Processing fails with RuntimeError when checkpoint API is missing."""
        # Context without checkpoint methods
        ctx = PluginContext(run_id="test-run", config={})

        rows = [{"text": "hello"}]

        with pytest.raises(RuntimeError) as exc_info:
            transform.process(rows, ctx)

        assert "checkpoint API" in str(exc_info.value)
        assert "get_checkpoint" in str(exc_info.value)


class TestAzureBatchLLMTransformSubmit:
    """Tests for batch submission (Phase 1)."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context with checkpoint support."""
        ctx = PluginContext(run_id="test-run", config={})
        ctx._checkpoint = {}  # type: ignore[attr-defined]
        ctx.get_checkpoint = lambda: ctx._checkpoint or None  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_fresh_batch_submits_and_raises_pending(
        self, ctx: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_checkpoint_saved_after_submit(
        self, ctx: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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


class TestAzureBatchLLMTransformTemplateErrors:
    """Tests for template rendering error handling."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create plugin context with checkpoint support."""
        ctx = PluginContext(run_id="test-run", config={})
        ctx._checkpoint = {}  # type: ignore[attr-defined]
        ctx.get_checkpoint = lambda: ctx._checkpoint or None  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]
        return ctx

    def test_all_templates_fail_returns_error(self, ctx: PluginContext) -> None:
        """When all templates fail, return error immediately."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ required_field }}",  # Requires specific field
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
        """Create plugin context with existing checkpoint."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        # Use a recent timestamp to avoid timeout
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint = {  # type: ignore[attr-defined]
            "batch_id": "batch-456",
            "input_file_id": "file-123",
            "row_mapping": {"row-0-abc12345": 0},
            "template_errors": [],
            "submitted_at": recent_timestamp,
            "row_count": 1,
        }
        ctx.get_checkpoint = lambda: ctx._checkpoint if ctx._checkpoint else None  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_resume_with_checkpoint_checks_status(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_completed_batch_downloads_results(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_failed_batch_returns_error(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_cancelled_batch_returns_error(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_expired_batch_returns_error(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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

    def test_checkpoint_cleared_on_completion(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "max_wait_hours": 1,  # Short timeout for testing
            }
        )

    def test_batch_timeout_returns_error(
        self, transform: AzureBatchLLMTransform
    ) -> None:
        """Batch exceeding max_wait_hours returns error."""
        ctx = PluginContext(run_id="test-run", config={})
        # Checkpoint from 2 hours ago
        ctx._checkpoint = {  # type: ignore[attr-defined]
            "batch_id": "batch-456",
            "input_file_id": "file-123",
            "row_mapping": {},
            "template_errors": [],
            "submitted_at": "2020-01-01T10:00:00+00:00",  # Old timestamp
            "row_count": 1,
        }
        ctx.get_checkpoint = lambda: ctx._checkpoint  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]

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
        """Create plugin context with checkpoint support."""
        ctx = PluginContext(run_id="test-run", config={})
        ctx._checkpoint = {}  # type: ignore[attr-defined]
        ctx.get_checkpoint = lambda: ctx._checkpoint or None  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_single_row_input_raises_pending(
        self, ctx: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_results_assembled_in_original_order(
        self, transform: AzureBatchLLMTransform
    ) -> None:
        """Results are assembled in original row order."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint = {  # type: ignore[attr-defined]
            "batch_id": "batch-456",
            "input_file_id": "file-123",
            "row_mapping": {
                "row-0-aaa": 0,
                "row-1-bbb": 1,
                "row-2-ccc": 2,
            },
            "template_errors": [],
            "submitted_at": recent_timestamp,
            "row_count": 3,
        }
        ctx.get_checkpoint = lambda: ctx._checkpoint  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]

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
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint = {  # type: ignore[attr-defined]
            "batch_id": "batch-456",
            "input_file_id": "file-123",
            "row_mapping": {"row-0-aaa": 0, "row-1-bbb": 1},
            "template_errors": [],
            "submitted_at": recent_timestamp,
            "row_count": 2,
        }
        ctx.get_checkpoint = lambda: ctx._checkpoint  # type: ignore[attr-defined]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[attr-defined]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[attr-defined]

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


class TestAzureBatchLLMTransformClose:
    """Tests for resource cleanup."""

    def test_close_clears_client(self) -> None:
        """close() clears the client reference."""
        transform = AzureBatchLLMTransform(
            {
                "deployment_name": "my-gpt4o-batch",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # No client set
        assert transform._client is None

        # Should not raise
        transform.close()
