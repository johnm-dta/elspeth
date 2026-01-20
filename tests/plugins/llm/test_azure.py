# tests/plugins/llm/test_azure.py
"""Tests for Azure OpenAI LLM transform."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


@contextmanager
def mock_azure_openai_client(
    content: str = "Response",
    model: str = "my-gpt4o-deployment",
    usage: dict[str, Any] | None = None,
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Context manager to mock openai.AzureOpenAI.

    Creates a mock client that returns a properly structured response
    from chat.completions.create().

    Args:
        content: Response content to return
        model: Model name in response
        usage: Token usage dict (defaults to prompt_tokens=10, completion_tokens=5)
        side_effect: Exception to raise instead of returning response

    Yields:
        Mock client instance for assertions
    """
    if usage is None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5}

    mock_usage = Mock()
    mock_usage.prompt_tokens = usage.get("prompt_tokens", 0)
    mock_usage.completion_tokens = usage.get("completion_tokens", 0)

    mock_message = Mock()
    mock_message.content = content

    mock_choice = Mock()
    mock_choice.message = mock_message

    mock_response = Mock()
    mock_response.choices = [mock_choice]
    mock_response.model = model
    mock_response.usage = mock_usage
    mock_response.model_dump = Mock(return_value={"model": model})

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        if side_effect:
            mock_client.chat.completions.create.side_effect = side_effect
        else:
            mock_client.chat.completions.create.return_value = mock_response
        mock_azure_class.return_value = mock_client
        yield mock_client


class TestAzureOpenAIConfig:
    """Tests for AzureOpenAIConfig validation."""

    def test_config_requires_deployment_name(self) -> None:
        """AzureOpenAIConfig requires deployment_name."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'deployment_name'

    def test_config_requires_endpoint(self) -> None:
        """AzureOpenAIConfig requires endpoint."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'endpoint'

    def test_config_requires_api_key(self) -> None:
        """AzureOpenAIConfig requires API key."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'api_key'

    def test_config_requires_template(self) -> None:
        """AzureOpenAIConfig requires template (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_schema(self) -> None:
        """AzureOpenAIConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            AzureOpenAIConfig.from_dict(
                {
                    "deployment_name": "my-gpt4o-deployment",
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "model": "gpt-4",
                    "template": "Analyze: {{ row.text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.deployment_name == "my-gpt4o-deployment"
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.api_key == "azure-api-key"
        assert config.template == "Analyze: {{ row.text }}"

    def test_default_api_version(self) -> None:
        """Config has default api_version of 2024-10-21."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.api_version == "2024-10-21"

    def test_custom_api_version(self) -> None:
        """Config accepts custom api_version."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "api_version": "2023-12-01-preview",
            }
        )
        assert config.api_version == "2023-12-01-preview"

    def test_config_inherits_llm_config_defaults(self) -> None:
        """Config inherits defaults from LLMConfig."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        # Inherited from LLMConfig
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"


class TestAzureLLMTransformInit:
    """Tests for AzureLLMTransform initialization."""

    def test_transform_name(self) -> None:
        """Transform has correct name."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.name == "azure_llm"

    def test_transform_stores_azure_config(self) -> None:
        """Transform stores Azure-specific config."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "api_version": "2023-12-01-preview",
            }
        )
        assert transform._azure_endpoint == "https://my-resource.openai.azure.com"
        assert transform._azure_api_key == "azure-api-key"
        assert transform._azure_api_version == "2023-12-01-preview"
        assert transform._deployment_name == "my-gpt4o-deployment"

    def test_model_set_to_deployment_name(self) -> None:
        """Model is set to deployment_name for API calls."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform._model == "my-gpt4o-deployment"

    def test_azure_config_property(self) -> None:
        """azure_config property returns correct values."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "api_version": "2023-12-01-preview",
            }
        )

        config = transform.azure_config
        assert config["endpoint"] == "https://my-resource.openai.azure.com"
        assert config["api_version"] == "2023-12-01-preview"
        assert config["provider"] == "azure"
        # api_key is NOT included in azure_config (security)
        assert "api_key" not in config

    def test_deployment_name_property(self) -> None:
        """deployment_name property returns correct value."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.deployment_name == "my-gpt4o-deployment"

    def test_determinism_is_non_deterministic(self) -> None:
        """Azure transforms are marked as non-deterministic."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_config_validation_failure_deployment_name(self) -> None:
        """Missing deployment_name raises PluginConfigError."""
        with pytest.raises(PluginConfigError):
            AzureLLMTransform(
                {
                    "endpoint": "https://my-resource.openai.azure.com",
                    "api_key": "azure-api-key",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )


class TestAzureLLMTransformProcess:
    """Tests for AzureLLMTransform processing.

    These tests verify that AzureLLMTransform creates its own AuditedLLMClient
    and processes rows correctly. The tests mock openai.AzureOpenAI.
    """

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape and state_id."""
        return PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
        )

    @pytest.fixture
    def transform(self) -> AzureLLMTransform:
        """Create a basic Azure transform."""
        return AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_successful_llm_call_returns_enriched_row(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Successful LLM call returns row with response."""
        with mock_azure_openai_client(
            content="The analysis is positive.",
            model="my-gpt4o-deployment",
            usage={"prompt_tokens": 10, "completion_tokens": 25},
        ):
            result = transform.process({"text": "hello world"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "The analysis is positive."
        assert result.row["llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 25,
        }
        assert "llm_response_template_hash" in result.row
        assert "llm_response_variables_hash" in result.row
        assert result.row["llm_response_model"] == "my-gpt4o-deployment"
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_model_passed_to_azure_client_is_deployment_name(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """deployment_name is used as model in Azure client calls."""
        with mock_azure_openai_client() as mock_client:
            transform.process({"text": "hello"}, ctx)

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "my-gpt4o-deployment"

    def test_template_rendering_error_returns_transform_error(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Template rendering failure returns TransformResult.error()."""
        # Missing required 'text' field triggers template error
        with mock_azure_openai_client():
            result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_llm_client_error_returns_transform_error(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """LLM client failure returns TransformResult.error()."""
        # Mock the underlying client to raise an exception
        with mock_azure_openai_client(side_effect=Exception("API Error")):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"
        assert "API Error" in result.reason["error"]
        assert result.retryable is False

    def test_rate_limit_error_is_retryable(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Rate limit errors marked retryable=True."""
        # Rate limit errors contain "rate" or "429" in the message
        with mock_azure_openai_client(side_effect=Exception("Rate limit exceeded 429")):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "rate_limited"
        assert result.retryable is True

    def test_missing_landscape_raises_runtime_error(
        self, transform: AzureLLMTransform
    ) -> None:
        """Missing landscape in context raises RuntimeError."""
        ctx = PluginContext(run_id="test-run", config={}, state_id="test-state")
        ctx.landscape = None

        with pytest.raises(RuntimeError, match="requires landscape recorder"):
            transform.process({"text": "hello"}, ctx)

    def test_missing_state_id_raises_runtime_error(
        self, mock_recorder: Mock, transform: AzureLLMTransform
    ) -> None:
        """Missing state_id in context raises RuntimeError."""
        ctx = PluginContext(
            run_id="test-run", config={}, landscape=mock_recorder, state_id=None
        )

        with pytest.raises(RuntimeError, match="requires landscape recorder"):
            transform.process({"text": "hello"}, ctx)

    def test_system_prompt_included_in_messages(self, ctx: PluginContext) -> None:
        """System prompt is included when configured."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "system_prompt": "You are a helpful assistant.",
            }
        )

        with mock_azure_openai_client() as mock_client:
            transform.process({"text": "hello"}, ctx)

            # Verify messages passed to client
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "hello"

    def test_temperature_and_max_tokens_passed_to_client(
        self, ctx: PluginContext
    ) -> None:
        """Temperature and max_tokens are passed to Azure client."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )

        with mock_azure_openai_client() as mock_client:
            transform.process({"text": "hello"}, ctx)

            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "my-gpt4o-deployment"
            assert call_args.kwargs["temperature"] == 0.7
            assert call_args.kwargs["max_tokens"] == 500

    def test_custom_response_field(self, ctx: PluginContext) -> None:
        """Custom response_field name is used."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "response_field": "analysis",
            }
        )

        with mock_azure_openai_client(content="Result"):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row
        assert "analysis_model" in result.row

    def test_close_is_noop(self, transform: AzureLLMTransform) -> None:
        """close() does nothing but doesn't raise."""
        transform.close()  # Should not raise

    def test_azure_client_created_with_correct_credentials(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """AzureOpenAI client is created with correct credentials."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            # Set up the mock to return a properly configured client
            mock_client = Mock()
            mock_usage = Mock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 5
            mock_message = Mock()
            mock_message.content = "Response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.model = "my-gpt4o-deployment"
            mock_response.usage = mock_usage
            mock_response.model_dump = Mock(return_value={})
            mock_client.chat.completions.create.return_value = mock_response
            mock_azure_class.return_value = mock_client

            transform.process({"text": "hello"}, ctx)

            # Verify AzureOpenAI was called with correct args
            mock_azure_class.assert_called_once_with(
                azure_endpoint="https://my-resource.openai.azure.com",
                api_key="azure-api-key",
                api_version="2024-10-21",
            )


class TestAzureLLMTransformIntegration:
    """Integration-style tests for Azure-specific edge cases."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape and state_id."""
        return PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
        )

    def test_azure_config_with_default_api_version(self, ctx: PluginContext) -> None:
        """azure_config uses default api_version when not specified."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        config = transform.azure_config
        assert config["api_version"] == "2024-10-21"

    def test_complex_template_with_multiple_variables(self, ctx: PluginContext) -> None:
        """Complex template with multiple variables works correctly."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": """
                    Analyze the following data:
                    Name: {{ row.name }}
                    Score: {{ row.score }}
                    Category: {{ row.category }}

                    Provide a summary.
                """,
                "schema": DYNAMIC_SCHEMA,
            }
        )

        with mock_azure_openai_client(content="Summary text") as mock_client:
            result = transform.process(
                {"name": "Test Item", "score": 95, "category": "A"},
                ctx,
            )

            assert result.status == "success"
            # Check the prompt was rendered correctly
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[0]["content"]
            assert "Test Item" in user_message
            assert "95" in user_message
            assert "A" in user_message

    def test_calls_are_recorded_to_landscape(self, ctx: PluginContext) -> None:
        """LLM calls are recorded via AuditedLLMClient."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        with mock_azure_openai_client():
            transform.process({"text": "hello"}, ctx)

        # Verify record_call was called (by AuditedLLMClient)
        assert ctx.landscape is not None
        assert ctx.landscape.record_call.called  # type: ignore[attr-defined]
