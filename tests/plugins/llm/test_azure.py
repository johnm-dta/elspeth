# tests/plugins/llm/test_azure.py
"""Tests for Azure OpenAI LLM transform."""

from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.clients.llm import LLMClientError, LLMResponse, RateLimitError
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


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
                    "template": "Analyze: {{ text }}",
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
                    "template": "Analyze: {{ text }}",
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
                    "template": "Analyze: {{ text }}",
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
                    "template": "Analyze: {{ text }}",
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
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.deployment_name == "my-gpt4o-deployment"
        assert config.endpoint == "https://my-resource.openai.azure.com"
        assert config.api_key == "azure-api-key"
        assert config.template == "Analyze: {{ text }}"

    def test_default_api_version(self) -> None:
        """Config has default api_version of 2024-10-21."""
        config = AzureOpenAIConfig.from_dict(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "model": "gpt-4",
                "template": "{{ text }}",
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
                "template": "{{ text }}",
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
                "template": "{{ text }}",
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
                "template": "{{ text }}",
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
                "template": "{{ text }}",
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        # BaseLLMTransform stores model from config
        assert transform._model == "my-gpt4o-deployment"

    def test_azure_config_property(self) -> None:
        """azure_config property returns correct values for executor."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "api_version": "2023-12-01-preview",
            }
        )

        config = transform.azure_config
        assert config["endpoint"] == "https://my-resource.openai.azure.com"
        assert config["api_key"] == "azure-api-key"
        assert config["api_version"] == "2023-12-01-preview"
        assert config["provider"] == "azure"

    def test_deployment_name_property(self) -> None:
        """deployment_name property returns correct value."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
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
                "template": "{{ text }}",
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
                    "template": "{{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )


class TestAzureLLMTransformProcess:
    """Tests for AzureLLMTransform processing.

    These tests verify that AzureLLMTransform inherits process() behavior
    from BaseLLMTransform correctly. The tests mock the llm_client which
    the executor would provide.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> AzureLLMTransform:
        """Create a basic Azure transform."""
        return AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_successful_llm_call_returns_enriched_row(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Successful LLM call returns row with response."""
        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="The analysis is positive.",
            model="my-gpt4o-deployment",
            usage={"prompt_tokens": 10, "completion_tokens": 25},
            latency_ms=150.0,
        )

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
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_model_passed_to_llm_client_is_deployment_name(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """deployment_name is used as model in LLM client calls."""
        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="my-gpt4o-deployment",
            usage={},
        )

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.llm_client.chat_completion.call_args
        assert call_args.kwargs["model"] == "my-gpt4o-deployment"

    def test_template_rendering_error_returns_transform_error(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Template rendering failure returns TransformResult.error()."""
        ctx.llm_client = Mock()

        # Missing required_field triggers template error
        result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_llm_client_error_returns_transform_error(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """LLM client failure returns TransformResult.error()."""
        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.side_effect = LLMClientError("API Error")

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
        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.side_effect = RateLimitError(
            "Rate limit exceeded"
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "rate_limited"
        assert result.retryable is True

    def test_missing_llm_client_raises_runtime_error(
        self, ctx: PluginContext, transform: AzureLLMTransform
    ) -> None:
        """Missing llm_client in context raises RuntimeError."""
        ctx.llm_client = None

        with pytest.raises(RuntimeError, match="LLM client not available"):
            transform.process({"text": "hello"}, ctx)

    def test_system_prompt_included_in_messages(self, ctx: PluginContext) -> None:
        """System prompt is included when configured."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "system_prompt": "You are a helpful assistant.",
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="my-gpt4o-deployment",
            usage={},
        )

        transform.process({"text": "hello"}, ctx)

        # Verify messages passed to client
        call_args = ctx.llm_client.chat_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"

    def test_temperature_and_max_tokens_passed_to_client(
        self, ctx: PluginContext
    ) -> None:
        """Temperature and max_tokens are passed to LLM client."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="my-gpt4o-deployment",
            usage={},
        )

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.llm_client.chat_completion.call_args
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
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "response_field": "analysis",
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Result",
            model="my-gpt4o-deployment",
            usage={},
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row

    def test_close_is_noop(self, transform: AzureLLMTransform) -> None:
        """close() does nothing but doesn't raise."""
        transform.close()  # Should not raise


class TestAzureLLMTransformIntegration:
    """Integration-style tests for Azure-specific edge cases."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_azure_config_with_default_api_version(self, ctx: PluginContext) -> None:
        """azure_config uses default api_version when not specified."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
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
                    Name: {{ name }}
                    Score: {{ score }}
                    Category: {{ category }}

                    Provide a summary.
                """,
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Summary text",
            model="my-gpt4o-deployment",
            usage={},
        )

        result = transform.process(
            {"name": "Test Item", "score": 95, "category": "A"},
            ctx,
        )

        assert result.status == "success"
        # Check the prompt was rendered correctly
        call_args = ctx.llm_client.chat_completion.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[0]["content"]
        assert "Test Item" in user_message
        assert "95" in user_message
        assert "A" in user_message

    def test_retryable_llm_error_propagates_flag(self, ctx: PluginContext) -> None:
        """LLMClientError retryable flag is propagated."""
        transform = AzureLLMTransform(
            {
                "deployment_name": "my-gpt4o-deployment",
                "endpoint": "https://my-resource.openai.azure.com",
                "api_key": "azure-api-key",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        # Non-rate-limit but retryable error
        ctx.llm_client.chat_completion.side_effect = LLMClientError(
            "Server overloaded", retryable=True
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.retryable is True
