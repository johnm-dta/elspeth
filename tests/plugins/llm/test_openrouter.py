# tests/plugins/llm/test_openrouter.py
"""Tests for OpenRouter LLM transform."""

from unittest.mock import Mock

import httpx
import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestOpenRouterConfig:
    """Tests for OpenRouterConfig validation."""

    def test_config_requires_api_key(self) -> None:
        """OpenRouterConfig requires API key."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "model": "anthropic/claude-3-opus",
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'api_key'

    def test_config_requires_model(self) -> None:
        """OpenRouterConfig requires model name (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'model'

    def test_config_requires_template(self) -> None:
        """OpenRouterConfig requires template (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "anthropic/claude-3-opus",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_schema(self) -> None:
        """OpenRouterConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "anthropic/claude-3-opus",
                    "template": "Analyze: {{ text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.api_key == "sk-test-key"
        assert config.model == "anthropic/claude-3-opus"
        assert config.template == "Analyze: {{ text }}"

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Hello, {{ name }}!",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.timeout_seconds == 60.0
        # Inherited from LLMConfig
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"

    def test_config_custom_base_url(self) -> None:
        """Config accepts custom base URL."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "base_url": "https://custom.openrouter.proxy/api/v1",
            }
        )
        assert config.base_url == "https://custom.openrouter.proxy/api/v1"

    def test_config_custom_timeout(self) -> None:
        """Config accepts custom timeout."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "timeout_seconds": 120.0,
            }
        )
        assert config.timeout_seconds == 120.0

    def test_timeout_must_be_positive(self) -> None:
        """Timeout must be positive."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "openai/gpt-4",
                    "template": "{{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "timeout_seconds": 0,
                }
            )

        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "openai/gpt-4",
                    "template": "{{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "timeout_seconds": -10,
                }
            )


class TestOpenRouterLLMTransformInit:
    """Tests for OpenRouterLLMTransform initialization."""

    def test_transform_name(self) -> None:
        """Transform has correct name."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.name == "openrouter_llm"

    def test_transform_stores_openrouter_config(self) -> None:
        """Transform stores OpenRouter-specific config."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "base_url": "https://custom.example.com/api/v1",
                "timeout_seconds": 90.0,
            }
        )
        assert transform._api_key == "sk-test-key"
        assert transform._base_url == "https://custom.example.com/api/v1"
        assert transform._timeout == 90.0

    def test_determinism_is_non_deterministic(self) -> None:
        """OpenRouter transforms are marked as non-deterministic."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.determinism == Determinism.NON_DETERMINISTIC


class TestOpenRouterLLMTransformProcess:
    """Tests for OpenRouterLLMTransform processing."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> OpenRouterLLMTransform:
        """Create a basic OpenRouter transform."""
        return OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def _mock_response(
        self,
        content: str = "Analysis result",
        model: str = "anthropic/claude-3-opus",
        usage: dict[str, int] | None = None,
        status_code: int = 200,
    ) -> Mock:
        """Create a mock HTTP response."""
        response = Mock(spec=httpx.Response)
        response.status_code = status_code
        response.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "model": model,
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
        }
        response.raise_for_status = Mock()
        return response

    def test_successful_api_call_returns_enriched_row(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Successful API call returns row with LLM response."""
        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response(
            content="The analysis is positive.",
            usage={"prompt_tokens": 10, "completion_tokens": 25},
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
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus"
        # Original data preserved
        assert result.row["text"] == "hello world"

    def test_template_rendering_error_returns_transform_error(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Template rendering failure returns TransformResult.error()."""
        ctx.http_client = Mock()

        # Missing required_field triggers template error
        result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_http_error_returns_transform_error(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """HTTP error returns TransformResult.error()."""
        ctx.http_client = Mock()
        ctx.http_client.post.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=Mock(),
            response=Mock(status_code=500),
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False

    def test_rate_limit_429_is_retryable(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Rate limit (429) errors are marked retryable."""
        ctx.http_client = Mock()
        ctx.http_client.post.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=Mock(),
            response=Mock(status_code=429),
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is True

    def test_request_error_not_retryable(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Network/connection errors (RequestError) are not retryable."""
        ctx.http_client = Mock()
        ctx.http_client.post.side_effect = httpx.ConnectError("Connection refused")

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False

    def test_missing_http_client_raises_runtime_error(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Missing http_client in context raises RuntimeError."""
        ctx.http_client = None

        with pytest.raises(RuntimeError, match="HTTP client not available"):
            transform.process({"text": "hello"}, ctx)

    def test_system_prompt_included_in_request(self, ctx: PluginContext) -> None:
        """System prompt is included when configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "system_prompt": "You are a helpful assistant.",
            }
        )

        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        # Verify request body
        call_args = ctx.http_client.post.call_args
        request_body = call_args.kwargs["json"]
        messages = request_body["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"

    def test_no_system_prompt_single_message(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Without system prompt, only user message is sent."""
        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.http_client.post.call_args
        request_body = call_args.kwargs["json"]
        messages = request_body["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_custom_base_url_used_in_request(self, ctx: PluginContext) -> None:
        """Custom base_url is used in HTTP request."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "base_url": "https://custom.proxy.com/api/v1",
            }
        )

        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.http_client.post.call_args
        url = call_args.args[0]
        assert url == "https://custom.proxy.com/api/v1/chat/completions"

    def test_authorization_header_included(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Authorization header is included with API key."""
        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.http_client.post.call_args
        headers = call_args.kwargs["headers"]

        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_temperature_and_max_tokens_in_request(self, ctx: PluginContext) -> None:
        """Temperature and max_tokens are passed to API."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )

        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.http_client.post.call_args
        request_body = call_args.kwargs["json"]

        assert request_body["model"] == "openai/gpt-4"
        assert request_body["temperature"] == 0.7
        assert request_body["max_tokens"] == 500

    def test_max_tokens_omitted_when_none(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """max_tokens is not included when not set."""
        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response()

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.http_client.post.call_args
        request_body = call_args.kwargs["json"]

        assert "max_tokens" not in request_body

    def test_custom_response_field(self, ctx: PluginContext) -> None:
        """Custom response_field name is used."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "response_field": "analysis",
            }
        )

        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response(content="Result text")

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result text"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row
        assert "analysis_model" in result.row

    def test_model_from_response_used_when_available(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """Model name from response is used if different from request."""
        ctx.http_client = Mock()
        ctx.http_client.post.return_value = self._mock_response(
            model="anthropic/claude-3-opus-20240229"  # Different from request
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.row is not None
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus-20240229"

    def test_raise_for_status_called(
        self, ctx: PluginContext, transform: OpenRouterLLMTransform
    ) -> None:
        """raise_for_status is called on response to check errors."""
        ctx.http_client = Mock()
        response = self._mock_response()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request",
            request=Mock(),
            response=Mock(status_code=400),
        )
        ctx.http_client.post.return_value = response

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        response.raise_for_status.assert_called_once()

    def test_close_is_noop(self, transform: OpenRouterLLMTransform) -> None:
        """close() does nothing but doesn't raise."""
        transform.close()  # Should not raise


class TestOpenRouterLLMTransformIntegration:
    """Integration-style tests for edge cases."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_complex_template_with_multiple_variables(self, ctx: PluginContext) -> None:
        """Complex template with multiple variables works correctly."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
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

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Summary text"}}],
            "model": "openai/gpt-4",
            "usage": {},
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        result = transform.process(
            {"name": "Test Item", "score": 95, "category": "A"},
            ctx,
        )

        assert result.status == "success"
        # Check the prompt was rendered correctly
        call_args = ctx.http_client.post.call_args
        request_body = call_args.kwargs["json"]
        user_message = request_body["messages"][0]["content"]
        assert "Test Item" in user_message
        assert "95" in user_message
        assert "A" in user_message

    def test_empty_usage_handled_gracefully(self, ctx: PluginContext) -> None:
        """Empty usage dict from API is handled."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "openai/gpt-4",
            # No "usage" field at all
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response_usage"] == {}

    def test_connection_error_returns_transform_error(self, ctx: PluginContext) -> None:
        """Network connection error returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.http_client = Mock()
        ctx.http_client.post.side_effect = httpx.ConnectError(
            "Failed to connect to server"
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert "connect" in result.reason["error"].lower()
        assert result.retryable is False  # Connection errors not auto-retryable

    def test_timeout_passed_to_http_client(self, ctx: PluginContext) -> None:
        """Custom timeout_seconds is passed to HTTP client post call."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "timeout_seconds": 120.0,  # Custom timeout
            }
        )

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "openai/gpt-4",
            "usage": {},
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        transform.process({"text": "hello"}, ctx)

        # Verify timeout was passed to the HTTP client
        call_kwargs = ctx.http_client.post.call_args.kwargs
        assert call_kwargs["timeout"] == 120.0

    def test_empty_choices_returns_error(self, ctx: PluginContext) -> None:
        """Empty choices array returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [],  # Empty choices
            "model": "openai/gpt-4",
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_choices"
        assert result.retryable is False

    def test_missing_choices_key_returns_error(self, ctx: PluginContext) -> None:
        """Missing 'choices' key in response returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "error": {"message": "Invalid request"},  # Error payload with 200
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"
        assert "KeyError" in result.reason["error"]
        assert result.retryable is False

    def test_malformed_choice_structure_returns_error(self, ctx: PluginContext) -> None:
        """Malformed choice structure returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.http_client = Mock()
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"wrong_key": "no message field"}],
            "model": "openai/gpt-4",
        }
        response.raise_for_status = Mock()
        ctx.http_client.post.return_value = response

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"
        assert result.retryable is False
