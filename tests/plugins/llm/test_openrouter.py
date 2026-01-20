# tests/plugins/llm/test_openrouter.py
"""Tests for OpenRouter LLM transform."""

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import Mock, patch

import httpx
import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


def _create_mock_response(
    content: str = "Analysis result",
    model: str = "anthropic/claude-3-opus",
    usage: dict[str, int] | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    raise_for_status_error: Exception | None = None,
) -> Mock:
    """Create a mock HTTP response for testing."""
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {"content-type": "application/json"}
    response.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "model": model,
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20},
    }
    if raise_for_status_error:
        response.raise_for_status.side_effect = raise_for_status_error
    else:
        response.raise_for_status = Mock()
    response.content = b""
    response.text = ""
    return response


@contextmanager
def mock_httpx_client(response: Mock | None = None, side_effect: Exception | None = None) -> Generator[Mock, None, None]:
    """Context manager to mock httpx.Client with a response or side_effect."""
    with patch("httpx.Client") as mock_client_class:
        mock_client = Mock()
        if side_effect:
            mock_client.post.side_effect = side_effect
        elif response:
            mock_client.post.return_value = response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=None)
        yield mock_client


class TestOpenRouterConfig:
    """Tests for OpenRouterConfig validation."""

    def test_config_requires_api_key(self) -> None:
        """OpenRouterConfig requires API key."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "model": "anthropic/claude-3-opus",
                    "template": "Analyze: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'api_key'

    def test_config_requires_model(self) -> None:
        """OpenRouterConfig requires model name (from LLMConfig)."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "template": "Analyze: {{ row.text }}",
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
                    "template": "Analyze: {{ row.text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.api_key == "sk-test-key"
        assert config.model == "anthropic/claude-3-opus"
        assert config.template == "Analyze: {{ row.text }}"

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Hello, {{ row.name }}!",
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
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "base_url": "https://custom.proxy.com/api/v1",
            }
        )
        assert config.base_url == "https://custom.proxy.com/api/v1"

    def test_config_custom_timeout(self) -> None:
        """Config accepts custom timeout."""
        config = OpenRouterConfig.from_dict(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "timeout_seconds": 120.0,
            }
        )
        assert config.timeout_seconds == 120.0

    def test_config_invalid_timeout_rejected(self) -> None:
        """Config rejects non-positive timeout."""
        with pytest.raises(PluginConfigError):
            OpenRouterConfig.from_dict(
                {
                    "api_key": "sk-test-key",
                    "model": "openai/gpt-4",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "timeout_seconds": 0.0,  # Must be > 0
                }
            )


class TestOpenRouterLLMTransformInit:
    """Tests for OpenRouterLLMTransform initialization."""

    def test_transform_stores_config_values(self) -> None:
        """Transform stores config values as attributes."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "base_url": "https://custom.example.com/api/v1",
                "timeout_seconds": 90.0,
            }
        )

        assert transform._model == "anthropic/claude-3-opus"
        assert transform._api_key == "sk-test-key"
        assert transform._base_url == "https://custom.example.com/api/v1"
        assert transform._timeout == 90.0

    def test_determinism_is_non_deterministic(self) -> None:
        """OpenRouter transforms are marked as non-deterministic."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert transform.determinism == Determinism.NON_DETERMINISTIC


class TestOpenRouterLLMTransformProcess:
    """Tests for OpenRouterLLMTransform processing."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock()
        recorder.record_call = Mock()
        return recorder

    @pytest.fixture
    def ctx(self, mock_recorder: Mock) -> PluginContext:
        """Create plugin context with landscape and state_id for audited HTTP."""
        return PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_recorder,
            state_id="test-state-id",
        )

    @pytest.fixture
    def transform(self) -> OpenRouterLLMTransform:
        """Create a basic OpenRouter transform."""
        return OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_successful_api_call_returns_enriched_row(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Successful API call returns row with LLM response."""
        mock_response = _create_mock_response(
            content="The analysis is positive.",
            usage={"prompt_tokens": 10, "completion_tokens": 25},
        )

        with mock_httpx_client(response=mock_response):
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

    def test_template_rendering_error_returns_transform_error(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Template rendering failure returns TransformResult.error()."""
        # Missing required_field triggers template error (no HTTP call needed)
        result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_http_error_returns_transform_error(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """HTTP error returns TransformResult.error()."""
        with mock_httpx_client(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=Mock(status_code=500),
            )
        ):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False

    def test_rate_limit_429_is_retryable(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Rate limit (429) errors are marked retryable."""
        with mock_httpx_client(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=Mock(),
                response=Mock(status_code=429),
            )
        ):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is True

    def test_request_error_not_retryable(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Network/connection errors (RequestError) are not retryable."""
        with mock_httpx_client(side_effect=httpx.ConnectError("Connection refused")):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False

    def test_missing_landscape_raises_runtime_error(self, transform: OpenRouterLLMTransform) -> None:
        """Missing landscape in context raises RuntimeError."""
        ctx = PluginContext(run_id="test-run", config={}, landscape=None, state_id=None)

        with pytest.raises(RuntimeError, match="requires landscape"):
            transform.process({"text": "hello"}, ctx)

    def test_system_prompt_included_in_request(self, ctx: PluginContext, mock_recorder: Mock) -> None:
        """System prompt is included when configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "system_prompt": "You are a helpful assistant.",
            }
        )

        mock_response = _create_mock_response()
        with mock_httpx_client(response=mock_response) as mock_client:
            transform.process({"text": "hello"}, ctx)

            # Verify request body
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            messages = request_body["messages"]

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "hello"

    def test_no_system_prompt_single_message(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Without system prompt, only user message is sent."""
        mock_response = _create_mock_response()
        with mock_httpx_client(response=mock_response) as mock_client:
            transform.process({"text": "hello"}, ctx)

            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            messages = request_body["messages"]

            assert len(messages) == 1
            assert messages[0]["role"] == "user"

    def test_custom_response_field(self, ctx: PluginContext) -> None:
        """Custom response_field name is used."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "response_field": "analysis",
            }
        )

        mock_response = _create_mock_response(content="Result text")
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result text"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row
        assert "analysis_model" in result.row

    def test_model_from_response_used_when_available(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """Model name from response is used if different from request."""
        mock_response = _create_mock_response(
            model="anthropic/claude-3-opus-20240229"  # Different from request
        )
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.row is not None
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus-20240229"

    def test_raise_for_status_called(self, ctx: PluginContext, transform: OpenRouterLLMTransform) -> None:
        """raise_for_status is called on response to check errors."""
        mock_response = _create_mock_response(
            raise_for_status_error=httpx.HTTPStatusError(
                "400 Bad Request",
                request=Mock(),
                response=Mock(status_code=400),
            )
        )
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"

    def test_close_is_noop(self, transform: OpenRouterLLMTransform) -> None:
        """close() does nothing but doesn't raise."""
        transform.close()  # Should not raise


class TestOpenRouterLLMTransformIntegration:
    """Integration-style tests for edge cases."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
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

    def test_complex_template_with_multiple_variables(self, ctx: PluginContext) -> None:
        """Complex template with multiple variables works correctly."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
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

        mock_response = _create_mock_response(content="Summary text")
        with mock_httpx_client(response=mock_response) as mock_client:
            result = transform.process(
                {"name": "Test Item", "score": 95, "category": "A"},
                ctx,
            )

            assert result.status == "success"
            # Check the prompt was rendered correctly
            call_args = mock_client.post.call_args
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
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}],
            "model": "openai/gpt-4",
            # No "usage" field at all
        }
        response.raise_for_status = Mock()
        response.content = b""
        response.text = ""

        with mock_httpx_client(response=response):
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
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        with mock_httpx_client(side_effect=httpx.ConnectError("Failed to connect to server")):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert "connect" in result.reason["error"].lower()
        assert result.retryable is False  # Connection errors not auto-retryable

    def test_timeout_passed_to_http_client(self, ctx: PluginContext) -> None:
        """Custom timeout_seconds is used when creating HTTP client."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "timeout_seconds": 120.0,  # Custom timeout
            }
        )

        # The transform stores the timeout internally
        assert transform._timeout == 120.0

        mock_response = _create_mock_response()
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"

    def test_empty_choices_returns_error(self, ctx: PluginContext) -> None:
        """Empty choices array returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {
            "choices": [],  # Empty choices
            "model": "openai/gpt-4",
            "usage": {},
        }
        response.raise_for_status = Mock()
        response.content = b""
        response.text = ""

        with mock_httpx_client(response=response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_choices"

    def test_missing_choices_key_returns_error(self, ctx: PluginContext) -> None:
        """Missing 'choices' key in response returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {
            "model": "openai/gpt-4",
            "usage": {},
            # No "choices" key
        }
        response.raise_for_status = Mock()
        response.content = b""
        response.text = ""

        with mock_httpx_client(response=response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"

    def test_malformed_choice_structure_returns_error(self, ctx: PluginContext) -> None:
        """Malformed choice structure returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {
            "choices": [{"wrong_key": "no message"}],  # Missing "message"
            "model": "openai/gpt-4",
            "usage": {},
        }
        response.raise_for_status = Mock()
        response.content = b""
        response.text = ""

        with mock_httpx_client(response=response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "malformed_response"

    def test_invalid_json_response_returns_error(self, ctx: PluginContext) -> None:
        """Non-JSON response body returns TransformResult.error()."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.headers = {"content-type": "text/html"}
        # Simulate non-JSON response (e.g., HTML error page from proxy)
        response.json.side_effect = ValueError("No JSON object could be decoded")
        response.raise_for_status = Mock()
        response.text = "<html><body>Error: Service Unavailable</body></html>"
        response.content = b"<html><body>Error: Service Unavailable</body></html>"

        with mock_httpx_client(response=response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "invalid_json_response"
        assert "content_type" in result.reason
        assert result.reason["content_type"] == "text/html"
        assert "body_preview" in result.reason
        assert "Error: Service Unavailable" in result.reason["body_preview"]


class TestOpenRouterTemplateFeatures:
    """Tests for template files and lookup features in OpenRouter transform."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
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

    def test_lookup_data_accessible_in_template(self, ctx: PluginContext) -> None:
        """Lookup data is accessible via lookup.* namespace in templates."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Classify as {{ lookup.categories[0] }} or {{ lookup.categories[1] }}: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "lookup": {"categories": ["positive", "negative"]},
            }
        )

        mock_response = _create_mock_response(content="positive")
        with mock_httpx_client(response=mock_response) as mock_client:
            result = transform.process({"text": "I love this!"}, ctx)

            assert result.status == "success"
            # Verify template rendered correctly with lookup data
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            user_message = request_body["messages"][0]["content"]
            assert "Classify as positive or negative:" in user_message
            assert "I love this!" in user_message

    def test_two_dimensional_lookup_row_plus_lookup(self, ctx: PluginContext) -> None:
        """Two-dimensional lookup: lookup.X[row.Y] works correctly."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Use tone: {{ lookup.tones[row.tone_id] }}. Message: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "lookup": {"tones": {"formal": "professional", "casual": "friendly"}},
            }
        )

        mock_response = _create_mock_response(content="Processed")
        with mock_httpx_client(response=mock_response) as mock_client:
            result = transform.process({"text": "Hello", "tone_id": "formal"}, ctx)

            assert result.status == "success"
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            user_message = request_body["messages"][0]["content"]
            assert "Use tone: professional" in user_message

    def test_lookup_hash_included_in_output(self, ctx: PluginContext) -> None:
        """Output includes lookup_hash when lookup data is configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Categories: {{ lookup.cats }}. Input: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                "lookup": {"cats": ["A", "B", "C"]},
            }
        )

        mock_response = _create_mock_response(content="Result")
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "test"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        # New audit fields should be present
        assert "llm_response_lookup_hash" in result.row
        assert result.row["llm_response_lookup_hash"] is not None
        # lookup_source is None when lookup is inline (not from file)
        assert "llm_response_lookup_source" in result.row

    def test_template_source_included_in_output(self, ctx: PluginContext) -> None:
        """Output includes template_source when provided."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Analyze: {{ row.text }}",
                "template_source": "prompts/analysis.j2",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        mock_response = _create_mock_response(content="Analysis result")
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response_template_source"] == "prompts/analysis.j2"

    def test_all_audit_fields_present_with_lookup(self, ctx: PluginContext) -> None:
        """All audit metadata fields are present when using template with lookup."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "{{ lookup.prompt_prefix }} {{ row.text }}",
                "template_source": "prompts/prefixed.j2",
                "schema": DYNAMIC_SCHEMA,
                "lookup": {"prompt_prefix": "Please analyze:"},
                "lookup_source": "prompts/lookups.yaml",
            }
        )

        mock_response = _create_mock_response(content="Done")
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "data"}, ctx)

        assert result.status == "success"
        assert result.row is not None

        # All audit fields should be present
        assert "llm_response_template_hash" in result.row
        assert "llm_response_variables_hash" in result.row
        assert "llm_response_template_source" in result.row
        assert "llm_response_lookup_hash" in result.row
        assert "llm_response_lookup_source" in result.row
        assert "llm_response_model" in result.row

        # Values should be set correctly
        assert result.row["llm_response_template_source"] == "prompts/prefixed.j2"
        assert result.row["llm_response_lookup_source"] == "prompts/lookups.yaml"
        assert result.row["llm_response_lookup_hash"] is not None

    def test_no_lookup_has_none_hash_in_output(self, ctx: PluginContext) -> None:
        """Output has None for lookup fields when no lookup configured."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Simple: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
                # No lookup configured
            }
        )

        mock_response = _create_mock_response(content="Result")
        with mock_httpx_client(response=mock_response):
            result = transform.process({"text": "test"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        # Fields should be present but None
        assert "llm_response_lookup_hash" in result.row
        assert result.row["llm_response_lookup_hash"] is None
        assert "llm_response_lookup_source" in result.row
        assert result.row["llm_response_lookup_source"] is None

    def test_lookup_iteration_in_template(self, ctx: PluginContext) -> None:
        """Lookup data can be iterated in templates."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": """Classify into one of:
{% for cat in lookup.categories %}
- {{ cat.name }}: {{ cat.description }}
{% endfor %}
Text: {{ row.text }}""",
                "schema": DYNAMIC_SCHEMA,
                "lookup": {
                    "categories": [
                        {"name": "spam", "description": "unwanted messages"},
                        {"name": "ham", "description": "legitimate messages"},
                    ]
                },
            }
        )

        mock_response = _create_mock_response(content="spam")
        with mock_httpx_client(response=mock_response) as mock_client:
            result = transform.process({"text": "Buy now! Limited offer!"}, ctx)

            assert result.status == "success"
            call_args = mock_client.post.call_args
            request_body = call_args.kwargs["json"]
            user_message = request_body["messages"][0]["content"]
            assert "spam: unwanted messages" in user_message
            assert "ham: legitimate messages" in user_message

    def test_template_error_includes_source_in_error_details(self, ctx: PluginContext) -> None:
        """Template rendering error includes template_source for debugging."""
        transform = OpenRouterLLMTransform(
            {
                "api_key": "sk-test-key",
                "model": "openai/gpt-4",
                "template": "Missing: {{ row.required_field }}",
                "template_source": "prompts/requires_field.j2",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Process with missing required_field - should fail template rendering
        result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert result.reason["template_source"] == "prompts/requires_field.j2"
