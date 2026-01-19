# tests/plugins/llm/test_base.py
"""Tests for base LLM transform."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from elspeth.contracts import Determinism
from elspeth.plugins.clients.llm import LLMClientError, LLMResponse, RateLimitError
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestLLMConfig:
    """Tests for LLMConfig validation."""

    def test_config_requires_template(self) -> None:
        """LLMConfig requires a prompt template."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'template'

    def test_config_requires_model(self) -> None:
        """LLMConfig requires model name."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "template": "Analyze: {{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )  # Missing 'model'

    def test_config_requires_schema(self) -> None:
        """LLMConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(PluginConfigError, match="schema"):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "Analyze: {{ text }}",
                }
            )  # Missing 'schema'

    def test_valid_config(self) -> None:
        """Valid config passes validation."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.model == "gpt-4"
        assert config.template == "Analyze: {{ text }}"

    def test_invalid_template_syntax_rejected(self) -> None:
        """Invalid Jinja2 syntax rejected at config time."""
        with pytest.raises(PluginConfigError, match="Invalid Jinja2 template"):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ unclosed",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_empty_template_rejected(self) -> None:
        """Empty template rejected."""
        with pytest.raises(PluginConfigError, match="cannot be empty"):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "   ",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "Hello, {{ name }}!",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.response_field == "llm_response"
        assert config.on_error is None

    def test_config_custom_values(self) -> None:
        """Config accepts custom values."""
        config = LLMConfig.from_dict(
            {
                "model": "claude-3-opus",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant.",
                "response_field": "analysis_result",
                "on_error": "quarantine_sink",
            }
        )
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.system_prompt == "You are a helpful assistant."
        assert config.response_field == "analysis_result"
        assert config.on_error == "quarantine_sink"

    def test_temperature_bounds(self) -> None:
        """Temperature must be between 0.0 and 2.0."""
        # Lower bound
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ x }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.0,
            }
        )
        assert config.temperature == 0.0

        # Upper bound
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ x }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 2.0,
            }
        )
        assert config.temperature == 2.0

        # Below lower bound
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "temperature": -0.1,
                }
            )

        # Above upper bound
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "temperature": 2.1,
                }
            )

    def test_max_tokens_must_be_positive(self) -> None:
        """max_tokens must be positive if specified."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "max_tokens": 0,
                }
            )

        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ x }}",
                    "schema": DYNAMIC_SCHEMA,
                    "max_tokens": -100,
                }
            )


class TestBaseLLMTransformInit:
    """Tests for BaseLLMTransform initialization."""

    def test_requires_name_attribute(self) -> None:
        """BaseLLMTransform requires name to be set by subclass.

        Note: BaseTransform defines name as a class attribute. Subclasses
        must set it. Without it, accessing self.name will raise AttributeError.
        """

        # BaseLLMTransform can be instantiated but name must be provided
        # by subclass as a class attribute
        class NoNameTransform(BaseLLMTransform):
            pass  # Deliberately missing name

        # Instantiation fails because __init__ accesses self.name for schema naming
        with pytest.raises(AttributeError):
            NoNameTransform(
                {
                    "model": "gpt-4",
                    "template": "{{ text }}",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_concrete_subclass_works(self) -> None:
        """Concrete subclass with name property can be instantiated."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        assert transform.name == "test_llm"
        assert transform._model == "gpt-4"
        assert transform._temperature == 0.0

    def test_determinism_is_non_deterministic(self) -> None:
        """LLM transforms are marked as non-deterministic."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_on_error_set_from_config(self) -> None:
        """on_error is set from config for error routing."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "on_error": "error_sink",
            }
        )

        assert transform._on_error == "error_sink"


class TestBaseLLMTransformProcess:
    """Tests for transform processing."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_template_rendering_error_returns_transform_error(
        self, ctx: PluginContext
    ) -> None:
        """Template rendering failure returns TransformResult.error()."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Hello, {{ required_field }}!",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()

        # Missing required_field should return error
        result = transform.process({"other_field": "value"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "template_rendering_failed"
        assert "template_hash" in result.reason

    def test_llm_client_error_returns_transform_error(self, ctx: PluginContext) -> None:
        """LLM client failure returns TransformResult.error()."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.side_effect = LLMClientError("API Error")

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"
        assert "API Error" in result.reason["error"]
        assert result.retryable is False

    def test_rate_limit_error_is_retryable(self, ctx: PluginContext) -> None:
        """Rate limit errors marked retryable=True."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.side_effect = RateLimitError(
            "Rate limit exceeded"
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "rate_limited"
        assert result.retryable is True

    def test_successful_transform_returns_enriched_row(
        self, ctx: PluginContext
    ) -> None:
        """Successful transform returns row with LLM response."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Analysis result",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=150.0,
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Analysis result"
        assert result.row["llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
        assert "llm_response_template_hash" in result.row
        assert "llm_response_variables_hash" in result.row
        # Original data preserved
        assert result.row["text"] == "hello"

    def test_custom_response_field(self, ctx: PluginContext) -> None:
        """Custom response_field name is used."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "response_field": "analysis",
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Result",
            model="gpt-4",
            usage={},
        )

        result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["analysis"] == "Result"
        assert "analysis_usage" in result.row
        assert "analysis_template_hash" in result.row
        assert "analysis_variables_hash" in result.row

    def test_missing_llm_client_raises_runtime_error(self, ctx: PluginContext) -> None:
        """Missing llm_client in context raises RuntimeError."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = None

        with pytest.raises(RuntimeError, match="LLM client not available"):
            transform.process({"text": "hello"}, ctx)

    def test_system_prompt_included_in_messages(self, ctx: PluginContext) -> None:
        """System prompt is included when configured."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "system_prompt": "You are a helpful assistant.",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="gpt-4",
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

    def test_no_system_prompt_single_message(self, ctx: PluginContext) -> None:
        """Without system prompt, only user message is sent."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="gpt-4",
            usage={},
        )

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.llm_client.chat_completion.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_temperature_and_max_tokens_passed_to_client(
        self, ctx: PluginContext
    ) -> None:
        """Temperature and max_tokens are passed to LLM client."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )

        ctx.llm_client = Mock()
        ctx.llm_client.chat_completion.return_value = LLMResponse(
            content="Response",
            model="gpt-4",
            usage={},
        )

        transform.process({"text": "hello"}, ctx)

        call_args = ctx.llm_client.chat_completion.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["temperature"] == 0.7
        assert call_args.kwargs["max_tokens"] == 500

    def test_retryable_llm_error_propagates_flag(self, ctx: PluginContext) -> None:
        """LLMClientError retryable flag is propagated."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
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

    def test_close_is_noop(self) -> None:
        """close() does nothing but doesn't raise."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Should not raise
        transform.close()


class TestBaseLLMTransformSchemaHandling:
    """Tests for schema configuration handling."""

    def test_schema_created_with_no_coercion(self) -> None:
        """Schema is created with allow_coercion=False (transform behavior)."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": {"mode": "strict", "fields": ["count: int"]},
            }
        )

        # Schema should reject string when int is expected (no coercion)
        with pytest.raises(ValidationError):
            transform.input_schema.model_validate({"count": "not_an_int"})

    def test_dynamic_schema_accepts_any_fields(self) -> None:
        """Dynamic schema accepts any fields."""

        class TestLLMTransform(BaseLLMTransform):
            name = "test_llm"

        transform = TestLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Should accept any data
        validated = transform.input_schema.model_validate(
            {
                "anything": "goes",
                "count": "string",
                "nested": {"data": 123},
            }
        )
        assert validated.anything == "goes"  # type: ignore[attr-defined]
