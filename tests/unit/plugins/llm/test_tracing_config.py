# tests/plugins/llm/test_tracing_config.py
"""Tests for Tier 2 tracing configuration models."""

from elspeth.plugins.llm.tracing import (
    AzureAITracingConfig,
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)


class TestTracingConfigParsing:
    """Tests for parse_tracing_config function."""

    def test_none_config_returns_none(self) -> None:
        """None input returns None."""
        result = parse_tracing_config(None)
        assert result is None

    def test_empty_dict_returns_base_config(self) -> None:
        """Empty dict returns base TracingConfig with provider='none'."""
        result = parse_tracing_config({})
        assert isinstance(result, TracingConfig)
        assert result.provider == "none"

    def test_azure_ai_provider_returns_azure_config(self) -> None:
        """Provider 'azure_ai' returns AzureAITracingConfig."""
        config = {
            "provider": "azure_ai",
            "connection_string": "InstrumentationKey=xxx",
            "enable_content_recording": True,
            "enable_live_metrics": False,
        }
        result = parse_tracing_config(config)

        assert isinstance(result, AzureAITracingConfig)
        assert result.provider == "azure_ai"
        assert result.connection_string == "InstrumentationKey=xxx"
        assert result.enable_content_recording is True
        assert result.enable_live_metrics is False

    def test_azure_ai_defaults(self) -> None:
        """AzureAITracingConfig has sensible defaults."""
        config = {"provider": "azure_ai"}
        result = parse_tracing_config(config)

        assert isinstance(result, AzureAITracingConfig)
        assert result.connection_string is None
        assert result.enable_content_recording is True  # Default: capture prompts
        assert result.enable_live_metrics is False  # Default: off

    def test_langfuse_provider_returns_langfuse_config(self) -> None:
        """Provider 'langfuse' returns LangfuseTracingConfig."""
        config = {
            "provider": "langfuse",
            "public_key": "pk-xxx",
            "secret_key": "sk-xxx",
            "host": "https://self-hosted.example.com",
        }
        result = parse_tracing_config(config)

        assert isinstance(result, LangfuseTracingConfig)
        assert result.provider == "langfuse"
        assert result.public_key == "pk-xxx"
        assert result.secret_key == "sk-xxx"
        assert result.host == "https://self-hosted.example.com"

    def test_langfuse_defaults(self) -> None:
        """LangfuseTracingConfig has sensible defaults."""
        config = {"provider": "langfuse"}
        result = parse_tracing_config(config)

        assert isinstance(result, LangfuseTracingConfig)
        assert result.public_key is None
        assert result.secret_key is None
        assert result.host == "https://cloud.langfuse.com"  # Default: cloud
        assert result.tracing_enabled is True  # v3: default enabled

    def test_langfuse_tracing_enabled_field(self) -> None:
        """LangfuseTracingConfig supports tracing_enabled field (v3)."""
        config = {
            "provider": "langfuse",
            "public_key": "pk-xxx",
            "secret_key": "sk-xxx",
            "tracing_enabled": False,  # Explicitly disabled
        }
        result = parse_tracing_config(config)

        assert isinstance(result, LangfuseTracingConfig)
        assert result.tracing_enabled is False

    def test_unknown_provider_returns_base_config(self) -> None:
        """Unknown provider returns base TracingConfig."""
        config = {"provider": "unknown_provider"}
        result = parse_tracing_config(config)

        assert isinstance(result, TracingConfig)
        assert result.provider == "unknown_provider"

    def test_none_provider_returns_base_config(self) -> None:
        """Provider 'none' returns base TracingConfig."""
        config = {"provider": "none"}
        result = parse_tracing_config(config)

        assert isinstance(result, TracingConfig)
        assert result.provider == "none"


class TestTracingConfigValidation:
    """Tests for validate_tracing_config function."""

    def test_azure_ai_without_connection_string_returns_error(self) -> None:
        """Azure AI without connection_string returns validation error."""
        config = AzureAITracingConfig(connection_string=None)
        errors = validate_tracing_config(config)
        assert len(errors) == 1
        assert "connection_string" in errors[0]

    def test_azure_ai_with_connection_string_returns_no_errors(self) -> None:
        """Azure AI with connection_string returns no errors."""
        config = AzureAITracingConfig(connection_string="InstrumentationKey=xxx")
        errors = validate_tracing_config(config)
        assert len(errors) == 0

    def test_langfuse_without_keys_returns_error(self) -> None:
        """Langfuse without public_key and secret_key returns validation error."""
        config = LangfuseTracingConfig(public_key=None, secret_key=None)
        errors = validate_tracing_config(config)
        assert len(errors) == 2
        assert any("public_key" in e for e in errors)
        assert any("secret_key" in e for e in errors)

    def test_langfuse_with_keys_returns_no_errors(self) -> None:
        """Langfuse with keys returns no errors."""
        config = LangfuseTracingConfig(public_key="pk-xxx", secret_key="sk-xxx")
        errors = validate_tracing_config(config)
        assert len(errors) == 0

    def test_none_provider_returns_no_errors(self) -> None:
        """Provider 'none' always valid."""
        config = TracingConfig(provider="none")
        errors = validate_tracing_config(config)
        assert len(errors) == 0

    def test_unknown_provider_returns_validation_error(self) -> None:
        """Unknown provider returns explicit validation error."""
        config = TracingConfig(provider="unknown_provider")
        errors = validate_tracing_config(config)
        assert len(errors) == 1
        assert "Unknown tracing provider" in errors[0]
        assert "unknown_provider" in errors[0]
