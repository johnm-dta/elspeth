# tests/plugins/llm/test_tracing_integration.py
"""Integration tests for Tier 2 tracing with mocked endpoints.

These tests verify end-to-end tracing behavior by:
1. Creating transforms with tracing configuration
2. Mocking external SDKs (Langfuse, Azure Monitor)
3. Verifying traces capture complete LLM call information

Note: Tests updated for Langfuse SDK v3 (context manager pattern).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.plugins.llm.azure import AzureLLMTransform
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
from elspeth.plugins.llm.tracing import AzureAITracingConfig, LangfuseTracingConfig


def _make_azure_config(**overrides: Any) -> dict[str, Any]:
    """Create base config for Azure LLM transform."""
    config = {
        "deployment_name": "gpt-4",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Hello {{ row.name }}",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


def _make_openrouter_config(**overrides: Any) -> dict[str, Any]:
    """Create base config for OpenRouter LLM transform."""
    config = {
        "model": "anthropic/claude-3-opus",
        "api_key": "test-key",
        "template": "Hello {{ row.name }}",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


def _make_mock_ctx(run_id: str = "test-run") -> MagicMock:
    """Create a mock PluginContext."""
    ctx = MagicMock()
    ctx.landscape = MagicMock()
    ctx.run_id = run_id
    ctx.telemetry_emit = lambda x: None
    ctx.rate_limit_registry = None
    return ctx


class TestLangfuseIntegration:
    """Integration tests for Langfuse tracing (v3 API)."""

    @pytest.fixture
    def mock_langfuse_client(self) -> MagicMock:
        """Create a mock Langfuse client that captures v3 observations.

        v3 API uses start_as_current_observation() context manager for both
        spans and generations, with update() to record outputs.
        """
        captured_observations: list[dict[str, Any]] = []

        mock_client = MagicMock()

        @contextmanager
        def mock_start_observation(**kwargs: Any):
            """Mock start_as_current_observation context manager."""
            obs = MagicMock()
            obs_record: dict[str, Any] = {"kwargs": kwargs, "updates": []}
            captured_observations.append(obs_record)

            def capture_update(**update_kwargs: Any) -> None:
                obs_record["updates"].append(update_kwargs)

            obs.update = capture_update
            yield obs

        mock_client.start_as_current_observation = mock_start_observation
        mock_client.captured_observations = captured_observations
        mock_client.flush = MagicMock()

        return mock_client

    def test_langfuse_captures_llm_call_end_to_end(self, mock_langfuse_client: MagicMock) -> None:
        """Langfuse captures complete LLM call with prompt, response, and usage."""
        # Setup transform with Langfuse tracing
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = AzureLLMTransform(config)

        # Inject mock Langfuse client and activate tracing
        transform._langfuse_client = mock_langfuse_client
        transform._tracing_active = True

        # Create mock context
        ctx = _make_mock_ctx()

        # Record a trace (v3 pattern - single method call after LLM response)
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="token-123",
            prompt="Hello world",
            response_content="Hi there!",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            latency_ms=150.0,
        )

        # Verify observations were created (span + generation)
        assert len(mock_langfuse_client.captured_observations) == 2

        # First observation is the outer span
        span_kwargs = mock_langfuse_client.captured_observations[0]["kwargs"]
        assert span_kwargs["as_type"] == "span"
        assert span_kwargs["name"] == "elspeth.azure_llm"
        assert span_kwargs["metadata"]["token_id"] == "token-123"
        assert span_kwargs["metadata"]["plugin"] == "azure_llm"
        assert span_kwargs["metadata"]["deployment"] == "gpt-4"

        # Second observation is the generation
        gen_kwargs = mock_langfuse_client.captured_observations[1]["kwargs"]
        assert gen_kwargs["as_type"] == "generation"
        assert gen_kwargs["name"] == "llm_call"
        assert gen_kwargs["model"] == "gpt-4"
        assert gen_kwargs["input"] == [{"role": "user", "content": "Hello world"}]

        # Verify update() was called with output and usage_details
        gen_updates = mock_langfuse_client.captured_observations[1]["updates"]
        assert len(gen_updates) == 1
        assert gen_updates[0]["output"] == "Hi there!"
        assert gen_updates[0]["usage_details"]["input"] == 10
        assert gen_updates[0]["usage_details"]["output"] == 5
        assert gen_updates[0]["metadata"]["latency_ms"] == 150.0

    def test_langfuse_captures_openrouter_call(self, mock_langfuse_client: MagicMock) -> None:
        """Langfuse captures OpenRouter HTTP call."""
        # Setup transform with Langfuse tracing
        config = _make_openrouter_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = OpenRouterLLMTransform(config)

        # Inject mock Langfuse client and activate tracing
        transform._langfuse_client = mock_langfuse_client
        transform._tracing_active = True

        # Create mock context
        ctx = _make_mock_ctx()

        # Record a trace (v3 pattern)
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="token-456",
            prompt="Analyze this",
            response_content="Analysis complete",
            model="anthropic/claude-3-opus",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            latency_ms=250.0,
        )

        # Verify observations were created
        assert len(mock_langfuse_client.captured_observations) == 2

        # Span has OpenRouter-specific metadata
        span_kwargs = mock_langfuse_client.captured_observations[0]["kwargs"]
        assert span_kwargs["name"] == "elspeth.openrouter_llm"
        assert span_kwargs["metadata"]["plugin"] == "openrouter_llm"
        assert span_kwargs["metadata"]["model"] == "anthropic/claude-3-opus"

        # Generation captures OpenRouter response
        gen_kwargs = mock_langfuse_client.captured_observations[1]["kwargs"]
        assert gen_kwargs["model"] == "anthropic/claude-3-opus"
        assert gen_kwargs["input"] == [{"role": "user", "content": "Analyze this"}]

        gen_updates = mock_langfuse_client.captured_observations[1]["updates"]
        assert gen_updates[0]["output"] == "Analysis complete"

    def test_langfuse_client_created_on_start(self) -> None:
        """Langfuse client is created during on_start when config is valid."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://custom.langfuse.com",
            }
        )
        transform = AzureLLMTransform(config)

        # Mock langfuse module
        mock_langfuse_instance = MagicMock()
        mock_langfuse_class = MagicMock(return_value=mock_langfuse_instance)

        mock_module = MagicMock()
        mock_module.Langfuse = mock_langfuse_class

        with patch.dict(sys.modules, {"langfuse": mock_module}):
            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            # Verify client was created with correct parameters (v3 includes tracing_enabled)
            mock_langfuse_class.assert_called_once_with(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://custom.langfuse.com",
                tracing_enabled=True,
            )
            assert transform._tracing_active is True
            assert transform._langfuse_client is mock_langfuse_instance

    def test_langfuse_flush_called_on_close(self) -> None:
        """Langfuse client is flushed when transform closes."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = AzureLLMTransform(config)

        # Setup mock client
        mock_langfuse = MagicMock()
        transform._langfuse_client = mock_langfuse
        transform._tracing_active = True

        # Close the transform
        transform.close()

        # Verify flush was called
        mock_langfuse.flush.assert_called_once()


class TestAzureAIAutoInstrumentation:
    """Tests for Azure AI auto-instrumentation verification."""

    def test_azure_ai_configures_opentelemetry(self) -> None:
        """Azure AI calls configure_azure_monitor with correct parameters."""
        config = _make_azure_config(
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
                "enable_content_recording": True,
                "enable_live_metrics": True,
            }
        )
        transform = AzureLLMTransform(config)

        # Verify parsed config
        assert transform._tracing_config is not None
        assert isinstance(transform._tracing_config, AzureAITracingConfig)
        assert transform._tracing_config.connection_string == "InstrumentationKey=xxx"
        assert transform._tracing_config.enable_content_recording is True
        assert transform._tracing_config.enable_live_metrics is True

        # Mock configure_azure_monitor and OTEL provider check
        with patch("elspeth.plugins.llm.azure._configure_azure_monitor") as mock_configure:
            mock_configure.return_value = True

            with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.__class__.__name__ = "ProxyTracerProvider"
                mock_get_provider.return_value = mock_provider

                ctx = _make_mock_ctx()
                transform.on_start(ctx)

                # Verify configure was called with the tracing config
                mock_configure.assert_called_once()
                call_args = mock_configure.call_args[0][0]
                assert isinstance(call_args, AzureAITracingConfig)
                assert call_args.connection_string == "InstrumentationKey=xxx"
                assert call_args.enable_live_metrics is True

                assert transform._tracing_active is True

    def test_azure_ai_warns_on_existing_otel_provider(self) -> None:
        """Azure AI logs warning when OTEL already configured (Tier 1 conflict)."""
        config = _make_azure_config(
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )
        transform = AzureLLMTransform(config)

        with patch("elspeth.plugins.llm.azure._configure_azure_monitor") as mock_configure:
            mock_configure.return_value = True

            # Simulate existing OTEL provider (not ProxyTracerProvider)
            with patch("opentelemetry.trace.get_tracer_provider") as mock_get_provider:
                mock_provider = MagicMock()
                mock_provider.__class__.__name__ = "TracerProvider"  # Not Proxy
                mock_get_provider.return_value = mock_provider

                with patch("structlog.get_logger") as mock_get_logger:
                    mock_logger = MagicMock()
                    mock_get_logger.return_value = mock_logger

                    ctx = _make_mock_ctx()
                    transform.on_start(ctx)

                    # Verify warning was logged about potential conflict
                    mock_logger.warning.assert_called()
                    call_args = mock_logger.warning.call_args
                    assert "Existing OpenTelemetry tracer detected" in call_args[0][0]

    def test_azure_ai_not_supported_for_openrouter(self) -> None:
        """Azure AI tracing is rejected for OpenRouter with appropriate warning."""
        config = _make_openrouter_config(
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )
        transform = OpenRouterLLMTransform(config)

        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            # Verify warning about unsupported provider
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "Azure AI tracing not supported" in call_args[0][0]
            assert transform._tracing_active is False


class TestGracefulDegradation:
    """Tests for graceful degradation when SDKs are not installed."""

    def test_langfuse_warning_when_not_installed(self) -> None:
        """Warning logged when Langfuse SDK not installed."""
        import builtins

        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = AzureLLMTransform(config)

        # Store the original import function
        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse":
                raise ImportError("No module named 'langfuse'")
            return original_import(name, *args, **kwargs)

        # Ensure langfuse is not already imported
        with (
            patch.dict(sys.modules, {"langfuse": None}),
            patch.object(builtins, "__import__", side_effect=mock_import),
            patch("structlog.get_logger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            # Verify warning was logged
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "package not installed" in call_args[0][0]
            assert transform._tracing_active is False

    def test_azure_ai_warning_when_not_installed(self) -> None:
        """Warning logged when Azure Monitor SDK not installed."""
        config = _make_azure_config(
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )
        transform = AzureLLMTransform(config)

        # Mock OTEL check to pass
        mock_provider = MagicMock()
        mock_provider.__class__.__name__ = "ProxyTracerProvider"

        with (
            patch("opentelemetry.trace.get_tracer_provider", return_value=mock_provider),
            patch(
                "elspeth.plugins.llm.azure._configure_azure_monitor",
                side_effect=ImportError("No module named 'azure.monitor.opentelemetry'"),
            ),
            patch("structlog.get_logger") as mock_get_logger,
        ):
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            # Verify warning was logged about missing package
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert "package not installed" in call_args[0][0]
            assert transform._tracing_active is False

    def test_tracing_inactive_when_config_validation_fails(self) -> None:
        """Tracing not activated when configuration is incomplete."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                # Missing public_key and secret_key
            }
        )
        transform = AzureLLMTransform(config)

        # Verify config was parsed as LangfuseTracingConfig
        assert transform._tracing_config is not None
        assert isinstance(transform._tracing_config, LangfuseTracingConfig)

        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            # Verify warnings about missing keys
            assert mock_logger.warning.call_count >= 2  # At least 2 errors (public_key, secret_key)
            assert transform._tracing_active is False


class TestTracingDisabled:
    """Tests for behavior when tracing is disabled or not configured."""

    def test_no_tracing_when_config_is_none(self) -> None:
        """No tracing setup when tracing config is None."""
        config = _make_azure_config()  # No tracing config
        transform = AzureLLMTransform(config)

        assert transform._tracing_config is None
        assert transform._tracing_active is False

    def test_no_tracing_when_provider_is_none(self) -> None:
        """No tracing setup when provider is 'none'."""
        config = _make_azure_config(tracing={"provider": "none"})
        transform = AzureLLMTransform(config)

        assert transform._tracing_config is not None
        assert transform._tracing_config.provider == "none"

        ctx = _make_mock_ctx()
        transform.on_start(ctx)

        assert transform._tracing_active is False

    def test_record_trace_does_nothing_when_tracing_inactive(self) -> None:
        """_record_langfuse_trace is a no-op when tracing is inactive."""
        config = _make_azure_config()
        transform = AzureLLMTransform(config)

        ctx = _make_mock_ctx()

        # This should not raise any errors (no-op when tracing inactive)
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="test-token",
            prompt="test",
            response_content="response",
            usage=None,
            latency_ms=None,
        )
        # If we get here without error, test passes


class TestTracingMetadata:
    """Tests for tracing metadata completeness (v3 API)."""

    def test_trace_includes_token_id_for_correlation(self) -> None:
        """Trace includes token_id for correlation with Landscape audit trail."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = AzureLLMTransform(config)

        captured_observations: list[dict[str, Any]] = []
        mock_langfuse = MagicMock()

        @contextmanager
        def mock_start_observation(**kwargs: Any):
            obs = MagicMock()
            obs.update = MagicMock()
            captured_observations.append(kwargs)
            yield obs

        mock_langfuse.start_as_current_observation = mock_start_observation

        transform._langfuse_client = mock_langfuse
        transform._tracing_active = True

        ctx = _make_mock_ctx()
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="token-abc-123",
            prompt="test",
            response_content="response",
            usage=None,
            latency_ms=None,
        )

        # First observation is the span, which has token_id in metadata
        assert len(captured_observations) >= 1
        assert captured_observations[0]["metadata"]["token_id"] == "token-abc-123"

    def test_generation_includes_usage_metrics(self) -> None:
        """Generation includes token usage for cost tracking (v3: usage_details)."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = AzureLLMTransform(config)

        captured_updates: list[dict[str, Any]] = []

        @contextmanager
        def mock_start_observation(**kwargs: Any):
            obs = MagicMock()
            obs.update = lambda **uk: captured_updates.append(uk)
            yield obs

        mock_langfuse = MagicMock()
        mock_langfuse.start_as_current_observation = mock_start_observation

        transform._langfuse_client = mock_langfuse
        transform._tracing_active = True

        ctx = _make_mock_ctx()
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="test-token",
            prompt="test prompt",
            response_content="test response",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500.0,
        )

        # Find the update call with usage_details (from the generation observation)
        usage_update = next((u for u in captured_updates if "usage_details" in u), None)
        assert usage_update is not None
        assert usage_update["usage_details"]["input"] == 100
        assert usage_update["usage_details"]["output"] == 50

    def test_generation_includes_latency(self) -> None:
        """Generation includes latency for performance monitoring."""
        config = _make_openrouter_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = OpenRouterLLMTransform(config)

        captured_updates: list[dict[str, Any]] = []

        @contextmanager
        def mock_start_observation(**kwargs: Any):
            obs = MagicMock()
            obs.update = lambda **uk: captured_updates.append(uk)
            yield obs

        mock_langfuse = MagicMock()
        mock_langfuse.start_as_current_observation = mock_start_observation

        transform._langfuse_client = mock_langfuse
        transform._tracing_active = True

        ctx = _make_mock_ctx()
        transform._record_langfuse_trace(
            ctx=ctx,
            token_id="test-token",
            prompt="test",
            response_content="response",
            model="anthropic/claude-3-opus",
            usage=None,
            latency_ms=1234.5,
        )

        # Find the update call with metadata (from the generation observation)
        metadata_update = next((u for u in captured_updates if "metadata" in u), None)
        assert metadata_update is not None
        assert metadata_update["metadata"]["latency_ms"] == 1234.5


class TestTracingProviderValidation:
    """Tests for explicit tracing provider validation behavior."""

    def test_unknown_provider_logs_validation_error(self) -> None:
        """Unknown providers emit a visible validation warning and disable tracing."""
        config = _make_openrouter_config(
            tracing={
                "provider": "langfusee",
                "public_key": "pk-test",
                "secret_key": "sk-test",
            }
        )
        transform = OpenRouterLLMTransform(config)

        with patch("structlog.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            ctx = _make_mock_ctx()
            transform.on_start(ctx)

            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args
            assert warning_call.args[0] == "Tracing configuration error"
            assert "Unknown tracing provider" in warning_call.kwargs["error"]
            assert transform._tracing_active is False
