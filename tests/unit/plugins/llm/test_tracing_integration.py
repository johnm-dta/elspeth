# tests/plugins/llm/test_tracing_integration.py
"""Integration tests for Tier 2 tracing with mocked endpoints.

These tests verify end-to-end tracing behavior by:
1. Creating transforms with tracing configuration
2. Mocking external SDKs (Langfuse)
3. Verifying traces capture complete LLM call information

Note: Tests updated for unified LLMTransform and Langfuse SDK v3
(context manager pattern).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.transforms.llm.langfuse import ActiveLangfuseTracer, NoOpLangfuseTracer
from elspeth.plugins.transforms.llm.provider import LLMQueryResult
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.testing import make_pipeline_row

# A valid OpenRouter catalog model id (the retired anthropic/claude-3-opus was
# dropped from the litellm-derived catalog; OpenRouterConfig now rejects models
# absent from it). Mirrors test_openrouter.py.
_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"


def _make_azure_config(**overrides: Any) -> dict[str, Any]:
    """Create base config for Azure LLM transform."""
    config: dict[str, Any] = {
        "provider": "azure",
        "deployment_name": "gpt-4",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "prompt_template": "Hello {{ row.name }}",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


def _make_openrouter_config(**overrides: Any) -> dict[str, Any]:
    """Create base config for OpenRouter LLM transform."""
    config: dict[str, Any] = {
        "provider": "openrouter",
        "model": _OPENROUTER_MODEL,
        "api_key": "test-key",
        "prompt_template": "Hello {{ row.name }}",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
    }
    config.update(overrides)
    return config


def _make_multi_query_config(**overrides: Any) -> dict[str, Any]:
    """Create base config for LLMTransform with multi-query (OpenRouter provider)."""
    config: dict[str, Any] = {
        "provider": "openrouter",
        "model": _OPENROUTER_MODEL,
        "api_key": "test-key",
        "prompt_template": "Case: {{ row.field1 }} Criterion: {{ row.criterion_name }}",
        "schema": {"mode": "observed"},
        "required_input_fields": [],
        "queries": {
            "cs1_crit1": {
                "input_fields": {"field1": "field1"},
                "output_fields": [{"suffix": "score", "type": "integer"}],
            },
        },
    }
    config.update(overrides)
    return config


class _LandscapeSentinel:
    pass


class _TracingContext:
    """Minimal transform context for LLMTransform._process_row tests."""

    def __init__(self, run_id: str = "test-run") -> None:
        self.landscape = _LandscapeSentinel()
        self.run_id = run_id
        self.telemetry_emit = lambda x: None
        self.rate_limit_registry = None
        self.state_id = "state-123"
        self.token: TokenInfo | None = None


class _RecordingObservation:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def update(self, **update_kwargs: Any) -> None:
        self._record["updates"].append(update_kwargs)


class _RecordingLangfuseClient:
    def __init__(self) -> None:
        self.captured_observations: list[dict[str, Any]] = []
        self.flush_count = 0

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        record: dict[str, Any] = {"kwargs": kwargs, "updates": []}
        self.captured_observations.append(record)
        yield _RecordingObservation(record)

    def flush(self) -> None:
        self.flush_count += 1


class _ErroringProvider:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def execute_query(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        state_id: str,
        token_id: str,
        response_format: dict[str, Any] | None = None,
    ) -> LLMQueryResult:
        del messages, model, temperature, max_tokens, state_id, token_id, response_format
        raise self.error

    def runtime_preflight(self, *, operation_id: str, model: str) -> None:
        del operation_id, model

    def close(self) -> None:
        return None


def _make_ctx(run_id: str = "test-run") -> _TracingContext:
    """Create a minimal PluginContext-like object."""
    return _TracingContext(run_id)


def _make_token(token_id: str = "test-token-id") -> TokenInfo:
    return TokenInfo(row_id="row-1", token_id=token_id, row_data=make_pipeline_row({}))


class TestLangfuseIntegration:
    """Integration tests for Langfuse tracing (v3 API)."""

    @pytest.fixture
    def mock_langfuse_client(self) -> _RecordingLangfuseClient:
        """Create a Langfuse client double that captures v3 observations.

        v3 API uses start_as_current_observation() context manager for both
        spans and generations, with update() to record outputs.
        """
        return _RecordingLangfuseClient()

    def test_langfuse_captures_llm_call_end_to_end(self, mock_langfuse_client: _RecordingLangfuseClient) -> None:
        """Langfuse captures complete LLM call with prompt, response, and usage."""
        # Setup transform with Langfuse tracing
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://langfuse.example.com",
            }
        )
        transform = LLMTransform(config)

        # Inject mock Langfuse client via ActiveLangfuseTracer
        transform._tracer = ActiveLangfuseTracer(
            transform_name=transform.name,
            client=mock_langfuse_client,
        )

        # Record a trace via tracer
        transform._tracer.record_success(
            token_id="token-123",
            query_name=transform.name,
            prompt="Hello world",
            response_content="Hi there!",
            model="gpt-4",
            usage=TokenUsage.known(10, 5),
            latency_ms=150.0,
            extra_metadata={"deployment": "gpt-4"},
        )

        # Verify observations were created (span + generation)
        assert len(mock_langfuse_client.captured_observations) == 2

        # First observation is the outer span
        span_kwargs = mock_langfuse_client.captured_observations[0]["kwargs"]
        assert span_kwargs["as_type"] == "span"
        assert span_kwargs["name"] == "elspeth.llm"
        assert span_kwargs["metadata"]["token_id"] == "token-123"
        assert span_kwargs["metadata"]["plugin"] == "llm"
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

    def test_langfuse_tracer_created_at_init(self) -> None:
        """ActiveLangfuseTracer is created at __init__ time when Langfuse config is valid."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://custom.langfuse.com",
            }
        )
        # Langfuse is installed in test env, so factory returns ActiveLangfuseTracer
        transform = LLMTransform(config)

        assert isinstance(transform._tracer, ActiveLangfuseTracer)
        assert transform._tracer.transform_name == "llm"

    def test_langfuse_flush_called_on_close(self) -> None:
        """Langfuse client is flushed when transform closes."""
        config = _make_azure_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://langfuse.example.com",
            }
        )
        transform = LLMTransform(config)

        # Setup mock tracer
        mock_langfuse = _RecordingLangfuseClient()
        transform._tracer = ActiveLangfuseTracer(
            transform_name=transform.name,
            client=mock_langfuse,
        )

        # Close the transform
        transform.close()

        # Verify flush was called
        assert mock_langfuse.flush_count == 1


class TestGracefulDegradation:
    """Tests for graceful degradation when SDKs are not installed."""

    def test_langfuse_raises_when_not_installed(self) -> None:
        """RuntimeError raised when Langfuse SDK not installed but configured."""
        import builtins

        # Store the original import function
        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse":
                raise ImportError("No module named 'langfuse'")
            return original_import(name, *args, **kwargs)

        # Missing Langfuse package with explicit config is a startup error —
        # the user has a reasonable expectation that configured tracing is active.
        with (
            patch.dict(sys.modules, {"langfuse": None}),
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(RuntimeError, match=r"langfuse.*not installed"),
        ):
            config = _make_azure_config(
                tracing={
                    "provider": "langfuse",
                    "public_key": "pk-test",
                    "secret_key": "sk-test",
                    "host": "https://langfuse.example.com",
                }
            )
            LLMTransform(config)

    def test_langfuse_raises_when_not_installed_regardless_of_config(self) -> None:
        """RuntimeError when Langfuse package is missing, even with incomplete config.

        Even with incomplete config (missing keys), the user explicitly asked
        for langfuse tracing. Missing package is a startup error.
        """
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse":
                raise ImportError("No module named 'langfuse'")
            return original_import(name, *args, **kwargs)

        with (
            patch.dict(sys.modules, {"langfuse": None}),
            patch.object(builtins, "__import__", side_effect=mock_import),
            pytest.raises(RuntimeError, match=r"langfuse.*not installed"),
        ):
            config = _make_azure_config(
                tracing={
                    "provider": "langfuse",
                    "public_key": "pk-test",
                    "secret_key": "sk-test",
                    "host": "https://langfuse.example.com",
                }
            )
            LLMTransform(config)


class TestTracingDisabled:
    """Tests for behavior when tracing is disabled or not configured."""

    def test_no_tracing_when_config_is_none(self) -> None:
        """NoOpLangfuseTracer when tracing config is None."""
        config = _make_azure_config()  # No tracing config
        transform = LLMTransform(config)

        assert isinstance(transform._tracer, NoOpLangfuseTracer)

    def test_no_tracing_when_provider_is_none(self) -> None:
        """NoOpLangfuseTracer when provider is 'none'."""
        config = _make_azure_config(tracing={"provider": "none"})
        transform = LLMTransform(config)

        # parse_tracing_config returns TracingConfig(provider="none"), which is
        # not LangfuseTracingConfig, so create_langfuse_tracer returns NoOp.
        assert isinstance(transform._tracer, NoOpLangfuseTracer)

    def test_record_trace_does_nothing_when_tracing_inactive(self) -> None:
        """NoOpLangfuseTracer.record_success is a no-op when tracing is not configured."""
        config = _make_azure_config()
        transform = LLMTransform(config)

        assert isinstance(transform._tracer, NoOpLangfuseTracer)

        # This should not raise any errors (no-op tracer)
        transform._tracer.record_success(
            token_id="test-token",
            query_name=transform.name,
            prompt="test",
            response_content="response",
            model="gpt-4",
            usage=None,
            latency_ms=None,
        )
        # If we get here without error, test passes

    def test_azure_ai_tracing_results_in_noop(self) -> None:
        """Azure AI tracing config results in NoOpLangfuseTracer.

        Azure AI tracing is handled via _configure_azure_monitor() in on_start(),
        not through the Langfuse tracer. The Langfuse factory returns
        NoOpLangfuseTracer for AzureAITracingConfig since it's not a Langfuse config.
        """
        config = _make_azure_config(
            tracing={
                "provider": "azure_ai",
                "connection_string": "InstrumentationKey=xxx",
            }
        )
        transform = LLMTransform(config)

        assert isinstance(transform._tracer, NoOpLangfuseTracer)


class TestProcessRowErrorTracing:
    """Tests for Langfuse tracing of failed LLM calls at _process_row level.

    Verifies that _process_row records Langfuse traces for failed LLM calls,
    including both non-retryable errors (returned as error results) and
    retryable errors (re-raised after recording the trace).
    """

    def _create_transform_with_langfuse(self) -> tuple[LLMTransform, _RecordingLangfuseClient, list[dict[str, Any]]]:
        """Create transform with a recording Langfuse client via LangfuseTracer."""
        config = _make_azure_config()
        transform = LLMTransform(config)

        mock_langfuse = _RecordingLangfuseClient()

        transform._tracer = ActiveLangfuseTracer(
            transform_name=transform.name,
            client=mock_langfuse,
        )

        return transform, mock_langfuse, mock_langfuse.captured_observations

    def test_process_row_records_error_trace_on_llm_failure(self) -> None:
        """_process_row records Langfuse trace when LLM call fails."""
        from elspeth.plugins.infrastructure.clients.llm import LLMClientError

        transform, _mock_langfuse, captured_observations = self._create_transform_with_langfuse()

        transform._provider = _ErroringProvider(LLMClientError("Content policy violation", retryable=False))

        ctx = _make_ctx()
        ctx.state_id = "test-state"
        ctx.token = _make_token()

        result = transform._process_row(make_pipeline_row({"name": "test"}), ctx)

        # Should return error result
        assert result.status == "error"
        assert result.reason is not None and result.reason["reason"] == "llm_call_failed"

        # And also have recorded the error trace in Langfuse
        assert len(captured_observations) == 2  # span + generation
        gen_record = captured_observations[1]
        assert gen_record["kwargs"]["as_type"] == "generation"
        assert len(gen_record["updates"]) == 1
        assert gen_record["updates"][0]["level"] == "ERROR"
        assert "Content policy violation" in gen_record["updates"][0]["status_message"]

    def test_process_row_records_error_trace_on_retryable_failure(self) -> None:
        """_process_row records Langfuse trace even for retryable errors before re-raising."""
        from elspeth.plugins.infrastructure.clients.llm import LLMClientError

        transform, _mock_langfuse, captured_observations = self._create_transform_with_langfuse()

        transform._provider = _ErroringProvider(LLMClientError("Rate limit exceeded", retryable=True))

        ctx = _make_ctx()
        ctx.state_id = "test-state"
        ctx.token = _make_token()

        try:
            transform._process_row(make_pipeline_row({"name": "test"}), ctx)
            raise AssertionError("Should have raised LLMClientError")
        except LLMClientError:
            pass  # Expected

        # Error trace should still have been recorded before re-raising
        assert len(captured_observations) == 2  # span + generation
        gen_record = captured_observations[1]
        assert gen_record["updates"][0]["level"] == "ERROR"
        assert "Rate limit exceeded" in gen_record["updates"][0]["status_message"]


class TestTracingProviderValidation:
    """Tests for explicit tracing provider validation behavior."""

    def test_unknown_provider_raises_at_config_time(self) -> None:
        """Unknown tracing providers raise ValueError during config parsing (fail-fast).

        parse_tracing_config raises ValueError for unknown providers, which
        propagates through LLMTransform.__init__.
        """
        config = _make_openrouter_config(
            tracing={
                "provider": "langfusee",
                "public_key": "pk-test",
                "secret_key": "sk-test",
                "host": "https://langfuse.example.com",
            }
        )
        with pytest.raises(ValueError, match="Unknown tracing provider"):
            LLMTransform(config)


class TestAzureAITracingRejection:
    """Tests for azure_ai tracing rejection on non-Azure providers."""

    def test_azure_ai_tracing_rejected_for_openrouter(self) -> None:
        """Azure AI tracing with openrouter provider raises ValueError at init.

        Azure Monitor auto-instruments the OpenAI SDK, which only the Azure
        provider uses. OpenRouter uses httpx directly, so azure_ai tracing
        would silently do nothing.
        """
        with pytest.raises(ValueError, match=r"azure_ai tracing.*azure provider"):
            config = _make_openrouter_config(
                tracing={
                    "provider": "azure_ai",
                    "connection_string": "InstrumentationKey=xxx",
                }
            )
            LLMTransform(config)

    def test_azure_ai_tracing_rejected_for_openrouter_multi_query(self) -> None:
        """Azure AI tracing with openrouter multi-query raises ValueError at init."""
        with pytest.raises(ValueError, match=r"azure_ai tracing.*azure provider"):
            config = _make_multi_query_config(
                tracing={
                    "provider": "azure_ai",
                    "connection_string": "InstrumentationKey=xxx",
                }
            )
            LLMTransform(config)


class TestOpenRouterMissingTracingKeys:
    """Tests for OpenRouter tracing with missing Langfuse keys."""

    def test_tracing_config_rejects_missing_keys_at_construction(self) -> None:
        """Langfuse config crashes at construction without required keys.

        Construction-time enforcement prevents invalid configs from reaching
        the SDK — fails fast with clear error rather than deferred auth failure.
        """
        config = _make_openrouter_config(
            tracing={
                "provider": "langfuse",
                "host": "https://langfuse.example.com",
                # Missing public_key and secret_key
            }
        )
        with pytest.raises(ValueError, match="public_key"):
            LLMTransform(config)


class TestMultiQueryLangfuseTracingViaStrategy:
    """Tests for Langfuse tracing in multi-query transforms via strategy execution.

    In the unified LLMTransform, multi-query tracing happens per-query inside
    MultiQueryStrategy.execute() via tracer.record_success/record_error.
    These tests verify the tracer is correctly wired through the strategy path.
    """

    def _create_multi_query_transform_with_langfuse(
        self,
    ) -> tuple[LLMTransform, _RecordingLangfuseClient, list[dict[str, Any]]]:
        """Create multi-query LLMTransform with a recording Langfuse client."""
        config = _make_multi_query_config(
            tracing={
                "provider": "langfuse",
                "public_key": "pk-xxx",
                "secret_key": "sk-xxx",
                "host": "https://langfuse.example.com",
            }
        )
        transform = LLMTransform(config)

        mock_langfuse = _RecordingLangfuseClient()

        transform._tracer = ActiveLangfuseTracer(
            transform_name=transform.name,
            client=mock_langfuse,
        )

        return transform, mock_langfuse, mock_langfuse.captured_observations

    def test_multi_query_tracer_is_active_with_langfuse_config(self) -> None:
        """Multi-query LLMTransform has ActiveLangfuseTracer when Langfuse configured."""
        transform, _mock_langfuse, _captured = self._create_multi_query_transform_with_langfuse()
        assert isinstance(transform._tracer, ActiveLangfuseTracer)

    def test_multi_query_tracer_records_per_query_success(self) -> None:
        """Tracer records success per-query during multi-query execution."""
        transform, _mock_langfuse, captured_observations = self._create_multi_query_transform_with_langfuse()

        # Simulate what MultiQueryStrategy.execute() does for each query:
        # it calls tracer.record_success after each successful LLM call
        transform._tracer.record_success(
            token_id="test-token",
            query_name="cs1_crit1",
            prompt="Case: data Criterion: criterion_name",
            response_content='{"score": 5}',
            model=_OPENROUTER_MODEL,
            usage=TokenUsage.known(100, 50),
            latency_ms=500.0,
        )

        # Verify observations were created (span + generation)
        assert len(captured_observations) == 2
        span_record = captured_observations[0]
        assert span_record["kwargs"]["as_type"] == "span"
        assert span_record["kwargs"]["metadata"]["query"] == "cs1_crit1"

        gen_record = captured_observations[1]
        assert gen_record["kwargs"]["as_type"] == "generation"
        assert gen_record["kwargs"]["model"] == _OPENROUTER_MODEL

        # Check update() recorded output and usage
        assert len(gen_record["updates"]) == 1
        assert gen_record["updates"][0]["output"] == '{"score": 5}'
        assert gen_record["updates"][0]["usage_details"]["input"] == 100
        assert gen_record["updates"][0]["usage_details"]["output"] == 50

    def test_multi_query_tracer_records_per_query_error(self) -> None:
        """Tracer records error per-query during multi-query execution."""
        transform, _mock_langfuse, captured_observations = self._create_multi_query_transform_with_langfuse()

        # Simulate what MultiQueryStrategy.execute() does on query failure
        transform._tracer.record_error(
            token_id="test-token",
            query_name="cs1_crit1",
            prompt="Case: data Criterion: criterion_name",
            error_message="Rate limit exceeded",
            model=_OPENROUTER_MODEL,
            latency_ms=50.0,
        )

        # Verify error observations were created (span + generation)
        assert len(captured_observations) == 2
        gen_record = captured_observations[1]
        assert gen_record["kwargs"]["as_type"] == "generation"
        assert len(gen_record["updates"]) == 1
        assert gen_record["updates"][0]["level"] == "ERROR"
        assert "Rate limit exceeded" in gen_record["updates"][0]["status_message"]
