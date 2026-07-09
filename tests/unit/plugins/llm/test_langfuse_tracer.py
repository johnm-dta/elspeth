# tests/unit/plugins/llm/test_langfuse_tracer.py
"""Tests for LangfuseTracer extraction.

Verifies the extracted Langfuse tracing utilities work correctly:
- Factory returns correct tracer type based on config
- Active tracer records success/error with correct metadata
- No-op tracer is silent
- Failures are logged via structlog (No Silent Failures)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace, TracebackType
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.transforms.llm.langfuse import (
    ActiveLangfuseTracer,
    LangfuseTracer,
    NoOpLangfuseTracer,
    create_langfuse_tracer,
)
from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig, LangfuseTracingConfig


@dataclass
class FakeGeneration:
    update_calls: list[dict[str, Any]] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        self.update_calls.append(kwargs)


@dataclass
class RecordingObservation:
    value: Any
    enter_calls: int = 0
    exit_calls: int = 0

    def __enter__(self) -> Any:
        self.enter_calls += 1
        return self.value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.exit_calls += 1
        return False


@dataclass
class FakeLangfuseClient:
    generation: FakeGeneration = field(default_factory=FakeGeneration)
    observation_calls: list[dict[str, Any]] = field(default_factory=list)
    start_error: Exception | None = None
    flush_error: Exception | None = None
    flush_calls: int = 0

    def start_as_current_observation(self, **kwargs: Any) -> RecordingObservation:
        if self.start_error is not None:
            raise self.start_error
        self.observation_calls.append(kwargs)
        value = self.generation if kwargs.get("as_type") == "generation" else object()
        return RecordingObservation(value=value)

    def flush(self) -> None:
        self.flush_calls += 1
        if self.flush_error is not None:
            raise self.flush_error


@dataclass
class FakeLangfuseSDK:
    public_key: str
    secret_key: str
    host: str
    tracing_enabled: bool


# ── Factory tests ──────────────────────────────────────────────────


class TestCreateLangfuseTracer:
    """Tests for the create_langfuse_tracer factory."""

    def test_create_with_none_config_returns_noop(self) -> None:
        tracer = create_langfuse_tracer(
            transform_name="test_transform",
            tracing_config=None,
        )
        assert isinstance(tracer, NoOpLangfuseTracer)

    def test_create_with_non_langfuse_config_returns_noop(self) -> None:
        config = AzureAITracingConfig(connection_string="test")
        tracer = create_langfuse_tracer(
            transform_name="test_transform",
            tracing_config=config,
        )
        assert isinstance(tracer, NoOpLangfuseTracer)

    @patch.dict("sys.modules", {"langfuse": SimpleNamespace(Langfuse=FakeLangfuseSDK)})
    def test_create_with_langfuse_config_returns_active_tracer(self) -> None:
        # Must import inside patched context so the langfuse import resolves
        from elspeth.plugins.transforms.llm.langfuse import (
            ActiveLangfuseTracer as PatchedActiveTracer,
        )
        from elspeth.plugins.transforms.llm.langfuse import (
            create_langfuse_tracer as patched_create,
        )

        config = LangfuseTracingConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://test.langfuse.com",
        )
        tracer = patched_create(
            transform_name="test_transform",
            tracing_config=config,
        )
        assert isinstance(tracer, PatchedActiveTracer)
        assert tracer.transform_name == "test_transform"

    def test_create_langfuse_not_installed_raises_runtime_error(self) -> None:
        import builtins

        config = LangfuseTracingConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://test.langfuse.com",
        )

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "langfuse":
                raise ImportError("No module named 'langfuse'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", new=mock_import),
            pytest.raises(RuntimeError, match=r"langfuse.*not installed"),
        ):
            create_langfuse_tracer(
                transform_name="test_transform",
                tracing_config=config,
            )


# ── NoOp tracer tests ─────────────────────────────────────────────


class TestNoOpLangfuseTracer:
    """Tests for the no-op tracer implementation."""

    def test_noop_tracer_record_success_is_silent(self) -> None:
        tracer = NoOpLangfuseTracer()
        # Should not raise
        tracer.record_success(
            token_id="tok-1",
            query_name="test",
            prompt="hello",
            response_content="world",
            model="gpt-4",
            usage=TokenUsage.known(10, 20),
        )

    def test_noop_tracer_record_error_is_silent(self) -> None:
        tracer = NoOpLangfuseTracer()
        # Should not raise
        tracer.record_error(
            token_id="tok-1",
            query_name="test",
            prompt="hello",
            error_message="something failed",
            model="gpt-4",
        )

    def test_flush_when_noop_is_silent(self) -> None:
        tracer = NoOpLangfuseTracer()
        # Should not raise
        tracer.flush()

    def test_noop_tracer_matches_protocol_signature(self) -> None:
        """Verify NoOpLangfuseTracer has explicit parameter signatures.

        This ensures mypy can catch signature drift between Protocol and
        implementations — no *args/**kwargs escape hatches.
        """
        import inspect

        for method_name in ("record_success", "record_error", "flush"):
            protocol_sig = inspect.signature(getattr(LangfuseTracer, method_name))
            impl_sig = inspect.signature(getattr(NoOpLangfuseTracer, method_name))

            # Parameter names and kinds must match (ignoring self)
            protocol_params = [(name, p.kind, p.default) for name, p in protocol_sig.parameters.items() if name != "self"]
            impl_params = [(name, p.kind, p.default) for name, p in impl_sig.parameters.items() if name != "self"]

            assert protocol_params == impl_params, (
                f"NoOpLangfuseTracer.{method_name} signature drifted from Protocol: Protocol={protocol_params}, Impl={impl_params}"
            )


# ── Active tracer tests ───────────────────────────────────────────


class TestActiveLangfuseTracer:
    """Tests for the active Langfuse tracer."""

    def _make_tracer(self) -> tuple[ActiveLangfuseTracer, FakeLangfuseClient]:
        """Create an ActiveLangfuseTracer with a recording Langfuse client."""
        client = FakeLangfuseClient()
        tracer = ActiveLangfuseTracer(transform_name="test_transform", client=client)
        return tracer, client

    def test_record_success_creates_span_and_generation(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_success(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            response_content="positive",
            model="gpt-4",
        )

        assert [call["as_type"] for call in client.observation_calls] == ["span", "generation"]
        assert len(client.generation.update_calls) == 1
        call_kwargs = client.generation.update_calls[0]
        assert call_kwargs["output"] == "positive"

    def test_record_success_with_usage_updates_generation(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_success(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            response_content="positive",
            model="gpt-4",
            usage=TokenUsage.known(10, 20),
        )

        call_kwargs = client.generation.update_calls[0]
        assert call_kwargs["usage_details"] == {"input": 10, "output": 20}

    def test_record_success_without_usage_skips_usage_details(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_success(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            response_content="positive",
            model="gpt-4",
            usage=None,
        )

        call_kwargs = client.generation.update_calls[0]
        assert "usage_details" not in call_kwargs

    def test_record_success_with_latency_includes_metadata(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_success(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            response_content="positive",
            model="gpt-4",
            latency_ms=42.5,
        )

        call_kwargs = client.generation.update_calls[0]
        assert call_kwargs["metadata"] == {"latency_ms": 42.5}

    def test_record_success_with_extra_metadata_merges(self) -> None:
        """Verify extra_metadata is merged into span metadata."""
        client = FakeLangfuseClient()
        tracer = ActiveLangfuseTracer(transform_name="test_transform", client=client)

        tracer.record_success(
            token_id="tok-1",
            query_name="classify",
            prompt="test",
            response_content="result",
            model="gpt-4",
            extra_metadata={"deployment": "prod-east"},
        )

        # The first call to start_as_current_observation creates the span
        span_call_kwargs = client.observation_calls[0]
        assert span_call_kwargs["metadata"]["deployment"] == "prod-east"
        assert span_call_kwargs["metadata"]["token_id"] == "tok-1"

    def test_record_error_sets_error_level(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_error(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            error_message="rate limited",
            model="gpt-4",
        )

        call_kwargs = client.generation.update_calls[0]
        assert call_kwargs["level"] == "ERROR"
        assert call_kwargs["status_message"] == "rate limited"

    def test_record_error_with_latency_includes_metadata(self) -> None:
        tracer, client = self._make_tracer()

        tracer.record_error(
            token_id="tok-1",
            query_name="classify",
            prompt="Classify this",
            error_message="timeout",
            model="gpt-4",
            latency_ms=5000.0,
        )

        call_kwargs = client.generation.update_calls[0]
        assert call_kwargs["metadata"] == {"latency_ms": 5000.0}

    def test_record_exception_logs_warning(self) -> None:
        """Tracing failures go to structlog only — No Silent Failures."""
        client = FakeLangfuseClient(start_error=RuntimeError("Langfuse down"))

        tracer = ActiveLangfuseTracer(transform_name="test_transform", client=client)

        with patch("elspeth.plugins.transforms.llm.langfuse._handle_trace_failure", autospec=True) as mock_handler:
            tracer.record_success(
                token_id="tok-1",
                query_name="classify",
                prompt="test",
                response_content="result",
                model="gpt-4",
            )
            mock_handler.assert_called_once()
            assert mock_handler.call_args[0][0] == "langfuse_trace_failed"
            assert mock_handler.call_args[0][1] == "test_transform"
            assert isinstance(mock_handler.call_args[0][2], RuntimeError)

    def test_flush_calls_client_flush(self) -> None:
        client = FakeLangfuseClient()
        tracer = ActiveLangfuseTracer(transform_name="test_transform", client=client)

        tracer.flush()
        assert client.flush_calls == 1

    def test_flush_failure_logs_warning(self) -> None:
        """Flush failures should be logged, not raised — No Silent Failures."""
        client = FakeLangfuseClient(flush_error=RuntimeError("Flush failed"))

        tracer = ActiveLangfuseTracer(transform_name="test_transform", client=client)

        with patch("elspeth.plugins.transforms.llm.langfuse._handle_trace_failure", autospec=True) as mock_handler:
            tracer.flush()
            mock_handler.assert_called_once()
            assert mock_handler.call_args[0][0] == "langfuse_flush_failed"
            assert isinstance(mock_handler.call_args[0][2], RuntimeError)
