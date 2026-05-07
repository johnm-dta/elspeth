"""Langfuse tracing utilities for LLM transforms.

Extracts the Langfuse v3 span/generation recording pattern that was duplicated
across all 6 LLM transform files. Uses the OpenTelemetry-based context manager
API (start_as_current_observation).

Uses factory pattern to avoid mutable two-phase initialization. The factory
returns either an ActiveLangfuseTracer or NoOpLangfuseTracer — both frozen,
both satisfying the LangfuseTracer protocol.

Follows No Silent Failures: tracing failures are logged at warning level via
structlog. Tracing failures do NOT go to the ELSPETH telemetry stream because
TelemetryEmitCallback expects ExternalCallCompleted dataclass instances, and
tracing failures are a different event class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.transforms.llm.tracing import LangfuseTracingConfig, TracingConfig

logger = structlog.get_logger(__name__)


class LangfuseTracer(Protocol):
    """What the transform needs from tracing. Narrow interface."""

    def record_success(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        response_content: str,
        model: str | None,
        usage: TokenUsage | None = None,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def record_error(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        error_message: str,
        model: str,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def flush(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpLangfuseTracer:
    """No-op tracer for when Langfuse is not configured.

    Matches LangfuseTracer Protocol signatures exactly — enables mypy to
    catch signature drift between Protocol and implementations.
    """

    def record_success(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        response_content: str,
        model: str | None,
        usage: TokenUsage | None = None,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def record_error(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        error_message: str,
        model: str,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class ActiveLangfuseTracer:
    """Fully-initialized Langfuse tracer. Immutable after construction."""

    transform_name: str
    client: Any  # Langfuse instance — typed as Any since it's an optional import

    def record_success(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        response_content: str,
        model: str | None,
        usage: TokenUsage | None = None,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Record successful LLM call as Langfuse span + generation."""
        # Build metadata and kwargs (OUR CODE — let bugs crash immediately)
        metadata = {"token_id": token_id, "plugin": self.transform_name, "query": query_name}
        if extra_metadata:
            metadata.update(extra_metadata)

        update_kwargs: dict[str, Any] = {"output": response_content}
        if usage is not None and usage.has_data:
            usage_details: dict[str, int] = {}
            if usage.prompt_tokens is not None:
                usage_details["input"] = usage.prompt_tokens
            if usage.completion_tokens is not None:
                usage_details["output"] = usage.completion_tokens
            if usage_details:
                update_kwargs["usage_details"] = usage_details
        if latency_ms is not None:
            update_kwargs["metadata"] = {"latency_ms": latency_ms}

        # Build the full message list — include system prompt if present
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Langfuse SDK calls (EXTERNAL boundary — catch SDK/transport errors)
        try:
            with (
                self.client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.transform_name}",
                    metadata=metadata,
                ),
                self.client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=model,
                    input=messages,
                ) as generation,
            ):
                generation.update(**update_kwargs)
        except Exception as e:
            _handle_trace_failure("langfuse_trace_failed", self.transform_name, e)

    def record_error(
        self,
        token_id: str,
        query_name: str,
        prompt: str,
        error_message: str,
        model: str,
        latency_ms: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Record failed LLM call as Langfuse span + generation with ERROR level."""
        # Build metadata and kwargs (OUR CODE — let bugs crash immediately)
        metadata = {"token_id": token_id, "plugin": self.transform_name, "query": query_name}
        if extra_metadata:
            metadata.update(extra_metadata)

        update_kwargs: dict[str, Any] = {
            "level": "ERROR",
            "status_message": error_message,
        }
        if latency_ms is not None:
            update_kwargs["metadata"] = {"latency_ms": latency_ms}

        # Build the full message list — include system prompt if present
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Langfuse SDK calls (EXTERNAL boundary — catch SDK/transport errors)
        try:
            with (
                self.client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.transform_name}",
                    metadata=metadata,
                ),
                self.client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=model,
                    input=messages,
                ) as generation,
            ):
                generation.update(**update_kwargs)
        except Exception as e:
            _handle_trace_failure("langfuse_error_trace_failed", self.transform_name, e)

    def flush(self) -> None:
        """Flush pending tracing data."""
        try:
            self.client.flush()
        except Exception as e:
            _handle_trace_failure("langfuse_flush_failed", self.transform_name, e)


def _handle_trace_failure(
    event_name: str,
    transform_name: str,
    error: Exception,
) -> None:
    """Handle trace recording failure — No Silent Failures via structlog.

    Tracing failures go to structlog only, not the ELSPETH telemetry stream.
    TelemetryEmitCallback expects ExternalCallCompleted (from plugins/clients/base.py),
    which does not match tracing failure events.
    """
    if isinstance(error, (TypeError, AttributeError, NameError)):
        raise error  # Programming errors must crash — not Langfuse SDK failures
    logger.warning(
        event_name,
        plugin=transform_name,
        error=str(error),
        error_type=type(error).__name__,
    )


def create_langfuse_tracer(
    transform_name: str,
    tracing_config: TracingConfig | None,
) -> LangfuseTracer:
    """Factory: returns ActiveLangfuseTracer or NoOpLangfuseTracer.

    Fully constructs the tracer — no deferred setup() needed. The transform
    holds the returned object from __init__ through the entire lifecycle.
    """
    if tracing_config is None:
        return NoOpLangfuseTracer()
    if not isinstance(tracing_config, LangfuseTracingConfig):
        # Non-Langfuse configs are valid no-ops for the Langfuse factory:
        # - AzureAITracingConfig: handled separately in LLMTransform.on_start()
        # - TracingConfig(provider="none"): documented no-tracing setting
        # Unknown providers are rejected by parse_tracing_config() at parse time.
        return NoOpLangfuseTracer()

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=tracing_config.public_key,
            secret_key=tracing_config.secret_key,
            host=tracing_config.host,
            tracing_enabled=tracing_config.tracing_enabled,
        )
        logger.info(
            "Langfuse tracing initialized (v3)",
            provider="langfuse",
            host=tracing_config.host,
            tracing_enabled=tracing_config.tracing_enabled,
        )
        return ActiveLangfuseTracer(transform_name=transform_name, client=client)
    except ImportError as exc:
        # User explicitly configured Langfuse tracing but the package is missing.
        # This is a startup error, not a silent degradation — the user has a
        # reasonable expectation that configured tracing is active.
        raise RuntimeError(
            "Langfuse tracing is configured but the 'langfuse' package is not installed. "
            "Install with: uv pip install 'elspeth[tracing-langfuse]'"
        ) from exc
