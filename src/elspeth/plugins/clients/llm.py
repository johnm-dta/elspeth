# src/elspeth/plugins/clients/llm.py
"""Audited LLM client with automatic call recording."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import CallStatus, CallType
from elspeth.core.canonical import stable_hash
from elspeth.plugins.clients.base import AuditedClientBase, TelemetryEmitCallback
from elspeth.telemetry.events import ExternalCallCompleted

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder

logger = structlog.get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call.

    Provides structured access to LLM response data including:
    - Generated content
    - Model used (may differ from requested model)
    - Token usage statistics
    - Latency measurement
    - Raw response for debugging

    Attributes:
        content: The generated text response
        model: The actual model that processed the request
        usage: Token counts (prompt_tokens, completion_tokens)
        latency_ms: Round-trip time in milliseconds
        raw_response: Full response object for debugging (optional)
    """

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)


class LLMClientError(Exception):
    """Error from LLM client.

    Base exception for all LLM client errors. Includes retryable
    flag to indicate if the operation might succeed on retry.

    Attributes:
        retryable: Whether the error is likely transient and retryable
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class RateLimitError(LLMClientError):
    """Rate limit exceeded - retryable.

    Raised when the LLM provider returns a rate limit error (HTTP 429).
    Always marked as retryable since rate limits are transient.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class NetworkError(LLMClientError):
    """Network/connection error - retryable.

    Raised for transient network issues like timeouts, connection refused,
    DNS failures, etc. These errors are typically transient and should be retried.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ServerError(LLMClientError):
    """Server error (5xx) - retryable.

    Raised for server-side errors that are typically transient:
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    - 529 Model Overloaded (Azure-specific)

    These errors indicate temporary infrastructure issues that may
    resolve on retry.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ContentPolicyError(LLMClientError):
    """Content policy violation - not retryable.

    Raised when the LLM provider rejects the request due to content
    policy violations. Retrying with the same prompt will always fail.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class ContextLengthError(LLMClientError):
    """Context length exceeded - not retryable.

    Raised when the prompt exceeds the model's maximum context length.
    Retrying with the same prompt will always fail.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def _is_retryable_error(exception: Exception) -> bool:
    """Determine if an LLM error is retryable.

    Retryable errors (transient):
    - Rate limits (429)
    - Server errors (500, 502, 503, 504, 529)
    - Network/connection errors (timeout, connection refused, etc.)

    Non-retryable errors (permanent):
    - Client errors (400, 401, 403, 404, 422)
    - Content policy violations
    - Context length exceeded

    Returns:
        True if error is likely transient and should be retried
    """
    error_str = str(exception).lower()

    # Rate limits - always retryable
    if "rate" in error_str or "429" in error_str:
        return True

    # Server errors (5xx) - usually transient
    # Include Microsoft Azure-specific codes (529 = model overloaded)
    server_error_codes = ["500", "502", "503", "504", "529"]
    if any(code in error_str for code in server_error_codes):
        return True

    # Network/connection errors - transient
    network_error_patterns = [
        "timeout",
        "timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "network unreachable",
        "host unreachable",
        "dns",
        "getaddrinfo failed",
    ]
    if any(pattern in error_str for pattern in network_error_patterns):
        return True

    # Client errors (4xx except 429) - permanent
    client_error_codes = ["400", "401", "403", "404", "422"]
    if any(code in error_str for code in client_error_codes):
        return False

    # LLM-specific permanent errors
    permanent_error_patterns = [
        "content_policy_violation",
        "content policy",
        "safety system",
        "context_length_exceeded",
        "context length",
        "maximum context",
    ]
    if any(pattern in error_str for pattern in permanent_error_patterns):
        return False

    # Unknown error - be conservative, do NOT retry
    # This prevents infinite retries on unexpected errors
    return False


class AuditedLLMClient(AuditedClientBase):
    """LLM client that automatically records all calls to audit trail.

    Wraps an OpenAI-compatible client to ensure every LLM call is
    recorded to the Landscape audit trail. Supports:
    - Automatic request/response recording
    - Latency measurement
    - Error recording with retry classification
    - Token usage tracking
    - Telemetry emission after successful audit recording
    - Rate limiting (when limiter provided)

    Example:
        client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            run_id=run_id,
            telemetry_emit=telemetry_emit,
            underlying_client=openai.OpenAI(api_key="..."),
            provider="openai",
            limiter=registry.get_limiter("openai"),
        )

        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.content)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        underlying_client: Any,  # openai.OpenAI or openai.AzureOpenAI
        *,
        provider: str = "openai",
        limiter: Any = None,  # RateLimiter | NoOpLimiter | None
    ) -> None:
        """Initialize audited LLM client.

        Args:
            recorder: LandscapeRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events
            underlying_client: OpenAI-compatible client instance
            provider: Provider name for audit trail (default: "openai")
            limiter: Optional rate limiter for throttling requests
        """
        super().__init__(recorder, state_id, run_id, telemetry_emit, limiter=limiter)
        self._client = underlying_client
        self._provider = provider

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make chat completion call with automatic audit recording.

        Args:
            model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (default: 0.0 for determinism)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments passed to the underlying client

        Returns:
            LLMResponse with content, model, usage, and latency

        Raises:
            RateLimitError: If rate limited (retryable)
            LLMClientError: For other errors (check retryable flag)
        """
        # Acquire rate limit permission before making external call
        self._acquire_rate_limit()

        call_index = self._next_call_index()

        # Build request_data - only include max_tokens if explicitly set
        # (None vs omitted changes request hash semantics and SDK behavior)
        request_data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "provider": self._provider,
            **kwargs,
        }
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        # Build SDK call kwargs - omit max_tokens when None to avoid
        # serializing as JSON null (which can trigger provider validation errors)
        sdk_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            sdk_kwargs["max_tokens"] = max_tokens

        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(**sdk_kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            # Guard against providers that omit usage data (streaming, certain configs)
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            else:
                usage = {}

            # Capture full raw response for audit completeness
            # raw_response includes: all choices, finish_reason, tool_calls, logprobs, etc.
            # NOTE: model_dump() is guaranteed present - we require openai>=2.15 in pyproject.toml
            raw_response = response.model_dump()

            response_data = {
                # Summary fields for convenience
                "content": content,
                "model": response.model,
                "usage": usage,
                # Full response for audit completeness (tool_calls, multiple choices, etc.)
                "raw_response": raw_response,
            }

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data=response_data,
                latency_ms=latency_ms,
            )

            # Telemetry emitted AFTER successful Landscape recording
            # Wrapped in try/except to prevent telemetry failures from corrupting audit trail
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.LLM,
                        provider=self._provider,
                        status=CallStatus.SUCCESS,
                        latency_ms=latency_ms,
                        state_id=self._state_id,  # Transform context
                        operation_id=None,  # Not in source/sink context
                        request_hash=stable_hash(request_data),
                        response_hash=stable_hash(response_data),
                        request_payload=request_data,  # Full request for observability
                        response_payload=response_data,  # Full response for observability
                        token_usage=usage if usage else None,
                    )
                )
            except Exception as tel_err:
                # Telemetry failure must not corrupt the successful call
                # Landscape has the record - telemetry is operational visibility only
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="llm",
                )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=raw_response,  # Reuse captured response from audit recording
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            error_type = type(e).__name__
            error_str = str(e).lower()

            # Classify error for retry decision
            is_retryable = _is_retryable_error(e)

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_data,
                error={
                    "type": error_type,
                    "message": str(e),
                    "retryable": is_retryable,
                },
                latency_ms=latency_ms,
            )

            # Telemetry emitted AFTER successful Landscape recording (even for call errors)
            # Wrapped in try/except to prevent telemetry failures from corrupting audit trail
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.LLM,
                        provider=self._provider,
                        status=CallStatus.ERROR,
                        latency_ms=latency_ms,
                        state_id=self._state_id,  # Transform context
                        operation_id=None,  # Not in source/sink context
                        request_hash=stable_hash(request_data),
                        response_hash=None,  # No response on error
                        request_payload=request_data,  # Full request for observability
                        response_payload=None,  # No response on error
                        token_usage=None,
                    )
                )
            except Exception as tel_err:
                # Telemetry failure must not corrupt the error handling flow
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="llm",
                )

            # Raise specific exception type based on error classification
            if "rate" in error_str or "429" in error_str:
                raise RateLimitError(str(e)) from e
            elif "content_policy" in error_str or "safety system" in error_str:
                raise ContentPolicyError(str(e)) from e
            elif "context_length" in error_str or "maximum context" in error_str:
                raise ContextLengthError(str(e)) from e
            elif is_retryable:
                # Server error or network error - determine which
                server_error_codes = ["500", "502", "503", "504", "529"]
                if any(code in error_str for code in server_error_codes):
                    raise ServerError(str(e)) from e
                else:
                    # Must be network error (timeout, connection refused, etc.)
                    raise NetworkError(str(e)) from e
            else:
                # Client error or unknown - not retryable
                raise LLMClientError(str(e), retryable=False) from e
