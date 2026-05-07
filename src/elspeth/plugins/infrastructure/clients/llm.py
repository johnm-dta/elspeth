"""Audited LLM client with automatic call recording."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import structlog

import elspeth.contracts.errors as contract_errors
from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import LLMCallError, LLMCallRequest, LLMCallResponse, RawCallPayload
from elspeth.contracts.errors import PluginRetryableError
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.canonical import stable_hash
from elspeth.plugins.infrastructure.clients.base import AuditedClientBase, TelemetryEmitCallback

if TYPE_CHECKING:
    from elspeth.contracts.audit_protocols import CallRecorder
    from elspeth.core.rate_limit import NoOpLimiter
    from elspeth.core.rate_limit.limiter import RateLimiter

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Response from an LLM call.

    Frozen: LLM responses are immutable evidence — the content, model,
    usage, and raw response must not be modified after construction.

    Attributes:
        content: The generated text response
        model: The actual model that processed the request
        usage: Token counts (prompt_tokens, completion_tokens)
        latency_ms: Round-trip time in milliseconds
        raw_response: Full response object for debugging (optional)
    """

    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage.unknown)
    latency_ms: float = 0.0
    raw_response: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or not math.isfinite(self.latency_ms):
            raise ValueError(f"LLMResponse.latency_ms must be non-negative and finite, got {self.latency_ms}")
        if self.raw_response is not None and not isinstance(self.raw_response, MappingProxyType):
            object.__setattr__(self, "raw_response", deep_freeze(self.raw_response))

    @property
    def total_tokens(self) -> int | None:
        """Total tokens used (prompt + completion), or None if unknown."""
        return self.usage.total_tokens


class LLMClientError(PluginRetryableError):
    """Error from LLM client.

    Base exception for all LLM client errors. Includes retryable
    flag to indicate if the operation might succeed on retry.

    Attributes:
        retryable: Whether the error is likely transient and retryable
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


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


_RATE_LIMIT_PATTERNS = (
    re.compile(r"\brate[\s_-]*limit(?:ed|ing)?\b"),
    re.compile(r"\brate(?:\s+has\s+been)?\s+exceeded\b"),
    re.compile(r"\btoo many requests\b"),
    re.compile(r"\bthrottl(?:e|ed|ing)\b"),
)
_SERVER_ERROR_CODE_PATTERN = re.compile(r"\b(?:500|502|503|504|529)\b")
_CLIENT_ERROR_CODE_PATTERN = re.compile(r"\b(?:400|401|403|404|422)\b")
_NETWORK_ERROR_PATTERNS = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "connection aborted",
    "network unreachable",
    "host unreachable",
    "dns",
    "getaddrinfo failed",
)
_CONTENT_POLICY_PATTERNS = (
    "content_policy_violation",
    "content policy",
    "safety system",
)
_CONTEXT_LENGTH_PATTERNS = (
    "context_length_exceeded",
    "context length",
    "maximum context",
)


def _classify_llm_error(exception: Exception) -> str:
    """Classify an LLM error into a canonical category."""
    error_str = str(exception).lower()

    if any(pattern in error_str for pattern in _CONTENT_POLICY_PATTERNS):
        return "content_policy"
    if any(pattern in error_str for pattern in _CONTEXT_LENGTH_PATTERNS):
        return "context_length"

    # Match explicit rate-limit indicators only; do not match arbitrary "rate" substrings.
    if re.search(r"\b429\b", error_str) or any(pattern.search(error_str) for pattern in _RATE_LIMIT_PATTERNS):
        return "rate_limit"

    if _SERVER_ERROR_CODE_PATTERN.search(error_str):
        return "server"
    if any(pattern in error_str for pattern in _NETWORK_ERROR_PATTERNS):
        return "network"
    if _CLIENT_ERROR_CODE_PATTERN.search(error_str):
        return "client"
    return "unknown"


def _extract_usage_from_provider_response(usage: Any) -> TokenUsage:
    """Normalize provider usage objects at the Tier 3 boundary.

    Providers may return usage as an SDK object with attributes, a mapping, or
    a partial aggregate-only payload. Reconstruct through ``TokenUsage.from_dict``
    so missing and non-int fields become explicit ``None`` rather than raising.
    """
    if usage is None:
        return TokenUsage.unknown()

    if isinstance(usage, Mapping):
        usage_data = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    else:
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    return TokenUsage.from_dict(usage_data)


def _validate_provider_response_model(model: Any) -> str:
    """Require provider model metadata to be a non-empty string."""
    if not isinstance(model, str):
        raise ValueError(f"LLM response model is {type(model).__name__}, expected non-empty str")
    if not model.strip():
        raise ValueError("LLM response model must be non-empty")
    return model


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
            execution=execution_repo,
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
        execution: CallRecorder,
        state_id: str,
        run_id: str,
        telemetry_emit: TelemetryEmitCallback,
        underlying_client: Any,  # openai.OpenAI or openai.AzureOpenAI
        *,
        provider: str = "openai",
        limiter: RateLimiter | NoOpLimiter | None = None,
        token_id: str | None = None,
    ) -> None:
        """Initialize audited LLM client.

        Args:
            execution: CallRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            run_id: Pipeline run ID for telemetry correlation
            telemetry_emit: Callback to emit telemetry events
            underlying_client: OpenAI-compatible client instance
            provider: Provider name for audit trail (default: "openai")
            limiter: Optional rate limiter for throttling requests
            token_id: Optional token identity for telemetry correlation
        """
        super().__init__(execution, state_id, run_id, telemetry_emit, limiter=limiter, token_id=token_id)
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

        # Build request DTO - frozen dataclass ensures construction-time type safety;
        # to_dict() conditionally omits max_tokens when None (hash-stable).
        # DTO stays alive for typed telemetry payload; dict form used for Landscape hashing.
        request_dto = LLMCallRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            provider=self._provider,
            max_tokens=max_tokens,
            extra_kwargs=kwargs,
        )
        request_data = request_dto.to_dict()

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
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            error_type = type(e).__name__
            error_class = _classify_llm_error(e)

            # Classify error for retry decision
            is_retryable = error_class in {"rate_limit", "server", "network"}

            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                error=LLMCallError(
                    type=error_type,
                    message=str(e),
                    retryable=is_retryable,
                ),
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
                        token_id=self._telemetry_token_id(),
                        request_hash=stable_hash(request_data),
                        response_hash=None,  # No response on error
                        request_payload=request_dto,  # Typed DTO, not dict
                        response_payload=None,  # No response on error
                        token_usage=None,
                    )
                )
            except contract_errors.TIER_1_ERRORS:
                raise  # System bugs and audit integrity violations must crash
            except (TypeError, AttributeError, KeyError, NameError):
                raise  # Programming errors must crash
            except Exception as tel_err:
                # Telemetry failure must not corrupt the error handling flow
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="llm",
                    exc_info=True,
                )

            # Raise specific exception type based on error classification
            if error_class == "rate_limit":
                raise RateLimitError(str(e)) from e
            elif error_class == "content_policy":
                raise ContentPolicyError(str(e)) from e
            elif error_class == "context_length":
                raise ContextLengthError(str(e)) from e
            elif error_class == "server":
                raise ServerError(str(e)) from e
            elif error_class == "network":
                raise NetworkError(str(e)) from e
            else:
                # Client error or unknown - not retryable
                raise LLMClientError(str(e), retryable=False) from e

        # Success path — OUTSIDE try/except so internal processing bugs crash
        # instead of being misclassified as LLM errors
        latency_ms = (time.perf_counter() - start) * 1000

        # Capture the provider response once, then validate/normalize the Tier 3
        # fields from that snapshot. This keeps malformed responses on the
        # audited path instead of letting them leak into success handling.
        usage = _extract_usage_from_provider_response(response.usage)
        try:
            raw_response = response.model_dump()
        except (TypeError, ValueError, RecursionError, AttributeError) as dump_exc:
            # The LLM call happened — record it before re-raising so the
            # audit trail reflects the consumed tokens even though we can't
            # fully serialize the response.
            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                error=LLMCallError(
                    type="ResponseProcessingError",
                    message=f"Failed to serialize LLM response: {dump_exc}",
                    retryable=False,
                ),
                latency_ms=latency_ms,
            )
            raise LLMClientError(
                f"Failed to serialize LLM response: {dump_exc}",
                retryable=False,
            ) from dump_exc

        try:
            response_model = _validate_provider_response_model(response.model)
        except ValueError as model_exc:
            error_msg = f"{model_exc}. Provider returned malformed data at Tier 3 boundary."
            response_payload = RawCallPayload(raw_response)
            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                response_data=response_payload,
                error=LLMCallError(
                    type="MalformedResponseError",
                    message=error_msg,
                    retryable=False,
                ),
                latency_ms=latency_ms,
            )

            response_data = response_payload.to_dict()
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.LLM,
                        provider=self._provider,
                        status=CallStatus.ERROR,
                        latency_ms=latency_ms,
                        state_id=self._state_id,
                        operation_id=None,
                        token_id=self._telemetry_token_id(),
                        request_hash=stable_hash(request_data),
                        response_hash=stable_hash(response_data),
                        request_payload=request_dto,
                        response_payload=response_payload,
                        token_usage=usage if usage.has_data else None,
                    )
                )
            except contract_errors.TIER_1_ERRORS:
                raise  # System bugs and audit integrity violations must crash
            except (TypeError, AttributeError, KeyError, NameError):
                raise  # Programming errors must crash
            except Exception as tel_err:
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="llm",
                    exc_info=True,
                )

            raise LLMClientError(error_msg, retryable=False) from model_exc

        # Tier 3 boundary: validate LLM response structure immediately.
        if not response.choices:
            error_msg = "LLM returned empty choices array — abnormal response"
            response_dto = LLMCallResponse(
                content="",  # No content available
                model=response_model,
                usage=usage,
                raw_response=raw_response,
            )
            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                response_data=response_dto,
                error=LLMCallError(
                    type="EmptyChoicesError",
                    message=error_msg,
                    retryable=False,
                ),
                latency_ms=latency_ms,
            )
            raise LLMClientError(error_msg, retryable=False)

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if content is None:
            # Tool call responses have no text content — ELSPETH does not support
            # tool_calls, so this is an error (not a fabrication opportunity).
            # Record the call as ERROR with raw response preserved, then raise.
            if finish_reason == "tool_calls":
                error_msg = "LLM returned tool_calls response (not supported by ELSPETH)"
                response_dto = LLMCallResponse(
                    content="",  # No text content available
                    model=response_model,
                    usage=usage,
                    raw_response=raw_response,
                )
                self._execution.record_call(
                    state_id=self._state_id,
                    call_index=call_index,
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data=request_dto,
                    response_data=response_dto,
                    error=LLMCallError(
                        type="UnsupportedResponseError",
                        message=error_msg,
                        retryable=False,
                    ),
                    latency_ms=latency_ms,
                )
                raise LLMClientError(error_msg, retryable=False)

            # Record the call BEFORE raising — the LLM call happened and must
            # appear in the audit trail even though the response is unusable.
            # Without this, content-filtered calls vanish from the audit trail
            # and create unexplained call-index gaps.
            error_msg = "LLM returned null content (likely content-filtered by provider)"
            response_dto = LLMCallResponse(
                content="",  # Null content normalized for DTO
                model=response_model,
                usage=usage,
                raw_response=raw_response,
            )
            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                response_data=response_dto,
                error=LLMCallError(
                    type="ContentPolicyError",
                    message=error_msg,
                    retryable=False,
                ),
                latency_ms=latency_ms,
            )

            # Telemetry emitted AFTER successful Landscape recording (even for null-content errors)
            # Unlike SDK errors, we have response data here — the HTTP call succeeded
            response_data = response_dto.to_dict()
            try:
                self._telemetry_emit(
                    ExternalCallCompleted(
                        timestamp=datetime.now(UTC),
                        run_id=self._run_id,
                        call_type=CallType.LLM,
                        provider=self._provider,
                        status=CallStatus.ERROR,
                        latency_ms=latency_ms,
                        state_id=self._state_id,
                        operation_id=None,
                        token_id=self._telemetry_token_id(),
                        request_hash=stable_hash(request_data),
                        response_hash=stable_hash(response_data),
                        request_payload=request_dto,
                        response_payload=response_dto,
                        token_usage=usage if usage.has_data else None,
                    )
                )
            except contract_errors.TIER_1_ERRORS:
                raise  # System bugs and audit integrity violations must crash
            except (TypeError, AttributeError, KeyError, NameError):
                raise  # Programming errors must crash
            except Exception as tel_err:
                # Telemetry failure must not corrupt the error handling flow
                logger.warning(
                    "telemetry_emit_failed",
                    error=str(tel_err),
                    error_type=type(tel_err).__name__,
                    run_id=self._run_id,
                    state_id=self._state_id,
                    call_type="llm",
                    exc_info=True,
                )

            raise ContentPolicyError(error_msg)

        # Tier 3 boundary: validate content is actually str.
        # Provider bugs or SDK schema drift could return non-str content
        # (e.g., list for multi-part, int, dict). Recording non-str as SUCCESS
        # would violate the response contract and crash downstream .strip() calls.
        if not isinstance(content, str):
            error_msg = (
                f"LLM response content is {type(content).__name__}, expected str. Provider returned malformed data at Tier 3 boundary."
            )
            response_payload = RawCallPayload(raw_response)
            self._execution.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_dto,
                response_data=response_payload,
                error=LLMCallError(
                    type="MalformedResponseError",
                    message=error_msg,
                    retryable=False,
                ),
                latency_ms=latency_ms,
            )
            raise LLMClientError(error_msg, retryable=False)

        response_dto = LLMCallResponse(
            content=content,
            model=response_model,
            usage=usage,
            raw_response=raw_response,
        )
        response_data = response_dto.to_dict()

        self._execution.record_call(
            state_id=self._state_id,
            call_index=call_index,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_dto,
            response_data=response_dto,
            latency_ms=latency_ms,
        )

        # Telemetry emitted AFTER successful Landscape recording
        usage_snapshot = usage if usage.has_data else None
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
                    token_id=self._telemetry_token_id(),
                    request_hash=stable_hash(request_data),
                    response_hash=stable_hash(response_data),
                    request_payload=request_dto,  # Typed DTO, not dict
                    response_payload=response_dto,  # Typed DTO, not dict
                    token_usage=usage_snapshot,
                )
            )
        except contract_errors.TIER_1_ERRORS:
            raise  # System bugs and audit integrity violations must crash
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors must crash
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
                exc_info=True,
            )

        return LLMResponse(
            content=content,
            model=response_model,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=raw_response,  # Reuse captured response from audit recording
        )
