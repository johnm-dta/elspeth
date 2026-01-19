# src/elspeth/plugins/clients/llm.py
"""Audited LLM client with automatic call recording."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.base import AuditedClientBase

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


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
        return self.usage.get("prompt_tokens", 0) + self.usage.get(
            "completion_tokens", 0
        )


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


class AuditedLLMClient(AuditedClientBase):
    """LLM client that automatically records all calls to audit trail.

    Wraps an OpenAI-compatible client to ensure every LLM call is
    recorded to the Landscape audit trail. Supports:
    - Automatic request/response recording
    - Latency measurement
    - Error recording with retry classification
    - Token usage tracking

    Example:
        client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=openai.OpenAI(api_key="..."),
            provider="openai",
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
        underlying_client: Any,  # openai.OpenAI or openai.AzureOpenAI
        *,
        provider: str = "openai",
    ) -> None:
        """Initialize audited LLM client.

        Args:
            recorder: LandscapeRecorder for audit trail storage
            state_id: Node state ID to associate calls with
            underlying_client: OpenAI-compatible client instance
            provider: Provider name for audit trail (default: "openai")
        """
        super().__init__(recorder, state_id)
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
        call_index = self._next_call_index()

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": self._provider,
            **kwargs,
        }

        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data={
                    "content": content,
                    "model": response.model,
                    "usage": usage,
                },
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=response.model_dump()
                if hasattr(response, "model_dump")
                else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            error_type = type(e).__name__
            error_str = str(e).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_data,
                error={
                    "type": error_type,
                    "message": str(e),
                    "retryable": is_rate_limit,
                },
                latency_ms=latency_ms,
            )

            if is_rate_limit:
                raise RateLimitError(str(e)) from e
            raise LLMClientError(str(e), retryable=False) from e
