"""LLM provider protocol and response DTOs.

The LLMProvider protocol defines the narrow interface between LLMTransform
(shared logic) and provider-specific transport (Azure SDK, OpenRouter HTTP).

Providers are responsible for:
1. Client lifecycle (creation, caching per state_id, cleanup)
2. LLM API calls (transport-specific)
3. Tier 3 boundary validation (response parsing, NaN rejection)
4. Error classification (raising typed exceptions)
5. Audit trail recording (via their Audited*Client)
6. Finish reason normalization (provider-specific → FinishReason enum)

The transform above the provider never sees raw SDK/HTTP responses.
raw_response is NOT on LLMQueryResult — providers record audit data
via their Audited*Client (D2 from architecture remediation).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from elspeth.contracts.token_usage import TokenUsage


class FinishReason(StrEnum):
    """Validated finish reasons from LLM providers."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class UnrecognizedFinishReason:
    """Sentinel for finish reasons not in our FinishReason enum.

    Preserves the raw value for audit trail recording, unlike None which
    conflates "absent" (no finish_reason in response) with "unrecognized"
    (provider sent a value we don't know about).
    """

    raw: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw, str):
            raise TypeError(f"raw must be a string, got {type(self.raw).__name__}: {self.raw!r}")


#: Type alias for parsed finish reasons.  ``None`` means the provider did
#: not include a finish_reason field at all (absent).
ParsedFinishReason = FinishReason | UnrecognizedFinishReason | None


def parse_finish_reason(raw: str | None) -> ParsedFinishReason:
    """Parse raw finish_reason string into validated enum.

    Returns:
        FinishReason: If the raw value is a known enum member.
        UnrecognizedFinishReason: If the raw value is not recognized.
            Preserves the raw string for audit recording.
        None: If raw is None (no finish_reason in response).

    Providers should normalize their known finish reasons BEFORE calling
    this function (e.g. Anthropic "end_turn" → "stop").

    IMPORTANT: Callers MUST NOT call this with raw=None to represent
    "no finish_reason in response" — pass None directly instead. This
    function should only be called when a non-None raw value exists.
    """
    if raw is None:
        return None
    try:
        return FinishReason(raw)
    except ValueError:
        return UnrecognizedFinishReason(raw)


@dataclass(frozen=True, slots=True)
class LLMQueryResult:
    """Normalized, validated result from any LLM provider.

    All Tier 3 validation has already happened inside the provider.
    Content is guaranteed non-null, non-empty, non-whitespace-only string.
    Usage is normalized via TokenUsage.known/unknown.

    NOTE: raw_response is NOT included here. Providers own audit recording
    via their Audited*Client (chat_completion/post methods record internally
    via their Landscape recorder) — the raw SDK/HTTP response stays within
    the provider boundary (D2 principle).
    """

    content: str
    usage: TokenUsage
    model: str
    finish_reason: ParsedFinishReason = None

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("LLMQueryResult.content must be non-empty (whitespace-only rejected)")
        if not self.model or not self.model.strip():
            raise ValueError("LLMQueryResult.model must be non-empty")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError(f"LLMQueryResult.usage must be a TokenUsage instance, got {type(self.usage).__name__}")
        if self.finish_reason is not None and not isinstance(self.finish_reason, (FinishReason, UnrecognizedFinishReason)):
            raise TypeError(
                f"LLMQueryResult.finish_reason must be FinishReason, UnrecognizedFinishReason, or None, "
                f"got {type(self.finish_reason).__name__}: {self.finish_reason!r}"
            )


@runtime_checkable
class LLMProvider(Protocol):
    """What LLMTransform needs from a provider. Narrow interface.

    Providers raise typed exceptions from elspeth.plugins.infrastructure.clients.llm:
    - RateLimitError: 429 / rate limit (retryable)
    - NetworkError: connection failures (retryable)
    - ServerError: 5xx errors (retryable)
    - ContentPolicyError: content filtering (not retryable)
    - ContextLengthError: context too long (not retryable)
    - LLMClientError: other failures (not retryable)

    Note: LLMClientError (exception in plugins/clients/llm.py) is NOT the
    same as LLMCallError (frozen dataclass in contracts/call_data.py for
    audit recording). Providers RAISE LLMClientError; they RECORD LLMCallError.
    """

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
    ) -> LLMQueryResult: ...

    def close(self) -> None: ...
