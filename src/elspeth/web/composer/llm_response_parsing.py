"""LiteLLM response-parsing helpers for the composer service.

This module is the Tier 3 trust boundary between the composer service and
the external LiteLLM response objects. Every public helper here treats its
input as untrusted external data: response shapes vary across providers,
usage objects can be missing fields, and provider artefacts can be opaque
SDK wrappers. The helpers:

- Extract token usage, provider cost, response model, request id,
  reasoning metadata, and free-form provider artefacts in a JSON-safe form.
- Construct ``ComposerLLMCall`` audit records from the normalised values.
- Attach the recorder's buffered LLM-call snapshot to escaping exceptions
  so the route layer can persist a complete decision trail.
- Apply Anthropic prompt-cache markers to messages and tools before the
  request goes on the wire.

Tier 3 boundary discipline (per ELSPETH CLAUDE.md):

- ``getattr(...)`` and ``dict.get(...)`` are appropriate at this boundary —
  LiteLLM response objects are external data, not internal contracts.
- ``isinstance(...)`` guards distinguish provider-Mapping shapes from
  attribute-style provider response objects.
- Missing fields surface as ``None`` (absence), not as fabricated zeros.

The module imports only from contracts (L0) and stdlib. It must not import
from ``service.py`` — that would create a cycle.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict, cast

from elspeth.contracts.composer_llm_audit import (
    PROVIDER_COST_SOURCE_NOT_AVAILABLE,
    PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST,
    ComposerLLMCall,
    ComposerLLMCallStatus,
    ComposerLLMProviderCostSource,
)
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.canonical import stable_hash

if TYPE_CHECKING:
    from elspeth.web.composer.audit import BufferingRecorder

__all__ = [
    "apply_anthropic_cache_markers",
    "attach_llm_calls",
    "build_llm_call_record",
    "safe_response_model",
    "supports_anthropic_prompt_cache_markers",
    "token_usage_from_response",
]


class _ReasoningMetadata(TypedDict):
    reasoning_content: str | None
    reasoning_details: Any | None
    thinking_blocks: Any | None


def _provider_details_payload(value: Any, *, fields: tuple[str, ...]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    return {field: getattr(value, field, None) for field in fields}


def token_usage_from_response(response: Any | None) -> TokenUsage:
    """Normalize provider usage metadata without fabricating missing counts.

    Captures provider-reported prompt-cache statistics in addition to the
    base prompt/completion/total counts. Cache fields are exposed as:

    - OpenAI / OpenRouter: nested ``usage.prompt_tokens_details.cached_tokens``
    - Anthropic: sibling ``usage.cache_creation_input_tokens`` and
      ``usage.cache_read_input_tokens``

    All cache fields are read defensively: a non-Mapping ``usage`` whose
    attributes are missing yields ``None`` rather than a fabricated zero.
    See elspeth-4e79436719 §Bug C.

    LiteLLM-shape deduplication: for Anthropic-family responses (direct
    Anthropic, Bedrock-Claude, Vertex-Claude/Gemini), LiteLLM's transformation
    layer populates ``prompt_tokens_details.cached_tokens`` as a synthetic
    copy of the Anthropic sibling ``cache_read_input_tokens`` (and uses ``0``
    when no read occurred). Treating both as independent signals would
    double-record the same provider counter into ``cached_prompt_tokens`` and
    ``cache_read_input_tokens`` — and, for creation-only responses, fabricate
    a zero into ``cached_prompt_tokens`` that the provider never asserted.
    When *either* Anthropic sibling is present we therefore drop the nested
    OpenAI shape: presence of a sibling is the provenance signal that the
    nested view is LiteLLM-derived, not provider-native.

    Primary-source evidence (LiteLLM transformations):
    - ``litellm/llms/anthropic/chat/transformation.py`` (Anthropic direct)
    - ``litellm/llms/bedrock/chat/converse_transformation.py`` (Bedrock Claude)
    - ``litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py``
      (Vertex Gemini / Claude)

    Each constructs ``PromptTokensDetailsWrapper(cached_tokens=cache_read_input_tokens)``
    and emits both fields on the returned ``Usage`` object.
    """
    if response is None:
        return TokenUsage.unknown()
    usage = getattr(response, "usage", None)
    if isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        output_details = usage.get("output_tokens_details")
        usage_data = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "prompt_tokens_details": details if isinstance(details, Mapping) else None,
            "completion_tokens_details": completion_details if isinstance(completion_details, Mapping) else None,
            "output_tokens_details": output_details if isinstance(output_details, Mapping) else None,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
        }
    else:
        details_attr = getattr(usage, "prompt_tokens_details", None)
        details_payload = _provider_details_payload(details_attr, fields=("cached_tokens",))
        completion_details_payload = _provider_details_payload(
            getattr(usage, "completion_tokens_details", None),
            fields=("reasoning_tokens",),
        )
        output_details_payload = _provider_details_payload(
            getattr(usage, "output_tokens_details", None),
            fields=("reasoning_tokens",),
        )
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "prompt_tokens_details": details_payload,
            "completion_tokens_details": completion_details_payload,
            "output_tokens_details": output_details_payload,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", None),
        }
    if usage_data["cache_read_input_tokens"] is not None or usage_data["cache_creation_input_tokens"] is not None:
        usage_data["prompt_tokens_details"] = None
    return TokenUsage.from_dict(usage_data)


def _provider_cost_from_response(response: Any | None) -> tuple[float | None, ComposerLLMProviderCostSource]:
    """Extract provider-reported request cost without fabricating a value.

    OpenRouter exposes request cost as ``response.usage.cost`` through the
    LiteLLM response object. This is external provider metadata, so malformed,
    negative, non-finite, or absent values are treated as unavailable rather
    than propagated into the audit row.
    """
    if response is None:
        return None, PROVIDER_COST_SOURCE_NOT_AVAILABLE
    usage = getattr(response, "usage", None)
    if isinstance(usage, Mapping):
        raw_cost = usage["cost"] if "cost" in usage else None
    else:
        raw_cost = getattr(usage, "cost", None)
    if type(raw_cost) is bool or type(raw_cost) not in (int, float):
        return None, PROVIDER_COST_SOURCE_NOT_AVAILABLE
    cost = float(cast(int | float, raw_cost))
    if not math.isfinite(cost) or cost < 0:
        return None, PROVIDER_COST_SOURCE_NOT_AVAILABLE
    return cost, PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST


def safe_response_model(response: Any | None) -> str | None:
    if response is None:
        return None
    model = getattr(response, "model", None)
    if isinstance(model, str) and model.strip():
        return model
    return None


def _safe_provider_request_id(response: Any | None) -> str | None:
    if response is None:
        return None
    for attr in ("id", "request_id"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value and len(value) <= 256:
            return value
    return None


def _response_field(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _first_response_message(response: Any | None) -> Any | None:
    if response is None:
        return None
    choices = _response_field(response, "choices")
    if not isinstance(choices, list | tuple) or not choices:
        return None
    return _response_field(choices[0], "message")


def _json_safe_provider_artifact(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_provider_artifact(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_provider_artifact(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe_provider_artifact(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe_provider_artifact(model_dump(mode="json"))
        except TypeError:
            return _json_safe_provider_artifact(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe_provider_artifact(to_dict())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _json_safe_provider_artifact(dict_method())
    return repr(value)


def _reasoning_metadata_from_response(response: Any | None) -> _ReasoningMetadata:
    message = _first_response_message(response)
    if message is None:
        return {
            "reasoning_content": None,
            "reasoning_details": None,
            "thinking_blocks": None,
        }
    reasoning_content = _response_field(message, "reasoning")
    if not isinstance(reasoning_content, str):
        reasoning_content = _response_field(message, "reasoning_content")
    if not isinstance(reasoning_content, str):
        reasoning_content = None

    reasoning_details = _response_field(message, "reasoning_details")
    if reasoning_details is None:
        provider_specific = _response_field(message, "provider_specific_fields")
        reasoning_details = _response_field(provider_specific, "reasoning_details") if provider_specific is not None else None

    return {
        "reasoning_content": reasoning_content,
        "reasoning_details": _json_safe_provider_artifact(reasoning_details),
        "thinking_blocks": _json_safe_provider_artifact(_response_field(message, "thinking_blocks")),
    }


def build_llm_call_record(
    *,
    model_requested: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    status: ComposerLLMCallStatus,
    started_at: datetime,
    started_ns: int,
    temperature: float | None,
    seed: int | None,
    response: Any | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
) -> ComposerLLMCall:
    usage = token_usage_from_response(response)
    provider_cost, provider_cost_source = _provider_cost_from_response(response)
    reasoning_metadata = _reasoning_metadata_from_response(response)
    return ComposerLLMCall(
        model_requested=model_requested,
        model_returned=safe_response_model(response),
        status=status,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        reasoning_content=reasoning_metadata["reasoning_content"],
        reasoning_details=reasoning_metadata["reasoning_details"],
        thinking_blocks=reasoning_metadata["thinking_blocks"],
        provider_cost=provider_cost,
        provider_cost_source=provider_cost_source,
        latency_ms=(time.monotonic_ns() - started_ns) // 1_000_000,
        provider_request_id=_safe_provider_request_id(response),
        messages_hash=stable_hash(messages),
        tools_spec_hash=stable_hash(tools) if tools is not None else None,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        error_class=error_class,
        error_message=error_message,
        temperature=temperature,
        seed=seed,
    )


def attach_llm_calls(exc: BaseException, recorder: BufferingRecorder | None) -> None:
    """Attach buffered LLM calls to exception objects that otherwise lack carriers."""
    if recorder is None:
        return
    exc_with_calls = cast(Any, exc)
    exc_with_calls.llm_calls = recorder.llm_calls


# ---------------------------------------------------------------------------
# Provider prompt-cache markers (elspeth-4e79436719)
#
# Anthropic-family providers (Anthropic direct, OpenRouter Anthropic routing,
# AWS Bedrock Anthropic, Google Vertex Anthropic) use explicit
# ``cache_control: {"type": "ephemeral"}`` markers placed on the system
# message and on the trailing function tool to indicate the static prefix
# that should be cached for follow-up requests within a session.
#
# OpenAI / OpenRouter OpenAI / Azure OpenAI providers use *automatic* prefix
# caching above a 1024-token threshold and do NOT honor the ``cache_control``
# field — for them, the marker is silently ignored. We still skip the
# transform on those providers to keep the wire payload smaller and the
# audit ``messages_hash`` clean.
#
# LiteLLM's Anthropic adapter (verified against
# ``litellm/llms/anthropic/chat/transformation.py``) accepts cache_control
# either at the message level OR per-content-block; we use the simpler
# message-level shape so existing message-construction code in prompts.py
# stays untouched. Tools accept cache_control either at the tool level OR
# inside the ``function`` field; we use the tool-level shape for symmetry.
# ---------------------------------------------------------------------------


def supports_anthropic_prompt_cache_markers(model: str | None) -> bool:
    """Return True when the configured model honors Anthropic cache_control markers.

    Coverage: Anthropic direct (``anthropic/...``), OpenRouter Anthropic
    routing (``openrouter/anthropic/...``), AWS Bedrock Anthropic
    (``bedrock/anthropic.*``), Google Vertex Anthropic
    (``vertex_ai/claude-*``), and bare ``claude-*`` model identifiers.

    Returns False for OpenAI, Azure OpenAI, Google Gemini, and any model
    not matched by the prefix heuristics — those providers use automatic
    prefix caching that doesn't need explicit markers.
    """
    if not isinstance(model, str):
        return False
    lowered = model.lower()
    return (
        lowered.startswith("anthropic/")
        or lowered.startswith("openrouter/anthropic/")
        or lowered.startswith("bedrock/anthropic.")
        or lowered.startswith("vertex_ai/claude-")
        or "claude-" in lowered
    )


def apply_anthropic_cache_markers(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Return new messages/tools lists with Anthropic ``cache_control`` markers.

    Behavior:
    - The first message with ``role == "system"`` receives a top-level
      ``cache_control: {"type": "ephemeral"}`` field. ``build_messages()``
      keeps this first system message to the stable skill/deployment prompt;
      the dynamic current-state JSON is emitted as a later system message.
      LiteLLM's Anthropic transform recognizes this marker and propagates it
      onto the corresponding ``AnthropicSystemMessageContent`` block on the
      wire.
    - The LAST tool in ``tools`` receives the same marker at the tool
      level. Anthropic caches all tools up to and including the marker,
      so marking the trailing tool covers the full tools array.

    Inputs are NOT mutated; new lists are returned. Messages and tools
    other than the marked entries are passed through by reference (their
    contents are not deep-copied) — this keeps the transform cheap and
    is safe because the receiver (LiteLLM) does not mutate them.
    """
    new_messages: list[dict[str, Any]] = list(messages)
    for index, message in enumerate(new_messages):
        # ``messages`` is our outbound request payload built by
        # ``build_messages()`` (prompts.py), not an external response object.
        # Every message dict we construct carries a ``"role"`` key by
        # contract; a missing role is a first-party bug, so access it
        # directly and let ``KeyError`` surface it rather than masking it
        # with ``.get()``. This is Tier-2 data we authored, not the Tier-3
        # provider responses the rest of this module normalizes.
        if message["role"] == "system":
            new_messages[index] = {**message, "cache_control": {"type": "ephemeral"}}
            break

    new_tools: list[dict[str, Any]] | None = None
    if tools:
        new_tools = list(tools)
        # Cache-key contract: the marker is placed on the trailing tool because
        # Anthropic caches "all content up to and including the marker." Tool
        # ORDER is therefore part of the cache key — a reordering of
        # ``get_tool_definitions()`` would silently shift which tool is "last"
        # and invalidate the prompt cache for every follow-up turn. The order
        # contract is locked by ``TestToolListOrderIsCacheKeyContract`` in
        # ``tests/unit/web/composer/test_provider_cache_markers.py``.
        new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

    return new_messages, new_tools
