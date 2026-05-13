"""ComposerServiceImpl — bounded LLM tool-use loop for pipeline composition.

Uses LiteLLM for provider abstraction. Model configured via
WebSettings.composer_model. Tool calls are executed against
CompositionState + CatalogService.

Dual-counter budget: separate limits for discovery and composition turns.
Discovery cache: cacheable discovery tool results cached per-compose-call
in a local dict variable (not an instance field) to avoid concurrent-request
races.

Layer: L3 (application).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, TypedDict, cast

if TYPE_CHECKING:
    from elspeth.web.composer.guided.state_machine import TerminalState
    from elspeth.web.sessions.protocol import SessionServiceProtocol

import structlog
from opentelemetry import metrics
from sqlalchemy import Engine, update
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_llm_audit import (
    PROVIDER_COST_SOURCE_NOT_AVAILABLE,
    PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST,
    ComposerLLMCall,
    ComposerLLMCallStatus,
    ComposerLLMProviderCostSource,
)
from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.canonical import stable_hash
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.anti_anchor import AntiAnchorTracker
from elspeth.web.composer.audit import (
    BufferingRecorder,
    begin_dispatch,
    begin_dispatch_or_arg_error,
    canonicalize_pydantic_cause,
    dispatch_with_audit,
    finish_arg_error,
    finish_plugin_crash,
    finish_success,
)
from elspeth.web.composer.progress import (
    ComposerProgressEvent,
    ComposerProgressSink,
    convergence_progress_event,
)
from elspeth.web.composer.prompts import build_messages, build_run_diagnostics_messages, build_system_prompt
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerResult,
    ComposerRuntimePreflightError,
    ComposerServiceError,
    ComposerSettings,
    ToolArgumentError,
)
from elspeth.web.composer.state import CompositionState, ValidationSummary
from elspeth.web.composer.state_claim_grounding import (
    check_state_claim_grounding,
    compose_grounded_message,
)
from elspeth.web.composer.tools import (
    ADVISOR_TRIGGER_REACTIVE,
    ADVISOR_TRIGGER_VALUES,
    RuntimePreflight,
    ToolResult,
    compute_proof_diagnostics,
    execute_tool,
    get_tool_definitions,
    is_cacheable_discovery_tool,
    is_discovery_tool,
)
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightEntry,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import ValidationCheck, ValidationError, ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.sessions.models import sessions_table

slog = structlog.get_logger()

_ARRAY_ITEM_SEGMENT = "[]"
_LLM_API_MAX_ATTEMPTS = 3
_LLM_API_RETRY_BASE_DELAY_SECONDS = 1.0

# Composer LLM sampling constants. Hardcoded for deterministic tool-call
# construction (RGR investigation 2026-05-06 §4.4 traced ~33% hard-GREEN
# ceiling on URL→download→line-explode primarily to LiteLLM/OpenRouter
# default sampling at ~1.0). Pipeline composition is closer to "extraction"
# than "creative writing" in the LLM-debugging-skill temperature guide;
# 0.0 is the right point for tool-construction tasks.
#
# The temperature value is recorded on every audit row via ComposerLLMCall.
# The seed is recorded when LiteLLM advertises support for the configured
# provider/model, otherwise it is omitted from the provider request and
# recorded as ``None`` so the audit row mirrors the actual request shape.
#
# Configurability is Tier 2 — do not read from settings/env without an ADR.
_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0
_COMPOSER_LLM_SEED: Final[int] = 42
_COMPOSER_LLM_SEED_PARAM: Final[str] = "seed"

type RequiredPath = tuple[str, ...]

# Bounded set of exception class names emitted as `exception_class` attribute on
# the runtime-preflight counter. Anything not in this set is bucketed as "other"
# to prevent unbounded cardinality from plugin class names leaking into metric labels.
_KNOWN_PREFLIGHT_EXCEPTION_CLASSES: frozenset[str] = frozenset(
    {
        "TimeoutError",
        "PluginNotFoundError",
        "PluginConfigError",
        "GraphValidationError",
        "ValidationError",  # pydantic.ValidationError
    }
)

# Module-level OTel counter for runtime preflight outcomes.
# Attributes: outcome (success | failure), exception_class (bounded closed-list | other)
_RUNTIME_PREFLIGHT_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.runtime_preflight.total",
    description="Total runtime-equivalent preflight invocations in the composer service",
)

# Module-level OTel counter for producer-side audit-integrity invariant violations.
# Increments before the AuditIntegrityError raise so an SRE has a count even if the
# uncaught exception kills the request before any other telemetry flushes. The crash
# is the right operational signal (CLAUDE.md: "Crash > silent wrong result"), but
# operators benefit from a counter trend ("how often is this firing?") to decide
# whether the field-level paydown at elspeth-7ae1732ab2 is becoming urgent.
# Attributes: invariant (augmentation_prefix | replacement_non_prefix), branch (the
# specific producer branch that violated; closed-list at type level via Literal).
_COMPOSER_AUDIT_INTEGRITY_VIOLATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.audit_integrity_violation.total",
    description="Producer-side audit-integrity invariant violations (augmentation/replacement prefix contracts)",
)


class _MalformedLLMResponseError(ComposerServiceError):
    """Internal carrier for malformed provider responses after the call returned."""

    def __init__(self, message: str, *, response: Any) -> None:
        super().__init__(message)
        self.response = response


class _BadRequestLLMError(ComposerServiceError):
    """Internal carrier for provider bad-request failures."""


class _ReasoningMetadata(TypedDict):
    reasoning_content: str | None
    reasoning_details: Any | None
    thinking_blocks: Any | None


async def _litellm_acompletion(**kwargs: Any) -> Any:
    """Call LiteLLM lazily so app startup never imports provider machinery."""
    import litellm

    return await litellm.acompletion(**kwargs)


def _litellm_completion_supports_param(model: str, param: str) -> bool:
    """Return whether LiteLLM advertises chat-completion support for ``param``."""
    import litellm

    supported_params = litellm.get_supported_openai_params(model=model)
    return isinstance(supported_params, list) and param in supported_params


def _composer_llm_seed_for_model(model: str) -> int | None:
    """Seed value to send to LiteLLM, or ``None`` when the provider rejects it."""
    if _litellm_completion_supports_param(model, _COMPOSER_LLM_SEED_PARAM):
        return _COMPOSER_LLM_SEED
    return None


def _provider_details_payload(value: Any, *, fields: tuple[str, ...]) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    return {field: getattr(value, field, None) for field in fields}


def _token_usage_from_response(response: Any | None) -> TokenUsage:
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


def _safe_response_model(response: Any | None) -> str | None:
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


def _build_llm_call_record(
    *,
    model_requested: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    status: ComposerLLMCallStatus,
    started_at: datetime,
    started_ns: int,
    temperature: float,
    seed: int | None,
    response: Any | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
) -> ComposerLLMCall:
    usage = _token_usage_from_response(response)
    provider_cost, provider_cost_source = _provider_cost_from_response(response)
    reasoning_metadata = _reasoning_metadata_from_response(response)
    return ComposerLLMCall(
        model_requested=model_requested,
        model_returned=_safe_response_model(response),
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


def _attach_llm_calls(exc: BaseException, recorder: BufferingRecorder | None) -> None:
    """Attach buffered LLM calls to exception objects that otherwise lack carriers."""
    if recorder is None:
        return
    exc_with_calls = cast(Any, exc)
    exc_with_calls.llm_calls = recorder.llm_calls


# ---------------------------------------------------------------------------
# Provider prompt-cache markers (elspeth-4e79436719 §Phase 3)
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


def _supports_anthropic_prompt_cache_markers(model: str | None) -> bool:
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


def _apply_anthropic_cache_markers(
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
        if message.get("role") == "system":
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


@dataclass(frozen=True, slots=True)
class _CompiledRequiredPath:
    """One schema-required path with conditional-on-presence semantics.

    JSON-Schema required semantics: a nested object's ``required`` list
    only applies when that object itself is present. The ``optional_ancestor``
    field captures the deepest path segment whose containing object is
    optional at its parent level — when any ancestor on that path is absent
    in the value, the inner ``path`` check short-circuits.

    For paths that are required at every level of the tree,
    ``optional_ancestor`` is the empty tuple and the path is enforced
    unconditionally.
    """

    path: RequiredPath
    optional_ancestor: RequiredPath = ()


def _collect_required_paths(
    schema: Mapping[str, object],
    prefix: RequiredPath = (),
    optional_ancestor: RequiredPath = (),
) -> tuple[_CompiledRequiredPath, ...]:
    """Compile schema-declared required fields into compiled-path records.

    The schema tree is system-owned tool metadata, so direct key access is
    intentional: a malformed tool definition should crash at import time.

    Each emitted :class:`_CompiledRequiredPath` carries the deepest optional
    ancestor seen on the way down. When the walker descends into a property
    that is NOT in the parent's ``required`` list, that property becomes the
    new ``optional_ancestor`` for everything emitted below it; the validator
    then short-circuits the inner check when that ancestor is absent in the
    value (correct JSON-Schema semantics: nested ``required`` applies only
    when the parent is present). Required-at-every-level paths keep an empty
    ``optional_ancestor`` and are enforced unconditionally.
    """
    schema_type = cast(str, schema["type"])

    if schema_type == "object":
        compiled: list[_CompiledRequiredPath] = []
        required_fields: set[str] = set()
        if "required" in schema:
            raw_required = cast(list[str], schema["required"])
            required_fields = set(raw_required)
            for field in raw_required:
                compiled.append(_CompiledRequiredPath(path=(*prefix, field), optional_ancestor=optional_ancestor))
        if "properties" in schema:
            properties = cast(Mapping[str, Mapping[str, object]], schema["properties"])
            for key, child_schema in properties.items():
                child_prefix = (*prefix, key)
                # If this property is NOT required at the current level, it
                # becomes the deepest optional ancestor for any nested-required
                # paths emitted below. If it IS required, propagate whatever
                # ancestor we already had (which is itself a required-at-this
                # -level path or empty).
                child_ancestor = optional_ancestor if key in required_fields else child_prefix
                compiled.extend(_collect_required_paths(child_schema, child_prefix, child_ancestor))
        return tuple(compiled)

    if schema_type == "array" and "items" in schema:
        item_schema = cast(Mapping[str, object], schema["items"])
        # Array items inherit the array's optional_ancestor: required fields
        # inside an item only matter if the array itself is present (and per
        # _find_missing_path_instances semantics, an empty array produces no
        # missing-path entries).
        return _collect_required_paths(item_schema, (*prefix, _ARRAY_ITEM_SEGMENT), optional_ancestor)

    return ()


def _build_tool_required_paths_index() -> dict[str, tuple[_CompiledRequiredPath, ...]]:
    """Build a lookup of required argument paths per tool definition."""
    index: dict[str, tuple[_CompiledRequiredPath, ...]] = {}
    for defn in get_tool_definitions():
        parameters = cast(Mapping[str, object], defn["parameters"])
        index[defn["name"]] = _collect_required_paths(parameters)
    return index


def _optional_ancestor_present(value: object, ancestor: RequiredPath) -> bool:
    """Walk down ``value`` along ``ancestor``; return False as soon as a segment is absent.

    Empty ``ancestor`` is the always-required case — treated as present.

    Today ``_collect_required_paths`` only sets ``optional_ancestor`` to a new
    path when descending into an OBJECT property that's not in ``required``;
    descending into array items propagates the existing ancestor unchanged.
    So array segments never appear in ``optional_ancestor`` under the current
    schema set. A future schema with an optional sub-object inside array items
    (e.g., ``tags: array<{ details?: { name: required } }>``) WOULD produce
    such an ancestor, and the all-or-nothing semantics here can't express
    "present in some items, absent in others." Per CLAUDE.md offensive
    programming: crash loudly with a diagnostic that points the maintainer at
    the extension site, rather than silently producing wrong validation.
    """
    if not ancestor:
        return True
    cursor: object = value
    for segment in ancestor:
        if segment == _ARRAY_ITEM_SEGMENT:
            raise AssertionError(
                "Array-segment in optional_ancestor is not yet supported by this walker. "
                f"Saw ancestor={ancestor!r}. To handle optional sub-objects inside array "
                "items, extend _find_missing_required_paths to evaluate ancestor presence "
                "per-array-item (rather than once globally) and update this walker to "
                "descend through array segments accordingly."
            )
        if not isinstance(cursor, Mapping) or segment not in cursor:
            return False
        cursor = cursor[segment]
    return True


def _find_missing_path_instances(
    value: object,
    required_path: RequiredPath,
    *,
    current_path: str = "",
) -> list[str]:
    """Return concrete missing-path instances for one required path."""
    if not required_path:
        return []

    head = required_path[0]
    tail = required_path[1:]

    if head == _ARRAY_ITEM_SEGMENT:
        match value:
            case list() as items:
                missing_paths: list[str] = []
                for index, item in enumerate(items):
                    item_path = f"{current_path}[{index}]" if current_path else f"[{index}]"
                    missing_paths.extend(_find_missing_path_instances(item, tail, current_path=item_path))
                return missing_paths
            case _:
                return []

    match value:
        case dict() as mapping:
            next_path = f"{current_path}.{head}" if current_path else head
            if head not in mapping:
                return [next_path]
            return _find_missing_path_instances(mapping[head], tail, current_path=next_path)
        case _:
            return []


def _find_missing_required_paths(
    value: object,
    required_paths: tuple[_CompiledRequiredPath, ...],
) -> list[str]:
    """Return dotted/indexed paths for missing schema-required fields.

    Skips any compiled path whose ``optional_ancestor`` is absent in the
    value: that mirrors JSON-Schema semantics where nested ``required``
    only applies when the containing optional object is itself present.
    """
    missing_paths: list[str] = []
    for compiled in required_paths:
        if not _optional_ancestor_present(value, compiled.optional_ancestor):
            continue
        missing_paths.extend(_find_missing_path_instances(value, compiled.path))
    return missing_paths


_TOOL_REQUIRED_PATHS: dict[str, tuple[_CompiledRequiredPath, ...]] = _build_tool_required_paths_index()


def _state_is_structurally_empty(state: CompositionState) -> bool:
    """Return True when no composition tools have produced visible state.

    Used by ``_finalize_no_tool_response`` to short-circuit the synthetic
    preflight-failed message: when the model gave up after failing to
    converge on a valid pipeline build, its prose is more truthful than
    the synthesizer's Pydantic-noise replacement, and we surface the prose
    instead.

    "Structurally empty" means no source, no nodes, no outputs — the three
    user-meaningful state fields. ``edges`` is implied (edges only exist
    between declared nodes) and ``metadata`` is not load-bearing for this
    check.
    """
    return state.source is None and not state.nodes and not state.outputs


def _tool_result_mutated_composition_state(
    *,
    version_before: int,
    result: ToolResult,
) -> bool:
    """Return True when a successful tool advanced the CompositionState version."""
    return result.success and result.updated_state.version > version_before


# Suffix appended to the model's prose when finalize-time runtime preflight
# fails on a structurally-empty state. Single source of truth so tests can
# pin the contract without duplicating the prose. Stable, system-attributed
# so a UI can detect and re-style it if desired.
_EMPTY_STATE_FINALIZE_SUFFIX = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] The pipeline is still empty — the composer did not "
    "complete a valid build this turn. To continue: refine your request "
    "with more specifics, or reply telling the composer to retry with the "
    "plan it described above."
)
_EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] The pipeline is still empty — the composer did not "
    "complete a valid build this turn.\n\nCause: {blocker}\n\n"
    "To continue: refine your request with more specifics, or reply telling "
    "the composer to retry with the plan it described above."
)

# Suffix appended to the model's prose when finalize-time runtime preflight
# fails on a non-empty state. Companion to the empty-state pair above; the
# difference is the suffix wording — empty-state asks the operator to refine
# or retry, non-empty-state names the validator's specific objection so the
# operator sees both the model's analysis and the validator's reason.
# Single source of truth for the contract (issue elspeth-9cfbad6901).
_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] Runtime preflight failed before this build could be "
    "marked complete.\n\nCause: {detail}{suggestion_block}\n\n"
    "The composer's analysis above is preserved verbatim; the validator's "
    "objection is recorded here."
)
_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] Runtime preflight failed before this build could be "
    "marked complete.\n\nThe composer's analysis above is preserved verbatim; "
    "the validator's objection is recorded here."
)


_BUILD_INTENT_PHRASES: Final[tuple[str, ...]] = (
    "set up",
    "setup",
    "build",
    "create",
    "make",
    "wire",
    "add",
    "update",
    "modify",
    "change",
    "run",
    "execute",
    "process",
    "route",
    "split",
    "save",
    "workflow",
    "automation",
    "pipeline",
    "runnable",
)
_INFORMATION_ONLY_PREFIXES: Final[tuple[str, ...]] = (
    "what ",
    "what's ",
    "which ",
    "why ",
    "how ",
    "explain ",
    "tell me ",
    "show me ",
    "list ",
)


_AugmentationBranch = Literal[
    "no_mutation_empty_state_augmentation",
    "preflight_invalid_empty_state_augmentation",
    "preflight_invalid_non_empty_state_augmentation",
    "state_claim_grounding_correction",
]


def _enforce_augmentation_prefix_invariant(
    *,
    branch: _AugmentationBranch,
    content: str,
    augmented: str,
) -> None:
    """Mechanically enforce the producer-side augmentation prefix contract.

    All composer synthesis shapes are augmentations — the consumer-side
    discriminator at ``routes._composer_history_content`` strips the
    operator-facing suffix from LLM history by detecting the structural
    property ``message.startswith(raw_content)``. A producer that breaks
    the prefix property would emit a row whose suffix cannot be
    distinguished from the model's own prose at read time; the LLM
    would see synthesized operator-facing text in its prior-turn
    history. Crash here rather than commit a corrupt audit row.

    Empty ``content`` is permitted — ``"".startswith("")`` is trivially
    True and the augmentation builders degenerate to suffix-only output
    for empty inputs. Non-empty ``content`` MUST appear at the start of
    ``augmented``.
    """
    if not augmented.startswith(content):
        _COMPOSER_AUDIT_INTEGRITY_VIOLATION_COUNTER.add(1, {"invariant": "augmentation_prefix", "branch": branch})
        raise AuditIntegrityError(
            f"Tier 1: composer augmentation contract violated on branch={branch!r}. "
            "Producer constructed an augmented message that does not have the "
            "model's pre-synthesis prose as a strict prefix. The consumer-side "
            "discriminator at routes._composer_history_content uses "
            "content.startswith(raw_content) to detect synthesis and strip "
            "the operator-facing suffix from LLM history; a producer that "
            "breaks the prefix property misroutes synthesized operator-facing "
            "text into the model's prior-turn history. Fix the augmented-"
            "message constructor so the prose appears verbatim at the start "
            "of message."
        )


def _compose_empty_state_message(content: str, *, blocker: str | None = None) -> str:
    """Build the user-facing message for the empty-state finalize path.

    Surfaces the model's content (which audit-DB inspection shows is
    typically an honest report of what the model tried and what blocked
    convergence) and appends a system-attributed suffix telling the user
    how to proceed.

    Two suffix templates are interpolated:
    ``_EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER`` when ``blocker`` is a
    non-empty string (carries ``Cause: {blocker}``);
    ``_EMPTY_STATE_FINALIZE_SUFFIX`` otherwise. Future maintainers
    extending the function MUST keep the two templates in sync —
    edits should generally apply to both.

    Args:
        content: The model's actual prose. Preserved verbatim at the start
            of the message. If the model produced no content at all (empty
            string), the suffix alone becomes the message — better than
            silence.
        blocker: When set to a non-empty string, the concrete cause that
            prevented mutation (e.g., a failed tool call's error). Included
            in the suffix so the operator gets the cause from the suffix
            even if the model's prose did not mention it. Used by the
            no-mutation empty-state augmentation; the preflight-invalid
            empty-state augmentation passes ``None`` because
            ``runtime_result`` already carries multi-error structured data.
            ``None`` and empty-string are treated identically (no blocker)
            to avoid emitting a degenerate ``Cause: \\n\\n`` suffix.
    """
    suffix = _EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER.format(blocker=blocker) if blocker else _EMPTY_STATE_FINALIZE_SUFFIX
    if not content:
        return suffix.lstrip("\n").lstrip("-").lstrip()
    return content + suffix


def _compose_preflight_failure_message(content: str, *, runtime_result: ValidationResult) -> str:
    """Build the user-facing message for the non-empty-state preflight-invalid path.

    Surfaces the model's prose verbatim (panel-evals evidence shows it
    typically carries substantive disclosure: removed nodes, chosen
    operational semantics, the model's own diagnosis — see issue
    elspeth-9cfbad6901). Appends a system-attributed suffix carrying the
    technical preflight error so the operator sees both the model's
    analysis and the validator's objection.

    Companion to ``_compose_empty_state_message`` (preflight-invalid
    empty-state branch). Both branches augment; the difference is the
    suffix wording — empty-state asks the operator to refine or retry,
    non-empty-state names the validator's specific objection.

    Suffix template selection (longest-detail-wins fallback chain):
    ``_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL`` when
    ``runtime_result`` carries a ValidationError or a failed check;
    ``_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE`` when neither.
    Future maintainers extending this function MUST keep the templates
    in sync — edits should generally apply to both.

    Args:
        content: The model's actual prose. Preserved verbatim at the
            start of the message. If empty, the suffix becomes the whole
            message (matches ``_compose_empty_state_message``); the
            augmentation prefix invariant holds trivially because every
            string startswith "".
        runtime_result: The failed ValidationResult from the final-gate
            preflight. The first ValidationError's ``message`` and
            optional ``suggestion`` populate the suffix; falls back to
            the first failed check's ``detail``; falls back to the bare
            template when the result has neither.

    Boundary contract: the suffix is echoed verbatim into chat history
    and the OpenAI-format chat completion. Secrets are guaranteed
    absent because validate_pipeline() resolves secret refs before
    settings load and validation errors are derived from typed plugin
    configs, never from raw secret values. However, the suffix MAY
    carry operator-supplied path fragments (the operator's own
    configured CSV paths, sink output paths) and exception text that
    names plugin config field values — both surface verbatim from
    ``ValidationError.message`` / ``ValidationCheck.detail``. In the
    current single-operator deployment model this is acceptable: the
    operator already knows their own paths and config. If ELSPETH ever
    takes a multi-tenant deployment shape, the suffix builder must be
    sanitized before merge.
    """
    detail: str | None = None
    suggestion: str | None = None
    if runtime_result.errors:
        first_error = runtime_result.errors[0]
        detail = first_error.message
        suggestion = first_error.suggestion
    else:
        failed_checks = [check for check in runtime_result.checks if not check.passed]
        if failed_checks:
            detail = failed_checks[0].detail
    if detail:
        suggestion_block = f"\n\nSuggested fix: {suggestion}" if suggestion else ""
        suffix = _PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL.format(
            detail=detail,
            suggestion_block=suggestion_block,
        )
    else:
        suffix = _PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE
    if not content:
        return suffix.lstrip("\n").lstrip("-").lstrip()
    return content + suffix


def _user_request_expects_pipeline_mutation(message: str) -> bool:
    """Return True when the user is asking the composer to build/edit/run."""
    normalized = " ".join(message.lower().split())
    if not normalized:
        return False
    if normalized.startswith(_INFORMATION_ONLY_PREFIXES) and "set up" not in normalized and "setup" not in normalized:
        return False
    return any(f" {phrase} " in f" {normalized} " for phrase in _BUILD_INTENT_PHRASES)


def _tool_failure_detail(payload: Mapping[str, Any]) -> str:
    """Extract a concise semantic failure detail from a ToolResult payload."""
    match payload:
        case {"data": {"error": error}}:
            return f": {error}"
        case {"validation": {"errors": [first_error, *_]}}:
            match first_error:
                case {"message": message}:
                    return f": {message}"
    if "error" in payload:
        return f": {payload['error']}"
    return "."


def _blocking_result_from_tool_invocations(tool_invocations: tuple[ComposerToolInvocation, ...]) -> str:
    """Name the most recent failed build/edit tool result, if one exists."""
    for invocation in reversed(tool_invocations):
        if invocation.status is ComposerToolStatus.ARG_ERROR:
            return f"{invocation.tool_name} failed before mutation ({invocation.error_class}: {invocation.error_message})."
        if invocation.status is ComposerToolStatus.PLUGIN_CRASH:
            return (
                f"{invocation.tool_name} crashed before a safe mutation completed ({invocation.error_class}: {invocation.error_message})."
            )
        if invocation.status is ComposerToolStatus.SUCCESS and invocation.result_canonical is not None:
            payload = json.loads(invocation.result_canonical)
            if type(payload) is dict and "success" in payload and payload["success"] is False:
                return f"{invocation.tool_name} returned success=false{_tool_failure_detail(payload)}"
            if (
                type(payload) is dict
                and "success" in payload
                and payload["success"] is True
                and invocation.version_after == invocation.version_before
                and not is_discovery_tool(invocation.tool_name)
            ):
                return f"{invocation.tool_name} succeeded without mutating CompositionState (version stayed {invocation.version_before})."
    return "the model ended the turn without calling any build/edit tool."


def _no_mutation_empty_state_validation(blocker: str) -> ValidationResult:
    """Build the synthetic final-gate result for empty-state no-mutation replies."""
    detail = f"No composition-state mutation completed successfully; state_exists=false. Blocking result: {blocker}"
    suggestion = (
        "Call set_pipeline with source.blob_id or source.inline_blob, call "
        "set_source_from_blob/set_source plus set_output, or ask for the specific "
        "missing file/configuration."
    )
    return ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name="state_exists", passed=False, detail=detail)],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message=detail,
                suggestion=suggestion,
            )
        ],
    )


@dataclass(frozen=True, slots=True)
class ComposerAvailability:
    """Boot-time availability snapshot for the composer service."""

    available: bool
    model: str
    provider: str | None
    reason: str | None = None
    missing_keys: tuple[str, ...] = ()


_PROVIDER_REQUIRED_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_API_KEY",),
    "azure_ai": ("AZURE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


@dataclass(frozen=True, slots=True)
class _CachedDiscoveryPayload:
    """State-independent portion of a cacheable discovery tool result."""

    success: bool
    affected_nodes: tuple[str, ...]
    data: Any


_RuntimePreflightCache = dict[RuntimePreflightKey, RuntimePreflightEntry]


# The per-dispatch audit envelope (DispatchAudit, begin_dispatch, finish_*)
# and the structural enforcement helper (dispatch_with_audit) live in
# web/composer/audit.py next to the BufferingRecorder. Hoisting them out of
# this module localises the "audit fires before return on every path"
# invariant inside a single helper rather than spreading it across seven
# procedural recorder.record() call sites in _compose_loop. See the audit.py
# module docstring for the contract details.


# Hard cap on proof-step-driven repair turns. When the assistant claims
# completion but preview_pipeline's proof_diagnostics still has blocking
# entries, the loop may inject a synthetic repair message and continue for
# at most this many additional iterations. After the cap, the original
# termination path runs — preventing indefinite spin against a model that
# refuses to apply the suggested repair.
_MAX_REPAIR_TURNS: Final[int] = 2


def _proof_repair_is_applicable(state: CompositionState) -> bool:
    """Return True iff the proof step has any input it can inspect.

    The forced-repair gate must fire whenever ``compute_proof_diagnostics``
    might find blocking diagnostics. The proof step is a no-op for sources
    that aren't blob-backed (no bytes to read), so the gate's predicate is
    "source is present AND options carries a ``blob_ref``" — *not* "state
    changed this turn", because a blocker can survive session resume into
    a turn where the LLM does no mutations.

    ``SourceSpec.options`` is internally typed as ``Mapping[str, Any]``
    (Tier-1 dataclass invariant — no isinstance probe needed). ``blob_ref``
    is an optional, well-known key set by the binding tools; its absence
    is a documented part of the contract (path-based sources don't have
    one), so containment checking is the appropriate primitive here.
    """
    if state.source is None:
        return False
    return "blob_ref" in state.source.options


class ComposerServiceImpl:
    """LLM-driven pipeline composer with dual-counter budget and discovery caching.

    Runs a bounded tool-use loop with separate budgets for discovery
    and composition turns. Cacheable discovery tool results are cached
    per-compose-call in a local dict (not an instance field) to avoid
    concurrent-request races.

    Budget classification: a turn containing at least one mutation tool
    call charges the composition budget. A turn containing only discovery
    tool calls charges the discovery budget. Cache hits do not charge
    any budget.

    Args:
        catalog: CatalogService for discovery tool delegation.
        settings: ComposerSettings with composer_max_composition_turns,
            composer_max_discovery_turns, composer_timeout_seconds,
            composer_model, data_dir.
    """

    def __init__(
        self,
        catalog: CatalogService,
        settings: ComposerSettings,
        *,
        sessions_service: SessionServiceProtocol | None = None,
        session_engine: Engine | None = None,
        secret_service: Any | None = None,
        runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
    ) -> None:
        self._catalog = catalog
        self._sessions_service = sessions_service
        self._model = settings.composer_model
        self._max_composition_turns = settings.composer_max_composition_turns
        self._max_discovery_turns = settings.composer_max_discovery_turns
        self._timeout_seconds = settings.composer_timeout_seconds
        self._data_dir: str = str(settings.data_dir)
        self._session_engine = session_engine
        self._secret_service = secret_service
        self._settings = settings
        self._runtime_preflight_timeout_seconds = settings.composer_runtime_preflight_timeout_seconds
        self._runtime_preflight_coordinator = runtime_preflight_coordinator or RuntimePreflightCoordinator()
        self._availability = self._compute_availability()

    def _require_sessions_service(self) -> SessionServiceProtocol:
        """Return the wired sessions service or fail at the persistence boundary."""

        if self._sessions_service is None:
            raise RuntimeError("sessions_service not wired")
        return self._sessions_service

    def get_availability(self) -> ComposerAvailability:
        """Return the boot-time composer availability snapshot."""
        return self._availability

    def _runtime_preflight(self, state: CompositionState, user_id: str | None) -> ValidationResult:
        return validate_pipeline(
            state,
            self._settings,
            yaml_generator,
            secret_service=self._secret_service,
            user_id=user_id,
        )

    def _new_runtime_preflight_cache(self) -> _RuntimePreflightCache:
        return {}

    def _raise_cached_runtime_preflight_failure(
        self,
        failure: RuntimePreflightFailure,
        *,
        state: CompositionState,
        initial_version: int,
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> NoReturn:
        raise ComposerRuntimePreflightError.capture(
            failure.original_exc,
            state=state,
            initial_version=initial_version,
            llm_calls=llm_calls,
        ) from failure.original_exc

    async def _cached_runtime_preflight(
        self,
        state: CompositionState,
        *,
        user_id: str | None,
        cache: _RuntimePreflightCache,
        initial_version: int,
        session_scope: str,
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> ValidationResult:
        key = RuntimePreflightKey(
            session_scope=session_scope,
            state_version=state.version,
            settings_hash=runtime_preflight_settings_hash(self._settings),
        )
        cached = cache.get(key)
        if isinstance(cached, ValidationResult):
            return cached
        if isinstance(cached, RuntimePreflightFailure):
            self._raise_cached_runtime_preflight_failure(
                cached,
                state=state,
                initial_version=initial_version,
                llm_calls=llm_calls,
            )

        async def worker() -> ValidationResult:
            return await asyncio.wait_for(
                run_sync_in_worker(self._runtime_preflight, state, user_id),
                timeout=self._runtime_preflight_timeout_seconds,
            )

        entry = await self._runtime_preflight_coordinator.run(key, worker)
        cache[key] = entry
        if isinstance(entry, RuntimePreflightFailure):
            exc_name = type(entry.original_exc).__name__
            exc_class = exc_name if exc_name in _KNOWN_PREFLIGHT_EXCEPTION_CLASSES else "other"
            _RUNTIME_PREFLIGHT_COUNTER.add(
                1,
                {"outcome": "failure", "exception_class": exc_class},
            )
            self._raise_cached_runtime_preflight_failure(
                entry,
                state=state,
                initial_version=initial_version,
                llm_calls=llm_calls,
            )
        _RUNTIME_PREFLIGHT_COUNTER.add(1, {"outcome": "success"})
        return entry

    def _attempt_proof_repair(
        self,
        *,
        state: CompositionState,
        llm_messages: list[dict[str, Any]],
        session_id: str | None,
        repair_turns_used: int,
    ) -> bool:
        """Pre-finalize proof gate.

        When the assistant emits no tool_calls (claiming completion), check
        ``preview_pipeline``'s ``proof_diagnostics`` for blocking entries.
        If any are found AND the repair-turn budget has not been exhausted,
        synthesize a user-attributed message describing each diagnostic plus
        its ``suggested_repair`` and append it to ``llm_messages``. The
        outer compose loop then continues for one more iteration so the
        model can apply the suggested fix.

        Returns True when a repair message was injected (the loop should
        ``continue`` and skip finalization). Returns False when there are
        no blocking diagnostics OR the repair budget is exhausted.

        Boundary contract: this helper NEVER catches plugin exceptions.
        It only repairs *configurations* via composer-tool calls. Plugin
        bugs (transform.process raising) propagate to the operator per the
        Plugin Ownership policy in CLAUDE.md.

        The synthesised message is appended verbatim into chat history. It
        contains operator-supplied column names and the diagnostic-message
        text (which may name CSV paths the operator wrote). No secrets are
        carried — proof_diagnostics never reads source bytes through any
        path that retains decoded content; only inspect_blob_content's
        bounded-summary facts are surfaced.
        """
        if repair_turns_used >= _MAX_REPAIR_TURNS:
            return False

        diagnostics = compute_proof_diagnostics(
            state,
            session_engine=self._session_engine,
            session_id=session_id,
        )
        # The diagnostic dict shape is the documented contract of
        # ``compute_proof_diagnostics`` (see ``tools.py``): every entry
        # has ``severity``, ``code``, ``message``, ``suggested_repair``,
        # ``evidence_locator``. This is an internal-package invariant,
        # not a Tier-3 trust boundary — a missing key is a bug in the
        # diagnostic builder, not malformed external data, so direct
        # subscript access is correct and a ``KeyError`` here is the
        # right failure mode (informative crash) per CLAUDE.md
        # offensive-programming policy. ``.get()`` fallbacks would bury
        # contract drift and ship ``[unknown]`` codes / empty messages
        # into the audit trail and the LLM's repair-message context.
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        if not blocking:
            return False

        # Cap at 3 blocking entries in the synthesised message to keep the
        # context window manageable. The model can call preview_pipeline to
        # see the full list.
        rendered = []
        for i, d in enumerate(blocking[:3], start=1):
            rendered.append(f"{i}. [{d['code']}] {d['message']}\n   Suggested repair: {d['suggested_repair']}")

        next_turn = repair_turns_used + 1
        budget_note = (
            f"This is forced repair turn {next_turn} of {_MAX_REPAIR_TURNS}. "
            "Apply the suggested repair via the appropriate composer tool, then call "
            "preview_pipeline to verify the diagnostics are cleared before finalising again."
        )

        message = (
            "[composer-system] Pre-finalisation proof step found blocking "
            "diagnostic(s) — the pipeline cannot run as currently configured. "
            "Do not respond to the user yet; resolve these first.\n\n" + "\n\n".join(rendered) + "\n\n" + budget_note
        )

        llm_messages.append({"role": "user", "content": message})
        return True

    async def _finalize_no_tool_response(
        self,
        *,
        content: str,
        state: CompositionState,
        initial_version: int,
        user_id: str | None,
        last_runtime_preflight: ValidationResult | None,
        runtime_preflight_cache: _RuntimePreflightCache,
        session_scope: str,
        user_message: str = "",
        mutation_success_seen: bool = False,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> ComposerResult:
        """Apply the deterministic final-gate check and build a ComposerResult.

        Four augmentation exit shapes are produced (the
        ``routes._composer_history_content`` discriminator depends on the
        ``content`` / ``raw_assistant_content`` relationship below):

        1. **No-mutation empty-state augmentation** — user asked for a
           build-style action, no successful mutation has been seen this
           turn, and the state is structurally empty. The model's prose
           is passed through verbatim with an operator-facing suffix
           appended (concrete blocker if a tool failed). ``content``
           starts with ``raw_assistant_content`` so the LLM keeps seeing
           its own prose on subsequent turns.
        2. **Preflight-invalid empty-state augmentation** — preflight
           is invalid AND the state is structurally empty. Same
           augmentation shape as (1); ``raw_assistant_content`` carries
           the unaugmented prose so the LLM context is unaffected by the
           operator-facing suffix.
        3. **Preflight-invalid non-empty-state augmentation** —
           preflight is invalid and the state is non-empty. Panel-evals
           evidence (issue elspeth-9cfbad6901) showed the model's prose
           in this case is typically substantive disclosure rather than
           a false completion claim — the model names removed nodes,
           chosen operational semantics, and its own diagnosis. The
           prose is preserved verbatim with a system-attributed suffix
           naming the validator's specific objection appended;
           ``raw_assistant_content`` carries the unaugmented prose so
           the LLM-context channel sees the model's own prose on
           subsequent turns. Self-correction continues to land via the
           model's next ``preview_pipeline`` call.

        When preflight is valid (or skipped because state did not
        change and ``last_runtime_preflight`` is ``None``) the
        ``check_state_claim_grounding`` content check runs to detect
        un-grounded prose (state-field contradictions or unmotivated
        action claims — see issues elspeth-c028f7d186 and
        elspeth-905fe2a3d8). When the check returns violations, the
        response is augmented with an ``[ELSPETH-SYSTEM]`` correction
        suffix and ``raw_assistant_content`` carries the unaugmented
        prose. Otherwise the response passes through unchanged and
        ``raw_assistant_content`` is left unset.

        Gate logic (no regex on natural-language text governs the
        routing-shape decision):
        - If ``state.version > initial_version`` (state changed this turn),
          run ``_cached_runtime_preflight`` for the current state.
        - Otherwise, reuse ``last_runtime_preflight`` from the most recent
          ``preview_pipeline`` call (may be ``None``).
        - Whichever of the two paths populated ``runtime_result``, the
          state-claim grounding check runs after preflight-invalid
          branches are dispatched. The grounding check's regex
          patterns are content-grounding (additive correction suffix),
          not routing-shape decisions.

        Args:
            content: The model's assistant prose for this turn.
            state: The post-tool composition state. ``state.version``
                versus ``initial_version`` decides whether preflight
                must re-run.
            initial_version: The composition version at turn start.
            user_id: Authenticated user identity for cache scoping.
            last_runtime_preflight: Most recent preflight outcome from
                ``preview_pipeline`` calls this turn; ``None`` if none.
            runtime_preflight_cache: Per-turn cache to avoid redundant
                preflight invocations on identical state.
            session_scope: Scope identifier for cache + telemetry.
            user_message: The user's message that triggered this turn.
                Used to detect "build-style" requests for the
                no-mutation empty-state augmentation path.
            mutation_success_seen: Whether any mutating tool call
                succeeded this turn. Suppresses the no-mutation
                augmentation path.
            tool_invocations: Tool calls made this turn; the most
                recent failure feeds the operator-facing blocker
                suffix.
            llm_calls: LLM call audit rows for this turn.

        Unexpected preflight exceptions (anything other than a
        ``RuntimePreflightFailure`` caught inside ``_cached_runtime_preflight``)
        are already converted to ``ComposerRuntimePreflightError`` with
        partial-state preservation by the coordinator's exception handling
        path — they are not caught here.
        """
        if _user_request_expects_pipeline_mutation(user_message) and not mutation_success_seen and _state_is_structurally_empty(state):
            # No-mutation empty-state augmentation. The model produced
            # honest diagnostic prose about what it tried and what blocked
            # convergence (audit-DB inspection across 2026-05-08 panel-cohort
            # cells confirms this). Pass the prose through and append a
            # system-attributed suffix carrying the concrete blocker — the
            # earlier full-replacement behavior hid the model's actual
            # output from both the user and (via routes._composer_history_content)
            # from the model itself on subsequent turns
            # (cf. elspeth-861b0c58f5).
            blocker = _blocking_result_from_tool_invocations(tool_invocations)
            empty_state_runtime_result = _no_mutation_empty_state_validation(blocker)
            augmented_message = _compose_empty_state_message(content, blocker=blocker)
            _enforce_augmentation_prefix_invariant(
                branch="no_mutation_empty_state_augmentation",
                content=content,
                augmented=augmented_message,
            )
            return ComposerResult(
                message=augmented_message,
                state=state,
                runtime_preflight=empty_state_runtime_result,
                raw_assistant_content=content,
                tool_invocations=tool_invocations,
                llm_calls=llm_calls,
            )

        runtime_result: ValidationResult | None = last_runtime_preflight
        if state.version > initial_version:
            runtime_result = await self._cached_runtime_preflight(
                state,
                user_id=user_id,
                cache=runtime_preflight_cache,
                initial_version=initial_version,
                session_scope=session_scope,
                llm_calls=llm_calls,
            )

        if runtime_result is not None and not runtime_result.is_valid:
            # Two finalize shapes for invalid preflight, dispatched on
            # state structure. Both augment — the difference is suffix
            # wording (issue elspeth-9cfbad6901 unified the policy after
            # the original replacement-on-non-empty-state branch was
            # found to discard substantive model disclosure):
            #
            # 1. Preflight-invalid empty-state augmentation: state is
            #    structurally empty. The model produced honest diagnostic
            #    prose about what it tried and what blocked convergence.
            #    Pass the prose through and append a system suffix asking
            #    the operator to refine or retry. raw_assistant_content
            #    carries the unaugmented prose so
            #    routes._composer_history_content replays it to the LLM
            #    on subsequent turns without the synthetic system text
            #    (cf. elspeth-861b0c58f5 — the original synthesizer-
            #    replaces-prose behavior corrupted both user view and
            #    LLM context).
            #
            # 2. Preflight-invalid non-empty-state augmentation: state
            #    has been populated AND preflight failed. Panel-evals
            #    evidence (fork_coalesce__p4_adversarial_engineer,
            #    boolean_routing__p3_marketingops; see issue
            #    elspeth-9cfbad6901) shows the model's prose in this
            #    case is typically substantive disclosure — what was
            #    attempted, removed nodes, chosen semantics — rather
            #    than a false completion claim. Pass the prose through
            #    and append a system suffix naming the validator's
            #    specific objection. The model's next preview_pipeline
            #    call surfaces the failure on the self-correction loop;
            #    the operator-facing suffix is stripped from LLM history
            #    so the LLM sees only its own prose.
            if _state_is_structurally_empty(state):
                augmented_message = _compose_empty_state_message(content)
                _enforce_augmentation_prefix_invariant(
                    branch="preflight_invalid_empty_state_augmentation",
                    content=content,
                    augmented=augmented_message,
                )
                return ComposerResult(
                    message=augmented_message,
                    state=state,
                    runtime_preflight=runtime_result,
                    raw_assistant_content=content,
                    tool_invocations=tool_invocations,
                    llm_calls=llm_calls,
                )
            augmented_message = _compose_preflight_failure_message(content, runtime_result=runtime_result)
            _enforce_augmentation_prefix_invariant(
                branch="preflight_invalid_non_empty_state_augmentation",
                content=content,
                augmented=augmented_message,
            )
            return ComposerResult(
                message=augmented_message,
                state=state,
                runtime_preflight=runtime_result,
                raw_assistant_content=content,
                tool_invocations=tool_invocations,
                llm_calls=llm_calls,
            )

        # State-claim grounding correction (Path 3 of issue elspeth-c028f7d186,
        # widened by issue elspeth-905fe2a3d8 to also catch verbal
        # acknowledgement without state mutation). The check runs in both
        # of the remaining cases:
        #
        #   - runtime_result is_valid: state has reached passing preflight
        #     with non-empty contents (T4 forward-contradiction case
        #     covered here).
        #   - runtime_result is None: state did not change AND no
        #     preview_pipeline was called this turn. Without grounding,
        #     the prior version of this function bare-passed-through —
        #     the panel-evals T5 case (model said "I just fixed it" with
        #     state unchanged from T4) and the cells #2/#4 cases (model
        #     said "you're right, I'll change that" with no mutation
        #     tool call) both landed in this hole. Running the grounding
        #     check here is what closes both bugs.
        #
        # The model's prose may contradict state — claiming a field has
        # its old value when the mutation already landed (forward
        # contradiction), claiming a fresh action when state was
        # unchanged (backward contradiction), or agreeing with the user
        # without acting (verbal acknowledgement). Detect and append an
        # [ELSPETH-SYSTEM] correction so amateur personas (compliance,
        # marketing-ops) cannot read confidently-wrong prose as
        # authoritative.
        #
        # Augmentation shape preserves the model's prose verbatim per the
        # ``_enforce_augmentation_prefix_invariant`` contract; the
        # raw_assistant_content field carries the original prose for the
        # LLM history-replay path (consistent with the empty-state
        # augmentation branches above). The natural-language regex used
        # by ``check_state_claim_grounding`` is content-grounding, not
        # gate routing — the gate decision (state non-empty, preflight
        # not failed) was already taken above without consulting prose.
        grounding_violations = check_state_claim_grounding(
            prose=content,
            state=state,
            mutation_success_seen=mutation_success_seen,
            state_changed=state.version > initial_version,
        )
        if grounding_violations:
            augmented_message = compose_grounded_message(
                prose=content,
                violations=grounding_violations,
            )
            _enforce_augmentation_prefix_invariant(
                branch="state_claim_grounding_correction",
                content=content,
                augmented=augmented_message,
            )
            return ComposerResult(
                message=augmented_message,
                state=state,
                runtime_preflight=runtime_result,
                raw_assistant_content=content,
                tool_invocations=tool_invocations,
                llm_calls=llm_calls,
            )

        return ComposerResult(
            message=content,
            state=state,
            runtime_preflight=runtime_result,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
        )

    async def explain_run_diagnostics(self, snapshot: Mapping[str, object]) -> str:
        """Return a plain-language explanation of a bounded run snapshot.

        The explanation is advisory UI text only: it does not call composer
        tools, mutate CompositionState, or persist chat messages.
        """
        if not self._availability.available:
            raise ComposerServiceError(self._availability.reason or "Composer is unavailable.")

        try:
            messages = build_run_diagnostics_messages(snapshot, data_dir=self._data_dir)
        except OSError as exc:
            raise ComposerServiceError(f"Failed to load deployment skill ({type(exc).__name__})") from exc

        try:
            from litellm.exceptions import APIError as LiteLLMAPIError

            response = await asyncio.wait_for(
                self._call_text_llm(messages),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            raise ComposerServiceError("Run diagnostics explanation timed out") from None
        except LiteLLMAPIError as exc:
            raise ComposerServiceError(f"LLM unavailable ({type(exc).__name__})") from exc

        content = cast(str | None, response.choices[0].message.content)
        if content is None or not content.strip():
            raise ComposerServiceError("LLM returned an empty diagnostics explanation")
        return content.strip()

    async def compose(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
        session_id: str | None = None,
        user_id: str | None = None,
        progress: ComposerProgressSink | None = None,
        guided_terminal: TerminalState | None = None,
    ) -> ComposerResult:
        """Run the LLM composition loop with dual-counter budget.

        Args:
            message: The user's chat message.
            messages: Chat history as plain dicts (pre-converted from
                ChatMessageRecord by route handler; seam contract B).
            state: The current CompositionState.
            guided_terminal: When set, the resolved TerminalState from the
                completed guided session; triggers the layered mode-transition
                prompt for this first freeform turn (spec §8.2). The caller
                is responsible for gate logic and ``transition_consumed`` flip.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If a budget is exhausted or
                the timeout is exceeded.
        """
        if not self._availability.available:
            raise ComposerServiceError(self._availability.reason or "Composer is unavailable.")

        deadline = asyncio.get_event_loop().time() + self._timeout_seconds
        from litellm.exceptions import APIError as LiteLLMAPIError

        try:
            return await self._compose_loop(message, messages, state, session_id, user_id, deadline, progress, guided_terminal)
        except ComposerConvergenceError as exc:
            await _emit_progress(
                progress,
                convergence_progress_event(budget_exhausted=exc.budget_exhausted),
            )
            # Has its own partial_state; route handler persists. Do not intercept.
            raise
        except ComposerPluginCrashError as crash:
            # Plugin-bug crash path. The exception already carries
            # partial_state (populated by _compose_loop at the execute_tool
            # site when state.version > initial_version), so the route
            # handler can persist the accumulated mutations into
            # composition_states symmetrically with the convergence path.
            #
            # Here we only add the session-row audit breadcrumb (updated_at
            # bump — richer crash-marker columns tracked as a follow-up
            # migration: elspeth-23b0987938).
            if self._session_engine is not None and session_id is not None:
                try:
                    # Offload to a worker — _persist_crashed_session
                    # executes a synchronous SQLAlchemy ``Engine.begin()``
                    # + UPDATE, which would otherwise block the event
                    # loop for the duration of the DB round-trip,
                    # stalling websocket heartbeats, rate-limit checks,
                    # and concurrent progress broadcasts. Symmetric with
                    # the execute_tool offload at the top of
                    # _compose_loop: every other sync DB path in this
                    # file runs through run_sync_in_worker, and this
                    # crash-path call was missed when it was hoisted
                    # out of the main loop.
                    await run_sync_in_worker(self._persist_crashed_session, session_id)
                except (SQLAlchemyError, OSError) as audit_failure:
                    # Audit-persistence is best-effort on the crash path —
                    # failure to persist MUST NOT mask the original plugin
                    # bug. Log via slog.error (audit system itself is failing
                    # here, which is one of the three permitted slog use
                    # cases per the logging-telemetry-policy skill).
                    #
                    # Catch is narrowed to (SQLAlchemyError, OSError) so that
                    # programmer-bug exceptions in _persist_crashed_session
                    # itself — RuntimeError from the engine guard,
                    # AttributeError from a drifted table column, TypeError
                    # from a signature change — propagate instead of being
                    # laundered as "audit failure". Mirrors the cleanup-
                    # rollback pattern in the ``fork_from_message`` route
                    # handler (web/sessions/routes.py); see also the
                    # tier-model enforcer entry for this call site.
                    #
                    # exc_info is deliberately omitted: the original plugin
                    # exception's message / __cause__ chain may carry DB
                    # URLs, filesystem paths, or secret fragments from
                    # deeper layers (the response-body redaction in
                    # routes.py exists for the same reason). The two
                    # exc_class fields give the operator enough correlation
                    # to triage from structured logs alone.
                    slog.error(
                        "composer_crash_persistence_failed",
                        session_id=session_id,
                        original_exc_class=crash.exc_class,
                        audit_exc_class=type(audit_failure).__name__,
                    )
            await _emit_progress(
                progress,
                ComposerProgressEvent(
                    phase="failed",
                    headline="The composer could not safely finish this request.",
                    evidence=("A pipeline tool failed on the server side.",),
                    likely_next="Review the visible error message, then retry after the issue is resolved.",
                    reason="plugin_crash",
                ),
            )
            raise
        except (ComposerServiceError, LiteLLMAPIError):
            # Generic service-level failure (prompt prep, availability check,
            # or a LiteLLMAPIError surfacing through the inner loop). The
            # route handlers further narrow LiteLLMAPIError into the
            # provider_unavailable progress code; here the service emits the
            # safe catch-all because we may not know which class fired.
            await _emit_progress(
                progress,
                ComposerProgressEvent(
                    phase="failed",
                    headline="The composer could not finish this request.",
                    evidence=("The model call or prompt preparation failed safely.",),
                    likely_next="Retry once the composer service is available.",
                    reason="service_setup_failed",
                ),
            )
            raise

    async def _compose_loop(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
        session_id: str | None = None,
        user_id: str | None = None,
        deadline: float = 0.0,
        progress: ComposerProgressSink | None = None,
        guided_terminal: TerminalState | None = None,
    ) -> ComposerResult:
        """Inner composition loop with dual-counter budget tracking.

        Uses cooperative timeout: the deadline is checked at safe
        checkpoints (before LLM calls, after tool batches) rather
        than using asyncio.wait_for() cancellation.  This ensures
        tool calls that have filesystem/DB side effects always run
        to completion with their state published — no split between
        committed side effects and the response.

        LLM calls are wrapped in per-call asyncio.wait_for(remaining)
        because they are pure network I/O with no side effects and
        can be safely cancelled.

        Args:
            guided_terminal: When set, this is the first freeform turn after
                guided-mode exit; the layered transition prompt is used.
        """
        initial_version = state.version
        llm_messages = self._build_messages(messages, state, message, guided_terminal)
        tools = self._get_litellm_tools()
        # Per-call audit recorder. Surfaced on ComposerResult and on
        # the three partial-state-carrier exceptions so the route handler
        # always has the per-call decision trail — including failure paths.
        recorder = BufferingRecorder()
        # Stable actor string for every invocation in this compose() call.
        # Falls back to "anonymous" when user_id is None (CLI/test paths);
        # the real web composer always has user_id from auth dependency.
        actor = f"composer-web:user-{user_id}" if user_id is not None else "composer-web:anonymous"
        await _emit_progress(
            progress,
            ComposerProgressEvent(
                phase="starting",
                headline="I'm reading your request and current pipeline.",
                evidence=(
                    "The current pipeline state is prepared for the composer.",
                    "The pipeline composer skill pack and deployment overlay are included.",
                ),
                likely_next="ELSPETH will ask the model for the next safe pipeline action.",
            ),
        )

        composition_turns_used = 0
        discovery_turns_used = 0
        mutation_success_seen = False

        # Discovery cache: local variable scoped to this compose() call.
        # Keyed by (tool_name, canonical_args_json). Each concurrent
        # compose() call gets its own independent cache dict.
        discovery_cache: dict[str, _CachedDiscoveryPayload] = {}

        # Validation threading: compute once for the initial state, then
        # carry forward from each ToolResult.validation. Avoids redundant
        # validate() calls — CompositionState is immutable so validation
        # is deterministic for a given state object.
        last_validation: ValidationSummary | None = None

        # Runtime preflight cache: scoped to this compose() call. Keyed by
        # (session_scope, state_version, settings_hash). A timeout or failure
        # is cached for the lifetime of this compose call so subsequent
        # preview_pipeline calls don't re-fire an already-failed worker.
        runtime_preflight_cache = self._new_runtime_preflight_cache()
        last_runtime_preflight: ValidationResult | None = None
        session_scope = f"session:{session_id}" if session_id is not None else "session:unsaved"

        # §7.7 anti-anchor tracker: detects 3-in-a-row identical failed tool
        # calls and injects a STRUCTURAL HINT before the next LLM turn so the
        # model breaks out of the anchored-loop pattern observed in the Tier 1
        # final cohort's residual RED. Per-compose-call instance — never
        # shared across requests.
        anti_anchor = AntiAnchorTracker()

        # Advisor escape-hatch budget. Local to this compose() call —
        # each fresh user request starts with the full configured budget.
        # Per-compose-request scope (matching the setting name
        # ``composer_advisor_max_calls_per_compose``) is the useful budget:
        # an LLM that breaks out of an anchored loop in one request should
        # not have its budget penalised in the next. There is intentionally
        # no session-lifetime cap; ``composer_rate_limit_per_minute`` and
        # the per-compose budget together bound advisor cost. When the
        # toggle is disabled the counter is never read.
        advisor_calls_used = 0

        # Forced-repair counter. When the assistant emits no tool_calls but
        # the proof step found blocking diagnostics, the loop synthesises a
        # repair message and continues for at most _MAX_REPAIR_TURNS
        # additional iterations. NEVER catches plugin exceptions — only
        # configuration diagnostics.
        repair_turns_used = 0
        persisted_assistant_message_id: str | None = None
        persisted_tool_call_turn = False
        failed_turn: FailedTurnMetadata | None = None

        while True:
            # Phase 3 Step 1 captures the state id observed before the
            # provider call. Step 2 passes this exact value to
            # persist_compose_turn_async as expected_current_state_id.
            current_state_id: str | None = None
            await _emit_progress(progress, _model_call_progress_event(message))
            response = await self._call_llm_before_deadline(
                llm_messages,
                tools,
                state,
                initial_version,
                deadline,
                recorder=recorder,
            )
            assistant_message = response.choices[0].message
            raw_assistant_content = assistant_message.content
            assistant_tool_calls = assistant_message.tool_calls or ()
            if len(assistant_tool_calls) > self._max_tool_calls_per_turn:  # type: ignore[attr-defined]
                telemetry = self._telemetry  # type: ignore[attr-defined]
                if telemetry is not None:
                    telemetry.tool_call_cap_exceeded_total.add(1)
                raise ComposerConvergenceError.capture(
                    max_turns=composition_turns_used + discovery_turns_used,
                    budget_exhausted="composition",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=recorder.invocations,
                    llm_calls=recorder.llm_calls,
                    reason="tool_call_cap_exceeded",
                    evidence={
                        "observed": len(assistant_tool_calls),
                        "cap": self._max_tool_calls_per_turn,  # type: ignore[attr-defined]
                    },
                )

            # If no tool calls, the LLM is done — apply the final gate and return
            if not assistant_message.tool_calls:
                # Forced-repair gate: when the model claims completion but
                # the proof step still has blocking diagnostics, inject a
                # repair message and continue. Capped at _MAX_REPAIR_TURNS so
                # the loop can never spin indefinitely. NEVER catches plugin
                # exceptions — only repairs configurations.
                #
                # The gate fires whenever the proof step is applicable —
                # i.e. there is a blob-backed source to inspect. The earlier
                # ``state.version > initial_version`` guard skipped the gate
                # on the first compose turn of a resumed session whose
                # blob-backed source was bound on a prior turn (state already
                # carries the source, no mutation this turn). That is exactly
                # the cross-turn scenario the gate exists to catch (e.g.
                # ``csv_fixed_schema_omits_observed_columns`` blockers
                # surviving session resume). For chat-only turns where the
                # source is absent or not blob-backed, ``_attempt_proof_repair``
                # short-circuits cheaply via ``compute_proof_diagnostics``'s
                # own early return.
                if _proof_repair_is_applicable(state) and self._attempt_proof_repair(
                    state=state,
                    llm_messages=llm_messages,
                    session_id=session_id,
                    repair_turns_used=repair_turns_used,
                ):
                    repair_turns_used += 1
                    continue

                await _emit_progress(
                    progress,
                    ComposerProgressEvent(
                        phase="complete",
                        headline="The composer response is ready.",
                        evidence=("The model did not request any more pipeline tools.",),
                        likely_next="ELSPETH will save any accepted pipeline update.",
                        reason="composer_complete",
                    ),
                )
                result = await self._finalize_no_tool_response(
                    content=assistant_message.content or "",
                    state=state,
                    initial_version=initial_version,
                    user_id=user_id,
                    last_runtime_preflight=last_runtime_preflight,
                    runtime_preflight_cache=runtime_preflight_cache,
                    session_scope=session_scope,
                    user_message=message,
                    mutation_success_seen=mutation_success_seen,
                    tool_invocations=recorder.invocations,
                    llm_calls=recorder.llm_calls,
                )
                # Thread repair_turns_used through to the result so the
                # route handler can persist it onto the new
                # ``composition_states.composer_meta`` row (and the API state
                # response can surface ``composer_meta.repair_turns_used``)
                # — see web/sessions/routes.py::_state_data_from_composer_state
                # call sites in the compose / recompose paths.
                return replace(
                    result,
                    repair_turns_used=repair_turns_used,
                    persisted_assistant_message_id=persisted_assistant_message_id,
                    persisted_tool_call_turn=persisted_tool_call_turn,
                )

            await _emit_progress(
                progress,
                _tool_batch_progress_event(
                    tuple(tool_call.function.name for tool_call in assistant_message.tool_calls),
                ),
            )

            # Append the assistant message (with tool_calls metadata)
            llm_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            # Execute each tool call, tracking whether this turn has
            # budgeted discovery or mutation work. Advisor-only turns are
            # intentionally neither: they spend the advisor-specific budget,
            # not the generic discovery/composition turn budgets.
            turn_has_mutation = False
            turn_has_discovery = False
            all_cache_hits = True
            # Step 1 — execute tool calls in async land while accumulating
            # immutable _ToolOutcome records. Step 2 performs audit writes
            # from this list; cancellation before Step 2 leaves the DB
            # unchanged for the current assistant response.
            from elspeth.web.sessions._persist_payload import _ToolOutcome

            tool_outcomes: list[_ToolOutcome] = []
            plugin_crash: ComposerPluginCrashError | None = None
            plugin_crash_cause: Exception | None = None
            pre_state_id = current_state_id
            self._phase3_last_expected_current_state_id = pre_state_id
            decoded_args_by_call_id: dict[str, dict[str, Any]] = {}

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                pre_version = state.version

                def _append_tool_outcome(
                    *,
                    response: Any,
                    error_class: str | None,
                    error_message: str | None,
                    post_version: int,
                    _tool_outcomes: list[_ToolOutcome] = tool_outcomes,
                    _tool_call: Any = tool_call,
                    _pre_version: int = pre_version,
                ) -> None:
                    _tool_outcomes.append(
                        _ToolOutcome(
                            call=_tool_call,
                            response=response,
                            error_class=error_class,
                            error_message=error_message,
                            pre_version=_pre_version,
                            post_version=post_version,
                        )
                    )

                try:
                    decoded_arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    # Track budget class even when args are unparseable.
                    if is_discovery_tool(tool_name):
                        turn_has_discovery = True
                    else:
                        turn_has_mutation = True
                    # ARG_ERROR pre-dispatch site (1/3): JSON-decode failure.
                    # Open the audit envelope with the raw (pre-parse) string
                    # so the trail records what the LLM tried, even when it
                    # wasn't valid JSON. ``error_message`` is class-name only
                    # because ``str(exc)`` for JSONDecodeError can echo column
                    # offsets that reference the un-truncated raw bytes.
                    audit = begin_dispatch(
                        tool_call.id,
                        tool_name,
                        tool_call.function.arguments,
                        version_before=state.version,
                        actor=actor,
                    )
                    error_payload = {"error": f"Invalid JSON in arguments: {exc}"}
                    recorder.record(
                        finish_arg_error(
                            audit,
                            error_class=type(exc).__name__,
                            error_message=type(exc).__name__,
                            error_payload=error_payload,
                        )
                    )
                    _append_tool_outcome(
                        response=None,
                        error_class=type(exc).__name__,
                        error_message=type(exc).__name__,
                        post_version=state.version,
                    )
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_payload),
                        }
                    )
                    all_cache_hits = False
                    continue

                if not isinstance(decoded_arguments, dict):
                    if is_discovery_tool(tool_name):
                        turn_has_discovery = True
                    else:
                        turn_has_mutation = True
                    # ARG_ERROR pre-dispatch site (2/3): non-dict arguments.
                    # The LLM produced valid JSON but not a JSON object. The
                    # canonicalized record wraps the (possibly scalar/list)
                    # value under ``_decoded_non_object`` so the audit trail
                    # captures it deterministically.
                    audit, canonicalization_failed = begin_dispatch_or_arg_error(
                        tool_call.id,
                        tool_name,
                        {"_decoded_non_object": decoded_arguments},
                        version_before=state.version,
                        actor=actor,
                    )
                    if canonicalization_failed is None:
                        err_msg = f"Tool '{tool_name}' arguments must be a JSON object, got {type(decoded_arguments).__name__}."
                        error_class = "TypeError"
                        error_message = f"non-object arguments ({type(decoded_arguments).__name__})"
                    else:
                        err_msg = f"Tool '{tool_name}' arguments are not canonical JSON ({type(canonicalization_failed).__name__})."
                        error_class = type(canonicalization_failed).__name__
                        error_message = type(canonicalization_failed).__name__
                    error_payload = {"error": err_msg}
                    recorder.record(
                        finish_arg_error(
                            audit,
                            error_class=error_class,
                            error_message=error_message,
                            error_payload=error_payload,
                        )
                    )
                    _append_tool_outcome(
                        response=None,
                        error_class=error_class,
                        error_message=error_message,
                        post_version=state.version,
                    )
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_payload),
                        }
                    )
                    all_cache_hits = False
                    continue

                arguments = cast(dict[str, Any], decoded_arguments)
                decoded_args_by_call_id[tool_call.id] = arguments

                # Open the audit envelope ONCE per dispatch — the cache,
                # required-paths, ToolArgumentError, plugin-crash, and
                # success branches below all read from this envelope so
                # ``started_at``/``arguments_canonical``/``version_before``
                # are consistent regardless of which branch fires.
                audit, canonicalization_failed = begin_dispatch_or_arg_error(
                    tool_call.id,
                    tool_name,
                    arguments,
                    version_before=state.version,
                    actor=actor,
                )
                if canonicalization_failed is not None:
                    if is_discovery_tool(tool_name):
                        turn_has_discovery = True
                    else:
                        turn_has_mutation = True
                    error_payload = {
                        "error": f"Tool '{tool_name}' arguments are not canonical JSON ({type(canonicalization_failed).__name__})."
                    }
                    recorder.record(
                        finish_arg_error(
                            audit,
                            error_class=type(canonicalization_failed).__name__,
                            error_message=type(canonicalization_failed).__name__,
                            error_payload=error_payload,
                        )
                    )
                    _append_tool_outcome(
                        response=None,
                        error_class=type(canonicalization_failed).__name__,
                        error_message=type(canonicalization_failed).__name__,
                        post_version=state.version,
                    )
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_payload),
                        }
                    )
                    all_cache_hits = False
                    continue

                # Check discovery cache before executing
                if is_cacheable_discovery_tool(tool_name):
                    cache_key = _make_cache_key(tool_name, arguments)
                    if cache_key in discovery_cache:
                        # Cache hit — return cached result, no budget charge.
                        # Audit-recorded with cache_hit=True so the trail
                        # captures every LLM decision-point, even those
                        # served from cache without re-running the handler.
                        await _emit_progress(
                            progress,
                            ComposerProgressEvent(
                                phase="using_tools",
                                headline="I'm reusing recently checked tool information.",
                                evidence=("The same discovery request was already answered for this compose step.",),
                                likely_next="ELSPETH will continue from the cached tool result.",
                            ),
                        )
                        cached_result = _result_from_cached_discovery_payload(
                            state,
                            discovery_cache[cache_key],
                        )
                        cached_payload = {
                            "success": cached_result.success,
                            "data": cached_result.data,
                            "cache_hit": True,
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=cached_payload,
                                version_after=state.version,
                                cache_hit=True,
                            )
                        )
                        _append_tool_outcome(
                            response=cached_result,
                            error_class=None,
                            error_message=None,
                            post_version=state.version,
                        )
                        # Cache hits are exclusively for discovery tools
                        # (`is_cacheable_discovery_tool` gates above). A
                        # discovery success is an *observation* — the model
                        # gained schema knowledge but did not change state.
                        # The §7.7 anchor is broken only by mutation
                        # progress, so tracker stays untouched here.
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": _serialize_tool_result(cached_result),
                            }
                        )
                        continue

                all_cache_hits = False

                # Validate schema-declared required arguments at the
                # Tier 3 boundary BEFORE entering tool handler code.
                # This walks nested object/array schemas, so malformed
                # set_pipeline payloads like source.plugin omissions are
                # caught here; any KeyError that still escapes
                # execute_tool() is an internal bug and must crash.
                # Unknown tool names skip validation — execute_tool()
                # handles them with a failure result downstream.
                required_paths = _TOOL_REQUIRED_PATHS[tool_name] if tool_name in _TOOL_REQUIRED_PATHS else ()
                missing = _find_missing_required_paths(arguments, required_paths)
                if missing:
                    if is_discovery_tool(tool_name):
                        turn_has_discovery = True
                    else:
                        turn_has_mutation = True
                    # ARG_ERROR pre-dispatch site (3/3): schema-required
                    # paths missing. ``missing`` is a list of dotted/indexed
                    # path strings — operator-controlled schema field names,
                    # safe to echo verbatim.
                    err_msg = f"Tool '{tool_name}' missing required argument(s): {', '.join(missing)}"
                    error_payload = {"error": err_msg}
                    recorder.record(
                        finish_arg_error(
                            audit,
                            error_class="MissingRequiredPaths",
                            error_message=f"missing: {', '.join(missing)}",
                            error_payload=error_payload,
                        )
                    )
                    _append_tool_outcome(
                        response=None,
                        error_class="MissingRequiredPaths",
                        error_message=f"missing: {', '.join(missing)}",
                        post_version=state.version,
                    )
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_payload),
                        }
                    )
                    continue

                await _emit_progress(progress, _tool_started_progress_event(tool_name))

                # Advisor escape-hatch interception. The request_advisor_hint
                # tool is intercepted here BEFORE execute_tool() because the
                # action is an async LiteLLM call to a frontier model rather
                # than a sync state mutation. The tool is not registered in
                # _DISCOVERY_TOOLS or _MUTATION_TOOLS, so execute_tool() would
                # return "Unknown tool" if this branch did not handle it.
                #
                # Audit envelope was already opened above by
                # begin_dispatch_or_arg_error; this branch closes it with the
                # truthful dispatch status: ARG_ERROR for local advisor-argument
                # rejection, SUCCESS for completed policy/provider outcomes
                # whose semantic status is encoded in result_payload. The inner
                # LLM call is recorded separately via _call_advisor_with_audit
                # firing a ComposerLLMCall record.
                if tool_name == "request_advisor_hint":
                    # Successful advisor guidance is governed solely by the
                    # advisor budget so the composer can read it. Advisor
                    # policy/error feedback with no usable guidance is still
                    # a non-mutating correction turn, so it consumes discovery
                    # budget before the loop asks the primary model again.
                    if not self._settings.composer_advisor_enabled:
                        # Defense-in-depth: the tool was filtered out of
                        # _get_litellm_tools() but the LLM somehow named it
                        # anyway (replayed transcript, stale state, prompt
                        # injection). Fail without making any outbound call.
                        error_payload = {
                            "error": "request_advisor_hint is disabled on this deployment.",
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=error_payload,
                                version_after=state.version,
                            )
                        )
                        anti_anchor.record_failure(tool_name, audit.arguments_hash)
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(error_payload),
                            }
                        )
                        turn_has_discovery = True
                        continue

                    budget = self._settings.composer_advisor_max_calls_per_compose
                    if advisor_calls_used >= budget:
                        budget_payload = {
                            "status": "BUDGET_EXHAUSTED",
                            "budget_used": advisor_calls_used,
                            "budget_remaining": 0,
                            "guidance": (
                                f"Advisor budget exhausted ({advisor_calls_used}/{budget} calls "
                                "used this compose request). Return to the validator output and "
                                "the recovery cheat sheet — no more frontier hints are available "
                                "until the operator raises the budget or the next compose request."
                            ),
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=budget_payload,
                                version_after=state.version,
                            )
                        )
                        # Budget exhaustion is a structural signal back to
                        # the LLM, not a tool-failure pattern — do NOT count
                        # it for §7.7 anchor tracking, since the issue isn't
                        # the LLM repeating an identical request, it's the
                        # operator's policy refusing further hints.
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(budget_payload),
                            }
                        )
                        turn_has_discovery = True
                        continue

                    # F3: validate argument types and total prompt size at the
                    # Tier-3 trust boundary. _TOOL_REQUIRED_PATHS only checks
                    # key presence, not value shape. Without this check the
                    # LLM could send a non-list (silently iterated char-by-
                    # char) or a megabyte-scale value (unbounded provider
                    # cost). ARG_ERRORs do NOT consume advisor budget — no
                    # outbound call is made — but anti-anchor counts them
                    # so repeated identical bad-arg calls trigger the §7.7
                    # structural hint.
                    advisor_arg_error = self._validate_advisor_arguments(arguments)
                    if advisor_arg_error is not None:
                        recorder.record(
                            finish_arg_error(
                                audit,
                                error_class=str(advisor_arg_error["error_class"]),
                                error_message=str(advisor_arg_error["error"]),
                                error_payload=advisor_arg_error,
                            )
                        )
                        anti_anchor.record_failure(tool_name, audit.arguments_hash)
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(advisor_arg_error),
                            }
                        )
                        turn_has_discovery = True
                        continue

                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        timeout_payload: dict[str, Any] = {
                            "status": "COMPOSE_TIMEOUT",
                            "error": "Advisor call exceeded the remaining compose deadline.",
                            "error_class": "TimeoutError",
                            "budget_used": advisor_calls_used,
                            "budget_remaining": budget - advisor_calls_used,
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=timeout_payload,
                                version_after=state.version,
                            )
                        )
                        raise ComposerConvergenceError.capture(
                            max_turns=0,
                            budget_exhausted="timeout",
                            state=state,
                            initial_version=initial_version,
                            tool_invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                        )

                    advisor_timeout = self._settings.composer_advisor_timeout_seconds
                    effective_advisor_timeout = min(advisor_timeout, remaining)
                    advisor_deadline_limited = remaining <= advisor_timeout

                    # F2: consume budget BEFORE the outbound call, not after.
                    # The cost guard's purpose is to bound outbound LiteLLM
                    # calls regardless of outcome. Counting only successes
                    # would let a flaky provider rack up unlimited failed
                    # outbound calls until anti-anchor or the discovery-
                    # turn limit fired. ARG_ERROR (above) and pre-call compose
                    # timeout (above) do NOT consume advisor budget because no
                    # outbound advisor call is made.
                    advisor_calls_used += 1

                    try:
                        guidance, advisor_meta = await self._call_advisor_with_audit(
                            arguments,
                            recorder=recorder,
                            timeout=effective_advisor_timeout,
                        )
                    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                        # Lifecycle exceptions: do not absorb. Propagate so
                        # cancellation and shutdown work normally; the inner
                        # ComposerLLMCall record was already fired by the
                        # advisor method's finally block, so the audit trail
                        # captures the cancelled call regardless.
                        raise
                    except TimeoutError as advisor_exc:
                        if advisor_deadline_limited:
                            timeout_payload = {
                                "status": "COMPOSE_TIMEOUT",
                                "error": "Advisor call exceeded the remaining compose deadline.",
                                "error_class": "TimeoutError",
                                "budget_used": advisor_calls_used,
                                "budget_remaining": budget - advisor_calls_used,
                            }
                            recorder.record(
                                finish_success(
                                    audit,
                                    result_payload=timeout_payload,
                                    version_after=state.version,
                                )
                            )
                            raise ComposerConvergenceError.capture(
                                max_turns=0,
                                budget_exhausted="timeout",
                                state=state,
                                initial_version=initial_version,
                                tool_invocations=recorder.invocations,
                                llm_calls=recorder.llm_calls,
                            ) from None
                        # Advisor-specific timeout with compose budget still
                        # remaining: return structured tool feedback so the
                        # composer can continue within its global deadline.
                        advisor_error_payload = {
                            "status": "ADVISOR_ERROR",
                            "error": "Advisor call failed; no guidance returned.",
                            "error_class": type(advisor_exc).__name__,
                            "budget_used": advisor_calls_used,
                            "budget_remaining": budget - advisor_calls_used,
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=advisor_error_payload,
                                version_after=state.version,
                            )
                        )
                        anti_anchor.record_failure(tool_name, audit.arguments_hash)
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(advisor_error_payload),
                            }
                        )
                        turn_has_discovery = True
                        continue
                    except Exception as advisor_exc:
                        # Tier 3 boundary: outbound LLM call failed. Convert
                        # to a structured tool-result error so the composer
                        # LLM gets feedback rather than a silent stall. The
                        # inner ComposerLLMCall record was already fired by
                        # _call_advisor_with_audit's finally block, so the
                        # audit trail captures the failure mode regardless.
                        # Budget was already consumed above (F2) — the
                        # outbound call attempt counts whether or not it
                        # produced guidance.
                        advisor_error_payload = {
                            "status": "ADVISOR_ERROR",
                            "error": "Advisor call failed; no guidance returned.",
                            "error_class": type(advisor_exc).__name__,
                            "budget_used": advisor_calls_used,
                            "budget_remaining": budget - advisor_calls_used,
                        }
                        recorder.record(
                            finish_success(
                                audit,
                                result_payload=advisor_error_payload,
                                version_after=state.version,
                            )
                        )
                        # Advisor failure IS counted for §7.7 anchor
                        # tracking — repeated identical failed advisor
                        # calls indicate the LLM is spamming a broken
                        # prompt, exactly the pattern the tracker exists
                        # to break.
                        anti_anchor.record_failure(tool_name, audit.arguments_hash)
                        llm_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(advisor_error_payload),
                            }
                        )
                        turn_has_discovery = True
                        continue

                    success_payload = {
                        "status": "SUCCESS",
                        "guidance": guidance,
                        "model": advisor_meta["model"],
                        "prompt_tokens": advisor_meta["prompt_tokens"],
                        "completion_tokens": advisor_meta["completion_tokens"],
                        "cached_prompt_tokens": advisor_meta["cached_prompt_tokens"],
                        "advisor_latency_ms": advisor_meta["latency_ms"],
                        "budget_used": advisor_calls_used,
                        "budget_remaining": budget - advisor_calls_used,
                        "note": (
                            "ADVICE only — call the appropriate mutation tool to "
                            "apply any change. Do not echo this guidance back as "
                            "configuration without verifying it against the schema."
                        ),
                    }
                    recorder.record(
                        finish_success(
                            audit,
                            result_payload=success_payload,
                            version_after=state.version,
                        )
                    )
                    # Successful advisor call is progress, not a failure —
                    # the §7.7 tracker is reset for this tool name so a
                    # subsequent set_pipeline failure does not inherit a
                    # stale advisor anchor count.
                    anti_anchor.record_success()
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(success_payload),
                        }
                    )
                    continue

                # Precompute runtime preflight for preview_pipeline outside
                # the general side-effectful tool worker. This keeps
                # execute_tool() synchronous and bounds the async I/O cost
                # before it enters the worker thread pool.
                runtime_preflight_callback: RuntimePreflight | None = None
                if tool_name == "preview_pipeline":
                    try:
                        preview_preflight = await self._cached_runtime_preflight(
                            state,
                            user_id=user_id,
                            cache=runtime_preflight_cache,
                            initial_version=initial_version,
                            session_scope=session_scope,
                            llm_calls=recorder.llm_calls,
                        )
                    except ComposerRuntimePreflightError as preflight_exc:
                        recorder.record(finish_plugin_crash(audit, exc=preflight_exc.original_exc))
                        raise ComposerRuntimePreflightError.capture(
                            preflight_exc.original_exc,
                            state=state,
                            initial_version=initial_version,
                            tool_invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                        ) from preflight_exc.original_exc

                    def _make_preflight_callback(
                        _result: ValidationResult = preview_preflight,
                    ) -> RuntimePreflight:
                        def _callback(_state: CompositionState) -> ValidationResult:
                            return _result

                        return _callback

                    runtime_preflight_callback = _make_preflight_callback()

                # All tool calls are offloaded to a worker to avoid blocking
                # the event loop.
                # Blob and secret tools perform synchronous filesystem
                # writes and SQLAlchemy transactions that would otherwise
                # stall the single-process web server for all concurrent
                # requests (rate-limit checks, websocket heartbeats,
                # progress broadcasts).
                #
                # Cancel-safety: tool calls are NOT wrapped in
                # asyncio.wait_for — they always run to completion.
                # The cooperative deadline is checked BETWEEN operations
                # (before LLM calls, after tool batches), so side effects
                # and state publication are never split.  LLM calls use
                # per-call wait_for because they are pure network I/O
                # with no side effects.
                #
                # Tool handlers raise ToolArgumentError at Tier-3 boundaries
                # (LLM supplied wrong types, semantically invalid values,
                # or malformed encodings that cannot be coerced).  The
                # compose loop catches ONLY that class and feeds the error
                # back to the LLM for retry.
                #
                # Any other exception — TypeError, ValueError, UnicodeError,
                # KeyError, AttributeError — escaping execute_tool() is a
                # plugin bug (Tier 1/2) and MUST crash.  Per CLAUDE.md,
                # silently laundering a plugin bug as an LLM-argument error
                # is worse than crashing: it pollutes the audit trail with
                # a confident but wrong Tier-3 story, and the LLM's "retry"
                # cannot correct a fault in our own code.
                # Dispatch under the structural audit envelope. The helper
                # records exactly one ComposerToolInvocation before this
                # await returns or raises, on every path:
                #   SUCCESS         → finish_success (with sentinel-canonical
                #                     fallback if canonical_json raises)
                #   ARG_ERROR       → finish_arg_error (caller's except block
                #                     below builds the LLM message)
                #   PLUGIN_CRASH (narrow re-raise) → finish_plugin_crash, then
                #                     re-raised so the outer compose loop exits
                #   PLUGIN_CRASH (general) → finish_plugin_crash, then the
                #                     caller's except block below wraps with
                #                     ComposerPluginCrashError.capture()
                #
                # Closes panel-review blockers B1 (canonical_json failure on
                # success path silently bypassed audit) and B2 (narrow
                # re-raise exited the loop unrecorded).
                # Bind loop-locals as default arguments so the closures
                # capture this iteration's values. The compose loop
                # awaits the helper synchronously inside the same
                # iteration, so late binding would not actually misfire
                # — but `B023 do not let function definitions reference
                # late-bound loop variables` is the project's structural
                # safeguard against future refactors that batch
                # iterations or introduce concurrency. Same pattern the
                # adjacent `_make_preflight_callback` already uses for
                # ``preview_preflight``.
                async def _do_dispatch(
                    _tool_name: str = tool_name,
                    _arguments: dict[str, Any] = arguments,
                    _state: CompositionState = state,
                    _last_validation: ValidationSummary | None = last_validation,
                    _runtime_preflight_callback: RuntimePreflight | None = runtime_preflight_callback,
                ) -> Any:
                    return await run_sync_in_worker(
                        execute_tool,
                        _tool_name,
                        _arguments,
                        _state,
                        self._catalog,
                        data_dir=self._data_dir,
                        session_engine=self._session_engine,
                        session_id=session_id,
                        secret_service=self._secret_service,
                        user_id=user_id,
                        prior_validation=_last_validation,
                        runtime_preflight=_runtime_preflight_callback,
                    )

                # ``_arg_error_payload`` is a module-level helper (F2 — testable
                # without spinning up the full compose loop). The nested
                # factory below binds ``tool_name`` to match the dispatch
                # ``arg_error_payload_factory`` signature. Default-arg binding
                # captures the loop-local ``tool_name`` at definition time.
                def _arg_error_payload_factory(
                    _exc: ToolArgumentError,
                    _tool_name: str = tool_name,
                ) -> Mapping[str, Any]:
                    return _arg_error_payload(_exc, _tool_name)

                def _version_after(_result: Any) -> int:
                    # The handler's ToolResult carries the new state on
                    # ``updated_state``. Read here so the helper records
                    # the post-mutation version inside the recorder call.
                    return cast(int, _result.updated_state.version)

                try:
                    outcome = await dispatch_with_audit(
                        recorder=recorder,
                        audit=audit,
                        do_dispatch=_do_dispatch,
                        version_after_provider=_version_after,
                        arg_error_payload_factory=_arg_error_payload_factory,
                    )
                except ToolArgumentError as exc:
                    # The audit record was already written by the helper.
                    # Build the LLM-facing tool message and continue.
                    if is_discovery_tool(tool_name):
                        turn_has_discovery = True
                    else:
                        turn_has_mutation = True
                    # Trust-boundary redaction: the echoed message reaches the
                    # LLM API and (via audit) the Landscape. ToolArgumentError
                    # is structurally safe by construction — the keyword-only
                    # constructor accepts (argument, expected, actual_type)
                    # and composes args[0] from those fields alone, so the
                    # message cannot carry a raw LLM-supplied value. Belt-
                    # and-suspenders: read ``exc.args[0]`` rather than
                    # ``str(exc)`` so a future subclass that overrides
                    # ``__str__`` to embed ``__cause__`` context (which may
                    # carry DB URLs, filesystem paths, or secret fragments
                    # from deeper layers) cannot leak through this path.
                    # Handlers that use
                    # ``raise ToolArgumentError(...) from exc`` get the
                    # cause preserved on ``__cause__`` for debug/audit but
                    # NOT echoed to the LLM.
                    await _emit_progress(
                        progress,
                        ComposerProgressEvent(
                            phase="using_tools",
                            headline="A tool request needed correction.",
                            evidence=("The tool rejected the request shape without exposing raw values.",),
                            likely_next="ELSPETH will ask the model to adjust the visible tool request.",
                        ),
                    )
                    arg_error_payload = _arg_error_payload(exc, tool_name)
                    _append_tool_outcome(
                        response=None,
                        error_class="ToolArgumentError",
                        error_message=str(exc.args[0] if exc.args else "ToolArgumentError"),
                        post_version=state.version,
                    )
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(arg_error_payload),
                        }
                    )
                    continue
                except (AssertionError, MemoryError, RecursionError, SystemError):
                    # CLAUDE.md policy exception — DOCUMENTED DIVERGENCE.
                    #
                    # CLAUDE.md "Plugin Ownership" says a defective plugin
                    # MUST crash rather than be wrapped and laundered as a
                    # recoverable error.  The web server relaxes this for
                    # ordinary exception classes (see the wider except
                    # Exception below) because crashing the whole ASGI
                    # process on one bad request would take down every
                    # other concurrent session.
                    #
                    # The exceptions listed on this handler are NOT
                    # relaxed: they represent states where the interpreter
                    # or our own Tier-1 invariants are compromised and any
                    # subsequent work — including the partial-state
                    # persistence inside ``ComposerPluginCrashError.capture``
                    # — would be operating on potentially-poisoned memory
                    # or data.
                    #
                    # - AssertionError: a plain ``assert`` fired inside
                    #   plugin code.  Asserts encode Tier-1 invariants
                    #   (CLAUDE.md: "crash on any anomaly").  Writing the
                    #   composition_states row after an invariant failure
                    #   would persist data the invariant said was
                    #   impossible.
                    # - MemoryError / RecursionError: interpreter-level
                    #   resource exhaustion.  The subsequent DB write may
                    #   itself fail or corrupt state; better to unwind.
                    # - SystemError: CPython internal invariant breach.
                    #
                    # ``BaseException``-only classes (SystemExit,
                    # KeyboardInterrupt, GeneratorExit) already propagate
                    # through ``except Exception`` below without any
                    # handling here.
                    #
                    # Audit-record-then-raise discipline: ``dispatch_with_audit``
                    # writes a ``PLUGIN_CRASH`` invocation BEFORE the
                    # narrow-class exception leaves the helper. Building the
                    # invocation reads only pre-captured scalars from the
                    # frozen :class:`DispatchAudit` envelope plus
                    # ``type(exc).__name__`` — no poisoned memory work.
                    # Closes blocker B2 from the panel review (2026-05-04).
                    raise
                except AuditIntegrityError:
                    # Tier-1 audit invariant. Do not let the plugin-bug
                    # catch-all below launder it into ComposerPluginCrashError.
                    raise
                except Exception as tool_exc:
                    # Plugin-bug path: any exception class OTHER than
                    # ToolArgumentError escaping execute_tool() is a plugin
                    # bug (CLAUDE.md tier 1/2). Capture the loop-local
                    # ``state`` — which has been rebound to
                    # result.updated_state on every successful prior
                    # iteration — so the route layer can persist the
                    # accumulated mutations into composition_states before
                    # returning the 500. Without this, any tool call that
                    # successfully mutated state prior to the crash would
                    # be silently dropped from the state history.
                    #
                    # Web-server policy exception: CLAUDE.md says a
                    # defective plugin must crash.  In the pipeline engine
                    # (single-shot CLI process) that is straightforward —
                    # abort the run.  In the web server a single malformed
                    # request reaching a buggy tool handler would take the
                    # ASGI worker down and abort every other concurrent
                    # session, including audit writes, websocket progress
                    # streams, and unrelated users.  We wrap the exception
                    # into a typed ComposerPluginCrashError that surfaces
                    # to the operator as an HTTP 500 with
                    # ``type(exc).__name__`` in the structured log, and
                    # preserves the original on ``__cause__`` for the ASGI
                    # error machinery.  The handler directly above
                    # re-raises the narrow set of exception classes that
                    # MUST NOT be laundered, so the concession below is
                    # bounded.
                    #
                    # Wrap narrow-scope: only exceptions from the
                    # execute_tool call are wrapped here. Bugs in
                    # _call_llm_before_deadline / _build_messages surface
                    # through their own exception classes
                    # (ComposerServiceError, ComposerConvergenceError).
                    # Record PLUGIN_CRASH BEFORE raising — done structurally
                    # by ``dispatch_with_audit``. The ``capture()`` helper
                    # takes the recorder buffer (including the helper's
                    # final crash record) so the route handler's
                    # ``_handle_plugin_crash`` gets the complete sequence.
                    _append_tool_outcome(
                        response=None,
                        error_class=type(tool_exc).__name__,
                        error_message=str(tool_exc),
                        post_version=state.version,
                    )
                    plugin_crash = ComposerPluginCrashError.capture(
                        tool_exc,
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                    )
                    plugin_crash_cause = tool_exc
                    break

                # SUCCESS path — the helper already recorded
                # ComposerToolStatus.SUCCESS via finish_success (with the
                # sentinel-canonical fallback for non-finite floats /
                # non-serializable result types). Update loop-local state
                # from outcome.result and continue with the LLM-message
                # append.
                version_before_tool = state.version
                result = outcome.result
                state = result.updated_state
                last_validation = result.validation
                last_runtime_preflight = result.runtime_preflight or last_runtime_preflight
                # §7.7 anchor tracking. ``finish_success`` records the audit
                # invocation regardless of ``result.success`` — the dispatch
                # itself ran without raising. But for anchor purposes we look
                # at ToolResult.success: a set_pipeline that returned a
                # validation-rejected state is the dominant anchor pattern
                # observed in the Tier 1 RED, and indistinguishable from a
                # ToolArgumentError as far as the LLM's retry loop is concerned.
                #
                # Successes only break the anchor when they are MUTATION
                # successes. Discovery successes (get_plugin_schema, list_*,
                # get_pipeline_state) are observations — empirically the model
                # interleaves them between failed mutation retries (see the
                # smoke session 55895523-... where 4 set_pipeline failures
                # were broken up by 1 get_pipeline_state and 1 get_plugin_schema,
                # both successful, both irrelevant to whether the model has
                # progressed on the anchored mutation).
                #
                # Mutation means a CompositionState version advance here.
                # Blob-store side effects such as create_blob/update_blob are
                # useful work, but they do not make an empty pipeline exist.
                if result.success:
                    if _tool_result_mutated_composition_state(version_before=version_before_tool, result=result):
                        mutation_success_seen = True
                        anti_anchor.record_success()
                else:
                    anti_anchor.record_failure(tool_name, audit.arguments_hash)
                result_json = _serialize_tool_result(result)
                await _emit_progress(progress, _tool_completed_progress_event(tool_name, result.success))
                _append_tool_outcome(
                    response=result,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
                )

                # Cache cacheable discovery results
                if is_cacheable_discovery_tool(tool_name):
                    cache_key = _make_cache_key(tool_name, arguments)
                    discovery_cache[cache_key] = _cached_discovery_payload(result)

                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    }
                )

                if not is_discovery_tool(tool_name):
                    turn_has_mutation = True
                else:
                    turn_has_discovery = True

            self._phase3_last_tool_outcomes = tuple(tool_outcomes)
            # Step 2a — redact in async land. The walkers are pure and
            # non-blocking; building this payload before the sync persistence
            # dispatch keeps the eventual worker transaction narrow.
            from elspeth.web.composer.redaction import MANIFEST, redact_tool_call_arguments
            from elspeth.web.sessions._persist_payload import RedactedToolRow

            phase3_self = cast(Any, self)
            redaction_telemetry = phase3_self._redaction_telemetry
            redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...] = ()
            for tool_outcome in tool_outcomes:
                tc = tool_outcome.call
                if tc.id in decoded_args_by_call_id:
                    decoded_args = decoded_args_by_call_id[tc.id]
                else:
                    decoded_args = {"_raw_arguments": tc.function.arguments}
                if tc.function.name in MANIFEST:
                    persisted_arguments = redact_tool_call_arguments(
                        tc.function.name,
                        decoded_args,
                        telemetry=redaction_telemetry,
                    )
                else:
                    # Unknown tool names are Tier-3 LLM hallucinations handled
                    # by execute_tool as a semantic failure ToolResult. The
                    # manifest is intentionally closed, so do not call the
                    # walker for names it cannot know about.
                    persisted_arguments = decoded_args
                redacted_assistant_tool_calls = (
                    *redacted_assistant_tool_calls,
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": json.dumps(persisted_arguments),
                        },
                    },
                )
            redacted_tool_rows = tuple(
                RedactedToolRow(
                    tool_call_id=tool_outcome.call.id,
                    content=phase3_self._serialize_response_via_walker(tool_outcome, telemetry=redaction_telemetry),
                    composition_state_payload=(
                        phase3_self._state_payload_for_compose_turn_for_test(tool_outcome.response)
                        if tool_outcome.post_version > tool_outcome.pre_version
                        else None
                    ),
                )
                for tool_outcome in tool_outcomes
            )
            self._phase3_last_redacted_assistant_tool_calls = redacted_assistant_tool_calls
            self._phase3_last_redacted_tool_rows = redacted_tool_rows
            if session_id is not None:
                sessions_service = self._require_sessions_service()
                try:
                    audit_outcome = await sessions_service.persist_compose_turn_async(
                        session_id=session_id,
                        assistant_content=assistant_message.content or "",
                        raw_content=raw_assistant_content,
                        redacted_assistant_tool_calls=redacted_assistant_tool_calls,
                        redacted_tool_rows=redacted_tool_rows,
                        parent_composition_state_id=current_state_id,
                        expected_current_state_id=current_state_id,
                        writer_principal="compose_loop",
                        plugin_crash_pending=plugin_crash is not None,
                    )
                except AuditIntegrityError as exc:
                    exc.failed_turn = FailedTurnMetadata(
                        assistant_message_id=None,
                        tool_calls_attempted=len(assistant_tool_calls),
                        tool_responses_persisted=0,
                    )
                    raise
                self._phase3_last_audit_outcome = audit_outcome
                failed_turn = FailedTurnMetadata(
                    assistant_message_id=audit_outcome.assistant_id,
                    tool_calls_attempted=len(assistant_tool_calls),
                )
                persisted_assistant_message_id = audit_outcome.assistant_id
                persisted_tool_call_turn = True
            if plugin_crash is not None:
                if persisted_tool_call_turn:
                    persisted_plugin_crash = ComposerPluginCrashError.capture(
                        plugin_crash.original_exc,
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=(),
                        llm_calls=recorder.llm_calls,
                        failed_turn=failed_turn,
                    )
                    if plugin_crash_cause is None:
                        raise persisted_plugin_crash
                    raise persisted_plugin_crash from plugin_crash_cause
                if plugin_crash_cause is None:
                    raise plugin_crash
                raise plugin_crash from plugin_crash_cause

            # §7.7 anti-anchor hint: if the last 3 failed tool calls share the
            # same (tool_name, arguments_hash), the model has stopped reading
            # validator feedback. Inject a synthetic role="user" hint before
            # the next LLM turn so the model breaks the anchor. consume_fire()
            # clears the deque so the hint cannot re-fire on the same anchor.
            # Persisted via the normal llm_messages → chat_messages path; the
            # operator-visible audit row carries the [ELSPETH-SYSTEM-HINT]
            # marker so its system origin is unambiguous.
            if anti_anchor.should_fire():
                hint_text = anti_anchor.build_hint()
                anti_anchor.consume_fire()
                llm_messages.append({"role": "user", "content": hint_text})
                await _emit_progress(
                    progress,
                    ComposerProgressEvent(
                        phase="using_tools",
                        headline="ELSPETH detected an anchored retry pattern.",
                        evidence=(
                            "The last 3 tool calls used identical arguments and produced the same error.",
                            "A structural hint was injected to help the model converge.",
                        ),
                        likely_next="The model will see the hint and try a different argument shape.",
                    ),
                )

            # If ALL tool calls in this turn were cache hits, no budget
            # charge — continue to next turn without incrementing.
            if all_cache_hits:
                continue

            # Classify turn and charge the appropriate budget.
            # The current turn has already been executed (tool results
            # are in the message history). We increment first, then
            # check whether the budget is now exhausted. If so, we give
            # the LLM one last chance (B-4D-3) for composition, or
            # raise immediately for discovery (discovery exhaustion
            # doesn't benefit from a bonus call — no state was mutated).
            if turn_has_mutation:
                composition_turns_used += 1
                if composition_turns_used >= self._max_composition_turns:
                    # B-4D-3 fix: give the LLM one last chance to see the
                    # tool results and produce a text response.
                    await _emit_progress(progress, _model_call_progress_event(message))
                    response = await self._call_llm_before_deadline(
                        llm_messages,
                        tools,
                        state,
                        initial_version,
                        deadline,
                        recorder=recorder,
                    )
                    assistant_message = response.choices[0].message
                    if not assistant_message.tool_calls:
                        await _emit_progress(
                            progress,
                            ComposerProgressEvent(
                                phase="complete",
                                headline="The composer response is ready.",
                                evidence=("The model stopped requesting pipeline tools.",),
                                likely_next="ELSPETH will save any accepted pipeline update.",
                                reason="composer_complete",
                            ),
                        )
                        result = await self._finalize_no_tool_response(
                            content=assistant_message.content or "",
                            state=state,
                            initial_version=initial_version,
                            user_id=user_id,
                            last_runtime_preflight=last_runtime_preflight,
                            runtime_preflight_cache=runtime_preflight_cache,
                            session_scope=session_scope,
                            user_message=message,
                            mutation_success_seen=mutation_success_seen,
                            tool_invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                        )
                        return replace(
                            result,
                            persisted_assistant_message_id=persisted_assistant_message_id,
                            persisted_tool_call_turn=persisted_tool_call_turn,
                        )
                    raise ComposerConvergenceError.capture(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="composition",
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=() if persisted_tool_call_turn else recorder.invocations,
                        llm_calls=recorder.llm_calls,
                        failed_turn=failed_turn,
                    )
            elif turn_has_discovery:
                discovery_turns_used += 1
                if discovery_turns_used >= self._max_discovery_turns:
                    raise ComposerConvergenceError.capture(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="discovery",
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=() if persisted_tool_call_turn else recorder.invocations,
                        llm_calls=recorder.llm_calls,
                        failed_turn=failed_turn,
                    )
            else:
                # The only non-cache tool currently handled outside the
                # discovery/mutation registries is request_advisor_hint. It
                # has its own per-compose budget above, so give the primary
                # model the returned guidance instead of charging discovery.
                continue

    def _persist_crashed_session(self, session_id: str) -> None:
        """Best-effort timestamp bump to mark that a compose session crashed.

        NOTE: The sessions-table schema does not yet have a dedicated crash
        marker column. Bumping updated_at is the minimum viable breadcrumb
        until a migration adds (e.g.) a ``status`` or ``crashed_at`` column.
        The schema addition is tracked separately as elspeth-23b0987938;
        when that lands, this method expands to populate the new columns
        and its signature gains ``exc_class``.

        The crash's exc_class is NOT written to the session row — no column
        exists to hold it. The operator correlates the updated_at bump with
        the crash via the slog.error emission at the call site, which
        includes session_id and exc_class in structured fields.

        Signature intentionally minimal — only the data that actually gets
        persisted is accepted. When the schema migration lands, this
        method's signature expands to take last_state and exc_class, and
        callers are updated at that point. Today, the caller passes
        session_id and logs the rest via slog.

        The caller's outer try/except absorbs any failure — this method
        MUST NOT mask the original plugin-bug exception if persistence
        itself fails.
        """
        # Offensive guard (explicit raise, not assert): ``python -O`` strips
        # assert statements, so a caller that somehow reaches this method
        # with ``_session_engine is None`` would silently no-op under the
        # optimised interpreter — turning a recoverable audit failure into
        # a missed ``updated_at`` write with no trace.  A typed
        # ``RuntimeError`` always fires.
        if self._session_engine is None:
            raise RuntimeError("_persist_crashed_session must only be called when session_engine is set")
        now = datetime.now(UTC)
        with self._session_engine.begin() as conn:
            conn.execute(update(sessions_table).where(sessions_table.c.id == session_id).values(updated_at=now))

    def _build_messages(
        self,
        chat_history: list[dict[str, Any]],
        state: CompositionState,
        user_message: str,
        guided_terminal: TerminalState | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list. Returns a NEW list on every call.

        This is critical: the tool-use loop appends to this list during
        iteration. Returning a cached reference would cause cross-turn
        contamination.

        OSError from deployment skill loading (PermissionError,
        IsADirectoryError) is translated into ComposerServiceError so
        the route handler returns a structured 502 rather than a raw 500.

        The HTTP body carries only ``type(exc).__name__`` — NOT
        ``str(exc)`` — because ``OSError.__str__`` expands to a string
        that includes the absolute filename (``[Errno 13] Permission
        denied: '/var/lib/elspeth/data/skills/...'``) which would
        leak filesystem layout and the operator's data-dir path into
        the 502 response body.  Full detail including the filename is
        preserved via ``raise ... from exc`` for the ASGI / server-log
        machinery only.  Mirrors the redaction contract landed by
        commits 1a30d985 (SQLAlchemy 422 path) and 127417cb (sibling
        HTTP-path slog sites) — both narrow the HTTP surface to
        class-name-only while preserving structured server-side detail.

        Args:
            guided_terminal: When set, forward to ``build_messages`` so the
                layered mode-transition prompt is used for this turn.
        """
        try:
            return build_messages(
                chat_history=chat_history,
                state=state,
                user_message=user_message,
                catalog=self._catalog,
                data_dir=self._data_dir,
                advisor_enabled=self._settings.composer_advisor_enabled,
                guided_terminal=guided_terminal,
            )
        except OSError as exc:
            raise ComposerServiceError(f"Failed to load deployment skill ({type(exc).__name__})") from exc

    def _get_litellm_tools(self) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM function format.

        When ``composer_advisor_enabled`` is False (the default), the
        ``request_advisor_hint`` tool is filtered out of the LLM-visible
        list. This is the strongest off-switch available — the composer
        LLM never sees the tool name or description, so it cannot call
        it even if instructed to. The CLI MCP server (composer_mcp/) is
        not affected; advisor is web-composer only by design.
        """
        definitions = get_tool_definitions()
        if not self._settings.composer_advisor_enabled:
            definitions = [defn for defn in definitions if defn["name"] != "request_advisor_hint"]
        return [
            {
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                },
            }
            for defn in definitions
        ]

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking."""
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
        except LiteLLMBadRequestError as exc:
            raise _BadRequestLLMError(f"LLM request rejected ({type(exc).__name__})") from exc
        # Tier 3 boundary: LiteLLM can return empty choices on content-filter,
        # rate-limit, or malformed upstream responses.  Validate before callers
        # index into choices[0].
        if not response.choices:
            raise _MalformedLLMResponseError("LLM returned empty choices array — cannot continue composition", response=response)
        return response

    async def _call_text_llm(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """Call the LLM for non-tool text generation."""
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
        except LiteLLMBadRequestError as exc:
            raise _BadRequestLLMError(f"LLM request rejected ({type(exc).__name__})") from exc
        if not response.choices:
            raise _MalformedLLMResponseError("LLM returned empty choices array — cannot explain run diagnostics", response=response)
        return response

    def _validate_advisor_arguments(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Validate advisor tool arguments at the Tier-3 trust boundary.

        Returns ``None`` if valid; otherwise returns an ARG_ERROR payload
        ready to embed in the outer tool-result envelope.

        The compose-loop's ``_TOOL_REQUIRED_PATHS`` check upstream guarantees
        ``trigger``, ``problem_summary``, ``recent_errors``, and
        ``attempted_actions`` are present in ``arguments`` — but only their
        *presence*, not their
        type or size. Without this validator:

        - A non-list ``recent_errors`` would be silently iterated by Python
          (string → char-by-char, int → TypeError, dict → keys), producing
          a corrupt prompt that we would still pay full provider cost for.
        - A megabyte-scale value would be sent verbatim to LiteLLM,
          rendering ``composer_advisor_max_prompt_tokens`` (declared as a
          cap) into dead config — operators would believe they had a cap
          and they would not.

        Both are Tier-3 trust-boundary failures: the LLM is providing
        external input, and CLAUDE.md's tier model permits ``isinstance``
        checks (and other defensive validation) at this boundary. Anti-
        anchor tracking on the caller side ensures repeated identical
        ARG_ERRORs surface the §7.7 structural hint.
        """
        trigger = arguments["trigger"]
        if not isinstance(trigger, str):
            return {
                "status": "ARG_ERROR",
                "error": "trigger must be a string",
                "error_class": "TypeError",
            }
        if trigger not in ADVISOR_TRIGGER_VALUES:
            return {
                "status": "ARG_ERROR",
                "error": f"trigger must be one of: {', '.join(ADVISOR_TRIGGER_VALUES)}",
                "error_class": "ValueError",
            }

        if not isinstance(arguments["problem_summary"], str):
            return {
                "status": "ARG_ERROR",
                "error": "problem_summary must be a string",
                "error_class": "TypeError",
            }

        recent = arguments["recent_errors"]
        if not isinstance(recent, list) or not all(isinstance(e, str) for e in recent):
            return {
                "status": "ARG_ERROR",
                "error": "recent_errors must be a list of strings",
                "error_class": "TypeError",
            }

        attempted = arguments["attempted_actions"]
        if not isinstance(attempted, list) or not all(isinstance(a, str) for a in attempted):
            return {
                "status": "ARG_ERROR",
                "error": "attempted_actions must be a list of strings",
                "error_class": "TypeError",
            }

        if trigger == ADVISOR_TRIGGER_REACTIVE and (len(recent) < 2 or len(attempted) < 2):
            return {
                "status": "ARG_ERROR",
                "error": (
                    "reactive_validation_loop trigger requires at least two recent_errors "
                    "and two attempted_actions showing the unchanged validation loop"
                ),
                "error_class": "ValueError",
            }

        if "schema_excerpt" in arguments and arguments["schema_excerpt"] is not None:
            candidate = arguments["schema_excerpt"]
            if not isinstance(candidate, str):
                return {
                    "status": "ARG_ERROR",
                    "error": "schema_excerpt must be a string when provided",
                    "error_class": "TypeError",
                }

        # Approximate provider cost cap: rough 4 chars / token. Compute the
        # exact formatted user-message char count we would emit if the call
        # proceeded, including section labels, bullets, and newlines. The
        # fixed system side is bounded separately by the packaged skill plus
        # load_deployment_skill's byte cap; this setting bounds the
        # LLM-controlled variable part.
        total_chars = len(_build_advisor_user_message(arguments))
        char_cap = self._settings.composer_advisor_max_prompt_tokens * 4
        if total_chars > char_cap:
            return {
                "status": "ARG_ERROR",
                "error": (
                    f"prompt size {total_chars} chars exceeds cap {char_cap} chars "
                    f"(composer_advisor_max_prompt_tokens={self._settings.composer_advisor_max_prompt_tokens}). "
                    "Truncate your error/action lists or schema excerpt and retry."
                ),
                "error_class": "ValueError",
            }

        return None

    async def _call_advisor_with_audit(
        self,
        arguments: dict[str, Any],
        *,
        recorder: BufferingRecorder | None,
        timeout: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Phone the configured advisor (frontier) model for a hint.

        Builds a structured prompt from the composer LLM's stuck-message
        arguments (``problem_summary``, ``recent_errors``, ``attempted_actions``,
        optional ``schema_excerpt``), forwards it to ``composer_advisor_model``
        via LiteLLM as a text-only completion (no tools), and returns
        ``(guidance_text, metadata)``. The caller may pass ``timeout`` to
        bound the advisor-specific timeout by the compose-loop deadline. The
        metadata dict carries inner-LLM accounting (model returned,
        prompt/completion tokens, cached prompt tokens, latency) so the outer
        tool-result envelope can embed it for audit-trail completeness.

        A :class:`ComposerLLMCall` record is fired into ``recorder`` in
        the ``finally`` block so the audit captures failure modes
        (timeouts, auth errors, malformed responses) just as cleanly as
        the success path. The outer ``ComposerToolInvocation`` record is
        the caller's responsibility — the compose-loop interception
        wraps this call with ``finish_success`` either way.

        Anthropic prompt-cache markers are deliberately NOT applied here.
        Advisor calls now include the same composer skill stack as normal
        composer requests, but their model and accounting are independent
        from the primary composer path. If advisor prompt caching becomes
        required, add it with focused usage-accounting tests rather than
        inheriting the primary-composer marker placement by accident.
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        advisor_model = self._settings.composer_advisor_model
        configured_timeout = self._settings.composer_advisor_timeout_seconds
        effective_timeout = configured_timeout if timeout is None else min(configured_timeout, timeout)
        max_completion = self._settings.composer_advisor_max_completion_tokens

        system_msg = build_system_prompt(self._data_dir, advisor_enabled=True) + "\n\n" + _ADVISOR_SYSTEM_INSTRUCTIONS
        # Required fields (trigger, problem_summary, recent_errors,
        # attempted_actions) are validated by _TOOL_REQUIRED_PATHS before this
        # method runs, so direct dict access is sound. schema_excerpt is the
        # only optional field — we test "in arguments" rather than .get() to
        # keep the Tier-3 trust-boundary rules clean.
        user_msg = _build_advisor_user_message(arguments)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        advisor_seed = _composer_llm_seed_for_model(advisor_model)
        kwargs: dict[str, Any] = {
            "model": advisor_model,
            "messages": messages,
            "temperature": _COMPOSER_LLM_TEMPERATURE,
            "max_tokens": max_completion,
        }
        if advisor_seed is not None:
            kwargs[_COMPOSER_LLM_SEED_PARAM] = advisor_seed
        try:
            response = await asyncio.wait_for(
                _litellm_acompletion(**kwargs),
                timeout=effective_timeout,
            )
            if not response.choices:
                raise _MalformedLLMResponseError(
                    "Advisor returned empty choices array",
                    response=response,
                )
            # F4: validate content BEFORE marking SUCCESS. None / empty /
            # whitespace-only content (content-filter triggered, malformed
            # provider output, tool-call-only response) must classify as
            # MALFORMED_RESPONSE rather than fall through to SUCCESS-with-
            # empty-guidance. Empty success would consume budget and tell
            # the composer LLM "you got advice" while no information was
            # actually produced.
            raw_content = response.choices[0].message.content
            if raw_content is None or not str(raw_content).strip():
                raise _MalformedLLMResponseError(
                    "Advisor returned empty or whitespace-only content",
                    response=response,
                )
            guidance = raw_content
            status = ComposerLLMCallStatus.SUCCESS
            usage = _token_usage_from_response(response)
            metadata = {
                "model": _safe_response_model(response) or advisor_model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "latency_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
            return guidance, metadata
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMBadRequestError as exc:
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except _MalformedLLMResponseError as exc:
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            response = exc.response
            error_class = type(exc).__name__
            error_message = "malformed_response"
            raise
        except Exception as exc:
            # F5: catch-all so the inner ComposerLLMCall record always
            # lands in the audit trail, even for exception classes not
            # in the typed clauses above (httpx ConnectionError, codec
            # ValueError, etc.). Without this, ``status`` would stay
            # ``None`` and the finally block would skip
            # ``record_llm_call``, leaving an audit gap for exactly the
            # broad-except failure path the compose-loop interception
            # relies on. API_ERROR is the closest semantic for "unknown
            # provider-side / transport failure"; the exception class
            # name is preserved in ``error_class`` for forensic detail.
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    _build_llm_call_record(
                        model_requested=advisor_model,
                        messages=messages,
                        tools=None,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=_COMPOSER_LLM_TEMPERATURE,
                        seed=advisor_seed,
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                    )
                )
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    _attach_llm_calls(current_exc, recorder)

    async def _call_llm_with_audit(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout: float,
        recorder: BufferingRecorder | None,
    ) -> Any:
        """Call the composer LLM once and record an audit sidecar.

        For Anthropic-family providers, ``cache_control`` markers are
        applied to the stable first system message and the trailing tool
        before the call. Dynamic composer state lives in the later context
        system message, outside the stable prompt-cache breakpoint. The
        transformed payload is what flows to LiteLLM and what the audit
        ``messages_hash`` / ``tools_spec_hash`` record — the hash is over
        the bytes actually sent, so the audit row is truthful about the
        wire payload (elspeth-4e79436719 §Phase 3).
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        if _supports_anthropic_prompt_cache_markers(self._model):
            messages, tools_or_none = _apply_anthropic_cache_markers(messages, tools)
            tools = tools_or_none if tools_or_none is not None else tools

        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any | None = None
        error_class: str | None = None
        error_message: str | None = None
        try:
            response = await asyncio.wait_for(
                self._call_llm(messages, tools),
                timeout=timeout,
            )
            status = ComposerLLMCallStatus.SUCCESS
            return response
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            _attach_llm_calls(exc, recorder)
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            _attach_llm_calls(exc, recorder)
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            _attach_llm_calls(exc, recorder)
            raise
        except _MalformedLLMResponseError as exc:
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            response = exc.response
            error_class = type(exc).__name__
            error_message = "malformed_response"
            _attach_llm_calls(exc, recorder)
            raise
        except _BadRequestLLMError as exc:
            cause = exc.__cause__
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(cause).__name__ if cause is not None else type(exc).__name__
            error_message = error_class
            _attach_llm_calls(exc, recorder)
            raise
        except Exception as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            _attach_llm_calls(exc, recorder)
            raise
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    _build_llm_call_record(
                        model_requested=self._model,
                        messages=messages,
                        tools=tools,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=_COMPOSER_LLM_TEMPERATURE,
                        seed=_composer_llm_seed_for_model(self._model),
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                    )
                )
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    _attach_llm_calls(current_exc, recorder)

    async def _call_llm_before_deadline(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        state: CompositionState,
        initial_version: int,
        deadline: float,
        recorder: BufferingRecorder | None = None,
    ) -> Any:
        """Call the LLM with a per-call timeout derived from the deadline.

        LLM calls are pure network I/O with no side effects, so they
        are safe to cancel via asyncio.wait_for.  If the deadline has
        already passed or the call exceeds the remaining budget, raise
        ComposerConvergenceError with the current partial state.

        ``recorder`` is the in-flight :class:`BufferingRecorder` from
        :meth:`_compose_loop` (or ``None`` from test paths). When set,
        timeout-based ``ComposerConvergenceError`` raises include the
        buffer's ``tool_invocations`` so the route handler's audit
        persistence has the per-call decision trail even when the
        budget exhaustion was a wall-clock timeout (no LLM mutation
        in this final call).
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        def _captured_invocations() -> tuple[ComposerToolInvocation, ...]:
            return recorder.invocations if recorder is not None else ()

        def _captured_llm_calls() -> tuple[ComposerLLMCall, ...]:
            return recorder.llm_calls if recorder is not None else ()

        attempt = 0
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise ComposerConvergenceError.capture(
                    max_turns=0,
                    budget_exhausted="timeout",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=_captured_invocations(),
                    llm_calls=_captured_llm_calls(),
                )
            try:
                return await self._call_llm_with_audit(
                    messages,
                    tools,
                    timeout=remaining,
                    recorder=recorder,
                )
            except TimeoutError:
                raise ComposerConvergenceError.capture(
                    max_turns=0,
                    budget_exhausted="timeout",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=_captured_invocations(),
                    llm_calls=_captured_llm_calls(),
                ) from None
            except LiteLLMAuthError:
                raise
            except LiteLLMAPIError:
                attempt += 1
                if attempt >= _LLM_API_MAX_ATTEMPTS:
                    raise
                delay_seconds = _LLM_API_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                remaining_after_error = deadline - asyncio.get_event_loop().time()
                if remaining_after_error <= delay_seconds:
                    raise
                await asyncio.sleep(delay_seconds)

    def _compute_availability(self) -> ComposerAvailability:
        """Infer whether the configured model has the required env at boot.

        This is a configuration/readiness signal, not a network health check.
        Keep it side-effect-free: LiteLLM provider probing has observable
        startup side effects in web lifespans, while the actual composer call
        path still validates provider requests through LiteLLM.
        """
        provider = _infer_provider_from_model_name(self._model) or _infer_provider_from_unprefixed_model_name(self._model)
        if provider is None:
            return ComposerAvailability(
                available=False,
                model=self._model,
                provider=provider,
                reason=(
                    f"Composer model {self._model} is unavailable: provider could not be inferred. "
                    "Use a provider-prefixed model name or a recognized OpenAI/Anthropic model name."
                ),
            )

        if provider not in _PROVIDER_REQUIRED_ENV_KEYS:
            return ComposerAvailability(
                available=False,
                model=self._model,
                provider=provider,
                reason=f"Composer model {self._model} is unavailable: provider {provider!r} has no configured environment contract.",
            )
        required_keys = _PROVIDER_REQUIRED_ENV_KEYS[provider]

        missing_keys = tuple(key for key in required_keys if key not in os.environ or not os.environ[key])
        if not missing_keys:
            return ComposerAvailability(
                available=True,
                model=self._model,
                provider=provider,
            )

        missing = ", ".join(missing_keys)
        reason = f"Composer model {self._model} is unavailable: missing {missing}."

        return ComposerAvailability(
            available=False,
            model=self._model,
            provider=provider,
            reason=reason,
            missing_keys=missing_keys,
        )


def _infer_provider_from_model_name(model: str) -> str | None:
    """Infer provider from a provider-prefixed model string."""
    if "/" not in model:
        return None
    return model.split("/", 1)[0]


def _infer_provider_from_unprefixed_model_name(model: str) -> str | None:
    """Infer provider for common unprefixed model families."""
    normalized = model.lower()
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    return None


async def _emit_progress(
    progress: ComposerProgressSink | None,
    event: ComposerProgressEvent,
) -> None:
    """Emit provider-safe progress when a sink is available."""
    if progress is None:
        return
    await progress(event)


def _model_call_progress_event(message: str) -> ComposerProgressEvent:
    headline = "I'm asking the model to choose the next safe pipeline update."
    normalized = message.lower()
    if "html" in normalized and "json" in normalized:
        headline = "I'm asking the model to choose an HTML input and JSON output."
    return ComposerProgressEvent(
        phase="calling_model",
        headline=headline,
        evidence=("The composer is using the prepared prompt and visible pipeline state.",),
        likely_next="The model may answer directly or request safe pipeline tools.",
    )


def _tool_batch_progress_event(tool_names: tuple[str, ...]) -> ComposerProgressEvent:
    if any(_is_schema_or_catalog_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model requested plugin schemas.",
            evidence=("Checking available source, transform, and sink tools.",),
            likely_next="ELSPETH will use visible schemas to guide the pipeline shape.",
        )
    if any(name in {"get_pipeline_state", "preview_pipeline", "diff_pipeline"} for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is checking the current pipeline.",
            evidence=("Reading the visible pipeline graph and validation summary.",),
            likely_next="ELSPETH will compare the request with the current setup.",
        )
    if any(_is_secret_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is checking available secret references.",
            evidence=("Checking available secret references without reading secret values.",),
            likely_next="ELSPETH will keep any credential references deferred.",
        )
    if any(not is_discovery_tool(name) for name in tool_names):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The model is updating the pipeline graph.",
            evidence=("A pipeline-editing tool was requested.",),
            likely_next="ELSPETH will validate the result before saving it.",
        )
    return ComposerProgressEvent(
        phase="using_tools",
        headline="The model requested composer tool information.",
        evidence=("Checking visible composer tool results.",),
        likely_next="ELSPETH will continue from the tool response.",
    )


def _tool_started_progress_event(tool_name: str) -> ComposerProgressEvent:
    if _is_schema_or_catalog_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking available source, transform, and sink tools.",
            evidence=("Reading plugin names and schemas only.",),
            likely_next="ELSPETH will choose compatible pipeline components.",
        )
    if _is_secret_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking available secret references.",
            evidence=("Secret names can be checked; secret values stay hidden.",),
            likely_next="ELSPETH will wire only deferred secret references if needed.",
        )
    if is_discovery_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="I'm checking the current pipeline and tool context.",
            evidence=("Reading visible composer state.",),
            likely_next="ELSPETH will use the result to decide the next action.",
        )
    return ComposerProgressEvent(
        phase="using_tools",
        headline="I'm updating the pipeline graph.",
        evidence=("A pipeline-editing tool is running.",),
        likely_next="ELSPETH will validate the updated pipeline.",
    )


def _tool_completed_progress_event(tool_name: str, success: bool) -> ComposerProgressEvent:
    if not success:
        return ComposerProgressEvent(
            phase="using_tools",
            headline="A composer tool reported a visible blocker.",
            evidence=("The tool result was returned without exposing raw request values.",),
            likely_next="ELSPETH will ask the model to adjust the pipeline request.",
        )
    if is_discovery_tool(tool_name):
        return ComposerProgressEvent(
            phase="using_tools",
            headline="The requested tool information is ready.",
            evidence=(_safe_tool_evidence(tool_name),),
            likely_next="ELSPETH will continue with the visible result.",
        )
    return ComposerProgressEvent(
        phase="validating",
        headline="The composer has updated the pipeline and is validating the result.",
        evidence=("A pipeline-editing tool completed successfully.",),
        likely_next="ELSPETH will save the updated pipeline if it is accepted.",
    )


def _is_schema_or_catalog_tool(tool_name: str) -> bool:
    return tool_name in {
        "list_sources",
        "list_transforms",
        "list_sinks",
        "get_plugin_schema",
        "list_models",
    }


def _is_secret_tool(tool_name: str) -> bool:
    return tool_name in {"list_secret_refs", "validate_secret_ref", "wire_secret_ref"}


def _safe_tool_evidence(tool_name: str) -> str:
    if _is_schema_or_catalog_tool(tool_name):
        return "Checking available source, transform, and sink tools."
    if _is_secret_tool(tool_name):
        return "Checking available secret references without reading secret values."
    if tool_name in {"get_pipeline_state", "preview_pipeline", "diff_pipeline"}:
        return "Reading the visible pipeline graph and validation summary."
    return "Using visible composer tool output."


def _pydantic_default(obj: Any) -> Any:
    """JSON serializer fallback for Pydantic models in tool results."""
    try:
        return obj.model_dump()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable") from None


def _serialize_tool_result(result: Any) -> str:
    """Serialize a ToolResult to JSON, handling Pydantic models in data."""
    return json.dumps(result.to_dict(), default=_pydantic_default)


def _cached_discovery_payload(result: ToolResult) -> _CachedDiscoveryPayload:
    """Extract the state-independent fields from a cacheable discovery result."""
    return _CachedDiscoveryPayload(
        success=result.success,
        affected_nodes=result.affected_nodes,
        data=result.data,
    )


def _result_from_cached_discovery_payload(
    state: CompositionState,
    cached: _CachedDiscoveryPayload,
) -> ToolResult:
    """Rebuild a cached discovery result with the current state envelope."""
    return ToolResult(
        success=cached.success,
        updated_state=state,
        validation=state.validate(),
        affected_nodes=cached.affected_nodes,
        data=cached.data,
    )


def _make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name + arguments."""
    # Sort keys for determinism. Arguments are simple JSON-serializable
    # dicts from the LLM — no MappingProxyType or frozen containers.
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


_ADVISOR_SYSTEM_INSTRUCTIONS: Final[str] = (
    "Advisor mode:\n"
    "- You are advising another LLM (a pipeline composer) that is stuck while building an ELSPETH pipeline.\n"
    "- Use the composer skill context and any deployment overlay above as binding local policy.\n"
    "- Read the problem summary, the verbatim validator errors, and the actions already attempted.\n"
    "- Return ONE concrete actionable hint: name specific fields, suggest values, and point at schema sections if provided.\n"
    "- Do not write YAML, do not produce final configuration, do not claim authority.\n"
    "- Your response is ADVICE; the composer LLM will decide what to apply.\n"
    "- Be specific and brief: under 250 words."
)


def _build_advisor_user_message(arguments: Mapping[str, Any]) -> str:
    """Build the exact variable user message sent to the advisor LLM.

    The validation path uses this same helper for prompt-size accounting, so
    bullets, section labels, and newlines cannot drift from the wire payload.
    Callers validate the Tier-3 argument shapes before invoking this helper.
    """
    user_msg_parts: list[str] = [
        f"Advisor trigger: {arguments['trigger']}",
        f"Problem: {arguments['problem_summary']}",
    ]
    recent = cast(list[str], arguments["recent_errors"])
    if recent:
        joined = "\n".join(f"- {e}" for e in recent)
        user_msg_parts.append(f"\nRecent validator errors (most recent first):\n{joined}")
    attempted = cast(list[str], arguments["attempted_actions"])
    if attempted:
        joined = "\n".join(f"- {a}" for a in attempted)
        user_msg_parts.append(f"\nAlready attempted:\n{joined}")
    if "schema_excerpt" in arguments and arguments["schema_excerpt"]:
        user_msg_parts.append(f"\nRelevant schema excerpt:\n{cast(str, arguments['schema_excerpt'])}")
    return "\n".join(user_msg_parts)


# ---------------------------------------------------------------------------
# F2 — module-level ARG_ERROR payload helper.
#
# Defined at module scope (not nested inside the compose loop) so the helper
# is directly importable and testable from
# ``tests/unit/web/composer/test_audit_arg_error_validation_errors.py``.
# Placed at end-of-file so callers above it use a forward reference (resolved
# at call time, never at import); inserting a new module-level def in the
# middle of the file would rotate every downstream fingerprint and force a
# churn of allowlist re-keying that has nothing to do with this change.
# ---------------------------------------------------------------------------


def _arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]:
    """Build the structured payload for an ARG_ERROR audit record + LLM tool message.

    The payload serves two consumers (spec §4.2.6, F2 disposition):

    1. ``dispatch_with_audit`` canonicalizes this into ``result_canonical``
       on the ARG_ERROR audit record. Persisted in Tier-1 Landscape.
    2. The compose loop's ARG_ERROR handler ``json.dumps`` this and sends
       it back to the LLM as the ``role=tool`` content. Tier-3 echo.

    Fields
    ------
    ``error``
        Operator-safe, LLM-facing message. Composed from the
        ``ToolArgumentError`` ``args[0]`` (structurally safe by
        construction — see ``ToolArgumentError`` docstring) plus the
        operator-chosen tool name. NEVER contains a raw LLM-supplied
        value.

    ``validation_errors`` (present iff ``exc.__cause__`` is a
        ``pydantic.ValidationError``)
        Leak-safe canonicalization of the Pydantic chained cause —
        ``loc``/``msg``/``type`` only, ``input``/``url``/``ctx``
        stripped. Provides field-name detail for recovery flows in
        Phase 3+ that need to know which specific Pydantic field
        failed validation. Absent (key omitted) when there is no
        chained Pydantic cause or the chained cause is not a
        ``ValidationError``; recording an empty list has no audit
        value.
    """
    safe_message = exc.args[0] if exc.args else "tool argument error"
    payload: dict[str, Any] = {"error": f"Tool '{tool_name}' failed: {safe_message}"}
    validation_errors = canonicalize_pydantic_cause(exc.__cause__)
    if validation_errors is not None:
        payload["validation_errors"] = validation_errors
    return payload


# ---------------------------------------------------------------------------
# Phase 3 test-only compose-loop driver.
#
# Appended at end-of-file for the same reason as ``_arg_error_payload``:
# inserting helper code above the existing composer implementation rotates
# tier-model allowlist fingerprints for unrelated trust-boundary code.
# Task 6 moves the sessions-service dependency into the constructor when the
# production persistence path is wired; Task 0 only needs the explicit test
# harness and first-use guard.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComposeLoopTestResult:
    """Structured result returned by the Phase 3 one-turn test driver."""

    assistant_message: str
    tool_outcomes: tuple[Any, ...] = ()
    persisted_assistant_row: Any | None = None
    persisted_assistant_tool_calls: tuple[Any, ...] = ()
    persisted_tool_row_content: tuple[Any, ...] = ()

    @property
    def tool_outcomes_for_assertion(self) -> tuple[Any, ...]:
        """Backward-compatible assertion surface named by the Phase 3 plan."""

        return self.tool_outcomes


async def _phase3_run_one_turn_for_test(
    self: ComposerServiceImpl,
    *,
    llm: Any | None = None,
    session_id: str | None = None,
    initial_state: CompositionState | None = None,
    user_message_id: str | None = None,
) -> ComposeLoopTestResult:
    """Drive exactly one compose-loop turn for Phase 3 tests.

    Test-only helper: it bypasses HTTP route setup but exercises the
    same ``_compose_loop`` body, including ``_require_sessions_service()``.
    Missing ``sessions_service`` must therefore fail with
    ``RuntimeError("sessions_service not wired")``, not ``AttributeError``
    or a constructor ``TypeError``.
    """

    from elspeth.web.composer.state import PipelineMetadata

    del user_message_id
    self._require_sessions_service()
    state = initial_state or CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    resolved_session_id = session_id or "00000000-0000-0000-0000-000000000000"
    original_call_llm = self._call_llm

    async def _call_fake_llm(messages: Any, tools: Any) -> Any:
        if llm is None:
            return await original_call_llm(messages, tools)
        return await llm(messages, tools)

    self._call_llm = _call_fake_llm  # type: ignore[method-assign]
    try:
        result = await self._compose_loop(
            "Phase 3 one-turn test driver",
            [],
            state,
            session_id=resolved_session_id,
            deadline=asyncio.get_event_loop().time() + self._timeout_seconds,
        )
    finally:
        self._call_llm = original_call_llm  # type: ignore[method-assign]

    return ComposeLoopTestResult(
        assistant_message=result.message,
        tool_outcomes=tuple(self._phase3_last_tool_outcomes),
        persisted_assistant_tool_calls=tuple(self._phase3_last_redacted_assistant_tool_calls),
        persisted_tool_row_content=tuple(row.content for row in self._phase3_last_redacted_tool_rows),
    )


ComposerServiceImpl._run_one_turn_for_test = _phase3_run_one_turn_for_test  # type: ignore[attr-defined]


def _phase3_serialize_response_via_walker(
    self: ComposerServiceImpl,
    outcome: Any,
    *,
    telemetry: Any,
) -> str:
    """Serialize one Step 1 outcome through the Phase 2 response walker."""

    from elspeth.core.canonical import canonical_json
    from elspeth.web.composer.redaction import MANIFEST, redact_tool_call_response

    if outcome.error_class is None:
        result = cast(ToolResult, outcome.response)
        if outcome.call.function.name not in MANIFEST:
            return canonical_json(result.to_dict())
        redacted = redact_tool_call_response(
            tool_name=outcome.call.function.name,
            response=result.to_dict(),
            telemetry=telemetry,
        )
        return canonical_json(redacted)
    return canonical_json(
        {
            "error_class": outcome.error_class,
            "error_message": outcome.error_message,
        }
    )


def _phase3_state_payload_for_compose_turn_for_test(
    self: ComposerServiceImpl,
    response: Any,
) -> Any:
    """Build a StatePayload for the current interim Step 2 redacted row."""

    del self
    from elspeth.web.sessions._persist_payload import StatePayload
    from elspeth.web.sessions.protocol import CompositionStateData

    result = cast(ToolResult, response)
    state_d = result.updated_state.to_dict()
    return StatePayload(
        data=CompositionStateData(
            source=state_d["source"],
            nodes=state_d["nodes"],
            edges=state_d["edges"],
            outputs=state_d["outputs"],
            metadata_=state_d["metadata"],
            is_valid=result.validation.is_valid,
            validation_errors=tuple(error.message for error in result.validation.errors),
            composer_meta=None,
        ),
        # Phase 1 inserts composition state rows inside
        # persist_compose_turn under the session write lock and re-derives
        # lineage from per-session version ordering when this is None
        # (spec §5.7.1). The async loop deliberately does not fabricate a
        # predecessor id for a row that has not been allocated yet.
        derived_from_state_id=None,
    )


ComposerServiceImpl._serialize_response_via_walker = _phase3_serialize_response_via_walker  # type: ignore[attr-defined]
ComposerServiceImpl._state_payload_for_compose_turn_for_test = _phase3_state_payload_for_compose_turn_for_test  # type: ignore[attr-defined]

_PHASE3_ORIGINAL_COMPOSER_INIT = ComposerServiceImpl.__init__


def _phase3_composer_init(self: ComposerServiceImpl, *args: Any, **kwargs: Any) -> None:
    """Store the Phase 3 per-turn tool-call cap without moving class code."""

    _PHASE3_ORIGINAL_COMPOSER_INIT(self, *args, **kwargs)
    from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry

    try:
        self._max_tool_calls_per_turn = self._settings.composer_max_tool_calls_per_turn  # type: ignore[attr-defined]
    except AttributeError:
        self._max_tool_calls_per_turn = 16  # type: ignore[attr-defined]
    self._telemetry = None  # type: ignore[attr-defined]
    self._redaction_telemetry = OtelRedactionTelemetry()  # type: ignore[attr-defined]
    self._phase3_last_tool_outcomes = ()
    self._phase3_last_expected_current_state_id = None
    self._phase3_last_redacted_assistant_tool_calls = ()
    self._phase3_last_redacted_tool_rows = ()
    self._phase3_last_audit_outcome = None  # type: ignore[assignment]


ComposerServiceImpl.__init__ = _phase3_composer_init  # type: ignore[method-assign]
