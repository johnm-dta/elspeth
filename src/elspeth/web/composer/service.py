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
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn, cast

import structlog
from opentelemetry import metrics
from sqlalchemy import Engine, update
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation
from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.canonical import stable_hash
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.audit import (
    BufferingRecorder,
    begin_dispatch,
    begin_dispatch_or_arg_error,
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
from elspeth.web.composer.prompts import build_messages, build_run_diagnostics_messages
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
from elspeth.web.composer.tools import (
    RuntimePreflight,
    ToolResult,
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
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.sessions.models import sessions_table

slog = structlog.get_logger()

_ARRAY_ITEM_SEGMENT = "[]"
_LLM_API_MAX_ATTEMPTS = 3
_LLM_API_RETRY_BASE_DELAY_SECONDS = 1.0
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


class _MalformedLLMResponseError(ComposerServiceError):
    """Internal carrier for malformed provider responses after the call returned."""

    def __init__(self, message: str, *, response: Any) -> None:
        super().__init__(message)
        self.response = response


class _BadRequestLLMError(ComposerServiceError):
    """Internal carrier for provider bad-request failures."""


async def _litellm_acompletion(**kwargs: Any) -> Any:
    """Call LiteLLM lazily so app startup never imports provider machinery."""
    import litellm

    return await litellm.acompletion(**kwargs)


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
    """
    if response is None:
        return TokenUsage.unknown()
    usage = getattr(response, "usage", None)
    if isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
        usage_data = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "prompt_tokens_details": details if isinstance(details, Mapping) else None,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        }
    else:
        details_attr = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details_attr, Mapping):
            details_payload: Any = details_attr
        elif details_attr is not None:
            # OpenAI Pydantic-shaped objects expose cached_tokens as an attribute.
            details_payload = {"cached_tokens": getattr(details_attr, "cached_tokens", None)}
        else:
            details_payload = None
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "prompt_tokens_details": details_payload,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        }
    return TokenUsage.from_dict(usage_data)


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


def _build_llm_call_record(
    *,
    model_requested: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    status: ComposerLLMCallStatus,
    started_at: datetime,
    started_ns: int,
    response: Any | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
) -> ComposerLLMCall:
    usage = _token_usage_from_response(response)
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
        latency_ms=(time.monotonic_ns() - started_ns) // 1_000_000,
        provider_request_id=_safe_provider_request_id(response),
        messages_hash=stable_hash(messages),
        tools_spec_hash=stable_hash(tools) if tools is not None else None,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        error_class=error_class,
        error_message=error_message,
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
      ``cache_control: {"type": "ephemeral"}`` field. LiteLLM's Anthropic
      transform recognizes this and propagates it onto the corresponding
      ``AnthropicSystemMessageContent`` block on the wire.
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
        session_engine: Engine | None = None,
        secret_service: Any | None = None,
        runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
    ) -> None:
        self._catalog = catalog
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

    @staticmethod
    def _runtime_preflight_failure_message(result: ValidationResult) -> str:
        """Compose a synthetic user-visible message for a failed runtime preflight.

        Prefers the first ValidationError's message and suggestion, falls back
        to the first failed check's detail, then to a bare fallback if the
        ValidationResult carries neither.

        Boundary contract: the message is echoed verbatim into chat history
        and the OpenAI-format chat completion. Secrets are guaranteed absent
        because validate_pipeline() resolves secret refs before settings load
        and validation errors are derived from typed plugin configs, never
        from raw secret values. However, the message MAY carry operator-
        supplied path fragments (the operator's own configured CSV paths,
        sink output paths) and exception text that names plugin config field
        values — both are surfaced verbatim from ValidationError.message and
        ValidationCheck.detail. In the current single-operator deployment
        model this is acceptable: the operator already knows their own
        paths and config. If ELSPETH ever takes a multi-tenant deployment
        shape, this synthesizer must be sanitized before merge.
        """
        if result.errors:
            first = result.errors[0]
            suggestion = f" Suggested fix: {first.suggestion}" if first.suggestion else ""
            return f"I cannot mark this pipeline complete yet because runtime preflight failed: {first.message}.{suggestion}"
        failed_checks = [check for check in result.checks if not check.passed]
        if failed_checks:
            return f"I cannot mark this pipeline complete yet because runtime preflight failed: {failed_checks[0].detail}."
        return "I cannot mark this pipeline complete yet because runtime preflight failed."

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
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> ComposerResult:
        """Apply the deterministic final-gate check and build a ComposerResult.

        Gate logic (no regex on natural-language text):
        - If ``state.version > initial_version`` (state changed this turn),
          run ``_cached_runtime_preflight`` for the current state.
        - Otherwise, reuse ``last_runtime_preflight`` from the most recent
          ``preview_pipeline`` call (may be ``None``).

        If the preflight result is invalid, replace the assistant message with
        a synthetic failure notice and preserve the original LLM text in
        ``raw_assistant_content``.

        Unexpected preflight exceptions (anything other than a
        ``RuntimePreflightFailure`` caught inside ``_cached_runtime_preflight``)
        are already converted to ``ComposerRuntimePreflightError`` with
        partial-state preservation by the coordinator's exception handling
        path — they are not caught here.
        """
        runtime_result = last_runtime_preflight
        if state.version > initial_version:
            runtime_result = await self._cached_runtime_preflight(
                state,
                user_id=user_id,
                cache=runtime_preflight_cache,
                initial_version=initial_version,
                session_scope=session_scope,
                llm_calls=llm_calls,
            )

        if runtime_result is None:
            return ComposerResult(message=content, state=state, tool_invocations=tool_invocations, llm_calls=llm_calls)

        if not runtime_result.is_valid:
            return ComposerResult(
                message=self._runtime_preflight_failure_message(runtime_result),
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
    ) -> ComposerResult:
        """Run the LLM composition loop with dual-counter budget.

        Args:
            message: The user's chat message.
            messages: Chat history as plain dicts (pre-converted from
                ChatMessageRecord by route handler; seam contract B).
            state: The current CompositionState.

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
            return await self._compose_loop(message, messages, state, session_id, user_id, deadline, progress)
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
        """
        initial_version = state.version
        llm_messages = self._build_messages(messages, state, message)
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

        while True:
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

            # If no tool calls, the LLM is done — apply the final gate and return
            if not assistant_message.tool_calls:
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
                return await self._finalize_no_tool_response(
                    content=assistant_message.content or "",
                    state=state,
                    initial_version=initial_version,
                    user_id=user_id,
                    last_runtime_preflight=last_runtime_preflight,
                    runtime_preflight_cache=runtime_preflight_cache,
                    session_scope=session_scope,
                    tool_invocations=recorder.invocations,
                    llm_calls=recorder.llm_calls,
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
            # any mutation calls for budget classification.
            turn_has_mutation = False
            all_cache_hits = True

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    decoded_arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    # Track mutation intent even when args are unparseable
                    if not is_discovery_tool(tool_name):
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
                    if not is_discovery_tool(tool_name):
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
                    if not is_discovery_tool(tool_name):
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
                    if not is_discovery_tool(tool_name):
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
                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_payload),
                        }
                    )
                    continue

                await _emit_progress(progress, _tool_started_progress_event(tool_name))

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

                def _arg_error_payload(
                    _exc: ToolArgumentError,
                    _tool_name: str = tool_name,
                ) -> Mapping[str, Any]:
                    safe_message = _exc.args[0] if _exc.args else "tool argument error"
                    return {"error": f"Tool '{_tool_name}' failed: {safe_message}"}

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
                        arg_error_payload_factory=_arg_error_payload,
                    )
                except ToolArgumentError as exc:
                    # The audit record was already written by the helper.
                    # Build the LLM-facing tool message and continue.
                    if not is_discovery_tool(tool_name):
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
                    arg_error_payload = _arg_error_payload(exc)
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
                    raise ComposerPluginCrashError.capture(
                        tool_exc,
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                    ) from tool_exc

                # SUCCESS path — the helper already recorded
                # ComposerToolStatus.SUCCESS via finish_success (with the
                # sentinel-canonical fallback for non-finite floats /
                # non-serializable result types). Update loop-local state
                # from outcome.result and continue with the LLM-message
                # append.
                result = outcome.result
                state = result.updated_state
                last_validation = result.validation
                last_runtime_preflight = result.runtime_preflight or last_runtime_preflight
                result_json = _serialize_tool_result(result)
                await _emit_progress(progress, _tool_completed_progress_event(tool_name, result.success))

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
                        return await self._finalize_no_tool_response(
                            content=assistant_message.content or "",
                            state=state,
                            initial_version=initial_version,
                            user_id=user_id,
                            last_runtime_preflight=last_runtime_preflight,
                            runtime_preflight_cache=runtime_preflight_cache,
                            session_scope=session_scope,
                            tool_invocations=recorder.invocations,
                            llm_calls=recorder.llm_calls,
                        )
                    raise ComposerConvergenceError.capture(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="composition",
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                    )
            else:
                discovery_turns_used += 1
                if discovery_turns_used >= self._max_discovery_turns:
                    raise ComposerConvergenceError.capture(
                        max_turns=composition_turns_used + discovery_turns_used,
                        budget_exhausted="discovery",
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                    )

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
        """
        try:
            return build_messages(
                chat_history=chat_history,
                state=state,
                user_message=user_message,
                catalog=self._catalog,
                data_dir=self._data_dir,
            )
        except OSError as exc:
            raise ComposerServiceError(f"Failed to load deployment skill ({type(exc).__name__})") from exc

    def _get_litellm_tools(self) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM function format."""
        definitions = get_tool_definitions()
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
            response = await _litellm_acompletion(
                model=self._model,
                messages=messages,
                tools=tools,
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
            response = await _litellm_acompletion(
                model=self._model,
                messages=messages,
            )
        except LiteLLMBadRequestError as exc:
            raise _BadRequestLLMError(f"LLM request rejected ({type(exc).__name__})") from exc
        if not response.choices:
            raise _MalformedLLMResponseError("LLM returned empty choices array — cannot explain run diagnostics", response=response)
        return response

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
        applied to the system message and the trailing tool before the
        call. The transformed payload is what flows to LiteLLM and what
        the audit ``messages_hash`` / ``tools_spec_hash`` record — the
        hash is over the bytes actually sent, so the audit row is
        truthful about the wire payload (elspeth-4e79436719 §Phase 3).
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
