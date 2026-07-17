"""Bounded, discovery-only planner for canonical pipeline proposals.

The planner deliberately has no mutation or persistence authority.  It can
read through a pinned core/blob/secret discovery palette, validate a complete
``set_pipeline`` candidate through the production candidate builder, settle
inline source custody only after that candidate is acceptable, and return an
immutable :class:`PipelineProposal`.  Publishing the proposal remains a route
or session-service responsibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Final, Literal, Protocol, cast
from uuid import UUID

from sqlalchemy import Engine

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.composer_progress import ComposerProgressSink
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder, begin_dispatch, dispatch_with_audit
from elspeth.web.composer.discovery_cache import serialize_tool_result
from elspeth.web.composer.llm_response_parsing import (
    apply_anthropic_cache_markers,
    build_llm_call_record,
    supports_anthropic_prompt_cache_markers,
)
from elspeth.web.composer.pipeline_custody import finalize_pipeline_custody, prepare_pipeline_custody
from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface, ProposalBase
from elspeth.web.composer.progress import (
    emit_progress,
    model_call_progress_event,
    tool_batch_progress_event,
    tool_completed_progress_event,
    tool_started_progress_event,
)
from elspeth.web.composer.redaction import SetPipelineArgumentsModel
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._common import RuntimePreflight, ToolContext, ToolResult
from elspeth.web.composer.tools._dispatch import (
    execute_discovery_tool_with_context,
    get_tool_definitions,
)
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema
from elspeth.web.composer.tools.sessions import build_set_pipeline_candidate
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot

PLANNER_DISCOVERY_TOOL_NAMES: Final[tuple[str, ...]] = (
    # Core discovery.
    "diff_pipeline",
    "explain_validation_error",
    "get_audit_info",
    "get_expression_grammar",
    "get_pipeline_state",
    "get_plugin_assistance",
    "get_plugin_schema",
    "list_models",
    "list_recipes",
    "list_sinks",
    "list_sources",
    "list_transforms",
    "preview_pipeline",
    # Blob discovery.
    "get_blob_content",
    "get_blob_metadata",
    "inspect_source",
    "list_blobs",
    "list_composer_blobs",
    # Secret discovery.  Values remain outside this surface.
    "list_secret_refs",
    "validate_secret_ref",
)
_PLANNER_DISCOVERY_TOOL_NAME_SET: Final[frozenset[str]] = frozenset(PLANNER_DISCOVERY_TOOL_NAMES)
_TERMINAL_TOOL_NAME: Final[str] = "emit_pipeline_proposal"


class _Completion(Protocol):
    async def __call__(self, **kwargs: Any) -> Any: ...


class PipelinePlannerError(RuntimeError):
    """Leak-safe failure raised when the bounded planner cannot continue."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PlannerBudgetPolicy:
    """Request-wide hard bounds, except cost which is a continuation cap.

    ``max_request_bytes`` covers the exact canonical UTF-8 bytes of the full
    post-cache-marker ``{messages, tools}`` request payload.  Provider cost is
    necessarily known only after a call: the call is audited first, then a
    missing/malformed value or cumulative overage prevents all response
    parsing, dispatch, custody, and proposal construction.  The final call may
    therefore overshoot the configured amount; this is not a pre-spend cap.
    """

    max_total_provider_calls: int
    max_request_bytes: int
    max_completion_tokens: int
    max_cumulative_provider_cost: Decimal

    def __post_init__(self) -> None:
        for name in ("max_total_provider_calls", "max_request_bytes", "max_completion_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.max_cumulative_provider_cost) is not Decimal:
            raise TypeError("max_cumulative_provider_cost must be Decimal")
        if not self.max_cumulative_provider_cost.is_finite() or self.max_cumulative_provider_cost < 0:
            raise ValueError("max_cumulative_provider_cost must be finite and non-negative")

    @property
    def audit_hash(self) -> str:
        return stable_hash(
            {
                "schema": "composer.planner-budget.v1",
                "max_total_provider_calls": self.max_total_provider_calls,
                "max_request_bytes": self.max_request_bytes,
                "max_completion_tokens": self.max_completion_tokens,
                "max_cumulative_provider_cost": str(self.max_cumulative_provider_cost),
            }
        )


@dataclass(frozen=True, slots=True)
class PlannerModelConfig:
    completion: _Completion
    model_identifier: str
    model_version: str
    provider: str
    temperature: float | None
    seed: int | None
    timeout_seconds: float
    max_composition_turns: int
    max_discovery_turns: int
    max_tool_calls_per_turn: int
    max_api_attempts: int
    api_retry_base_seconds: float

    def __post_init__(self) -> None:
        for name in ("model_identifier", "model_version", "provider"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty exact string")
        for name in ("max_composition_turns", "max_discovery_turns", "max_tool_calls_per_turn", "max_api_attempts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int | float):
            raise TypeError("timeout_seconds must be a finite positive number")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        if isinstance(self.api_retry_base_seconds, bool) or not isinstance(self.api_retry_base_seconds, int | float):
            raise TypeError("api_retry_base_seconds must be a finite non-negative number")
        if not math.isfinite(float(self.api_retry_base_seconds)) or self.api_retry_base_seconds < 0:
            raise ValueError("api_retry_base_seconds must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class PlannerOriginatingMessage:
    session_id: str
    message_id: str
    content: str
    user_id: str | None

    def __post_init__(self) -> None:
        for name in ("session_id", "message_id"):
            value = getattr(self, name)
            try:
                parsed = UUID(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a canonical UUID string") from exc
            if str(parsed) != value:
                raise ValueError(f"{name} must be a canonical UUID string")
        if type(self.content) is not str:
            raise TypeError("content must be an exact string")
        if self.user_id is not None and (type(self.user_id) is not str or not self.user_id.strip()):
            raise ValueError("user_id must be a non-empty exact string or None")


@dataclass(frozen=True, slots=True)
class PlannerCustodyConfig:
    data_dir: str
    session_engine: Engine | None
    max_storage_per_session: int
    secret_service: WebSecretResolver | None
    runtime_preflight: RuntimePreflight | None

    def __post_init__(self) -> None:
        if type(self.data_dir) is not str or not self.data_dir.strip():
            raise ValueError("data_dir must be a non-empty exact string")
        if type(self.max_storage_per_session) is not int or self.max_storage_per_session <= 0:
            raise ValueError("max_storage_per_session must be a positive exact integer")


PlannerSettlement = Literal["complete", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class PlannerRequestLifecycle:
    """Injected route lifecycle adapters; the planner owns no route globals."""

    before_start: Callable[[], Awaitable[None]]
    request_scope: Callable[[], AbstractAsyncContextManager[None]]
    on_settled: Callable[[PlannerSettlement], Awaitable[None]]
    progress: ComposerProgressSink | None


@dataclass(frozen=True, slots=True)
class _ParsedToolCall:
    call_id: str
    name: str
    raw_arguments: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _AuditedDiscoveryResult:
    """Carry the real result while exposing only a closed audit projection."""

    result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.result.success,
            "validation": {"is_valid": self.result.validation.is_valid},
            "version": self.result.updated_state.version,
        }


def planner_terminal_tool_definition() -> dict[str, Any]:
    """Return the sole terminal with the exact registered pipeline schema."""
    return {
        "type": "function",
        "function": {
            "name": _TERMINAL_TOOL_NAME,
            "description": "Return one complete canonical pipeline proposal for server validation.",
            "parameters": {
                "type": "object",
                "properties": {"pipeline": canonical_set_pipeline_schema()},
                "required": ["pipeline"],
                "additionalProperties": False,
            },
        },
    }


def planner_tool_definitions() -> list[dict[str, Any]]:
    """Return the pinned read-only palette followed by the sole terminal."""
    registered = {definition["name"]: definition for definition in get_tool_definitions()}
    missing = _PLANNER_DISCOVERY_TOOL_NAME_SET - registered.keys()
    if missing:
        raise RuntimeError(f"planner discovery declarations are missing: {sorted(missing)}")
    discovery = [
        {
            "type": "function",
            "function": {
                "name": registered[name]["name"],
                "description": registered[name]["description"],
                "parameters": registered[name]["parameters"],
            },
        }
        for name in PLANNER_DISCOVERY_TOOL_NAMES
    ]
    return [*discovery, planner_terminal_tool_definition()]


def _provider_fields(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    try:
        fields = vars(value)
    except TypeError:
        return None
    try:
        extra = object.__getattribute__(value, "__pydantic_extra__")
    except AttributeError:
        return cast(Mapping[str, Any], fields)
    if isinstance(extra, Mapping) and extra:
        return cast(Mapping[str, Any], {**extra, **fields})
    return cast(Mapping[str, Any], fields)


def _provider_field(value: Any, name: str) -> Any:
    fields = _provider_fields(value)
    return fields[name] if fields is not None and name in fields else None


def _parse_json_object(raw: object, *, label: str) -> Mapping[str, Any]:
    if type(raw) is not str or not raw.strip():
        raise PipelinePlannerError(f"{label} must be a non-empty JSON string", code="MALFORMED_RESPONSE")

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite JSON number")

    try:
        parsed = json.loads(raw, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelinePlannerError(f"{label} is not strict JSON", code="MALFORMED_RESPONSE") from exc
    if type(parsed) is not dict:
        raise PipelinePlannerError(f"{label} must decode to an object", code="MALFORMED_RESPONSE")
    return cast(Mapping[str, Any], parsed)


def _parse_response_tool_calls(response: Any) -> tuple[Any, tuple[_ParsedToolCall, ...]]:
    choices = _provider_field(response, "choices")
    if not isinstance(choices, list | tuple) or len(choices) != 1:
        raise PipelinePlannerError("planner response must contain exactly one choice", code="MALFORMED_RESPONSE")
    message = _provider_field(choices[0], "message")
    if message is None:
        raise PipelinePlannerError("planner response choice is missing its message", code="MALFORMED_RESPONSE")
    raw_calls = _provider_field(message, "tool_calls")
    if not isinstance(raw_calls, list | tuple) or not raw_calls:
        raise PipelinePlannerError("planner response must call a declared tool", code="MALFORMED_RESPONSE")
    parsed: list[_ParsedToolCall] = []
    for raw_call in raw_calls:
        call_id = _provider_field(raw_call, "id")
        function = _provider_field(raw_call, "function")
        name = _provider_field(function, "name")
        raw_arguments = _provider_field(function, "arguments")
        if type(call_id) is not str or not call_id or type(name) is not str or not name:
            raise PipelinePlannerError("planner tool call metadata is malformed", code="MALFORMED_RESPONSE")
        arguments = _parse_json_object(raw_arguments, label=f"{name} arguments")
        parsed.append(_ParsedToolCall(call_id, name, cast(str, raw_arguments), arguments))
    return message, tuple(parsed)


def _assistant_tool_calls_message(message: Any, calls: tuple[_ParsedToolCall, ...]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": _provider_field(message, "content"),
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in calls
        ],
    }


def _allowlisted_candidate_feedback(result: ToolResult) -> dict[str, Any]:
    """Project only structured validation fields already safe for tool output."""
    validation = result.validation
    return {
        "success": False,
        "validation": {
            "is_valid": validation.is_valid,
            "errors": [
                {
                    "component": entry.component,
                    "severity": entry.severity,
                    "error_code": entry.error_code or "validation_error",
                    "error_class": "ValidationError",
                }
                for entry in validation.errors
            ],
        },
    }


def _canonical_schema_feedback() -> dict[str, Any]:
    return {
        "success": False,
        "validation": {
            "is_valid": False,
            "errors": [
                {
                    "component": "pipeline",
                    "severity": "high",
                    "error_code": "canonical_schema",
                    "error_class": "SchemaValidationError",
                }
            ],
        },
    }


async def _await_custody_settlement(awaitable: Awaitable[Any]) -> Any:
    """Finish idempotent custody after cancellation, then preserve cancellation."""

    async def settle() -> Any:
        return await awaitable

    task: asyncio.Task[Any] = asyncio.create_task(settle())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)
        # Observe a custody failure without replacing the active cancellation.
        with suppress(BaseException):
            task.result()
        raise


async def _settle_lifecycle(lifecycle: PlannerRequestLifecycle, outcome: PlannerSettlement) -> None:
    async def settle() -> None:
        await lifecycle.on_settled(outcome)

    task: asyncio.Task[None] = asyncio.create_task(settle())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        # Settlement is operator/UI bookkeeping.  Let it finish even while the
        # request task is being torn down, then preserve the original cancel.
        while not task.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)
        raise


async def plan_pipeline(
    *,
    intent: str,
    current_state: CompositionState,
    reviewed_facts: Mapping[str, Any],
    surface: PlannerSurface,
    policy_catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
    originating_message: PlannerOriginatingMessage,
    base: ProposalBase,
    model_config: PlannerModelConfig,
    rendered_skill: str,
    repair_budget: int,
    budget_policy: PlannerBudgetPolicy,
    custody_config: PlannerCustodyConfig,
    lifecycle: PlannerRequestLifecycle,
    recorder: BufferingRecorder,
) -> PipelineProposal:
    """Plan and validate one proposal without publishing state or DB rows."""
    if type(intent) is not str or not intent.strip():
        raise ValueError("intent must be a non-empty exact string")
    if type(rendered_skill) is not str or not rendered_skill.strip():
        raise ValueError("rendered_skill must be a non-empty exact string")
    if type(repair_budget) is not int or repair_budget < 0:
        raise ValueError("repair_budget must be a non-negative exact integer")
    if policy_catalog.snapshot is not plugin_snapshot:
        raise ValueError("plugin_snapshot_catalog_mismatch")

    outcome: PlannerSettlement = "failed"
    try:
        await lifecycle.before_start()
        async with lifecycle.request_scope():
            proposal = await _plan_pipeline_inner(
                intent=intent,
                current_state=current_state,
                reviewed_facts=reviewed_facts,
                surface=surface,
                policy_catalog=policy_catalog,
                plugin_snapshot=plugin_snapshot,
                originating_message=originating_message,
                base=base,
                model_config=model_config,
                rendered_skill=rendered_skill,
                repair_budget=repair_budget,
                budget_policy=budget_policy,
                custody_config=custody_config,
                lifecycle=lifecycle,
                recorder=recorder,
            )
        outcome = "complete"
        return proposal
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    finally:
        await _settle_lifecycle(lifecycle, outcome)


async def _plan_pipeline_inner(
    *,
    intent: str,
    current_state: CompositionState,
    reviewed_facts: Mapping[str, Any],
    surface: PlannerSurface,
    policy_catalog: PolicyCatalogView,
    plugin_snapshot: PluginAvailabilitySnapshot,
    originating_message: PlannerOriginatingMessage,
    base: ProposalBase,
    model_config: PlannerModelConfig,
    rendered_skill: str,
    repair_budget: int,
    budget_policy: PlannerBudgetPolicy,
    custody_config: PlannerCustodyConfig,
    lifecycle: PlannerRequestLifecycle,
    recorder: BufferingRecorder,
) -> PipelineProposal:
    skill_hash = hashlib.sha256(rendered_skill.encode("utf-8")).hexdigest()
    request_context = ToolContext(
        catalog=policy_catalog,
        plugin_snapshot=plugin_snapshot,
        data_dir=custody_config.data_dir,
        require_data_dir_for_paths=True,
        session_engine=custody_config.session_engine,
        session_id=originating_message.session_id,
        secret_service=custody_config.secret_service,
        user_id=originating_message.user_id,
        baseline=current_state,
        current_validation=policy_catalog.validate_composition_state(current_state).validation,
        runtime_preflight=custody_config.runtime_preflight,
        max_blob_storage_per_session_bytes=custody_config.max_storage_per_session,
        user_message_id=originating_message.message_id,
        user_message_content=originating_message.content,
        composer_model_identifier=model_config.model_identifier,
        composer_model_version=model_config.model_version,
        composer_provider=model_config.provider,
        composer_skill_hash=skill_hash,
        tool_arguments_hash=None,
    )
    tools = planner_tool_definitions()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": rendered_skill},
        {
            "role": "user",
            "content": canonical_json(
                {
                    "intent": intent,
                    "current_state": current_state.to_dict(),
                    "reviewed_facts": reviewed_facts,
                    "surface": surface.value,
                    "instruction": (
                        "Use read-only discovery as needed, then call emit_pipeline_proposal exactly once "
                        "with one complete canonical set_pipeline argument object."
                    ),
                }
            ),
        },
    ]
    deadline = asyncio.get_running_loop().time() + model_config.timeout_seconds
    total_calls = 0
    total_cost = Decimal("0")
    discovery_turns = 0
    composition_turns = 0
    repair_count = 0
    seen_discovery: set[tuple[str, str]] = set()

    async def call_model() -> Any:
        nonlocal total_calls, total_cost
        marked_messages, marked_tools = (
            apply_anthropic_cache_markers(messages, tools)
            if supports_anthropic_prompt_cache_markers(model_config.model_identifier)
            else (list(messages), list(tools))
        )
        assert marked_tools is not None
        request_size = len(canonical_json({"messages": marked_messages, "tools": marked_tools}).encode("utf-8"))
        if request_size > budget_policy.max_request_bytes:
            raise PipelinePlannerError("planner request byte budget exhausted", code="REQUEST_BYTES_EXHAUSTED")
        await emit_progress(lifecycle.progress, model_call_progress_event(intent))

        for attempt in range(1, model_config.max_api_attempts + 1):
            if total_calls >= budget_policy.max_total_provider_calls:
                raise PipelinePlannerError("planner provider call budget exhausted", code="PROVIDER_CALLS_EXHAUSTED")
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise PipelinePlannerError("planner wall-clock budget exhausted", code="TIMEOUT")
            total_calls += 1
            ordinal = total_calls
            started_at = datetime.now(UTC)
            started_ns = time.monotonic_ns()
            response: Any = None
            kwargs: dict[str, Any] = {
                "model": model_config.model_identifier,
                "messages": marked_messages,
                "tools": marked_tools,
                "max_tokens": budget_policy.max_completion_tokens,
            }
            if model_config.temperature is not None:
                kwargs["temperature"] = model_config.temperature
            if model_config.seed is not None:
                kwargs["seed"] = model_config.seed
            try:
                response = await asyncio.wait_for(model_config.completion(**kwargs), timeout=remaining)
            except asyncio.CancelledError as exc:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=model_config.model_identifier,
                        messages=marked_messages,
                        tools=marked_tools,
                        status=ComposerLLMCallStatus.CANCELLED,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=model_config.temperature,
                        seed=model_config.seed,
                        error_class=type(exc).__name__,
                        error_message=type(exc).__name__,
                        max_completion_tokens_requested=budget_policy.max_completion_tokens,
                        planner_policy_hash=budget_policy.audit_hash,
                        planner_call_ordinal=ordinal,
                    )
                )
                raise
            except TimeoutError as exc:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=model_config.model_identifier,
                        messages=marked_messages,
                        tools=marked_tools,
                        status=ComposerLLMCallStatus.TIMEOUT,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=model_config.temperature,
                        seed=model_config.seed,
                        error_class=type(exc).__name__,
                        error_message=type(exc).__name__,
                        max_completion_tokens_requested=budget_policy.max_completion_tokens,
                        planner_policy_hash=budget_policy.audit_hash,
                        planner_call_ordinal=ordinal,
                    )
                )
                raise PipelinePlannerError("planner wall-clock budget exhausted", code="TIMEOUT") from exc
            except Exception as exc:
                from litellm.exceptions import APIError as LiteLLMAPIError
                from litellm.exceptions import AuthenticationError as LiteLLMAuthError
                from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

                status = ComposerLLMCallStatus.API_ERROR
                if isinstance(exc, LiteLLMAuthError):
                    status = ComposerLLMCallStatus.AUTH_ERROR
                elif isinstance(exc, LiteLLMBadRequestError):
                    status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=model_config.model_identifier,
                        messages=marked_messages,
                        tools=marked_tools,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=model_config.temperature,
                        seed=model_config.seed,
                        error_class=type(exc).__name__,
                        error_message=type(exc).__name__,
                        max_completion_tokens_requested=budget_policy.max_completion_tokens,
                        planner_policy_hash=budget_policy.audit_hash,
                        planner_call_ordinal=ordinal,
                    )
                )
                if (
                    isinstance(exc, LiteLLMAPIError)
                    and status is ComposerLLMCallStatus.API_ERROR
                    and attempt < model_config.max_api_attempts
                ):
                    retry_delay = model_config.api_retry_base_seconds * (2 ** (attempt - 1))
                    if retry_delay > 0:
                        await asyncio.sleep(min(retry_delay, max(0.0, deadline - asyncio.get_running_loop().time())))
                    continue
                raise PipelinePlannerError(
                    f"planner provider call failed ({type(exc).__name__})",
                    code="PROVIDER_ERROR",
                ) from None

            call = build_llm_call_record(
                model_requested=model_config.model_identifier,
                messages=marked_messages,
                tools=marked_tools,
                status=ComposerLLMCallStatus.SUCCESS,
                started_at=started_at,
                started_ns=started_ns,
                temperature=model_config.temperature,
                seed=model_config.seed,
                response=response,
                max_completion_tokens_requested=budget_policy.max_completion_tokens,
                planner_policy_hash=budget_policy.audit_hash,
                planner_call_ordinal=ordinal,
            )
            recorder.record_llm_call(call)
            # Cost enforcement is intentionally post-call and pre-parse.  Do
            # not inspect provider content or dispatch tools before it passes.
            if call.provider_cost is None:
                raise PipelinePlannerError("planner provider cost metadata is missing or malformed", code="COST_UNAVAILABLE")
            if call.completion_tokens is not None and call.completion_tokens > budget_policy.max_completion_tokens:
                raise PipelinePlannerError(
                    "planner provider reported a completion token limit overage",
                    code="COMPLETION_TOKENS_EXCEEDED",
                )
            total_cost += Decimal(str(call.provider_cost))
            if total_cost > budget_policy.max_cumulative_provider_cost:
                raise PipelinePlannerError("planner provider cost continuation cap exceeded", code="COST_CAP_EXCEEDED")
            return response
        raise AssertionError("provider attempt loop exited without return or exception")

    while True:
        response = await call_model()
        message, calls = _parse_response_tool_calls(response)
        if len(calls) > model_config.max_tool_calls_per_turn:
            raise PipelinePlannerError("planner per-turn tool call budget exhausted", code="TOOL_CALLS_EXHAUSTED")

        terminal_calls = tuple(call for call in calls if call.name == _TERMINAL_TOOL_NAME)
        if terminal_calls:
            if len(calls) != 1:
                raise PipelinePlannerError("terminal proposal call must be the only tool call", code="MALFORMED_RESPONSE")
            composition_turns += 1
            if composition_turns > model_config.max_composition_turns:
                raise PipelinePlannerError("planner composition turn budget exhausted", code="COMPOSITION_EXHAUSTED")
            call = terminal_calls[0]
            schema_feedback: dict[str, Any] | None = None
            if set(call.arguments) != {"pipeline"} or type(call.arguments.get("pipeline")) is not dict:
                schema_feedback = _canonical_schema_feedback()
                pipeline = None
            else:
                pipeline = cast(dict[str, Any], call.arguments["pipeline"])
                try:
                    SetPipelineArgumentsModel.model_validate(pipeline)
                except ValueError:
                    schema_feedback = _canonical_schema_feedback()
            if schema_feedback is not None:
                repair_count += 1
                if repair_count > repair_budget:
                    raise PipelinePlannerError("planner repair budget exhausted", code="REPAIR_EXHAUSTED")
                messages.append(_assistant_tool_calls_message(message, calls))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": canonical_json(schema_feedback),
                    }
                )
                continue
            assert pipeline is not None
            candidate_context = replace(request_context, tool_arguments_hash=stable_hash(call.arguments))
            candidate = build_set_pipeline_candidate(pipeline, current_state, candidate_context)
            if not candidate.acceptable:
                repair_count += 1
                if repair_count > repair_budget:
                    raise PipelinePlannerError("planner repair budget exhausted", code="REPAIR_EXHAUSTED")
                messages.append(_assistant_tool_calls_message(message, calls))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": canonical_json(_allowlisted_candidate_feedback(candidate.result)),
                    }
                )
                continue

            safe_pipeline: Mapping[str, Any] = pipeline
            if candidate.prepared_inline_blob is not None:
                if custody_config.session_engine is None:
                    raise AuditIntegrityError("inline pipeline custody requires session_engine")
                preparation = prepare_pipeline_custody(
                    pipeline,
                    candidate.prepared_inline_blob,
                    session_id=originating_message.session_id,
                )
                await _await_custody_settlement(
                    finalize_pipeline_custody(
                        preparation,
                        engine=custody_config.session_engine,
                        data_dir=custody_config.data_dir,
                        max_storage_per_session=custody_config.max_storage_per_session,
                    )
                )
                # Custody freezes its preparation envelope for hand-off.  The
                # proposal contract intentionally accepts strict JSON
                # containers only, so thaw the already-validated safe shape
                # back to dict/list containers before final validation/hash.
                safe_pipeline = cast(dict[str, Any], deep_thaw(preparation.arguments))
                safe_context = replace(request_context, tool_arguments_hash=stable_hash({"pipeline": safe_pipeline}))
                safe_candidate = build_set_pipeline_candidate(safe_pipeline, current_state, safe_context)
                if not safe_candidate.acceptable or safe_candidate.prepared_inline_blob is not None:
                    # Custody is idempotently retained for session
                    # reconciliation; no proposal or state is published.
                    raise AuditIntegrityError("custody-safe pipeline failed canonical revalidation")

            return PipelineProposal.create(
                pipeline=safe_pipeline,
                base=base,
                reviewed_facts=reviewed_facts,
                surface=surface,
                repair_count=repair_count,
                skill_hash=skill_hash,
            )

        if any(call.name not in _PLANNER_DISCOVERY_TOOL_NAME_SET for call in calls):
            raise PipelinePlannerError(
                "planner may execute read-only discovery tools only before its terminal proposal",
                code="DISCOVERY_ONLY",
            )
        discovery_turns += 1
        if discovery_turns > model_config.max_discovery_turns:
            raise PipelinePlannerError("planner discovery turn budget exhausted", code="DISCOVERY_EXHAUSTED")
        keys = tuple((call.name, stable_hash(call.arguments)) for call in calls)
        if any(key in seen_discovery for key in keys) or len(set(keys)) != len(keys):
            raise PipelinePlannerError("planner discovery repetition/cycle guard fired", code="DISCOVERY_CYCLE")
        seen_discovery.update(keys)
        await emit_progress(lifecycle.progress, tool_batch_progress_event(tuple(call.name for call in calls)))
        messages.append(_assistant_tool_calls_message(message, calls))
        for call in calls:
            await emit_progress(lifecycle.progress, tool_started_progress_event(call.name))
            dispatch = begin_dispatch(
                call.call_id,
                call.name,
                call.arguments,
                version_before=current_state.version,
                actor=originating_message.user_id or "pipeline-planner",
            )

            async def execute_discovery(call_to_execute: _ParsedToolCall = call) -> _AuditedDiscoveryResult:
                return _AuditedDiscoveryResult(
                    execute_discovery_tool_with_context(
                        call_to_execute.name,
                        call_to_execute.arguments,
                        current_state,
                        request_context,
                    )
                )

            audited = await dispatch_with_audit(
                recorder=recorder,
                audit=dispatch,
                do_dispatch=execute_discovery,
                version_after_provider=lambda carrier: carrier.result.updated_state.version,
                arg_error_payload_factory=lambda exc: {
                    "error_class": "ToolArgumentError",
                    "error_code": exc.code or "argument_error",
                },
            )
            result = cast(_AuditedDiscoveryResult, audited.result).result
            if result.updated_state != current_state:
                raise AuditIntegrityError("read-only planner discovery changed composition state")
            messages.append({"role": "tool", "tool_call_id": call.call_id, "content": serialize_tool_result(result)})
            await emit_progress(lifecycle.progress, tool_completed_progress_event(call.name, result.success))


__all__ = [
    "PLANNER_DISCOVERY_TOOL_NAMES",
    "PipelinePlannerError",
    "PlannerBudgetPolicy",
    "PlannerCustodyConfig",
    "PlannerModelConfig",
    "PlannerOriginatingMessage",
    "PlannerRequestLifecycle",
    "plan_pipeline",
    "planner_terminal_tool_definition",
    "planner_tool_definitions",
]
