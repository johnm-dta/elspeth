"""Per-tool-call dispatch pipeline for the composer compose loop.

Extracted verbatim from ComposerServiceImpl._dispatch_tool_batch (service.py)
to take the single largest method out of the god class. The loop body is
UNCHANGED; only its enclosing context is made explicit via the two carriers
below, replacing the prior nested-closure capture of loop-invariant inputs and
loop-carried accumulators.

Behaviour-preservation contract: every terminal arm's
recorder.record(finish_*) / anti_anchor.record_* / llm_messages.append /
budget-class side-effect is identical to the pre-extraction method. Pinned by
tests/unit/web/composer/test_dispatch_arms_characterization.py.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import UUID

from elspeth.contracts.composer_progress import ComposerProgressEvent, ComposerProgressSink
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.composer._compose_loop_carriers import (
    _CallModelOutcome,
    _DispatchOutcome,
    _ToolOutcome,
)
from elspeth.web.composer._required_paths_validator import (
    _TOOL_REQUIRED_PATHS,
    _find_missing_required_paths,
)
from elspeth.web.composer.anti_anchor import AntiAnchorTracker
from elspeth.web.composer.audit import (
    BufferingRecorder,
    begin_dispatch,
    begin_dispatch_or_arg_error,
    dispatch_with_audit,
    finish_arg_error,
    finish_plugin_crash,
    finish_success,
)
from elspeth.web.composer.discovery_cache import (
    CachedDiscoveryPayload as _CachedDiscoveryPayload,
)
from elspeth.web.composer.discovery_cache import (
    RuntimePreflightCache as _RuntimePreflightCache,
)
from elspeth.web.composer.discovery_cache import (
    cached_discovery_payload as _cached_discovery_payload,
)
from elspeth.web.composer.discovery_cache import (
    make_cache_key as _make_cache_key,
)
from elspeth.web.composer.discovery_cache import (
    result_from_cached_discovery_payload as _result_from_cached_discovery_payload,
)
from elspeth.web.composer.discovery_cache import (
    serialize_tool_result as _serialize_tool_result,
)
from elspeth.web.composer.discovery_cache import (
    tool_result_mutated_composition_state as _tool_result_mutated_composition_state,
)
from elspeth.web.composer.llm_response_parsing import safe_response_model
from elspeth.web.composer.progress import (
    emit_progress,
    tool_batch_progress_event,
    tool_completed_progress_event,
    tool_started_progress_event,
)
from elspeth.web.composer.proposals import build_tool_proposal_summary
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerRuntimePreflightError,
    ComposerServiceError,
    ToolArgumentError,
)
from elspeth.web.composer.state import CompositionState, ValidationSummary
from elspeth.web.composer.tool_error_payloads import arg_error_payload as _arg_error_payload
from elspeth.web.composer.tools import (
    RuntimePreflight,
    ToolResult,
    execute_tool,
    is_approval_required_blob_store_only_mutation_tool,
    is_blob_store_only_mutation_tool,
    is_cacheable_discovery_tool,
    is_discovery_tool,
    is_mutation_tool,
    is_session_aware_tool,
)
from elspeth.web.execution.schemas import ValidationResult

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl
    from elspeth.web.sessions.protocol import (
        ComposerSessionPreferencesRecord,
        SessionServiceProtocol,
    )


_MAX_PENDING_PROPOSALS_PER_TURN: Final[int] = 10


@dataclass(frozen=True, slots=True)
class ToolBatchContext:
    """Loop-invariant inputs to the dispatch loop, built once per batch.

    ``frozen=True`` prevents rebinding the field *references* only. The
    ``discovery_cache`` and ``runtime_preflight_cache`` fields are
    intentionally-mutable shared caches the dispatch loop writes into; they
    are deliberately exempt from the project's ``deep_freeze`` contract
    because deep-freezing them would break the loop's cache-write behaviour.
    No ``__post_init__`` freeze guard is added for that reason.
    """

    service: ComposerServiceImpl
    recorder: BufferingRecorder
    anti_anchor: AntiAnchorTracker
    discovery_cache: dict[str, _CachedDiscoveryPayload]
    runtime_preflight_cache: _RuntimePreflightCache
    session_id: str | None
    user_id: str | None
    user_message_id: str | None
    user_message_content: str | None
    current_state_id: str | None
    actor: str
    initial_version: int
    deadline: float
    progress: ComposerProgressSink | None
    session_scope: str
    turn_sessions_service: SessionServiceProtocol | None
    turn_session_uuid: UUID | None
    turn_preferences: ComposerSessionPreferencesRecord | None


@dataclass(slots=True)
class BatchAccumulator:
    """Driver-owned state threaded across compose-loop turns.

    These four are the only live loop-carried inputs: ``run_tool_batch``
    reads each exactly once (alias preamble) and the driver carries the
    updated values forward via ``_DispatchOutcome``. The batch body keeps
    every other accumulator (``tool_outcomes``, ``turn_has_mutation``, ...)
    as inline locals, so they are NOT fields here — adding them back would
    create a second, unread source of truth shadowing those locals. The
    deferred Phase 3 (locals->carriers) migration is what would move that
    state onto this carrier; until then this is exactly the live surface.
    """

    state: CompositionState
    last_validation: ValidationSummary | None
    last_runtime_preflight: ValidationResult | None
    advisor_calls_used: int


async def run_tool_batch(
    *,
    call_model: _CallModelOutcome,
    ctx: ToolBatchContext,
    acc: BatchAccumulator,
    llm_messages: list[dict[str, Any]],
) -> tuple[_DispatchOutcome, int]:
    """Execute one tool batch — extracted verbatim from
    ``ComposerServiceImpl._dispatch_tool_batch``.

    See the module docstring and ``ToolBatchContext`` for the
    behaviour-preservation contract. The body below is the former method
    body with ``self.`` rewritten to ``ctx.service.`` and the
    loop-invariant / driver-owned locals supplied by the alias preamble.
    """
    # ------------------------------------------------------------------
    # Alias preamble: reconstruct the original ``_dispatch_tool_batch``
    # local namespace so the body below is genuinely verbatim. Loop-invariant
    # inputs come from ``ctx``; the four driver-owned loop-carried inputs come
    # from ``acc``. Every other body init (``tool_outcomes = []``,
    # ``turn_has_mutation = False``, ...) stays inline exactly as before, so
    # the closures, the ``_append_tool_outcome`` default-arg capture, and
    # every loop reassignment keep working against the same local names.
    # ``self.`` is rewritten to ``ctx.service.`` — the only token change.
    # ------------------------------------------------------------------
    recorder = ctx.recorder
    anti_anchor = ctx.anti_anchor
    discovery_cache = ctx.discovery_cache
    runtime_preflight_cache = ctx.runtime_preflight_cache
    session_id = ctx.session_id
    user_id = ctx.user_id
    user_message_id = ctx.user_message_id
    user_message_content = ctx.user_message_content
    current_state_id = ctx.current_state_id
    actor = ctx.actor
    initial_version = ctx.initial_version
    deadline = ctx.deadline
    progress = ctx.progress
    session_scope = ctx.session_scope
    turn_sessions_service = ctx.turn_sessions_service
    turn_session_uuid = ctx.turn_session_uuid
    turn_preferences = ctx.turn_preferences
    state = acc.state
    last_validation = acc.last_validation
    last_runtime_preflight = acc.last_runtime_preflight
    advisor_calls_used = acc.advisor_calls_used

    assistant_message = call_model.assistant_message
    raw_assistant_content = call_model.raw_assistant_content
    assistant_tool_calls = call_model.assistant_tool_calls
    response = call_model.response

    await emit_progress(
        progress,
        tool_batch_progress_event(
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
    # immutable _ToolOutcome records. Step 2 (in _persist_turn_audit)
    # performs audit writes from this list; cancellation before Step 2
    # leaves the DB unchanged for the current assistant response.
    tool_outcomes: list[_ToolOutcome] = []
    plugin_crash: ComposerPluginCrashError | None = None
    plugin_crash_cause: BaseException | None = None
    pre_state_id: str | None = current_state_id
    ctx.service._phase3_last_expected_current_state_id = pre_state_id
    decoded_args_by_call_id: dict[str, dict[str, Any]] = {}
    proposals_this_turn = 0
    mutation_success_observed = False

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
            error_payload = {"error": f"Tool '{tool_name}' arguments are not canonical JSON ({type(canonicalization_failed).__name__})."}
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
                await emit_progress(
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

        if (
            turn_sessions_service is not None
            and turn_session_uuid is not None
            and turn_preferences is not None
            and turn_preferences.trust_mode == "explicit_approve"
            and is_mutation_tool(tool_name)
            # create_blob stays immediate because it allocates the blob id a
            # later composition proposal may reference. Destructive blob-only
            # writes such as update_blob/delete_blob still require approval.
            and (not is_blob_store_only_mutation_tool(tool_name) or is_approval_required_blob_store_only_mutation_tool(tool_name))
        ):
            if proposals_this_turn >= _MAX_PENDING_PROPOSALS_PER_TURN:
                raise ComposerServiceError(
                    f"Composer produced too many pending tool proposals in one turn ({_MAX_PENDING_PROPOSALS_PER_TURN} maximum)."
                )

            from pydantic import ValidationError as PydanticValidationError

            from elspeth.web.composer.redaction import MANIFEST, redact_tool_call_arguments

            # The LLM may produce arguments that fail the redaction
            # MANIFEST's argument_model — most commonly a misplaced
            # field like ``nodes[*].schema`` belonging at
            # ``nodes[*].options.schema``. The runtime validator
            # would reject these the same way. On redaction
            # ValidationError we fall through to normal dispatch:
            # execute_tool emits a clean ToolArgumentError, the
            # existing post-dispatch arg-error handling records
            # the failure for audit (with the
            # ``_redaction_status: invalid_tool_arguments`` marker),
            # and the compose loop continues so the model can
            # self-correct. The previous behaviour — letting the
            # ValidationError propagate — crashed the compose
            # request with HTTP 500 (session 100dc5cb… 2026-05-14:
            # frontend rendered a generic ApiError as a bare
            # "retry" button with no diagnostic).
            redacted_arguments: Mapping[str, Any] | None
            if tool_name in MANIFEST:
                try:
                    redacted_arguments = redact_tool_call_arguments(
                        tool_name,
                        arguments,
                        telemetry=ctx.service._redaction_telemetry,
                    )
                except PydanticValidationError:
                    redacted_arguments = None
            else:
                redacted_arguments = arguments

            if redacted_arguments is not None:
                proposal_summary = build_tool_proposal_summary(
                    tool_name=tool_name,
                    arguments=arguments,
                    redacted_arguments=redacted_arguments,
                )
                proposal = await turn_sessions_service.create_composition_proposal(
                    session_id=turn_session_uuid,
                    tool_call_id=tool_call.id,
                    tool_name=tool_name,
                    summary=proposal_summary.summary,
                    rationale=proposal_summary.rationale,
                    affects=proposal_summary.affects,
                    arguments_json=arguments,
                    arguments_redacted_json=proposal_summary.arguments_redacted_json,
                    base_state_id=UUID(current_state_id) if current_state_id is not None else None,
                    actor=f"composer-web:user:{user_id}" if user_id is not None else "composer-web:anonymous",
                    user_message_id=UUID(user_message_id) if user_message_id is not None else None,
                    composer_model_identifier=ctx.service._model,
                    composer_model_version=safe_response_model(response) or ctx.service._model,
                    composer_provider=ctx.service._availability.provider or "unknown",
                    composer_skill_hash=ctx.service._composer_skill_hash,
                    tool_arguments_hash=audit.arguments_hash,
                )
                proposals_this_turn += 1
                proposal_payload = {
                    "success": True,
                    "status": "APPROVAL_REQUIRED",
                    "proposal_id": str(proposal.id),
                    "tool_name": tool_name,
                    "summary": proposal.summary,
                    "message": "The requested pipeline change is pending human approval and has not been applied.",
                }
                proposal_result = ToolResult(
                    success=True,
                    updated_state=state,
                    validation=last_validation if last_validation is not None else state.validate(),
                    affected_nodes=(),
                    data=proposal_payload,
                )
                recorder.record(
                    finish_success(
                        audit,
                        result_payload=proposal_result.to_dict(),
                        version_after=state.version,
                    )
                )
                _append_tool_outcome(
                    response=proposal_result,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
                )
                anti_anchor.record_success()
                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _serialize_tool_result(proposal_result),
                    }
                )
                turn_has_mutation = True
                continue
            # redacted_arguments is None → invalid LLM arguments;
            # fall through to normal dispatch, which will emit a
            # ToolArgumentError through the standard arg-error path.

        await emit_progress(progress, tool_started_progress_event(tool_name))

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
            budget = ctx.service._settings.composer_advisor_max_calls_per_compose
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
                _append_tool_outcome(
                    response=budget_payload,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
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
            advisor_arg_error = ctx.service._validate_advisor_arguments(arguments)
            if advisor_arg_error is not None:
                recorder.record(
                    finish_arg_error(
                        audit,
                        error_class=str(advisor_arg_error["error_class"]),
                        error_message=str(advisor_arg_error["error"]),
                        error_payload=advisor_arg_error,
                    )
                )
                _append_tool_outcome(
                    response=None,
                    error_class=str(advisor_arg_error["error_class"]),
                    error_message=str(advisor_arg_error["error"]),
                    post_version=state.version,
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
                _append_tool_outcome(
                    response=timeout_payload,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
                )
                raise ComposerConvergenceError.capture(
                    max_turns=0,
                    budget_exhausted="timeout",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=recorder.invocations,
                    llm_calls=recorder.llm_calls,
                )

            advisor_timeout = ctx.service._settings.composer_advisor_timeout_seconds
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
                guidance, advisor_meta = await ctx.service._call_advisor_with_audit(
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
                    _append_tool_outcome(
                        response=timeout_payload,
                        error_class=None,
                        error_message=None,
                        post_version=state.version,
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
                _append_tool_outcome(
                    response=advisor_error_payload,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
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
                _append_tool_outcome(
                    response=advisor_error_payload,
                    error_class=None,
                    error_message=None,
                    post_version=state.version,
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
            _append_tool_outcome(
                response=success_payload,
                error_class=None,
                error_message=None,
                post_version=state.version,
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

        # Session-aware async tool dispatch.
        #
        # ``_SESSION_AWARE_TOOL_HANDLERS`` holds handlers that AWAIT
        # session-service writers/readers; they cannot be dispatched
        # through the sync ``run_sync_in_worker(execute_tool, ...)``
        # path because execute_tool() is sync. The compose loop
        # intercepts these tool names BEFORE the generic dispatch
        # and awaits the handler directly, then routes the result
        # through the same audit envelope discipline as the sync
        # path (finish_success / finish_arg_error / finish_plugin_crash).
        #
        # Currently registered: ``request_interpretation_review``.
        # Adding another session-aware tool extends
        # ``_SESSION_AWARE_TOOL_HANDLERS`` and the per-tool
        # kwarg-build dict below; no new dispatch branch is needed.
        if is_session_aware_tool(tool_name):
            session_aware_outcome = await ctx.service._dispatch_session_aware_tool(
                tool_name=tool_name,
                tool_call_id=tool_call.id,
                arguments=arguments,
                state=state,
                audit=audit,
                recorder=recorder,
                session_id=session_id,
                current_state_id=current_state_id,
                response=response,
                llm_messages=llm_messages,
                anti_anchor=anti_anchor,
            )
            all_cache_hits = False
            _append_tool_outcome(
                response=session_aware_outcome.result,
                error_class=session_aware_outcome.error_class,
                error_message=session_aware_outcome.error_message,
                post_version=session_aware_outcome.post_version,
            )
            if session_aware_outcome.is_discovery:
                turn_has_discovery = True
            else:
                turn_has_mutation = True
            if session_aware_outcome.result is not None:
                state = session_aware_outcome.result.updated_state
                last_validation = session_aware_outcome.result.validation
            continue

        # Precompute runtime preflight for preview_pipeline outside
        # the general side-effectful tool worker. This keeps
        # execute_tool() synchronous and bounds the async I/O cost
        # before it enters the worker thread pool.
        runtime_preflight_callback: RuntimePreflight | None = None
        if tool_name == "preview_pipeline":
            try:
                preview_preflight = await ctx.service._cached_runtime_preflight(
                    state,
                    user_id=user_id,
                    session_id=session_id,
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
            _user_message_id: str | None = user_message_id,
            _user_message_content: str | None = user_message_content,
            _composer_model_identifier: str = ctx.service._model,
            _composer_model_version: str = safe_response_model(response) or ctx.service._model,
            _composer_provider: str = ctx.service._availability.provider or "unknown",
            _composer_skill_hash: str = ctx.service._composer_skill_hash,
            _tool_arguments_hash: str = audit.arguments_hash,
        ) -> Any:
            return await run_sync_in_worker(
                execute_tool,
                _tool_name,
                _arguments,
                _state,
                ctx.service._catalog,
                data_dir=ctx.service._data_dir,
                session_engine=ctx.service._session_engine,
                session_id=session_id,
                secret_service=ctx.service._secret_service,
                user_id=user_id,
                prior_validation=_last_validation,
                runtime_preflight=_runtime_preflight_callback,
                max_blob_storage_per_session_bytes=ctx.service._settings.max_blob_storage_per_session_bytes,
                user_message_id=_user_message_id,
                user_message_content=_user_message_content,
                composer_model_identifier=_composer_model_identifier,
                composer_model_version=_composer_model_version,
                composer_provider=_composer_provider,
                composer_skill_hash=_composer_skill_hash,
                tool_arguments_hash=_tool_arguments_hash,
                raise_schema_argument_errors=True,
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
            await emit_progress(
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
                error_message=type(tool_exc).__name__,
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
        # Mark the (plugin_type, plugin_name) pair as
        # schema-loaded for this session when a get_plugin_schema
        # call returned a discovery success. The per-turn system
        # context renders this set as
        # ``composer_progress.schemas_loaded_this_session`` so
        # the model can compute its own schemas_gap without
        # re-introspecting plugins it has already seen. See
        # ``ComposerServiceImpl._mark_plugin_schema_loaded`` and
        # ``prompts.build_context_string``.
        #
        # Lifted from the prior inline compose-loop body into the
        # extracted ``_dispatch_tool_batch`` helper as part of the
        # parallel decomposition (commit f918f4269). The RC5.2
        # commit that introduced this tracker explicitly anticipated
        # the lift in its message: "the single line that will need
        # lift-and-shift into ``dispatch_tools`` when the parallel
        # ``_compose_loop`` decomposition lands".
        if tool_name == "get_plugin_schema" and result.success:
            ctx.service._mark_plugin_schema_loaded(
                session_id,
                str(arguments["plugin_type"]),
                str(arguments["name"]),
            )
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
                mutation_success_observed = True
                anti_anchor.record_success()
        else:
            anti_anchor.record_failure(tool_name, audit.arguments_hash)
        result_json = _serialize_tool_result(result)
        await emit_progress(progress, tool_completed_progress_event(tool_name, result.success))
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

    dispatch = _DispatchOutcome(
        state=state,
        last_validation=last_validation,
        last_runtime_preflight=last_runtime_preflight,
        tool_outcomes=tuple(tool_outcomes),
        decoded_args_by_call_id=decoded_args_by_call_id,
        turn_has_mutation=turn_has_mutation,
        turn_has_discovery=turn_has_discovery,
        all_cache_hits=all_cache_hits,
        plugin_crash=plugin_crash,
        plugin_crash_cause=plugin_crash_cause,
        assistant_message=assistant_message,
        raw_assistant_content=raw_assistant_content,
        assistant_tool_calls=assistant_tool_calls,
        mutation_success_observed=mutation_success_observed,
        pre_state_id=pre_state_id,
    )
    return dispatch, advisor_calls_used
