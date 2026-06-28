"""Chain solver: invoke LLM with guided skill + GUIDED CONTEXT block, parse propose_chain."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from typing import Any, Final

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided._discovery import _assistant_tool_calls_message, _execute_discovery_call

# Redundant alias marks this an explicit re-export (mypy no_implicit_reexport):
# back-compat call sites still do ``from ...chain_solver import ChainSolverResponseShapeError``.
from elspeth.web.composer.guided.errors import ChainSolverResponseShapeError as ChainSolverResponseShapeError
from elspeth.web.composer.guided.prompts import (
    build_repair_addendum,
    build_revise_addendum,
    build_step_3_context_block,
    load_guided_skill,
)
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.llm_response_parsing import (
    attach_llm_calls,
    build_llm_call_record,
)
from elspeth.web.composer.service import _litellm_acompletion
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._dispatch import get_discovery_tool_definitions

# ``ChainSolverResponseShapeError`` now lives in ``errors.py`` (the guided
# exception taxonomy) so the discovery loop (``_discovery``, which this module
# imports) can raise it on malformed tool-call arguments without a circular
# import. It is imported above and re-exported here for backward compatibility
# (existing call sites do ``from ...chain_solver import ChainSolverResponseShapeError``).


# CLOSED LIST — turn_type is intentionally restricted to "propose_chain"
# because this is the only turn type the chain-solver consumer handles.
# The full TurnType enum lives in protocol.py; do not relist values here
# without updating the consumer and the auto-drop contract.
#
# The schema is the wire contract.  Strict-mode-capable providers (OpenAI)
# enforce it; other providers may not.  The consumer (`solve_chain` below)
# carries a backstop that catches shape failures and routes them through
# ``solve_chain_with_auto_drop`` via :class:`ChainSolverResponseShapeError`.
#
# ``additionalProperties: False`` is set on the outer parameters object and
# on each step item to make the contract strict -- LLMs sometimes invent
# extra keys ("notes", "confidence") that the consumer would silently drop.
_GUIDED_LLM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "emit_turn",
            "description": "Emit one turn to the user. The only way to interact in guided mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_type": {
                        "type": "string",
                        "enum": ["propose_chain"],
                    },
                    "payload": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "plugin": {"type": "string"},
                                        # ``options`` is intentionally left as a bare ``object`` (no
                                        # ``additionalProperties: false``, no inner ``properties``)
                                        # because the shape varies by plugin: ``csv`` takes
                                        # ``path``/``schema``, ``json`` takes
                                        # ``path``/``schema``/``collision_policy``, ``llm`` takes
                                        # ``model``/``template``/``api_key_secret``/..., etc.
                                        # Constraining it here would block legitimate plugin
                                        # shapes.  Plugin-specific validation happens downstream
                                        # in ``handle_step_3_chain_accept`` -> ``_execute_set_pipeline``.
                                        "options": {"type": "object"},
                                        "rationale": {"type": "string"},
                                    },
                                    "required": ["plugin", "options", "rationale"],
                                    "additionalProperties": False,
                                },
                            },
                            "why": {"type": "string"},
                        },
                        "required": ["steps", "why"],
                        "additionalProperties": False,
                    },
                },
                "required": ["turn_type", "payload"],
                "additionalProperties": False,
            },
        },
    },
]


_STEP_3_TRANSFORM_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset({"list_transforms", "get_plugin_schema", "list_models"})
"""Read-only discovery tools the transform stage offers the composer model.

``list_transforms`` answers "which transform plugins exist", ``get_plugin_schema``
answers "what options does this transform take" (the authority on the
``web_scrape`` required fields + ``http.*`` nesting the model otherwise guesses
wrong), and ``list_models`` answers "which model ids may an ``llm`` step pin" —
the three facts a model needs to build a valid chain without a hand-maintained
inventory baked into the prompt. Every name here is asserted
``<= _DISCOVERY_TOOL_NAMES`` inside :func:`get_discovery_tool_definitions`.
"""

_DEFAULT_MAX_DISCOVERY_ITERS: Final[int] = 6
"""Fallback discovery-iteration cap when the route does not pass one.

Production threads ``settings.composer_max_discovery_turns``; this default keeps
direct callers (and tests) bounded. Reaching the cap raises
:class:`ChainSolverResponseShapeError` (routed to the auto-drop path), never
loops forever.
"""


def _parse_emit_turn_call(call: Any) -> ChainProposal:
    """Validate one ``emit_turn`` tool call into a :class:`ChainProposal`.

    The wire-boundary backstop that previously lived inline in ``solve_chain``:
    the LLM is an external system (Tier 3) and the JSON tool schema is the
    contract, but providers vary in how strictly they enforce it. Any shape
    violation raises :class:`ChainSolverResponseShapeError` (the typed-except
    clause maps it to ``MALFORMED_RESPONSE`` and the auto-drop wrapper routes
    it); a ``json.JSONDecodeError`` on ``arguments`` propagates unchanged (also
    ``MALFORMED_RESPONSE``).

    Error messages carry only the offending value(s) — no LLM response body, no
    system prompt, no GUIDED CONTEXT. The wrapper's audit emission redacts down
    to ``type(exc).__name__``; the message text is for slog frame context only.
    """
    name = call.function.name
    if name != "emit_turn":
        raise ChainSolverResponseShapeError(f"chain solver expected emit_turn, got {name!r}")
    arguments = json.loads(call.function.arguments)
    try:
        turn_type = arguments["turn_type"]
        payload = arguments["payload"]
    except (KeyError, TypeError) as exc:
        raise ChainSolverResponseShapeError(
            f"chain solver emit_turn arguments missing required keys (turn_type/payload): {type(exc).__name__}"
        ) from exc
    if turn_type != "propose_chain":
        raise ChainSolverResponseShapeError(f"chain solver expected propose_chain turn, got {turn_type!r}")
    try:
        steps_raw = payload["steps"]
        why_raw = payload["why"]
    except (KeyError, TypeError) as exc:
        raise ChainSolverResponseShapeError(
            f"chain solver propose_chain payload missing required keys (steps/why): {type(exc).__name__}"
        ) from exc
    try:
        steps = tuple(dict(s) for s in steps_raw)
    except (TypeError, ValueError) as exc:
        # ``tuple(dict(s) for s in steps_raw)`` fails when steps_raw is not
        # iterable (TypeError), or when an element is not dict-coercible
        # (TypeError or ValueError). All such failures are LLM shape violations.
        raise ChainSolverResponseShapeError(
            f"chain solver propose_chain payload.steps is not a list of dicts: {type(exc).__name__}"
        ) from exc
    return ChainProposal(steps=steps, why=str(why_raw))


async def solve_chain(
    *,
    model: str,
    source: SourceResolved,
    sink: SinkResolved,
    intent: str | None = None,
    repair_context: str | None = None,
    revise_context: str | None = None,
    recorder: BufferingRecorder | None = None,
    temperature: float | None,
    seed: int | None,
    state: CompositionState | None = None,
    catalog: CatalogService | None = None,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    max_discovery_iters: int | None = None,
) -> ChainProposal:
    """Invoke the LLM with the guided skill, expect a propose_chain turn back.

    Reuses the same _litellm_acompletion path as freeform composer so audit,
    telemetry, and token accounting flow through the same plumbing.

    Args:
        model: LiteLLM model identifier from settings.composer_model.  Required —
            callers must be explicit; no hard-coded default.
        intent: The user's originating free-text transform request (e.g. the
            frozen tutorial prompt "scrape these pages and have an LLM extract
            ..."). Rendered as a USER-role message so the model builds the chain
            FROM the request — the freeform mirror. ``None`` (legacy callers)
            reproduces the prior source/sink-contract-only build.
        repair_context: Verbatim validation error text from the failing ToolResult.
            When set, the LLM is asked to correct the named errors rather than
            propose an independent first-pass chain.
        revise_context: Verbatim user revise instruction. When set, the LLM is
            asked to UPDATE the current proposal per the instruction (via
            build_revise_addendum) — distinct from repair_context, which frames the
            text as a validation-failure to correct. The STEP_3 /guided/chat branch
            uses this; the genuine validation-repair loop uses repair_context.
        recorder: Optional :class:`BufferingRecorder` to receive a
            :class:`ComposerLLMCall` audit row for the LLM invocation. When
            supplied, exactly one record is appended via ``record_llm_call`` on
            every exit path — SUCCESS, the six error-classification statuses
            (TIMEOUT / CANCELLED / AUTH_ERROR / BAD_REQUEST_ERROR / API_ERROR /
            MALFORMED_RESPONSE), and the F5 catch-all. Recording mirrors the
            freeform composer pattern at ``composer/service.py:3173-3309``:
            ``started_at`` / ``started_ns`` captured before the call, status
            assigned in each typed ``except`` clause, the record built in
            ``finally`` so even non-typed failures land in the audit trail.

            When ``None``, no audit row is recorded — the call still happens
            and the same exception semantics hold. The default is ``None`` to
            keep ``solve_chain`` callable from contexts that pre-date this
            audit wiring (older test fixtures).
    """
    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    skill = load_guided_skill()
    context_block = build_step_3_context_block(source=source, sink=sink)
    # Additive, branch-total render: repair_context and revise_context are
    # mutually exclusive by call-site, but appending each independently keeps the
    # function total (a raise here would 500 and brick the phase). Neither-set and
    # repair-only render byte-identically to the prior nested form.
    system_prompt = f"{skill}\n\n{context_block}"
    if repair_context is not None:
        system_prompt = f"{system_prompt}\n\n{build_repair_addendum(validation_error=repair_context)}"
    if revise_context is not None:
        system_prompt = f"{system_prompt}\n\n{build_revise_addendum(revise_instruction=revise_context)}"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    # The originating transform request rides as a USER-role message — the freeform
    # mirror. Freeform feeds the frozen prompt as the final user message
    # (composer/prompts.py:401); the guided sink resolver already does the same
    # (chat_solver.py:582-585). Without it solve_chain proposed the chain blind
    # from the source/sink contract alone (the passthrough-instead-of-web_scrape
    # bug). repair_context/revise_context stay system-prompt addenda — they frame
    # an EXISTING proposal to correct/change, not the first-pass build request.
    if intent is not None:
        messages.append({"role": "user", "content": intent})

    # Discovery is active only when both ``state`` and ``catalog`` are supplied
    # (the guided routes thread them). Without them the loop runs exactly once
    # with FORCED ``tool_choice: emit_turn`` and no discovery tools — the prior
    # single-shot behaviour, byte-identical for every pre-discovery caller/test.
    # With them, the model may call ``list_transforms`` / ``get_plugin_schema`` /
    # ``list_models`` (under ``tool_choice: required``, which keeps the chain
    # solver's "act only via a tool" invariant while letting the model choose
    # discovery before it commits) and the solver threads results back until the
    # model emits ``propose_chain``.
    discovery_enabled = state is not None and catalog is not None
    discovery_defs = get_discovery_tool_definitions(_STEP_3_TRANSFORM_DISCOVERY_TOOL_NAMES) if discovery_enabled else []
    allowed_discovery = _STEP_3_TRANSFORM_DISCOVERY_TOOL_NAMES if discovery_enabled else frozenset()
    tools = [*_GUIDED_LLM_TOOLS, *discovery_defs]
    actor = user_id or "guided-composer"
    if discovery_enabled:
        iteration_cap = max_discovery_iters if max_discovery_iters is not None else _DEFAULT_MAX_DISCOVERY_ITERS
    else:
        iteration_cap = 1

    for _iteration in range(max(1, iteration_cap)):
        # Snapshot the messages the wire request carries THIS round, so the audit
        # row records exactly what was sent even as the loop appends tool results
        # for the next round.
        request_messages = list(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": request_messages, "tools": tools}
        kwargs["tool_choice"] = "required" if discovery_enabled else {"type": "function", "function": {"name": "emit_turn"}}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

        # Capture timing bracket BEFORE the call so latency_ms is end-to-end and
        # the audit row stamps the moment the wire request started, not the moment
        # the response landed.  Mirrors composer/service.py:3195-3196 and 3337-3338.
        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        try:
            response = await _litellm_acompletion(**kwargs)
            message = response.choices[0].message
            tool_calls = message.tool_calls or ()

            # ``emit_turn`` (propose_chain) is terminal — take it regardless of
            # any sibling discovery call in the same message. ``_parse_emit_turn_call``
            # carries the exact wire-boundary shape validation (and its messages)
            # that previously lived inline here.
            emit_call = next(
                (tc for tc in tool_calls if tc.function is not None and tc.function.name == "emit_turn"),
                None,
            )
            if emit_call is not None:
                proposal = _parse_emit_turn_call(emit_call)
                # Mark SUCCESS only AFTER parsing succeeded — any shape failure is
                # MALFORMED_RESPONSE, not SUCCESS-with-bad-data.
                status = ComposerLLMCallStatus.SUCCESS
                return proposal

            if not discovery_enabled:
                # Single-shot path: forced ``tool_choice`` guarantees a call, so
                # an empty response or a wrong-named call is a shape violation.
                # Messages preserved verbatim for the existing test matchers
                # (``match="emit_turn"`` / empty-tool_calls).
                if not tool_calls:
                    raise ChainSolverResponseShapeError("chain solver response had no tool_calls")
                first = tool_calls[0].function
                first_name = first.name if first is not None else None
                raise ChainSolverResponseShapeError(f"chain solver expected emit_turn, got {first_name!r}")

            # Discovery path execution-side safety gate: the only non-terminal
            # calls we dispatch are allowed read-only discovery tools. ANY other
            # tool (a hallucinated mutation / secret call) raises WITHOUT
            # dispatching — ``execute_tool``'s handler union would otherwise
            # happily run it. The raise routes to the auto-drop path exactly like
            # today's transient/no-proposal outcome.
            discovery_calls = [tc for tc in tool_calls if tc.function is not None and tc.function.name in allowed_discovery]
            if not discovery_calls or len(discovery_calls) != len(tool_calls):
                raise ChainSolverResponseShapeError(
                    "chain solver emitted a tool call that is neither emit_turn nor an allowed discovery tool"
                )

            # Thread the assistant tool-call request once, then answer every call
            # id with its result, or the next round 400s.
            assert state is not None and catalog is not None  # implied by discovery_enabled
            messages.append(_assistant_tool_calls_message(message, tool_calls))
            for tool_call in tool_calls:
                messages.append(
                    _execute_discovery_call(
                        tool_call=tool_call,
                        state=state,
                        catalog=catalog,
                        secret_service=secret_service,
                        user_id=user_id,
                        actor=actor,
                        recorder=recorder,
                    )
                )
            status = ComposerLLMCallStatus.SUCCESS
            # fall through to finally (records this round), then loop again
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            # Client disconnect or task cancellation. Record the call (so the audit
            # row exists for forensics) but re-raise unchanged — auto-drop wrappers
            # deliberately do NOT absorb CancelledError per ``solve_chain_with_auto_drop``.
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
            # Listed after the more specific Auth/BadRequest subclasses.
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except (IndexError, AttributeError, json.JSONDecodeError, ChainSolverResponseShapeError) as exc:
            # LLM-response shape failures: empty ``choices`` (IndexError), missing
            # ``message`` / ``tool_calls`` attribute (AttributeError), invalid JSON
            # in ``arguments`` (JSONDecodeError), and the consumer-side
            # ``ChainSolverResponseShapeError`` (wrong tool / turn_type, malformed
            # payload, gate-trip, no-tool_calls). All classify as MALFORMED_RESPONSE.
            # Mirrors composer/service.py:3373-3378.
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            error_class = type(exc).__name__
            error_message = "malformed_response"
            raise
        except Exception as exc:
            # F5 catch-all (mirrors composer/service.py:3275-3289). Ensures the
            # audit row lands even for exception classes outside the typed clauses
            # above — e.g. httpx ConnectionError, codec ValueError, or any other
            # transport / runtime failure. ``API_ERROR`` is the closest semantic
            # for "unknown provider-side or server-side failure"; the exception
            # class name is preserved in ``error_class`` for forensics.
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=model,
                        messages=request_messages,
                        tools=tools,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=temperature,
                        seed=seed,
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                    )
                )
                # Attach the buffered call list to the in-flight exception (if any)
                # so callers further out the stack can recover the audit trail even
                # when the exception escapes the recorder's drain scope.  Mirrors
                # composer/service.py:3307-3309 and :3404-3406.
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    attach_llm_calls(current_exc, recorder)

    # Discovery iteration cap reached without an ``emit_turn`` — a shape failure
    # that routes to the auto-drop path (``ChainSolverResponseShapeError`` is in
    # both transient sets), exactly like a transient/no-proposal outcome.
    raise ChainSolverResponseShapeError("chain solver discovery loop reached the iteration cap without an emit_turn")
