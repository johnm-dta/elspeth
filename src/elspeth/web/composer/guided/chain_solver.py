"""Chain solver: invoke LLM with guided skill + GUIDED CONTEXT block, parse propose_chain."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from typing import Any

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.prompts import (
    build_repair_addendum,
    build_step_3_context_block,
    load_guided_skill,
)
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.llm_response_parsing import (
    attach_llm_calls,
    build_llm_call_record,
)
from elspeth.web.composer.service import (
    _COMPOSER_LLM_TEMPERATURE,
    _composer_llm_seed_for_model,
    _litellm_acompletion,
)


class ChainSolverResponseShapeError(Exception):
    """The LLM emitted an emit_turn response that violated the chain solver's
    contract (wrong tool name, wrong turn_type, missing/malformed payload).

    Distinct from :class:`InvariantError` (server-side bug) and ``ValueError``
    (client-payload bug): this exception means an *external system* (the LLM)
    produced an unexpected response shape. ``solve_chain_with_auto_drop``
    catches this class and routes the request through the SOLVER_EXHAUSTED
    auto-drop path -- the same bucket as other malformed-response-shape
    failures (``IndexError`` from empty ``choices``, ``json.JSONDecodeError``
    on tool-call arguments, etc.).

    NOT a subclass of ``ValueError`` because the auto-drop wrapper docstring
    explicitly excludes ``ValueError`` to preserve client-payload-bug routing.
    NOT a subclass of ``InvariantError`` because that class is documented as
    "server invariant violated" -- the wrong category for "external LLM
    misbehaved."

    Inside ``solve_chain`` the audit wrap's typed-except clause maps this
    class to :attr:`ComposerLLMCallStatus.MALFORMED_RESPONSE` -- semantically
    the LLM did respond, but the response failed contract.

    Spec gap (tracked separately):
        Spec §5.4 case 1 calls for "reject + grant one retry → second illegal
        emission triggers auto-drop with reason=protocol_violation."  The
        current implementation single-shots the auto-drop with
        reason=solver_exhausted (the existing wrapper's outcome) because
        wiring the retry state machine is feature work, not a bug fix.  The
        spec-mandated retry-then-PROTOCOL_VIOLATION flow is filed as a
        follow-up observation.
    """


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


async def solve_chain(
    *,
    model: str,
    source: SourceResolved,
    sink: SinkResolved,
    recipe_match: RecipeMatch | None = None,
    repair_context: str | None = None,
    recorder: BufferingRecorder | None = None,
) -> ChainProposal:
    """Invoke the LLM with the guided skill, expect a propose_chain turn back.

    Reuses the same _litellm_acompletion path as freeform composer so audit,
    telemetry, and token accounting flow through the same plumbing.

    Args:
        model: LiteLLM model identifier from settings.composer_model.  Required —
            callers must be explicit; no hard-coded default.
        repair_context: Verbatim validation error text from the failing ToolResult.
            When set, the LLM is asked to correct the named errors rather than
            propose an independent first-pass chain.
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
    context_block = build_step_3_context_block(source=source, sink=sink, recipe_match=recipe_match)
    if repair_context is not None:
        repair_addendum = build_repair_addendum(validation_error=repair_context)
        system_prompt = f"{skill}\n\n{context_block}\n\n{repair_addendum}"
    else:
        system_prompt = f"{skill}\n\n{context_block}"
    seed = _composer_llm_seed_for_model(model)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": _GUIDED_LLM_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "emit_turn"}},
        "temperature": _COMPOSER_LLM_TEMPERATURE,
    }
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
        name, arguments = _extract_tool_call(response)

        # Wire-boundary validation of the LLM-produced emit_turn shape.  The
        # LLM is an external system (Tier 3); the JSON tool schema above is
        # the contract, but providers vary in how strictly they enforce it.
        # This block is the consumer-side backstop: any shape violation raises
        # ``ChainSolverResponseShapeError``, which is caught by the typed-
        # except clause below and routed through the auto-drop path.
        #
        # Error messages carry only the offending value(s) -- no LLM response
        # body content, no system prompt, no GUIDED CONTEXT.  The wrapper's
        # audit emission redacts down to ``type(exc).__name__``; the message
        # text is for slog frame context only.
        if name != "emit_turn":
            raise ChainSolverResponseShapeError(f"chain solver expected emit_turn, got {name!r}")
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
            # (TypeError or ValueError, depending on the Python version and the
            # specific malformed shape).  All such failures are LLM shape
            # violations.
            raise ChainSolverResponseShapeError(
                f"chain solver propose_chain payload.steps is not a list of dicts: {type(exc).__name__}"
            ) from exc
        proposal = ChainProposal(
            steps=steps,
            why=str(why_raw),
        )
        # Mark SUCCESS only AFTER all post-call parsing succeeded — any shape
        # failure between _litellm_acompletion returning and ChainProposal
        # construction is MALFORMED_RESPONSE, not SUCCESS-with-bad-data.
        status = ComposerLLMCallStatus.SUCCESS
        return proposal
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
        # LLM-response shape failures.  This covers two layers:
        #
        # - ``_extract_tool_call`` shape failures: empty ``choices``
        #   (IndexError), missing ``message`` / ``tool_calls`` attribute
        #   (AttributeError), invalid JSON in ``arguments`` (JSONDecodeError).
        #
        # - Consumer-side shape failures: the parser above wraps KeyError /
        #   TypeError / ValueError on payload / steps access into
        #   ``ChainSolverResponseShapeError`` and the no-tool_calls branch
        #   in ``_extract_tool_call`` raises ``ChainSolverResponseShapeError``
        #   directly.  All such failures classify as MALFORMED_RESPONSE.
        #
        # Mirrors the freeform composer mapping at composer/service.py:
        # 3373-3378 for empty-choices MALFORMED_RESPONSE.
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
                    messages=messages,
                    tools=_GUIDED_LLM_TOOLS,
                    status=status,
                    started_at=started_at,
                    started_ns=started_ns,
                    temperature=_COMPOSER_LLM_TEMPERATURE,
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


def _extract_tool_call(response: Any) -> tuple[str, dict[str, Any]]:
    """Extract (name, parsed_arguments) from a LiteLLM response.

    LiteLLM returns attribute-access objects (not dicts); see _FakeLLMResponse
    in tests/unit/web/composer/test_compose_loop_llm_audit.py for the shape.

    Sibling shape failures from this function (``IndexError`` from empty
    ``choices``, ``AttributeError`` from missing ``message`` / ``tool_calls``,
    ``json.JSONDecodeError`` from malformed arguments JSON) are caught by
    :func:`solve_chain_with_auto_drop`.  The empty-tool_calls branch raises
    :class:`ChainSolverResponseShapeError` to route through the same path --
    the LLM responded but skipped the tool, which is an external-system
    shape failure, not a server invariant violation.
    """
    message = response.choices[0].message
    tool_calls = message.tool_calls
    if not tool_calls:
        raise ChainSolverResponseShapeError("chain solver response had no tool_calls")
    call = tool_calls[0]
    return call.function.name, json.loads(call.function.arguments)
