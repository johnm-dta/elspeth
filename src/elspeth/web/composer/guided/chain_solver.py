"""Chain solver: invoke LLM with guided skill + GUIDED CONTEXT block, parse propose_chain."""

from __future__ import annotations

import json
from typing import Any

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
    """
    skill = load_guided_skill()
    context_block = build_step_3_context_block(source=source, sink=sink, recipe_match=recipe_match)
    if repair_context is not None:
        repair_addendum = build_repair_addendum(validation_error=repair_context)
        system_prompt = f"{skill}\n\n{context_block}\n\n{repair_addendum}"
    else:
        system_prompt = f"{skill}\n\n{context_block}"
    seed = _composer_llm_seed_for_model(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}],
        "tools": _GUIDED_LLM_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "emit_turn"}},
        "temperature": _COMPOSER_LLM_TEMPERATURE,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = await _litellm_acompletion(**kwargs)
    name, arguments = _extract_tool_call(response)

    # Wire-boundary validation of the LLM-produced emit_turn shape.  The
    # LLM is an external system (Tier 3); the JSON tool schema above is the
    # contract, but providers vary in how strictly they enforce it.  This
    # block is the consumer-side backstop: any shape violation raises
    # ``ChainSolverResponseShapeError``, which ``solve_chain_with_auto_drop``
    # catches and routes through the SOLVER_EXHAUSTED auto-drop path.
    #
    # Routing all three layers (wrong tool name, wrong turn_type, malformed
    # payload) through the same exception class keeps the dispatch
    # single-pathed.  The ``tool_choice`` constraint above forces ``emit_turn``,
    # so the name-mismatch case is rare in practice -- but a provider bug or
    # an upstream proxy mutation could still produce it, and treating it the
    # same as the other shape failures is correct.
    #
    # Error messages carry only the offending value(s) -- no LLM response
    # body content, no system prompt, no GUIDED CONTEXT.  The wrapper's
    # audit emission already redacts down to ``type(exc).__name__``; the
    # message text is for slog frame context only.
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
    return ChainProposal(
        steps=steps,
        why=str(why_raw),
    )


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
