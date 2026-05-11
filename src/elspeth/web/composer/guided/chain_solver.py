"""Chain solver: invoke LLM with guided skill + GUIDED CONTEXT block, parse propose_chain."""

from __future__ import annotations

import json
from typing import Any

from elspeth.web.composer.guided.prompts import (
    build_step_3_context_block,
    load_guided_skill,
)
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.service import _litellm_acompletion

# CLOSED LIST — turn_type enum mirrors TurnType in protocol.py; do not extend
# without updating the legal-turn matrix.
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
                        "enum": [
                            "inspect_and_confirm",
                            "single_select",
                            "multi_select_with_custom",
                            "schema_form",
                            "propose_chain",
                            "recipe_offer",
                        ],
                    },
                    "payload": {"type": "object"},
                },
                "required": ["turn_type", "payload"],
            },
        },
    },
]


async def solve_chain(
    *,
    source: SourceResolved,
    sink: SinkResolved,
    recipe_match: RecipeMatch | None = None,
) -> ChainProposal:
    """Invoke the LLM with the guided skill, expect a propose_chain turn back.

    Reuses the same _litellm_acompletion path as freeform composer so audit,
    telemetry, and token accounting flow through the same plumbing.
    """
    skill = load_guided_skill()
    context_block = build_step_3_context_block(source=source, sink=sink, recipe_match=recipe_match)
    system_prompt = f"{skill}\n\n{context_block}"
    response = await _litellm_acompletion(
        model="anthropic/claude-3.5-sonnet",
        messages=[{"role": "system", "content": system_prompt}],
        tools=_GUIDED_LLM_TOOLS,
        tool_choice={"type": "function", "function": {"name": "emit_turn"}},
    )
    name, arguments = _extract_tool_call(response)
    if name != "emit_turn":
        raise ValueError(f"chain solver expected emit_turn, got {name!r}")
    if arguments["turn_type"] != "propose_chain":
        raise ValueError(f"chain solver expected propose_chain turn, got {arguments['turn_type']!r}")
    payload = arguments["payload"]
    return ChainProposal(
        steps=tuple(dict(s) for s in payload["steps"]),
        why=str(payload["why"]),
    )


def _extract_tool_call(response: Any) -> tuple[str, dict[str, Any]]:
    """Extract (name, parsed_arguments) from a LiteLLM response.

    LiteLLM returns attribute-access objects (not dicts); see _FakeLLMResponse
    in tests/unit/web/composer/test_compose_loop_llm_audit.py for the shape.
    """
    message = response.choices[0].message
    tool_calls = message.tool_calls
    if not tool_calls:
        raise ValueError("solve_chain: response had no tool_calls")
    call = tool_calls[0]
    return call.function.name, json.loads(call.function.arguments)
