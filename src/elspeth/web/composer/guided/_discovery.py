"""Shared discovery-loop primitives for the guided per-phase solvers.

Both the sink solver (``chat_solver.maybe_resolve_step_2_sink_chat``) and the
transform-chain solver (``chain_solver.solve_chain``) expose the freeform MCP
discovery tools (``list_*`` / ``get_plugin_schema`` / ``list_models``) to the
composer model so it can look up real plugins and their option schemas before it
commits a stage. The two solvers differ only in their *terminal* tool
(``resolve_sink`` vs ``emit_turn``/``propose_chain``) and their terminal parse;
the provider-protocol plumbing — re-materialising the assistant tool-call turn
and dispatching one read-only discovery call with its audit row — is identical
and lives here so a fix to either lands in both.

These functions carry NO solver-specific policy: the *caller* owns the
execution-side safety gate (proving a call is an allowed discovery tool before
dispatching), the terminal handling, and the iteration cap. ``_execute_discovery_call``
trusts that its ``tool_call`` was already gated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.audit import BufferingRecorder, begin_dispatch, finish_success
from elspeth.web.composer.guided.errors import ChainSolverResponseShapeError
from elspeth.web.composer.service import _serialize_tool_result
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._dispatch import execute_tool


def _assistant_tool_calls_message(message: Any, tool_calls: Any) -> dict[str, Any]:
    """Re-materialise the assistant turn that requested *tool_calls*.

    The OpenAI/LiteLLM tool protocol requires the assistant message carrying
    the tool-call request to precede the ``role=tool`` results in the next
    round, and every ``tool_call_id`` it names must be answered. We rebuild it
    explicitly (rather than re-appending the raw provider object) so the wire
    shape is deterministic and provider-agnostic.
    """
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
            for tool_call in tool_calls
        ],
    }


def _execute_discovery_call(
    *,
    tool_call: Any,
    state: CompositionState,
    catalog: CatalogService,
    secret_service: WebSecretResolver | None,
    user_id: str | None,
    actor: str,
    recorder: BufferingRecorder | None,
) -> dict[str, Any]:
    """Dispatch one read-only discovery call, audit it, and build its result message.

    The caller has already proven ``tool_call.function.name`` is an allowed
    discovery tool (the execution-side safety gate), so dispatching through
    ``execute_tool`` — whose handler union also contains mutation tools — is
    safe here. Discovery tools never advance ``CompositionState`` so
    ``version_before == version_after``.

    A semantically-failed result (e.g. ``get_plugin_schema`` for an unknown
    plugin) is still threaded back to the model verbatim so it can correct
    itself; that matches the freeform loop's behaviour.
    """
    function = tool_call.function
    name = function.name
    raw_arguments = function.arguments
    # Malformed tool-call arguments are an LLM RESPONSE-SHAPE failure, not a
    # server/client bug: route them through ``ChainSolverResponseShapeError`` so
    # the solver's ``solve_chain_with_auto_drop`` wrapper auto-drops to freeform
    # (same bucket as a malformed ``emit_turn``). A bare ``ValueError`` /
    # ``JSONDecodeError`` here would escape that wrapper (it deliberately excludes
    # ``ValueError`` to preserve client-payload-bug routing) and surface as a 500.
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) and raw_arguments.strip() else {}
    except json.JSONDecodeError as exc:
        raise ChainSolverResponseShapeError(f"{name} arguments are not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ChainSolverResponseShapeError(f"{name} arguments must decode to an object; got {type(parsed).__name__}")
    arguments = dict(parsed)
    result = execute_tool(name, arguments, state, catalog, secret_service=secret_service, user_id=user_id)
    if recorder is not None:
        audit = begin_dispatch(tool_call.id, name, arguments, version_before=state.version, actor=actor)
        invocation = finish_success(audit, result_payload=result.to_dict(), version_after=state.version)
        recorder.record(invocation)
    return {"role": "tool", "tool_call_id": tool_call.id, "content": _serialize_tool_result(result)}
