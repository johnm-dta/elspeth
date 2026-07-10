"""Sink stage discovery-tool loop.

The guided sink solver was single-shot (only ``resolve_sink``), so the
composer "light" model had no plugin knowledge and replied in prose — the
route then fell back to the manual ``single_select`` picker. These tests pin
the agentic discovery loop: the model calls read-only discovery tools
(``list_sinks`` / ``get_plugin_schema``) at runtime, the solver executes them
via the production ``execute_tool`` dispatcher and threads the results back,
and the model then calls ``resolve_sink``.

The loop is exercised against the REAL ``execute_tool`` + a ``spec``-locked
catalog mock, so these tests cover dispatch integration and the
execution-side safety gate, not just message threading.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_2_sink_chat
from elspeth.web.composer.guided.resolved import SinkResolved
from elspeth.web.composer.state import CompositionState, PipelineMetadata


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


class _SinkCatalog:
    def list_sinks(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="json", description="JSON Lines sink", plugin_type="sink", config_fields=[]),
        ]

    def get_schema(self, plugin_type: str, name: str) -> PluginSchemaInfo:
        assert (plugin_type, name) == ("sink", "json")
        return PluginSchemaInfo(
            name="json",
            plugin_type="sink",
            description="JSON Lines sink",
            json_schema={"title": "Config", "properties": {}},
            knob_schema={"fields": []},
        )


def _catalog() -> CatalogService:
    return _SinkCatalog()


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(*, content: str | None = None, tool_calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


_RESOLVE_SINK_ARGS = {
    "resolution": "sink",
    "outputs": [
        {
            "plugin": "json",
            "options": {"path": "out.jsonl", "schema": {"mode": "observed"}},
            "required_fields": [],
            "schema_mode": "observed",
        }
    ],
    "assistant_message": "Saved the results as a JSON Lines file.",
}


@pytest.mark.asyncio
async def test_sink_loop_lists_sinks_then_resolves() -> None:
    """list_sinks -> (tool result threaded back) -> resolve_sink."""
    responses = [
        _response(tool_calls=[_tool_call("c1", "list_sinks", {})]),
        _response(tool_calls=[_tool_call("c2", "resolve_sink", _RESOLVE_SINK_ARGS)]),
    ]
    recorder = BufferingRecorder()
    captured_messages: list[list[dict[str, Any]]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured_messages.append(kwargs["messages"])
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save the results as a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
        )

    sink, assistant = result.sink, result.assistant_message
    assert sink is not None
    assert isinstance(sink, SinkResolved)
    assert sink.outputs[0].plugin == "json"
    assert assistant == "Saved the results as a JSON Lines file."

    # The second LLM call must carry the list_sinks tool result keyed to c1.
    second_call_messages = captured_messages[1]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in second_call_messages)
    # ... and the assistant tool-call request that precedes it.
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_call_messages)

    # Audit parity: one ComposerLLMCall per provider round, one tool invocation.
    assert len(recorder.llm_calls) == 2
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].tool_name == "list_sinks"


@pytest.mark.asyncio
async def test_sink_loop_refuses_to_dispatch_mutation_tool() -> None:
    """Execution-side safety gate: a hallucinated mutation call is NOT dispatched.

    Advertising only the read-only discovery subset is not the gate —
    ``execute_tool``'s handler union includes every mutation registry, so the
    solver must refuse to dispatch any name that is neither ``resolve_sink``
    nor an allowed discovery tool. We spy on ``execute_tool`` to prove it is
    never reached for ``set_pipeline``.
    """
    responses = [_response(tool_calls=[_tool_call("c1", "set_pipeline", {"source": {}})])]

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    with (
        patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake),
        # ``execute_tool`` is dispatched from the shared ``_discovery`` helper
        # (``_execute_discovery_call``) after the discovery-loop primitives were
        # extracted there; spy at that seam, not at ``chat_solver``.
        patch("elspeth.web.composer.guided._discovery.execute_tool", autospec=True) as execute_tool_spy,
    ):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="just build the whole thing",
            current_sink=None,
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
        )

    assert result.sink is None
    assert result.assistant_message is None
    execute_tool_spy.assert_not_called()


@pytest.mark.asyncio
async def test_sink_loop_threads_parallel_tool_calls() -> None:
    """Two discovery calls in one turn each get a matching role=tool result.

    A single assistant turn may carry multiple tool_calls; every id must be
    answered before the next round or the provider 400s. Round 1 emits
    list_sinks + get_plugin_schema; round 2 resolves.
    """
    responses = [
        _response(
            tool_calls=[
                _tool_call("c1", "list_sinks", {}),
                _tool_call("c2", "get_plugin_schema", {"plugin_type": "sink", "name": "json"}),
            ]
        ),
        _response(tool_calls=[_tool_call("c3", "resolve_sink", _RESOLVE_SINK_ARGS)]),
    ]
    recorder = BufferingRecorder()
    captured_messages: list[list[dict[str, Any]]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured_messages.append(kwargs["messages"])
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save as jsonl",
            current_sink=None,
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
        )

    assert result.sink is not None
    # The second round must answer BOTH tool_call ids.
    second = captured_messages[1]
    tool_ids = {m.get("tool_call_id") for m in second if m.get("role") == "tool"}
    assert tool_ids == {"c1", "c2"}
    # Exactly one assistant message carrying both tool_calls precedes them.
    assistant_msgs = [m for m in second if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_msgs) == 1
    assert {tc["id"] for tc in assistant_msgs[0]["tool_calls"]} == {"c1", "c2"}
    # Both discovery calls were audited.
    assert {inv.tool_name for inv in recorder.invocations} == {"list_sinks", "get_plugin_schema"}


@pytest.mark.asyncio
async def test_sink_loop_returns_none_at_iteration_cap() -> None:
    """A model that never resolves is bounded by the iteration cap, not hung."""
    recorder = BufferingRecorder()

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        # Always ask for list_sinks again; never resolve.
        return _response(tool_calls=[_tool_call("loop", "list_sinks", {})])

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="keep looking",
            current_sink=None,
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=3,
        )

    assert result.sink is None
    assert result.assistant_message is None
    assert len(recorder.llm_calls) == 3


@pytest.mark.asyncio
async def test_sink_loop_malformed_discovery_args_classify_malformed_response() -> None:
    """A malformed discovery call classifies MALFORMED_RESPONSE, not API_ERROR.

    An *allowed* discovery tool whose ``arguments`` decode to a non-object makes
    the production ``_execute_discovery_call`` raise ``ChainSolverResponseShapeError``.
    The loop must list that class in its typed shape-failure except (mirroring
    ``solve_chain``'s ``chain_solver.py`` clause) so the audit row records
    MALFORMED_RESPONSE — not fall through to the API_ERROR catch-all. The class
    still re-raises; the wrapper turns it into the advisory fallback.
    """
    from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
    from elspeth.web.composer.guided.errors import ChainSolverResponseShapeError

    recorder = BufferingRecorder()
    # ``list_sinks`` is allowed (passes the dispatch gate), but its arguments
    # decode to a non-object, so dispatch raises ChainSolverResponseShapeError.
    malformed = _response(tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="list_sinks", arguments="[1, 2, 3]"))])

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return malformed

    with (
        patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake),
        pytest.raises(ChainSolverResponseShapeError),
    ):
        await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save as jsonl",
            current_sink=None,
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
        )

    assert recorder.llm_calls[-1].status == ComposerLLMCallStatus.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_sink_loop_progress_events_advance_through_discovery_and_resolve() -> None:
    """Progress phases published to a real ``ComposerProgressSink`` across a
    list_sinks -> resolve_sink round trip, exercising the actual production
    seam (elspeth-a8eeebb3aa): calling_model precedes EVERY provider round
    (including the round after the discovery tool result is threaded back),
    and using_tools precedes the discovery dispatch. The frontend folds both
    calling_model and using_tools into the same substep (ChatPanel.tsx
    tutorialStep2ActiveSubstep) precisely because this loop re-emits
    calling_model a second time — that mapping is exercised in the frontend
    test suite; this test pins the backend event sequence it must handle.
    """
    responses = [
        _response(tool_calls=[_tool_call("c1", "list_sinks", {})]),
        _response(tool_calls=[_tool_call("c2", "resolve_sink", _RESOLVE_SINK_ARGS)]),
    ]
    phases: list[str] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    async def _capture_progress(event: Any) -> None:
        phases.append(event.phase)

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save the results as a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            progress=_capture_progress,
        )

    assert result.sink is not None
    assert phases == ["calling_model", "using_tools", "calling_model"]


@pytest.mark.asyncio
async def test_sink_loop_progress_single_shot_resolve_emits_calling_model_only() -> None:
    """A single-round resolve (no discovery tool call at all) still emits
    calling_model — the common case for the tutorial's one-line "Save the
    pipeline's results to a JSON file." prompt, where the composer model
    resolves ``json`` directly without needing list_sinks/get_plugin_schema.
    Without this event the substep indicator would never leave substep 0 for
    the tutorial's actual traffic shape, even with the loop instrumented.
    """
    responses = [_response(tool_calls=[_tool_call("c1", "resolve_sink", _RESOLVE_SINK_ARGS)])]
    phases: list[str] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    async def _capture_progress(event: Any) -> None:
        phases.append(event.phase)

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save the results as a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
            progress=_capture_progress,
        )

    assert result.sink is not None
    assert phases == ["calling_model"]


@pytest.mark.asyncio
async def test_sink_loop_single_shot_when_no_catalog() -> None:
    """Without state+catalog the loop degrades to the pre-loop single-shot path."""
    responses = [_response(tool_calls=[_tool_call("c1", "resolve_sink", _RESOLVE_SINK_ARGS)])]
    captured: list[list[dict[str, Any]]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs.get("tools", []))
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chat_solver._litellm_acompletion", side_effect=_fake):
        result = await maybe_resolve_step_2_sink_chat(
            model="m",
            user_message="save as jsonl",
            current_sink=None,
            temperature=None,
            seed=None,
        )

    assert result.sink is not None
    # Only the resolve_sink tool is offered — no discovery tools.
    offered_names = {t["function"]["name"] for t in captured[0]}
    assert offered_names == {"resolve_sink"}
