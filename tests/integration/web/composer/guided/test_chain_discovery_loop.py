"""Transform-chain stage discovery-tool loop (the sink loop's mirror).

``solve_chain`` was single-shot with a FORCED ``tool_choice: emit_turn`` and no
discovery tools, so the composer "light" model proposed transform chains blind —
guessing plugin names and option shapes — and the accept seam
(``_execute_set_pipeline``) rejected them (e.g. ``web_scrape`` requires
``url_field`` top-level + ``http.{abuse_contact,scraping_reason}``). These tests
pin the agentic discovery loop: with ``state`` + ``catalog`` supplied the model
calls read-only discovery tools (``list_transforms`` / ``get_plugin_schema`` /
``list_models``) under ``tool_choice: required``, the solver dispatches them via
the production ``execute_tool`` and threads results back, and the model then
emits ``propose_chain``. Without ``state``/``catalog`` the solver degrades to the
prior byte-identical single-shot (forced ``emit_turn``) path.

Exercised against the REAL ``execute_tool`` + a ``spec``-locked catalog mock, so
these cover dispatch integration and the execution-side safety gate, not just
message threading.
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
from elspeth.web.composer.guided.chain_solver import ChainSolverResponseShapeError, solve_chain
from elspeth.web.composer.guided.state_machine import SinkOutputResolved, SinkResolved, SourceResolved
from elspeth.web.composer.state import CompositionState, PipelineMetadata


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


class _TransformCatalog:
    def list_transforms(self) -> list[PluginSummary]:
        return [
            PluginSummary(name="web_scrape", description="Fetch URL rows", plugin_type="transform", config_fields=[]),
            PluginSummary(name="llm", description="LLM transform", plugin_type="transform", config_fields=[]),
        ]

    def get_schema(self, plugin_type: str, name: str) -> PluginSchemaInfo:
        assert (plugin_type, name) == ("transform", "web_scrape")
        return PluginSchemaInfo(
            name="web_scrape",
            plugin_type="transform",
            description="Fetch URL rows",
            json_schema={"title": "Config", "properties": {}},
            knob_schema={"fields": []},
        )


def _catalog() -> CatalogService:
    return _TransformCatalog()


def _source() -> SourceResolved:
    return SourceResolved(
        plugin="json",
        options={},
        observed_columns=("url",),
        sample_rows=({"url": "https://example/a"},),
    )


def _sink() -> SinkResolved:
    return SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "outputs/ratings.json"},
                required_fields=("rating",),
                schema_mode="observed",
            ),
        )
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def _response(*, content: str | None = None, tool_calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


_PROPOSE_CHAIN_ARGS = {
    "turn_type": "propose_chain",
    "payload": {
        "steps": [
            {
                "plugin": "web_scrape",
                "options": {
                    "url_field": "url",
                    "content_field": "page_text",
                    "fingerprint_field": "page_fp",
                    "schema": {"mode": "observed"},
                    "http": {"abuse_contact": "noreply@demo.com", "scraping_reason": "tutorial demo"},
                },
                "rationale": "fetch each URL row into page_text",
            }
        ],
        "why": "fetch the pages so the next step can rate them",
    },
}


@pytest.mark.asyncio
async def test_chain_loop_lists_transforms_then_proposes() -> None:
    """list_transforms -> (tool result threaded back) -> emit_turn(propose_chain)."""
    responses = [
        _response(tool_calls=[_tool_call("c1", "list_transforms", {})]),
        _response(tool_calls=[_tool_call("c2", "emit_turn", _PROPOSE_CHAIN_ARGS)]),
    ]
    recorder = BufferingRecorder()
    captured: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake):
        proposal = await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )

    assert proposal.steps[0]["plugin"] == "web_scrape"
    assert proposal.why == "fetch the pages so the next step can rate them"

    # Discovery path uses tool_choice "required" (NOT forced emit_turn), so the
    # model is free to call discovery tools before it commits a chain.
    assert captured[0]["tool_choice"] == "required"
    # The second round must carry the list_transforms tool result keyed to c1,
    # preceded by the assistant tool-call request.
    second = captured[1]["messages"]
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in second)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)

    assert len(recorder.llm_calls) == 2
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].tool_name == "list_transforms"


@pytest.mark.asyncio
async def test_chain_loop_refuses_to_dispatch_mutation_tool() -> None:
    """Execution-side safety gate: a hallucinated mutation call is NOT dispatched.

    ``execute_tool``'s handler union includes every mutation registry, so the
    solver must refuse to dispatch any name that is neither ``emit_turn`` nor an
    allowed discovery tool. We spy on ``execute_tool`` to prove it is never
    called, and the loop raises ChainSolverResponseShapeError (routing to the
    auto-drop path exactly like today's no-proposal outcome).
    """
    responses = [_response(tool_calls=[_tool_call("c1", "set_pipeline", {"sources": {}})])]

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    with (
        patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake),
        patch("elspeth.web.composer.guided._discovery.execute_tool", autospec=True) as execute_tool_spy,
        pytest.raises(ChainSolverResponseShapeError),
    ):
        await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )

    execute_tool_spy.assert_not_called()


@pytest.mark.asyncio
async def test_chain_loop_malformed_discovery_args_raise_shape_error() -> None:
    """A discovery tool call whose arguments decode to a NON-object (e.g. a JSON
    list) is an LLM RESPONSE-SHAPE failure, not a client ``ValueError`` — it must
    raise ChainSolverResponseShapeError so ``solve_chain_with_auto_drop`` routes
    it to the auto-drop path. A bare ValueError/JSONDecodeError here would escape
    that wrapper (it excludes ValueError) and surface as a 500.
    """
    # ``list_transforms`` is an allowed discovery tool, so the execution-side gate
    # dispatches it — but its arguments decode to ``[]`` (a list, not an object).
    bad_call = SimpleNamespace(id="c1", function=SimpleNamespace(name="list_transforms", arguments="[]"))
    responses = [_response(tool_calls=[bad_call])]

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    with (
        patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake),
        pytest.raises(ChainSolverResponseShapeError),
    ):
        await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )


@pytest.mark.asyncio
async def test_chain_loop_threads_parallel_tool_calls() -> None:
    """Two discovery calls in one assistant message: BOTH results are threaded
    back keyed to their ids before the next round, or the provider 400s."""
    responses = [
        _response(
            tool_calls=[
                _tool_call("a1", "list_transforms", {}),
                _tool_call("a2", "get_plugin_schema", {"plugin_type": "transform", "name": "web_scrape"}),
            ]
        ),
        _response(tool_calls=[_tool_call("a3", "emit_turn", _PROPOSE_CHAIN_ARGS)]),
    ]
    recorder = BufferingRecorder()
    captured: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake):
        proposal = await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )

    assert proposal.steps[0]["plugin"] == "web_scrape"
    second = captured[1]["messages"]
    answered = {m.get("tool_call_id") for m in second if m.get("role") == "tool"}
    assert {"a1", "a2"} <= answered
    assert len(recorder.invocations) == 2


@pytest.mark.asyncio
async def test_chain_loop_raises_at_iteration_cap() -> None:
    """Discovery that never proposes a chain hits the cap and raises
    ChainSolverResponseShapeError (auto-drop), never loops forever."""

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        # Always answer with a fresh discovery call — never an emit_turn.
        return _response(tool_calls=[_tool_call("loop", "list_transforms", {})])

    with (
        patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake),
        pytest.raises(ChainSolverResponseShapeError),
    ):
        await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=2,
        )


@pytest.mark.asyncio
async def test_chain_loop_emit_turn_terminal_regardless_of_siblings() -> None:
    """An emit_turn alongside a discovery call in ONE message takes the proposal
    (terminal) and returns — it does not dispatch the sibling discovery call."""
    responses = [
        _response(
            tool_calls=[
                _tool_call("s1", "list_transforms", {}),
                _tool_call("s2", "emit_turn", _PROPOSE_CHAIN_ARGS),
            ]
        )
    ]
    recorder = BufferingRecorder()

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        return responses.pop(0)

    with patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake):
        proposal = await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
            recorder=recorder,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )

    assert proposal.steps[0]["plugin"] == "web_scrape"
    # Terminal on the first round: no discovery call dispatched.
    assert len(recorder.invocations) == 0


@pytest.mark.asyncio
async def test_chain_solve_single_shot_when_no_catalog() -> None:
    """Without state/catalog the solver degrades to the prior single-shot path:
    FORCED tool_choice emit_turn, one round, no discovery tools offered."""
    captured: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return _response(tool_calls=[_tool_call("only", "emit_turn", _PROPOSE_CHAIN_ARGS)])

    with patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake):
        proposal = await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            temperature=None,
            seed=None,
        )

    assert proposal.steps[0]["plugin"] == "web_scrape"
    assert len(captured) == 1
    # Byte-identical disabled path: forced emit_turn, no discovery tools.
    assert captured[0]["tool_choice"] == {"type": "function", "function": {"name": "emit_turn"}}
    tool_names = {t["function"]["name"] for t in captured[0]["tools"]}
    assert tool_names == {"emit_turn"}


@pytest.mark.asyncio
async def test_solve_chain_threads_intent_as_user_role_message() -> None:
    """The originating transform intent renders as a USER-role message.

    Mirrors freeform (the frozen prompt arrives as the final user message,
    ``prompts.py:401``) and the working sink resolver (``chat_solver.py:582-585``).
    Before this, ``solve_chain`` built a SYSTEM-ONLY messages list, so the model
    proposed the chain blind from the source/sink contract alone — which is why
    the tutorial built a passthrough instead of web_scrape->llm->field_mapper.
    """
    intent = "Scrape these three project-brief pages and have an LLM write a short summary of each page into one JSON row."
    captured: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        return _response(tool_calls=[_tool_call("only", "emit_turn", _PROPOSE_CHAIN_ARGS)])

    with patch("elspeth.web.composer.guided.chain_solver._litellm_acompletion", side_effect=_fake):
        await solve_chain(
            model="m",
            source=_source(),
            sink=_sink(),
            intent=intent,
            temperature=None,
            seed=None,
            state=_empty_state(),
            catalog=_catalog(),
            user_id="u1",
            max_discovery_iters=6,
        )

    assert captured, "solve_chain never invoked the LLM"
    user_messages = [m for m in captured[0]["messages"] if m.get("role") == "user"]
    assert len(user_messages) == 1, f"expected exactly one user-role message, got {user_messages}"
    assert intent in user_messages[0]["content"]
