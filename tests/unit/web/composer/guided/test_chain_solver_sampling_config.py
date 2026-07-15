"""Guided chain solver uses caller-threaded composer sampling."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.guided import chain_solver
from elspeth.web.composer.guided.prompts import build_step_3_context_block, load_guided_skill
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot


def _source() -> SourceResolved:
    return SourceResolved(plugin="csv", options={}, observed_columns=("name",), sample_rows=({"name": "alice"},))


def _sink() -> SinkResolved:
    return SinkResolved(outputs=(SinkOutputResolved(plugin="csv", options={}, required_fields=("name",), schema_mode="fixed"),))


def _proposal_response() -> Any:
    function = type(
        "Function",
        (),
        {
            "name": "emit_turn",
            "arguments": json.dumps({"turn_type": "propose_chain", "payload": {"steps": [], "why": "x"}}),
        },
    )()
    tool_call = type("ToolCall", (), {"function": function})()
    message = type("Message", (), {"tool_calls": [tool_call], "content": None})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


@pytest.mark.asyncio
async def test_solve_chain_omits_sampling_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _proposal_response()

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    await chain_solver.solve_chain(model="gpt-5", source=_source(), sink=_sink(), temperature=None, seed=None)

    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_solve_chain_sends_configured_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _proposal_response()

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    await chain_solver.solve_chain(model="gpt-4o", source=_source(), sink=_sink(), temperature=0.0, seed=42)

    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_solve_chain_redacts_sample_rows_in_outbound_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _proposal_response()

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    source = SourceResolved(
        plugin="csv",
        options={},
        observed_columns=("email", "api_key", "profile_url"),
        sample_rows=(
            {
                "email": "person@example.test",
                "api_key": "sk-test-secret-row-value",
                "profile_url": "https://example.test/private?token=secret",
            },
        ),
    )

    await chain_solver.solve_chain(model="gpt-5", source=source, sink=_sink(), temperature=None, seed=None)

    # messages[0] is now the pure, stable skill head (the markable cache prefix);
    # the server-resolved GUIDED CONTEXT block — including the redacted sample
    # summaries — moved to messages[1]. Relocation, not a regression: redaction
    # still happens, just in the dynamic context message.
    context_content = captured["messages"][1]["content"]
    assert "person@example.test" not in context_content
    assert "sk-test-secret-row-value" not in context_content
    assert "https://example.test/private" not in context_content
    assert "<sample:email-like>" in context_content
    assert "<sample:secret-like>" in context_content
    assert "<sample:url>" in context_content


def _empty_state_for_discovery() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _mock_catalog_for_discovery() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_transforms.return_value = [
        PluginSummary(name="web_scrape", description="Fetch URL rows", plugin_type="transform", config_fields=[]),
    ]
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="web_scrape",
        plugin_type="transform",
        description="Fetch URL rows",
        json_schema={"title": "Config", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _emit_turn_call(call_id: str) -> SimpleNamespace:
    args = {"turn_type": "propose_chain", "payload": {"steps": [], "why": "x"}}
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name="emit_turn", arguments=json.dumps(args)))


def _discovery_call(call_id: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps({})))


def _ns_response(*, tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))])


@pytest.mark.asyncio
async def test_solve_chain_splits_skill_head_and_marks_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused system message is SPLIT: messages[0] is the verbatim, stable skill
    head (the markable cache prefix); the GUIDED CONTEXT block rides in messages[1];
    the intent follows as a user message. The skill head carries cache_control."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _proposal_response()

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    source = _source()
    sink = _sink()
    await chain_solver.solve_chain(
        model="openrouter/anthropic/claude-sonnet-4-6",
        source=source,
        sink=sink,
        intent="scrape these pages",
        temperature=None,
        seed=None,
    )

    msgs = captured["messages"]
    # Byte-stable head: messages[0] is EXACTLY load_guided_skill() so a future
    # dynamic insertion into the stable prefix fails this guard loudly.
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == load_guided_skill()
    assert msgs[0]["cache_control"] == {"type": "ephemeral"}
    # Dynamic context rides in messages[1], unmarked.
    assert msgs[1]["role"] == "system"
    assert msgs[1]["content"] == build_step_3_context_block(source=source, sink=sink)
    assert "cache_control" not in msgs[1]
    # Intent follows as a user message.
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "scrape these pages"
    assert "cache_control" not in msgs[2]


@pytest.mark.asyncio
async def test_solve_chain_no_marker_for_non_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Anthropic route is split the same way but carries no marker."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _proposal_response()

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    await chain_solver.solve_chain(
        model="openrouter/openai/gpt-5.5",
        source=_source(),
        sink=_sink(),
        temperature=None,
        seed=None,
    )

    msgs = captured["messages"]
    assert msgs[0]["content"] == load_guided_skill()
    assert "cache_control" not in msgs[0]
    assert "cache_control" not in msgs[1]


@pytest.mark.asyncio
async def test_solve_chain_marks_each_discovery_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Across >=2 discovery rounds, each captured wire payload's messages[0] carries
    the marker and stays the verbatim skill head — re-marking per round is idempotent
    and never mutates the growing messages list's original system dict."""
    responses = [
        _ns_response(tool_calls=[_discovery_call("c1", "list_transforms")]),
        _ns_response(tool_calls=[_emit_turn_call("c2")]),
    ]
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.append(dict(kwargs))
        return responses.pop(0)

    monkeypatch.setattr(chain_solver, "_litellm_acompletion", fake_acompletion)

    full_catalog = _mock_catalog_for_discovery()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(full_catalog)
    catalog = PolicyCatalogView.for_trained_operator(full_catalog, snapshot)
    await chain_solver.solve_chain(
        model="openrouter/anthropic/claude-sonnet-4-6",
        source=_source(),
        sink=_sink(),
        temperature=None,
        seed=None,
        state=_empty_state_for_discovery(),
        catalog=catalog,
        plugin_snapshot=snapshot,
        user_id="u1",
        max_discovery_iters=6,
    )

    assert len(captured) >= 2, "expected at least two discovery rounds"
    for wire in captured:
        assert wire["messages"][0]["content"] == load_guided_skill()
        assert wire["messages"][0]["cache_control"] == {"type": "ephemeral"}
    # Each round's marked head is a FRESH dict (non-mutation of the shared list).
    assert captured[0]["messages"][0] is not captured[1]["messages"][0]
    # Round 1's context block is unmarked.
    assert "cache_control" not in captured[0]["messages"][1]
