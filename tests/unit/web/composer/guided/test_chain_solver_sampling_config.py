"""Guided chain solver uses caller-threaded composer sampling."""

from __future__ import annotations

import json
from typing import Any

import pytest

from elspeth.web.composer.guided import chain_solver
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved


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
