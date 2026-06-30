"""Guided chat solver uses caller-threaded composer sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.protocol import GuidedStep


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


def _text_response(text: str = "reply") -> _FakeResponse:
    return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=text))])


@pytest.mark.asyncio
async def test_solve_step_chat_omits_sampling_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("reply")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    out = await chat_solver.solve_step_chat(
        model="gpt-5",
        step=GuidedStep.STEP_1_SOURCE,
        user_message="hi",
        temperature=None,
        seed=None,
    )

    assert out == "reply"
    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_step_1_source_resolution_sends_configured_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("advice")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    result = await chat_solver.maybe_resolve_step_1_source_chat(
        model="gpt-4o",
        user_message="how should I configure csv?",
        plugin_hint="csv",
        temperature=0.0,
        seed=42,
    )

    assert result is None
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_solve_step_chat_marks_system_message_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Anthropic-family route gets ``cache_control`` on the stable skill head."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("reply")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    await chat_solver.solve_step_chat(
        model="openrouter/anthropic/claude-sonnet-4-6",
        step=GuidedStep.STEP_1_SOURCE,
        user_message="hi",
        temperature=None,
        seed=None,
    )

    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["cache_control"] == {"type": "ephemeral"}
    # The user message must NOT be marked.
    assert "cache_control" not in captured["messages"][1]


@pytest.mark.asyncio
async def test_solve_step_chat_no_marker_for_non_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Anthropic route carries no ``cache_control`` marker."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("reply")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    await chat_solver.solve_step_chat(
        model="openrouter/openai/gpt-5.5",
        step=GuidedStep.STEP_1_SOURCE,
        user_message="hi",
        temperature=None,
        seed=None,
    )

    assert "cache_control" not in captured["messages"][0]
    assert "cache_control" not in captured["messages"][1]
