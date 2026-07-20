"""Guided chat solver uses caller-threaded composer sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.core.canonical import stable_hash
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.prompts import load_step_chat_skill
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
        timeout_seconds=30.0,
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
        current_source=None,
        available_source_plugins=("csv",),
        temperature=0.0,
        seed=42,
        timeout_seconds=30.0,
    )

    assert type(result) is chat_solver.GuidedChatProseOutcome
    assert result.assistant_message == "advice"
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
        timeout_seconds=30.0,
    )

    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["cache_control"] == {"type": "ephemeral"}
    # The no-tools addendum (messages[1]) and the user message (messages[2])
    # must NOT be marked.
    assert "cache_control" not in captured["messages"][1]
    assert "cache_control" not in captured["messages"][2]


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
        timeout_seconds=30.0,
    )

    assert "cache_control" not in captured["messages"][0]
    assert "cache_control" not in captured["messages"][1]


@pytest.mark.asyncio
async def test_step_1_source_splits_skill_head_and_marks_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The step_1 source prompt is SPLIT: messages[0] is the stable step skill
    (the markable head), messages[1] holds the dynamic hint/revise block + tool
    instructions, messages[2] is the user message. Anthropic routes mark the
    skill head AND the trailing tool; the dynamic block stays unmarked."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("advice")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    await chat_solver.maybe_resolve_step_1_source_chat(
        model="openrouter/anthropic/claude-sonnet-4-6",
        user_message="make a csv of urls",
        plugin_hint="csv",
        current_source=None,
        available_source_plugins=("csv",),
        temperature=None,
        seed=None,
        timeout_seconds=30.0,
    )

    msgs = captured["messages"]
    # Stable, markable skill head — byte-stable so a future dynamic insertion
    # into the cache prefix fails this guard loudly.
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()
    assert msgs[0]["cache_control"] == {"type": "ephemeral"}
    # Dynamic block: hint + tool instructions, unmarked.
    assert msgs[1]["role"] == "system"
    assert "The current source plugin selected in the wizard is 'csv'." in msgs[1]["content"]
    assert "call `resolve_source`" in msgs[1]["content"]
    assert "cache_control" not in msgs[1]
    # User message follows, unmarked.
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "make a csv of urls"
    assert "cache_control" not in msgs[2]
    # Trailing tool carries the tools-array marker.
    assert captured["tools"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_step_1_source_no_marker_for_non_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Anthropic route is split the same way but carries no marker."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("advice")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    await chat_solver.maybe_resolve_step_1_source_chat(
        model="openrouter/openai/gpt-5.5",
        user_message="make a csv of urls",
        plugin_hint="csv",
        current_source=None,
        available_source_plugins=("csv",),
        temperature=None,
        seed=None,
        timeout_seconds=30.0,
    )

    msgs = captured["messages"]
    assert msgs[0]["content"] == load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()
    assert "cache_control" not in msgs[0]
    assert "cache_control" not in msgs[1]
    for tool in captured["tools"]:
        assert "cache_control" not in tool


@pytest.mark.asyncio
async def test_solve_step_chat_audit_hash_matches_marked_wire_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit truthfulness: the recorded ``messages_hash`` is the stable hash of
    the EXACT marked list sent to the wire (the same-object invariant). If the
    marker were ever applied to a copy that diverged from the audited list, this
    fails."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _text_response("reply")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    recorder = BufferingRecorder()
    await chat_solver.solve_step_chat(
        model="openrouter/anthropic/claude-sonnet-4-6",
        step=GuidedStep.STEP_1_SOURCE,
        user_message="hi",
        temperature=None,
        seed=None,
        timeout_seconds=30.0,
        recorder=recorder,
    )

    # The wire payload IS marked (precondition for a meaningful hash check).
    assert captured["messages"][0]["cache_control"] == {"type": "ephemeral"}
    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].messages_hash == stable_hash(captured["messages"])
