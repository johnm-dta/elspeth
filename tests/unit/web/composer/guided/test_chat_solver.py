"""Unit tests for the per-step chat solver (Phase A — advisory text only).

The solver builds a step-scoped system prompt + user message + temp/seed
kwargs, invokes ``_litellm_acompletion``, and returns the assistant message
content as a plain string.

Phase B (separate slice) introduces the per-step tool palette + Tier-3 args
validation; tests for that surface live in test_step_tool_scope.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.chat_solver import Step1SourceChatResolution, solve_step_chat
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import GuidedStep


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


def _ok_response(text: str) -> _FakeLLMResponse:
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=text))])


def test_step_1_source_chat_resolution_deep_freezes_container_fields() -> None:
    resolution = Step1SourceChatResolution(
        assistant_message="Created a CSV source.",
        plugin="csv",
        filename="rows.csv",
        mime_type="text/csv",
        content="name\nalice\n",
        options={"schema": {"fields": ["name"]}},
        observed_columns=("name",),
        sample_rows=({"name": "alice"},),
    )

    with pytest.raises(TypeError):
        resolution.options["delimiter"] = ","  # type: ignore[index]
    with pytest.raises(TypeError):
        resolution.options["schema"]["fields"] = ["other"]  # type: ignore[index,call-overload]
    with pytest.raises(TypeError):
        resolution.sample_rows[0]["name"] = "bob"  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", list(GuidedStep))
async def test_solver_sends_step_scoped_system_prompt(monkeypatch: pytest.MonkeyPatch, step: GuidedStep) -> None:
    """The solver's system prompt must be the per-step skill, NOT the full skill.

    This is the entire point of Phase A — verify scoping is mechanical, not
    a comment.  We capture the kwargs the solver sends to _litellm_acompletion
    and assert the system prompt matches load_step_chat_skill(step).
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeLLMResponse:
        captured.update(kwargs)
        return _ok_response("here's some advice")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    # Disable seed lookup network/litellm call by stubbing it
    monkeypatch.setattr(chat_solver, "_composer_llm_seed_for_model", lambda _model: None)

    reply = await solve_step_chat(model="test/model", step=step, user_message="hi")

    assert reply == "here's some advice"
    messages = captured["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hi"}

    from elspeth.web.composer.guided.prompts import load_step_chat_skill

    assert messages[0]["content"] == load_step_chat_skill(step), (
        f"system prompt for {step.value} did not match load_step_chat_skill output — per-step scoping is broken"
    )


@pytest.mark.asyncio
async def test_empty_user_message_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty user message is a route-handler-validation gap; we crash loudly."""
    # Stub seed lookup to avoid litellm import
    monkeypatch.setattr(chat_solver, "_composer_llm_seed_for_model", lambda _model: None)

    with pytest.raises(InvariantError, match="user_message is empty"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="",
        )


@pytest.mark.asyncio
async def test_missing_response_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns None for message.content is defective; we crash loudly."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=None))])

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)
    monkeypatch.setattr(chat_solver, "_composer_llm_seed_for_model", lambda _model: None)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="hello",
        )


@pytest.mark.asyncio
async def test_whitespace_only_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns only whitespace is also defective — same crash path."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _ok_response("   \n  \t  \n")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)
    monkeypatch.setattr(chat_solver, "_composer_llm_seed_for_model", lambda _model: None)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_2_SINK,
            user_message="hello",
        )
