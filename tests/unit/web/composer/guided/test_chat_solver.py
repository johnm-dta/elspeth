"""Unit tests for the per-step chat solver (Phase A — advisory text only).

The solver builds a step-scoped system prompt + user message + temp/seed
kwargs, invokes ``_litellm_acompletion``, and returns the assistant message
content as a plain string.

Phase B (separate slice) introduces the per-step tool palette + Tier-3 args
validation; tests for that surface live in test_step_tool_scope.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.chat_solver import (
    Step1SourceChatResolution,
    _build_step_1_source_tool_prompt,
    _build_step_2_sink_tool_prompt,
    _parse_step_1_source_tool_arguments,
    solve_step_chat,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved


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
        on_validation_failure="discard",
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

    reply = await solve_step_chat(model="test/model", step=step, user_message="hi", temperature=None, seed=None)

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
    with pytest.raises(InvariantError, match="user_message is empty"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="",
            temperature=None,
            seed=None,
        )


@pytest.mark.asyncio
async def test_missing_response_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns None for message.content is defective; we crash loudly."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=None))])

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="hello",
            temperature=None,
            seed=None,
        )


@pytest.mark.asyncio
async def test_whitespace_only_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns only whitespace is also defective — same crash path."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _ok_response("   \n  \t  \n")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_2_SINK,
            user_message="hello",
            temperature=None,
            seed=None,
        )


def _source_tool_args(**overrides: Any) -> str:
    """A valid resolve_source argument blob (json-encoded), overridable per test."""
    args: dict[str, Any] = {
        "resolution": "source",
        "plugin": "json",
        "filename": "rows.json",
        "mime_type": "application/json",
        "content": '[{"url": "https://example.test/a"}]',
        "options": {"schema": {"mode": "observed"}},
        "observed_columns": ["url"],
        "sample_rows": [{"url": "https://example.test/a"}],
        "assistant_message": "Created the source.",
    }
    args.update(overrides)
    return json.dumps(args)


def test_parse_defaults_on_validation_failure_to_discard_when_omitted() -> None:
    """The optional knob is absent -> default to 'discard' so a passive walk never stalls."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(), plugin_hint="json")
    assert resolution.on_validation_failure == "discard"


def test_parse_preserves_explicit_on_validation_failure() -> None:
    """A composer-chosen value (non-default sentinel) survives the parse verbatim."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure="quarantine_sink"), plugin_hint="json")
    assert resolution.on_validation_failure == "quarantine_sink"


def test_parse_empty_on_validation_failure_defaults_to_discard() -> None:
    """An empty string is treated as 'not set' and defaults to 'discard'."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure=""), plugin_hint="json")
    assert resolution.on_validation_failure == "discard"


def test_parse_non_string_on_validation_failure_raises() -> None:
    """When the model sends a non-string value, reject at the Tier-3 boundary."""
    with pytest.raises(ValueError, match="on_validation_failure must be a string"):
        _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure=123), plugin_hint="json")


def test_step_1_revision_prompt_uses_llm_safe_source_context() -> None:
    current_source = SourceResolved(
        plugin="csv",
        options={
            "schema": {"mode": "observed", "guaranteed_fields": ["email", "profile_url"]},
            "blob_ref": {"id": "blob-private-source-id", "storage_path": "/srv/elspeth/blobs/private.csv"},
            "raw_option_key_should_not_leave": "sk-option-secret",
        },
        observed_columns=("email", "profile_url", "note"),
        sample_rows=(
            {
                "email": "person@example.test",
                "profile_url": "https://example.test/private?token=secret",
                "note": "customer asked for refunds",
            },
        ),
        on_validation_failure="quarantine",
    )

    prompt = _build_step_1_source_tool_prompt(plugin_hint="csv", current_source=current_source)

    assert "person@example.test" not in prompt
    assert "https://example.test/private" not in prompt
    assert "customer asked for refunds" not in prompt
    assert "blob-private-source-id" not in prompt
    assert "/srv/elspeth/blobs/private.csv" not in prompt
    assert "raw_option_key_should_not_leave" not in prompt
    assert "sk-option-secret" not in prompt
    assert '"plugin": "csv"' in prompt
    assert '"mode": "observed"' in prompt
    assert '"guaranteed_fields": ["email", "profile_url"]' in prompt
    assert "<sample:email-like>" in prompt
    assert "<sample:url>" in prompt
    assert "<sample:string:" in prompt


def test_step_2_revision_prompt_uses_llm_safe_sink_context() -> None:
    current_sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="azure_blob",
                options={
                    "path": "/srv/elspeth/exports/private-output.jsonl",
                    "sas_token": "sv=private-token",
                    "raw_sink_option_key_should_not_leave": {"secret_ref": "PROD_BLOB_SECRET"},
                },
                required_fields=("email_hash", "profile_url"),
                schema_mode="fixed",
            ),
        )
    )

    prompt = _build_step_2_sink_tool_prompt(current_sink=current_sink)

    assert "/srv/elspeth/exports/private-output.jsonl" not in prompt
    assert "sv=private-token" not in prompt
    assert "raw_sink_option_key_should_not_leave" not in prompt
    assert "PROD_BLOB_SECRET" not in prompt
    assert '"plugin": "azure_blob"' in prompt
    assert '"schema_mode": "fixed"' in prompt
    assert '"required_fields": ["email_hash", "profile_url"]' in prompt
    assert '"option_count": 3' in prompt
