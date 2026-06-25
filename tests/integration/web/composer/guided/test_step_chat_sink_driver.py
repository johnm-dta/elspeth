"""p1 Task 2 — sink driver resolves free text into a SinkResolved."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_2_sink_chat
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved
from elspeth.web.sessions._guided_step_chat import (
    Step2SinkChatResult,
    resolve_step_2_sink_chat_with_auto_drop,
)


def _fake_resolve_sink_response(args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_sink",
                                arguments=json.dumps(args),
                            )
                        )
                    ],
                )
            )
        ]
    )


_JSON_SINK_ARGS = {
    "resolution": "sink",
    "outputs": [
        {
            "plugin": "json",
            "options": {"path": "out.jsonl", "schema": {"mode": "observed"}},
            "required_fields": [],
            "schema_mode": "observed",
        }
    ],
    "assistant_message": "I set the output to a JSON Lines file.",
}


@pytest.mark.asyncio
async def test_sink_driver_resolves_json_output() -> None:
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=_fake_resolve_sink_response(_JSON_SINK_ARGS)),
    ):
        result = await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="write the results to a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert result is not None
    sink, assistant_message = result
    assert isinstance(sink, SinkResolved)
    assert len(sink.outputs) == 1
    assert sink.outputs[0].plugin == "json"
    assert sink.outputs[0].options["path"] == "out.jsonl"
    assert assistant_message == "I set the output to a JSON Lines file."


@pytest.mark.asyncio
async def test_sink_driver_returns_none_on_prose() -> None:
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A sink writes rows out.", tool_calls=None))])
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=prose),
    ):
        result = await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert result is None


@pytest.mark.asyncio
async def test_sink_driver_revise_threads_current_sink() -> None:
    current = SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": "old.jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_resolve_sink_response(_JSON_SINK_ARGS)

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(side_effect=_capture),
    ):
        await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="rename the file to out.jsonl",
            current_sink=current,
            temperature=None,
            seed=None,
        )
    assert "old.jsonl" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_sink_wrapper_absorbs_transient_into_synthetic_unavailable() -> None:
    with patch(
        "elspeth.web.sessions._guided_step_chat.maybe_resolve_step_2_sink_chat",
        new=AsyncMock(side_effect=TimeoutError("provider timeout")),
    ):
        result = await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="write a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert isinstance(result, Step2SinkChatResult)
    assert result.sink_resolution is None
    assert result.fallback_chat is not None
    assert result.fallback_chat.error_class == "TimeoutError"
    assert result.fallback_chat.status == ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE
