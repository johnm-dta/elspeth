"""p1 Task 1 — source driver accepts current applied source for in-place revise."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
from elspeth.web.composer.guided.resolved import SourceResolved


def _fake_resolve_source_response(args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_source",
                                arguments=json.dumps(args),
                            )
                        )
                    ],
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_source_driver_includes_current_source_in_prompt() -> None:
    current = SourceResolved(
        plugin="json",
        options={"schema": {"mode": "observed"}, "blob_ref": "abc"},
        observed_columns=("url",),
        sample_rows=({"url": "https://example.test/a"},),
    )
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_resolve_source_response(
            {
                "resolution": "source",
                "plugin": "json",
                "filename": "urls.json",
                "mime_type": "application/json",
                "content": '[{"url": "https://example.test/a"}]',
                "options": {"schema": {"mode": "observed"}},
                "observed_columns": ["url"],
                "sample_rows": [{"url": "https://example.test/a"}],
                "assistant_message": "Updated the URL list.",
            }
        )

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(side_effect=_capture),
    ):
        result = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="add a second url",
            plugin_hint="json",
            current_source=current,
            temperature=None,
            seed=None,
        )

    assert result is not None
    assert result.plugin == "json"
    system_prompt = captured["messages"][0]["content"]
    # The current applied source MUST be threaded so "add" resolves relative to it.
    assert "https://example.test/a" in system_prompt


@pytest.mark.asyncio
async def test_source_driver_returns_none_on_prose() -> None:
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Here is some advice.", tool_calls=None))])
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=prose),
    ):
        result = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a source?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert result is None
