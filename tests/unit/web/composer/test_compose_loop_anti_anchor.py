"""Integration test for §7.7 anti-anchor hint injection through the real
``ComposerServiceImpl._compose_loop``.

Unit tests in ``test_anti_anchor.py`` exercise the trigger logic in isolation;
this test exercises the wiring — that the loop calls ``record_failure`` at
each failure site, ``record_success`` on each success path, and injects the
hint into ``llm_messages`` BEFORE the next LLM call (not after, where the LLM
would never see it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.config import WebSettings


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(
            name="csv",
            description="CSV source",
            plugin_type="source",
            config_fields=[],
        ),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"title": "Config", "properties": {}},
    )
    return catalog


def _make_settings() -> WebSettings:
    return WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
    )


def _make_response_with_tool(tool_id: str, tool_name: str, args: dict[str, Any]) -> _FakeLLMResponse:
    return _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id=tool_id,
                            function=_FakeFunction(name=tool_name, arguments=json.dumps(args)),
                        )
                    ],
                )
            )
        ]
    )


def _make_text_only_response(content: str) -> _FakeLLMResponse:
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=None))])


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.mark.asyncio
async def test_three_identical_arg_error_failures_inject_hint_before_fourth_turn() -> None:
    """The hint must reach the model BEFORE the next LLM call, not after.

    Drive 3 turns where the LLM issues the same set_metadata call with the
    same arguments and execute_tool raises ToolArgumentError each time. On
    turn 4 the LLM produces a text-only response so compose() returns. Then
    inspect the messages passed to _call_llm on turn 4: the hint must appear
    after the third tool result.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()

    identical_args = {"patch": {"name": "Anchored Build"}}

    def turn_with_failure(call_id: str) -> _FakeLLMResponse:
        return _make_response_with_tool(call_id, "set_metadata", identical_args)

    turns = [
        turn_with_failure("call_1"),
        turn_with_failure("call_2"),
        turn_with_failure("call_3"),
        _make_text_only_response("I give up."),
    ]

    arg_error = ToolArgumentError(argument="patch", expected="non-anchored payload", actual_type="dict")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service.execute_tool",
            side_effect=[arg_error, arg_error, arg_error],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    # The fourth LLM call is the post-hint LLM call. Inspect its messages
    # argument and find the system-injected hint.
    assert mock_llm.call_count == 4, f"expected 4 LLM calls (3 mutating + final), got {mock_llm.call_count}"
    fourth_call_messages = mock_llm.call_args_list[3].args[0]

    hint_messages = [
        m
        for m in fourth_call_messages
        if isinstance(m, dict) and m.get("role") == "user" and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))
    ]
    assert len(hint_messages) == 1, (
        f"expected exactly one [ELSPETH-SYSTEM-HINT] in fourth-turn messages; got {len(hint_messages)} "
        f"(messages: {[(m.get('role'), str(m.get('content', ''))[:60]) for m in fourth_call_messages]})"
    )
    hint_text = hint_messages[0]["content"]
    assert "set_metadata" in hint_text, "hint should name the anchored tool"
    assert "byte-identical" in hint_text or "identical" in hint_text


@pytest.mark.asyncio
async def test_two_identical_failures_do_not_inject_hint() -> None:
    """Below threshold (N=3) the hint must not fire."""
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()

    args = {"patch": {"name": "Two Strikes"}}
    turns = [
        _make_response_with_tool("c1", "set_metadata", args),
        _make_response_with_tool("c2", "set_metadata", args),
        _make_text_only_response("not stuck yet"),
    ]
    arg_error = ToolArgumentError(argument="patch", expected="x", actual_type="dict")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service.execute_tool",
            side_effect=[arg_error, arg_error],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    # Inspect the THIRD (final) LLM call — should contain no hint.
    third_call_messages = mock_llm.call_args_list[2].args[0]
    hint_messages = [m for m in third_call_messages if isinstance(m, dict) and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))]
    assert hint_messages == [], "hint must not fire below the 3-failure threshold"
