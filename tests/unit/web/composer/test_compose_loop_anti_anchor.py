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
        knob_schema={"fields": []},
    )
    return catalog


def _make_settings() -> WebSettings:
    return WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
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
            "elspeth.web.composer.tool_batch.execute_tool",
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
async def test_three_distinct_arg_error_failures_inject_drift_hint_before_fourth_turn() -> None:
    """Same-tool failed payload drift must also reach the model before surrender."""
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()

    turns = [
        _make_response_with_tool("call_1", "set_metadata", {"patch": {"name": "Draft A"}}),
        _make_response_with_tool("call_2", "set_metadata", {"patch": {"name": "Draft B"}}),
        _make_response_with_tool("call_3", "set_metadata", {"patch": {"name": "Draft C"}}),
        _make_text_only_response("I am stuck."),
    ]

    arg_error = ToolArgumentError(argument="patch", expected="valid metadata patch", actual_type="dict")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=[arg_error, arg_error, arg_error],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    assert mock_llm.call_count == 4
    fourth_call_messages = mock_llm.call_args_list[3].args[0]
    hint_messages = [
        m
        for m in fourth_call_messages
        if isinstance(m, dict) and m.get("role") == "user" and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))
    ]
    assert len(hint_messages) == 1
    hint_text = hint_messages[0]["content"]
    assert "set_metadata" in hint_text
    assert "drift" in hint_text.lower()
    assert "different arguments" in hint_text


@pytest.mark.asyncio
async def test_discovery_success_between_mutation_failures_does_not_break_anchor() -> None:
    """Regression test for the smoke-session bug discovered 2026-05-06.

    The Tier 1 RED's failure pattern interleaves discovery tool calls
    (`get_plugin_schema`, `get_pipeline_state`) between mutation retries.
    Discovery successes are observations, not progress — they must NOT
    clear the anti-anchor tracker. Without this guard, the threshold of 3
    consecutive identical mutation failures is never reached because each
    discovery call between retries clears the deque.

    This test drives:
      turn 1: set_pipeline #1   (fails)
      turn 2: get_plugin_schema (succeeds — discovery, must not clear)
      turn 3: set_pipeline #2   (fails, identical args)
      turn 4: get_plugin_schema (succeeds — discovery, must not clear)
      turn 5: set_pipeline #3   (fails, identical args)
      turn 6: text-only         (compose returns)

    Asserts that turn 6's LLM call sees the [ELSPETH-SYSTEM-HINT].
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()

    identical_args = {"patch": {"name": "Anchored Across Discoveries"}}
    discovery_args = {"name": "csv", "plugin_type": "source"}
    turns = [
        _make_response_with_tool("c1", "set_metadata", identical_args),
        _make_response_with_tool("d1", "get_plugin_schema", discovery_args),
        _make_response_with_tool("c2", "set_metadata", identical_args),
        _make_response_with_tool("d2", "get_plugin_schema", discovery_args),
        _make_response_with_tool("c3", "set_metadata", identical_args),
        _make_text_only_response("Either the hint will help me or I give up."),
    ]

    arg_error = ToolArgumentError(argument="patch", expected="x", actual_type="dict")

    # Mutations raise ToolArgumentError; discovery calls return a successful
    # ToolResult with no state change (mock the cache miss path so the
    # discovery success goes through the full dispatch instead of the cache).
    from dataclasses import replace as dc_replace

    from elspeth.web.composer.state import ValidationSummary
    from elspeth.web.composer.tools import ToolResult

    discovery_success = ToolResult(
        success=True,
        updated_state=dc_replace(state),
        validation=ValidationSummary(
            is_valid=False,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
    )

    # `get_plugin_schema` is cacheable — turn d2's identical args hit the
    # discovery cache and bypass execute_tool entirely. So execute_tool is
    # called only 4 times (c1, d1, c2, c3); d2 is served from cache. The
    # bug-under-test is whether the cache-hit-discovery still counts as a
    # mutation success that clears the tracker. (It must not — cache-hit
    # path was already gated above; this test verifies the dispatch path
    # gate too.)
    side_effects = [
        arg_error,  # c1 set_metadata fail
        discovery_success,  # d1 get_plugin_schema success (cache miss → dispatch)
        arg_error,  # c2 set_metadata fail
        # d2 get_plugin_schema → CACHE HIT, no execute_tool call
        arg_error,  # c3 set_metadata fail
    ]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch("elspeth.web.composer.tool_batch.execute_tool", side_effect=side_effects),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    # 6th LLM call should see the hint after the 3rd identical mutation failure.
    assert mock_llm.call_count == 6, f"expected 6 LLM calls, got {mock_llm.call_count}"
    sixth_call_messages = mock_llm.call_args_list[5].args[0]
    hint_messages = [m for m in sixth_call_messages if isinstance(m, dict) and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))]
    assert len(hint_messages) == 1, (
        f"expected 1 hint after 3 identical mutation failures interleaved with discovery successes; got {len(hint_messages)}"
    )


@pytest.mark.asyncio
async def test_mutation_success_breaks_anchor() -> None:
    """A successful mutation between failures must clear the tracker.

    Verifies the inverse of the discovery test: a *mutation* success (not a
    discovery success) does break the anchor, because it represents real
    progress. After: 2 fails -> 1 mutation-success -> 2 fails -> text. The
    hint must NOT fire (only 2 consecutive failures post-success).
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()

    args = {"patch": {"name": "Two-Then-Reset"}}
    turns = [
        _make_response_with_tool("c1", "set_metadata", args),
        _make_response_with_tool("c2", "set_metadata", args),
        _make_response_with_tool("c3", "set_metadata", args),  # this one succeeds
        _make_response_with_tool("c4", "set_metadata", args),
        _make_response_with_tool("c5", "set_metadata", args),
        _make_text_only_response("Below threshold post-reset."),
    ]

    arg_error = ToolArgumentError(argument="patch", expected="x", actual_type="dict")

    from dataclasses import replace as dc_replace

    from elspeth.web.composer.state import ValidationSummary
    from elspeth.web.composer.tools import ToolResult

    mutation_success = ToolResult(
        success=True,
        updated_state=dc_replace(state, version=state.version + 1),
        validation=ValidationSummary(
            is_valid=True,
            errors=(),
            warnings=(),
            suggestions=(),
            semantic_contracts=(),
        ),
        affected_nodes=(),
    )

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=[arg_error, arg_error, mutation_success, arg_error, arg_error],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    sixth_call_messages = mock_llm.call_args_list[-1].args[0]
    hint_messages = [m for m in sixth_call_messages if isinstance(m, dict) and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))]
    assert hint_messages == [], (
        "mutation success between failure pairs must reset the tracker — two post-success failures alone are below threshold"
    )


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
            "elspeth.web.composer.tool_batch.execute_tool",
            side_effect=[arg_error, arg_error],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("Build something", [], state)

    # Inspect the THIRD (final) LLM call — should contain no hint.
    third_call_messages = mock_llm.call_args_list[2].args[0]
    hint_messages = [m for m in third_call_messages if isinstance(m, dict) and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))]
    assert hint_messages == [], "hint must not fire below the 3-failure threshold"
