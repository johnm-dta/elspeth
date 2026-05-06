"""Composer LLM-call audit tests for the async LiteLLM path."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerServiceError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationSummary
from elspeth.web.composer.tools import ToolResult
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationResult


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
    model: str | None = "provider/model-returned"
    usage: Any = None
    id: str | None = "chatcmpl-123"


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
        PluginSummary(name="csv", description="CSV source", plugin_type="source", config_fields=[]),
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


def _make_settings(**overrides: Any) -> WebSettings:
    defaults: dict[str, Any] = {
        "data_dir": Path("/data"),
        "composer_model": "openrouter/openai/gpt-5.5",
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
    }
    defaults.update(overrides)
    return WebSettings(**defaults)


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    model: str | None = "provider/model-returned",
    prompt_tokens: int | None = 11,
    completion_tokens: int | None = 7,
    total_tokens: int | None = None,
    request_id: str | None = "chatcmpl-123",
) -> _FakeLLMResponse:
    fake_tool_calls: list[_FakeToolCall] | None = None
    if tool_calls:
        fake_tool_calls = [
            _FakeToolCall(
                id=tc["id"],
                function=_FakeFunction(name=tc["name"], arguments=json.dumps(tc["arguments"])),
            )
            for tc in tool_calls
        ]
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
    return _FakeLLMResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=fake_tool_calls))],
        model=model,
        usage=usage,
        id=request_id,
    )


def _captured_llm_calls(exc: BaseException) -> Sequence[ComposerLLMCall]:
    calls = exc.__dict__["llm_calls"]
    assert isinstance(calls, tuple)
    return cast(tuple[ComposerLLMCall, ...], calls)


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.mark.asyncio
async def test_text_only_success_records_llm_call_metadata() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.")

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response) as mock_acomp:
        result = await service.compose("Build a CSV pipeline", [], state)

    assert result.message == "Done."
    assert len(result.llm_calls) == 1
    call = result.llm_calls[0]
    request_kwargs = mock_acomp.call_args.kwargs
    assert call.status is ComposerLLMCallStatus.SUCCESS
    assert call.model_requested == "openrouter/openai/gpt-5.5"
    assert call.model_returned == "provider/model-returned"
    assert call.prompt_tokens == 11
    assert call.completion_tokens == 7
    assert call.total_tokens == 18
    assert call.provider_request_id == "chatcmpl-123"
    assert call.messages_hash == stable_hash(request_kwargs["messages"])
    assert call.tools_spec_hash == stable_hash(request_kwargs["tools"])


@pytest.mark.asyncio
async def test_seed_omission_for_unsupported_provider_is_reflected_in_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_supported_openai_params",
        lambda model: ["temperature", "tools"],
    )
    service = ComposerServiceImpl(
        catalog=_mock_catalog(),
        settings=_make_settings(composer_model="anthropic/claude-3-5-sonnet-20241022"),
    )
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.")

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response) as mock_acomp:
        result = await service.compose("Build a CSV pipeline", [], state)

    request_kwargs = mock_acomp.call_args.kwargs
    assert request_kwargs["temperature"] == 0.0
    assert "seed" not in request_kwargs
    assert result.llm_calls[0].temperature == 0.0
    assert result.llm_calls[0].seed is None


@pytest.mark.asyncio
async def test_tool_call_then_final_response_records_both_llm_calls() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    tool_turn = _make_llm_response(
        tool_calls=[
            {
                "id": "call-1",
                "name": "set_metadata",
                "arguments": {"patch": {"name": "After tool"}},
            }
        ],
        request_id="chatcmpl-tool",
    )
    final_turn = _make_llm_response(content="Pipeline updated.", request_id="chatcmpl-final")
    mutated_state = replace(state, metadata=PipelineMetadata(name="After tool"), version=2)
    tool_result = ToolResult(
        success=True,
        updated_state=mutated_state,
        validation=ValidationSummary(is_valid=True, errors=(), warnings=(), suggestions=(), semantic_contracts=()),
        affected_nodes=(),
    )

    with (
        patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, side_effect=[tool_turn, final_turn]),
        patch("elspeth.web.composer.service.execute_tool", return_value=tool_result),
        patch.object(
            service, "_cached_runtime_preflight", new_callable=AsyncMock, return_value=ValidationResult(is_valid=True, checks=[], errors=[])
        ),
    ):
        result = await service.compose("Set a name", [], state)

    assert result.message == "Pipeline updated."
    assert [call.provider_request_id for call in result.llm_calls] == ["chatcmpl-tool", "chatcmpl-final"]
    assert [call.status for call in result.llm_calls] == [ComposerLLMCallStatus.SUCCESS, ComposerLLMCallStatus.SUCCESS]
    assert len(result.tool_invocations) == 1


@pytest.mark.asyncio
async def test_deadline_timeout_records_llm_call_on_convergence_error() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()

    async def timeout_llm(*_args: Any, **_kwargs: Any) -> _FakeLLMResponse:
        raise TimeoutError

    with (
        patch.object(service, "_call_llm", side_effect=timeout_llm),
        pytest.raises(ComposerConvergenceError) as exc_info,
    ):
        await service.compose("Hello", [], state)

    assert exc_info.value.budget_exhausted == "timeout"
    assert len(exc_info.value.llm_calls) == 1
    assert exc_info.value.llm_calls[0].status is ComposerLLMCallStatus.TIMEOUT


@pytest.mark.asyncio
async def test_bad_request_records_redacted_llm_call_on_service_error() -> None:
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    bad_request = LiteLLMBadRequestError(message="bad request leaked detail", model="bad", llm_provider="test")

    with (
        patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, side_effect=bad_request),
        pytest.raises(ComposerServiceError) as exc_info,
    ):
        await service.compose("Hello", [], _empty_state())

    assert str(exc_info.value) == "LLM request rejected (BadRequestError)"
    llm_calls = _captured_llm_calls(exc_info.value)
    assert len(llm_calls) == 1
    call = llm_calls[0]
    assert call.status is ComposerLLMCallStatus.BAD_REQUEST_ERROR
    assert call.error_class == "BadRequestError"
    assert call.error_message == "BadRequestError"


@pytest.mark.asyncio
async def test_empty_choices_records_malformed_response() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    empty_response = _FakeLLMResponse(choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=None, total_tokens=3))

    with (
        patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=empty_response),
        pytest.raises(ComposerServiceError) as exc_info,
    ):
        await service.compose("Hello", [], _empty_state())

    llm_calls = _captured_llm_calls(exc_info.value)
    assert len(llm_calls) == 1
    call = llm_calls[0]
    assert call.status is ComposerLLMCallStatus.MALFORMED_RESPONSE
    assert call.prompt_tokens == 3
    assert call.completion_tokens is None
    assert call.total_tokens == 3


@pytest.mark.asyncio
async def test_cancelled_model_call_records_cancelled_status() -> None:
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    cancelled = asyncio.CancelledError()

    with (
        patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, side_effect=cancelled),
        pytest.raises(asyncio.CancelledError) as exc_info,
    ):
        await service.compose("Hello", [], _empty_state())

    llm_calls = _captured_llm_calls(exc_info.value)
    assert len(llm_calls) == 1
    assert llm_calls[0].status is ComposerLLMCallStatus.CANCELLED
