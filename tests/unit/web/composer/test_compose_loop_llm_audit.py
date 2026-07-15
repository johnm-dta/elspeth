"""Composer LLM-call audit tests for the async LiteLLM path."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.core.canonical import stable_hash
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer import llm_response_parsing as llm_response_parsing_module
from elspeth.web.composer.llm_response_parsing import build_llm_call_record, token_usage_from_response
from elspeth.web.composer.protocol import ComposerConvergenceError, ComposerServiceError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationSummary
from elspeth.web.composer.tools import ToolResult
from elspeth.web.config import WebSettings
from elspeth.web.execution.schemas import ValidationReadiness, ValidationResult
from tests.unit.web.composer._helpers import _stub_advisor_end_gate_clean  # noqa: F401  (autouse end-gate CLEAN stub)


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
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None
    thinking_blocks: list[dict[str, Any]] | None = None


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


def _passing_preflight() -> ValidationResult:
    return ValidationResult(
        is_valid=True,
        checks=[],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[]),
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
        knob_schema={"fields": []},
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
        "shareable_link_signing_key": b"\x00" * 32,
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
    cost: float | int | str | None = None,
    reasoning_tokens: int | None = None,
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
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
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost=cost,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=content,
                    tool_calls=fake_tool_calls,
                    reasoning=reasoning,
                    reasoning_content=reasoning_content,
                    reasoning_details=reasoning_details,
                    thinking_blocks=thinking_blocks,
                )
            )
        ],
        model=model,
        usage=usage,
        id=request_id,
    )


def _captured_llm_calls(exc: BaseException) -> Sequence[ComposerLLMCall]:
    calls = exc.__dict__["llm_calls"]
    assert isinstance(calls, tuple)
    return cast(tuple[ComposerLLMCall, ...], calls)


def test_llm_response_parser_does_not_call_dynamic_getattr() -> None:
    import inspect

    assert "getattr(" not in inspect.getsource(llm_response_parsing_module)


def test_token_usage_ignores_provider_properties_and_reads_own_fields() -> None:
    class _ProviderUsage:
        def __init__(self) -> None:
            self.completion_tokens = 7

        @property
        def prompt_tokens(self) -> int:
            raise AssertionError("provider property should not be invoked")

    usage = token_usage_from_response(SimpleNamespace(usage=_ProviderUsage()))

    assert usage.prompt_tokens is None
    assert usage.completion_tokens == 7


def test_token_usage_reads_real_litellm_pydantic_extra_fields() -> None:
    """LiteLLM ModelResponse keeps ``usage`` in ``__pydantic_extra__``, not ``__dict__``.

    Pydantic v2 models with ``extra="allow"`` store undeclared fields outside
    ``vars()``; the parser must still see them or every real provider response
    audits as unknown token counts with no provider cost.
    """
    from litellm.types.utils import ModelResponse, Usage

    usage_obj = Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    usage_obj.cost = 0.0123
    response = ModelResponse(usage=usage_obj)
    assert "usage" not in vars(response)  # the storage split this test pins

    usage = token_usage_from_response(response)

    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 7
    assert usage.reported_total == 18

    record = build_llm_call_record(
        model_requested="openrouter/openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        status=ComposerLLMCallStatus.SUCCESS,
        started_at=datetime.now(UTC),
        started_ns=time.monotonic_ns(),
        temperature=None,
        seed=None,
        response=response,
    )
    assert record.prompt_tokens == 11
    assert record.provider_cost == 0.0123


def test_llm_call_record_redacts_raw_provider_error_detail() -> None:
    openai_key = "sk-" + ("A" * 48)
    bearer_token = "Bearer " + ("x" * 32)
    raw_error = (
        "ProviderError headers={'Authorization': '"
        + bearer_token
        + "'} api_key="
        + openai_key
        + " prompt_template='classify {email}' "
        + "sample_rows=[{'email':'person@example.com','ssn':'123-45-6789'}] "
        + ("row detail " * 80)
    )

    call = build_llm_call_record(
        model_requested="provider/model",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        status=ComposerLLMCallStatus.API_ERROR,
        started_at=datetime.now(UTC),
        started_ns=time.monotonic_ns(),
        temperature=None,
        seed=None,
        response=None,
        error_class="ProviderError",
        error_message=raw_error,
    )

    stored = call.error_message
    assert stored is not None
    assert stored.startswith("provider error detail redacted")
    assert "raw_error_hash=" in stored
    assert len(stored) <= 512
    for leaked in (
        openai_key,
        bearer_token,
        "person@example.com",
        "123-45-6789",
        "classify {email}",
        "row detail",
    ):
        assert leaked not in stored


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


@pytest.mark.asyncio
async def test_text_only_success_records_llm_call_metadata() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.")

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response) as mock_acomp:
        result = await service.compose("Build a CSV pipeline", [], state)

    # New contract (post elspeth-861b0c58f5): model prose preserved verbatim,
    # system suffix appended; the synthetic [ELSPETH-SYSTEM] marker is the
    # operator-facing signal, not a wholesale replacement of the model's text.
    assert result.message.startswith("Done.")
    assert "[ELSPETH-SYSTEM]" in result.message
    assert result.raw_assistant_content == "Done."
    assert len(result.llm_calls) == 1
    call = result.llm_calls[0]
    request_kwargs = mock_acomp.call_args.kwargs
    assert call.status is ComposerLLMCallStatus.SUCCESS
    assert call.model_requested == "openrouter/openai/gpt-5.5"
    assert call.model_returned == "provider/model-returned"
    assert call.prompt_tokens == 11
    assert call.completion_tokens == 7
    assert call.total_tokens == 18
    assert call.provider_cost is None
    assert call.provider_cost_source == "not_available"
    assert call.provider_request_id == "chatcmpl-123"
    assert call.messages_hash == stable_hash(request_kwargs["messages"])
    assert call.tools_spec_hash == stable_hash(request_kwargs["tools"])


@pytest.mark.asyncio
async def test_success_records_provider_cost_from_usage_metadata() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.", cost=0.0037)

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response):
        result = await service.compose("Build a CSV pipeline", [], state)

    call = result.llm_calls[0]
    assert call.provider_cost == 0.0037
    assert call.provider_cost_source == "response_usage.cost"


@pytest.mark.asyncio
async def test_success_records_provider_reasoning_metadata() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    reasoning_details = [{"type": "reasoning.text", "text": "selected set_pipeline"}]
    thinking_blocks = [{"type": "thinking", "thinking": "checked required output options"}]
    llm_response = _make_llm_response(
        content="Done.",
        reasoning_tokens=12,
        reasoning="openrouter native reasoning",
        reasoning_content="litellm normalized reasoning",
        reasoning_details=reasoning_details,
        thinking_blocks=thinking_blocks,
    )

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response):
        result = await service.compose("Build a CSV pipeline", [], state)

    call = result.llm_calls[0]
    payload = call.to_dict()
    assert call.reasoning_tokens == 12
    assert call.reasoning_content == "openrouter native reasoning"
    assert payload["reasoning_details"] == reasoning_details
    assert payload["thinking_blocks"] == thinking_blocks


@pytest.mark.asyncio
async def test_malformed_provider_cost_is_recorded_as_unavailable() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.", cost="not-a-number")

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response):
        result = await service.compose("Build a CSV pipeline", [], state)

    call = result.llm_calls[0]
    assert call.provider_cost is None
    assert call.provider_cost_source == "not_available"


@pytest.mark.asyncio
async def test_unset_sampling_is_omitted_and_reflected_in_audit() -> None:
    service = ComposerServiceImpl.for_trained_operator(
        catalog=_mock_catalog(),
        settings=_make_settings(composer_model="anthropic/claude-3-5-sonnet-20241022"),
    )
    state = _empty_state()
    llm_response = _make_llm_response(content="Done.")

    with patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, return_value=llm_response) as mock_acomp:
        result = await service.compose("Build a CSV pipeline", [], state)

    request_kwargs = mock_acomp.call_args.kwargs
    assert "temperature" not in request_kwargs
    assert "seed" not in request_kwargs
    assert result.llm_calls[0].temperature is None
    assert result.llm_calls[0].seed is None


@pytest.mark.asyncio
async def test_tool_call_then_final_response_records_both_llm_calls() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
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
        patch("elspeth.web.composer.tool_batch.execute_tool", return_value=tool_result),
        patch.object(service, "_cached_runtime_preflight", new_callable=AsyncMock, return_value=_passing_preflight()),
    ):
        result = await service.compose("Set a name", [], state)

    assert result.message == "Pipeline updated."
    assert [call.provider_request_id for call in result.llm_calls] == ["chatcmpl-tool", "chatcmpl-final"]
    assert [call.status for call in result.llm_calls] == [ComposerLLMCallStatus.SUCCESS, ComposerLLMCallStatus.SUCCESS]
    assert len(result.tool_invocations) == 1


@pytest.mark.asyncio
async def test_deadline_timeout_records_llm_call_on_convergence_error() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
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

    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
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
async def test_unclassified_provider_exception_records_api_error_call() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())

    with (
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=ValueError("unexpected codec failure"),
        ),
        pytest.raises(ValueError, match="unexpected codec failure") as exc_info,
    ):
        await service.compose("Hello", [], _empty_state())

    llm_calls = _captured_llm_calls(exc_info.value)
    assert len(llm_calls) == 1
    call = llm_calls[0]
    assert call.status is ComposerLLMCallStatus.API_ERROR
    assert call.error_class == "ValueError"
    assert call.error_message == "ValueError"


@pytest.mark.asyncio
async def test_empty_choices_records_malformed_response() -> None:
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
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
    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=_make_settings())
    cancelled = asyncio.CancelledError()

    with (
        patch("elspeth.web.composer.service._litellm_acompletion", new_callable=AsyncMock, side_effect=cancelled),
        pytest.raises(asyncio.CancelledError) as exc_info,
    ):
        await service.compose("Hello", [], _empty_state())

    llm_calls = _captured_llm_calls(exc_info.value)
    assert len(llm_calls) == 1
    assert llm_calls[0].status is ComposerLLMCallStatus.CANCELLED
