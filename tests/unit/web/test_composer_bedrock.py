"""Real Composer service contracts for Bedrock's default AWS credential chain."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.web.composer import service as service_module
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.service import AdvisorCheckpointVerdict, ComposerServiceImpl
from tests.unit.web.composer._helpers import _empty_state, _make_llm_response, _make_settings, _mock_catalog

_BEDROCK_PRIMARY = "bedrock/global.anthropic.claude-sonnet-4-6"
_BEDROCK_ADVISOR = "bedrock/global.anthropic.claude-opus-4-6-v1"
_STATIC_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "AZURE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
)
_FORBIDDEN_BEDROCK_KWARGS = {
    "api_base",
    "base_url",
    "api_key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_profile_name",
    "aws_role_name",
    "aws_bedrock_runtime_endpoint",
    "gateway_arn",
    "agentcore_gateway_arn",
}


def _clear_static_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _STATIC_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)


def _bedrock_service() -> ComposerServiceImpl:
    settings = _make_settings(
        composer_model=_BEDROCK_PRIMARY,
        composer_advisor_model=_BEDROCK_ADVISOR,
        composer_temperature=None,
        composer_seed=None,
    )
    return ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=settings)


@pytest.mark.asyncio
async def test_bedrock_primary_uses_real_service_path_without_static_provider_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_static_provider_keys(monkeypatch)
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return _make_llm_response(content="No pipeline changes are needed.")

    async def clean_checkpoint(*_args: object, **_kwargs: object) -> AdvisorCheckpointVerdict:
        return AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN")

    monkeypatch.setattr(service_module, "_litellm_acompletion", fake_acompletion)
    service = _bedrock_service()
    monkeypatch.setattr(service, "_run_advisor_checkpoint", clean_checkpoint)

    availability = service.get_availability()
    assert availability.available is True
    assert availability.provider == "bedrock"
    expected_tool_names = {tool["function"]["name"] for tool in service._get_litellm_tools()}

    await service.compose("No pipeline changes are needed.", [], _empty_state())

    assert len(captured) == 1
    request = captured[0]
    assert request["model"] == _BEDROCK_PRIMARY
    assert {tool["function"]["name"] for tool in request["tools"]} == expected_tool_names
    assert all(tool["type"] == "function" and set(tool["function"]) == {"name", "description", "parameters"} for tool in request["tools"])
    assert set(request) == {"model", "messages", "tools"}
    assert not (_FORBIDDEN_BEDROCK_KWARGS & set(request))


@pytest.mark.asyncio
async def test_bedrock_advisor_uses_default_chain_without_tools_or_gateway_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_static_provider_keys(monkeypatch)
    captured: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return SimpleNamespace(
            model=_BEDROCK_ADVISOR,
            choices=[SimpleNamespace(message=SimpleNamespace(content="CLEAN"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2, total_tokens=13),
        )

    monkeypatch.setattr(service_module, "_litellm_acompletion", fake_acompletion)
    service = _bedrock_service()
    recorder = BufferingRecorder()

    guidance, metadata = await service._call_advisor_with_audit(
        {
            "trigger": "end",
            "problem_summary": "Review the complete pipeline.",
            "recent_errors": [],
            "attempted_actions": [],
        },
        recorder=recorder,
    )

    assert guidance == "CLEAN"
    assert metadata["model"] == _BEDROCK_ADVISOR
    assert len(captured) == 1
    request = captured[0]
    assert request["model"] == _BEDROCK_ADVISOR
    assert set(request) == {"model", "messages", "max_tokens"}
    assert "tools" not in request
    assert not (_FORBIDDEN_BEDROCK_KWARGS & set(request))
    assert recorder.llm_calls[-1].model_requested == _BEDROCK_ADVISOR


def test_openrouter_primary_still_fails_closed_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_static_provider_keys(monkeypatch)
    settings = _make_settings(
        composer_model="openrouter/openai/gpt-5.4",
        composer_advisor_model=_BEDROCK_ADVISOR,
    )

    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=settings)

    availability = service.get_availability()
    assert availability.available is False
    assert availability.provider == "openrouter"
    assert availability.missing_keys == ("OPENROUTER_API_KEY",)


def test_openrouter_advisor_fails_closed_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_static_provider_keys(monkeypatch)
    settings = _make_settings(
        composer_model=_BEDROCK_PRIMARY,
        composer_advisor_model="openrouter/anthropic/claude-opus-4.6",
    )

    service = ComposerServiceImpl.for_trained_operator(catalog=_mock_catalog(), settings=settings)

    availability = service.get_availability()
    assert availability.available is False
    assert availability.provider == "bedrock"
    assert availability.missing_keys == ("OPENROUTER_API_KEY",)
    assert availability.reason is not None
    assert "advisor model" in availability.reason
