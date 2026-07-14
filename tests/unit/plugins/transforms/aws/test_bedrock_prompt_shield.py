from __future__ import annotations

from typing import Any

import pytest
from tests.fixtures.factories import make_context
from tests.unit.plugins.transforms.aws.test_guardrails_client import FakeExecution

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_capabilities import ControlRole, PluginCapability, WebConfigAuthority
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.aws.bedrock_prompt_shield import AWSBedrockPromptShield
from elspeth.plugins.transforms.aws.guardrails_client import (
    BedrockGuardrailsClient,
    GuardrailDecision,
    GuardrailServiceError,
    GuardrailUsage,
)
from elspeth.testing import make_pipeline_row


def _config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "guardrail_identifier": "privateguardrail",
        "guardrail_version": "7",
        "region": "us-east-1",
        "fields": ["prompt"],
        "schema": {"mode": "observed"},
    }
    values.update(overrides)
    return values


def _decision(*, detected: bool = False, intervened: bool = False) -> GuardrailDecision:
    return GuardrailDecision(
        detected=detected,
        intervened=intervened,
        matched_filters=("PROMPT_ATTACK",) if detected else (),
        usage=GuardrailUsage((("contentPolicyUnits", 1),)),
        request_id="request-1",
    )


def _started_transform(config: dict[str, object] | None = None) -> tuple[AWSBedrockPromptShield, Any]:
    transform = AWSBedrockPromptShield(config or _config())
    transform._sdk_client = object()
    context = make_context(landscape=FakeExecution())
    transform.on_start(context)
    return transform, context


def test_prompt_shield_config_requires_closed_private_binding() -> None:
    for missing in ("guardrail_identifier", "guardrail_version", "region", "fields", "schema"):
        config = _config()
        del config[missing]
        with pytest.raises(PluginConfigError):
            AWSBedrockPromptShield(config)

    for forbidden in ("access_key", "secret_key", "endpoint_url"):
        with pytest.raises(PluginConfigError):
            AWSBedrockPromptShield(_config(**{forbidden: "forbidden"}))


def test_prompt_shield_declares_operator_profiled_blocking_capability() -> None:
    declaration = next(iter(AWSBedrockPromptShield.policy_capabilities))

    assert AWSBedrockPromptShield.determinism is Determinism.EXTERNAL_CALL
    assert AWSBedrockPromptShield.web_config_authority is WebConfigAuthority.OPERATOR_PROFILED
    assert declaration.capability is PluginCapability.PROMPT_SHIELD
    assert declaration.control_role is ControlRole.INPUT
    assert declaration.blocks_positive_detection is True
    assert AWSBedrockPromptShield.get_agent_assistance() is not None


def test_prompt_shield_passes_original_row_only_when_explicitly_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    transform, context = _started_transform()
    calls: list[dict[str, object]] = []

    def apply(_client: BedrockGuardrailsClient, **kwargs: object) -> GuardrailDecision:
        calls.append(kwargs)
        return _decision()

    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", apply)
    row = make_pipeline_row({"prompt": "safe", "kept": 7})
    result = transform.process(row, context)

    assert result.status == "success"
    assert result.row is not None
    assert result.row.to_dict() == row.to_dict()
    assert calls == [{"text": "safe", "source": "INPUT", "required_filters": ("PROMPT_ATTACK",)}]


@pytest.mark.parametrize("intervened", [False, True])
def test_prompt_shield_blocks_detect_only_and_intervention(monkeypatch: pytest.MonkeyPatch, intervened: bool) -> None:
    transform, context = _started_transform()
    monkeypatch.setattr(
        BedrockGuardrailsClient,
        "apply_guardrail",
        lambda _client, **_kwargs: _decision(detected=True, intervened=intervened),
    )

    result = transform.process(make_pipeline_row({"prompt": "marker"}), context)

    assert result.status == "error"
    assert result.reason == {
        "reason": "prompt_injection_detected",
        "field": "prompt",
        "categories": ["PROMPT_ATTACK"],
        "error_type": "intervened" if intervened else "detect_only",
    }
    assert result.retryable is False


@pytest.mark.parametrize(
    ("row", "reason"),
    [({}, "missing_field"), ({"prompt": 7}, "non_string_field")],
)
def test_prompt_shield_fails_closed_before_call_for_invalid_fields(
    monkeypatch: pytest.MonkeyPatch,
    row: dict[str, object],
    reason: str,
) -> None:
    transform, context = _started_transform()
    called = False

    def apply(_client: BedrockGuardrailsClient, **_kwargs: object) -> GuardrailDecision:
        nonlocal called
        called = True
        return _decision()

    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", apply)
    result = transform.process(make_pipeline_row(row), context)

    assert result.status == "error"
    assert result.reason is not None and result.reason["reason"] == reason
    assert called is False


def test_prompt_shield_evaluates_every_field_and_stops_on_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    transform, context = _started_transform(_config(fields=["first", "second", "third"]))
    calls: list[str] = []

    def apply(_client: BedrockGuardrailsClient, **kwargs: object) -> GuardrailDecision:
        text = str(kwargs["text"])
        calls.append(text)
        return _decision(detected=text == "blocked")

    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", apply)
    result = transform.process(
        make_pipeline_row({"first": "safe", "second": "blocked", "third": "not-called"}),
        context,
    )

    assert result.status == "error"
    assert calls == ["safe", "blocked"]


@pytest.mark.parametrize(
    "failure",
    [GuardrailServiceError(retryable=True), ValueError("malformed provider marker")],
)
def test_prompt_shield_sanitizes_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    transform, context = _started_transform()

    def apply(_client: BedrockGuardrailsClient, **_kwargs: object) -> GuardrailDecision:
        raise failure

    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", apply)
    result = transform.process(make_pipeline_row({"prompt": "marker"}), context)

    assert result.status == "error"
    assert result.reason == {
        "reason": "api_call_failed",
        "field": "prompt",
        "error_type": "guardrail_service_error",
    }
    assert "malformed provider marker" not in repr(result)
