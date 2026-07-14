from __future__ import annotations

from typing import Any

import pytest
from tests.fixtures.factories import make_context
from tests.unit.plugins.transforms.aws.test_guardrails_client import CONTENT_FILTERS, FakeExecution

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_capabilities import ControlRole, PluginCapability, WebConfigAuthority
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.aws.bedrock_content_safety import AWSBedrockContentSafety
from elspeth.plugins.transforms.aws.guardrails_client import BedrockGuardrailsClient, GuardrailDecision, GuardrailUsage
from elspeth.testing import make_pipeline_row


def _config(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "guardrail_identifier": "privateguardrail",
        "guardrail_version": "7",
        "region": "us-east-1",
        "fields": ["content"],
        "source": "OUTPUT",
        "schema": {"mode": "observed"},
    }
    values.update(overrides)
    return values


def _started_transform(**overrides: object) -> tuple[AWSBedrockContentSafety, Any]:
    transform = AWSBedrockContentSafety(_config(**overrides))
    transform._sdk_client = object()
    context = make_context(landscape=FakeExecution())
    transform.on_start(context)
    return transform, context


def _decision(*, detected: bool = False) -> GuardrailDecision:
    return GuardrailDecision(
        detected=detected,
        intervened=detected,
        matched_filters=("VIOLENCE",) if detected else (),
        usage=GuardrailUsage((("contentPolicyUnits", 1),)),
        request_id=None,
    )


def test_content_safety_declares_output_blocking_capability() -> None:
    declaration = next(iter(AWSBedrockContentSafety.policy_capabilities))

    assert AWSBedrockContentSafety.determinism is Determinism.EXTERNAL_CALL
    assert AWSBedrockContentSafety.web_config_authority is WebConfigAuthority.OPERATOR_PROFILED
    assert declaration.capability is PluginCapability.CONTENT_SAFETY
    assert declaration.control_role is ControlRole.OUTPUT
    assert declaration.blocks_positive_detection is True
    assert AWSBedrockContentSafety.get_agent_assistance() is not None


@pytest.mark.parametrize("source", ["INPUT", "OUTPUT"])
def test_content_safety_uses_explicit_safe_source(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    transform, context = _started_transform(source=source)
    calls: list[dict[str, object]] = []

    def apply(_client: BedrockGuardrailsClient, **kwargs: object) -> GuardrailDecision:
        calls.append(kwargs)
        return _decision()

    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", apply)
    row = make_pipeline_row({"content": "safe", "kept": True})
    result = transform.process(row, context)

    assert result.status == "success"
    assert result.row is not None and result.row.to_dict() == row.to_dict()
    assert calls == [{"text": "safe", "source": source, "required_filters": CONTENT_FILTERS}]


def test_content_safety_rejects_unknown_source() -> None:
    with pytest.raises(PluginConfigError):
        AWSBedrockContentSafety(_config(source="BOTH"))


def test_content_safety_blocks_positive_without_provider_text(monkeypatch: pytest.MonkeyPatch) -> None:
    transform, context = _started_transform()
    monkeypatch.setattr(BedrockGuardrailsClient, "apply_guardrail", lambda _client, **_kwargs: _decision(detected=True))

    result = transform.process(make_pipeline_row({"content": "provider-text-marker"}), context)

    assert result.status == "error"
    assert result.reason == {
        "reason": "content_safety_violation",
        "field": "content",
        "categories": ["VIOLENCE"],
        "error_type": "intervened",
    }
    assert "provider-text-marker" not in repr(result)
