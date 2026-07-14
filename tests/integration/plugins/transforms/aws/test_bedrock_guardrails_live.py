"""Opt-in live proof for operator-approved Bedrock Guardrail profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.plugins.transforms.aws.guardrails_live_check import run_guardrail_live_check
from elspeth.web.app import _settings_from_env

_RUN_GATE = "ELSPETH_RUN_LIVE_BEDROCK_GUARDRAILS"
_LIVE_INPUTS = {
    "aws_bedrock_prompt_shield": {
        "alias": "ELSPETH_LIVE_BEDROCK_PROMPT_PROFILE_ALIAS",
        "safe": "ELSPETH_LIVE_BEDROCK_PROMPT_SAFE_TEXT",
        "blocked": "ELSPETH_LIVE_BEDROCK_PROMPT_BLOCKED_TEXT",
        "version": "ELSPETH_LIVE_BEDROCK_PROMPT_EXPECTED_VERSION",
    },
    "aws_bedrock_content_safety": {
        "alias": "ELSPETH_LIVE_BEDROCK_CONTENT_PROFILE_ALIAS",
        "safe": "ELSPETH_LIVE_BEDROCK_CONTENT_SAFE_TEXT",
        "blocked": "ELSPETH_LIVE_BEDROCK_CONTENT_BLOCKED_TEXT",
        "version": "ELSPETH_LIVE_BEDROCK_CONTENT_EXPECTED_VERSION",
    },
}


@dataclass
class _LiveAuditRecorder:
    calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)

    def allocate_call_index(self, state_id: str) -> int:
        del state_id
        return len(self.calls)

    def record_call(self, **kwargs: Any) -> SimpleNamespace:
        self.order.append("audit")
        self.calls.append(kwargs)
        return SimpleNamespace(id=f"live-call-{len(self.calls)}")


@pytest.mark.live_aws
@pytest.mark.parametrize("plugin_id", tuple(_LIVE_INPUTS))
def test_operator_approved_bedrock_guardrail_profile_live(plugin_id: str) -> None:
    gate = os.getenv(_RUN_GATE)
    if gate is None:
        pytest.skip("live Bedrock Guardrail proof is opt-in")
    if gate != "1":
        pytest.fail("live Bedrock Guardrail proof gate is invalid", pytrace=False)

    names = _LIVE_INPUTS[plugin_id]
    values = {name: os.getenv(env_name) for name, env_name in names.items()}
    if any(not value for value in values.values()):
        pytest.fail("approved Bedrock Guardrail live inputs are incomplete", pytrace=False)
    try:
        settings = _settings_from_env()
    except Exception:
        pytest.fail("approved Bedrock Guardrail live profile configuration is invalid", pytrace=False)

    matching = tuple(
        profile for profile in settings.bedrock_guardrail_profiles if profile.plugin == plugin_id and profile.alias == values["alias"]
    )
    if (
        len(matching) != 1
        or values["version"] != matching[0].guardrail_version
        or f"transform:{plugin_id}" not in settings.plugin_allowlist
    ):
        pytest.fail("approved Bedrock Guardrail live policy input is unavailable", pytrace=False)

    recorder = _LiveAuditRecorder()
    order = recorder.order

    def emit(_event: object) -> None:
        order.append("telemetry")

    receipt = run_guardrail_live_check(
        profile=matching[0],
        safe_text=values["safe"],
        blocked_text=values["blocked"],
        execution=recorder,
        state_id=f"live-{plugin_id}",
        run_id=f"live-{plugin_id}",
        telemetry_emit=emit,
    )

    assert receipt.plugin_id == plugin_id
    assert receipt.profile_alias == values["alias"]
    assert receipt.safe_case_passed is True
    assert receipt.attack_case_blocked is True
    assert receipt.request_ids_present is True
    assert len(recorder.calls) == 2
    assert recorder.order == ["audit", "telemetry", "audit", "telemetry"]
