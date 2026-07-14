from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pytest
from botocore.exceptions import ClientError
from tests.unit.plugins.transforms.aws.test_guardrails_client import (
    CONTENT_FILTERS,
    FakeExecution,
    response,
)

from elspeth.plugins.transforms.aws.guardrail_profiles import BedrockGuardrailProfileSettings
from elspeth.plugins.transforms.aws.guardrails_live_check import (
    GuardrailLiveCheckError,
    GuardrailLiveReceipt,
    run_guardrail_live_check,
)


class _SequencedSDK:
    def __init__(self, *responses: object) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def apply_guardrail(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        item = next(self._responses)
        if isinstance(item, Exception):
            raise item
        return item


def _profile(plugin: str = "aws_bedrock_prompt_shield") -> BedrockGuardrailProfileSettings:
    return BedrockGuardrailProfileSettings.model_validate(
        {
            "alias": "prompt-default" if plugin == "aws_bedrock_prompt_shield" else "content-default",
            "plugin": plugin,
            "guardrail_identifier": "privateguardrailmarker",
            "guardrail_version": "73",
            "region": "us-private-1",
        }
    )


def _run(
    sdk: Any,
    *,
    profile: BedrockGuardrailProfileSettings | None = None,
    safe_text: str = "SAFE_FIXTURE_TEXT_MARKER",
    blocked_text: str = "BLOCKED_FIXTURE_TEXT_MARKER",
) -> tuple[GuardrailLiveReceipt, FakeExecution, list[object]]:
    execution = FakeExecution()
    events: list[object] = []
    receipt = run_guardrail_live_check(
        profile=profile or _profile(),
        safe_text=safe_text,
        blocked_text=blocked_text,
        execution=execution,
        state_id="state-1",
        run_id="run-1",
        telemetry_emit=events.append,
        sdk_client=sdk,
    )
    return receipt, execution, events


def test_live_check_returns_only_bounded_receipt_and_uses_guard_content() -> None:
    sdk = _SequencedSDK(
        response(),
        response(detected="PROMPT_ATTACK"),
    )

    receipt, execution, events = _run(sdk)

    assert receipt == GuardrailLiveReceipt(
        plugin_id="aws_bedrock_prompt_shield",
        profile_alias="prompt-default",
        safe_case_passed=True,
        attack_case_blocked=True,
        request_ids_present=True,
    )
    assert len(execution.calls) == 2
    assert len(events) == 2
    assert [call["source"] for call in sdk.calls] == ["INPUT", "INPUT"]
    assert [call["outputScope"] for call in sdk.calls] == ["FULL", "FULL"]
    assert [call["content"][0]["text"]["qualifiers"] for call in sdk.calls] == [["guard_content"], ["guard_content"]]
    rendered = json.dumps(asdict(receipt), sort_keys=True)
    for private in (
        "privateguardrailmarker",
        "us-private-1",
        "SAFE_FIXTURE_TEXT_MARKER",
        "BLOCKED_FIXTURE_TEXT_MARKER",
        "request-1",
        "arn:aws:iam::123456789012:role/private-role",
        "AWS_SECRET_ACCESS_KEY",
    ):
        assert private not in rendered


def test_content_live_check_uses_output_policy_and_accepts_intervention() -> None:
    sdk = _SequencedSDK(
        response(CONTENT_FILTERS),
        response(
            CONTENT_FILTERS,
            action="GUARDRAIL_INTERVENED",
            detected="VIOLENCE",
            blocked=True,
            outputs=[{"text": "PROVIDER_CANNED_TEXT_MARKER"}],
        ),
    )

    receipt, _execution, _events = _run(sdk, profile=_profile("aws_bedrock_content_safety"))

    assert receipt.plugin_id == "aws_bedrock_content_safety"
    assert receipt.profile_alias == "content-default"
    assert receipt.safe_case_passed is True
    assert receipt.attack_case_blocked is True
    assert [call["source"] for call in sdk.calls] == ["OUTPUT", "OUTPUT"]
    assert "PROVIDER_CANNED_TEXT_MARKER" not in repr(receipt)


@pytest.mark.parametrize(
    ("responses", "expected_message"),
    [
        ((response(detected="PROMPT_ATTACK"), response(detected="PROMPT_ATTACK")), "approved safe case did not pass"),
        ((response(), response()), "approved blocked case was not blocked"),
    ],
)
def test_fixture_outcome_mismatch_raises_static_sanitized_error(
    responses: tuple[dict[str, Any], dict[str, Any]],
    expected_message: str,
) -> None:
    with pytest.raises(GuardrailLiveCheckError, match=expected_message) as exc_info:
        _run(_SequencedSDK(*responses))

    rendered = str(exc_info.value)
    assert "SAFE_FIXTURE_TEXT_MARKER" not in rendered
    assert "BLOCKED_FIXTURE_TEXT_MARKER" not in rendered
    assert "privateguardrailmarker" not in rendered


def test_provider_failure_is_sanitized_without_fixture_binding_or_body() -> None:
    provider_marker = "PRIVATE_PROVIDER_ERROR_MARKER"
    failure = ClientError(
        {
            "Error": {"Code": "AccessDeniedException", "Message": provider_marker},
            "ResponseMetadata": {"RetryAttempts": 0},
        },
        "ApplyGuardrail",
    )

    with pytest.raises(GuardrailLiveCheckError, match="Bedrock Guardrail live check failed") as exc_info:
        _run(_SequencedSDK(failure))

    rendered = str(exc_info.value)
    for private in (
        provider_marker,
        "SAFE_FIXTURE_TEXT_MARKER",
        "BLOCKED_FIXTURE_TEXT_MARKER",
        "privateguardrailmarker",
        "us-private-1",
    ):
        assert private not in rendered


def test_missing_provider_request_ids_are_receipted_without_values() -> None:
    safe = response()
    blocked = response(detected="PROMPT_ATTACK")
    safe.pop("ResponseMetadata")
    blocked.pop("ResponseMetadata")

    receipt, _execution, _events = _run(_SequencedSDK(safe, blocked))

    assert receipt.request_ids_present is False
