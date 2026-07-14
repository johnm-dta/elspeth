from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from elspeth.contracts import CallStatus
from elspeth.plugins.transforms.aws.guardrails_client import (
    ALL_USAGE_KEYS,
    BedrockGuardrailsClient,
    GuardrailResponseError,
)

PROMPT_FILTERS = ("PROMPT_ATTACK",)
CONTENT_FILTERS = ("INSULTS", "HATE", "SEXUAL", "VIOLENCE", "MISCONDUCT")


@dataclass
class FakeExecution:
    calls: list[dict[str, Any]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    fail_record: bool = False

    def allocate_call_index(self, state_id: str) -> int:
        assert state_id == "state-1"
        return len(self.calls)

    def record_call(self, **kwargs: Any) -> SimpleNamespace:
        self.order.append("audit")
        if self.fail_record:
            raise RuntimeError("audit unavailable")
        self.calls.append(kwargs)
        return SimpleNamespace(id="call-1")


def _sdk_client() -> Any:
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _usage() -> dict[str, int]:
    return {key: index for index, key in enumerate(sorted(ALL_USAGE_KEYS))}


def _filters(required: tuple[str, ...], *, detected: str | None = None, blocked: bool = False) -> list[dict[str, object]]:
    return [
        {
            "type": name,
            "confidence": "HIGH" if name == detected else "NONE",
            "action": "BLOCKED" if blocked and name == detected else "NONE",
            "detected": name == detected,
        }
        for name in required
    ]


def response(
    required: tuple[str, ...] = PROMPT_FILTERS,
    *,
    action: str = "NONE",
    detected: str | None = None,
    blocked: bool = False,
    outputs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "usage": _usage(),
        "action": action,
        "outputs": [] if outputs is None else outputs,
        "assessments": [{"contentPolicy": {"filters": _filters(required, detected=detected, blocked=blocked)}}],
        "ResponseMetadata": {"RequestId": "request-1", "RetryAttempts": 0, "HTTPStatusCode": 200, "HTTPHeaders": {}},
    }


def _client(sdk: Any, execution: FakeExecution, events: list[Any]) -> BedrockGuardrailsClient:
    return BedrockGuardrailsClient(
        execution=execution,
        state_id="state-1",
        run_id="run-1",
        telemetry_emit=lambda event: (execution.order.append("telemetry"), events.append(event)),
        guardrail_identifier="privateguardrail",
        guardrail_version="7",
        region="us-east-1",
        sdk_client=sdk,
        audit_salt=b"0123456789abcdef0123456789abcdef",
        token_id="token-1",
    )


def test_apply_guardrail_uses_exact_guard_content_request() -> None:
    sdk = _sdk_client()
    execution = FakeExecution()
    events: list[Any] = []
    expected = {
        "guardrailIdentifier": "privateguardrail",
        "guardrailVersion": "7",
        "source": "INPUT",
        "outputScope": "FULL",
        "content": [{"text": {"text": "untrusted text", "qualifiers": ["guard_content"]}}],
    }
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", response(), expected_params=expected)
        decision = _client(sdk, execution, events).apply_guardrail(
            text="untrusted text",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert decision.detected is False
    assert decision.intervened is False
    assert decision.matched_filters == ()
    assert decision.request_id == "request-1"
    assert execution.order == ["audit", "telemetry"]
    request_payload = execution.calls[0]["request_data"].to_dict()
    response_payload = execution.calls[0]["response_data"].to_dict()
    rendered = repr((request_payload, response_payload, events))
    assert "untrusted text" not in rendered
    assert "privateguardrail" not in rendered
    assert execution.calls[0]["status"] is CallStatus.SUCCESS
    assert decision.usage.units == tuple(sorted(_usage().items()))


def test_detect_only_positive_blocks_even_when_top_action_is_none() -> None:
    sdk = _sdk_client()
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", response(detected="PROMPT_ATTACK"))
        decision = _client(sdk, FakeExecution(), []).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert decision.detected is True
    assert decision.intervened is False
    assert decision.matched_filters == ("PROMPT_ATTACK",)


def test_intervention_accepts_multiple_outputs_and_discards_text() -> None:
    sdk = _sdk_client()
    marker = "PROVIDER_CANNED_OUTPUT_MARKER"
    with Stubber(sdk) as stubber:
        stubber.add_response(
            "apply_guardrail",
            response(
                action="GUARDRAIL_INTERVENED",
                detected="PROMPT_ATTACK",
                blocked=True,
                outputs=[{"text": marker}, {"text": f"{marker}2"}],
            ),
        )
        decision = _client(sdk, FakeExecution(), []).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert decision.detected is True
    assert decision.intervened is True
    assert marker not in repr(decision)


def test_content_policy_requires_exact_approved_filter_set() -> None:
    sdk = _sdk_client()
    malformed = response(CONTENT_FILTERS[:-1])
    marker = "PROVIDER_TEXT_MARKER"
    malformed["outputs"] = [{"text": marker}]
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", malformed)
        with pytest.raises(GuardrailResponseError) as exc_info:
            _client(sdk, FakeExecution(), []).apply_guardrail(
                text="marker",
                source="OUTPUT",
                required_filters=CONTENT_FILTERS,
            )

    assert marker not in str(exc_info.value)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(action="UNKNOWN"),
        lambda value: value.update(assessments=[]),
        lambda value: value.update(assessments=value["assessments"] * 2),
        lambda value: value["assessments"][0]["contentPolicy"]["filters"][0].update(detected="true"),
        lambda value: value["usage"].update(unknownUnits=1),
        lambda value: value["usage"].update(contentPolicyUnits=-1),
        lambda value: value.update(outputs=[{"text": "unexpected"}]),
    ],
)
def test_malformed_responses_fail_closed(mutator: Any) -> None:
    malformed = response()
    mutator(malformed)

    class MalformedSDK:
        def apply_guardrail(self, **_kwargs: object) -> dict[str, Any]:
            return malformed

    with pytest.raises(GuardrailResponseError):
        _client(MalformedSDK(), FakeExecution(), []).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )


def test_audit_failure_suppresses_telemetry() -> None:
    sdk = _sdk_client()
    execution = FakeExecution(fail_record=True)
    events: list[Any] = []
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", response())
        with pytest.raises(RuntimeError, match="audit unavailable"):
            _client(sdk, execution, events).apply_guardrail(
                text="marker",
                source="INPUT",
                required_filters=PROMPT_FILTERS,
            )

    assert execution.order == ["audit"]
    assert events == []


def test_service_error_is_sanitized_audited_then_telemetered() -> None:
    marker = "PRIVATE_PROVIDER_ERROR_BODY"

    class FailingSDK:
        def apply_guardrail(self, **_kwargs: object) -> dict[str, Any]:
            raise ClientError(
                {
                    "Error": {"Code": "ThrottlingException", "Message": marker},
                    "ResponseMetadata": {"RetryAttempts": 2},
                },
                "ApplyGuardrail",
            )

    execution = FakeExecution()
    events: list[Any] = []
    with pytest.raises(RuntimeError) as exc_info:
        _client(FailingSDK(), execution, events).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert marker not in str(exc_info.value)
    assert execution.order == ["audit", "telemetry"]
    assert execution.calls[0]["status"] is CallStatus.ERROR
    assert execution.calls[0]["response_data"].to_dict()["attempts"] == 3
    assert marker not in repr((execution.calls, events))


def test_sdk_retry_configuration_has_one_owner() -> None:
    client = BedrockGuardrailsClient(
        execution=FakeExecution(),
        state_id="state-1",
        run_id="run-1",
        telemetry_emit=lambda _event: None,
        guardrail_identifier="privateguardrail",
        guardrail_version="7",
        region="us-east-1",
        audit_salt=b"0123456789abcdef0123456789abcdef",
    )

    config = client.sdk_client.meta.config
    assert config.retries["mode"] == "standard"
    assert config.retries["total_max_attempts"] == 3
    assert config.connect_timeout == 5
    assert config.read_timeout == 15
