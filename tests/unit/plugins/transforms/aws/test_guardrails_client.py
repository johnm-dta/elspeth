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
    GuardrailPartialCoverageError,
    GuardrailResponseError,
    parse_guardrail_response,
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


def _required_usage() -> dict[str, int]:
    optional = {"automatedReasoningPolicies", "automatedReasoningPolicyUnits", "contentPolicyImageUnits"}
    return {key: value for key, value in _usage().items() if key not in optional}


def _coverage() -> dict[str, dict[str, int]]:
    return {
        "textCharacters": {"guarded": 12, "total": 12},
        "images": {"guarded": 0, "total": 0},
    }


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
    full_optional_sections: bool = False,
) -> dict[str, Any]:
    assessment: dict[str, Any] = {"contentPolicy": {"filters": _filters(required, detected=detected, blocked=blocked)}}
    result: dict[str, Any] = {
        "usage": _usage(),
        "action": action,
        "outputs": [] if outputs is None else outputs,
        "assessments": [assessment],
        "ResponseMetadata": {"RequestId": "request-1", "RetryAttempts": 0, "HTTPStatusCode": 200, "HTTPHeaders": {}},
    }
    if full_optional_sections:
        result["actionReason"] = "Guardrail policy assessment complete"
        result["guardrailCoverage"] = _coverage()
        assessment["invocationMetrics"] = {
            "guardrailProcessingLatency": 17,
            "usage": _usage(),
            "guardrailCoverage": _coverage(),
        }
        assessment["appliedGuardrailDetails"] = {
            "guardrailId": "privateguardrailmarker",
            "guardrailVersion": "7",
            "guardrailArn": "arn:aws:bedrock:us-east-1:123456789012:guardrail/privateguardrailmarker",
            "guardrailOrigin": ["REQUEST"],
            "guardrailOwnership": "SELF",
        }
    return result


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


def test_full_optional_sections_are_validated_and_private_details_are_discarded() -> None:
    sdk = _sdk_client()
    execution = FakeExecution()
    events: list[Any] = []
    private_markers = ("privateguardrailmarker", "123456789012")
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", response(full_optional_sections=True))
        decision = _client(sdk, execution, events).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    rendered = repr((decision, execution.calls, events))
    for marker in private_markers:
        assert marker not in rendered


@pytest.mark.parametrize(
    "invocation_metrics",
    [
        {},
        {"guardrailProcessingLatency": 17},
        {"usage": _required_usage()},
        {"guardrailCoverage": {"textCharacters": {}}},
        {"guardrailCoverage": {"textCharacters": {"guarded": 3}, "images": {"total": 0}}},
    ],
    ids=("empty", "latency-only", "required-usage-only", "empty-coverage-bucket", "one-sided-coverage"),
)
def test_model_conforming_partial_invocation_metrics_are_accepted(invocation_metrics: dict[str, object]) -> None:
    provider_response = response(full_optional_sections=True)
    provider_response["assessments"][0]["invocationMetrics"] = invocation_metrics

    decision, attempts = parse_guardrail_response(provider_response, required_filters=PROMPT_FILTERS)

    assert decision.detected is False
    assert attempts == 1


@pytest.mark.parametrize(
    "coverage",
    [
        {},
        {"textCharacters": {}},
        {"textCharacters": {"guarded": 3}},
        {"textCharacters": {"total": 12}},
        {"images": {"guarded": 0}},
    ],
    ids=("empty", "empty-bucket", "guarded-only", "total-only", "images-guarded-only"),
)
def test_model_conforming_partial_root_coverage_is_accepted(coverage: dict[str, object]) -> None:
    provider_response = response(full_optional_sections=True)
    provider_response["guardrailCoverage"] = coverage

    decision, attempts = parse_guardrail_response(provider_response, required_filters=PROMPT_FILTERS)

    assert decision.detected is False
    assert attempts == 1


@pytest.mark.parametrize("location", ["root", "invocation_metrics"])
def test_partial_guardrail_coverage_fails_closed(location: str) -> None:
    provider_response = response(full_optional_sections=True)
    partial = {"textCharacters": {"guarded": 11, "total": 12}, "images": {"guarded": 0, "total": 0}}
    if location == "root":
        provider_response["guardrailCoverage"] = partial
    else:
        provider_response["assessments"][0]["invocationMetrics"]["guardrailCoverage"] = partial

    with pytest.raises(GuardrailPartialCoverageError) as exc_info:
        parse_guardrail_response(provider_response, required_filters=PROMPT_FILTERS)

    # Partial coverage is fail-closed but distinctly classified (operator
    # decision 2026-07-17): still a GuardrailResponseError subclass so every
    # existing handler keeps failing closed, with counts for diagnosis.
    assert isinstance(exc_info.value, GuardrailResponseError)
    assert exc_info.value.coverage_key == "textCharacters"
    assert exc_info.value.guarded == 11
    assert exc_info.value.total == 12


def test_partial_coverage_audits_distinct_error_payload() -> None:
    partial_response = response(full_optional_sections=True)
    partial_response["guardrailCoverage"] = {"textCharacters": {"guarded": 11, "total": 12}}

    class PartialSDK:
        def apply_guardrail(self, **_kwargs: object) -> dict[str, Any]:
            return partial_response

    execution = FakeExecution()
    with pytest.raises(GuardrailPartialCoverageError):
        _client(PartialSDK(), execution, []).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    recorded = execution.calls[0]
    assert recorded["status"] is CallStatus.ERROR
    response_data = recorded["response_data"].to_dict()
    assert response_data["status"] == "partial_coverage"
    assert response_data["coverage_key"] == "textCharacters"
    assert response_data["guarded_units"] == 11
    assert response_data["total_units"] == 12
    assert recorded["error"].to_dict()["type"] == "partial_coverage"


def test_root_usage_accepts_required_six_without_optional_counters() -> None:
    provider_response = response()
    provider_response["usage"] = _required_usage()

    decision, _attempts = parse_guardrail_response(provider_response, required_filters=PROMPT_FILTERS)

    assert decision.usage.units == tuple(sorted(_required_usage().items()))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(actionReason=123),
        lambda value: value.update(actionReason="x" * 4097),
        lambda value: value["guardrailCoverage"].update(unknown={"guarded": 0, "total": 0}),
        lambda value: value["guardrailCoverage"]["textCharacters"].update(guarded=True),
        lambda value: value["guardrailCoverage"]["textCharacters"].update(guarded=13, total=12),
        lambda value: value["guardrailCoverage"]["textCharacters"].update(total=2**31),
        lambda value: value["assessments"][0]["invocationMetrics"].update(unknown=1),
        lambda value: value["assessments"][0]["invocationMetrics"].update(guardrailProcessingLatency=-1),
        lambda value: value["assessments"][0]["invocationMetrics"].update(guardrailProcessingLatency=2**63),
        lambda value: value["assessments"][0]["invocationMetrics"]["usage"].pop("topicPolicyUnits"),
        lambda value: value["assessments"][0]["appliedGuardrailDetails"].update(unknown="private"),
        lambda value: value["assessments"][0]["appliedGuardrailDetails"].update(guardrailId="x" * 2049),
        lambda value: value["assessments"][0]["appliedGuardrailDetails"].update(guardrailOrigin=["UNKNOWN"]),
    ],
)
def test_malformed_full_optional_sections_fail_closed(mutator: Any) -> None:
    malformed = response(full_optional_sections=True)
    mutator(malformed)

    with pytest.raises(GuardrailResponseError):
        parse_guardrail_response(malformed, required_filters=PROMPT_FILTERS)


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
        lambda value: value["usage"].pop("topicPolicyUnits"),
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


def test_malformed_response_audits_sdk_attempt_count_before_decision_parsing() -> None:
    malformed = response()
    malformed["action"] = "UNKNOWN"
    malformed["ResponseMetadata"]["RetryAttempts"] = 2

    class MalformedSDK:
        def apply_guardrail(self, **_kwargs: object) -> dict[str, Any]:
            return malformed

    execution = FakeExecution()
    with pytest.raises(GuardrailResponseError):
        _client(MalformedSDK(), execution, []).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert execution.calls[0]["response_data"].to_dict()["attempts"] == 3


def test_invalid_response_metadata_fails_closed_without_private_value_leak() -> None:
    marker = "PRIVATE_REQUEST_ID_MARKER_" * 20
    malformed = response()
    malformed["ResponseMetadata"]["RequestId"] = marker

    class MalformedSDK:
        def apply_guardrail(self, **_kwargs: object) -> dict[str, Any]:
            return malformed

    execution = FakeExecution()
    events: list[Any] = []
    with pytest.raises(GuardrailResponseError) as exc_info:
        _client(MalformedSDK(), execution, events).apply_guardrail(
            text="marker",
            source="INPUT",
            required_filters=PROMPT_FILTERS,
        )

    assert marker not in str(exc_info.value)
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
