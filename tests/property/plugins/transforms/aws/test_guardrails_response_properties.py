from __future__ import annotations

from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.unit.plugins.transforms.aws.test_guardrails_client import PROMPT_FILTERS, FakeExecution, response

from elspeth.plugins.transforms.aws.guardrails_client import BedrockGuardrailsClient, GuardrailResponseError, parse_guardrail_response


@settings(max_examples=25)
@given(st.lists(st.text(min_size=0, max_size=1024), min_size=1, max_size=8))
def test_bounded_intervention_outputs_are_discarded(texts: list[str]) -> None:
    marked_texts = [f"PROVIDER_OUTPUT_MARKER_{index}_{text}" for index, text in enumerate(texts)]
    sdk: Any = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    provider_response = response(
        action="GUARDRAIL_INTERVENED",
        detected="PROMPT_ATTACK",
        blocked=True,
        outputs=[{"text": text} for text in marked_texts],
    )
    with Stubber(sdk) as stubber:
        stubber.add_response("apply_guardrail", provider_response)
        decision = BedrockGuardrailsClient(
            execution=FakeExecution(),
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=lambda _event: None,
            guardrail_identifier="privateguardrail",
            guardrail_version="7",
            region="us-east-1",
            sdk_client=sdk,
            audit_salt=b"0123456789abcdef0123456789abcdef",
        ).apply_guardrail(text="marker", source="INPUT", required_filters=PROMPT_FILTERS)

    assert decision.intervened is True
    for text in marked_texts:
        assert text not in repr(decision)


@settings(max_examples=30)
@given(st.text(min_size=1, max_size=16))
def test_oversized_provider_output_fails_without_text_leak(seed: str) -> None:
    text = (seed * (4097 // len(seed) + 1))[:4097]
    malformed = response(
        action="GUARDRAIL_INTERVENED",
        detected="PROMPT_ATTACK",
        blocked=True,
        outputs=[{"text": text}],
    )

    with pytest.raises(GuardrailResponseError) as exc_info:
        parse_guardrail_response(malformed, required_filters=PROMPT_FILTERS)

    assert text not in str(exc_info.value)
