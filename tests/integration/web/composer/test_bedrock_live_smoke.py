"""Explicit operator-selected smoke test for composer calls through Bedrock."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import pytest

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.web.composer.llm_response_parsing import build_llm_call_record
from elspeth.web.composer.service import _litellm_acompletion

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.timeout(60),
]


async def test_bedrock_live_smoke_uses_aws_default_credential_chain() -> None:
    if os.environ.get("ELSPETH_RUN_BEDROCK_LIVE") != "1":
        pytest.skip("set ELSPETH_RUN_BEDROCK_LIVE=1 to select the live Bedrock smoke test")

    model = os.environ.get("ELSPETH_BEDROCK_LIVE_TEST_MODEL")
    if model is None or not model.startswith("bedrock/"):
        pytest.fail("ELSPETH_BEDROCK_LIVE_TEST_MODEL must name a bedrock/ model")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        pytest.fail("AWS_REGION or AWS_DEFAULT_REGION is required for the live Bedrock smoke test")

    messages = [{"role": "user", "content": "Reply with exactly: Bedrock smoke passed."}]
    started_at = datetime.now(UTC)
    started_ns = time.monotonic_ns()
    response = await _litellm_acompletion(
        model=model,
        messages=messages,
        max_tokens=16,
        aws_region_name=region,
    )

    assert response.choices
    content = response.choices[0].message.content
    assert isinstance(content, str)
    assert content.strip()

    record = build_llm_call_record(
        model_requested=model,
        messages=messages,
        tools=None,
        status=ComposerLLMCallStatus.SUCCESS,
        started_at=started_at,
        started_ns=started_ns,
        temperature=None,
        seed=None,
        response=response,
    )
    assert record.prompt_tokens is not None
    assert record.completion_tokens is not None
    assert record.model_returned is not None
    assert record.provider_request_id is not None
    assert record.provider_cost_source in {
        "not_available",
        "response_usage.cost",
        "_hidden_params.response_cost",
    }
    record.to_dict()
