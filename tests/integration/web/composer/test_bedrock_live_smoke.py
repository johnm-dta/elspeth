"""Explicit operator-selected smoke test for Composer calls through Bedrock."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio,
    pytest.mark.timeout(60),
]


async def test_bedrock_live_smoke_uses_real_composer_service_and_default_credential_chain(tmp_path: Path) -> None:
    if os.environ.get("ELSPETH_RUN_BEDROCK_LIVE") != "1":
        pytest.skip("set ELSPETH_RUN_BEDROCK_LIVE=1 to select the live Bedrock smoke test")

    model = os.environ.get("ELSPETH_BEDROCK_LIVE_TEST_MODEL")
    if model is None or not model.startswith("bedrock/"):
        pytest.fail("ELSPETH_BEDROCK_LIVE_TEST_MODEL must name a bedrock/ model")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        pytest.fail("AWS_REGION or AWS_DEFAULT_REGION is required for the live Bedrock smoke test")
    advisor_model = os.environ.get("ELSPETH_BEDROCK_LIVE_ADVISOR_MODEL", "bedrock/global.anthropic.claude-opus-4-6-v1")
    if advisor_model == model:
        pytest.fail("ELSPETH_BEDROCK_LIVE_ADVISOR_MODEL must differ from the primary live model")

    settings = WebSettings(
        data_dir=tmp_path,
        composer_model=model,
        composer_advisor_model=advisor_model,
        composer_temperature=None,
        composer_seed=None,
        shareable_link_signing_key=b"\x00" * 32,
    )
    service = ComposerServiceImpl.for_trained_operator(catalog=create_catalog_service(), settings=settings)
    assert service.get_availability().available is True

    messages = [{"role": "user", "content": "Reply with exactly: Bedrock smoke passed."}]
    tools = service._get_litellm_tools()
    recorder = BufferingRecorder()
    response = await service._call_llm_with_audit(
        messages,
        tools,
        timeout=55.0,
        recorder=recorder,
    )

    assert response.choices
    record = recorder.llm_calls[-1]
    assert record.model_requested == model
    assert record.tools_sha256 is not None
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
