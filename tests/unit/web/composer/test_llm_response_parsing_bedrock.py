"""Bedrock-shaped LiteLLM response parsing tests."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.web.composer.llm_response_parsing import build_llm_call_record


def _response(*, response_cost: object = 0.01234) -> ModelResponse:
    usage = Usage(
        prompt_tokens=19,
        completion_tokens=5,
        total_tokens=24,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=7),
        cache_creation_input_tokens=11,
        cache_read_input_tokens=7,
    )
    response = ModelResponse(id="bedrock-request-123", model="bedrock/returned-model", usage=usage)
    response._hidden_params = {"response_cost": response_cost}
    return response


def _record(response: Any):
    return build_llm_call_record(
        model_requested="bedrock/requested-model",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        status=ComposerLLMCallStatus.SUCCESS,
        started_at=datetime.now(UTC),
        started_ns=time.monotonic_ns(),
        temperature=None,
        seed=None,
        response=response,
    )


def test_real_litellm_bedrock_response_reads_private_cost_and_preserves_metadata() -> None:
    response = _response()

    assert "_hidden_params" not in response.model_dump()

    record = _record(response)

    assert record.provider_cost == 0.01234
    assert record.provider_cost_source == "_hidden_params.response_cost"
    assert record.prompt_tokens == 19
    assert record.completion_tokens == 5
    assert record.total_tokens == 24
    assert record.cached_prompt_tokens is None
    assert record.cache_creation_input_tokens == 11
    assert record.cache_read_input_tokens == 7
    assert record.model_returned == "bedrock/returned-model"
    assert record.provider_request_id == "bedrock-request-123"


def test_usage_cost_takes_precedence_over_private_response_cost() -> None:
    response = _response(response_cost=0.01234)
    response.usage.cost = 0.0042

    record = _record(response)

    assert record.provider_cost == 0.0042
    assert record.provider_cost_source == "response_usage.cost"


def test_zero_private_response_cost_is_preserved() -> None:
    record = _record(_response(response_cost=0))

    assert record.provider_cost == 0.0
    assert record.provider_cost_source == "_hidden_params.response_cost"


@pytest.mark.parametrize("bad_cost", [True, "0.01", -0.01, math.nan, math.inf, -math.inf, None])
def test_malformed_private_response_cost_is_unavailable(bad_cost: object) -> None:
    record = _record(_response(response_cost=bad_cost))

    assert record.provider_cost is None
    assert record.provider_cost_source == "not_available"


@pytest.mark.parametrize("bad_usage_cost", [True, "0.01", -0.01, math.nan, math.inf, None])
def test_present_malformed_usage_cost_does_not_fall_back_to_private_cost(bad_usage_cost: object) -> None:
    response = _response(response_cost=0.01234)
    response.usage.cost = bad_usage_cost

    record = _record(response)

    assert record.provider_cost is None
    assert record.provider_cost_source == "not_available"


def test_private_cost_read_does_not_invoke_provider_named_property() -> None:
    class _Response:
        def __init__(self) -> None:
            self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            self.model = "bedrock/returned-model"
            self.id = "bedrock-request-123"
            self.__pydantic_private__ = {"_hidden_params": {"response_cost": 0.75}}

        @property
        def _hidden_params(self) -> object:
            raise AssertionError("provider property must not be invoked")

    record = _record(_Response())

    assert record.provider_cost == 0.75
    assert record.provider_cost_source == "_hidden_params.response_cost"
