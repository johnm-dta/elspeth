"""Tests for the LiteLLM-backed AWS Bedrock pipeline provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from litellm.exceptions import (
    ContextWindowExceededError,
    ServiceUnavailableError,
)
from litellm.exceptions import (
    RateLimitError as LiteLLMRateLimitError,
)
from litellm.exceptions import (
    Timeout as LiteLLMTimeout,
)
from litellm.types.utils import ModelResponse, Usage

from elspeth.plugins.infrastructure.clients.llm import (
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider
from elspeth.plugins.transforms.llm.providers.bedrock import BedrockConfig, BedrockLLMProvider

MODEL = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
DYNAMIC_SCHEMA = {"mode": "observed"}


@dataclass
class FakeAuditRecorder:
    allocated_state_ids: list[str | None] = field(default_factory=list)
    allocated_operation_ids: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    operation_calls: list[dict[str, Any]] = field(default_factory=list)

    def allocate_call_index(self, state_id: str | None) -> int:
        self.allocated_state_ids.append(state_id)
        return len(self.allocated_state_ids) - 1

    def allocate_operation_call_index(self, operation_id: str) -> int:
        self.allocated_operation_ids.append(operation_id)
        return len(self.allocated_operation_ids) - 1

    def record_call(self, **call: Any) -> SimpleNamespace:
        self.calls.append(call)
        return SimpleNamespace(request_ref=f"request-{len(self.calls)}", response_ref=f"response-{len(self.calls)}")

    def record_operation_call(self, **call: Any) -> SimpleNamespace:
        self.operation_calls.append(call)
        return SimpleNamespace(
            request_ref=f"operation-request-{len(self.operation_calls)}",
            response_ref=f"operation-response-{len(self.operation_calls)}",
        )


@dataclass
class FakeTelemetryEmit:
    events: list[Any] = field(default_factory=list)

    def __call__(self, event: Any) -> None:
        self.events.append(event)


def _config(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "provider": "bedrock",
        "model": MODEL,
        "prompt_template": "Classify: {{ row.text }}",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": ["text"],
    }
    raw.update(overrides)
    return raw


def _provider(
    *,
    region_name: str | None = None,
    recorder: FakeAuditRecorder | None = None,
    telemetry: FakeTelemetryEmit | None = None,
) -> BedrockLLMProvider:
    return BedrockLLMProvider(
        region_name=region_name,
        recorder=recorder if recorder is not None else FakeAuditRecorder(),
        run_id="run-1",
        telemetry_emit=telemetry if telemetry is not None else FakeTelemetryEmit(),
    )


def _response(*, content: str = "Hello", finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        model=MODEL,
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


def _execute(provider: BedrockLLMProvider) -> Any:
    return provider.execute_query(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
        temperature=0.25,
        max_tokens=64,
        state_id="state-1",
        token_id="token-1",
    )


class TestBedrockConfig:
    def test_valid_config_uses_default_credential_chain(self) -> None:
        config = BedrockConfig.from_dict(_config())

        assert config.provider == "bedrock"
        assert config.model == MODEL
        assert config.region_name is None
        assert "api_key" not in BedrockConfig.model_fields
        assert "aws_access_key_id" not in BedrockConfig.model_fields
        assert "aws_secret_access_key" not in BedrockConfig.model_fields

    def test_explicit_region_is_accepted(self) -> None:
        assert BedrockConfig.from_dict(_config(region_name="us-gov-west-1")).region_name == "us-gov-west-1"

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-v2",
            "bedrock/",
            " bedrock/model",
            "bedrock/model ",
            "bedrock/model\n",
            "bedrock/\x7fmodel",
        ],
    )
    def test_invalid_model_identifier_is_rejected(self, model: str) -> None:
        with pytest.raises(PluginConfigError, match="model"):
            BedrockConfig.from_dict(_config(model=model))

    def test_model_bound_accepts_512_characters(self) -> None:
        model = "bedrock/" + "x" * (512 - len("bedrock/"))

        assert BedrockConfig.from_dict(_config(model=model)).model == model

    def test_model_bound_rejects_513_characters(self) -> None:
        model = "bedrock/" + "x" * (513 - len("bedrock/"))

        with pytest.raises(PluginConfigError, match="model"):
            BedrockConfig.from_dict(_config(model=model))

    @pytest.mark.parametrize("region_name", ["", "US-EAST-1", "us_east_1", "x" * 65])
    def test_invalid_region_is_rejected(self, region_name: str) -> None:
        with pytest.raises(PluginConfigError, match="region_name"):
            BedrockConfig.from_dict(_config(region_name=region_name))

    @pytest.mark.parametrize(
        "field_name",
        [
            "api_key",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "profile",
            "profile_name",
            "role",
            "role_arn",
            "endpoint",
            "endpoint_url",
        ],
    )
    def test_credential_and_endpoint_fields_are_rejected(self, field_name: str) -> None:
        with pytest.raises(PluginConfigError, match=field_name):
            BedrockConfig.from_dict(_config(**{field_name: "forbidden"}))


class TestBedrockAdapter:
    @pytest.mark.parametrize("region_name", [None, "ap-southeast-2"])
    def test_forwards_only_normal_request_fields_and_optional_region(self, region_name: str | None) -> None:
        provider = _provider(region_name=region_name)
        response = _response()

        with patch("litellm.completion", return_value=response) as completion:
            result = _execute(provider)

        kwargs = completion.call_args.kwargs
        assert kwargs["model"] == MODEL
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["temperature"] == 0.25
        assert kwargs["max_tokens"] == 64
        if region_name is None:
            assert "aws_region_name" not in kwargs
        else:
            assert kwargs["aws_region_name"] == region_name
        for forbidden in (
            "api_key",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_profile_name",
            "aws_role_name",
            "endpoint_url",
        ):
            assert forbidden not in kwargs
        assert result.content == "Hello"

    def test_region_does_not_override_an_explicit_call_kwarg(self) -> None:
        from elspeth.plugins.transforms.llm.providers.bedrock import _LiteLLMSDKAdapter

        adapter = _LiteLLMSDKAdapter(region_name="ap-southeast-2")
        with patch("litellm.completion", return_value=_response()) as completion:
            adapter.create(model=MODEL, messages=[], aws_region_name="us-east-1")

        assert completion.call_args.kwargs["aws_region_name"] == "us-east-1"


class TestBedrockProvider:
    def test_satisfies_llm_provider_protocol(self) -> None:
        assert isinstance(_provider(), LLMProvider)

    def test_execute_query_returns_normalized_result(self) -> None:
        with patch("litellm.completion", return_value=_response()):
            result = _execute(_provider())

        assert result.content == "Hello"
        assert result.model == MODEL
        assert result.usage.prompt_tokens == 11
        assert result.usage.completion_tokens == 7
        assert result.finish_reason is FinishReason.STOP

    def test_execute_query_content_filter_raises_content_policy_error(self) -> None:
        with (
            patch("litellm.completion", return_value=_response(content="", finish_reason="content_filter")),
            pytest.raises(ContentPolicyError, match="empty content"),
        ):
            _execute(_provider())

    @pytest.mark.parametrize(
        ("provider_error", "expected_type", "retryable"),
        [
            (
                LiteLLMRateLimitError(
                    message="BedrockException: Rate Limit Error - ThrottlingException: Rate exceeded",
                    llm_provider="bedrock",
                    model=MODEL,
                ),
                RateLimitError,
                True,
            ),
            (
                ContextWindowExceededError(
                    message="BedrockException: Context Window Error - Input is too long for requested model.",
                    model=MODEL,
                    llm_provider="bedrock",
                ),
                ContextLengthError,
                False,
            ),
            (
                ServiceUnavailableError(
                    message="BedrockException - Internal server error",
                    llm_provider="bedrock",
                    model=MODEL,
                    response=httpx.Response(
                        status_code=500,
                        request=httpx.Request("POST", "https://example.invalid/"),
                    ),
                ),
                ServerError,
                True,
            ),
            (
                LiteLLMTimeout(
                    message="BedrockException: Timeout Error - Connect timeout on endpoint URL",
                    model=MODEL,
                    llm_provider="bedrock",
                ),
                NetworkError,
                True,
            ),
        ],
    )
    def test_execute_query_preserves_typed_category_but_redacts_provider_detail(
        self,
        provider_error: Exception,
        expected_type: type[LLMClientError],
        retryable: bool,
    ) -> None:
        recorder = FakeAuditRecorder()
        provider = _provider(recorder=recorder)

        with patch("litellm.completion", side_effect=provider_error), pytest.raises(expected_type) as exc_info:
            _execute(provider)

        escaping = exc_info.value
        assert str(escaping) == "Bedrock LLM request failed"
        assert repr(escaping) == f"{expected_type.__name__}('Bedrock LLM request failed')"
        assert escaping.retryable is retryable
        assert escaping.__cause__ is None
        assert escaping.__context__ is None
        assert recorder.calls[0]["error"].message == str(provider_error)
        assert str(provider_error) not in str(escaping)

    def test_runtime_preflight_redacts_raw_error_and_preserves_retryability(self) -> None:
        sentinel = "arn:aws:bedrock:ap-southeast-2:123456789012:raw-sentinel-request-id"
        recorder = FakeAuditRecorder()
        provider = _provider(recorder=recorder)
        provider_error = LiteLLMRateLimitError(
            message=f"BedrockException: Rate Limit Error - {sentinel}",
            llm_provider="bedrock",
            model=MODEL,
        )

        with patch("litellm.completion", side_effect=provider_error), pytest.raises(RateLimitError) as exc_info:
            provider.runtime_preflight(operation_id="operation-1", model=MODEL)

        escaping = exc_info.value
        assert str(escaping) == "Bedrock LLM request failed"
        assert sentinel not in str(escaping)
        assert sentinel not in repr(escaping)
        assert escaping.retryable is True
        assert escaping.__cause__ is None
        assert escaping.__context__ is None
        assert sentinel in recorder.operation_calls[0]["error"].message

    def test_close_closes_underlying_adapter_once(self) -> None:
        provider = _provider()
        adapter = provider._get_underlying_client()
        close = Mock(wraps=adapter.close)
        adapter.close = close

        provider.close()
        provider.close()

        close.assert_called_once_with()
