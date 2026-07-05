# tests/unit/plugins/llm/test_provider_openrouter.py
"""Tests for OpenRouterLLMProvider."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest

from elspeth.plugins.infrastructure.clients.llm import (
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider, LLMQueryResult, UnrecognizedFinishReason
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterLLMProvider

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


@dataclass
class FakeAuditRecorder:
    call_indexes: list[int] = field(default_factory=list)
    allocated_state_ids: list[str | None] = field(default_factory=list)
    allocated_operation_ids: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    operation_calls: list[dict[str, Any]] = field(default_factory=list)

    def allocate_call_index(self, state_id: str | None) -> int:
        self.allocated_state_ids.append(state_id)
        if self.call_indexes:
            return self.call_indexes.pop(0)
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


@dataclass(frozen=True)
class HTTPPostCall:
    url: str
    kwargs: dict[str, Any]


@dataclass
class FakeHTTPClient:
    response: httpx.Response | None = None
    error: BaseException | None = None
    post_calls: list[HTTPPostCall] = field(default_factory=list)
    closed: bool = False

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.post_calls.append(HTTPPostCall(url=url, kwargs=dict(kwargs)))
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("FakeHTTPClient.response must be set before post()")
        return self.response

    def close(self) -> None:
        self.closed = True

    @property
    def last_post(self) -> HTTPPostCall:
        assert self.post_calls
        return self.post_calls[-1]


@contextmanager
def _provider_http_client(provider: OpenRouterLLMProvider, http_client: FakeHTTPClient) -> Iterator[None]:
    original_get = provider._get_http_client
    original_release = provider._release_http_client

    def get_http_client(state_id: str, *, token_id: str | None = None) -> FakeHTTPClient:
        _ = state_id, token_id
        return http_client

    def release_http_client(state_id: str) -> None:
        _ = state_id

    provider._get_http_client = get_http_client  # type: ignore[method-assign]
    provider._release_http_client = release_http_client  # type: ignore[method-assign]
    try:
        yield
    finally:
        provider._get_http_client = original_get  # type: ignore[method-assign]
        provider._release_http_client = original_release  # type: ignore[method-assign]


@pytest.fixture()
def audit_recorder() -> FakeAuditRecorder:
    return FakeAuditRecorder()


@pytest.fixture()
def telemetry_emit() -> FakeTelemetryEmit:
    return FakeTelemetryEmit()


@pytest.fixture()
def provider(audit_recorder: FakeAuditRecorder, telemetry_emit: FakeTelemetryEmit) -> OpenRouterLLMProvider:
    return OpenRouterLLMProvider(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=30.0,
        recorder=audit_recorder,
        run_id="run-1",
        telemetry_emit=telemetry_emit,
    )


def _make_http_response(
    content: str = "Hello",
    model: str = "gpt-4o",
    finish_reason: str | None = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    status_code: int = 200,
) -> httpx.Response:
    """Build a mock httpx.Response."""
    body = {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def _make_error_response(
    status_code: int,
    body: str = '{"error": "test"}',
    *,
    content_type: str = "application/json",
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=body.encode(),
        headers={"content-type": content_type},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


def _assert_provider_body_redacted(message: str, *forbidden_fragments: str) -> None:
    assert "provider error body redacted" in message
    assert "body_present=true" in message
    for fragment in forbidden_fragments:
        assert fragment not in message


def test_provider_accepts_loopback_http_base_url(
    audit_recorder: FakeAuditRecorder,
    telemetry_emit: FakeTelemetryEmit,
) -> None:
    # Loopback HTTP is permitted for local OpenAI-compatible dev servers (the
    # shipped ChaosLLM examples target http://127.0.0.1:8199/v1); the bearer
    # token never leaves the machine. Remote HTTP is still rejected.
    provider = OpenRouterLLMProvider(
        api_key="test-key",
        base_url="http://127.0.0.1:8199/v1",
        timeout_seconds=30.0,
        recorder=audit_recorder,
        run_id="run-1",
        telemetry_emit=telemetry_emit,
    )
    assert provider._base_url == "http://127.0.0.1:8199/v1"


def test_provider_rejects_remote_http_base_url(
    audit_recorder: FakeAuditRecorder,
    telemetry_emit: FakeTelemetryEmit,
) -> None:
    # The loopback exception must NOT widen to network-reachable hosts — a remote
    # HTTP base_url would send the bearer token over plaintext. Guards against the
    # allow_http_loopback exception being silently broadened.
    with pytest.raises(ValueError, match="base_url"):
        OpenRouterLLMProvider(
            api_key="test-key",
            base_url="http://api.example.com/v1",
            timeout_seconds=30.0,
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )


class TestExecuteQuery:
    """Tests for execute_query method."""

    def test_parses_json_response(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_http_response())
        with _provider_http_client(provider, http_client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

        assert isinstance(result, LLMQueryResult)
        assert result.content == "Hello"
        assert result.model == "gpt-4o"
        assert result.usage.is_known
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_max_tokens_none_omitted_from_request_body(self, provider: OpenRouterLLMProvider) -> None:
        """When max_tokens=None, it should NOT appear in the request body."""
        http_client = FakeHTTPClient(response=_make_http_response())
        with _provider_http_client(provider, http_client):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=None,
                state_id="state-1",
                token_id="tok-1",
            )

            # Verify the POST body does NOT contain max_tokens
            request_body = http_client.last_post.kwargs["json"]
            assert "max_tokens" not in request_body

    def test_rejects_nan_in_response(self, provider: OpenRouterLLMProvider) -> None:
        # NaN in JSON
        resp = httpx.Response(
            status_code=200,
            content=b'{"choices": [{"message": {"content": "hi"}}], "value": NaN}',
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(LLMClientError, match="not valid JSON"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_rejects_null_content(self, provider: OpenRouterLLMProvider) -> None:
        body = json.dumps(
            {
                "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                "model": "gpt-4o",
            }
        )
        resp = httpx.Response(
            status_code=200,
            content=body.encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(ContentPolicyError, match="null content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_rejects_non_string_content(self, provider: OpenRouterLLMProvider) -> None:
        body = json.dumps(
            {
                "choices": [{"message": {"content": 42}, "finish_reason": "stop"}],
                "model": "gpt-4o",
            }
        )
        resp = httpx.Response(
            status_code=200,
            content=body.encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(LLMClientError, match="Expected string content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_validates_usage_non_finite_via_json_parse(self, provider: OpenRouterLLMProvider) -> None:
        """Infinity in JSON is caught at the parse level by reject_nonfinite_constant."""
        # Infinity literal in JSON — caught by reject_nonfinite_constant during json.loads
        raw_json = b'{"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": Infinity}}'
        resp = httpx.Response(
            status_code=200,
            content=raw_json,
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(LLMClientError, match="not valid JSON"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_validates_usage_non_finite_float(self, provider: OpenRouterLLMProvider) -> None:
        """Non-finite float that survives JSON parsing (e.g., provider returns huge float)
        is caught by the post-parse usage validation."""
        http_client = FakeHTTPClient(response=_make_http_response())
        with _provider_http_client(provider, http_client):
            # Inject non-finite value after JSON parse
            import elspeth.plugins.transforms.llm.providers.openrouter as mod

            original_loads = json.loads

            def patched_loads(s: Any, **kwargs: Any) -> Any:
                result = original_loads(s, **kwargs)
                if isinstance(result, dict) and "usage" in result:
                    result["usage"]["prompt_tokens"] = float("inf")
                return result

            with (
                patch.object(mod.json, "loads", side_effect=patched_loads),  # type: ignore[attr-defined]
                pytest.raises(LLMClientError, match="Non-finite value in usage"),
            ):
                provider.execute_query(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                    temperature=0.0,
                    max_tokens=100,
                    state_id="state-1",
                    token_id="tok-1",
                )

    def test_empty_string_content_raises_content_policy_error(self, provider: OpenRouterLLMProvider) -> None:
        """Empty string content (not null) must raise ContentPolicyError,
        not ValueError from LLMQueryResult invariant."""
        body = json.dumps(
            {
                "choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            }
        )
        resp = httpx.Response(
            status_code=200,
            content=body.encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(ContentPolicyError, match="empty content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_whitespace_only_content_raises_content_policy_error(self, provider: OpenRouterLLMProvider) -> None:
        """Whitespace-only content must raise ContentPolicyError."""
        body = json.dumps(
            {
                "choices": [{"message": {"content": "   "}, "finish_reason": "stop"}],
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }
        )
        resp = httpx.Response(
            status_code=200,
            content=body.encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(ContentPolicyError, match="empty content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_unknown_finish_reason(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_http_response(finish_reason="end_turn"))
        with _provider_http_client(provider, http_client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

        assert isinstance(result.finish_reason, UnrecognizedFinishReason)
        assert result.finish_reason.raw == "end_turn"

    def test_maps_finish_reason_stop(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_http_response(finish_reason="stop"))
        with _provider_http_client(provider, http_client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

        assert result.finish_reason is FinishReason.STOP

    def test_sends_current_openrouter_app_attribution_headers(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            timeout_seconds=30.0,
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

        http_client = FakeHTTPClient(response=_make_http_response())
        with patch("elspeth.plugins.infrastructure.clients.http.httpx.Client", autospec=True) as client_class:
            client_class.return_value = http_client

            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

        request_headers = http_client.last_post.kwargs["headers"]
        assert request_headers["HTTP-Referer"] == "https://github.com/johnm-dta/elspeth"
        assert request_headers["X-OpenRouter-Title"] == "Elspeth"
        assert "elspeth-rapid" not in set(request_headers.values())


class TestHTTPErrorMapping:
    """Tests for HTTP status code → exception mapping."""

    def test_429_raises_rate_limit_error(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_error_response(429))
        with _provider_http_client(provider, http_client), pytest.raises(RateLimitError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_500_raises_server_error(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_error_response(500))
        with _provider_http_client(provider, http_client), pytest.raises(ServerError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_502_raises_server_error(self, provider: OpenRouterLLMProvider) -> None:
        """Verify 502 Bad Gateway maps to ServerError (5xx range, not just 500)."""
        http_client = FakeHTTPClient(response=_make_error_response(502))
        with _provider_http_client(provider, http_client), pytest.raises(ServerError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_503_raises_server_error(self, provider: OpenRouterLLMProvider) -> None:
        """Verify 503 Service Unavailable maps to ServerError."""
        http_client = FakeHTTPClient(response=_make_error_response(503))
        with _provider_http_client(provider, http_client), pytest.raises(ServerError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_network_error_raises_network_error(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(error=httpx.ConnectError("connection refused"))
        with _provider_http_client(provider, http_client), pytest.raises(NetworkError, match="Network error"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_timeout_raises_network_error(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(error=httpx.ReadTimeout("timed out"))
        with _provider_http_client(provider, http_client), pytest.raises(NetworkError, match="Network error"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_4xx_raises_llm_client_error(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(response=_make_error_response(400))
        with _provider_http_client(provider, http_client), pytest.raises(LLMClientError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_400_context_length_raises_context_length_error(self, provider: OpenRouterLLMProvider) -> None:
        provider_body = (
            '{"error": {"message": "This model\'s maximum context length is 8192 tokens", "api_key": "sk-or-v1-context-secret"}}'
        )
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body))
        with _provider_http_client(provider, http_client):
            with pytest.raises(ContextLengthError) as exc_info:
                provider.execute_query(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-4o",
                    temperature=0.0,
                    max_tokens=100,
                    state_id="state-1",
                    token_id="tok-1",
                )

            message = str(exc_info.value)
            assert "Context length exceeded" in message
            _assert_provider_body_redacted(
                message,
                "maximum context length",
                "api_key",
                "sk-or-v1-context-secret",
            )

    def test_400_context_length_exceeded_pattern(self, provider: OpenRouterLLMProvider) -> None:
        http_client = FakeHTTPClient(
            response=_make_error_response(
                400,
                body='{"error": {"code": "context_length_exceeded"}}',
            )
        )
        with _provider_http_client(provider, http_client), pytest.raises(ContextLengthError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_400_anthropic_prompt_too_long_pattern(self, provider: OpenRouterLLMProvider) -> None:
        # Anthropic via OpenRouter returns "prompt is too long: N tokens > M maximum"
        # (an envelope wrapping Anthropic's invalid_request_error). Previously
        # misclassified as a generic LLMClientError → audit reason llm_call_failed;
        # must classify as ContextLengthError → audit reason context_length_exceeded.
        anthropic_body = (
            '{"error":{"message":"Provider returned error","code":400,'
            '"metadata":{"raw":"{\\"type\\":\\"error\\",\\"error\\":{\\"type\\":'
            '\\"invalid_request_error\\",\\"message\\":\\"prompt is too long: '
            '202814 tokens > 200000 maximum\\"}}","provider_name":"Anthropic"}}}'
        )
        http_client = FakeHTTPClient(response=_make_error_response(400, body=anthropic_body))
        with _provider_http_client(provider, http_client), pytest.raises(ContextLengthError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="anthropic/claude-sonnet-4",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_empty_choices_raises(self, provider: OpenRouterLLMProvider) -> None:
        body = json.dumps({"choices": [], "model": "gpt-4o"})
        resp = httpx.Response(
            status_code=200,
            content=body.encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        )
        http_client = FakeHTTPClient(response=resp)
        with _provider_http_client(provider, http_client), pytest.raises(LLMClientError, match="Empty or missing choices"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )


class TestClientCaching:
    """Tests for client creation and caching."""

    def test_client_cached_per_state_id(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

        client1 = provider._get_http_client("state-a", token_id="tok-1")
        client2 = provider._get_http_client("state-a", token_id="tok-1")
        client3 = provider._get_http_client("state-b", token_id="tok-2")

        assert client1 is client2
        assert client1 is not client3

    def test_concurrent_client_creation_same_state_id(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

        clients: list[AuditedHTTPClient] = []
        collect_lock = threading.Lock()
        barrier = threading.Barrier(50)

        def create_client() -> None:
            barrier.wait()
            c = provider._get_http_client("state-race", token_id="tok-1")
            with collect_lock:
                clients.append(c)

        threads = [threading.Thread(target=create_client) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(clients) == 50
        assert all(c is clients[0] for c in clients)

    def test_close_clears_clients(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

        provider._get_http_client("state-1", token_id="tok-1")
        assert len(provider._http_clients) == 1

        provider.close()
        assert len(provider._http_clients) == 0


class TestProtocolCompliance:
    """Verify OpenRouterLLMProvider satisfies LLMProvider protocol."""

    def test_satisfies_llm_provider_protocol(self) -> None:
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            recorder=FakeAuditRecorder(),
            run_id="run-1",
            telemetry_emit=FakeTelemetryEmit(),
        )
        assert isinstance(provider, LLMProvider)


class TestRuntimePreflight:
    """Tests for runtime_preflight — provider warm-up probe before row processing.

    The preflight probe sends a minimal chat-completions request and asserts that
    the provider accepts it. Two contracts are pinned here because both have
    bitten us in production:

    1. max_tokens must clear the underlying-provider minimum. Azure-backed
       routes enforce max_output_tokens >= 16; values below floor 400 with
       "integer_below_min_value" and kill the entire pipeline before any
       row processing. Run 8294aab2 on 2026-05-13 failed this way.

    2. When the provider returns a 4xx/5xx, the wrapped exception's message
       must disclose that a provider error body existed without copying that
       remote body into persisted diagnostics. The full body remains in the
       audited HTTP payload, not web-visible exception text.
    """

    def test_preflight_request_max_tokens_meets_azure_floor(self, provider: OpenRouterLLMProvider) -> None:
        """Preflight request body must include max_tokens >= 16 (Azure backend floor)."""
        http_client = FakeHTTPClient(response=_make_http_response(content="OK"))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            request_body = http_client.last_post.kwargs["json"]
            assert "max_tokens" in request_body, (
                "Preflight must send max_tokens; Azure-backed OpenRouter routes reject requests without it."
            )
            assert request_body["max_tokens"] >= 16, (
                f"Preflight max_tokens={request_body['max_tokens']} is below the "
                "Azure backend floor of 16; will 400 with integer_below_min_value."
            )

    def test_preflight_400_redacts_response_body_in_exception(self, provider: OpenRouterLLMProvider) -> None:
        """A 4xx from the provider must not surface raw response-body text."""
        provider_body = (
            '{"error":{"message":"Invalid \'max_output_tokens\': integer below minimum '
            'value. Expected a value >= 16, but got 4 instead.",'
            '"authorization":"Bearer sk-or-v1-secret","code":400}}'
        )
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(LLMClientError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            assert "HTTP 400" in message
            assert f"chars={len(provider_body)}" in message
            _assert_provider_body_redacted(
                message,
                "max_output_tokens",
                "authorization",
                "sk-or-v1-secret",
            )

    def test_preflight_429_redacts_response_body_in_rate_limit_error(self, provider: OpenRouterLLMProvider) -> None:
        """A 429 from the provider must not surface raw response-body text."""
        provider_body = '{"error":{"message":"Rate limit exceeded: 60 RPM","api_key":"sk-or-v1-rate-secret","code":429}}'
        http_client = FakeHTTPClient(response=_make_error_response(429, body=provider_body))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(RateLimitError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            assert "Rate limited" in message
            _assert_provider_body_redacted(message, "60 RPM", "api_key", "sk-or-v1-rate-secret")

    def test_preflight_500_redacts_response_body_in_server_error(self, provider: OpenRouterLLMProvider) -> None:
        """A 5xx from the provider must not surface raw response-body text."""
        provider_body = (
            '{"error":{"message":"Upstream provider unavailable",'
            '"request_echo":"Authorization: Bearer sk-or-v1-upstream-secret","code":500}}'
        )
        http_client = FakeHTTPClient(response=_make_error_response(500, body=provider_body))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(ServerError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            assert "Server error (HTTP 500)" in message
            _assert_provider_body_redacted(
                message,
                "Upstream provider unavailable",
                "Authorization",
                "sk-or-v1-upstream-secret",
            )

    def test_execute_query_400_redacts_response_body(self, provider: OpenRouterLLMProvider) -> None:
        """The same redaction contract applies to per-row execute_query calls."""
        provider_body = '{"error":{"message":"Model openai/gpt-5-mini not found","token":"sk-or-v1-query-secret","code":400}}'
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body))
        with _provider_http_client(provider, http_client):
            with pytest.raises(LLMClientError) as exc_info:
                provider.execute_query(
                    messages=[{"role": "user", "content": "hi"}],
                    model="gpt-5-mini",
                    temperature=0.0,
                    max_tokens=100,
                    state_id="state-1",
                    token_id="tok-1",
                )

            message = str(exc_info.value)
            assert "HTTP 400" in message
            _assert_provider_body_redacted(
                message,
                "openai/gpt-5-mini not found",
                "token",
                "sk-or-v1-query-secret",
            )

    def test_preflight_redacts_oversized_response_body(self, provider: OpenRouterLLMProvider) -> None:
        """Pathologically large response bodies are summarized without body text."""
        huge_body = '{"error":"' + ("x" * 10_000) + '"}'
        http_client = FakeHTTPClient(response=_make_error_response(400, body=huge_body))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(LLMClientError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            assert len(message) < 600, (
                f"Exception message length {len(message)} exceeds reasonable bound; "
                "_summarize_http_error_body must not copy oversized bodies."
            )
            assert "chars=10012" in message
            assert "x" * 100 not in message

    def test_preflight_redacts_malformed_content_type_metadata(self, provider: OpenRouterLLMProvider) -> None:
        """Provider-controlled content-type metadata must not become a second diagnostic leak."""
        provider_body = '{"error":{"message":"invalid","code":400}}'
        content_type = "application/json authorization=Bearer sk-or-v1-header-secret"
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body, content_type=content_type))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(LLMClientError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            _assert_provider_body_redacted(message, "authorization", "sk-or-v1-header-secret")
            assert "content_type" not in message

    def test_preflight_redacts_valid_secret_bearing_content_type_metadata(
        self,
        provider: OpenRouterLLMProvider,
    ) -> None:
        """Even syntactically valid media types are provider-controlled and not diagnostic text."""
        provider_body = '{"error":{"message":"invalid","code":400}}'
        content_type = "application/sk-or-v1-header-secret"
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body, content_type=content_type))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(LLMClientError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            _assert_provider_body_redacted(message, "application/sk-or-v1-header-secret", "sk-or-v1-header-secret")
            assert "content_type" not in message

    def test_preflight_error_message_omits_request_url(self, audit_recorder: FakeAuditRecorder) -> None:
        """Configurable base URLs can carry query strings; exception text must not copy them."""
        provider = OpenRouterLLMProvider(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1?token=sk-or-v1-url-secret",
            timeout_seconds=30.0,
            recorder=audit_recorder,
            run_id="run-1",
            telemetry_emit=FakeTelemetryEmit(),
        )
        provider_body = '{"error":{"message":"invalid","code":400}}'
        http_client = FakeHTTPClient(response=_make_error_response(400, body=provider_body))
        with patch("elspeth.plugins.transforms.llm.providers.openrouter.AuditedHTTPClient", autospec=True) as client_cls:
            client_cls.return_value = http_client

            with pytest.raises(LLMClientError) as exc_info:
                provider.runtime_preflight(operation_id="op-1", model="gpt-4o")

            message = str(exc_info.value)
            assert "HTTP 400" in message
            _assert_provider_body_redacted(message, "sk-or-v1-url-secret", "openrouter.ai", "chat/completions")
