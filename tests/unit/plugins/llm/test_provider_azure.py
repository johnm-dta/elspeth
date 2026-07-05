# tests/unit/plugins/llm/test_provider_azure.py
"""Tests for AzureLLMProvider."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.infrastructure.clients.llm import (
    AuditedLLMClient,
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    LLMResponse,
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider, LLMQueryResult, UnrecognizedFinishReason
from elspeth.plugins.transforms.llm.providers.azure import AzureLLMProvider


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


@dataclass
class FakeUnderlyingAzureClient:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class ChatCompletionCall:
    model: str
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int | None
    response_format: dict[str, Any] | None
    resolved_prompt_template_hash: str | None


@dataclass
class FakeLLMClient:
    response: LLMResponse | None = None
    error: Exception | None = None
    calls: list[ChatCompletionCall] = field(default_factory=list)

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            ChatCompletionCall(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                resolved_prompt_template_hash=resolved_prompt_template_hash,
            )
        )
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("FakeLLMClient.response must be set before chat_completion()")
        return self.response


def _make_provider(
    recorder: FakeAuditRecorder | None = None,
    telemetry_emit: FakeTelemetryEmit | None = None,
) -> AzureLLMProvider:
    return AzureLLMProvider(
        endpoint="https://test.openai.azure.com/",
        api_key="test-key",
        api_version="2024-10-21",
        deployment_name="gpt-4o",
        recorder=recorder if recorder is not None else FakeAuditRecorder(),
        run_id="run-1",
        telemetry_emit=telemetry_emit if telemetry_emit is not None else FakeTelemetryEmit(),
    )


@contextmanager
def _provider_llm_client(provider: AzureLLMProvider, client: FakeLLMClient) -> Iterator[None]:
    original_get = provider._get_llm_client

    def get_llm_client(state_id: str, *, token_id: str | None = None) -> FakeLLMClient:
        _ = state_id, token_id
        return client

    provider._get_llm_client = get_llm_client  # type: ignore[method-assign]
    try:
        yield
    finally:
        provider._get_llm_client = original_get  # type: ignore[method-assign]


@pytest.fixture()
def audit_recorder() -> FakeAuditRecorder:
    return FakeAuditRecorder()


@pytest.fixture()
def telemetry_emit() -> FakeTelemetryEmit:
    return FakeTelemetryEmit()


@pytest.fixture()
def provider(audit_recorder: FakeAuditRecorder, telemetry_emit: FakeTelemetryEmit) -> AzureLLMProvider:
    return _make_provider(audit_recorder, telemetry_emit)


def _make_llm_response(
    content: str = "Hello",
    model: str = "gpt-4o",
    finish_reason: str | None = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResponse:
    """Build a mock LLMResponse."""
    raw = {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return LLMResponse(
        content=content,
        model=model,
        usage=TokenUsage.known(prompt_tokens, completion_tokens),
        latency_ms=50.0,
        raw_response=raw,
    )


class TestExecuteQuery:
    """Tests for execute_query method."""

    def test_returns_llm_query_result(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(response=_make_llm_response())

        with _provider_llm_client(provider, client):
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

    def test_maps_finish_reason(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(response=_make_llm_response(finish_reason="stop"))

        with _provider_llm_client(provider, client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

        assert result.finish_reason is FinishReason.STOP

    def test_unknown_finish_reason_returns_unrecognized(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(response=_make_llm_response(finish_reason="end_turn"))

        with _provider_llm_client(provider, client):
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

    def test_propagates_rate_limit_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=RateLimitError("429 rate limited"))

        with _provider_llm_client(provider, client), pytest.raises(RateLimitError, match="429 rate limited"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_propagates_content_policy_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=ContentPolicyError("content_policy_violation"))

        with _provider_llm_client(provider, client), pytest.raises(ContentPolicyError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_propagates_server_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=ServerError("503 overloaded"))

        with _provider_llm_client(provider, client), pytest.raises(ServerError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_propagates_network_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=NetworkError("connection refused"))

        with _provider_llm_client(provider, client), pytest.raises(NetworkError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_propagates_llm_client_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=LLMClientError("bad request", retryable=False))

        with _provider_llm_client(provider, client), pytest.raises(LLMClientError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_propagates_context_length_error(self, provider: AzureLLMProvider) -> None:
        client = FakeLLMClient(error=ContextLengthError("context_length_exceeded"))

        with _provider_llm_client(provider, client), pytest.raises(ContextLengthError):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_execute_query_timeout_propagates_as_network_error(self, provider: AzureLLMProvider) -> None:
        """Timeout errors from AuditedLLMClient propagate as NetworkError (retryable)."""
        client = FakeLLMClient(error=NetworkError("Request timed out"))

        with _provider_llm_client(provider, client), pytest.raises(NetworkError, match="timed out"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_no_raw_response_still_works(self, provider: AzureLLMProvider) -> None:
        """finish_reason gracefully handles missing raw_response."""
        resp = LLMResponse(
            content="hi",
            model="gpt-4o",
            usage=TokenUsage.unknown(),
            raw_response=None,
        )
        client = FakeLLMClient(response=resp)

        with _provider_llm_client(provider, client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=None,
                state_id="state-1",
                token_id="tok-1",
            )

        assert result.finish_reason is None
        assert result.content == "hi"

    def test_empty_choices_returns_none_finish_reason(self, provider: AzureLLMProvider) -> None:
        """Empty choices list yields finish_reason=None without crashing."""
        resp = LLMResponse(
            content="hi",
            model="gpt-4o",
            usage=TokenUsage.unknown(),
            raw_response={"choices": []},
        )
        client = FakeLLMClient(response=resp)

        with _provider_llm_client(provider, client):
            result = provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=None,
                state_id="state-1",
                token_id="tok-1",
            )

        assert result.finish_reason is None
        assert result.content == "hi"

    def test_empty_content_raises_content_policy_error(self, provider: AzureLLMProvider) -> None:
        """Empty string content (from AuditedLLMClient's None→'' conversion)
        must raise ContentPolicyError, not ValueError from LLMQueryResult invariant."""
        client = FakeLLMClient(
            response=_make_llm_response(
                content="",
                finish_reason="content_filter",
            )
        )

        with _provider_llm_client(provider, client), pytest.raises(ContentPolicyError, match="empty content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_whitespace_only_content_raises_content_policy_error(self, provider: AzureLLMProvider) -> None:
        """Whitespace-only content from provider must raise ContentPolicyError."""
        client = FakeLLMClient(
            response=_make_llm_response(
                content="   ",
                finish_reason="stop",
            )
        )

        with _provider_llm_client(provider, client), pytest.raises(ContentPolicyError, match="empty content"):
            provider.execute_query(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o",
                temperature=0.0,
                max_tokens=100,
                state_id="state-1",
                token_id="tok-1",
            )

    def test_empty_content_with_tool_calls_finish_reason(self, provider: AzureLLMProvider) -> None:
        """Tool-call responses (content=None→'', finish_reason=tool_calls)
        must raise LLMClientError, not ValueError."""
        client = FakeLLMClient(
            response=_make_llm_response(
                content="",
                finish_reason="tool_calls",
            )
        )

        with _provider_llm_client(provider, client), pytest.raises(LLMClientError, match="tool_calls"):
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
        provider = _make_provider(audit_recorder, telemetry_emit)
        provider._underlying_client = FakeUnderlyingAzureClient()

        client1 = provider._get_llm_client("state-a", token_id="tok-1")
        client2 = provider._get_llm_client("state-a", token_id="tok-1")
        client3 = provider._get_llm_client("state-b", token_id="tok-2")

        assert client1 is client2  # Same state_id → same client
        assert client1 is not client3  # Different state_id → different client

    def test_concurrent_client_creation_same_state_id(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        """50 threads racing to create a client for the same state_id.
        Verify exactly one client instance created.
        """
        provider = _make_provider(audit_recorder, telemetry_emit)
        provider._underlying_client = FakeUnderlyingAzureClient()

        clients: list[AuditedLLMClient] = []
        collect_lock = threading.Lock()
        barrier = threading.Barrier(50)

        def create_client() -> None:
            barrier.wait()
            c = provider._get_llm_client("state-race", token_id="tok-1")
            with collect_lock:
                clients.append(c)

        threads = [threading.Thread(target=create_client) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 50 threads should have gotten the same client instance
        assert len(clients) == 50
        assert all(c is clients[0] for c in clients)

    def test_close_clears_clients(
        self,
        audit_recorder: FakeAuditRecorder,
        telemetry_emit: FakeTelemetryEmit,
    ) -> None:
        provider = _make_provider(audit_recorder, telemetry_emit)
        underlying_client = FakeUnderlyingAzureClient()
        provider._underlying_client = underlying_client

        provider._get_llm_client("state-1", token_id="tok-1")

        assert len(provider._llm_clients) == 1
        provider.close()
        assert len(provider._llm_clients) == 0
        assert provider._underlying_client is None
        assert underlying_client.closed


class TestProtocolCompliance:
    """Verify AzureLLMProvider satisfies LLMProvider protocol."""

    def test_satisfies_llm_provider_protocol(self) -> None:
        # LLMProvider is runtime_checkable
        provider = _make_provider()
        assert isinstance(provider, LLMProvider)
