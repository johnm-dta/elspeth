# tests/plugins/clients/test_audited_llm_client.py
"""Tests for AuditedLLMClient."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.contracts.token_usage import TokenUsage
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.plugins.infrastructure.clients.llm import (
    AuditedLLMClient,
    ContentPolicyError,
    LLMClientError,
    LLMResponse,
    RateLimitError,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self) -> None:
        """LLMResponse holds response data."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            usage=TokenUsage.known(10, 5),
            latency_ms=150.0,
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.usage == TokenUsage.known(10, 5)
        assert response.latency_ms == 150.0

    def test_total_tokens_property(self) -> None:
        """total_tokens sums prompt and completion tokens."""
        response = LLMResponse(
            content="test",
            model="gpt-4",
            usage=TokenUsage.known(10, 5),
        )

        assert response.total_tokens == 15

    def test_total_tokens_with_unknown_usage(self) -> None:
        """total_tokens returns None when usage is unknown."""
        response = LLMResponse(
            content="test",
            model="gpt-4",
            usage=TokenUsage.unknown(),
        )

        assert response.total_tokens is None

    def test_default_values(self) -> None:
        """LLMResponse has sensible defaults."""
        response = LLMResponse(content="test", model="gpt-4")

        assert response.usage == TokenUsage.unknown()
        assert response.latency_ms == 0.0
        assert response.raw_response is None


class TestLLMClientErrors:
    """Tests for LLM client exceptions."""

    def test_llm_client_error_default_not_retryable(self) -> None:
        """LLMClientError is not retryable by default."""
        error = LLMClientError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.retryable is False

    def test_llm_client_error_retryable_flag(self) -> None:
        """LLMClientError can be marked retryable."""
        error = LLMClientError("Temporary issue", retryable=True)

        assert error.retryable is True

    def test_rate_limit_error_always_retryable(self) -> None:
        """RateLimitError is always retryable."""
        error = RateLimitError("Rate limit exceeded")

        assert str(error) == "Rate limit exceeded"
        assert error.retryable is True


class TestAuditedLLMClient:
    """Tests for AuditedLLMClient."""

    def _create_mock_execution(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        import itertools

        execution = MagicMock(spec=ExecutionRepository)
        execution.record_call = MagicMock()
        counter = itertools.count()
        execution.allocate_call_index.side_effect = lambda _: next(counter)
        return execution

    def _create_mock_openai_client(
        self,
        *,
        content: str = "Hello!",
        model: str = "gpt-4",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> MagicMock:
        """Create a mock OpenAI client."""
        # Build the nested response structure
        message = Mock()
        message.content = content

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens

        response = Mock()
        response.choices = [choice]
        response.model = model
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_123"})

        client = MagicMock()
        client.chat.completions.create.return_value = response

        return client

    def test_successful_call_records_to_audit_trail(self) -> None:
        """Successful LLM call is recorded to audit trail."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
            provider="openai",
        )

        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify response
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.usage == TokenUsage.known(10, 5)
        assert response.latency_ms > 0

        # Verify audit record
        execution.record_call.assert_called_once()
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["state_id"] == "state_123"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] == CallType.LLM
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["request_data"].to_dict()["model"] == "gpt-4"
        assert call_kwargs["request_data"].to_dict()["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["response_data"].to_dict()["content"] == "Hello!"
        assert call_kwargs["latency_ms"] > 0

    def test_telemetry_emits_token_id_when_configured(self) -> None:
        """Telemetry event should carry token_id when provided."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()
        emitted_events: list[ExternalCallCompleted] = []

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: emitted_events.append(event),
            underlying_client=openai_client,
            token_id="tok-123",
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "tok-123"

    def test_telemetry_uses_none_token_id_when_unset(self) -> None:
        """Telemetry event should allow token_id=None when not provided."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()
        emitted_events: list[ExternalCallCompleted] = []

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: emitted_events.append(event),
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id is None

    def test_call_index_increments(self) -> None:
        """Each call gets a unique, incrementing call index."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        # Make multiple calls
        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "First"}],
        )
        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Second"}],
        )

        # Check call indices
        calls = execution.record_call.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["call_index"] == 0
        assert calls[1][1]["call_index"] == 1

    def test_failed_call_records_error(self) -> None:
        """Failed LLM call records error details."""
        execution = self._create_mock_execution()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("API connection failed")

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match="API connection failed"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify error was recorded
        execution.record_call.assert_called_once()
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "Exception"
        assert "API connection failed" in call_kwargs["error"].message
        assert call_kwargs["error"].retryable is False

    def test_rate_limit_error_marked_retryable(self) -> None:
        """Rate limit errors are marked as retryable."""
        execution = self._create_mock_execution()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("Rate limit exceeded (429)")

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(RateLimitError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify error was recorded with retryable=True
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["error"].retryable is True

    def test_rate_limit_detected_by_keyword(self) -> None:
        """Rate limit is detected by 'rate' keyword in error."""
        execution = self._create_mock_execution()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("You have exceeded your rate limit")

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(RateLimitError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_non_rate_substring_does_not_raise_rate_limit_error(self) -> None:
        """Errors like 'enumerate' should not be misclassified as rate limit."""
        execution = self._create_mock_execution()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("400 Bad Request: enumerate at least one item")

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert type(exc_info.value) is LLMClientError
        assert exc_info.value.retryable is False
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["error"].retryable is False

    def test_temperature_and_max_tokens_recorded(self) -> None:
        """Temperature and max_tokens are recorded in request."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["temperature"] == 0.7
        assert call_kwargs["request_data"].to_dict()["max_tokens"] == 100

    def test_provider_recorded_in_request(self) -> None:
        """Provider name is recorded in request data."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
            provider="azure",
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["provider"] == "azure"

    def test_extra_kwargs_passed_to_client(self) -> None:
        """Extra kwargs are passed to underlying client and recorded."""
        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            top_p=0.9,
            presence_penalty=0.5,
        )

        # Verify extra kwargs were passed to underlying client
        openai_client.chat.completions.create.assert_called_once()
        create_kwargs = openai_client.chat.completions.create.call_args[1]
        assert create_kwargs["top_p"] == 0.9
        assert create_kwargs["presence_penalty"] == 0.5

        # Verify extra kwargs were recorded
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["top_p"] == 0.9
        assert call_kwargs["request_data"].to_dict()["presence_penalty"] == 0.5

    # NOTE: test_response_without_model_dump was removed because we require
    # openai>=2.15 which guarantees model_dump() exists on all responses.
    # Per CLAUDE.md "No Legacy Code Policy" - no backwards compatibility code.

    def test_null_content_raises_content_policy_error(self) -> None:
        """None content from LLM raises ContentPolicyError — not fabricated to "".

        The OpenAI API returns content=None when the model refuses (content
        filtering). Previously this was fabricated to "" via `or ""`, hiding
        the refusal from the audit trail.
        """
        execution = self._create_mock_execution()

        message = Mock()
        message.content = None  # Content-filtered response

        choice = Mock()
        choice.message = message
        choice.finish_reason = "content_filter"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(ContentPolicyError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_full_raw_response_recorded_in_audit_trail(self) -> None:
        """Full raw_response from model_dump() is recorded in audit trail.

        This ensures audit completeness per CLAUDE.md:
        "External calls - Full request AND response recorded"
        """
        execution = self._create_mock_execution()

        # Create a realistic OpenAI response with all fields
        full_response = {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699500000,
            "model": "gpt-4-0613",
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "tool_calls": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        message = Mock()
        message.content = "Hello!"

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4-0613"
        response.usage = usage
        response.model_dump = Mock(return_value=full_response)

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify raw_response is recorded in audit trail
        call_kwargs = execution.record_call.call_args[1]
        response_data = call_kwargs["response_data"].to_dict()

        # Summary fields still present for convenience
        assert response_data["content"] == "Hello!"
        assert response_data["model"] == "gpt-4-0613"
        assert response_data["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}

        # Full raw_response now also recorded
        assert response_data["raw_response"] == full_response
        assert response_data["raw_response"]["id"] == "chatcmpl-abc123"
        assert response_data["raw_response"]["system_fingerprint"] == "fp_44709d6fcb"
        assert response_data["raw_response"]["choices"][0]["finish_reason"] == "stop"

    def test_multiple_choices_preserved_in_raw_response(self) -> None:
        """Multiple choices (n>1) are preserved in raw_response."""
        execution = self._create_mock_execution()

        # Response with multiple choices
        full_response = {
            "id": "chatcmpl-multi",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Option A"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Option B"},
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "message": {"role": "assistant", "content": "Option C"},
                    "finish_reason": "length",
                },
            ],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40},
        }

        message = Mock()
        message.content = "Option A"  # First choice extracted

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 30

        response = Mock()
        response.choices = [choice]  # Only first choice for content extraction
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value=full_response)

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Give me options"}],
        )

        # Verify all choices are preserved in raw_response
        call_kwargs = execution.record_call.call_args[1]
        raw_response = call_kwargs["response_data"].to_dict()["raw_response"]

        assert len(raw_response["choices"]) == 3
        assert raw_response["choices"][0]["message"]["content"] == "Option A"
        assert raw_response["choices"][1]["message"]["content"] == "Option B"
        assert raw_response["choices"][2]["message"]["content"] == "Option C"
        assert raw_response["choices"][2]["finish_reason"] == "length"

    def test_tool_calls_raises_error_and_records_audit(self) -> None:
        """Tool calls response raises LLMClientError and records ERROR in audit trail.

        ELSPETH does not support tool_calls — a provider returning tool_calls
        must be recorded as ERROR (not fabricated to SUCCESS with content="").
        The raw response is preserved so the audit trail shows what the provider
        actually returned.
        """
        execution = self._create_mock_execution()

        full_response = {
            "id": "chatcmpl-tools",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        }

        message = Mock()
        message.content = None

        choice = Mock()
        choice.message = message
        choice.finish_reason = "tool_calls"

        usage = Mock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 20

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value=full_response)

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match=r"tool_calls response.*not supported"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "What's the weather?"}],
            )

        # Verify ERROR call recorded in audit trail (not dropped)
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "UnsupportedResponseError"

        # Verify raw response preserved for audit
        raw_response = call_kwargs["response_data"].to_dict()["raw_response"]
        assert raw_response["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = raw_response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

    # NOTE: test_raw_response_none_when_model_dump_unavailable was removed because
    # we require openai>=2.15 which guarantees model_dump() exists on all responses.
    # Per CLAUDE.md "No Legacy Code Policy" - no backwards compatibility code.

    def test_successful_call_with_missing_usage(self) -> None:
        """LLM call succeeds when provider returns no usage data.

        Some LLM providers/modes (streaming, certain Azure configs) omit
        usage data entirely. The client should record this as SUCCESS
        with an empty usage dict, not crash with AttributeError.
        """
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Hello, I'm working!"

        choice = Mock()
        choice.message = message

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = None  # Provider omits usage data
        response.model_dump = Mock(return_value={"id": "resp_no_usage"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Call should succeed (not crash)
        assert result.content == "Hello, I'm working!"
        assert result.model == "gpt-4"
        assert result.usage == TokenUsage.unknown()  # Unknown, not crash

        # Audit trail should record SUCCESS with empty usage
        execution.record_call.assert_called_once()
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["response_data"].to_dict()["content"] == "Hello, I'm working!"
        assert call_kwargs["response_data"].to_dict()["usage"] == {}

    def test_success_path_preserves_aggregate_only_usage(self) -> None:
        """Aggregate-only provider usage must not crash before audit recording."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Hello, aggregate usage!"

        choice = Mock()
        choice.message = message
        choice.finish_reason = "stop"

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = SimpleNamespace(total_tokens=15)
        response.model_dump = Mock(return_value={"id": "resp_total_only", "usage": {"total_tokens": 15}})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_aggregate",
            run_id="run_aggregate",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result.content == "Hello, aggregate usage!"
        assert result.usage.prompt_tokens is None
        assert result.usage.completion_tokens is None
        assert result.usage.total_tokens == 15

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["response_data"].to_dict()["usage"] == {"total_tokens": 15}

    def test_empty_choices_preserves_aggregate_only_usage_in_error_record(self) -> None:
        """Malformed responses with total-only usage must still record the call."""
        execution = self._create_mock_execution()

        response = Mock()
        response.choices = []
        response.model = "gpt-4"
        response.usage = SimpleNamespace(total_tokens=15)
        response.model_dump = Mock(return_value={"id": "resp_empty_total_only", "usage": {"total_tokens": 15}})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_empty_choices",
            run_id="run_empty_choices",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match="empty choices"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["response_data"].to_dict()["usage"] == {"total_tokens": 15}

    def test_invalid_provider_model_records_error_with_raw_response(self) -> None:
        """Missing provider model must be rejected before success is recorded."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Hello!"

        choice = Mock()
        choice.message = message
        choice.finish_reason = "stop"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        response = Mock()
        response.choices = [choice]
        response.model = None
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_bad_model", "model": None})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        emitted_events: list[ExternalCallCompleted] = []
        client = AuditedLLMClient(
            execution=execution,
            state_id="state_bad_model",
            run_id="run_bad_model",
            telemetry_emit=lambda event: emitted_events.append(event),
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match="model"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "MalformedResponseError"
        assert "model" in call_kwargs["error"].message.lower()
        assert call_kwargs["response_data"].to_dict()["model"] is None

        assert len(emitted_events) == 1
        assert emitted_events[0].status == CallStatus.ERROR
        assert emitted_events[0].response_payload is not None

    def test_non_string_content_records_raw_response_payload(self) -> None:
        """Non-string content must preserve the provider payload for audit."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = ["part1", "part2"]

        choice = Mock()
        choice.message = message
        choice.finish_reason = "stop"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        raw_response = {
            "id": "resp_non_string_content",
            "choices": [{"message": {"content": ["part1", "part2"]}, "finish_reason": "stop"}],
        }

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value=raw_response)

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_non_string_content",
            run_id="run_non_string_content",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match="expected str"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["response_data"] is not None
        assert call_kwargs["response_data"].to_dict()["choices"][0]["message"]["content"] == ["part1", "part2"]


class TestBug4_6_SuccessPathOutsideTryExcept:
    """Bug 4.6: Internal processing errors in success path crash directly.

    Previously, the success path (content extraction, usage building,
    audit recording) was inside the same try/except that caught SDK errors.
    This meant an AttributeError in our code would be misclassified as an
    LLMClientError. Now the success path is OUTSIDE the try/except block.
    """

    @staticmethod
    def _create_mock_execution() -> Mock:
        execution = Mock(spec=ExecutionRepository)
        execution.allocate_call_index = Mock(return_value=0)
        execution.record_call = Mock()
        return execution

    def test_internal_error_in_success_path_crashes_directly(self) -> None:
        """Bug in success processing crashes as AttributeError, not LLMClientError.

        If response.choices[0].message has no 'content' attribute (simulating
        an internal processing bug), it should raise AttributeError directly,
        NOT get caught and wrapped as LLMClientError.
        """
        execution = self._create_mock_execution()

        # Create a response where .choices[0].message.content raises AttributeError
        # This simulates a bug in our success path processing
        message = Mock(spec=[])  # Empty spec means no attributes at all
        choice = Mock()
        choice.message = message  # message.content will raise AttributeError

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = Mock()
        response.usage.prompt_tokens = 10
        response.usage.completion_tokens = 5
        response.model_dump = Mock(return_value={"id": "resp_test"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_bug46",
            run_id="run_bug46",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        # Should raise AttributeError directly (not LLMClientError)
        with pytest.raises(AttributeError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )


class TestContentFabrication:
    """Null content from LLM must raise, not fabricate empty string.

    The OpenAI API returns content=None when the model refuses to respond
    (content filtering). `content or ""` conflates None (refused) with ""
    (legitimately empty), fabricating data in the audit trail.
    """

    @staticmethod
    def _create_mock_execution() -> Mock:
        execution = Mock(spec=ExecutionRepository)
        execution.allocate_call_index = Mock(return_value=0)
        execution.record_call = Mock()
        return execution

    @staticmethod
    def _create_client(execution: Mock, openai_client: MagicMock) -> AuditedLLMClient:
        return AuditedLLMClient(
            execution=execution,
            state_id="state_fab",
            run_id="run_fab",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

    def test_null_content_raises_content_policy_error(self) -> None:
        """content=None from LLM must raise ContentPolicyError, not fabricate ""."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = None  # Content-filtered response

        choice = Mock()
        choice.message = message
        choice.finish_reason = "content_filter"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_null"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = self._create_client(execution, openai_client)

        with pytest.raises(ContentPolicyError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_null_content_records_call_before_raising(self) -> None:
        """Null-content responses must be recorded in audit trail before raising.

        The LLM call actually happened — the provider responded with content=None
        (content-filtered). The audit trail must record this call so operators can
        see why the call failed and there are no unexplained call-index gaps.
        """
        execution = self._create_mock_execution()

        message = Mock()
        message.content = None

        choice = Mock()
        choice.message = message
        choice.finish_reason = "content_filter"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_null", "choices": [{"finish_reason": "content_filter"}]})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = self._create_client(execution, openai_client)

        with pytest.raises(ContentPolicyError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # The call MUST be recorded despite raising ContentPolicyError
        execution.record_call.assert_called_once()
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["state_id"] == "state_fab"
        assert call_kwargs["call_type"] == CallType.LLM
        # Error should describe the content policy issue
        assert call_kwargs["error"] is not None
        assert "null content" in call_kwargs["error"].message.lower() or "content-filtered" in call_kwargs["error"].message.lower()
        assert call_kwargs["error"].retryable is False
        # Response data should still be recorded for audit completeness
        assert call_kwargs["response_data"] is not None

    def test_null_content_emits_telemetry_before_raising(self) -> None:
        """Null-content responses must emit ExternalCallCompleted telemetry.

        The audit trail records the call, but telemetry dashboards also need
        the event to avoid undercounting content-filtered failures.
        """
        execution = self._create_mock_execution()

        message = Mock()
        message.content = None

        choice = Mock()
        choice.message = message
        choice.finish_reason = "content_filter"

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_null", "choices": [{"finish_reason": "content_filter"}]})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        emitted_events: list[ExternalCallCompleted] = []
        client = AuditedLLMClient(
            execution=execution,
            state_id="state_fab",
            run_id="run_fab",
            telemetry_emit=lambda event: emitted_events.append(event),
            underlying_client=openai_client,
        )

        with pytest.raises(ContentPolicyError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.status == CallStatus.ERROR
        assert event.call_type == CallType.LLM
        assert event.state_id == "state_fab"
        assert event.run_id == "run_fab"
        # Unlike SDK errors, null-content has response data (HTTP call succeeded)
        assert event.response_hash is not None
        assert event.response_payload is not None
        assert event.token_usage is not None

    def test_empty_choices_raises_error_and_records_audit(self) -> None:
        """Empty choices array must raise AND record ERROR in audit trail.

        The LLM call happened (consuming call_index and tokens), so the
        audit trail must reflect it — raising without recording creates
        an unexplained audit gap.
        """
        execution = self._create_mock_execution()

        response = Mock()
        response.choices = []  # No choices at all
        response.model = "gpt-4"
        response.usage = Mock(prompt_tokens=10, completion_tokens=0)
        response.model_dump = Mock(return_value={"id": "resp_empty"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = self._create_client(execution, openai_client)

        with pytest.raises(LLMClientError, match="empty choices"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify ERROR call recorded in audit trail (not dropped)
        call_kwargs = execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "EmptyChoicesError"

    def test_empty_string_content_is_valid(self) -> None:
        """content="" is a legitimate response — must NOT raise."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = ""  # Legitimately empty, not None

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 0

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_empty_str"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = self._create_client(execution, openai_client)

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.content == ""


class TestModelDumpFailureRecordsCall:
    """Regression: model_dump() failure must still record the LLM call.

    Bug: If response.model_dump() raises during success-path processing,
    record_call(SUCCESS) never fires and the audit trail has no evidence
    that tokens were consumed.
    """

    @staticmethod
    def _create_mock_execution() -> Mock:
        execution = Mock(spec=ExecutionRepository)
        execution.allocate_call_index = Mock(return_value=0)
        execution.record_call = Mock()
        return execution

    def test_model_dump_failure_records_error_call(self) -> None:
        """When model_dump() raises, an ERROR call must be recorded."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Hello!"

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(side_effect=TypeError("Unserializable field"))

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_dump_fail",
            run_id="run_dump_fail",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        with pytest.raises(LLMClientError, match="serialize"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # The critical assertion: record_call was invoked despite the failure
        execution.record_call.assert_called_once()
        call_kwargs = execution.record_call.call_args
        assert call_kwargs.kwargs["status"] == CallStatus.ERROR


class TestTier3UsageBoundary:
    """Bug: TokenUsage.known() used on Tier 3 response.usage fields.

    LLM providers return usage data as Tier 3 external data. Fields like
    prompt_tokens/completion_tokens may be floats, non-finite, or non-int.
    TokenUsage.known() trusts values implicitly — from_dict() must be used
    at the Tier 3 boundary instead.
    """

    @staticmethod
    def _create_mock_execution() -> Mock:
        execution = Mock(spec=ExecutionRepository)
        execution.allocate_call_index = Mock(return_value=0)
        execution.record_call = Mock()
        return execution

    def test_float_usage_values_coerced_via_from_dict(self) -> None:
        """Float token counts from provider must be coerced to None, not crash."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Response text"

        choice = Mock()
        choice.message = message
        choice.finish_reason = "stop"

        usage = Mock()
        usage.prompt_tokens = 10.5  # Float — not int
        usage.completion_tokens = 20.7  # Float — not int

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_float_usage"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_float",
            run_id="run_float",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Float values should be coerced to None (unknown), not passed through
        assert result.usage.prompt_tokens is None
        assert result.usage.completion_tokens is None

    def test_bool_usage_values_rejected_as_invalid_by_from_dict(self) -> None:
        """Bool token counts must be rejected as invalid (bool is subclass of int but not a valid count)."""
        execution = self._create_mock_execution()

        message = Mock()
        message.content = "Response text"

        choice = Mock()
        choice.message = message
        choice.finish_reason = "stop"

        usage = Mock()
        usage.prompt_tokens = True  # bool — technically int subclass
        usage.completion_tokens = False

        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_bool_usage"})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_bool",
            run_id="run_bool",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Bool values should be rejected by from_dict
        assert result.usage.prompt_tokens is None
        assert result.usage.completion_tokens is None
