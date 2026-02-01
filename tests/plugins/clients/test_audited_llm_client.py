# tests/plugins/clients/test_audited_llm_client.py
"""Tests for AuditedLLMClient."""

from unittest.mock import MagicMock, Mock

import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
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
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=150.0,
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert response.latency_ms == 150.0

    def test_total_tokens_property(self) -> None:
        """total_tokens sums prompt and completion tokens."""
        response = LLMResponse(
            content="test",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        assert response.total_tokens == 15

    def test_total_tokens_with_missing_fields(self) -> None:
        """total_tokens handles missing usage fields."""
        response = LLMResponse(
            content="test",
            model="gpt-4",
            usage={},
        )

        assert response.total_tokens == 0

    def test_default_values(self) -> None:
        """LLMResponse has sensible defaults."""
        response = LLMResponse(content="test", model="gpt-4")

        assert response.usage == {}
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

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock LandscapeRecorder."""
        import itertools

        recorder = MagicMock()
        recorder.record_call = MagicMock()
        counter = itertools.count()
        recorder.allocate_call_index.side_effect = lambda _: next(counter)
        return recorder

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
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
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
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert response.latency_ms > 0

        # Verify audit record
        recorder.record_call.assert_called_once()
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["state_id"] == "state_123"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] == CallType.LLM
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["request_data"]["model"] == "gpt-4"
        assert call_kwargs["request_data"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["response_data"]["content"] == "Hello!"
        assert call_kwargs["latency_ms"] > 0

    def test_call_index_increments(self) -> None:
        """Each call gets a unique, incrementing call index."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
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
        calls = recorder.record_call.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["call_index"] == 0
        assert calls[1][1]["call_index"] == 1

    def test_failed_call_records_error(self) -> None:
        """Failed LLM call records error details."""
        recorder = self._create_mock_recorder()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("API connection failed")

        client = AuditedLLMClient(
            recorder=recorder,
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
        recorder.record_call.assert_called_once()
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"]["type"] == "Exception"
        assert "API connection failed" in call_kwargs["error"]["message"]
        assert call_kwargs["error"]["retryable"] is False

    def test_rate_limit_error_marked_retryable(self) -> None:
        """Rate limit errors are marked as retryable."""
        recorder = self._create_mock_recorder()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("Rate limit exceeded (429)")

        client = AuditedLLMClient(
            recorder=recorder,
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
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["error"]["retryable"] is True

    def test_rate_limit_detected_by_keyword(self) -> None:
        """Rate limit is detected by 'rate' keyword in error."""
        recorder = self._create_mock_recorder()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("You have exceeded your rate limit")

        client = AuditedLLMClient(
            recorder=recorder,
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

    def test_temperature_and_max_tokens_recorded(self) -> None:
        """Temperature and max_tokens are recorded in request."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
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

        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["request_data"]["temperature"] == 0.7
        assert call_kwargs["request_data"]["max_tokens"] == 100

    def test_provider_recorded_in_request(self) -> None:
        """Provider name is recorded in request data."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
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

        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["request_data"]["provider"] == "azure"

    def test_extra_kwargs_passed_to_client(self) -> None:
        """Extra kwargs are passed to underlying client and recorded."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
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
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["request_data"]["top_p"] == 0.9
        assert call_kwargs["request_data"]["presence_penalty"] == 0.5

    # NOTE: test_response_without_model_dump was removed because we require
    # openai>=2.15 which guarantees model_dump() exists on all responses.
    # Per CLAUDE.md "No Legacy Code Policy" - no backwards compatibility code.

    def test_empty_content_handled(self) -> None:
        """Handles responses with None content."""
        recorder = self._create_mock_recorder()

        message = Mock()
        message.content = None  # Explicitly None

        choice = Mock()
        choice.message = message

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
            recorder=recorder,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # None content should become empty string
        assert result.content == ""

    def test_full_raw_response_recorded_in_audit_trail(self) -> None:
        """Full raw_response from model_dump() is recorded in audit trail.

        This ensures audit completeness per CLAUDE.md:
        "External calls - Full request AND response recorded"
        """
        recorder = self._create_mock_recorder()

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
            recorder=recorder,
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
        call_kwargs = recorder.record_call.call_args[1]
        response_data = call_kwargs["response_data"]

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
        recorder = self._create_mock_recorder()

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
            recorder=recorder,
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
        call_kwargs = recorder.record_call.call_args[1]
        raw_response = call_kwargs["response_data"]["raw_response"]

        assert len(raw_response["choices"]) == 3
        assert raw_response["choices"][0]["message"]["content"] == "Option A"
        assert raw_response["choices"][1]["message"]["content"] == "Option B"
        assert raw_response["choices"][2]["message"]["content"] == "Option C"
        assert raw_response["choices"][2]["finish_reason"] == "length"

    def test_tool_calls_preserved_in_raw_response(self) -> None:
        """Tool calls are preserved in raw_response for function calling."""
        recorder = self._create_mock_recorder()

        # Response with tool calls
        full_response = {
            "id": "chatcmpl-tools",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,  # No text content when tool calls
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                            {
                                "id": "call_def456",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"timezone": "UTC"}',
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
        message.content = None  # Tool call responses have no text content

        choice = Mock()
        choice.message = message

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
            recorder=recorder,
            state_id="state_123",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=openai_client,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
        )

        # Verify tool calls are preserved in raw_response
        call_kwargs = recorder.record_call.call_args[1]
        raw_response = call_kwargs["response_data"]["raw_response"]

        assert raw_response["choices"][0]["finish_reason"] == "tool_calls"
        tool_calls = raw_response["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[1]["function"]["name"] == "get_time"

    # NOTE: test_raw_response_none_when_model_dump_unavailable was removed because
    # we require openai>=2.15 which guarantees model_dump() exists on all responses.
    # Per CLAUDE.md "No Legacy Code Policy" - no backwards compatibility code.

    def test_successful_call_with_missing_usage(self) -> None:
        """LLM call succeeds when provider returns no usage data.

        Some LLM providers/modes (streaming, certain Azure configs) omit
        usage data entirely. The client should record this as SUCCESS
        with an empty usage dict, not crash with AttributeError.
        """
        recorder = self._create_mock_recorder()

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
            recorder=recorder,
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
        assert result.usage == {}  # Empty dict, not crash

        # Audit trail should record SUCCESS with empty usage
        recorder.record_call.assert_called_once()
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["response_data"]["content"] == "Hello, I'm working!"
        assert call_kwargs["response_data"]["usage"] == {}
