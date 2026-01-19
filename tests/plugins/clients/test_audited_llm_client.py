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
        recorder = MagicMock()
        recorder.record_call = MagicMock()
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
        assert call_kwargs["request_data"]["messages"] == [
            {"role": "user", "content": "Hello"}
        ]
        assert call_kwargs["response_data"]["content"] == "Hello!"
        assert call_kwargs["latency_ms"] > 0

    def test_call_index_increments(self) -> None:
        """Each call gets a unique, incrementing call index."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
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
        openai_client.chat.completions.create.side_effect = Exception(
            "API connection failed"
        )

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
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
        openai_client.chat.completions.create.side_effect = Exception(
            "Rate limit exceeded (429)"
        )

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
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
        openai_client.chat.completions.create.side_effect = Exception(
            "You have exceeded your rate limit"
        )

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
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

    def test_response_without_model_dump(self) -> None:
        """Handles responses without model_dump method."""
        recorder = self._create_mock_recorder()

        # Create response without model_dump
        message = Mock()
        message.content = "Hello!"

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        response = Mock(spec=["choices", "model", "usage"])  # No model_dump
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = usage

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Should work without raw_response
        assert result.content == "Hello!"
        assert result.raw_response is None

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
            underlying_client=openai_client,
        )

        result = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # None content should become empty string
        assert result.content == ""
