# tests/plugins/clients/test_llm_error_classification.py
"""Tests for LLM client error classification and retry behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from elspeth.contracts import CallStatus
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
    ContentPolicyError,
    ContextLengthError,
    LLMClientError,
    NetworkError,
    RateLimitError,
    ServerError,
    _is_retryable_error,
)


class TestErrorClassification:
    """Test error classification logic."""

    def test_rate_limit_errors_are_retryable(self) -> None:
        """Rate limit errors (429) should be classified as retryable."""
        # Error with explicit 429 code
        error_429 = Exception("Error 429: Rate limit exceeded")
        assert _is_retryable_error(error_429) is True

        # Error with "rate" in message
        error_rate = Exception("Rate limit has been exceeded")
        assert _is_retryable_error(error_rate) is True

    def test_server_errors_are_retryable(self) -> None:
        """Server errors (5xx) should be classified as retryable."""
        server_errors = [
            Exception("500 Internal Server Error"),
            Exception("502 Bad Gateway"),
            Exception("503 Service Unavailable"),
            Exception("504 Gateway Timeout"),
            Exception("529 Model Overloaded"),  # Azure-specific
        ]

        for error in server_errors:
            assert _is_retryable_error(error) is True, f"Failed for: {error}"

    def test_network_errors_are_retryable(self) -> None:
        """Network/connection errors should be classified as retryable."""
        network_errors = [
            Exception("Connection timeout"),
            Exception("Request timed out"),
            Exception("Connection refused by server"),
            Exception("Connection reset by peer"),
            Exception("Connection aborted"),
            Exception("Network unreachable"),
            Exception("Host unreachable"),
            Exception("DNS resolution failed"),
            Exception("getaddrinfo failed"),
        ]

        for error in network_errors:
            assert _is_retryable_error(error) is True, f"Failed for: {error}"

    def test_client_errors_are_not_retryable(self) -> None:
        """Client errors (4xx except 429) should not be retryable."""
        client_errors = [
            Exception("400 Bad Request"),
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("404 Not Found"),
            Exception("422 Unprocessable Entity"),
        ]

        for error in client_errors:
            assert _is_retryable_error(error) is False, f"Failed for: {error}"

    def test_content_policy_errors_are_not_retryable(self) -> None:
        """Content policy violations should not be retryable."""
        policy_errors = [
            Exception("Your request was rejected by our safety system"),
            Exception("Content policy violation detected"),
            Exception("content_policy_violation"),
        ]

        for error in policy_errors:
            assert _is_retryable_error(error) is False, f"Failed for: {error}"

    def test_context_length_errors_are_not_retryable(self) -> None:
        """Context length exceeded errors should not be retryable."""
        context_errors = [
            Exception("This model's maximum context length is 8192 tokens"),
            Exception("context_length_exceeded"),
            Exception("Maximum context length exceeded"),
        ]

        for error in context_errors:
            assert _is_retryable_error(error) is False, f"Failed for: {error}"

    def test_unknown_errors_are_not_retryable(self) -> None:
        """Unknown errors should default to non-retryable (conservative)."""
        unknown_errors = [
            Exception("Something went wrong"),
            Exception("Unexpected error occurred"),
            Exception("No clue what happened"),
        ]

        for error in unknown_errors:
            assert _is_retryable_error(error) is False, f"Failed for: {error}"


class TestLLMClientExceptionTypes:
    """Test that correct exception types are raised."""

    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create a mock LandscapeRecorder."""
        recorder = Mock(spec=LandscapeRecorder)
        return recorder

    @pytest.fixture
    def mock_openai_client(self) -> Mock:
        """Create a mock OpenAI client."""
        return Mock()

    def test_rate_limit_raises_rate_limit_error(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Rate limit error (429) should raise RateLimitError."""
        # Configure mock to raise rate limit error
        mock_openai_client.chat.completions.create.side_effect = Exception("429 Rate limit exceeded")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(RateLimitError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is True
        assert "429" in str(exc_info.value)

    def test_server_error_raises_server_error(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Server error (503) should raise ServerError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(ServerError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is True
        assert "503" in str(exc_info.value)

    def test_network_error_raises_network_error(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Network timeout should raise NetworkError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("Connection timeout")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(NetworkError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is True
        assert "timeout" in str(exc_info.value).lower()

    def test_content_policy_raises_content_policy_error(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Content policy violation should raise ContentPolicyError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("Your request was rejected by our safety system")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(ContentPolicyError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is False
        assert "safety system" in str(exc_info.value).lower()

    def test_context_length_raises_context_length_error(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Context length exceeded should raise ContextLengthError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("This model's maximum context length is 8192 tokens")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(ContextLengthError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is False
        assert "context length" in str(exc_info.value).lower()

    def test_client_error_raises_llm_client_error_non_retryable(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Client error (401) should raise non-retryable LLMClientError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API key")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(LLMClientError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        assert exc_info.value.retryable is False
        assert "401" in str(exc_info.value)

    def test_audit_trail_records_retryable_flag(
        self,
        mock_recorder: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Audit trail should record correct retryable flag."""
        # Test retryable error
        mock_openai_client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

        client = AuditedLLMClient(
            recorder=mock_recorder,
            state_id="test-state",
            run_id="run_abc",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai_client,
        )

        with pytest.raises(ServerError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        # Verify recorder was called with retryable=True
        mock_recorder.record_call.assert_called_once()
        call_args = mock_recorder.record_call.call_args
        assert call_args.kwargs["status"] == CallStatus.ERROR
        assert call_args.kwargs["error"]["retryable"] is True

        # Reset and test non-retryable error
        mock_recorder.reset_mock()
        mock_openai_client.chat.completions.create.side_effect = Exception("401 Unauthorized")

        with pytest.raises(LLMClientError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
            )

        # Verify recorder was called with retryable=False
        mock_recorder.record_call.assert_called_once()
        call_args = mock_recorder.record_call.call_args
        assert call_args.kwargs["status"] == CallStatus.ERROR
        assert call_args.kwargs["error"]["retryable"] is False


class TestAzureSpecificCodes:
    """Test Azure-specific error codes."""

    def test_azure_529_model_overloaded_is_retryable(self) -> None:
        """Azure 529 (model overloaded) should be retryable."""
        error = Exception("529: The model is currently overloaded")
        assert _is_retryable_error(error) is True

    def test_azure_other_5xx_codes_are_retryable(self) -> None:
        """Other 5xx codes used by Azure should be retryable."""
        # Azure may use various 5xx codes for capacity issues
        azure_errors = [
            Exception("500: Internal server error from Azure"),
            Exception("502: Bad gateway from Azure OpenAI"),
            Exception("504: Gateway timeout from Azure"),
        ]

        for error in azure_errors:
            assert _is_retryable_error(error) is True, f"Failed for: {error}"
