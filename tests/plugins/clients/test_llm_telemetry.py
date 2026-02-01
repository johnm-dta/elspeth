# tests/plugins/clients/test_llm_telemetry.py
"""Tests for AuditedLLMClient telemetry integration."""

import itertools
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
    LLMClientError,
)
from elspeth.telemetry.events import ExternalCallCompleted


class TestLLMClientTelemetry:
    """Tests for telemetry emission from AuditedLLMClient."""

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock LandscapeRecorder that returns recorded calls."""
        recorder = MagicMock()
        counter = itertools.count()
        recorder.allocate_call_index.side_effect = lambda _: next(counter)

        # record_call returns a Call object with hashes
        recorded_call = MagicMock()
        recorded_call.request_hash = "req_hash_123"
        recorded_call.response_hash = "resp_hash_456"
        recorder.record_call.return_value = recorded_call

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

    def test_successful_call_emits_telemetry(self) -> None:
        """Successful LLM call emits ExternalCallCompleted event."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        # Track emitted events
        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="azure",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify response
        assert response.content == "Hello!"

        # Verify telemetry event
        assert len(emitted_events) == 1
        event = emitted_events[0]

        assert isinstance(event, ExternalCallCompleted)
        assert event.run_id == "run_abc"
        assert event.state_id == "state_123"
        assert event.call_type == CallType.LLM
        assert event.provider == "azure"
        assert event.status == CallStatus.SUCCESS
        assert event.latency_ms > 0
        # Hashes are computed from request/response data
        assert event.request_hash is not None
        assert len(event.request_hash) == 64  # SHA-256 hex digest
        assert event.response_hash is not None
        assert len(event.response_hash) == 64  # SHA-256 hex digest
        # Full payloads are included for observability
        assert event.request_payload is not None
        assert event.request_payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert event.request_payload["model"] == "gpt-4"
        assert event.response_payload is not None
        assert event.response_payload["content"] == "Hello!"
        assert event.response_payload["model"] == "gpt-4"
        assert event.token_usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert isinstance(event.timestamp, datetime)

    def test_failed_call_emits_telemetry_with_error_status(self) -> None:
        """Failed LLM call emits ExternalCallCompleted with ERROR status."""
        recorder = self._create_mock_recorder()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("API error")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        with pytest.raises(LLMClientError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # Verify telemetry event
        assert len(emitted_events) == 1
        event = emitted_events[0]

        assert event.run_id == "run_abc"
        assert event.state_id == "state_123"
        assert event.call_type == CallType.LLM
        assert event.provider == "openai"
        assert event.status == CallStatus.ERROR
        assert event.latency_ms > 0
        # Hash is computed from request data
        assert event.request_hash is not None
        assert len(event.request_hash) == 64  # SHA-256 hex digest
        assert event.response_hash is None  # No response on error
        # Request payload is still included on error for debugging
        assert event.request_payload is not None
        assert event.request_payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert event.response_payload is None  # No response on error
        assert event.token_usage is None

    def test_noop_callback_works(self) -> None:
        """No-op callback (telemetry disabled) works without error."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        # No-op callback (simulates telemetry disabled)
        def noop_callback(event: Any) -> None:
            pass

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=noop_callback,
        )

        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Call succeeds without error
        assert response.content == "Hello!"
        # Audit trail is still recorded
        recorder.record_call.assert_called_once()

    def test_telemetry_emitted_after_landscape_recording(self) -> None:
        """Telemetry is emitted AFTER Landscape recording succeeds."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        call_order: list[str] = []

        def mock_record_call(**kwargs):
            call_order.append("landscape")
            recorded_call = MagicMock()
            recorded_call.request_hash = "req_hash"
            recorded_call.response_hash = "resp_hash"
            return recorded_call

        recorder.record_call.side_effect = mock_record_call

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            call_order.append("telemetry")

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify order: Landscape first, then telemetry
        assert call_order == ["landscape", "telemetry"]

    def test_telemetry_handles_empty_usage(self) -> None:
        """Telemetry emits None token_usage when provider omits usage data."""
        recorder = self._create_mock_recorder()

        # Create response without usage
        message = Mock()
        message.content = "Hello!"
        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        response.model = "gpt-4"
        response.usage = None  # Provider omits usage
        response.model_dump = Mock(return_value={})

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify telemetry event has None token_usage
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.token_usage is None

    def test_multiple_calls_emit_multiple_events(self) -> None:
        """Each LLM call emits a separate telemetry event."""
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        # Make multiple calls
        client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "First"}])
        client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Second"}])
        client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Third"}])

        # Verify one event per call
        assert len(emitted_events) == 3

        # All events have same run_id and state_id
        for event in emitted_events:
            assert event.run_id == "run_abc"
            assert event.state_id == "state_123"
            assert event.status == CallStatus.SUCCESS

    def test_telemetry_failure_does_not_corrupt_successful_call(self) -> None:
        """Telemetry callback failure should not corrupt audit trail or cause retry.

        Regression test for bug: If telemetry_emit raises (e.g., when
        fail_on_total_exporter_failure=True), the exception should not:
        1. Cause a second audit record with ERROR status
        2. Change the call outcome from SUCCESS to ERROR
        3. Trigger retry logic for a successful call

        The fix isolates telemetry emission in its own try/except.
        """
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        def failing_telemetry_emit(event: ExternalCallCompleted) -> None:
            raise RuntimeError("Telemetry exporter failed!")

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="azure",
            run_id="run_abc",
            telemetry_emit=failing_telemetry_emit,  # Will raise!
        )

        # Call should succeed despite telemetry failure
        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Verify call succeeded
        assert response.content == "Hello!"

        # CRITICAL: Only ONE audit record, with SUCCESS status
        assert recorder.record_call.call_count == 1
        call_kwargs = recorder.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_no_telemetry_when_landscape_recording_fails(self) -> None:
        """Telemetry is NOT emitted if Landscape recording fails.

        This is a critical invariant: Landscape is the legal record.
        If audit recording fails, telemetry should NOT be emitted because
        the event was never properly recorded.
        """
        recorder = self._create_mock_recorder()
        openai_client = self._create_mock_openai_client()

        # Make record_call raise an exception (simulating DB failure)
        recorder.record_call.side_effect = Exception("Database connection failed")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            recorder=recorder,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        # The call should fail (Landscape recording fails)
        with pytest.raises(Exception, match="Database connection failed"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # CRITICAL: No telemetry should have been emitted
        assert len(emitted_events) == 0, "Telemetry was emitted before Landscape recording!"
