# tests/plugins/clients/test_llm_telemetry.py
"""Tests for AuditedLLMClient telemetry integration."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from elspeth.contracts import Call, CallStatus, CallType, TokenUsage
from elspeth.contracts.call_data import LLMCallRequest, LLMCallResponse
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.plugins.infrastructure.clients.llm import (
    AuditedLLMClient,
    LLMClientError,
)


@dataclass(slots=True)
class ProviderUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int | None = None


@dataclass(slots=True)
class ProviderMessage:
    content: Any


@dataclass(slots=True)
class ProviderChoice:
    message: ProviderMessage
    finish_reason: str = "stop"


@dataclass(slots=True)
class ProviderResponse:
    choices: list[ProviderChoice]
    model: str = "gpt-4"
    usage: ProviderUsage | None = field(default_factory=ProviderUsage)
    raw_response: dict[str, Any] = field(default_factory=lambda: {"id": "resp_123"})
    model_dump_error: Exception | None = None

    def model_dump(self) -> dict[str, Any]:
        if self.model_dump_error is not None:
            raise self.model_dump_error
        return self.raw_response


def provider_response(
    *,
    content: Any = "Hello!",
    finish_reason: str = "stop",
    choices_empty: bool = False,
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    include_usage: bool = True,
    raw_response: dict[str, Any] | None = None,
    model_dump_error: Exception | None = None,
) -> ProviderResponse:
    choices = [] if choices_empty else [ProviderChoice(message=ProviderMessage(content), finish_reason=finish_reason)]
    usage = ProviderUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens) if include_usage else None
    return ProviderResponse(
        choices=choices,
        model=model,
        usage=usage,
        raw_response=raw_response if raw_response is not None else {"id": "resp_123"},
        model_dump_error=model_dump_error,
    )


@dataclass(slots=True)
class FakeChatCompletions:
    response: ProviderResponse | None = None
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> ProviderResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("FakeChatCompletions requires a response or error")
        return self.response


@dataclass(slots=True)
class FakeChat:
    completions: FakeChatCompletions


@dataclass(slots=True)
class FakeOpenAIClient:
    chat: FakeChat


def fake_openai_client(
    *,
    response: ProviderResponse | None = None,
    error: Exception | None = None,
) -> FakeOpenAIClient:
    completions = FakeChatCompletions(response=response if response is not None else provider_response(), error=error)
    return FakeOpenAIClient(chat=FakeChat(completions=completions))


class FakeCallRecorder:
    def __init__(self) -> None:
        self._call_counter = itertools.count()
        self._operation_call_counter = itertools.count()
        self.recorded_calls: list[dict[str, Any]] = []
        self.record_call_error: Exception | None = None
        self.record_call_observer: Callable[[dict[str, Any]], None] | None = None

    def allocate_call_index(self, state_id: str) -> int:
        return next(self._call_counter)

    def allocate_operation_call_index(self, operation_id: str) -> int:
        return next(self._operation_call_counter)

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        call_kwargs = {
            "state_id": state_id,
            "call_index": call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        if self.record_call_observer is not None:
            self.record_call_observer(call_kwargs)
        if self.record_call_error is not None:
            raise self.record_call_error
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def record_operation_call(
        self,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: Any,
        response_data: Any | None = None,
        error: Any | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> Call:
        actual_call_index = call_index if call_index is not None else self.allocate_operation_call_index(operation_id)
        call_kwargs = {
            "operation_id": operation_id,
            "call_index": actual_call_index,
            "call_type": call_type,
            "status": status,
            "request_data": request_data,
            "response_data": response_data,
            "error": error,
            "latency_ms": latency_ms,
            "request_ref": request_ref,
            "response_ref": response_ref,
            "resolved_prompt_template_hash": resolved_prompt_template_hash,
        }
        if self.record_call_observer is not None:
            self.record_call_observer(call_kwargs)
        if self.record_call_error is not None:
            raise self.record_call_error
        self.recorded_calls.append(call_kwargs)
        return self._recorded_call(call_kwargs)

    def _recorded_call(self, call_kwargs: dict[str, Any]) -> Call:
        return Call(
            call_id=f"call_{len(self.recorded_calls)}",
            call_index=call_kwargs["call_index"],
            call_type=call_kwargs["call_type"],
            status=call_kwargs["status"],
            request_hash="req_hash_123",
            response_hash="resp_hash_456" if call_kwargs["response_data"] is not None else None,
            created_at=datetime.now(UTC),
            state_id=call_kwargs.get("state_id"),
            operation_id=call_kwargs.get("operation_id"),
            latency_ms=call_kwargs["latency_ms"],
            resolved_prompt_template_hash=call_kwargs["resolved_prompt_template_hash"],
        )


class TestLLMClientErrorBranchTelemetry:
    """elspeth-a960d22540: malformed-response error branches must emit telemetry.

    Several provider-response error branches recorded a Landscape ERROR call
    then raised WITHOUT emitting ExternalCallCompleted(ERROR), unlike the SDK
    error / null-content / success branches that do. Telemetry dashboards
    therefore undercounted malformed/content-filtered failures even though
    audit primacy was preserved.
    """

    def _response(
        self,
        *,
        content: Any = "Hi",
        finish_reason: str = "stop",
        choices_empty: bool = False,
        model_dump_fails: bool = False,
    ) -> ProviderResponse:
        return provider_response(
            content=content,
            finish_reason=finish_reason,
            choices_empty=choices_empty,
            model_dump_error=TypeError("unserializable response") if model_dump_fails else None,
        )

    def _run_expecting_error(self, response: ProviderResponse) -> list[ExternalCallCompleted]:
        events: list[ExternalCallCompleted] = []
        client = AuditedLLMClient(
            execution=FakeCallRecorder(),
            state_id="state_123",
            underlying_client=fake_openai_client(response=response),
            provider="azure",
            run_id="run_abc",
            telemetry_emit=events.append,
        )
        with pytest.raises(LLMClientError):
            client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
        return events

    def test_empty_choices_emits_error_telemetry(self) -> None:
        events = self._run_expecting_error(self._response(choices_empty=True))
        assert len(events) == 1
        assert events[0].status == CallStatus.ERROR

    def test_unsupported_tool_calls_emits_error_telemetry(self) -> None:
        events = self._run_expecting_error(self._response(content=None, finish_reason="tool_calls"))
        assert len(events) == 1
        assert events[0].status == CallStatus.ERROR

    def test_non_string_content_emits_error_telemetry(self) -> None:
        events = self._run_expecting_error(self._response(content=[1, 2, 3]))
        assert len(events) == 1
        assert events[0].status == CallStatus.ERROR

    def test_response_serialization_failure_emits_error_telemetry(self) -> None:
        events = self._run_expecting_error(self._response(model_dump_fails=True))
        assert len(events) == 1
        assert events[0].status == CallStatus.ERROR


class TestLLMClientTelemetry:
    """Tests for telemetry emission from AuditedLLMClient."""

    def _create_execution(self) -> FakeCallRecorder:
        """Create a fake call recorder that returns real Call contracts."""
        return FakeCallRecorder()

    def _create_openai_client(
        self,
        *,
        content: str = "Hello!",
        model: str = "gpt-4",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> FakeOpenAIClient:
        """Create a fake OpenAI client."""
        return fake_openai_client(
            response=provider_response(
                content=content,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def test_successful_call_emits_telemetry(self) -> None:
        """Successful LLM call emits ExternalCallCompleted event."""
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        # Track emitted events
        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
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
        # Typed DTO payloads are included for observability
        assert event.request_payload is not None
        assert isinstance(event.request_payload, LLMCallRequest)
        assert event.request_payload.to_dict()["messages"] == [{"role": "user", "content": "Hello"}]
        assert event.request_payload.model == "gpt-4"
        assert event.response_payload is not None
        assert isinstance(event.response_payload, LLMCallResponse)
        assert event.response_payload.content == "Hello!"
        assert event.response_payload.model == "gpt-4"
        assert event.token_usage == TokenUsage(prompt_tokens=10, completion_tokens=5)
        assert isinstance(event.timestamp, datetime)

    def test_failed_call_emits_telemetry_with_error_status(self) -> None:
        """Failed LLM call emits ExternalCallCompleted with ERROR status."""
        execution = self._create_execution()
        openai_client = fake_openai_client(error=Exception("API error"))

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
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
        # Typed request DTO is still included on error for debugging
        assert event.request_payload is not None
        assert isinstance(event.request_payload, LLMCallRequest)
        assert event.request_payload.to_dict()["messages"] == [{"role": "user", "content": "Hello"}]
        assert event.response_payload is None  # No response on error
        assert event.token_usage is None

    def test_noop_callback_works(self) -> None:
        """No-op callback (telemetry disabled) works without error."""
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        # No-op callback (simulates telemetry disabled)
        def noop_callback(event: Any) -> None:
            pass

        client = AuditedLLMClient(
            execution=execution,
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
        assert len(execution.recorded_calls) == 1

    def test_telemetry_emitted_after_landscape_recording(self) -> None:
        """Telemetry is emitted AFTER Landscape recording succeeds."""
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        call_order: list[str] = []

        def observe_record_call(_call_kwargs: dict[str, Any]) -> None:
            call_order.append("landscape")

        execution.record_call_observer = observe_record_call

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            call_order.append("telemetry")

        client = AuditedLLMClient(
            execution=execution,
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
        execution = self._create_execution()
        response = provider_response(content="Hello!", include_usage=False, raw_response={})
        openai_client = fake_openai_client(response=response)

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
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
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
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
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        def failing_telemetry_emit(event: ExternalCallCompleted) -> None:
            raise RuntimeError("Telemetry exporter failed!")

        client = AuditedLLMClient(
            execution=execution,
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
        assert len(execution.recorded_calls) == 1
        call_kwargs = execution.recorded_calls[0]
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_no_telemetry_when_landscape_recording_fails(self) -> None:
        """Telemetry is NOT emitted if Landscape recording fails.

        This is a critical invariant: Landscape is the legal record.
        If audit recording fails, telemetry should NOT be emitted because
        the event was never properly recorded.
        """
        execution = self._create_execution()
        openai_client = self._create_openai_client()

        # Make record_call raise an exception (simulating DB failure)
        execution.record_call_error = Exception("Database connection failed")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
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
