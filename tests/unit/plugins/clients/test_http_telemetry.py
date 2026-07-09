# tests/plugins/clients/test_http_telemetry.py
"""Tests for AuditedHTTPClient telemetry integration."""

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import HTTPCallRequest, HTTPCallResponse
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


@dataclass(frozen=True)
class _RecordedCall:
    request_hash: str = "req_hash_123"
    response_hash: str = "resp_hash_456"


class _RecordingExecution:
    def __init__(self) -> None:
        self._call_counter = itertools.count()
        self._operation_call_counter = itertools.count()
        self.record_call_calls: list[dict[str, Any]] = []
        self.record_call_effect: Callable[..., _RecordedCall] | Exception | None = None

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
    ) -> _RecordedCall:
        kwargs = {
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
        self.record_call_calls.append(kwargs)
        if isinstance(self.record_call_effect, Exception):
            raise self.record_call_effect
        if self.record_call_effect is not None:
            return self.record_call_effect(**kwargs)
        return _RecordedCall()

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
    ) -> _RecordedCall:
        return _RecordedCall()


def _json_response(body: Mapping[str, Any], *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        headers={"content-type": "application/json"},
    )


class TestHTTPClientTelemetry:
    """Tests for telemetry emission from AuditedHTTPClient."""

    def _create_execution(self) -> _RecordingExecution:
        """Create an ExecutionRepository fake that returns recorded calls."""
        return _RecordingExecution()

    def test_successful_post_emits_telemetry(self) -> None:
        """Successful HTTP POST emits ExternalCallCompleted event."""
        execution = self._create_execution()

        # Track emitted events
        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            response = client.post("/endpoint", json={"input": "test"})

        # Verify response
        assert response.status_code == 200

        # Verify telemetry event
        assert len(emitted_events) == 1
        event = emitted_events[0]

        assert isinstance(event, ExternalCallCompleted)
        assert event.run_id == "run_abc"
        assert event.state_id == "state_123"
        assert event.call_type == CallType.HTTP
        assert event.provider == "api.example.com"  # Extracted from base_url
        assert event.status == CallStatus.SUCCESS
        assert event.latency_ms > 0
        # Hashes are computed from request/response data
        assert event.request_hash is not None
        assert len(event.request_hash) == 64  # SHA-256 hex digest
        assert event.response_hash is not None
        assert len(event.response_hash) == 64  # SHA-256 hex digest
        # Typed DTO payloads are included for observability
        assert event.request_payload is not None
        assert isinstance(event.request_payload, HTTPCallRequest)
        assert event.request_payload.method == "POST"
        assert event.request_payload.json == {"input": "test"}
        assert event.response_payload is not None
        assert isinstance(event.response_payload, HTTPCallResponse)
        assert event.response_payload.status_code == 200
        assert event.response_payload.body == {"result": "success"}
        assert event.token_usage is None  # Not applicable for HTTP
        assert isinstance(event.timestamp, datetime)

    def test_failed_post_emits_telemetry_with_error_status(self) -> None:
        """Failed HTTP POST emits ExternalCallCompleted with ERROR status."""
        execution = self._create_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(httpx.ConnectError):
                client.post("/endpoint", json={"input": "test"})

        # Verify telemetry event
        assert len(emitted_events) == 1
        event = emitted_events[0]

        assert event.run_id == "run_abc"
        assert event.state_id == "state_123"
        assert event.call_type == CallType.HTTP
        assert event.status == CallStatus.ERROR
        assert event.latency_ms >= 0
        assert event.response_hash is None  # No response on error
        # Typed request DTO is still included on error for debugging
        assert event.request_payload is not None
        assert isinstance(event.request_payload, HTTPCallRequest)
        assert event.request_payload.method == "POST"
        assert event.response_payload is None  # No response on error

    def test_noop_callback_works(self) -> None:
        """No-op callback (telemetry disabled) works without error."""
        execution = self._create_execution()

        # No-op callback (simulates telemetry disabled)
        def noop_callback(event: Any) -> None:
            pass

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=noop_callback,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            response = client.post("/endpoint", json={"input": "test"})

        # Call succeeds without error
        assert response.status_code == 200
        # Audit trail is still recorded
        assert len(execution.record_call_calls) == 1

    def test_telemetry_emitted_after_landscape_recording(self) -> None:
        """Telemetry is emitted AFTER Landscape recording succeeds."""
        execution = self._create_execution()

        call_order: list[str] = []

        def record_call_with_order(**kwargs: Any) -> _RecordedCall:
            call_order.append("landscape")
            return _RecordedCall(request_hash="req_hash", response_hash="resp_hash")

        execution.record_call_effect = record_call_with_order

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            call_order.append("telemetry")

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            client.post("/endpoint", json={"input": "test"})

        # Verify order: Landscape first, then telemetry
        assert call_order == ["landscape", "telemetry"]

    def test_no_telemetry_when_landscape_recording_fails(self) -> None:
        """Telemetry is NOT emitted if Landscape recording fails.

        This is a critical invariant: Landscape is the legal record.
        If audit recording fails, telemetry should NOT be emitted because
        the event was never properly recorded.
        """
        execution = self._create_execution()

        # Make record_call raise an exception (simulating DB failure)
        execution.record_call_effect = Exception("Database connection failed")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            # The call should fail (Landscape recording fails)
            with pytest.raises(Exception, match="Database connection failed"):
                client.post("/endpoint", json={"input": "test"})

        # CRITICAL: No telemetry should have been emitted
        assert len(emitted_events) == 0, "Telemetry was emitted before Landscape recording!"

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

        def failing_telemetry_emit(event: ExternalCallCompleted) -> None:
            raise RuntimeError("Telemetry exporter failed!")

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=failing_telemetry_emit,  # Will raise!
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            # Call should succeed despite telemetry failure
            response = client.post("/endpoint", json={"input": "test"})

        # Verify call succeeded
        assert response.status_code == 200

        # CRITICAL: Only ONE audit record, with SUCCESS status
        assert len(execution.record_call_calls) == 1
        call_kwargs = execution.record_call_calls[0]
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_programmer_bug_in_telemetry_callback_crashes_after_audit_recording(self) -> None:
        """TypeError/KeyError-style telemetry bugs must crash instead of being downgraded to warnings."""
        execution = self._create_execution()

        def failing_telemetry_emit(event: ExternalCallCompleted) -> None:
            raise TypeError("telemetry bug")

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=failing_telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            with pytest.raises(TypeError, match="telemetry bug"):
                client.post("/endpoint", json={"input": "test"})

        assert len(execution.record_call_calls) == 1
        call_kwargs = execution.record_call_calls[0]
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_http_error_response_emits_telemetry(self) -> None:
        """4xx/5xx response emits telemetry with ERROR status."""
        execution = self._create_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        response_from_server = _json_response({"error": "Internal Server Error"}, status_code=500)

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            response = client.post("/endpoint", json={"input": "test"})

        # Response is returned (not raised as exception)
        assert response.status_code == 500

        # Verify telemetry event with ERROR status
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.call_type == CallType.HTTP
        assert event.status == CallStatus.ERROR

    def test_provider_extraction_strips_credentials_from_url(self) -> None:
        """Provider extraction MUST NOT include credentials from URL.

        SECURITY: URLs may contain embedded credentials (e.g., https://user:pass@host/).
        The telemetry provider field must contain only the hostname, not the userinfo
        component. Leaking credentials into telemetry violates the secret-handling policy.

        Regression test for credential leak vulnerability.
        """
        execution = self._create_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            # URL with embedded credentials
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api_user:super_secret_password@api.example.com:8443",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            client.post("/endpoint", json={"input": "test"})

        # Verify telemetry was emitted
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CRITICAL: Provider must NOT contain credentials
        assert "api_user" not in event.provider, f"Credentials leaked in provider: {event.provider}"
        assert "super_secret_password" not in event.provider, f"Password leaked in provider: {event.provider}"
        assert "@" not in event.provider, f"Userinfo separator leaked in provider: {event.provider}"

        # Provider should contain only hostname (optionally with port)
        assert event.provider == "api.example.com"

    def test_provider_extraction_handles_url_without_credentials(self) -> None:
        """Provider extraction works correctly for URLs without credentials."""
        execution = self._create_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        response_from_server = _json_response({"result": "success"})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            # URL without credentials
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com:443",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            client.post("/endpoint", json={"input": "test"})

        # Verify provider is correct
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.provider == "api.example.com"


class TestHTTPClientPerCallTokenId:
    """Tests for per-call token_id override on post()/get().

    Batch transforms share one AuditedHTTPClient across multiple tokens.
    The per-call token_id parameter ensures correct telemetry attribution.
    """

    def _create_execution(self) -> _RecordingExecution:
        return _RecordingExecution()

    def test_per_call_token_id_overrides_client_default(self) -> None:
        """post(token_id=...) overrides the constructor token_id in telemetry."""
        execution = self._create_execution()
        emitted_events: list[ExternalCallCompleted] = []

        response_from_server = _json_response({"ok": True})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_batch",
                base_url="https://api.example.com",
                run_id="run_batch",
                telemetry_emit=emitted_events.append,
                token_id="token-constructor",  # Client-level default
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            # Call with per-call override
            client.post("/v1/chat", json={"msg": "hello"}, token_id="token-row-42")

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "token-row-42"

    def test_client_default_token_id_when_no_override(self) -> None:
        """Without per-call override, client-level token_id is used."""
        execution = self._create_execution()
        emitted_events: list[ExternalCallCompleted] = []

        response_from_server = _json_response({"ok": True})

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_single",
                base_url="https://api.example.com",
                run_id="run_single",
                telemetry_emit=emitted_events.append,
                token_id="token-constructor",
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = response_from_server

            client.post("/v1/chat", json={"msg": "hello"})

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "token-constructor"

    def test_per_call_token_id_on_error_path(self) -> None:
        """Per-call token_id is used even when the request fails."""
        execution = self._create_execution()
        emitted_events: list[ExternalCallCompleted] = []

        with patch("httpx.Client", autospec=True) as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_err",
                base_url="https://api.example.com",
                run_id="run_err",
                telemetry_emit=emitted_events.append,
                token_id="token-default",
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(httpx.ConnectError):
                client.post("/v1/chat", json={"msg": "hello"}, token_id="token-row-99")

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "token-row-99"
        assert emitted_events[0].status == CallStatus.ERROR
