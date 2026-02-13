# tests/plugins/clients/test_http_telemetry.py
"""Tests for AuditedHTTPClient telemetry integration."""

import itertools
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.plugins.clients.http import AuditedHTTPClient


class TestHTTPClientTelemetry:
    """Tests for telemetry emission from AuditedHTTPClient."""

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

    def test_successful_post_emits_telemetry(self) -> None:
        """Successful HTTP POST emits ExternalCallCompleted event."""
        recorder = self._create_mock_recorder()

        # Track emitted events
        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

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
        # Full payloads are included for observability
        assert event.request_payload is not None
        assert event.request_payload["method"] == "POST"
        assert event.request_payload["json"] == {"input": "test"}
        assert event.response_payload is not None
        assert event.response_payload["status_code"] == 200
        assert event.response_payload["body"] == {"result": "success"}
        assert event.token_usage is None  # Not applicable for HTTP
        assert isinstance(event.timestamp, datetime)

    def test_failed_post_emits_telemetry_with_error_status(self) -> None:
        """Failed HTTP POST emits ExternalCallCompleted with ERROR status."""
        recorder = self._create_mock_recorder()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
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
        # Request payload is still included on error for debugging
        assert event.request_payload is not None
        assert event.request_payload["method"] == "POST"
        assert event.response_payload is None  # No response on error

    def test_noop_callback_works(self) -> None:
        """No-op callback (telemetry disabled) works without error."""
        recorder = self._create_mock_recorder()

        # No-op callback (simulates telemetry disabled)
        def noop_callback(event: Any) -> None:
            pass

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=noop_callback,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            response = client.post("/endpoint", json={"input": "test"})

        # Call succeeds without error
        assert response.status_code == 200
        # Audit trail is still recorded
        recorder.record_call.assert_called_once()

    def test_telemetry_emitted_after_landscape_recording(self) -> None:
        """Telemetry is emitted AFTER Landscape recording succeeds."""
        recorder = self._create_mock_recorder()

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

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            client.post("/endpoint", json={"input": "test"})

        # Verify order: Landscape first, then telemetry
        assert call_order == ["landscape", "telemetry"]

    def test_no_telemetry_when_landscape_recording_fails(self) -> None:
        """Telemetry is NOT emitted if Landscape recording fails.

        This is a critical invariant: Landscape is the legal record.
        If audit recording fails, telemetry should NOT be emitted because
        the event was never properly recorded.
        """
        recorder = self._create_mock_recorder()

        # Make record_call raise an exception (simulating DB failure)
        recorder.record_call.side_effect = Exception("Database connection failed")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

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
        recorder = self._create_mock_recorder()

        def failing_telemetry_emit(event: ExternalCallCompleted) -> None:
            raise RuntimeError("Telemetry exporter failed!")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=failing_telemetry_emit,  # Will raise!
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            # Call should succeed despite telemetry failure
            response = client.post("/endpoint", json={"input": "test"})

        # Verify call succeeded
        assert response.status_code == 200

        # CRITICAL: Only ONE audit record, with SUCCESS status
        assert recorder.record_call.call_count == 1
        call_kwargs = recorder.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_http_error_response_emits_telemetry(self) -> None:
        """4xx/5xx response emits telemetry with ERROR status."""
        recorder = self._create_mock_recorder()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal Server Error"}
        mock_response.text = '{"error": "Internal Server Error"}'
        mock_response.content = b'{"error": "Internal Server Error"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

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
        recorder = self._create_mock_recorder()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            # URL with embedded credentials
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api_user:super_secret_password@api.example.com:8443",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            client.post("/endpoint", json={"input": "test"})

        # Verify telemetry was emitted
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CRITICAL: Provider must NOT contain credentials
        assert "api_user" not in event.provider, f"Credentials leaked in provider: {event.provider}"
        assert "super_secret_password" not in event.provider, f"Password leaked in provider: {event.provider}"
        assert "@" not in event.provider, f"Userinfo separator leaked in provider: {event.provider}"

        # Provider should contain only hostname (optionally with port)
        assert "api.example.com" in event.provider

    def test_provider_extraction_handles_url_without_credentials(self) -> None:
        """Provider extraction works correctly for URLs without credentials."""
        recorder = self._create_mock_recorder()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            # URL without credentials
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_123",
                base_url="https://api.example.com:443",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

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

    def _create_mock_recorder(self) -> MagicMock:
        recorder = MagicMock()
        counter = itertools.count()
        recorder.allocate_call_index.side_effect = lambda _: next(counter)
        return recorder

    def test_per_call_token_id_overrides_client_default(self) -> None:
        """post(token_id=...) overrides the constructor token_id in telemetry."""
        recorder = self._create_mock_recorder()
        emitted_events: list[ExternalCallCompleted] = []

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"ok": true}'
        mock_response.content = b'{"ok": true}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_batch",
                base_url="https://api.example.com",
                run_id="run_batch",
                telemetry_emit=emitted_events.append,
                token_id="token-constructor",  # Client-level default
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            # Call with per-call override
            client.post("/v1/chat", json={"msg": "hello"}, token_id="token-row-42")

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "token-row-42"

    def test_client_default_token_id_when_no_override(self) -> None:
        """Without per-call override, client-level token_id is used."""
        recorder = self._create_mock_recorder()
        emitted_events: list[ExternalCallCompleted] = []

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"ok": true}'
        mock_response.content = b'{"ok": true}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
                state_id="state_single",
                base_url="https://api.example.com",
                run_id="run_single",
                telemetry_emit=emitted_events.append,
                token_id="token-constructor",
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            client.post("/v1/chat", json={"msg": "hello"})

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "token-constructor"

    def test_per_call_token_id_on_error_path(self) -> None:
        """Per-call token_id is used even when the request fails."""
        recorder = self._create_mock_recorder()
        emitted_events: list[ExternalCallCompleted] = []

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                recorder=recorder,
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
