# tests/plugins/clients/test_audited_http_client.py
"""Tests for AuditedHTTPClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.http import AuditedHTTPClient


class TestAuditedHTTPClient:
    """Tests for AuditedHTTPClient."""

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock LandscapeRecorder."""
        recorder = MagicMock()
        recorder.record_call = MagicMock()
        return recorder

    def test_successful_post_records_to_audit_trail(self) -> None:
        """Successful HTTP POST is recorded to audit trail."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            timeout=30.0,
        )

        # Mock httpx.Client
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"result": "success"}'

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = client.post(
                "https://api.example.com/v1/process",
                json={"data": "value"},
            )

        # Verify response
        assert response.status_code == 200

        # Verify audit record
        recorder.record_call.assert_called_once()
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["state_id"] == "state_123"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] == CallType.HTTP
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["request_data"]["method"] == "POST"
        assert (
            call_kwargs["request_data"]["url"] == "https://api.example.com/v1/process"
        )
        assert call_kwargs["request_data"]["json"] == {"data": "value"}
        assert call_kwargs["response_data"]["status_code"] == 200
        assert call_kwargs["latency_ms"] > 0

    def test_call_index_increments(self) -> None:
        """Each call gets a unique, incrementing call index."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Make multiple calls
            client.post("https://api.example.com/first")
            client.post("https://api.example.com/second")

        # Check call indices
        calls = recorder.record_call.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["call_index"] == 0
        assert calls[1][1]["call_index"] == 1

    def test_failed_call_records_error(self) -> None:
        """Failed HTTP call records error details."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                client.post("https://api.example.com/v1/process")

        # Verify error was recorded
        recorder.record_call.assert_called_once()
        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"]["type"] == "ConnectError"
        assert "Connection refused" in call_kwargs["error"]["message"]

    def test_auth_headers_filtered_from_recorded_request(self) -> None:
        """Auth-related headers are not recorded for security."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            headers={
                "Authorization": "Bearer secret-token",
                "X-API-Key": "api-key-12345",
                "Content-Type": "application/json",
                "X-Request-Id": "req-123",
            },
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://api.example.com/v1/process")

        # Verify auth headers were filtered
        call_kwargs = recorder.record_call.call_args[1]
        recorded_headers = call_kwargs["request_data"]["headers"]

        # Auth headers should NOT be recorded
        assert "Authorization" not in recorded_headers
        assert "X-API-Key" not in recorded_headers

        # Non-auth headers SHOULD be recorded
        assert "Content-Type" in recorded_headers
        assert "X-Request-Id" in recorded_headers

    def test_base_url_prepended(self) -> None:
        """Base URL is prepended to request path."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            base_url="https://api.example.com",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("/v1/process", json={"data": "value"})

        # Verify full URL was recorded
        call_kwargs = recorder.record_call.call_args[1]
        assert (
            call_kwargs["request_data"]["url"] == "https://api.example.com/v1/process"
        )

        # Verify httpx was called with full URL
        mock_client.post.assert_called_once()
        actual_url = mock_client.post.call_args[0][0]
        assert actual_url == "https://api.example.com/v1/process"

    def test_response_body_size_recorded(self) -> None:
        """Response body size is recorded in bytes."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"result": "data", "count": 42}'

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://api.example.com/v1/process")

        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["response_data"]["body_size"] == len(mock_response.content)

    def test_request_headers_merged(self) -> None:
        """Per-request headers are merged with default headers."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            headers={"Content-Type": "application/json"},
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post(
                "https://api.example.com/v1/process",
                headers={"X-Custom-Header": "custom-value"},
            )

        # Verify headers were passed to httpx
        mock_client.post.assert_called_once()
        actual_headers = mock_client.post.call_args[1]["headers"]
        assert actual_headers["Content-Type"] == "application/json"
        assert actual_headers["X-Custom-Header"] == "custom-value"

    def test_timeout_passed_to_client(self) -> None:
        """Timeout configuration is passed to httpx client."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            timeout=60.0,
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://api.example.com/v1/process")

        # Verify timeout was passed to Client constructor
        mock_client_class.assert_called_once_with(timeout=60.0)

    def test_response_headers_recorded(self) -> None:
        """Response headers are recorded in audit trail."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers(
            {"content-type": "application/json", "x-request-id": "req-456"}
        )
        mock_response.content = b"{}"

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://api.example.com/v1/process")

        call_kwargs = recorder.record_call.call_args[1]
        response_headers = call_kwargs["response_data"]["headers"]
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "req-456"

    def test_no_base_url_uses_full_url(self) -> None:
        """When no base_url, the full URL is used as-is."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
            # No base_url
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://other-api.example.com/endpoint")

        call_kwargs = recorder.record_call.call_args[1]
        assert (
            call_kwargs["request_data"]["url"]
            == "https://other-api.example.com/endpoint"
        )

    def test_none_json_body(self) -> None:
        """HTTP call with None json body is handled."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Call without json
            client.post("https://api.example.com/endpoint")

        call_kwargs = recorder.record_call.call_args[1]
        assert call_kwargs["request_data"]["json"] is None

    def test_sensitive_response_headers_filtered(self) -> None:
        """Sensitive response headers (cookies, auth) are filtered from audit trail."""
        recorder = self._create_mock_recorder()

        client = AuditedHTTPClient(
            recorder=recorder,
            state_id="state_123",
        )

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "x-request-id": "req-456",
                "set-cookie": "session=secret-session-token; HttpOnly",
                "www-authenticate": "Bearer realm=api",
                "x-auth-token": "sensitive-token-value",
            }
        )
        mock_response.content = b"{}"

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            client.post("https://api.example.com/v1/process")

        call_kwargs = recorder.record_call.call_args[1]
        response_headers = call_kwargs["response_data"]["headers"]

        # Sensitive headers should NOT be recorded
        assert "set-cookie" not in response_headers
        assert "www-authenticate" not in response_headers
        assert "x-auth-token" not in response_headers

        # Non-sensitive headers SHOULD be recorded
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "req-456"
