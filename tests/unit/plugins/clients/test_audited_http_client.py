# tests/plugins/clients/test_audited_http_client.py
"""Tests for AuditedHTTPClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


class TestAuditedHTTPClient:
    """Tests for AuditedHTTPClient."""

    def _create_mock_execution(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        import itertools

        mock_execution = MagicMock()
        mock_execution.record_call = MagicMock()
        counter = itertools.count()
        mock_execution.allocate_call_index.side_effect = lambda _: next(counter)
        return mock_execution

    def test_successful_post_records_to_audit_trail(self) -> None:
        """Successful HTTP POST is recorded to audit trail with full response body."""
        mock_execution = self._create_mock_execution()

        # Mock httpx.Client with JSON response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"result": "success"}'
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                timeout=30.0,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post(
                "https://api.example.com/v1/process",
                json={"data": "value"},
            )

        # Verify response
        assert response.status_code == 200

        # Verify audit record
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["state_id"] == "state_123"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] == CallType.HTTP
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["request_data"].to_dict()["method"] == "POST"
        assert call_kwargs["request_data"].to_dict()["url"] == "https://api.example.com/v1/process"
        assert call_kwargs["request_data"].to_dict()["json"] == {"data": "value"}
        assert call_kwargs["response_data"].to_dict()["status_code"] == 200
        assert call_kwargs["response_data"].to_dict()["body"] == {"result": "success"}
        assert call_kwargs["latency_ms"] > 0

    def test_call_index_increments(self) -> None:
        """Each call gets a unique, incrementing call index."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Make multiple calls
            client.post("https://api.example.com/first")
            client.post("https://api.example.com/second")

        # Check call indices
        calls = mock_execution.record_call.call_args_list
        assert len(calls) == 2
        assert calls[0][1]["call_index"] == 0
        assert calls[1][1]["call_index"] == 1

    def test_failed_call_records_error(self) -> None:
        """Failed HTTP call records error details."""
        mock_execution = self._create_mock_execution()

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(httpx.ConnectError):
                client.post("https://api.example.com/v1/process")

        # Verify error was recorded
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "ConnectError"
        assert "Connection refused" in call_kwargs["error"].message

    def test_internal_success_path_bug_crashes_without_recording_synthetic_http_error(self) -> None:
        """ELSPETH success-path bugs must not be misclassified as provider failures."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"ok": true}'
        mock_response.text = '{"ok": true}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            with (
                patch.object(client, "_filter_response_headers", side_effect=TypeError("header filter bug")),
                pytest.raises(TypeError, match="header filter bug"),
            ):
                client.post("https://api.example.com/v1/process")

        mock_execution.record_call.assert_not_called()

    def test_auth_headers_fingerprinted_in_recorded_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Auth headers are fingerprinted (not removed) so different credentials produce different hashes.

        This is critical for audit integrity: requests with different credentials must
        have different request_hash values to distinguish them in replay/verify mode.
        The raw secret values must NOT appear in the audit trail.
        """
        # Set fingerprint key for deterministic testing, ensure dev mode is off
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key-for-http-client")
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={
                    "Authorization": "Bearer secret-token",
                    "X-API-Key": "api-key-12345",
                    "Content-Type": "application/json",
                    "X-Request-Id": "req-123",
                },
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_headers = call_kwargs["request_data"].to_dict()["headers"]

        # Auth headers should be FINGERPRINTED, not removed
        # The fingerprint format is "<fingerprint:64hexchars>"
        assert "Authorization" in recorded_headers
        assert "X-API-Key" in recorded_headers

        # Raw secrets must NOT appear
        assert "Bearer secret-token" not in recorded_headers["Authorization"]
        assert "api-key-12345" not in recorded_headers["X-API-Key"]

        # Fingerprints should be in the expected format
        assert recorded_headers["Authorization"].startswith("<fingerprint:")
        assert recorded_headers["X-API-Key"].startswith("<fingerprint:")
        # Fingerprints are 64 hex chars
        assert len(recorded_headers["Authorization"]) == len("<fingerprint:") + 64 + len(">")

        # Non-auth headers SHOULD be recorded unchanged
        assert recorded_headers["Content-Type"] == "application/json"
        assert recorded_headers["X-Request-Id"] == "req-123"

    def test_different_auth_headers_produce_different_request_hashes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requests with different auth credentials must produce different request_hash values.

        This is the core bug fix: previously, auth headers were removed entirely,
        causing identical hashes for requests that differed only by credentials.
        With fingerprinting, each unique credential produces a unique fingerprint,
        which produces a unique request_hash.
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key-for-http-client")
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        from elspeth.core.canonical import stable_hash

        mock_execution1 = self._create_mock_execution()
        mock_execution2 = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            # Client 1 with credential A
            client1 = AuditedHTTPClient(
                execution=mock_execution1,
                state_id="state_1",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={"Authorization": "Bearer credential-A"},
            )

            # Client 2 with credential B (different)
            client2 = AuditedHTTPClient(
                execution=mock_execution2,
                state_id="state_2",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={"Authorization": "Bearer credential-B"},
            )

            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Same URL and body, different credentials
            client1.post("https://api.example.com/v1/process", json={"data": "test"})
            client2.post("https://api.example.com/v1/process", json={"data": "test"})

        # Get the recorded request_data for each call (now CallPayload DTOs)
        request_data_1 = mock_execution1.record_call.call_args[1]["request_data"].to_dict()
        request_data_2 = mock_execution2.record_call.call_args[1]["request_data"].to_dict()

        # The request_hash values MUST be different
        hash1 = stable_hash(request_data_1)
        hash2 = stable_hash(request_data_2)

        assert hash1 != hash2, (
            "CRITICAL: Requests with different credentials produced identical hashes! This breaks replay/verify mode and audit integrity."
        )

    def test_auth_headers_removed_when_no_fingerprint_key_dev_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """In dev mode (ELSPETH_ALLOW_RAW_SECRETS=true), auth headers are removed without fingerprinting.

        This allows development without setting up fingerprint keys while still
        not leaking secrets into the audit trail.
        """
        # No fingerprint key, but dev mode enabled
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={
                    "Authorization": "Bearer secret-token",
                    "Content-Type": "application/json",
                },
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_headers = call_kwargs["request_data"].to_dict()["headers"]

        # In dev mode, auth headers are removed (no fingerprint available)
        assert "Authorization" not in recorded_headers

        # Non-auth headers still recorded
        assert recorded_headers["Content-Type"] == "application/json"

    def test_sensitive_header_without_fingerprint_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sensitive headers without fingerprint key (and not dev mode) raise FrameworkBugError.

        Regression test for elspeth-780f6e39b1: previously this path silently
        dropped the header with only a logger.warning, leaving no audit trail
        for missing auth configuration.
        """
        from elspeth.contracts.errors import FrameworkBugError

        # No fingerprint key AND not dev mode
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={
                    "Authorization": "Bearer secret-token",
                    "Content-Type": "application/json",
                },
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            with pytest.raises(FrameworkBugError, match="Sensitive header 'Authorization' cannot be fingerprinted"):
                client.post("https://api.example.com/v1/process")

    def test_base_url_prepended(self) -> None:
        """Base URL is prepended to request path."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                base_url="https://api.example.com",
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("/v1/process", json={"data": "value"})

        # Verify full URL was recorded
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["url"] == "https://api.example.com/v1/process"

        # Verify httpx was called with full URL
        mock_client.post.assert_called_once()
        actual_url = mock_client.post.call_args[0][0]
        assert actual_url == "https://api.example.com/v1/process"

    def test_base_url_trailing_slash_url_leading_slash(self) -> None:
        """Base URL with trailing slash + URL with leading slash produces single slash."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            # Both have slashes - would cause double slash with naive concat
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                base_url="https://api.example.com/",
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("/v1/process")

        # Should have exactly one slash, not double
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["url"] == "https://api.example.com/v1/process"

        actual_url = mock_client.post.call_args[0][0]
        assert actual_url == "https://api.example.com/v1/process"

    def test_base_url_no_trailing_slash_url_no_leading_slash(self) -> None:
        """Base URL without trailing slash + URL without leading slash produces correct URL."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            # Neither has slashes - would cause missing slash with naive concat
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                base_url="https://api.example.com/v1",
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("process")

        # Should have slash separator inserted
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["url"] == "https://api.example.com/v1/process"

        actual_url = mock_client.post.call_args[0][0]
        assert actual_url == "https://api.example.com/v1/process"

    def test_response_body_size_recorded(self) -> None:
        """Response body size is recorded in bytes."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b'{"result": "data", "count": 42}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["response_data"].to_dict()["body_size"] == len(mock_response.content)

    def test_request_headers_merged(self) -> None:
        """Per-request headers are merged with default headers."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={"Content-Type": "application/json"},
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

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
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                timeout=60.0,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        # Verify timeout was passed to Client constructor
        mock_client_class.assert_called_once_with(timeout=60.0, follow_redirects=False)

    def test_response_headers_recorded(self) -> None:
        """Response headers are recorded in audit trail."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({"content-type": "application/json", "x-request-id": "req-456"})
        mock_response.content = b"{}"
        mock_response.text = "{}"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        call_kwargs = mock_execution.record_call.call_args[1]
        response_headers = call_kwargs["response_data"].to_dict()["headers"]
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "req-456"

    def test_no_base_url_uses_full_url(self) -> None:
        """When no base_url, the full URL is used as-is."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                # No base_url
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://other-api.example.com/endpoint")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["url"] == "https://other-api.example.com/endpoint"

    def test_none_json_body(self) -> None:
        """HTTP call with None json body is handled."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Call without json
            client.post("https://api.example.com/endpoint")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["json"] is None

    def test_sensitive_response_headers_filtered(self) -> None:
        """Sensitive response headers (cookies, auth) are filtered from audit trail."""
        mock_execution = self._create_mock_execution()

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
        mock_response.text = "{}"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/v1/process")

        call_kwargs = mock_execution.record_call.call_args[1]
        response_headers = call_kwargs["response_data"].to_dict()["headers"]

        # Sensitive headers should NOT be recorded
        assert "set-cookie" not in response_headers
        assert "www-authenticate" not in response_headers
        assert "x-auth-token" not in response_headers

        # Non-sensitive headers SHOULD be recorded
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "req-456"

    def test_per_request_timeout_overrides_default(self) -> None:
        """Per-request timeout overrides the client's default timeout."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            # Client has default timeout of 30s
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                timeout=30.0,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Request with per-request timeout of 120s
            client.post("https://api.example.com/endpoint", timeout=120.0)

        # Verify per-request timeout was passed to the post() call
        mock_client.post.assert_called_once()
        assert mock_client.post.call_args[1]["timeout"] == 120.0

    def test_none_timeout_uses_default(self) -> None:
        """When timeout=None is passed, the client's default timeout is used."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                timeout=45.0,  # Default
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Request without explicit timeout
            client.post("https://api.example.com/endpoint")

        # Verify default timeout was passed to the post() call
        mock_client.post.assert_called_once()
        assert mock_client.post.call_args[1]["timeout"] == 45.0

    def test_non_json_response_body_recorded_as_text(self) -> None:
        """Non-JSON response body is recorded as text."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"Plain text response"
        mock_response.text = "Plain text response"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/endpoint")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["response_data"].to_dict()["body"] == "Plain text response"

    def test_json_response_body_recorded_as_dict(self) -> None:
        """JSON response body is recorded as parsed dict."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}
        mock_response.content = b'{"choices": [{"message": {"content": "Hello"}}]}'
        mock_response.text = '{"choices": [{"message": {"content": "Hello"}}]}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/endpoint")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["response_data"].to_dict()["body"] == {"choices": [{"message": {"content": "Hello"}}]}

    def test_4xx_response_recorded_as_error(self) -> None:
        """HTTP 4xx responses are recorded with ERROR status."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"error": "Unauthorized"}'
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_response.text = '{"error": "Unauthorized"}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/v1/process")

        # Response is still returned (caller decides what to do)
        assert response.status_code == 401

        # Verify ERROR status was recorded
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["response_data"].to_dict()["status_code"] == 401
        assert call_kwargs["error"].type == "HTTPError"
        assert call_kwargs["error"].status_code == 401
        assert "401" in call_kwargs["error"].message

    def test_5xx_response_recorded_as_error(self) -> None:
        """HTTP 5xx responses are recorded with ERROR status."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 503
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"Service Unavailable"
        mock_response.text = "Service Unavailable"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/v1/process")

        # Response is still returned
        assert response.status_code == 503

        # Verify ERROR status was recorded
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["response_data"].to_dict()["status_code"] == 503
        assert call_kwargs["error"].type == "HTTPError"
        assert call_kwargs["error"].status_code == 503

    def test_2xx_responses_recorded_as_success(self) -> None:
        """HTTP 2xx responses (200-299) are recorded with SUCCESS status."""
        mock_execution = self._create_mock_execution()

        # Test 201 Created
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.headers = {}
        mock_response.content = b"{}"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/v1/resource")

        assert response.status_code == 201

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["error"] is None

    def test_3xx_responses_recorded_as_success(self) -> None:
        """HTTP 3xx responses (redirects) are currently recorded as non-2xx = ERROR.

        Note: httpx follows redirects by default, so this tests the case
        when redirects are not followed.
        """
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 302
        mock_response.headers = {"location": "https://new-location.example.com"}
        mock_response.content = b""
        mock_response.text = ""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/v1/resource")

        assert response.status_code == 302

        # 3xx is not in 2xx range, so recorded as ERROR
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].status_code == 302

    def test_large_text_response_not_truncated(self) -> None:
        """Large text responses (>100KB) are recorded completely, not truncated.

        This test verifies that the HTTP client does not truncate large non-JSON
        responses before passing them to record_call(). The payload store auto-persist
        mechanism in ExecutionRepository is designed to handle large payloads.
        """
        mock_execution = self._create_mock_execution()

        # Create 150KB text response (exceeds old 100KB truncation limit)
        large_text = "x" * 150_000

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = large_text.encode("utf-8")
        mock_response.text = large_text

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/large-response")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]

        # CRITICAL: Must record ALL 150KB, not truncated to 100KB
        assert len(recorded_body) == 150_000, f"Expected 150000 chars, got {len(recorded_body)}"
        assert recorded_body == large_text

    def test_large_json_response_not_truncated(self) -> None:
        """Large JSON responses (>100KB) are recorded as complete dict.

        This test verifies that large JSON responses are parsed and recorded
        completely without truncation.
        """
        import json

        mock_execution = self._create_mock_execution()

        # Create large JSON with >100KB when serialized
        large_json = {"items": [{"id": i, "data": "x" * 1000} for i in range(200)]}
        json_str = json.dumps(large_json)
        assert len(json_str) > 100_000, "Test setup error: JSON not large enough"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_str.encode("utf-8")
        mock_response.text = json_str

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/large-json")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]

        # JSON should be parsed as dict, not truncated
        assert isinstance(recorded_body, dict)
        assert len(recorded_body["items"]) == 200
        assert recorded_body == large_json

    def test_binary_response_recorded_as_base64(self) -> None:
        """Binary responses (images, PDFs) are recorded as base64.

        This test verifies that binary content (which cannot be decoded as UTF-8)
        is properly handled by encoding it as base64 for JSON serialization.
        """
        import base64

        mock_execution = self._create_mock_execution()

        # Simulate a small PNG image (binary data that's not valid UTF-8)
        # PNG file signature + minimal IHDR chunk
        binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/png"}
        mock_response.content = binary_data
        # Note: .text would raise UnicodeDecodeError for real binary data
        # For this test, we're verifying the code path that uses .content instead

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            client.post("https://api.example.com/image")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]

        # Binary should be encoded as base64 in a dict
        assert isinstance(recorded_body, dict), f"Expected dict, got {type(recorded_body)}"
        assert "_binary" in recorded_body, "Expected '_binary' key in response body"

        # Verify we can decode it back to original binary data
        decoded = base64.b64decode(recorded_body["_binary"])
        assert decoded == binary_data, "Decoded binary data doesn't match original"

    def test_json_response_with_nan_recorded_as_parse_failure(self) -> None:
        """JSON response containing NaN is recorded as parse failure.

        This is a P1 bug fix (P1-2026-02-05): HTTP client must reject NaN/Infinity
        at the Tier 3 boundary because canonicalization cannot handle them. Recording
        as parse failure with raw text preserved ensures audit completeness without
        crashing.
        """
        mock_execution = self._create_mock_execution()

        # JSON with NaN (Python's json.loads accepts this)
        json_with_nan = '{"value": NaN, "other": "data"}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_with_nan.encode("utf-8")
        mock_response.text = json_with_nan

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Should NOT crash - should record as parse failure
            response = client.post("https://api.example.com/endpoint")

        # Response object is still returned to caller
        assert response.status_code == 200

        # Verify audit record was created successfully (not crashed)
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]

        # Body should be recorded as parse failure with raw text
        recorded_body = call_kwargs["response_data"].to_dict()["body"]
        assert isinstance(recorded_body, dict)
        assert recorded_body["_json_parse_failed"] is True
        assert "NaN" in recorded_body["_error"] or "non-finite" in recorded_body["_error"]
        assert recorded_body["_raw_text"] == json_with_nan

    def test_json_response_with_infinity_recorded_as_parse_failure(self) -> None:
        """JSON response containing Infinity is recorded as parse failure.

        Similar to NaN test - Infinity is also non-canonicalizable and must be
        rejected at the HTTP boundary.
        """
        mock_execution = self._create_mock_execution()

        # JSON with Infinity (Python's json.loads accepts this)
        json_with_infinity = '{"value": Infinity, "count": 42}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_with_infinity.encode("utf-8")
        mock_response.text = json_with_infinity

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            # Should NOT crash - should record as parse failure
            response = client.post("https://api.example.com/endpoint")

        assert response.status_code == 200
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]

        recorded_body = call_kwargs["response_data"].to_dict()["body"]
        assert recorded_body["_json_parse_failed"] is True
        assert "Infinity" in recorded_body["_error"] or "non-finite" in recorded_body["_error"]
        assert recorded_body["_raw_text"] == json_with_infinity

    def test_json_response_with_negative_infinity_recorded_as_parse_failure(self) -> None:
        """JSON response containing -Infinity is recorded as parse failure."""
        mock_execution = self._create_mock_execution()

        json_with_neg_infinity = '{"value": -Infinity}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_with_neg_infinity.encode("utf-8")
        mock_response.text = json_with_neg_infinity

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/endpoint")

        assert response.status_code == 200
        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]
        assert recorded_body["_json_parse_failed"] is True
        assert recorded_body["_raw_text"] == json_with_neg_infinity

    def test_json_response_with_nested_nan_recorded_as_parse_failure(self) -> None:
        """JSON response with NaN nested in object/array is detected and rejected.

        NaN can appear anywhere in the JSON structure - we must check recursively.
        """
        mock_execution = self._create_mock_execution()

        # NaN nested in array within object
        json_with_nested_nan = '{"data": {"values": [1, 2, NaN, 4]}}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_with_nested_nan.encode("utf-8")
        mock_response.text = json_with_nested_nan

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/endpoint")

        assert response.status_code == 200
        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]
        assert recorded_body["_json_parse_failed"] is True
        assert recorded_body["_raw_text"] == json_with_nested_nan

    def test_valid_json_response_still_works(self) -> None:
        """Ensure the NaN/Infinity fix doesn't break normal JSON parsing.

        Regression test to verify that valid JSON (including edge cases like
        null, empty objects, negative numbers) is still parsed correctly.
        """
        mock_execution = self._create_mock_execution()

        # Valid JSON with edge cases
        valid_json = '{"null_value": null, "negative": -42.5, "empty": {}, "array": [1, 2, 3]}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = valid_json.encode("utf-8")
        mock_response.text = valid_json

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.post.return_value = mock_response

            response = client.post("https://api.example.com/endpoint")

        assert response.status_code == 200
        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_body = call_kwargs["response_data"].to_dict()["body"]

        # Should be parsed as dict, NOT as parse failure
        assert isinstance(recorded_body, dict)
        assert "_json_parse_failed" not in recorded_body
        assert recorded_body["null_value"] is None
        assert recorded_body["negative"] == -42.5
        assert recorded_body["empty"] == {}
        assert recorded_body["array"] == [1, 2, 3]


class TestAuditedHTTPClientGet:
    """Tests for AuditedHTTPClient.get() method."""

    def _create_mock_execution(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        import itertools

        mock_execution = MagicMock()
        mock_execution.record_call = MagicMock()
        counter = itertools.count()
        mock_execution.allocate_call_index.side_effect = lambda _: next(counter)
        return mock_execution

    def test_successful_get_records_to_audit_trail(self) -> None:
        """Successful HTTP GET is recorded to audit trail with full response body."""
        mock_execution = self._create_mock_execution()

        # Mock httpx.Client with HTML response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html><body>Content</body></html>"
        mock_response.text = "<html><body>Content</body></html>"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                timeout=30.0,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            response = client.get("https://example.com/page")

        # Verify response
        assert response.status_code == 200
        assert response.text == "<html><body>Content</body></html>"

        # Verify audit record
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["state_id"] == "state_123"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] == CallType.HTTP
        assert call_kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs["request_data"].to_dict()["method"] == "GET"
        assert call_kwargs["request_data"].to_dict()["url"] == "https://example.com/page"
        assert call_kwargs["request_data"].to_dict()["params"] is None
        assert call_kwargs["response_data"].to_dict()["status_code"] == 200
        assert call_kwargs["response_data"].to_dict()["body"] == "<html><body>Content</body></html>"
        assert call_kwargs["latency_ms"] > 0

    def test_get_with_query_params(self) -> None:
        """GET with query params records params in audit trail and passes to httpx."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"result": "ok"}'
        mock_response.text = '{"result": "ok"}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            response = client.get("https://api.example.com/search", params={"q": "test", "limit": 10})

        # Verify response
        assert response.status_code == 200

        # Verify params were passed to httpx
        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://api.example.com/search"
        assert call_args[1]["params"] == {"q": "test", "limit": 10}

        # Verify params were recorded in audit trail
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["params"] == {"q": "test", "limit": 10}

    def test_get_with_base_url(self) -> None:
        """GET with base_url prepends base to path correctly."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                base_url="https://api.example.com",
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            client.get("/v1/resource")

        # Verify full URL was recorded
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["request_data"].to_dict()["url"] == "https://api.example.com/v1/resource"

        # Verify httpx was called with full URL
        mock_client.get.assert_called_once()
        actual_url = mock_client.get.call_args[0][0]
        assert actual_url == "https://api.example.com/v1/resource"

    def test_get_failed_call_records_error(self) -> None:
        """Failed HTTP GET call records error details."""
        mock_execution = self._create_mock_execution()

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(httpx.ConnectError):
                client.get("https://example.com/page")

        # Verify error was recorded
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"].type == "ConnectError"
        assert "Connection refused" in call_kwargs["error"].message

    def test_get_auth_headers_fingerprinted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GET requests fingerprint auth headers in audit trail."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key-for-http-client")
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.content = b""

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
                headers={"Authorization": "Bearer secret-token"},
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            client.get("https://api.example.com/resource")

        call_kwargs = mock_execution.record_call.call_args[1]
        recorded_headers = call_kwargs["request_data"].to_dict()["headers"]

        # Auth header should be fingerprinted
        assert "Authorization" in recorded_headers
        assert "Bearer secret-token" not in recorded_headers["Authorization"]
        assert recorded_headers["Authorization"].startswith("<fingerprint:")

    def test_get_json_response_recorded_as_dict(self) -> None:
        """GET with JSON response records parsed dict."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = b'{"items": [1, 2, 3]}'
        mock_response.text = '{"items": [1, 2, 3]}'

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            client.get("https://api.example.com/data")

        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["response_data"].to_dict()["body"] == {"items": [1, 2, 3]}

    def test_get_4xx_response_recorded_as_error(self) -> None:
        """GET with 4xx response is recorded with ERROR status."""
        mock_execution = self._create_mock_execution()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"Not Found"
        mock_response.text = "Not Found"

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            response = client.get("https://example.com/missing")

        # Response is still returned
        assert response.status_code == 404

        # Verify ERROR status was recorded
        call_kwargs = mock_execution.record_call.call_args[1]
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["response_data"].to_dict()["status_code"] == 404
        assert call_kwargs["error"].type == "HTTPError"
        assert call_kwargs["error"].status_code == 404

    def test_get_json_response_with_nan_recorded_as_parse_failure(self) -> None:
        """GET JSON response containing NaN is recorded as parse failure.

        Ensures the NaN/Infinity fix applies to GET requests as well as POST.
        """
        mock_execution = self._create_mock_execution()

        json_with_nan = '{"metric": NaN}'

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = json_with_nan.encode("utf-8")
        mock_response.text = json_with_nan

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=mock_execution,
                state_id="state_123",
                run_id="run_abc",
                telemetry_emit=lambda event: None,
            )
            mock_client = mock_client_class.return_value
            mock_client.get.return_value = mock_response

            response = client.get("https://api.example.com/metrics")

        assert response.status_code == 200
        mock_execution.record_call.assert_called_once()
        call_kwargs = mock_execution.record_call.call_args[1]

        recorded_body = call_kwargs["response_data"].to_dict()["body"]
        assert recorded_body["_json_parse_failed"] is True
        assert "NaN" in recorded_body["_error"] or "non-finite" in recorded_body["_error"]
        assert recorded_body["_raw_text"] == json_with_nan
