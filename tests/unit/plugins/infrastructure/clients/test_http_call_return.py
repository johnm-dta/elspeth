"""Tests for AuditedHTTPClient.get_ssrf_safe() Call return."""

from unittest.mock import MagicMock, patch

import httpx

from elspeth.contracts.audit import Call
from elspeth.core.security.web import SSRFSafeRequest
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


class TestGetSsrfSafeCallReturn:
    """Verify get_ssrf_safe() returns a Call with request/response refs.

    Uses Mock() for the recorder — AuditedHTTPClient is what's being tested,
    not the recorder. The recorder mock returns a Call with known ref hashes.
    """

    def _make_mock_call(self) -> Call:
        """Create a mock Call with known request/response refs."""
        from datetime import UTC, datetime

        from elspeth.contracts import CallStatus, CallType

        return Call(
            call_id="test-call-id",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_hash="test-request-hash",
            created_at=datetime.now(UTC),
            state_id="state-1",
            request_ref="test-request-ref-hash",
            response_hash="test-response-hash",
            response_ref="test-response-ref-hash",
            latency_ms=100.0,
        )

    def _make_client_with_mock_recorder(self) -> tuple[AuditedHTTPClient, MagicMock]:
        """Create AuditedHTTPClient with a mock recorder that returns a known Call."""
        mock_recorder = MagicMock()
        mock_call = self._make_mock_call()
        mock_recorder.record_call.return_value = mock_call
        mock_recorder.allocate_call_index.return_value = 0

        client = AuditedHTTPClient(
            execution=mock_recorder,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=lambda event: None,
            timeout=5.0,
        )
        return client, mock_recorder

    def test_returns_three_tuple_with_call_on_success(self):
        """get_ssrf_safe() returns (Response, str, Call) on success."""
        client, _mock_recorder = self._make_client_with_mock_recorder()

        # Mock the actual HTTP call
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html>test</html>"
        mock_response.text = "<html>test</html>"
        mock_response.url = httpx.URL("http://93.184.216.34/")
        mock_response.is_success = True
        mock_response.is_redirect = False

        safe_request = SSRFSafeRequest(
            original_url="http://example.com/",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/",
            scheme="http",
            bare_hostname="example.com",
        )

        with patch("httpx.Client") as mock_client_class:
            mock_http_instance = MagicMock()
            mock_http_instance.__enter__ = MagicMock(return_value=mock_http_instance)
            mock_http_instance.__exit__ = MagicMock(return_value=False)
            mock_http_instance.get.return_value = mock_response
            mock_client_class.return_value = mock_http_instance

            result = client.get_ssrf_safe(safe_request)

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        _response, final_url, call = result
        assert isinstance(final_url, str)
        assert isinstance(call, Call)
        assert call.request_ref == "test-request-ref-hash"
        assert call.response_ref == "test-response-ref-hash"

    def test_record_and_emit_returns_call(self):
        """_record_and_emit() returns a Call object."""
        client, _mock_recorder = self._make_client_with_mock_recorder()

        from elspeth.contracts import CallStatus
        from elspeth.contracts.call_data import HTTPCallRequest

        request_dto = HTTPCallRequest(
            method="GET",
            url="http://example.com/",
            headers={},
        )

        result = client._record_and_emit(
            call_index=0,
            full_url="http://example.com/",
            request_data=request_dto.to_dict(),
            response=None,
            response_data=None,
            error_data=None,
            latency_ms=10.0,
            call_status=CallStatus.SUCCESS,
            request_payload=request_dto,
        )

        assert isinstance(result, Call)
        assert result.request_ref == "test-request-ref-hash"
