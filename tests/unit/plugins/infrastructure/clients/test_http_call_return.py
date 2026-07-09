"""Tests for AuditedHTTPClient.get_ssrf_safe() Call return."""

from unittest.mock import patch

import httpx

from elspeth.contracts import CallStatus
from elspeth.contracts.audit import Call
from elspeth.core.security.web import SSRFSafeRequest
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


class _CallRecord:
    def __init__(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        self.args = args
        self.kwargs = kwargs


class _CallRecorder:
    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value
        self.call_args: _CallRecord | None = None
        self.call_args_list: list[_CallRecord] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        record = _CallRecord(args, kwargs)
        self.call_args = record
        self.call_args_list.append(record)
        return self.return_value

    def assert_called_once(self) -> None:
        assert len(self.call_args_list) == 1


class _ExecutionRecorderDouble:
    def __init__(self, call: Call) -> None:
        self.record_call = _CallRecorder(call)
        self.allocate_call_index = _CallRecorder(0)


class _HTTPClientDouble:
    def __init__(self, response: httpx.Response) -> None:
        self.get = _CallRecorder(response)

    def __enter__(self) -> "_HTTPClientDouble":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


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

    def _make_client_with_recorder(self) -> tuple[AuditedHTTPClient, _ExecutionRecorderDouble]:
        """Create AuditedHTTPClient with a recorder double that returns a known Call."""
        mock_call = self._make_mock_call()
        recorder = _ExecutionRecorderDouble(mock_call)

        client = AuditedHTTPClient(
            execution=recorder,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=lambda event: None,
            timeout=5.0,
        )
        return client, recorder

    def test_returns_three_tuple_with_call_on_success(self):
        """get_ssrf_safe() returns (Response, str, Call) on success."""
        client, _recorder = self._make_client_with_recorder()

        response = httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<html>test</html>",
            request=httpx.Request("GET", "http://93.184.216.34/"),
        )

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
            mock_client_class.return_value = _HTTPClientDouble(response)

            result = client.get_ssrf_safe(safe_request)

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        _response, final_url, call = result
        assert isinstance(final_url, str)
        assert isinstance(call, Call)
        assert call.request_ref == "test-request-ref-hash"
        assert call.response_ref == "test-response-ref-hash"

    def test_record_and_emit_returns_call(self):
        """_record_and_emit() returns a Call object."""
        client, _recorder = self._make_client_with_recorder()

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

    def test_nonstandard_three_digit_status_is_recorded(self):
        """Remote 6xx/9xx statuses must not crash before audit recording."""
        client, recorder = self._make_client_with_recorder()
        response = httpx.Response(
            999,
            headers={"content-type": "text/plain"},
            text="nonstandard",
            request=httpx.Request("GET", "https://example.com/nonstandard"),
        )
        client._client.get = _CallRecorder(response)

        result = client.get("https://example.com/nonstandard")

        assert result is response
        recorder.record_call.assert_called_once()
        assert recorder.record_call.call_args is not None
        record_kwargs = recorder.record_call.call_args.kwargs
        assert record_kwargs["status"] is CallStatus.ERROR
        assert record_kwargs["response_data"].status_code == 999
        assert record_kwargs["error"].status_code == 999

    def test_ssrf_safe_nonstandard_three_digit_status_is_recorded(self):
        """SSRF-safe requests also preserve non-standard three-digit statuses in audit."""
        client, recorder = self._make_client_with_recorder()
        response = httpx.Response(
            999,
            headers={"content-type": "text/plain"},
            text="nonstandard",
            request=httpx.Request("GET", "http://93.184.216.34/"),
        )
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
            mock_client_class.return_value = _HTTPClientDouble(response)

            result, _final_url, call = client.get_ssrf_safe(safe_request)

        assert result is response
        assert isinstance(call, Call)
        recorder.record_call.assert_called_once()
        assert recorder.record_call.call_args is not None
        record_kwargs = recorder.record_call.call_args.kwargs
        assert record_kwargs["status"] is CallStatus.ERROR
        assert record_kwargs["response_data"].status_code == 999
        assert record_kwargs["error"].status_code == 999
