"""Tests for redirect URL resolution and audit recording in _follow_redirects_safe().

Verifies that:
1. Relative redirects resolve against the original hostname URL, not the
   IP-based connection URL.
2. Each redirect hop is individually recorded in the audit trail as
   CallType.HTTP_REDIRECT with correct lineage data.
"""

from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.core.security.web import SSRFSafeRequest
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


@pytest.fixture
def http_client():
    """Create AuditedHTTPClient with mocked dependencies.

    Patches httpx.Client so that:
    - The shared self._client (created in __init__) is a mock
    - Ephemeral clients created via `with httpx.Client(...) as c:` return
      a controllable mock — accessible as http_client._ephemeral_mock
    """
    ephemeral_mock = MagicMock()
    # Context manager protocol: __enter__ returns the mock itself
    ephemeral_mock.__enter__ = Mock(return_value=ephemeral_mock)
    ephemeral_mock.__exit__ = Mock(return_value=False)

    with patch("elspeth.plugins.infrastructure.clients.http.httpx.Client") as MockClient:
        # First call: shared client in __init__
        # Subsequent calls: ephemeral clients in get_ssrf_safe / _follow_redirects_safe
        shared_mock = MagicMock()
        MockClient.side_effect = [shared_mock, ephemeral_mock, ephemeral_mock, ephemeral_mock, ephemeral_mock, ephemeral_mock]

        execution = Mock()
        execution.record_call = Mock()
        client = AuditedHTTPClient(
            execution=execution,
            state_id="test-state-001",
            run_id="test-run-001",
            telemetry_emit=Mock(),
            timeout=30.0,
        )
        # Expose the ephemeral mock for tests to set up return values
        client._ephemeral_mock = ephemeral_mock
        yield client


def _make_redirect_response(location: str, status_code: int = 301, url: str = "https://93.184.216.34:443/old-path") -> httpx.Response:
    """Create a redirect response with an IP-based URL (as httpx would see it)."""
    request = httpx.Request("GET", url)
    return httpx.Response(
        status_code=status_code,
        headers={"location": location},
        request=request,
    )


def _make_final_response(url: str = "https://93.184.216.34:443/final") -> httpx.Response:
    """Create a 200 OK response."""
    request = httpx.Request("GET", url)
    return httpx.Response(200, text="OK", request=request)


def _make_ssrf_request(url: str, ip: str = "93.184.216.34") -> SSRFSafeRequest:
    """Create an SSRFSafeRequest for the given URL."""
    parsed = httpx.URL(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = str(parsed.raw_path, "ascii") if parsed.raw_path else "/"
    return SSRFSafeRequest(
        original_url=url,
        resolved_ip=ip,
        host_header=parsed.host,
        port=port,
        path=path,
        scheme=parsed.scheme,
        bare_hostname=parsed.host,
    )


class TestRelativeRedirectResolution:
    """Relative redirects must resolve against hostname, not IP."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_relative_redirect_resolves_against_hostname(self, mock_validate, http_client):
        """Location: /new-path should produce https://example.com/new-path, not https://93.184.216.34/new-path."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://example.com/new-path")
        http_client._ephemeral_mock.get.return_value = final_response

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test"},
            original_url="https://example.com/old-path",
        )

        # validate_url_for_ssrf should receive hostname-based URL
        mock_validate.assert_called_once_with("https://example.com/new-path", allowed_ranges=())
        assert result.status_code == 200
        assert count == 1

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_relative_redirect_preserves_scheme_and_host(self, mock_validate, http_client):
        """Relative redirect should preserve scheme and host from original URL."""
        redirect_response = _make_redirect_response("/api/v2/resource")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://api.example.com/api/v2/resource")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://api.example.com/api/v1/resource",
        )

        mock_validate.assert_called_once_with("https://api.example.com/api/v2/resource", allowed_ranges=())


class TestAbsoluteRedirectResolution:
    """Absolute redirects carry their own hostname — should work regardless."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_absolute_redirect_to_different_host(self, mock_validate, http_client):
        """Location: https://other.com/page should use other.com, not original host."""
        redirect_response = _make_redirect_response("https://other.com/page")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://other.com/page", ip="198.51.100.1")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        mock_validate.assert_called_once_with("https://other.com/page", allowed_ranges=())


class TestChainedRedirects:
    """Redirect chains must track hostname through each hop."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_chained_relative_redirects_track_hostname(self, mock_validate, http_client):
        """Relative -> relative should keep resolving against the logical hostname."""
        redirect1 = _make_redirect_response("/step2")
        redirect2 = _make_redirect_response("/step3", url="https://93.184.216.34:443/step2")
        final_response = _make_final_response()

        mock_validate.side_effect = [
            _make_ssrf_request("https://example.com/step2"),
            _make_ssrf_request("https://example.com/step3"),
        ]
        http_client._ephemeral_mock.get.side_effect = [redirect2, final_response]

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/step1",
        )

        assert mock_validate.call_count == 2
        mock_validate.assert_any_call("https://example.com/step2", allowed_ranges=())
        mock_validate.assert_any_call("https://example.com/step3", allowed_ranges=())
        assert result.status_code == 200
        assert count == 2

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_absolute_redirect_updates_hostname_for_subsequent_relative(self, mock_validate, http_client):
        """Absolute redirect to new.com, then relative /page, should resolve as https://new.com/page."""
        redirect1 = _make_redirect_response("https://new.com/")
        redirect2 = _make_redirect_response("/page", url="https://203.0.113.1:443/")
        final_response = _make_final_response()

        mock_validate.side_effect = [
            _make_ssrf_request("https://new.com/", ip="203.0.113.1"),
            _make_ssrf_request("https://new.com/page", ip="203.0.113.1"),
        ]
        http_client._ephemeral_mock.get.side_effect = [redirect2, final_response]

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        assert mock_validate.call_count == 2
        mock_validate.assert_any_call("https://new.com/", allowed_ranges=())
        mock_validate.assert_any_call("https://new.com/page", allowed_ranges=())
        assert result.status_code == 200
        assert count == 2


class TestHostHeaderAndSNI:
    """Host header and SNI must use hostname from validate_url_for_ssrf, not IP."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_host_header_uses_hostname_not_ip(self, mock_validate, http_client):
        """Host header on redirect hop should be the hostname, not the resolved IP."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_req = _make_ssrf_request("https://example.com/new-path")
        mock_validate.return_value = ssrf_req
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test", "Host": "should-be-overwritten"},
            original_url="https://example.com/old-path",
        )

        # Check the headers passed to client.get()
        call_kwargs = http_client._ephemeral_mock.get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Host"] == "example.com"

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_sni_hostname_set_for_https_redirect(self, mock_validate, http_client):
        """TLS SNI should use the hostname from the redirect target, not IP."""
        redirect_response = _make_redirect_response("/secure-path")
        final_response = _make_final_response()

        ssrf_req = _make_ssrf_request("https://secure.example.com/secure-path")
        mock_validate.return_value = ssrf_req
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://secure.example.com/old-path",
        )

        call_kwargs = http_client._ephemeral_mock.get.call_args
        extensions = call_kwargs.kwargs.get("extensions") or call_kwargs[1].get("extensions")
        assert extensions["sni_hostname"] == "secure.example.com"


class TestNonRedirectPassthrough:
    """Non-redirect responses should pass through unchanged."""

    def test_non_redirect_response_returned_as_is(self, http_client):
        """A 200 response should be returned without modification."""
        response = _make_final_response()

        result, count, _final_url = http_client._follow_redirects_safe(
            response=response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/page",
        )

        assert result is response
        assert result.status_code == 200
        assert count == 0


class TestRedirectAuditRecording:
    """Each redirect hop must be individually recorded in the audit trail."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_single_redirect_records_one_hop(self, mock_validate, http_client):
        """A single redirect should produce exactly one HTTP_REDIRECT record_call."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://example.com/new-path")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test"},
            original_url="https://example.com/old-path",
        )

        recorder = http_client._execution
        recorder.record_call.assert_called_once()
        kw = recorder.record_call.call_args.kwargs
        assert kw["call_type"] == CallType.HTTP_REDIRECT
        assert kw["status"] == CallStatus.SUCCESS
        assert kw["state_id"] == "test-state-001"
        assert kw["request_data"].to_dict()["url"] == "https://example.com/new-path"
        assert kw["request_data"].to_dict()["hop_number"] == 1
        assert kw["response_data"].to_dict()["status_code"] == 200

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_chained_redirects_record_multiple_hops(self, mock_validate, http_client):
        """Two redirects should produce two HTTP_REDIRECT record_call invocations."""
        redirect1 = _make_redirect_response("/step2")
        redirect2 = _make_redirect_response("/step3", url="https://93.184.216.34:443/step2")
        final_response = _make_final_response()

        mock_validate.side_effect = [
            _make_ssrf_request("https://example.com/step2"),
            _make_ssrf_request("https://example.com/step3"),
        ]
        http_client._ephemeral_mock.get.side_effect = [redirect2, final_response]

        http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/step1",
        )

        recorder = http_client._execution
        assert recorder.record_call.call_count == 2

        # First hop
        kw1 = recorder.record_call.call_args_list[0].kwargs
        assert kw1["call_type"] == CallType.HTTP_REDIRECT
        assert kw1["request_data"].to_dict()["hop_number"] == 1
        assert kw1["request_data"].to_dict()["url"] == "https://example.com/step2"

        # Second hop
        kw2 = recorder.record_call.call_args_list[1].kwargs
        assert kw2["call_type"] == CallType.HTTP_REDIRECT
        assert kw2["request_data"].to_dict()["hop_number"] == 2
        assert kw2["request_data"].to_dict()["url"] == "https://example.com/step3"

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_hop_records_include_redirect_from(self, mock_validate, http_client):
        """redirect_from captures the URL we're redirecting FROM (lineage within chain)."""
        redirect_response = _make_redirect_response("https://other.com/page")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://other.com/page", ip="198.51.100.1")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        kw = http_client._execution.record_call.call_args.kwargs
        assert kw["request_data"].to_dict()["redirect_from"] == "https://example.com/start"
        assert kw["request_data"].to_dict()["url"] == "https://other.com/page"

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_hop_records_include_resolved_ip(self, mock_validate, http_client):
        """Each hop record must include the resolved IP from SSRF validation."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://example.com/new-path", ip="93.184.216.34")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/old-path",
        )

        kw = http_client._execution.record_call.call_args.kwargs
        assert kw["request_data"].to_dict()["resolved_ip"] == "93.184.216.34"

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_hop_records_have_latency(self, mock_validate, http_client):
        """Each hop record must include latency_ms."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        mock_validate.return_value = _make_ssrf_request("https://example.com/new-path")
        http_client._ephemeral_mock.get.return_value = final_response

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/old-path",
        )

        kw = http_client._execution.record_call.call_args.kwargs
        assert "latency_ms" in kw
        assert isinstance(kw["latency_ms"], float)
        assert kw["latency_ms"] >= 0

    def test_no_hop_recorded_when_no_redirects(self, http_client):
        """A non-redirect response should produce zero HTTP_REDIRECT records."""
        response = _make_final_response()

        http_client._follow_redirects_safe(
            response=response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/page",
        )

        http_client._execution.record_call.assert_not_called()


class TestBug4_7_FailedHopRecordsAuditTrail:
    """Bug 4.7: Failed redirect hops are recorded in audit trail.

    Previously, if a redirect hop's HTTP request failed (e.g., connection
    error), the hop was never recorded in the audit trail because the
    recording happened after the request. Now the audit trail records
    the failed hop with CallStatus.ERROR before re-raising the exception.
    """

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_failed_hop_recorded_with_error_status(self, mock_validate, http_client):
        """Redirect hop that raises exception is still recorded in audit trail."""
        redirect_response = _make_redirect_response("/new-path")

        mock_validate.return_value = _make_ssrf_request("https://example.com/new-path")
        # Simulate a connection error during the hop request
        http_client._ephemeral_mock.get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            http_client._follow_redirects_safe(
                response=redirect_response,
                max_redirects=5,
                timeout=10.0,
                original_headers={},
                original_url="https://example.com/old-path",
            )

        # The failed hop MUST still be recorded in the audit trail
        http_client._execution.record_call.assert_called_once()
        call_kwargs = http_client._execution.record_call.call_args.kwargs
        assert call_kwargs["call_type"] == CallType.HTTP_REDIRECT
        assert call_kwargs["status"] == CallStatus.ERROR
        assert "ConnectError" in call_kwargs["error"].type
        assert "Connection refused" in call_kwargs["error"].message
        assert isinstance(call_kwargs["latency_ms"], float)
        assert call_kwargs["latency_ms"] >= 0


class TestRedirectValidationFailuresPreserveEvidence:
    """Blocked redirect validation must retain both hop and triggering response evidence."""

    @patch("elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf")
    def test_blocked_redirect_records_hop_error_and_triggering_3xx_response(self, mock_validate, http_client):
        """Redirect validation failures must preserve both redirect target and 3xx response."""
        from elspeth.core.security.web import SSRFBlockedError

        redirect_url = "http://169.254.169.254/latest/meta-data/"
        initial_request = _make_ssrf_request("http://example.com/start")
        redirect_response = httpx.Response(
            301,
            headers={"location": redirect_url},
            request=httpx.Request("GET", "http://93.184.216.34:80/start"),
        )

        mock_validate.side_effect = SSRFBlockedError("blocked redirect")
        http_client._ephemeral_mock.get.return_value = redirect_response

        with pytest.raises(SSRFBlockedError, match="blocked redirect"):
            http_client.get_ssrf_safe(initial_request, follow_redirects=True)

        calls = [call.kwargs for call in http_client._execution.record_call.call_args_list]
        assert len(calls) == 2

        initial_call = next(call for call in calls if call["call_type"] == CallType.HTTP)
        redirect_call = next(call for call in calls if call["call_type"] == CallType.HTTP_REDIRECT)

        assert initial_call["status"] == CallStatus.ERROR
        assert initial_call["response_data"].to_dict()["status_code"] == 301
        assert initial_call["response_data"].to_dict()["headers"]["location"] == redirect_url

        assert redirect_call["status"] == CallStatus.ERROR
        assert redirect_call["request_data"].to_dict()["url"] == redirect_url
        assert redirect_call["request_data"].to_dict()["redirect_from"] == "http://example.com/start"
        assert redirect_call["error"].type == "SSRFBlockedError"
