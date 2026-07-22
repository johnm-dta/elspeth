"""Tests for redirect URL resolution and audit recording in _follow_redirects_safe().

Verifies that:
1. Relative redirects resolve against the original hostname URL, not the
   IP-based connection URL.
2. Each redirect hop is individually recorded in the audit trail as
   CallType.HTTP_REDIRECT with correct lineage data.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import HTTPCallResponse
from elspeth.core.security.web import SSRFSafeRequest
from elspeth.plugins.infrastructure.clients import http as http_client_module
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient, HTTPResponseBodyTooLargeError


@dataclass(frozen=True)
class RecordedGetCall:
    url: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ValidationCall:
    url: str
    allowed_ranges: tuple[Any, ...]


@dataclass
class FakeHTTPXClient:
    get_results: list[httpx.Response | BaseException] = field(default_factory=list)
    get_calls: list[RecordedGetCall] = field(default_factory=list)

    def queue_get(self, *results: httpx.Response | BaseException) -> None:
        self.get_results.extend(results)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.get_calls.append(RecordedGetCall(url=url, kwargs=kwargs))
        if not self.get_results:
            raise AssertionError(f"No queued fake HTTP GET result for {url}")

        result = self.get_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def __enter__(self) -> "FakeHTTPXClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def close(self) -> None:
        pass


@dataclass
class FakeHTTPXClientFactory:
    shared_client: FakeHTTPXClient
    ephemeral_client: FakeHTTPXClient
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> FakeHTTPXClient:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return self.shared_client
        return self.ephemeral_client


@dataclass
class FakeSSRFValidator:
    results: list[SSRFSafeRequest | BaseException] = field(default_factory=list)
    calls: list[ValidationCall] = field(default_factory=list)

    def queue(self, *results: SSRFSafeRequest | BaseException) -> None:
        self.results.extend(results)

    def __call__(self, url: str, *, allowed_ranges: Sequence[Any] = ()) -> SSRFSafeRequest:
        self.calls.append(ValidationCall(url=url, allowed_ranges=tuple(allowed_ranges)))
        if not self.results:
            raise AssertionError(f"No queued fake SSRF validation result for {url}")

        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


@dataclass
class FakeCallRecorder:
    calls: list[dict[str, Any]] = field(default_factory=list)
    next_call_index: int = 0

    def allocate_call_index(self, state_id: str) -> int:
        assert state_id == "test-state-001"
        self.next_call_index += 1
        return self.next_call_index

    def allocate_operation_call_index(self, operation_id: str) -> int:
        raise AssertionError(f"Unexpected operation call index allocation for {operation_id}")

    def record_call(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(**kwargs)


def _ignore_telemetry(_event: object) -> None:
    return None


@pytest.fixture
def ssrf_validator(monkeypatch):
    validator = FakeSSRFValidator()
    monkeypatch.setattr(http_client_module, "validate_url_for_ssrf", validator)
    return validator


@pytest.fixture
def http_client(monkeypatch):
    """Create AuditedHTTPClient with fake HTTP and audit dependencies."""
    shared_client = FakeHTTPXClient()
    ephemeral_client = FakeHTTPXClient()
    client_factory = FakeHTTPXClientFactory(shared_client=shared_client, ephemeral_client=ephemeral_client)
    monkeypatch.setattr(http_client_module.httpx, "Client", client_factory)

    client = AuditedHTTPClient(
        execution=FakeCallRecorder(),
        state_id="test-state-001",
        run_id="test-run-001",
        telemetry_emit=_ignore_telemetry,
        timeout=30.0,
    )
    client._test_ephemeral_client = ephemeral_client
    return client


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

    def test_relative_redirect_resolves_against_hostname(self, ssrf_validator, http_client):
        """Location: /new-path should produce https://example.com/new-path, not https://93.184.216.34/new-path."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://example.com/new-path"))
        http_client._test_ephemeral_client.queue_get(final_response)

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test"},
            original_url="https://example.com/old-path",
        )

        # validate_url_for_ssrf should receive hostname-based URL
        assert ssrf_validator.calls == [ValidationCall("https://example.com/new-path", ())]
        assert result.status_code == 200
        assert count == 1

    def test_relative_redirect_preserves_scheme_and_host(self, ssrf_validator, http_client):
        """Relative redirect should preserve scheme and host from original URL."""
        redirect_response = _make_redirect_response("/api/v2/resource")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://api.example.com/api/v2/resource"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://api.example.com/api/v1/resource",
        )

        assert ssrf_validator.calls == [ValidationCall("https://api.example.com/api/v2/resource", ())]


class TestAbsoluteRedirectResolution:
    """Absolute redirects carry their own hostname — should work regardless."""

    def test_absolute_redirect_to_different_host(self, ssrf_validator, http_client):
        """Location: https://other.com/page should use other.com, not original host."""
        redirect_response = _make_redirect_response("https://other.com/page")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://other.com/page", ip="198.51.100.1"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        assert ssrf_validator.calls == [ValidationCall("https://other.com/page", ())]


class TestChainedRedirects:
    """Redirect chains must track hostname through each hop."""

    def test_chained_relative_redirects_track_hostname(self, ssrf_validator, http_client):
        """Relative -> relative should keep resolving against the logical hostname."""
        redirect1 = _make_redirect_response("/step2")
        redirect2 = _make_redirect_response("/step3", url="https://93.184.216.34:443/step2")
        final_response = _make_final_response()

        ssrf_validator.queue(
            _make_ssrf_request("https://example.com/step2"),
            _make_ssrf_request("https://example.com/step3"),
        )
        http_client._test_ephemeral_client.queue_get(redirect2, final_response)

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/step1",
        )

        assert ssrf_validator.calls == [
            ValidationCall("https://example.com/step2", ()),
            ValidationCall("https://example.com/step3", ()),
        ]
        assert result.status_code == 200
        assert count == 2

    def test_absolute_redirect_updates_hostname_for_subsequent_relative(self, ssrf_validator, http_client):
        """Absolute redirect to new.com, then relative /page, should resolve as https://new.com/page."""
        redirect1 = _make_redirect_response("https://new.com/")
        redirect2 = _make_redirect_response("/page", url="https://203.0.113.1:443/")
        final_response = _make_final_response()

        ssrf_validator.queue(
            _make_ssrf_request("https://new.com/", ip="203.0.113.1"),
            _make_ssrf_request("https://new.com/page", ip="203.0.113.1"),
        )
        http_client._test_ephemeral_client.queue_get(redirect2, final_response)

        result, count, _final_url = http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        assert ssrf_validator.calls == [
            ValidationCall("https://new.com/", ()),
            ValidationCall("https://new.com/page", ()),
        ]
        assert result.status_code == 200
        assert count == 2


class TestHostHeaderAndSNI:
    """Host header and SNI must use hostname from validate_url_for_ssrf, not IP."""

    def test_host_header_uses_hostname_not_ip(self, ssrf_validator, http_client):
        """Host header on redirect hop should be the hostname, not the resolved IP."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_req = _make_ssrf_request("https://example.com/new-path")
        ssrf_validator.queue(ssrf_req)
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test", "Host": "should-be-overwritten"},
            original_url="https://example.com/old-path",
        )

        # Check the headers passed to client.get()
        headers = http_client._test_ephemeral_client.get_calls[-1].kwargs["headers"]
        assert headers["Host"] == "example.com"

    def test_sni_hostname_set_for_https_redirect(self, ssrf_validator, http_client):
        """TLS SNI should use the hostname from the redirect target, not IP."""
        redirect_response = _make_redirect_response("/secure-path")
        final_response = _make_final_response()

        ssrf_req = _make_ssrf_request("https://secure.example.com/secure-path")
        ssrf_validator.queue(ssrf_req)
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://secure.example.com/old-path",
        )

        extensions = http_client._test_ephemeral_client.get_calls[-1].kwargs["extensions"]
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

    def test_single_redirect_records_one_hop(self, ssrf_validator, http_client):
        """A single redirect should produce exactly one HTTP_REDIRECT record_call."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://example.com/new-path"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={"User-Agent": "test"},
            original_url="https://example.com/old-path",
        )

        recorder = http_client._execution
        assert len(recorder.calls) == 1
        kw = recorder.calls[0]
        assert kw["call_type"] == CallType.HTTP_REDIRECT
        assert kw["status"] == CallStatus.SUCCESS
        assert kw["state_id"] == "test-state-001"
        assert kw["request_data"].to_dict()["url"] == "https://example.com/new-path"
        assert kw["request_data"].to_dict()["hop_number"] == 1
        assert kw["response_data"].to_dict()["status_code"] == 200

    def test_chained_redirects_record_multiple_hops(self, ssrf_validator, http_client):
        """Two redirects should produce two HTTP_REDIRECT record_call invocations."""
        redirect1 = _make_redirect_response("/step2")
        redirect2 = _make_redirect_response("/step3", url="https://93.184.216.34:443/step2")
        final_response = _make_final_response()

        ssrf_validator.queue(
            _make_ssrf_request("https://example.com/step2"),
            _make_ssrf_request("https://example.com/step3"),
        )
        http_client._test_ephemeral_client.queue_get(redirect2, final_response)

        http_client._follow_redirects_safe(
            response=redirect1,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/step1",
        )

        recorder = http_client._execution
        assert len(recorder.calls) == 2

        # First hop
        kw1 = recorder.calls[0]
        assert kw1["call_type"] == CallType.HTTP_REDIRECT
        assert kw1["request_data"].to_dict()["hop_number"] == 1
        assert kw1["request_data"].to_dict()["url"] == "https://example.com/step2"

        # Second hop
        kw2 = recorder.calls[1]
        assert kw2["call_type"] == CallType.HTTP_REDIRECT
        assert kw2["request_data"].to_dict()["hop_number"] == 2
        assert kw2["request_data"].to_dict()["url"] == "https://example.com/step3"

    def test_hop_records_include_redirect_from(self, ssrf_validator, http_client):
        """redirect_from captures the URL we're redirecting FROM (lineage within chain)."""
        redirect_response = _make_redirect_response("https://other.com/page")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://other.com/page", ip="198.51.100.1"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/start",
        )

        kw = http_client._execution.calls[0]
        assert kw["request_data"].to_dict()["redirect_from"] == "https://example.com/start"
        assert kw["request_data"].to_dict()["url"] == "https://other.com/page"

    def test_hop_records_include_resolved_ip(self, ssrf_validator, http_client):
        """Each hop record must include the resolved IP from SSRF validation."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://example.com/new-path", ip="93.184.216.34"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/old-path",
        )

        kw = http_client._execution.calls[0]
        assert kw["request_data"].to_dict()["resolved_ip"] == "93.184.216.34"

    def test_hop_records_have_latency(self, ssrf_validator, http_client):
        """Each hop record must include latency_ms."""
        redirect_response = _make_redirect_response("/new-path")
        final_response = _make_final_response()

        ssrf_validator.queue(_make_ssrf_request("https://example.com/new-path"))
        http_client._test_ephemeral_client.queue_get(final_response)

        http_client._follow_redirects_safe(
            response=redirect_response,
            max_redirects=5,
            timeout=10.0,
            original_headers={},
            original_url="https://example.com/old-path",
        )

        kw = http_client._execution.calls[0]
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

        assert http_client._execution.calls == []


class TestBug4_7_FailedHopRecordsAuditTrail:
    """Bug 4.7: Failed redirect hops are recorded in audit trail.

    Previously, if a redirect hop's HTTP request failed (e.g., connection
    error), the hop was never recorded in the audit trail because the
    recording happened after the request. Now the audit trail records
    the failed hop with CallStatus.ERROR before re-raising the exception.
    """

    def test_failed_hop_recorded_with_error_status(self, ssrf_validator, http_client):
        """Redirect hop that raises exception is still recorded in audit trail."""
        redirect_response = _make_redirect_response("/new-path")

        ssrf_validator.queue(_make_ssrf_request("https://example.com/new-path"))
        # Simulate a connection error during the hop request
        http_client._test_ephemeral_client.queue_get(httpx.ConnectError("Connection refused"))

        with pytest.raises(httpx.ConnectError):
            http_client._follow_redirects_safe(
                response=redirect_response,
                max_redirects=5,
                timeout=10.0,
                original_headers={},
                original_url="https://example.com/old-path",
            )

        # The failed hop MUST still be recorded in the audit trail
        assert len(http_client._execution.calls) == 1
        call_kwargs = http_client._execution.calls[0]
        assert call_kwargs["call_type"] == CallType.HTTP_REDIRECT
        assert call_kwargs["status"] == CallStatus.ERROR
        assert "ConnectError" in call_kwargs["error"].type
        assert "Connection refused" in call_kwargs["error"].message
        assert isinstance(call_kwargs["latency_ms"], float)
        assert call_kwargs["latency_ms"] >= 0

    def test_failed_hop_audit_sanitizes_connection_url_without_changing_raised_error(
        self,
        ssrf_validator: FakeSSRFValidator,
        http_client: AuditedHTTPClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """IP-pinned redirect errors retain diagnostics without raw query credentials."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key-for-http-error-audit")
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)
        redirect_response = _make_redirect_response("https://redirect.example.com/huge?token=HOP_SECRET&view=summary#HOP_FRAGMENT")
        redirect_request = _make_ssrf_request(
            "https://redirect.example.com/huge?token=HOP_SECRET&view=summary#HOP_FRAGMENT",
            ip="198.51.100.10",
        )
        ssrf_validator.queue(redirect_request)
        response_payload = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=6,
            body={"_truncated": True, "_reason": "body_too_large"},
        )
        failure = HTTPResponseBodyTooLargeError(
            url=redirect_request.connection_url,
            body_size=6,
            max_body_bytes=5,
            response_payload=response_payload,
        )
        http_client._test_ephemeral_client.queue_get(failure)

        with pytest.raises(HTTPResponseBodyTooLargeError) as exc_info:
            http_client._follow_redirects_safe(
                response=redirect_response,
                max_redirects=5,
                timeout=10.0,
                original_headers={},
                original_url="https://example.com/start",
            )

        assert exc_info.value is failure
        assert len(http_client._execution.calls) == 1
        recorded_message = http_client._execution.calls[0]["error"].message
        assert "response body 6 bytes exceeds max_response_body_bytes 5" in recorded_message
        assert "198.51.100.10:443/huge" in recorded_message
        assert "view=summary" in recorded_message
        assert "token=" in recorded_message
        assert "HOP_SECRET" not in recorded_message
        assert "HOP_FRAGMENT" not in recorded_message


class TestRedirectValidationFailuresPreserveEvidence:
    """Blocked redirect validation must retain both hop and triggering response evidence."""

    def test_blocked_redirect_records_hop_error_and_triggering_3xx_response(self, ssrf_validator, http_client):
        """Redirect validation failures must preserve both redirect target and 3xx response."""
        from elspeth.core.security.web import SSRFBlockedError

        redirect_url = "http://169.254.169.254/latest/meta-data/"
        initial_request = _make_ssrf_request("http://example.com/start")
        redirect_response = httpx.Response(
            301,
            headers={"location": redirect_url},
            request=httpx.Request("GET", "http://93.184.216.34:80/start"),
        )

        ssrf_validator.queue(SSRFBlockedError("blocked redirect"))
        http_client._test_ephemeral_client.queue_get(redirect_response)

        with pytest.raises(SSRFBlockedError, match="blocked redirect"):
            http_client.get_ssrf_safe(initial_request, follow_redirects=True)

        calls = http_client._execution.calls
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
