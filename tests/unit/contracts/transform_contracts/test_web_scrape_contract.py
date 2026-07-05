# tests/unit/contracts/transform_contracts/test_web_scrape_contract.py
"""Contract tests for WebScrapeTransform.

Note: WebScrapeTransform does NOT inherit BatchTransformMixin yet,
so we use TransformContractPropertyTestBase. This will change when
we add concurrency in a later task.
"""

from __future__ import annotations

import socket
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.audit import Call
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.plugins.transforms.web_scrape_errors import ForbiddenError, NotFoundError, UnauthorizedError
from elspeth.testing import make_pipeline_row

from .test_transform_protocol import TransformContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol


_HTTPX_CLIENT_CLASS = httpx.Client
_TEST_IP = "93.184.216.34"

# SSRF blocked-IP coverage matrix.
#
# Each entry is (resolved_ip, blocklist_tier) where blocklist_tier is the
# verbatim prefix of the SSRFBlockedError message produced by
# elspeth.core.security.web._validate_ip_address(). The two prefixes are
# load-bearing: they distinguish ALWAYS_BLOCKED_RANGES (cannot be bypassed
# by allowed_ranges) from BLOCKED_IP_RANGES (bypassable). Both must be
# covered to verify the full SSRF surface — the previous single-case test
# (169.254.169.254 only) only exercised the ALWAYS_BLOCKED tier and would
# pass even if the entire BLOCKED_IP_RANGES check was deleted.
#
# Sourced from BLOCKED_IP_RANGES + ALWAYS_BLOCKED_RANGES in
# src/elspeth/core/security/web.py. Do NOT shrink this list without a
# matching change there — the ranges and these test cases must move together.
#
# Falsifiability check (run manually after any change to either the
# production blocklist or this matrix): remove a single range from
# production and confirm exactly the matching parametrised case fails.
# Production code at security/web.py:61 specifically calls
# ::ffff:0:0/96 a "CRITICAL: bypass vector!" — the ipv4_mapped_ipv6 case
# below is the falsifiable proof that the IPv6-side check fires before
# any IPv4-mapping path could leak through. ::ffff:10.0.0.1 is chosen
# because it is in ::ffff:0:0/96 but NOT in ::ffff:169.254.0.0/112
# (which would short-circuit on the always-blocked tier first).
_SSRF_BLOCKED_CASES: tuple[tuple[str, str, str], ...] = (
    # ALWAYS_BLOCKED_RANGES (unconditional — no allowlist bypass)
    ("aws_metadata_v4", "169.254.169.254", "Always-blocked IP range"),
    ("ipv4_mapped_metadata", "::ffff:169.254.169.254", "Always-blocked IP range"),
    ("aws_metadata_v6", "fd00:ec2::254", "Always-blocked IP range"),
    ("ipv6_link_local", "fe80::1", "Always-blocked IP range"),
    ("broadcast_v4", "255.255.255.255", "Always-blocked IP range"),
    ("multicast_v4", "224.0.0.1", "Always-blocked IP range"),
    ("multicast_v6", "ff02::1", "Always-blocked IP range"),
    # BLOCKED_IP_RANGES (default blocklist; bypassable only via allowed_ranges)
    ("current_network_v4", "0.0.0.1", "Blocked IP range"),
    ("loopback_v4", "127.0.0.1", "Blocked IP range"),
    ("loopback_v6", "::1", "Blocked IP range"),
    ("rfc1918_class_a", "10.0.0.1", "Blocked IP range"),
    ("rfc1918_class_b", "172.16.0.1", "Blocked IP range"),
    ("rfc1918_class_c", "192.168.1.1", "Blocked IP range"),
    ("cgnat_v4", "100.64.0.1", "Blocked IP range"),
    ("ipv6_ula", "fc00::1", "Blocked IP range"),
    ("ipv4_mapped_ipv6", "::ffff:10.0.0.1", "Blocked IP range"),
)


def _mock_getaddrinfo(ip: str = _TEST_IP) -> Any:
    """Create a deterministic DNS resolver for SSRF validation.

    The shape of the returned tuple matches what ``socket.getaddrinfo``
    actually produces and what
    ``elspeth.core.security.web._resolve_hostname`` consumes:
    ``(family, type, proto, canonname, sockaddr)``.

    For IPv4: sockaddr = ``(ip, port)``
    For IPv6: sockaddr = ``(ip, port, flowinfo, scopeid)``

    The IP family is detected from the literal so callers can pass either
    form without a separate flag.
    """

    is_ipv6 = ":" in ip
    family = socket.AF_INET6 if is_ipv6 else socket.AF_INET
    sockaddr: tuple[Any, ...] = (ip, 0, 0, 0) if is_ipv6 else (ip, 0)

    def _getaddrinfo(
        host: str,
        port: Any,
        family_arg: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]

    return _getaddrinfo


def _create_http_response() -> httpx.Response:
    """Create a real HTTP response for testing the audited-client boundary."""
    # WebScrapeTransform records the IP-pinned destination from the final
    # SSRF-safe request. Contract fixtures must therefore emulate the
    # real AuditedHTTPClient behavior rather than leaving request.url.host
    # as an untyped Mock.
    return httpx.Response(
        200,
        content=b"<html><body>Test content</body></html>",
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://93.184.216.34/contract-test"),
    )


def _set_stream_response(client: Mock, response: httpx.Response) -> None:
    client.stream.return_value = nullcontext(response)


def _create_audit_call() -> Call:
    return Call(
        call_id="test-call-id",
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.SUCCESS,
        request_hash="test-request-hash",
        created_at=datetime.now(UTC),
        state_id="test-state-001",
        request_ref="test-request-ref-hash",
        response_hash="test-response-hash",
        response_ref="test-response-ref-hash",
        latency_ms=100.0,
    )


def _context_mock(value: object, name: str) -> Mock:
    assert isinstance(value, Mock), f"expected mocked {name}"
    return value


class TestWebScrapeContract(TransformContractPropertyTestBase):
    """Verify WebScrapeTransform satisfies plugin contract."""

    @pytest.fixture(autouse=True)
    def mock_httpx(self):
        """Mock httpx.Client for all contract tests."""
        with (
            patch("socket.getaddrinfo", side_effect=_mock_getaddrinfo()),
            patch("httpx.Client", autospec=True) as mock_client_class,
        ):
            mock_response = _create_http_response()
            mock_client_instance = MagicMock(spec_set=_HTTPX_CLIENT_CLASS)
            _set_stream_response(mock_client_instance, mock_response)
            mock_client_instance.__enter__.return_value = mock_client_instance
            mock_client_instance.__exit__.return_value = False
            mock_client_class.return_value = mock_client_instance
            yield mock_client_class

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a WebScrapeTransform instance with valid configuration."""
        return WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Contract testing web scrape transform",
                },
            }
        )

    @pytest.fixture(autouse=True)
    def _init_lifecycle(self, transform, ctx) -> None:
        """Call on_start() to capture infrastructure before tests call process()."""
        transform.on_start(ctx)

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row with a URL field."""
        return {"url": "https://example.com"}

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Provide a PluginContext with required dependencies for WebScrapeTransform."""
        mock_limiter = Mock(spec_set=["acquire", "close"])
        mock_limiter.acquire.return_value = None

        mock_registry = Mock(spec_set=["get_limiter", "close"])
        mock_registry.get_limiter.return_value = mock_limiter

        # Create mock landscape recorder — must return a proper Call so process()
        # can read call.request_ref and call.response_ref without FrameworkBugError.
        mock_landscape = Mock(spec_set=["allocate_call_index", "record_call", "store_payload"])
        mock_landscape.record_call.return_value = _create_audit_call()
        mock_landscape.allocate_call_index.return_value = 0
        mock_landscape.store_payload.return_value = "test-processed-hash"

        # PayloadStore mock — WebScrapeTransform.process() stores processed
        # content via self._payload_store.store() captured during on_start().
        mock_payload_store = Mock(spec_set=["store", "retrieve", "exists", "delete"])
        mock_payload_store.store.return_value = "test-processed-content-hash"

        return PluginContext(
            run_id="test-run-001",
            config={},
            node_id="test-transform",
            rate_limit_registry=mock_registry,
            landscape=mock_landscape,
            payload_store=mock_payload_store,
            state_id="test-state-001",
        )

    def test_success_records_audit_call_and_processed_payload(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Successful fetches must leave both call audit and processed-payload evidence."""
        result = transform.process(make_pipeline_row(valid_input), ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["fetch_status"] == 200
        assert result.row["fetch_url_final_ip"] == _TEST_IP
        assert result.success_reason is not None
        assert result.success_reason["metadata"] == {
            "fetch_request_hash": "test-request-ref-hash",
            "fetch_response_raw_hash": "test-response-ref-hash",
            "fetch_response_processed_hash": "test-processed-content-hash",
        }

        limiter = _context_mock(ctx.rate_limit_registry.get_limiter.return_value, "rate limiter")
        landscape = _context_mock(ctx.landscape, "landscape")
        payload_store = _context_mock(ctx.payload_store, "payload_store")

        limiter.acquire.assert_called_once_with()
        landscape.allocate_call_index.assert_called_once_with("test-state-001")
        landscape.record_call.assert_called_once()
        call_kwargs = landscape.record_call.call_args.kwargs
        assert call_kwargs["state_id"] == "test-state-001"
        assert call_kwargs["call_index"] == 0
        assert call_kwargs["call_type"] is CallType.HTTP
        assert call_kwargs["status"] is CallStatus.SUCCESS
        payload_store.store.assert_called_once_with(result.row["page_content"].encode())

    @pytest.mark.parametrize(
        ("case_id", "resolved_ip", "tier_prefix"),
        _SSRF_BLOCKED_CASES,
        ids=[case[0] for case in _SSRF_BLOCKED_CASES],
    )
    def test_ssrf_blocked_resolved_ip_quarantines_before_external_call(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        mock_httpx: Mock,
        case_id: str,
        resolved_ip: str,
        tier_prefix: str,
    ) -> None:
        """SSRF rejection is a pre-fetch validation result, not an audited HTTP call.

        Coverage rationale: WebScrapeTransform delegates to
        ``elspeth.core.security.web.validate_url_for_ssrf``, which checks
        resolved IPs against two distinct blocklist tiers
        (``ALWAYS_BLOCKED_RANGES`` and ``BLOCKED_IP_RANGES``). A test that
        only exercises one tier — as the previous single-case version did
        with ``169.254.169.254`` — would silently pass even if the other
        tier's check was deleted. This parametrisation covers both tiers
        across IPv4 loopback, IPv4 link-local/metadata, IPv4 RFC-1918
        (Class A/B/C), IPv6 loopback, and IPv6 link-local. See
        ``_SSRF_BLOCKED_CASES`` for the matrix and its source-of-truth
        cross-reference.

        DNS path: the only resolution path in the SSRF check is
        ``_resolve_hostname`` → ``socket.getaddrinfo`` (called via the
        bounded ``_dns_pool`` ThreadPoolExecutor). No alternative DNS APIs
        (``gethostbyname``, async resolvers, etc.) are used by the
        production code, so a single ``socket.getaddrinfo`` patch covers
        the whole surface. Redirect re-validation in
        ``AuditedHTTPClient._follow_redirects`` also funnels through
        ``validate_url_for_ssrf``, so it inherits the same DNS path.
        """
        row = make_pipeline_row({"url": "https://blocked.example/path"})

        with patch("socket.getaddrinfo", side_effect=_mock_getaddrinfo(resolved_ip)):
            result = transform.process(row, ctx)

        assert result.status == "error", f"case {case_id}: expected error, got {result.status!r}"
        assert result.reason is not None, f"case {case_id}: missing reason payload"
        assert result.reason["reason"] == "validation_failed", (
            f"case {case_id}: expected reason=validation_failed, got {result.reason['reason']!r}"
        )
        assert result.reason["error_type"] == "SSRFBlockedError", (
            f"case {case_id}: expected SSRFBlockedError, got {result.reason['error_type']!r}"
        )
        # The error message must include the verbatim resolved IP and the
        # blocklist-tier prefix. Both are load-bearing for audit clarity:
        # downstream tooling distinguishes the tiers when reporting.
        assert f"{tier_prefix}: {resolved_ip}" in result.reason["error"], (
            f"case {case_id}: error message {result.reason['error']!r} did not contain {tier_prefix!r}: {resolved_ip!r}"
        )
        assert result.retryable is False, f"case {case_id}: SSRF rejection must not be retryable"

        # SSRF is rejected before any HTTP call; the audit recorder, payload
        # store, and httpx client must not see this request.
        _context_mock(ctx.landscape, "landscape").record_call.assert_not_called()
        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()
        mock_httpx.return_value.stream.assert_not_called()

    def test_response_without_request_crashes_before_payload_storage(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Audited clients must return an IP-pinned response request for provenance."""

        def fake_fetch_url(safe_request: Any, process_ctx: PluginContext) -> tuple[httpx.Response, str, Call]:
            assert process_ctx is ctx
            return (
                httpx.Response(200, content=b"<html><body>missing request</body></html>"),
                safe_request.original_url,
                _create_audit_call(),
            )

        monkeypatch.setattr(transform, "_fetch_url", fake_fetch_url)

        with pytest.raises(
            FrameworkBugError,
            match=r"SSRF-safe HTTP response has no request; cannot record final resolved IP\.",
        ):
            transform.process(make_pipeline_row({"url": "https://example.com"}), ctx)

        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()

    @pytest.mark.parametrize(
        ("row_data", "error_type"),
        [
            ({}, "KeyError"),
            ({"url": 123}, "TypeError"),
            ({"url": ["https://example.com"]}, "TypeError"),
        ],
        ids=["missing-url", "integer-url", "list-url"],
    )
    def test_invalid_url_field_values_fail_before_external_call(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        mock_httpx: Mock,
        row_data: dict[str, Any],
        error_type: str,
    ) -> None:
        """Missing or malformed URL field values are row validation failures."""
        result = transform.process(make_pipeline_row(row_data), ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["error_type"] == error_type
        assert result.retryable is False

        _context_mock(ctx.landscape, "landscape").record_call.assert_not_called()
        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()
        _context_mock(ctx.rate_limit_registry, "rate limit registry").get_limiter.assert_not_called()
        mock_httpx.return_value.stream.assert_not_called()

    def test_response_request_without_host_crashes_before_payload_storage(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Audited clients must return a response request URL with a host."""

        def fake_fetch_url(safe_request: Any, process_ctx: PluginContext) -> tuple[Any, str, Call]:
            assert process_ctx is ctx
            response = Mock(spec_set=httpx.Response)
            response.request.url.host = None
            return response, safe_request.original_url, _create_audit_call()

        monkeypatch.setattr(transform, "_fetch_url", fake_fetch_url)

        with pytest.raises(
            FrameworkBugError,
            match=r"SSRF-safe HTTP response request URL has no host; cannot record final resolved IP\.",
        ):
            transform.process(make_pipeline_row({"url": "https://example.com"}), ctx)

        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()

    def test_response_request_with_non_ip_host_crashes_before_payload_storage(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Audited clients must return the IP-pinned request, not the logical hostname."""

        def fake_fetch_url(safe_request: Any, process_ctx: PluginContext) -> tuple[httpx.Response, str, Call]:
            assert process_ctx is ctx
            return (
                httpx.Response(
                    200,
                    content=b"<html><body>hostname request</body></html>",
                    request=httpx.Request("GET", "https://example.com/contract-test"),
                ),
                safe_request.original_url,
                _create_audit_call(),
            )

        monkeypatch.setattr(transform, "_fetch_url", fake_fetch_url)

        with pytest.raises(
            FrameworkBugError,
            match=(
                r"SSRF-safe HTTP response request host 'example\.com' is not an IP address; "
                r"AuditedHTTPClient must return the IP-pinned final request\."
            ),
        ):
            transform.process(make_pipeline_row({"url": "https://example.com"}), ctx)

        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()

    @pytest.mark.parametrize(
        ("status_code", "error_type"),
        [
            (401, UnauthorizedError.__name__),
            (403, ForbiddenError.__name__),
            (404, NotFoundError.__name__),
        ],
    )
    def test_non_retryable_http_status_returns_api_error_without_processed_payload(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        mock_httpx: Mock,
        status_code: int,
        error_type: str,
    ) -> None:
        """Non-retryable HTTP failures are audited fetches, not successful rows."""
        _set_stream_response(
            mock_httpx.return_value,
            httpx.Response(
                status_code,
                content=b"<html><body>not ok</body></html>",
                request=httpx.Request("GET", "https://93.184.216.34/contract-test"),
            ),
        )

        result = transform.process(make_pipeline_row({"url": "https://example.com"}), ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == error_type
        assert result.retryable is False

        _context_mock(ctx.landscape, "landscape").record_call.assert_called_once()
        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()

    def test_processed_payload_store_failure_propagates_after_fetch_audit(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
    ) -> None:
        """Processed-content storage failures must not be hidden as success."""
        payload_store = _context_mock(ctx.payload_store, "payload_store")
        payload_store.store.side_effect = RuntimeError("processed payload store failed")

        with pytest.raises(RuntimeError, match="processed payload store failed"):
            transform.process(make_pipeline_row({"url": "https://example.com"}), ctx)

        _context_mock(ctx.landscape, "landscape").record_call.assert_called_once()
        payload_store.store.assert_called_once()
