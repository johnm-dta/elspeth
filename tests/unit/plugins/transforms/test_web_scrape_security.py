"""Security tests for WebScrapeTransform (SSRF prevention).

Tests the complete SSRF-safe pipeline:
1. validate_url_for_ssrf() validates URL and pins IP
2. get_ssrf_safe() connects to the pinned IP
3. Redirects are independently validated at each hop
"""

import socket
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.audit import Call
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.testing import make_pipeline_row


def _mock_getaddrinfo(ip: str) -> Any:
    """Create a mock getaddrinfo that returns the given IP."""
    is_ipv6 = ":" in ip

    def _getaddrinfo(
        host: str,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        if is_ipv6:
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return _getaddrinfo


@pytest.fixture
def transform(mock_ctx):
    t = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Security testing",
            },
        }
    )
    t.on_start(mock_ctx)
    return t


@pytest.fixture
def mock_ctx():
    """Create PluginContext with required attributes for security testing."""
    landscape = Mock()

    # Configure record_call to return a proper Call object so process() can
    # read call.request_ref and call.response_ref without FrameworkBugError.
    mock_call = Call(
        call_id="test-call-id",
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.SUCCESS,
        request_hash="test-request-hash",
        created_at=datetime.now(UTC),
        state_id="state-123",
        request_ref="test-request-ref-hash",
        response_hash="test-response-hash",
        response_ref="test-response-ref-hash",
        latency_ms=100.0,
    )
    landscape.record_call.return_value = mock_call
    landscape.allocate_call_index.return_value = 0
    landscape.store_payload.return_value = "test-processed-hash"

    rate_limit_registry = Mock()
    rate_limit_registry.get_limiter.return_value = None

    payload_store = Mock()
    payload_store.store.return_value = "test-processed-hash"

    ctx = PluginContext(
        run_id="test-run-456",
        config={},
        landscape=landscape,
        payload_store=payload_store,
        rate_limit_registry=rate_limit_registry,
        state_id="state-123",
    )

    return ctx


def test_ssrf_blocks_file_scheme(transform, mock_ctx):
    """file:// URLs should be blocked."""
    result = transform.process(make_pipeline_row({"url": "file:///etc/passwd"}), mock_ctx)

    assert result.status == "error"
    assert result.reason["error_type"] == "SSRFBlockedError"
    assert "file" in result.reason["error"].lower()


def test_ssrf_blocks_private_ip(transform, mock_ctx):
    """Private IPs should be blocked."""
    with patch("socket.getaddrinfo", _mock_getaddrinfo("192.168.1.1")):
        result = transform.process(make_pipeline_row({"url": "https://internal.example.com"}), mock_ctx)

        assert result.status == "error"
        assert result.reason["error_type"] == "SSRFBlockedError"


def test_ssrf_blocks_loopback(transform, mock_ctx):
    """Loopback IPs should be blocked."""
    with patch("socket.getaddrinfo", _mock_getaddrinfo("127.0.0.1")):
        result = transform.process(make_pipeline_row({"url": "http://localhost/admin"}), mock_ctx)

        assert result.status == "error"
        assert result.reason["error_type"] == "SSRFBlockedError"


def test_ssrf_blocks_cloud_metadata(transform, mock_ctx):
    """Cloud metadata endpoints should be blocked."""
    with patch("socket.getaddrinfo", _mock_getaddrinfo("169.254.169.254")):
        result = transform.process(
            make_pipeline_row({"url": "http://metadata.google.internal/computeMetadata/v1/"}),
            mock_ctx,
        )

        assert result.status == "error"
        assert result.reason["error_type"] == "SSRFBlockedError"


@respx.mock
def test_ssrf_allows_public_ip(transform, mock_ctx):
    """Public IPs should be allowed (request goes to pinned IP)."""
    # Mock the IP-based URL that get_ssrf_safe() will actually request
    respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Content</html>"))

    with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

        assert result.status == "success"


# ============================================================================
# DNS Rebinding Prevention Tests
# ============================================================================


@respx.mock
def test_connection_uses_validated_ip(transform, mock_ctx):
    """Verify the actual HTTP request goes to the validated IP, not re-resolved hostname.

    This is the core DNS rebinding test: even if an attacker could change
    DNS between validation and connection, get_ssrf_safe() connects to the
    IP that was validated, not a re-resolved IP.
    """
    # The request should go to the pinned IP (93.184.216.34), NOT to example.com
    ip_route = respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Safe</html>"))
    # If httpx re-resolves DNS, it would hit the hostname route instead
    hostname_route = respx.get("https://example.com/page").mock(return_value=httpx.Response(200, text="<html>Unsafe</html>"))

    with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    assert ip_route.called, "Request should go to pinned IP, not hostname"
    assert not hostname_route.called, "Request should NOT re-resolve DNS to hostname"


# ============================================================================
# Audit Trail Tests
# ============================================================================


@respx.mock
def test_resolved_ip_recorded_in_audit(transform, mock_ctx):
    """Audit trail should record the resolved IP for forensic traceability."""
    respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Content</html>"))

    with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
        transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    # Check that record_call was invoked with resolved_ip in request_data
    record_call_args = mock_ctx.landscape.record_call.call_args
    assert record_call_args is not None, "record_call should have been called"
    request_data = record_call_args.kwargs.get("request_data") or record_call_args[1].get("request_data")
    assert request_data.to_dict()["resolved_ip"] == "93.184.216.34"


# ============================================================================
# Contract Propagation Tests
# ============================================================================


@respx.mock
def test_contract_includes_output_fields(transform, mock_ctx):
    """Output contract must include content_field, fingerprint_field, and fetch_* fields.

    P2 bug: WebScrapeTransform returned success without contract, causing executor
    to build contract from output_schema (which is same as input_schema). New fields
    were not accessible under FIXED schema mode.
    """
    respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Content</html>"))

    with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

        assert result.status == "success"
        # Contract MUST be provided so executor can use it (now inside PipelineRow)
        assert isinstance(result.row, PipelineRow), "TransformResult must include PipelineRow with contract for new fields"

        # Contract must include all output fields
        field_names = {f.normalized_name for f in result.row.contract.fields}

        # Input field should be preserved
        assert "url" in field_names

        # New operational fields from transform must be in contract
        assert "page_content" in field_names, "content_field must be in output contract"
        assert "page_fingerprint" in field_names, "fingerprint_field must be in output contract"
        assert "fetch_status" in field_names, "fetch_status must be in output contract"
        assert "fetch_url_final" in field_names, "fetch_url_final must be in output contract"
        assert "fetch_url_final_ip" in field_names, "fetch_url_final_ip must be in output contract"
        # Hash fields are audit-only (in success_reason["metadata"]) — not in row contract
        assert "fetch_request_hash" not in field_names, "fetch_request_hash is audit-only, must not be in row contract"
        assert "fetch_response_raw_hash" not in field_names, "fetch_response_raw_hash is audit-only, must not be in row contract"
        assert "fetch_response_processed_hash" not in field_names, (
            "fetch_response_processed_hash is audit-only, must not be in row contract"
        )


@respx.mock
def test_contract_field_types_are_correct(transform, mock_ctx):
    """Output contract fields should have correct types inferred from values."""
    respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Content</html>"))

    with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

        assert isinstance(result.row, PipelineRow)

        # Get field contracts by name
        field_by_name = {f.normalized_name: f for f in result.row.contract.fields}

        # Check types for operational fields in the row
        assert field_by_name["page_content"].python_type is str
        assert field_by_name["page_fingerprint"].python_type is str
        assert field_by_name["fetch_status"].python_type is int
        assert field_by_name["fetch_url_final"].python_type is str
        assert field_by_name["fetch_url_final_ip"].python_type is str
        # Hash fields are audit-only — they live in success_reason["metadata"], not in the row


# ===========================================================================
# allowed_hosts end-to-end tests
# ===========================================================================


class TestAllowedHostsEndToEnd:
    """End-to-end tests for allowed_hosts config through the transform."""

    @pytest.fixture
    def allowed_loopback_transform(self, mock_ctx):
        """Transform with allowed_hosts: ["127.0.0.0/8"]."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing allowed_hosts",
                    "allowed_hosts": ["127.0.0.0/8"],
                },
            }
        )
        t.on_start(mock_ctx)
        return t

    @pytest.fixture
    def allow_private_transform(self, mock_ctx):
        """Transform with allowed_hosts: allow_private."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing allow_private",
                    "allowed_hosts": "allow_private",
                },
            }
        )
        t.on_start(mock_ctx)
        return t

    @respx.mock
    def test_loopback_allowed_with_config(self, allowed_loopback_transform, mock_ctx):
        """Loopback succeeds when explicitly allowed."""
        respx.get("http://127.0.0.1:80/page").mock(return_value=httpx.Response(200, text="<html>Local content</html>"))
        with patch("socket.getaddrinfo", _mock_getaddrinfo("127.0.0.1")):
            result = allowed_loopback_transform.process(make_pipeline_row({"url": "http://localhost/page"}), mock_ctx)
        assert result.status == "success"

    def test_non_allowed_private_still_blocked(self, allowed_loopback_transform, mock_ctx):
        """192.168.x.x still blocked when only 127.0.0.0/8 is allowed."""
        with patch("socket.getaddrinfo", _mock_getaddrinfo("192.168.1.1")):
            result = allowed_loopback_transform.process(make_pipeline_row({"url": "http://internal.example.com/page"}), mock_ctx)
        assert result.status == "error"
        assert result.reason["error_type"] == "SSRFBlockedError"

    def test_cloud_metadata_blocked_with_allow_private(self, allow_private_transform, mock_ctx):
        """Cloud metadata always blocked even with allow_private."""
        with patch("socket.getaddrinfo", _mock_getaddrinfo("169.254.169.254")):
            result = allow_private_transform.process(make_pipeline_row({"url": "http://metadata.internal/latest"}), mock_ctx)
        assert result.status == "error"
        assert result.reason["error_type"] == "SSRFBlockedError"

    @respx.mock
    def test_public_ip_still_works_with_default(self, transform, mock_ctx):
        """Default (public_only) still allows public IPs — regression check."""
        respx.get("https://93.184.216.34:443/page").mock(return_value=httpx.Response(200, text="<html>Content</html>"))
        with patch("socket.getaddrinfo", _mock_getaddrinfo("93.184.216.34")):
            result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)
        assert result.status == "success"


# ===========================================================================
# Redirect chain behavior tests (design doc requirement)
# ===========================================================================


class TestRedirectAllowedRangesBehavior:
    """Behavioral redirect tests with allowed_ranges active.

    Required by the design doc (2026-03-10-allowed-hosts-design.md):
    1. Redirect to allowed private IP succeeds
    2. Redirect to non-allowed private IP raises SSRFBlockedError
    3. Redirect to always-blocked IP (cloud metadata) despite allow_private raises SSRFBlockedError
    """

    @pytest.fixture
    def allowed_loopback_transform(self, mock_ctx):
        """Transform with allowed_hosts: ["127.0.0.0/8"]."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing redirect allowed_hosts",
                    "allowed_hosts": ["127.0.0.0/8"],
                },
            }
        )
        t.on_start(mock_ctx)
        return t

    @pytest.fixture
    def allow_private_transform(self, mock_ctx):
        """Transform with allowed_hosts: allow_private."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing allow_private redirects",
                    "allowed_hosts": "allow_private",
                },
            }
        )
        t.on_start(mock_ctx)
        return t

    def test_redirect_to_allowed_private_ip_succeeds(self, allowed_loopback_transform, mock_ctx):
        """Redirect chain where hop targets an allowed private IP — must succeed."""
        from elspeth.core.security.web import SSRFSafeRequest

        initial_safe = SSRFSafeRequest(
            original_url="http://example.com/start",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/start",
            scheme="http",
            bare_hostname="example.com",
        )
        redirect_safe = SSRFSafeRequest(
            original_url="http://localhost/redirected",
            resolved_ip="127.0.0.1",
            host_header="localhost",
            port=80,
            path="/redirected",
            scheme="http",
            bare_hostname="localhost",
        )

        with (
            patch(
                "elspeth.plugins.transforms.web_scrape.validate_url_for_ssrf",
                return_value=initial_safe,
            ),
            patch(
                "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                return_value=redirect_safe,
            ),
            patch("httpx.Client") as MockClient,
        ):
            shared_client = Mock()
            shared_client.close = Mock()
            initial_client = Mock()
            initial_client.__enter__ = Mock(return_value=initial_client)
            initial_client.__exit__ = Mock(return_value=False)
            initial_client.get.return_value = httpx.Response(
                301,
                headers={"location": "http://localhost/redirected"},
                request=httpx.Request("GET", "http://93.184.216.34:80/start"),
            )
            hop_client = Mock()
            hop_client.__enter__ = Mock(return_value=hop_client)
            hop_client.__exit__ = Mock(return_value=False)
            hop_client.get.return_value = httpx.Response(
                200,
                text="<html>Local content</html>",
                request=httpx.Request("GET", "http://127.0.0.1:80/redirected"),
            )
            MockClient.side_effect = [shared_client, initial_client, hop_client]

            result = allowed_loopback_transform.process(make_pipeline_row({"url": "http://example.com/start"}), mock_ctx)

        assert result.status == "success"
        assert result.row["fetch_url_final"] == "http://localhost/redirected"
        assert result.row["fetch_url_final_ip"] == "127.0.0.1"

    def test_redirect_to_non_allowed_private_ip_blocked(self, allowed_loopback_transform, mock_ctx):
        """Redirect chain where hop targets a non-allowed private IP — must block."""
        from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest

        initial_safe = SSRFSafeRequest(
            original_url="http://example.com/start",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/start",
            scheme="http",
            bare_hostname="example.com",
        )

        with (
            patch(
                "elspeth.plugins.transforms.web_scrape.validate_url_for_ssrf",
                return_value=initial_safe,
            ),
            patch(
                "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                side_effect=SSRFBlockedError("Blocked IP range: 192.168.1.1"),
            ),
            patch("httpx.Client") as MockClient,
        ):
            initial_client = Mock()
            initial_client.__enter__ = Mock(return_value=initial_client)
            initial_client.__exit__ = Mock(return_value=False)
            initial_client.get.return_value = httpx.Response(
                301,
                headers={"location": "http://192.168.1.1/internal"},
                request=httpx.Request("GET", "http://93.184.216.34:80/start"),
            )
            MockClient.return_value = initial_client

            result = allowed_loopback_transform.process(make_pipeline_row({"url": "http://example.com/start"}), mock_ctx)

        assert result.status == "error"
        assert "SSRFBlockedError" in result.reason.get("error_type", "")

    def test_redirect_to_cloud_metadata_blocked_even_allow_private(self, allow_private_transform, mock_ctx):
        """Redirect to always-blocked IP (cloud metadata) despite allow_private — must block."""
        from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest

        initial_safe = SSRFSafeRequest(
            original_url="http://example.com/start",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/start",
            scheme="http",
            bare_hostname="example.com",
        )

        with (
            patch(
                "elspeth.plugins.transforms.web_scrape.validate_url_for_ssrf",
                return_value=initial_safe,
            ),
            patch(
                "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                side_effect=SSRFBlockedError("Always-blocked IP range: 169.254.169.254"),
            ),
            patch("httpx.Client") as MockClient,
        ):
            initial_client = Mock()
            initial_client.__enter__ = Mock(return_value=initial_client)
            initial_client.__exit__ = Mock(return_value=False)
            initial_client.get.return_value = httpx.Response(
                301,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
                request=httpx.Request("GET", "http://93.184.216.34:80/start"),
            )
            MockClient.return_value = initial_client

            result = allow_private_transform.process(make_pipeline_row({"url": "http://example.com/start"}), mock_ctx)

        assert result.status == "error"
        assert "SSRFBlockedError" in result.reason.get("error_type", "")
