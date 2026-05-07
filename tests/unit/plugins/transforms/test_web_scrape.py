"""Tests for WebScrapeTransform plugin.

All tests mock socket.getaddrinfo so that validate_url_for_ssrf() produces a
known resolved IP, then mock respx to match the IP-based URL that
get_ssrf_safe() actually sends.
"""

import socket
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

from elspeth.contracts import CallStatus, CallType, check_compatibility
from elspeth.contracts.audit import Call
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.plugins.transforms.web_scrape_errors import (
    NetworkError,
    RateLimitError,
    ServerError,
)
from elspeth.testing import make_field, make_pipeline_row, make_row

# Stable test IP used for all DNS resolution mocks
_TEST_IP = "104.18.27.120"


def _mock_getaddrinfo(ip: str = _TEST_IP) -> Any:
    """Create a mock getaddrinfo that returns the given IP."""

    def _getaddrinfo(
        host: str,
        port: Any,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    return _getaddrinfo


@pytest.fixture
def mock_ctx():
    """Create PluginContext with required attributes for web scraping."""
    # Mock landscape recorder
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
    # Mock payload store (WebScrapeTransform uses self._payload_store.store())
    payload_store = Mock()
    payload_store.store.return_value = "test-processed-hash"

    # Mock rate limit registry
    rate_limit_registry = Mock()
    rate_limit_registry.get_limiter.return_value = None

    # Create context
    ctx = PluginContext(
        run_id="test-run-456",
        config={},
        landscape=landscape,
        payload_store=payload_store,
        rate_limit_registry=rate_limit_registry,
        state_id="state-123",
    )

    return ctx


def test_web_scrape_wires_on_success_and_on_error() -> None:
    """WebScrapeTransform routing is set via BaseTransform properties (bridge injection)."""
    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Unit testing web scrape transform",
            },
        }
    )

    # Routing is injected by the instantiation bridge, not config
    assert transform.on_success is None
    assert transform.on_error is None

    # Verify bridge-style injection works
    transform.on_success = "output"
    transform.on_error = "errors"
    assert transform.on_success == "output"
    assert transform.on_error == "errors"


@respx.mock
def test_web_scrape_success_markdown(mock_ctx):
    """Successful scrape should enrich row with content and fingerprint."""
    html_content = "<html><body><h1>Title</h1><p>Content here</p></body></html>"

    # Mock the IP-based URL that get_ssrf_safe() sends
    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Unit testing web scrape transform",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    assert "# Title" in result.row["page_content"]
    assert "Content here" in result.row["page_content"]
    assert result.row["page_fingerprint"] is not None
    assert len(result.row["page_fingerprint"]) == 64  # SHA-256
    assert result.row["fetch_status"] == 200


@respx.mock
def test_web_scrape_404_returns_error(mock_ctx):
    """404 should return error result (non-retryable)."""
    respx.get(f"https://{_TEST_IP}:443/missing").mock(return_value=httpx.Response(404))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/missing"}), mock_ctx)

    assert result.status == "error"
    assert "NotFoundError" in result.reason["error_type"]
    assert "404" in result.reason["error"]


@respx.mock
def test_web_scrape_500_raises_for_retry(mock_ctx):
    """HTTP 500 should raise ServerError (retryable)."""
    respx.get(f"https://{_TEST_IP}:443/error").mock(return_value=httpx.Response(500))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()), pytest.raises(ServerError) as exc_info:
        transform.process(make_pipeline_row({"url": "https://example.com/error"}), mock_ctx)

    assert exc_info.value.retryable is True
    assert "500" in str(exc_info.value)


@respx.mock
def test_web_scrape_429_raises_for_retry(mock_ctx):
    """HTTP 429 should raise RateLimitError (retryable)."""
    respx.get(f"https://{_TEST_IP}:443/throttled").mock(return_value=httpx.Response(429))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()), pytest.raises(RateLimitError) as exc_info:
        transform.process(make_pipeline_row({"url": "https://example.com/throttled"}), mock_ctx)

    assert exc_info.value.retryable is True
    assert "429" in str(exc_info.value)


def test_web_scrape_invalid_scheme_returns_error(mock_ctx):
    """Non-HTTP scheme should return error (security violation)."""
    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    result = transform.process(make_pipeline_row({"url": "ftp://example.com/file"}), mock_ctx)

    assert result.status == "error"
    assert "SSRFBlockedError" in result.reason["error_type"]
    assert "scheme" in result.reason["error"].lower()


def test_web_scrape_private_ip_returns_error(mock_ctx):
    """Private IP should return error (SSRF prevention)."""
    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo("169.254.169.254")):
        result = transform.process(make_pipeline_row({"url": "http://169.254.169.254/metadata"}), mock_ctx)

    assert result.status == "error"
    assert "SSRFBlockedError" in result.reason["error_type"]


@respx.mock
def test_web_scrape_text_format(mock_ctx):
    """Test text extraction format."""
    html_content = "<html><body><h1>Title</h1><p>Content here</p></body></html>"

    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "text",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    # Text format should not include markdown
    assert result.row["page_content"] == "Title Content here"
    assert "#" not in result.row["page_content"]
    assert "Title" in result.row["page_content"]
    assert "Content here" in result.row["page_content"]


@respx.mock
def test_web_scrape_text_format_uses_configured_separator(mock_ctx):
    """Text extraction can preserve line boundaries before line_explode consumes it."""
    html_content = "<html><body><h1>Title</h1><p>Content here</p><ul><li>One</li><li>Two</li></ul></body></html>"

    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "text",
            "text_separator": "\n",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    assert result.row["page_content"] == "Title\nContent here\nOne\nTwo"


@respx.mock
def test_web_scrape_strips_script_tags(mock_ctx):
    """Test that script tags are stripped by default."""
    html_content = """
    <html>
        <body>
            <h1>Title</h1>
            <script>alert('malicious');</script>
            <p>Content here</p>
        </body>
    </html>
    """

    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "text",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    assert "alert" not in result.row["page_content"]
    assert "malicious" not in result.row["page_content"]
    assert "Title" in result.row["page_content"]


@respx.mock
def test_web_scrape_payload_storage(mock_ctx):
    """Test that payload hashes are stored in success_reason metadata (not in row)."""
    html_content = "<html><body><h1>Title</h1></body></html>"

    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    assert result.status == "success"
    # Hash fields are now audit-only — they live in success_reason["metadata"], not in the row
    assert "fetch_request_hash" not in result.row
    assert "fetch_response_raw_hash" not in result.row
    assert "fetch_response_processed_hash" not in result.row
    # Verify hashes are in success_reason["metadata"]
    metadata = result.success_reason["metadata"]
    assert metadata["fetch_request_hash"] == "test-request-ref-hash"
    assert metadata["fetch_response_raw_hash"] == "test-response-ref-hash"
    assert metadata["fetch_response_processed_hash"] == "test-processed-hash"


@respx.mock
def test_web_scrape_framework_bug_error_when_call_refs_none(mock_ctx):
    """FrameworkBugError raised when Call.request_ref or response_ref is None.

    This guards against a misconfigured PayloadStore (no payload_store),
    which would produce Calls with None refs and silently lose audit provenance.
    """
    from elspeth.contracts.errors import FrameworkBugError

    # Override the fixture's mock_call to return None refs
    mock_call_no_refs = Call(
        call_id="test-call-id",
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.SUCCESS,
        request_hash="test-request-hash",
        created_at=datetime.now(UTC),
        state_id="state-123",
        request_ref=None,
        response_hash="test-response-hash",
        response_ref=None,
        latency_ms=100.0,
    )
    mock_ctx.landscape.record_call.return_value = mock_call_no_refs

    html_content = "<html><body><h1>Title</h1></body></html>"
    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()), pytest.raises(FrameworkBugError, match="request_ref/response_ref"):
        transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)


@respx.mock
def test_web_scrape_timeout_raises_network_error(mock_ctx):
    """Timeout should raise NetworkError (retryable)."""
    # Mock timeout by raising httpx.TimeoutException on the IP-based URL
    respx.get(f"https://{_TEST_IP}:443/slow").mock(side_effect=httpx.TimeoutException("Connection timeout"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
                "timeout": 1,
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()), pytest.raises(NetworkError) as exc_info:
        transform.process(make_pipeline_row({"url": "https://example.com/slow"}), mock_ctx)

    assert exc_info.value.retryable is True
    assert "timeout" in str(exc_info.value).lower()


@respx.mock
def test_web_scrape_connection_error_raises_network_error(mock_ctx):
    """Connection error should raise NetworkError (retryable)."""
    respx.get(f"https://{_TEST_IP}:443/unreachable").mock(side_effect=httpx.ConnectError("Connection refused"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()), pytest.raises(NetworkError) as exc_info:
        transform.process(make_pipeline_row({"url": "https://example.com/unreachable"}), mock_ctx)

    assert exc_info.value.retryable is True
    assert "connection" in str(exc_info.value).lower()


@respx.mock
def test_web_scrape_403_returns_error(mock_ctx):
    """HTTP 403 should return error result (non-retryable)."""
    respx.get(f"https://{_TEST_IP}:443/forbidden").mock(return_value=httpx.Response(403))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/forbidden"}), mock_ctx)

    assert result.status == "error"
    assert "ForbiddenError" in result.reason["error_type"]
    assert "403" in result.reason["error"]


@respx.mock
def test_web_scrape_401_returns_error(mock_ctx):
    """HTTP 401 should return error result (non-retryable)."""
    respx.get(f"https://{_TEST_IP}:443/unauthorized").mock(return_value=httpx.Response(401))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/unauthorized"}), mock_ctx)

    assert result.status == "error"
    assert "UnauthorizedError" in result.reason["error_type"]
    assert "401" in result.reason["error"]


@respx.mock
def test_web_scrape_with_pipeline_row(mock_ctx):
    """Test that web_scrape works correctly with PipelineRow input."""
    html_content = "<html><body><h1>Test</h1></body></html>"
    respx.get(f"https://{_TEST_IP}:443/test").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing PipelineRow compatibility",
            },
        }
    )
    transform.on_start(mock_ctx)

    # Create PipelineRow input (simulates what engine passes to transforms)
    fields = (make_field("url", str, original_name="url", required=True, source="declared"),)
    contract = SchemaContract(mode="FIXED", fields=fields, locked=True)
    pipeline_row = make_row({"url": "https://example.com/test"}, contract=contract)

    # Process should work with PipelineRow (uses row.to_dict() internally)
    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(pipeline_row, mock_ctx)

    assert result.status == "success"
    assert "# Test" in result.row["page_content"]
    assert result.row["page_fingerprint"] is not None
    assert result.row["fetch_status"] == 200


@respx.mock
def test_web_scrape_output_contract_matches_declared_enriched_fields(mock_ctx):
    """Declared web_scrape fields must keep declared metadata at emission."""
    html_content = "<html><body><h1>Title</h1></body></html>"
    respx.get(f"https://{_TEST_IP}:443/page").mock(return_value=httpx.Response(200, text=html_content))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing declared output contract metadata",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/page"}), mock_ctx)

    field_by_name = {field.normalized_name: field for field in result.row.contract.fields}
    for field_name in (
        "page_content",
        "page_fingerprint",
        "fetch_status",
        "fetch_url_final",
        "fetch_url_final_ip",
    ):
        assert field_by_name[field_name].required is True
        assert field_by_name[field_name].source == "declared"


@respx.mock
def test_web_scrape_follows_redirects_301(mock_ctx):
    """HTTP 301 redirect should be followed and final URL recorded.

    Edge case: 301 Moved Permanently redirects are common for URL migrations.
    The scraper should follow the redirect and record both the requested URL
    and the final URL after redirect resolution. Both the logical hostname URL
    and the IP-based connection URL are recorded for audit comparison.
    """
    respx.get(f"https://{_TEST_IP}:443/old").mock(return_value=httpx.Response(301, headers={"Location": "https://example.com/new"}))
    respx.get(f"https://{_TEST_IP}:443/new").mock(return_value=httpx.Response(200, text="<html><body><h1>New Location</h1></body></html>"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing redirect handling",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/old"}), mock_ctx)

    assert result.status == "success"
    assert "# New Location" in result.row["page_content"]
    assert result.row["fetch_status"] == 200
    assert result.row["fetch_url_final"] == "https://example.com/new"
    assert result.row["fetch_url_final_ip"] == _TEST_IP


@respx.mock
def test_web_scrape_follows_redirect_chain(mock_ctx):
    """Multiple redirects (301->302->200) should be followed to final destination."""
    respx.get(f"https://{_TEST_IP}:443/start").mock(return_value=httpx.Response(301, headers={"Location": "https://example.com/middle"}))
    respx.get(f"https://{_TEST_IP}:443/middle").mock(return_value=httpx.Response(302, headers={"Location": "https://example.com/end"}))
    respx.get(f"https://{_TEST_IP}:443/end").mock(
        return_value=httpx.Response(200, text="<html><body><h1>Final Destination</h1></body></html>")
    )

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing redirect chain",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/start"}), mock_ctx)

    assert result.status == "success"
    assert "# Final Destination" in result.row["page_content"]
    assert result.row["fetch_status"] == 200
    assert result.row["fetch_url_final"] == "https://example.com/end"
    assert result.row["fetch_url_final_ip"] == _TEST_IP


@respx.mock
def test_web_scrape_redirect_limit_exceeded(mock_ctx):
    """Excessive redirects should return non-retryable error result.

    A redirect loop is a configuration problem, not a transient failure.
    The TooManyRedirects exception from httpx is caught by _fetch_url and
    re-raised as InvalidURLError (non-retryable), which process() converts
    to a TransformResult.error() for row quarantine.
    """
    respx.get(f"https://{_TEST_IP}:443/a").mock(return_value=httpx.Response(301, headers={"Location": "https://example.com/b"}))
    respx.get(f"https://{_TEST_IP}:443/b").mock(return_value=httpx.Response(301, headers={"Location": "https://example.com/a"}))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing redirect limit",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/a"}), mock_ctx)

    assert result.status == "error"
    assert "InvalidURLError" in result.reason["error_type"]
    assert "redirect" in result.reason["error"].lower()


@respx.mock
def test_web_scrape_malformed_html_graceful_degradation(mock_ctx):
    """Malformed HTML should still extract text content without crashing."""
    malformed_html = """
    <html>
    <body>
        <h1>Title
        <p>Paragraph without closing tag
        <div>
            <p>Nested paragraph
        </div>
        <!-- Missing closing tags for body and html -->
    """

    respx.get(f"https://{_TEST_IP}:443/malformed").mock(return_value=httpx.Response(200, text=malformed_html))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "fingerprint_mode": "content",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing malformed HTML",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/malformed"}), mock_ctx)

    # Should succeed despite malformed HTML
    assert result.status == "success"
    # Content should be extracted (BeautifulSoup auto-closes tags)
    assert "Title" in result.row["page_content"]
    assert "Paragraph without closing tag" in result.row["page_content"]
    assert "Nested paragraph" in result.row["page_content"]
    assert result.row["fetch_status"] == 200


@respx.mock
def test_web_scrape_extract_content_exception_returns_error(mock_ctx):
    """When extract_content() raises, process() returns TransformResult.error() instead of crashing.

    This is the Tier 3 boundary protection: response.text is external data and parsing it
    can fail in ways we can't predict. The pipeline must quarantine the row, not crash.
    """
    respx.get(f"https://{_TEST_IP}:443/bad").mock(return_value=httpx.Response(200, text="<html>valid</html>"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing parse error",
            },
        }
    )
    transform.on_start(mock_ctx)

    # Simulate extract_content() raising an unexpected exception
    with (
        patch("socket.getaddrinfo", _mock_getaddrinfo()),
        patch(
            "elspeth.plugins.transforms.web_scrape.extract_content",
            side_effect=RuntimeError("html2text internal error"),
        ),
    ):
        result = transform.process(make_pipeline_row({"url": "https://example.com/bad"}), mock_ctx)

    assert result.status == "error"
    assert result.reason["reason"] == "content_extraction_failed"
    assert "html2text internal error" in result.reason["error"]
    assert result.reason["error_type"] == "RuntimeError"
    assert result.reason["url"] == "https://example.com/bad"


@respx.mock
def test_web_scrape_extract_content_tier3_valueerror_returns_error(mock_ctx):
    """Tier 3 boundary: ValueError from extract_content (wrapping library exceptions) returns error.

    extract_content() catches AttributeError/TypeError from BeautifulSoup/html2text
    on malformed HTML and re-raises as ValueError. The caller catches ValueError
    and returns TransformResult.error() — pipeline continues, row is quarantined.
    """
    respx.get(f"https://{_TEST_IP}:443/malformed").mock(return_value=httpx.Response(200, text="<html>malformed</html>"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing Tier 3 parse error",
            },
        }
    )
    transform.on_start(mock_ctx)

    with (
        patch("socket.getaddrinfo", _mock_getaddrinfo()),
        patch(
            "elspeth.plugins.transforms.web_scrape.extract_content",
            side_effect=ValueError("HTML extraction failed on malformed content: NoneType"),
        ),
    ):
        result = transform.process(make_pipeline_row({"url": "https://example.com/malformed"}), mock_ctx)

    assert result.status == "error"
    assert result.reason["reason"] == "content_extraction_failed"
    assert result.reason["error_type"] == "ValueError"


@respx.mock
def test_web_scrape_binary_response_does_not_crash(mock_ctx):
    """Binary content in response.text should be handled gracefully.

    When a server returns binary data (image, PDF) with a text content-type,
    httpx will attempt UTF-8 decoding. If extraction fails on the result,
    we should get an error result, not a pipeline crash.
    """
    # Binary data that httpx will lossy-decode to replacement characters
    binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff\xfe" * 100
    respx.get(f"https://{_TEST_IP}:443/binary").mock(
        return_value=httpx.Response(200, content=binary_content, headers={"content-type": "text/html"})
    )

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing binary response",
            },
        }
    )
    transform.on_start(mock_ctx)

    with patch("socket.getaddrinfo", _mock_getaddrinfo()):
        result = transform.process(make_pipeline_row({"url": "https://example.com/binary"}), mock_ctx)

    # Should either succeed (BeautifulSoup is very forgiving) or return error — never crash
    assert result.status in ("success", "error")


@respx.mock
def test_web_scrape_unicode_decode_error_returns_error(mock_ctx):
    """UnicodeDecodeError during extraction returns error result."""
    respx.get(f"https://{_TEST_IP}:443/encoding").mock(return_value=httpx.Response(200, text="<html>ok</html>"))

    transform = WebScrapeTransform(
        {
            "schema": {"mode": "observed"},
            "url_field": "url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing encoding error",
            },
        }
    )
    transform.on_start(mock_ctx)

    with (
        patch("socket.getaddrinfo", _mock_getaddrinfo()),
        patch(
            "elspeth.plugins.transforms.web_scrape.extract_content",
            side_effect=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        ),
    ):
        result = transform.process(make_pipeline_row({"url": "https://example.com/encoding"}), mock_ctx)

    assert result.status == "error"
    assert result.reason["reason"] == "content_extraction_failed"
    assert result.reason["error_type"] == "UnicodeDecodeError"


# --- Config validation tests (WebScrapeHTTPConfig sub-model) ---


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a valid WebScrapeTransform config dict, with overrides applied."""
    cfg: dict[str, Any] = {
        "schema": {"mode": "observed"},
        "url_field": "url",
        "content_field": "page_content",
        "fingerprint_field": "page_fingerprint",
        "http": {
            "abuse_contact": "test@example.com",
            "scraping_reason": "Testing",
        },
    }
    cfg.update(overrides)
    return cfg


def test_http_config_missing_abuse_contact_raises() -> None:
    """Missing required abuse_contact must raise PluginConfigError."""
    with pytest.raises(PluginConfigError, match="abuse_contact"):
        WebScrapeTransform(_base_config(http={"scraping_reason": "Testing"}))


def test_http_config_missing_scraping_reason_raises() -> None:
    """Missing required scraping_reason must raise PluginConfigError."""
    with pytest.raises(PluginConfigError, match="scraping_reason"):
        WebScrapeTransform(_base_config(http={"abuse_contact": "test@example.com"}))


def test_http_config_extra_field_raises() -> None:
    """Unknown fields in http config must be rejected (extra=forbid)."""
    with pytest.raises(PluginConfigError, match="extra_field"):
        WebScrapeTransform(
            _base_config(
                http={
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing",
                    "extra_field": "should fail",
                }
            )
        )


def test_http_config_timeout_zero_raises() -> None:
    """Timeout must be > 0."""
    with pytest.raises(PluginConfigError, match="timeout"):
        WebScrapeTransform(
            _base_config(
                http={
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing",
                    "timeout": 0,
                }
            )
        )


def test_http_config_timeout_negative_raises() -> None:
    """Negative timeout must be rejected."""
    with pytest.raises(PluginConfigError, match="timeout"):
        WebScrapeTransform(
            _base_config(
                http={
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing",
                    "timeout": -5,
                }
            )
        )


def test_http_config_timeout_default() -> None:
    """Timeout defaults to 30 when not specified."""
    transform = WebScrapeTransform(_base_config())
    assert transform._timeout == 30


def test_http_config_timeout_custom() -> None:
    """Custom timeout value is respected."""
    transform = WebScrapeTransform(
        _base_config(
            http={
                "abuse_contact": "test@example.com",
                "scraping_reason": "Testing",
                "timeout": 60,
            }
        )
    )
    assert transform._timeout == 60


def test_web_scrape_forward_probe_preserves_baseline_and_restores_payload_store(mock_ctx) -> None:
    """Invariant probe should use a hermetic fetch seam and restore injected state."""
    transform = WebScrapeTransform(WebScrapeTransform.probe_config())

    assert WebScrapeTransform.passes_through_input is True

    original_fetch = transform._fetch_url
    original_payload_store = Mock()
    original_payload_store.store.return_value = "existing-hash"
    transform._payload_store = original_payload_store

    base_row = make_pipeline_row({"baseline": "kept"})
    result = transform.execute_forward_invariant_probe(
        transform.forward_invariant_probe_rows(base_row),
        mock_ctx,
    )

    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"
    assert result.row["page_content"]
    assert result.row["page_fingerprint"]
    assert result.row["fetch_status"] == 200
    assert transform._payload_store is original_payload_store
    assert transform._fetch_url.__func__ is original_fetch.__func__


class TestWebScrapeDeclaredOutputFields:
    """Tests for declared_output_fields — centralized collision detection support.

    Field collision detection is enforced centrally by TransformExecutor
    (see TestTransformExecutor in test_executors.py). These tests verify
    that WebScrapeTransform correctly declares its output fields so the
    executor can perform pre-execution collision checks.
    """

    def test_declared_output_fields_contains_hardcoded_fields(self):
        """declared_output_fields includes hardcoded fetch_* operational fields.

        Hash fields are audit-only (in success_reason["metadata"]) — not in declared_output_fields.
        """
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing declared fields",
                },
            }
        )

        assert "fetch_status" in transform.declared_output_fields
        assert "fetch_url_final" in transform.declared_output_fields
        assert "fetch_url_final_ip" in transform.declared_output_fields
        # Hash fields are audit-only — not declared as output fields
        assert "fetch_request_hash" not in transform.declared_output_fields
        assert "fetch_response_raw_hash" not in transform.declared_output_fields
        assert "fetch_response_processed_hash" not in transform.declared_output_fields

    def test_declared_output_fields_contains_configurable_fields(self):
        """declared_output_fields includes configurable content and fingerprint fields."""
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing declared fields",
                },
            }
        )

        assert "page_content" in transform.declared_output_fields
        assert "page_fingerprint" in transform.declared_output_fields

    def test_declared_output_fields_adapts_to_config(self):
        """declared_output_fields changes when content/fingerprint field names change."""
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "scraped_html",
                "fingerprint_field": "content_hash",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing declared fields",
                },
            }
        )

        assert "scraped_html" in transform.declared_output_fields
        assert "content_hash" in transform.declared_output_fields
        # Old names should NOT be present
        assert "page_content" not in transform.declared_output_fields
        assert "page_fingerprint" not in transform.declared_output_fields

    def test_declared_output_fields_drives_schema_evolution(self):
        """declared_output_fields is non-empty, enabling schema evolution."""
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Testing declared fields",
                },
            }
        )

        assert transform.declared_output_fields


# ===========================================================================
# allowed_hosts config validation
# ===========================================================================


class TestAllowedHostsConfig:
    """Config validation for allowed_hosts field."""

    def test_default_is_public_only(self) -> None:
        """Default allowed_hosts is public_only (no allowlist)."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                },
            }
        )
        assert t._allowed_ranges == ()

    def test_public_only_keyword(self) -> None:
        """public_only keyword produces empty allowed_ranges."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": "public_only",
                },
            }
        )
        assert t._allowed_ranges == ()

    def test_allow_private_keyword(self) -> None:
        """allow_private keyword produces 0.0.0.0/0 + ::/0."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": "allow_private",
                },
            }
        )
        range_strs = {str(r) for r in t._allowed_ranges}
        assert "0.0.0.0/0" in range_strs
        assert "::/0" in range_strs

    def test_cidr_list(self) -> None:
        """List of CIDR ranges parsed correctly."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": ["127.0.0.0/8", "10.0.0.0/8"],
                },
            }
        )
        range_strs = {str(r) for r in t._allowed_ranges}
        assert "127.0.0.0/8" in range_strs
        assert "10.0.0.0/8" in range_strs

    def test_single_ip_expanded_to_32(self) -> None:
        """Single IP address expanded to /32."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": ["127.0.0.1"],
                },
            }
        )
        range_strs = {str(r) for r in t._allowed_ranges}
        assert "127.0.0.1/32" in range_strs

    def test_ipv6_cidr_accepted(self) -> None:
        """IPv6 CIDR entries are accepted."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": ["::1/128"],
                },
            }
        )
        assert len(t._allowed_ranges) == 1

    def test_empty_list_rejected(self) -> None:
        """Empty list is rejected (ambiguous — use allow_private if intended)."""
        with pytest.raises((PluginConfigError, ValueError)):
            WebScrapeTransform(
                {
                    "schema": {"mode": "observed"},
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "http": {
                        "abuse_contact": "test@example.com",
                        "scraping_reason": "Test",
                        "allowed_hosts": [],
                    },
                }
            )

    def test_invalid_cidr_rejected(self) -> None:
        """Unparseable CIDR entry crashes at config time."""
        with pytest.raises((PluginConfigError, ValueError)):
            WebScrapeTransform(
                {
                    "schema": {"mode": "observed"},
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "http": {
                        "abuse_contact": "test@example.com",
                        "scraping_reason": "Test",
                        "allowed_hosts": ["not-a-cidr"],
                    },
                }
            )

    def test_invalid_keyword_rejected(self) -> None:
        """Unknown string keyword is rejected."""
        with pytest.raises((PluginConfigError, ValueError)):
            WebScrapeTransform(
                {
                    "schema": {"mode": "observed"},
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "http": {
                        "abuse_contact": "test@example.com",
                        "scraping_reason": "Test",
                        "allowed_hosts": "allow_all",
                    },
                }
            )

    def test_keyword_case_sensitive(self) -> None:
        """Keywords are case-sensitive — 'Public_Only' is rejected."""
        with pytest.raises((PluginConfigError, ValueError)):
            WebScrapeTransform(
                {
                    "schema": {"mode": "observed"},
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "http": {
                        "abuse_contact": "test@example.com",
                        "scraping_reason": "Test",
                        "allowed_hosts": "Public_Only",
                    },
                }
            )

    def test_host_bits_set_normalized_by_strict_false(self) -> None:
        """CIDR with host bits set is silently normalized via strict=False."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": ["127.0.0.1/8"],
                },
            }
        )
        range_strs = {str(r) for r in t._allowed_ranges}
        assert "127.0.0.0/8" in range_strs

    def test_always_blocked_overlap_accepted_in_config(self) -> None:
        """Entries overlapping ALWAYS_BLOCKED_RANGES are accepted in config."""
        t = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Test",
                    "allowed_hosts": ["169.254.0.0/16"],
                },
            }
        )
        assert len(t._allowed_ranges) == 1


# ===========================================================================
# _parse_allowed_ranges unit tests
# ===========================================================================


class TestParseAllowedRanges:
    """Direct unit tests for _parse_allowed_ranges helper."""

    def test_mixed_ipv4_and_ipv6(self) -> None:
        """Mixed address families in a single list."""
        from elspeth.plugins.transforms.web_scrape import _parse_allowed_ranges

        result = _parse_allowed_ranges(["10.0.0.0/8", "::1/128"])
        assert len(result) == 2
        range_strs = {str(r) for r in result}
        assert "10.0.0.0/8" in range_strs
        assert "::1/128" in range_strs

    def test_single_ipv6_expanded_to_128(self) -> None:
        """Single IPv6 address without prefix is expanded to /128."""
        from elspeth.plugins.transforms.web_scrape import _parse_allowed_ranges

        result = _parse_allowed_ranges(["::1"])
        assert str(result[0]) == "::1/128"

    def test_host_bits_normalized(self) -> None:
        """Host bits are cleared by strict=False."""
        from elspeth.plugins.transforms.web_scrape import _parse_allowed_ranges

        result = _parse_allowed_ranges(["10.0.0.1/8"])
        assert str(result[0]) == "10.0.0.0/8"

    def test_returns_tuple(self) -> None:
        """Return type is tuple (immutable)."""
        from elspeth.plugins.transforms.web_scrape import _parse_allowed_ranges

        result = _parse_allowed_ranges(["127.0.0.0/8"])
        assert isinstance(result, tuple)


class TestOutputSchemaConfig:
    def test_fixed_input_output_schema_exposes_enriched_fields_for_type_validation(self):
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "fixed", "fields": ["url: str"]},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_hash",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Unit testing output schema config",
                },
            }
        )
        consumer_schema = create_schema_from_config(
            SchemaConfig.from_dict(
                {
                    "mode": "flexible",
                    "fields": [
                        "page_content: str",
                        "page_hash: str",
                        "fetch_status: int",
                        "fetch_url_final: str",
                        "fetch_url_final_ip: str",
                    ],
                }
            ),
            "WebScrapeDownstreamConsumer",
            allow_coercion=False,
        )

        result = check_compatibility(transform.output_schema, consumer_schema)

        assert result.compatible, result.error_message
        output_fields = transform.output_schema.model_fields
        assert output_fields["page_content"].annotation is str
        assert output_fields["page_hash"].annotation is str
        assert output_fields["fetch_status"].annotation is int
        assert output_fields["fetch_url_final"].annotation is str
        assert output_fields["fetch_url_final_ip"].annotation is str

        assert transform._output_schema_config is not None
        config_field_types = {field.name: field.field_type for field in transform._output_schema_config.fields or ()}
        assert config_field_types["page_content"] == "str"
        assert config_field_types["page_hash"] == "str"
        assert config_field_types["fetch_status"] == "int"
        assert config_field_types["fetch_url_final"] == "str"
        assert config_field_types["fetch_url_final_ip"] == "str"

    def test_guaranteed_fields(self):
        transform = WebScrapeTransform(
            {
                "schema": {"mode": "observed"},
                "url_field": "url",
                "content_field": "page_content",
                "fingerprint_field": "page_hash",
                "http": {
                    "abuse_contact": "test@example.com",
                    "scraping_reason": "Unit testing output schema config",
                },
            }
        )
        expected = frozenset(
            {
                "page_content",
                "page_hash",
                "fetch_status",
                "fetch_url_final",
                "fetch_url_final_ip",
            }
        )
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == expected


class TestWebScrapeOutputSemantics:
    def _build(self, **option_overrides):
        # WebScrapeConfig is a TransformDataConfig subclass — schema is
        # REQUIRED at construction. Omitting it raises PluginConfigError
        # which the validator's tolerant probe path silently absorbs;
        # the test would then pass vacuously without exercising
        # output_semantics() at all.
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        defaults = {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "fingerprint",
            "format": "markdown",
            "http": {
                "abuse_contact": "x@example.com",
                "scraping_reason": "t",
                "timeout": 5,
                "allowed_hosts": "public_only",
            },
        }
        defaults.update(option_overrides)
        return get_shared_plugin_manager().create_transform("web_scrape", defaults)

    def test_text_compact_separator_declares_plain_text_compact(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming

        ws = self._build(format="text", text_separator=" ")
        decl = ws.output_semantics()
        facts = next(f for f in decl.fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.COMPACT
        assert facts.fact_code == "web_scrape.content.compact_text"
        assert facts.configured_by == ("format", "text_separator")

    def test_text_newline_separator_declares_plain_text_newline_framed(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming

        ws = self._build(format="text", text_separator="\n")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.NEWLINE_FRAMED

    def test_markdown_declares_markdown_line_compatible(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming

        ws = self._build(format="markdown")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.MARKDOWN
        assert facts.text_framing is TextFraming.LINE_COMPATIBLE

    def test_raw_declares_html_raw_not_text(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming

        ws = self._build(format="raw")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.HTML_RAW
        assert facts.text_framing is TextFraming.NOT_TEXT

    def test_custom_content_field_changes_semantic_field_name(self):
        ws = self._build(format="text", text_separator="\n", content_field="body")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "body")
        assert facts.field_name == "body"


class TestWebScrapeAssistance:
    def test_returns_assistance_for_compact_text_issue(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        result = WebScrapeTransform.get_agent_assistance(
            issue_code="web_scrape.content.compact_text",
        )
        assert result is not None
        assert result.plugin_name == "web_scrape"
        assert result.issue_code == "web_scrape.content.compact_text"
        # Suggested fixes mention configuration knobs only — no values.
        assert any("text_separator" in fix for fix in result.suggested_fixes)
        assert any("markdown" in fix.lower() for fix in result.suggested_fixes)

    def test_returns_none_for_unknown_issue(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        assert WebScrapeTransform.get_agent_assistance(issue_code="nope.unknown") is None

    def test_returns_none_when_no_issue_code(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        assert WebScrapeTransform.get_agent_assistance(issue_code=None) is None

    def test_assistance_does_not_leak_secret_options(self):
        """Sentinel test: configured option values must not bleed into assistance prose.

        Construct a plugin with sentinel-laced options (abuse_contact, scraping_reason
        — the realistic credential-shaped leak surface) and assert the returned
        PluginAssistance carries none of those raw values. Also covers
        output_semantics() against accidental value-bearing fields.

        Field names must be Python identifiers (config layer enforces this), so
        the sentinel only appears in HTTP option values where leakage would
        actually be a B5 incident.
        """
        from elspeth.contracts.plugin_assistance import PluginAssistance, PluginAssistanceExample
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        sentinel_contact = "SENTINEL_LEAK_CONTACT@example.com"
        sentinel_reason = "SENTINEL_LEAK_REASON_credential_shape"

        plugin = get_shared_plugin_manager().create_transform(
            "web_scrape",
            {
                "schema": {"mode": "flexible", "fields": ["url: str"]},
                "required_input_fields": ["url"],
                "url_field": "url",
                "content_field": "content",
                "fingerprint_field": "fingerprint",
                "format": "text",
                "text_separator": " ",
                "http": {
                    "abuse_contact": sentinel_contact,
                    "scraping_reason": sentinel_reason,
                    "timeout": 5,
                    "allowed_hosts": "public_only",
                },
            },
        )

        # output_semantics() must not echo any HTTP option value.
        decl = plugin.output_semantics()
        for fact in decl.fields:
            assert sentinel_contact not in fact.field_name
            assert sentinel_contact not in fact.fact_code
            assert all(sentinel_contact not in entry for entry in fact.configured_by)
            assert sentinel_reason not in fact.field_name
            assert sentinel_reason not in fact.fact_code
            assert all(sentinel_reason not in entry for entry in fact.configured_by)

        result = WebScrapeTransform.get_agent_assistance(
            issue_code="web_scrape.content.compact_text",
        )
        assert result is not None
        assert isinstance(result, PluginAssistance)

        def _scan(text: str) -> None:
            assert sentinel_contact not in text
            assert sentinel_reason not in text

        _scan(result.plugin_name)
        if result.issue_code is not None:
            _scan(result.issue_code)
        _scan(result.summary)
        for fix in result.suggested_fixes:
            _scan(fix)
        for hint in result.composer_hints:
            _scan(hint)
        for example in result.examples:
            assert isinstance(example, PluginAssistanceExample)
            _scan(example.title)
            for mapping_field in (example.before, example.after):
                if mapping_field is None:
                    continue
                for key, value in mapping_field.items():
                    _scan(str(key))
                    _scan(str(value))


class TestWebScrapeSecretLeakage:
    SENTINEL = "PASSWORD_SENTINEL_x9q7r3"

    def test_sentinel_url_not_in_output_semantics_or_assistance(self):
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        ws = get_shared_plugin_manager().create_transform(
            "web_scrape",
            {
                "schema": {"mode": "flexible", "fields": ["url: str"]},
                "required_input_fields": ["url"],
                "url_field": "url",
                "content_field": f"content_{self.SENTINEL}",  # field name SHOULD appear
                "fingerprint_field": "fingerprint",
                "format": "text",
                "text_separator": " ",
                "http": {
                    "abuse_contact": f"x+{self.SENTINEL}@example.com",
                    "scraping_reason": f"reason-{self.SENTINEL}",
                    "timeout": 5,
                    "allowed_hosts": "public_only",
                },
            },
        )

        # Output semantics: the configured content_field name DOES include
        # the sentinel - that's not a leak, the user wrote that field name.
        # What MUST NOT appear: the abuse_contact email, the scraping_reason.
        decl = ws.output_semantics()
        decl_repr = repr(decl)
        assert f"x+{self.SENTINEL}" not in decl_repr
        assert f"reason-{self.SENTINEL}" not in decl_repr

        # Assistance is class-level, no instance state - but verify anyway.
        assistance = WebScrapeTransform.get_agent_assistance(
            issue_code="web_scrape.content.compact_text",
        )
        assistance_repr = repr(assistance)
        assert self.SENTINEL not in assistance_repr
