# tests/unit/contracts/transform_contracts/test_web_scrape_contract.py
"""Contract tests for WebScrapeTransform.

Note: WebScrapeTransform does NOT inherit BatchTransformMixin yet,
so we use TransformContractPropertyTestBase. This will change when
we add concurrency in a later task.
"""

from __future__ import annotations

import socket
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
from elspeth.testing import make_pipeline_row

from .test_transform_protocol import TransformContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol


_HTTPX_CLIENT_CLASS = httpx.Client
_TEST_IP = "93.184.216.34"
_METADATA_IP = "169.254.169.254"


def _mock_getaddrinfo(ip: str = _TEST_IP) -> Any:
    """Create a deterministic DNS resolver for SSRF validation."""

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
            mock_client_instance.get.return_value = mock_response
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

    def test_ssrf_blocked_url_returns_validation_error_before_external_call(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
        mock_httpx: Mock,
    ) -> None:
        """SSRF rejection is a pre-fetch validation result, not an audited HTTP call."""
        row = make_pipeline_row({"url": "https://metadata.example/latest/meta-data"})

        with patch("socket.getaddrinfo", side_effect=_mock_getaddrinfo(_METADATA_IP)):
            result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["error_type"] == "SSRFBlockedError"
        assert f"Always-blocked IP range: {_METADATA_IP}" in result.reason["error"]
        assert result.retryable is False

        _context_mock(ctx.landscape, "landscape").record_call.assert_not_called()
        _context_mock(ctx.payload_store, "payload_store").store.assert_not_called()
        mock_httpx.return_value.get.assert_not_called()

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
