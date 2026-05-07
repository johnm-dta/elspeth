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
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

from .test_transform_protocol import TransformContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol


_HTTPX_CLIENT_CLASS = httpx.Client
_TEST_IP = "93.184.216.34"


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
        # Create mock rate limiter
        mock_limiter = Mock()
        mock_limiter.try_acquire.return_value = True

        # Create mock rate limit registry
        mock_registry = Mock()
        mock_registry.get_limiter.return_value = mock_limiter

        # Create mock landscape recorder — must return a proper Call so process()
        # can read call.request_ref and call.response_ref without FrameworkBugError.
        mock_landscape = Mock()
        mock_call = Call(
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
        mock_landscape.record_call.return_value = mock_call
        mock_landscape.allocate_call_index.return_value = 0
        mock_landscape.store_payload.return_value = "test-processed-hash"

        # PayloadStore mock — WebScrapeTransform.process() stores processed
        # content via self._payload_store.store() captured during on_start().
        mock_payload_store = Mock()
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
