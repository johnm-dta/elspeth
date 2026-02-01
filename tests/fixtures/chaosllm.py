# tests/fixtures/chaosllm.py
"""Pytest fixture for ChaosLLM fake LLM server testing.

This module provides the `chaosllm_server` fixture which creates an in-process
ChaosLLM server using Starlette's TestClient. This is faster and simpler than
subprocess-based server management.

Usage:
    # Basic usage with defaults
    def test_llm_client(chaosllm_server):
        response = requests.post(
            f"{chaosllm_server.url}/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert response.status_code == 200

    # Using presets via pytest marker
    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_under_stress(chaosllm_server):
        # Server configured with stress_aimd preset
        ...

    # Override specific settings via marker
    @pytest.mark.chaosllm(preset="gentle", rate_limit_pct=20.0)
    def test_with_some_errors(chaosllm_server):
        ...

    # Runtime configuration updates
    def test_dynamic_config(chaosllm_server):
        chaosllm_server.update_config(rate_limit_pct=50.0)
        # Now 50% of requests will get 429
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from starlette.testclient import TestClient

from elspeth.testing.chaosllm.config import (
    ChaosLLMConfig,
    load_config,
)
from elspeth.testing.chaosllm.server import ChaosLLMServer

if TYPE_CHECKING:
    import httpx


@dataclass
class ChaosLLMFixture:
    """Pytest fixture object for ChaosLLM server.

    Provides a convenient interface for test interaction with the ChaosLLM
    fake LLM server. Uses Starlette's TestClient for in-process testing.

    Attributes:
        client: Starlette TestClient for making HTTP requests
        server: ChaosLLMServer instance with access to internal state

    Example:
        def test_completion(chaosllm_server):
            response = chaosllm_server.client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": []}
            )
            assert response.status_code == 200
    """

    client: TestClient
    server: ChaosLLMServer
    metrics_db_path: Path
    _request_count: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def url(self) -> str:
        """Get the base URL for the server.

        Note: TestClient makes requests in-process, so this URL is just
        a placeholder. The actual base URL used by TestClient is handled
        internally. Use this for constructing full URL paths when needed.
        """
        return "http://testserver"

    @property
    def port(self) -> int:
        """Get the server port.

        Note: TestClient uses in-process testing, so there's no real port.
        This returns a placeholder value for API compatibility.
        """
        return 8000

    @property
    def metrics_db(self) -> Path:
        """Get the path to the metrics SQLite database."""
        return self.metrics_db_path

    @property
    def admin_url(self) -> str:
        """Get the admin endpoint base URL."""
        return f"{self.url}/admin"

    @property
    def run_id(self) -> str:
        """Get the current run ID from the server."""
        return self.server.run_id

    def get_stats(self) -> dict[str, Any]:
        """Get current server statistics.

        Returns:
            Dictionary with:
            - run_id: Current run identifier
            - started_utc: Run start time
            - total_requests: Total requests processed
            - requests_by_outcome: Count by outcome type
            - requests_by_status_code: Count by HTTP status
            - latency_stats: Latency percentiles
            - error_rate: Percentage of non-success requests
        """
        return self.server.get_stats()

    def export_metrics(self) -> dict[str, Any]:
        """Export raw metrics data for pushback."""
        return self.server.export_metrics()

    def update_config(
        self,
        *,
        # Error injection percentages
        rate_limit_pct: float | None = None,
        capacity_529_pct: float | None = None,
        service_unavailable_pct: float | None = None,
        bad_gateway_pct: float | None = None,
        gateway_timeout_pct: float | None = None,
        internal_error_pct: float | None = None,
        timeout_pct: float | None = None,
        connection_reset_pct: float | None = None,
        slow_response_pct: float | None = None,
        invalid_json_pct: float | None = None,
        truncated_pct: float | None = None,
        empty_body_pct: float | None = None,
        missing_fields_pct: float | None = None,
        wrong_content_type_pct: float | None = None,
        selection_mode: str | None = None,
        # Latency settings
        base_ms: int | None = None,
        jitter_ms: int | None = None,
        # Response settings
        mode: str | None = None,
    ) -> None:
        """Update server configuration at runtime.

        All parameters are optional. Only specified parameters will be updated.

        Args:
            rate_limit_pct: 429 Rate Limit error percentage
            capacity_529_pct: 529 Capacity error percentage
            service_unavailable_pct: 503 Service Unavailable percentage
            bad_gateway_pct: 502 Bad Gateway percentage
            gateway_timeout_pct: 504 Gateway Timeout percentage
            internal_error_pct: 500 Internal Error percentage
            timeout_pct: Connection timeout percentage
            connection_reset_pct: Connection reset percentage
            slow_response_pct: Slow response percentage
            invalid_json_pct: Invalid JSON response percentage
            truncated_pct: Truncated response percentage
            empty_body_pct: Empty body response percentage
            missing_fields_pct: Missing fields response percentage
            wrong_content_type_pct: Wrong Content-Type percentage
            selection_mode: Error selection strategy (priority or weighted)
            base_ms: Base latency in milliseconds
            jitter_ms: Latency jitter in milliseconds
            mode: Response generation mode
        """
        updates: dict[str, Any] = {}

        # Collect error injection updates
        error_updates: dict[str, float] = {}
        if rate_limit_pct is not None:
            error_updates["rate_limit_pct"] = rate_limit_pct
        if capacity_529_pct is not None:
            error_updates["capacity_529_pct"] = capacity_529_pct
        if service_unavailable_pct is not None:
            error_updates["service_unavailable_pct"] = service_unavailable_pct
        if bad_gateway_pct is not None:
            error_updates["bad_gateway_pct"] = bad_gateway_pct
        if gateway_timeout_pct is not None:
            error_updates["gateway_timeout_pct"] = gateway_timeout_pct
        if internal_error_pct is not None:
            error_updates["internal_error_pct"] = internal_error_pct
        if timeout_pct is not None:
            error_updates["timeout_pct"] = timeout_pct
        if connection_reset_pct is not None:
            error_updates["connection_reset_pct"] = connection_reset_pct
        if slow_response_pct is not None:
            error_updates["slow_response_pct"] = slow_response_pct
        if invalid_json_pct is not None:
            error_updates["invalid_json_pct"] = invalid_json_pct
        if truncated_pct is not None:
            error_updates["truncated_pct"] = truncated_pct
        if empty_body_pct is not None:
            error_updates["empty_body_pct"] = empty_body_pct
        if missing_fields_pct is not None:
            error_updates["missing_fields_pct"] = missing_fields_pct
        if wrong_content_type_pct is not None:
            error_updates["wrong_content_type_pct"] = wrong_content_type_pct
        if selection_mode is not None:
            error_updates["selection_mode"] = selection_mode

        if error_updates:
            updates["error_injection"] = error_updates

        # Collect latency updates
        latency_updates: dict[str, int] = {}
        if base_ms is not None:
            latency_updates["base_ms"] = base_ms
        if jitter_ms is not None:
            latency_updates["jitter_ms"] = jitter_ms

        if latency_updates:
            updates["latency"] = latency_updates

        # Collect response updates
        response_updates: dict[str, str] = {}
        if mode is not None:
            response_updates["mode"] = mode

        if response_updates:
            updates["response"] = response_updates

        if updates:
            self.server.update_config(updates)

    def reset(self) -> str:
        """Reset server state and metrics.

        Clears all recorded metrics and starts a new run with a fresh run_id.

        Returns:
            The new run_id
        """
        return self.server.reset()

    def wait_for_requests(self, count: int, timeout: float = 10.0) -> bool:
        """Wait until the server has processed at least `count` requests.

        This is useful for tests that make async requests and need to verify
        the server has processed them before checking metrics.

        Args:
            count: Minimum number of requests to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if count was reached, False if timeout occurred
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            stats = self.get_stats()
            if stats["total_requests"] >= count:
                return True
            time.sleep(0.01)  # 10ms poll interval
        return False

    def post_completion(
        self,
        messages: list[dict[str, str]] | None = None,
        model: str = "gpt-4",
        **kwargs: Any,
    ) -> httpx.Response:
        """Convenience method to post a chat completion request.

        Args:
            messages: List of message dicts (default: single user message)
            model: Model name
            **kwargs: Additional request body fields

        Returns:
            httpx Response object (Starlette TestClient uses httpx)
        """
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        body = {"model": model, "messages": messages, **kwargs}
        return self.client.post("/v1/chat/completions", json=body)

    def post_azure_completion(
        self,
        deployment: str,
        messages: list[dict[str, str]] | None = None,
        api_version: str = "2024-02-01",
        **kwargs: Any,
    ) -> httpx.Response:
        """Convenience method to post an Azure chat completion request.

        Args:
            deployment: Azure deployment name
            messages: List of message dicts (default: single user message)
            api_version: Azure API version
            **kwargs: Additional request body fields

        Returns:
            httpx Response object (Starlette TestClient uses httpx)
        """
        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        body = {"messages": messages, **kwargs}
        url = f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        return self.client.post(url, json=body)


def _build_config_from_marker(
    marker: pytest.Mark | None,
    tmp_path: Path,
) -> ChaosLLMConfig:
    """Build ChaosLLMConfig from pytest marker and defaults.

    Args:
        marker: The @pytest.mark.chaosllm marker (may be None)
        tmp_path: Temp directory for metrics database

    Returns:
        Configured ChaosLLMConfig
    """
    metrics_db_path = tmp_path / "chaosllm-metrics.db"

    # Start with defaults optimized for testing (no latency)
    base_config: dict[str, Any] = {
        "metrics": {"database": str(metrics_db_path)},
        "latency": {"base_ms": 0, "jitter_ms": 0},
    }

    if marker is None:
        return ChaosLLMConfig(**base_config)

    # Check for preset in marker
    preset = marker.kwargs.get("preset")

    # Collect all other kwargs as potential overrides
    overrides: dict[str, Any] = {}

    # Error injection overrides
    error_overrides: dict[str, float] = {}
    for key in [
        "rate_limit_pct",
        "capacity_529_pct",
        "service_unavailable_pct",
        "bad_gateway_pct",
        "gateway_timeout_pct",
        "internal_error_pct",
        "timeout_pct",
        "connection_reset_pct",
        "slow_response_pct",
        "invalid_json_pct",
        "truncated_pct",
        "empty_body_pct",
        "missing_fields_pct",
        "wrong_content_type_pct",
        "selection_mode",
    ]:
        if key in marker.kwargs:
            error_overrides[key] = marker.kwargs[key]

    if error_overrides:
        overrides["error_injection"] = error_overrides

    # Latency overrides
    latency_overrides: dict[str, int] = {}
    if "base_ms" in marker.kwargs:
        latency_overrides["base_ms"] = marker.kwargs["base_ms"]
    if "jitter_ms" in marker.kwargs:
        latency_overrides["jitter_ms"] = marker.kwargs["jitter_ms"]

    if latency_overrides:
        overrides["latency"] = latency_overrides

    # Response overrides
    if "mode" in marker.kwargs:
        overrides["response"] = {"mode": marker.kwargs["mode"]}

    # Load config with precedence: overrides > preset > base
    if preset or overrides:
        config = load_config(
            preset=preset,
            cli_overrides={**base_config, **overrides} if overrides else base_config,
        )
        return config

    return ChaosLLMConfig(**base_config)


@pytest.fixture
def chaosllm_server(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[ChaosLLMFixture, None, None]:
    """Create a ChaosLLM fake LLM server for testing.

    This fixture creates an in-process ChaosLLM server using Starlette's
    TestClient. Configuration can be customized via the `@pytest.mark.chaosllm`
    marker.

    Marker usage:
        @pytest.mark.chaosllm(preset="gentle")
        @pytest.mark.chaosllm(rate_limit_pct=10.0)
        @pytest.mark.chaosllm(preset="stress_aimd", rate_limit_pct=50.0)

    Available marker parameters:
        preset: Name of preset configuration (e.g., "gentle", "stress_aimd")
        rate_limit_pct: 429 Rate Limit error percentage (0-100)
        capacity_529_pct: 529 Capacity error percentage (0-100)
        service_unavailable_pct: 503 Service Unavailable percentage
        internal_error_pct: 500 Internal Error percentage
        timeout_pct: Connection timeout percentage
        base_ms: Base latency in milliseconds
        jitter_ms: Latency jitter in milliseconds
        mode: Response generation mode

    Returns:
        ChaosLLMFixture with client, server access, and convenience methods

    Example:
        def test_basic(chaosllm_server):
            response = chaosllm_server.post_completion()
            assert response.status_code == 200

        @pytest.mark.chaosllm(rate_limit_pct=100.0)
        def test_rate_limits(chaosllm_server):
            response = chaosllm_server.post_completion()
            assert response.status_code == 429
    """
    # Get the chaosllm marker if present
    marker = request.node.get_closest_marker("chaosllm")

    # Build config from marker
    config = _build_config_from_marker(marker, tmp_path)

    # Create server and client
    server = ChaosLLMServer(config)
    client = TestClient(server.app)

    # Build fixture
    metrics_db_path = Path(config.metrics.database)
    fixture = ChaosLLMFixture(
        client=client,
        server=server,
        metrics_db_path=metrics_db_path,
    )

    yield fixture

    # Cleanup - close the test client
    client.close()


# Register the marker so pytest doesn't warn about unknown markers
def pytest_configure(config: pytest.Config) -> None:
    """Register the chaosllm marker."""
    config.addinivalue_line(
        "markers",
        "chaosllm(preset=None, **kwargs): Configure ChaosLLM server for the test. "
        "Use preset='name' to load a preset, and keyword args to override specific settings.",
    )
