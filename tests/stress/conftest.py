# tests/stress/conftest.py
"""Pytest fixtures for ChaosLLM HTTP stress testing.

This module provides fixtures for running ChaosLLM as a real HTTP server
(not TestClient) for stress testing LLM plugins that need actual TCP endpoints.

Usage:
    @pytest.mark.stress
    @pytest.mark.chaosllm(preset="stress_aimd")
    def test_llm_under_stress(chaosllm_http_server):
        # Server is running with real HTTP endpoint
        response = requests.post(
            f"{chaosllm_http_server.url}/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert response.status_code == 200
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.testing.chaosllm.config import (
    ChaosLLMConfig,
    load_config,
)
from elspeth.testing.chaosllm.server import ChaosLLMServer

if TYPE_CHECKING:
    import httpx


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0.

    The OS assigns an available port, which we capture before closing.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return port


def _wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Wait for server health endpoint to respond.

    Args:
        url: Base URL of the server
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is healthy, False if timeout occurred
    """
    import httpx

    health_url = f"{url}/health"
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            response = httpx.get(health_url, timeout=1.0)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(0.05)  # 50ms poll interval

    return False


@dataclass
class ChaosLLMHTTPFixture:
    """Pytest fixture object for ChaosLLM HTTP server.

    Unlike the TestClient-based ChaosLLMFixture, this runs a real HTTP server
    that can be accessed by actual HTTP clients (OpenAI SDK, httpx).

    Attributes:
        url: Base URL for the server (e.g., "http://127.0.0.1:8001")
        port: TCP port the server is listening on
        server: ChaosLLMServer instance with access to internal state
        _shutdown_event: Event to signal server shutdown
    """

    url: str
    port: int
    server: ChaosLLMServer
    metrics_db_path: Path
    _shutdown_event: threading.Event = field(default_factory=threading.Event)
    _server_thread: threading.Thread | None = field(default=None, init=False)
    _request_count: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

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

    def wait_for_requests(self, count: int, timeout: float = 30.0) -> bool:
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
            time.sleep(0.05)  # 50ms poll interval
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
            httpx Response object
        """
        import httpx

        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        body = {"model": model, "messages": messages, **kwargs}
        return httpx.post(f"{self.url}/v1/chat/completions", json=body, timeout=30.0)

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
            httpx Response object
        """
        import httpx

        if messages is None:
            messages = [{"role": "user", "content": "Hello"}]

        body = {"messages": messages, **kwargs}
        url = f"{self.url}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        return httpx.post(url, json=body, timeout=30.0)

    def shutdown(self) -> None:
        """Signal the server to shutdown."""
        self._shutdown_event.set()


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

    # Start with defaults - minimal latency for faster stress tests
    base_config: dict[str, Any] = {
        "metrics": {"database": str(metrics_db_path)},
        "latency": {"base_ms": 5, "jitter_ms": 2},  # Fast but non-zero for realism
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
    response_overrides: dict[str, Any] = {}
    if "mode" in marker.kwargs:
        response_overrides["mode"] = marker.kwargs["mode"]
    if "template_body" in marker.kwargs:
        response_overrides["mode"] = "template"  # Auto-set mode when body specified
        response_overrides["template"] = {"body": marker.kwargs["template_body"]}

    if response_overrides:
        overrides["response"] = response_overrides

    # Load config with precedence: overrides > preset > base
    if preset or overrides:
        config = load_config(
            preset=preset,
            cli_overrides={**base_config, **overrides} if overrides else base_config,
        )
        return config

    return ChaosLLMConfig(**base_config)


def _run_uvicorn_server(
    app: Any,
    host: str,
    port: int,
    shutdown_event: threading.Event,
) -> None:
    """Run uvicorn server in a thread with shutdown support.

    Args:
        app: ASGI application
        host: Host to bind to
        port: Port to bind to
        shutdown_event: Event to signal shutdown
    """
    import asyncio

    import uvicorn

    async def serve() -> None:
        """Async server runner with shutdown support."""
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",  # Reduce noise in tests
            access_log=False,
        )
        server = uvicorn.Server(config)

        # Create task to watch shutdown event
        async def watch_shutdown() -> None:
            while not shutdown_event.is_set():
                await asyncio.sleep(0.1)
            server.should_exit = True

        # Run server and shutdown watcher concurrently
        shutdown_task = asyncio.create_task(watch_shutdown())
        try:
            await server.serve()
        finally:
            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task

    # Run the async server in a new event loop
    asyncio.run(serve())


@pytest.fixture
def chaosllm_http_server(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Generator[ChaosLLMHTTPFixture, None, None]:
    """Create a ChaosLLM HTTP server for stress testing.

    This fixture creates a real HTTP server using uvicorn in a background thread.
    Unlike the TestClient-based fixture, this provides an actual TCP endpoint
    that can be used by real HTTP clients (OpenAI SDK, httpx).

    Configuration can be customized via the `@pytest.mark.chaosllm` marker.

    Marker usage:
        @pytest.mark.chaosllm(preset="stress_aimd")
        @pytest.mark.chaosllm(rate_limit_pct=30.0)
        @pytest.mark.chaosllm(preset="stress_extreme", rate_limit_pct=50.0)

    Returns:
        ChaosLLMHTTPFixture with url, server access, and convenience methods

    Example:
        @pytest.mark.stress
        @pytest.mark.chaosllm(preset="stress_aimd")
        def test_llm_under_stress(chaosllm_http_server):
            response = chaosllm_http_server.post_completion()
            assert response.status_code in (200, 429)
    """
    # Get the chaosllm marker if present
    marker = request.node.get_closest_marker("chaosllm")

    # Build config from marker
    config = _build_config_from_marker(marker, tmp_path)

    # Create server
    server = ChaosLLMServer(config)

    # Find free port and build URL
    port = _find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    # Create shutdown event
    shutdown_event = threading.Event()

    # Start uvicorn in background thread
    server_thread = threading.Thread(
        target=_run_uvicorn_server,
        args=(server.app, host, port, shutdown_event),
        daemon=True,
        name=f"chaosllm-stress-{port}",
    )
    server_thread.start()

    # Wait for server to be ready
    if not _wait_for_server(url, timeout=10.0):
        shutdown_event.set()
        raise RuntimeError(f"ChaosLLM HTTP server failed to start at {url}")

    # Build fixture
    metrics_db_path = Path(config.metrics.database)
    fixture = ChaosLLMHTTPFixture(
        url=url,
        port=port,
        server=server,
        metrics_db_path=metrics_db_path,
        _shutdown_event=shutdown_event,
    )
    fixture._server_thread = server_thread

    yield fixture

    # Cleanup - signal shutdown and wait for thread
    shutdown_event.set()
    server_thread.join(timeout=5.0)


# Register the markers so pytest doesn't warn about unknown markers
def pytest_configure(config: pytest.Config) -> None:
    """Register stress testing markers."""
    config.addinivalue_line(
        "markers",
        "stress: marks tests as stress tests requiring ChaosLLM HTTP server (deselect with '-m \"not stress\"')",
    )
    config.addinivalue_line(
        "markers",
        "chaosllm(preset=None, **kwargs): Configure ChaosLLM server for the test. "
        "Use preset='name' to load a preset, and keyword args to override specific settings.",
    )
