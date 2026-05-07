# tests/performance/stress/conftest.py
"""Stress test configuration and shared infrastructure.

Provides:
- ChaosLLM HTTP server fixture (real TCP endpoint for stress testing)
- CollectingOutputPort (thread-safe output collector, consolidated from 5 v1 files)
- create_factory_and_run helper (parameterized factory+run creator)
- LLM config factories (Azure, OpenRouter, multi-query variants)
- Test row generators (single-query and multi-query)
- StressTestContext / StressTestResult dataclasses
- verify_audit_integrity helper

Migrated and consolidated from:
- tests/stress/conftest.py (ChaosLLM HTTP fixture)
- tests/stress/llm/conftest.py (LLM-specific fixtures)
- CollectingOutputPort extracted from 5 duplicate definitions across test files
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from errorworks.llm.config import ChaosLLMConfig, load_config
from errorworks.llm.server import ChaosLLMServer

from elspeth.contracts import NodeType, PipelineRow, TransformErrorReason, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

if TYPE_CHECKING:
    import httpx

    from elspeth.contracts.plugin_context import PluginContext
    from elspeth.engine.batch_adapter import ExceptionResult

# Dynamic schema for LLM transforms
DYNAMIC_SCHEMA = {"mode": "observed"}


# ---------------------------------------------------------------------------
# ChaosLLM HTTP Server Infrastructure
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find a free TCP port by binding to port 0.

    The OS assigns an available port, which we capture before closing.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port: int = s.getsockname()[1]
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
        """Get current server statistics."""
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
        error_updates: dict[str, float | str] = {}
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

        Returns:
            The new run_id
        """
        return self.server.reset()

    def wait_for_requests(self, count: int, timeout: float = 30.0) -> bool:
        """Wait until the server has processed at least `count` requests.

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
        """Convenience method to post a chat completion request."""
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
        """Convenience method to post an Azure chat completion request."""
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


@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers():
    """Cleanup TelemetryManager instances after each test."""
    from tests.fixtures.telemetry import telemetry_manager_cleanup

    with telemetry_manager_cleanup():
        yield


@pytest.fixture
def chaosllm_http_server(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> Generator[ChaosLLMHTTPFixture, None, None]:
    """Create a ChaosLLM HTTP server for stress testing.

    This fixture creates a real HTTP server using uvicorn in a background thread.
    Unlike the TestClient-based fixture, this provides an actual TCP endpoint
    that can be used by real HTTP clients (OpenAI SDK, httpx).

    Configuration can be customized via the ``@pytest.mark.chaosllm`` marker.

    Marker usage::

        @pytest.mark.chaosllm(preset="stress_aimd")
        @pytest.mark.chaosllm(rate_limit_pct=30.0)
        @pytest.mark.chaosllm(preset="stress_extreme", rate_limit_pct=50.0)

    Returns:
        ChaosLLMHTTPFixture with url, server access, and convenience methods
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


# ---------------------------------------------------------------------------
# CollectingOutputPort (consolidated from 5 duplicate v1 definitions)
# ---------------------------------------------------------------------------


class CollectingOutputPort:
    """Thread-safe output port that collects results for verification.

    Implements the OutputPort protocol to receive results from BatchTransformMixin.
    Results are stored as (row_data, token) tuples for success,
    (error_reason, token) for errors.

    Extracted from v1 test files where it was duplicated in each of:
    - test_azure_llm_stress.py
    - test_azure_multi_query_stress.py
    - test_openrouter_llm_stress.py
    - test_openrouter_multi_query_stress.py
    - test_mixed_errors.py
    """

    def __init__(self) -> None:
        self.results: list[tuple[dict[str, Any] | PipelineRow, TokenInfo]] = []
        self.errors: list[tuple[TransformErrorReason, TokenInfo]] = []
        self._lock = threading.Lock()

    def emit(
        self,
        token: TokenInfo,
        result: TransformResult | ExceptionResult,
        state_id: str | None,
    ) -> None:
        """Accept a result from upstream transform.

        Routes to results or errors based on TransformResult.status.
        ExceptionResults are treated as errors.
        """
        # Handle ExceptionResult (plugin bugs)
        if hasattr(result, "exception"):
            with self._lock:
                self.errors.append(({"reason": "test_error", "error": f"exception: {result.exception}"}, token))
            return

        # Handle TransformResult
        if result.status == "success":
            row_data: dict[str, Any] | PipelineRow = result.row if result.row is not None else {}
            with self._lock:
                self.results.append((row_data, token))
        else:
            error_data: TransformErrorReason = result.reason if result.reason is not None else {"reason": "test_error"}
            with self._lock:
                self.errors.append((error_data, token))

    @property
    def total_count(self) -> int:
        """Total processed (success + error)."""
        with self._lock:
            return len(self.results) + len(self.errors)

    @property
    def success_count(self) -> int:
        """Count of successful results."""
        with self._lock:
            return len(self.results)

    @property
    def error_count(self) -> int:
        """Count of errors."""
        with self._lock:
            return len(self.errors)


# ---------------------------------------------------------------------------
# Recorder + Run Factory (consolidated from 5 duplicate v1 definitions)
# ---------------------------------------------------------------------------


def create_factory_and_run(
    tmp_path_factory: pytest.TempPathFactory,
    plugin_name: str = "llm",
) -> tuple[RecorderFactory, str, str]:
    """Create a factory and start a run, returning (factory, run_id, node_id).

    Args:
        tmp_path_factory: Pytest temp path factory for creating audit DB
        plugin_name: Plugin name to register as node (default: "llm")

    Returns:
        Tuple of (factory, run_id, node_id)
    """
    tmp_path = tmp_path_factory.mktemp("stress_audit")
    db_path = tmp_path / "audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    factory = RecorderFactory(db)

    run = factory.run_lifecycle.begin_run(
        config={"test": f"{plugin_name}_stress"},
        run_id=f"stress-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    schema = SchemaConfig.from_dict({"mode": "observed"})
    node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name=plugin_name,
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return factory, run.run_id, node.node_id


# ---------------------------------------------------------------------------
# Token + Row Helpers
# ---------------------------------------------------------------------------


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    contract = SchemaContract(mode="FLEXIBLE", fields=(), locked=True)
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=PipelineRow({}, contract),  # Not used in these tests
    )


def make_pipeline_row(data: dict[str, Any]) -> PipelineRow:
    """Create a PipelineRow with OBSERVED contract for testing.

    Args:
        data: Row data dictionary

    Returns:
        PipelineRow wrapping the data with appropriate schema contract
    """
    fields = tuple(
        FieldContract(
            normalized_name=key,
            original_name=key,
            python_type=object,
            required=False,
            source="inferred",
        )
        for key in data
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return PipelineRow(data, contract)


def make_plugin_context(
    factory: RecorderFactory,
    run_id: str,
    state_id: str | None = None,
    token: TokenInfo | None = None,
) -> PluginContext:
    """Create a PluginContext with real landscape.

    Delegates to the shared make_context() factory.

    Args:
        factory: RecorderFactory for audit trail
        run_id: Run ID for context
        state_id: Optional state ID (generated if not provided)
        token: Optional token info (generated if not provided)

    Returns:
        PluginContext ready for use
    """
    from tests.fixtures.factories import make_context

    if state_id is None:
        state_id = f"state-{uuid.uuid4().hex[:12]}"
    if token is None:
        token = make_token()

    return make_context(
        run_id=run_id,
        landscape=factory.plugin_audit_writer(),
        state_id=state_id,
        token=token,
    )


# ---------------------------------------------------------------------------
# LLM Config Factories
# ---------------------------------------------------------------------------


def make_azure_llm_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid Azure LLM config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for LLMTransform with provider="azure"
    """
    config = {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": chaosllm_url,
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "system_prompt": "You are a helpful assistant.",
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
        "temperature": 0.7,
        "max_tokens": 500,
        "required_input_fields": [],  # Explicit opt-out for tests
    }
    config.update(overrides)
    return config


def make_openrouter_llm_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid OpenRouter LLM config pointed at ChaosLLM.

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for LLMTransform with provider="openrouter"

    Note:
        OpenRouter uses /chat/completions, but ChaosLLM serves at /v1/chat/completions.
        We append /v1 to the base_url so the paths match.
    """
    config = {
        "provider": "openrouter",
        "model": "anthropic/claude-3-opus",
        "base_url": f"{chaosllm_url}/v1",  # Append /v1 for ChaosLLM compatibility
        "api_key": "test-key",
        "template": "Analyze: {{ row.text }}",
        "system_prompt": "You are a helpful assistant.",
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
        "temperature": 0.7,
        "max_tokens": 500,
        "required_input_fields": [],  # Explicit opt-out for tests
    }
    config.update(overrides)
    return config


def make_azure_multi_query_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid Azure multi-query config pointed at ChaosLLM.

    Uses the unified LLMTransform queries dict format with provider="azure".

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for LLMTransform with provider="azure" and queries
    """
    config: dict[str, Any] = {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": chaosllm_url,
        "api_key": "test-key",
        "template": "Input: {{ row.text_content }}\nCriterion: {{ row.criterion_name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_diag_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_treat_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_diag_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_treat_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for tests
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
    }
    config.update(overrides)
    return config


def make_openrouter_multi_query_config(
    chaosllm_url: str,
    **overrides: Any,
) -> dict[str, Any]:
    """Create valid OpenRouter multi-query config pointed at ChaosLLM.

    Uses the unified LLMTransform queries dict format with provider="openrouter".

    Args:
        chaosllm_url: Base URL of ChaosLLM server
        **overrides: Override any config values

    Returns:
        Config dict ready for LLMTransform with provider="openrouter" and queries

    Note:
        OpenRouter uses /chat/completions, but ChaosLLM serves at /v1/chat/completions.
        We append /v1 to the base_url so the paths match.
    """
    config: dict[str, Any] = {
        "provider": "openrouter",
        "model": "anthropic/claude-3-opus",
        "base_url": f"{chaosllm_url}/v1",  # Append /v1 for ChaosLLM compatibility
        "api_key": "test-key",
        "template": "Input: {{ row.text_content }}\nCriterion: {{ row.criterion_name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "queries": {
            "cs1_diagnosis": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_diag_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs1_treatment": {
                "input_fields": {"text_content": "cs1_bg", "criterion_name": "cs1_treat_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_diagnosis": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_diag_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
            "cs2_treatment": {
                "input_fields": {"text_content": "cs2_bg", "criterion_name": "cs2_treat_name"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],  # Explicit opt-out for tests
        "pool_size": 4,
        "max_capacity_retry_seconds": 30,
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Test Row Generators
# ---------------------------------------------------------------------------


def generate_test_rows(count: int, prefix: str = "row") -> list[dict[str, Any]]:
    """Generate test rows for stress testing.

    Args:
        count: Number of rows to generate
        prefix: Prefix for row IDs

    Returns:
        List of row dicts with text and id fields
    """
    return [{"id": f"{prefix}-{i}", "text": f"Test input for row {i}"} for i in range(count)]


def generate_multi_query_rows(count: int, prefix: str = "row") -> list[dict[str, Any]]:
    """Generate test rows for multi-query stress testing.

    Creates rows with all named input fields that multi-query configs reference.

    Args:
        count: Number of rows to generate
        prefix: Prefix for row IDs

    Returns:
        List of row dicts with named input fields for each query
    """
    return [
        {
            "id": f"{prefix}-{i}",
            # Case study 1 fields
            "cs1_bg": f"Background info for case study 1, row {i}",
            "cs1_sym": f"Symptoms for case study 1, row {i}",
            "cs1_hist": f"History for case study 1, row {i}",
            # Case study 2 fields
            "cs2_bg": f"Background info for case study 2, row {i}",
            "cs2_sym": f"Symptoms for case study 2, row {i}",
            "cs2_hist": f"History for case study 2, row {i}",
            # Criterion name fields for each query
            "cs1_diag_name": "diagnosis",
            "cs1_treat_name": "treatment",
            "cs2_diag_name": "diagnosis",
            "cs2_treat_name": "treatment",
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Stress Test Context + Result
# ---------------------------------------------------------------------------


@dataclass
class StressTestContext:
    """Context container for stress test execution.

    Holds all the resources needed for running LLM transforms
    against ChaosLLM with proper audit recording.
    """

    factory: RecorderFactory
    run_id: str
    chaosllm_url: str


@dataclass
class StressTestResult:
    """Results from a stress test run.

    Attributes:
        total_rows: Total number of rows processed
        successful_rows: Rows that completed successfully
        failed_rows: Rows that failed
        error_rate_observed: Actual error rate from ChaosLLM
        total_requests: Total requests made to ChaosLLM
        fifo_preserved: Whether output order matched input order
    """

    total_rows: int
    successful_rows: int
    failed_rows: int
    error_rate_observed: float
    total_requests: int
    fifo_preserved: bool

    @property
    def success_rate(self) -> float:
        """Calculate row success rate."""
        if self.total_rows == 0:
            return 0.0
        return self.successful_rows / self.total_rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stress_landscape_factory(tmp_path: Path) -> Generator[RecorderFactory, None, None]:
    """Create a fresh RecorderFactory for stress testing.

    Uses SQLite file database for persistence during test.

    Yields:
        RecorderFactory ready for use
    """
    db_path = tmp_path / "stress-audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    factory = RecorderFactory(db)

    yield factory


@pytest.fixture
def stress_test_context(
    chaosllm_http_server: ChaosLLMHTTPFixture,
    stress_landscape_factory: RecorderFactory,
) -> Generator[StressTestContext, None, None]:
    """Create a complete stress test context.

    Combines ChaosLLM server with landscape factory and begins a run.

    Yields:
        StressTestContext with all resources ready
    """
    run = stress_landscape_factory.run_lifecycle.begin_run(
        config={"test": "stress"},
        run_id=f"stress-run-{uuid.uuid4().hex[:8]}",
        canonical_version="v1",
    )

    yield StressTestContext(
        factory=stress_landscape_factory,
        run_id=run.run_id,
        chaosllm_url=chaosllm_http_server.url,
    )

    from elspeth.contracts import RunStatus

    stress_landscape_factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Verification Helpers
# ---------------------------------------------------------------------------


def verify_audit_integrity(
    factory: RecorderFactory,
    run_id: str,
    expected_rows: int,
) -> bool:
    """Verify that audit trail is complete for all rows.

    Checks that every row has:
    - A token record
    - Node state(s)
    - Either completed or failed outcome

    Args:
        factory: RecorderFactory with test data
        run_id: Run ID to verify
        expected_rows: Expected number of source rows

    Returns:
        True if audit trail is complete, False otherwise
    """
    with factory._db.connection() as conn:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import tokens_table

        result = conn.execute(select(tokens_table).where(tokens_table.c.run_id == run_id))
        tokens = list(result.fetchall())

    # Every row should have at least one token
    unique_row_ids = {t.row_id for t in tokens}

    if len(unique_row_ids) < expected_rows:
        return False

    # Get outcomes
    with factory._db.connection() as conn:
        from elspeth.core.landscape.schema import token_outcomes_table

        result = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_id))
        outcomes = list(result.fetchall())

    # Every token should have an outcome
    token_ids_with_outcome = {o.token_id for o in outcomes}
    all_token_ids = {t.token_id for t in tokens}

    return all_token_ids.issubset(token_ids_with_outcome)
