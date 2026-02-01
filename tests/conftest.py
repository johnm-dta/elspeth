# tests/conftest.py
"""Shared test fixtures and helpers.

This module provides reusable test utilities for creating test plugins that
properly implement the Protocol interfaces required by ELSPETH.

Test Base Classes:
- _TestSchema: Minimal PluginSchema for test fixtures
- _TestSourceBase: Base class for SourceProtocol test implementations
- _TestSinkBase: Base class for SinkProtocol test implementations
- _TestTransformBase: Base class for TransformProtocol test implementations

Hypothesis Configuration:
- "ci" profile: Fast tests for CI (100 examples) - default
- "nightly" profile: Thorough tests (1000 examples)
- "debug" profile: Minimal tests with verbose output (10 examples)

Set profile via environment variable:
    HYPOTHESIS_PROFILE=nightly pytest tests/property/

Test Fixture Philosophy: Validation Bypass Pattern
===================================================

ELSPETH has two distinct plugin instantiation paths:

1. PRODUCTION PATH (PluginManager):
   - Configuration loaded from YAML
   - PluginManager validates config against plugin's validation_schema()
   - ONLY THEN does PluginManager call plugin.__init__() with validated config
   - Plugins can assume __init__ receives valid data

2. TEST PATH (Direct Instantiation):
   - Tests call MyPlugin.__init__() directly
   - NO validation occurs - bypasses PluginManager entirely
   - Tests pass whatever arguments they need for the test scenario

Why Tests Bypass Validation (This is CORRECT):
-----------------------------------------------

Interface tests verify that plugins implement their Protocols correctly:
- Do they have all required attributes? (name, schema, version)
- Do their methods return the right types? (TransformResult, ArtifactDescriptor)
- Do lifecycle hooks work? (on_start, on_complete, close)

Interface tests do NOT verify:
- Configuration validation logic (that's tested separately in config tests)
- Production instantiation flow (that's tested in integration tests)

Benefits of Direct Instantiation in Tests:
-------------------------------------------
1. FASTER: No config parsing, no validation overhead
2. SIMPLER: Test code is just `MySource(data=[...])`, not YAML + manager
3. FOCUSED: Each test controls exact plugin state without config indirection

Example Comparison:
-------------------

# Production path (used by elspeth CLI):
config = ElspethSettings.from_yaml("pipeline.yaml")
manager = PluginManager()
source = manager.instantiate_source(config.source)  # Validates FIRST
# source.__init__ called ONLY if config is valid

# Test path (used by interface tests):
source = ListSource(data=[{"x": 1}])  # Direct instantiation, no validation
# Useful for testing that ListSource.load() works correctly

Both paths are correct for their context. The test base classes in this module
support the direct instantiation pattern by providing Protocol-compliant defaults.
"""

import contextlib
import hashlib
import hmac
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast
from uuid import uuid4

import pytest
from hypothesis import Phase, Verbosity, settings

from elspeth.contracts import Determinism, PluginSchema, SourceRow
from elspeth.contracts.payload_store import IntegrityError, PayloadStore
from elspeth.plugins.manager import PluginManager
from tests.fixtures.chaosllm import ChaosLLMFixture, chaosllm_server
from tests.fixtures.chaosllm import pytest_configure as _chaosllm_pytest_configure

if TYPE_CHECKING:
    from elspeth.contracts import TransformResult
    from elspeth.plugins.protocols import (
        GateProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
    )


# =============================================================================
# TelemetryManager Cleanup Fixture (Thread Leak Prevention)
# =============================================================================


@pytest.fixture(autouse=True)
def _auto_close_telemetry_managers():
    """Automatically close all TelemetryManager instances created during tests.

    TelemetryManager starts a non-daemon background thread for async export.
    If tests create TelemetryManager instances without calling close(), the
    thread keeps running and blocks pytest from exiting.

    This fixture tracks all TelemetryManager instances created during each test
    and closes them in teardown, preventing thread leaks.

    Special handling for tests that replace manager._queue:
        Some tests replace the internal queue to test backpressure. This can
        cause the export thread to be blocked on the OLD queue while close()
        sends the sentinel to the NEW queue. We handle this by also sending
        sentinels to any old queues we detect.

    Thread Safety:
        Uses module-level list tracking. Each test gets isolated via fixture
        setup/teardown. The list is cleared at fixture start, populated during
        the test, and drained at fixture end.
    """
    import queue as queue_module

    from elspeth.telemetry.manager import TelemetryManager

    # Track managers AND their original queues
    created_managers: list[tuple[TelemetryManager, queue_module.Queue]] = []
    original_init = TelemetryManager.__init__

    def tracking_init(self: TelemetryManager, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        # Store manager AND its original queue (in case tests replace _queue)
        created_managers.append((self, self._queue))

    # Monkey-patch for tracking
    TelemetryManager.__init__ = tracking_init  # type: ignore[method-assign]

    try:
        yield
    finally:
        # Restore original __init__
        TelemetryManager.__init__ = original_init  # type: ignore[method-assign]

        # Close all managers created during this test
        for manager, original_queue in created_managers:
            try:
                # First, unblock any SlowExporter-style waiters
                # by setting shutdown event
                manager._shutdown_event.set()

                # If queue was replaced, send sentinel to BOTH queues
                current_queue = manager._queue
                if current_queue is not original_queue:
                    # Test replaced the queue - thread may be blocked on old one
                    try:
                        original_queue.put_nowait(None)  # Sentinel to old queue
                    except queue_module.Full:
                        # Queue full - drain and retry
                        try:
                            original_queue.get_nowait()
                            original_queue.put_nowait(None)
                        except (queue_module.Full, queue_module.Empty):
                            pass

                # Only close if not already closed (check if thread is alive)
                if manager._export_thread.is_alive():
                    manager.close()

                # Final safety: if thread still alive, give it time then force-join
                if manager._export_thread.is_alive():
                    manager._export_thread.join(timeout=1.0)
            except Exception:
                # Best effort - don't fail test teardown
                pass


# =============================================================================
# Test Infrastructure
# =============================================================================


class MockPayloadStore:
    """In-memory PayloadStore for testing.

    Implements PayloadStore protocol using a dictionary.
    Each test gets a fresh instance for isolation.
    """

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    def store(self, content: bytes) -> str:
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash not in self._storage:
            self._storage[content_hash] = content
        return content_hash

    def retrieve(self, content_hash: str) -> bytes:
        if content_hash not in self._storage:
            raise KeyError(f"Payload not found: {content_hash}")
        content = self._storage[content_hash]
        # Integrity verification matching production FilesystemPayloadStore
        actual_hash = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(actual_hash, content_hash):
            raise IntegrityError(f"Payload integrity check failed: expected {content_hash}, got {actual_hash}")
        return content

    def exists(self, content_hash: str) -> bool:
        return content_hash in self._storage

    def delete(self, content_hash: str) -> bool:
        if content_hash not in self._storage:
            return False
        del self._storage[content_hash]
        return True


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def payload_store() -> PayloadStore:
    """PayloadStore fixture for tests that call orchestrator.run().

    Returns a fresh MockPayloadStore for each test. Required for audit
    compliance - all pipeline runs must have a payload store.
    """
    return MockPayloadStore()


# =============================================================================
# Azurite (Azure Blob Emulator) Fixtures
# =============================================================================


def _find_azurite_bin() -> str | None:
    """Locate the Azurite CLI binary.

    Prefers repo-local install (node_modules/.bin/azurite). Falls back to PATH.
    """
    repo_root = Path(__file__).resolve().parents[1]
    local_bin = repo_root / "node_modules" / ".bin" / "azurite"
    if local_bin.exists():
        return str(local_bin)
    return shutil.which("azurite")


def _get_free_port() -> int:
    """Get an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(host: str, port: int, timeout_seconds: float = 5.0) -> bool:
    """Wait until a TCP port is accepting connections."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _build_azurite_connection_string(host: str, port: int) -> str:
    """Build Azure Storage connection string for Azurite (blob only)."""
    account_name = "devstoreaccount1"
    # Standard Azurite account key (public default, not a secret)
    account_key = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
    return (
        "DefaultEndpointsProtocol=http;"
        f"AccountName={account_name};"
        f"AccountKey={account_key};"
        f"BlobEndpoint=http://{host}:{port}/{account_name};"
    )


@pytest.fixture(scope="session")
def azurite_blob_service(tmp_path_factory):
    """Start Azurite (blob-only) and provide connection details."""
    pytest.importorskip("azure.storage.blob")

    azurite_bin = _find_azurite_bin()
    if azurite_bin is None:
        pytest.skip("Azurite CLI not found. Run 'npm install' to install dev dependencies.")

    host = "127.0.0.1"
    port = _get_free_port()
    data_dir = tmp_path_factory.mktemp("azurite")

    cmd = [
        azurite_bin,
        "--silent",
        "--skipApiVersionCheck",
        "--disableProductStyleUrl",
        "--location",
        str(data_dir),
        "--blobHost",
        host,
        "--blobPort",
        str(port),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    if not _wait_for_port(host, port):
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        pytest.skip("Azurite failed to start (blob endpoint not reachable).")

    connection_string = _build_azurite_connection_string(host, port)
    previous = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = connection_string

    try:
        yield {
            "connection_string": connection_string,
            "host": host,
            "port": port,
            "process": process,
        }
    finally:
        if previous is None:
            os.environ.pop("AZURE_STORAGE_CONNECTION_STRING", None)
        else:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = previous

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture
def azurite_blob_container(azurite_blob_service):
    """Create a temporary container in Azurite for blob tests."""
    from azure.storage.blob import BlobServiceClient

    connection_string = azurite_blob_service["connection_string"]
    container_name = f"test-{uuid4().hex}"

    try:
        service_client = BlobServiceClient.from_connection_string(connection_string)
        service_client.create_container(container_name)
    except Exception as exc:
        pytest.skip(f"Azurite connection failed: {exc}")

    try:
        yield {
            "connection_string": connection_string,
            "container": container_name,
        }
    finally:
        with contextlib.suppress(Exception):
            # Best-effort cleanup for emulator containers
            service_client.delete_container(container_name)


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Standard plugin manager with builtin plugins registered.

    Use this fixture in tests that need to build ExecutionGraph from config.
    Ensures all tests use consistent plugin registration.

    Example:
        def test_graph_building(plugin_manager):
            config = ElspethSettings(...)
            plugins = instantiate_plugins_from_config(config)

            graph = ExecutionGraph.from_plugin_instances(

                source=plugins["source"],

                transforms=plugins["transforms"],

                sinks=plugins["sinks"],

                aggregations=plugins["aggregations"],

                gates=list(config.gates),

                default_sink=config.default_sink,

            )
    """
    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager


# =============================================================================
# Hypothesis Configuration
# =============================================================================

# CI profile: Fast tests for continuous integration
settings.register_profile(
    "ci",
    max_examples=100,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,  # Disable deadline for CI (timing varies)
)

# Nightly profile: Thorough testing for scheduled runs
settings.register_profile(
    "nightly",
    max_examples=1000,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,
)

# Debug profile: Minimal examples with verbose output for debugging
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,
)

# Load profile from environment, default to "ci"
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))


# =============================================================================
# Shared Test Base Classes
# =============================================================================
# These provide all required Protocol attributes and methods so test classes
# only need to override what's specific to the test.


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures.

    Use this when tests don't need specific fields - just pass it as
    input_schema or output_schema to satisfy Protocol requirements.
    """

    pass


class _TestSourceBase:
    """Base class for test sources that implements SourceProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - output_schema: type[PluginSchema]
    - load(ctx) -> Iterator[SourceRow]

    NOTE: Validation Bypass Pattern
    --------------------------------
    Test sources instantiated from this base class bypass PluginManager validation.
    This is CORRECT for interface tests:

    - Interface tests verify Protocol compliance (attributes, method signatures)
    - Config validation is tested separately in config-specific tests
    - Direct instantiation is faster and simpler than YAML + PluginManager

    Production path: PluginManager validates config BEFORE calling __init__
    Test path: Tests call __init__ directly with whatever data they need

    Usage:
        class MyTestSource(_TestSourceBase):
            name = "my_source"
            output_schema = MySchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield from self.wrap_rows(self._data)
    """

    # Required by SourceProtocol - child classes must override
    name: str
    output_schema: type[PluginSchema]

    # Protocol-required attributes with defaults
    # ClassVar for class-level default; __init__ creates instance attribute that shadows it
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    _on_validation_failure: str = "discard"  # Default: drop invalid rows in tests

    def __init__(self) -> None:
        """Initialize test source with empty config."""
        self.config = {"schema": {"fields": "dynamic"}}

    def wrap_rows(self, rows: list[dict[str, Any]]) -> Iterator[SourceRow]:
        """Wrap plain dicts in SourceRow.valid() as required by source protocol."""
        for row in rows:
            yield SourceRow.valid(row)

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        """Return field resolution mapping for audit trail.

        Test sources don't do field normalization, so return None.
        """
        return None


class CallbackSource(_TestSourceBase):
    """Source with callbacks for deterministic MockClock testing.

    Enables tests to advance a MockClock between row yields, allowing
    deterministic testing of timeout-dependent code paths without time.sleep().

    The callback is called AFTER each yield resumes (when the orchestrator
    asks for the next row). This timing is critical:

    Flow:
        1. yield row 0 (pause)
        2. Orchestrator: timeout check (empty), process row 0 (buffered)
        3. Orchestrator asks for next row → generator resumes
        4. after_yield_callback(0) called → can advance clock here
        5. Loop continues
        6. yield row 1 (pause)
        7. Orchestrator: timeout check (sees advanced clock, may trigger!)

    Example:
        clock = MockClock()

        def advance_after_row(row_idx: int) -> None:
            if row_idx == 0:
                clock.advance(0.25)  # Advance 250ms after first row

        source = CallbackSource(
            rows=[{"id": 1}, {"id": 2}, {"id": 3}],
            output_schema=MySchema,
            after_yield_callback=advance_after_row,
        )

        orchestrator = Orchestrator(..., clock=clock)
        result = orchestrator.run(source=source, ...)

        # Timeout fires before row 2 is processed because clock
        # was advanced to 0.25s (past 0.1s timeout threshold)
    """

    name: str = "callback_source"
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        rows: list[dict[str, Any]],
        output_schema: type[PluginSchema] | None = None,
        after_yield_callback: Callable[[int], None] | None = None,
        source_name: str = "callback_source",
    ) -> None:
        """Initialize callback source.

        Args:
            rows: List of row dicts to yield.
            output_schema: Schema for this source (defaults to _TestSchema).
            after_yield_callback: Called with row index after each yield resumes.
                Use this to advance MockClock between rows.
            source_name: Name for this source instance.
        """
        super().__init__()
        self._rows = rows
        self._after_yield_callback = after_yield_callback
        self.name = source_name
        if output_schema is not None:
            self.output_schema = output_schema

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        """Yield rows with callbacks between them for clock advancement."""
        for i, row in enumerate(self._rows):
            yield SourceRow.valid(row)
            # Called when generator resumes (after row i processed, before row i+1 yields)
            if self._after_yield_callback is not None:
                self._after_yield_callback(i)


class _TestSinkBase:
    """Base class for test sinks that implements SinkProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - write(rows, ctx) -> ArtifactDescriptor

    NOTE: Validation Bypass Pattern
    --------------------------------
    Test sinks instantiated from this base class bypass PluginManager validation.
    This is CORRECT for interface tests:

    - Interface tests verify Protocol compliance (attributes, method signatures)
    - Config validation is tested separately in config-specific tests
    - Direct instantiation is faster and simpler than YAML + PluginManager

    Production path: PluginManager validates config BEFORE calling __init__
    Test path: Tests call __init__ directly with whatever data they need

    Usage:
        class MyTestSink(_TestSinkBase):
            name = "my_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )
    """

    # Required by SinkProtocol - child classes must override
    name: str

    # Protocol-required attributes with defaults
    # ClassVar for class-level default; __init__ creates instance attribute that shadows it
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    input_schema: type[PluginSchema] = _TestSchema
    idempotent: bool = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        """Initialize test sink with empty config."""
        self.config = {"schema": {"fields": "dynamic"}}

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def flush(self) -> None:
        """Flush buffered data - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass


class _TestTransformBase:
    """Base class for test transforms that implements TransformProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - process(row, ctx) -> TransformResult

    NOTE: Validation Bypass Pattern
    --------------------------------
    Test transforms instantiated from this base class bypass PluginManager validation.
    This is CORRECT for interface tests:

    - Interface tests verify Protocol compliance (attributes, method signatures)
    - Config validation is tested separately in config-specific tests
    - Direct instantiation is faster and simpler than YAML + PluginManager

    Production path: PluginManager validates config BEFORE calling __init__
    Test path: Tests call __init__ directly with whatever data they need

    Usage:
        class MyTestTransform(_TestTransformBase):
            name = "my_transform"

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})
    """

    # Required by TransformProtocol - child classes must override
    name: str

    # Protocol-required attributes with defaults
    # ClassVar for class-level default; __init__ creates instance attribute that shadows it
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    is_batch_aware: bool = False
    creates_tokens: bool = False
    _on_error: str | None = None

    def __init__(self) -> None:
        """Initialize test transform with empty config."""
        self.config = {"schema": {"fields": "dynamic"}}

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass


# =============================================================================
# Type Cast Helpers for Tests
# =============================================================================
# These helpers provide type-safe ways to create test fixtures that satisfy
# the strict Protocol requirements. They use cast() internally to tell mypy
# that our test classes satisfy the protocols, even though they have minor
# differences (like more specific output_schema types).


def as_source(source: Any) -> "SourceProtocol":
    """Cast a test source to SourceProtocol.

    Use this when passing test source instances to functions expecting
    SourceProtocol. The cast tells mypy to trust that the test source
    satisfies the protocol.
    """
    return cast("SourceProtocol", source)


def as_transform(transform: Any) -> "TransformProtocol":
    """Cast a test transform to TransformProtocol."""
    return cast("TransformProtocol", transform)


def as_sink(sink: Any) -> "SinkProtocol":
    """Cast a test sink to SinkProtocol."""
    return cast("SinkProtocol", sink)


def as_gate(gate: Any) -> "GateProtocol":
    """Cast a test gate to GateProtocol."""
    return cast("GateProtocol", gate)


def as_transform_result(result: Any) -> "TransformResult":
    """Assert and cast a result to TransformResult.

    Used when extracting results from CollectorOutputPort which stores
    TransformResult | ExceptionResult. In normal test scenarios, we expect
    TransformResult - ExceptionResult only occurs for plugin bugs.

    Args:
        result: The result to cast (typically from collector.results[i][1])

    Returns:
        The result cast to TransformResult

    Raises:
        AssertionError: If result is ExceptionResult (unexpected plugin bug)
    """
    from elspeth.engine.batch_adapter import ExceptionResult

    if isinstance(result, ExceptionResult):
        raise AssertionError(f"Expected TransformResult but got ExceptionResult: {result.exception}\n{result.traceback}")
    return cast("TransformResult", result)


# =============================================================================
# Integration Test Fixtures (INFRA-02)
# =============================================================================
# These provide REAL database and recorder instances with FK constraints enabled.
# Use these for integration tests that validate audit trail integrity.


@pytest.fixture(scope="module")
def real_landscape_db(tmp_path_factory):
    """Real LandscapeDB with FK constraints enabled (module-scoped).

    Use this for integration tests that validate:
    - FK constraints are satisfied
    - Unique constraints work correctly
    - Audit trail completeness

    Returns an in-memory SQLite database with all tables created
    and FK constraints ENABLED (enforced).

    Module scope avoids repeated schema creation (15+ tables, indexes)
    which takes ~5-10ms per instantiation.

    Tests should use unique run_ids to isolate their data.

    Example:
        def test_batch_fk_constraints(real_landscape_db):
            recorder = LandscapeRecorder(real_landscape_db)
            # ... test validates no FK violations ...
    """
    from elspeth.core.landscape.database import LandscapeDB

    # Use in-memory database for fast tests
    # Tables are created automatically during initialization
    db = LandscapeDB.in_memory()
    return db


@pytest.fixture
def real_landscape_recorder(real_landscape_db):
    """Real LandscapeRecorder with FK constraint enforcement.

    Combines real_landscape_db with a recorder instance.
    Use for integration tests that record audit trail data.

    Example:
        def test_call_recording(real_landscape_recorder):
            run = real_landscape_recorder.begin_run(...)
            # ... test validates calls are recorded correctly ...
    """
    from elspeth.core.landscape.recorder import LandscapeRecorder

    return LandscapeRecorder(real_landscape_db)


@pytest.fixture
def real_landscape_recorder_with_payload_store(real_landscape_db, tmp_path):
    """Real LandscapeRecorder with payload store enabled.

    Use for integration tests that validate large payload storage
    (e.g., JSONL batch requests/responses).

    Example:
        def test_batch_payload_recording(real_landscape_recorder_with_payload_store):
            recorder = real_landscape_recorder_with_payload_store
            # ... test validates payloads are stored and retrievable ...
    """
    from elspeth.core.landscape.recorder import LandscapeRecorder
    from elspeth.core.payload_store import FilesystemPayloadStore

    payload_dir = tmp_path / "payloads"
    payload_store = FilesystemPayloadStore(payload_dir)
    return LandscapeRecorder(real_landscape_db, payload_store=payload_store)


# =============================================================================
# ChaosLLM Fixtures
# =============================================================================
# chaosllm_server fixture and ChaosLLMFixture are imported at top of file


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    # Register ChaosLLM marker
    _chaosllm_pytest_configure(config)


# Re-export for convenient import
__all__ = [
    "CallbackSource",
    "ChaosLLMFixture",
    "_TestSchema",
    "_TestSinkBase",
    "_TestSourceBase",
    "_TestTransformBase",
    "as_gate",
    "as_sink",
    "as_source",
    "as_transform",
    "as_transform_result",
    "chaosllm_server",
    "plugin_manager",
    "real_landscape_db",
    "real_landscape_recorder",
    "real_landscape_recorder_with_payload_store",
]
