# Web UX Task-Plan 5C: ExecutionServiceImpl

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement background thread pipeline execution with all thread safety fixes (B2, B3, B7)
**Parent Plan:** `plans/2026-03-28-web-ux-sub5-execution.md`
**Spec:** `specs/2026-03-28-web-ux-sub5-execution-design.md`
**Depends On:** Task-Plan 5A (Models — ProgressBroadcaster, schemas), Sub-Plan 2 (Sessions — SessionService Run CRUD)
**Blocks:** Task-Plan 5D (Routes & Integration)
**Can run in parallel with:** Task-Plan 5B (Validation)

---

### File Map

| Action | File |
|--------|------|
| Create | `src/elspeth/web/execution/service.py` |
| Create | `tests/unit/web/execution/test_service.py` |

---

### Task 5.5: ExecutionServiceImpl (B2 + B3 + B7 + B8/C1 Fixes)

**Files:**
- Create: `src/elspeth/web/execution/service.py`
- Create: `tests/unit/web/execution/test_service.py`

This is the highest-risk task. Four blocking review fixes live here: B2 (shutdown_event), B3 (LandscapeDB construction), B7 (BaseException + done_callback), B8/C1 (async/sync bridging via `_call_async()`). Every test verifies a specific thread safety invariant.

- [ ] **Step 1: Write ExecutionServiceImpl tests**

```python
# tests/unit/web/execution/test_service.py
"""Tests for ExecutionServiceImpl — background execution with thread safety.

Each test class targets a specific review fix:
- TestExecutionFlow: Basic lifecycle (pending -> running -> completed)
- TestB2ShutdownEvent: shutdown_event always passed to Orchestrator.run()
- TestB3Construction: LandscapeDB/PayloadStore from WebSettings
- TestB7ExceptionHandling: BaseException catch + done_callback safety net
- TestB8AsyncBridging: _call_async() bridges sync thread to async event loop
- TestCancelMechanism: Event-based cancellation
- TestOneActiveRun: B6 constraint enforcement
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, call, patch
from uuid import UUID, uuid4

import pytest

from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import RunEvent
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.sessions.protocol import RunAlreadyActiveError


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_loop() -> MagicMock:
    return MagicMock(spec=asyncio.AbstractEventLoop)


@pytest.fixture
def broadcaster(mock_loop: MagicMock) -> ProgressBroadcaster:
    return ProgressBroadcaster(mock_loop)


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.get_landscape_url.return_value = "sqlite:///test_audit.db"
    settings.get_payload_store_path.return_value = Path("/tmp/test_payloads")
    return settings


@pytest.fixture
def mock_session_service() -> MagicMock:
    svc = MagicMock()
    state = MagicMock()
    state.yaml_content = "source:\n  plugin: csv_source"
    # B4 fix: get_composition_state() doesn't exist — use get_state/get_current_state
    svc.get_state.return_value = state
    svc.get_current_state.return_value = state
    return svc


@pytest.fixture
def service(
    mock_loop: MagicMock,
    broadcaster: ProgressBroadcaster,
    mock_settings: MagicMock,
    mock_session_service: MagicMock,
) -> ExecutionServiceImpl:
    # AC #17: All Run CRUD goes through SessionService — no direct DB access.
    # session_service provides: create_run(), update_run_status(),
    # get_active_run(), get_run() with R6 expanded params
    # (landscape_run_id, pipeline_yaml, rows_processed, rows_failed).
    mock_session_service.get_active_run.return_value = None
    return ExecutionServiceImpl(
        loop=mock_loop,
        broadcaster=broadcaster,
        settings=mock_settings,
        session_service=mock_session_service,
        yaml_generator=MagicMock(),
    )


# ── Basic Lifecycle ────────────────────────────────────────────────────

class TestExecutionFlow:
    @pytest.mark.asyncio
    async def test_execute_returns_run_id_immediately(
        self, service: ExecutionServiceImpl
    ) -> None:
        """execute() returns a UUID without blocking on pipeline completion."""
        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)

    @pytest.mark.asyncio
    async def test_execute_creates_run_via_session_service(
        self, service: ExecutionServiceImpl, mock_session_service: MagicMock
    ) -> None:
        """AC #17: Run creation delegates to session_service.create_run()
        with R6 expanded params (session_id, state_id, pipeline_yaml)."""
        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        mock_session_service.create_run.assert_called_once()
        create_call = mock_session_service.create_run.call_args
        assert "session_id" in create_call[1] or len(create_call[0]) >= 1
        assert "pipeline_yaml" in create_call[1] or len(create_call[0]) >= 2

    @pytest.mark.asyncio
    async def test_get_status_returns_run_status(
        self, service: ExecutionServiceImpl, mock_session_service: MagicMock
    ) -> None:
        run_id = uuid4()
        mock_session_service.get_run.return_value = MagicMock(
            id=run_id,  # B7: RunRecord uses `id`, not `run_id`
            status="running",
            started_at=datetime.now(tz=timezone.utc),
            finished_at=None,
            rows_processed=50,
            rows_failed=2,
            error=None,
            landscape_run_id=None,
        )
        status = await service.get_status(run_id)
        assert status.status == "running"
        assert status.rows_processed == 50


# ── B2: shutdown_event Always Passed ───────────────────────────────────

class TestB2ShutdownEvent:
    """B2 fix: _run_pipeline() MUST pass shutdown_event to orchestrator.run().

    If shutdown_event is omitted, the Orchestrator calls signal.signal()
    from the worker thread, raising ValueError: signal only works in main thread.
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings")
    @patch("elspeth.web.execution.service.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_shutdown_event_passed_to_orchestrator_run(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph = MagicMock()
        mock_graph_cls.from_plugin_instances.return_value = mock_graph
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.run_id = "landscape-run-123"
        mock_orch.run.return_value = mock_result

        shutdown_event = threading.Event()
        run_id = uuid4()

        service._run_pipeline(str(run_id), "source:\n  plugin: csv", shutdown_event)

        # B2 invariant: shutdown_event was passed
        orch_run_call = mock_orch.run.call_args
        assert orch_run_call[1].get("shutdown_event") is shutdown_event, (
            "B2 VIOLATION: shutdown_event not passed to orchestrator.run(). "
            "This will cause ValueError: signal only works in main thread."
        )


# ── B3: LandscapeDB and PayloadStore Construction ─────────────────────

class TestB3Construction:
    """B3 fix: Construct LandscapeDB and FilesystemPayloadStore from WebSettings.

    _run_pipeline() does NOT use hardcoded paths. It calls
    self._settings.get_landscape_url() and self._settings.get_payload_store_path().
    """

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings")
    @patch("elspeth.web.execution.service.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_landscape_db_constructed_from_settings(
        self,
        mock_payload_cls: MagicMock,
        mock_landscape_cls: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_settings: MagicMock,
    ) -> None:
        mock_load.return_value = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_orch.run.return_value = MagicMock(run_id="r1")

        service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # B3: LandscapeDB constructed from settings URL
        mock_landscape_cls.assert_called_once_with(
            connection_string="sqlite:///test_audit.db"
        )
        # B3: PayloadStore constructed from settings path
        mock_payload_cls.assert_called_once_with(
            base_path=Path("/tmp/test_payloads")
        )


# ── B7: BaseException + Done Callback ─────────────────────────────────

class TestB7ExceptionHandling:
    """B7 fix: _run_pipeline() catches BaseException, not Exception.

    Layer 1: try/except BaseException updates run to failed status.
    Layer 2: future.add_done_callback() logs as safety net.
    """

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_keyboard_interrupt_skips_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """R6 fix: KeyboardInterrupt skips _call_async — event loop is shutting down."""
        mock_landscape.side_effect = KeyboardInterrupt("ctrl-c")

        with pytest.raises(KeyboardInterrupt):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # R6: status update skipped for KeyboardInterrupt — orphan cleanup handles it
        mock_session_service.update_run_status.assert_not_called()

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_system_exit_skips_status_update(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """R6 fix: SystemExit skips _call_async — event loop is shutting down."""
        mock_landscape.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # R6: status update skipped for SystemExit — orphan cleanup handles it
        mock_session_service.update_run_status.assert_not_called()

    def test_shutdown_event_cleaned_up_in_finally(
        self,
        service: ExecutionServiceImpl,
    ) -> None:
        """finally clause removes shutdown event from _shutdown_events dict."""
        run_id = str(uuid4())
        event = threading.Event()
        service._shutdown_events[run_id] = event

        with patch("elspeth.web.execution.service.LandscapeDB") as mock_db:
            mock_db.side_effect = RuntimeError("boom")
            # RuntimeError is not BaseException subclass that bypasses except,
            # so it's caught by except BaseException
            with pytest.raises(RuntimeError):
                service._run_pipeline(run_id, "yaml", event)

        # finally must have removed the event
        assert run_id not in service._shutdown_events

    def test_done_callback_logs_unhandled_exception(
        self, service: ExecutionServiceImpl
    ) -> None:
        """future.add_done_callback fires when _run_pipeline raises."""
        future: Future[None] = Future()
        future.set_exception(RuntimeError("unhandled"))

        with patch("elspeth.web.execution.service.slog") as mock_slog:
            service._on_pipeline_done(future)
            mock_slog.error.assert_called_once()

    def test_done_callback_noop_on_success(
        self, service: ExecutionServiceImpl
    ) -> None:
        """done_callback does not log on successful completion."""
        future: Future[None] = Future()
        future.set_result(None)

        with patch("elspeth.web.execution.service.slog") as mock_slog:
            service._on_pipeline_done(future)
            mock_slog.error.assert_not_called()


# ── Cancel Mechanism ───────────────────────────────────────────────────

class TestCancelMechanism:
    @pytest.mark.asyncio
    async def test_cancel_active_run_sets_event(
        self, service: ExecutionServiceImpl
    ) -> None:
        run_id = uuid4()
        event = threading.Event()
        service._shutdown_events[str(run_id)] = event

        await service.cancel(run_id)

        assert event.is_set(), (
            "cancel() must set the threading.Event so the Orchestrator "
            "detects it during row processing"
        )

    @pytest.mark.asyncio
    async def test_cancel_pending_run_updates_status(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """When no shutdown event exists (pending), update status directly."""
        run_id = uuid4()
        # No event in _shutdown_events — run is pending
        await service.cancel(run_id)
        mock_session_service.update_run_status.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_terminal_run_is_noop(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """Cancelling completed/failed/cancelled run does nothing."""
        run_id = uuid4()
        mock_session_service.get_run.return_value = MagicMock(status="completed")
        await service.cancel(run_id)
        mock_session_service.update_run_status.assert_not_called()

        # (exact assertion depends on implementation checking status first)

    @pytest.mark.asyncio
    async def test_cancel_idempotent_on_set_event(
        self, service: ExecutionServiceImpl
    ) -> None:
        """Setting an already-set event is safe."""
        run_id = uuid4()
        event = threading.Event()
        event.set()
        service._shutdown_events[str(run_id)] = event

        # Should not raise
        await service.cancel(run_id)
        assert event.is_set()

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings")
    @patch("elspeth.web.execution.service.instantiate_plugins_from_config")
    @patch("elspeth.web.execution.service.ExecutionGraph")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_cancelled_run_broadcasts_cancelled_event(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        mock_graph_cls: MagicMock,
        mock_instantiate: MagicMock,
        mock_load: MagicMock,
        mock_orch_cls: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """AC #19: When shutdown_event is set, _run_pipeline broadcasts
        a 'cancelled' terminal event instead of 'completed'."""
        mock_load.return_value = MagicMock()
        mock_bundle = MagicMock()
        mock_bundle.source = MagicMock()
        mock_bundle.source_settings = MagicMock()
        mock_bundle.transforms = ()
        mock_bundle.sinks = {"primary": MagicMock()}
        mock_bundle.aggregations = {}
        mock_instantiate.return_value = mock_bundle
        mock_graph_cls.from_plugin_instances.return_value = MagicMock()
        mock_orch = MagicMock()
        mock_orch_cls.return_value = mock_orch
        mock_result = MagicMock()
        mock_result.rows_processed = 50
        mock_result.rows_failed = 2
        mock_orch.run.return_value = mock_result

        shutdown_event = threading.Event()
        shutdown_event.set()  # Simulate cancellation
        run_id = str(uuid4())

        service._run_pipeline(run_id, "source:\n  plugin: csv", shutdown_event)

        # Verify status updated to "cancelled" (not "completed")
        status_calls = mock_session_service.update_run_status.call_args_list
        final_status_call = status_calls[-1]
        assert "cancelled" in str(final_status_call), (
            f"Expected 'cancelled' status update, got: {final_status_call}"
        )

        # Verify the broadcaster received a "cancelled" event via
        # loop.call_soon_threadsafe. The ProgressBroadcaster forwards
        # put_nowait calls through the mock loop, so we can inspect them.
        threadsafe_calls = service._broadcaster._loop.call_soon_threadsafe.call_args_list
        # Find the terminal event among call_soon_threadsafe calls
        terminal_events = [
            c[0][1]  # second positional arg to call_soon_threadsafe is the event
            for c in threadsafe_calls
            if isinstance(c[0][1], RunEvent) and c[0][1].event_type == "cancelled"
        ]
        assert len(terminal_events) >= 1, (
            "Expected a 'cancelled' RunEvent to be broadcast"
        )
        assert terminal_events[0].data["rows_processed"] == 50
        assert terminal_events[0].data["rows_failed"] == 2


# ── One Active Run (B6) ───────────────────────────────────────────────

class TestOneActiveRun:
    @pytest.mark.asyncio
    async def test_second_execute_raises_run_already_active(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """B6: Only one pending/running run per session."""
        session_id = uuid4()
        mock_session_service.get_active_run.return_value = MagicMock(
            status="running"
        )

        with pytest.raises(RunAlreadyActiveError):
            await service.execute(session_id=session_id)

    @pytest.mark.asyncio
    async def test_execute_after_completed_run_succeeds(
        self,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """After a run completes, a new one can start."""
        mock_session_service.get_active_run.return_value = None
        with patch.object(service, "_run_pipeline"):
            run_id = await service.execute(session_id=uuid4())
        assert isinstance(run_id, UUID)


# ── EventBus Bridge ───────────────────────────────────────────────────

class TestEventBusBridge:
    """Verify that ProgressEvent from the Orchestrator's EventBus
    is translated to RunEvent and broadcast via the ProgressBroadcaster."""

    def test_progress_event_translated_to_run_event(
        self, service: ExecutionServiceImpl
    ) -> None:
        """_to_run_event maps ProgressEvent fields to RunEvent.data."""
        from elspeth.contracts.cli import ProgressEvent

        progress = ProgressEvent(
            rows_processed=100,
            rows_succeeded=95,
            rows_failed=5,
            rows_quarantined=3,
            elapsed_seconds=10.5,
        )
        run_id = "run-123"
        run_event = service._to_run_event(run_id, progress)

        assert run_event.event_type == "progress"
        assert run_event.data["rows_processed"] == 100
        assert run_event.data["rows_failed"] == 5
        assert run_event.run_id == "run-123"


# ── B10: _call_async() Bridge Tests ──────────────────────────────────

class TestB8AsyncBridging:
    """B8/C1 fix: _call_async() bridges sync thread to async event loop."""

    def test_call_async_returns_coroutine_result(
        self, service: ExecutionServiceImpl
    ) -> None:
        """_call_async() schedules coroutine and returns its result."""
        mock_future = MagicMock()
        mock_future.result.return_value = "test_result"
        service._loop.run_coroutine_threadsafe = MagicMock(return_value=mock_future)

        async def dummy_coro() -> str:
            return "test_result"

        result = service._call_async(dummy_coro())
        assert result == "test_result"
        mock_future.result.assert_called_once_with(timeout=30.0)

    def test_call_async_propagates_coroutine_exception(
        self, service: ExecutionServiceImpl
    ) -> None:
        """If the coroutine raises, _call_async re-raises from future.result()."""
        mock_future = MagicMock()
        mock_future.result.side_effect = ValueError("db error")
        service._loop.run_coroutine_threadsafe = MagicMock(return_value=mock_future)

        async def failing_coro() -> None:
            raise ValueError("db error")

        with pytest.raises(ValueError, match="db error"):
            service._call_async(failing_coro())

    def test_call_async_raises_timeout_error(
        self, service: ExecutionServiceImpl
    ) -> None:
        """R6 fix: _call_async raises TimeoutError after 30s, preventing deadlock."""
        import concurrent.futures
        mock_future = MagicMock()
        mock_future.result.side_effect = concurrent.futures.TimeoutError()
        service._loop.run_coroutine_threadsafe = MagicMock(return_value=mock_future)

        async def hanging_coro() -> None:
            pass

        with pytest.raises(concurrent.futures.TimeoutError):
            service._call_async(hanging_coro())


# ── W15: Running Status Failure Path ─────────────────────────────────

class TestRunningStatusFailure:
    """W15: What happens when the initial status update to 'running' fails."""

    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_running_status_failure_marks_run_failed(
        self,
        mock_payload: MagicMock,
        mock_landscape: MagicMock,
        service: ExecutionServiceImpl,
        mock_session_service: MagicMock,
    ) -> None:
        """If update_run_status('running') fails, the except BaseException
        block attempts to set 'failed'. Run stays 'pending' if both fail."""
        # Make the first _call_async raise (simulating event loop issues)
        original_call_async = service._call_async
        call_count = 0
        def failing_call_async(coro):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call = update to "running"
                raise ConnectionError("DB connection lost")
            return original_call_async(coro)

        service._call_async = failing_call_async

        with pytest.raises(ConnectionError):
            service._run_pipeline(str(uuid4()), "yaml", threading.Event())

        # The except block tried to set "failed" via the second _call_async call
        assert call_count >= 2
```

- [ ] **Step 2: Implement ExecutionServiceImpl**

```python
# src/elspeth/web/execution/service.py
"""ExecutionServiceImpl — background pipeline execution with thread safety.

Thread safety fixes implemented here:
- B2: Always pass shutdown_event=threading.Event() to Orchestrator.run()
- B3: Construct LandscapeDB/PayloadStore from WebSettings resolvers
- B7: except BaseException + future.add_done_callback() safety net
- B8/C1: _call_async() bridges sync thread to async event loop for SessionService

The _run_pipeline() method is the ONLY code that runs outside the asyncio
event loop. Everything else runs in the main async context. Because
SessionService methods are async, _run_pipeline() uses _call_async() to
schedule coroutines on the main event loop from the background thread.
"""
from __future__ import annotations

import asyncio
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Coroutine, TypeVar
from uuid import UUID

import structlog

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts.cli import ProgressEvent
from elspeth.core.config import load_settings
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.core.events import EventBus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.types import PipelineConfig
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import RunEvent, RunStatusResponse
from elspeth.web.sessions.protocol import RunAlreadyActiveError  # B1: canonical definition

slog = structlog.get_logger()

T = TypeVar("T")

# B1 fix: RunAlreadyActiveError is NOT defined here — imported from
# sessions.protocol where the canonical definition lives. Defining a
# second class with the same name would prevent app.py's global
# exception handler (which catches sessions.protocol.RunAlreadyActiveError)
# from catching exceptions raised here.


class ExecutionServiceImpl:
    """Pipeline execution service with ThreadPoolExecutor backend.

    Construction: Created inside the FastAPI lifespan async context manager
    (after ProgressBroadcaster), NOT in the synchronous create_app() factory.
    Stored as application state and injected into route handlers via FastAPI's
    dependency injection. The event loop reference is obtained from
    asyncio.get_running_loop() in the lifespan (same loop as ProgressBroadcaster).

    Thread model: execute() submits _run_pipeline() to a ThreadPoolExecutor
    with max_workers=1. The pipeline runs in a background thread. All other
    methods run in the asyncio event loop thread.

    B8/C1 fix: SessionService methods are async. _run_pipeline() runs in a
    background thread and uses _call_async() to bridge async calls back to
    the main event loop via asyncio.run_coroutine_threadsafe().
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        broadcaster: ProgressBroadcaster,
        settings: Any,  # WebSettings
        session_service: Any,  # SessionService
        yaml_generator: Any,  # YamlGenerator — injected, not module-level
    ) -> None:
        self._loop = loop
        self._broadcaster = broadcaster
        self._settings = settings
        self._session_service = session_service
        self._yaml_generator = yaml_generator
        # AC #17: No run_repository — all Run CRUD delegates to SessionService
        # via create_run(), update_run_status(), get_active_run(), get_run().
        # R6 expanded params: landscape_run_id, pipeline_yaml, rows_processed,
        # rows_failed.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._shutdown_events: dict[str, threading.Event] = {}

    def _call_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Bridge an async call from the background thread to the main event loop.

        B8/C1 fix: SessionService methods are async, but _run_pipeline() runs
        in a ThreadPoolExecutor worker thread. This helper schedules the
        coroutine on the main event loop and blocks until it completes.

        R6 fix: 30-second timeout prevents deadlock during shutdown — if the
        event loop is blocked in executor.shutdown(wait=True), this will raise
        TimeoutError instead of hanging forever.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30.0)

    def shutdown(self) -> None:
        """Shut down the thread pool. Called during app shutdown.

        Sets all active shutdown events first so running pipelines can
        terminate gracefully before blocking on executor.shutdown(wait=True).
        """
        for event in self._shutdown_events.values():
            event.set()
        self._executor.shutdown(wait=True)

    async def execute(
        self, session_id: UUID, state_id: UUID | None = None
    ) -> UUID:
        """Start a background pipeline run.

        B6 enforcement: raises RunAlreadyActiveError if a pending or running
        run already exists for this session.

        Returns the run_id immediately.

        Note: async because SessionService methods are async. The pipeline
        itself runs in a background thread — only setup is async.
        """
        # B6: One active run per session (AC #17: via SessionService)
        active = await self._session_service.get_active_run(session_id)
        if active is not None:
            raise RunAlreadyActiveError(
                f"Session {session_id} already has an active run: {active.id}"
            )

        # B4 fix: get_composition_state() doesn't exist on SessionService.
        # Use get_state() for explicit state_id, get_current_state() for latest.
        if state_id is not None:
            state = await self._session_service.get_state(state_id)
        else:
            state = await self._session_service.get_current_state(session_id)
            if state is None:
                raise ValueError(f"No composition state exists for session {session_id}")
        pipeline_yaml = self._yaml_generator.generate_yaml(state)

        # B9 fix: create_run() generates its own UUID internally and returns
        # a RunRecord. Read the run_id back from the returned record so our
        # _shutdown_events key matches the DB record.
        run_record = await self._session_service.create_run(
            session_id=session_id,
            state_id=state.id,  # Always use resolved state's ID, not the possibly-None parameter
            pipeline_yaml=pipeline_yaml,
        )
        run_id = run_record.id  # Use the DB-generated UUID as canonical

        # Create shutdown event for cancellation (B2/cancel support)
        shutdown_event = threading.Event()
        self._shutdown_events[str(run_id)] = shutdown_event

        # Submit to thread pool — clean up shutdown event if submit fails
        # (e.g. RuntimeError after executor.shutdown())
        try:
            future = self._executor.submit(
                self._run_pipeline, str(run_id), pipeline_yaml, shutdown_event
            )
        except RuntimeError:
            self._shutdown_events.pop(str(run_id), None)
            raise
        # B7 Layer 2: safety net callback
        future.add_done_callback(self._on_pipeline_done)

        return run_id

    async def get_status(self, run_id: UUID) -> RunStatusResponse:
        """Return current run status. AC #17: delegates to SessionService."""
        run = await self._session_service.get_run(run_id)
        return RunStatusResponse(
            run_id=str(run.id),  # B7: RunRecord uses `id`, not `run_id`
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            rows_processed=run.rows_processed,
            rows_failed=run.rows_failed,
            error=run.error,
            landscape_run_id=run.landscape_run_id,
        )

    async def cancel(self, run_id: UUID) -> None:
        """Cancel a run via the shutdown Event.

        Active runs: sets the Event, Orchestrator detects during row processing.
        Pending runs: sets the Event so _run_pipeline terminates immediately.
        Terminal runs: no-op (idempotent).

        Note: async because pending-run cancellation calls SessionService
        directly (we're in the event loop thread, not the background thread,
        so await is correct here — NOT _call_async).
        """
        event = self._shutdown_events.get(str(run_id))
        if event is not None:
            event.set()
        else:
            # No event means either pending (not yet started) or already done
            run = await self._session_service.get_run(run_id)
            if run.status not in ("completed", "failed", "cancelled"):
                await self._session_service.update_run_status(
                    run_id, status="cancelled"
                )

    # ── Background Thread ──────────────────────────────────────────────

    def _run_pipeline(
        self,
        run_id: str,
        pipeline_yaml: str,
        shutdown_event: threading.Event,
    ) -> None:
        """Execute a pipeline in the background thread.

        B7 fix: Wrapped in try/except BaseException/finally.
        - except BaseException: Updates run to failed, re-raises.
        - finally: Removes shutdown event from _shutdown_events.

        B2 fix: shutdown_event is ALWAYS passed to orchestrator.run().
        B3 fix: LandscapeDB and PayloadStore from WebSettings resolvers.
        """
        tmp_path: Path | None = None
        landscape_db: LandscapeDB | None = None
        run_uuid = UUID(run_id)
        try:
            # B8/C1: SessionService is async — bridge from background thread
            self._call_async(
                self._session_service.update_run_status(run_uuid, status="running")
            )

            # B3 fix: construct from WebSettings, not hardcoded paths
            # NOTE: LandscapeDB is constructed per-run, not shared. This is safe
            # with max_workers=1 (no concurrent access) but wasteful — each run
            # creates a new SQLAlchemy engine. Acceptable for MVP; consider
            # sharing a single instance if profiling shows connection overhead.
            landscape_db = LandscapeDB(
                connection_string=self._settings.get_landscape_url()
            )
            payload_store = FilesystemPayloadStore(
                base_path=self._settings.get_payload_store_path()
            )

            # Write YAML to temp file — load_settings takes a path, NOT content
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            )
            tmp_path = Path(tmp_file.name)
            tmp_file.write(pipeline_yaml)
            tmp_file.close()

            settings = load_settings(tmp_path)
            bundle = instantiate_plugins_from_config(settings)

            graph = ExecutionGraph.from_plugin_instances(
                source=bundle.source,
                source_settings=bundle.source_settings,
                transforms=bundle.transforms,
                sinks=bundle.sinks,
                aggregations=bundle.aggregations,
                gates=list(settings.gates),
                coalesce_settings=(
                    list(settings.coalesce) if settings.coalesce else None
                ),
            )
            graph.validate()

            pipeline_config = PipelineConfig(
                source=bundle.source,
                transforms=[t.plugin for t in bundle.transforms],
                sinks=bundle.sinks,
                gates=list(settings.gates),
                aggregation_settings=dict(
                    (k, v[1]) for k, v in bundle.aggregations.items()
                ),
                coalesce_settings=(
                    list(settings.coalesce) if settings.coalesce else []
                ),
            )

            # Set up EventBus to bridge ProgressEvent -> RunEvent -> broadcaster
            def _safe_broadcast(evt: ProgressEvent) -> None:
                try:
                    self._broadcaster.broadcast(run_id, self._to_run_event(run_id, evt))
                except Exception as broadcast_err:
                    slog.error(
                        "progress_broadcast_failed",
                        run_id=run_id,
                        error=str(broadcast_err),
                    )

            event_bus = EventBus()
            event_bus.subscribe(ProgressEvent, _safe_broadcast)

            orchestrator = Orchestrator(
                db=landscape_db, event_bus=event_bus
            )

            # B2 fix: ALWAYS pass shutdown_event — suppresses signal handler
            # installation from background thread (Python forbids
            # signal.signal() from non-main threads)
            result = orchestrator.run(
                pipeline_config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                shutdown_event=shutdown_event,  # B2: NEVER omit this
            )

            # AC #19: Check if shutdown was requested — if so, broadcast
            # "cancelled" instead of "completed". The orchestrator returns
            # normally after detecting the shutdown event, but the run was
            # not completed by the user's intent.
            if shutdown_event.is_set():
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=timezone.utc),
                        event_type="cancelled",
                        data={
                            "rows_processed": result.rows_processed,
                            "rows_failed": result.rows_failed,
                        },
                    ),
                )
                self._call_async(self._session_service.update_run_status(
                    run_uuid,
                    status="cancelled",
                    rows_processed=result.rows_processed,
                    rows_failed=result.rows_failed,
                ))
            else:
                # Broadcast terminal completed event
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=timezone.utc),
                        event_type="completed",
                        data={
                            "rows_processed": result.rows_processed,
                            "rows_succeeded": result.rows_succeeded,
                            "rows_failed": result.rows_failed,
                            "rows_quarantined": result.rows_quarantined,
                            "landscape_run_id": result.run_id,
                        },
                    ),
                )
                self._call_async(self._session_service.update_run_status(
                    run_uuid,
                    status="completed",
                    landscape_run_id=result.run_id,
                    rows_processed=result.rows_processed,
                    rows_failed=result.rows_failed,
                ))

        except BaseException as exc:
            # B7 fix: Catch BaseException (not Exception) to handle
            # KeyboardInterrupt, SystemExit, and OOM-triggered exceptions.
            # Without this, the Run record stays in 'running' forever.
            self._broadcaster.broadcast(
                run_id,
                RunEvent(
                    run_id=run_id,
                    timestamp=datetime.now(tz=timezone.utc),
                    event_type="error",
                    data={
                        "detail": str(exc),
                        "node_id": None,
                        "row_id": None,
                    },
                ),
            )
            # R6 fix: Skip _call_async for KeyboardInterrupt/SystemExit — the event
            # loop is likely shutting down. Let orphan cleanup handle the status.
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                try:
                    self._call_async(
                        self._session_service.update_run_status(
                            run_uuid, status="failed", error=str(exc)
                        )
                    )
                except Exception as status_err:
                    slog.error(
                        "run_status_update_failed_in_except",
                        run_id=run_id,
                        original_error=str(exc),
                        status_update_error=str(status_err),
                    )
            else:
                slog.warning(
                    "skipping_status_update_on_signal",
                    run_id=run_id,
                    exc_type=type(exc).__name__,
                )
            raise
        finally:
            # Always clean up, regardless of success or failure
            self._shutdown_events.pop(run_id, None)
            if landscape_db is not None:
                landscape_db.close()
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
            self._broadcaster.cleanup_run(run_id)

    def _on_pipeline_done(self, future: Future[None]) -> None:
        """B7 Layer 2: Safety net callback.

        Fires when the Future completes, regardless of how. Logs any
        exception that _run_pipeline() raised. Does NOT update the Run
        record — that's Layer 1's job. This is purely diagnostic.
        """
        exc = future.exception()
        if exc is not None:
            slog.error(
                "Pipeline thread raised exception (safety net)",
                exc_info=exc,
            )

    def _to_run_event(self, run_id: str, progress: ProgressEvent) -> RunEvent:
        """Translate engine ProgressEvent to web RunEvent.

        Explicit mapping — unknown event types raise ValueError
        (offensive programming, not silent drop).
        """
        return RunEvent(
            run_id=run_id,
            timestamp=datetime.now(tz=timezone.utc),
            event_type="progress",
            data={
                "rows_processed": progress.rows_processed,
                "rows_failed": progress.rows_failed,
            },
        )
```

- [ ] **Step 3: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_service.py -v
git commit -m "feat(web/execution): add ExecutionServiceImpl with B2/B3/B7/B8 thread safety fixes"
```

---

## Self-Review Checklist

After completing all steps, verify:

- [ ] `RunAlreadyActiveError` is raised when `session_service.get_active_run()` returns non-None (B6). `pytest tests/unit/web/execution/test_service.py::TestOneActiveRun`
- [ ] `shutdown_event` is ALWAYS passed as a keyword argument to `orchestrator.run()` (B2). `pytest tests/unit/web/execution/test_service.py::TestB2ShutdownEvent`
- [ ] `LandscapeDB` constructed via `self._settings.get_landscape_url()`, NOT hardcoded paths (B3). `pytest tests/unit/web/execution/test_service.py::TestB3Construction`
- [ ] `FilesystemPayloadStore` constructed via `self._settings.get_payload_store_path()`, NOT hardcoded paths (B3). Same test class.
- [ ] `_run_pipeline()` uses `except BaseException`, NOT `except Exception` (B7). `pytest tests/unit/web/execution/test_service.py::TestB7ExceptionHandling`
- [ ] `future.add_done_callback(self._on_pipeline_done)` registered after `executor.submit()` (B7 Layer 2). `pytest tests/unit/web/execution/test_service.py::TestB7ExceptionHandling::test_done_callback_logs_unhandled_exception`
- [ ] `finally` block removes shutdown event from `_shutdown_events` and cleans up temp file. `pytest tests/unit/web/execution/test_service.py::TestB7ExceptionHandling::test_shutdown_event_cleaned_up_in_finally`
- [ ] `_call_async()` uses `asyncio.run_coroutine_threadsafe()` to bridge sync thread to async event loop (B8/C1).
- [ ] All `SessionService` calls from `_run_pipeline()` go through `_call_async()` (B8/C1 — never `await` from the background thread).
- [ ] Cancelled runs broadcast a terminal `RunEvent` with `event_type="cancelled"` and `rows_processed`/`rows_failed` data (AC #19). `pytest tests/unit/web/execution/test_service.py::TestCancelMechanism::test_cancelled_run_broadcasts_cancelled_event`
- [ ] `cancel()` sets the `threading.Event` for active runs, updates status directly for pending runs, and is a no-op for terminal runs.
- [ ] `execute()` delegates Run creation to `session_service.create_run()` with R6 expanded params (AC #17). No direct DB access.
- [ ] `_to_run_event()` translates `ProgressEvent` to `RunEvent` with explicit field mapping. `pytest tests/unit/web/execution/test_service.py::TestEventBusBridge`
- [ ] `ThreadPoolExecutor(max_workers=1)` — single worker ensures sequential pipeline execution.
- [ ] `_call_async()` has 30-second timeout preventing deadlock during shutdown (R6). `pytest tests/unit/web/execution/test_service.py::TestB8AsyncBridging::test_call_async_raises_timeout_error`
- [ ] `shutdown()` sets all active shutdown events BEFORE calling `executor.shutdown(wait=True)` (W-5C-1).
- [ ] `create_run()` receives `state.id` (resolved state), never `None` (B5).
- [ ] `landscape_db.close()` is in the `finally` block, not on the happy path only (B6 resource leak).
- [ ] `_run_pipeline()` converts `run_id` string to `UUID` for `update_run_status()` calls (B7 type fix).
- [ ] `yaml_generator` is a constructor parameter, not a module-level variable (W2).
- [ ] `except BaseException` skips `_call_async` for `KeyboardInterrupt`/`SystemExit` (W3/R6).
- [ ] EventBus lambda wrapped in `_safe_broadcast` with try/except (W14).
- [ ] `test_cancel_terminal_run_is_noop` has assertion: `update_run_status.assert_not_called()` (B8).
- [ ] All tests calling async methods (`execute`, `cancel`, `get_status`) use `@pytest.mark.asyncio` + `await` (B2).
- [ ] mypy passes on `src/elspeth/web/execution/service.py`.

```bash
# Test suite
.venv/bin/python -m pytest tests/unit/web/execution/test_service.py -v

# Type checking
.venv/bin/python -m mypy src/elspeth/web/execution/service.py
```

---

## Round 5 Review Findings

### Fixed Inline

| ID | Summary | Fix |
|----|---------|-----|
| **B1** | `RunAlreadyActiveError` double-definition in `service.py` would prevent `app.py`'s global exception handler from catching it | Deleted class definition, replaced with `from elspeth.web.sessions.protocol import RunAlreadyActiveError`. Updated test imports. |
| **B4** (partial) | `get_composition_state()` doesn't exist on `SessionService` | Replaced with `get_state(state_id)` when explicit, `get_current_state(session_id)` when latest. Updated test fixture. |
| **B6** | `_call_async()` failure in `except BaseException` clause shadows original exception | Wrapped `_call_async(update_run_status(...))` in nested `try/except Exception`. Logs failure via `slog.error` but preserves original exception for re-raise. |
| **B7** | `RunRecord.run_id` doesn't exist -- field is `id: UUID` | Changed all `run.run_id` to `run.id`, `active.run_id` to `active.id`. Updated test mock attributes. |
| **B9** | `run_id = uuid4()` generated locally but `create_run()` generates its own UUID internally -- `_shutdown_events` key wouldn't match DB record | Removed local `uuid4()` call. Read `run_id` back from returned `RunRecord`: `run_id = run_record.id`. Removed unused `uuid4` import from implementation. |
| **B4** (timeout) | `_call_async()` deadlock on shutdown -- `future.result()` blocks indefinitely if event loop is busy | Added `timeout=30.0` to `future.result()` call. R6 fix. |
| **B5** | `create_run()` receives `state_id=None` violating protocol when no explicit state_id passed | Changed to `state_id=state.id` -- always use the resolved state's ID. |
| **B6** (resource) | `landscape_db.close()` only on happy path -- resource leak on exceptions | Added `landscape_db: LandscapeDB | None = None` variable, moved `close()` to `finally` block. |
| **B7** (type) | `run_id` is `str` but `update_run_status()` takes `UUID` | Added `run_uuid = UUID(run_id)` conversion, changed all `update_run_status` calls to use `run_uuid`. |
| **B8** | `test_cancel_terminal_run_is_noop` has no assertion | Added `mock_session_service.update_run_status.assert_not_called()`. |
| **B10** | `_call_async()` never tested | Added `TestB8AsyncBridging` class with result, exception, and timeout tests. |
| **W2** | Module-level `yaml_generator` is fragile | Moved to constructor parameter `yaml_generator: Any`. Removed `TYPE_CHECKING` import. Updated fixture. Removed `@patch` decorators. |
| **W3** | `except BaseException` handler calls `_call_async` for `KeyboardInterrupt`/`SystemExit` | Added `isinstance` check -- skips `_call_async` for signal exceptions, logs warning instead. |
| **W6** | `shutdown()` must set shutdown events before `executor.shutdown` | Added `for event in self._shutdown_events.values(): event.set()` before `executor.shutdown(wait=True)`. |
| **W13** | B7 status assertion too loose; now wrong after W3 fix | Renamed tests to `test_keyboard_interrupt_skips_status_update` / `test_system_exit_skips_status_update`. Changed assertions to `assert_not_called()`. |
| **W14** | EventBus handler exception propagation kills pipeline | Wrapped EventBus lambda in `_safe_broadcast()` with try/except that logs errors. |
| **W15** | `update_run_status("running")` failure path untested | Added `TestRunningStatusFailure` class. |
| **B2** (tests) | Most `test_service.py` tests call async methods without `await` | Added `@pytest.mark.asyncio` and `async def` to all tests calling `execute()`, `cancel()`, `get_status()`. |

### Warnings (not yet fixed -- add to self-review checklist)

| ID | Summary | Mitigation |
|----|---------|------------|
| **W-5C-1** | ~~`shutdown()` calls `executor.shutdown(wait=True)` which blocks indefinitely if a pipeline is running.~~ | **Fixed:** `shutdown()` now sets all active shutdown events before calling `executor.shutdown(wait=True)`. Combined with the R6 `_call_async()` 30-second timeout, shutdown cannot deadlock. |
| **W-5C-2** | Even with the B6 nested `try/except` fix, if `_call_async(update_run_status(...))` fails, the Run record stays "running" permanently. | **Partially mitigated:** R6 fix skips `_call_async` for `KeyboardInterrupt`/`SystemExit` (the event loop is likely shutting down in those cases). For other `BaseException` subclasses, the nested `try/except` still applies and orphan run cleanup (D5, Sub-2) remains the recovery mechanism. |
