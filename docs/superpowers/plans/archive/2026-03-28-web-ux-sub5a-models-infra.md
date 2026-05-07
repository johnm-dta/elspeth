# Web UX Task-Plan 5A: Models & Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement Pydantic response models, ExecutionService protocol, and thread-safe ProgressBroadcaster
**Parent Plan:** `plans/2026-03-28-web-ux-sub5-execution.md`
**Spec:** `specs/2026-03-28-web-ux-sub5-execution-design.md`
**Depends On:** Sub-Plan 1 (Foundation — lifespan hook), Sub-Plan 2 (Sessions — RunEvent shapes)
**Blocks:** Task-Plans 5B (Validation), 5C (ExecutionServiceImpl)

---

## File Map

| Action | File |
|--------|------|
| Create | `src/elspeth/web/execution/__init__.py` |
| Create | `src/elspeth/web/execution/schemas.py` |
| Create | `src/elspeth/web/execution/protocol.py` |
| Create | `src/elspeth/web/execution/progress.py` |
| Create | `tests/unit/web/execution/__init__.py` |
| Create | `tests/unit/web/execution/test_schemas.py` |
| Create | `tests/unit/web/execution/test_progress.py` |

---

### Task 5.1: Pydantic Response Models

**Files:**
- Create: `src/elspeth/web/execution/__init__.py`
- Create: `src/elspeth/web/execution/schemas.py`
- Create: `tests/unit/web/execution/__init__.py`
- Create: `tests/unit/web/execution/test_schemas.py`

- [ ] **Step 1: Write tests for response models**

```python
# tests/unit/web/execution/test_schemas.py
"""Tests for execution response models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from elspeth.web.execution.schemas import (
    RunEvent,
    RunStatusResponse,
    ValidationCheck,
    ValidationError,
    ValidationResult,
)


class TestValidationResult:
    def test_valid_result(self) -> None:
        result = ValidationResult(
            is_valid=True,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=True,
                    detail="Settings loaded successfully",
                ),
                ValidationCheck(
                    name="plugin_instantiation",
                    passed=True,
                    detail="All plugins instantiated",
                ),
                ValidationCheck(
                    name="graph_structure",
                    passed=True,
                    detail="Graph is valid",
                ),
                ValidationCheck(
                    name="schema_compatibility",
                    passed=True,
                    detail="All edge schemas compatible",
                ),
            ],
            errors=[],
        )
        assert result.is_valid is True
        assert len(result.checks) == 4
        assert all(c.passed for c in result.checks)

    def test_invalid_result_with_attributed_error(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load", passed=True, detail="OK"
                ),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Graph validation failed",
                ),
            ],
            errors=[
                ValidationError(
                    component_id="gate_1",
                    component_type="gate",
                    message="Route destination 'nonexistent_sink' not found",
                    suggestion="Check sink names in gate configuration",
                ),
            ],
        )
        assert result.is_valid is False
        assert result.errors[0].component_id == "gate_1"
        assert result.errors[0].component_type == "gate"

    def test_structural_error_has_null_component(self) -> None:
        err = ValidationError(
            component_id=None,
            component_type=None,
            message="Graph contains a cycle",
            suggestion=None,
        )
        assert err.component_id is None
        assert err.component_type is None

    def test_skipped_check_recorded(self) -> None:
        """When settings_load fails, downstream checks are skipped but recorded."""
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Invalid YAML syntax",
                ),
                ValidationCheck(
                    name="plugin_instantiation",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
                ValidationCheck(
                    name="schema_compatibility",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="Invalid YAML syntax",
                    suggestion=None,
                ),
            ],
        )
        assert result.is_valid is False
        skipped = [c for c in result.checks if "Skipped" in c.detail]
        assert len(skipped) == 3


class TestRunEvent:
    def test_progress_event(self) -> None:
        event = RunEvent(
            run_id="run-123",
            timestamp=datetime.now(tz=timezone.utc),
            event_type="progress",
            data={"rows_processed": 50, "rows_failed": 2},
        )
        assert event.event_type == "progress"
        assert event.data["rows_processed"] == 50

    def test_completed_event(self) -> None:
        event = RunEvent(
            run_id="run-123",
            timestamp=datetime.now(tz=timezone.utc),
            event_type="completed",
            data={
                "rows_processed": 100,
                "rows_succeeded": 98,
                "rows_failed": 2,
                "rows_quarantined": 0,
                "landscape_run_id": "lscape-456",
            },
        )
        assert event.event_type == "completed"
        assert event.data["landscape_run_id"] == "lscape-456"

    def test_error_event(self) -> None:
        event = RunEvent(
            run_id="run-123",
            timestamp=datetime.now(tz=timezone.utc),
            event_type="error",
            data={"detail": "Division by zero", "node_id": "transform_1", "row_id": "row-5"},
        )
        assert event.event_type == "error"
        assert event.data["node_id"] == "transform_1"

    def test_invalid_event_type_rejected(self) -> None:
        """event_type is a Literal — Pydantic rejects unknown values."""
        with pytest.raises(Exception):  # ValidationError from Pydantic
            RunEvent(
                run_id="run-123",
                timestamp=datetime.now(tz=timezone.utc),
                event_type="unknown",  # type: ignore[arg-type]
                data={},
            )


class TestRunStatusResponse:
    def test_pending_status(self) -> None:
        status = RunStatusResponse(
            run_id="run-123",
            status="pending",
            started_at=None,
            finished_at=None,
            rows_processed=0,
            rows_failed=0,
            error=None,
            landscape_run_id=None,
        )
        assert status.status == "pending"
        assert status.started_at is None

    def test_completed_status(self) -> None:
        now = datetime.now(tz=timezone.utc)
        status = RunStatusResponse(
            run_id="run-123",
            status="completed",
            started_at=now,
            finished_at=now,
            rows_processed=100,
            rows_failed=0,
            error=None,
            landscape_run_id="lscape-456",
        )
        assert status.landscape_run_id == "lscape-456"

    def test_failed_status_has_error(self) -> None:
        now = datetime.now(tz=timezone.utc)
        status = RunStatusResponse(
            run_id="run-123",
            status="failed",
            started_at=now,
            finished_at=now,
            rows_processed=50,
            rows_failed=50,
            error="Connection refused",
            landscape_run_id=None,
        )
        assert status.error == "Connection refused"
```

- [ ] **Step 2: Implement response models**

```python
# src/elspeth/web/execution/schemas.py
"""Pydantic response models for execution endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class ValidationCheck(BaseModel):
    """Individual check result from dry-run validation."""

    name: str
    passed: bool
    detail: str


class ValidationError(BaseModel):
    """Error with per-component attribution."""

    component_id: str | None
    component_type: str | None
    message: str
    suggestion: str | None


class ValidationResult(BaseModel):
    """Result of dry-run validation against real engine code."""

    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[ValidationError]


class RunEvent(BaseModel):
    """WebSocket event payload for live progress streaming."""

    run_id: str
    timestamp: datetime  # NOTE: Fast pipelines may produce identical timestamps.
    # Event ordering is guaranteed by the asyncio.Queue FIFO, not by timestamp.
    # Frontend must NOT sort by timestamp — use arrival order instead.
    event_type: Literal["progress", "error", "completed", "cancelled"]
    data: dict[str, Any]


class RunStatusResponse(BaseModel):
    """REST response for run status queries."""

    run_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    started_at: datetime | None
    finished_at: datetime | None
    rows_processed: int
    rows_failed: int
    error: str | None
    landscape_run_id: str | None
```

- [ ] **Step 3: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py -v
git commit -m "feat(web/execution): add Pydantic response models for validation, status, events"
```

---

### Task 5.2: ExecutionService Protocol

**Files:**
- Create: `src/elspeth/web/execution/protocol.py`

- [ ] **Step 1: Define the protocol**

```python
# src/elspeth/web/execution/protocol.py
"""ExecutionService protocol — called from FastAPI route handlers."""
from __future__ import annotations

from typing import Protocol
from uuid import UUID

from elspeth.web.execution.schemas import RunStatusResponse, ValidationResult


class ExecutionService(Protocol):
    """Protocol for pipeline execution operations.

    All methods are called from FastAPI route handlers in the async context.
    execute() returns immediately; the pipeline runs in a background thread.
    """

    async def validate(self, session_id: UUID) -> ValidationResult:
        """Dry-run validation using real engine code paths.

        Loads the current CompositionState for the session, generates YAML,
        and runs it through load_settings -> instantiate_plugins_from_config
        -> ExecutionGraph.from_plugin_instances -> graph.validate().

        Async because the implementation wraps the sync validate_pipeline()
        call via run_in_executor to avoid blocking the event loop.
        """
        ...

    async def execute(self, session_id: UUID, state_id: UUID | None = None) -> UUID:
        """Start a background pipeline run.

        Returns the run_id immediately. Raises RunAlreadyActiveError if
        a pending or running Run already exists for this session.

        Note: async because it calls SessionService (async) for active-run
        check and run creation. The actual pipeline runs in a background
        thread via ThreadPoolExecutor — only the setup is async.
        """
        ...

    async def get_status(self, run_id: UUID) -> RunStatusResponse:
        """Return current run status from the Run database record."""
        ...

    async def cancel(self, run_id: UUID) -> None:
        """Cancel a run. Sets the shutdown Event for active runs.

        Idempotent — cancelling a terminal run is a no-op.
        Note: async because cancelling a pending run calls
        SessionService.update_run_status() directly (not via _call_async,
        since we're in the event loop thread).
        """
        ...
```

No tests needed for a pure protocol. This exists for type checking and dependency injection.

- [ ] **Step 2: Commit**

```bash
git commit -m "feat(web/execution): add ExecutionService protocol"
```

---

### Task 5.3: ProgressBroadcaster (B1 Fix — Thread Safety)

**Files:**
- Create: `src/elspeth/web/execution/progress.py`
- Create: `tests/unit/web/execution/test_progress.py`

This is the critical async/thread bridge. Every test must verify that the thread boundary is crossed correctly.

- [ ] **Step 1: Write ProgressBroadcaster tests**

```python
# tests/unit/web/execution/test_progress.py
"""Tests for ProgressBroadcaster — thread-safe async event delivery.

These tests verify the B1 fix: broadcast() uses loop.call_soon_threadsafe()
to safely push events from a background thread into asyncio queues.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.schemas import RunEvent


def _make_event(run_id: str = "run-1", event_type: str = "progress") -> RunEvent:
    return RunEvent(
        run_id=run_id,
        timestamp=datetime.now(tz=timezone.utc),
        event_type=event_type,  # type: ignore[arg-type]
        data={"rows_processed": 10, "rows_failed": 0},
    )


class TestProgressBroadcasterSubscription:
    def test_subscribe_returns_queue(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue = broadcaster.subscribe("run-1")
            assert isinstance(queue, asyncio.Queue)
        finally:
            loop.close()

    def test_unsubscribe_removes_queue(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue = broadcaster.subscribe("run-1")
            broadcaster.unsubscribe("run-1", queue)
            # Verify subscriber set is empty after unsubscribe
            assert len(broadcaster._subscribers.get("run-1", set())) == 0
        finally:
            loop.close()

    def test_unsubscribe_unknown_run_is_noop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            queue: asyncio.Queue[RunEvent] = asyncio.Queue()
            # Should not raise
            broadcaster.unsubscribe("nonexistent", queue)
        finally:
            loop.close()

    def test_multiple_subscribers_for_same_run(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            q1 = broadcaster.subscribe("run-1")
            q2 = broadcaster.subscribe("run-1")
            assert q1 is not q2
            assert len(broadcaster._subscribers["run-1"]) == 2
        finally:
            loop.close()


class TestProgressBroadcasterThreadSafety:
    """Verify B1 fix: broadcast() uses loop.call_soon_threadsafe()."""

    def test_broadcast_uses_call_soon_threadsafe(self) -> None:
        """broadcast() MUST use loop.call_soon_threadsafe, not direct put_nowait."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        queue = broadcaster.subscribe("run-1")
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        # Verify call_soon_threadsafe was called with put_nowait
        mock_loop.call_soon_threadsafe.assert_called_once_with(
            queue.put_nowait, event
        )

    def test_broadcast_to_multiple_subscribers(self) -> None:
        """Each subscriber queue gets its own call_soon_threadsafe call."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        broadcaster.subscribe("run-1")
        broadcaster.subscribe("run-1")
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        assert mock_loop.call_soon_threadsafe.call_count == 2

    def test_broadcast_no_subscribers_is_noop(self) -> None:
        """Broadcasting with no subscribers does not raise."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        event = _make_event()

        # Should not raise
        broadcaster.broadcast("run-1", event)
        mock_loop.call_soon_threadsafe.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_from_background_thread_delivers_events(self) -> None:
        """End-to-end: background thread broadcasts, async consumer receives.

        This is the actual B1 scenario — _run_pipeline() calls broadcast()
        from a ThreadPoolExecutor worker thread, and the WebSocket handler
        awaits queue.get() on the event loop thread.
        """
        loop = asyncio.get_running_loop()
        broadcaster = ProgressBroadcaster(loop)
        queue = broadcaster.subscribe("run-1")
        event = _make_event()

        def background_broadcast() -> None:
            """Simulate _run_pipeline() calling broadcast from worker thread."""
            broadcaster.broadcast("run-1", event)

        # Run broadcast in a real background thread
        thread = threading.Thread(target=background_broadcast)
        thread.start()
        thread.join(timeout=5.0)

        # The event should be in the queue via call_soon_threadsafe
        received = await asyncio.wait_for(queue.get(), timeout=5.0)
        assert received.run_id == "run-1"
        assert received.event_type == "progress"
        assert received.data["rows_processed"] == 10

    @pytest.mark.asyncio
    async def test_multiple_broadcasts_from_thread_arrive_in_order(self) -> None:
        """Events broadcast from a single thread arrive in FIFO order."""
        loop = asyncio.get_running_loop()
        broadcaster = ProgressBroadcaster(loop)
        queue = broadcaster.subscribe("run-1")

        events = [
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=timezone.utc),
                event_type="progress",
                data={"rows_processed": i, "rows_failed": 0},
            )
            for i in range(5)
        ]

        def background_broadcast() -> None:
            for event in events:
                broadcaster.broadcast("run-1", event)

        thread = threading.Thread(target=background_broadcast)
        thread.start()
        thread.join(timeout=5.0)

        received = []
        for _ in range(5):
            item = await asyncio.wait_for(queue.get(), timeout=5.0)
            received.append(item.data["rows_processed"])

        assert received == [0, 1, 2, 3, 4]

    def test_broadcast_after_unsubscribe_does_not_deliver(self) -> None:
        """After unsubscribe, broadcast does not target the removed queue."""
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        broadcaster = ProgressBroadcaster(mock_loop)
        queue = broadcaster.subscribe("run-1")
        broadcaster.unsubscribe("run-1", queue)
        event = _make_event()

        broadcaster.broadcast("run-1", event)

        mock_loop.call_soon_threadsafe.assert_not_called()


class TestProgressBroadcasterCleanup:
    def test_cleanup_run_removes_all_subscribers(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            broadcaster.subscribe("run-1")
            broadcaster.subscribe("run-1")
            broadcaster.cleanup_run("run-1")
            assert "run-1" not in broadcaster._subscribers
        finally:
            loop.close()

    def test_cleanup_unknown_run_is_noop(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            broadcaster = ProgressBroadcaster(loop)
            broadcaster.cleanup_run("nonexistent")  # Should not raise
        finally:
            loop.close()
```

- [ ] **Step 2: Implement ProgressBroadcaster**

```python
# src/elspeth/web/execution/progress.py
"""ProgressBroadcaster — thread-safe bridge from sync pipeline to async WebSocket.

B1 Fix: All cross-thread pushes use loop.call_soon_threadsafe() to schedule
queue.put_nowait() on the event loop thread. Direct queue.put_nowait() from
a background thread would corrupt the event loop's internal state.

Construction timing: Created inside the FastAPI lifespan async context manager,
NOT in the synchronous create_app() factory. The loop reference is captured via
asyncio.get_running_loop() inside the lifespan, which guarantees a running event
loop exists and is Python 3.12+ compatible (asyncio.get_event_loop() emits a
deprecation warning when no running loop exists). The ExecutionServiceImpl is
also constructed inside the lifespan (after ProgressBroadcaster), not in
create_app(), because it depends on the broadcaster instance.
"""
from __future__ import annotations

import asyncio
import threading

from elspeth.web.execution.schemas import RunEvent


class ProgressBroadcaster:
    """In-process event broadcaster for pipeline progress.

    Bridges the synchronous background thread (where the Orchestrator runs)
    to the async WebSocket handlers (where clients receive events).

    Thread safety contract:
    - subscribe() and unsubscribe() may be called from the asyncio event loop thread.
    - broadcast() is called from the background pipeline thread.
    - broadcast() uses self._loop.call_soon_threadsafe() for every queue.put_nowait().
    - All mutations to and iterations over self._subscribers are protected by
      self._lock (a threading.Lock). This ensures correctness regardless of
      GIL presence (safe under PEP 703 / free-threaded Python).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue[RunEvent]:
        """Create and register a subscriber queue for a run.

        Called from the WebSocket handler when a client connects.
        Returns the queue that the handler awaits on.
        """
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue[RunEvent]) -> None:
        """Remove a subscriber queue. Called on WebSocket disconnect.

        Always called from a finally block to ensure cleanup.
        """
        with self._lock:
            if run_id in self._subscribers:
                self._subscribers[run_id].discard(queue)

    def broadcast(self, run_id: str, event: RunEvent) -> None:
        """Thread-safe broadcast — callable from background threads.

        B1 fix: Uses self._loop.call_soon_threadsafe() to schedule
        queue.put_nowait() on the event loop thread. This is the standard
        Python pattern for pushing data from a synchronous thread into
        an asyncio context.

        B3 fix: Copies the subscriber set under the lock before iterating,
        so concurrent subscribe/unsubscribe cannot cause RuntimeError.
        """
        with self._lock:
            subs = set(self._subscribers.get(run_id, ()))
        for queue in subs:
            self._loop.call_soon_threadsafe(queue.put_nowait, event)

    def cleanup_run(self, run_id: str) -> None:
        """Remove all subscribers for a completed/failed run.

        Called after the terminal event has been broadcast and all
        WebSocket handlers have disconnected.
        """
        with self._lock:
            self._subscribers.pop(run_id, None)
```

- [ ] **Step 3: Run tests, commit**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_progress.py -v
git commit -m "feat(web/execution): add ProgressBroadcaster with B1 thread safety fix"
```

---

## Self-Review Checklist

- [ ] `schemas.py` defines all five models: `ValidationCheck`, `ValidationError`, `ValidationResult`, `RunEvent`, `RunStatusResponse`
- [ ] `RunEvent.event_type` is `Literal["progress", "error", "completed", "cancelled"]` -- Pydantic rejects unknown values
- [ ] `RunStatusResponse.status` is `Literal["pending", "running", "completed", "failed", "cancelled"]`
- [ ] `ValidationError` supports `component_id=None` for structural (non-component) errors
- [ ] `protocol.py` defines `ExecutionService` as a `Protocol` with `validate`, `execute`, `get_status`, `cancel`
- [ ] `validate()` is async (wraps sync call via `run_in_executor`); `execute()`, `get_status()`, `cancel()` are async
- [ ] `ProgressBroadcaster.__init__` captures the event loop reference
- [ ] `broadcast()` uses `self._loop.call_soon_threadsafe(queue.put_nowait, event)` -- never direct `put_nowait()`
- [ ] `subscribe()` / `unsubscribe()` / `broadcast()` / `cleanup_run()` all acquire `self._lock` before mutating or iterating `_subscribers`
- [ ] `unsubscribe()` with unknown run_id is a no-op (no KeyError)
- [ ] `cleanup_run()` removes the entire subscriber set for a run_id
- [ ] Thread safety test verifies `call_soon_threadsafe` is used via mock loop
- [ ] End-to-end test broadcasts from a real `threading.Thread` and receives via `await queue.get()`
- [ ] FIFO ordering test confirms events arrive in broadcast order
- [ ] All `__init__.py` files created for both `src/elspeth/web/execution/` and `tests/unit/web/execution/`
- [ ] No defensive `.get()` with fabricated defaults on our own data structures

---

## Round 5 Review Findings

**No blocking issues.**

**Warnings (fixed):**
- **W-5A-1: Concurrent subscribe/broadcast not thread-safe.** Fixed: Added `threading.Lock` (`self._lock`) to protect all mutations and iterations of `self._subscribers`. `broadcast()` copies the subscriber set under the lock before iterating. All four methods (`subscribe`, `unsubscribe`, `broadcast`, `cleanup_run`) acquire the lock.
- **W-5A-2: GIL reliance undocumented.** Fixed: No longer relies on GIL. The `threading.Lock` provides explicit synchronization, safe under PEP 703 / free-threaded Python. Class docstring documents the locking strategy.
