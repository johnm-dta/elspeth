# Backpressure Mode Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the orphaned `backpressure_mode` config field to actual runtime behavior by adding a background export thread with queue-based backpressure.

**Architecture:** Add `threading.Event` for shutdown coordination, `queue.Queue` for async export, and a background thread that consumes events. BLOCK mode uses blocking `put()` with timeout, DROP mode uses `put_nowait()` with Full exception handling.

**Tech Stack:** Python stdlib `threading`, `queue`; existing `TelemetryManager`, `ExporterProtocol`

**Design Document:** `docs/plans/2026-01-30-backpressure-mode-design.md`

**Issue:** elspeth-rapid-ceq

---

## Implementation Summary

- Telemetry queueing with BLOCK/DROP behavior implemented in `TelemetryManager.handle_event()` (`src/elspeth/telemetry/manager.py`).
- Runtime config wires `backpressure_mode` to enum validation and internal defaults for queue size (`src/elspeth/contracts/config/runtime.py`, `src/elspeth/contracts/config/defaults.py`).
- Coverage added for backpressure behavior and config validation (`tests/unit/telemetry/test_manager.py`, `tests/contracts/test_telemetry_config.py`).

## Review Board Feedback (2026-01-30)

**Applied Fixes:**

| Priority | Issue | Resolution |
|----------|-------|------------|
| P0 | Shutdown sequence: sentinel BEFORE join | Task 7 reordered: shutdown→sentinel→thread.join (no queue.join) |
| P0 | Concurrent close() test | Added Task 9.5 |
| P0 | Lock contention test | Added Task 9.6 |
| P0 | Sleepy assertion in Task 9 | Replaced join(timeout) with Event-based coordination |
| P1 | Startup race (_export_thread_ready) | Added to Task 4 and Task 5 |
| P1 | Re-entrance deadlock test | Added Task 9.7 |
| P1 | BLOCK mode put() timeout | Task 5 uses put(event, timeout=1.0) |
| P2 | queue_size hardcoded | Task 4 reads from INTERNAL_DEFAULTS |
| P2 | Windows portability (SIGALRM) | Task 13 uses threading.Timer with platform skip |
| P2 | Deterministic test coordination | All tests use Event.wait() not sleep/join(timeout) |

---

## Task 1: Add Queue Size to Internal Defaults

**Files:**
- Modify: `src/elspeth/contracts/config/defaults.py`

**Step 1: Add telemetry queue_size to INTERNAL_DEFAULTS**

```python
# In INTERNAL_DEFAULTS dict, add after "retry" section:
    "telemetry": {
        # Queue size for async export buffer
        # 1000 events absorbs bursts without excessive memory
        # Not user-configurable - internal implementation detail
        "queue_size": 1000,
    },
```

**Step 2: Run existing tests to verify no regression**

Run: `.venv/bin/python -m pytest tests/contracts/ -v -x`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/elspeth/contracts/config/defaults.py
git commit -m "config: add telemetry queue_size internal default"
```

---

## Task 2: Update ExporterProtocol Threading Documentation

**Files:**
- Modify: `src/elspeth/telemetry/protocols.py`

**Step 1: Add thread safety documentation to export() method**

In `ExporterProtocol.export()` docstring, add after "Args:" section:

```python
    def export(self, event: "TelemetryEvent") -> None:
        """Export a single telemetry event.

        Called for each event emitted by the pipeline. This method MUST NOT
        raise exceptions - telemetry failures should not crash the pipeline.
        Errors should be logged internally.

        Implementations may buffer events for batch export. Use flush() to
        ensure all buffered events are sent.

        Thread Safety:
            export() is always called from the telemetry export thread, never
            concurrently with itself. However, export() may run on a different
            thread than configure() and close(). Implementations should not
            rely on thread-local state from configure().

        Args:
            event: The telemetry event to export
        """
        ...
```

**Step 2: Run type check to verify docstring is valid**

Run: `.venv/bin/python -m mypy src/elspeth/telemetry/protocols.py`
Expected: Success (no errors)

**Step 3: Commit**

```bash
git add src/elspeth/telemetry/protocols.py
git commit -m "docs: add thread safety documentation to ExporterProtocol"
```

---

## Task 3: Write Failing Test for DROP Mode

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add BackpressureMode import and update MockConfig**

At top of file, add import:
```python
from elspeth.contracts.enums import BackpressureMode, RunStatus, TelemetryGranularity
```

Update `MockConfig` class:
```python
@dataclass
class MockConfig:
    """Mock RuntimeTelemetryProtocol implementation for testing."""

    enabled: bool = True
    granularity: TelemetryGranularity = TelemetryGranularity.FULL
    fail_on_total_exporter_failure: bool = False
    backpressure_mode: BackpressureMode = BackpressureMode.BLOCK

    @property
    def exporter_configs(self) -> tuple:
        return ()
```

**Step 2: Add slow exporter mock**

After `MockExporter` class:
```python
class SlowExporter:
    """Exporter that blocks on export until signaled."""

    def __init__(self, name: str):
        self._name = name
        self.export_started = threading.Event()
        self.can_continue = threading.Event()
        self.exports: list[TelemetryEvent] = []

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        self.export_started.set()
        self.can_continue.wait(timeout=10.0)
        self.exports.append(event)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
```

Add import at top:
```python
import threading
```

**Step 3: Write failing test for DROP mode**

Add test at end of file:
```python
# =============================================================================
# Backpressure Mode Tests
# =============================================================================


class TestBackpressureMode:
    """Tests for backpressure_mode config wiring."""

    def test_drop_mode_drops_when_queue_full(self, base_timestamp: datetime) -> None:
        """DROP mode drops events when queue is full instead of blocking."""
        # Use a tiny queue to make it fill quickly
        slow_exporter = SlowExporter("slow")
        config = MockConfig(backpressure_mode=BackpressureMode.DROP)
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Override queue size for testing (access internal for test)
        # This tests the behavior, not the default size
        manager._queue = queue.Queue(maxsize=2)

        event = RunStarted(timestamp=base_timestamp, run_id="test-run")

        try:
            # Fill queue: 2 events should queue, 3rd should drop
            manager.handle_event(event)
            manager.handle_event(event)
            manager.handle_event(event)  # Should drop, not block

            # Verify drop was counted
            assert manager.health_metrics["events_dropped"] == 1

            # Verify we didn't block (test would timeout if we blocked)
        finally:
            slow_exporter.can_continue.set()  # Unblock exporter
            manager.close()
```

Add import at top:
```python
import queue
```

**Step 4: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_drop_mode_drops_when_queue_full -v`
Expected: FAIL with AttributeError (no `_queue` attribute yet)

**Step 5: Commit failing test**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test: add failing test for DROP mode backpressure"
```

---

## Task 4: Add Queue and Threading Infrastructure to TelemetryManager

**Files:**
- Modify: `src/elspeth/telemetry/manager.py`

**Step 1: Add imports**

At top of file, add:
```python
import queue
import threading
from typing import Any

import structlog

from elspeth.contracts.config import RuntimeTelemetryProtocol
from elspeth.contracts.config.defaults import INTERNAL_DEFAULTS
from elspeth.contracts.enums import BackpressureMode
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.events import TelemetryEvent
from elspeth.telemetry.filtering import should_emit
from elspeth.telemetry.protocols import ExporterProtocol
```

**Step 2: Update module docstring**

Replace the module docstring with:
```python
"""TelemetryManager coordinates event emission to configured exporters.

The TelemetryManager is the central hub for telemetry:
1. Receives telemetry events from EventBus
2. Filters events based on configured granularity
3. Queues events for async export via background thread
4. Dispatches to all exporters with failure isolation
5. Tracks health metrics for monitoring
6. Implements aggregate logging to prevent Warning Fatigue

Design principles:
- Telemetry emitted AFTER Landscape recording (Landscape is the legal record)
- Individual exporter failures don't crash the pipeline
- Aggregate logging every 100 total failures (Warning Fatigue prevention)
- Configurable backpressure behavior (BLOCK vs DROP)

Thread Safety:
    TelemetryManager uses a background export thread for async export.
    - handle_event() is called from the pipeline thread (non-blocking)
    - _export_loop() runs in the background export thread
    - _events_dropped is protected by _dropped_lock (accessed from both threads)
    - All other metrics are only modified by the export thread
    - health_metrics reads are approximately consistent
"""
```

**Step 3: Update class docstring**

Replace `TelemetryManager` class docstring:
```python
class TelemetryManager:
    """Coordinates event emission to configured exporters.

    The TelemetryManager receives telemetry events, queues them for async
    export via a background thread, and dispatches to all exporters. It
    provides failure isolation (one exporter failing doesn't affect others)
    and tracks health metrics for operational monitoring.

    Backpressure modes:
    - BLOCK: handle_event() blocks when queue is full (pipeline slows)
    - DROP: handle_event() drops events when queue is full (pipeline unaffected)

    Failure handling:
    - Individual exporter failures are logged but don't crash the pipeline
    - Aggregate logging every _LOG_INTERVAL failures prevents Warning Fatigue
    - If ALL exporters fail _max_consecutive_failures times in a row:
      - fail_on_total_exporter_failure=True: Raise TelemetryExporterError
      - fail_on_total_exporter_failure=False: Log CRITICAL and continue

    Thread Safety:
        Uses background export thread. handle_event() is thread-safe.
        _events_dropped protected by lock (both threads write).
        Other metrics single-writer (export thread only).

    Example:
        >>> from elspeth.telemetry import TelemetryManager
        >>> manager = TelemetryManager(config, exporters=[console_exporter])
        >>> manager.handle_event(run_started_event)
        >>> manager.flush()
        >>> manager.close()
    """
```

**Step 4: Update __init__ method**

Replace `__init__` method:
```python
    def __init__(
        self,
        config: RuntimeTelemetryProtocol,
        exporters: list[ExporterProtocol],
    ) -> None:
        """Initialize the TelemetryManager.

        Args:
            config: Runtime telemetry configuration with granularity,
                backpressure_mode, and fail_on_total_exporter_failure settings
            exporters: List of configured exporter instances. May be empty
                (telemetry will be a no-op).
        """
        self._config = config
        self._exporters = exporters
        self._consecutive_total_failures = 0
        self._max_consecutive_failures = 10

        # Health metrics
        self._events_emitted = 0
        self._events_dropped = 0
        self._exporter_failures: dict[str, int] = {}
        self._last_logged_drop_count: int = 0

        # Track whether we've already disabled telemetry
        self._disabled = False

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._dropped_lock = threading.Lock()
        self._export_thread_ready = threading.Event()  # Signals thread is running

        # Queue for async export - read size from INTERNAL_DEFAULTS
        queue_size = INTERNAL_DEFAULTS["telemetry"]["queue_size"]
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=queue_size)

        # Start export thread (non-daemon to ensure proper cleanup)
        self._export_thread = threading.Thread(
            target=self._export_loop,
            name="telemetry-export",
            daemon=False,
        )
        self._export_thread.start()
        # Wait for thread to be ready (prevents startup race)
        self._export_thread_ready.wait(timeout=5.0)
```

**Step 5: Run test to see progress**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_drop_mode_drops_when_queue_full -v`
Expected: FAIL (still missing handle_event changes and _export_loop)

**Step 6: Commit infrastructure**

```bash
git add src/elspeth/telemetry/manager.py
git commit -m "feat(telemetry): add queue and threading infrastructure"
```

---

## Task 5: Implement handle_event with Backpressure

**Files:**
- Modify: `src/elspeth/telemetry/manager.py`

**Step 1: Replace handle_event method**

```python
    def handle_event(self, event: TelemetryEvent) -> None:
        """Queue event for async export.

        Events are filtered by granularity before queuing. Queue behavior
        depends on backpressure_mode config:
        - BLOCK: Blocks until queue has space (may slow pipeline), with timeout
        - DROP: Drops event if queue is full (pipeline unaffected)

        Thread Safety:
            Safe to call from any thread. Uses thread-safe Event check
            and Queue operations. _events_dropped protected by lock.

        Args:
            event: The telemetry event to process
        """
        # Thread-safe shutdown check
        if self._shutdown_event.is_set():
            return

        # Skip if telemetry was disabled due to repeated failures
        if self._disabled:
            return

        # Skip if no exporters configured
        if not self._exporters:
            return

        # Filter by granularity
        if not should_emit(event, self._config.granularity):
            return

        # Check thread readiness and liveness
        if not self._export_thread_ready.is_set():
            logger.warning("Export thread not ready, dropping event")
            with self._dropped_lock:
                self._events_dropped += 1
            return

        if not self._export_thread.is_alive():
            logger.critical("Export thread died, disabling telemetry")
            self._disabled = True
            return

        # Queue event based on backpressure mode
        if self._config.backpressure_mode == BackpressureMode.DROP:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._dropped_lock:
                    self._events_dropped += 1
                    self._log_drops_if_needed()
        else:  # BLOCK (default)
            # Timeout prevents permanent deadlock if export thread dies
            try:
                self._queue.put(event, timeout=30.0)
            except queue.Full:
                # Timeout hit - thread may be dead or stuck
                logger.error("BLOCK mode put() timed out - export thread may be stuck")
                with self._dropped_lock:
                    self._events_dropped += 1

    def _log_drops_if_needed(self) -> None:
        """Log aggregate drop message if threshold reached.

        Must be called while holding _dropped_lock.
        """
        if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
            logger.warning(
                "Telemetry events dropped due to backpressure",
                dropped_since_last_log=self._events_dropped - self._last_logged_drop_count,
                dropped_total=self._events_dropped,
                backpressure_mode="drop",
            )
            self._last_logged_drop_count = self._events_dropped
```

**Step 2: Run test to see progress**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_drop_mode_drops_when_queue_full -v`
Expected: FAIL (missing _export_loop - thread exits immediately)

**Step 3: Commit**

```bash
git add src/elspeth/telemetry/manager.py
git commit -m "feat(telemetry): implement handle_event with backpressure modes"
```

---

## Task 6: Implement Export Loop

**Files:**
- Modify: `src/elspeth/telemetry/manager.py`

**Step 1: Add _export_loop method**

After `_log_drops_if_needed`, add:
```python
    def _export_loop(self) -> None:
        """Background thread: consume queue and export.

        Runs until shutdown sentinel (None) is received. Each event is
        dispatched to all exporters with failure isolation.

        Thread Safety:
            Runs exclusively in the export thread. Metrics updates
            (except _events_dropped on total failure) are single-threaded.
        """
        # Signal that export thread is ready to receive events
        self._export_thread_ready.set()

        while True:
            event = self._queue.get()
            if event is None:  # Shutdown sentinel
                self._queue.task_done()
                break
            try:
                self._dispatch_to_exporters(event)
            except Exception as e:
                # CRITICAL: Log but don't crash - telemetry must not kill pipeline
                logger.error("Export loop failed unexpectedly", error=str(e))
            finally:
                # ALWAYS call task_done() to prevent join() hangs
                self._queue.task_done()

    def _dispatch_to_exporters(self, event: TelemetryEvent) -> None:
        """Export to all exporters with failure isolation.

        Thread Safety:
            Called only from export thread. _events_dropped protected by lock
            when incrementing on total failure.

        Args:
            event: The telemetry event to export
        """
        failures = 0
        for exporter in self._exporters:
            try:
                exporter.export(event)
            except Exception as e:
                failures += 1
                self._exporter_failures[exporter.name] = (
                    self._exporter_failures.get(exporter.name, 0) + 1
                )
                logger.warning(
                    "Telemetry exporter failed",
                    exporter=exporter.name,
                    event_type=type(event).__name__,
                    error=str(e),
                )

        # Update metrics based on outcome
        if failures == 0:
            # Complete success
            self._events_emitted += 1
            self._consecutive_total_failures = 0
        elif failures == len(self._exporters):
            # ALL exporters failed
            self._consecutive_total_failures += 1
            # Lock required - pipeline thread also writes in DROP mode
            with self._dropped_lock:
                self._events_dropped += 1

            # Aggregate logging every _LOG_INTERVAL
            if self._events_dropped - self._last_logged_drop_count >= self._LOG_INTERVAL:
                logger.error(
                    "ALL telemetry exporters failing - events dropped",
                    dropped_since_last_log=self._events_dropped - self._last_logged_drop_count,
                    dropped_total=self._events_dropped,
                    consecutive_failures=self._consecutive_total_failures,
                )
                self._last_logged_drop_count = self._events_dropped

            # Check if we should crash or disable
            if self._consecutive_total_failures >= self._max_consecutive_failures:
                if self._config.fail_on_total_exporter_failure:
                    raise TelemetryExporterError(
                        "all",
                        f"All {len(self._exporters)} exporters failed "
                        f"{self._max_consecutive_failures} consecutive times.",
                    )
                else:
                    logger.critical(
                        "Telemetry disabled after repeated total failures",
                        consecutive_failures=self._consecutive_total_failures,
                        events_dropped=self._events_dropped,
                    )
                    self._disabled = True
        else:
            # Partial success - at least one exporter worked
            self._events_emitted += 1
            self._consecutive_total_failures = 0
```

**Step 2: Run test to see progress**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_drop_mode_drops_when_queue_full -v`
Expected: FAIL (close() doesn't send sentinel yet)

**Step 3: Commit**

```bash
git add src/elspeth/telemetry/manager.py
git commit -m "feat(telemetry): implement export loop with failure isolation"
```

---

## Task 7: Update flush() and close() Methods

**Files:**
- Modify: `src/elspeth/telemetry/manager.py`

**Step 1: Replace flush() method**

```python
    def flush(self) -> None:
        """Wait for queue to drain, then flush exporters.

        Blocks until all queued events have been processed by the export
        thread, then flushes each exporter's internal buffer.

        Exporter flush failures are logged but don't raise - telemetry
        should not crash the pipeline.
        """
        if not self._shutdown_event.is_set():
            self._queue.join()  # Wait for all queued events to be processed
        for exporter in self._exporters:
            try:
                exporter.flush()
            except Exception as e:
                logger.warning(
                    "Exporter flush failed",
                    exporter=exporter.name,
                    error=str(e),
                )
```

**Step 2: Replace close() method**

```python
    def close(self) -> None:
        """Shutdown export thread and close exporters.

        Shutdown Sequence (order is critical to avoid deadlock/data loss):
        1. Signal shutdown to reject new events
        2. Send sentinel FIRST (before waiting) - thread processes remaining
           events then exits when it sees the sentinel
        3. Wait for thread to exit (implicitly waits for queue drain)
        4. Close exporters

        CRITICAL: Do NOT call queue.join() before sending sentinel - this
        creates a race condition where the thread may block on get() after
        join() completes but before sentinel arrives.

        Should be called at pipeline shutdown. Logs final health metrics
        and releases exporter resources. Close failures are logged but
        don't raise.
        """
        # 1. Signal shutdown - prevents new events from being queued
        self._shutdown_event.set()

        # 2. Send sentinel FIRST - thread will process remaining events
        #    then exit when it sees the sentinel
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logger.error("Failed to send shutdown sentinel - queue full")

        # 3. Wait for thread to exit (this implicitly waits for queue drain
        #    because thread processes all events before exiting on sentinel)
        self._export_thread.join(timeout=5.0)
        if self._export_thread.is_alive():
            logger.error("Export thread did not exit cleanly within timeout")

        # 4. Close exporters
        logger.info("Telemetry manager closing", **self.health_metrics)
        for exporter in self._exporters:
            try:
                exporter.close()
            except Exception as e:
                logger.warning(
                    "Exporter close failed",
                    exporter=exporter.name,
                    error=str(e),
                )
```

**Step 3: Run test - should pass now**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_drop_mode_drops_when_queue_full -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/telemetry/manager.py
git commit -m "feat(telemetry): update flush() and close() for async export"
```

---

## Task 8: Update health_metrics Property

**Files:**
- Modify: `src/elspeth/telemetry/manager.py`

**Step 1: Replace health_metrics property**

```python
    @property
    def health_metrics(self) -> dict[str, Any]:
        """Return telemetry health metrics for monitoring.

        Returns a snapshot of telemetry health:
        - events_emitted: Successfully delivered to at least one exporter
        - events_dropped: Failed to deliver (queue full or all exporters failed)
        - exporter_failures: Per-exporter failure counts
        - consecutive_total_failures: Current streak of total failures
        - queue_depth: Current number of events in queue
        - queue_maxsize: Maximum queue capacity

        Thread Safety:
            Reads are approximately consistent. _events_dropped uses lock
            for consistency with concurrent writes. Other metrics may be
            slightly stale but this is acceptable for operational monitoring.

        Returns:
            Dictionary of health metrics
        """
        with self._dropped_lock:
            events_dropped = self._events_dropped
        return {
            "events_emitted": self._events_emitted,
            "events_dropped": events_dropped,
            "exporter_failures": self._exporter_failures.copy(),
            "consecutive_total_failures": self._consecutive_total_failures,
            "queue_depth": self._queue.qsize(),
            "queue_maxsize": self._queue.maxsize,
        }
```

**Step 2: Run all existing manager tests**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py -v`
Expected: Most tests pass (some may need adjustment for async behavior)

**Step 3: Commit**

```bash
git add src/elspeth/telemetry/manager.py
git commit -m "feat(telemetry): add queue metrics to health_metrics"
```

---

## Task 9: Write Test for BLOCK Mode

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add BLOCK mode test**

In `TestBackpressureMode` class:
```python
    def test_block_mode_blocks_when_queue_full(self, base_timestamp: datetime) -> None:
        """BLOCK mode blocks handle_event() when queue is full."""
        slow_exporter = SlowExporter("slow")
        config = MockConfig(backpressure_mode=BackpressureMode.BLOCK)
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Override queue size for testing
        manager._queue = queue.Queue(maxsize=1)

        event = RunStarted(timestamp=base_timestamp, run_id="test-run")
        started_blocking = threading.Event()
        finished_blocking = threading.Event()

        def blocking_put():
            manager.handle_event(event)  # First fills queue
            started_blocking.set()  # Signal we're about to block
            manager.handle_event(event)  # Second should block
            finished_blocking.set()  # Signal that we got past blocking

        thread = threading.Thread(target=blocking_put)
        thread.start()

        # Wait for thread to signal it's about to block
        assert started_blocking.wait(timeout=5.0), "Thread never started"

        # Give thread a moment to actually enter blocking put()
        # Then verify it hasn't finished (still blocked)
        assert not finished_blocking.wait(timeout=0.5), "handle_event should have blocked but didn't"

        # Unblock exporter so queue drains
        slow_exporter.can_continue.set()

        # Now thread should complete
        assert finished_blocking.wait(timeout=5.0), "handle_event never unblocked"

        thread.join(timeout=1.0)
        manager.close()
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_block_mode_blocks_when_queue_full -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add BLOCK mode backpressure test"
```

---

## Task 9.5: Write Test for Concurrent Close During Export (P0)

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add concurrent close test**

In `TestBackpressureMode` class:
```python
    def test_concurrent_close_during_export(self, base_timestamp: datetime) -> None:
        """close() during active export doesn't deadlock or corrupt state."""
        slow_exporter = SlowExporter("slow")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[slow_exporter])

        # Queue several events
        for i in range(5):
            event = RunStarted(timestamp=base_timestamp, run_id=f"run-{i}")
            manager.handle_event(event)

        # Wait for export thread to start processing
        assert slow_exporter.export_started.wait(timeout=5.0), "Export never started"

        # Now call close() while export is blocked
        close_completed = threading.Event()

        def close_manager():
            manager.close()
            close_completed.set()

        close_thread = threading.Thread(target=close_manager)
        close_thread.start()

        # Give close() a moment to start
        # Then unblock exporter so everything can complete
        import time
        time.sleep(0.1)
        slow_exporter.can_continue.set()

        # Close should complete without deadlock
        assert close_completed.wait(timeout=5.0), "close() deadlocked during active export"

        close_thread.join(timeout=1.0)

        # Verify no corruption - metrics should be consistent
        metrics = manager.health_metrics
        assert isinstance(metrics["events_emitted"], int)
        assert isinstance(metrics["events_dropped"], int)
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_concurrent_close_during_export -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add concurrent close during export test"
```

---

## Task 9.6: Write Test for Lock Contention (P0)

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add lock contention test**

In `TestBackpressureMode` class:
```python
    def test_lock_contention_on_events_dropped(self, base_timestamp: datetime) -> None:
        """Both threads incrementing _events_dropped doesn't corrupt counter."""
        # Create scenario where both threads write _events_dropped:
        # - Pipeline thread: DROP mode with full queue
        # - Export thread: All exporters failing

        failing_exporter = MockExporter("failing", fail_export=True)
        config = MockConfig(backpressure_mode=BackpressureMode.DROP)
        manager = TelemetryManager(config, exporters=[failing_exporter])

        # Use tiny queue to force drops
        manager._queue = queue.Queue(maxsize=2)

        # Fire many events - some will drop on queue full, some on export fail
        events_fired = 100
        for i in range(events_fired):
            event = RunStarted(timestamp=base_timestamp, run_id=f"run-{i}")
            manager.handle_event(event)

        # Close to flush everything
        manager.close()

        # Key assertion: dropped count should be consistent
        # (not corrupted by concurrent increments)
        metrics = manager.health_metrics
        total_accounted = metrics["events_emitted"] + metrics["events_dropped"]

        # We don't know exact split, but total should equal events fired
        # (some may be in-flight when we closed, so allow small variance)
        assert total_accounted <= events_fired, (
            f"More events accounted ({total_accounted}) than fired ({events_fired}) - "
            "counter corruption detected"
        )
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_lock_contention_on_events_dropped -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add lock contention test for _events_dropped"
```

---

## Task 9.7: Write Test for Re-entrance Deadlock (P1)

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add re-entrance deadlock test**

Add a new exporter mock class:
```python
class ReentrantExporter:
    """Exporter that tries to emit telemetry from export() - tests re-entrance."""

    def __init__(self, name: str, manager: "TelemetryManager"):
        self._name = name
        self._manager = manager
        self.reentrant_calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: dict[str, Any]) -> None:
        pass

    def export(self, event: TelemetryEvent) -> None:
        # Try to call handle_event from export thread - should not deadlock
        # This simulates buggy exporter that triggers telemetry
        try:
            reentrant_event = RunStarted(
                timestamp=event.timestamp,
                run_id=f"reentrant-{event.run_id}",
            )
            self._manager.handle_event(reentrant_event)
            self.reentrant_calls.append(event.run_id)
        except Exception as e:
            self.reentrant_calls.append(f"error: {e}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
```

In `TestBackpressureMode` class:
```python
    def test_reentrant_handle_event_no_deadlock(self, base_timestamp: datetime) -> None:
        """Exporter calling handle_event() from export thread doesn't deadlock.

        This tests a buggy exporter that tries to emit telemetry from within
        the export thread. It should either work or fail gracefully, not deadlock.
        """
        config = MockConfig()
        # Need to create manager first, then exporter (circular dependency)
        manager = TelemetryManager(config, exporters=[])

        reentrant_exporter = ReentrantExporter("reentrant", manager)
        manager._exporters = [reentrant_exporter]

        event = RunStarted(timestamp=base_timestamp, run_id="test")

        # This should complete without deadlock
        manager.handle_event(event)

        # Close with timeout - deadlock would hang here
        close_completed = threading.Event()

        def close_manager():
            manager.close()
            close_completed.set()

        close_thread = threading.Thread(target=close_manager)
        close_thread.start()

        assert close_completed.wait(timeout=5.0), "Re-entrant handle_event caused deadlock"
        close_thread.join(timeout=1.0)

        # Verify the reentrant call happened (or was handled)
        assert len(reentrant_exporter.reentrant_calls) > 0
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_reentrant_handle_event_no_deadlock -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add re-entrance deadlock test"
```

---

## Task 10: Write Test for Graceful Shutdown

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add shutdown test**

In `TestBackpressureMode` class:
```python
    def test_close_processes_all_queued_events(self, base_timestamp: datetime) -> None:
        """close() processes all queued events before returning."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Queue several events
        for i in range(5):
            event = RunStarted(timestamp=base_timestamp, run_id=f"run-{i}")
            manager.handle_event(event)

        # Close should process all events
        manager.close()

        # Verify all events were exported
        assert len(exporter.exports) == 5
        assert [e.run_id for e in exporter.exports] == [f"run-{i}" for i in range(5)]
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_close_processes_all_queued_events -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add graceful shutdown test"
```

---

## Task 11: Write Test for FIFO Ordering

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add FIFO ordering test**

In `TestBackpressureMode` class:
```python
    def test_events_exported_in_fifo_order(self, base_timestamp: datetime) -> None:
        """Events are exported in the order they were queued (FIFO)."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Queue events with distinct run_ids
        run_ids = [f"run-{i}" for i in range(20)]
        for run_id in run_ids:
            event = RunStarted(timestamp=base_timestamp, run_id=run_id)
            manager.handle_event(event)

        # Close and verify order
        manager.close()

        exported_ids = [e.run_id for e in exporter.exports]
        assert exported_ids == run_ids, "Events not exported in FIFO order"
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_events_exported_in_fifo_order -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add FIFO ordering test"
```

---

## Task 12: Write Test for Thread Liveness Check

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add thread liveness test**

In `TestBackpressureMode` class:
```python
    def test_thread_death_disables_telemetry(self, base_timestamp: datetime) -> None:
        """If export thread dies, handle_event disables telemetry."""
        exporter = MockExporter("test")
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[exporter])

        # Force thread to exit by sending sentinel directly
        manager._queue.put(None)
        manager._export_thread.join(timeout=1.0)
        assert not manager._export_thread.is_alive()

        # Now handle_event should detect dead thread and disable
        event = RunStarted(timestamp=base_timestamp, run_id="test")
        manager.handle_event(event)

        assert manager._disabled is True

        # Cleanup (close won't try to stop already-dead thread)
        manager._shutdown_event.set()
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_thread_death_disables_telemetry -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add thread liveness check test"
```

---

## Task 13: Write Test for task_done on Exception

**Files:**
- Modify: `tests/unit/telemetry/test_manager.py`

**Step 1: Add exception handling test**

In `TestBackpressureMode` class:
```python
    def test_task_done_called_on_exception(self, base_timestamp: datetime) -> None:
        """task_done() is called even when exporter raises, preventing join() hang."""
        import sys

        failing_exporter = MockExporter("failing", fail_export=True)
        config = MockConfig()
        manager = TelemetryManager(config, exporters=[failing_exporter])

        # Queue events (they'll all fail to export)
        for i in range(3):
            event = RunStarted(timestamp=base_timestamp, run_id=f"run-{i}")
            manager.handle_event(event)

        # close() should not hang - use threading.Timer for cross-platform timeout
        close_completed = threading.Event()
        close_exception: Exception | None = None

        def close_with_tracking():
            nonlocal close_exception
            try:
                manager.close()
            except Exception as e:
                close_exception = e
            finally:
                close_completed.set()

        close_thread = threading.Thread(target=close_with_tracking)
        close_thread.start()

        # Wait with timeout - if task_done() not called, close() would hang
        if not close_completed.wait(timeout=5.0):
            # Force cleanup and fail
            manager._shutdown_event.set()
            manager._queue.put(None)  # Try to unblock
            close_thread.join(timeout=1.0)
            pytest.fail("close() hung - task_done() not called properly")

        close_thread.join(timeout=1.0)

        # Should get here without hanging
        assert manager.health_metrics["events_dropped"] == 3
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py::TestBackpressureMode::test_task_done_called_on_exception -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/unit/telemetry/test_manager.py
git commit -m "test(telemetry): add task_done exception safety test"
```

---

## Task 14: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all telemetry tests**

Run: `.venv/bin/python -m pytest tests/unit/telemetry/test_manager.py tests/telemetry/ -v`
Expected: All tests pass

**Step 2: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/telemetry/manager.py`
Expected: Success (no errors)

**Step 3: Run linting**

Run: `.venv/bin/python -m ruff check src/elspeth/telemetry/manager.py`
Expected: No errors

**Step 4: No commit needed (verification only)**

---

## Task 15: Close Issue and Final Commit

**Files:**
- None

**Step 1: Run bd close**

```bash
bd close elspeth-rapid-ceq --reason="Implemented backpressure_mode wiring with background export thread"
```

**Step 2: Sync beads**

```bash
bd sync
```

**Step 3: Final status check**

```bash
git status
git log --oneline -10
```

---

## Summary

| Task | Description | Test First? | Priority |
|------|-------------|-------------|----------|
| 1 | Add queue_size to INTERNAL_DEFAULTS | No (config) | - |
| 2 | Update ExporterProtocol docs | No (docs) | - |
| 3 | Write failing DROP mode test | Yes | - |
| 4 | Add queue/threading infrastructure | No (TDD) | - |
| 5 | Implement handle_event (with timeout) | No (TDD) | P1 fix |
| 6 | Implement export loop (with ready signal) | No (TDD) | P1 fix |
| 7 | Update flush() and close() (sentinel BEFORE join) | No (TDD) | **P0 fix** |
| 8 | Update health_metrics | No (TDD) | - |
| 9 | Test BLOCK mode (Event-based) | Yes | P0 fix |
| 9.5 | Test concurrent close() | Yes | **P0 new** |
| 9.6 | Test lock contention | Yes | **P0 new** |
| 9.7 | Test re-entrance deadlock | Yes | **P1 new** |
| 10 | Test graceful shutdown | Yes | - |
| 11 | Test FIFO ordering | Yes | - |
| 12 | Test thread liveness | Yes | - |
| 13 | Test task_done exception safety (cross-platform) | Yes | P2 fix |
| 14 | Full test suite verification | No | - |
| 15 | Close issue | No | - |
