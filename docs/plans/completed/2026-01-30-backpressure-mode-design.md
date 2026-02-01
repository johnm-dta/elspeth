# Backpressure Mode Implementation Design

**Date:** 2026-01-30
**Issue:** elspeth-rapid-ceq
**Status:** ✅ IMPLEMENTED (2026-02-01)
**Review:** Passed 4-perspective review (Architecture, Python, QA, Systems Thinking)

## Implementation Summary

- Design executed in `TelemetryManager` with queue-based BLOCK/DROP behavior (`src/elspeth/telemetry/manager.py`).
- Runtime config now validates and wires `backpressure_mode` with internal queue defaults (`src/elspeth/contracts/config/runtime.py`, `src/elspeth/contracts/config/defaults.py`).
- Behavior verified via telemetry manager and config tests (`tests/unit/telemetry/test_manager.py`, `tests/contracts/test_telemetry_config.py`).

## Problem Statement

The `backpressure_mode` configuration field is orphaned - validated in `TelemetrySettings` but never used at runtime. Users can configure `backpressure_mode: drop` expecting behavior changes, but the setting is silently ignored.

This is the P2-2026-01-21 config orphaning pattern: config fields that pass validation but have no effect at runtime.

## Solution Overview

Decouple event production from export via a bounded queue with a background export thread. The `backpressure_mode` setting controls queue behavior when full:

- **BLOCK:** `queue.put()` blocks until space available (pipeline slows)
- **DROP:** `queue.put_nowait()` drops event on `queue.Full` (pipeline unaffected)

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Pipeline Thread │────▶│ Queue        │────▶│ Export Thread   │
│                 │     │ maxsize=1000 │     │                 │
│ handle_event()  │     │              │     │ _export_loop()  │
│ - filter        │     │ BLOCK: wait  │     │ - get event     │
│ - enqueue       │     │ DROP: drop   │     │ - export to all │
│ - return fast   │     │              │     │ - track metrics │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

### Important: DROP Mode Semantics

**DROP mode is for BURST protection only, not sustained slow export tolerance.**

If exporters cannot keep up with sustained load, the correct response is to:
1. Fix/optimize the slow exporter
2. Reduce telemetry granularity
3. Disable problematic exporters

DROP mode absorbs temporary spikes; it does NOT make broken exporters acceptable.

## Detailed Design

### TelemetryManager Changes

```python
import queue
import threading
from elspeth.contracts.enums import BackpressureMode

class TelemetryManager:
    """Coordinates event emission to configured exporters.

    Thread Safety:
        After this change, TelemetryManager uses a background export thread.
        - handle_event() is called from the pipeline thread
        - _export_loop() runs in the background export thread
        - _events_dropped is protected by _dropped_lock (accessed from both threads)
        - All other metrics are only modified by the export thread
        - health_metrics reads are approximately consistent (see note below)
    """

    def __init__(self, config, exporters):
        # Existing initialization...
        self._config = config
        self._exporters = exporters

        # Thread coordination - use Event for thread-safe signaling
        self._shutdown_event = threading.Event()
        self._dropped_lock = threading.Lock()

        # Queue for async export
        self._queue: queue.Queue[TelemetryEvent | None] = queue.Queue(maxsize=1000)

        # Start export thread (non-daemon to ensure proper cleanup)
        self._export_thread = threading.Thread(
            target=self._export_loop,
            name="telemetry-export",
            daemon=False,  # Force proper cleanup via close()
        )
        self._export_thread.start()

    def handle_event(self, event: TelemetryEvent) -> None:
        """Queue event for async export.

        Thread Safety:
            Safe to call from any thread. Uses thread-safe Event check
            and Queue operations.
        """
        # Thread-safe shutdown check
        if self._shutdown_event.is_set():
            return
        if self._disabled:
            return
        if not self._exporters:
            return
        if not should_emit(event, self._config.granularity):
            return

        # Check thread liveness - if export thread died, disable telemetry
        if not self._export_thread.is_alive():
            logger.critical("Export thread died, disabling telemetry")
            self._disabled = True
            return

        if self._config.backpressure_mode == BackpressureMode.DROP:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._dropped_lock:
                    self._events_dropped += 1
                    self._log_drops_if_needed()
        else:  # BLOCK (default)
            self._queue.put(event)

    def _export_loop(self) -> None:
        """Background thread: consume queue and export.

        Thread Safety:
            Runs exclusively in the export thread. Metrics updates
            (except _events_dropped) are single-threaded.
        """
        while True:
            event = self._queue.get()
            if event is None:  # Shutdown sentinel
                self._queue.task_done()
                break
            try:
                self._dispatch_to_exporters(event)
            except Exception as e:
                # CRITICAL: Log but don't crash - telemetry must not kill pipeline
                logger.error("Export loop failed", error=str(e))
            finally:
                # ALWAYS call task_done() to prevent join() hangs
                self._queue.task_done()

    def _dispatch_to_exporters(self, event: TelemetryEvent) -> None:
        """Export to all exporters with failure isolation.

        Thread Safety:
            Called only from export thread.
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

        # Update metrics - all in export thread except _events_dropped
        if failures == 0:
            self._events_emitted += 1
            self._consecutive_total_failures = 0
        elif failures == len(self._exporters):
            self._consecutive_total_failures += 1
            # Lock required - pipeline thread also writes in DROP mode
            with self._dropped_lock:
                self._events_dropped += 1
            # ... existing failure handling (aggregate logging, disable check) ...
        else:
            self._events_emitted += 1
            self._consecutive_total_failures = 0

    @property
    def health_metrics(self) -> dict[str, Any]:
        """Return telemetry health metrics for monitoring.

        Thread Safety:
            Reads are approximately consistent. _events_dropped uses lock
            for consistency with concurrent writes. Other metrics may be
            slightly stale but this is acceptable for operational monitoring.
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

    def flush(self) -> None:
        """Wait for queue to drain, then flush exporters."""
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

    def close(self) -> None:
        """Shutdown export thread and close exporters.

        Shutdown Sequence (order is critical to avoid deadlock/data loss):
        1. Signal shutdown to reject new events
        2. Wait for pending events to process (queue.join)
        3. Send sentinel to exit thread
        4. Wait for thread to exit
        5. Close exporters
        """
        # 1. Signal shutdown - prevents new events from being queued
        self._shutdown_event.set()

        # 2. Wait for pending events to be processed
        self._queue.join()

        # 3. Send sentinel to stop export loop (with timeout to avoid deadlock)
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            logger.error("Failed to send shutdown sentinel - queue full after join")

        # 4. Wait for thread to exit
        self._export_thread.join(timeout=5.0)
        if self._export_thread.is_alive():
            logger.error("Export thread did not exit cleanly within timeout")

        # 5. Close exporters
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

### Thread Safety Summary

| Component | Access Pattern | Protection | Notes |
|-----------|----------------|------------|-------|
| `_shutdown_event` | Both threads | `threading.Event` (atomic) | Thread-safe by design |
| `_events_dropped` | Both threads | `_dropped_lock` | Pipeline (DROP) + Export (failure) |
| `_events_emitted` | Export thread only | None needed | Single writer |
| `_exporter_failures` | Export thread only | None needed | Single writer |
| `_consecutive_total_failures` | Export thread only | None needed | Single writer |
| `_queue` | Both threads | `queue.Queue` (thread-safe) | Built-in synchronization |
| `_disabled` | Both threads | Atomic bool read | Write only in export thread |

### Exporter Protocol Thread Safety Note

**IMPORTANT:** Exporters must now handle being called from a different thread than configure().

Add to `ExporterProtocol.export()` docstring:
```python
def export(self, event: "TelemetryEvent") -> None:
    """Export a single telemetry event.

    Thread Safety:
        export() is always called from the telemetry export thread, never
        concurrently with itself. However, export() may run on a different
        thread than configure() and close(). Implementations should not
        rely on thread-local state from configure().
    """
```

### Configuration

Queue size is an internal default (not user-configurable):

```python
# contracts/config/defaults.py
INTERNAL_DEFAULTS = {
    # ... existing ...
    "telemetry": {
        "queue_size": 1000,
    },
}
```

1000 events is generous for burst absorption. If tuning is needed, we can expose it later (YAGNI).

### Shutdown Sequence Diagram

```
Pipeline Thread                    Export Thread
      │                                  │
      │ close() called                   │
      ▼                                  │
 ┌────────────────┐                      │
 │ shutdown_event │                      │
 │    .set()      │                      │
 └───────┬────────┘                      │
         │                               │
         ▼                               │
 ┌────────────────┐                      │
 │  queue.join()  │◄─────────────────────┤ (processes remaining events)
 │  (blocks)      │                      │
 └───────┬────────┘                      │
         │ (all events processed)        │
         ▼                               │
 ┌────────────────┐                      │
 │ queue.put(None)│─────────────────────►│
 │  (sentinel)    │                      ▼
 └───────┬────────┘               ┌──────────────┐
         │                        │ Sees None,   │
         ▼                        │ breaks loop  │
 ┌────────────────┐               └──────┬───────┘
 │ thread.join()  │◄─────────────────────┘
 │  (waits)       │                (thread exits)
 └───────┬────────┘
         │
         ▼
 ┌────────────────┐
 │ Close exporters│
 └────────────────┘
```

### API Compatibility

| Method | Before | After |
|--------|--------|-------|
| `handle_event(event)` | Synchronous, blocks on export | Non-blocking (queues) |
| `flush()` | Flushes exporters | Drains queue + flushes exporters |
| `close()` | Closes exporters | Stops thread + drains + closes |
| `health_metrics` | Returns dict | Same + queue_depth, queue_maxsize |

## Files to Modify

| File | Changes |
|------|---------|
| `src/elspeth/telemetry/manager.py` | Add queue, background thread, backpressure logic |
| `src/elspeth/telemetry/protocols.py` | Document threading model in ExporterProtocol |
| `src/elspeth/contracts/config/defaults.py` | Add `telemetry.queue_size` internal default |
| `tests/telemetry/test_manager.py` | Add backpressure mode tests |

## Test Plan

### Unit Tests - Queue Mechanics

1. **Queue blocks on put when full (BLOCK mode)**
   ```python
   def test_queue_blocks_when_full_block_mode():
       # Fill queue, verify put() blocks, drain one, verify unblocks
       # Use threading.Event for deterministic coordination, not sleep()
   ```

2. **Queue drops on put_nowait when full (DROP mode)**
   ```python
   def test_queue_drops_when_full_drop_mode():
       # Fill queue, verify put_nowait raises Full, verify _events_dropped increments
   ```

3. **Sentinel not exported as event**
   ```python
   def test_sentinel_not_exported():
       # Verify None sentinel doesn't reach exporters
   ```

### Unit Tests - Thread Safety

4. **_events_dropped increment is atomic under contention**
   ```python
   def test_events_dropped_atomic_under_contention():
       # Multiple threads incrementing, verify no lost increments
   ```

5. **health_metrics readable during concurrent drops**
   ```python
   def test_health_metrics_consistent_during_drops():
       # Read metrics while drops happening, verify no torn reads
   ```

6. **Thread liveness check disables telemetry**
   ```python
   def test_thread_death_disables_telemetry():
       # Kill export thread, verify handle_event sets _disabled=True
   ```

### Unit Tests - Shutdown

7. **Graceful shutdown processes all queued events**
   ```python
   def test_close_processes_all_queued_events():
       # Queue events, close(), verify all exported
   ```

8. **Shutdown respects timeout**
   ```python
   def test_close_respects_timeout_with_slow_exporter():
       # Slow exporter, close(), verify returns within timeout+buffer
   ```

9. **task_done always called (even on exception)**
   ```python
   def test_task_done_called_on_exception():
       # Exporter raises, verify task_done still called, join() doesn't hang
   ```

### Unit Tests - Re-entrance Protection

10. **Export thread cannot cause re-entrance**
    ```python
    def test_no_reentrance_from_export_thread():
        # Exporter calls handle_event(), verify bounded/blocked
    ```

### Unit Tests - Concurrent Operations

11. **Concurrent close() during active export**
    ```python
    def test_concurrent_close_during_export():
        # Export in flight, close() called, verify clean shutdown
    ```

12. **Lock not held during export (no pipeline blocking)**
    ```python
    def test_lock_not_held_during_export():
        # Verify _dropped_lock released before export() call
    ```

### Integration Tests

13. **Config wiring - DROP mode**
    ```python
    def test_config_drop_mode_wired():
        # Configure backpressure_mode: drop, verify behavior
    ```

14. **Config wiring - BLOCK mode**
    ```python
    def test_config_block_mode_wired():
        # Configure backpressure_mode: block, verify behavior
    ```

### Property-Based Tests

15. **Event ordering preserved (FIFO)**
    ```python
    @given(event_count=st.integers(10, 100))
    def test_fifo_ordering_preserved(event_count):
        # Queue N events, verify exported in same order
    ```

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Thread leak if close() not called | Medium | Medium | Non-daemon thread forces proper cleanup; add atexit handler |
| Lost events on crash | Medium | Low | Best-effort; telemetry is ephemeral (for bursts, not audit) |
| Metrics race on _events_dropped | Low | Low | Lock protects all writes from both threads |
| Deadlock in close() | Low | High | queue.join() BEFORE sentinel; timeout on thread.join() |
| Export thread dies silently | Low | Medium | Liveness check in handle_event() disables + logs CRITICAL |
| Re-entrance from exporter | Low | Medium | shutdown_event check prevents infinite loop |

## Systemic Considerations

### Archetype Warning: Fixes that Fail

This design addresses the symptom (slow exports blocking pipeline) but not the root cause (broken/slow exporters). To prevent the "Fixes that Fail" pattern:

1. **Make exporter health visible** - Added `queue_depth` and `queue_maxsize` to health_metrics
2. **Consider fail_on_total_exporter_failure=True as default** - Crash on broken telemetry aligns with ELSPETH philosophy
3. **Document that DROP is for bursts** - Not a license to tolerate broken exporters

### Observability of Telemetry Itself

| Metric | What it tells you |
|--------|-------------------|
| `queue_depth > 0.8 * maxsize` | Exporters can't keep up - investigate |
| `events_dropped > 0` | Bursts exceeded buffer OR exporters are broken |
| `consecutive_total_failures > 0` | ALL exporters failing - urgent |

## Alternatives Considered

### A. Timeout-based inline export
Keep synchronous export but timeout after N ms in DROP mode. Rejected because you can't timeout a synchronous HTTP call mid-flight without async.

### B. Remove DROP mode entirely
Simplest option - just don't support it. Rejected because the config already claims to support it (`_IMPLEMENTED_BACKPRESSURE_MODES` includes DROP).

### C. Use BoundedBuffer for backpressure
The existing `BoundedBuffer` is for exporter-level batching, not manager-level backpressure. Using it would conflate two concerns.

### D. Daemon thread
Simpler (exits with process), but loses events on abrupt shutdown. Rejected in favor of non-daemon + proper close() to maximize event delivery.

## Implementation Notes

- Use `threading.Event` for shutdown coordination (atomic, supports `is_set()`)
- Use `queue.Queue` from stdlib (thread-safe, blocking semantics built-in)
- Thread should be **non-daemon** to force proper cleanup via close()
- Use `queue.task_done()` / `queue.join()` for flush semantics
- Sentinel value `None` signals shutdown (distinct from any real event)
- Shutdown sequence: `join()` → sentinel → thread join (order prevents deadlock)
- Always call `task_done()` in finally block to prevent `join()` hangs
