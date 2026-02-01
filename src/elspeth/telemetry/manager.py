# src/elspeth/telemetry/manager.py
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

import queue
import threading
from typing import Any

import structlog

from elspeth.contracts.config import RuntimeTelemetryProtocol
from elspeth.contracts.config.defaults import INTERNAL_DEFAULTS
from elspeth.contracts.enums import BackpressureMode
from elspeth.contracts.events import TelemetryEvent
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.filtering import should_emit
from elspeth.telemetry.protocols import ExporterProtocol

logger = structlog.get_logger(__name__)


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

    _LOG_INTERVAL = 100  # Log every 100 total failures

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

        # Store exception for fail_on_total=True to re-raise on flush()
        self._stored_exception: TelemetryExporterError | None = None

        # Thread coordination
        self._shutdown_event = threading.Event()
        self._dropped_lock = threading.Lock()
        self._export_thread_ready = threading.Event()  # Signals thread is running

        # Queue for async export - read size from INTERNAL_DEFAULTS
        queue_size = int(INTERNAL_DEFAULTS["telemetry"]["queue_size"])
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
            try:
                if event is None:  # Shutdown sentinel
                    break
                self._dispatch_to_exporters(event)
            except TelemetryExporterError as e:
                # Store for re-raise on flush() when fail_on_total=True
                logger.error("Export loop failed unexpectedly", error=str(e))
                self._stored_exception = e
            except Exception as e:
                # CRITICAL: Log but don't crash - telemetry must not kill pipeline
                logger.error("Export loop failed unexpectedly", error=str(e))
            finally:
                # ALWAYS call task_done() to prevent join() hangs
                # This runs for ALL cases including sentinel (break still executes finally)
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
                self._exporter_failures[exporter.name] = self._exporter_failures.get(exporter.name, 0) + 1
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
                        f"All {len(self._exporters)} exporters failed {self._max_consecutive_failures} consecutive times.",
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

    def flush(self) -> None:
        """Wait for queue to drain, then flush exporters.

        Blocks until all queued events have been processed by the export
        thread, then flushes each exporter's internal buffer.

        Exporter flush failures are logged but don't raise - telemetry
        should not crash the pipeline.

        Raises:
            TelemetryExporterError: If fail_on_total_exporter_failure=True
                and all exporters failed repeatedly during processing.
        """
        if not self._shutdown_event.is_set():
            self._queue.join()  # Wait for all queued events to be processed

        # Re-raise stored exception from background thread (fail_on_total=True)
        if self._stored_exception is not None:
            exc = self._stored_exception
            self._stored_exception = None  # Clear to allow recovery
            raise exc

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
        #    then exit when it sees the sentinel.
        #
        #    CRITICAL: We MUST guarantee sentinel insertion. If queue is full,
        #    drain items until we can insert. Since shutdown is signaled, no
        #    new events will be queued, so draining is safe.
        sentinel_sent = False
        for _ in range(self._queue.maxsize + 10):  # +10 for safety margin
            try:
                self._queue.put(None, timeout=0.1)
                sentinel_sent = True
                break
            except queue.Full:
                # Queue full - drain one item and retry
                try:
                    discarded = self._queue.get_nowait()
                    self._queue.task_done()
                    if discarded is not None:
                        logger.debug(
                            "Discarded event during shutdown drain",
                            event_type=type(discarded).__name__,
                        )
                except queue.Empty:
                    # Queue became empty between put and get - try put again
                    pass

        if not sentinel_sent:
            logger.error("Failed to send shutdown sentinel after drain attempts - export thread may hang")

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
