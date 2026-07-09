"""Graceful-shutdown signal handling for the orchestrator run loop.

This module owns the context manager that installs SIGINT/SIGTERM handlers
for the duration of a pipeline run, surfacing an :class:`threading.Event`
that the processing loop polls to stop cleanly after the current row.

It is a pure delegation target for the Orchestrator (same pattern as
aggregation.py / outcomes.py / graph_wiring.py): it holds no orchestrator
state and operates only on the process-global signal handlers and a freshly
created event.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def shutdown_handler_context() -> Iterator[threading.Event]:
    """Install SIGINT/SIGTERM handlers that set a shutdown event.

    On first signal: sets the event, restores default SIGINT/SIGTERM handlers
    (so a second signal force-kills instead of re-entering graceful handling).

    When called from a non-main thread (e.g., programmatic/embedded usage),
    signal registration is skipped — Python raises ValueError if
    signal.signal() is called outside the main thread.  The returned
    Event still works; it just won't be triggered by OS signals.

    Yields the Event for the processing loop to check.
    Restores original handlers in finally block (main thread only).
    """
    shutdown_event = threading.Event()

    # signal.signal() can only be called from the main thread.
    # In embedded/programmatic usage the orchestrator may run on a
    # worker thread — fall back to a plain event without handlers.
    if threading.current_thread() is not threading.main_thread():
        yield shutdown_event
        return

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _handler(signum: int, frame: Any) -> None:
        shutdown_event.set()
        # Restore explicit second-signal policies so repeated SIGINT/SIGTERM
        # escalates instead of re-entering graceful handling while draining.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    try:
        yield shutdown_event
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
