"""IdleTimeoutPump: one long-lived idle-flush worker per source-iteration run.

Extracted from ``SourceIterationDriver._next_source_item_with_idle_timeout_flushes``
(filigree elspeth-735df9576d). The driver used to construct, start, and join a
fresh daemon ``threading.Thread`` around EVERY source ``next()`` call in timed
aggregation/coalesce pipelines — one thread lifecycle per row at a 10ms poll
interval — and relied on the implicit fact that the orchestrator thread was
blocked inside ``next()`` for the poller's whole lifetime (with ``Thread.join``
as the only happens-before edge) to serialize the helper thread's mutations of
shared loop state (``loop_ctx.counters`` / ``loop_ctx.pending_tokens``).

This pump replaces that with ONE persistent worker thread per source-iteration
run and an explicit condition-variable handshake. Behaviour-preservation
invariants held exactly:

- **Iterator advancement stays on the caller thread.** ``fetch()`` runs the
  supplied callable on the calling (orchestrator) thread; only the idle-flush
  closure runs on the worker (pinned by ``ThreadAffinitySource`` in
  ``tests/integration/pipeline/orchestrator/test_execution_loop.py``).
- **Flushes fire WHILE the caller is blocked in the fetch.** The documented
  timeout contract requires idle flushes to execute during ``next()`` — a
  blocked source generator may depend on the flush to make progress
  (``IdleBlockingSource``; ``tests/unit/docs/test_timeout_contract_docs.py``).
- **One full poll interval elapses before the first flush**, and a fetch that
  completes sooner sees zero flushes (mirrors the old
  ``while not stop_event.wait(interval)`` loop shape).
- **First error wins and the worker parks permanently.** A flush failure is
  recorded once and stops all further flushing until ``stop()`` (mirrors the
  old one-slot error queue + stop-on-error).
- **Error precedence on ``fetch()``:** recorded flush error > fetch-callable
  exception > ``StopIteration`` — identical to the historical arbitration
  order.
- **The end-of-fetch handshake reproduces ``poller.join()``.** ``fetch()``
  does not return until the worker has parked (any in-flight flush has
  completed), re-establishing the happens-before edge that makes every
  cross-thread mutation of loop state visible to — and serialized against —
  the orchestrator thread before it touches that state again.

The flush closure is invoked with the condition lock RELEASED so a
long-running flush never blocks ``stop()``/state changes from being
requested; the park handshake alone provides the serialization.

The pump knows nothing about LoopContext or the row processor: it takes a
zero-arg ``flush`` closure and a ``poll_interval``, and ``fetch()`` is generic
over the fetch callable's return type. Single-caller contract: ``start()``,
every ``fetch()``, and ``stop()`` are called from one orchestrator thread;
overlapping ``fetch()`` calls are an invariant violation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from elspeth.contracts.errors import OrchestrationInvariantError

_T = TypeVar("_T")


class IdleTimeoutPump:
    """Persistent idle-flush worker with a locked fetch/park handshake.

    While a ``fetch()`` is in flight the worker invokes ``flush`` every
    ``poll_interval`` seconds; between fetches (and after ``stop()`` or a
    flush error) it parks and touches nothing.
    """

    def __init__(self, *, flush: Callable[[], None], poll_interval: float) -> None:
        self._flush = flush
        self._poll_interval = poll_interval
        self._cond = threading.Condition()
        # --- State guarded by _cond ---
        self._fetch_active = False
        self._stopped = False
        # The worker is "parked" whenever it is guaranteed not to run another
        # flush until reactivated by a later fetch(). True initially so a
        # degenerate fetch against a never-started pump cannot hang.
        self._parked = True
        self._error: BaseException | None = None
        # --- Worker lifecycle (orchestrator thread only) ---
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        """Start the persistent worker thread. Must be called exactly once."""
        if self._worker is not None:
            raise OrchestrationInvariantError("IdleTimeoutPump.start() called more than once.")
        self._worker = threading.Thread(target=self._run, name="idle-timeout-pump", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        """Stop and join the worker. Idempotent; safe on a never-started pump."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._worker is not None:
            self._worker.join()

    def fetch(self, fn: Callable[[], _T]) -> _T:
        """Run ``fn`` on the calling thread while the worker flushes on idle.

        Captures the ``fn`` result / ``StopIteration`` / exception trichotomy,
        then performs the join-equivalent handshake (deactivate polling, wait
        until the worker has parked) before arbitrating errors in the
        historical order: recorded flush error, then the ``fn`` exception,
        then ``StopIteration``.
        """
        with self._cond:
            if self._worker is None:
                raise OrchestrationInvariantError("IdleTimeoutPump.fetch() called before start().")
            if self._stopped:
                raise OrchestrationInvariantError("IdleTimeoutPump.fetch() called after stop().")
            if self._fetch_active:
                raise OrchestrationInvariantError("IdleTimeoutPump.fetch() re-entered while a fetch is in flight.")
            self._fetch_active = True
            self._cond.notify_all()

        result: list[_T] = []
        fetch_stop = False
        fetch_exc: BaseException | None = None
        try:
            try:
                result.append(fn())
            except StopIteration:
                # Re-raised below AFTER the park handshake and error
                # arbitration — exhaustion must not skip serialization.
                fetch_stop = True
            except BaseException as exc:
                # Captured (not swallowed) so the park handshake still runs;
                # re-raised below unless a flush error takes precedence.
                fetch_exc = exc
        finally:
            with self._cond:
                self._fetch_active = False
                self._cond.notify_all()
                # Join-equivalence: do not hand control back to the caller
                # until the worker has parked (no flush is mid-flight). This
                # is the happens-before edge for the flush closure's
                # mutations of shared loop state.
                while not self._parked:
                    self._cond.wait()
                idle_error = self._error

        if idle_error is not None:
            raise idle_error
        if fetch_exc is not None:
            raise fetch_exc
        if fetch_stop:
            raise StopIteration
        if not result:
            raise OrchestrationInvariantError("IdleTimeoutPump.fetch() callable neither returned nor raised.")
        return result[0]

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                # Parked: wait until a fetch activates polling. A recorded
                # error latches the pump parked until stop() — mirrors the
                # old per-fetch poller's stop-on-error behaviour.
                while not self._stopped and not (self._fetch_active and self._error is None):
                    self._cond.wait()
                if self._stopped:
                    return
                self._parked = False
            self._poll_until_fetch_completes()

    def _poll_until_fetch_completes(self) -> None:
        while True:
            with self._cond:
                # Checked BEFORE the wait so a fetch that ended while a flush
                # was running parks immediately instead of waiting a full
                # interval; and AFTER the wait for state changes during it.
                if self._should_park():
                    self._park_and_notify()
                    return
                self._cond.wait(timeout=self._poll_interval)
                if self._should_park():
                    self._park_and_notify()
                    return
            try:
                # Lock released: the flush may be slow and must not block
                # stop()/fetch-end requests from being RECORDED (the park
                # handshake still serializes their completion behind it).
                self._flush()
            except BaseException as exc:
                # Cross-thread propagation, not suppression: first error wins,
                # is re-raised on the orchestrator thread at the end of the
                # owning fetch(), and parks the worker permanently.
                with self._cond:
                    if self._error is None:
                        self._error = exc
                    self._park_and_notify()
                return

    def _should_park(self) -> bool:
        """Caller must hold ``_cond``."""
        return self._stopped or not self._fetch_active or self._error is not None

    def _park_and_notify(self) -> None:
        """Caller must hold ``_cond``."""
        self._parked = True
        self._cond.notify_all()
