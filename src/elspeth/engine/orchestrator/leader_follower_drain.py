"""LeaderFollowerDrain: bounded peer-lease wait + follower pending-sink drain.

Extracted from ``Orchestrator._execute_run`` (filigree elspeth-6630fb3e31). The
fresh-run method used to inline two near-duplicate bespoke poll / latch / reap /
timeout / sleep loops:

1. **Peer-lease wait** — before the unresolved-scheduler-work invariant fires,
   wait (bounded) for peer followers to finish any in-flight LEASED items so a
   just-claimed row (LEASED, ``pending_sink_name IS NULL``) doesn't trip the
   invariant. In the single-worker case ``has_peer_active_leases()`` is False
   immediately and the loop is skipped.
2. **Follower pending-sink drain** — after the leader's own sink writes, re-drain
   the PENDING_SINK rows produced by follower workers until both no peer holds an
   in-flight lease and no undrained PENDING_SINK row remains.

Both are now ``wait_for_peer_leases()`` and ``drain_pending_sink_work()`` here.
The node_states-UNIQUE-constraint-critical coupling (clear pending_tokens →
accumulate → flush) stays in the caller's ``drain_and_flush`` closure; this
coordinator owns only the bounded-poll mechanics.

Behaviour-preserving. Invariants held exactly:
  - per-iteration order: shutdown → latch → (drain) → reap → recheck → timeout → sleep;
  - timeout BREAKS (never raises) in both loops;
  - a productive drain iteration ``continue``\\ s without sleeping;
  - the ``slog.warning`` payloads (run_id, still_leased_peers, waited_seconds,
    has_scheduled_work) are unchanged. The event text is preserved; only the
    emitting logger name moves from ``…orchestrator.core`` to this module.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

import structlog

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

if TYPE_CHECKING:
    import threading

    from elspeth.contracts.errors import GracefulShutdownError

slog = structlog.get_logger(__name__)


class PeerLeaseProcessor(Protocol):
    """The narrow processor surface the drain coordinator polls (duck-typed)."""

    def peer_lease_wait_budget_seconds(self) -> float: ...

    def has_peer_active_leases(self) -> bool: ...

    def has_scheduled_work(self) -> bool: ...

    def reap_expired_peer_leases(self) -> int: ...

    def peer_active_lease_owners(self) -> Sequence[str]: ...


class LeaderFollowerDrain:
    """Bounded peer-lease wait and follower pending-sink drain for a fresh run.

    All wall-clock and processor seams are injected so the loops are unit-testable
    with a fake processor and fake ``monotonic`` / ``sleep``. ``make_shutdown_error``
    is a zero-arg callable (NOT a pre-built exception) so it reads the run's live
    counters at raise time, matching the source loop / sink-flush shutdown path.
    """

    def __init__(
        self,
        *,
        processor: PeerLeaseProcessor,
        run_id: str,
        shutdown_event: threading.Event | None,
        check_coordination_latch: Callable[[], None] | None,
        make_shutdown_error: Callable[[], GracefulShutdownError],
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_interval: float = 0.5,
    ) -> None:
        self._processor = processor
        self._run_id = run_id
        self._shutdown_event = shutdown_event
        self._check_coordination_latch = check_coordination_latch
        self._make_shutdown_error = make_shutdown_error
        # Resolve the wall-clock seams HERE (construction time), NOT as import-time
        # default args: the fresh-run call site constructs this inside the run, so a
        # test that patches ``…orchestrator.core.time.monotonic`` / ``.sleep`` (the
        # shared time module) around the run is honoured — matching the pre-extraction
        # inline loops, which resolved ``time.monotonic()`` / ``time.sleep()`` at call
        # time. Unit tests inject fakes explicitly.
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._poll_interval = poll_interval

    def wait_for_peer_leases(self) -> None:
        """Wait (bounded) for peer followers to finish in-flight LEASED items.

        Bounded by ``peer_lease_wait_budget_seconds()``. Each iteration honours
        shutdown (SIGINT) and the coordination latch (epoch deposition), and
        reaps a dead peer's expired lease to READY within the liveness window.
        On timeout it warns and returns — the caller's unresolved-work invariant
        raise (which names the still-leased peers) is the fall-through.
        """
        peer_wait_seconds = self._processor.peer_lease_wait_budget_seconds()
        peer_wait_deadline = self._monotonic() + peer_wait_seconds
        while self._processor.has_peer_active_leases():
            # SIGINT during the wait: surface the graceful-shutdown path
            # rather than spinning.
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                raise self._make_shutdown_error()
            # Epoch deposition during the wait: check_and_raise surfaces
            # RunWorkerEvictedError so the deposed leader runs its INTERRUPTED
            # ceremony instead of spinning.
            if self._check_coordination_latch is not None:
                self._check_coordination_latch()
            # Actively reap a dead peer's expired lease (recovers it to READY
            # within the liveness window).
            self._processor.reap_expired_peer_leases()
            if not self._processor.has_peer_active_leases():
                break
            if self._monotonic() >= peer_wait_deadline:
                still_leased = self._processor.peer_active_lease_owners()
                slog.warning(
                    "Bounded peer-lease wait timed out; falling through to the unresolved-work invariant",
                    run_id=self._run_id,
                    still_leased_peers=list(still_leased),
                    waited_seconds=peer_wait_seconds,
                )
                break
            self._sleep(self._poll_interval)

    def drain_pending_sink_work(self, drain_and_flush: Callable[[], bool]) -> None:
        """Re-drain follower PENDING_SINK rows until both conditions are false.

        ``drain_and_flush`` performs the caller-owned drain→clear→accumulate→flush
        block and returns True when it made progress (follower results were
        drained and flushed) so the loop re-checks immediately without sleeping.
        Bounded by ``3 x DEFAULT_RUN_LIVENESS_WINDOW_SECONDS``; on timeout it warns
        and returns, relying on complete_run's quiescence backstop (a residual
        PENDING_SINK row fails the run loudly — the resumable exactly-once
        fail-direction — never a silent lost row).
        """
        drain_deadline = self._monotonic() + (3.0 * DEFAULT_RUN_LIVENESS_WINDOW_SECONDS)
        while self._processor.has_peer_active_leases() or self._processor.has_scheduled_work():
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                raise self._make_shutdown_error()
            if self._check_coordination_latch is not None:
                self._check_coordination_latch()

            if self._processor.has_scheduled_work() and drain_and_flush():
                # Made progress this iteration; re-check immediately.
                continue

            # No drainable work this pass but a peer still holds a lease that
            # could yet produce a PENDING_SINK row.  Actively reap a dead peer,
            # then wait briefly — bounded.
            if not (self._processor.has_peer_active_leases() or self._processor.has_scheduled_work()):
                break
            self._processor.reap_expired_peer_leases()
            if not (self._processor.has_peer_active_leases() or self._processor.has_scheduled_work()):
                break
            if self._monotonic() >= drain_deadline:
                slog.warning(
                    "Bounded follower-drain wait timed out; relying on complete_run quiescence backstop",
                    run_id=self._run_id,
                    still_leased_peers=list(self._processor.peer_active_lease_owners()),
                    has_scheduled_work=self._processor.has_scheduled_work(),
                    waited_seconds=3.0 * DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
                )
                break
            self._sleep(self._poll_interval)
