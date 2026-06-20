"""Dedicated run-level heartbeat thread (ADR-030 §A.3, slice 4).

Each run worker (currently only the N=1 leader) starts one
:class:`RunHeartbeatThread` after the coordination seat is minted and joins
it (via :meth:`RunHeartbeatThread.stop`) before every ``release_seat`` call.
The thread shares ZERO Python state with the ``RowProcessor`` — it
communicates through two :class:`threading.Event` flags:

- ``_stop_event``: set by the owner to signal the thread to exit; also used
  as the clock/sleep primitive via ``_stop_event.wait(interval)`` so that unit
  tests can inject a fast-forwarding ``wait_fn`` without real wall-clock sleeps.
- ``_coordination_lost_event``: set by the thread when ``worker_heartbeat``
  returns ``worker_active=False`` (this worker's registry row left ``active``)
  OR when the snapshot's ``leader_worker_id`` differs from our own worker_id
  (deposed — another process took the seat). The drain loop raises
  :class:`~elspeth.contracts.errors.RunWorkerEvictedError` at the next
  boundary by polling :meth:`check_and_raise`.

Design invariants enforced here:

- **BUSY = liveness-unknown** — a heartbeat ``OperationalError`` (SQLite
  busy-timeout) is logged at DEBUG and counted toward the ``heartbeat_degraded``
  threshold ``k``; the thread never sets the latch on a DB error.
- **Never self-terminate on DB errors** — the per-tick try/except swallows all
  exceptions and continues looping; only a deliberate ``_stop_event.set()``
  exits the loop.
- **Both rows in one transaction** — ``worker_heartbeat`` (the repository
  verb) updates ``run_workers`` and, for the leader, ``run_coordination`` in a
  single BEGIN IMMEDIATE transaction; the two liveness clocks cannot skew in
  the worker-fresher-than-seat direction.
- **Deterministic testability** — inject ``now_fn`` (defaults to
  ``datetime.now(UTC)``) and ``wait_fn`` (defaults to ``stop_event.wait``);
  unit tests step the beat with ``_step_beat()`` and an injected
  ``wait_fn=lambda _: False`` to drive ticks without sleeping.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from elspeth.contracts.coordination import (
    DEFAULT_RUN_HEARTBEAT_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.errors import RunWorkerEvictedError

__all__ = ["RunHeartbeatThread"]

logger = logging.getLogger(__name__)

# After this many consecutive BUSY failures (DB error ticks) the thread emits
# a ``heartbeat_degraded`` coordination event so a later eviction is
# diagnosable as "could not reach the DB" rather than "process died".
_DEFAULT_DEGRADED_THRESHOLD: int = 3


class RunHeartbeatThread:
    """Daemon thread that beats the run-worker (and, for leaders, the seat) row.

    Lifecycle::

        thread = RunHeartbeatThread(repo, token=token, now_fn=..., ...)
        thread.start()
        try:
            # run body — call thread.check_and_raise() at every boundary
            ...
        finally:
            thread.stop()   # blocks until the thread exits

    Parameters
    ----------
    repo:
        :class:`~elspeth.core.landscape.run_coordination_repository.RunCoordinationRepository`
        instance backed by the run's engine.  Passed as ``Any`` to avoid a
        circular import — the caller has the real type.
    token:
        The leader's coordination token (worker_id + run_id); used to detect
        seat deposition (snapshot ``leader_worker_id`` ≠ our ``worker_id``).
    heartbeat_seconds:
        Beat cadence; defaults to
        :data:`~elspeth.contracts.coordination.DEFAULT_RUN_HEARTBEAT_SECONDS`
        (15 s).
    window_seconds:
        Liveness window passed to ``worker_heartbeat``; defaults to
        :data:`~elspeth.contracts.coordination.DEFAULT_RUN_LIVENESS_WINDOW_SECONDS`
        (80 s).
    now_fn:
        Callable returning the current UTC datetime.  Inject a fixed-clock
        callable in unit tests.
    wait_fn:
        Called as ``wait_fn(interval_seconds) -> bool`` (same signature as
        :meth:`threading.Event.wait`); returns ``True`` when the stop event
        was signalled.  Defaults to ``_stop_event.wait``.  Inject
        ``lambda _: False`` in unit tests to step beats synchronously.
    degraded_threshold:
        Number of consecutive busy failures before a ``heartbeat_degraded``
        event is recorded.
    """

    def __init__(
        self,
        repo: object,
        *,
        token: CoordinationToken,
        heartbeat_seconds: float = DEFAULT_RUN_HEARTBEAT_SECONDS,
        window_seconds: float = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        now_fn: Callable[[], datetime] | None = None,
        wait_fn: Callable[[float], bool] | None = None,
        degraded_threshold: int = _DEFAULT_DEGRADED_THRESHOLD,
    ) -> None:
        self._repo = repo
        self._token = token
        self._heartbeat_seconds = heartbeat_seconds
        self._window_seconds = window_seconds
        self._now_fn: Callable[[], datetime] = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._degraded_threshold = degraded_threshold

        self._stop_event = threading.Event()
        # wait_fn defaults to _stop_event.wait after construction so that
        # injected wait_fn overrides are honoured while still allowing the
        # default binding to reference the instance's own Event.
        self._wait_fn: Callable[[float], bool] = wait_fn if wait_fn is not None else self._stop_event.wait

        self._coordination_lost_event = threading.Event()
        # Stores the eviction reason for the error message raised at the
        # boundary; written by the heartbeat thread, read by check_and_raise.
        self._coordination_lost_reason: str = "worker registry row left 'active' (evicted or departed)"

        self._thread = threading.Thread(target=self._run, daemon=True, name=f"run-heartbeat:{token.run_id[:8]}")
        self._consecutive_busy: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to stop and block until it exits.

        Must be called (in a ``finally`` block) before every ``release_seat``
        call so that the thread cannot beat the seat after the release vacates
        it.  Idempotent — safe to call more than once.
        """
        self._stop_event.set()
        self._thread.join()

    @property
    def coordination_lost(self) -> bool:
        """True when the heartbeat latch is set (worker row left 'active').

        Read-only view for callers that need to test the flag WITHOUT raising
        (e.g. the follower drain loop discriminates between a finalize-departure
        clean exit and a true eviction before propagating
        :class:`~elspeth.contracts.errors.RunWorkerEvictedError`).
        """
        return self._coordination_lost_event.is_set()

    def check_and_raise(self) -> None:
        """Raise :class:`~elspeth.contracts.errors.RunWorkerEvictedError` if the latch is set.

        Call this at every claim/node boundary in the drain loop.  The latch
        is set when ``worker_heartbeat`` returns ``worker_active=False`` or the
        snapshot reveals seat deposition; the thread itself never raises on the
        drain thread — it only sets the flag and lets the drain raise cleanly at
        its next checkpoint.
        """
        if self._coordination_lost_event.is_set():
            raise RunWorkerEvictedError(
                worker_id=self._token.worker_id,
                run_id=self._token.run_id,
            )

    # ------------------------------------------------------------------
    # Internal — the beat loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Thread entry point: beat loop, exits only when stop_event is set."""
        while not self._wait_fn(self._heartbeat_seconds):
            self._beat_once()
        # Final beat on exit: keeps the seat live until release_seat is called
        # (stop() is called in the finally block just before release_seat).
        # Best-effort — never raises.
        self._beat_once()

    def _beat_once(self) -> None:
        """Execute one heartbeat tick; NEVER raises.

        Three outcomes:
        1. ``worker_active=True``, and (if leader) snapshot
           ``leader_worker_id==our_id`` → healthy; reset busy counter.
        2. ``worker_active=False`` → coordination lost; latch the flag.
           For a LEADER ONLY: foreign ``leader_worker_id`` (deposed) also
           latches.  A follower seeing a foreign leader is the NORMAL case
           (the follower is never the leader); follower deposed-latch is
           triggered only by ``worker_active=False`` (eviction/departure).
        3. Any exception (including SQLITE_BUSY ``OperationalError``) →
           liveness-unknown; log, count toward degraded threshold, continue.

        ADR-030 §B: ``snapshot.worker_role`` discriminates leader vs follower
        so that the deposed-latch is role-gated.  The field defaults to
        "leader" for backward-compat with tests that construct snapshots
        without the field.
        """
        try:
            snapshot = self._repo.worker_heartbeat(  # type: ignore[attr-defined]
                worker_id=self._token.worker_id,
                now=self._now_fn(),
                window_seconds=self._window_seconds,
            )
            # Reset busy counter on any successful DB round-trip.
            self._consecutive_busy = 0

            # LATCH: evicted/departed — our registry row left 'active'.
            # Applies to BOTH leaders and followers: if the run_workers row is
            # no longer active the worker must stop (evicted by leader, or
            # departed at finalize).
            if not snapshot.worker_active:
                logger.warning(
                    "run_heartbeat: worker %r row left 'active' in run %r (evicted or departed)",
                    self._token.worker_id,
                    self._token.run_id,
                )
                self._coordination_lost_reason = (
                    f"worker {self._token.worker_id!r} registry row left 'active' (evicted or departed at finalize)"
                )
                self._coordination_lost_event.set()
                return

            # LATCH: deposed — seat taken by another process.
            # LEADER ONLY: a leader seeing a foreign leader_worker_id means it
            # has been deposed and must stop.  A FOLLOWER seeing a foreign
            # leader_worker_id is the NORMAL, HEALTHY case (§B.2: trigger
            # evaluation is leader-only; the follower always has a different
            # leader_worker_id from its own worker_id) and must NOT latch.
            is_leader = snapshot.worker_role == "leader"
            if is_leader and snapshot.leader_worker_id is not None and snapshot.leader_worker_id != self._token.worker_id:
                logger.warning(
                    "run_heartbeat: seat taken by %r (our worker_id=%r) in run %r",
                    snapshot.leader_worker_id,
                    self._token.worker_id,
                    self._token.run_id,
                )
                self._coordination_lost_reason = f"seat taken by {snapshot.leader_worker_id!r} (our worker_id={self._token.worker_id!r})"
                self._coordination_lost_event.set()
                return

        except Exception as exc:
            # BUSY = liveness-unknown (design §A.3): count toward degraded
            # threshold but never crash and never set the latch.
            self._consecutive_busy += 1
            logger.debug(
                "run_heartbeat: busy/error for worker %r in run %r (consecutive=%d): %s",
                self._token.worker_id,
                self._token.run_id,
                self._consecutive_busy,
                exc,
            )
            if self._consecutive_busy >= self._degraded_threshold:
                self._emit_degraded()

    def _emit_degraded(self) -> None:
        """Emit ``heartbeat_degraded`` event; best-effort, never raises."""
        with contextlib.suppress(Exception):
            self._repo.record_heartbeat_degraded(  # type: ignore[attr-defined]
                run_id=self._token.run_id,
                worker_id=self._token.worker_id,
                failures=self._consecutive_busy,
                now=self._now_fn(),
            )

    # ------------------------------------------------------------------
    # Test seam: step a single beat synchronously
    # ------------------------------------------------------------------

    def _step_beat(self) -> None:
        """Execute exactly one ``_beat_once()`` call on the caller's thread.

        FOR UNIT TESTS ONLY.  Do NOT call from the drain loop.
        """
        self._beat_once()
