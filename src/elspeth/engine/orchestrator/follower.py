"""Follower-mode RowProcessor (ADR-030 §B.1, slice 5).

A follower is a cooperating elspeth process that attaches to a RUNNING run
(already led by another process) via ``elspeth join <run_id>``.  It claims
and advances READY scheduler work but never performs source ingest, barrier
trigger evaluation, checkpoint writes, or sink I/O — those are leader-only.

Construction
------------
Build via :func:`build_follower_processor`.  It wires the scheduler and
payload store but does NOT wire:

- source plugin (``source_plugin=None``) — follower has no source
- barrier restore (``barrier_restore=None``) — follower has no barrier memory
- coalesce executor (``coalesce_executor=None``) — barrier/coalesce plane is
  leader-only
- aggregation settings (``aggregation_settings={}``) — no trigger evaluation;
  instead ``follower_barrier_node_ids`` carries the aggregation node ID set so
  the processor intercepts batch-aware transforms at those nodes and calls
  ``mark_blocked`` rather than executing them row-wise (§B.2)

The ``ExecutionGraph`` is passed so the follower can recognise barrier and
sink nodes by their ``NodeID`` for hand-off routing.  Aggregation node IDs are
extracted via ``graph.get_aggregation_id_map()`` and threaded as
``follower_barrier_node_ids`` into :class:`RowProcessor`; no barrier executor
memory is needed.

Drain loop
----------
:meth:`FollowerProcessor.run` starts the heartbeat thread, then drives the
drain until one of four stop conditions:

``run_terminal``
    ``runs.status`` is not RUNNING.  The follower calls ``depart_worker``
    (CAS ``active → departed``, idempotent — finalize may have already done
    it) and exits 0.

``seat_dead``
    The heartbeat's coordination snapshot shows no live leader (or the thread
    latched ``coordination_lost_event``).  The follower finishes or abandons
    the current claim, takes no new claims after the grace period, then exits
    with a message naming ``elspeth resume``.

``SIGINT``
    Standard :exc:`KeyboardInterrupt` propagated from ``claim_ready``.
    Finish or abandon the current claim, depart, exit.

``evicted``
    The heartbeat thread latched its coordination-lost event
    (:class:`RunWorkerEvictedError`).  The exception propagates out of the
    drain loop; the caller receives it.

Follower dispositions (claim_ready only — never claim_pending_sink)
--------------------------------------------------------------------
For each claimed token the traversal runs the existing per-node processing
logic behind :meth:`RowProcessor.drain_follower_ready_work` — the processor's
explicit follower drain surface (its ``ProcessorMode.FOLLOWER`` contract owns
the claim_ready-only / no-pending-sink-recovery policy).  The four
disposition arms are handled by the standard RowProcessor machinery:

barrier node reached
    ``mark_blocked(barrier_key=…)`` — durable hold, **no in-memory accept**
    (§E); the traversal ends.

lossy coalesce fork-lineage token
    Durable branch-loss record written in the same lease-fenced transaction
    as the ``mark_failed``/divert (§E.5); the leader's next intake adopts it.

sink-bound
    ``mark_pending_sink`` with the full handoff bundle; the leader picks it
    up.  Followers **never perform sink I/O**.

otherwise
    ``mark_terminal`` / ``mark_failed``.

Child continuations use the idempotent ``enqueue_ready`` (now membership-
fenced with the follower's worker_id, task (e) in slice 5).

Per-worker JSONL journal
------------------------
FORENSIC-ONLY at N>1 (§C.4 row 13 / §F): per-statement timestamps are not
WAL commit order across processes.  Each worker appends to its own journal
file ``<db>.<worker_uuid>.jsonl``.  This module documents the doctrine but
does NOT create the journal file — journal paths are resolved by the
``LandscapeDB`` layer on open; this module records the path in the
coordination event for operator forensics.

Stop conditions and invariants are documented inline.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from elspeth.contracts.coordination import (
    DEFAULT_RUN_HEARTBEAT_SECONDS,
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.errors import FollowerSeatDeadError, RunWorkerEvictedError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread

if TYPE_CHECKING:
    from collections.abc import Callable

    from elspeth.contracts import RowResult
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
    from elspeth.engine.clock import Clock
    from elspeth.engine.orchestrator.types import PipelineConfig

__all__ = ["FollowerProcessor", "FollowerWorkSource", "build_follower_processor"]

logger = logging.getLogger(__name__)

# How long to wait between idle polls (no READY work found).
_IDLE_POLL_SECONDS: float = 2.0

# After no live leader is detected, allow up to this many seconds for the
# current claim to complete before taking no further claims.
_SEAT_DEAD_GRACE_SECONDS: float = 5.0


class FollowerWorkSource(Protocol):
    """The narrow processor surface the follower drain loop drives (duck-typed).

    One method: the explicit follower drain entry
    (:meth:`RowProcessor.drain_follower_ready_work`).  The follower-mode
    contract — ``claim_ready`` only, never ``claim_pending_sink`` /
    pending-sink recovery, no pre-claimed items (ADR-030 §B.1/§C.3) — lives
    behind that name on the processor's ``ProcessorMode.FOLLOWER``; this loop
    only threads its leader-liveness probe as ``before_claim``.
    """

    def drain_follower_ready_work(
        self,
        ctx: PluginContext,
        *,
        before_claim: Callable[[], None] | None = None,
    ) -> list[RowResult]: ...


class _SeatDeadError(Exception):
    """Internal signal: leader seat expired; follower should exit after grace."""

    def __init__(self, worker_id: str, run_id: str) -> None:
        super().__init__(f"Follower {worker_id!r} detected no live leader for run {run_id!r}; run can be taken over via `elspeth resume`")
        self.worker_id = worker_id
        self.run_id = run_id


class FollowerProcessor:
    """Follower-mode drain loop wrapping a claim-only :class:`RowProcessor`.

    Construction is via :func:`build_follower_processor`, which assembles the
    inner ``RowProcessor`` with the correct follower-mode parameters.  Direct
    construction is available for testing.

    Parameters
    ----------
    processor:
        An already-constructed follower-mode work source (production: a
        ``ProcessorMode.FOLLOWER`` :class:`RowProcessor` — no source, no
        barrier executors, no checkpoint coordinator). Only the narrow
        :class:`FollowerWorkSource` surface is driven.
    token:
        The follower's coordination token (worker_id + run_id, role=follower).
        Used to start the heartbeat thread and to call ``depart_worker`` on
        exit.
    run_coordination:
        :class:`RunCoordinationRepository` for ``depart_worker`` and run-status
        polling.
    factory:
        :class:`RecorderFactory` for run-status read-backs.
    clock:
        Optional clock injection for tests.
    heartbeat_seconds:
        Beat cadence for the heartbeat thread (default 15 s).
    window_seconds:
        Liveness window for the heartbeat (default 80 s).
    idle_poll_seconds:
        Sleep interval between idle polls (default 2 s).
    now_fn:
        Injectable ``datetime.now(UTC)`` supplier for tests.
    wait_fn:
        Injectable sleep function ``wait_fn(seconds)`` for tests.
    """

    def __init__(
        self,
        processor: FollowerWorkSource,
        *,
        token: CoordinationToken,
        run_coordination: RunCoordinationRepository,
        factory: RecorderFactory,
        clock: Clock | None = None,
        heartbeat_seconds: float = DEFAULT_RUN_HEARTBEAT_SECONDS,
        window_seconds: float = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
        idle_poll_seconds: float = _IDLE_POLL_SECONDS,
        now_fn: Callable[[], datetime] | None = None,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._processor = processor
        self._token = token
        self._run_coordination = run_coordination
        self._factory = factory
        self._heartbeat_seconds = heartbeat_seconds
        self._window_seconds = window_seconds
        self._idle_poll_seconds = idle_poll_seconds
        self._now_fn: Callable[[], datetime] = now_fn if now_fn is not None else lambda: datetime.now(UTC)
        self._wait_fn: Callable[[float], None] = wait_fn if wait_fn is not None else time.sleep
        from elspeth.engine.clock import DEFAULT_CLOCK

        self._clock = clock if clock is not None else DEFAULT_CLOCK

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, ctx: PluginContext) -> None:
        """Start the heartbeat thread and drive the follower drain loop.

        Returns normally when the run is terminal (depart + exit 0 semantics).
        Raises :class:`RunWorkerEvictedError` when the heartbeat thread detects
        eviction.  Propagates :class:`KeyboardInterrupt` after a clean depart.

        Stop conditions (design §B.1 steps 4-6):

        ``run_terminal``
            ``runs.status`` transitions out of RUNNING.  Depart (idempotent)
            and return.

        ``seat_dead``
            Leader seat expired (heartbeat snapshot or direct poll).  Finish or
            abandon the current claim, take no new claims, depart, return.

        ``evicted``
            Heartbeat thread latched coordination-lost.  Caller receives
            :class:`RunWorkerEvictedError` AFTER a best-effort depart attempt.

        ``SIGINT``
            :exc:`KeyboardInterrupt`.  Depart, re-raise (or return if the
            caller catches it).
        """
        heartbeat = RunHeartbeatThread(
            self._run_coordination,
            token=self._token,
            heartbeat_seconds=self._heartbeat_seconds,
            window_seconds=self._window_seconds,
            now_fn=self._now_fn,
        )
        heartbeat.start()
        try:
            self._drain_loop(ctx, heartbeat)
        except RunWorkerEvictedError:
            # The membership fence can raise mid-drain-pass when the leader's
            # complete_run departed our run_workers row in the SAME txn that
            # stamped the run terminal (finalize-departure, design case a).
            # The top-of-loop discrimination only covers the heartbeat-latch
            # path; recheck run status here before reporting a true eviction.
            if self._run_is_terminal():
                logger.info(
                    "follower %r: eviction raised mid-drain but run %r is terminal (finalize-departure) — exiting cleanly",
                    self._token.worker_id,
                    self._token.run_id,
                )
                self._best_effort_depart()
                return
            # True eviction: leader's housekeeping sweep evicted us while the
            # run is still RUNNING (design case b). Propagate (CLI exit 3).
            self._best_effort_depart()
            raise
        except KeyboardInterrupt:
            logger.info(
                "follower %r: SIGINT received — departing from run %r",
                self._token.worker_id,
                self._token.run_id,
            )
            self._best_effort_depart()
            raise
        except _SeatDeadError:
            # The leader seat expired while we were draining. Depart cleanly,
            # then raise FollowerSeatDeadError so the CLI can surface the
            # "use elspeth resume" guidance and exit with a distinct code
            # (design §B.1 step 5).
            self._best_effort_depart()
            raise FollowerSeatDeadError(
                worker_id=self._token.worker_id,
                run_id=self._token.run_id,
            ) from None
        else:
            # Clean terminal exit.
            self._best_effort_depart()
        finally:
            heartbeat.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _drain_loop(self, ctx: PluginContext, heartbeat: RunHeartbeatThread) -> None:
        """Drive claim_ready in a loop until a stop condition is reached.

        Raises:
            RunWorkerEvictedError: Propagated from heartbeat.check_and_raise().
            _SeatDeadError: No live leader detected.
            KeyboardInterrupt: SIGINT.
        """
        run_id = self._token.run_id
        worker_id = self._token.worker_id

        leader_recheck_interval = max(self._window_seconds / 4.0, 1.0)
        last_leader_check_monotonic: float | None = None

        def ensure_leader_live(*, force: bool = False) -> None:
            nonlocal last_leader_check_monotonic
            monotonic_now = time.monotonic()
            if (
                not force
                and last_leader_check_monotonic is not None
                and monotonic_now - last_leader_check_monotonic < leader_recheck_interval
            ):
                return
            last_leader_check_monotonic = monotonic_now
            now = self._now_fn()
            seat = self._run_coordination.live_leader(run_id=run_id, now=now)
            if seat is None or not seat.seat_live:
                raise _SeatDeadError(worker_id, run_id)

        def ensure_leader_live_before_claim() -> None:
            ensure_leader_live(force=True)

        while True:
            # Eviction / finalize-departure discrimination (design §B.1 step 5,
            # §D finalize flip).  The heartbeat latch is set on TWO distinct
            # events that differ in meaning:
            #
            #   (a) finalize-departure: complete_run flipped our run_workers row
            #       to 'departed' AND stamped runs.status as terminal in ONE
            #       IMMEDIATE txn.  worker_heartbeat returns worker_active=False
            #       because the row is no longer 'active'.  Correct exit: clean
            #       depart + return (exit 0).
            #
            #   (b) true eviction: the leader's §C.2 housekeeping sweep evicted
            #       this worker while the run is still RUNNING.  Correct exit:
            #       propagate RunWorkerEvictedError (exit 3 at the CLI).
            #
            # We cannot discriminate from the latch alone; read runs.status
            # first.  If the run is terminal → clean exit (case a).  Only then
            # propagate the eviction error (case b).
            if heartbeat.coordination_lost:
                if self._run_is_terminal():
                    logger.info(
                        "follower %r: heartbeat latch set + run %r is terminal (finalize-departure) — exiting cleanly",
                        worker_id,
                        run_id,
                    )
                    return
                heartbeat.check_and_raise()

            # ── check run status ──────────────────────────────────────
            if self._run_is_terminal():
                logger.info(
                    "follower %r: run %r is terminal — departing",
                    worker_id,
                    run_id,
                )
                return

            # ── check leader liveness ─────────────────────────────────
            ensure_leader_live()

            # ── attempt one claim-and-drain pass ──────────────────────
            # drain_follower_ready_work is the processor's EXPLICIT follower
            # drain surface (elspeth-577179bba1): it claims ALL currently
            # READY work until the queue is empty. The follower-mode contract
            # — claim_ready only, never pending-sink recovery (sink work is
            # leader-only), no pre-claimed items — is owned by the
            # processor's ProcessorMode.FOLLOWER, not by flag wiring here.
            # The outer loop handles idle/terminal/seat-dead between drain
            # passes.
            drained = self._processor.drain_follower_ready_work(
                ctx,
                before_claim=ensure_leader_live_before_claim,
            )

            if drained:
                # Work was found and advanced; loop immediately.
                logger.debug(
                    "follower %r: drained %d results from run %r",
                    worker_id,
                    len(drained),
                    run_id,
                )
                continue

            # ── idle: no READY work right now ─────────────────────────
            logger.debug(
                "follower %r: idle — no READY work in run %r; polling in %.1fs",
                worker_id,
                run_id,
                self._idle_poll_seconds,
            )
            self._wait_fn(self._idle_poll_seconds)

    def _run_is_terminal(self) -> bool:
        """Return True when the run's status is no longer RUNNING."""
        from elspeth.contracts.enums import RunStatus

        run = self._factory.run_lifecycle.get_run(self._token.run_id)
        if run is None:
            # Run disappeared — treat as terminal.
            return True
        return run.status != RunStatus.RUNNING

    def _best_effort_depart(self) -> None:
        """Call depart_worker, swallowing all exceptions (best-effort hygiene)."""
        try:
            self._run_coordination.depart_worker(
                worker_id=self._token.worker_id,
                now=self._now_fn(),
            )
        except Exception:
            logger.debug(
                "follower %r: best-effort depart_worker raised (idempotent — finalize may have already departed this row)",
                self._token.worker_id,
                exc_info=True,
            )


def build_follower_processor(
    *,
    factory: RecorderFactory,
    run_id: str,
    worker_id: str,
    graph: ExecutionGraph,
    config: PipelineConfig,
    payload_store: PayloadStore,
    clock: Clock | None = None,
    scheduler_lease_seconds: int = 300,
    scheduler_heartbeat_seconds: int = 60,
) -> FollowerProcessor:
    """Assemble a follower-mode :class:`FollowerProcessor`.

    The inner :class:`RowProcessor` is constructed through the SHARED builder
    (``processor_factory.build_row_processor``) with ``mode=ProcessorMode.FOLLOWER``
    (elspeth-577179bba1, absorbing elspeth-07b2031e41 part (b)) — previously a
    hand-assembled argument list that could drift from the leader/resume path.
    The builder's FOLLOWER gates own the follower-specific wiring:

    - ``source_plugin=None`` — no source ingest
    - ``barrier_restore=None`` — no barrier memory
    - ``coalesce_executor=None`` — no coalesce executor (leader-only plane;
      no settings.coalesce required even on a coalesce graph)
    - ``aggregation_settings={}`` — no trigger evaluation (barrier counting is
      leader-only per §B.2); ``follower_barrier_node_ids`` carries the
      aggregation node ID set so the processor intercepts batch-aware transforms
      at those nodes and calls ``mark_blocked`` rather than running them row-wise
    - ``run_coordination=None`` — followers do NOT run the §C.2 housekeeping
      eviction sweep (leader-only)
    - ``coordination_token=None`` — followers present no epoch fence (they
      only drive the item-layer CAS verbs, which fence on lease_owner)
    - ``scheduler_lease_owner=worker_id`` — the registered worker identity IS
      the scheduler lease_owner (§A.1); also threads the membership fence into
      ``enqueue_ready`` (task e, slice 5)

    The token/run_coordination Nones remain correct ABSENCES; the explicit
    mode flag, not their None-ness, now drives follower branch selection, and
    RowProcessor validates the combination fail-closed at construction.

    Args:
        factory: RecorderFactory bound to the run's audit DB.
        run_id: The run being joined.
        worker_id: The follower's registered worker identity (from
            :meth:`Orchestrator.join_run`).
        graph: The run's :class:`ExecutionGraph` (from the pipeline config).
        config: The run's :class:`PipelineConfig`.
        payload_store: PayloadStore for row payload persistence.
        clock: Optional clock injection for tests.
        scheduler_lease_seconds: Item lease TTL (default 300 s).
        scheduler_heartbeat_seconds: Item heartbeat cadence (default 60 s).

    Returns:
        A :class:`FollowerProcessor` ready to drive via :meth:`FollowerProcessor.run`.
    """
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.engine.orchestrator.graph_wiring import (
        assign_plugin_node_ids,
        build_source_id_map,
        load_edge_map,
    )
    from elspeth.engine.orchestrator.processor_factory import build_row_processor
    from elspeth.engine.scheduler_drain import ProcessorMode
    from elspeth.engine.spans import SpanFactory

    # Assign node_id to all plugin instances before building the traversal
    # context.  build_dag_traversal_context (inside build_row_processor)
    # requires transform.node_id to be set on every transform; the leader does
    # this via run_context_factory.py's assign_plugin_node_ids call, but the follower
    # path omitted it (slice 5 bug: build_follower_processor never called
    # assign_plugin_node_ids).
    #
    # source_id_map comes from the SAME loader the leader and resume use
    # (elspeth-07b2031e41 — the loop was previously copy-pasted per seam).
    source_id_map = build_source_id_map(graph)

    assign_plugin_node_ids(
        sources=config.sources,
        transforms=config.transforms,
        sinks=config.sinks,
        source_id_map=source_id_map,
        transform_id_map=graph.get_transform_id_map(),
        sink_id_map=graph.get_sink_id_map(),
        aggregation_node_ids=frozenset(graph.get_aggregation_id_map().values()),
    )

    # Source node (first source in the graph).  Followers need this to
    # construct the step_resolver closure but never call on_start/load/etc.
    sources = graph.get_sources()
    if not sources:
        raise ValueError(f"ExecutionGraph for run {run_id!r} has no source nodes")
    source_id = sources[0]

    # Load the edge_map through the shared loader (same helper as the resume
    # path) so GateExecutor has the correct edge_id for routing events.
    edge_map = load_edge_map(factory.data_flow, run_id)

    # Coordination token: followers carry their own token for the heartbeat
    # thread, but do NOT use it as an epoch fence (no leader-fenced verbs).
    # coordination_token=None keeps _require_coordination_token unreachable
    # and derives run_coordination=None (no §C.2 housekeeping sweep) — both
    # remain correct ABSENCES; ProcessorMode.FOLLOWER, not their None-ness,
    # drives follower branch selection, and RowProcessor validates the
    # combination fail-closed.
    #
    # The follower's run_workers row is required for the membership fence on
    # enqueue_ready / claim_ready — the fence is keyed on scheduler_lease_owner,
    # which equals worker_id here (§A.1).
    processor, _coalesce_node_map, _coalesce_executor = build_row_processor(
        graph=graph,
        config=config,
        settings=None,  # follower: no retry policy, no coalesce registration
        factory=factory,
        run_id=run_id,
        source_id=source_id,
        edge_map=edge_map,
        route_resolution_map=graph.get_route_resolution_map(),
        config_gate_id_map=graph.get_config_gate_id_map(),
        coalesce_id_map=graph.get_coalesce_id_map(),
        payload_store=payload_store,
        span_factory=SpanFactory(),
        clock=clock,
        max_workers=None,
        telemetry=None,
        mode=ProcessorMode.FOLLOWER,
        scheduler_lease_owner=worker_id,  # §A.1: registered identity = lease owner
        scheduler_lease_seconds=scheduler_lease_seconds,
        scheduler_heartbeat_seconds=scheduler_heartbeat_seconds,
        barrier_restore=None,  # follower: never restores barriers
        coordination_token=None,  # follower: no epoch fence
    )

    # Follower token for the heartbeat thread (worker_id, run_id, epoch=0
    # sentinel — followers have no epoch; heartbeat uses worker_id only).
    follower_token = CoordinationToken(run_id=run_id, worker_id=worker_id, leader_epoch=0)

    return FollowerProcessor(
        processor=processor,
        token=follower_token,
        run_coordination=factory.run_coordination,
        factory=factory,
        clock=clock,
    )
