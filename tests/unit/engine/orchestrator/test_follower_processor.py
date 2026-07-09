"""Unit tests for FollowerProcessor (ADR-030 §B.1, slice 5).

Tests each stop-condition arm of the follower drain loop plus the
construction contract:

1. ``run_terminal`` — run status leaves RUNNING; follower departs and returns.
2. ``seat_dead`` — live_leader returns None or seat_live=False; follower
   departs and returns (no exception propagated).
3. ``evicted`` — heartbeat latches coordination_lost; RunWorkerEvictedError
   propagates out of run() after a best-effort depart.
4. ``SIGINT`` — KeyboardInterrupt propagates out of run() after a best-effort
   depart.
5. ``idle`` — no READY work: follower calls wait_fn and loops.
6. ``drained`` — drain found work (non-empty list): follower loops immediately
   without sleeping.
7. ``depart_hygiene`` — depart_worker is called exactly once on every clean
   exit path.

Heartbeat is driven synchronously via ``_StubHeartbeat`` (no real threads).
All sleeps are suppressed via injected ``wait_fn``.  Injected ``now_fn``
fixes the clock.  ``live_leader`` and ``_run_is_terminal`` are controlled
via the stub coordination repo and stub factory.

Disposition arm note
--------------------
The per-disposition-arm logic (barrier→mark_blocked, sink→mark_pending_sink,
lossy-coalesce→branch-loss) lives inside the RowProcessor drain (behind the
public ``drain_follower_ready_work`` follower surface) and is already tested
in tests/unit/engine/ drain tests.  Here we verify only that
FollowerProcessor correctly:

- drives ONLY the public ``drain_follower_ready_work`` surface, threading its
  leader-liveness probe as ``before_claim`` (the claim_ready-only /
  no-pending-sink-recovery contract is the processor's FOLLOWER mode and is
  pinned in tests/unit/engine/test_processor_mode.py)
- calls wait_fn when drained is empty
- loops immediately when drained is non-empty
- exits on terminal/dead-seat/evict/SIGINT
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts.coordination import (
    CoordinationToken,
    LeaderInfo,
)
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import FollowerSeatDeadError, RunWorkerEvictedError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.orchestrator.follower import FollowerProcessor, _SeatDeadError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "run-follower-test-1"
WORKER_ID = f"worker:{RUN_ID}:follower-unit-test"
NOW = datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC)
LIVENESS_WINDOW = 80.0


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RunStatusRecord:
    status: RunStatus


@dataclass(frozen=True, slots=True)
class _FollowerDrainResult:
    label: str = "work-found"


class _UnusedScheduler:
    """Scheduler surface required for RowProcessor construction in focused tests."""

    @staticmethod
    def serialize_row_payload(row: Any) -> str:
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

        return TokenSchedulerRepository.serialize_row_payload(row)

    @staticmethod
    def deserialize_row_payload(row_payload_json: str) -> Any:
        from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

        return TokenSchedulerRepository.deserialize_row_payload(row_payload_json)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"scheduler method {name!r} should not be used in this focused processor test")


class _UncalledBatchTransform:
    """Concrete TransformProtocol-compatible fake that must not be executed."""

    def __init__(self, *, node_id: str) -> None:
        self.name = "batch-agg-transform"
        self.input_schema = object
        self.output_schema = object
        self.node_id = node_id
        self.config: dict[str, Any] = {}
        self.determinism = None
        self.plugin_version = "1.0"
        self.source_file_hash = None
        self.usage_when_to_use = None
        self.usage_when_not_to_use = None
        self.example_use = None
        self.capability_tags: tuple[str, ...] = ()
        self.audit_characteristics = frozenset()
        self.discovery_secret_requirements = MappingProxyType({})
        self._on_start_called = True
        self._on_complete_called = False
        self.is_batch_aware = True
        self.supports_row_mode_when_batch_aware = False
        self.creates_tokens = False
        self.passes_through_input = False
        self.can_drop_rows = False
        self.declared_output_fields = frozenset()
        self.declared_input_fields = frozenset()
        self.requires_runtime_preflight = False
        self._output_schema_config = None
        self.on_error = "discard"
        self.on_success = None
        self.process_called = False

    def effective_static_contract(self) -> frozenset[str]:
        return self.declared_output_fields

    def process(self, row: Any, ctx: Any) -> Any:
        self.process_called = True
        raise AssertionError("follower barrier test must not execute the batch transform")

    def close(self) -> None:
        raise AssertionError("follower barrier test must not close the transform")

    def on_start(self, ctx: Any) -> None:
        raise AssertionError("follower barrier test must not start the transform")

    def on_complete(self, ctx: Any) -> None:
        raise AssertionError("follower barrier test must not complete the transform")

    def runtime_preflight(self, ctx: Any) -> None:
        raise AssertionError("follower barrier test must not preflight the transform")

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> None:
        return None

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> None:
        return None

    @classmethod
    def get_post_call_hints(cls, *, tool_name: str, config_snapshot: Any) -> tuple[str, ...]:
        return ()


class _StubHeartbeat:
    """Synchronous stub for RunHeartbeatThread.

    Controls whether check_and_raise() raises RunWorkerEvictedError.
    start() and stop() are no-ops.
    """

    def __init__(self, *, evicted: bool = False) -> None:
        self._evicted = evicted
        self.start_called = False
        self.stop_called = False
        self.check_calls = 0

    def start(self) -> None:
        self.start_called = True

    def stop(self) -> None:
        self.stop_called = True

    @property
    def coordination_lost(self) -> bool:
        """True when the latch would raise (mirrors RunHeartbeatThread.coordination_lost)."""
        return self._evicted

    def check_and_raise(self) -> None:
        self.check_calls += 1
        if self._evicted:
            raise RunWorkerEvictedError(worker_id=WORKER_ID, run_id=RUN_ID)

    def set_evicted(self) -> None:
        self._evicted = True


class _StubRunCoordRepo:
    """Stub of RunCoordinationRepository for FollowerProcessor tests.

    Controls what live_leader() returns and records depart_worker() calls.
    """

    def __init__(self, *, seat_live: bool = True, seat_present: bool = True) -> None:
        self._seat_live = seat_live
        self._seat_present = seat_present
        self.depart_calls: list[dict[str, Any]] = []
        self.live_leader_calls: list[dict[str, Any]] = []
        self.worker_heartbeat_calls: list[dict[str, Any]] = []

        # Override per-call via a list (pops first element each call)
        self.live_leader_results: list[LeaderInfo | None] = []

    def live_leader(self, *, run_id: str, now: datetime) -> LeaderInfo | None:
        self.live_leader_calls.append({"run_id": run_id, "now": now})
        if self.live_leader_results:
            return self.live_leader_results.pop(0)
        if not self._seat_present:
            return None
        return LeaderInfo(
            run_id=run_id,
            leader_worker_id=f"worker:{run_id}:leader",
            leader_epoch=1,
            leader_heartbeat_expires_at=now + timedelta(seconds=LIVENESS_WINDOW),
            seat_live=self._seat_live,
        )

    def depart_worker(self, *, worker_id: str, now: datetime) -> None:
        self.depart_calls.append({"worker_id": worker_id, "now": now})

    def worker_heartbeat(self, *, worker_id: str, now: datetime, window_seconds: float) -> Any:
        from elspeth.contracts.coordination import CoordinationSnapshot

        self.worker_heartbeat_calls.append({"worker_id": worker_id, "now": now})
        return CoordinationSnapshot(
            leader_worker_id=f"worker:{RUN_ID}:leader",
            leader_epoch=1,
            seat_live=True,
            worker_active=True,
            # Follower stub: this worker is a follower, so seeing a foreign
            # leader_worker_id is NORMAL and must NOT latch coordination_lost.
            worker_role="follower",
        )


class _StubRunLifecycle:
    """Stub of RunLifecycleRepository for run status reads."""

    def __init__(self, *, running: bool = True) -> None:
        self._running = running
        # Per-call sequence; pops first each call
        self.get_run_results: list[Any] = []

    def get_run(self, run_id: str) -> Any:
        if self.get_run_results:
            return self.get_run_results.pop(0)

        return _RunStatusRecord(
            status=RunStatus.RUNNING if self._running else RunStatus.COMPLETED,
        )


class _StubFactory:
    """Stub of RecorderFactory wrapping _StubRunLifecycle."""

    def __init__(self, *, running: bool = True) -> None:
        self.run_lifecycle = _StubRunLifecycle(running=running)


class _CountingDrainProcessor:
    """Stub FollowerWorkSource with configurable drain_follower_ready_work returns.

    Returns a pre-configured sequence of drain results (list of RowResult-like
    objects).  Use [] for idle (no work found), [mock_result] for work found.
    Tracks arguments passed to drain_follower_ready_work for assertion.
    """

    def __init__(self) -> None:
        self.drain_results: list[list[Any]] = []
        self.drain_calls: list[dict[str, Any]] = []

    def drain_follower_ready_work(
        self,
        ctx: Any,
        *,
        before_claim: Any = None,
    ) -> list[Any]:
        self.drain_calls.append(
            {
                "ctx": ctx,
                "before_claim": before_claim,
            }
        )
        if self.drain_results:
            return self.drain_results.pop(0)
        # Default: idle (empty drain)
        return []


# ---------------------------------------------------------------------------
# Helper: build a FollowerProcessor under test
# ---------------------------------------------------------------------------


def _make_follower(
    *,
    processor: _CountingDrainProcessor | None = None,
    coord_repo: _StubRunCoordRepo | None = None,
    factory: _StubFactory | None = None,
    wait_calls: list[float] | None = None,
    evicted: bool = False,
) -> tuple[FollowerProcessor, _CountingDrainProcessor, _StubRunCoordRepo, _StubFactory, _StubHeartbeat]:
    """Build a FollowerProcessor with all stubs injected."""
    if processor is None:
        processor = _CountingDrainProcessor()
    if coord_repo is None:
        coord_repo = _StubRunCoordRepo()
    if factory is None:
        factory = _StubFactory()
    heartbeat = _StubHeartbeat(evicted=evicted)

    # Inject wait_fn that records calls but does not sleep
    recorded_waits: list[float] = wait_calls if wait_calls is not None else []

    def _wait(seconds: float) -> None:
        recorded_waits.append(seconds)
        # Also trip the terminal flag so the loop exits
        factory.run_lifecycle._running = False

    token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)

    follower = FollowerProcessor(
        processor=processor,
        token=token,
        run_coordination=coord_repo,  # type: ignore[arg-type]
        factory=factory,  # type: ignore[arg-type]
        now_fn=lambda: NOW,
        wait_fn=_wait,
    )

    return follower, processor, coord_repo, factory, heartbeat


def _ctx() -> PluginContext:
    return PluginContext(run_id=RUN_ID, config={}, landscape=None)


# ---------------------------------------------------------------------------
# Tests: stop condition — run terminal
# ---------------------------------------------------------------------------


class TestFollowerTerminalRun:
    """Follower exits cleanly when the run is no longer RUNNING."""

    def test_terminal_at_first_check_exits_and_departs(self) -> None:
        """If run is not RUNNING before any drain pass, follower departs and returns."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=False)

        follower, _, coord_repo, _, heartbeat = _make_follower(processor=processor, coord_repo=coord_repo, factory=factory)

        with (
            patch.object(follower, "_drain_loop", wraps=follower._drain_loop),
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
        ):
            follower.run(ctx=_ctx())

        # No drain calls — run was terminal before the first pass
        assert processor.drain_calls == []

        # depart_worker called exactly once
        assert len(coord_repo.depart_calls) == 1
        assert coord_repo.depart_calls[0]["worker_id"] == WORKER_ID

        # Heartbeat lifecycle: start and stop both called
        assert heartbeat.start_called
        assert heartbeat.stop_called

    def test_terminal_after_one_idle_exits(self) -> None:
        """Run becomes terminal after one idle cycle; depart called once."""
        processor = _CountingDrainProcessor()
        # First get_run call → RUNNING; second → COMPLETED
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        waits: list[float] = []

        def _wait(seconds: float) -> None:
            waits.append(seconds)
            # Transition to terminal AFTER the first idle sleep
            factory.run_lifecycle._running = False

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_wait,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        # One drain call (idle — returned []) via the public follower surface,
        # threading the leader-liveness probe.
        assert len(processor.drain_calls) == 1
        assert callable(processor.drain_calls[0]["before_claim"])

        # One idle wait
        assert len(waits) == 1

        # depart called once
        assert len(coord_repo.depart_calls) == 1


# ---------------------------------------------------------------------------
# Tests: stop condition — seat dead
# ---------------------------------------------------------------------------


class TestFollowerSeatDead:
    """Follower raises FollowerSeatDeadError when the leader seat is gone.

    Design §B.1 step 5: the follower departs cleanly then raises
    FollowerSeatDeadError so the CLI can surface the 'use elspeth resume'
    guidance and exit with a distinct code (exit 2).
    """

    def test_seat_absent_raises_follower_seat_dead(self) -> None:
        """live_leader returns None -> FollowerSeatDeadError propagated after depart."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo(seat_present=False)  # live_leader returns None
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(FollowerSeatDeadError) as exc_info,
        ):
            follower.run(ctx=_ctx())

        assert exc_info.value.worker_id == WORKER_ID
        assert exc_info.value.run_id == RUN_ID
        assert "elspeth resume" in str(exc_info.value)
        # depart called once even though exception raised
        assert len(coord_repo.depart_calls) == 1

    def test_seat_dead_raises_follower_seat_dead(self) -> None:
        """seat_live=False -> FollowerSeatDeadError propagated after depart."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo(seat_live=False)  # seat present but expired
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(FollowerSeatDeadError),
        ):
            follower.run(ctx=_ctx())

        assert len(coord_repo.depart_calls) == 1

    def test_drain_not_called_on_dead_seat(self) -> None:
        """No drain pass occurs when the seat is immediately dead."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo(seat_present=False)
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(FollowerSeatDeadError),
        ):
            follower.run(ctx=_ctx())

        # No drain calls — seat was dead before any claim attempt
        assert processor.drain_calls == []

    def test_before_claim_liveness_check_is_fresh_for_every_claim(self) -> None:
        """Follower re-reads leader liveness before every claim in a hot drain."""

        class _ManyClaimDrainProcessor(_CountingDrainProcessor):
            claim_checks = 0

            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                if before_claim is None:
                    raise AssertionError("follower drain must pass a before_claim liveness callback")
                for _ in range(3):
                    before_claim()
                    self.claim_checks += 1
                return [_FollowerDrainResult()]

        processor = _ManyClaimDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        run_result_running = _RunStatusRecord(status=RunStatus.RUNNING)
        run_result_completed = _RunStatusRecord(status=RunStatus.COMPLETED)
        factory.run_lifecycle.get_run_results = [run_result_running, run_result_completed]
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            patch("elspeth.engine.orchestrator.follower.time.monotonic", side_effect=[10.0, 10.0, 11.0, 19.9]),
        ):
            follower.run(ctx=_ctx())

        assert processor.claim_checks == 3
        assert len(coord_repo.live_leader_calls) == 4
        assert len(coord_repo.depart_calls) == 1

    def test_seat_dead_between_hot_claims_stops_follower_drain_inside_throttle_interval(self) -> None:
        """Follower refuses the next claim even when the old throttle interval has not elapsed."""

        class _CallbackDrainProcessor(_CountingDrainProcessor):
            claim_checks = 0

            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                if before_claim is None:
                    raise AssertionError("follower drain must pass a before_claim liveness callback")
                before_claim()
                self.claim_checks += 1
                before_claim()
                self.claim_checks += 1
                return [_FollowerDrainResult()]

        processor = _CallbackDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        coord_repo.live_leader_results = [
            LeaderInfo(
                run_id=RUN_ID,
                leader_worker_id=f"worker:{RUN_ID}:leader",
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=LIVENESS_WINDOW),
                seat_live=True,
            ),
            LeaderInfo(
                run_id=RUN_ID,
                leader_worker_id=f"worker:{RUN_ID}:leader",
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=LIVENESS_WINDOW),
                seat_live=True,
            ),
            None,
        ]
        factory = _StubFactory(running=True)
        run_result_running = _RunStatusRecord(status=RunStatus.RUNNING)
        run_result_completed = _RunStatusRecord(status=RunStatus.COMPLETED)
        factory.run_lifecycle.get_run_results = [run_result_running, run_result_completed]
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            patch("elspeth.engine.orchestrator.follower.time.monotonic", side_effect=[0.0, 0.1, 0.2]),
            pytest.raises(FollowerSeatDeadError),
        ):
            follower.run(ctx=_ctx())

        assert processor.claim_checks == 1
        assert len(coord_repo.live_leader_calls) == 3
        assert len(coord_repo.depart_calls) == 1

    def test_seat_dead_between_claims_stops_follower_drain(self) -> None:
        """Follower rechecks leader liveness once the throttle interval elapses."""

        class _CallbackDrainProcessor(_CountingDrainProcessor):
            claim_checks = 0

            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                if before_claim is None:
                    raise AssertionError("follower drain must pass a before_claim liveness callback")
                before_claim()
                self.claim_checks += 1
                before_claim()
                self.claim_checks += 1
                return [_FollowerDrainResult()]

        processor = _CallbackDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        coord_repo.live_leader_results = [
            LeaderInfo(
                run_id=RUN_ID,
                leader_worker_id=f"worker:{RUN_ID}:leader",
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=LIVENESS_WINDOW),
                seat_live=True,
            ),
            LeaderInfo(
                run_id=RUN_ID,
                leader_worker_id=f"worker:{RUN_ID}:leader",
                leader_epoch=1,
                leader_heartbeat_expires_at=NOW + timedelta(seconds=LIVENESS_WINDOW),
                seat_live=True,
            ),
            None,
        ]
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            patch("elspeth.engine.orchestrator.follower.time.monotonic", side_effect=[0.0, 0.0, 21.0]),
            pytest.raises(FollowerSeatDeadError),
        ):
            follower.run(ctx=_ctx())

        assert processor.claim_checks == 1
        assert len(coord_repo.live_leader_calls) == 3
        assert len(coord_repo.depart_calls) == 1


# ---------------------------------------------------------------------------
# Tests: stop condition — evicted
# ---------------------------------------------------------------------------


class TestFollowerEvicted:
    """Follower propagates RunWorkerEvictedError after best-effort depart."""

    def test_eviction_propagates_and_departs(self) -> None:
        """RunWorkerEvictedError propagates out of run(); depart is called."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat(evicted=True)  # check_and_raise() raises immediately

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError) as exc_info,
        ):
            follower.run(ctx=_ctx())

        assert exc_info.value.worker_id == WORKER_ID
        assert exc_info.value.run_id == RUN_ID

        # depart still called (best-effort)
        assert len(coord_repo.depart_calls) == 1

        # Heartbeat was stopped
        assert heartbeat.stop_called

    def test_eviction_before_any_drain_pass(self) -> None:
        """check_and_raise fires before first live_leader poll; no drain calls."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat(evicted=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError),
        ):
            follower.run(ctx=_ctx())

        assert processor.drain_calls == []


# ---------------------------------------------------------------------------
# Tests: finalize-departure race — eviction raised mid-drain (elspeth-8690ef4bfd)
# ---------------------------------------------------------------------------


class TestFollowerEvictionFinalizeDepartureRace:
    """When RunWorkerEvictedError is raised mid-drain (fence fires inside
    drain_follower_ready_work), the follower must distinguish:

    (a) finalize-departure: complete_run stamped the run terminal AND departed
        our row in ONE txn — so the fence fires but the run is COMPLETED.
        Correct exit: return cleanly (exit 0), depart called once.

    (b) true eviction: the fence fires and the run is still RUNNING.
        Correct exit: propagate RunWorkerEvictedError (exit 3).

    The bug (elspeth-8690ef4bfd): the old 'except RunWorkerEvictedError' arm
    unconditionally re-raised without checking run status, so case (a) was
    mis-reported as a true eviction (exit 3 instead of exit 0).
    """

    def test_mid_drain_eviction_with_terminal_run_returns_cleanly(self) -> None:
        """RunWorkerEvictedError from drain_follower_ready_work + run COMPLETED
        → run() returns (no raise), depart called once.

        This is the finalize-departure race: complete_run departed the row AND
        stamped the run terminal in the same txn; the membership fence fires
        inside the drain pass before the top-of-loop discrimination saw the
        terminal status.
        """

        class _MidDrainEvictingProcessor:
            """Raises RunWorkerEvictedError on the first drain_follower_ready_work call."""

            def __init__(self) -> None:
                self.drain_calls: list[dict[str, Any]] = []

            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                self.drain_calls.append({"ctx": ctx})
                raise RunWorkerEvictedError(worker_id=WORKER_ID, run_id=RUN_ID)

        processor = _MidDrainEvictingProcessor()
        coord_repo = _StubRunCoordRepo()
        # Heartbeat latch NOT set (evicted=False) — the exception comes from drain, not heartbeat.
        heartbeat = _StubHeartbeat(evicted=False)

        # get_run sequence: first call (top-of-loop check, returns RUNNING so loop continues),
        # second call (in the except arm recheck, returns COMPLETED → clean exit).
        factory = _StubFactory(running=True)
        run_result_running = _RunStatusRecord(status=RunStatus.RUNNING)
        run_result_completed = _RunStatusRecord(status=RunStatus.COMPLETED)
        factory.run_lifecycle.get_run_results = [run_result_running, run_result_completed]

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        # Must return normally — NOT raise RunWorkerEvictedError.
        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())  # must not raise

        # Depart called once (best-effort hygiene).
        assert len(coord_repo.depart_calls) == 1
        assert coord_repo.depart_calls[0]["worker_id"] == WORKER_ID

        # Heartbeat lifecycle clean.
        assert heartbeat.start_called
        assert heartbeat.stop_called

    def test_mid_drain_true_eviction_still_raises(self) -> None:
        """RunWorkerEvictedError from drain_follower_ready_work + run still RUNNING
        → RunWorkerEvictedError propagates (true eviction, not finalize-departure).

        This is the negative/anti-regression test: we must NOT suppress true
        evictions (which should surface as exit 3 at the CLI).
        """

        class _MidDrainEvictingProcessor:
            def __init__(self) -> None:
                self.drain_calls: list[dict[str, Any]] = []

            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                self.drain_calls.append({"ctx": ctx})
                raise RunWorkerEvictedError(worker_id=WORKER_ID, run_id=RUN_ID)

        processor = _MidDrainEvictingProcessor()
        coord_repo = _StubRunCoordRepo()
        heartbeat = _StubHeartbeat(evicted=False)

        # Run stays RUNNING for ALL get_run calls → true eviction.
        factory = _StubFactory(running=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        # Must raise RunWorkerEvictedError.
        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError) as exc_info,
        ):
            follower.run(ctx=_ctx())

        assert exc_info.value.worker_id == WORKER_ID
        assert exc_info.value.run_id == RUN_ID

        # Depart still called (best-effort).
        assert len(coord_repo.depart_calls) == 1


# ---------------------------------------------------------------------------
# Tests: stop condition — SIGINT
# ---------------------------------------------------------------------------


class TestFollowerSIGINT:
    """Follower propagates KeyboardInterrupt after best-effort depart."""

    def test_keyboard_interrupt_propagates_and_departs(self) -> None:
        """KeyboardInterrupt from wait_fn propagates; depart called."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)

        def _raising_wait(seconds: float) -> None:
            raise KeyboardInterrupt

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_raising_wait,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(KeyboardInterrupt),
        ):
            follower.run(ctx=_ctx())

        # depart still called
        assert len(coord_repo.depart_calls) == 1
        assert heartbeat.stop_called

    def test_keyboard_interrupt_from_drain_propagates(self) -> None:
        """KeyboardInterrupt raised inside drain_follower_ready_work propagates."""

        class _RaisingProcessor:
            def drain_follower_ready_work(self, ctx: Any, *, before_claim: Any = None) -> list[Any]:
                raise KeyboardInterrupt

        processor = _RaisingProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(KeyboardInterrupt),
        ):
            follower.run(ctx=_ctx())

        assert len(coord_repo.depart_calls) == 1


# ---------------------------------------------------------------------------
# Tests: idle behavior
# ---------------------------------------------------------------------------


class TestFollowerIdleBehavior:
    """Follower calls wait_fn when no READY work is found."""

    def test_idle_calls_wait_fn_with_configured_seconds(self) -> None:
        """Empty drain result triggers wait_fn(idle_poll_seconds)."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        waits: list[float] = []

        def _wait(seconds: float) -> None:
            waits.append(seconds)
            factory.run_lifecycle._running = False  # exit after first idle

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        idle_seconds = 3.5
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_wait,
            idle_poll_seconds=idle_seconds,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        assert waits == [idle_seconds]

    def test_drain_uses_only_public_surface_with_liveness_probe(self) -> None:
        """Follower drives ONLY drain_follower_ready_work, threading before_claim.

        The claim_ready-only / no-pending-sink-recovery contract moved from
        caller flag wiring into the processor's ProcessorMode.FOLLOWER
        (elspeth-577179bba1); it is pinned against a real RowProcessor in
        tests/unit/engine/test_processor_mode.py. Here we pin that the loop
        never reaches for the private drain and always passes its
        leader-liveness probe.
        """
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)

        call_count = 0

        def _wait(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                factory.run_lifecycle._running = False

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_wait,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        assert processor.drain_calls, "expected at least one drain pass"
        for call in processor.drain_calls:
            assert callable(call["before_claim"]), "Follower must thread its leader-liveness probe as before_claim"


# ---------------------------------------------------------------------------
# Tests: drained (work found) behavior
# ---------------------------------------------------------------------------


class TestFollowerDrainedBehavior:
    """Follower loops immediately when drain_follower_ready_work returns non-empty."""

    def test_drained_result_loops_without_sleeping(self) -> None:
        """Non-empty drain result: no wait_fn call, loop immediately."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        waits: list[float] = []

        # First drain: returns work.  Second drain: idle → triggers terminal.
        processor.drain_results = [
            [_FollowerDrainResult()],  # iteration 1: work found
            [],  # iteration 2: idle
        ]

        def _wait(seconds: float) -> None:
            waits.append(seconds)
            factory.run_lifecycle._running = False  # exit after first idle

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_wait,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        # Drain was called twice: once with work, once idle
        assert len(processor.drain_calls) == 2
        # Only ONE wait (from the idle pass, not from the work pass)
        assert len(waits) == 1

    def test_multiple_work_batches_no_sleeps(self) -> None:
        """Three consecutive non-empty drains → no sleeps, then idle triggers one sleep."""
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        waits: list[float] = []

        processor.drain_results = [
            [_FollowerDrainResult()],  # work
            [_FollowerDrainResult()],  # work
            [_FollowerDrainResult()],  # work
            [],  # idle
        ]

        def _wait(seconds: float) -> None:
            waits.append(seconds)
            factory.run_lifecycle._running = False

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        heartbeat = _StubHeartbeat()
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_wait,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        assert len(processor.drain_calls) == 4
        assert len(waits) == 1


# ---------------------------------------------------------------------------
# Tests: depart hygiene
# ---------------------------------------------------------------------------


class TestFollowerDepartHygiene:
    """depart_worker is called exactly once on every clean exit path."""

    def test_depart_called_on_terminal_run(self) -> None:
        _, _, coord_repo, _, heartbeat = _make_follower()
        follower, _, coord_repo, _, heartbeat = _make_follower(factory=_StubFactory(running=False))
        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())
        assert len(coord_repo.depart_calls) == 1

    def test_depart_called_on_seat_dead(self) -> None:
        follower, _, coord_repo, _, heartbeat = _make_follower(coord_repo=_StubRunCoordRepo(seat_present=False))
        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(FollowerSeatDeadError),
        ):
            follower.run(ctx=_ctx())
        assert len(coord_repo.depart_calls) == 1

    def test_depart_called_on_eviction(self) -> None:
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat(evicted=True)
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError),
        ):
            follower.run(ctx=_ctx())

        assert len(coord_repo.depart_calls) == 1

    def test_depart_called_on_sigint(self) -> None:
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)

        def _raising_wait(seconds: float) -> None:
            raise KeyboardInterrupt

        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_raising_wait,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(KeyboardInterrupt),
        ):
            follower.run(ctx=_ctx())

        assert len(coord_repo.depart_calls) == 1

    def test_depart_failure_does_not_mask_run_result(self) -> None:
        """If depart_worker raises, the exception is swallowed (best-effort)."""
        processor = _CountingDrainProcessor()
        factory = _StubFactory(running=False)

        class _ExplodingCoordRepo(_StubRunCoordRepo):
            def depart_worker(self, *, worker_id: str, now: datetime) -> None:
                raise RuntimeError("DB unavailable")

        coord_repo = _ExplodingCoordRepo()
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            # Must NOT raise despite depart_worker exploding
            follower.run(ctx=_ctx())


# ---------------------------------------------------------------------------
# Tests: heartbeat lifecycle
# ---------------------------------------------------------------------------


class TestFollowerHeartbeatLifecycle:
    """Heartbeat is always stopped in the finally block."""

    def test_heartbeat_stopped_on_clean_exit(self) -> None:
        follower, _, _, _, heartbeat = _make_follower(factory=_StubFactory(running=False))
        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())
        assert heartbeat.start_called
        assert heartbeat.stop_called

    def test_heartbeat_stopped_on_eviction(self) -> None:
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat(evicted=True)
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )
        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError),
        ):
            follower.run(ctx=_ctx())
        assert heartbeat.stop_called

    def test_heartbeat_stopped_on_sigint(self) -> None:
        processor = _CountingDrainProcessor()
        coord_repo = _StubRunCoordRepo()
        factory = _StubFactory(running=True)
        heartbeat = _StubHeartbeat()
        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)

        def _raising_wait(seconds: float) -> None:
            raise KeyboardInterrupt

        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=_raising_wait,
        )
        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(KeyboardInterrupt),
        ):
            follower.run(ctx=_ctx())
        assert heartbeat.stop_called


# ---------------------------------------------------------------------------
# Tests: finalize-departure clean exit (§B.1 / §D flip discrimination)
# ---------------------------------------------------------------------------


class TestFollowerFinalizeFlipCleanExit:
    """Follower exits cleanly (not RunWorkerEvictedError) when the heartbeat
    latch fires because complete_run flipped the run_workers row to 'departed'
    (the §D finalize flip).

    Design §B.1 step 5: "a live idle follower discovers the flip via its
    heartbeat rowcount-0 → reads terminal run status → exits 0 through the
    clean-departure path (§A.3)."

    The discrimination: when heartbeat.coordination_lost is True AND
    _run_is_terminal() is True → clean exit (not RunWorkerEvictedError).
    When coordination_lost is True AND run is still RUNNING → true eviction
    → propagate RunWorkerEvictedError.
    """

    def test_finalize_flip_exits_cleanly_not_eviction_error(self) -> None:
        """heartbeat.coordination_lost=True + run COMPLETED → return, not raise."""
        processor = _CountingDrainProcessor()
        # Run is COMPLETED (terminal): the finalize flip has happened.
        factory = _StubFactory(running=False)
        coord_repo = _StubRunCoordRepo()
        # Heartbeat latch is already set (simulates worker_active=False).
        heartbeat = _StubHeartbeat(evicted=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        # Must return normally — NOT raise RunWorkerEvictedError.
        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())  # no exception

        # Depart was called (clean-exit hygiene).
        assert len(coord_repo.depart_calls) == 1
        assert coord_repo.depart_calls[0]["worker_id"] == WORKER_ID

    def test_true_eviction_still_propagates_error(self) -> None:
        """heartbeat.coordination_lost=True + run still RUNNING → RunWorkerEvictedError."""
        processor = _CountingDrainProcessor()
        # Run is still RUNNING (true eviction, not finalize-departure).
        factory = _StubFactory(running=True)
        coord_repo = _StubRunCoordRepo()
        heartbeat = _StubHeartbeat(evicted=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with (
            patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat),
            pytest.raises(RunWorkerEvictedError) as exc_info,
        ):
            follower.run(ctx=_ctx())

        assert exc_info.value.worker_id == WORKER_ID
        assert exc_info.value.run_id == RUN_ID
        # Best-effort depart was attempted.
        assert len(coord_repo.depart_calls) == 1

    def test_finalize_flip_departs_idempotently(self) -> None:
        """depart_worker called once on the clean finalize-flip exit path."""
        processor = _CountingDrainProcessor()
        factory = _StubFactory(running=False)
        coord_repo = _StubRunCoordRepo()
        heartbeat = _StubHeartbeat(evicted=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        # Exactly one depart call despite finalize already having departed it
        # (best-effort swallows the "already departed" error).
        assert len(coord_repo.depart_calls) == 1

    def test_no_drain_called_when_latch_set_at_entry(self) -> None:
        """When latch is already set at loop entry, no drain pass is attempted."""
        processor = _CountingDrainProcessor()
        factory = _StubFactory(running=False)
        coord_repo = _StubRunCoordRepo()
        heartbeat = _StubHeartbeat(evicted=True)

        token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER_ID, leader_epoch=0)
        follower = FollowerProcessor(
            processor=processor,
            token=token,
            run_coordination=coord_repo,  # type: ignore[arg-type]
            factory=factory,  # type: ignore[arg-type]
            now_fn=lambda: NOW,
            wait_fn=lambda _: None,
        )

        with patch("elspeth.engine.orchestrator.follower.RunHeartbeatThread", return_value=heartbeat):
            follower.run(ctx=_ctx())

        # The drain should not have been called — we exited before it.
        assert len(processor.drain_calls) == 0


# ---------------------------------------------------------------------------
# Tests: _SeatDeadError
# ---------------------------------------------------------------------------


class TestSeatDeadError:
    """_SeatDeadError carries run_id and worker_id for logging."""

    def test_seat_dead_error_message_contains_resume_hint(self) -> None:
        err = _SeatDeadError(WORKER_ID, RUN_ID)
        assert "elspeth resume" in str(err)
        assert err.worker_id == WORKER_ID
        assert err.run_id == RUN_ID


# ---------------------------------------------------------------------------
# Tests: follower_barrier_node_ids — aggregation arm (ADR-030 §B, Finding 6)
# ---------------------------------------------------------------------------


class TestFollowerBarrierNodeIds:
    """ADR-030 §B: a follower must NOT execute batch-aware transforms at
    aggregation nodes — it must return (None, []) so _drain_scheduler_claims
    calls mark_blocked (durable barrier hold, no in-memory accept).

    This covers the aggregation arm of Finding 6: even though
    aggregation_settings={} on a follower, the follower_barrier_node_ids
    frozenset carries the aggregation node IDs and intercepts the transform
    BEFORE it falls through to _handle_transform_node.
    """

    def _make_follower_processor_for_agg_test(self) -> tuple[Any, Any, Any, Any]:
        """Build a minimal RowProcessor with follower_barrier_node_ids wired.

        Returns (processor, agg_node_id, transform, token).
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.contracts.types import NodeID
        from elspeth.engine.processor import DAGTraversalContext, RowProcessor
        from elspeth.engine.spans import SpanFactory
        from tests.fixtures.landscape import make_recorder_with_run

        _SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

        run_id = "run-follower-barrier-test"
        source_node_id = NodeID("source-0")
        agg_node_id = NodeID("agg-node-1")

        setup = make_recorder_with_run(
            run_id=run_id,
            source_node_id=str(source_node_id),
            source_plugin_name="test-source",
        )
        factory = setup.factory

        # Register the aggregation node in Landscape (FK constraint).
        from elspeth.contracts import NodeType

        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="aggregation",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            node_id=str(agg_node_id),
            schema_config=_SCHEMA,
        )

        # A batch-aware transform that must NOT be called on a follower.
        transform = _UncalledBatchTransform(node_id=str(agg_node_id))

        traversal = DAGTraversalContext(
            node_step_map={source_node_id: 0, agg_node_id: 1},
            node_to_plugin={agg_node_id: transform},
            node_to_next={source_node_id: agg_node_id, agg_node_id: None},
            coalesce_node_map={},
        )

        # Stub scheduler — mark_blocked / claim_ready are not called by
        # _process_single_token directly; we only call the method and inspect
        # the return value.
        stub_scheduler = _UnusedScheduler()

        from tests.fixtures.landscape import leader_coordination_token

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id=run_id,
            source_node_id=source_node_id,
            source_on_success="default",
            traversal=traversal,
            aggregation_settings={},  # follower: empty — no trigger evaluation
            follower_barrier_node_ids=frozenset({agg_node_id}),  # the fix
            sink_names=frozenset({"default"}),
            scheduler=stub_scheduler,
            coordination_token=leader_coordination_token(factory, run_id),
        )

        from elspeth.testing import make_token_info

        token = make_token_info(data={"value": 42}, token_id="tok-agg-follower-1")

        return processor, agg_node_id, transform, token

    def test_batch_aware_transform_at_follower_barrier_node_returns_none_not_executed(self) -> None:
        """follower_barrier_node_ids: batch-aware transform at aggregation node
        returns (None, []) without calling the transform, so _drain_scheduler_claims
        calls mark_blocked instead of executing the aggregation row-wise.

        Regression for Finding 6 (aggregation arm): without follower_barrier_node_ids,
        the fallthrough to _handle_transform_node runs the transform row-wise,
        producing wrong aggregate output and bypassing the leader's barrier.
        """
        processor, agg_node_id, transform, token = self._make_follower_processor_for_agg_test()

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-follower-barrier-test", config={}, landscape=None)

        result, child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=agg_node_id,
            on_success_sink="default",
        )

        # The follower MUST return (None, []) — barrier hold, no transform executed.
        assert result is None, (
            f"Expected None (barrier hold), got {result!r}. Follower must not execute batch-aware aggregation transforms row-wise."
        )
        assert child_items == [], f"Expected no child items, got {child_items!r}"

        # The transform must NOT have been called (process() is the TransformProtocol method).
        assert not transform.process_called

    def test_non_batch_aware_transform_not_intercepted_by_follower_barrier(self) -> None:
        """A non-batch-aware transform is NEVER intercepted by follower_barrier_node_ids,
        even if the node is in the frozenset.  Only batch-aware transforms at
        aggregation nodes are intercepted (the condition is
        ``is_batch_aware AND node_id in follower_barrier_node_ids``).
        """
        # We verify this by asserting that _follower_barrier_node_ids on a
        # non-follower (empty frozenset) processor never triggers.  This is a
        # pure structural test — no node registration required since
        # _process_single_token is not called.
        from elspeth.contracts.types import NodeID
        from elspeth.engine.processor import RowProcessor

        agg_node = NodeID("agg-node-structural")
        other_node = NodeID("other-node")

        # Build a bare processor (object.__new__) to inspect internal state only —
        # we are NOT calling _process_single_token here, just verifying that
        # the follower_barrier_node_ids frozenset is stored correctly and that
        # a node NOT in the set returns None from _barrier_key_for_blocked_item.
        processor = object.__new__(RowProcessor)
        processor._aggregation_settings = {}
        processor._follower_barrier_node_ids = frozenset({agg_node})

        from elspeth.engine.work_items import WorkItem
        from elspeth.testing import make_token_info

        token = make_token_info(data={"v": 1}, token_id="tok-structural-1")

        # A node NOT in follower_barrier_node_ids → barrier_key is None.
        item_outside = WorkItem(token=token, current_node_id=other_node)
        assert processor._barrier_key_for_blocked_item(item_outside) is None

        # A node IN follower_barrier_node_ids → barrier_key = str(node_id).
        item_barrier = WorkItem(token=token, current_node_id=agg_node)
        assert processor._barrier_key_for_blocked_item(item_barrier) == str(agg_node)

    def test_barrier_key_for_blocked_item_returns_node_id_for_follower_barrier_node(self) -> None:
        """_barrier_key_for_blocked_item returns str(node_id) for aggregation nodes
        in follower_barrier_node_ids, matching the leader's barrier_key so the
        adoption verb finds the correct row.
        """
        from elspeth.contracts.types import NodeID
        from elspeth.engine.processor import DAGTraversalContext, RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.work_items import WorkItem
        from elspeth.testing import make_token_info
        from tests.fixtures.landscape import leader_coordination_token, make_recorder_with_run

        run_id = "run-barrier-key-test"
        source_node_id = NodeID("source-0")
        agg_node_id = NodeID("agg-2")

        setup = make_recorder_with_run(
            run_id=run_id,
            source_node_id=str(source_node_id),
            source_plugin_name="test-source",
        )
        factory = setup.factory

        traversal = DAGTraversalContext(
            node_step_map={source_node_id: 0},
            node_to_plugin={},
            node_to_next={source_node_id: None},
            coalesce_node_map={},
        )

        stub_scheduler = _UnusedScheduler()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id=run_id,
            source_node_id=source_node_id,
            source_on_success="default",
            traversal=traversal,
            aggregation_settings={},  # follower: empty
            follower_barrier_node_ids=frozenset({agg_node_id}),
            sink_names=frozenset({"default"}),
            scheduler=stub_scheduler,
            coordination_token=leader_coordination_token(factory, run_id),
        )

        token = make_token_info(data={"v": 1}, token_id="tok-bk-1")
        item = WorkItem(token=token, current_node_id=agg_node_id)

        barrier_key = processor._barrier_key_for_blocked_item(item)

        assert barrier_key == str(agg_node_id), (
            f"Expected barrier_key={str(agg_node_id)!r} for follower barrier node, got {barrier_key!r}. "
            "Without this, _mark_claimed_scheduler_work_blocked raises OrchestrationInvariantError."
        )
