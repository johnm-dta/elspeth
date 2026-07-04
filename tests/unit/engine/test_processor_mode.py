# tests/unit/engine/test_processor_mode.py
"""ProcessorMode contract tests (elspeth-577179bba1).

Follower-ness is an EXPLICIT construction-time decision (``ProcessorMode``),
not an inference re-derived from the ``coordination_token`` /
``run_coordination`` None-sentinels. These tests pin the mode contract at the
RowProcessor surface over a real scheduler DB:

1. ``drain_follower_ready_work`` (the public follower drain surface) owns the
   follower-mode contract: ``claim_ready`` only — never ``claim_pending_sink``
   / pending-sink recovery, never ``recover_expired_leases`` — and threads the
   caller's ``before_claim`` leader-liveness probe ahead of every claim
   attempt (ADR-030 §B.1/§C.3).
2. The surface fails closed on mode: a LEADER-mode processor driven through
   the follower surface raises OrchestrationInvariantError (the wrong-mode
   bug the named contract exists to catch).
3. FOLLOWER wiring invariants are validated fail-closed at construction:
   a coordination token (epoch fence), a run_coordination repository (§C.2
   housekeeping), or a missing explicit lease owner each refuse construction.
4. Maintenance parity: the in-drain maintenance cadence never reaps leases in
   FOLLOWER mode — the explicit-mode replacement for the old triple-None
   skip (the inverse LEADER-mode pin lives in
   test_scheduler_drain_characterization.py).

Harness: the real-RowProcessor-over-real-scheduler-DB ``_build`` net from
test_scheduler_drain_characterization.py (typed fakes and a delegating
recording scheduler — no bare mocks).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts import RowResult
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.engine.dag_navigator import WorkItem
from elspeth.engine.processor import SCHEDULER_MAINTENANCE_INTERVAL
from elspeth.engine.scheduler_drain import ProcessorMode
from tests.unit.engine.test_scheduler_drain_characterization import (
    LEADER_OWNER,
    _build,
    _ctx,
    _dropped_result,
    _enqueue_ready,
    _register_worker,
)

FOLLOWER_OWNER = "follower-1"


# ---------------------------------------------------------------------------
# 1. The public follower drain surface owns the follower-mode contract
# ---------------------------------------------------------------------------


def test_drain_follower_ready_work_claims_ready_only_and_threads_before_claim() -> None:
    """FOLLOWER drain: claim_ready only, no pending-sink recovery, probe threaded."""
    follower, spy, setup, clock = _build(lease_owner=FOLLOWER_OWNER, mode=ProcessorMode.FOLLOWER)
    _register_worker(setup, FOLLOWER_OWNER)
    _work_item_id, token = _enqueue_ready(setup, spy, clock, sequence=0)

    probe_calls: list[bool] = []

    def probe() -> None:
        probe_calls.append(True)

    def fake_process(**kwargs: Any) -> tuple[RowResult, list[WorkItem]]:
        return _dropped_result(kwargs["token"]), []

    spy.calls.clear()
    with patch.object(follower, "_process_single_token", new=fake_process):
        results = follower.drain_follower_ready_work(_ctx(setup), before_claim=probe)

    assert [result.token.token_id for result in results] == [token.token_id]

    verbs = spy.verbs()
    assert "claim_ready" in verbs, "the follower surface must drain via claim_ready"
    assert "claim_pending_sink" not in verbs, "pending-sink recovery is leader-only (ADR-030 §C.3)"
    assert "recover_expired_leases" not in verbs, "lease recovery is leader-only (ADR-030 §C.3)"
    # One probe per claim attempt: a non-recovery drain with no in-memory
    # continuations returns after the terminal disposition, so the single
    # claimed item means exactly one claim attempt (the follower's outer
    # loop re-drives the drain between passes).
    assert len(probe_calls) == 1


# ---------------------------------------------------------------------------
# 2. Wrong-mode guard on the public surface
# ---------------------------------------------------------------------------


def test_drain_follower_ready_work_refuses_leader_mode_processor() -> None:
    """Driving a default (LEADER) processor through the follower surface raises."""
    leader, _spy, setup, _clock = _build(lease_owner=LEADER_OWNER, bind_leader_token=True)

    with pytest.raises(OrchestrationInvariantError, match=r"ProcessorMode\.FOLLOWER"):
        leader.drain_follower_ready_work(_ctx(setup))


# ---------------------------------------------------------------------------
# 3. FOLLOWER construction invariants are validated fail-closed
# ---------------------------------------------------------------------------


def test_follower_mode_with_coordination_token_refuses_construction() -> None:
    """A follower must never present an epoch fence."""
    with pytest.raises(OrchestrationInvariantError, match="forbids a coordination_token"):
        _build(lease_owner=FOLLOWER_OWNER, bind_leader_token=True, mode=ProcessorMode.FOLLOWER)


def test_follower_mode_with_run_coordination_refuses_construction() -> None:
    """A follower must never hold the §C.2 housekeeping repository."""
    # _build never wires run_coordination, so drive the invariant through the
    # real builder shape: construct via _build to get a valid setup, then
    # re-run the constructor with run_coordination attached.
    _follower, _spy, setup, clock = _build(lease_owner=FOLLOWER_OWNER, mode=ProcessorMode.FOLLOWER)
    from elspeth.contracts.types import NodeID
    from elspeth.engine.processor import DAGTraversalContext, RowProcessor
    from elspeth.engine.spans import SpanFactory

    with pytest.raises(OrchestrationInvariantError, match="forbids run_coordination"):
        RowProcessor(
            execution=setup.execution,
            data_flow=setup.data_flow,
            span_factory=SpanFactory(),
            run_id=setup.run_id,
            source_node_id=NodeID(setup.source_node_id),
            source_on_success="default",
            traversal=DAGTraversalContext(
                node_step_map={NodeID(setup.source_node_id): 0},
                node_to_plugin={},
                node_to_next={},
                coalesce_node_map={},
            ),
            scheduler=setup.factory.scheduler,
            scheduler_lease_owner=FOLLOWER_OWNER,
            coordination_token=None,
            run_coordination=setup.factory.run_coordination,
            clock=clock,
            mode=ProcessorMode.FOLLOWER,
        )


def test_follower_mode_without_explicit_lease_owner_refuses_construction() -> None:
    """A follower's registered worker identity IS its lease owner (§A.1)."""
    with pytest.raises(OrchestrationInvariantError, match="requires an explicit scheduler_lease_owner"):
        _build(lease_owner=None, register_leader=None, mode=ProcessorMode.FOLLOWER)


# ---------------------------------------------------------------------------
# 4. Maintenance parity: the in-drain cadence never reaps in FOLLOWER mode
# ---------------------------------------------------------------------------


def test_follower_drain_cadence_never_runs_lease_recovery() -> None:
    """The explicit-mode replacement for the old triple-None maintenance skip.

    Drive enough idle follower drains to trip the SCHEDULER_MAINTENANCE_INTERVAL
    cadence: run_maintenance fires, but FOLLOWER mode returns 0 up front —
    recover_expired_leases (whose None-token call would take the unfenced
    legacy arm and reap peers' in-flight leases) is never issued.
    """
    follower, spy, setup, _clock = _build(lease_owner=FOLLOWER_OWNER, mode=ProcessorMode.FOLLOWER)
    _register_worker(setup, FOLLOWER_OWNER)
    ctx = _ctx(setup)

    spy.calls.clear()
    for _ in range(SCHEDULER_MAINTENANCE_INTERVAL):
        assert follower.drain_follower_ready_work(ctx) == []

    assert spy.calls_for("recover_expired_leases") == [], "follower maintenance must never reap peer leases"
    assert spy.calls_for("claim_pending_sink") == []
