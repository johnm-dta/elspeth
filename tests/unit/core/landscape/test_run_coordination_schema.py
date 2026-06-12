"""Epoch-21 coordination-substrate schema tests (ADR-030, slice 2).

Dedicated unit tests for the multi-worker coordination tables and the shared
``active_worker_fence_clause`` construct (design §G: one definition, one
dedicated unit test each). Repository verbs are tested elsewhere — these
tests pin the *schema-level* invariants:

- the membership fence predicate semantics (active-only, run-scoped),
- the CHECK constraints on run_coordination / run_workers /
  run_coordination_events,
- run_coordination_events.seq strict monotonicity (AUTOINCREMENT took, so
  seq values are never reused after deletion),
- coalesce_branch_losses natural-key idempotency.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from elspeth.core.landscape.schema import (
    active_worker_fence_clause,
    coalesce_branch_losses_table,
    metadata,
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
)

NOW = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=60)


@pytest.fixture
def engine() -> Iterator[Engine]:
    """In-memory schema-only engine (FK pragma off; CHECKs always enforced)."""
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    yield eng
    eng.dispose()


def _insert_worker(
    engine: Engine,
    *,
    worker_id: str,
    run_id: str = "run-1",
    role: str = "leader",
    status: str = "active",
    evicted_at: datetime | None = None,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            run_workers_table.insert().values(
                worker_id=worker_id,
                run_id=run_id,
                role=role,
                status=status,
                registered_at=NOW,
                heartbeat_expires_at=LATER,
                evicted_at=evicted_at,
            )
        )


def _fence_result(engine: Engine, *, worker_id: str, run_id: str) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(select(active_worker_fence_clause(worker_id=worker_id, run_id=run_id))).scalar_one())


class TestActiveWorkerFenceClause:
    """Membership fence: True iff the worker holds an *active* row in *this* run."""

    def test_true_for_active_row(self, engine: Engine) -> None:
        _insert_worker(engine, worker_id="worker:run-1:aaa", status="active")
        assert _fence_result(engine, worker_id="worker:run-1:aaa", run_id="run-1") is True

    def test_false_for_departed_row(self, engine: Engine) -> None:
        _insert_worker(engine, worker_id="worker:run-1:bbb", status="departed")
        assert _fence_result(engine, worker_id="worker:run-1:bbb", run_id="run-1") is False

    def test_false_for_evicted_row(self, engine: Engine) -> None:
        _insert_worker(engine, worker_id="worker:run-1:ccc", status="evicted", evicted_at=NOW)
        assert _fence_result(engine, worker_id="worker:run-1:ccc", run_id="run-1") is False

    def test_false_for_absent_worker(self, engine: Engine) -> None:
        assert _fence_result(engine, worker_id="worker:run-1:ghost", run_id="run-1") is False

    def test_false_for_active_row_in_other_run(self, engine: Engine) -> None:
        """Run-scoped: an active registration in run-1 is no fence pass for run-2."""
        _insert_worker(engine, worker_id="worker:run-1:ddd", run_id="run-1", status="active")
        assert _fence_result(engine, worker_id="worker:run-1:ddd", run_id="run-2") is False


class TestRunCoordinationSeatLivenessCheck:
    """ck_run_coordination_seat_liveness_paired: vacant seat ⇔ no liveness clock."""

    def test_vacant_seat_without_clock_is_valid(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute(
                run_coordination_table.insert().values(
                    run_id="run-1", leader_worker_id=None, leader_heartbeat_expires_at=None, updated_at=NOW
                )
            )

    def test_occupied_seat_with_clock_is_valid(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute(
                run_coordination_table.insert().values(
                    run_id="run-1",
                    leader_worker_id="worker:run-1:aaa",
                    leader_epoch=1,
                    leader_heartbeat_expires_at=LATER,
                    updated_at=NOW,
                )
            )

    def test_vacant_seat_with_clock_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                run_coordination_table.insert().values(
                    run_id="run-1", leader_worker_id=None, leader_heartbeat_expires_at=LATER, updated_at=NOW
                )
            )

    def test_occupied_seat_without_clock_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                run_coordination_table.insert().values(
                    run_id="run-1",
                    leader_worker_id="worker:run-1:aaa",
                    leader_heartbeat_expires_at=None,
                    updated_at=NOW,
                )
            )

    def test_leader_epoch_defaults_to_zero(self, engine: Engine) -> None:
        """server_default=0: the seat row is born at epoch 0; acquisition CAS bumps it."""
        with engine.begin() as conn:
            conn.execute(run_coordination_table.insert().values(run_id="run-1", updated_at=NOW))
        with engine.connect() as conn:
            epoch = conn.execute(select(run_coordination_table.c.leader_epoch)).scalar_one()
        assert epoch == 0


class TestRunWorkersChecks:
    def test_unknown_role_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError):
            _insert_worker(engine, worker_id="w-bad-role", role="observer")

    def test_unknown_status_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError):
            _insert_worker(engine, worker_id="w-bad-status", status="zombie")

    def test_evicted_without_evicted_at_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError):
            _insert_worker(engine, worker_id="w-evicted-bare", status="evicted", evicted_at=None)

    def test_active_with_evicted_at_is_rejected(self, engine: Engine) -> None:
        """The pairing CHECK is biconditional: evicted_at on a non-evicted row is invalid."""
        with pytest.raises(IntegrityError):
            _insert_worker(engine, worker_id="w-active-evicted", status="active", evicted_at=NOW)

    def test_follower_role_is_valid(self, engine: Engine) -> None:
        _insert_worker(engine, worker_id="w-follower", role="follower")


class TestRunCoordinationEvents:
    @staticmethod
    def _insert_event(engine: Engine, *, event_id: str, event_type: str) -> int:
        with engine.begin() as conn:
            result = conn.execute(
                run_coordination_events_table.insert().values(
                    event_id=event_id,
                    run_id="run-1",
                    event_type=event_type,
                    worker_id="worker:run-1:aaa",
                    leader_epoch=1,
                    recorded_at=NOW,
                    context_json="{}",
                )
            )
            pk = result.inserted_primary_key
            assert pk is not None
        return int(pk[0])

    @pytest.mark.parametrize(
        "event_type",
        [
            "worker_register",
            "worker_depart",
            "worker_evict",
            "worker_stalled",
            "leader_acquire",
            "leader_release",
            "leadership_lost",
            "fence_refusal",
            "heartbeat_degraded",
            "finalize",
        ],
    )
    def test_all_ten_event_types_are_accepted(self, engine: Engine, event_type: str) -> None:
        """All design-§A.2 event types — including the slice-4 producers — pass the CHECK."""
        self._insert_event(engine, event_id=f"ev-{event_type}", event_type=event_type)

    def test_unknown_event_type_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(IntegrityError):
            self._insert_event(engine, event_id="ev-bad", event_type="leader_coup")

    def test_duplicate_event_id_is_rejected(self, engine: Engine) -> None:
        """uq_run_coordination_events_event_id delivers the scheduler-events dedup discipline."""
        self._insert_event(engine, event_id="ev-dup", event_type="worker_register")
        with pytest.raises(IntegrityError):
            self._insert_event(engine, event_id="ev-dup", event_type="worker_depart")

    def test_seq_is_strictly_monotonic_across_deletion(self, engine: Engine) -> None:
        """AUTOINCREMENT took: deleting the max row never frees its seq for reuse.

        Without ``sqlite_autoincrement=True`` the PK is a bare rowid alias and
        SQLite reuses max(rowid)+1 after a delete, letting a later event sort
        *under* an earlier one in replay order.
        """
        first = self._insert_event(engine, event_id="ev-1", event_type="worker_register")
        second = self._insert_event(engine, event_id="ev-2", event_type="leader_acquire")
        assert second > first

        with engine.begin() as conn:
            conn.execute(run_coordination_events_table.delete().where(run_coordination_events_table.c.seq == second))

        third = self._insert_event(engine, event_id="ev-3", event_type="leader_release")
        assert third > second, "seq was reused after deletion — AUTOINCREMENT is not in effect"

    def test_sqlite_ddl_contains_autoincrement(self) -> None:
        ddl = str(CreateTable(run_coordination_events_table).compile(dialect=sqlite.dialect()))
        assert "AUTOINCREMENT" in ddl

    def test_context_json_defaults_to_empty_object(self, engine: Engine) -> None:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO run_coordination_events (event_id, run_id, event_type, worker_id, recorded_at) "
                    "VALUES ('ev-default', 'run-1', 'worker_register', 'worker:run-1:aaa', '2026-06-12 00:00:00')"
                )
            )
        with engine.connect() as conn:
            context = conn.execute(
                select(run_coordination_events_table.c.context_json).where(run_coordination_events_table.c.event_id == "ev-default")
            ).scalar_one()
        assert context == "{}"


class TestCoalesceBranchLossesNaturalKey:
    @staticmethod
    def _insert_loss(engine: Engine, *, loss_id: str, branch_name: str = "branch-a", token_id: str = "tok-1") -> None:
        with engine.begin() as conn:
            conn.execute(
                coalesce_branch_losses_table.insert().values(
                    loss_id=loss_id,
                    run_id="run-1",
                    coalesce_name="merge-1",
                    row_id="row-1",
                    branch_name=branch_name,
                    token_id=token_id,
                    reason="failed",
                    recorded_by="worker:run-1:aaa",
                    recorded_at=NOW,
                )
            )

    def test_duplicate_natural_key_is_rejected(self, engine: Engine) -> None:
        """uq_coalesce_branch_losses_natural: (run_id, coalesce_name, row_id, branch_name)."""
        self._insert_loss(engine, loss_id="loss-1")
        with pytest.raises(IntegrityError):
            self._insert_loss(engine, loss_id="loss-2", token_id="tok-2")

    def test_distinct_branch_is_accepted(self, engine: Engine) -> None:
        self._insert_loss(engine, loss_id="loss-1", branch_name="branch-a")
        self._insert_loss(engine, loss_id="loss-2", branch_name="branch-b")

    def test_adopted_epoch_defaults_to_null(self, engine: Engine) -> None:
        """NULL = not yet replayed into leader memory (slice-3 replay verb stamps it)."""
        self._insert_loss(engine, loss_id="loss-1")
        with engine.connect() as conn:
            adopted = conn.execute(select(coalesce_branch_losses_table.c.adopted_epoch)).scalar_one()
        assert adopted is None
