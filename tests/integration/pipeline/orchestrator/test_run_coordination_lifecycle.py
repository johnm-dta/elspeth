# tests/integration/pipeline/orchestrator/test_run_coordination_lifecycle.py
"""Epoch-21 uniformity pin: every N=1 run exercises the coordination substrate.

ADR-030 uniformity rule (slice 2): ``begin_run`` mints the ``run_coordination``
leader seat at epoch 1 atomically with the runs row, registers the origin
worker as leader, and the run teardown paths release the seat. These tests
drive the REAL orchestrator end-to-end and pin the substrate's in-DB image on
both the success and the failure arms.

The §H pin includes the ``finalize`` event written inside the fenced
``complete_run`` terminal transaction (slice-2 step 4, ADR-030 §D).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from elspeth.contracts import RunStatus
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import (
    run_coordination_events_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, FailingSink, ListSource, PassTransform


def _coordination_image(db: LandscapeDB, run_id: str) -> tuple[object, list[object], list[str]]:
    """Read (seat row, worker rows, ordered event types) for ``run_id``."""
    with db.engine.connect() as conn:
        seat = conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == run_id)).one()
        workers = conn.execute(select(run_workers_table).where(run_workers_table.c.run_id == run_id)).all()
        event_types = (
            conn.execute(
                select(run_coordination_events_table.c.event_type)
                .where(run_coordination_events_table.c.run_id == run_id)
                .order_by(run_coordination_events_table.c.seq)
            )
            .scalars()
            .all()
        )
    return seat, list(workers), list(event_types)


class TestRunCoordinationUniformityPin:
    """ADR-030 §H uniformity pin (slice-2 half): N=1 = leader-of-its-own-run."""

    def test_normal_run_mints_epoch_1_seat_and_releases_it(self, tmp_path: Path, payload_store) -> None:
        """A plain successful ``run()``:

        - the run_coordination seat row exists, minted at epoch 1 and never
          bumped (no takeover happened);
        - the origin worker is registered as the run's leader with the
          ``worker:{run_id}:`` identity (§A.1) and pid/hostname forensics;
        - on normal completion the seat is RELEASED (vacant) and the leader
          row departs — graceful-shutdown hygiene (§D);
        - the event ledger reads worker_register → leader_acquire →
          finalize → leader_release in seq order (§D: the finalize event is
          written inside the fenced complete_run transaction, before the
          seat release).
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        source = ListSource([{"value": 1}, {"value": 2}])
        transform = PassTransform(name="passthrough", on_success="default", on_error="discard")
        sink = CollectSink()
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        result = Orchestrator(db).run(config, graph=build_production_graph(config), payload_store=payload_store)
        assert result.status == RunStatus.COMPLETED

        run_id = result.run_id
        seat, workers, event_types = _coordination_image(db, run_id)

        assert seat.leader_epoch == 1, "fresh path mints epoch 1; nothing may bump it"
        assert seat.leader_worker_id is None, "normal completion must release the seat"
        assert seat.leader_heartbeat_expires_at is None, "vacant seat carries no expiry (biconditional CHECK)"

        assert len(workers) == 1, "an N=1 run registers exactly one worker"
        (leader,) = workers
        assert leader.role == "leader"
        assert leader.status == "departed", "graceful completion departs the leader row"
        assert leader.entry_point == "run"
        assert isinstance(leader.worker_id, str) and leader.worker_id.startswith(f"worker:{run_id}:")
        assert leader.pid is not None and leader.hostname is not None

        # §D: complete_run writes the 'finalize' event in the terminal
        # transaction, BEFORE the seat release (step-4 fence work).
        assert event_types == ["worker_register", "leader_acquire", "finalize", "leader_release"]
        db.close()

    def test_failed_run_finalizes_failed_then_releases_seat(self, tmp_path: Path, payload_store) -> None:
        """The failure ceremony arm releases the seat AFTER the FAILED finalize.

        A failed resume's (or run's) wedged seat would block retries for the
        liveness window — §3 seat hygiene. The release happens only after the
        terminal status is durably recorded, so the pin checks both.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        source = ListSource([{"value": 1}])
        transform = PassTransform(name="passthrough", on_success="default", on_error="discard")
        sink = FailingSink(error_message="boom — sink never writes")
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        with pytest.raises(RuntimeError, match="boom"):
            Orchestrator(db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        with db.engine.connect() as conn:
            run_row = conn.execute(select(runs_table)).one()
        run_id = run_row.run_id
        assert run_row.status == RunStatus.FAILED.value

        seat, workers, event_types = _coordination_image(db, run_id)
        assert seat.leader_epoch == 1
        assert seat.leader_worker_id is None, "the FAILED ceremony must release the seat after finalize"
        (leader,) = workers
        assert leader.status == "departed"
        # The FAILED ceremony finalizes (writing the 'finalize' event) and
        # only then releases the seat.
        assert event_types == ["worker_register", "leader_acquire", "finalize", "leader_release"]
        db.close()
