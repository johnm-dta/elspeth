"""N=1 uniformity pin (ADR-030 §H "Additional pinned doctrine", design :478).

The uniformity rule (design :489): "a single worker is simply
leader-of-its-own-run" — ``begin_run`` mints the ``run_coordination`` seat at
epoch 1 in the same transaction as the runs row, the origin worker registers
as leader, every fenced verb the run executes doubles as a seat heartbeat,
and a clean completion writes the ``finalize`` event then releases the seat.

E2E because it must drive a REAL ``Orchestrator.run()`` (the harness's 3-row
pipeline, no interrupt, MockClock-injected): every existing e2e run now
exercises this substrate, so the full suite is the broader regression net
for the property; this file is the explicit pin.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from elspeth.contracts import RunStatus
from elspeth.core.landscape import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import Orchestrator
from tests.e2e.recovery.harness import (
    _SOURCE_ROWS,
    _T0,
    _build_pipeline,
    _coordination_events,
    _coordination_row,
    _run_workers,
)


class TestRunCoordinationUniformity:
    def test_single_worker_run_is_leader_of_its_own_run(self, tmp_path: Path) -> None:
        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        config, graph, sink, _source = _build_pipeline(_SOURCE_ROWS)

        result = Orchestrator(db, clock=MockClock(start=_T0)).run(config, graph=graph, payload_store=payload_store)

        # 4. N=1 behavior preserved: sink/audit unchanged from today's
        # expectations — 3 rows, COMPLETED.
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3
        assert sink.results == _SOURCE_ROWS
        run_id = result.run_id

        # 1. Exactly one run_coordination row at leader_epoch 1 (begin_run
        # minted it; nothing bumped it). Clean completion RELEASED the seat,
        # so the seat is vacant with the biconditional NULL expiry.
        seat = _coordination_row(db, run_id)
        assert seat["leader_epoch"] == 1
        assert seat["leader_worker_id"] is None
        assert seat["leader_heartbeat_expires_at"] is None

        # 2. Exactly one registered worker: the origin worker, role=leader,
        # §A.1 identity shape, pid/hostname forensics. Post-completion status
        # is 'departed' — the graceful-release semantics (release_seat CASes
        # the leader's own registry row active→departed after the terminal
        # finalize succeeded); pinned as non-'evicted' single-use identity.
        workers = _run_workers(db, run_id)
        assert len(workers) == 1
        (leader,) = workers
        assert leader["role"] == "leader"
        assert re.fullmatch(rf"worker:{re.escape(run_id)}:[0-9a-f]{{32}}", str(leader["worker_id"]))
        assert leader["entry_point"] == "run"
        assert leader["pid"] is not None
        assert leader["hostname"] is not None
        assert leader["status"] == "departed"
        assert leader["evicted_at"] is None

        # 3. The event ledger: worker_register and leader_acquire at epoch 1,
        # exactly one finalize (inside the fenced complete_run transaction),
        # zero fence_refusal / worker_evict / heartbeat_degraded on a clean
        # run, and strictly increasing seq.
        events = _coordination_events(db, run_id)
        event_types = [str(event["event_type"]) for event in events]
        assert event_types == ["worker_register", "leader_acquire", "finalize", "leader_release"]
        acquire = events[1]
        assert acquire["leader_epoch"] == 1
        finalize = events[2]
        assert json.loads(str(finalize["context_json"]))["status"] == RunStatus.COMPLETED.value
        assert finalize["worker_id"] == leader["worker_id"]
        seqs = [int(event["seq"]) for event in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "seq strictly increasing"
        db.close()
