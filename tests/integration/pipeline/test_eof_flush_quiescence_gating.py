# tests/integration/pipeline/test_eof_flush_quiescence_gating.py
"""ADR-030 §D steps 2-3 — the end-of-input flush is journal-quiescence gated.

Sibling of test_rc6_eof_resume_proof.py, real pipelines end-to-end. The §D
rule: the EOF barrier flush may run ONLY when the journal holds zero READY
rows and zero non-pending-sink LEASED rows — otherwise a still-in-flight work
item (a slice-4/5 slow follower's claim) could deposit a barrier arrival
AFTER the "final" batch flushed, silently splitting it into two batches.

At N=1 the gate is a loud refusal (not a wait — the leader's own drain always
returns quiesced, so unquiesced work at EOF means a peer): the run FAILS with
``OrchestrationInvariantError`` and NOTHING flushes. When the simulated peer
then completes into the barrier (its row goes LEASED→BLOCKED via the
production ``mark_blocked``), the resume leader adopts the inherited arrival
journal-first (§E.2 rehydration under resume provenance) and the §D step-3
loop emits exactly ONE batch containing EVERY member, the late one included —
never two batches.

Construction discipline: the simulated peer's rows are written by the
PRODUCTION writers only (RecorderFactory / TokenSchedulerRepository — the
``_craft_crashed_lease`` discipline from tests/e2e/recovery/harness.py),
crafted from inside the source plugin at the exact EOF instant.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    batches_table,
    rows_table,
    runs_table,
    token_work_items_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator
from tests.fixtures.base_classes import _TestSourceBase
from tests.fixtures.plugins import CollectSink, ListSource
from tests.integration.pipeline.test_aggregation_recovery import (
    _build_eof_aggregation_pipeline,
    _SumBatchTransform,
)

PEER_OWNER = "peer-follower"
PEER_INGEST_SEQUENCE = 99


def _observed_contract(row: dict[str, Any]) -> SchemaContract:
    fields = tuple(
        FieldContract(normalized_name=key, original_name=key, python_type=object, required=False, source="inferred") for key in row
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


class _PeerSimulatingSource(_TestSourceBase):
    """Source that, at its own EOF instant, writes the journal rows a slow
    PEER worker would hold: row + token + READY enqueue + LEASED claim under
    a foreign ``lease_owner`` (production writers only). Optionally the peer
    "completes into the barrier" immediately (LEASED→BLOCKED via the
    production ``mark_blocked``)."""

    name = "peer_simulating_source"
    output_schema = ListSource.output_schema

    def __init__(
        self,
        rows: list[dict[str, int]],
        *,
        on_success: str,
        db: LandscapeDB,
        payload_store: FilesystemPayloadStore,
        craft_peer_row: bool,
        peer_completes_into_barrier: bool,
    ) -> None:
        super().__init__()
        self._rows = rows
        self.on_success = on_success
        self._db = db
        self._payload_store = payload_store
        self._craft_peer_row = craft_peer_row
        self._peer_completes_into_barrier = peer_completes_into_barrier
        self.load_invocations = 0
        self.peer_token_id: str | None = None

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        self.load_invocations += 1
        for source_row_index, row in enumerate(self._rows):
            contract = _observed_contract(row)
            self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)
        # Generator epilogue: runs when the engine pulls past the last row —
        # i.e. at the exact source-EOF instant, BEFORE the engine's
        # end-of-input work. This is where the slow peer's in-flight claim
        # becomes visible to the §D step-2 gate.
        if self._craft_peer_row:
            self._write_peer_journal_rows()

    def _write_peer_journal_rows(self) -> None:
        now = datetime.now(UTC)
        with self._db.connection() as conn:
            run_id = str(conn.execute(select(runs_table.c.run_id)).scalar_one())
            source_node_id = str(
                conn.execute(select(rows_table.c.source_node_id).where(rows_table.c.run_id == run_id).limit(1)).scalar_one()
            )
            # The engine's own buffered members show the journal shape to mirror.
            sample = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == "blocked")
                    .limit(1)
                )
                .mappings()
                .one()
            )
        factory = RecorderFactory(self._db, payload_store=self._payload_store)
        repo = TokenSchedulerRepository(self._db.engine)
        data = {"value": 99}
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=PEER_INGEST_SEQUENCE,
            data=data,
            source_row_index=PEER_INGEST_SEQUENCE,
            ingest_sequence=PEER_INGEST_SEQUENCE,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        self.peer_token_id = token.token_id
        repo.enqueue_ready(
            run_id=run_id,
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=str(sample["node_id"]),
            step_index=int(sample["step_index"]),
            ingest_sequence=PEER_INGEST_SEQUENCE,
            row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
            available_at=now,
        )
        claimed = repo.claim_ready(run_id=run_id, lease_owner=PEER_OWNER, lease_seconds=3600, now=now)
        assert claimed is not None and claimed.token_id == token.token_id
        if self._peer_completes_into_barrier:
            repo.mark_blocked(
                work_item_id=claimed.work_item_id,
                queue_key=None,
                barrier_key=str(sample["barrier_key"]),
                now=now,
                expected_lease_owner=PEER_OWNER,
            )


def _env(tmp_path: Any) -> tuple[LandscapeDB, FilesystemPayloadStore, CheckpointManager, RuntimeCheckpointConfig]:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    checkpoint_mgr = CheckpointManager(db)
    checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
    return db, payload_store, checkpoint_mgr, checkpoint_config


def _journal(db: LandscapeDB) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [dict(row) for row in conn.execute(select(token_work_items_table)).mappings()]


def _batches(db: LandscapeDB) -> list[dict[str, Any]]:
    with db.connection() as conn:
        return [dict(row) for row in conn.execute(select(batches_table)).mappings()]


@pytest.mark.timeout(120)
class TestEofFlushQuiescenceGating:
    def test_in_flight_peer_lease_at_eof_refuses_the_flush_then_single_batch_on_resume(self, tmp_path: Any) -> None:
        """The full §D story in one run+resume:

        Phase 1 (step 2, refusal): a foreign LEASED row exists at EOF — the
        flush is REFUSED before completing any batch or emitting any
        PENDING_SINK from the barrier; the run fails loudly.

        Phase 2 (the peer completes into the barrier + resume): the row goes
        BLOCKED via production ``mark_blocked``; the resume leader inherits
        it intake-pending, adopts it journal-first, and the §D step-3 loop
        produces exactly ONE batch containing ALL four members. COMPLETED in
        one resume pass, no second FAILED/resume cycle.
        """
        db, payload_store, checkpoint_mgr, checkpoint_config = _env(tmp_path)
        source = _PeerSimulatingSource(
            [{"value": 10}, {"value": 20}, {"value": 30}],
            on_success="batch_in",
            db=db,
            payload_store=payload_store,
            craft_peer_row=True,
            peer_completes_into_barrier=False,
        )
        transform = _SumBatchTransform()
        sink = CollectSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, transform, sink)
        orchestrator = Orchestrator(db=db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        with pytest.raises(OrchestrationInvariantError, match="End-of-input barrier flush refused"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        # (1) The EOF flush did NOT run while the LEASED row existed: no
        # batch completed, no PENDING_SINK emitted from the barrier, the
        # engine's three members still BLOCKED, the peer's claim untouched.
        assert transform.batch_calls == 0
        assert sink.results == []
        assert [b for b in _batches(db) if b["status"] == "completed"] == []
        journal = _journal(db)
        assert [row for row in journal if row["status"] == "pending_sink"] == []
        blocked = [row for row in journal if row["status"] == "blocked"]
        assert len(blocked) == 3, "the engine's own members are journal-held, nothing consumed"
        (peer_row,) = [row for row in journal if row["status"] == "leased"]
        assert peer_row["lease_owner"] == PEER_OWNER
        assert peer_row["token_id"] == source.peer_token_id

        # ── Phase 2: the slow peer completes into the barrier... ───────────
        repo = TokenSchedulerRepository(db.engine)
        now = datetime.now(UTC)
        repo.mark_blocked(
            work_item_id=str(peer_row["work_item_id"]),
            queue_key=None,
            barrier_key=str(blocked[0]["barrier_key"]),
            now=now,
            expected_lease_owner=PEER_OWNER,
        )

        # ...and the resume leader finishes the run in ONE pass.
        recovery = RecoveryManager(db, checkpoint_mgr)
        check = recovery.can_resume(str(peer_row["run_id"]), graph)
        assert check.can_resume, f"expected resumable run, got: {check.reason}"
        resume_point = recovery.get_resume_point(str(peer_row["run_id"]), graph)
        assert resume_point is not None
        result = orchestrator.resume(resume_point=resume_point, config=config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        # (2) Exactly ONE completed batch with ALL members, late peer included.
        completed_batches = [b for b in _batches(db) if b["status"] == "completed"]
        assert len(completed_batches) == 1, "the late arrival joins the SINGLE final batch — never a second batch"
        assert transform.batch_calls == 1
        assert sink.results == [{"value": 159, "count": 4}], "10+20+30 plus the peer's 99, in one aggregate"
        # (3) No residue: every journal row terminal, source never re-invoked.
        assert {row["status"] for row in _journal(db)} == {"terminal"}
        assert source.load_invocations == 1

    def test_quiescent_journal_at_eof_flushes_immediately_negative_control(self, tmp_path: Any) -> None:
        """(4) With no in-flight rows the EOF flush proceeds in the same pass —
        the gate changes nothing at quiescent N=1 (protects the battery)."""
        db, payload_store, checkpoint_mgr, checkpoint_config = _env(tmp_path)
        source = _PeerSimulatingSource(
            [{"value": 10}, {"value": 20}, {"value": 30}],
            on_success="batch_in",
            db=db,
            payload_store=payload_store,
            craft_peer_row=False,
            peer_completes_into_barrier=False,
        )
        transform = _SumBatchTransform()
        sink = CollectSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, transform, sink)
        orchestrator = Orchestrator(db=db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED
        assert transform.batch_calls == 1
        assert sink.results == [{"value": 60, "count": 3}]
        completed_batches = [b for b in _batches(db) if b["status"] == "completed"]
        assert len(completed_batches) == 1
        assert {row["status"] for row in _journal(db)} == {"terminal"}
        assert source.load_invocations == 1
