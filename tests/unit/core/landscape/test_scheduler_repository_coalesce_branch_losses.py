"""Slice 3 (ADR-030 §E.5): durable coalesce branch-loss hand-off verbs.

``record_coalesce_branch_loss`` rides the CALLER's lease-fenced disposition
transaction (record-then-notify uniformity rule); rows are append-only with an
``adopted_epoch`` replay cursor — never deleted, never consumed destructively.
``list_unadopted_coalesce_branch_losses`` is the leader's per-iteration replay
read; ``adopt_coalesce_branch_losses`` is the fenced cursor mark;
``list_coalesce_branch_losses`` is the §E.4 takeover-restore full read.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, insert, select, update

from elspeth.contracts import NodeType, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunLeadershipLostError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, begin_write
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import (
    BranchLossSpec,
    TokenSchedulerRepository,
    record_coalesce_branch_loss,
)
from elspeth.core.landscape.schema import (
    coalesce_branch_losses_table,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    runs_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

RUN_ID = "run-loss-1"
WORKER = f"worker:{RUN_ID}:deadbeef"
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
NODE_ID = "transform-1"
SOURCE_NODE_ID = "source-1"


def _payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


@pytest.fixture
def token(db: LandscapeDB) -> CoordinationToken:
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="v1",
                status=RunStatus.RUNNING.value,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type in ((SOURCE_NODE_ID, NodeType.SOURCE), (NODE_ID, NodeType.TRANSFORM)):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name="test",
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="cfg",
                    config_json="{}",
                    registered_at=NOW,
                )
            )
    return RunCoordinationRepository(db.engine).register_run_leader(run_id=RUN_ID, worker_id=WORKER, now=NOW, window_seconds=80.0)


def _bump_epoch(db: LandscapeDB) -> None:
    with db.engine.begin() as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == RUN_ID)
            .values(leader_epoch=run_coordination_table.c.leader_epoch + 1)
        )


def _fence_refusals(db: LandscapeDB, verb: str) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        rows = (
            conn.execute(
                select(run_coordination_events_table)
                .where(run_coordination_events_table.c.run_id == RUN_ID)
                .where(run_coordination_events_table.c.event_type == "fence_refusal")
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows if json.loads(str(row["context_json"])).get("verb") == verb]


def _loss_rows(db: LandscapeDB) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                select(coalesce_branch_losses_table).order_by(
                    coalesce_branch_losses_table.c.recorded_at, coalesce_branch_losses_table.c.loss_id
                )
            )
            .mappings()
            .all()
        ]


def _record(db: LandscapeDB, *, branch: str = "left", token_id: str = "tok-left", reason: str = "failed", now: datetime = NOW) -> bool:
    with begin_write(db.engine) as conn:
        return record_coalesce_branch_loss(
            conn,
            run_id=RUN_ID,
            coalesce_name="merge",
            row_id="row-1",
            branch_name=branch,
            token_id=token_id,
            reason=reason,
            recorded_by=WORKER,
            now=now,
        )


class TestRecordCoalesceBranchLoss:
    def test_insert_then_idempotent_reinsert(self, db: LandscapeDB, token: CoordinationToken) -> None:
        assert _record(db) is True
        assert _record(db) is False, "natural-key idempotency: second record is a no-op"
        rows = _loss_rows(db)
        assert len(rows) == 1
        (row,) = rows
        assert row["loss_id"].startswith("loss_")
        assert row["adopted_epoch"] is None
        assert row["reason"] == "failed"
        assert row["recorded_by"] == WORKER

    def test_token_id_mismatch_is_tier1(self, db: LandscapeDB, token: CoordinationToken) -> None:
        assert _record(db, token_id="tok-left") is True
        with pytest.raises(AuditIntegrityError, match="two distinct tokens"):
            _record(db, token_id="tok-other")
        assert len(_loss_rows(db)) == 1

    def test_reason_mismatch_is_tolerated_first_record_wins(self, db: LandscapeDB, token: CoordinationToken) -> None:
        assert _record(db, reason="failed") is True
        assert _record(db, reason="quarantined") is False
        rows = _loss_rows(db)
        assert len(rows) == 1
        assert rows[0]["reason"] == "failed", "the first durable record wins"

    def test_rides_caller_transaction_rollback_loses_the_record(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """§E.5: the loss commits iff the caller's disposition commits."""
        with pytest.raises(RuntimeError, match="disposition failed"), begin_write(db.engine) as conn:
            inserted = record_coalesce_branch_loss(
                conn,
                run_id=RUN_ID,
                coalesce_name="merge",
                row_id="row-1",
                branch_name="left",
                token_id="tok-left",
                reason="failed",
                recorded_by=WORKER,
                now=NOW,
            )
            assert inserted is True
            raise RuntimeError("disposition failed")
        assert _loss_rows(db) == [], "the loss rolled back with the caller's transaction"


class TestReplayAndAdoption:
    def test_list_unadopted_filters_and_orders(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        assert _record(db, branch="left", token_id="tok-left", now=NOW) is True
        assert _record(db, branch="right", token_id="tok-right", now=NOW + timedelta(seconds=5)) is True

        unadopted = repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID)
        assert [(loss.branch_name, loss.adopted_epoch) for loss in unadopted] == [("left", None), ("right", None)]
        assert unadopted[0].recorded_at == NOW
        assert unadopted[1].recorded_at == NOW + timedelta(seconds=5)

        marked = repo.adopt_coalesce_branch_losses(
            run_id=RUN_ID, loss_ids=(unadopted[0].loss_id,), now=NOW + timedelta(seconds=10), coordination_token=token
        )
        assert marked == 1

        remaining = repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID)
        assert [loss.branch_name for loss in remaining] == ["right"], "adopted rows leave the replay read"
        # §E.4 restore read returns EVERYTHING, adopted or not.
        full = repo.list_coalesce_branch_losses(run_id=RUN_ID)
        assert [(loss.branch_name, loss.adopted_epoch) for loss in full] == [("left", token.leader_epoch), ("right", None)]

    def test_adopt_is_idempotent_second_mark_is_zero(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        _record(db)
        (loss,) = repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID)
        assert repo.adopt_coalesce_branch_losses(run_id=RUN_ID, loss_ids=(loss.loss_id,), now=NOW, coordination_token=token) == 1
        assert repo.adopt_coalesce_branch_losses(run_id=RUN_ID, loss_ids=(loss.loss_id,), now=NOW, coordination_token=token) == 0
        assert _loss_rows(db)[0]["adopted_epoch"] == token.leader_epoch

    def test_adopt_empty_loss_ids_is_zero_without_fencing(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        assert repo.adopt_coalesce_branch_losses(run_id=RUN_ID, loss_ids=(), now=NOW, coordination_token=token) == 0

    def test_stale_token_adopt_refused_with_fence_refusal(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        _record(db)
        (loss,) = repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID)
        _bump_epoch(db)

        with pytest.raises(RunLeadershipLostError):
            repo.adopt_coalesce_branch_losses(run_id=RUN_ID, loss_ids=(loss.loss_id,), now=NOW, coordination_token=token)

        assert _loss_rows(db)[0]["adopted_epoch"] is None, "a deposed leader cannot move the replay cursor"
        refusals = _fence_refusals(db, "adopt_coalesce_branch_losses")
        assert len(refusals) == 1
        assert refusals[0]["leader_epoch"] == token.leader_epoch


class TestDispositionComposition:
    """The §E.5 composition hook: branch_loss rides the disposition verb's txn."""

    def _leased_item(self, db: LandscapeDB) -> tuple[TokenSchedulerRepository, str]:
        repo = TokenSchedulerRepository(db.engine)
        with db.engine.begin() as conn:
            conn.execute(
                insert(rows_table).values(
                    row_id="row-1",
                    run_id=RUN_ID,
                    source_node_id=SOURCE_NODE_ID,
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="hash-row-1",
                    created_at=NOW,
                )
            )
            conn.execute(insert(tokens_table).values(token_id="tok-left", row_id="row-1", run_id=RUN_ID, created_at=NOW))
        repo.enqueue_ready(
            run_id=RUN_ID,
            token_id="tok-left",
            row_id="row-1",
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=0,
            row_payload_json=_payload_json(),
            available_at=NOW,
        )
        claimed = repo.claim_ready(run_id=RUN_ID, lease_owner=WORKER, lease_seconds=60, now=NOW)
        assert claimed is not None
        return repo, claimed.work_item_id

    def _spec(self) -> BranchLossSpec:
        return BranchLossSpec(
            coalesce_name="merge",
            row_id="row-1",
            branch_name="left",
            token_id="tok-left",
            reason="failed",
            recorded_by=WORKER,
        )

    def test_mark_failed_with_branch_loss_commits_both(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, work_item_id = self._leased_item(db)

        item = repo.mark_failed(work_item_id=work_item_id, now=NOW, expected_lease_owner=WORKER, branch_loss=self._spec())

        assert item.status == TokenWorkStatus.FAILED
        rows = _loss_rows(db)
        assert len(rows) == 1
        assert rows[0]["token_id"] == "tok-left"
        assert rows[0]["adopted_epoch"] is None

    def test_failed_disposition_records_no_loss(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """The loss record commits iff the disposition commits: a refused
        disposition (lease-owner CAS miss) leaves no loss row."""
        repo, work_item_id = self._leased_item(db)

        with pytest.raises(AuditIntegrityError):
            repo.mark_failed(work_item_id=work_item_id, now=NOW, expected_lease_owner="some-other-worker", branch_loss=self._spec())

        assert _loss_rows(db) == []

    def test_mark_pending_sink_with_branch_loss_commits_both(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo, work_item_id = self._leased_item(db)

        item = repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=_payload_json(),
            sink_name="quarantine",
            outcome=TerminalOutcome.FAILURE.value,
            path=TerminalPath.ON_ERROR_ROUTED.value,
            error_hash="quarantined-error-hash",
            error_message="quarantined",
            now=NOW,
            expected_lease_owner=WORKER,
            branch_loss=BranchLossSpec(
                coalesce_name="merge",
                row_id="row-1",
                branch_name="left",
                token_id="tok-left",
                reason="quarantined",
                recorded_by=WORKER,
            ),
        )

        assert item.status == TokenWorkStatus.PENDING_SINK
        rows = _loss_rows(db)
        assert len(rows) == 1
        assert rows[0]["reason"] == "quarantined"


def test_loss_table_count_helper_sanity(db: LandscapeDB, token: CoordinationToken) -> None:
    """Append-only ledger sanity: records accumulate, nothing deletes them."""
    repo = TokenSchedulerRepository(db.engine)
    _record(db, branch="left", token_id="tok-left")
    _record(db, branch="right", token_id="tok-right", now=NOW + timedelta(seconds=1))
    losses = repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID)
    repo.adopt_coalesce_branch_losses(run_id=RUN_ID, loss_ids=[loss.loss_id for loss in losses], now=NOW, coordination_token=token)
    with db.engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(coalesce_branch_losses_table)).scalar_one()
    assert count == 2
    assert repo.list_unadopted_coalesce_branch_losses(run_id=RUN_ID) == []
    assert len(repo.list_coalesce_branch_losses(run_id=RUN_ID)) == 2
