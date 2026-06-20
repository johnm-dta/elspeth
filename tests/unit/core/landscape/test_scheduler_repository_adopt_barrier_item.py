"""Slice 3 (ADR-030 §E.2): ``adopt_blocked_barrier_item`` — fenced, backdated adoption.

The journal-first intake verb: epoch fence, then the
``barrier_adopted_epoch NULL → epoch`` CAS, then (aggregation arm) the
``batch_members`` row and the BUFFERED ``token_outcomes`` row — ONE
``BEGIN IMMEDIATE`` transaction. ``token_outcomes`` has NO non-terminal
uniqueness (the only unique index is partial on ``completed=1``), so the
adoption CAS is the ONLY double-BUFFERED guard: the idempotent
``adopted=False`` skip is what makes the F2 duplicate-acceptance regression
structurally impossible.

Stale-epoch refusal mirrors tests/unit/core/landscape/test_leader_fence_stale_token.py:
raise :class:`RunLeadershipLostError` + exactly one ``fence_refusal`` event +
ZERO payload mutation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, insert, select, update

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunLeadershipLostError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import (
    BatchMembershipSpec,
    BufferedOutcomeSpec,
    TokenSchedulerRepository,
)
from elspeth.core.landscape.schema import (
    batch_members_table,
    batches_table,
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    runs_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

RUN_ID = "run-adopt-1"
WORKER = f"worker:{RUN_ID}:deadbeef"
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
NODE_ID = "transform-1"
SOURCE_NODE_ID = "source-1"
BARRIER_KEY = "agg-node-1"
BATCH_ID = "batch-1"


def _payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


@pytest.fixture
def token(db: LandscapeDB) -> CoordinationToken:
    """Seed a RUNNING run + nodes + a DRAFT batch; mint the epoch-1 seat."""
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
        conn.execute(
            insert(batches_table).values(
                batch_id=BATCH_ID,
                run_id=RUN_ID,
                aggregation_node_id=NODE_ID,
                attempt=0,
                status="draft",
                created_at=NOW,
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
                .order_by(run_coordination_events_table.c.seq)
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows if json.loads(str(row["context_json"])).get("verb") == verb]


def _seed_blocked_barrier_hold(db: LandscapeDB, *, sequence: int, barrier_key: str = BARRIER_KEY) -> tuple[str, str, datetime]:
    """READY → LEASED → BLOCKED barrier hold; returns (token_id, work_item_id, blocked_at)."""
    repo = TokenSchedulerRepository(db.engine)
    token_id = f"token-{sequence}"
    row_id = f"row-{sequence}"
    with db.engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id=SOURCE_NODE_ID,
                row_index=sequence,
                source_row_index=sequence,
                ingest_sequence=sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=RUN_ID, created_at=NOW))
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        row_payload_json=_payload_json(),
        available_at=NOW,
    )
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner=WORKER, lease_seconds=60, now=NOW)
    assert claimed is not None and claimed.token_id == token_id
    blocked_at = NOW + timedelta(seconds=2)
    repo.mark_blocked(
        work_item_id=claimed.work_item_id,
        queue_key=None,
        barrier_key=barrier_key,
        now=blocked_at,
        expected_lease_owner=WORKER,
    )
    return token_id, claimed.work_item_id, blocked_at


def _work_item(db: LandscapeDB, token_id: str) -> dict[str, object]:
    with db.engine.connect() as conn:
        row = conn.execute(select(token_work_items_table).where(token_work_items_table.c.token_id == token_id)).mappings().one()
    return dict(row)


def _batch_members(db: LandscapeDB) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        return [dict(r) for r in conn.execute(select(batch_members_table)).mappings().all()]


def _outcomes(db: LandscapeDB, token_id: str) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                select(token_outcomes_table)
                .where(token_outcomes_table.c.token_id == token_id)
                .order_by(token_outcomes_table.c.recorded_at, token_outcomes_table.c.outcome_id)
            )
            .mappings()
            .all()
        ]


def _count(db: LandscapeDB, table) -> int:  # type: ignore[no-untyped-def]
    with db.engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


ADOPT_NOW = NOW + timedelta(seconds=30)


def _adopt(db: LandscapeDB, token: CoordinationToken, *, token_id: str, work_item_id: str, aggregation: bool = True, **overrides):  # type: ignore[no-untyped-def]
    repo = TokenSchedulerRepository(db.engine)
    kwargs = {
        "run_id": RUN_ID,
        "work_item_id": work_item_id,
        "token_id": token_id,
        "barrier_key": BARRIER_KEY,
        "membership": BatchMembershipSpec(batch_id=BATCH_ID, ordinal=0) if aggregation else None,
        "buffered_outcome": BufferedOutcomeSpec(batch_id=BATCH_ID, context={"branch": "left"}) if aggregation else None,
        "now": ADOPT_NOW,
        "coordination_token": token,
    }
    kwargs.update(overrides)
    return repo.adopt_blocked_barrier_item(**kwargs)


class TestAdoption:
    def test_aggregation_arm_adopts_with_membership_and_backdated_buffered_outcome(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        result = _adopt(db, token, token_id=token_id, work_item_id=work_item_id)

        assert result.adopted is True
        assert result.barrier_adopted_epoch == token.leader_epoch
        assert result.outcome_id is not None

        item = _work_item(db, token_id)
        assert item["status"] == TokenWorkStatus.BLOCKED.value, "adoption marks, it does not transition"
        assert item["barrier_adopted_epoch"] == token.leader_epoch

        members = _batch_members(db)
        assert len(members) == 1
        assert members[0] == {"batch_id": BATCH_ID, "run_id": RUN_ID, "token_id": token_id, "ordinal": 0}

        outcomes = _outcomes(db, token_id)
        assert len(outcomes) == 1
        (outcome,) = outcomes
        assert outcome["outcome_id"] == result.outcome_id
        assert outcome["outcome"] is None
        assert outcome["path"] == "buffered"
        assert outcome["completed"] == 0
        assert outcome["batch_id"] == BATCH_ID
        recorded_at = outcome["recorded_at"]
        assert recorded_at.replace(tzinfo=UTC) == blocked_at, "§E.2 backdated accept: recorded_at == barrier_blocked_at, NOT now"
        context = json.loads(str(outcome["context_json"]))
        assert context["adopted_epoch"] == token.leader_epoch
        assert context["adopted_at"] == ADOPT_NOW.isoformat()
        assert context["branch"] == "left", "caller context merges in"

    def test_idempotent_readoption_skips_all_inserts(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """The F2 regression pin: re-adoption is a success-SKIP — no second
        batch_members row, no second BUFFERED outcome (the adoption CAS is the
        ONLY double-BUFFERED guard)."""
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)
        first = _adopt(db, token, token_id=token_id, work_item_id=work_item_id)
        assert first.adopted is True

        second = _adopt(db, token, token_id=token_id, work_item_id=work_item_id)

        assert second.adopted is False
        assert second.barrier_adopted_epoch == token.leader_epoch
        assert second.outcome_id is None
        assert len(_batch_members(db)) == 1, "no second batch_members row"
        assert len(_outcomes(db, token_id)) == 1, "no second BUFFERED outcome"

    def test_earlier_epoch_marker_is_still_idempotent_skip(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """Any non-NULL epoch counts as adopted: §E.4 treats adopted-at-any-epoch
        rows as restorable members; the intake filter is epoch-NULL."""
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)
        prior_epoch = token.leader_epoch  # pretend a prior leader at this epoch adopted
        with db.engine.begin() as conn:
            conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.token_id == token_id)
                .values(barrier_adopted_epoch=prior_epoch)
            )
        # A NEW epoch's leader (same worker identity, bumped seat epoch)
        # re-encounters the row (e.g. restore/intake re-entry).
        _bump_epoch(db)
        new_token = CoordinationToken(run_id=RUN_ID, worker_id=WORKER, leader_epoch=token.leader_epoch + 1)

        result = _adopt(db, new_token, token_id=token_id, work_item_id=work_item_id)

        assert result.adopted is False
        assert result.barrier_adopted_epoch == prior_epoch, "the prior adopter's epoch is reported, not ours"
        assert _batch_members(db) == []
        assert _outcomes(db, token_id) == []

    def test_coalesce_arm_is_cas_only(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0, barrier_key="coalesce:merge")

        repo = TokenSchedulerRepository(db.engine)
        result = repo.adopt_blocked_barrier_item(
            run_id=RUN_ID,
            work_item_id=work_item_id,
            token_id=token_id,
            barrier_key="coalesce:merge",
            membership=None,
            buffered_outcome=None,
            now=ADOPT_NOW,
            coordination_token=token,
        )

        assert result.adopted is True
        assert result.outcome_id is None
        assert _work_item(db, token_id)["barrier_adopted_epoch"] == token.leader_epoch
        assert _batch_members(db) == []
        assert _outcomes(db, token_id) == []


class TestAdoptionRefusals:
    def test_stale_token_refused_with_fence_refusal_and_zero_mutation(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)
        _bump_epoch(db)

        with pytest.raises(RunLeadershipLostError) as exc_info:
            _adopt(db, token, token_id=token_id, work_item_id=work_item_id)
        assert not isinstance(exc_info.value, AuditIntegrityError)

        item = _work_item(db, token_id)
        assert item["barrier_adopted_epoch"] is None, "CAS marker untouched"
        assert item["status"] == TokenWorkStatus.BLOCKED.value
        assert _batch_members(db) == [], "zero batch_members rows"
        assert _outcomes(db, token_id) == [], "zero token_outcomes rows"
        refusals = _fence_refusals(db, "adopt_blocked_barrier_item")
        assert len(refusals) == 1
        assert refusals[0]["worker_id"] == WORKER
        assert refusals[0]["leader_epoch"] == token.leader_epoch

    def test_row_not_blocked_is_tier1_with_zero_mutation(self, db: LandscapeDB, token: CoordinationToken) -> None:
        repo = TokenSchedulerRepository(db.engine)
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)
        # Terminalize it out from under the (hypothetical) intake listing.
        repo.mark_blocked_barrier_terminal(run_id=RUN_ID, barrier_key=BARRIER_KEY, token_ids=(token_id,), now=NOW)

        with pytest.raises(AuditIntegrityError, match="journal corruption"):
            _adopt(db, token, token_id=token_id, work_item_id=work_item_id)

        assert _batch_members(db) == []
        assert _outcomes(db, token_id) == []

    def test_missing_row_is_tier1(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, _work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        with pytest.raises(AuditIntegrityError, match="missing"):
            _adopt(db, token, token_id=token_id, work_item_id="no-such-work-item")

        assert _work_item(db, token_id)["barrier_adopted_epoch"] is None

    def test_wrong_barrier_key_is_tier1(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        with pytest.raises(AuditIntegrityError, match="journal corruption"):
            _adopt(db, token, token_id=token_id, work_item_id=work_item_id, barrier_key="some-other-barrier")

        assert _work_item(db, token_id)["barrier_adopted_epoch"] is None

    def test_mixed_specs_are_tier1(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        with pytest.raises(AuditIntegrityError, match="BOTH specs"):
            _adopt(db, token, token_id=token_id, work_item_id=work_item_id, buffered_outcome=None)
        with pytest.raises(AuditIntegrityError, match="BOTH specs"):
            _adopt(db, token, token_id=token_id, work_item_id=work_item_id, membership=None)

        assert _work_item(db, token_id)["barrier_adopted_epoch"] is None

    def test_batch_id_mismatch_between_specs_is_tier1(self, db: LandscapeDB, token: CoordinationToken) -> None:
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        with pytest.raises(AuditIntegrityError, match="exactly one batch"):
            _adopt(
                db,
                token,
                token_id=token_id,
                work_item_id=work_item_id,
                buffered_outcome=BufferedOutcomeSpec(batch_id="some-other-batch"),
            )

        assert _work_item(db, token_id)["barrier_adopted_epoch"] is None

    def test_crash_after_cas_rolls_back_everything(self, db: LandscapeDB, token: CoordinationToken) -> None:
        """Atomicity: a Tier-1 raised AFTER the CAS (missing batches row) rolls
        the whole transaction back — barrier_adopted_epoch stays NULL, the
        legitimate restore-reconcile disposition (crash-walk: leader crash
        mid-adoption). Mirrors test_complete_barrier_crash_atomicity."""
        token_id, work_item_id, _blocked_at = _seed_blocked_barrier_hold(db, sequence=0)

        with pytest.raises(AuditIntegrityError, match="not found"):
            _adopt(
                db,
                token,
                token_id=token_id,
                work_item_id=work_item_id,
                membership=BatchMembershipSpec(batch_id="batch-missing", ordinal=0),
                buffered_outcome=BufferedOutcomeSpec(batch_id="batch-missing"),
            )

        item = _work_item(db, token_id)
        assert item["barrier_adopted_epoch"] is None, "the CAS rolled back with the failed insert"
        assert item["status"] == TokenWorkStatus.BLOCKED.value
        assert _batch_members(db) == []
        assert _outcomes(db, token_id) == []
