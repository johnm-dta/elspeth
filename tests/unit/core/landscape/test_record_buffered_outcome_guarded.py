"""``record_buffered_outcome_guarded`` — the shared BUFFERED adoption writer.

The scheduler barrier's §E.2 adoption transaction records its backdated
BUFFERED ``token_outcomes`` row through this writer so the ADR-019
(NULL, BUFFERED) discriminator rule is enforced by the SAME validator as
``TokenOutcomeRepository.record_token_outcome`` — a future constraint change
cannot enforce in one path and silently not the other.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, select

from elspeth.contracts import NodeType, RunStatus
from elspeth.core.landscape.data_flow.outcomes import record_buffered_outcome_guarded
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    batches_table,
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db

RUN_ID = "run-buffered-guarded-1"
NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)
BLOCKED_AT = datetime(2026, 6, 12, 11, 59, 30, tzinfo=UTC)
NODE_ID = "agg-node-1"
SOURCE_NODE_ID = "source-1"
BATCH_ID = "batch-1"
TOKEN_ID = "token-1"
ROW_ID = "row-1"


@pytest.fixture
def db() -> LandscapeDB:
    db = make_landscape_db()
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
        conn.execute(
            insert(rows_table).values(
                row_id=ROW_ID,
                run_id=RUN_ID,
                source_node_id=SOURCE_NODE_ID,
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="hash-row-1",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=TOKEN_ID, row_id=ROW_ID, run_id=RUN_ID, created_at=NOW))
    return db


def _outcomes(db: LandscapeDB) -> list[dict[str, object]]:
    with db.engine.connect() as conn:
        return [dict(r) for r in conn.execute(select(token_outcomes_table)).mappings().all()]


class TestRecordBufferedOutcomeGuarded:
    def test_writes_backdated_buffered_row_in_caller_transaction(self, db: LandscapeDB) -> None:
        with db.engine.begin() as conn:
            outcome_id = record_buffered_outcome_guarded(
                conn,
                run_id=RUN_ID,
                token_id=TOKEN_ID,
                batch_id=BATCH_ID,
                recorded_at=BLOCKED_AT,
                context={"branch": "left", "adopted_epoch": 1},
            )

        outcomes = _outcomes(db)
        assert len(outcomes) == 1
        (outcome,) = outcomes
        assert outcome["outcome_id"] == outcome_id
        assert outcome["run_id"] == RUN_ID
        assert outcome["token_id"] == TOKEN_ID
        assert outcome["outcome"] is None
        assert outcome["path"] == "buffered"
        assert outcome["completed"] == 0
        assert outcome["batch_id"] == BATCH_ID
        recorded_at = outcome["recorded_at"]
        assert isinstance(recorded_at, datetime)
        assert recorded_at.replace(tzinfo=UTC) == BLOCKED_AT, "backdated accept: recorded_at is the caller's stamp, NOT now"
        assert json.loads(str(outcome["context_json"])) == {"branch": "left", "adopted_epoch": 1}

    def test_missing_batch_id_is_rejected_by_the_shared_adr019_validator(self, db: LandscapeDB) -> None:
        with pytest.raises(ValueError, match=r"requires batch_id"), db.engine.begin() as conn:
            record_buffered_outcome_guarded(
                conn,
                run_id=RUN_ID,
                token_id=TOKEN_ID,
                batch_id=None,  # type: ignore[arg-type]
                recorded_at=BLOCKED_AT,
            )

        assert _outcomes(db) == [], "the ADR-019 refusal must precede the audit write"

    def test_caller_rollback_discards_the_write(self, db: LandscapeDB) -> None:
        class _Boom(Exception):
            pass

        with pytest.raises(_Boom), db.engine.begin() as conn:
            record_buffered_outcome_guarded(
                conn,
                run_id=RUN_ID,
                token_id=TOKEN_ID,
                batch_id=BATCH_ID,
                recorded_at=BLOCKED_AT,
            )
            raise _Boom

        assert _outcomes(db) == [], "the write lives and dies with the caller's transaction"
