"""RC6 storage-discipline verification on scheduler-bearing SQLite connections.

The token scheduler (``TokenSchedulerRepository``) persists its work queue in
``token_work_items`` / ``scheduler_events`` and depends on four SQLite PRAGMA
invariants for correctness: ``journal_mode=WAL`` (writer/reader concurrency),
``synchronous=NORMAL`` (crash-safe WAL pairing), ``foreign_keys=ON``
(composite-FK run ownership actually enforced), and ``busy_timeout=5000``
(claim races wait instead of failing fast).  ``LandscapeDB._configure_sqlite``
applies them per-connection and ``_verify_sqlite_pragmas`` probe-and-asserts
at engine open; ``TokenSchedulerRepository.__init__`` re-probes as defence in
depth.  These tests pin the discipline on the exact connection shapes the
scheduler uses (filigree elspeth-8536552dcb, elspeth-97f8509b35,
elspeth-addd3dc41f):

1. Every pooled connection — including concurrent ones and the
   ``engine.begin()`` write-transaction shape the repository uses — carries
   the full PRAGMA contract.
2. FK enforcement on scheduler tables is IMMEDIATE, not deferred: the schema
   declares no DEFERRABLE constraints and ``defer_foreign_keys`` stays 0, so
   an orphan work item is rejected at statement time, not commit time.
3. G27: ``claim_ready``'s SELECT + CAS-UPDATE transaction cannot double-lease
   under a second writer connection — the CAS WHERE re-verifies READY, so a
   lost race surfaces as ``None`` (graceful), never as two live leases.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import NodeType
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-rc6-pragma"
SCHEDULER_TABLES = ("token_work_items", "scheduler_events")


@pytest.fixture
def landscape_db(tmp_path: Path) -> Iterator[LandscapeDB]:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    yield db
    db.close()


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _insert_run_node_row_token(db: LandscapeDB, *, now: datetime) -> None:
    """Insert the minimal FK graph one READY work item needs."""
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=now,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin in (
            ("source-a", NodeType.SOURCE, "csv"),
            ("normalize", NodeType.TRANSFORM, "identity"),
        ):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=now,
                )
            )
        conn.execute(
            insert(rows_table).values(
                row_id="row-1",
                run_id=RUN_ID,
                source_node_id="source-a",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="hash-row-1",
                created_at=now,
            )
        )
        conn.execute(
            insert(tokens_table).values(
                token_id="token-1",
                row_id="row-1",
                run_id=RUN_ID,
                created_at=now,
            )
        )


def _enqueue_one_ready(repo: TokenSchedulerRepository, *, now: datetime) -> TokenWorkItem:
    return repo.enqueue_ready(
        run_id=RUN_ID,
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=0,
        ingest_sequence=0,
        row_payload_json=_row_payload_json(),
        available_at=now,
    )


def _assert_pragma_contract(conn: Connection) -> None:
    assert conn.execute(text("PRAGMA journal_mode")).scalar_one() == "wal"
    assert conn.execute(text("PRAGMA synchronous")).scalar_one() == 1
    assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    assert conn.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000


class TestSchedulerConnectionPragmaDiscipline:
    """The PRAGMA contract holds on every connection shape the scheduler uses."""

    def test_every_pooled_connection_carries_the_pragma_contract(self, landscape_db: LandscapeDB) -> None:
        """Three simultaneously open connections each honour all four PRAGMAs.

        Holding the connections open concurrently forces the pool to mint
        distinct DBAPI connections, proving the ``connect`` listener fires per
        connection — not just on the probe connection at engine open.
        """
        conns = [landscape_db.engine.connect() for _ in range(3)]
        try:
            distinct = {id(conn.connection.dbapi_connection) for conn in conns}
            assert len(distinct) == 3
            for conn in conns:
                _assert_pragma_contract(conn)
        finally:
            for conn in conns:
                conn.close()

    def test_write_transaction_connection_carries_the_pragma_contract(self, landscape_db: LandscapeDB) -> None:
        """The ``engine.begin()`` shape — the repository's only write path — is covered.

        Every mutating method on ``TokenSchedulerRepository`` opens
        ``self._engine.begin()``; this pins that the transaction-scoped
        connection carries the same contract as a bare ``connect()``.
        """
        with landscape_db.engine.begin() as conn:
            _assert_pragma_contract(conn)

    def test_foreign_key_enforcement_is_immediate_not_deferred(self, landscape_db: LandscapeDB) -> None:
        """Scheduler-table FKs enforce at statement time, not at COMMIT.

        The schema deliberately declares no DEFERRABLE constraints — the
        idempotent-insert reconcile path (``_insert_work_item_idempotent``)
        relies on ``IntegrityError`` surfacing at INSERT time.  Two facets:
        the DDL carries no DEFERRABLE clause, and ``defer_foreign_keys``
        (the per-connection override that would postpone enforcement) is off.
        """
        with landscape_db.engine.connect() as conn:
            assert conn.execute(text("PRAGMA defer_foreign_keys")).scalar_one() == 0
            for table in SCHEDULER_TABLES:
                ddl = conn.execute(
                    text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :name"),
                    {"name": table},
                ).scalar_one()
                assert "deferrable" not in ddl.lower(), f"{table} DDL declares a DEFERRABLE constraint: {ddl}"

    def test_orphan_work_item_insert_is_rejected_at_the_database(self, landscape_db: LandscapeDB) -> None:
        """``foreign_keys=ON`` is enforcement, not documentation.

        A raw INSERT into ``token_work_items`` referencing a token that does
        not exist must violate the composite ``(token_id, run_id)`` FK at the
        database layer — bypassing the repository's reference validation to
        prove the constraint itself is live on a scheduler-bearing connection.
        """
        now = datetime.now(UTC)
        _insert_run_node_row_token(landscape_db, now=now)
        with pytest.raises(IntegrityError), landscape_db.engine.begin() as conn:
            conn.execute(
                insert(token_work_items_table).values(
                    work_item_id="orphan-item",
                    run_id=RUN_ID,
                    token_id="token-does-not-exist",
                    row_id="row-1",
                    node_id="normalize",
                    step_index=0,
                    ingest_sequence=0,
                    row_payload_json=_row_payload_json(),
                    status=TokenWorkStatus.READY.value,
                    attempt=1,
                    available_at=now,
                    created_at=now,
                    updated_at=now,
                )
            )


class _RivalInterposingRepository(TokenSchedulerRepository):
    """Test seam: a rival connection claims between our SELECT and CAS-UPDATE.

    ``claim_ready`` SELECTs the next READY row, then ``_claim_ready_row``
    issues the CAS-UPDATE.  Overriding the seam lets a second writer
    connection complete a full claim in exactly that window, then delegates
    to the real production code — the SQL under test is unchanged.
    """

    def __init__(self, engine: object, rival: TokenSchedulerRepository, *, now: datetime) -> None:
        super().__init__(engine)  # type: ignore[arg-type]  # branded engine forwarded verbatim
        self._rival = rival
        self._now = now
        self.rival_claim: TokenWorkItem | None = None

    def _claim_ready_row(
        self,
        conn: Connection,
        *,
        row: RowMapping,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
        now: datetime,
    ) -> RowMapping | None:
        self.rival_claim = self._rival.claim_ready(
            run_id=run_id,
            lease_owner="owner-rival",
            lease_seconds=300,
            now=self._now,
        )
        return super()._claim_ready_row(
            conn,
            row=row,
            run_id=run_id,
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
            now=now,
        )


class TestClaimReadyUnderSecondWriterConnection:
    """G27: the SELECT + CAS-UPDATE claim cannot double-lease across connections."""

    def _open_second_db(self, db_path: Path) -> LandscapeDB:
        return LandscapeDB.from_url(f"sqlite:///{db_path}")

    def test_leased_item_is_not_reclaimable_from_a_second_connection(self, tmp_path: Path) -> None:
        """A live lease taken on connection A is refused to connection B.

        Two independent engines over the same file (the closest unit-level
        analogue to two processes) each pass the PRAGMA probe; once A holds
        the lease, B's claim returns ``None`` and exactly one CLAIM_READY
        event exists in the durable journal.
        """
        now = datetime.now(UTC)
        db_a = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
        try:
            _insert_run_node_row_token(db_a, now=now)
            repo_a = TokenSchedulerRepository(db_a.engine)
            _enqueue_one_ready(repo_a, now=now)

            db_b = self._open_second_db(tmp_path / "audit.db")
            try:
                repo_b = TokenSchedulerRepository(db_b.engine)

                claimed_a = repo_a.claim_ready(run_id=RUN_ID, lease_owner="owner-a", lease_seconds=300, now=now)
                assert claimed_a is not None
                assert claimed_a.lease_owner == "owner-a"

                claimed_b = repo_b.claim_ready(run_id=RUN_ID, lease_owner="owner-b", lease_seconds=300, now=now)
                assert claimed_b is None

                with db_a.engine.connect() as conn:
                    events = conn.execute(
                        select(scheduler_events_table.c.event_type, scheduler_events_table.c.to_lease_owner).where(
                            scheduler_events_table.c.run_id == RUN_ID
                        )
                    ).all()
                claim_events = [event for event in events if event.event_type == SchedulerEventType.CLAIM_READY.value]
                assert len(claim_events) == 1
                assert claim_events[0].to_lease_owner == "owner-a"
            finally:
                db_b.close()
        finally:
            db_a.close()

    def test_rival_claim_between_select_and_cas_update_loses_gracefully(self, tmp_path: Path) -> None:
        """A rival commit inside the SELECT→UPDATE window yields ``None``, not a double lease.

        The CAS-UPDATE re-verifies ``status == READY`` in its WHERE clause, so
        even though ``engine.begin()`` opens a DEFERRED transaction (pysqlite
        emits BEGIN lazily, so the SELECT holds no read lock), the stolen row
        matches zero rows and the loser backs off.  Exactly one lease and one
        CLAIM_READY event exist afterward — the work item is never executed
        twice and never wedged.
        """
        now = datetime.now(UTC)
        db_a = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
        try:
            _insert_run_node_row_token(db_a, now=now)
            db_b = self._open_second_db(tmp_path / "audit.db")
            try:
                rival = TokenSchedulerRepository(db_b.engine)
                repo_a = _RivalInterposingRepository(db_a.engine, rival, now=now)
                _enqueue_one_ready(repo_a, now=now)

                claimed_a = repo_a.claim_ready(run_id=RUN_ID, lease_owner="owner-a", lease_seconds=300, now=now)

                assert claimed_a is None
                assert repo_a.rival_claim is not None
                assert repo_a.rival_claim.lease_owner == "owner-rival"

                with db_a.engine.connect() as conn:
                    item = conn.execute(select(token_work_items_table).where(token_work_items_table.c.run_id == RUN_ID)).mappings().one()
                    events = conn.execute(
                        select(scheduler_events_table.c.event_type, scheduler_events_table.c.to_lease_owner).where(
                            scheduler_events_table.c.run_id == RUN_ID
                        )
                    ).all()
                assert item["status"] == TokenWorkStatus.LEASED.value
                assert item["lease_owner"] == "owner-rival"
                assert item["attempt"] == 1
                claim_events = [event for event in events if event.event_type == SchedulerEventType.CLAIM_READY.value]
                assert len(claim_events) == 1
                assert claim_events[0].to_lease_owner == "owner-rival"
            finally:
                db_b.close()
        finally:
            db_a.close()
