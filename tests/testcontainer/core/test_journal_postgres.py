"""Real PostgreSQL commit-failure durability tests for the JSONL journal."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock, Thread
from typing import cast

import pytest
from sqlalchemy import Column, ForeignKey, Integer, MetaData, Table, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.core.landscape.journal import JournalRecord, LandscapeJournal
from elspeth.core.landscape.schema import sidecar_journal_outbox_table

pytestmark = pytest.mark.testcontainer


class _ConcurrentProbeJournal(LandscapeJournal):
    """Force unfenced PostgreSQL drains past publication checks together."""

    def __init__(self, path: str, barrier: Barrier) -> None:
        super().__init__(path, fail_on_error=True)
        self._barrier = barrier
        self._probe_lock = Lock()
        self._probe_used = False

    def _append_payload_locked(self, payload: str, record_count: int) -> bool:
        with self._probe_lock:
            probe = not self._probe_used
            self._probe_used = True
        if probe:
            with suppress(BrokenBarrierError):
                self._barrier.wait(timeout=1)
        return super()._append_payload_locked(payload, record_count)


@pytest.fixture
def postgres_engine() -> Iterator[Engine]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        engine = create_engine(postgres.get_connection_url())
        try:
            yield engine
        finally:
            engine.dispose()


def test_postgres_deferred_constraint_commit_failure_publishes_no_records(
    postgres_engine: Engine,
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    metadata = MetaData()
    parent = Table("journal_parent", metadata, Column("id", Integer, primary_key=True))
    child = Table(
        "journal_child",
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "parent_id",
            Integer,
            ForeignKey(parent.c.id, deferrable=True, initially="DEFERRED"),
            nullable=False,
        ),
    )
    metadata.create_all(postgres_engine)
    sidecar_journal_outbox_table.create(postgres_engine)
    journal = LandscapeJournal(str(journal_path), fail_on_error=True)
    journal.attach(postgres_engine)

    with postgres_engine.connect() as connection:
        transaction = connection.begin()
        connection.execute(child.insert().values(id=1, parent_id=999))
        with pytest.raises(SQLAlchemyIntegrityError):
            transaction.commit()
        connection.rollback()

    assert not journal_path.exists() or journal_path.read_text(encoding="utf-8") == ""
    with postgres_engine.connect() as connection:
        assert connection.scalar(select(child.c.id)) is None


def test_postgres_committed_batch_is_published_and_outbox_acknowledged(
    postgres_engine: Engine,
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    metadata = MetaData()
    rows = Table("journal_rows", metadata, Column("id", Integer, primary_key=True))
    metadata.create_all(postgres_engine)
    sidecar_journal_outbox_table.create(postgres_engine)
    journal = LandscapeJournal(str(journal_path), fail_on_error=True)
    journal.attach(postgres_engine)

    with postgres_engine.begin() as connection:
        connection.execute(rows.insert().values(id=1))

    records = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    with postgres_engine.connect() as connection:
        assert connection.execute(select(sidecar_journal_outbox_table)).all() == []


def test_postgres_concurrent_same_owner_recovery_publishes_batch_once(
    postgres_engine: Engine,
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "shared.jsonl"
    sidecar_journal_outbox_table.create(postgres_engine)
    barrier = Barrier(2)
    journal_a = _ConcurrentProbeJournal(str(journal_path), barrier)
    journal_b = _ConcurrentProbeJournal(str(journal_path), barrier)
    batch_id = "f" * 32
    record = cast(
        JournalRecord,
        {
            "timestamp": "2026-01-15T12:00:00+00:00",
            "statement": "INSERT INTO rows (id) VALUES (%s)",
            "parameters": ["row-1"],
            "executemany": False,
            "journal_batch_id": batch_id,
            "journal_batch_ordinal": 0,
            "journal_batch_size": 1,
        },
    )
    with postgres_engine.begin() as connection:
        connection.execute(
            sidecar_journal_outbox_table.insert().values(
                batch_id=batch_id,
                journal_owner=journal_a._owner_key,
                created_at=datetime.now(UTC),
                records_json=json.dumps([record]),
            )
        )

    url = postgres_engine.url.render_as_string(hide_password=False)
    engine_a = create_engine(url)
    engine_b = create_engine(url)
    journal_a.attach(engine_a)
    journal_b.attach(engine_b)
    errors: list[BaseException] = []

    def recover(journal: LandscapeJournal, engine: Engine) -> None:
        try:
            journal.recover_pending(engine)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        Thread(target=recover, args=(journal_a, engine_a)),
        Thread(target=recover, args=(journal_b, engine_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1
    with postgres_engine.connect() as connection:
        assert connection.execute(select(sidecar_journal_outbox_table)).all() == []
    engine_a.dispose()
    engine_b.dispose()
