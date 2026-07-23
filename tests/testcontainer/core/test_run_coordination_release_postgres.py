"""PostgreSQL proof for release-seat/takeover lock ordering (RC-04)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, insert, select
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts.coordination import CoordinationToken, mint_worker_id
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.schema import run_coordination_table, run_workers_table, runs_table

pytestmark = pytest.mark.testcontainer

NOW = datetime(2026, 7, 23, tzinfo=UTC)
RUN_ID = "release-seat-vs-takeover"


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


@pytest.mark.timeout(120)
def test_release_and_takeover_share_seat_then_membership_lock_order(postgres_url: str) -> None:
    """A release racing a takeover completes or loses silently, never deadlocks."""
    db = LandscapeDB.from_url(postgres_url)
    repo = RunCoordinationRepository(db.engine)
    with db.engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="failed",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
    incumbent_id = mint_worker_id(RUN_ID)
    token = repo.register_run_leader(run_id=RUN_ID, worker_id=incumbent_id, now=NOW, window_seconds=1)
    successor_id = mint_worker_id(RUN_ID)

    release_has_first_lock = threading.Event()
    release_attempting_seat = threading.Event()
    acquire_attempting_seat = threading.Event()
    acquire_has_seat = threading.Event()
    allow_release = threading.Event()
    allow_acquire = threading.Event()
    release_lock_kind: list[str] = []
    outcomes: dict[str, object] = {}

    def before_sql(_conn: Any, _cursor: Any, statement: str, _params: Any, _context: Any, _many: bool) -> None:
        normalized = " ".join(statement.upper().split())
        name = threading.current_thread().name
        if name == "release" and normalized.startswith("UPDATE RUN_COORDINATION"):
            release_attempting_seat.set()
        elif name == "acquire" and normalized.startswith("UPDATE RUN_COORDINATION"):
            acquire_attempting_seat.set()

    def after_sql(_conn: Any, _cursor: Any, statement: str, _params: Any, _context: Any, _many: bool) -> None:
        normalized = " ".join(statement.upper().split())
        name = threading.current_thread().name
        if name == "release" and not release_lock_kind:
            if normalized.startswith("SELECT") and "FROM RUN_WORKERS" in normalized and "FOR UPDATE" in normalized:
                release_lock_kind.append("membership")
            elif normalized.startswith("UPDATE RUN_COORDINATION"):
                release_lock_kind.append("seat")
            if release_lock_kind:
                release_has_first_lock.set()
                if not allow_release.wait(timeout=30):
                    raise TimeoutError("release interleaving gate timed out")
        elif name == "acquire" and normalized.startswith("UPDATE RUN_COORDINATION"):
            acquire_has_seat.set()
            if not allow_acquire.wait(timeout=30):
                raise TimeoutError("acquire interleaving gate timed out")

    event.listen(db.engine, "before_cursor_execute", before_sql)
    event.listen(db.engine, "after_cursor_execute", after_sql)

    def release() -> None:
        try:
            repo.release_seat(token=token, now=NOW + timedelta(seconds=3))
            outcomes["release"] = "returned"
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes["release"] = exc

    def acquire() -> None:
        try:
            outcomes["acquire"] = repo.acquire_run_leadership(
                run_id=RUN_ID,
                worker_id=successor_id,
                now=NOW + timedelta(seconds=3),
                window_seconds=30,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes["acquire"] = exc

    release_thread = threading.Thread(target=release, name="release")
    acquire_thread = threading.Thread(target=acquire, name="acquire")
    try:
        release_thread.start()
        assert release_has_first_lock.wait(timeout=30), "release never acquired its first coordination lock"
        acquire_thread.start()
        assert acquire_attempting_seat.wait(timeout=30), "takeover never attempted its seat CAS"

        if release_lock_kind == ["membership"]:
            assert acquire_has_seat.wait(timeout=30), "takeover never acquired the seat behind membership-first release"
            allow_release.set()
            assert release_attempting_seat.wait(timeout=30), "release never attempted the seat behind takeover"
            allow_acquire.set()
        else:
            assert release_lock_kind == ["seat"]
            allow_release.set()
            assert acquire_has_seat.wait(timeout=30), "takeover never acquired the seat after release committed"
            allow_acquire.set()

        release_thread.join(timeout=60)
        acquire_thread.join(timeout=60)
        assert not release_thread.is_alive() and not acquire_thread.is_alive(), "coordination race threads wedged"
    finally:
        allow_release.set()
        allow_acquire.set()
        if release_thread.ident is not None:
            release_thread.join(timeout=30)
        if acquire_thread.ident is not None:
            acquire_thread.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", before_sql)
        event.remove(db.engine, "after_cursor_execute", after_sql)

    try:
        assert outcomes["release"] == "returned"
        acquired = outcomes["acquire"]
        assert isinstance(acquired, CoordinationToken), f"takeover returned {acquired!r}"
        assert acquired.worker_id == successor_id
        with db.engine.connect() as conn:
            seat = conn.execute(
                select(run_coordination_table.c.leader_worker_id, run_coordination_table.c.leader_epoch).where(
                    run_coordination_table.c.run_id == RUN_ID
                )
            ).one()
            workers = dict(
                conn.execute(
                    select(run_workers_table.c.worker_id, run_workers_table.c.status).where(run_workers_table.c.run_id == RUN_ID)
                ).all()
            )
        assert tuple(seat) == (successor_id, 2)
        assert workers == {incumbent_id: "departed", successor_id: "active"}
    finally:
        db.close()
