"""PostgreSQL serialization proof: membership fencing vs worker eviction.

Ticket elspeth-6903f82511. The membership fence rides the claim/heartbeat CAS
UPDATE as an EXISTS predicate over ``run_workers``; MVCC predicate reads take
no row locks, so before the fix a fenced claim or renewal could commit around
an in-flight eviction that had already observed no live lease — leaving an
evicted worker holding a live lease.  These tests drive the exact
interleavings on a real PostgreSQL backend:

* an eviction paused with its registry UPDATE uncommitted must BLOCK a fenced
  claim / heartbeat renewal, which then observes the committed eviction and
  is refused with ``RunWorkerEvictedError`` (no lease granted or renewed);
* an in-flight fenced claim (shared membership lock held, lease uncommitted)
  must BLOCK ``evict_worker``, whose live-lease precondition then sees the
  committed lease and returns ``False`` (no eviction).

Interleavings are event-fenced (pause hooks on cursor execution); the only
polling is a bounded wait on ``pg_stat_activity`` for the deterministic
"blocked on a row lock" state — never a timing assumption.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, insert, select
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
)
from elspeth.contracts.errors import RunWorkerEvictedError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)

pytestmark = pytest.mark.testcontainer

WINDOW = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
GRACE = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        yield postgres.get_connection_url()


def _seed(
    engine: Any,
    *,
    run_id: str,
    leader_id: str,
    worker_id: str,
    worker_heartbeat_expires_at: datetime,
    now: datetime,
) -> CoordinationToken:
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=run_id,
                started_at=now,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin in (("source-a", "source", "csv"), ("transform-1", "transform", "identity")):
            conn.execute(
                insert(nodes_table).values(
                    run_id=run_id,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=now,
                )
            )
        conn.execute(
            insert(run_coordination_table).values(
                run_id=run_id,
                leader_worker_id=leader_id,
                leader_epoch=1,
                leader_heartbeat_expires_at=now + timedelta(seconds=WINDOW),
                updated_at=now,
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=leader_id,
                run_id=run_id,
                role="leader",
                status="active",
                registered_at=now,
                heartbeat_expires_at=now + timedelta(seconds=WINDOW),
            )
        )
        conn.execute(
            insert(run_workers_table).values(
                worker_id=worker_id,
                run_id=run_id,
                role="follower",
                status="active",
                registered_at=now,
                heartbeat_expires_at=worker_heartbeat_expires_at,
            )
        )
    return CoordinationToken(run_id=run_id, worker_id=leader_id, leader_epoch=1)


def _enqueue_ready_item(engine: Any, *, run_id: str, token_id: str, now: datetime) -> None:
    row_id = f"row-{token_id}"
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=run_id,
                source_node_id="source-a",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash=f"hash-{token_id}",
                created_at=now,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=run_id, created_at=now))
    repo = TokenSchedulerRepository(engine)
    payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    repo.enqueue_ready(
        run_id=run_id,
        token_id=token_id,
        row_id=row_id,
        node_id="transform-1",
        step_index=1,
        ingest_sequence=0,
        row_payload_json=payload,
        available_at=now,
    )


def _backend_pid(conn: Any) -> int:
    return int(conn.connection.driver_connection.info.backend_pid)


def _await_done_or_lock_wait(
    db: LandscapeDB,
    *,
    done: threading.Event,
    pid_holder: dict[str, int],
    pid_key: str,
    timeout: float = 30.0,
) -> str:
    """Wait until the observed thread finished OR its backend blocks on a row lock.

    Returns ``"done"`` or ``"lock_wait"``.  Bounded condition poll on
    ``pg_stat_activity`` — the loop exits on an observed state, never on time.
    """
    deadline = time.monotonic() + timeout
    with db.engine.connect() as monitor:
        while time.monotonic() < deadline:
            if done.is_set():
                return "done"
            pid = pid_holder.get(pid_key)
            if pid is not None:
                wait_event_type = monitor.exec_driver_sql(
                    "SELECT wait_event_type FROM pg_stat_activity WHERE pid = %(pid)s",
                    {"pid": pid},
                ).scalar()
                if wait_event_type == "Lock":
                    return "lock_wait"
            time.sleep(0.02)
    raise AssertionError(f"thread neither finished nor blocked on a lock within {timeout}s")


def _worker_status_and_live_leases(db: LandscapeDB, *, run_id: str, worker_id: str, now: datetime) -> tuple[str, list[str]]:
    with db.read_only_connection() as conn:
        status = conn.execute(
            select(run_workers_table.c.status).where(
                run_workers_table.c.worker_id == worker_id,
                run_workers_table.c.run_id == run_id,
            )
        ).scalar_one()
        leases = (
            conn.execute(
                select(token_work_items_table.c.work_item_id, token_work_items_table.c.lease_expires_at)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status == "leased")
                .where(token_work_items_table.c.lease_owner == worker_id)
            )
            .mappings()
            .all()
        )
    live = [str(lease["work_item_id"]) for lease in leases if lease["lease_expires_at"] and lease["lease_expires_at"] > now]
    return str(status), live


def _pause_evictor_after_registry_update(
    pids: dict[str, int],
    paused: threading.Event,
    release: threading.Event,
) -> Callable[..., None]:
    def hook(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        if threading.current_thread().name != "evictor":
            return
        pids.setdefault("evictor", _backend_pid(conn))
        if " ".join(statement.upper().split()).startswith("UPDATE RUN_WORKERS"):
            paused.set()
            if not release.wait(timeout=30):
                raise TimeoutError("test never released the eviction transaction")

    return hook


def _record_thread_pid(pids: dict[str, int], thread_name: str, key: str) -> Callable[..., None]:
    def hook(conn, _cursor, _statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        if threading.current_thread().name == thread_name:
            pids.setdefault(key, _backend_pid(conn))

    return hook


@pytest.mark.timeout(120)
def test_fenced_claim_blocks_behind_in_flight_eviction_and_is_refused(postgres_url: str) -> None:
    """A claim racing an uncommitted eviction must observe it and be refused."""
    now = datetime(2026, 7, 16, 10, 0, 0, tzinfo=UTC)
    run_id = "run-claim-vs-evict"
    worker_id = "worker-claim"
    db = LandscapeDB.from_url(postgres_url)
    token = _seed(
        db.engine,
        run_id=run_id,
        leader_id="leader-claim",
        worker_id=worker_id,
        worker_heartbeat_expires_at=now - timedelta(seconds=GRACE + 10),
        now=now,
    )
    _enqueue_ready_item(db.engine, run_id=run_id, token_id="tok-claim", now=now)
    scheduler = TokenSchedulerRepository(db.engine)
    coord = RunCoordinationRepository(db.engine)

    pids: dict[str, int] = {}
    evictor_paused = threading.Event()
    release_evictor = threading.Event()
    claimant_done = threading.Event()
    results: dict[str, object] = {}

    pause_hook = _pause_evictor_after_registry_update(pids, evictor_paused, release_evictor)
    pid_hook = _record_thread_pid(pids, "claimant", "claimant")
    event.listen(db.engine, "after_cursor_execute", pause_hook)
    event.listen(db.engine, "before_cursor_execute", pid_hook)

    def evict() -> None:
        try:
            results["evict"] = coord.evict_worker(
                token=token,
                target_worker_id=worker_id,
                now=now,
                grace_seconds=GRACE,
                window_seconds=WINDOW,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            results["evict"] = f"RAISED {type(exc).__name__}: {exc}"

    def claim() -> None:
        try:
            results["claim"] = scheduler.claim_ready(run_id=run_id, lease_owner=worker_id, lease_seconds=300, now=now)
        except BaseException as exc:
            results["claim"] = exc
        finally:
            claimant_done.set()

    evictor = threading.Thread(target=evict, name="evictor")
    claimant = threading.Thread(target=claim, name="claimant")
    try:
        evictor.start()
        assert evictor_paused.wait(timeout=30), "eviction never reached its registry UPDATE"
        claimant.start()
        state = _await_done_or_lock_wait(db, done=claimant_done, pid_holder=pids, pid_key="claimant")
        assert state == "lock_wait", (
            "the fenced claim must BLOCK behind the in-flight eviction's "
            f"registry row lock, but it {state} with result {results.get('claim')!r} "
            "— the membership fence did not serialize with eviction "
            "(elspeth-6903f82511)"
        )
        release_evictor.set()
        evictor.join(timeout=60)
        claimant.join(timeout=60)
        assert not evictor.is_alive() and not claimant.is_alive(), "race threads wedged"
    finally:
        release_evictor.set()
        if evictor.ident is not None:
            evictor.join(timeout=30)
        if claimant.ident is not None:
            claimant.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", pid_hook)
        event.remove(db.engine, "after_cursor_execute", pause_hook)

    try:
        assert results["evict"] is True, f"eviction must commit, got: {results['evict']!r}"
        assert isinstance(results["claim"], RunWorkerEvictedError), (
            f"the racing claim must be refused with RunWorkerEvictedError, got: {results['claim']!r}"
        )
        status, live = _worker_status_and_live_leases(db, run_id=run_id, worker_id=worker_id, now=now)
        assert status == "evicted"
        assert live == [], f"an evicted worker must not hold a live lease, found: {live}"
    finally:
        db.close()


@pytest.mark.timeout(120)
def test_heartbeat_renewal_blocks_behind_in_flight_eviction_and_is_refused(postgres_url: str) -> None:
    """A renewal racing an uncommitted eviction must observe it and be refused."""
    t0 = datetime(2026, 7, 16, 11, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(seconds=GRACE + 100)
    run_id = "run-heartbeat-vs-evict"
    worker_id = "worker-heartbeat"
    db = LandscapeDB.from_url(postgres_url)
    token = _seed(
        db.engine,
        run_id=run_id,
        leader_id="leader-heartbeat",
        worker_id=worker_id,
        worker_heartbeat_expires_at=t0,  # stale by t1
        now=t0,
    )
    _enqueue_ready_item(db.engine, run_id=run_id, token_id="tok-heartbeat", now=t0)
    scheduler = TokenSchedulerRepository(db.engine)
    coord = RunCoordinationRepository(db.engine)
    item = scheduler.claim_ready(run_id=run_id, lease_owner=worker_id, lease_seconds=10, now=t0)
    assert item is not None  # lease is expired well before t1

    pids: dict[str, int] = {}
    evictor_paused = threading.Event()
    release_evictor = threading.Event()
    heartbeat_done = threading.Event()
    results: dict[str, object] = {}

    pause_hook = _pause_evictor_after_registry_update(pids, evictor_paused, release_evictor)
    pid_hook = _record_thread_pid(pids, "heartbeater", "heartbeater")
    event.listen(db.engine, "after_cursor_execute", pause_hook)
    event.listen(db.engine, "before_cursor_execute", pid_hook)

    def evict() -> None:
        try:
            results["evict"] = coord.evict_worker(
                token=token,
                target_worker_id=worker_id,
                now=t1,
                grace_seconds=GRACE,
                window_seconds=WINDOW,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            results["evict"] = f"RAISED {type(exc).__name__}: {exc}"

    def heartbeat() -> None:
        try:
            results["heartbeat"] = scheduler.heartbeat_lease(
                run_id=run_id,
                work_item_id=item.work_item_id,
                lease_owner=worker_id,
                lease_seconds=300,
                now=t1,
                membership_fenced=True,
            )
        except BaseException as exc:
            results["heartbeat"] = exc
        finally:
            heartbeat_done.set()

    evictor = threading.Thread(target=evict, name="evictor")
    heartbeater = threading.Thread(target=heartbeat, name="heartbeater")
    try:
        evictor.start()
        assert evictor_paused.wait(timeout=30), "eviction never reached its registry UPDATE"
        heartbeater.start()
        state = _await_done_or_lock_wait(db, done=heartbeat_done, pid_holder=pids, pid_key="heartbeater")
        assert state == "lock_wait", (
            "the fenced renewal must BLOCK behind the in-flight eviction's "
            f"registry row lock, but it {state} with result {results.get('heartbeat')!r} "
            "— the membership fence did not serialize with eviction "
            "(elspeth-6903f82511)"
        )
        release_evictor.set()
        evictor.join(timeout=60)
        heartbeater.join(timeout=60)
        assert not evictor.is_alive() and not heartbeater.is_alive(), "race threads wedged"
    finally:
        release_evictor.set()
        if evictor.ident is not None:
            evictor.join(timeout=30)
        if heartbeater.ident is not None:
            heartbeater.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", pid_hook)
        event.remove(db.engine, "after_cursor_execute", pause_hook)

    try:
        assert results["evict"] is True, f"eviction must commit, got: {results['evict']!r}"
        assert isinstance(results["heartbeat"], RunWorkerEvictedError), (
            f"the racing renewal must be refused with RunWorkerEvictedError, got: {results['heartbeat']!r}"
        )
        status, live = _worker_status_and_live_leases(db, run_id=run_id, worker_id=worker_id, now=t1)
        assert status == "evicted"
        assert live == [], f"an evicted worker must not retain a renewed lease, found: {live}"
    finally:
        db.close()


@pytest.mark.timeout(120)
def test_eviction_defers_to_in_flight_fenced_claim(postgres_url: str) -> None:
    """Eviction racing an uncommitted fenced claim must block, then skip.

    The mirror interleaving: the claimant holds its shared membership lock
    with the lease CAS uncommitted.  ``evict_worker`` must block on the
    registry row BEFORE its no-unexpired-leases precondition, then observe the
    committed lease and return False — never evict a worker that just
    acquired a live lease.
    """
    now = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
    run_id = "run-evict-vs-claim"
    worker_id = "worker-defer"
    db = LandscapeDB.from_url(postgres_url)
    token = _seed(
        db.engine,
        run_id=run_id,
        leader_id="leader-defer",
        worker_id=worker_id,
        worker_heartbeat_expires_at=now - timedelta(seconds=GRACE + 10),  # evictable heartbeat
        now=now,
    )
    _enqueue_ready_item(db.engine, run_id=run_id, token_id="tok-defer", now=now)
    scheduler = TokenSchedulerRepository(db.engine)
    coord = RunCoordinationRepository(db.engine)

    pids: dict[str, int] = {}
    claimant_paused = threading.Event()
    release_claimant = threading.Event()
    evictor_done = threading.Event()
    results: dict[str, object] = {}

    def pause_claimant_after_cas(conn, _cursor, statement, _parameters, _context, _executemany) -> None:  # type: ignore[no-untyped-def]
        if threading.current_thread().name != "claimant":
            return
        if " ".join(statement.upper().split()).startswith("UPDATE TOKEN_WORK_ITEMS"):
            claimant_paused.set()
            if not release_claimant.wait(timeout=30):
                raise TimeoutError("test never released the claim transaction")

    pid_hook = _record_thread_pid(pids, "evictor", "evictor")
    event.listen(db.engine, "after_cursor_execute", pause_claimant_after_cas)
    event.listen(db.engine, "before_cursor_execute", pid_hook)

    def claim() -> None:
        try:
            results["claim"] = scheduler.claim_ready(run_id=run_id, lease_owner=worker_id, lease_seconds=300, now=now)
        except BaseException as exc:  # pragma: no cover - asserted below
            results["claim"] = f"RAISED {type(exc).__name__}: {exc}"

    def evict() -> None:
        try:
            results["evict"] = coord.evict_worker(
                token=token,
                target_worker_id=worker_id,
                now=now,
                grace_seconds=GRACE,
                window_seconds=WINDOW,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            results["evict"] = f"RAISED {type(exc).__name__}: {exc}"
        finally:
            evictor_done.set()

    claimant = threading.Thread(target=claim, name="claimant")
    evictor = threading.Thread(target=evict, name="evictor")
    try:
        claimant.start()
        assert claimant_paused.wait(timeout=30), "claim never reached its CAS UPDATE"
        evictor.start()
        state = _await_done_or_lock_wait(db, done=evictor_done, pid_holder=pids, pid_key="evictor")
        assert state == "lock_wait", (
            "eviction must BLOCK behind the in-flight fenced claim's shared "
            f"membership lock, but it {state} with result {results.get('evict')!r} "
            "— evict_worker read its live-lease precondition without "
            "serializing with the fence (elspeth-6903f82511)"
        )
        release_claimant.set()
        claimant.join(timeout=60)
        evictor.join(timeout=60)
        assert not claimant.is_alive() and not evictor.is_alive(), "race threads wedged"
    finally:
        release_claimant.set()
        if claimant.ident is not None:
            claimant.join(timeout=30)
        if evictor.ident is not None:
            evictor.join(timeout=30)
        event.remove(db.engine, "before_cursor_execute", pid_hook)
        event.remove(db.engine, "after_cursor_execute", pause_claimant_after_cas)

    try:
        assert results["evict"] is False, f"eviction must defer to the committed live lease and skip, got: {results['evict']!r}"
        assert results["claim"] is not None and not isinstance(results["claim"], str), (
            f"the fenced claim must succeed, got: {results['claim']!r}"
        )
        status, live = _worker_status_and_live_leases(db, run_id=run_id, worker_id=worker_id, now=now)
        assert status == "active", "a worker holding a live lease must not be evicted"
        assert len(live) == 1
    finally:
        db.close()
