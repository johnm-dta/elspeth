"""Task 1.3 (F1 durability unification): ``barrier_blocked_at`` write + read verbs.

``mark_blocked`` stamps ``token_work_items.barrier_blocked_at`` (epoch 20
column, previously written by nothing), and ``list_blocked_barrier_items`` is
the resume-side read verb that sweeps barrier holds — and ONLY barrier holds —
back out of the journal.

D1 discrimination rule: BLOCKED rows are dual-use. Barrier holds carry a
non-NULL ``barrier_key``; ADR-028 queue-holds carry a ``queue_key`` with a
NULL ``barrier_key``. Every barrier query must filter ``barrier_key IS NOT
NULL``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import insert, select

from elspeth.contracts import NodeType
from elspeth.contracts.scheduler import TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)


def _make_scheduler_engine() -> Tier1Engine:
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _seed_run(engine: Tier1Engine, *, run_id: str, tokens: list[tuple[str, str, int]], now: datetime) -> str:
    """Insert run/node/row/token prerequisites; return a serialized row payload.

    ``tokens`` is a list of ``(row_id, token_id, ingest_sequence)`` triples.
    """
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    row_payload_json = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
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
        conn.execute(
            insert(nodes_table).values(
                run_id=run_id,
                node_id="source-0",
                plugin_name="csv",
                node_type=NodeType.SOURCE.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="config",
                config_json="{}",
                registered_at=now,
            )
        )
        conn.execute(
            insert(nodes_table).values(
                run_id=run_id,
                node_id="normalize",
                plugin_name="identity",
                node_type=NodeType.TRANSFORM.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="config",
                config_json="{}",
                registered_at=now,
            )
        )
        for index, (row_id, token_id, ingest_sequence) in enumerate(tokens):
            conn.execute(
                insert(rows_table).values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="source-0",
                    row_index=index,
                    source_row_index=index,
                    ingest_sequence=ingest_sequence,
                    source_data_hash=f"hash-{row_id}",
                    created_at=now,
                )
            )
            conn.execute(
                insert(tokens_table).values(
                    token_id=token_id,
                    row_id=row_id,
                    run_id=run_id,
                    created_at=now,
                )
            )
    return row_payload_json


def _enqueue_and_block(
    repo,
    *,
    run_id: str,
    token_id: str,
    row_id: str,
    ingest_sequence: int,
    payload: str,
    queue_key: str | None,
    barrier_key: str | None,
    now: datetime,
) -> TokenWorkItem:
    """Enqueue one READY item, claim it, and block it at a queue or barrier.

    Performed one item at a time so ``claim_ready`` cannot pick up an earlier
    seeded READY item.
    """
    item = repo.enqueue_ready(
        run_id=run_id,
        token_id=token_id,
        row_id=row_id,
        node_id="normalize",
        step_index=1,
        ingest_sequence=ingest_sequence,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id=run_id, lease_owner="w1", lease_seconds=30, now=now + timedelta(seconds=1))
    assert claimed is not None
    assert claimed.work_item_id == item.work_item_id
    return repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=queue_key,
        barrier_key=barrier_key,
        now=now + timedelta(seconds=2),
        expected_lease_owner="w1",
    )


def test_mark_blocked_stamps_barrier_blocked_at() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime(2026, 6, 11, 3, 0, tzinfo=UTC)
    payload = _seed_run(engine, run_id="run-1", tokens=[("row-1", "token-1", 0)], now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="w1", lease_seconds=30, now=now) is not None

    blocked = repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=None,
        barrier_key="agg-1",
        now=now,
        expected_lease_owner="w1",
    )

    assert blocked.status is TokenWorkStatus.BLOCKED
    assert blocked.barrier_blocked_at == now
    # Raw row check: SQLite's DateTime adapter stores tz-aware values naive.
    with engine.connect() as conn:
        raw = (
            conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == item.work_item_id)).mappings().one()
        )
    assert raw["status"] == "blocked"
    assert raw["barrier_blocked_at"] == now.replace(tzinfo=None)


def test_mark_blocked_queue_hold_is_stamped_but_never_swept_as_barrier() -> None:
    """Queue-holds get the stamp too (single UPDATE shape, nothing reads it on that arm)."""
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime(2026, 6, 11, 3, 0, tzinfo=UTC)
    payload = _seed_run(engine, run_id="run-1", tokens=[("row-1", "token-1", 0)], now=now)
    blocked = _enqueue_and_block(
        repo,
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        ingest_sequence=0,
        payload=payload,
        queue_key="queue-1",
        barrier_key=None,
        now=now,
    )
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert blocked.barrier_blocked_at is not None
    assert repo.list_blocked_barrier_items(run_id="run-1") == []


def test_list_blocked_barrier_items_returns_only_barrier_blocked_for_run() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime(2026, 6, 11, 3, 0, tzinfo=UTC)
    payload_a = _seed_run(
        engine,
        run_id="run-A",
        tokens=[("row-a1", "token-a1", 0), ("row-a2", "token-a2", 1), ("row-a3", "token-a3", 2)],
        now=now,
    )
    payload_b = _seed_run(engine, run_id="run-B", tokens=[("row-b1", "token-b1", 0)], now=now)

    # run-A: one barrier-BLOCKED item.
    _enqueue_and_block(
        repo,
        run_id="run-A",
        token_id="token-a1",
        row_id="row-a1",
        ingest_sequence=0,
        payload=payload_a,
        queue_key=None,
        barrier_key="agg-1",
        now=now,
    )
    # run-A: one ADR-028 queue-hold (status=blocked, queue_key set, barrier_key NULL).
    _enqueue_and_block(
        repo,
        run_id="run-A",
        token_id="token-a3",
        row_id="row-a3",
        ingest_sequence=2,
        payload=payload_a,
        queue_key="queue-1",
        barrier_key=None,
        now=now,
    )
    # run-B: one barrier-BLOCKED item (must not bleed into run-A's sweep).
    _enqueue_and_block(
        repo,
        run_id="run-B",
        token_id="token-b1",
        row_id="row-b1",
        ingest_sequence=0,
        payload=payload_b,
        queue_key=None,
        barrier_key="agg-1",
        now=now,
    )
    # run-A: one READY item (enqueued last so the claims above stay deterministic).
    repo.enqueue_ready(
        run_id="run-A",
        token_id="token-a2",
        row_id="row-a2",
        node_id="normalize",
        step_index=1,
        ingest_sequence=1,
        available_at=now,
        row_payload_json=payload_a,
    )

    items = repo.list_blocked_barrier_items(run_id="run-A")

    assert [item.status for item in items] == [TokenWorkStatus.BLOCKED]
    assert items[0].token_id == "token-a1"
    assert items[0].barrier_key is not None  # queue-hold NOT swept in
    assert items[0].barrier_blocked_at is not None


def test_list_blocked_barrier_items_orders_by_barrier_key_ingest_sequence_work_item_id() -> None:
    """Deterministic iteration order: (barrier_key, ingest_sequence, work_item_id).

    Buffer ORDER at restore comes from batch_members.ordinal (Task 2.1), not
    from this verb — this ordering is for deterministic sweeps only.
    """
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime(2026, 6, 11, 3, 0, tzinfo=UTC)
    payload = _seed_run(
        engine,
        run_id="run-1",
        tokens=[("row-1", "token-1", 0), ("row-2", "token-2", 1), ("row-3", "token-3", 2)],
        now=now,
    )
    # Block out of order relative to the expected sweep order.
    _enqueue_and_block(
        repo,
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        ingest_sequence=0,
        payload=payload,
        queue_key=None,
        barrier_key="barrier-b",
        now=now,
    )
    _enqueue_and_block(
        repo,
        run_id="run-1",
        token_id="token-3",
        row_id="row-3",
        ingest_sequence=2,
        payload=payload,
        queue_key=None,
        barrier_key="barrier-a",
        now=now,
    )
    _enqueue_and_block(
        repo,
        run_id="run-1",
        token_id="token-2",
        row_id="row-2",
        ingest_sequence=1,
        payload=payload,
        queue_key=None,
        barrier_key="barrier-a",
        now=now,
    )

    items = repo.list_blocked_barrier_items(run_id="run-1")

    assert [(item.barrier_key, item.ingest_sequence) for item in items] == [
        ("barrier-a", 1),
        ("barrier-a", 2),
        ("barrier-b", 0),
    ]
