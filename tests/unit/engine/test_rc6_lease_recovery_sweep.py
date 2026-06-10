"""Characterization tests for multi-item expired-lease recovery sweeps.

These tests pin the CURRENT behavior of ``TokenSchedulerRepository.
recover_expired_leases`` against a real Tier-1 SQLite engine when the sweep
faces a POPULATION of leases rather than a single item (filigree
elspeth-0bae6d8a52):

1. A sweep by a fresh ``lease_owner`` recovers every expired lease in one
   call — exactly once each, with an attempt bump and ``work_item_id``
   rotation — and never touches a live (unexpired) lease.
2. Recovery order is the deterministic 3-key ORDER BY
   ``(ingest_sequence, step_index, work_item_id)`` — including the
   ``work_item_id`` last-resort tiebreaker for exact same-key collisions
   (the same determinism contract as ``claim_ready``, filigree
   elspeth-6cb89db535).
3. The G1 self-steal guard extends PAST lease expiry: an expired lease is
   invisible to its own holder's sweep, even while that same sweep recovers
   other owners' expired leases. Recovery therefore requires a DIFFERENT
   ``lease_owner`` — the resume-sweep path, not in-run self-recovery.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, insert, select, text

from elspeth.contracts import NodeType
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-rc6-lease-sweep"


def _make_scheduler_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _insert_run_and_nodes(engine: Tier1Engine, *, now: datetime) -> None:
    with engine.begin() as conn:
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


def _insert_row_with_tokens(
    engine: Tier1Engine,
    *,
    row_id: str,
    ingest_sequence: int,
    token_ids: tuple[str, ...],
    now: datetime,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id="source-a",
                row_index=ingest_sequence,
                source_row_index=ingest_sequence,
                ingest_sequence=ingest_sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=now,
            )
        )
        for token_id in token_ids:
            conn.execute(
                insert(tokens_table).values(
                    token_id=token_id,
                    row_id=row_id,
                    run_id=RUN_ID,
                    created_at=now,
                )
            )


def _enqueue_single_token_rows(
    repo: TokenSchedulerRepository,
    engine: Tier1Engine,
    token_ids: tuple[str, ...],
    *,
    now: datetime,
) -> dict[str, TokenWorkItem]:
    """One row + one token per entry, ingest_sequence in tuple order."""
    payload = _row_payload_json()
    items: dict[str, TokenWorkItem] = {}
    for ingest_sequence, token_id in enumerate(token_ids):
        row_id = f"row-{ingest_sequence}"
        _insert_row_with_tokens(engine, row_id=row_id, ingest_sequence=ingest_sequence, token_ids=(token_id,), now=now)
        items[token_id] = repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=1,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )
    return items


def _work_item_states(engine: Tier1Engine) -> dict[str, dict[str, object]]:
    with engine.connect() as conn:
        return {
            row["token_id"]: dict(row)
            for row in conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.work_item_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.attempt,
                    token_work_items_table.c.lease_owner,
                ).where(token_work_items_table.c.run_id == RUN_ID)
            ).mappings()
        }


def _recovery_events(engine: Tier1Engine) -> list[dict[str, object]]:
    """Recovery events in insertion order.

    All events of one sweep share a single ``recorded_at`` and ``event_id`` is
    a non-monotonic opaque id, so SQLite's rowid is the only durable witness
    of the sweep's per-item iteration order.
    """
    with engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == RUN_ID)
                .where(scheduler_events_table.c.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value)
                .order_by(text("rowid"))
            ).mappings()
        ]


def test_sweep_recovers_every_expired_lease_exactly_once_and_never_live_leases() -> None:
    """A fresh-owner sweep over 3 expired + 2 live leases recovers exactly the
    3 expired items in one call — each exactly once with attempt bump and
    work_item_id rotation — leaves the live leases untouched, and is
    idempotent across repeated calls."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_run_and_nodes(engine, now=now)

    token_ids = ("token-0", "token-1", "token-2", "token-3", "token-4")
    originals = _enqueue_single_token_rows(repo, engine, token_ids, now=now)

    # claim_ready admits in ingest_sequence order, so the Nth claim leases
    # token-N. Tokens 0/2/4 get a 30s lease (expired at sweep time); tokens
    # 1/3 get a 3600s lease (still live at sweep time).
    expired_tokens = ("token-0", "token-2", "token-4")
    live_tokens = ("token-1", "token-3")
    lease_seconds_by_token = {"token-0": 30, "token-1": 3600, "token-2": 30, "token-3": 3600, "token-4": 30}
    for token_id in token_ids:
        claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=lease_seconds_by_token[token_id], now=now)
        assert claimed is not None
        assert claimed.token_id == token_id

    sweep_at = now + timedelta(seconds=60)
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 3

    states = _work_item_states(engine)
    for token_id in expired_tokens:
        assert states[token_id]["status"] == TokenWorkStatus.READY.value
        assert states[token_id]["attempt"] == 2
        assert states[token_id]["lease_owner"] is None
        assert states[token_id]["work_item_id"] != originals[token_id].work_item_id
    for token_id in live_tokens:
        assert states[token_id]["status"] == TokenWorkStatus.LEASED.value
        assert states[token_id]["attempt"] == 1
        assert states[token_id]["lease_owner"] == "worker-a"
        assert states[token_id]["work_item_id"] == originals[token_id].work_item_id

    # Exactly one recovery event per expired item, each bumping 1 -> 2.
    events = _recovery_events(engine)
    assert sorted(str(event["token_id"]) for event in events) == sorted(expired_tokens)
    assert all(event["from_attempt"] == 1 and event["to_attempt"] == 2 for event in events)
    assert all(event["caller_owner"] == "resume-sweeper" for event in events)

    # Idempotent: a second sweep finds nothing left to recover.
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 0
    assert len(_recovery_events(engine)) == 3

    # The recovered continuations are claimable in ingest order; the live
    # leases still block their own tokens.
    reclaimed: list[str] = []
    while True:
        item = repo.claim_ready(run_id=RUN_ID, lease_owner="resume-sweeper", lease_seconds=300, now=sweep_at)
        if item is None:
            break
        assert item.attempt == 2
        reclaimed.append(item.token_id)
    assert reclaimed == list(expired_tokens)


def test_sweep_recovery_order_is_ingest_sequence_then_step_index_then_work_item_id() -> None:
    """Multi-item recovery walks expired leases in the deterministic 3-key
    order (ingest_sequence, step_index, work_item_id) — the work_item_id
    last-resort tiebreaker resolves exact same-key collisions, mirroring the
    claim_ready determinism contract (elspeth-6cb89db535)."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)

    # Fork-family shape: three sibling tokens on row-0 (ingest_sequence 0) —
    # token-y/token-z collide exactly on (ingest_sequence=0, step_index=1),
    # token-w trails at step_index=2 — plus token-c on row-1 (ingest_sequence 1).
    _insert_row_with_tokens(engine, row_id="row-0", ingest_sequence=0, token_ids=("token-w", "token-y", "token-z"), now=now)
    _insert_row_with_tokens(engine, row_id="row-1", ingest_sequence=1, token_ids=("token-c",), now=now)
    items: dict[str, TokenWorkItem] = {}
    for token_id, row_id, step_index, ingest_sequence in (
        ("token-w", "row-0", 2, 0),
        ("token-y", "row-0", 1, 0),
        ("token-z", "row-0", 1, 0),
        ("token-c", "row-1", 1, 1),
    ):
        items[token_id] = repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id="normalize",
            step_index=step_index,
            ingest_sequence=ingest_sequence,
            available_at=now,
            row_payload_json=payload,
        )

    for _ in range(4):
        assert repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=30, now=now) is not None

    sweep_at = now + timedelta(seconds=60)
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 4

    tied_pair = sorted(("token-y", "token-z"), key=lambda token_id: items[token_id].work_item_id)
    recovery_order = [event["token_id"] for event in _recovery_events(engine)]
    assert recovery_order == [*tied_pair, "token-w", "token-c"]


def test_expired_lease_is_invisible_to_its_own_holders_sweep() -> None:
    """G1 self-steal guard extends past expiry: a sweep recovers other owners'
    expired leases but NEVER the caller's own — even when the caller's lease is
    itself expired. The wedged item is recovered only when a DIFFERENT
    lease_owner (the resume-sweep path) runs the sweep."""
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_run_and_nodes(engine, now=now)

    _enqueue_single_token_rows(repo, engine, ("token-0", "token-1"), now=now)

    # claim_ready admits in ingest order: worker-a leases token-0, worker-b
    # leases token-1. Both leases expire before the sweep.
    claimed_a = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=30, now=now)
    claimed_b = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=30, now=now)
    assert claimed_a is not None and claimed_a.token_id == "token-0"
    assert claimed_b is not None and claimed_b.token_id == "token-1"

    sweep_at = now + timedelta(seconds=60)

    # worker-a's sweep recovers ONLY worker-b's expired lease; its own expired
    # lease stays LEASED under worker-a (invisible to its own holder).
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="worker-a") == 1
    states = _work_item_states(engine)
    assert states["token-1"]["status"] == TokenWorkStatus.READY.value
    assert states["token-1"]["attempt"] == 2
    assert states["token-0"]["status"] == TokenWorkStatus.LEASED.value
    assert states["token-0"]["attempt"] == 1
    assert states["token-0"]["lease_owner"] == "worker-a"

    # Repeating its own sweep never reaps it.
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="worker-a") == 0

    # A different lease_owner — the resume-sweep identity — recovers it.
    assert repo.recover_expired_leases(run_id=RUN_ID, now=sweep_at, caller_owner="resume-sweeper") == 1
    states = _work_item_states(engine)
    assert states["token-0"]["status"] == TokenWorkStatus.READY.value
    assert states["token-0"]["attempt"] == 2
    assert states["token-0"]["lease_owner"] is None

    # No recovery event was ever attributed to the lease's own holder.
    events = _recovery_events(engine)
    assert [(event["token_id"], event["caller_owner"], event["from_lease_owner"]) for event in events] == [
        ("token-1", "worker-a", "worker-b"),
        ("token-0", "resume-sweeper", "worker-a"),
    ]
