"""Task 2.3 (F1 durability unification): ``complete_barrier`` — atomic consume+emit.

Barrier completion (consume the BLOCKED inputs + emit the outputs) is ONE
journal transaction. Out-of-claim flush outputs (timeout/EOF) previously
travelled in memory with no journal row until the sink write — the
unresumable crash window of elspeth-ae5183307b. ``complete_barrier`` closes
it: a single transaction validates the consumed set against the durable
BLOCKED set, terminalizes consumed rows, and inserts/transitions emissions.

``mark_blocked_barrier_terminal`` and ``mark_blocked_barrier_pending_sink_many``
remain public permanently as delegating wrappers (their own tests pin the
legacy partial-release semantics and event shapes; they MUST stay green
unmodified).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, select

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import BarrierEmission, SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "R"
BARRIER_KEY = "agg-1"
NOW = datetime(2026, 6, 11, 4, 0, tzinfo=UTC)


def _make_scheduler_engine() -> Tier1Engine:
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _seed_run(engine: Tier1Engine, *, run_id: str, tokens: list[tuple[str, str, int]], now: datetime) -> str:
    """Insert run/node/row/token prerequisites; return a serialized row payload.

    ``tokens`` is a list of ``(row_id, token_id, ingest_sequence)`` triples,
    one row per token.
    """
    return _seed_run_grouped(
        engine,
        run_id=run_id,
        rows=[(row_id, ingest_sequence) for row_id, _token_id, ingest_sequence in tokens],
        tokens=[(row_id, token_id) for row_id, token_id, _ingest_sequence in tokens],
        now=now,
    )


def _seed_run_grouped(
    engine: Tier1Engine,
    *,
    run_id: str,
    rows: list[tuple[str, int]],
    tokens: list[tuple[str, str]],
    now: datetime,
) -> str:
    """Insert run/node/row/token prerequisites; return a serialized row payload.

    ``rows`` is a list of ``(row_id, ingest_sequence)`` pairs; ``tokens`` is a
    list of ``(row_id, token_id)`` pairs — multiple tokens may share a row,
    the coalesce-group shape (fork/expand children inherit their parent's
    row_id).
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
        for index, (row_id, ingest_sequence) in enumerate(rows):
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
        for row_id, token_id in tokens:
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
    token_id: str,
    row_id: str,
    ingest_sequence: int,
    payload: str,
    now: datetime,
    barrier_key: str = BARRIER_KEY,
) -> None:
    """Enqueue one READY item, claim it, and block it at the barrier."""
    item = repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id="normalize",
        step_index=1,
        ingest_sequence=ingest_sequence,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="w1", lease_seconds=30, now=now + timedelta(seconds=1))
    assert claimed is not None
    assert claimed.work_item_id == item.work_item_id
    repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=None,
        barrier_key=barrier_key,
        now=now + timedelta(seconds=2),
        expected_lease_owner="w1",
    )


def _seed_three_blocked(engine: Tier1Engine, repo, *, extra_tokens: list[tuple[str, str, int]] | None = None) -> str:
    payload = _seed_run(
        engine,
        run_id=RUN_ID,
        tokens=[("r1", "t1", 0), ("r2", "t2", 1), ("r3", "t3", 2), *(extra_tokens or [])],
        now=NOW,
    )
    for ingest_sequence, (row_id, token_id) in enumerate([("r1", "t1"), ("r2", "t2"), ("r3", "t3")]):
        _enqueue_and_block(repo, token_id=token_id, row_id=row_id, ingest_sequence=ingest_sequence, payload=payload, now=NOW)
    return payload


def _statuses(engine: Tier1Engine, token_ids: list[str]) -> set[str]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(token_work_items_table.c.status)
                .where(token_work_items_table.c.run_id == RUN_ID)
                .where(token_work_items_table.c.token_id.in_(token_ids))
            )
            .scalars()
            .all()
        )
    assert len(rows) == len(token_ids)
    return set(rows)


def _row_for_token(engine: Tier1Engine, token_id: str):
    with engine.connect() as conn:
        return (
            conn.execute(
                select(token_work_items_table)
                .where(token_work_items_table.c.run_id == RUN_ID)
                .where(token_work_items_table.c.token_id == token_id)
            )
            .mappings()
            .one()
        )


def _events(engine: Tier1Engine) -> list[dict]:
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(scheduler_events_table)).mappings().all()]


def _terminal_lane_work_item_id(token_id: str, attempt: int = 1) -> str:
    return hashlib.sha256(f"{RUN_ID}:{token_id}:<terminal>:{attempt}".encode()).hexdigest()


def _make_repo():
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    return engine, TokenSchedulerRepository(engine)


def test_complete_barrier_consumes_and_emits_atomically() -> None:
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo, extra_tokens=[("r-agg", "t-agg-out", 3)])

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2", "t3"],
        emitted_pending_sink=[
            BarrierEmission(
                token_id="t-agg-out",
                row_id="r-agg",
                row_payload_json=payload,
                sink_name="out",
                outcome="success",
                path="aggregated",
                step_index=4,
                ingest_sequence=3,
            )
        ],
        emitted_ready=[],
        now=NOW,
    )

    assert n == 3
    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.TERMINAL.value}
    # Consumed payloads are scrubbed.
    for token_id in ("t1", "t2", "t3"):
        scrubbed = json.loads(_row_for_token(engine, token_id)["row_payload_json"])
        assert scrubbed["row_payload"] == "purged"

    out = _row_for_token(engine, "t-agg-out")
    assert out["status"] == TokenWorkStatus.PENDING_SINK.value
    assert out["node_id"] is None  # terminal lane
    assert out["work_item_id"] == _terminal_lane_work_item_id("t-agg-out")
    assert out["attempt"] == 1
    assert out["pending_sink_name"] == "out"
    assert out["pending_outcome"] == "success"
    assert out["pending_path"] == "aggregated"
    assert out["row_payload_json"] == payload
    assert out["lease_owner"] is None

    events = _events(engine)
    terminal_events = [e for e in events if e["event_type"] == SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL.value]
    assert len(terminal_events) == 3
    # Consumed events keep the existing barrier event context shape.
    assert all(json.loads(e["context_json"]) == {"barrier_key": BARRIER_KEY} for e in terminal_events)
    emission_events = [e for e in events if e["event_type"] == SchedulerEventType.MARK_PENDING_SINK.value]
    assert len(emission_events) == 1
    assert emission_events[0]["token_id"] == "t-agg-out"
    assert emission_events[0]["from_status"] is None
    assert emission_events[0]["to_status"] == TokenWorkStatus.PENDING_SINK.value
    # Emission events carry the atomic-completion context.
    assert json.loads(emission_events[0]["context_json"]) == {"barrier_key": BARRIER_KEY, "consumed_count": 3}


def test_complete_barrier_refuses_partial_consumed_set() -> None:
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match=r"uncovered token_ids.*t3"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_refuses_consumed_tokens_missing_from_blocked_set() -> None:
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match=r"missing token_ids.*t4"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2", "t3", "t4"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_crash_atomicity() -> None:
    """A natural failure INSIDE the txn rolls back the consumed transitions.

    The fresh PENDING_SINK insert uses the deterministic sha256 work_item_id
    (run_id:token_id:<terminal>:attempt). Seeding a terminal-lane row for the
    emitted token first makes the INSERT collide inside the transaction AFTER
    the consumed rows have been terminalized — everything must roll back.
    """
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo, extra_tokens=[("r-dup", "t-dup", 3)])
    # Occupy the terminal lane identity for t-dup (attempt=1, node_id NULL).
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id="t-dup",
        row_id="r-dup",
        node_id=None,
        step_index=4,
        ingest_sequence=3,
        available_at=NOW,
        row_payload_json=payload,
    )
    events_before = len(_events(engine))

    # The strict insert wraps the in-txn IntegrityError into LandscapeRecordError.
    with pytest.raises(LandscapeRecordError):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2", "t3"],
            emitted_pending_sink=[
                BarrierEmission(
                    token_id="t-dup",
                    row_id="r-dup",
                    row_payload_json=payload,
                    sink_name="out",
                    outcome="success",
                    path="aggregated",
                    step_index=4,
                    ingest_sequence=3,
                )
            ],
            emitted_ready=[],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}
    assert _row_for_token(engine, "t-dup")["status"] == TokenWorkStatus.READY.value
    # No event from the failed completion survives the rollback.
    assert len(_events(engine)) == events_before


def test_complete_barrier_passthrough_handoff_counts_toward_blocked_coverage() -> None:
    """A buffered token handed off via emitted_pending_sink transitions its OWN BLOCKED row."""
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo)
    blocked_row_before = dict(_row_for_token(engine, "t3"))

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2"],
        emitted_pending_sink=[
            BarrierEmission(
                token_id="t3",
                row_payload_json=payload,
                sink_name="out",
                outcome="success",
                path="completed",
            )
        ],
        emitted_ready=[],
        now=NOW,
    )

    assert n == 2
    assert _statuses(engine, ["t1", "t2"]) == {TokenWorkStatus.TERMINAL.value}
    out = _row_for_token(engine, "t3")
    # Passthrough: same work_item_id and node cursor — BLOCKED -> PENDING_SINK in place.
    assert out["work_item_id"] == blocked_row_before["work_item_id"]
    assert out["node_id"] == blocked_row_before["node_id"]
    assert out["status"] == TokenWorkStatus.PENDING_SINK.value
    assert out["pending_sink_name"] == "out"
    assert out["row_payload_json"] == payload

    emission_events = [e for e in _events(engine) if e["event_type"] == SchedulerEventType.MARK_PENDING_SINK.value]
    assert len(emission_events) == 1
    assert emission_events[0]["from_status"] == TokenWorkStatus.BLOCKED.value
    assert json.loads(emission_events[0]["context_json"]) == {"barrier_key": BARRIER_KEY, "consumed_count": 2}


def test_complete_barrier_leased_triggering_token_is_excluded() -> None:
    """An in-claim LEASED trigger under the same barrier_key is neither required nor terminalized."""
    engine, repo = _make_repo()
    payload = _seed_run(
        engine,
        run_id=RUN_ID,
        tokens=[("r1", "t1", 0), ("r2", "t2", 1), ("r-trig", "t-trigger", 2)],
        now=NOW,
    )
    for ingest_sequence, (row_id, token_id) in enumerate([("r1", "t1"), ("r2", "t2")]):
        _enqueue_and_block(repo, token_id=token_id, row_id=row_id, ingest_sequence=ingest_sequence, payload=payload, now=NOW)
    trigger = repo.enqueue_ready(
        run_id=RUN_ID,
        token_id="t-trigger",
        row_id="r-trig",
        node_id="normalize",
        step_index=1,
        ingest_sequence=2,
        available_at=NOW,
        row_payload_json=payload,
        barrier_key=BARRIER_KEY,
    )
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="w-trigger", lease_seconds=30, now=NOW)
    assert claimed is not None
    assert claimed.work_item_id == trigger.work_item_id

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2"],
        emitted_pending_sink=[],
        emitted_ready=[],
        now=NOW,
        leased_exclusion_token_id="t-trigger",
    )

    assert n == 2
    assert _statuses(engine, ["t1", "t2"]) == {TokenWorkStatus.TERMINAL.value}
    assert _row_for_token(engine, "t-trigger")["status"] == TokenWorkStatus.LEASED.value


def test_complete_barrier_leased_exclusion_exempts_blocked_row_from_coverage() -> None:
    """The exclusion arm: a BLOCKED row for the excluded token is not required in consumed_token_ids."""
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2"],
        emitted_pending_sink=[],
        emitted_ready=[],
        now=NOW,
        leased_exclusion_token_id="t3",
    )

    assert n == 2
    assert _statuses(engine, ["t1", "t2"]) == {TokenWorkStatus.TERMINAL.value}
    assert _row_for_token(engine, "t3")["status"] == TokenWorkStatus.BLOCKED.value


def test_complete_barrier_emitted_ready_inserts_ready_rows_with_enqueue_events() -> None:
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo, extra_tokens=[("r-next", "t-next", 3)])

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2", "t3"],
        emitted_pending_sink=[],
        emitted_ready=[
            BarrierEmission(
                token_id="t-next",
                row_id="r-next",
                row_payload_json=payload,
                node_id="normalize",
                step_index=4,
                ingest_sequence=3,
            )
        ],
        now=NOW,
    )

    assert n == 3
    out = _row_for_token(engine, "t-next")
    assert out["status"] == TokenWorkStatus.READY.value
    assert out["node_id"] == "normalize"
    assert out["attempt"] == 1
    enqueue_events = [e for e in _events(engine) if e["event_type"] == SchedulerEventType.ENQUEUE.value]
    emission_enqueues = [e for e in enqueue_events if e["token_id"] == "t-next"]
    assert len(emission_enqueues) == 1
    assert json.loads(emission_enqueues[0]["context_json"]) == {"barrier_key": BARRIER_KEY, "consumed_count": 3}

    # The emitted READY continuation is claimable like any other.
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="w-next", lease_seconds=30, now=NOW + timedelta(seconds=3))
    assert claimed is not None
    assert claimed.token_id == "t-next"


def test_complete_barrier_rejects_duplicate_consumed_token_ids() -> None:
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match="duplicate"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t1", "t2", "t3"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_rejects_consumed_token_also_emitted() -> None:
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match="both consumed and emitted"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2", "t3"],
            emitted_pending_sink=[
                BarrierEmission(
                    token_id="t3",
                    row_payload_json=payload,
                    sink_name="out",
                    outcome="success",
                    path="completed",
                )
            ],
            emitted_ready=[],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_wrappers_delegate_preserving_legacy_partial_release() -> None:
    """The two public wrappers keep their pinned semantics over complete_barrier.

    Partial release (blocked rows not named stay BLOCKED) is pinned by
    test_scheduler_barrier_completion_only_terminalizes_consumed_tokens and the
    lifecycle state machine; this is a delegation smoke test, not the pin.
    """
    from elspeth.contracts.scheduler import BlockedPendingSinkHandoff

    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo)

    transitioned = repo.mark_blocked_barrier_pending_sink_many(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        handoffs={
            "t1": BlockedPendingSinkHandoff(
                row_payload_json=payload,
                sink_name="out",
                outcome="success",
                path="completed",
                error_hash=None,
                error_message=None,
            )
        },
        now=NOW,
    )
    assert transitioned == 1
    # Legacy wrapper event context carries NO consumed_count.
    handoff_events = [e for e in _events(engine) if e["event_type"] == SchedulerEventType.MARK_PENDING_SINK.value]
    assert json.loads(handoff_events[0]["context_json"]) == {"barrier_key": BARRIER_KEY}

    terminalized = repo.mark_blocked_barrier_terminal(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        token_ids=("t2",),  # t3 left blocked: legacy partial release
        now=NOW,
    )
    assert terminalized == 1
    assert _row_for_token(engine, "t2")["status"] == TokenWorkStatus.TERMINAL.value
    assert _row_for_token(engine, "t3")["status"] == TokenWorkStatus.BLOCKED.value


COALESCE_KEY = "coalesce:merge"


def _seed_two_coalesce_groups(engine: Tier1Engine, repo, *, extra_tokens: list[tuple[str, str]] | None = None) -> str:
    """Two rows' branch groups BLOCKED under ONE shared coalesce barrier_key.

    Coalesce barriers key on coalesce_name, shared by ALL row_ids pending at
    the coalesce; fork children inherit their parent's row_id, so each pending
    group is (run_id, barrier_key, row_id). ``extra_tokens`` registers
    additional (row_id, token_id) pairs without blocking them (e.g. the merged
    output child of a coalesce fire).
    """
    payload = _seed_run_grouped(
        engine,
        run_id=RUN_ID,
        rows=[("r1", 0), ("r2", 1)],
        tokens=[("r1", "t1a"), ("r1", "t1b"), ("r2", "t2a"), ("r2", "t2b"), *(extra_tokens or [])],
        now=NOW,
    )
    for token_id, row_id, ingest_sequence in [("t1a", "r1", 0), ("t1b", "r1", 0), ("t2a", "r2", 1), ("t2b", "r2", 1)]:
        _enqueue_and_block(
            repo,
            token_id=token_id,
            row_id=row_id,
            ingest_sequence=ingest_sequence,
            payload=payload,
            now=NOW,
            barrier_key=COALESCE_KEY,
        )
    return payload


def test_complete_barrier_scope_row_id_isolates_coalesce_group() -> None:
    """A strict coalesce fire scoped to one row consumes ONLY that row's branches.

    Without scope_row_id the shared coalesce barrier_key would make r2's held
    branches look uncovered and spuriously trip the would-orphan check.
    """
    engine, repo = _make_repo()
    _seed_two_coalesce_groups(engine, repo)

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=COALESCE_KEY,
        consumed_token_ids=["t1a", "t1b"],
        emitted_pending_sink=[],
        emitted_ready=[],
        now=NOW,
        scope_row_id="r1",
    )

    assert n == 2
    assert _statuses(engine, ["t1a", "t1b"]) == {TokenWorkStatus.TERMINAL.value}
    # The other row's pending group is untouched.
    assert _statuses(engine, ["t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_scoped_group_still_catches_cross_group_consumed_token() -> None:
    """scope_row_id narrows the missing-token cross-check too: another group's token is refused."""
    engine, repo = _make_repo()
    _seed_two_coalesce_groups(engine, repo)

    with pytest.raises(AuditIntegrityError, match=r"missing token_ids.*t2a.*scope_row_id='r1'"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=COALESCE_KEY,
            consumed_token_ids=["t1a", "t1b", "t2a"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
            scope_row_id="r1",
        )

    assert _statuses(engine, ["t1a", "t1b", "t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_scoped_group_still_catches_uncovered_blocked_row() -> None:
    """Exhaustiveness still holds WITHIN the scoped group: a held branch left behind raises."""
    engine, repo = _make_repo()
    _seed_two_coalesce_groups(engine, repo)

    with pytest.raises(AuditIntegrityError, match=r"uncovered token_ids.*t1b.*scope_row_id='r1'"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=COALESCE_KEY,
            consumed_token_ids=["t1a"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
            scope_row_id="r1",
        )

    assert _statuses(engine, ["t1a", "t1b", "t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_scope_row_id_requires_exhaustive_release() -> None:
    """scope_row_id is meaningless on the legacy partial-release arm and is refused."""
    engine, repo = _make_repo()
    _seed_two_coalesce_groups(engine, repo)

    with pytest.raises(AuditIntegrityError, match="meaningless on the legacy partial-release arm"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=COALESCE_KEY,
            consumed_token_ids=["t1a", "t1b"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
            scope_row_id="r1",
            require_exhaustive_release=False,
        )

    assert _statuses(engine, ["t1a", "t1b", "t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_combined_lanes_one_call() -> None:
    """Consumed + passthrough + fresh pending-sink + ready emissions in ONE atomic call.

    Pins the arm ordering (terminalizations, then passthrough transitions, then
    fresh inserts, then ready inserts) via the scheduler_events insertion order
    and the shared emission context on BOTH emission lanes.
    """
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo, extra_tokens=[("r-agg", "t-agg-out", 3), ("r-next", "t-next", 4)])
    events_before = len(_events(engine))

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=BARRIER_KEY,
        consumed_token_ids=["t1", "t2"],
        emitted_pending_sink=[
            # Passthrough: t3 holds a BLOCKED row under the barrier.
            BarrierEmission(token_id="t3", row_payload_json=payload, sink_name="out", outcome="success", path="completed"),
            # Fresh terminal-lane insert: t-agg-out has no BLOCKED row.
            BarrierEmission(
                token_id="t-agg-out",
                row_id="r-agg",
                row_payload_json=payload,
                sink_name="out",
                outcome="success",
                path="aggregated",
                step_index=4,
                ingest_sequence=3,
            ),
        ],
        emitted_ready=[
            BarrierEmission(
                token_id="t-next",
                row_id="r-next",
                row_payload_json=payload,
                node_id="normalize",
                step_index=4,
                ingest_sequence=4,
            )
        ],
        now=NOW,
    )

    assert n == 2
    assert _statuses(engine, ["t1", "t2"]) == {TokenWorkStatus.TERMINAL.value}
    assert _statuses(engine, ["t3", "t-agg-out"]) == {TokenWorkStatus.PENDING_SINK.value}
    assert _row_for_token(engine, "t-agg-out")["node_id"] is None
    assert _row_for_token(engine, "t-next")["status"] == TokenWorkStatus.READY.value

    new_events = _events(engine)[events_before:]
    assert [(event["event_type"], event["token_id"]) for event in new_events] == [
        (SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL.value, "t1"),
        (SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL.value, "t2"),
        (SchedulerEventType.MARK_PENDING_SINK.value, "t3"),
        (SchedulerEventType.MARK_PENDING_SINK.value, "t-agg-out"),
        (SchedulerEventType.ENQUEUE.value, "t-next"),
    ]
    # Consumed events keep the legacy context shape; ALL emission events on
    # BOTH lanes share the atomic-completion context.
    for event in new_events[:2]:
        assert json.loads(event["context_json"]) == {"barrier_key": BARRIER_KEY}
    for event in new_events[2:]:
        assert json.loads(event["context_json"]) == {"barrier_key": BARRIER_KEY, "consumed_count": 2}


def test_complete_barrier_scoped_coalesce_fire_emits_merged_ready_child() -> None:
    """The Task 3.4 coalesce shape end-to-end: scoped consume + merged READY child.

    The merged child's row_id is the inherited parent row_id (= scope_row_id);
    the other row's held branches are untouched.
    """
    engine, repo = _make_repo()
    payload = _seed_two_coalesce_groups(engine, repo, extra_tokens=[("r1", "t1-merged")])

    n = repo.complete_barrier(
        run_id=RUN_ID,
        barrier_key=COALESCE_KEY,
        consumed_token_ids=["t1a", "t1b"],
        emitted_pending_sink=[],
        emitted_ready=[
            BarrierEmission(
                token_id="t1-merged",
                row_id="r1",  # fork children inherit the parent row_id
                row_payload_json=payload,
                node_id="normalize",
                step_index=2,
                ingest_sequence=0,
            )
        ],
        now=NOW,
        scope_row_id="r1",
    )

    assert n == 2
    assert _statuses(engine, ["t1a", "t1b"]) == {TokenWorkStatus.TERMINAL.value}
    assert _statuses(engine, ["t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}
    merged = _row_for_token(engine, "t1-merged")
    assert merged["status"] == TokenWorkStatus.READY.value
    assert merged["row_id"] == "r1"
    enqueue_events = [
        event for event in _events(engine) if event["event_type"] == SchedulerEventType.ENQUEUE.value and event["token_id"] == "t1-merged"
    ]
    assert len(enqueue_events) == 1
    assert json.loads(enqueue_events[0]["context_json"]) == {"barrier_key": COALESCE_KEY, "consumed_count": 2}
    # The merged continuation is claimable like any other.
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="w-merged", lease_seconds=30, now=NOW + timedelta(seconds=3))
    assert claimed is not None
    assert claimed.token_id == "t1-merged"


def test_complete_barrier_scoped_fire_rejects_emission_outside_scope_group() -> None:
    """A scoped completion must not emit into another row's pending group."""
    engine, repo = _make_repo()
    payload = _seed_two_coalesce_groups(engine, repo, extra_tokens=[("r1", "t1-merged")])

    with pytest.raises(AuditIntegrityError, match="outside the scoped pending group"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=COALESCE_KEY,
            consumed_token_ids=["t1a", "t1b"],
            emitted_pending_sink=[],
            emitted_ready=[
                BarrierEmission(
                    token_id="t1-merged",
                    row_id="r2",  # confused caller: emitting into the OTHER row's group
                    row_payload_json=payload,
                    node_id="normalize",
                    step_index=2,
                    ingest_sequence=1,
                )
            ],
            now=NOW,
            scope_row_id="r1",
        )

    assert _statuses(engine, ["t1a", "t1b", "t2a", "t2b"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_leased_exclusion_requires_exhaustive_release() -> None:
    """leased_exclusion_token_id only affects the strict exhaustiveness check; the legacy arm refuses it."""
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match="exclusion only exempts"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
            leased_exclusion_token_id="t3",
            require_exhaustive_release=False,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_rejects_duplicate_ready_emissions() -> None:
    engine, repo = _make_repo()
    payload = _seed_three_blocked(engine, repo, extra_tokens=[("r-next", "t-next", 3)])

    with pytest.raises(AuditIntegrityError, match="duplicate ready emissions"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2", "t3"],
            emitted_pending_sink=[],
            emitted_ready=[
                BarrierEmission(
                    token_id="t-next",
                    row_id="r-next",
                    row_payload_json=payload,
                    node_id="normalize",
                    step_index=4,
                    ingest_sequence=3,
                ),
                BarrierEmission(
                    token_id="t-next",
                    row_id="r-next",
                    row_payload_json=payload,
                    node_id="normalize",
                    step_index=5,
                    ingest_sequence=3,
                ),
            ],
            now=NOW,
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}


def test_complete_barrier_rejects_leased_exclusion_token_in_consumed_set() -> None:
    """The in-claim trigger must be excluded, not completed: exclusion ∩ consumed raises."""
    engine, repo = _make_repo()
    _seed_three_blocked(engine, repo)

    with pytest.raises(AuditIntegrityError, match="must be excluded, not completed"):
        repo.complete_barrier(
            run_id=RUN_ID,
            barrier_key=BARRIER_KEY,
            consumed_token_ids=["t1", "t2", "t3"],
            emitted_pending_sink=[],
            emitted_ready=[],
            now=NOW,
            leased_exclusion_token_id="t3",
        )

    assert _statuses(engine, ["t1", "t2", "t3"]) == {TokenWorkStatus.BLOCKED.value}
