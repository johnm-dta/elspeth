"""Characterization tests for lease-recovery races and mid-sweep crash atomicity.

Ticket elspeth-0bae6d8a52, items (c) and (d). These tests pin the durable,
journal-level guarantees of ``TokenSchedulerRepository.recover_expired_leases``
and ``claim_ready`` under deterministic interleavings against a REAL
file-backed Tier-1 SQLite database (two engine handles on the same file —
``:memory:`` cannot be shared across engines):

(c) ``recover_expired_leases`` racing a concurrent ``claim_ready`` for the
    same item. The invariant under test is *exactly one effective owner and a
    consistent attempt count — never a lost item, never a double-bump*. Three
    deterministic interleavings are pinned:

    1. A claimant probing AFTER the sweep's per-row UPDATE has executed but
       BEFORE the sweep transaction commits is excluded at its own ``BEGIN
       IMMEDIATE`` with the retryable "database is locked" error — the
       uncommitted READY row is unobservable by construction. After the
       sweep commits, the rotated attempt is claimable.
    2. A peer sweeper attempting a competing recovery inside that same
       window is likewise lock-excluded; the caller's sweep wins, and the
       peer's serialized retry returns 0 cleanly (records no event) — the
       attempt is bumped exactly once, never twice.
    3. The mirror-image window inside ``claim_ready``: a peer claim inside
       the caller claim's SELECT→CAS window is lock-excluded; the caller
       wins and the peer's retry returns ``None`` cleanly (no
       ``AuditIntegrityError``), leaving a single lease owner and an
       unchanged attempt.

    SQLite serialization note — UPDATED for the option-c slice-1 write-intent
    discipline (ADR-030 §D5): every scheduler write transaction now begins
    with ``BEGIN IMMEDIATE``, taking the single WAL write lock AT BEGIN. The
    old SELECT->UPDATE window — in which a second engine could COMMIT a
    competing claim/recovery before the caller's first per-row UPDATE — no
    longer exists: a peer write transaction attempted inside that window is
    excluded at its own ``BEGIN IMMEDIATE`` with the retryable
    "database is locked" ``OperationalError`` (after its ``busy_timeout``
    poll). These tests therefore pin LOCK EXCLUSION inside the window plus
    the clean serialized loser path immediately after commit (recovery
    returns 0 / claim returns None — no ``AuditIntegrityError``, no double
    bump). The CAS predicate on the per-row UPDATE remains as belt-and-braces
    and still serves same-connection interleavings; its
    ``lease_expires_at < now`` expiry leg is pinned directly by
    ``test_scheduler_recover_expired_leases_skips_pending_sink_row_with_fresh_lease``
    in tests/unit/core/test_multi_source_foundation.py.

(d) Crash mid-``recover_expired_leases`` with multiple expired items. The
    sweep runs in ONE ``engine.begin()`` transaction, so a crash after some
    per-row UPDATEs have executed must roll back atomically: NOTHING is
    recovered — no half-bumped item (attempt advanced but status still
    LEASED), no rotated ``work_item_id`` with stale status, no orphaned
    recovery event — and a repeated sweep then completes recovery for ALL
    items.

The interleavings are driven by SQLAlchemy ``before_cursor_execute`` /
``after_cursor_execute`` event listeners on the TEST harness side (the same
pattern as the elspeth-28aaa36a62
PENDING_SINK ABA regression test in test_multi_source_foundation.py); the
repository code under test is never monkeypatched. All clocks are injected
fixed datetimes; assertions go only through the durable scheduler journal
(``token_work_items`` columns) and the audit ``scheduler_events`` table, so
they survive the F1 durability unification that deletes the checkpoint-blob
layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, insert, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from elspeth.contracts import NodeType
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkItem, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-rc6-lease-races"
BASE = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
SWEEP_AT = BASE + timedelta(seconds=60)


@pytest.fixture
def engines(tmp_path: Path) -> Iterator[tuple[Tier1Engine, Tier1Engine]]:
    """Two independent Tier-1 engine handles onto ONE file-backed SQLite DB.

    Cross-engine interleavings need two real connections; ``:memory:``
    databases are per-engine, so the canonical in-memory helper cannot be
    shared. Both handles go through ``LandscapeDB._configure_sqlite`` +
    ``_verify_sqlite_pragmas`` so they satisfy the same Tier-1 PRAGMA
    invariants (WAL, foreign_keys=ON) as production engines.
    """
    url = f"sqlite:///{tmp_path / 'landscape.db'}"
    raw_engines: list[Engine] = []
    for _ in range(2):
        raw = create_engine(url, echo=False)
        LandscapeDB._configure_sqlite(raw)
        LandscapeDB._verify_sqlite_pragmas(raw, url)
        raw_engines.append(raw)
    metadata.create_all(raw_engines[0])
    yield Tier1Engine(raw_engines[0]), Tier1Engine(raw_engines[1])
    for raw in raw_engines:
        raw.dispose()


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _seed_run_rows_tokens(engine: Tier1Engine, token_ids: tuple[str, ...]) -> None:
    """Insert the run, source/transform nodes, and one row per token."""
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=BASE,
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
                    registered_at=BASE,
                )
            )
        for ingest_sequence, token_id in enumerate(token_ids):
            row_id = f"row-{ingest_sequence}"
            conn.execute(
                insert(rows_table).values(
                    row_id=row_id,
                    run_id=RUN_ID,
                    source_node_id="source-a",
                    row_index=ingest_sequence,
                    source_row_index=ingest_sequence,
                    ingest_sequence=ingest_sequence,
                    source_data_hash=f"hash-{row_id}",
                    created_at=BASE,
                )
            )
            conn.execute(
                insert(tokens_table).values(
                    token_id=token_id,
                    row_id=row_id,
                    run_id=RUN_ID,
                    created_at=BASE,
                )
            )


def _enqueue_tokens(repo: TokenSchedulerRepository, token_ids: tuple[str, ...]) -> dict[str, TokenWorkItem]:
    payload = _row_payload_json()
    return {
        token_id: repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=f"row-{ingest_sequence}",
            node_id="normalize",
            step_index=1,
            ingest_sequence=ingest_sequence,
            available_at=BASE,
            row_payload_json=payload,
        )
        for ingest_sequence, token_id in enumerate(token_ids)
    }


def _expire_leases(repo: TokenSchedulerRepository, token_ids: tuple[str, ...], *, lease_owner: str = "worker-dead") -> None:
    """Lease every enqueued token for 30s at BASE — all expired by SWEEP_AT."""
    for token_id in token_ids:
        claimed = repo.claim_ready(run_id=RUN_ID, lease_owner=lease_owner, lease_seconds=30, now=BASE)
        assert claimed is not None
        assert claimed.token_id == token_id


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


def _event_counts(engine: Tier1Engine) -> dict[str, int]:
    with engine.connect() as conn:
        rows = conn.execute(select(scheduler_events_table.c.event_type).where(scheduler_events_table.c.run_id == RUN_ID)).scalars()
        counts: dict[str, int] = {}
        for event_type in rows:
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts


def _is_token_work_items_update(statement: str) -> bool:
    return statement.lstrip().upper().startswith("UPDATE TOKEN_WORK_ITEMS")


def _lower_busy_timeout(engine: Tier1Engine, ms: int = 100) -> None:
    """Shrink the pooled connection's busy_timeout so lock-exclusion paths
    fail fast in tests instead of polling the production 5000 ms window.

    The engine's QueuePool holds a single DBAPI connection under sequential
    single-threaded use, so adjusting it directly is deterministic.
    """
    raw = engine.raw_connection()
    try:
        driver = raw.driver_connection
        assert driver is not None
        driver.execute(f"PRAGMA busy_timeout={ms}")
    finally:
        raw.close()


def test_explicit_none_preserves_pre_coordination_direct_harness_recovery(
    engines: tuple[Tier1Engine, Tier1Engine],
) -> None:
    """The load-bearing None arm remains available to direct harnesses that
    have no coordination seat and must recover a crashed owner's lease.
    """
    engine, _ = engines
    repo = TokenSchedulerRepository(engine)
    _seed_run_rows_tokens(engine, ("token-0",))
    original = _enqueue_tokens(repo, ("token-0",))["token-0"]
    _expire_leases(repo, ("token-0",))
    with engine.connect() as conn:
        assert conn.execute(select(run_coordination_table.c.run_id)).first() is None

    recovered = repo.recover_expired_leases(
        run_id=RUN_ID,
        now=SWEEP_AT,
        caller_owner="direct-harness-sweeper",
        coordination_token=None,
    )

    assert recovered == 1
    state = _work_item_states(engine)["token-0"]
    assert state["status"] == TokenWorkStatus.READY.value
    assert state["attempt"] == 2
    assert state["lease_owner"] is None
    assert state["work_item_id"] != original.work_item_id
    assert _event_counts(engine)[SchedulerEventType.RECOVER_EXPIRED_LEASE.value] == 1


def test_claim_ready_probing_mid_sweep_is_lock_excluded_until_recovery_commits(
    engines: tuple[Tier1Engine, Tier1Engine],
) -> None:
    """Item (c), interleaving 1 — write-intent discipline: a concurrent
    claimant that probes AFTER the sweep's per-row UPDATE has executed but
    BEFORE the sweep transaction commits cannot even BEGIN its own write
    transaction. The sweep holds the WAL write lock from its ``BEGIN
    IMMEDIATE``, so the claimant's ``BEGIN IMMEDIATE`` polls busy_timeout and
    raises the retryable "database is locked" ``OperationalError``. The
    uncommitted READY row (bumped attempt inside the sweep's open
    transaction) is therefore trivially unobservable — the old
    WAL-snapshot-invisibility pin is subsumed by lock exclusion, and the
    BUSY_SNAPSHOT mid-transaction abort the old shape risked is impossible
    by construction. After the sweep commits, the rotated attempt becomes
    claimable. No torn intermediate state is ever observable through the
    journal."""
    sweep_engine, claim_engine = engines
    sweep_repo = TokenSchedulerRepository(sweep_engine)
    claim_repo = TokenSchedulerRepository(claim_engine)
    _seed_run_rows_tokens(sweep_engine, ("token-0",))
    original = _enqueue_tokens(sweep_repo, ("token-0",))["token-0"]
    _expire_leases(sweep_repo, ("token-0",))
    _lower_busy_timeout(claim_engine)

    mid_sweep_outcomes: list[TokenWorkItem | OperationalError | None] = []

    @event.listens_for(sweep_engine, "after_cursor_execute")
    def claimant_probes_mid_sweep(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        if mid_sweep_outcomes or not _is_token_work_items_update(statement):
            return
        # The sweep's per-row UPDATE has EXECUTED (the row is READY/attempt-2
        # in the sweep's uncommitted transaction) but nothing is committed: a
        # worker on a separate engine probes for claimable work right now.
        try:
            mid_sweep_outcomes.append(claim_repo.claim_ready(run_id=RUN_ID, lease_owner="claimant", lease_seconds=300, now=SWEEP_AT))
        except OperationalError as exc:
            mid_sweep_outcomes.append(exc)

    try:
        recovered = sweep_repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="resume-sweeper")
    finally:
        event.remove(sweep_engine, "after_cursor_execute", claimant_probes_mid_sweep)

    assert len(mid_sweep_outcomes) == 1
    outcome = mid_sweep_outcomes[0]
    assert isinstance(outcome, OperationalError) and "database is locked" in str(outcome), (
        "Write-intent discipline: the mid-sweep probe is excluded at BEGIN IMMEDIATE (retryable), never fed a torn state"
    )
    assert recovered == 1

    # After commit, the recovered item is claimable at the bumped attempt.
    claimed = claim_repo.claim_ready(run_id=RUN_ID, lease_owner="claimant", lease_seconds=300, now=SWEEP_AT)
    assert claimed is not None
    assert claimed.token_id == "token-0"
    assert claimed.attempt == 2, "Attempt bumped exactly once by the sweep"
    assert claimed.work_item_id != original.work_item_id

    states = _work_item_states(sweep_engine)
    assert len(states) == 1, "Never a lost or duplicated item"
    assert states["token-0"]["status"] == TokenWorkStatus.LEASED.value
    assert states["token-0"]["lease_owner"] == "claimant"
    assert states["token-0"]["attempt"] == 2

    counts = _event_counts(sweep_engine)
    assert counts[SchedulerEventType.RECOVER_EXPIRED_LEASE.value] == 1
    # Initial expired claim + the post-recovery claim; the mid-sweep probe
    # was excluded at BEGIN and recorded nothing.
    assert counts[SchedulerEventType.CLAIM_READY.value] == 2


def test_peer_recovery_in_select_update_window_is_lock_excluded_and_loses_cleanly_after_commit(
    engines: tuple[Tier1Engine, Tier1Engine],
) -> None:
    """Item (c), interleaving 2 — write-intent discipline: a peer sweeper
    that tries to recover the same item inside the caller sweep's
    SELECT→UPDATE window is excluded at its own ``BEGIN IMMEDIATE`` (the
    caller holds the write lock from BEGIN), so the window in which a peer
    could COMMIT a competing recovery no longer exists: the caller's sweep
    always wins its own window. The peer's retry AFTER the commit takes the
    clean serialized loser path: ``recover_expired_leases`` finds nothing
    expired (returns 0, records no event — the rotated attempt is READY,
    unleased), and the peer's claim leases the bumped attempt. Exactly one
    attempt bump, exactly one effective owner, never a double-recovered
    item. (The per-row CAS predicate remains as belt-and-braces for
    same-connection interleavings; its ``lease_expires_at < now`` leg is
    pinned by the foundation ABA test — see the module docstring.)"""
    sweep_engine, peer_engine = engines
    sweep_repo = TokenSchedulerRepository(sweep_engine)
    peer_repo = TokenSchedulerRepository(peer_engine)
    _seed_run_rows_tokens(sweep_engine, ("token-0",))
    original = _enqueue_tokens(sweep_repo, ("token-0",))["token-0"]
    _expire_leases(sweep_repo, ("token-0",))
    _lower_busy_timeout(peer_engine)

    peer_outcomes: list[OperationalError] = []

    @event.listens_for(sweep_engine, "before_cursor_execute")
    def peer_recovers_mid_sweep(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        if peer_outcomes or not _is_token_work_items_update(statement):
            return
        # Peer sweep attempts a competing recovery on a separate engine while
        # the caller's sweep is between its SELECT and its per-row UPDATE.
        with pytest.raises(OperationalError, match="database is locked") as excinfo:
            peer_repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="peer-sweeper")
        peer_outcomes.append(excinfo.value)

    try:
        winner_recovered = sweep_repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="resume-sweeper")
    finally:
        event.remove(sweep_engine, "before_cursor_execute", peer_recovers_mid_sweep)

    assert len(peer_outcomes) == 1, "The peer's mid-window recovery was excluded at BEGIN IMMEDIATE"
    assert winner_recovered == 1, "The caller's sweep always wins its own window under the write lock"

    # Serialized loser path after the commit: nothing left to recover (clean
    # 0, no event), and the rotated attempt is claimable by the peer.
    assert peer_repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="peer-sweeper") == 0
    peer_claim = peer_repo.claim_ready(run_id=RUN_ID, lease_owner="peer-claimant", lease_seconds=300, now=SWEEP_AT)
    assert peer_claim is not None and peer_claim.attempt == 2

    states = _work_item_states(sweep_engine)
    assert len(states) == 1, "Never a lost or duplicated item"
    assert states["token-0"]["status"] == TokenWorkStatus.LEASED.value
    assert states["token-0"]["lease_owner"] == "peer-claimant", "Exactly one effective owner — the peer's fresh lease survives"
    assert states["token-0"]["attempt"] == 2, "Attempt bumped exactly once — never double-bumped"
    assert states["token-0"]["work_item_id"] != original.work_item_id

    counts = _event_counts(sweep_engine)
    assert counts[SchedulerEventType.RECOVER_EXPIRED_LEASE.value] == 1, "Only the winning sweep recorded a recovery event"


def test_peer_claim_in_select_update_window_is_lock_excluded_single_owner(
    engines: tuple[Tier1Engine, Tier1Engine],
) -> None:
    """Item (c), interleaving 3 (mirror window inside claim_ready) —
    write-intent discipline: a peer claim attempted between the caller
    claim's SELECT and its CAS UPDATE is excluded at the peer's own ``BEGIN
    IMMEDIATE`` — the caller has held the write lock since BEGIN, so the
    "peer commits in the window" interleaving the CAS rowcount-0 path used
    to absorb can no longer occur. The caller's claim wins; the peer's
    retry after the commit returns ``None`` cleanly (the row is LEASED) —
    single lease owner, unchanged attempt, exactly one CLAIM_READY event."""
    claim_engine, peer_engine = engines
    claim_repo = TokenSchedulerRepository(claim_engine)
    peer_repo = TokenSchedulerRepository(peer_engine)
    _seed_run_rows_tokens(claim_engine, ("token-0",))
    original = _enqueue_tokens(claim_repo, ("token-0",))["token-0"]
    _lower_busy_timeout(peer_engine)

    peer_outcomes: list[OperationalError] = []

    @event.listens_for(claim_engine, "before_cursor_execute")
    def peer_claims_mid_claim(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        if peer_outcomes or not _is_token_work_items_update(statement):
            return
        with pytest.raises(OperationalError, match="database is locked") as excinfo:
            peer_repo.claim_ready(run_id=RUN_ID, lease_owner="peer-worker", lease_seconds=300, now=BASE)
        peer_outcomes.append(excinfo.value)

    try:
        winner_claim = claim_repo.claim_ready(run_id=RUN_ID, lease_owner="first-worker", lease_seconds=300, now=BASE)
    finally:
        event.remove(claim_engine, "before_cursor_execute", peer_claims_mid_claim)

    assert len(peer_outcomes) == 1, "The peer's mid-window claim was excluded at BEGIN IMMEDIATE"
    assert winner_claim is not None and winner_claim.lease_owner == "first-worker"

    # Serialized loser path: the peer's retry sees the committed lease and
    # returns None cleanly — never an AuditIntegrityError, never a steal.
    assert peer_repo.claim_ready(run_id=RUN_ID, lease_owner="peer-worker", lease_seconds=300, now=BASE) is None

    states = _work_item_states(claim_engine)
    assert len(states) == 1
    assert states["token-0"]["status"] == TokenWorkStatus.LEASED.value
    assert states["token-0"]["lease_owner"] == "first-worker", "Exactly one effective owner"
    assert states["token-0"]["attempt"] == 1, "Claim races never touch the attempt counter"
    assert states["token-0"]["work_item_id"] == original.work_item_id

    counts = _event_counts(claim_engine)
    assert counts[SchedulerEventType.CLAIM_READY.value] == 1, "Only the winning claim recorded an event"


@pytest.mark.parametrize("crash_before_update_number", [2, 3])
def test_crash_mid_sweep_rolls_back_atomically_and_repeat_sweep_completes(
    engines: tuple[Tier1Engine, Tier1Engine],
    crash_before_update_number: int,
) -> None:
    """Item (d): a crash partway through one ``recover_expired_leases`` sweep
    over 3 expired items (after 1 or 2 per-row UPDATEs executed, before
    commit) rolls back the WHOLE single transaction. The durable journal shows
    every item in exactly one coherent state — here, with full rollback, ALL
    items untouched (LEASED, original owner/attempt/work_item_id) and ZERO
    recovery events. There is NEVER a half-bumped item: an attempt advanced
    while status stayed LEASED, or a rotated work_item_id with stale status,
    or a recovery event for an unrecovered item. A repeated sweep then
    completes recovery for all items."""
    engine, _ = engines
    repo = TokenSchedulerRepository(engine)
    token_ids = ("token-0", "token-1", "token-2")
    _seed_run_rows_tokens(engine, token_ids)
    originals = _enqueue_tokens(repo, token_ids)
    _expire_leases(repo, token_ids)

    update_count = [0]

    @event.listens_for(engine, "before_cursor_execute")
    def crash_mid_sweep(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        if not _is_token_work_items_update(statement):
            return
        update_count[0] += 1
        if update_count[0] == crash_before_update_number:
            raise RuntimeError("simulated crash mid-recovery-sweep")

    try:
        with pytest.raises(RuntimeError, match="simulated crash mid-recovery-sweep"):
            repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="resume-sweeper")
    finally:
        event.remove(engine, "before_cursor_execute", crash_mid_sweep)
    assert update_count[0] == crash_before_update_number, "Crash fired at the intended per-row UPDATE"

    # Atomic-or-idempotent, durable check. Each item must be in exactly one
    # coherent state: UNTOUCHED (rollback) or FULLY RECOVERED (committed) —
    # never a half-bumped hybrid.
    states_after_crash = _work_item_states(engine)
    assert set(states_after_crash) == set(token_ids), "No item lost in the crash"
    untouched: list[str] = []
    fully_recovered: list[str] = []
    for token_id in token_ids:
        state = states_after_crash[token_id]
        if (
            state["status"] == TokenWorkStatus.LEASED.value
            and state["attempt"] == 1
            and state["lease_owner"] == "worker-dead"
            and state["work_item_id"] == originals[token_id].work_item_id
        ):
            untouched.append(token_id)
        elif (
            state["status"] == TokenWorkStatus.READY.value
            and state["attempt"] == 2
            and state["lease_owner"] is None
            and state["work_item_id"] != originals[token_id].work_item_id
        ):
            fully_recovered.append(token_id)
        else:
            pytest.fail(f"Half-bumped work item after mid-sweep crash: {token_id}={state!r}")

    # The sweep is ONE transaction, so the rollback is total: the UPDATEs that
    # executed before the crash (and their recovery events) are undone.
    assert untouched == list(token_ids), "Single-transaction sweep must roll back ALL per-row updates on crash"
    assert fully_recovered == []
    assert _event_counts(engine).get(SchedulerEventType.RECOVER_EXPIRED_LEASE.value, 0) == 0, (
        "No recovery event may survive for an unrecovered item"
    )

    # A repeated sweep completes recovery for every item.
    assert repo.recover_expired_leases(run_id=RUN_ID, now=SWEEP_AT, caller_owner="resume-sweeper") == 3
    final_states = _work_item_states(engine)
    for token_id in token_ids:
        assert final_states[token_id]["status"] == TokenWorkStatus.READY.value
        assert final_states[token_id]["attempt"] == 2, "Attempt bumped exactly once across crash + retry"
        assert final_states[token_id]["lease_owner"] is None
        assert final_states[token_id]["work_item_id"] != originals[token_id].work_item_id
    assert _event_counts(engine)[SchedulerEventType.RECOVER_EXPIRED_LEASE.value] == 3
