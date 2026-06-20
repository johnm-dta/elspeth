"""Pin ``TokenSchedulerRepository.count_ready_in_set`` and ``count_failed_in_set``
(ADR-030 M1/M2 remediation).

The leader's relinquish gate (processor.py) uses BOTH verbs: it relinquishes a
non-READY pending set only when ``count_ready_in_set == 0`` AND
``count_failed_in_set == 0`` AND a peer holds an active lease. ``count_ready_in_set``
answers "how many are still READY?" (a still-READY item is genuinely stranded);
``count_failed_in_set`` answers "how many are FAILED?" (FAILED is the only status
absent from BOTH backstops, so a single FAILED pending row keeps the leader
raising). ``count_failed_in_set`` is pinned at the bottom of this module. Both
share the same M2 run-scope + chunking contract:

  - it counts ONLY READY rows from the supplied set (LEASED / PENDING_SINK /
    TERMINAL / FAILED / absent do not count);
  - it is SCOPED to ``run_id`` like every sibling verb — a same-id work item
    belonging to another run must NOT count (the M2 fix; before it, the query
    had no run_id predicate);
  - it must not break on a large ``.in_()`` fan-out (>999 siblings) —
    SQLITE_MAX_VARIABLE_NUMBER (the M2 chunking fix).

The cross-run case is the regression that FAILS under the pre-fix
(un-run-scoped) query: two runs each enqueue a work item, and a count
restricted to run A's id-set must return only run A's READY rows even though
run B has identically-statused rows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from tests.fixtures.landscape import RecorderSetup, make_factory, make_recorder_with_run, register_test_node

NODE_ID = "normalize"
LEASE_OWNER = "worker-a"
NOW = datetime(2026, 1, 1, tzinfo=UTC)

_PAYLOAD = TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _enqueue_ready(setup: RecorderSetup, scheduler: TokenSchedulerRepository, *, sequence: int) -> str:
    row, token = setup.data_flow.create_row_with_token(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=sequence,
        data={"id": sequence},
        source_row_index=sequence,
        ingest_sequence=sequence,
    )
    item = scheduler.enqueue_ready(
        run_id=setup.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=sequence,
        available_at=NOW,
        row_payload_json=_PAYLOAD,
    )
    return item.work_item_id


def _single_run() -> tuple[RecorderSetup, TokenSchedulerRepository]:
    setup = make_recorder_setup("run-count-ready")
    return setup, setup.factory.scheduler


def make_recorder_setup(run_id: str) -> RecorderSetup:
    """A fresh run+source+NODE_ID on its own DB (per-run isolation)."""
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-1")
    register_test_node(setup.data_flow, setup.run_id, NODE_ID)
    return setup


def test_empty_input_returns_zero() -> None:
    setup, scheduler = _single_run()
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[]) == 0


def test_all_ready_returns_count() -> None:
    setup, scheduler = _single_run()
    ids = [_enqueue_ready(setup, scheduler, sequence=i) for i in range(3)]
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=ids) == 3


def test_mixed_statuses_counts_only_ready() -> None:
    """One READY, one LEASED, one PENDING_SINK → 1."""
    setup, scheduler = _single_run()
    ready_id = _enqueue_ready(setup, scheduler, sequence=0)
    leased_id = _enqueue_ready(setup, scheduler, sequence=1)
    pending_id = _enqueue_ready(setup, scheduler, sequence=2)

    # Drive leased_id → LEASED, pending_id → PENDING_SINK via production verbs.
    # claim_ready pops the lowest ingest_sequence first, so claim three times and
    # transition the specific ones we want non-READY.
    claims = {}
    for _ in range(3):
        item = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER, lease_seconds=300, now=NOW)
        assert item is not None
        claims[item.work_item_id] = item
    # Re-enqueue the one we want to stay READY by recovering its lease back.
    # Simpler: mark two non-READY, then re-READY the third via recover.
    scheduler.mark_pending_sink(
        work_item_id=pending_id,
        row_payload_json=_PAYLOAD,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=NOW + timedelta(seconds=1),
        expected_lease_owner=LEASE_OWNER,
    )
    # leased_id stays LEASED (claimed, untouched). ready_id: push it back to READY
    # by expiring its lease and recovering.
    from sqlalchemy import update

    from elspeth.core.landscape.schema import token_work_items_table

    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == ready_id)
            .values(status=TokenWorkStatus.READY.value, lease_owner=None, lease_expires_at=None)
        )

    result = scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=[ready_id, leased_id, pending_id])
    assert result == 1, "only the READY row counts (LEASED + PENDING_SINK excluded)"


def test_all_non_ready_returns_zero() -> None:
    setup, scheduler = _single_run()
    ids = [_enqueue_ready(setup, scheduler, sequence=i) for i in range(2)]
    for _ in range(2):
        item = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER, lease_seconds=300, now=NOW)
        assert item is not None
        scheduler.mark_terminal(work_item_id=item.work_item_id, now=NOW + timedelta(seconds=1), expected_lease_owner=LEASE_OWNER)
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=ids) == 0


def test_absent_work_item_id_does_not_count() -> None:
    setup, scheduler = _single_run()
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=["does-not-exist-uuid"]) == 0


def test_separate_db_isolation_baseline() -> None:
    """Baseline: behaviour on separate-DB deployments (NOT the M2 regression pin).

    Two independent runs on SEPARATE SQLite files each enqueue a READY work item.
    Counting run A's id against run B returns 0 — but this isolation is guaranteed
    by SQLite's per-file architecture, NOT by the run_id predicate, so it would
    pass even under the pre-M2 (un-run-scoped) query. The actual M2 regression (a
    count leaking rows from a different run) can only manifest on a SHARED DB; the
    genuine run_id-predicate pin is ``test_cross_run_isolation_shared_db_run_id_predicate``
    below. This test documents the separate-DB deployment contract.
    """
    setup_a = make_recorder_setup("run-A")
    scheduler_a = setup_a.factory.scheduler
    id_a = _enqueue_ready(setup_a, scheduler_a, sequence=0)

    # Same id queried against a DIFFERENT run must not count it.
    setup_b = make_recorder_setup("run-B")
    scheduler_b = setup_b.factory.scheduler
    _id_b = _enqueue_ready(setup_b, scheduler_b, sequence=0)

    assert scheduler_a.count_ready_in_set(run_id=setup_a.run_id, work_item_ids=[id_a]) == 1
    # Run B's scheduler asked for run A's id → 0 (id_a does not exist in B's DB,
    # but the contract that matters is the run_id predicate, pinned next).
    assert scheduler_b.count_ready_in_set(run_id=setup_b.run_id, work_item_ids=[id_a]) == 0


def test_cross_run_isolation_shared_db_run_id_predicate() -> None:
    """Strongest run_id-scope pin: two runs on ONE DB.

    Both runs enqueue a READY work item. count_ready_in_set(run=A, ids=[A_id,
    B_id]) must return 1 (only A's row), proving the ``.where(run_id == run_id)``
    predicate excludes B's identically-statused READY row even on a shared DB.
    Pre-fix (no run_id predicate) this returned 2.
    """
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    scheduler = factory.scheduler

    def _make_run(run_id: str, sequence: int) -> str:
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id=run_id,
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        register_test_node(factory.data_flow, run.run_id, NODE_ID)
        row, token = factory.data_flow.create_row_with_token(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=sequence,
            data={"id": sequence},
            source_row_index=sequence,
            ingest_sequence=sequence,
        )
        item = scheduler.enqueue_ready(
            run_id=run.run_id,
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=sequence,
            available_at=NOW,
            row_payload_json=_PAYLOAD,
        )
        return item.work_item_id

    a_id = _make_run("shared-run-A", 0)
    b_id = _make_run("shared-run-B", 1)

    # Both ids are READY on the shared DB.
    assert scheduler.count_ready_in_set(run_id="shared-run-A", work_item_ids=[a_id, b_id]) == 1, (
        "run_id predicate must exclude run B's READY row even when its id is in the set"
    )
    assert scheduler.count_ready_in_set(run_id="shared-run-B", work_item_ids=[a_id, b_id]) == 1


def test_large_in_list_does_not_break_on_sqlite_variable_limit() -> None:
    """The M2 chunking fix: a >999-element id set must not blow
    SQLITE_MAX_VARIABLE_NUMBER. Enqueue >999 READY siblings and count them all.

    n is chosen to exceed the HISTORICAL SQLITE_MAX_VARIABLE_NUMBER of 999
    (SQLite < 3.32) — plus one bind slot for the run_id predicate — while staying
    well under the current 32766 default (SQLite >= 3.32). On an old SQLite build
    the pre-fix single unchunked ``.in_()`` raises OperationalError ("too many SQL
    variables"); the chunked implementation (chunk_size=900) never does. Note: on
    a modern SQLite build (e.g. 3.47) the unchunked query would NOT raise at this
    n, so this is a forward-/backward-safety pin for older deployments rather than
    a hard regression pin on this build — the chunking is still the correct
    implementation and is exercised here (n > chunk_size forces >1 chunk).
    """
    setup, scheduler = _single_run()
    n = 1005
    ids = [_enqueue_ready(setup, scheduler, sequence=i) for i in range(n)]
    assert scheduler.count_ready_in_set(run_id=setup.run_id, work_item_ids=ids) == n


# ---------------------------------------------------------------------------
# count_failed_in_set — the FAILED-status refusal arm of the M1 discriminator.
# ---------------------------------------------------------------------------

LEASE_OWNER_LEADER = "leader"


def test_count_failed_empty_input_returns_zero() -> None:
    setup, scheduler = _single_run()
    assert scheduler.count_failed_in_set(run_id=setup.run_id, work_item_ids=[]) == 0


def test_count_failed_counts_only_failed_rows() -> None:
    """Only FAILED rows count.

    Seeds: one FAILED (counts), one TERMINAL (excluded), one LEASED (excluded),
    one READY (excluded). Expect 1. FAILED is the only status absent from both
    backstops, so the leader refuses to relinquish whenever this is > 0.
    """
    setup, scheduler = _single_run()
    failed_id = _enqueue_ready(setup, scheduler, sequence=0)
    terminal_id = _enqueue_ready(setup, scheduler, sequence=1)
    leased_id = _enqueue_ready(setup, scheduler, sequence=2)
    ready_id = _enqueue_ready(setup, scheduler, sequence=3)

    # Claim FAILED + TERMINAL + LEASED ids (lowest ingest_sequence first).
    claimed_failed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER_LEADER, lease_seconds=300, now=NOW)
    assert claimed_failed is not None and claimed_failed.work_item_id == failed_id
    scheduler.mark_failed(work_item_id=failed_id, now=NOW + timedelta(seconds=1), expected_lease_owner=LEASE_OWNER_LEADER)

    claimed_terminal = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER_LEADER, lease_seconds=300, now=NOW)
    assert claimed_terminal is not None and claimed_terminal.work_item_id == terminal_id
    scheduler.mark_terminal(work_item_id=terminal_id, now=NOW + timedelta(seconds=1), expected_lease_owner=LEASE_OWNER_LEADER)

    claimed_leased = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER_LEADER, lease_seconds=300, now=NOW)
    assert claimed_leased is not None and claimed_leased.work_item_id == leased_id
    # leased stays LEASED; ready_id untouched (still READY).

    result = scheduler.count_failed_in_set(run_id=setup.run_id, work_item_ids=[failed_id, terminal_id, leased_id, ready_id])
    assert result == 1, "only the FAILED row counts (TERMINAL + LEASED + READY excluded)"


def test_count_failed_is_run_scoped() -> None:
    """A FAILED row in another run (shared DB) must not count."""
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    scheduler = factory.scheduler

    def _make_failed(run_id: str, sequence: int) -> str:
        run = factory.run_lifecycle.begin_run(
            config={}, canonical_version="v1", run_id=run_id, openrouter_catalog_sha256="0" * 64, openrouter_catalog_source="bundled"
        )
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        register_test_node(factory.data_flow, run.run_id, NODE_ID)
        row, token = factory.data_flow.create_row_with_token(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=sequence,
            data={"id": sequence},
            source_row_index=sequence,
            ingest_sequence=sequence,
        )
        item = scheduler.enqueue_ready(
            run_id=run.run_id,
            token_id=token.token_id,
            row_id=row.row_id,
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=sequence,
            available_at=NOW,
            row_payload_json=_PAYLOAD,
        )
        claimed = scheduler.claim_ready(run_id=run.run_id, lease_owner=LEASE_OWNER_LEADER, lease_seconds=300, now=NOW)
        assert claimed is not None
        scheduler.mark_failed(work_item_id=item.work_item_id, now=NOW + timedelta(seconds=1), expected_lease_owner=LEASE_OWNER_LEADER)
        return item.work_item_id

    a_id = _make_failed("failed-run-A", 0)
    b_id = _make_failed("failed-run-B", 1)
    assert scheduler.count_failed_in_set(run_id="failed-run-A", work_item_ids=[a_id, b_id]) == 1


# ---------------------------------------------------------------------------
# has_peer_owned_work — arm (3): a peer is/was carrying work on this run.
# ---------------------------------------------------------------------------

PEER_OWNER = "follower-peer"


def test_has_peer_owned_work_false_for_solo_leader_own_rows() -> None:
    """An N=1 leader's own LEASED rows are NOT peer-owned → False."""
    setup, scheduler = _single_run()
    _enqueue_ready(setup, scheduler, sequence=0)
    claimed = scheduler.claim_ready(run_id=setup.run_id, lease_owner=LEASE_OWNER_LEADER, lease_seconds=300, now=NOW)
    assert claimed is not None  # own LEASED row
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEASE_OWNER_LEADER) is False


def test_has_peer_owned_work_true_for_peer_leased() -> None:
    """A LEASED row under a DIFFERENT owner → True."""
    from sqlalchemy import update

    from elspeth.core.landscape.schema import token_work_items_table

    setup, scheduler = _single_run()
    wid = _enqueue_ready(setup, scheduler, sequence=0)
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == wid)
            .values(status=TokenWorkStatus.LEASED.value, lease_owner=PEER_OWNER, lease_expires_at=NOW + timedelta(seconds=300))
        )
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEASE_OWNER_LEADER) is True


def test_has_peer_owned_work_true_for_peer_pending_sink_even_after_lease_lapses() -> None:
    """The decisive case: a peer's PENDING_SINK row (mark_pending_sink KEEPS the
    owner) is detected even though it holds NO active LEASE.

    This is exactly the in-claim-drain race: a follower parks all its claims as
    PENDING_SINK and lets its leases lapse before the leader's claim_ready returns
    None. peer_active_leases() would be empty here, but has_peer_owned_work must
    still be True so the leader relinquishes rather than raising.
    """
    from sqlalchemy import update

    from elspeth.core.landscape.schema import token_work_items_table

    setup, scheduler = _single_run()
    wid = _enqueue_ready(setup, scheduler, sequence=0)
    # Peer claims then parks PENDING_SINK (owner kept), with NO live lease.
    with setup.db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == wid)
            .values(status=TokenWorkStatus.LEASED.value, lease_owner=PEER_OWNER, lease_expires_at=NOW + timedelta(seconds=300))
        )
    scheduler.mark_pending_sink(
        work_item_id=wid,
        row_payload_json=_PAYLOAD,
        sink_name="sink-a",
        outcome="success",
        path="default",
        error_hash=None,
        error_message=None,
        now=NOW + timedelta(seconds=1),
        expected_lease_owner=PEER_OWNER,
    )
    # No active peer LEASE remains, but the PENDING_SINK row still carries the peer.
    assert scheduler.peer_active_leases(run_id=setup.run_id, caller_owner=LEASE_OWNER_LEADER, now=NOW + timedelta(seconds=2)) == ()
    assert scheduler.has_peer_owned_work(run_id=setup.run_id, caller_owner=LEASE_OWNER_LEADER) is True


def test_has_peer_owned_work_is_run_scoped() -> None:
    """A peer-owned row in another run (shared DB) must not leak."""
    from sqlalchemy import update

    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.landscape.schema import token_work_items_table
    from tests.fixtures.landscape import make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    scheduler = factory.scheduler

    run = factory.run_lifecycle.begin_run(
        config={}, canonical_version="v1", run_id="peer-owned-B", openrouter_catalog_sha256="0" * 64, openrouter_catalog_source="bundled"
    )
    node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )
    register_test_node(factory.data_flow, run.run_id, NODE_ID)
    row, token = factory.data_flow.create_row_with_token(
        run_id=run.run_id, source_node_id=node.node_id, row_index=0, data={"id": 0}, source_row_index=0, ingest_sequence=0
    )
    item = scheduler.enqueue_ready(
        run_id=run.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=0,
        available_at=NOW,
        row_payload_json=_PAYLOAD,
    )
    with db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == item.work_item_id)
            .values(status=TokenWorkStatus.LEASED.value, lease_owner=PEER_OWNER, lease_expires_at=NOW + timedelta(seconds=300))
        )
    # Querying a DIFFERENT run id must not see run-B's peer-owned row.
    assert scheduler.has_peer_owned_work(run_id="peer-owned-A", caller_owner=LEASE_OWNER_LEADER) is False
    assert scheduler.has_peer_owned_work(run_id="peer-owned-B", caller_owner=LEASE_OWNER_LEADER) is True
