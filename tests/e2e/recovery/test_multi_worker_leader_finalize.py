"""Leader finalize-phase peer coordination (ADR-030 H1 + M3 remediation).

These tests drive the REAL leader ``Orchestrator._execute_run`` finalize phase
— the 4b-pre bounded peer-lease wait (H1) and the 4b looped follower
PENDING_SINK drain (M3) — over a real WAL DB. A peer worker's durable scheduler
rows are crafted directly (the technique the follower e2e suites use) so the
leader's ``has_peer_active_leases()`` / ``has_scheduled_work()`` observe a real
multi-worker image at finalize time.

To keep the bounded wait fast and deterministic, the loop's ``time.monotonic``
and ``time.sleep`` are patched in ``elspeth.engine.orchestrator.core``: sleep is
a no-op and monotonic returns a scripted, advancing clock so the lease-aware
deadline is reached in a handful of iterations without any real wall-clock wait.

Contracts pinned (each FAILS under the pre-fix code):

  H1-7  Wedged-but-alive peer: an unexpired, never-resolving peer lease must NOT
        hang the leader forever — the bounded wait times out (logging the still-
        leased peer) and falls through to the unresolved-work invariant raise.
        Pre-fix: ``while has_peer_active_leases(): time.sleep(0.5)`` → infinite.
  H1-8  Deposed leader during wait: ``check_coordination_latch`` raising
        ``RunWorkerEvictedError`` inside the wait must break the leader out
        (INTERRUPTED ceremony), not spin. Pre-fix: the latch was ignored.
  H1-9  Dead peer mid-lease: a peer's EXPIRED lease with a dead registry row is
        actively reaped to READY by the in-loop ``reap_expired_peer_leases``
        within the liveness window (not the 300s item TTL). Pre-fix: no reap in
        the wait; only natural lease expiry.
  M3-5  Late PENDING_SINK: a follower PENDING_SINK row present at finalize is
        DRAINED to the sink by the looped 4b (exactly once), so the run COMPLETEs
        with the follower's row written — never a silent lost row.
  M3-10 Double-flush guard: the leader's own already-written tokens are NOT
        re-emitted when 4b accumulates follower results (no node_states UNIQUE
        IntegrityError, no duplicate sink row).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import select, update

from elspeth.contracts import Determinism, PipelineRow, PluginSchema, RunStatus
from elspeth.contracts.coordination import DEFAULT_ITEM_STALL_BUDGET_SECONDS, DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
from elspeth.contracts.errors import OrchestrationInvariantError, RunWorkerEvictedError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.config import QueueSettings, SourceSettings, TransformSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    node_states_table,
    scheduler_events_table,
    token_work_items_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.processor import RowProcessor
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink

PEER_OWNER = "follower-peer"


class _RowSchema(PluginSchema):
    id: int
    value: int


def _observed_contract(row: dict[str, Any]) -> SchemaContract:
    fields = tuple(FieldContract(normalized_name=k, original_name=k, python_type=object, required=False, source="inferred") for k in row)
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


class _SeedingSource(_TestSourceBase):
    """Yields rows, then runs an injected callback to craft peer scheduler state.

    The callback fires while the run is RUNNING and after all rows are journaled
    — so by the time the source loop exits and the leader enters 4b-pre, the
    crafted peer lease / follower PENDING_SINK row is durable and visible.
    """

    name = "seeding_source"
    output_schema = _RowSchema
    determinism = Determinism.IO_READ

    def __init__(self, rows: list[dict[str, Any]], seed_cb: Any, *, on_success: str) -> None:
        super().__init__()
        self._rows = rows
        self._seed_cb = seed_cb
        self.on_success = on_success
        self._seeded = False

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for idx, row in enumerate(self._rows):
            contract = _observed_contract(row)
            self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=idx)
        if not self._seeded:
            self._seeded = True
            self._seed_cb(ctx.run_id)

    def close(self) -> None:
        pass


class _PassthroughTransform(BaseTransform):
    name = "passthrough"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _RowSchema
    output_schema: type[PluginSchema] = _RowSchema
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.on_success = "output"
        self.input = "inbound"

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "pass"})


def _build(rows: list[dict[str, Any]], seed_cb: Any) -> tuple[PipelineConfig, ExecutionGraph, CollectSink, _SeedingSource]:
    source = _SeedingSource(rows, seed_cb, on_success="inbound")
    transform = _PassthroughTransform()
    sink = CollectSink("output")
    sources = {"primary": as_source(source)}
    wired = WiredTransform(
        plugin=as_transform(transform),
        settings=TransformSettings(
            name="passthrough_0", plugin=transform.name, input="inbound", on_success="output", on_error="discard", options={}
        ),
    )
    graph = ExecutionGraph.from_plugin_instances(
        sources=sources,
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="inbound", options={})},
        transforms=[wired],
        sinks={"output": as_sink(sink)},
        queues={"inbound": QueueSettings(description="multi-worker finalize")},
    )
    config = PipelineConfig(sources=sources, transforms=[as_transform(transform)], sinks={"output": as_sink(sink)})
    return config, graph, sink, source


class _FastMonotonic:
    """Scripted monotonic clock: every call advances by ``step`` seconds.

    Patched over ``core.time.monotonic`` so the 3x liveness deadline is crossed
    after a few iterations — no real wall-clock wait. ``step`` defaults to a
    fraction of the liveness window so a handful of iterations exhausts the
    240s budget.
    """

    def __init__(self, step: float = DEFAULT_RUN_LIVENESS_WINDOW_SECONDS) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        cur = self._t
        self._t += self._step
        return cur


def _seed_peer_leased_row(
    db: LandscapeDB,
    payload_store: FilesystemPayloadStore,
    run_id: str,
    *,
    ingest_sequence: int,
    lease_expires_offset: float,
    source_node_id: str,
) -> str:
    """Craft a peer-owned LEASED row directly. Returns its token_id.

    A real row+token+READY enqueue via production writers, then a direct SQL flip
    to LEASED under ``PEER_OWNER`` with the chosen expiry (positive=unexpired,
    negative=already expired). No run_workers row is created → the leader's
    liveness-aware reaper treats the owner as dead (owner_registry_dead=TRUE).
    """
    from datetime import UTC, datetime, timedelta

    factory = RecorderFactory(db, payload_store=payload_store)
    repo = TokenSchedulerRepository(db.engine)
    now = datetime.now(UTC)
    data = {"id": 1000 + ingest_sequence, "value": ingest_sequence}
    row = factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=source_node_id,
        row_index=1000 + ingest_sequence,
        data=data,
        source_row_index=1000 + ingest_sequence,
        ingest_sequence=1000 + ingest_sequence,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    item = repo.enqueue_ready(
        run_id=run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=source_node_id,
        step_index=0,
        ingest_sequence=1000 + ingest_sequence,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
        available_at=now,
    )
    with db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == item.work_item_id)
            .values(
                status=TokenWorkStatus.LEASED.value,
                lease_owner=PEER_OWNER,
                lease_expires_at=now + timedelta(seconds=lease_expires_offset),
            )
        )
    return token.token_id


def _seed_follower_pending_sink_row(
    db: LandscapeDB,
    payload_store: FilesystemPayloadStore,
    run_id: str,
    *,
    ingest_sequence: int,
    source_node_id: str,
    sink_name: str = "output",
    value: int | None = None,
) -> str:
    """Craft a follower-produced PENDING_SINK row via production verbs.

    Models a follower that claimed a sink-bound CONTINUATION (a child token of an
    existing source row the leader already ingested) and parked it for the leader
    to drain: child token on an existing row → enqueue READY → flip LEASED under
    PEER_OWNER → ``mark_pending_sink(sink_name)`` attributed to PEER_OWNER.
    Returns token_id. The leader's 4b drain claims this PENDING_SINK row and
    writes it to the sink.

    The child token reuses an EXISTING leader row_id (not a new source row) so it
    does not inflate ``count_distinct_source_rows_with_terminal_outcome`` — a
    follower processes continuations of journaled source rows, never phantom new
    source rows.
    """
    from datetime import UTC, datetime, timedelta

    from elspeth.core.landscape.schema import rows_table

    factory = RecorderFactory(db, payload_store=payload_store)
    repo = TokenSchedulerRepository(db.engine)
    now = datetime.now(UTC)
    # Reuse an existing leader source row (a continuation child), not a new row.
    # The child's scheduler cursor MUST carry the parent row's ingest_sequence
    # (enqueue_ready enforces row_id↔ingest_sequence ownership).
    with db.engine.connect() as conn:
        existing = (
            conn.execute(
                select(rows_table.c.row_id, rows_table.c.ingest_sequence)
                .where(rows_table.c.run_id == run_id)
                .order_by(rows_table.c.ingest_sequence)
                .limit(1)
            )
            .mappings()
            .one()
        )
    row_id = str(existing["row_id"])
    row_ingest_sequence = int(existing["ingest_sequence"])
    data = {"id": 2000 + ingest_sequence, "value": value if value is not None else ingest_sequence}
    token = factory.data_flow.create_token(row_id=row_id)
    payload = TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data)))
    item = repo.enqueue_ready(
        run_id=run_id,
        token_id=token.token_id,
        row_id=row_id,
        node_id=source_node_id,
        step_index=1,
        ingest_sequence=row_ingest_sequence,
        row_payload_json=payload,
        available_at=now,
    )
    # Flip LEASED under the peer so mark_pending_sink's owner CAS passes.
    with db.engine.begin() as conn:
        conn.execute(
            update(token_work_items_table)
            .where(token_work_items_table.c.work_item_id == item.work_item_id)
            .values(status=TokenWorkStatus.LEASED.value, lease_owner=PEER_OWNER, lease_expires_at=now + timedelta(seconds=300))
        )
    repo.mark_pending_sink(
        work_item_id=item.work_item_id,
        row_payload_json=payload,
        sink_name=sink_name,
        outcome="success",
        path="default_flow",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=1),
        expected_lease_owner=PEER_OWNER,
    )
    return token.token_id


def _source_node_id(db: LandscapeDB, run_id: str) -> str:
    from elspeth.core.landscape.schema import rows_table

    with db.engine.connect() as conn:
        return str(conn.execute(select(rows_table.c.source_node_id).where(rows_table.c.run_id == run_id).limit(1)).scalar_one())


def _make_orch(tmp_path: Path) -> tuple[Orchestrator, LandscapeDB, FilesystemPayloadStore]:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    return Orchestrator(db), db, payload_store


# ---------------------------------------------------------------------------
# Baseline: no peer state → loops are no-ops → run completes.
# ---------------------------------------------------------------------------


def test_no_peer_state_run_completes_normally(tmp_path: Path) -> None:
    """N=1, no peer: 4b-pre and 4b are immediate no-ops; the run COMPLETEs."""
    orch, db, payload_store = _make_orch(tmp_path)

    def _no_seed(_run_id: str) -> None:
        pass

    config, graph, sink, _source = _build([{"id": 1, "value": 10}, {"id": 2, "value": 20}], _no_seed)
    result = orch.run(config, graph=graph, payload_store=payload_store)

    assert result.status == RunStatus.COMPLETED
    assert len(sink.results) == 2
    db.close()


# ---------------------------------------------------------------------------
# H1-7  Wedged-but-alive peer → bounded timeout, not infinite hang.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_wedged_alive_peer_bounded_wait_times_out_then_raises(tmp_path: Path) -> None:
    """An unexpired, never-resolving peer lease must NOT hang the leader.

    The bounded 4b-pre wait times out (the fast monotonic crosses the
    item-lease plus stall-budget deadline) and falls through to
    has_unresolved_scheduler_work(), which raises naming the still-leased peer.
    Pre-fix this loop is unbounded and the test would hang (guarded by the
    60s timeout).
    """
    orch, db, payload_store = _make_orch(tmp_path)

    def _seed(run_id: str) -> None:
        # Unexpired lease, far in the future → reaper can never recover it.
        _seed_peer_leased_row(
            db, payload_store, run_id, ingest_sequence=0, lease_expires_offset=10_000.0, source_node_id=_source_node_id(db, run_id)
        )

    config, graph, _sink, _source = _build([{"id": 1, "value": 10}], _seed)

    import structlog.testing

    with (
        structlog.testing.capture_logs() as captured,
        patch("elspeth.engine.orchestrator.core.time.sleep", lambda _s: None),
        patch("elspeth.engine.orchestrator.core.time.monotonic", _FastMonotonic()),
        pytest.raises(OrchestrationInvariantError) as exc_info,
    ):
        orch.run(config, graph=graph, payload_store=payload_store)

    # The bounded wait fired (did not hang) and named the still-leased peer.
    timeout_logs = [e for e in captured if str(e.get("event", "")).startswith("Bounded peer-lease wait timed out")]
    assert timeout_logs, "the bounded peer-lease wait must time out and log (not hang)"
    assert PEER_OWNER in timeout_logs[0]["still_leased_peers"]
    assert timeout_logs[0]["waited_seconds"] == pytest.approx(300.0 + DEFAULT_ITEM_STALL_BUDGET_SECONDS)
    # And it fell through to the SPECIFIC unresolved-work invariant raise in
    # core.py (the LEASED row is unresolved) — not some other OrchestrationInvariantError.
    assert "non-terminal scheduler work" in str(exc_info.value).lower()
    db.close()


# ---------------------------------------------------------------------------
# H1-8  Deposed leader during wait → breaks out via the latch.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_deposed_leader_during_wait_breaks_out_via_latch(tmp_path: Path) -> None:
    """check_coordination_latch raising RunWorkerEvictedError inside the wait
    must break the leader out (it does not spin to the timeout).

    We seed an unexpired peer lease so the wait loop is entered, then force the
    heartbeat latch to raise on its first in-loop poll. Pre-fix the latch was
    ignored, so the loop would spin (and, unbounded, hang).
    """
    orch, db, payload_store = _make_orch(tmp_path)

    def _seed(run_id: str) -> None:
        _seed_peer_leased_row(
            db, payload_store, run_id, ingest_sequence=0, lease_expires_offset=10_000.0, source_node_id=_source_node_id(db, run_id)
        )

    config, graph, _sink, _source = _build([{"id": 1, "value": 10}], _seed)

    # The latch is RunHeartbeatThread.check_and_raise. Patch it to raise once a
    # peer lease exists (i.e. we are in the 4b-pre wait). It is also called
    # per-row in the source loop, so those earlier calls must pass through.
    from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread

    def _latch(self: Any) -> None:
        # Raise only once a peer lease has been seeded (i.e. we are in 4b-pre).
        from sqlalchemy import func

        with db.engine.connect() as conn:
            peer_leased = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.lease_owner == PEER_OWNER)
                .where(token_work_items_table.c.status == TokenWorkStatus.LEASED.value)
            ).scalar_one()
        if peer_leased and self._token is not None:
            raise RunWorkerEvictedError(worker_id=self._token.worker_id, run_id=self._token.run_id)

    with (
        patch("elspeth.engine.orchestrator.core.time.sleep", lambda _s: None),
        patch("elspeth.engine.orchestrator.core.time.monotonic", _FastMonotonic()),
        patch.object(RunHeartbeatThread, "check_and_raise", _latch),
        pytest.raises(RunWorkerEvictedError),
    ):
        orch.run(config, graph=graph, payload_store=payload_store)

    db.close()


# ---------------------------------------------------------------------------
# H1-9  Dead peer mid-lease → reaped to READY within the liveness window.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_dead_peer_expired_lease_reaped_to_ready_in_loop(tmp_path: Path) -> None:
    """A dead peer's EXPIRED lease is actively reaped to READY by the in-loop
    reap_expired_peer_leases — within the liveness window, not the 300s item TTL.

    Two peer leases are seeded: a still-ALIVE one (unexpired, keeps the bounded
    wait loop running) and a DEAD one (already-expired lease, no run_workers
    registry row → owner_registry_dead). Each loop iteration drives
    reap_expired_peer_leases, which recovers the dead one to READY (attempt
    bumped) while leaving the live one LEASED. Pre-fix the wait loop performed NO
    reap, so the dead peer's row would sit LEASED until its natural TTL.

    The live peer never resolves, so the bounded wait times out and the run
    raises — but the dead peer's row has already been reaped to READY (the
    contract under test).

    NOTE on "within the liveness window": the dead peer here is seeded with an
    ALREADY-expired lease AND no run_workers registry row, so it is
    owner_registry_dead and immediately reapable on the FIRST in-loop iteration —
    no real wall-clock wait is needed to demonstrate the reap. The "within the
    liveness window, not the 300s item TTL" claim is the GENERAL slice-4
    liveness-aware reap guarantee (a peer whose heartbeat stops is reaped once its
    lease expires within the liveness window); this test pins that the in-loop
    reaper actually FIRES, using the simplest already-reapable fixture.
    """
    orch, db, payload_store = _make_orch(tmp_path)
    dead_token: dict[str, str] = {}
    live_token: dict[str, str] = {}

    def _seed(run_id: str) -> None:
        src = _source_node_id(db, run_id)
        # Live peer (unexpired) keeps has_peer_active_leases() True → loop runs.
        live_token["id"] = _seed_peer_leased_row(
            db, payload_store, run_id, ingest_sequence=0, lease_expires_offset=10_000.0, source_node_id=src
        )
        # Dead peer (expired lease, no registry row) → reapable each iteration.
        dead_token["id"] = _seed_peer_leased_row(
            db, payload_store, run_id, ingest_sequence=1, lease_expires_offset=-1.0, source_node_id=src
        )

    config, graph, _sink, _source = _build([{"id": 1, "value": 10}], _seed)

    with (
        patch("elspeth.engine.orchestrator.core.time.sleep", lambda _s: None),
        patch("elspeth.engine.orchestrator.core.time.monotonic", _FastMonotonic()),
        pytest.raises(OrchestrationInvariantError),
    ):
        orch.run(config, graph=graph, payload_store=payload_store)

    with db.engine.connect() as conn:
        rows = {
            r["token_id"]: dict(r)
            for r in conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.attempt,
                ).where(token_work_items_table.c.token_id.in_([dead_token["id"], live_token["id"]]))
            ).mappings()
        }
    # Dead peer's expired lease was REAPED to READY (attempt bumped) in-loop.
    assert rows[dead_token["id"]]["status"] == TokenWorkStatus.READY.value, "dead peer's expired lease must be reaped to READY in-loop"
    assert rows[dead_token["id"]]["attempt"] == 2, "reaper bumps attempt on recovery"
    # Live peer's unexpired lease was LEFT LEASED (liveness gate protected it).
    assert rows[live_token["id"]]["status"] == TokenWorkStatus.LEASED.value, "a live peer's unexpired lease must NOT be reaped"
    db.close()


# ---------------------------------------------------------------------------
# M3-5  Late follower PENDING_SINK → drained to the sink exactly once.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_late_follower_pending_sink_is_drained_exactly_once(tmp_path: Path) -> None:
    """A follower PENDING_SINK row present at finalize is DRAINED by the looped
    4b and written to the sink exactly once — the run COMPLETEs, no lost row.

    The follower's parked row is attributed to PEER_OWNER. The leader's 4b loop
    (has_scheduled_work() True) claims it via claim_pending_sink, reconstructs
    its RowResult, and flushes it to the configured sink. We assert: the run
    COMPLETED; the follower's row reached TERMINAL; and the follower's row value
    appears in the sink output exactly once alongside the leader's own rows.
    """
    orch, db, payload_store = _make_orch(tmp_path)
    follower_token: dict[str, str] = {}

    def _seed(run_id: str) -> None:
        follower_token["id"] = _seed_follower_pending_sink_row(
            db, payload_store, run_id, ingest_sequence=0, source_node_id=_source_node_id(db, run_id), value=999
        )

    config, graph, sink, _source = _build([{"id": 1, "value": 10}], _seed)

    # No peer LEASE is held, so 4b-pre is a no-op; 4b drains the PENDING_SINK.
    result = orch.run(config, graph=graph, payload_store=payload_store)

    assert result.status == RunStatus.COMPLETED, "the run must COMPLETE after draining the follower's PENDING_SINK row"

    # The follower's parked row reached TERMINAL (drained exactly once).
    with db.engine.connect() as conn:
        status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == follower_token["id"])
        ).scalar_one()
    assert status == TokenWorkStatus.TERMINAL.value, "the follower's PENDING_SINK row must be terminalized after the sink write"

    # The follower's row value appears in the sink output EXACTLY ONCE.
    values = [r.get("value") for r in sink.results]
    assert values.count(999) == 1, f"follower's row must be written exactly once, got values={values}"
    # The leader's own row is also present (no row lost).
    assert 10 in values


@pytest.mark.timeout(60)
def test_late_follower_pending_sink_arriving_after_first_drain_pass_is_looped(tmp_path: Path) -> None:
    """The LOOP-vs-single-pass pin: a follower PENDING_SINK row that arrives AFTER
    the first 4b drain pass returns is STILL drained by the looped 4b.

    A single-pass 4b would write the early row and exit, silently leaving the
    late-arriving row → the run would FAIL the complete_run quiescence arm (safe
    but stuck). The looped 4b re-checks has_scheduled_work() and drains the late
    row too → the run COMPLETEs with BOTH follower rows written exactly once.

    We force the late arrival deterministically by wrapping the processor's
    ``drain_scheduled_work``: the FIRST call drains the early row as normal, then
    injects a SECOND follower PENDING_SINK row durably (modelling a follower that
    parked LEASED→PENDING_SINK after the leader's first claim loop returned). A
    single-pass implementation never sees it; the loop does.
    """
    orch, db, payload_store = _make_orch(tmp_path)
    state: dict[str, Any] = {"src": None, "injected": False, "late_value": 1313}

    def _seed(run_id: str) -> None:
        state["src"] = _source_node_id(db, run_id)
        # Early row: present before the source loop exits (drained on pass 1).
        _seed_follower_pending_sink_row(db, payload_store, run_id, ingest_sequence=0, source_node_id=state["src"], value=1212)

    config, graph, sink, _source = _build([{"id": 1, "value": 10}], _seed)

    orig_drain = RowProcessor.drain_scheduled_work

    def _wrapped_drain(self: RowProcessor, ctx: Any) -> Any:
        results = orig_drain(self, ctx)
        # After the FIRST drain pass returns, inject a late-arriving follower
        # PENDING_SINK row. A single-pass 4b would already be done.
        if not state["injected"]:
            state["injected"] = True
            _seed_follower_pending_sink_row(
                db, payload_store, self._run_id, ingest_sequence=1, source_node_id=state["src"], value=state["late_value"]
            )
        return results

    with patch.object(RowProcessor, "drain_scheduled_work", _wrapped_drain):
        result = orch.run(config, graph=graph, payload_store=payload_store)

    assert result.status == RunStatus.COMPLETED, "the looped 4b must drain the late row so the run COMPLETEs"
    values = [r.get("value") for r in sink.results]
    assert values.count(1212) == 1, f"early follower row written exactly once, got {values}"
    assert values.count(state["late_value"]) == 1, (
        f"the LATE follower row (arrived after the first drain pass) must be drained by the loop, got {values}"
    )
    assert 10 in values, "the leader's own row is present (no row lost)"


# ---------------------------------------------------------------------------
# M3-10  Double-flush guard: leader tokens not re-emitted by the 4b re-flush.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_double_flush_guard_no_duplicate_leader_rows(tmp_path: Path) -> None:
    """When 4b accumulates follower results and re-flushes, the leader's own
    already-written tokens are NOT re-emitted.

    The 4b clears pending_tokens before the second flush (write_pending_to_sinks
    does not consume), so a re-flush would otherwise re-write every leader token
    and hit the node_states UNIQUE constraint. We run a multi-row leader pipeline
    that ALSO has a follower PENDING_SINK row to drain, forcing the second flush,
    and assert: no exception (no node_states IntegrityError); every leader row
    appears in the sink exactly once; no node_states row is duplicated.
    """
    orch, db, payload_store = _make_orch(tmp_path)

    def _seed(run_id: str) -> None:
        _seed_follower_pending_sink_row(db, payload_store, run_id, ingest_sequence=0, source_node_id=_source_node_id(db, run_id), value=888)

    leader_rows = [{"id": 1, "value": 11}, {"id": 2, "value": 22}, {"id": 3, "value": 33}]
    config, graph, sink, _source = _build(leader_rows, _seed)

    result = orch.run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED

    # Each leader row written exactly once (no duplicate from the re-flush).
    values = [r.get("value") for r in sink.results]
    for v in (11, 22, 33):
        assert values.count(v) == 1, f"leader row value={v} must appear exactly once, got {values}"
    # The follower row drained too.
    assert values.count(888) == 1

    # No duplicate node_states rows (the UNIQUE the clear protects).
    from sqlalchemy import func

    with db.engine.connect() as conn:
        dup = conn.execute(
            select(
                node_states_table.c.token_id,
                node_states_table.c.node_id,
                node_states_table.c.attempt,
                func.count().label("n"),
            )
            .where(node_states_table.c.run_id == result.run_id)
            .group_by(node_states_table.c.token_id, node_states_table.c.node_id, node_states_table.c.attempt)
            .having(func.count() > 1)
        ).fetchall()
    assert dup == [], f"no node_states row may be written twice (double-flush), found {dup}"


# ---------------------------------------------------------------------------
# Test 11  Attribution: follower-processed row carries the follower's owner.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_follower_pending_sink_attribution_and_exactly_one_emission(tmp_path: Path) -> None:
    """A follower-processed row has a mark_pending_sink scheduler_event with
    from_lease_owner == the follower, and a TERMINAL token_work_items row with
    exactly one sink emission — locking the per-worker attribution query.
    """
    orch, db, payload_store = _make_orch(tmp_path)
    follower_token: dict[str, str] = {}

    def _seed(run_id: str) -> None:
        follower_token["id"] = _seed_follower_pending_sink_row(
            db, payload_store, run_id, ingest_sequence=0, source_node_id=_source_node_id(db, run_id), value=777
        )

    config, graph, sink, _source = _build([{"id": 1, "value": 10}], _seed)
    result = orch.run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED

    # A mark_pending_sink scheduler_event attributed to the follower exists.
    with db.engine.connect() as conn:
        events = (
            conn.execute(
                select(scheduler_events_table.c.event_type, scheduler_events_table.c.from_lease_owner).where(
                    scheduler_events_table.c.run_id == result.run_id
                )
            )
            .mappings()
            .all()
        )
    pending_sink_events = [e for e in events if e["event_type"] == "mark_pending_sink" and e["from_lease_owner"] == PEER_OWNER]
    assert pending_sink_events, f"a mark_pending_sink event attributed to {PEER_OWNER!r} must exist; events={[dict(e) for e in events]}"

    # The follower's row is TERMINAL with exactly one sink emission.
    with db.engine.connect() as conn:
        status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.token_id == follower_token["id"])
        ).scalar_one()
    assert status == TokenWorkStatus.TERMINAL.value
    assert [r.get("value") for r in sink.results].count(777) == 1


# ---------------------------------------------------------------------------
# Test 14  End-to-end exactly-once at N>=2 (leader + follower-attributed work).
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_end_to_end_exactly_once_with_follower_attributed_work(tmp_path: Path) -> None:
    """End-to-end exactly-once across a leader and follower-attributed work.

    The leader processes its own source rows; TWO follower-attributed
    PENDING_SINK continuations (parked under PEER_OWNER) are drained by the
    leader's 4b. We assert end-to-end:

      - every token_work_items row is TERMINAL (none stranded);
      - every sink-routed value appears in the sink output EXACTLY ONCE (no lost
        row, no double write);
      - the leader CollectSink holds each row once (the follower never wrote the
        sink itself — it only marked PENDING_SINK; the leader did the sink I/O);
      - at least two distinct continuations carry the follower's from_lease_owner
        on their mark_pending_sink scheduler_event.

    COVERAGE NOTE: "the follower never wrote the sink itself" is asserted here only
    for the in-memory CollectSink (no file I/O). The latent "follower truncates the
    shared sink file in on_start" hazard is downgraded to low/latent in the review
    precisely because lazy-I/O sinks (e.g. JSONSink) open the file only at first
    write() and close() is a no-op when never opened — and the follower never
    writes. A real-file sink (JSONSink) asserting the follower's output path was
    never created would complete this coverage; tracked as a follow-up, not a
    blocker (CollectSink is the right harness for the exactly-once contract).
    """
    orch, db, payload_store = _make_orch(tmp_path)
    follower_tokens: list[str] = []

    def _seed(run_id: str) -> None:
        src = _source_node_id(db, run_id)
        follower_tokens.append(_seed_follower_pending_sink_row(db, payload_store, run_id, ingest_sequence=0, source_node_id=src, value=501))
        follower_tokens.append(_seed_follower_pending_sink_row(db, payload_store, run_id, ingest_sequence=1, source_node_id=src, value=502))

    leader_rows = [{"id": 1, "value": 101}, {"id": 2, "value": 102}]
    config, graph, sink, _source = _build(leader_rows, _seed)

    result = orch.run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED

    # Every token_work_items row is TERMINAL — none stranded.
    with db.engine.connect() as conn:
        statuses = [
            r[0]
            for r in conn.execute(
                select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == result.run_id)
            ).fetchall()
        ]
    assert statuses, "the run must have scheduler rows"
    assert all(s == TokenWorkStatus.TERMINAL.value for s in statuses), f"every work item must be TERMINAL, got {statuses}"

    # Exactly-once at the sink: leader rows + both follower rows, each once.
    values = [r.get("value") for r in sink.results]
    for v in (101, 102, 501, 502):
        assert values.count(v) == 1, f"value {v} must appear in the sink exactly once, got {values}"
    assert len(values) == 4, f"exactly 4 rows written (2 leader + 2 follower-drained), got {values}"

    # >=2 follower-attributed mark_pending_sink events (per-worker attribution).
    with db.engine.connect() as conn:
        follower_pending = conn.execute(
            select(scheduler_events_table.c.token_id)
            .where(scheduler_events_table.c.run_id == result.run_id)
            .where(scheduler_events_table.c.event_type == "mark_pending_sink")
            .where(scheduler_events_table.c.from_lease_owner == PEER_OWNER)
        ).fetchall()
    assert len(follower_pending) >= 2, "at least two follower-attributed mark_pending_sink events expected"
