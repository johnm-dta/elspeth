"""E2E crash + concurrent-resume proofs over the durable scheduler journal.

Ticket elspeth-40886ef9f8: the old ``test_concurrent_resume.py`` only tested
resume *rejection* (now ``test_resume_rejection.py``). This file is the real
coverage for the three gaps that ticket names:

1. **Mid-claim crash, then resume** — a worker dies holding a LEASED
   ``token_work_items`` row; a later resume's recovery sweep reclaims it
   after lease expiry (attempt bump + work_item_id rotation), the run
   completes, and the audit DB shows exactly-once terminal token outcomes.
2. **Expired-lease reclaim under contention** — a population of crashed
   workers' leases with different expiries: the drain refuses atomically
   while ANY peer lease is still live (ADR-026 Precondition #9), then
   reclaims everything once all leases age out, with per-item attempt
   offsets visible in both the journal ``attempt`` column and the
   ``node_states`` attempt identity.
3. **Two resume() calls racing on the same run_id** — two interleavings,
   two registers. The loser-after-winner interleaving is CHARACTERIZATION
   (the run-immutability guard refuses it; full cross-process mutual
   exclusion remains an OPEN operator decision — option c). The mid-flight
   interleaving is now an ENFORCED CONTRACT: ``ResumeCoordinator.resume()``'s
   entry guard (elspeth-2f23292372, operator option b) re-checks run status
   via the same shared implementation ``can_resume`` uses and refuses a
   RUNNING run with ``NonResumableRunError`` before any mutation.

Durability-unification (F1) survival contract: every ASSERTION below reads
PUBLIC, durable surfaces only — the ``token_work_items`` journal columns
(status/attempt/lease_owner), ``scheduler_events``, ``token_outcomes``,
``node_states``, ``runs``, terminal RunStatus, and the public
``Orchestrator.resume()`` API. Nothing asserts on checkpoint internals
(the blob layer is deleted — Task 4.1), and resume points are treated as
opaque handles (never dereferenced).

The HARNESS, by necessity, enters resume through today's checkpoint-layer
surface: ``RecoveryManager.get_resume_point`` and the Orchestrator's
``checkpoint_manager``/``checkpoint_config`` kwargs. That dependency is
confined to exactly three seams — :func:`_run_to_interrupted_checkpoint`
(run setup), :class:`_CrashedRun` (``resume_orchestrator``), and
:func:`_recovery_manager` / :func:`_resume_point` (resume entry, shared by
every test). When F1 re-anchors resume on the scheduler journal and the
checkpoint constructors/signatures change or disappear, the edits land in
those seams only; the assertions themselves survive unchanged.

Construction note (kept honest, same technique as
test_rc6_eof_resume_proof.py): the crashed-mid-claim state cannot be reached
by a graceful shutdown — the engine always finishes the in-flight row before
honoring the shutdown event — and a real SIGKILL is not deterministic inside
a unit process. So each test first runs the REAL pipeline (real Orchestrator,
real checkpoint writer, real scheduler journal) to an interrupted-but-
checkpointed state, then crafts the kill instant through the production
Tier-1 writers themselves: ``RecorderFactory.data_flow.create_row`` /
``create_token`` for the row the dead worker was carrying, and
``TokenSchedulerRepository.enqueue_ready`` + ``claim_ready`` for its LEASED
journal row — the exact rows the engine writes before a hard kill. Two
columns are flipped to the engine's own crash-classification values
(``run_sources.lifecycle_state='loaded'`` — the completed-iteration value —
and ``runs.status=FAILED``). The resume side is then driven exclusively
through the public production path with an injected deterministic MockClock
(no sleeps, no monkeypatching of code under test).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from sqlalchemy import func, select

from elspeth.contracts import Determinism, PipelineRow, PluginSchema, RunStatus
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.errors import AuditIntegrityError, GracefulShutdownError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.checkpoint.recovery import NonResumableRunError
from elspeth.core.config import CheckpointSettings, QueueSettings, SourceSettings, TransformSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    node_states_table,
    rows_table,
    run_sources_table,
    runs_table,
    scheduler_events_table,
    token_outcomes_table,
    token_work_items_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.clock import MockClock
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink

# Deterministic epoch for the injected MockClock (UTC datetimes derive from it).
_T0 = 1_750_000_000.0

# The production default lease (processor.py scheduler_lease_seconds=300) —
# leases crafted at this value mirror what a real dead worker leaves behind.
_DEFAULT_LEASE_SECONDS = 300

_SOURCE_ROWS = [{"id": i, "value": i * 10} for i in range(3)]


class _CrashRowSchema(PluginSchema):
    """Typed row schema — resume reconstruction requires non-empty fields."""

    id: int
    value: int


class _InterruptibleSource(_TestSourceBase):
    """Source that requests graceful shutdown while yielding its final row.

    Also counts ``load()`` invocations: the scheduler-drain resume path must
    never replay the source (rows travel via the durable journal payload).
    """

    name = "interruptible_source"
    output_schema = _CrashRowSchema
    determinism = Determinism.IO_READ

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        on_success: str,
        shutdown_event: threading.Event | None = None,
        interrupt_after: int | None = None,
    ) -> None:
        super().__init__()
        self._rows = rows
        self.on_success = on_success
        self._event = shutdown_event
        self._interrupt_after = interrupt_after
        self.load_invocations = 0

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        self.load_invocations += 1
        for source_row_index, row in enumerate(self._rows):
            if self._event is not None and self._interrupt_after is not None and source_row_index + 1 >= self._interrupt_after:
                self._event.set()
            contract = _observed_contract(row)
            self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)


class _PassthroughTransform(BaseTransform):
    """Identity transform so the pipeline has a real mid-DAG node."""

    name = "passthrough"
    determinism = Determinism.DETERMINISTIC
    input_schema: ClassVar[type[PluginSchema]] = _CrashRowSchema
    output_schema: ClassVar[type[PluginSchema]] = _CrashRowSchema
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.on_success = "output"
        self.input = "inbound"

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "pass"})


def _observed_contract(row: dict[str, Any]) -> SchemaContract:
    fields = tuple(
        FieldContract(
            normalized_name=key,
            original_name=key,
            python_type=object,
            required=False,
            source="inferred",
        )
        for key in row
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def _build_pipeline(
    rows: list[dict[str, Any]],
    *,
    shutdown_event: threading.Event | None = None,
    interrupt_after: int | None = None,
) -> tuple[PipelineConfig, ExecutionGraph, CollectSink, _InterruptibleSource]:
    """Real production graph: source -> queue 'inbound' -> transform -> sink."""
    source = _InterruptibleSource(rows, on_success="inbound", shutdown_event=shutdown_event, interrupt_after=interrupt_after)
    transform = _PassthroughTransform()
    sink = CollectSink("output")
    sources = {"primary": as_source(source)}
    wired = WiredTransform(
        plugin=as_transform(transform),
        settings=TransformSettings(
            name="passthrough_0",
            plugin=transform.name,
            input="inbound",
            on_success="output",
            on_error="discard",
            options={},
        ),
    )
    graph = ExecutionGraph.from_plugin_instances(
        sources=sources,
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="inbound", options={})},
        transforms=[wired],
        sinks={"output": as_sink(sink)},
        queues={"inbound": QueueSettings(description="crash-resume fan-in")},
    )
    config = PipelineConfig(sources=sources, transforms=[as_transform(transform)], sinks={"output": as_sink(sink)})
    return config, graph, sink, source


@dataclass
class _CrashedRun:
    """Handles to a run reshaped into the crashed-mid-claim state."""

    db: LandscapeDB
    payload_store: FilesystemPayloadStore
    checkpoint_mgr: CheckpointManager
    checkpoint_config: RuntimeCheckpointConfig
    clock: MockClock
    run_id: str
    graph: ExecutionGraph
    repo: TokenSchedulerRepository
    factory: RecorderFactory
    source_node_id: str
    journal_node_id: str
    journal_step_index: int
    crashed_token_ids: dict[int, str]  # ingest_sequence -> token_id

    def resume_orchestrator(self) -> Orchestrator:
        return Orchestrator(
            self.db,
            checkpoint_manager=self.checkpoint_mgr,
            checkpoint_config=self.checkpoint_config,
            clock=self.clock,
        )


def _recovery_manager(crashed: _CrashedRun) -> RecoveryManager:
    """SINGLE construction seam for the checkpoint-coupled recovery entry.

    Today's RecoveryManager constructor takes a CheckpointManager; when the
    durability unification (F1) reshapes resume entry around the scheduler
    journal, this helper (plus :func:`_resume_point` below) is the only
    place in this file that changes.
    """
    return RecoveryManager(crashed.db, crashed.checkpoint_mgr)


def _resume_point(crashed: _CrashedRun) -> Any:
    """Fetch an OPAQUE resume point via the public recovery API.

    Every resume entry in this file funnels through here. Callers never
    dereference the returned handle — they only pass it back to
    ``Orchestrator.resume()`` — so its shape is free to change under F1.
    """
    return _recovery_manager(crashed).get_resume_point(crashed.run_id, crashed.graph)


def _run_to_interrupted_checkpoint(tmp_path: Path, clock: MockClock) -> _CrashedRun:
    """Run the REAL pipeline to an interrupted, checkpointed state.

    The graceful shutdown fires while the final source row is being yielded,
    so all three rows are fully processed (terminal outcomes + sink writes +
    post-sink checkpoints) and the run lands FAILED-with-checkpoint — the
    production precondition for resume. The crash instant is then crafted on
    top via :func:`_craft_crashed_lease`.
    """
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    checkpoint_mgr = CheckpointManager(db)
    checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

    shutdown_event = threading.Event()
    config, graph, _sink, _source = _build_pipeline(
        _SOURCE_ROWS,
        shutdown_event=shutdown_event,
        interrupt_after=len(_SOURCE_ROWS),
    )
    orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config, clock=clock)
    with pytest.raises(GracefulShutdownError) as exc_info:
        orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)
    run_id = exc_info.value.run_id
    assert run_id is not None

    # The engine recorded 'interrupted' for the source because the shutdown
    # event was honored before the iterator's StopIteration. At the simulated
    # kill instant the source HAD fully completed iteration, so flip the one
    # column to the engine's own completed-iteration value ('loaded' — the
    # same value finalize_source_iteration records, and the value resume.py's
    # _SOURCE_COMPLETE_LIFECYCLE_STATES accepts). FAILED is what the failure
    # ceremony records when the process dies abnormally.
    with db.engine.begin() as conn:
        conn.execute(
            run_sources_table.update()
            .where(run_sources_table.c.run_id == run_id, run_sources_table.c.source_name == "primary")
            .values(lifecycle_state="loaded")
        )
        conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(status=RunStatus.FAILED))

    # Mirror the journal shape the real run wrote (node_id is the queue node,
    # step_index its DAG step) instead of hardcoding hash-suffixed node IDs.
    with db.engine.connect() as conn:
        sample = conn.execute(select(token_work_items_table).where(token_work_items_table.c.run_id == run_id).limit(1)).mappings().one()
        source_node_id = conn.execute(select(rows_table.c.source_node_id).where(rows_table.c.run_id == run_id).limit(1)).scalar_one()

    crashed = _CrashedRun(
        db=db,
        payload_store=payload_store,
        checkpoint_mgr=checkpoint_mgr,
        checkpoint_config=checkpoint_config,
        clock=clock,
        run_id=run_id,
        graph=graph,
        repo=TokenSchedulerRepository(db.engine),
        factory=RecorderFactory(db, payload_store=payload_store),
        source_node_id=str(source_node_id),
        journal_node_id=str(sample["node_id"]),
        journal_step_index=int(sample["step_index"]),
        crashed_token_ids={},
    )
    # Setup precondition via the PUBLIC recovery API (not a direct
    # checkpoint-store read): the reshaped run must be resumable.
    assert _resume_point(crashed) is not None
    return crashed


def _craft_crashed_lease(
    crashed: _CrashedRun,
    *,
    ingest_sequence: int,
    lease_owner: str,
    lease_seconds: int,
) -> str:
    """Write the rows a hard-killed worker leaves behind, via production writers.

    A worker killed mid-claim has durably written, in order: the source row
    (``rows``), its token (``tokens``), the READY journal enqueue, and the
    LEASED claim under its own ``lease_owner`` — and nothing else (no
    node_states, no outcome: the kill predates them). Every one of those
    writes below goes through the same Tier-1 production writer the engine
    uses (RecorderFactory / TokenSchedulerRepository), never raw SQL.

    Returns the crashed token_id.
    """
    data = {"id": ingest_sequence, "value": ingest_sequence * 10}
    row = crashed.factory.data_flow.create_row(
        run_id=crashed.run_id,
        source_node_id=crashed.source_node_id,
        row_index=ingest_sequence,
        data=data,
        source_row_index=ingest_sequence,
        ingest_sequence=ingest_sequence,
    )
    token = crashed.factory.data_flow.create_token(row_id=row.row_id)
    crashed.repo.enqueue_ready(
        run_id=crashed.run_id,
        token_id=token.token_id,
        row_id=row.row_id,
        node_id=crashed.journal_node_id,
        step_index=crashed.journal_step_index,
        ingest_sequence=ingest_sequence,
        row_payload_json=TokenSchedulerRepository.serialize_row_payload(PipelineRow(data, _observed_contract(data))),
        available_at=crashed.clock.now_utc(),
    )
    claimed = crashed.repo.claim_ready(
        run_id=crashed.run_id,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
        now=crashed.clock.now_utc(),
    )
    assert claimed is not None and claimed.token_id == token.token_id
    crashed.crashed_token_ids[ingest_sequence] = token.token_id
    return token.token_id


def _resume(crashed: _CrashedRun) -> tuple[Any, CollectSink, _InterruptibleSource]:
    """Drive the PUBLIC resume path with fresh plugins (new-process reality)."""
    resume_point = _resume_point(crashed)
    assert resume_point is not None
    config, graph, sink, source = _build_pipeline(_SOURCE_ROWS)
    result = crashed.resume_orchestrator().resume(resume_point, config, graph, payload_store=crashed.payload_store)
    return result, sink, source


def _work_items_by_token(db: LandscapeDB, run_id: str) -> dict[str, dict[str, Any]]:
    with db.engine.connect() as conn:
        return {
            row["token_id"]: dict(row)
            for row in conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.work_item_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.attempt,
                    token_work_items_table.c.lease_owner,
                    token_work_items_table.c.ingest_sequence,
                ).where(token_work_items_table.c.run_id == run_id)
            ).mappings()
        }


def _duplicate_terminal_outcome_tokens(db: LandscapeDB, run_id: str) -> list[str]:
    """Token IDs with more than one completed terminal outcome (must be [])."""
    with db.engine.connect() as conn:
        return list(
            conn.execute(
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id, token_outcomes_table.c.completed == 1)
                .group_by(token_outcomes_table.c.token_id)
                .having(func.count() > 1)
            )
            .scalars()
            .all()
        )


def _completed_outcome_tokens(db: LandscapeDB, run_id: str) -> set[str]:
    with db.engine.connect() as conn:
        return set(
            conn.execute(
                select(token_outcomes_table.c.token_id).where(
                    token_outcomes_table.c.run_id == run_id, token_outcomes_table.c.completed == 1
                )
            )
            .scalars()
            .all()
        )


def _node_state_identities(db: LandscapeDB, run_id: str) -> list[tuple[str, str, int]]:
    with db.engine.connect() as conn:
        return [
            (str(row.token_id), str(row.node_id), int(row.attempt))
            for row in conn.execute(
                select(node_states_table.c.token_id, node_states_table.c.node_id, node_states_table.c.attempt).where(
                    node_states_table.c.run_id == run_id
                )
            ).all()
        ]


def _recovery_events(db: LandscapeDB, run_id: str) -> list[dict[str, Any]]:
    """RECOVER_EXPIRED_LEASE events in durable insertion (rowid) order."""
    from sqlalchemy import text

    with db.engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == run_id)
                .where(scheduler_events_table.c.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value)
                .order_by(text("rowid"))
            ).mappings()
        ]


@pytest.mark.timeout(120)
class TestMidClaimCrashResume:
    """Ticket item 1: mid-claim crash -> lease expiry -> sweep -> completion."""

    def test_mid_claim_crash_resume_recovers_leased_item_exactly_once(self, tmp_path: Path) -> None:
        """A LEASED journal row left by a dead worker is recovered exactly once.

        Invariants pinned (all durable surfaces):
        - the public resume claims the crashed item only AFTER lease expiry,
          via the recovery sweep: the journal row rotates to attempt=2 and
          reaches 'terminal'; exactly one RECOVER_EXPIRED_LEASE event exists
          (from_attempt=1, to_attempt=2, from_lease_owner=the dead worker,
          caller_owner=a different identity — the G1 self-steal guard);
        - the run completes COMPLETED with cumulative rows_processed=4 and
          the recovered row reaches the sink exactly once;
        - exactly-once token outcomes: every token has exactly one completed
          terminal outcome — including the crashed token, under its ORIGINAL
          token_id (the journal re-drive does not mint a replacement token);
        - the re-drive records the transform node_state at attempt=1 (the
          journal attempt offset, claimed.attempt-1), with no attempt-0 row
          for the crashed token (the dead worker never got that far) and no
          duplicate (token_id, node_id, attempt) identity anywhere;
        - no source replay: the resume-side source plugin is never load()ed.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )

        # The dead worker's lease must age out before recovery is possible.
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        result, resume_sink, resume_source = _resume(crashed)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 4
        assert resume_sink.results == [{"id": 3, "value": 30}]
        assert resume_source.load_invocations == 0, "scheduler-drain resume must not replay the source"

        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert len(items) == 4
        assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items.values())
        assert items[crashed_token]["attempt"] == 2
        assert all(item["attempt"] == 1 for token_id, item in items.items() if token_id != crashed_token)

        events = _recovery_events(crashed.db, crashed.run_id)
        assert len(events) == 1
        (event,) = events
        assert event["token_id"] == crashed_token
        assert event["from_attempt"] == 1
        assert event["to_attempt"] == 2
        assert event["from_lease_owner"] == "crashed-worker-1"
        assert event["caller_owner"] != "crashed-worker-1"

        # Exactly-once terminal outcomes, for the original token identity.
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        completed_tokens = _completed_outcome_tokens(crashed.db, crashed.run_id)
        assert crashed_token in completed_tokens
        assert len(completed_tokens) == 4

        # node_states attempt identity: re-drive at the journal attempt offset.
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities)), "duplicate node_states identity"
        crashed_attempts = {(node_id, attempt) for token_id, node_id, attempt in identities if token_id == crashed_token}
        transform_attempts = {attempt for node_id, attempt in crashed_attempts if node_id.startswith("transform_")}
        assert transform_attempts == {1}, f"expected re-drive at attempt offset 1, got {sorted(transform_attempts)}"

        # Full completion is already durably proven above (terminal
        # RunStatus, all-TERMINAL journal, exactly-once outcomes); the
        # checkpoint store's delete-on-completion lifecycle is deliberately
        # NOT asserted — that layer dissolves with F1.
        crashed.db.close()


@pytest.mark.timeout(120)
class TestExpiredLeaseReclaimUnderContention:
    """Ticket item 2: a population of crashed leases with attempt offsets."""

    def test_contended_reclaim_is_atomic_and_bumps_attempts_with_offset_identity(self, tmp_path: Path) -> None:
        """Two crashed workers' leases, different expiries and attempt history.

        Construction (every journal transition written by the production
        repository): worker-a dies holding token-3 at attempt=1 (300s lease).
        token-4 was already recovered once during the prior process's own
        maintenance sweep (a REAL recover_expired_leases call) and re-claimed
        by worker-b at attempt=2 with a long 7200s lease before the kill.

        Invariants pinned:
        - PARTIAL expiry refuses ATOMICALLY: a resume at a time when token-3's
          lease has expired but worker-b's is still live raises
          AuditIntegrityError naming the live peer (ADR-026 Precondition #9),
          and the refusal touches NEITHER journal row — worker-a's expired
          lease is NOT half-recovered (still LEASED/attempt=1/owner intact);
        - after all leases expire, one resume recovers BOTH: attempts bump
          1->2 (token-3) and 2->3 (token-4), recovery events carry the dead
          owners' identities, and recovery order follows ingest_sequence;
        - attempt offsets land in the audit DB: the re-driven transform
          node_state is recorded at attempt = journal_attempt - 1 (1 for
          token-3, 2 for token-4) with no duplicate node_states identities;
        - exactly-once terminal outcomes across all 5 tokens, run COMPLETED.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)

        token_3 = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-a",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        token_4 = _craft_crashed_lease(
            crashed,
            ingest_sequence=4,
            lease_owner="crashed-worker-b",
            lease_seconds=60,
        )

        # Production attempt history for token-4: its first lease expires and
        # the prior process's own maintenance sweep (a different owner — the
        # G1 self-steal guard requires it) recovers it, then worker-b
        # re-claims at attempt=2 with a long lease... and dies holding it.
        clock.advance(120)  # token-4's 60s lease is expired; token-3's 300s is not
        assert (
            crashed.repo.recover_expired_leases(
                run_id=crashed.run_id,
                now=clock.now_utc(),
                caller_owner="row-processor:prior-attempt-sweeper",
            )
            == 1
        )
        reclaimed = crashed.repo.claim_ready(
            run_id=crashed.run_id,
            lease_owner="crashed-worker-b",
            lease_seconds=7200,
            now=clock.now_utc(),
        )
        assert reclaimed is not None and reclaimed.token_id == token_4 and reclaimed.attempt == 2

        # ---- Resume #1: token-3 expired, worker-b's 7200s lease still live ----
        clock.advance(3600)
        resume_point = _resume_point(crashed)
        assert resume_point is not None
        config_1, graph_1, sink_1, _source_1 = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(AuditIntegrityError, match=r"peer worker\(s\).*crashed-worker-b"):
            crashed.resume_orchestrator().resume(resume_point, config_1, graph_1, payload_store=crashed.payload_store)
        assert sink_1.results == []

        # Atomicity: the refused drain recovered NOTHING — not even the
        # already-expired worker-a lease.
        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert items[token_3]["status"] == TokenWorkStatus.LEASED.value
        assert items[token_3]["attempt"] == 1
        assert items[token_3]["lease_owner"] == "crashed-worker-a"
        assert items[token_4]["status"] == TokenWorkStatus.LEASED.value
        assert items[token_4]["attempt"] == 2
        assert items[token_4]["lease_owner"] == "crashed-worker-b"

        # The failure ceremony re-finalized FAILED, so the run stays resumable.
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert status == RunStatus.FAILED.value

        # ---- Resume #2: every crashed lease has aged out ----
        clock.advance(7200)
        result, resume_sink, _resume_source = _resume(crashed)

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5
        assert resume_sink.results == [{"id": 3, "value": 30}, {"id": 4, "value": 40}]

        items = _work_items_by_token(crashed.db, crashed.run_id)
        assert all(item["status"] == TokenWorkStatus.TERMINAL.value for item in items.values())
        assert items[token_3]["attempt"] == 2
        assert items[token_4]["attempt"] == 3

        # Recovery journal: the crafted prior-process sweep (token-4 1->2),
        # then the final resume's sweep in ingest_sequence order.
        events = _recovery_events(crashed.db, crashed.run_id)
        assert [(event["token_id"], event["from_attempt"], event["to_attempt"], event["from_lease_owner"]) for event in events] == [
            (token_4, 1, 2, "crashed-worker-b"),
            (token_3, 1, 2, "crashed-worker-a"),
            (token_4, 2, 3, "crashed-worker-b"),
        ]
        final_sweep_callers = {event["caller_owner"] for event in events[1:]}
        assert len(final_sweep_callers) == 1
        assert final_sweep_callers.isdisjoint({"crashed-worker-a", "crashed-worker-b"})

        # Attempt offsets in the audit DB: node_states attempt identity is
        # journal_attempt - 1 for the re-driven transform segment.
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities)), "duplicate node_states identity"

        def _transform_attempts(token_id: str) -> set[int]:
            return {attempt for tid, node_id, attempt in identities if tid == token_id and node_id.startswith("transform_")}

        assert _transform_attempts(token_3) == {1}
        assert _transform_attempts(token_4) == {2}

        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert len(_completed_outcome_tokens(crashed.db, crashed.run_id)) == 5
        crashed.db.close()


@pytest.mark.timeout(120)
class TestTwoResumesSameRunId:
    """Ticket item 3 — the two deterministic interleavings of a resume race.

    Full cross-process resume mutual exclusion has NO designed policy today
    (operator option c, post-F1). The two interleavings are pinned in
    different registers:

    - loser-after-winner: CHARACTERIZATION of the run-immutability guard
      (``update_run_status`` raises AuditIntegrityError from COMPLETED) —
      what IS, recorded as input to the open operator decision;
    - loser-while-RUNNING: ENFORCED CONTRACT — the resume() entry guard
      (elspeth-2f23292372, option b) refuses before any mutation.
    """

    def test_characterization_two_resumes_same_run_id_loser_after_winner(self, tmp_path: Path) -> None:
        """Both resumes witness the same FAILED state; the loser starts last.

        Observed behavior today (the characterization):
        - WINNER: completes the run normally (COMPLETED, recovered row at
          the sink exactly once).
        - LOSER: ``resume()`` is REFUSED — not by any resume-specific mutual
          exclusion, but by the run-status immutability guard:
          ``update_run_status`` raises AuditIntegrityError("... from
          COMPLETED ... Successful terminal runs are immutable") during
          resume-state reconstruction, BEFORE any row is touched.
        - AUDIT INTEGRITY HOLDS: no duplicate terminal token outcomes, no
          duplicate node_states identities, the journal rows are untouched
          by the loser, runs.completed_at is unchanged, and the loser's
          sink receives nothing.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        # Both racing operators fetch equally-valid resume points from the
        # same durable FAILED state — the cross-process race decision point.
        resume_point_winner = _resume_point(crashed)
        resume_point_loser = _resume_point(crashed)
        assert resume_point_winner is not None and resume_point_loser is not None

        config_w, graph_w, sink_w, _source_w = _build_pipeline(_SOURCE_ROWS)
        result_winner = crashed.resume_orchestrator().resume(resume_point_winner, config_w, graph_w, payload_store=crashed.payload_store)
        assert result_winner.status == RunStatus.COMPLETED
        assert sink_w.results == [{"id": 3, "value": 30}]

        with crashed.db.engine.connect() as conn:
            completed_at_after_winner = conn.execute(
                select(runs_table.c.completed_at).where(runs_table.c.run_id == crashed.run_id)
            ).scalar_one()
        items_after_winner = _work_items_by_token(crashed.db, crashed.run_id)

        # The loser arrives with its (now stale) resume point.
        clock.advance(10)
        config_l, graph_l, sink_l, _source_l = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(AuditIntegrityError, match=r"from COMPLETED .*immutable"):
            crashed.resume_orchestrator().resume(resume_point_loser, config_l, graph_l, payload_store=crashed.payload_store)

        # Audit integrity after the race: the loser changed NOTHING.
        assert sink_l.results == []
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert len(_completed_outcome_tokens(crashed.db, crashed.run_id)) == 4
        identities = _node_state_identities(crashed.db, crashed.run_id)
        assert len(identities) == len(set(identities))
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
            completed_at_after_loser = conn.execute(
                select(runs_table.c.completed_at).where(runs_table.c.run_id == crashed.run_id)
            ).scalar_one()
        assert status == RunStatus.COMPLETED.value
        assert completed_at_after_loser == completed_at_after_winner
        assert _work_items_by_token(crashed.db, crashed.run_id) == items_after_winner
        assert items_after_winner[crashed_token]["attempt"] == 2
        crashed.db.close()

    def test_entry_guard_refuses_resume_while_run_status_running(self, tmp_path: Path) -> None:
        """The loser arrives while the winner is mid-flight (status=RUNNING).

        The winner's FIRST durable write on the resume path is
        ``update_run_status(run_id, RUNNING)`` (reconstruct_resume_state
        step 2). This test replays exactly that write through the production
        recorder, then drives a competing ``resume()`` whose resume point was
        fetched before the winner started — the deterministic image of the
        mid-flight race window.

        ENFORCED CONTRACT (elspeth-2f23292372, operator option b — this was
        formerly a characterization of ADMISSION): ``resume()`` re-checks the
        run status at entry through the same shared implementation
        ``can_resume`` uses (``check_run_status_resumable``) and REFUSES the
        RUNNING run with ``NonResumableRunError`` — carrying the run_id and
        the advisory check's reason — BEFORE any mutation.

        KNOWN RESIDUAL (deliberately NOT closed here — operator option c,
        cross-process coordination, post-F1): a TOCTOU window remains. Two
        resumes can BOTH observe FAILED at the guard before either flips the
        run to RUNNING; this guard closes the caller-convention gap only.

        Durable surfaces pinned — the refused loser changed NOTHING:
        - the crashed worker's LEASED journal row is untouched
          (status/attempt/lease_owner intact), no recovery events appended;
        - no new token outcomes, no new node_states identities;
        - the runs row still says RUNNING (the notional winner's state);
        - the loser's sink receives nothing.
        """
        clock = MockClock(start=_T0)
        crashed = _run_to_interrupted_checkpoint(tmp_path, clock)
        crashed_token = _craft_crashed_lease(
            crashed,
            ingest_sequence=3,
            lease_owner="crashed-worker-1",
            lease_seconds=_DEFAULT_LEASE_SECONDS,
        )
        clock.advance(_DEFAULT_LEASE_SECONDS + 60)

        resume_point_loser = _resume_point(crashed)
        assert resume_point_loser is not None

        # The winner's first durable resume write: FAILED -> RUNNING.
        crashed.factory.run_lifecycle.update_run_status(crashed.run_id, RunStatus.RUNNING)

        # Advisory surface refuses...
        check = _recovery_manager(crashed).can_resume(crashed.run_id, crashed.graph)
        assert not check.can_resume
        assert check.reason == "Run is still in progress"

        # Snapshot every durable surface the refused loser must not touch.
        items_before = _work_items_by_token(crashed.db, crashed.run_id)
        outcomes_before = _completed_outcome_tokens(crashed.db, crashed.run_id)
        identities_before = sorted(_node_state_identities(crashed.db, crashed.run_id))

        # ...and the public resume() now refuses with the SAME reason,
        # raised at entry before any mutation.
        config_l, graph_l, sink_l, _source_l = _build_pipeline(_SOURCE_ROWS)
        with pytest.raises(NonResumableRunError, match=r"Run is still in progress") as exc_info:
            crashed.resume_orchestrator().resume(resume_point_loser, config_l, graph_l, payload_store=crashed.payload_store)
        assert exc_info.value.run_id == crashed.run_id
        assert exc_info.value.reason == "Run is still in progress"

        # The refused loser changed NOTHING durable.
        assert sink_l.results == []
        items_after = _work_items_by_token(crashed.db, crashed.run_id)
        assert items_after == items_before
        assert items_after[crashed_token]["status"] == TokenWorkStatus.LEASED.value
        assert items_after[crashed_token]["attempt"] == 1
        assert items_after[crashed_token]["lease_owner"] == "crashed-worker-1"
        assert _recovery_events(crashed.db, crashed.run_id) == []
        assert _completed_outcome_tokens(crashed.db, crashed.run_id) == outcomes_before
        assert _duplicate_terminal_outcome_tokens(crashed.db, crashed.run_id) == []
        assert sorted(_node_state_identities(crashed.db, crashed.run_id)) == identities_before
        with crashed.db.engine.connect() as conn:
            status = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == crashed.run_id)).scalar_one()
        assert status == RunStatus.RUNNING.value
        crashed.db.close()
