"""Shared crashed-run harness for the e2e recovery suites.

Extracted VERBATIM from ``tests/e2e/recovery/test_concurrent_resume.py`` (the
crashed-mid-claim construction, the checkpoint-coupled resume seams, and the
durable-surface snapshot helpers) so the slice-2 coordination suites
(``test_suspended_winner_fences.py``, ``test_run_coordination_uniformity.py``)
reuse exactly the same production-writer state builders. Zero behavior
changes in the move; ``test_concurrent_resume.py`` imports everything back.

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

The HARNESS, by necessity, enters resume through today's checkpoint-layer
surface: ``RecoveryManager.get_resume_point`` and the Orchestrator's
``checkpoint_manager``/``checkpoint_config`` kwargs. That dependency is
confined to exactly three seams — :func:`_run_to_interrupted_checkpoint`
(run setup), :class:`_CrashedRun` (``resume_orchestrator``), and
:func:`_recovery_manager` / :func:`_resume_point` (resume entry, shared by
every test). When the checkpoint constructors/signatures change or
disappear, the edits land in those seams only; the assertions in the test
files survive unchanged.

EPOCH-21 SEAT ASSUMPTION (slice 2, ADR-030 — pinned here per the test
campaign spec): ``begin_run`` now mints the ``run_coordination`` leader seat
at epoch 1 for the real run this harness drives, and the graceful/FAILED
teardown ceremonies RELEASE the seat — so the crafted crashed image presents
a VACANT seat at epoch 1, which ``acquire_run_leadership`` admits directly.
Even if a future ceremony change left the seat HELD, every consuming resume
test calls ``clock.advance(360)`` before resuming, which exceeds the 80 s
default liveness window — the takeover CAS admits vacant-OR-expired, so the
assumption is safe in both worlds.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from sqlalchemy import func, select, update

from elspeth.contracts import Determinism, PipelineRow, PluginSchema, RunStatus
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.scheduler import SchedulerEventType
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings, QueueSettings, SourceSettings, TransformSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.database import begin_write
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    checkpoints_table,
    node_states_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    run_sources_table,
    run_workers_table,
    runs_table,
    scheduler_events_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
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

    Every resume entry in the consuming files funnels through here. Callers
    never dereference the returned handle — they only pass it back to
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


# ---------------------------------------------------------------------------
# Durable-surface snapshot helpers (moved verbatim)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Epoch-21 coordination helpers (slice 2 — new with the test campaign)
# ---------------------------------------------------------------------------


def _coord(crashed: _CrashedRun) -> RunCoordinationRepository:
    """Coordination repository over the crashed run's own engine."""
    return RunCoordinationRepository(crashed.db.engine)


def _usurp_seat(
    db: LandscapeDB,
    run_id: str,
    *,
    usurper: str,
    now: datetime,
) -> int:
    """The in-DB image of takeover (design §H): bump ``leader_epoch`` directly.

    Writes with the slice-1 write-intent discipline (``begin_write``). The
    usurped seat's expiry is stamped at ``now`` — fence admissibility is
    identity+epoch only (NEVER expiry), and a low-water expiry keeps the
    verify-AND-EXTEND side effect observable for every clock domain the
    fenced verbs use (MockClock-driven scheduler verbs and wall-clock-driven
    lifecycle/checkpoint verbs alike).

    Returns the post-usurpation ``leader_epoch``.
    """
    with begin_write(db.engine) as conn:
        conn.execute(
            update(run_coordination_table)
            .where(run_coordination_table.c.run_id == run_id)
            .values(
                leader_worker_id=usurper,
                leader_epoch=run_coordination_table.c.leader_epoch + 1,
                leader_heartbeat_expires_at=now,
                updated_at=now,
            )
        )
    with db.engine.connect() as conn:
        return int(
            conn.execute(select(run_coordination_table.c.leader_epoch).where(run_coordination_table.c.run_id == run_id)).scalar_one()
        )


def _coordination_row(db: LandscapeDB, run_id: str) -> dict[str, Any]:
    """The run_coordination seat row as a plain dict (byte-for-byte snapshots)."""
    with db.engine.connect() as conn:
        row = conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == run_id)).mappings().one()
    return dict(row)


def _coordination_events(db: LandscapeDB, run_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
    """run_coordination_events rows in ``seq`` (authoritative replay) order."""
    query = (
        select(run_coordination_events_table)
        .where(run_coordination_events_table.c.run_id == run_id)
        .order_by(run_coordination_events_table.c.seq)
    )
    if event_type is not None:
        query = query.where(run_coordination_events_table.c.event_type == event_type)
    with db.engine.connect() as conn:
        return [dict(row) for row in conn.execute(query).mappings()]


def _run_workers(db: LandscapeDB, run_id: str) -> list[dict[str, Any]]:
    """run_workers registry rows for the run, ordered by registration."""
    with db.engine.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                select(run_workers_table).where(run_workers_table.c.run_id == run_id).order_by(run_workers_table.c.registered_at)
            ).mappings()
        ]


def _checkpoint_count_and_max_seq(db: LandscapeDB, run_id: str) -> tuple[int, int | None]:
    """(checkpoint count, MAX(sequence_number)) for the run."""
    with db.engine.connect() as conn:
        row = conn.execute(
            select(func.count(), func.max(checkpoints_table.c.sequence_number)).where(checkpoints_table.c.run_id == run_id)
        ).one()
    return int(row[0]), None if row[1] is None else int(row[1])


def _rows_and_tokens_at(db: LandscapeDB, run_id: str, ingest_sequence: int) -> tuple[list[str], list[str]]:
    """(row_ids, token_ids) durably present at one ingest_sequence slot."""
    with db.engine.connect() as conn:
        row_ids = [
            str(row_id)
            for row_id in conn.execute(
                select(rows_table.c.row_id).where(
                    rows_table.c.run_id == run_id,
                    rows_table.c.ingest_sequence == ingest_sequence,
                )
            ).scalars()
        ]
        token_ids = (
            [
                str(token_id)
                for token_id in conn.execute(select(tokens_table.c.token_id).where(tokens_table.c.row_id.in_(row_ids))).scalars()
            ]
            if row_ids
            else []
        )
    return row_ids, token_ids
