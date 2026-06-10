# tests/integration/pipeline/test_rc6_eof_resume_proof.py
"""RC6 exhausted-source EOF resume proof (elspeth-b2142988b4).

The claim under test: a crash AFTER source exhaustion but BEFORE the
end-of-input engine work (count/EOF-triggered aggregation flush) completes
leaves the run resumable as *source-exhausted engine work*:

- ``run_sources.lifecycle_state == 'exhausted'`` is durably recorded BEFORE
  the EOF flush runs (source_iteration.py finalize_source_iteration), so a
  flush crash is distinguishable from a mid-source interruption;
- the resume path accepts exhausted sources (resume.py
  ``_SOURCE_COMPLETE_LIFECYCLE_STATES``) and drains the restored EOF
  aggregation work to the sink;
- the source plugin is NOT re-invoked on resume (rows replay from persisted
  payloads, never from the source — load() invocation count stays 1);
- an INTERRUPTED (non-exhausted) source still fails resume safely with
  ``IncompleteSourceResumeError`` instead of fabricating completion.

Everything below runs the production path: real Orchestrator.run, real
SQLite LandscapeDB, real CheckpointManager/RecoveryManager, real
``get_resume_point()`` + ``Orchestrator.resume()``. No resume seams are
mocked.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import Determinism, PipelineRow, RunStatus
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.errors import GracefulShutdownError, IncompleteSourceResumeError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import AggregationName
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import AggregationSettings, CheckpointSettings, SourceSettings, TriggerConfig
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import run_sources_table, runs_table, token_work_items_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink, ListSource


class _LoadCountingSource(_TestSourceBase):
    """Source that counts load() invocations — the source-replay tripwire."""

    name = "load_counting_source"
    output_schema = ListSource.output_schema

    def __init__(
        self,
        rows: list[dict[str, int]],
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
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)


class _FailOnceEOFBatchTransform(BaseTransform):
    """Batch transform whose FIRST batch (the EOF flush) crashes.

    Single-row calls buffer (count trigger never fires below the threshold);
    the first list-shaped call — the end-of-input flush — raises, simulating
    a crash after source exhaustion but before EOF work completes. Subsequent
    batch calls (the resume's flush) succeed and emit the batch sum.
    """

    name = "fail_once_eof_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(self, *, fail_first_batch: bool = True) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._fail_next_batch = fail_first_batch
        self.batch_calls = 0

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            self.batch_calls += 1
            if self._fail_next_batch:
                self._fail_next_batch = False
                raise RuntimeError("injected EOF flush crash")
            total = sum(r.get("value", 0) for r in row)
            contract = SchemaContract(
                mode="OBSERVED",
                fields=(
                    FieldContract(normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"),
                    FieldContract(normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"),
                ),
                locked=True,
            )
            return TransformResult.success(
                PipelineRow({"value": total, "count": len(row)}, contract),
                success_reason={"action": "batch_sum"},
            )
        return TransformResult.success(row, success_reason={"action": "buffer"})


def _build_eof_aggregation_pipeline(
    source: Any,
    transform: _FailOnceEOFBatchTransform,
) -> tuple[PipelineConfig, ExecutionGraph, CollectSink]:
    """Count-triggered aggregation pipeline whose flush only fires at EOF."""
    output_sink = CollectSink("output")
    agg_settings = AggregationSettings(
        name="eof_sum",
        plugin=transform.name,
        input="batch_in",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=100, timeout_seconds=3600),
        output_mode="transform",
    )

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": as_source(source)},
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="batch_in", options={})},
        transforms=[],
        sinks={"output": as_sink(output_sink)},
        aggregations={"eof_sum": (as_transform(transform), agg_settings)},
        gates=[],
    )
    agg_node_id = graph.get_aggregation_id_map()[AggregationName("eof_sum")]
    transform.node_id = agg_node_id

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(transform)],
        sinks={"output": as_sink(output_sink)},
        aggregation_settings={agg_node_id: agg_settings},
    )
    return config, graph, output_sink


def _run_sources_states(db: LandscapeDB, run_id: str) -> dict[str, str]:
    with db.engine.connect() as conn:
        rows = conn.execute(
            select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state).where(run_sources_table.c.run_id == run_id)
        ).all()
    return {str(row.source_name): str(row.lifecycle_state) for row in rows}


def _single_run_id(db: LandscapeDB) -> str:
    with db.engine.connect() as conn:
        return str(conn.execute(select(runs_table.c.run_id)).scalar_one())


@pytest.mark.timeout(120)
class TestExhaustedSourceEOFResume:
    """elspeth-b2142988b4: real-path EOF resume after source exhaustion."""

    def test_exhausted_source_eof_aggregation_resumes_without_source_replay(self, tmp_path: Any) -> None:
        """A FAILED run with an exhausted source and pending EOF work resumes cleanly.

        Invariants proven:
        - ``lifecycle_state='exhausted'`` sources are ACCEPTED by the public
          resume path (resume.py ``_SOURCE_COMPLETE_LIFECYCLE_STATES``) — the
          run resumes instead of raising IncompleteSourceResumeError;
        - the restored EOF aggregation work drains: the count-100 trigger
          never fired in-run, so the only flush is the resume's end-of-input
          flush, which must deliver exactly one batch covering ALL source
          rows to the sink;
        - NO source replay: the resume never invokes the source plugin
          (load_invocations stays at the original run's 1);
        - final sink/audit state is correct: one batch artifact, all durable
          scheduler work terminal, and the run finalized COMPLETED.

        Construction note (kept honest): every audit row here is written by
        the REAL production engine — the run executes with the real
        Orchestrator and is gracefully interrupted on its final source row,
        which is the production writer of buffered-aggregation checkpoints
        (``checkpoint_interrupted_progress``). Two columns are then flipped
        to the exact values the engine writes when it crashes DURING the EOF
        flush instead (source_iteration.py records 'exhausted' BEFORE the
        flush; the failure ceremony records FAILED): a virgin run cannot
        reach that state resumable today because no checkpoint writer runs
        between exhaustion and sink durability — that gap is pinned by
        test_real_eof_flush_crash_is_not_yet_resumable below.
        """
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        shutdown_event = threading.Event()
        source = _LoadCountingSource(
            [{"value": 10}, {"value": 20}, {"value": 30}],
            on_success="batch_in",
            shutdown_event=shutdown_event,
            interrupt_after=3,
        )
        transform = _FailOnceEOFBatchTransform(fail_first_batch=False)
        config, graph, output_sink = _build_eof_aggregation_pipeline(source, transform)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError):
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = _single_run_id(db)
        assert source.load_invocations == 1
        assert transform.batch_calls == 0, "count-100 trigger must not fire in-run; EOF work stays buffered"
        assert output_sink.results == []

        # Production wrote the buffered-aggregation checkpoint on shutdown.
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.aggregation_state_json is not None

        # Reshape the interruption into the exhausted-then-crashed-EOF-flush
        # state: 'exhausted' is exactly what finalize_source_iteration records
        # before the EOF flush runs; FAILED is what the failure ceremony
        # records when that flush crashes.
        with db.engine.begin() as conn:
            conn.execute(
                run_sources_table.update()
                .where(run_sources_table.c.run_id == run_id, run_sources_table.c.source_name == "primary")
                .values(lifecycle_state="exhausted")
            )
            conn.execute(runs_table.update().where(runs_table.c.run_id == run_id).values(status=RunStatus.FAILED))
        assert _run_sources_states(db, run_id) == {"primary": "exhausted"}

        recovery = RecoveryManager(db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable exhausted-source run, got: {check.reason}"
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # The resume drained the EOF work: exactly one batch of all 3 rows.
        assert result.status == RunStatus.COMPLETED
        assert output_sink.results == [{"value": 60, "count": 3}]
        assert transform.batch_calls == 1

        # No source replay: resume never re-invoked the source plugin.
        assert source.load_invocations == 1

        # All durable scheduler work is resolved and checkpoints are gone.
        with db.connection() as conn:
            work_statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
            )
        assert work_statuses
        assert set(work_statuses) <= {"terminal"}
        assert checkpoint_mgr.get_latest_checkpoint(run_id) is None

    def test_real_eof_flush_crash_is_not_yet_resumable(self, tmp_path: Any) -> None:
        """CHARACTERIZATION of a production gap (flip when fixed, do not delete).

        A virgin run that crashes DURING the EOF aggregation flush — after
        source exhaustion — durably records ``lifecycle_state='exhausted'``
        (the engine's own crash-classification write) and leaves the batch
        membership and BLOCKED scheduler work in the audit DB, but it leaves
        NO checkpoint: production checkpoints are written only after sink
        durability or by the graceful-shutdown handler, both of which run
        AFTER the EOF flush. ``can_resume`` therefore refuses the very state
        resume.py's exhausted-source acceptance was built for. When a
        checkpoint writer is added at the EOF-flush boundary, this test must
        be flipped into a full crash->resume roundtrip (see
        elspeth-b2142988b4).
        """
        from elspeth.core.landscape.schema import batch_members_table, batches_table
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        source = _LoadCountingSource([{"value": 10}, {"value": 20}, {"value": 30}], on_success="batch_in")
        transform = _FailOnceEOFBatchTransform(fail_first_batch=True)
        config, graph, output_sink = _build_eof_aggregation_pipeline(source, transform)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="injected EOF flush crash"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        run_id = _single_run_id(db)
        assert output_sink.results == []

        # The engine classified the crash correctly: exhaustion was durably
        # recorded BEFORE the flush, and the EOF work is durably represented
        # (batch membership + BLOCKED barrier scheduler rows).
        assert _run_sources_states(db, run_id) == {"primary": "exhausted"}
        with db.connection() as conn:
            member_count = (
                conn.execute(
                    select(batch_members_table.c.token_id)
                    .join(batches_table, batch_members_table.c.batch_id == batches_table.c.batch_id)
                    .where(batches_table.c.run_id == run_id)
                )
                .scalars()
                .all()
            )
            blocked = (
                conn.execute(
                    select(token_work_items_table.c.status).where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == "blocked",
                    )
                )
                .scalars()
                .all()
            )
        assert len(member_count) == 3
        assert len(blocked) == 3

        # THE GAP: no checkpoint exists, so the public resume path refuses.
        check = RecoveryManager(db, checkpoint_mgr).can_resume(run_id, graph)
        assert not check.can_resume
        assert check.reason == "No checkpoint found for recovery"

    def test_interrupted_source_still_fails_resume_safely(self, tmp_path: Any) -> None:
        """A non-exhausted (interrupted) source refuses resume with IncompleteSourceResumeError.

        Invariant proven: the exhausted-source acceptance above does NOT
        loosen the fail-safe — a graceful shutdown mid-source records
        lifecycle_state='interrupted', and the public resume path refuses
        rather than completing a run whose source never finished, leaving
        buffered work unflushed (the sink stays empty).
        """
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        shutdown_event = threading.Event()
        source = _LoadCountingSource(
            [{"value": 10}, {"value": 20}, {"value": 30}, {"value": 40}],
            on_success="batch_in",
            shutdown_event=shutdown_event,
            interrupt_after=2,
        )
        transform = _FailOnceEOFBatchTransform(fail_first_batch=False)
        config, graph, output_sink = _build_eof_aggregation_pipeline(source, transform)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError):
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = _single_run_id(db)
        assert _run_sources_states(db, run_id) == {"primary": "interrupted"}
        assert output_sink.results == []

        recovery = RecoveryManager(db, checkpoint_mgr)
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        with pytest.raises(IncompleteSourceResumeError, match=r"primary.*interrupted"):
            orchestrator.resume(
                resume_point=resume_point,
                config=config,
                graph=graph,
                payload_store=payload_store,
            )

        # The refusal did not fabricate output or flush buffered EOF work.
        assert output_sink.results == []
        assert transform.batch_calls == 0
        assert source.load_invocations == 1
