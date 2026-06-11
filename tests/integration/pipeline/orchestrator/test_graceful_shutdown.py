# tests/integration/pipeline/orchestrator/test_graceful_shutdown.py
"""Integration tests for graceful shutdown (SIGINT/SIGTERM).

Tests the full orchestrator interrupt flow: shutdown event triggers loop break,
pending work is flushed, run is marked INTERRUPTED, and is resumable.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Determinism, PipelineRow, RunStatus
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import GracefulShutdownError, IncompleteSourceResumeError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import AggregationName
from elspeth.core.canonical import canonical_json
from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
from elspeth.core.dag import ExecutionGraph
from elspeth.engine.orchestrator import PipelineConfig, prepare_for_run
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import (
    _TestSchema,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.pipeline import build_linear_pipeline, build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


def _runtime_val_manifest_json() -> str:
    """Mirror the run-header manifest production begin_run() stores."""
    prepare_for_run()
    return canonical_json(build_runtime_val_manifest())


class InterruptAfterN(BaseTransform):
    """Transform that sets a shutdown event after processing N rows."""

    name = "interrupt_after_n"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, n: int, shutdown_event: threading.Event) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._n = n
        self._event = shutdown_event
        self._count = 0

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        self._count += 1
        if self._count >= self._n:
            self._event.set()
        return TransformResult.success(row, success_reason={"action": "processed"})


class QuarantineSource(_TestSourceBase):
    """Source that emits quarantined rows, setting shutdown event after N."""

    name = "quarantine_source"
    output_schema = _TestSchema
    _on_validation_failure: str = "quarantine"

    def __init__(self, total: int, interrupt_after: int, shutdown_event: threading.Event) -> None:
        super().__init__()
        self._total = total
        self._interrupt_after = interrupt_after
        self._event = shutdown_event
        self._count = 0

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for i in range(self._total):
            self._count += 1
            if self._count >= self._interrupt_after:
                self._event.set()
            yield SourceRow.quarantined(
                row={"value": i},
                error=f"validation_error_{i}",
                destination="quarantine",
                source_row_index=i,
            )


class InterruptAfterNBufferedBatch(BaseTransform):
    """Batch transform that interrupts after buffering N rows.

    Used to verify graceful shutdown does not force END_OF_SOURCE aggregation
    semantics for partially buffered batches.
    """

    name = "interrupt_after_n_buffered_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(
        self,
        *,
        interrupt_after: int | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._interrupt_after = interrupt_after
        self._event = shutdown_event
        self._count = 0

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            total = sum(r.get("value", 0) for r in row)
            contract = SchemaContract(
                mode="OBSERVED",
                fields=(
                    FieldContract(
                        normalized_name="value",
                        original_name="value",
                        python_type=int,
                        required=False,
                        source="inferred",
                    ),
                    FieldContract(
                        normalized_name="count",
                        original_name="count",
                        python_type=int,
                        required=False,
                        source="inferred",
                    ),
                ),
                locked=True,
            )
            return TransformResult.success(
                PipelineRow({"value": total, "count": len(row)}, contract),
                success_reason={"action": "batch_sum"},
            )

        self._count += 1
        if self._event is not None and self._interrupt_after is not None and self._count >= self._interrupt_after:
            self._event.set()
        return TransformResult.success(row, success_reason={"action": "buffer"})


class FailFirstFlushBatch(InterruptAfterNBufferedBatch):
    """Batch transform whose FIRST list-shaped (flush) call crashes.

    Single-row calls buffer normally; the first end-of-source flush raises,
    producing the production-written resumable state: source exhausted,
    run FAILED, buffered tokens durable as BLOCKED journal rows (F1).
    Subsequent flush calls succeed (the batch-sum path of the parent class).
    """

    name = "fail_first_flush_batch"
    determinism = Determinism.DETERMINISTIC

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0
        self._failed_once = False

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            self.batch_calls += 1
            if not self._failed_once:
                self._failed_once = True
                raise RuntimeError("injected EOF flush crash")
        return super().process(row, ctx)


class InterruptingAggregationSource(_TestSourceBase):
    """Source that raises the shutdown event after yielding N rows."""

    name = "interrupting_aggregation_source"
    output_schema = ListSource.output_schema

    def __init__(self, rows: list[dict[str, int]], interrupt_after: int, shutdown_event: threading.Event) -> None:
        super().__init__()
        self._rows = rows
        self._interrupt_after = interrupt_after
        self._event = shutdown_event
        self.on_success = "source_out"

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for source_row_index, row in enumerate(self._rows):
            if source_row_index + 1 >= self._interrupt_after:
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


def _build_interruptible_aggregation_config(
    shutdown_event: threading.Event,
    *,
    interrupt_after: int = 2,
    transform: InterruptAfterNBufferedBatch | None = None,
) -> tuple[PipelineConfig, Any, CollectSink]:
    """Build a count-triggered aggregation pipeline with an interrupting batch transform.

    ``interrupt_after`` larger than the row count means the source exhausts
    normally (the shutdown event never fires); ``transform`` substitutes the
    buffering batch transform (e.g. ``FailFirstFlushBatch`` for crash paths).
    """
    source = InterruptingAggregationSource(
        rows=[{"value": 10}, {"value": 20}, {"value": 30}, {"value": 40}],
        interrupt_after=interrupt_after,
        shutdown_event=shutdown_event,
    )
    if transform is None:
        transform = InterruptAfterNBufferedBatch()
    output_sink = CollectSink("output")
    agg_settings = AggregationSettings(
        name="sum_agg",
        plugin=transform.name,
        input="source_out",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=100, timeout_seconds=3600),
        output_mode="transform",
    )

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": as_source(source)},
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
        transforms=[],
        sinks={"output": as_sink(output_sink)},
        aggregations={"sum_agg": (as_transform(transform), agg_settings)},
        gates=[],
    )

    agg_node_id = graph.get_aggregation_id_map()[AggregationName("sum_agg")]
    transform.node_id = agg_node_id

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(transform)],
        sinks={"output": as_sink(output_sink)},
        aggregation_settings={agg_node_id: agg_settings},
    )
    return config, graph, output_sink


def _build_interruptible_coalesce_config(
    shutdown_event: threading.Event,
) -> tuple[PipelineConfig, ExecutionGraph, Any, CollectSink]:
    """Build a fork -> buffered aggregation/direct branch -> coalesce pipeline."""
    from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings
    from elspeth.engine.orchestrator import PipelineConfig

    source = InterruptingAggregationSource(
        rows=[{"value": 10}],
        interrupt_after=1,
        shutdown_event=shutdown_event,
    )
    source.on_success = "fork_input"
    output_sink = CollectSink("output")
    batch_transform = InterruptAfterNBufferedBatch()
    batch_transform.on_success = "agg_ready"
    batch_transform.on_error = "discard"

    fork_gate = GateSettings(
        name="fork_gate",
        input="fork_input",
        condition="True",
        routes={"true": "fork", "false": "fork"},
        fork_to=["agg_branch", "direct_branch"],
    )
    coalesce = CoalesceSettings(
        name="merge_paths",
        branches={"agg_branch": "agg_ready", "direct_branch": "direct_branch"},
        policy="require_all",
        merge="nested",
        on_success="output",
    )
    agg_settings = AggregationSettings(
        name="agg_branch_hold",
        plugin=batch_transform.name,
        input="agg_branch",
        on_success="agg_ready",
        on_error="discard",
        trigger=TriggerConfig(count=100, timeout_seconds=3600),
        output_mode="transform",
    )

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": as_source(source)},
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="fork_input", options={})},
        transforms=[],
        sinks={"output": as_sink(output_sink)},
        aggregations={"agg_branch_hold": (as_transform(batch_transform), agg_settings)},
        gates=[fork_gate],
        coalesce_settings=[coalesce],
    )

    agg_node_id = graph.get_aggregation_id_map()[AggregationName("agg_branch_hold")]
    batch_transform.node_id = agg_node_id

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(batch_transform)],
        sinks={"output": as_sink(output_sink)},
        aggregation_settings={agg_node_id: agg_settings},
        gates=[fork_gate],
        coalesce_settings=[coalesce],
    )
    settings = ElspethSettings(
        sources={"primary": {"plugin": source.name, "on_success": "fork_input", "options": {}}},
        sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
        gates=[fork_gate],
        coalesce=[coalesce],
    )
    return config, graph, settings, output_sink


class TestShutdownBreaksLoop:
    """Tests that shutdown event correctly interrupts the processing loop."""

    def test_shutdown_breaks_loop_after_current_row(self, landscape_db: LandscapeDB, payload_store) -> None:
        """GracefulShutdownError raised with correct rows_processed."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(10)])
        transform = InterruptAfterN(3, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        assert exc_info.value.rows_processed == 3
        assert exc_info.value.run_id is not None

    def test_shutdown_writes_pending_tokens(self, landscape_db: LandscapeDB, payload_store) -> None:
        """All processed tokens reach sinks before shutdown error."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(10)])
        transform = InterruptAfterN(5, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError):
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        # All 5 processed rows should have been written to the sink
        assert len(sink.results) == 5

    def test_shutdown_run_status_is_interrupted(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Database shows INTERRUPTED status, not FAILED."""
        from sqlalchemy import select

        from elspeth.core.landscape.schema import runs_table
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(10)])
        transform = InterruptAfterN(3, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = exc_info.value.run_id

        with landscape_db.engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()

        assert run is not None
        assert run.status == RunStatus.INTERRUPTED

    def test_shutdown_calls_plugin_cleanup(self, landscape_db: LandscapeDB, payload_store) -> None:
        """close() is called on all plugins during shutdown."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(5)])
        transform = InterruptAfterN(2, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        # Track close() calls
        close_calls: list[str] = []
        original_source_close = source.close
        original_sink_close = sink.close

        def track_source_close() -> None:
            close_calls.append("source")
            original_source_close()

        def track_sink_close() -> None:
            close_calls.append("sink")
            original_sink_close()

        source.close = track_source_close  # type: ignore[method-assign]
        sink.close = track_sink_close  # type: ignore[method-assign]

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError):
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        assert "source" in close_calls
        assert "sink" in close_calls

    def test_no_interrupt_if_all_rows_consumed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Signal on last row still results in GracefulShutdownError with all rows processed."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        # Interrupt after row 5 out of 5 — all rows consumed, but event is set
        source = ListSource([{"value": i} for i in range(5)])
        transform = InterruptAfterN(5, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        # Event is set on the LAST row, so the shutdown check fires at end of loop
        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        # All 5 rows were processed before shutdown triggered
        assert exc_info.value.rows_processed == 5
        assert len(sink.results) == 5

    def test_shutdown_interrupts_quarantined_row_stream(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Shutdown event is checked on quarantine path, not just normal path."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        shutdown_event = threading.Event()
        # Source emits 100 quarantined rows; event set after row 5
        source = QuarantineSource(total=100, interrupt_after=5, shutdown_event=shutdown_event)
        quarantine_sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(CollectSink()), "quarantine": as_sink(quarantine_sink)},
        )

        orchestrator = Orchestrator(db=landscape_db)
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        # Should stop well before 100 rows — event fires at row 5,
        # so at most a few more rows may be processed before the check fires.
        assert exc_info.value.rows_processed <= 10
        assert exc_info.value.rows_processed >= 5

    def test_shutdown_does_not_flush_buffered_aggregation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Graceful shutdown must not synthesize END_OF_SOURCE aggregation output."""
        from elspeth.engine.orchestrator import Orchestrator

        shutdown_event = threading.Event()
        config, graph, output_sink = _build_interruptible_aggregation_config(shutdown_event)

        orchestrator = Orchestrator(db=landscape_db)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        assert exc_info.value.rows_processed == 2
        assert exc_info.value.rows_succeeded == 0
        assert exc_info.value.rows_failed == 0
        assert exc_info.value.rows_quarantined == 0
        assert exc_info.value.rows_routed_success == 0
        assert exc_info.value.rows_routed_failure == 0
        assert output_sink.results == []


class TestInterruptAndResume:
    """Tests for interrupt → resume pipeline lifecycle."""

    def test_interrupted_run_is_resumable(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Interrupt after N of M rows, verify checkpoint and resumability."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        total_rows = 10
        interrupt_after = 5

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        # Run and interrupt
        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(total_rows)])
        transform = InterruptAfterN(interrupt_after, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = exc_info.value.run_id
        assert exc_info.value.rows_processed == interrupt_after
        assert len(sink.results) == interrupt_after

        # Verify DB status is INTERRUPTED
        from sqlalchemy import select

        from elspeth.core.landscape.schema import runs_table

        with landscape_db.engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
        assert run is not None
        assert run.status == RunStatus.INTERRUPTED

        # Verify the run IS resumable
        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable, got: {check.reason}"

    def test_shutdown_creates_checkpoint(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Checkpoint exists after graceful shutdown."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        shutdown_event = threading.Event()
        source = ListSource([{"value": i} for i in range(10)])
        transform = InterruptAfterN(5, shutdown_event)
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        graph = build_production_graph(config)

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = exc_info.value.run_id

        # Verify checkpoint was created
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None

    def test_buffered_aggregation_shutdown_remains_resumable(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Buffered aggregation shutdown must persist a recovery checkpoint."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        shutdown_event = threading.Event()
        config, graph, output_sink = _build_interruptible_aggregation_config(shutdown_event)

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(config, graph=graph, payload_store=payload_store, shutdown_event=shutdown_event)

        run_id = exc_info.value.run_id
        assert run_id is not None
        assert output_sink.results == []

        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None

        # F1: the journal is the durable buffer truth — buffered aggregation
        # tokens persist as BLOCKED token_work_items rows at the barrier.
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with landscape_db.connection() as conn:
            blocked_barrier_rows = (
                conn.execute(
                    select(token_work_items_table.c.barrier_key).where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == "blocked",
                        token_work_items_table.c.barrier_key.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
        assert blocked_barrier_rows, "Expected buffered aggregation tokens as BLOCKED journal rows"

        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable buffered shutdown, got: {check.reason}"

    def test_buffered_coalesce_shutdown_refuses_completion_without_source_exhaustion(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Shutdown checkpoint persists pending coalesces but cannot complete an interrupted source."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        shutdown_event = threading.Event()
        config, graph, settings, output_sink = _build_interruptible_coalesce_config(shutdown_event)

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(
                config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                shutdown_event=shutdown_event,
            )

        run_id = exc_info.value.run_id
        assert run_id is not None
        assert output_sink.results == []

        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        # F1: pending aggregation/coalesce state persists as BLOCKED journal
        # rows, asserted on pre_resume_work below — the checkpoint carries only
        # scalar barrier metadata.

        # NOTE (multi-source-token-scheduler): the orchestrator now writes the
        # schema contract to ``run_sources`` on the first valid row (ADR-025 §3
        # Decision 5). Earlier revisions of this test fabricated the contract
        # here because the previous engine path skipped the in-loop
        # ``_record_schema_contract`` call when an observed source already had
        # a contract at ``_load_source_with_events`` time — leaving resume
        # broken. The fabrication is removed; the engine itself supplies the
        # contract.

        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with landscape_db.connection() as conn:
            pre_resume_work = (
                conn.execute(
                    select(
                        token_work_items_table.c.status,
                        token_work_items_table.c.branch_name,
                        token_work_items_table.c.fork_group_id,
                        token_work_items_table.c.barrier_key,
                        token_work_items_table.c.coalesce_name,
                    ).where(token_work_items_table.c.run_id == run_id)
                )
                .mappings()
                .all()
            )
        blocked_work = [row for row in pre_resume_work if row["status"] == "blocked"]
        assert blocked_work
        coalesce_blocked_work = [row for row in blocked_work if row["barrier_key"] == "merge_paths"]
        assert coalesce_blocked_work
        assert {row["branch_name"] for row in coalesce_blocked_work} == {"direct_branch"}
        assert {row["coalesce_name"] for row in coalesce_blocked_work} == {"merge_paths"}
        assert all(row["fork_group_id"] is not None for row in coalesce_blocked_work)

        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        assert recovery.get_unprocessed_rows(run_id) == []

        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        with pytest.raises(IncompleteSourceResumeError, match=r"source.*primary.*interrupted"):
            orchestrator.resume(
                resume_point=resume_point,
                config=config,
                graph=graph,
                payload_store=payload_store,
                settings=settings,
            )

        assert output_sink.results == []
        with landscape_db.connection() as conn:
            post_resume_work = (
                conn.execute(
                    select(token_work_items_table.c.status).where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.barrier_key == "merge_paths",
                    )
                )
                .scalars()
                .all()
            )
        assert post_resume_work
        assert set(post_resume_work) == {"blocked"}

    def test_buffered_only_resume_refuses_interrupted_source_before_pre_set_shutdown(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Buffered-only resume must not turn interrupted source work into completed output."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        initial_shutdown = threading.Event()
        config, graph, settings, output_sink = _build_interruptible_coalesce_config(initial_shutdown)

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError) as exc_info:
            orchestrator.run(
                config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                shutdown_event=initial_shutdown,
            )

        run_id = exc_info.value.run_id
        assert run_id is not None
        assert output_sink.results == []

        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        assert recovery.get_unprocessed_rows(run_id) == []

        first_resume_point = recovery.get_resume_point(run_id, graph)
        assert first_resume_point is not None
        # F1: pending coalesce state lives in BLOCKED journal rows, not on the
        # resume point — assert the durable pending-coalesce evidence directly.
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_work_items_table

        with landscape_db.connection() as conn:
            blocked_coalesce_rows = (
                conn.execute(
                    select(token_work_items_table.c.coalesce_name).where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == "blocked",
                        token_work_items_table.c.coalesce_name.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
        assert blocked_coalesce_rows, "Expected pending coalesce branches as BLOCKED journal rows"

        resume_shutdown = threading.Event()
        resume_shutdown.set()
        with pytest.raises(IncompleteSourceResumeError, match=r"source.*primary.*interrupted"):
            orchestrator.resume(
                resume_point=first_resume_point,
                config=config,
                graph=graph,
                payload_store=payload_store,
                settings=settings,
                shutdown_event=resume_shutdown,
            )

        assert output_sink.results == []

        second_resume_point = recovery.get_resume_point(run_id, graph)
        assert second_resume_point is not None
        assert second_resume_point.sequence_number == first_resume_point.sequence_number
        # F1: the pending coalesce branches must still be BLOCKED journal rows
        # after the refused resume (nothing consumed them).
        with landscape_db.connection() as conn:
            post_refusal_blocked = (
                conn.execute(
                    select(token_work_items_table.c.coalesce_name).where(
                        token_work_items_table.c.run_id == run_id,
                        token_work_items_table.c.status == "blocked",
                        token_work_items_table.c.coalesce_name.is_not(None),
                    )
                )
                .scalars()
                .all()
            )
        # Sorted compare: the two SELECTs carry no ORDER BY, so raw list
        # equality is a latent row-order flake.
        assert sorted(post_refusal_blocked) == sorted(blocked_coalesce_rows)
        assert recovery.get_unprocessed_rows(run_id) == []

    def _setup_failed_run(
        self,
        db: LandscapeDB,
        payload_store: Any,
        run_id: str,
        num_rows: int,
        processed_count: int,
    ) -> Any:
        """Set up a failed run with some rows processed and others pending.

        Creates DB records manually so resume has unprocessed rows to work with.
        Graph is built via production path (ExecutionGraph.from_plugin_instances)
        to prevent BUG-LINEAGE-01; node IDs are extracted from the graph for
        the manual SQL inserts.

        Args:
            db: LandscapeDB connection
            payload_store: PayloadStore for row data
            run_id: Run identifier
            num_rows: Total rows to create
            processed_count: Number of rows already processed (with terminal outcomes)

        Returns:
            ExecutionGraph for the run (with sink/transform ID maps already set)
        """
        import json as json_mod

        from sqlalchemy import insert

        from elspeth.contracts import NodeType
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.contracts.enums import Determinism, RoutingMode, TerminalOutcome, TerminalPath
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.landscape.schema import (
            edges_table,
            nodes_table,
            rows_table,
            run_sources_table,
            runs_table,
            tokens_table,
        )
        from tests.fixtures.landscape import make_factory
        from tests.fixtures.plugins import PassTransform

        now = datetime.now(UTC)

        # Build graph via production path — prevents BUG-LINEAGE-01
        source_data = [{"value": i} for i in range(num_rows)]
        transform = PassTransform()
        _, _, _, graph = build_linear_pipeline(source_data, transforms=[as_transform(transform)])

        # Extract production-generated node IDs
        source_nid = graph.get_sources()[0]
        assert source_nid is not None
        transform_id_map = graph.get_transform_id_map()
        sink_id_map = graph.get_sink_id_map()
        xform_nid = str(transform_id_map[0])
        sink_nid = str(next(iter(sink_id_map.values())))

        source_schema_json = json_mod.dumps({"properties": {"value": {"type": "integer"}}, "required": ["value"]})

        contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="value",
                    original_name="value",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        audit_record = ContractAuditRecord.from_contract(contract)
        schema_contract_json = audit_record.to_json()
        schema_contract_hash = contract.version_hash()

        with db.engine.begin() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            for node_id, plugin_name, node_type in [
                (source_nid, "list_source", NodeType.SOURCE),
                (xform_nid, "passthrough", NodeType.TRANSFORM),
                (sink_nid, "collect_sink", NodeType.SINK),
            ]:
                conn.execute(
                    insert(nodes_table).values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Per ADR-025 §3 Decision 5, ``verify_contract_integrity`` reads
            # from ``run_sources.schema_contract_json`` exclusively. The
            # fabricated failed run must populate the per-source contract row
            # so resume can verify Tier-1 integrity.
            conn.execute(
                insert(run_sources_table).values(
                    run_id=run_id,
                    source_node_id=source_nid,
                    source_name="primary",
                    plugin_name="list_source",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            for edge_id, from_node, to_node in [
                ("e1", source_nid, xform_nid),
                ("e2", xform_nid, sink_nid),
            ]:
                conn.execute(
                    insert(edges_table).values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            for i in range(num_rows):
                row_data = {"value": i}
                ref = payload_store.store(json_mod.dumps(row_data).encode())
                conn.execute(
                    insert(rows_table).values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id=source_nid,
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(tokens_table).values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        # Mark first N rows as completed
        factory = make_factory(db)
        for i in range(processed_count):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=f"t{i}", run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="default",
            )

        # Create checkpoint at last processed row
        if processed_count > 0:
            checkpoint_mgr = CheckpointManager(db)
            checkpoint_mgr.create_checkpoint(
                run_id=run_id,
                sequence_number=processed_count - 1,
                barrier_scalars=None,
                graph=graph,
            )

        return graph

    def test_resume_honors_shutdown_event(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Interrupt during resume: GracefulShutdownError raised, run marked INTERRUPTED."""
        from sqlalchemy import select

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import runs_table
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.null_source import NullSource

        run_id = "resume-shutdown-test"
        total_rows = 10
        processed_count = 3

        # Set up failed run: 10 rows, 3 processed, 7 remaining
        # Graph is built via production path; ID maps are already set.
        graph = self._setup_failed_run(
            landscape_db,
            payload_store,
            run_id,
            num_rows=total_rows,
            processed_count=processed_count,
        )

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)
        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        # Set up resume with shutdown event that fires after 2 rows
        resume_shutdown = threading.Event()
        resume_transform = InterruptAfterN(2, resume_shutdown)
        resume_transform.on_success = "default"
        resume_transform.on_error = "discard"
        resume_sink = CollectSink()
        null_source = NullSource({})
        null_source.on_success = "default"

        resume_config = PipelineConfig(
            sources={"primary": as_source(null_source)},
            transforms=[as_transform(resume_transform)],
            sinks={"default": as_sink(resume_sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(GracefulShutdownError) as resume_exc:
            orchestrator.resume(
                resume_point=resume_point,
                config=resume_config,
                graph=graph,
                payload_store=payload_store,
                shutdown_event=resume_shutdown,
            )

        # GracefulShutdownError has correct rows_processed and run_id
        assert resume_exc.value.rows_processed >= 2
        assert resume_exc.value.run_id == run_id

        # Run is INTERRUPTED in database (not FAILED or RUNNING)
        with landscape_db.engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
        assert run is not None
        assert run.status == RunStatus.INTERRUPTED

        # Processed rows reached the sink
        assert len(resume_sink.results) >= 2

    def test_resume_shutdown_recheckpoints_buffered_aggregation_without_sink_writes(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Pre-set shutdown during resume must preserve buffered aggregation state without sink writes.

        F1 rewrite: the buffered-aggregation run is produced by the PRODUCTION
        engine — a virgin run whose end-of-source flush crashes AFTER the
        source exhausted. The journal's BLOCKED barrier rows are the buffer
        truth, and the sequence-0 run-start checkpoint makes the crashed run
        resumable. The resume is then interrupted by a pre-set shutdown event:
        it must honor the shutdown BEFORE any end-of-source flush work — no
        sink writes, no flush attempt — and re-checkpoint progress so the
        buffered state survives as the same BLOCKED journal rows + checkpoint
        barrier scalars.
        """
        from sqlalchemy import select

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import runs_table, token_work_items_table
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        # interrupt_after=99: the source exhausts normally (no shutdown during
        # the run); the count=100 trigger never fires, so ALL rows are buffered
        # at the aggregation barrier when the EOF flush crashes.
        transform = FailFirstFlushBatch()
        config, graph, output_sink = _build_interruptible_aggregation_config(
            threading.Event(),
            interrupt_after=99,
            transform=transform,
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="injected EOF flush crash"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        with landscape_db.connection() as conn:
            run_id = str(conn.execute(select(runs_table.c.run_id)).scalar_one())
        assert output_sink.results == []
        assert transform.batch_calls == 1

        # F1: the journal is the buffer truth — the buffered rows persist as
        # BLOCKED token_work_items rows at the aggregation barrier.
        def blocked_barrier_tokens() -> list[str]:
            with landscape_db.connection() as conn:
                return sorted(
                    conn.execute(
                        select(token_work_items_table.c.token_id).where(
                            token_work_items_table.c.run_id == run_id,
                            token_work_items_table.c.status == "blocked",
                            token_work_items_table.c.barrier_key.is_not(None),
                        )
                    )
                    .scalars()
                    .all()
                )

        pre_resume_blocked = blocked_barrier_tokens()
        assert len(pre_resume_blocked) == 4, "Expected all 4 buffered rows as BLOCKED journal rows"

        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        first_resume_point = recovery.get_resume_point(run_id, graph)
        assert first_resume_point is not None

        resume_shutdown = threading.Event()
        resume_shutdown.set()
        with pytest.raises(GracefulShutdownError) as resume_exc:
            orchestrator.resume(
                resume_point=first_resume_point,
                config=config,
                graph=graph,
                payload_store=payload_store,
                shutdown_event=resume_shutdown,
            )
        assert resume_exc.value.run_id == run_id

        # The interrupted resume wrote nothing to the sink and never attempted
        # the end-of-source flush (the in-run crash remains the only batch call).
        assert output_sink.results == []
        assert transform.batch_calls == 1

        # The shutdown path re-checkpointed resume progress beyond the
        # original resume point...
        second_resume_point = recovery.get_resume_point(run_id, graph)
        assert second_resume_point is not None
        assert second_resume_point.sequence_number > first_resume_point.sequence_number

        # ...and the buffered state survived: identical BLOCKED journal rows
        # plus unchanged checkpoint-borne barrier scalars.
        assert blocked_barrier_tokens() == pre_resume_blocked
        assert second_resume_point.barrier_scalars == first_resume_point.barrier_scalars

    def test_resume_without_shutdown_completes_normally(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Resume without shutdown event completes all remaining rows."""
        from sqlalchemy import select

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import runs_table
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.null_source import NullSource
        from elspeth.plugins.transforms.passthrough import PassThrough

        run_id = "resume-no-shutdown-test"
        total_rows = 10
        processed_count = 5

        # Set up failed run: 10 rows, 5 processed, 5 remaining
        # Graph is built via production path; ID maps are already set.
        graph = self._setup_failed_run(
            landscape_db,
            payload_store,
            run_id,
            num_rows=total_rows,
            processed_count=processed_count,
        )

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)
        recovery = RecoveryManager(landscape_db, checkpoint_mgr)
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        # Set up resume WITHOUT shutdown event
        passthrough = PassThrough({"schema": {"mode": "observed"}})
        passthrough.on_success = "default"
        passthrough.on_error = "discard"
        resume_sink = CollectSink()
        null_source = NullSource({})
        null_source.on_success = "default"

        resume_config = PipelineConfig(
            sources={"primary": as_source(null_source)},
            transforms=[as_transform(passthrough)],
            sinks={"default": as_sink(resume_sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        result = orchestrator.resume(
            resume_point=resume_point,
            config=resume_config,
            graph=graph,
            payload_store=payload_store,
        )

        remaining_rows = total_rows - processed_count
        # F2 (resume-fork-reemit): the resume RunResult now reports CUMULATIVE
        # rows_processed reconstructed from the audit trail (distinct source rows
        # reaching a terminal outcome) — the whole run (total_rows), matching an
        # uninterrupted run. Pre-F2 it reported the resume-only `remaining_rows`.
        # The resume sink still only collects the rows THIS resume wrote, so its
        # result count remains `remaining_rows`.
        assert result.rows_processed == total_rows
        assert result.status == RunStatus.COMPLETED
        assert len(resume_sink.results) == remaining_rows

        # Run is COMPLETED in database
        with landscape_db.engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
        assert run is not None
        assert run.status == RunStatus.COMPLETED
