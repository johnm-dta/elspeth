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

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.contracts.enums import Determinism
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.results import SourceRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.fixtures.base_classes import (
    _TestSchema,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


class InterruptAfterN(BaseTransform):
    """Transform that sets a shutdown event after processing N rows."""

    name = "interrupt_after_n"
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


class QuarantineSource:
    """Source that emits quarantined rows, setting shutdown event after N."""

    name = "quarantine_source"
    output_schema = _TestSchema
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    _on_validation_failure: str = "quarantine"
    on_success: str = "default"

    def __init__(self, total: int, interrupt_after: int, shutdown_event: threading.Event) -> None:
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        self._total = total
        self._interrupt_after = interrupt_after
        self._event = shutdown_event
        self._count = 0

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for i in range(self._total):
            self._count += 1
            if self._count >= self._interrupt_after:
                self._event.set()
            yield SourceRow.quarantined(
                row={"value": i},
                error=f"validation_error_{i}",
                destination="quarantine",
            )

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def get_field_resolution(self) -> tuple[dict[str, str], str | None] | None:
        return None

    def get_schema_contract(self) -> None:
        return None


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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
            source=as_source(source),
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
        Uses the same pattern as TestResumeComprehensive._setup_failed_run.

        Args:
            db: LandscapeDB connection
            payload_store: PayloadStore for row data
            run_id: Run identifier
            num_rows: Total rows to create
            processed_count: Number of rows already processed (with terminal outcomes)

        Returns:
            ExecutionGraph for the run
        """
        import json as json_mod

        from sqlalchemy import insert

        from elspeth.contracts import NodeType, RowOutcome
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.contracts.enums import Determinism, RoutingMode
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import (
            edges_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )

        now = datetime.now(UTC)

        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="collect", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

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
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                )
            )

            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "collect", NodeType.SINK),
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

            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
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
                        source_node_id="src",
                        row_index=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    insert(tokens_table).values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        created_at=now,
                    )
                )

        # Mark first N rows as completed
        recorder = LandscapeRecorder(db)
        for i in range(processed_count):
            recorder.record_token_outcome(
                run_id=run_id,
                token_id=f"t{i}",
                outcome=RowOutcome.COMPLETED,
                sink_name="sink",
            )

        # Create checkpoint at last processed row
        if processed_count > 0:
            checkpoint_mgr = CheckpointManager(db)
            checkpoint_mgr.create_checkpoint(
                run_id=run_id,
                token_id=f"t{processed_count - 1}",
                node_id="xform",
                sequence_number=processed_count - 1,
                graph=graph,
            )

        return graph

    def test_resume_honors_shutdown_event(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Interrupt during resume: GracefulShutdownError raised, run marked INTERRUPTED."""
        from sqlalchemy import select

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.types import NodeID, SinkName
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import runs_table
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.null_source import NullSource

        run_id = "resume-shutdown-test"
        total_rows = 10
        processed_count = 3

        # Set up failed run: 10 rows, 3 processed, 7 remaining
        graph = self._setup_failed_run(
            landscape_db,
            payload_store,
            run_id,
            num_rows=total_rows,
            processed_count=processed_count,
        )
        graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        graph.set_transform_id_map({0: NodeID("xform")})

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
            source=as_source(null_source),
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

    def test_resume_without_shutdown_completes_normally(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Resume without shutdown event completes all remaining rows."""
        from sqlalchemy import select

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.types import NodeID, SinkName
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
        graph = self._setup_failed_run(
            landscape_db,
            payload_store,
            run_id,
            num_rows=total_rows,
            processed_count=processed_count,
        )
        graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        graph.set_transform_id_map({0: NodeID("xform")})

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
            source=as_source(null_source),
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
        assert result.rows_processed == remaining_rows
        assert result.status == RunStatus.COMPLETED
        assert len(resume_sink.results) == remaining_rows

        # Run is COMPLETED in database
        with landscape_db.engine.connect() as conn:
            run = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
        assert run is not None
        assert run.status == RunStatus.COMPLETED
