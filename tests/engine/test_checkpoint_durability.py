# tests/engine/test_checkpoint_durability.py
"""Tests for checkpoint durability guarantees.

These tests verify the core invariants of the checkpoint system:
1. Checkpoints are created AFTER sink writes, not during the main loop
2. Checkpoint node_id is the sink node, not the last transform
3. Crash recovery works correctly (unwritten rows are reprocessed)

The key insight: a checkpoint represents DURABLE OUTPUT. If we checkpoint
before the sink write completes, a crash after checkpoint but before sink
means we'd skip data that was never actually written.

IMPORTANT: The orchestrator batches all rows through processing, then writes
them all to the sink in a single batch. Checkpoints are created AFTER the
sink.write() call returns successfully. If sink.write() throws, NO checkpoints
are created for that batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import PluginSchema, RoutingMode, SourceRow
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)


def _build_test_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for testing.

    Creates: source -> transforms... -> sinks
    """
    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Add transforms
    transform_ids: dict[int, str] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        graph.add_node(
            node_id,
            node_type="transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Edge from last transform to default sink
    output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))
    graph.add_edge(prev, sink_ids[output_sink], label="continue", mode=RoutingMode.MOVE)

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = {}
    graph._route_resolution_map = {}
    graph._output_sink = output_sink

    return graph


def _get_latest_run_id(db: LandscapeDB) -> str:
    """Helper to get the most recent run_id from the database."""
    from sqlalchemy import desc, select

    from elspeth.core.landscape.schema import runs_table

    with db.engine.connect() as conn:
        result = conn.execute(
            select(runs_table.c.run_id).order_by(desc(runs_table.c.started_at))
        ).fetchone()
        if result is None:
            raise ValueError("No runs found in database")
        run_id: str = result.run_id
        return run_id


# ============================================================================
# Test Classes
# ============================================================================


class TestCheckpointDurability:
    """Tests that checkpoints represent durable sink output."""

    def test_successful_run_creates_checkpoints_for_all_rows(
        self, tmp_path: Path
    ) -> None:
        """A successful run should create checkpoints for all rows written.

        Checkpoints are created AFTER the sink write completes. For a successful
        run with N rows, we should see N checkpoints (with every_row frequency)
        before they are deleted on completion.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")

        # Track checkpoints during the run (before they're deleted)
        checkpoint_count_during_write: list[int] = []

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class TrackingSink(_TestSinkBase):
            """Sink that tracks checkpoint creation timing."""

            name = "tracking_sink"

            def __init__(
                self, checkpoint_mgr: CheckpointManager, db_ref: LandscapeDB
            ) -> None:
                self._checkpoint_mgr = checkpoint_mgr
                self._db_ref = db_ref
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                # Check checkpoint count at on_complete (before deletion)
                # on_complete is called AFTER write() returns successfully
                run_id = _get_latest_run_id(self._db_ref)
                cps = self._checkpoint_mgr.get_checkpoints(run_id)
                checkpoint_count_during_write.append(len(cps))

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = TrackingSink(checkpoint_manager, db)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        result = orchestrator.run(config, graph=_build_test_graph(config))
        assert result.status == "completed"
        assert result.rows_processed == 5

        # Verify: 5 rows written
        assert len(sink.results) == 5

        # on_complete is called after sink write completes
        # At that point, checkpoints should exist for all written rows
        assert len(checkpoint_count_during_write) == 1
        assert (
            checkpoint_count_during_write[0] == 5
        ), f"Expected 5 checkpoints after write, got {checkpoint_count_during_write[0]}"

        # After successful completion, checkpoints are deleted
        run_id = _get_latest_run_id(db)
        final_checkpoints = checkpoint_manager.get_checkpoints(run_id)
        assert len(final_checkpoints) == 0, "Checkpoints should be deleted on success"

    def test_crash_before_sink_write_recovers_correctly(self, tmp_path: Path) -> None:
        """End-to-end test: crash before sink write, then recover correctly.

        This test verifies the complete recovery flow:
        1. Start a run with 5 rows
        2. Successfully write 2 rows to sink (creating checkpoints)
        3. Crash/failure occurs before remaining 3 rows are written
        4. Resume from checkpoint
        5. Verify the 3 unwritten rows ARE reprocessed
        6. Verify the 2 already-written rows are NOT reprocessed

        This ensures the checkpoint system correctly differentiates between
        rows that were durably written vs rows that only reached processing.
        """
        import json

        from elspeth.contracts import NodeType
        from elspeth.core.checkpoint import RecoveryManager
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        # Track which rows are written during resume
        written_during_resume: list[dict[str, Any]] = []

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            """Source that yields from provided data list."""

            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class TrackingSink(_TestSinkBase):
            """Sink that tracks all written rows."""

            name = "tracking_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                # Track writes during resume for assertion
                written_during_resume.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        # --- Phase 1: Simulate a partial run with crash ---
        # We manually create the database state that would exist if:
        # - 5 rows were loaded from source
        # - 2 rows were successfully written to sink (checkpointed)
        # - Run failed before remaining 3 rows were written

        recorder = LandscapeRecorder(db, payload_store=payload_store)

        # Create the run
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        # Register source node
        from elspeth.contracts import Determinism
        from elspeth.contracts.schema import SchemaConfig

        recorder.register_node(
            run_id=run_id,
            node_id="source",
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )

        # Register transform node (must match graph's transform_0)
        recorder.register_node(
            run_id=run_id,
            node_id="transform_0",
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )

        # Register sink node
        recorder.register_node(
            run_id=run_id,
            node_id="sink_default",
            plugin_name="tracking_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode=None, fields=None, is_dynamic=True),
        )

        # Create 5 source rows with payload storage
        all_rows = [{"value": i} for i in range(5)]
        row_ids = []
        token_ids = []

        for i, row_data in enumerate(all_rows):
            # Store payload
            payload_ref = payload_store.store(json.dumps(row_data).encode("utf-8"))

            # Create row with payload reference
            row = recorder.create_row(
                run_id=run_id,
                source_node_id="source",
                row_index=i,
                data=row_data,
                payload_ref=payload_ref,
            )
            row_ids.append(row.row_id)

            # Create token for the row
            token = recorder.create_token(row_id=row.row_id)
            token_ids.append(token.token_id)

        # Simulate: First 2 rows were successfully written to sink and checkpointed
        # (In real scenario, this happens in orchestrator after sink.write() returns)
        for i in range(2):
            checkpoint_manager.create_checkpoint(
                run_id=run_id,
                token_id=token_ids[i],
                node_id="sink_default",  # Checkpoint at sink node
                sequence_number=i + 1,
            )

        # Mark run as failed (simulating crash after 2 rows written)
        recorder.complete_run(run_id, status="failed")

        # --- Phase 2: Verify preconditions ---
        # 2 checkpoints exist (for rows 0 and 1)
        checkpoints = checkpoint_manager.get_checkpoints(run_id)
        assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"

        # Recovery should identify 3 unprocessed rows (rows 2, 3, 4)
        recovery = RecoveryManager(db, checkpoint_manager)
        unprocessed_row_ids = recovery.get_unprocessed_rows(run_id)
        assert (
            len(unprocessed_row_ids) == 3
        ), f"Expected 3 unprocessed rows, got {len(unprocessed_row_ids)}"

        # Verify can_resume returns True
        resume_check = recovery.can_resume(run_id)
        assert resume_check.can_resume, f"Cannot resume: {resume_check.reason}"

        # --- Phase 3: Resume and verify recovery ---
        # Get resume point
        resume_point = recovery.get_resume_point(run_id)
        assert resume_point is not None

        # Create fresh components for resume (simulating new process)
        source = ListSource(all_rows)  # Same data
        transform = PassthroughTransform()
        sink = TrackingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        # Clear the tracking list before resume
        written_during_resume.clear()

        # Resume the run
        result = orchestrator.resume(
            resume_point,
            config,
            graph=_build_test_graph(config),
            payload_store=payload_store,
        )

        # --- Phase 4: Verify correct recovery behavior ---
        assert (
            result.status == "completed"
        ), f"Resume failed with status: {result.status}"

        # CRITICAL: Only 3 rows should have been reprocessed (rows 2, 3, 4)
        assert result.rows_processed == 3, (
            f"Expected 3 rows reprocessed during resume, got {result.rows_processed}. "
            "Rows 0-1 were already checkpointed and should NOT be reprocessed."
        )

        # Verify the correct rows were written (values 2, 3, 4)
        assert (
            len(written_during_resume) == 3
        ), f"Expected 3 rows written during resume, got {len(written_during_resume)}"

        # Extract values and verify they are the unwritten rows
        written_values = sorted([r["value"] for r in written_during_resume])
        assert written_values == [2, 3, 4], (
            f"Expected rows with values [2, 3, 4] to be reprocessed, "
            f"got {written_values}. "
            "The checkpoint system should skip already-written rows (0, 1)."
        )

        # Checkpoints should be deleted after successful completion
        final_checkpoints = checkpoint_manager.get_checkpoints(run_id)
        assert len(final_checkpoints) == 0, "Checkpoints should be deleted on success"

    def test_failed_batch_creates_no_checkpoints(self, tmp_path: Path) -> None:
        """If sink.write() fails, NO checkpoints should be created.

        The orchestrator batches all rows, then writes them all at once.
        If the batch write fails (sink.write() throws), the exception is raised
        BEFORE checkpoints are created. This is correct behavior - we don't
        want to checkpoint rows that weren't durably written.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class FailingSink(_TestSinkBase):
            """Sink that always fails."""

            name = "failing_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Simulated sink failure")

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = FailingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        with pytest.raises(RuntimeError, match="Simulated sink failure"):
            orchestrator.run(config, graph=_build_test_graph(config))

        # Verify: NO checkpoints created because batch failed
        run_id = _get_latest_run_id(db)
        checkpoints = checkpoint_manager.get_checkpoints(run_id)
        assert len(checkpoints) == 0, (
            f"Expected 0 checkpoints when batch fails, got {len(checkpoints)}. "
            "Checkpoints should only be created after successful sink writes."
        )

    def test_checkpoint_node_id_is_sink_node(self, tmp_path: Path) -> None:
        """Checkpoints should reference sink node, not last transform.

        The checkpoint node_id must be the SINK node, not the transform node.
        This is because the checkpoint represents "data has been durably written
        to this sink", not "data has been processed by this transform".
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")

        # We'll capture the checkpoint state before completion deletes them
        captured_checkpoints: list[Any] = []

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        class CapturingSink(_TestSinkBase):
            name = "capturing_sink"

            def __init__(
                self, checkpoint_mgr: CheckpointManager, db_ref: LandscapeDB
            ) -> None:
                self._checkpoint_mgr = checkpoint_mgr
                self._db_ref = db_ref
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                # Capture checkpoints before they're deleted
                run_id = _get_latest_run_id(self._db_ref)
                cps = self._checkpoint_mgr.get_checkpoints(run_id)
                captured_checkpoints.extend(cps)

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = DoubleTransform()
        sink = CapturingSink(checkpoint_manager, db)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        graph = _build_test_graph(config)
        sink_node_id = graph._sink_id_map["default"]
        transform_node_id = graph._transform_id_map[0]

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        result = orchestrator.run(config, graph=graph)
        assert result.status == "completed"

        # Verify: checkpoint node_id is the SINK node
        assert len(captured_checkpoints) == 1
        checkpoint = captured_checkpoints[0]
        assert checkpoint.node_id == sink_node_id, (
            f"Checkpoint node_id should be sink '{sink_node_id}', "
            f"got '{checkpoint.node_id}'. "
            "Checkpoints must reference the sink node, not the last transform."
        )
        assert checkpoint.node_id != transform_node_id, (
            "Checkpoint node_id should NOT be the transform node. "
            "Checkpoints represent durable output, not processing completion."
        )

    def test_no_checkpoint_until_sink_write_completes(self, tmp_path: Path) -> None:
        """Tokens processed but not yet written to sink have no checkpoint.

        This verifies that the checkpoint is created AFTER the sink write,
        not when the token is added to pending_tokens or processed by transforms.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")

        # Track checkpoint count at various points
        checkpoints_before_write: list[int] = []
        checkpoints_after_write: list[int] = []

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class TimingSink(_TestSinkBase):
            """Sink that records checkpoint count before/after write."""

            name = "timing_sink"

            def __init__(
                self, checkpoint_mgr: CheckpointManager, db_ref: LandscapeDB
            ) -> None:
                self._checkpoint_mgr = checkpoint_mgr
                self._db_ref = db_ref
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                # After write completes (and checkpoints created)
                run_id = _get_latest_run_id(self._db_ref)
                cps = self._checkpoint_mgr.get_checkpoints(run_id)
                checkpoints_after_write.append(len(cps))

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                # BEFORE we return (checkpoint callback not yet called)
                run_id = _get_latest_run_id(self._db_ref)
                cps = self._checkpoint_mgr.get_checkpoints(run_id)
                checkpoints_before_write.append(len(cps))

                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(3)])
        transform = PassthroughTransform()
        sink = TimingSink(checkpoint_manager, db)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        result = orchestrator.run(config, graph=_build_test_graph(config))
        assert result.status == "completed"

        # Verify timing: 0 checkpoints before write, N after
        assert len(checkpoints_before_write) == 1, "Expected one write call"
        assert checkpoints_before_write[0] == 0, (
            f"Expected 0 checkpoints before write returns, got {checkpoints_before_write[0]}. "
            "Checkpoints should be created AFTER sink.write() returns."
        )

        assert len(checkpoints_after_write) == 1, "Expected one on_complete call"
        assert (
            checkpoints_after_write[0] == 3
        ), f"Expected 3 checkpoints after write, got {checkpoints_after_write[0]}"


class TestCheckpointTimingInvariants:
    """Additional tests for checkpoint timing guarantees."""

    def test_every_n_checkpointing_respects_sink_writes(self, tmp_path: Path) -> None:
        """Checkpoints with frequency='every_n' still happen after sink writes.

        Even when checkpointing every N rows, the checkpoint represents
        the SINK write completion, not the transform processing.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        # Checkpoint every 2 rows
        checkpoint_settings = CheckpointSettings(
            enabled=True, frequency="every_n", checkpoint_interval=2
        )

        captured_checkpoints: list[Any] = []

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class CapturingSink(_TestSinkBase):
            name = "capturing_sink"

            def __init__(
                self, checkpoint_mgr: CheckpointManager, db_ref: LandscapeDB
            ) -> None:
                self._checkpoint_mgr = checkpoint_mgr
                self._db_ref = db_ref
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                # Capture checkpoints before they're deleted
                run_id = _get_latest_run_id(self._db_ref)
                cps = self._checkpoint_mgr.get_checkpoints(run_id)
                captured_checkpoints.extend(cps)

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        # Process 5 rows
        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = CapturingSink(checkpoint_manager, db)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        result = orchestrator.run(config, graph=_build_test_graph(config))
        assert result.status == "completed"

        # With 5 rows and checkpoint_interval=2:
        # Checkpoints at sequence 2 and 4 (sequence 1, 3, 5 are not boundaries)
        assert len(captured_checkpoints) == 2, (
            f"Expected 2 checkpoints (at sequence 2 and 4 with interval=2), "
            f"got {len(captured_checkpoints)}. "
            "every_n checkpointing should create checkpoints at interval boundaries."
        )

        # All checkpoint node_ids should be sinks
        for cp in captured_checkpoints:
            assert cp.node_id.startswith(
                "sink_"
            ), f"Checkpoint node_id should be a sink, got '{cp.node_id}'"

    def test_checkpointing_disabled_creates_no_checkpoints(
        self, tmp_path: Path
    ) -> None:
        """When checkpointing is disabled, no checkpoints should be created."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=False)

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_settings=checkpoint_settings,
        )

        result = orchestrator.run(config, graph=_build_test_graph(config))
        assert result.status == "completed"

        # No checkpoints when disabled
        run_id = _get_latest_run_id(db)
        checkpoints = checkpoint_manager.get_checkpoints(run_id)
        assert len(checkpoints) == 0

    def test_no_checkpoint_manager_skips_checkpointing(self, tmp_path: Path) -> None:
        """When no checkpoint manager is provided, checkpointing is skipped."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        # No checkpoint_manager provided
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                for row in self._data:
                    yield SourceRow.valid(row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash="abc123"
                )

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(3)])
        transform = PassthroughTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": as_sink(sink)},
        )

        # No checkpoint_manager, but settings enabled - should not crash
        orchestrator = Orchestrator(
            db,
            checkpoint_manager=None,  # No manager
            checkpoint_settings=checkpoint_settings,
        )

        # Should run successfully without checkpointing
        result = orchestrator.run(config, graph=_build_test_graph(config))
        assert result.status == "completed"
        assert len(sink.results) == 3
