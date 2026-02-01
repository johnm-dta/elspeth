# tests/engine/test_orchestrator_checkpointing.py
"""Tests for Orchestrator checkpointing functionality.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import GateName, NodeID, NodeType, RoutingMode, SinkName, SourceRow
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult
    from elspeth.core.landscape import LandscapeDB


@pytest.fixture(scope="module")
def checkpoint_db() -> LandscapeDB:
    """Module-scoped in-memory database for checkpointing tests.

    Each test generates a unique run_id via the Orchestrator, so tests
    do not interfere with each other despite sharing the database.
    """
    from elspeth.core.landscape import LandscapeDB

    return LandscapeDB.in_memory()


class TestOrchestratorCheckpointing:
    """Tests for checkpoint integration in Orchestrator."""

    def test_orchestrator_accepts_checkpoint_manager(self, checkpoint_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with CheckpointManager."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
        )
        assert orchestrator._checkpoint_manager is checkpoint_mgr

    def test_orchestrator_accepts_checkpoint_config(self, checkpoint_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with RuntimeCheckpointConfig."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        settings = CheckpointSettings(frequency="every_n", checkpoint_interval=10)
        config = RuntimeCheckpointConfig.from_settings(settings)
        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_config=config,
        )
        assert orchestrator._checkpoint_config == config

    def test_maybe_checkpoint_creates_on_every_row(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint creates checkpoint when frequency=every_row."""

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        # P2 Fix: Track checkpoint creation calls
        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3

        # P2 Fix: Verify checkpoint was called for each row
        # With frequency=every_row and 3 rows, we expect 3 checkpoint calls
        assert len(checkpoint_calls) == 3, f"Expected 3 checkpoint calls (one per row), got {len(checkpoint_calls)}"

    def test_maybe_checkpoint_respects_interval(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint only creates checkpoint every N rows."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        # Checkpoint every 3 rows
        settings = CheckpointSettings(enabled=True, frequency="every_n", checkpoint_interval=3)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        # P2 Fix: Track checkpoint creation calls
        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        # 7 rows: should checkpoint at rows 3, 6 (sequence 3, 6)
        source = ListSource([{"value": i} for i in range(7)])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 7

        # P2 Fix: Verify checkpoint was called at correct intervals
        # With frequency=every_n, interval=3, and 7 rows, we expect checkpoints at rows 3, 6
        # (rows 1, 2 don't checkpoint; row 3 checkpoints; rows 4, 5 don't; row 6 checkpoints; row 7 doesn't)
        assert len(checkpoint_calls) == 2, f"Expected 2 checkpoint calls (at rows 3 and 6), got {len(checkpoint_calls)}"

    def test_checkpoint_deleted_on_successful_completion(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are deleted when run completes successfully."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        # After successful completion, checkpoints should be deleted
        remaining_checkpoints = checkpoint_mgr.get_checkpoints(result.run_id)
        assert len(remaining_checkpoints) == 0

    def test_checkpoint_preserved_on_failure(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are preserved when run fails for recovery.

        IMPORTANT: Checkpoints are created AFTER sink writes, not during transform
        processing. This test verifies that when a pipeline fails with multiple
        sinks (where one succeeds and one fails), the checkpoints from the
        successful sink are preserved for potential recovery.

        The batched write model means:
        - All rows are processed through transforms first
        - Then pending tokens are written to each sink in sequence
        - Checkpoints are created after each sink.write() returns successfully
        - If a later sink fails, checkpoints from earlier sinks are preserved
        """
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings, GateSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "passthrough"})

        class GoodSink(_TestSinkBase):
            """Sink that succeeds."""

            name = "good_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="good123")

            def close(self) -> None:
                pass

        class BadSink(_TestSinkBase):
            """Sink that fails during write."""

            name = "bad_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Bad sink failure")

            def close(self) -> None:
                pass

        # Route odd values to 'good' sink, even values to 'bad' sink
        gate_config = GateSettings(
            name="split",
            condition="row['value'] % 2 == 1",
            routes={"true": "good", "false": "bad"},
        )

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassthroughTransform()
        good_sink = GoodSink()
        bad_sink = BadSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"good": as_sink(good_sink), "bad": as_sink(bad_sink)},
            gates=[gate_config],
        )

        # Build graph with routing
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="list_source", config=schema_config)
        graph.add_node("transform_0", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node(
            "config_gate_split",
            node_type=NodeType.GATE,
            plugin_name="config_gate:split",
            config={
                "schema": schema_config["schema"],
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
            },
        )
        graph.add_node("sink_good", node_type=NodeType.SINK, plugin_name="good_sink", config=schema_config)
        graph.add_node("sink_bad", node_type=NodeType.SINK, plugin_name="bad_sink", config=schema_config)

        graph.add_edge("source", "transform_0", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform_0", "config_gate_split", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("config_gate_split", "sink_good", label="true", mode=RoutingMode.MOVE)
        graph.add_edge("config_gate_split", "sink_bad", label="false", mode=RoutingMode.MOVE)

        graph._sink_id_map = {SinkName("good"): NodeID("sink_good"), SinkName("bad"): NodeID("sink_bad")}
        graph._transform_id_map = {0: NodeID("transform_0")}
        graph._config_gate_id_map = {GateName("split"): NodeID("config_gate_split")}
        graph._route_resolution_map = {
            (NodeID("config_gate_split"), "true"): "good",
            (NodeID("config_gate_split"), "false"): "bad",
        }
        graph._default_sink = "good"

        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        # The bad sink will fail, but good sink should have already written
        # Note: Sink iteration order in dict may vary, so we check checkpoints exist
        with pytest.raises(RuntimeError, match="Bad sink failure"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Query for the failed run (most recent)
        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(checkpoint_db)
        runs = recorder.list_runs()
        assert len(runs) >= 1
        # Get the most recent run (sorted by started_at descending)
        run_id = sorted(runs, key=lambda r: r.started_at, reverse=True)[0].run_id

        # Checkpoints should be preserved (at least for the good sink)
        remaining_checkpoints = checkpoint_mgr.get_checkpoints(run_id)

        # We may have 0, 1, or 2 checkpoints depending on which sink was written first
        # The key assertion is: if good sink was written first, its checkpoints exist
        # If bad sink was written first, it fails before any checkpoints
        # This test verifies the durability invariant: checkpoints that were created
        # are NOT deleted on failure (they're only deleted on success)

        # Check that the good sink did write (if it got a chance)
        if len(good_sink.results) > 0:
            # Good sink wrote - should have checkpoints for those rows
            assert len(remaining_checkpoints) == len(good_sink.results), (
                f"Expected {len(good_sink.results)} checkpoints for written rows, got {len(remaining_checkpoints)}"
            )
        # If good sink didn't write (bad sink failed first), that's also valid behavior

    def test_checkpoint_disabled_skips_checkpoint_creation(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """No checkpoints created when checkpointing is disabled."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(checkpoint_db)
        settings = CheckpointSettings(enabled=False)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        # P2 Fix: Track checkpoint creation calls to verify none are made
        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        # P2 Fix: Verify no checkpoint calls were made when disabled
        assert len(checkpoint_calls) == 0, f"Expected 0 checkpoint calls when disabled, got {len(checkpoint_calls)}"

    def test_no_checkpoint_manager_skips_checkpointing(self, checkpoint_db: LandscapeDB, payload_store) -> None:
        """Orchestrator works fine without checkpoint manager."""
        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class ValueSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = ValueSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = ValueSchema
            output_schema = ValueSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def close(self) -> None:
                pass

        source = ListSource([{"value": 1}])
        transform = IdentityTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # No checkpoint_manager passed - should work without checkpointing
        orchestrator = Orchestrator(
            db=checkpoint_db,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 1
