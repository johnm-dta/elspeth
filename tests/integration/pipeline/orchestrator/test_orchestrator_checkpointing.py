# tests/integration/pipeline/orchestrator/test_orchestrator_checkpointing.py
"""Tests for Orchestrator checkpointing functionality.

Migrated from tests/engine/test_orchestrator_checkpointing.py.
Uses v2 fixtures with function-scoped databases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import GateName, NodeID, NodeType, PipelineRow, RouteDestination, RoutingMode, SinkName
from elspeth.plugins.base import BaseTransform
from tests.fixtures.base_classes import (
    _TestSchema,
    _TestSinkBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult
    from elspeth.core.landscape import LandscapeDB


class TestOrchestratorCheckpointing:
    """Tests for checkpoint integration in Orchestrator."""

    def test_orchestrator_accepts_checkpoint_manager(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with CheckpointManager."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
        )
        assert orchestrator._checkpoint_manager is checkpoint_mgr

    def test_orchestrator_accepts_checkpoint_config(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with RuntimeCheckpointConfig."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        settings = CheckpointSettings(frequency="every_n", checkpoint_interval=10)
        config = RuntimeCheckpointConfig.from_settings(settings)
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_config=config,
        )
        assert orchestrator._checkpoint_config == config

    def test_maybe_checkpoint_creates_on_every_row(self, landscape_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint creates checkpoint when frequency=every_row."""

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = IdentityTransform()
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
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3

        assert len(checkpoint_calls) == 3, f"Expected 3 checkpoint calls (one per row), got {len(checkpoint_calls)}"

    def test_maybe_checkpoint_respects_interval(self, landscape_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint only creates checkpoint every N rows."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_n", checkpoint_interval=3)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

        source = ListSource([{"value": i} for i in range(7)])
        transform = IdentityTransform()
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
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 7

        assert len(checkpoint_calls) == 2, f"Expected 2 checkpoint calls (at rows 3 and 6), got {len(checkpoint_calls)}"

    def test_checkpoint_deleted_on_successful_completion(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are deleted when run completes successfully."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
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
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        remaining_checkpoints = checkpoint_mgr.get_checkpoints(result.run_id)
        assert len(remaining_checkpoints) == 0

    def test_checkpoint_preserved_on_failure(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are preserved when run fails for recovery."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings, GateSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = _TestSchema
            output_schema = _TestSchema
            on_error = "discard"

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "passthrough"})

        class GoodSink(_TestSinkBase):
            name = "good_sink"

            def __init__(self) -> None:
                super().__init__()
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
            name = "bad_sink"

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Bad sink failure")

            def close(self) -> None:
                pass

        gate_config = GateSettings(
            name="split",
            input="transform_out",
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

        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
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

        graph.set_sink_id_map({SinkName("good"): NodeID("sink_good"), SinkName("bad"): NodeID("sink_bad")})
        graph.set_transform_id_map({0: NodeID("transform_0")})
        graph.set_config_gate_id_map({GateName("split"): NodeID("config_gate_split")})
        graph.set_route_resolution_map(
            {
                (NodeID("config_gate_split"), "true"): RouteDestination.sink(SinkName("good")),
                (NodeID("config_gate_split"), "false"): RouteDestination.sink(SinkName("bad")),
            }
        )
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="Bad sink failure"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        from elspeth.core.landscape import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        runs = recorder.list_runs()
        assert len(runs) >= 1
        run_id = sorted(runs, key=lambda r: r.started_at, reverse=True)[0].run_id

        remaining_checkpoints = checkpoint_mgr.get_checkpoints(run_id)

        if len(good_sink.results) > 0:
            assert len(remaining_checkpoints) == len(good_sink.results), (
                f"Expected {len(good_sink.results)} checkpoints for written rows, got {len(remaining_checkpoints)}"
            )

    def test_checkpoint_disabled_skips_checkpoint_creation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """No checkpoints created when checkpointing is disabled."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=False)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = IdentityTransform()
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
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        assert len(checkpoint_calls) == 0, f"Expected 0 checkpoint calls when disabled, got {len(checkpoint_calls)}"

    def test_no_checkpoint_manager_skips_checkpointing(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Orchestrator works fine without checkpoint manager."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        class IdentityTransform(BaseTransform):
            name = "identity"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

        source = ListSource([{"value": 1}])
        transform = IdentityTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 1
