# tests/integration/pipeline/orchestrator/test_orchestrator_core.py
"""Core orchestrator tests.

Migrated from tests/engine/test_orchestrator_core.py.
Uses v2 fixtures and production assembly path (BUG-LINEAGE-01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism, NodeType, PipelineRow, RoutingMode, RunStatus, SinkName
from elspeth.plugins.base import BaseTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


# ---------------------------------------------------------------------------
# Shared test transforms (specific to orchestrator_core tests)
# ---------------------------------------------------------------------------


class DoubleTransform(BaseTransform):
    """Transform that doubles a value field."""

    name = "double"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(
            make_pipeline_row({"value": row["value"], "doubled": row["value"] * 2}),
            success_reason={"action": "double"},
        )


class AddOneTransform(BaseTransform):
    """Transform that adds 1 to a value field."""

    name = "add_one"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(make_pipeline_row({"value": row["value"] + 1}), success_reason={"action": "add_one"})


class MultiplyTwoTransform(BaseTransform):
    """Transform that multiplies value by 2."""

    name = "multiply_two"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(make_pipeline_row({"value": row["value"] * 2}), success_reason={"action": "multiply_two"})


class IdentityTransform(BaseTransform):
    """Transform that passes data through unchanged."""

    name = "identity"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.results import TransformResult

        return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "identity"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self, payload_store) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = DoubleTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 3
        assert len(sink.results) == 3
        assert sink.results[0] == {"value": 1, "doubled": 2}

    def test_run_with_gate_routing(self, payload_store) -> None:
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Config-driven gate: routes values > 50 to "high" sink, else to "default"
        threshold_gate = GateSettings(
            name="threshold",
            input="source_out",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "default"},
        )

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[threshold_gate],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        # value=10 and value=30 go to default, value=100 goes to high
        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1

    def test_nonterminal_coalesce_continues_to_downstream_gate(self, payload_store) -> None:
        """Merged fork paths at a non-terminal coalesce must continue downstream."""
        from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 1}, {"value": 2}], on_success="source_sink")
        transform = IdentityTransform()
        output_sink = CollectSink(name="output")
        source_sink = CollectSink(name="source_sink")

        fork_gate = GateSettings(
            name="fork_gate",
            input="transform_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        terminal_gate = GateSettings(
            name="terminal_gate",
            input="merge_paths",
            condition="True",
            routes={"true": "output", "false": "output"},
        )
        coalesce = CoalesceSettings(
            name="merge_paths",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "output": as_sink(output_sink),
                "source_sink": as_sink(source_sink),
            },
            gates=[fork_gate, terminal_gate],
            coalesce_settings=[coalesce],
        )

        settings = ElspethSettings(
            source={"plugin": "test", "on_success": "source_out", "options": {}},
            sinks={"output": {"plugin": "test"}, "source_sink": {"plugin": "test"}},
            gates=[fork_gate, terminal_gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=build_production_graph(config),
            settings=settings,
            payload_store=payload_store,
        )

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert len(output_sink.results) == 2
        assert len(source_sink.results) == 0

    def test_traversal_context_keeps_nonterminal_coalesce_in_graph_step_order(self) -> None:
        """Traversal context must preserve graph step order for non-terminal coalesce nodes."""
        from elspeth.contracts.types import CoalesceName, GateName
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 1}], on_success="source_sink")
        transform = IdentityTransform()
        output_sink = CollectSink(name="output")
        source_sink = CollectSink(name="source_sink")

        fork_gate = GateSettings(
            name="fork_gate",
            input="transform_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        terminal_gate = GateSettings(
            name="terminal_gate",
            input="merge_paths",
            condition="True",
            routes={"true": "output", "false": "output"},
        )
        coalesce = CoalesceSettings(
            name="merge_paths",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "output": as_sink(output_sink),
                "source_sink": as_sink(source_sink),
            },
            gates=[fork_gate, terminal_gate],
            coalesce_settings=[coalesce],
        )
        graph = build_production_graph(config)

        source_id = graph.get_source()
        assert source_id is not None

        orchestrator = Orchestrator(db)
        orchestrator._assign_plugin_node_ids(
            source=config.source,
            transforms=config.transforms,
            sinks=config.sinks,
            source_id=source_id,
            transform_id_map=graph.get_transform_id_map(),
            sink_id_map=graph.get_sink_id_map(),
        )

        graph_step_map = graph.build_step_map()
        coalesce_node_id = graph.get_coalesce_id_map()[CoalesceName("merge_paths")]
        downstream_gate_node_id = graph.get_config_gate_id_map()[GateName("terminal_gate")]
        assert graph_step_map[coalesce_node_id] < graph_step_map[downstream_gate_node_id]

        traversal = orchestrator._build_dag_traversal_context(
            graph=graph,
            config=config,
            config_gate_id_map=graph.get_config_gate_id_map(),
        )
        assert traversal.node_step_map[coalesce_node_id] == graph_step_map[coalesce_node_id]
        assert traversal.node_step_map[downstream_gate_node_id] == graph_step_map[downstream_gate_node_id]
        assert traversal.node_step_map[coalesce_node_id] < traversal.node_step_map[downstream_gate_node_id]


class TestOrchestratorMultipleTransforms:
    """Test pipelines with multiple transforms."""

    def test_run_multiple_transforms_in_sequence(self, payload_store) -> None:
        """Test that multiple transforms execute in order."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 5}])
        transform1 = AddOneTransform()
        transform2 = MultiplyTwoTransform()
        transform2.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform1), as_transform(transform2)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1
        # (5 + 1) * 2 = 12
        assert sink.results[0]["value"] == 12


class TestOrchestratorEmptyPipeline:
    """Test edge cases."""

    def test_run_no_transforms(self, payload_store) -> None:
        """Test pipeline with source directly to sink."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([{"value": 99}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 1
        assert len(sink.results) == 1
        assert sink.results[0] == {"value": 99}

    def test_run_empty_source(self, payload_store) -> None:
        """Test pipeline with no rows from source."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([])  # Empty source
        transform = IdentityTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 0
        assert len(sink.results) == 0

    def test_flexible_source_contract_persisted_when_all_rows_quarantined(self, tmp_path, payload_store) -> None:
        """All-invalid FLEXIBLE runs still persist declared run contract."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "all_invalid.json"
        json_file.write_text('[{"id": "bad"}, {"id": "still_bad"}]')

        db = LandscapeDB.in_memory()
        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "flexible", "fields": ["id: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        default_sink = CollectSink(name="default")
        quarantine_sink = CollectSink(name="quarantine")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),
            },
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert run_result.rows_quarantined == 2
        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 2

        recorder = LandscapeRecorder(db)
        contract = recorder.get_run_contract(run_result.run_id)
        assert contract is not None
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True
        assert [field.normalized_name for field in contract.fields] == ["id"]


class TestOrchestratorAcceptsGraph:
    """Orchestrator accepts ExecutionGraph parameter."""

    def test_orchestrator_uses_graph_node_ids(self, plugin_manager, payload_store) -> None:
        """Orchestrator uses node IDs from graph, not generated IDs."""
        from unittest.mock import MagicMock, PropertyMock

        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Build config and graph from settings
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="output",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}})},
        )
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Use PropertyMock to track node_id setter calls
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source._on_validation_failure = "discard"

        source_node_id_setter = PropertyMock()
        type(mock_source).node_id = source_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_source.output_schema = schema_mock
        mock_source.load.return_value = iter([])
        mock_source.get_field_resolution.return_value = None
        mock_source.get_schema_contract.return_value = None

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.determinism = Determinism.IO_WRITE
        mock_sink.plugin_version = "1.0.0"

        sink_node_id_setter = PropertyMock()
        type(mock_sink).node_id = sink_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_sink.input_schema = schema_mock

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Verify orchestrator called the node_id setter with correct value from graph
        expected_source_id = graph.get_source()
        source_node_id_setter.assert_called_once_with(expected_source_id)

        # Verify sink node_id was set with correct value from graph's sink_id_map
        sink_id_map = graph.get_sink_id_map()
        expected_sink_id = sink_id_map[SinkName("output")]
        sink_node_id_setter.assert_called_once_with(expected_sink_id)

    def test_orchestrator_assigns_unique_node_ids_to_multiple_sinks(self, plugin_manager, payload_store) -> None:
        """Each sink should get a unique node_id from the graph, not shared."""
        from unittest.mock import MagicMock, PropertyMock

        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Build config with MULTIPLE sinks
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="output_a",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output_a": SinkSettings(plugin="json", options={"path": "a.json", "schema": {"mode": "observed"}}),
                "output_b": SinkSettings(plugin="json", options={"path": "b.json", "schema": {"mode": "observed"}}),
            },
        )
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            source_settings=plugins["source_settings"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Track node_id assignments with PropertyMock
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source._on_validation_failure = "discard"

        source_node_id_setter = PropertyMock()
        type(mock_source).node_id = source_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_source.output_schema = schema_mock
        mock_source.load.return_value = iter([])
        mock_source.get_field_resolution.return_value = None
        mock_source.get_schema_contract.return_value = None

        mock_sink_a = MagicMock()
        mock_sink_a.name = "output_a"
        mock_sink_a.determinism = Determinism.IO_WRITE
        mock_sink_a.plugin_version = "1.0.0"

        sink_a_node_id_setter = PropertyMock()
        type(mock_sink_a).node_id = sink_a_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_sink_a.input_schema = schema_mock

        mock_sink_b = MagicMock()
        mock_sink_b.name = "output_b"
        mock_sink_b.determinism = Determinism.IO_WRITE
        mock_sink_b.plugin_version = "1.0.0"

        sink_b_node_id_setter = PropertyMock()
        type(mock_sink_b).node_id = sink_b_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_sink_b.input_schema = schema_mock

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output_a": mock_sink_a, "output_b": mock_sink_b},
        )

        orchestrator = Orchestrator(db)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Verify each sink got a unique node_id from the graph
        sink_id_map = graph.get_sink_id_map()
        expected_sink_a_id = sink_id_map[SinkName("output_a")]
        expected_sink_b_id = sink_id_map[SinkName("output_b")]

        sink_a_node_id_setter.assert_called_once_with(expected_sink_a_id)
        sink_b_node_id_setter.assert_called_once_with(expected_sink_b_id)

        # Verify node IDs are different
        assert expected_sink_a_id != expected_sink_b_id, "Sinks should have unique node IDs"

    def test_orchestrator_run_accepts_graph(self) -> None:
        """Orchestrator.run() accepts graph parameter."""
        import inspect

        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator

        db = LandscapeDB.in_memory()

        # Build a simple graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("source_1", node_type=NodeType.SOURCE, plugin_name="csv", config=schema_config)
        graph.add_node("sink_1", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source_1", "sink_1", label="continue", mode=RoutingMode.MOVE)

        orchestrator = Orchestrator(db)

        # Should accept graph parameter (signature check)
        sig = inspect.signature(orchestrator.run)
        assert "graph" in sig.parameters

    def test_orchestrator_run_requires_graph(self, payload_store) -> None:
        """Orchestrator.run() raises ValueError if graph is None."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()
        source = ListSource([])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        # graph=None should raise ValueError
        with pytest.raises(ValueError, match="ExecutionGraph is required"):
            orchestrator.run(config, graph=None, payload_store=payload_store)
