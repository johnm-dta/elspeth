# tests/integration/pipeline/orchestrator/test_orchestrator_core.py
"""Core orchestrator tests.

Migrated from tests/engine/test_orchestrator_core.py.
Uses v2 fixtures and production assembly path (BUG-LINEAGE-01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism, NodeType, PipelineRow, RoutingMode, RunStatus, SinkName, SourceRow
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.testing import make_pipeline_row, make_source_row
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.landscape import make_factory
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult
    from elspeth.core.landscape import LandscapeDB


# ---------------------------------------------------------------------------
# Shared test transforms (specific to orchestrator_core tests)
# ---------------------------------------------------------------------------


class DoubleTransform(BaseTransform):
    """Transform that doubles a value field."""

    name = "double"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.infrastructure.results import TransformResult

        return TransformResult.success(
            make_pipeline_row({"value": row["value"], "doubled": row["value"] * 2}),
            success_reason={"action": "double"},
        )


class AddOneTransform(BaseTransform):
    """Transform that adds 1 to a value field."""

    name = "add_one"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.infrastructure.results import TransformResult

        return TransformResult.success(make_pipeline_row({"value": row["value"] + 1}), success_reason={"action": "add_one"})


class MultiplyTwoTransform(BaseTransform):
    """Transform that multiplies value by 2."""

    name = "multiply_two"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.infrastructure.results import TransformResult

        return TransformResult.success(make_pipeline_row({"value": row["value"] * 2}), success_reason={"action": "multiply_two"})


class ValidationErrorAfterValidRowSource(_TestSourceBase):
    """Source that records a validation error after a valid row has been processed."""

    name = "validation_error_after_valid_row"
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__()
        self.on_success = "default"
        self._on_validation_failure = "quarantine"

    def load(self, ctx: Any) -> Any:
        yield make_source_row({"value": 1})
        ctx.record_validation_error(
            row={"value": "bad"},
            error="source parse failed on second row",
            schema_mode="parse",
            destination="quarantine",
        )
        yield SourceRow.quarantined(
            row={"value": "bad"},
            error="source parse failed on second row",
            destination="quarantine",
        )


class LoadTrackingSource(ListSource):
    """List source that records whether the processing loop reached load()."""

    determinism = Determinism.IO_READ

    def __init__(self, data: list[dict[str, Any]]) -> None:
        super().__init__(data)
        self.load_started = False

    def load(self, ctx: Any) -> Any:
        self.load_started = True
        yield from super().load(ctx)


class RuntimePreflightFailingTransform(PassTransform):
    """Transform whose runtime preflight fails before source iteration."""

    name = "runtime_preflight_failing"
    determinism = Determinism.DETERMINISTIC
    requires_runtime_preflight = True

    def runtime_preflight(self, ctx: Any) -> None:
        raise RuntimeError("pre_flight_failed: provider auth exploded")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self, landscape_db: LandscapeDB, payload_store) -> None:
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = DoubleTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 3
        assert len(sink.results) == 3
        assert sink.results[0] == {"value": 1, "doubled": 2}

    def test_runtime_preflight_failure_aborts_before_source_load(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Runtime preflight failures must fail the run before any row work starts."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = LoadTrackingSource([{"value": 1}])
        transform = RuntimePreflightFailingTransform()
        transform.on_success = "default"
        sink = CollectSink()
        run_id = "run-runtime-preflight-fails-before-load"

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)
        with pytest.raises(RuntimeError, match="pre_flight_failed: provider auth exploded"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store, run_id=run_id)

        assert source.load_started is False
        assert sink.results == []

        factory = make_factory(landscape_db)
        run_row = factory.run_lifecycle.get_run(run_id)
        assert run_row is not None
        assert run_row.status == RunStatus.FAILED

        operations = factory.execution.get_operations_for_run(run_id)
        runtime_preflight_ops = [op for op in operations if op.operation_type == "runtime_preflight"]
        assert len(runtime_preflight_ops) == 1
        operation = runtime_preflight_ops[0]
        assert operation.node_id == transform.node_id
        assert operation.status == "failed"
        assert operation.error_message is not None
        assert "pre_flight_failed" in operation.error_message
        assert "provider auth exploded" in operation.error_message

    def test_source_validation_error_after_transform_is_attributed_to_source_node(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Later generator-step validation errors must not inherit transform node_id."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ValidationErrorAfterValidRowSource()
        transform = PassTransform()
        transform.on_success = "default"
        default_sink = CollectSink(name="default")
        quarantine_sink = CollectSink(name="quarantine")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "default": as_sink(default_sink),
                "quarantine": as_sink(quarantine_sink),
            },
        )

        orchestrator = Orchestrator(landscape_db)
        graph = build_production_graph(config)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Phase 2.2: rows_succeeded > 0 alongside rows_quarantined > 0 maps
        # to `completed_with_failures` under the presence-indicator predicate.
        assert run_result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert run_result.rows_processed == 2
        assert run_result.rows_quarantined == 1
        assert len(default_sink.results) == 1
        assert len(quarantine_sink.results) == 1

        factory = make_factory(landscape_db)
        errors = factory.data_flow.get_validation_errors_for_run(run_result.run_id)
        assert len(errors) == 1
        assert errors[0].node_id == source.node_id
        assert errors[0].node_id != transform.node_id
        assert errors[0].error == "source parse failed on second row"

    def test_run_with_gate_routing(self, landscape_db: LandscapeDB, payload_store) -> None:
        from elspeth.core.config import GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

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

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # ADR-019: gate route_to_sink is intentional MOVE provenance on top of
        # lifecycle SUCCESS. Gate-routed-only run -> COMPLETED.
        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_succeeded == 3
        assert run_result.rows_routed_success == 3
        # value=10 and value=30 go to default, value=100 goes to high
        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1

    def test_nonterminal_coalesce_continues_to_downstream_gate(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Merged fork paths at a non-terminal coalesce must continue downstream."""
        from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ListSource([{"value": 1}, {"value": 2}], on_success="source_sink")
        transform = PassTransform()
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
        runtime_coalesce = CoalesceSettings(
            name="merge_paths",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="output",
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
            sinks={
                "output": {"plugin": "test", "on_write_failure": "discard"},
                "source_sink": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[fork_gate, terminal_gate],
            coalesce=[runtime_coalesce],
        )

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(
            config,
            graph=build_production_graph(config),
            settings=settings,
            payload_store=payload_store,
        )

        # ADR-019: gate route_to_sink is intentional MOVE provenance on top of
        # lifecycle SUCCESS. Gate-routed-only run -> COMPLETED.
        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 2
        assert run_result.rows_routed_success == 2
        assert len(output_sink.results) == 2
        assert len(source_sink.results) == 0

    def test_traversal_context_keeps_nonterminal_coalesce_in_graph_step_order(self, landscape_db: LandscapeDB) -> None:
        """Traversal context must preserve graph step order for non-terminal coalesce nodes."""
        from elspeth.contracts.types import CoalesceName, GateName
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.engine.orchestrator import PipelineConfig
        from elspeth.engine.orchestrator.graph_wiring import (
            assign_plugin_node_ids,
            build_dag_traversal_context,
        )

        source = ListSource([{"value": 1}], on_success="source_sink")
        transform = PassTransform()
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

        assign_plugin_node_ids(
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

        traversal = build_dag_traversal_context(
            graph=graph,
            config=config,
            config_gate_id_map=graph.get_config_gate_id_map(),
        )
        assert traversal.node_step_map[coalesce_node_id] == graph_step_map[coalesce_node_id]
        assert traversal.node_step_map[downstream_gate_node_id] == graph_step_map[downstream_gate_node_id]
        assert traversal.node_step_map[coalesce_node_id] < traversal.node_step_map[downstream_gate_node_id]


class TestOrchestratorMultipleTransforms:
    """Test pipelines with multiple transforms."""

    def test_run_multiple_transforms_in_sequence(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Test that multiple transforms execute in order."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

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

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert len(sink.results) == 1
        # (5 + 1) * 2 = 12
        assert sink.results[0]["value"] == 12


class TestOrchestratorEmptyPipeline:
    """Test edge cases."""

    def test_run_no_transforms(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Test pipeline with source directly to sink."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ListSource([{"value": 99}])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 1
        assert len(sink.results) == 1
        assert sink.results[0] == {"value": 99}

    def test_run_empty_source(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Test pipeline with no rows from source."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ListSource([])  # Empty source
        transform = PassTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Phase 2.2: status taxonomy now distinguishes this shape as EMPTY.
        assert run_result.status == RunStatus.EMPTY
        assert run_result.rows_processed == 0
        assert len(sink.results) == 0

    def test_flexible_source_contract_persisted_when_all_rows_quarantined(self, landscape_db: LandscapeDB, tmp_path, payload_store) -> None:
        """All-invalid FLEXIBLE runs still persist declared run contract."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "all_invalid.json"
        json_file.write_text('[{"id": "bad"}, {"id": "still_bad"}]')

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

        orchestrator = Orchestrator(landscape_db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # All rows fail source validation and are quarantined. Per
        # CLAUDE.md Tier-3 data manifesto, quarantine is a clean terminal
        # outcome — the pipeline made a deliberate classification on every
        # row. With ``terminal_clean_indicator`` satisfied via quarantine
        # and no uncaught ``failure_indicator``, the predicate returns
        # COMPLETED_WITH_FAILURES rather than FAILED.
        assert run_result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert run_result.rows_processed == 2
        assert run_result.rows_quarantined == 2
        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 2

        factory = make_factory(landscape_db)
        contract = factory.run_lifecycle.get_run_contract(run_result.run_id)
        assert contract is not None
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True
        assert [field.normalized_name for field in contract.fields] == ["id"]


class TestOrchestratorAcceptsGraph:
    """Orchestrator accepts ExecutionGraph parameter."""

    def test_orchestrator_uses_graph_node_ids(self, landscape_db: LandscapeDB, plugin_manager, payload_store) -> None:
        """Orchestrator uses node IDs from graph, not generated IDs."""
        from unittest.mock import MagicMock, PropertyMock

        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

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
            sinks={
                "output": SinkSettings(
                    plugin="json", on_write_failure="discard", options={"path": "output.json", "schema": {"mode": "observed"}}
                )
            },
        )
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Use PropertyMock to track node_id setter calls
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.source_file_hash = None
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
        mock_sink.source_file_hash = None
        mock_sink._on_write_failure = "discard"
        mock_sink._reset_diversion_log = MagicMock()

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

        orchestrator = Orchestrator(landscape_db)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Verify orchestrator called the node_id setter with correct value from graph
        expected_source_id = graph.get_source()
        source_node_id_setter.assert_called_once_with(expected_source_id)

        # Verify sink node_id was set with correct value from graph's sink_id_map
        sink_id_map = graph.get_sink_id_map()
        expected_sink_id = sink_id_map[SinkName("output")]
        sink_node_id_setter.assert_called_once_with(expected_sink_id)

    def test_orchestrator_assigns_unique_node_ids_to_multiple_sinks(self, landscape_db: LandscapeDB, plugin_manager, payload_store) -> None:
        """Each sink should get a unique node_id from the graph, not shared."""
        from unittest.mock import MagicMock, PropertyMock

        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

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
                "output_a": SinkSettings(
                    plugin="json", on_write_failure="discard", options={"path": "a.json", "schema": {"mode": "observed"}}
                ),
                "output_b": SinkSettings(
                    plugin="json", on_write_failure="discard", options={"path": "b.json", "schema": {"mode": "observed"}}
                ),
            },
        )
        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins.source,
            source_settings=plugins.source_settings,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
        )

        # Track node_id assignments with PropertyMock
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.determinism = Determinism.IO_READ
        mock_source.plugin_version = "1.0.0"
        mock_source.source_file_hash = None
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
        mock_sink_a.source_file_hash = None
        mock_sink_a._on_write_failure = "discard"
        mock_sink_a._reset_diversion_log = MagicMock()

        sink_a_node_id_setter = PropertyMock()
        type(mock_sink_a).node_id = sink_a_node_id_setter

        schema_mock = MagicMock()
        schema_mock.model_json_schema.return_value = {"type": "object"}
        mock_sink_a.input_schema = schema_mock

        mock_sink_b = MagicMock()
        mock_sink_b.name = "output_b"
        mock_sink_b.determinism = Determinism.IO_WRITE
        mock_sink_b.plugin_version = "1.0.0"
        mock_sink_b.source_file_hash = None
        mock_sink_b._on_write_failure = "discard"
        mock_sink_b._reset_diversion_log = MagicMock()

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

        orchestrator = Orchestrator(landscape_db)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Verify each sink got a unique node_id from the graph
        sink_id_map = graph.get_sink_id_map()
        expected_sink_a_id = sink_id_map[SinkName("output_a")]
        expected_sink_b_id = sink_id_map[SinkName("output_b")]

        sink_a_node_id_setter.assert_called_once_with(expected_sink_a_id)
        sink_b_node_id_setter.assert_called_once_with(expected_sink_b_id)

        # Verify node IDs are different
        assert expected_sink_a_id != expected_sink_b_id, "Sinks should have unique node IDs"

    def test_orchestrator_run_accepts_graph(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator.run() accepts graph parameter."""
        import inspect

        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator

        # Build a simple graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("source_1", node_type=NodeType.SOURCE, plugin_name="csv", config=schema_config)
        graph.add_node("sink_1", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source_1", "sink_1", label="continue", mode=RoutingMode.MOVE)

        orchestrator = Orchestrator(landscape_db)

        # Should accept graph parameter (signature check)
        sig = inspect.signature(orchestrator.run)
        assert "graph" in sig.parameters

    def test_orchestrator_run_requires_graph(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Orchestrator.run() raises ValueError if graph is None."""
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        source = ListSource([])
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db)

        # graph=None should raise ValueError
        with pytest.raises(OrchestrationInvariantError, match="ExecutionGraph is required"):
            orchestrator.run(config, graph=None, payload_store=payload_store)
