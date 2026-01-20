# tests/engine/test_engine_gates.py
"""Comprehensive integration tests for engine-level gates.

This module provides integration tests for WP-09 verification requirements:
- Composite conditions work: row['a'] > 0 and row['b'] == 'x'
- fork_to creates child tokens
- Route labels resolve correctly
- Security rejection at config time

Note: Basic gate tests exist in test_config_gates.py. This module focuses on
the WP-09 specific verification requirements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from elspeth.contracts import PluginSchema, RoutingMode, SourceRow
from elspeth.core.config import GateSettings
from tests.conftest import _TestSinkBase, _TestSourceBase, as_source

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


def _build_test_graph_with_config_gates(
    config: PipelineConfig,
) -> ExecutionGraph:
    """Build a test graph including config gates.

    Creates a linear graph matching the PipelineConfig structure:
    source -> transforms... -> config_gates... -> sinks
    """
    from elspeth.core.dag import ExecutionGraph

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

    # Add sinks first (needed for config gate edges)
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add config gates
    config_gate_ids: dict[str, str] = {}
    route_resolution_map: dict[tuple[str, str], str] = {}

    for gate_config in config.gates:
        node_id = f"config_gate_{gate_config.name}"
        config_gate_ids[gate_config.name] = node_id
        graph.add_node(
            node_id,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config={
                "condition": gate_config.condition,
                "routes": dict(gate_config.routes),
            },
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)

        # Add route edges and resolution map
        for route_label, target in gate_config.routes.items():
            route_resolution_map[(node_id, route_label)] = target
            if target not in ("continue", "fork") and target in sink_ids:
                graph.add_edge(
                    node_id, sink_ids[target], label=route_label, mode=RoutingMode.MOVE
                )

        prev = node_id

    # Edge to output sink - only add if no edge already exists to this sink
    # (gate routes may have created one)
    output_sink = "default" if "default" in sink_ids else next(iter(sink_ids))
    output_sink_node = sink_ids[output_sink]
    if not graph._graph.has_edge(prev, output_sink_node, key="continue"):
        graph.add_edge(prev, output_sink_node, label="continue", mode=RoutingMode.MOVE)

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = config_gate_ids
    graph._route_resolution_map = route_resolution_map
    graph._output_sink = output_sink

    return graph


# ============================================================================
# WP-09 Verification: Composite Conditions
# ============================================================================


class TestCompositeConditions:
    """WP-09 Verification: Composite conditions work correctly."""

    def test_composite_and_condition(self) -> None:
        """Verify: row['a'] > 0 and row['b'] == 'x' works correctly."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            a: int
            b: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Test data covering all combinations
        source = ListSource(
            [
                {"a": 5, "b": "x"},  # Both true - should route to "match"
                {"a": 0, "b": "x"},  # a=0 fails - should route to "no_match"
                {"a": 5, "b": "y"},  # b!='x' fails - should route to "no_match"
                {"a": 0, "b": "y"},  # Both fail - should route to "no_match"
            ]
        )
        match_sink = CollectSink()
        no_match_sink = CollectSink()

        # Composite AND condition
        gate = GateSettings(
            name="composite_and",
            condition="row['a'] > 0 and row['b'] == 'x'",
            routes={"true": "match", "false": "no_match"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"match": match_sink, "no_match": no_match_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 4
        # 1 match (a=5, b='x'), 3 no_match
        assert len(match_sink.results) == 1
        assert match_sink.results[0]["a"] == 5
        assert match_sink.results[0]["b"] == "x"
        assert len(no_match_sink.results) == 3

    def test_composite_or_condition(self) -> None:
        """Verify: row['status'] == 'active' or row['priority'] > 5 works."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            status: str
            priority: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource(
            [
                {"status": "active", "priority": 3},  # status active - true
                {"status": "inactive", "priority": 8},  # priority > 5 - true
                {"status": "inactive", "priority": 3},  # both false - false
            ]
        )
        pass_sink = CollectSink()
        fail_sink = CollectSink()

        gate = GateSettings(
            name="composite_or",
            condition="row['status'] == 'active' or row['priority'] > 5",
            routes={"true": "continue", "false": "fail"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": pass_sink, "fail": fail_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 2 pass, 1 fail
        assert len(pass_sink.results) == 2
        assert len(fail_sink.results) == 1

    def test_membership_condition(self) -> None:
        """Verify: row['status'] in ['active', 'pending'] works."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            status: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource(
            [
                {"status": "active"},
                {"status": "pending"},
                {"status": "deleted"},
                {"status": "suspended"},
            ]
        )
        allowed_sink = CollectSink()
        blocked_sink = CollectSink()

        gate = GateSettings(
            name="membership_check",
            condition="row['status'] in ['active', 'pending']",
            routes={"true": "continue", "false": "blocked"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": allowed_sink, "blocked": blocked_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 4
        # 2 allowed (active, pending), 2 blocked (deleted, suspended)
        assert len(allowed_sink.results) == 2
        assert {r["status"] for r in allowed_sink.results} == {"active", "pending"}
        assert len(blocked_sink.results) == 2
        assert {r["status"] for r in blocked_sink.results} == {"deleted", "suspended"}

    def test_optional_field_with_get(self) -> None:
        """Verify: row.get('optional') is not None works."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            required: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource(
            [
                {"required": "a", "optional": "present"},
                {"required": "b"},  # optional field missing
                {"required": "c", "optional": None},  # optional explicitly None
            ]
        )
        has_optional_sink = CollectSink()
        missing_optional_sink = CollectSink()

        gate = GateSettings(
            name="optional_check",
            condition="row.get('optional') is not None",
            routes={"true": "has_optional", "false": "continue"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": missing_optional_sink, "has_optional": has_optional_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 1 has optional, 2 missing/None
        assert len(has_optional_sink.results) == 1
        assert has_optional_sink.results[0]["required"] == "a"
        assert len(missing_optional_sink.results) == 2


# ============================================================================
# WP-09 Verification: Route Label Resolution
# ============================================================================


class TestRouteLabelResolution:
    """WP-09 Verification: Route labels resolve correctly."""

    def test_route_labels_resolve_to_sinks(self) -> None:
        """Verify route labels map to correct sinks."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "main_output": SinkSettings(plugin="csv"),
                "review_queue": SinkSettings(plugin="csv"),
                "archive": SinkSettings(plugin="csv"),
            },
            output_sink="main_output",
            gates=[
                GateSettingsConfig(
                    name="quality_router",
                    condition="row['confidence'] >= 0.85",
                    routes={
                        "true": "continue",  # Goes to main_output
                        "false": "review_queue",
                    },
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        # Verify route resolution map
        route_map = graph.get_route_resolution_map()
        config_gate_map = graph.get_config_gate_id_map()
        gate_id = config_gate_map["quality_router"]

        # Check route resolution
        assert (gate_id, "true") in route_map
        assert route_map[(gate_id, "true")] == "continue"
        assert (gate_id, "false") in route_map
        assert route_map[(gate_id, "false")] == "review_queue"

    def test_ternary_expression_returns_string_routes(self) -> None:
        """Verify ternary expressions can return different route labels."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            category: str

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Build settings with ternary condition that returns category directly
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "premium_sink": SinkSettings(plugin="csv"),
                "standard_sink": SinkSettings(plugin="csv"),
            },
            output_sink="standard_sink",
            gates=[
                GateSettingsConfig(
                    name="category_router",
                    condition="row['category']",  # Returns 'premium' or 'standard'
                    routes={
                        "premium": "premium_sink",
                        "standard": "standard_sink",
                    },
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        source = ListSource(
            [
                {"category": "premium"},
                {"category": "standard"},
                {"category": "premium"},
            ]
        )
        premium_sink = CollectSink()
        standard_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"premium_sink": premium_sink, "standard_sink": standard_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3
        assert len(premium_sink.results) == 2
        assert len(standard_sink.results) == 1


# ============================================================================
# WP-09 Verification: Fork Creates Child Tokens
# ============================================================================


class TestForkCreatesChildTokens:
    """WP-09 Verification: fork_to creates child tokens.

    Fork execution was implemented in WP-07 (Fork Work Queue).
    Tests verify configuration, graph construction, and execution.
    """

    def test_fork_config_accepted(self) -> None:
        """Verify fork_to configuration is accepted in GateSettings."""
        gate = GateSettings(
            name="fork_gate",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b", "path_c"],
        )

        assert gate.fork_to == ["path_a", "path_b", "path_c"]
        assert gate.routes["true"] == "fork"

    def test_fork_config_requires_fork_to(self) -> None:
        """Verify fork route requires fork_to list."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="fork_to is required"):
            GateSettings(
                name="bad_fork",
                condition="True",
                routes={"true": "fork", "false": "continue"},
                # Missing fork_to
            )

    def test_fork_to_without_fork_route_rejected(self) -> None:
        """Verify fork_to without fork route is rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="fork_to is only valid"):
            GateSettings(
                name="bad_config",
                condition="True",
                routes={"true": "continue", "false": "review"},
                fork_to=["path_a", "path_b"],  # Invalid - no fork route
            )

    def test_fork_gate_in_graph(self) -> None:
        """Verify fork gate is correctly represented in graph."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
            gates=[
                GateSettingsConfig(
                    name="parallel_processor",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["analysis_a", "analysis_b"],
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        # Verify gate node exists with fork config
        config_gate_map = graph.get_config_gate_id_map()
        gate_id = config_gate_map["parallel_processor"]
        node_info = graph.get_node_info(gate_id)

        assert node_info.config["fork_to"] == ["analysis_a", "analysis_b"]
        assert node_info.config["routes"]["true"] == "fork"

    def test_fork_children_route_to_branch_named_sinks(self) -> None:
        """Fork children with branch_name route to matching sinks.

        This is the core fork use case:
        - Gate forks to ["path_a", "path_b"]
        - Child with branch_name="path_a" goes to sink named "path_a"
        - Child with branch_name="path_b" goes to sink named "path_b"
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        path_a_sink = CollectSink()
        path_b_sink = CollectSink()

        # Config with fork gate and branch-named sinks
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="list_source"),
            sinks={
                "path_a": SinkSettings(plugin="collect"),
                "path_b": SinkSettings(plugin="collect"),
            },
            gates=[
                GateSettingsConfig(
                    name="forking_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            output_sink="path_a",  # Default, but fork should override for path_b
        )

        graph = ExecutionGraph.from_config(settings)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"path_a": path_a_sink, "path_b": path_b_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_forked == 1

        # CRITICAL: Each sink gets exactly one row (the fork child for that branch)
        assert (
            len(path_a_sink.results) == 1
        ), f"path_a should get 1 row, got {len(path_a_sink.results)}"
        assert (
            len(path_b_sink.results) == 1
        ), f"path_b should get 1 row, got {len(path_b_sink.results)}"

        # Both should have the same value (forked from same parent)
        assert path_a_sink.results[0]["value"] == 42
        assert path_b_sink.results[0]["value"] == 42

    def test_fork_unmatched_branch_falls_back_to_output_sink(self) -> None:
        """Fork child with branch_name not matching any sink goes to output_sink.

        Edge case: fork_to=["stats", "alerts"] but only "alerts" is a sink.
        Child with branch_name="stats" should fall back to output_sink.
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 99}])
        default_sink = CollectSink()  # output_sink
        alerts_sink = CollectSink()  # only one fork branch has matching sink

        # fork_to has "stats" and "alerts", but only "alerts" is a sink
        # "stats" child should fall back to default output_sink
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="list_source"),
            sinks={
                "default": SinkSettings(plugin="collect"),
                "alerts": SinkSettings(plugin="collect"),
            },
            gates=[
                GateSettingsConfig(
                    name="forking_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["stats", "alerts"],  # "stats" is NOT a sink
                ),
            ],
            output_sink="default",
        )

        graph = ExecutionGraph.from_config(settings)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": default_sink, "alerts": alerts_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_forked == 1

        # "alerts" child -> alerts_sink (branch matches sink)
        assert (
            len(alerts_sink.results) == 1
        ), f"alerts sink should get 1 row, got {len(alerts_sink.results)}"

        # "stats" child -> default_sink (no matching sink, falls back)
        assert (
            len(default_sink.results) == 1
        ), f"default sink should get 1 row (stats fallback), got {len(default_sink.results)}"

        # Both should have the same value (forked from same parent)
        assert alerts_sink.results[0]["value"] == 99
        assert default_sink.results[0]["value"] == 99

    def test_fork_multiple_source_rows_counts_correctly(self) -> None:
        """Multiple source rows fork correctly with proper counting.

        When processing multiple source rows through a fork gate:
        - rows_forked should count the NUMBER OF PARENT ROWS that forked
        - Each sink should receive one child per source row
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Three source rows - all will fork
        source = ListSource([{"value": 10}, {"value": 20}, {"value": 30}])
        analysis_sink = CollectSink()
        archive_sink = CollectSink()

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="list_source"),
            sinks={
                "analysis": SinkSettings(plugin="collect"),
                "archive": SinkSettings(plugin="collect"),
            },
            gates=[
                GateSettingsConfig(
                    name="parallel_fork",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["analysis", "archive"],
                ),
            ],
            output_sink="analysis",
        )

        graph = ExecutionGraph.from_config(settings)

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"analysis": analysis_sink, "archive": archive_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3

        # CRITICAL: rows_forked counts parent rows that forked
        assert (
            result.rows_forked == 3
        ), f"Expected 3 forked rows, got {result.rows_forked}"

        # Each sink should receive 3 rows (one child per source row)
        assert (
            len(analysis_sink.results) == 3
        ), f"analysis should get 3 rows, got {len(analysis_sink.results)}"
        assert (
            len(archive_sink.results) == 3
        ), f"archive should get 3 rows, got {len(archive_sink.results)}"

        # Verify all values are preserved
        analysis_values = {r["value"] for r in analysis_sink.results}
        archive_values = {r["value"] for r in archive_sink.results}
        expected_values = {10, 20, 30}

        assert analysis_values == expected_values
        assert archive_values == expected_values


# ============================================================================
# WP-09 Verification: Security Rejection at Config Time
# ============================================================================


class TestSecurityRejectionAtConfigTime:
    """WP-09 Verification: Malicious conditions rejected at config load.

    These tests verify that ExpressionSecurityError is raised when
    GateSettings validates malicious condition expressions.
    """

    def test_import_rejected_at_config_time(self) -> None:
        """SECURITY: __import__('os').system('rm -rf /') rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="__import__('os').system('rm -rf /')",
                routes={"true": "continue", "false": "review"},
            )

        assert "Forbidden" in str(exc_info.value)

    def test_eval_rejected_at_config_time(self) -> None:
        """SECURITY: eval('malicious') rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="eval('malicious')",
                routes={"true": "continue", "false": "review"},
            )

        assert "Forbidden" in str(exc_info.value)

    def test_exec_rejected_at_config_time(self) -> None:
        """SECURITY: exec('code') rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="exec('code')",
                routes={"true": "continue", "false": "review"},
            )

        assert "Forbidden" in str(exc_info.value)

    def test_lambda_rejected_at_config_time(self) -> None:
        """SECURITY: lambda: ... rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="(lambda: True)()",
                routes={"true": "continue", "false": "review"},
            )

        assert "Lambda" in str(exc_info.value)

    def test_list_comprehension_rejected_at_config_time(self) -> None:
        """SECURITY: [x for x in ...] rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="[x for x in row['items']]",
                routes={"true": "continue", "false": "review"},
            )

        assert "comprehension" in str(exc_info.value).lower()

    def test_attribute_access_rejected_at_config_time(self) -> None:
        """SECURITY: Attribute access beyond row[...] and row.get(...) rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="row.__class__.__bases__",
                routes={"true": "continue", "false": "review"},
            )

        assert "Forbidden" in str(exc_info.value)

    def test_arbitrary_function_call_rejected_at_config_time(self) -> None:
        """SECURITY: Function calls other than row.get() rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="len(row['items']) > 5",
                routes={"true": "continue", "false": "review"},
            )

        assert "Forbidden" in str(exc_info.value)

    def test_assignment_expression_rejected_at_config_time(self) -> None:
        """SECURITY: Assignment expressions (:=) rejected at config."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            GateSettings(
                name="malicious",
                condition="(x := row['value']) > 5",
                routes={"true": "continue", "false": "review"},
            )

        assert ":=" in str(exc_info.value) or "Assignment" in str(exc_info.value)


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================


class TestEndToEndPipeline:
    """Full pipeline integration tests with gates."""

    def test_source_transform_gate_sink_pipeline(self) -> None:
        """End-to-end: Source -> Transform -> Config Gate -> Sink."""
        from elspeth.contracts import SourceRow, TransformResult
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            raw_score: int

        class OutputSchema(PluginSchema):
            raw_score: int
            score: float

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class NormalizeTransform(BaseTransform):
            """Transform that normalizes score to 0-1 range."""

            name = "normalize"
            input_schema = InputSchema
            output_schema = OutputSchema
            plugin_version = "1.0.0"

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                normalized = row["raw_score"] / 100.0
                return TransformResult.success({**row, "score": normalized})

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource(
            [
                {"raw_score": 90},  # 0.9 - high confidence
                {"raw_score": 50},  # 0.5 - low confidence
                {"raw_score": 85},  # 0.85 - exactly threshold
            ]
        )
        transform = NormalizeTransform(config={})
        high_conf_sink = CollectSink()
        low_conf_sink = CollectSink()

        gate = GateSettings(
            name="confidence_gate",
            condition="row['score'] >= 0.85",
            routes={"true": "continue", "false": "low_conf"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[transform],
            sinks={"default": high_conf_sink, "low_conf": low_conf_sink},
            gates=[gate],
        )

        # Build graph manually with transform
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="list_source")
        graph.add_node("transform_0", node_type="transform", plugin_name="normalize")
        graph.add_node(
            "config_gate_confidence_gate",
            node_type="gate",
            plugin_name="config_gate:confidence_gate",
            config={"condition": gate.condition, "routes": dict(gate.routes)},
        )
        graph.add_node("sink_default", node_type="sink", plugin_name="collect")
        graph.add_node("sink_low_conf", node_type="sink", plugin_name="collect")

        graph.add_edge("source", "transform_0", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge(
            "transform_0",
            "config_gate_confidence_gate",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        graph.add_edge(
            "config_gate_confidence_gate",
            "sink_default",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        graph.add_edge(
            "config_gate_confidence_gate",
            "sink_low_conf",
            label="false",
            mode=RoutingMode.MOVE,
        )

        graph._sink_id_map = {"default": "sink_default", "low_conf": "sink_low_conf"}
        graph._transform_id_map = {0: "transform_0"}
        graph._config_gate_id_map = {"confidence_gate": "config_gate_confidence_gate"}
        graph._route_resolution_map = {
            ("config_gate_confidence_gate", "true"): "continue",
            ("config_gate_confidence_gate", "false"): "low_conf",
        }
        graph._output_sink = "default"

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 2 high confidence (0.9, 0.85), 1 low confidence (0.5)
        assert len(high_conf_sink.results) == 2
        assert len(low_conf_sink.results) == 1
        assert low_conf_sink.results[0]["score"] == 0.5

    def test_audit_trail_records_gate_evaluation(self) -> None:
        """Verify audit trail records gate condition and result."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        gate = GateSettings(
            name="audit_test_gate",
            condition="row['value'] > 0",
            routes={"true": "continue", "false": "reject"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": sink, "reject": CollectSink()},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        # Query Landscape for registered nodes
        with db.engine.connect() as conn:
            from sqlalchemy import text

            nodes = conn.execute(
                text("SELECT plugin_name, node_type FROM nodes WHERE run_id = :run_id"),
                {"run_id": result.run_id},
            ).fetchall()

        # Verify gate node is registered
        node_names = [n[0] for n in nodes]
        node_types = [n[1] for n in nodes]

        assert "config_gate:audit_test_gate" in node_names
        assert "gate" in node_types

    def test_gate_audit_trail_includes_evaluation_metadata(self) -> None:
        """WP-14b: Verify gate audit trail includes condition, result, and route.

        ELSPETH is built for high-stakes accountability. Every gate decision
        must be auditable - traceable to what condition was evaluated and
        what route was taken.

        This test verifies that for gate evaluations:
        1. The node_states table records the gate evaluation
        2. The routing_events table (for routed tokens) includes metadata
        3. The metadata contains: condition evaluated, evaluation result, route taken
        """
        import json

        from sqlalchemy import text

        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            priority: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                for _row in self._data:
                    yield SourceRow.valid(_row)

            def close(self) -> None:
                pass

        class CollectSink(_TestSinkBase):
            name = "collect"
            config: ClassVar[dict[str, Any]] = {}

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        # Two rows: one high priority (routes to "urgent"), one low (continues to default)
        source = ListSource([{"priority": 10}, {"priority": 2}])
        default_sink = CollectSink()
        urgent_sink = CollectSink()

        gate = GateSettings(
            name="priority_gate",
            condition="row['priority'] > 5",
            routes={"true": "urgent", "false": "continue"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": default_sink, "urgent": urgent_sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        assert result.status == "completed"
        assert result.rows_processed == 2

        # Verify sink routing
        assert len(urgent_sink.results) == 1, "High priority should route to urgent"
        assert len(default_sink.results) == 1, "Low priority should continue to default"
        assert urgent_sink.results[0]["priority"] == 10
        assert default_sink.results[0]["priority"] == 2

        # Query the database to verify audit trail completeness
        with db.engine.connect() as conn:
            # 1. Find the gate node
            gate_node = conn.execute(
                text("""
                    SELECT node_id, config_json
                    FROM nodes
                    WHERE run_id = :run_id
                    AND node_type = 'gate'
                """),
                {"run_id": result.run_id},
            ).fetchone()

            assert gate_node is not None, "Gate node should be registered"
            gate_node_id = gate_node[0]

            # Verify gate config is recorded (includes condition)
            gate_config = json.loads(gate_node[1])
            assert gate_config["condition"] == "row['priority'] > 5"
            assert "routes" in gate_config
            assert gate_config["routes"]["true"] == "urgent"
            assert gate_config["routes"]["false"] == "continue"

            # 2. Find node_states for the gate (one per token processed)
            gate_states = conn.execute(
                text("""
                    SELECT state_id, token_id, status, input_hash, output_hash
                    FROM node_states
                    WHERE node_id = :node_id
                    ORDER BY started_at
                """),
                {"node_id": gate_node_id},
            ).fetchall()

            assert len(gate_states) == 2, "Should have 2 node_states (one per row)"

            # All gate states should be "completed" (terminal state is derived)
            for state in gate_states:
                assert state[2] == "completed", "Gate state should be 'completed'"
                assert state[3] is not None, "input_hash must be recorded"
                assert state[4] is not None, "output_hash must be recorded"

            # 3. Find routing events for the gate evaluations
            # Only routed tokens (not "continue") generate routing events
            routing_events = conn.execute(
                text("""
                    SELECT re.event_id, re.state_id, re.edge_id, re.reason_hash,
                           e.label, e.to_node_id
                    FROM routing_events re
                    JOIN node_states ns ON re.state_id = ns.state_id
                    JOIN edges e ON re.edge_id = e.edge_id
                    WHERE ns.node_id = :node_id
                """),
                {"node_id": gate_node_id},
            ).fetchall()

            # We expect 1 routing event for the "true" -> "urgent" route
            # The "false" -> "continue" route doesn't generate a routing event
            # because it just continues to next node
            assert (
                len(routing_events) == 1
            ), f"Expected 1 routing event for routed token, got {len(routing_events)}"

            routing_event = routing_events[0]
            edge_label = routing_event[4]
            assert (
                edge_label == "true"
            ), f"Edge label should be 'true', got {edge_label}"

            # The routing event should have a reason_hash (metadata was recorded)
            reason_hash = routing_event[3]
            assert (
                reason_hash is not None
            ), "Routing event should have reason_hash for audit trail"

            # 4. Verify the edge points to the urgent sink
            to_node_id = routing_event[5]
            sink_node = conn.execute(
                text("""
                    SELECT plugin_name FROM nodes WHERE node_id = :node_id
                """),
                {"node_id": to_node_id},
            ).fetchone()
            assert sink_node is not None
            # The sink plugin name should indicate it's the urgent sink


# ============================================================================
# Error Handling Tests
# ============================================================================


# ============================================================================
# WP-14b: Gate Runtime Error Handling
# ============================================================================


class TestGateRuntimeErrors:
    """WP-14b: Verify runtime condition errors follow Three-Tier Trust Model.

    Per ELSPETH's data manifesto:
    - Row data is Tier 2 (elevated trust) - types are trustworthy but operations can fail
    - Missing fields or type errors during gate evaluation should fail clearly
    - row.get() is the correct pattern for optional fields

    These tests verify that gate conditions fail predictably when row data
    doesn't meet expectations, and that optional field patterns work correctly.
    """

    def test_missing_field_raises_key_error(self) -> None:
        """Gate condition referencing missing field should fail clearly.

        Per ELSPETH's design: operations on row values can fail, and when they
        do, the failure should be clear and auditable. A gate condition that
        references row['missing_field'] should raise KeyError when the field
        doesn't exist, and this should be recorded as a failed node state.
        """
        from elspeth.contracts import TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register a fake gate node for audit trail
        schema_config = SchemaConfig.from_dict({"fields": "dynamic"})
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="config_gate:missing_test",
            node_type="gate",
            plugin_version="1.0.0",
            config={"condition": "row['nonexistent'] > 0"},
            schema_config=schema_config,
        )

        span_factory = SpanFactory()  # No tracer = no-op spans
        executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map={},
            route_resolution_map={},
        )

        gate_config = GateSettings(
            name="missing_test",
            condition="row['nonexistent'] > 0",
            routes={"true": "continue", "false": "continue"},
        )

        token = TokenInfo(
            row_id="row_1",
            token_id="token_1",
            row_data={"existing_field": 42},  # Missing 'nonexistent' field
        )

        # Create row and token in landscape for audit trail
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        ctx = PluginContext(run_id=run.run_id, config={})

        # The expression parser evaluates row['nonexistent'], which raises KeyError
        # The executor should catch this and re-raise after recording failure
        with pytest.raises(KeyError) as exc_info:
            executor.execute_config_gate(
                gate_config=gate_config,
                node_id=node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

        # Verify the error message indicates the missing field
        assert "nonexistent" in str(exc_info.value)

        # Verify the failure was recorded in the audit trail
        with db.engine.connect() as conn:
            from sqlalchemy import text

            states = conn.execute(
                text("""
                    SELECT status, error_json
                    FROM node_states
                    WHERE node_id = :node_id
                """),
                {"node_id": node.node_id},
            ).fetchall()

            assert len(states) == 1, "Should have exactly one node state"
            status, error_json = states[0]
            assert status == "failed", "Node state should be failed"
            assert error_json is not None, "Error should be recorded"

            import json

            error = json.loads(error_json)
            assert (
                error["type"] == "KeyError"
            ), f"Expected KeyError, got {error['type']}"

    def test_optional_field_with_get_succeeds(self) -> None:
        """Gate using row.get() for optional field should succeed safely.

        The row.get() pattern is the correct way to handle optional fields
        in gate conditions. This test verifies that:
        1. row.get('field') returns None for missing fields (no exception)
        2. row.get('field', default) returns the default for missing fields
        3. The gate evaluates correctly and routes based on the result
        """
        from elspeth.contracts import TokenInfo
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register a fake gate node for audit trail
        schema_config = SchemaConfig.from_dict({"fields": "dynamic"})
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="config_gate:optional_test",
            node_type="gate",
            plugin_version="1.0.0",
            config={"condition": "row.get('optional', 0) > 5"},
            schema_config=schema_config,
        )

        span_factory = SpanFactory()  # No tracer = no-op spans
        executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map={},
            route_resolution_map={
                (node.node_id, "true"): "continue",
                (node.node_id, "false"): "continue",
            },
        )

        gate_config = GateSettings(
            name="optional_test",
            condition="row.get('optional', 0) > 5",
            routes={"true": "continue", "false": "continue"},
        )

        # Test 1: Missing optional field - should use default (0) and evaluate to false
        token_missing = TokenInfo(
            row_id="row_1",
            token_id="token_1",
            row_data={"required": "value"},  # Missing 'optional' field
        )

        # Create row and token in landscape for audit trail
        row1 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token_missing.row_data,
            row_id=token_missing.row_id,
        )
        recorder.create_token(row_id=row1.row_id, token_id=token_missing.token_id)

        ctx = PluginContext(run_id=run.run_id, config={})

        # Should NOT raise - row.get() handles missing fields safely
        outcome_missing = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=node.node_id,
            token=token_missing,
            ctx=ctx,
            step_in_pipeline=0,
        )

        # With default 0, condition "0 > 5" is false
        assert outcome_missing.result.action.kind.value == "continue"
        # Check that the raw evaluation result (captured in reason) shows "false"
        assert outcome_missing.result.action.reason["result"] == "false"

        # Test 2: Present optional field with value > 5 - should evaluate to true
        token_present = TokenInfo(
            row_id="row_2",
            token_id="token_2",
            row_data={"required": "value", "optional": 10},
        )

        row2 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=1,
            data=token_present.row_data,
            row_id=token_present.row_id,
        )
        recorder.create_token(row_id=row2.row_id, token_id=token_present.token_id)

        outcome_present = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=node.node_id,
            token=token_present,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # With value 10, condition "10 > 5" is true
        assert outcome_present.result.action.kind.value == "continue"
        assert outcome_present.result.action.reason["result"] == "true"

        # Test 3: Present optional field with value <= 5 - should evaluate to false
        token_low = TokenInfo(
            row_id="row_3",
            token_id="token_3",
            row_data={"required": "value", "optional": 3},
        )

        row3 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=2,
            data=token_low.row_data,
            row_id=token_low.row_id,
        )
        recorder.create_token(row_id=row3.row_id, token_id=token_low.token_id)

        outcome_low = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=node.node_id,
            token=token_low,
            ctx=ctx,
            step_in_pipeline=2,
        )

        # With value 3, condition "3 > 5" is false
        assert outcome_low.result.action.kind.value == "continue"
        assert outcome_low.result.action.reason["result"] == "false"

        # Verify all node states are completed (not failed)
        with db.engine.connect() as conn:
            from sqlalchemy import text

            states = conn.execute(
                text("""
                    SELECT status
                    FROM node_states
                    WHERE node_id = :node_id
                    ORDER BY step_index
                """),
                {"node_id": node.node_id},
            ).fetchall()

            assert len(states) == 3, "Should have three node states"
            for state in states:
                assert state[0] == "completed", "All states should be completed"


class TestErrorHandling:
    """Error handling scenarios for gates."""

    def test_invalid_condition_rejected_at_config_time(self) -> None:
        """Invalid condition syntax rejected when creating GateSettings."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Invalid"):
            GateSettings(
                name="bad_syntax",
                condition="row['field'] >",  # Incomplete expression
                routes={"true": "continue", "false": "review"},
            )

    def test_route_to_nonexistent_sink_caught_at_graph_construction(self) -> None:
        """Route to non-existent sink caught when building graph."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            output_sink="output",
            gates=[
                GateSettingsConfig(
                    name="bad_route",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "continue"},
                ),
            ],
        )

        with pytest.raises(GraphValidationError, match="nonexistent_sink"):
            ExecutionGraph.from_config(settings)
