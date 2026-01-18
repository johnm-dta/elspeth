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

from elspeth.contracts import Determinism, PluginSchema, RoutingMode
from elspeth.core.config import GateSettings

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph
    from elspeth.engine.orchestrator import PipelineConfig


# ============================================================================
# Test Fixture Base Classes
# ============================================================================


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes."""

    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    input_schema = _TestSchema
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def flush(self) -> None:
        pass

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass


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
                yield from self._data

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
            source=source,
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
                yield from self._data

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
            source=source,
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
                yield from self._data

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
            source=source,
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
                yield from self._data

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
            source=source,
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
                        "high_conf": "continue",  # Goes to main_output
                        "low_conf": "review_queue",
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
        assert (gate_id, "high_conf") in route_map
        assert route_map[(gate_id, "high_conf")] == "continue"
        assert (gate_id, "low_conf") in route_map
        assert route_map[(gate_id, "low_conf")] == "review_queue"

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
                yield from self._data

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
            source=source,
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

    Note: Fork execution is deferred to WP-07 (Fork Work Queue).
    These tests verify the configuration and graph construction only.
    """

    def test_fork_config_accepted(self) -> None:
        """Verify fork_to configuration is accepted in GateSettings."""
        gate = GateSettings(
            name="fork_gate",
            condition="True",
            routes={"all": "fork"},
            fork_to=["path_a", "path_b", "path_c"],
        )

        assert gate.fork_to == ["path_a", "path_b", "path_c"]
        assert gate.routes["all"] == "fork"

    def test_fork_config_requires_fork_to(self) -> None:
        """Verify fork route requires fork_to list."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="fork_to is required"):
            GateSettings(
                name="bad_fork",
                condition="True",
                routes={"all": "fork"},
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
                    routes={"all": "fork"},
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
        assert node_info.config["routes"]["all"] == "fork"

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
                yield from self._data

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
                    routes={"true": "fork"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            output_sink="path_a",  # Default, but fork should override for path_b
        )

        graph = ExecutionGraph.from_config(settings)

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"path_a": path_a_sink, "path_b": path_b_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"

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
                yield from self._data

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
                    routes={"true": "fork"},
                    fork_to=["stats", "alerts"],  # "stats" is NOT a sink
                ),
            ],
            output_sink="default",
        )

        graph = ExecutionGraph.from_config(settings)

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": default_sink, "alerts": alerts_sink},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"

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
        from elspeth.contracts import TransformResult
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
                yield from self._data

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
            source=source,
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
                yield from self._data

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
            source=source,
            transforms=[],
            sinks={"default": sink, "reject": CollectSink()},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        # Query Landscape for registered nodes
        with db._engine.connect() as conn:
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


# ============================================================================
# Error Handling Tests
# ============================================================================


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
