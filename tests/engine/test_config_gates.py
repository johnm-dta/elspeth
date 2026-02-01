# tests/engine/test_config_gates.py
"""Tests for config-driven gates integration.

Config gates are defined in YAML and evaluated by the engine using ExpressionParser.
They are processed AFTER plugin transforms but BEFORE sinks.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text

from elspeth.contracts import ArtifactDescriptor, GateName, PluginSchema, SourceRow
from elspeth.core.config import GateSettings
from elspeth.core.landscape import LandscapeDB
from tests.conftest import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

# =============================================================================
# Module-Scoped Database Fixture
# =============================================================================


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Module-scoped in-memory LandscapeDB for config gate tests."""
    return LandscapeDB.in_memory()


# =============================================================================
# Shared Test Plugin Classes (P3 Fix: Deduplicate from individual tests)
# =============================================================================


class _ValueSchema(PluginSchema):
    """Schema with integer value field."""

    value: int


class _CategorySchema(PluginSchema):
    """Schema with string category field."""

    category: str


class _PrioritySchema(PluginSchema):
    """Schema with integer priority field."""

    priority: int


class ListSource(_TestSourceBase):
    """Test source that yields rows from a list.

    This is the standard test source for config gate tests.
    Provides rows from an in-memory list with configurable schema.
    """

    name = "list_source"
    output_schema: type[PluginSchema] = _ValueSchema  # Default, can be overridden

    def __init__(self, data: list[dict[str, Any]], schema: type[PluginSchema] | None = None) -> None:
        super().__init__()
        self._data = data
        if schema is not None:
            self.output_schema = schema

    def load(self, ctx: Any) -> Any:
        for row in self._data:
            yield SourceRow.valid(row)

    def close(self) -> None:
        pass


class CollectSink(_TestSinkBase):
    """Test sink that collects rows into a list.

    This is the standard test sink for config gate tests.
    Collects all written rows for assertion after pipeline completion.
    """

    name = "collect"

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, Any]] = []

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory://collect", size_bytes=0, content_hash="")

    def close(self) -> None:
        pass


# =============================================================================
# Audit Trail Verification Helpers
# =============================================================================


def verify_audit_trail(
    db: Any,
    run_id: str,
    expected_rows: int,
    expected_gate_name: str,
    expected_terminal_outcomes: dict[str, int],
) -> None:
    """Verify audit trail completeness for config gate tests.

    Args:
        db: LandscapeDB instance
        run_id: Run ID to verify
        expected_rows: Expected number of source rows processed
        expected_gate_name: Name of the config gate (without "config_gate:" prefix)
        expected_terminal_outcomes: Dict mapping outcome type (lowercase) to expected count
            e.g., {"completed": 2, "routed": 1}
    """
    with db.engine.connect() as conn:
        # 1. Verify gate node is registered
        gate_node = conn.execute(
            text("""
                SELECT node_id, plugin_name, config_json
                FROM nodes
                WHERE run_id = :run_id
                AND plugin_name = :plugin_name
            """),
            {"run_id": run_id, "plugin_name": f"config_gate:{expected_gate_name}"},
        ).fetchone()

        assert gate_node is not None, f"Gate node 'config_gate:{expected_gate_name}' should be registered"
        gate_node_id = gate_node[0]

        # 2. Verify node_states for the gate (one per row processed)
        gate_states = conn.execute(
            text("""
                SELECT state_id, token_id, status, input_hash, output_hash, error_json
                FROM node_states
                WHERE node_id = :node_id
                ORDER BY started_at
            """),
            {"node_id": gate_node_id},
        ).fetchall()

        assert len(gate_states) == expected_rows, f"Expected {expected_rows} node_states for gate, got {len(gate_states)}"

        for state in gate_states:
            _state_id, _token_id, status, input_hash, output_hash, error_json = state
            assert status == "completed", f"Gate state should be 'completed', got '{status}'"
            assert input_hash is not None, "input_hash must be recorded"
            assert output_hash is not None, "output_hash must be recorded"
            assert error_json is None, "error_json should be None for successful gate evaluation"

        # 3. Verify token_outcomes have correct terminal outcomes
        outcomes = conn.execute(
            text("""
                SELECT outcome, sink_name, COUNT(*)
                FROM token_outcomes
                WHERE run_id = :run_id
                AND is_terminal = 1
                GROUP BY outcome, sink_name
            """),
            {"run_id": run_id},
        ).fetchall()

        outcome_counts: dict[str, int] = {}
        for outcome, _sink_name, count in outcomes:
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + count

        for expected_outcome, expected_count in expected_terminal_outcomes.items():
            actual = outcome_counts.get(expected_outcome, 0)
            assert actual == expected_count, (
                f"Expected {expected_count} {expected_outcome} outcomes, got {actual}. All outcomes: {outcome_counts}"
            )

        # 4. Verify artifacts exist for sinks that received rows
        artifacts = conn.execute(
            text("""
                SELECT artifact_id, artifact_type, content_hash, path_or_uri
                FROM artifacts
                WHERE run_id = :run_id
            """),
            {"run_id": run_id},
        ).fetchall()

        # At least one artifact should exist (from sink writes)
        # Note: CollectSink produces artifacts with empty content_hash
        assert len(artifacts) > 0 or sum(expected_terminal_outcomes.values()) == 0, "Should have artifacts for sink writes"


class TestConfigGateIntegration:
    """Integration tests for config-driven gates."""

    def test_config_gate_continue(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Config gate with 'continue' destination passes rows through.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource([{"value": 10}, {"value": 20}])
        sink = CollectSink()

        # Config gate that always continues
        gate = GateSettings(
            name="always_pass",
            condition="True",  # Always true
            routes={"true": "continue", "false": "continue"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2
        assert len(sink.results) == 2

        # P1 Fix: Verify audit trail
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=2,
            expected_gate_name="always_pass",
            expected_terminal_outcomes={"completed": 2},
        )

    def test_config_gate_routes_to_sink(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Config gate routes rows to different sinks based on condition.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        # Rows: 10 (low), 100 (high), 30 (low)
        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        default_sink = CollectSink()
        high_sink = CollectSink()

        # Config gate that routes high values to a different sink
        gate = GateSettings(
            name="threshold_gate",
            condition="row['value'] > 50",
            routes={
                "true": "high",  # High values go to high sink
                "false": "continue",  # Low values continue to default
            },
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        assert result.rows_succeeded == 2  # 10, 30 -> default
        assert result.rows_routed == 1  # 100 -> high

        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1
        assert high_sink.results[0]["value"] == 100

        # P1 Fix: Verify audit trail
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="threshold_gate",
            expected_terminal_outcomes={"completed": 2, "routed": 1},
        )

        # Additional verification: Check routing event for high-value row
        with db.engine.connect() as conn:
            gate_node = conn.execute(
                text("""
                    SELECT node_id FROM nodes
                    WHERE run_id = :run_id AND plugin_name = 'config_gate:threshold_gate'
                """),
                {"run_id": result.run_id},
            ).fetchone()

            assert gate_node is not None, "Gate node should exist"
            routing_events = conn.execute(
                text("""
                    SELECT re.event_id, e.label, e.to_node_id
                    FROM routing_events re
                    JOIN node_states ns ON re.state_id = ns.state_id
                    JOIN edges e ON re.edge_id = e.edge_id
                    WHERE ns.node_id = :gate_node_id
                """),
                {"gate_node_id": gate_node[0]},
            ).fetchall()

            # Should have routing events (both for true->high and false->continue per AUD-002)
            assert len(routing_events) == 3, f"Expected 3 routing events, got {len(routing_events)}"

            # Verify the "true" route event exists (routes to high sink)
            true_events = [e for e in routing_events if e[1] == "true"]
            assert len(true_events) == 1, "Should have exactly 1 'true' routing event"

    def test_config_gate_with_string_result(self, plugin_manager, landscape_db: LandscapeDB, payload_store) -> None:
        """Config gate condition can return a string route label.

        This test uses ExecutionGraph.from_plugin_instances() for proper edge building.
        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource(
            [{"category": "A"}, {"category": "B"}, {"category": "A"}],
            schema=_CategorySchema,
        )
        a_sink = CollectSink()
        b_sink = CollectSink()

        # Build settings for graph construction
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "a_sink": SinkSettings(plugin="json", options={"path": "a.json", "schema": {"fields": "dynamic"}}),
                "b_sink": SinkSettings(plugin="json", options={"path": "b.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="a_sink",
            gates=[
                GateSettingsConfig(
                    name="category_router",
                    condition="row['category']",  # Returns 'A' or 'B'
                    routes={
                        "A": "a_sink",
                        "B": "b_sink",
                    },
                ),
            ],
        )

        # Instantiate plugins from config
        plugins = instantiate_plugins_from_config(settings)

        # Build graph from plugin instances
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        # Build PipelineConfig with actual plugin instances
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"a_sink": as_sink(a_sink), "b_sink": as_sink(b_sink)},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # All rows are routed (none go to "continue" default)
        assert result.rows_routed == 3
        assert len(a_sink.results) == 2
        assert len(b_sink.results) == 1

        # P1 Fix: Verify audit trail
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="category_router",
            expected_terminal_outcomes={"routed": 3},
        )

    def test_config_gate_integer_route_label(self, plugin_manager, landscape_db: LandscapeDB, payload_store) -> None:
        """Config gate condition can return an integer that maps to route labels.

        When an expression returns an integer (e.g., row['priority'] returns 1, 2, 3),
        the executor converts it to a string for route lookup. So routes must use
        string keys like {"1": "priority_1", "2": "priority_2"}.

        This test uses ExecutionGraph.from_plugin_instances() for proper edge building.
        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        # 3 rows with priorities 1, 2, 1 -> expect 2 go to priority_1, 1 goes to priority_2
        source = ListSource(
            [{"priority": 1}, {"priority": 2}, {"priority": 1}],
            schema=_PrioritySchema,
        )
        priority_1_sink = CollectSink()
        priority_2_sink = CollectSink()

        # Build settings for graph construction
        # NOTE: Route keys must be strings because the executor converts
        # non-bool/non-string results to strings via str()
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "priority_1": SinkSettings(plugin="json", options={"path": "priority_1.json", "schema": {"fields": "dynamic"}}),
                "priority_2": SinkSettings(plugin="json", options={"path": "priority_2.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="priority_1",
            gates=[
                GateSettingsConfig(
                    name="priority_router",
                    condition="row['priority']",  # Returns 1 or 2 (integer)
                    routes={
                        "1": "priority_1",  # String key for integer result
                        "2": "priority_2",  # String key for integer result
                    },
                ),
            ],
        )

        # Instantiate plugins from config
        plugins = instantiate_plugins_from_config(settings)

        # Build graph from plugin instances
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        # Build PipelineConfig with actual plugin instances
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "priority_1": as_sink(priority_1_sink),
                "priority_2": as_sink(priority_2_sink),
            },
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # All rows are routed (none use "continue")
        assert result.rows_routed == 3
        # 2 rows with priority 1, 1 row with priority 2
        assert len(priority_1_sink.results) == 2
        assert len(priority_2_sink.results) == 1
        # Verify the right rows went to the right sinks
        assert all(row["priority"] == 1 for row in priority_1_sink.results)
        assert all(row["priority"] == 2 for row in priority_2_sink.results)

        # P1 Fix: Verify audit trail
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="priority_router",
            expected_terminal_outcomes={"routed": 3},
        )

    def test_config_gate_node_registered_in_landscape(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Config gates are registered as nodes in Landscape.

        P1 Fix: Added comprehensive audit trail verification beyond node registration.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource([{"value": 42}])
        sink = CollectSink()

        gate = GateSettings(
            name="my_gate",
            condition="True",
            routes={"true": "continue", "false": "continue"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # Query Landscape for registered nodes
        with db.engine.connect() as conn:
            nodes = conn.execute(
                text("SELECT plugin_name, node_type FROM nodes WHERE run_id = :run_id"),
                {"run_id": result.run_id},
            ).fetchall()

        # Should have source, config gate, and sink
        node_names = [n[0] for n in nodes]
        node_types = [n[1] for n in nodes]

        assert "config_gate:my_gate" in node_names
        assert "gate" in node_types

        # P1 Fix: Verify audit trail completeness
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=1,
            expected_gate_name="my_gate",
            expected_terminal_outcomes={"completed": 1},
        )


class TestConfigGateFromSettings:
    """Tests for config gates built via ExecutionGraph.from_plugin_instances()."""

    def test_from_config_builds_config_gates(self, plugin_manager) -> None:
        """ExecutionGraph.from_plugin_instances() includes config gates."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}}),
                "review": SinkSettings(plugin="json", options={"path": "review.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettings(
                    name="quality_check",
                    condition="row['confidence'] >= 0.85",
                    routes={"true": "continue", "false": "review"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        # Should have: source, config_gate, output_sink, review_sink
        assert graph.node_count == 4

        # Config gate should be in the graph
        config_gate_map = graph.get_config_gate_id_map()
        assert GateName("quality_check") in config_gate_map

        # Route resolution should include the gate
        route_map = graph.get_route_resolution_map()
        gate_id = config_gate_map[GateName("quality_check")]
        assert (gate_id, "true") in route_map
        assert route_map[(gate_id, "true")] == "continue"
        assert (gate_id, "false") in route_map
        assert route_map[(gate_id, "false")] == "review"

    def test_from_config_validates_gate_sink_targets(self, plugin_manager) -> None:
        """ExecutionGraph.from_plugin_instances() validates gate route targets."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph, GraphValidationError

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="bad_gate",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "continue"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        with pytest.raises(GraphValidationError) as exc_info:
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                default_sink=settings.default_sink,
            )

        assert "nonexistent_sink" in str(exc_info.value)

    def test_config_gates_ordered_after_transforms(self, plugin_manager) -> None:
        """Config gates come after plugin transforms in topological order."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"fields": "dynamic"},
                },
            ),
            sinks={"output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            transforms=[
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),
            ],
            gates=[
                GateSettings(
                    name="final_gate",
                    condition="True",
                    routes={"true": "continue", "false": "continue"},
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )
        order = graph.topological_order()

        # Find indices
        transform_idx = next(i for i, n in enumerate(order) if "passthrough" in n)
        gate_idx = next(i for i, n in enumerate(order) if "config_gate" in n)
        sink_idx = next(i for i, n in enumerate(order) if "sink" in n)

        # Transform before gate, gate before sink
        assert transform_idx < gate_idx
        assert gate_idx < sink_idx


class TestMultipleConfigGates:
    """Tests for multiple config gates in sequence."""

    def test_multiple_config_gates_processed_in_order(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Multiple config gates are processed in definition order.

        P1 Fix: Added audit trail verification for both gates in sequence.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        # Row: value=25 should pass gate1 (>10) but fail gate2 (>50)
        source = ListSource([{"value": 25}])
        default_sink = CollectSink()
        low_sink = CollectSink()
        mid_sink = CollectSink()

        gate1 = GateSettings(
            name="gate1",
            condition="row['value'] > 10",
            routes={"true": "continue", "false": "low"},
        )
        gate2 = GateSettings(
            name="gate2",
            condition="row['value'] > 50",
            routes={"true": "continue", "false": "mid"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={
                "default": as_sink(default_sink),
                "low": as_sink(low_sink),
                "mid": as_sink(mid_sink),
            },
            gates=[gate1, gate2],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        # Row passes gate1, routes to mid via gate2
        assert result.rows_routed == 1
        assert len(mid_sink.results) == 1
        assert len(default_sink.results) == 0
        assert len(low_sink.results) == 0

        # P1 Fix: Verify audit trail for both gates
        # Gate1 should process the row (continue)
        verify_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=1,
            expected_gate_name="gate1",
            expected_terminal_outcomes={"routed": 1},  # Final outcome is ROUTED via gate2
        )

        # Gate2 should also have processed the row
        with db.engine.connect() as conn:
            gate2_node = conn.execute(
                text("""
                    SELECT node_id FROM nodes
                    WHERE run_id = :run_id AND plugin_name = 'config_gate:gate2'
                """),
                {"run_id": result.run_id},
            ).fetchone()

            assert gate2_node is not None, "Gate2 should be registered"

            gate2_states = conn.execute(
                text("""
                    SELECT status, input_hash, output_hash
                    FROM node_states
                    WHERE node_id = :node_id
                """),
                {"node_id": gate2_node[0]},
            ).fetchall()

            assert len(gate2_states) == 1, "Gate2 should have processed 1 row"
            status, input_hash, output_hash = gate2_states[0]
            assert status == "completed"
            assert input_hash is not None
            assert output_hash is not None
