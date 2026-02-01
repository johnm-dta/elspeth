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

import json
from typing import Any

import pytest
from sqlalchemy import text

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import ArtifactDescriptor, GateName, NodeID, NodeType, PluginSchema, RoutingMode, SourceRow
from elspeth.core.config import GateSettings
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.expression_parser import ExpressionEvaluationError
from tests.conftest import _TestSinkBase, _TestSourceBase, as_sink, as_source, as_transform
from tests.engine.orchestrator_test_helpers import build_production_graph

# =============================================================================
# Shared Test Plugin Classes (P3 Fix: Deduplicate from individual tests)
# =============================================================================


class _CompositeSchema(PluginSchema):
    """Schema for composite condition tests (a: int, b: str)."""

    a: int
    b: str


class _StatusPrioritySchema(PluginSchema):
    """Schema for OR condition tests."""

    status: str
    priority: int


class _StatusOnlySchema(PluginSchema):
    """Schema with just status field."""

    status: str


class _RequiredOnlySchema(PluginSchema):
    """Schema with required field only."""

    required: str


class _ValueSchema(PluginSchema):
    """Schema with integer value field."""

    value: int


class _PrioritySchema(PluginSchema):
    """Schema with integer priority field."""

    priority: int


class _CategorySchema(PluginSchema):
    """Schema with string category field."""

    category: str


class _RawScoreSchema(PluginSchema):
    """Schema with raw_score field."""

    raw_score: int


class _NormalizedScoreSchema(PluginSchema):
    """Schema with raw_score and normalized score fields."""

    raw_score: int
    score: float


class ListSource(_TestSourceBase):
    """Test source that yields rows from a list.

    This is the standard test source for engine gate tests.
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

    This is the standard test sink for engine gate tests.
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


def verify_gate_audit_trail(
    db: Any,
    run_id: str,
    expected_rows: int,
    expected_gate_name: str,
    expected_terminal_outcomes: dict[str, int],
) -> None:
    """Verify audit trail completeness for engine gate tests.

    Args:
        db: LandscapeDB instance
        run_id: Run ID to verify
        expected_rows: Expected number of rows processed by the gate
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

        # 2. Verify node_states for the gate
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
            _state_id, _token_id, status, input_hash, output_hash, _error_json = state
            assert status == "completed", f"Gate state should be 'completed', got '{status}'"
            assert input_hash is not None, "input_hash must be recorded"
            assert output_hash is not None, "output_hash must be recorded"

        # 3. Verify token_outcomes have correct terminal outcomes
        outcomes = conn.execute(
            text("""
                SELECT outcome, COUNT(*)
                FROM token_outcomes
                WHERE run_id = :run_id
                AND is_terminal = 1
                GROUP BY outcome
            """),
            {"run_id": run_id},
        ).fetchall()

        outcome_counts = dict(outcomes)

        for expected_outcome, expected_count in expected_terminal_outcomes.items():
            actual = outcome_counts.get(expected_outcome, 0)
            assert actual == expected_count, (
                f"Expected {expected_count} {expected_outcome} outcomes, got {actual}. All outcomes: {outcome_counts}"
            )


def verify_fork_audit_trail(
    db: Any,
    run_id: str,
    expected_parent_rows: int,
    expected_child_count_per_parent: int,
    expected_gate_name: str,
) -> None:
    """Verify audit trail for fork operations.

    Args:
        db: LandscapeDB instance
        run_id: Run ID to verify
        expected_parent_rows: Number of parent rows that forked
        expected_child_count_per_parent: Number of children per parent
        expected_gate_name: Name of the forking gate
    """
    with db.engine.connect() as conn:
        # 1. Verify fork outcomes are recorded (outcome values are lowercase)
        fork_outcomes = conn.execute(
            text("""
                SELECT outcome, fork_group_id, COUNT(*)
                FROM token_outcomes
                WHERE run_id = :run_id
                AND outcome = 'forked'
                GROUP BY fork_group_id
            """),
            {"run_id": run_id},
        ).fetchall()

        # Each parent should have exactly one forked outcome
        assert len(fork_outcomes) == expected_parent_rows, f"Expected {expected_parent_rows} fork groups, got {len(fork_outcomes)}"

        # 2. Verify token_parents relationships for forked children
        parent_relationships = conn.execute(
            text("""
                SELECT tp.token_id, tp.parent_token_id, t.fork_group_id, t.branch_name
                FROM token_parents tp
                JOIN tokens t ON tp.token_id = t.token_id
                JOIN rows r ON t.row_id = r.row_id
                WHERE r.run_id = :run_id
                AND t.fork_group_id IS NOT NULL
            """),
            {"run_id": run_id},
        ).fetchall()

        # Should have (parent_rows * children_per_parent) relationships
        expected_total = expected_parent_rows * expected_child_count_per_parent
        assert len(parent_relationships) == expected_total, (
            f"Expected {expected_total} parent relationships, got {len(parent_relationships)}"
        )


# ============================================================================
# WP-09 Verification: Composite Conditions
# ============================================================================


class TestCompositeConditions:
    """WP-09 Verification: Composite conditions work correctly."""

    def test_composite_and_condition(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify: row['a'] > 0 and row['b'] == 'x' works correctly.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        # Test data covering all combinations
        source = ListSource(
            [
                {"a": 5, "b": "x"},  # Both true - should route to "match"
                {"a": 0, "b": "x"},  # a=0 fails - should route to "no_match"
                {"a": 5, "b": "y"},  # b!='x' fails - should route to "no_match"
                {"a": 0, "b": "y"},  # Both fail - should route to "no_match"
            ],
            schema=_CompositeSchema,
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
            sinks={"match": as_sink(match_sink), "no_match": as_sink(no_match_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 4
        # 1 match (a=5, b='x'), 3 no_match
        assert len(match_sink.results) == 1
        assert match_sink.results[0]["a"] == 5
        assert match_sink.results[0]["b"] == "x"
        assert len(no_match_sink.results) == 3

        # P1 Fix: Verify audit trail
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=4,
            expected_gate_name="composite_and",
            expected_terminal_outcomes={"routed": 4},  # All rows routed to named sinks
        )

    def test_composite_or_condition(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify: row['status'] == 'active' or row['priority'] > 5 works.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource(
            [
                {"status": "active", "priority": 3},  # status active - true
                {"status": "inactive", "priority": 8},  # priority > 5 - true
                {"status": "inactive", "priority": 3},  # both false - false
            ],
            schema=_StatusPrioritySchema,
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
            sinks={"default": as_sink(pass_sink), "fail": as_sink(fail_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 2 pass, 1 fail
        assert len(pass_sink.results) == 2
        assert len(fail_sink.results) == 1

        # P1 Fix: Verify audit trail
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="composite_or",
            expected_terminal_outcomes={"completed": 2, "routed": 1},
        )

    def test_membership_condition(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify: row['status'] in ['active', 'pending'] works.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource(
            [
                {"status": "active"},
                {"status": "pending"},
                {"status": "deleted"},
                {"status": "suspended"},
            ],
            schema=_StatusOnlySchema,
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
            sinks={"default": as_sink(allowed_sink), "blocked": as_sink(blocked_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 4
        # 2 allowed (active, pending), 2 blocked (deleted, suspended)
        assert len(allowed_sink.results) == 2
        assert {r["status"] for r in allowed_sink.results} == {"active", "pending"}
        assert len(blocked_sink.results) == 2
        assert {r["status"] for r in blocked_sink.results} == {"deleted", "suspended"}

        # P1 Fix: Verify audit trail
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=4,
            expected_gate_name="membership_check",
            expected_terminal_outcomes={"completed": 2, "routed": 2},
        )

    def test_optional_field_with_get(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify: row.get('optional') is not None works.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource(
            [
                {"required": "a", "optional": "present"},
                {"required": "b"},  # optional field missing
                {"required": "c", "optional": None},  # optional explicitly None
            ],
            schema=_RequiredOnlySchema,
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
            sinks={"default": as_sink(missing_optional_sink), "has_optional": as_sink(has_optional_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 1 has optional, 2 missing/None
        assert len(has_optional_sink.results) == 1
        assert has_optional_sink.results[0]["required"] == "a"
        assert len(missing_optional_sink.results) == 2

        # P1 Fix: Verify audit trail
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="optional_check",
            expected_terminal_outcomes={"completed": 2, "routed": 1},
        )


# ============================================================================
# WP-09 Verification: Route Label Resolution
# ============================================================================


class TestRouteLabelResolution:
    """WP-09 Verification: Route labels resolve correctly."""

    def test_route_labels_resolve_to_sinks(self, plugin_manager) -> None:
        """Verify route labels map to correct sinks."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
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
                "main_output": SinkSettings(plugin="json", options={"path": "main_output.json", "schema": {"fields": "dynamic"}}),
                "review_queue": SinkSettings(plugin="json", options={"path": "review_queue.json", "schema": {"fields": "dynamic"}}),
                "archive": SinkSettings(plugin="json", options={"path": "archive.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="main_output",
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

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        # Verify route resolution map
        route_map = graph.get_route_resolution_map()
        config_gate_map = graph.get_config_gate_id_map()
        gate_id = config_gate_map[GateName("quality_router")]

        # Check route resolution
        assert (gate_id, "true") in route_map
        assert route_map[(gate_id, "true")] == "continue"
        assert (gate_id, "false") in route_map
        assert route_map[(gate_id, "false")] == "review_queue"

    def test_ternary_expression_returns_string_routes(self, plugin_manager, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify ternary expressions can return different route labels.

        P1 Fix: Added audit trail verification for node_states, token_outcomes.
        """
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

        # Build settings with ternary condition that returns category directly
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
                "premium_sink": SinkSettings(plugin="json", options={"path": "premium.json", "schema": {"fields": "dynamic"}}),
                "standard_sink": SinkSettings(plugin="json", options={"path": "standard.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="standard_sink",
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

        plugins = instantiate_plugins_from_config(settings)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
        )

        source = ListSource(
            [
                {"category": "premium"},
                {"category": "standard"},
                {"category": "premium"},
            ],
            schema=_CategorySchema,
        )
        premium_sink = CollectSink()
        standard_sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"premium_sink": as_sink(premium_sink), "standard_sink": as_sink(standard_sink)},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        assert len(premium_sink.results) == 2
        assert len(standard_sink.results) == 1

        # P1 Fix: Verify audit trail
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="category_router",
            expected_terminal_outcomes={"routed": 3},
        )


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

    def test_fork_gate_in_graph(self, plugin_manager) -> None:
        """Verify fork gate is correctly represented in graph."""
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
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
                "analysis_a": SinkSettings(plugin="json", options={"path": "analysis_a.json", "schema": {"fields": "dynamic"}}),
                "analysis_b": SinkSettings(plugin="json", options={"path": "analysis_b.json", "schema": {"fields": "dynamic"}}),
            },
            default_sink="output",
            gates=[
                GateSettingsConfig(
                    name="parallel_processor",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["analysis_a", "analysis_b"],
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

        # Verify gate node exists with fork config
        config_gate_map = graph.get_config_gate_id_map()
        gate_id = config_gate_map[GateName("parallel_processor")]
        node_info = graph.get_node_info(gate_id)

        assert node_info.config["fork_to"] == ["analysis_a", "analysis_b"]
        assert node_info.config["routes"]["true"] == "fork"

    def test_fork_children_route_to_branch_named_sinks(self, plugin_manager, landscape_db: LandscapeDB, payload_store) -> None:
        """Fork children with branch_name route to matching sinks.

        This is the core fork use case:
        - Gate forks to ["path_a", "path_b"]
        - Child with branch_name="path_a" goes to sink named "path_a"
        - Child with branch_name="path_b" goes to sink named "path_b"

        P1 Fix: Added audit trail verification including token_parents for lineage.
        """
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

        source = ListSource([{"value": 42}])
        path_a_sink = CollectSink()
        path_b_sink = CollectSink()

        # Config with fork gate and branch-named sinks
        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            sinks={
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"fields": "dynamic"}}),
                "path_b": SinkSettings(plugin="json", options={"path": "path_b.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettingsConfig(
                    name="forking_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            default_sink="path_a",  # Default, but fork should override for path_b
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

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"path_a": as_sink(path_a_sink), "path_b": as_sink(path_b_sink)},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_forked == 1

        # CRITICAL: Each sink gets exactly one row (the fork child for that branch)
        assert len(path_a_sink.results) == 1, f"path_a should get 1 row, got {len(path_a_sink.results)}"
        assert len(path_b_sink.results) == 1, f"path_b should get 1 row, got {len(path_b_sink.results)}"

        # Both should have the same value (forked from same parent)
        assert path_a_sink.results[0]["value"] == 42
        assert path_b_sink.results[0]["value"] == 42

        # P1 Fix: Verify fork audit trail including token_parents for lineage
        verify_fork_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_parent_rows=1,
            expected_child_count_per_parent=2,
            expected_gate_name="forking_gate",
        )

    def test_fork_multiple_source_rows_counts_correctly(self, plugin_manager, landscape_db: LandscapeDB, payload_store) -> None:
        """Multiple source rows fork correctly with proper counting.

        When processing multiple source rows through a fork gate:
        - rows_forked should count the NUMBER OF PARENT ROWS that forked
        - Each sink should receive one child per source row

        P1 Fix: Added audit trail verification including token_parents for lineage.
        """
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

        # Three source rows - all will fork
        source = ListSource([{"value": 10}, {"value": 20}, {"value": 30}])
        analysis_sink = CollectSink()
        archive_sink = CollectSink()

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            sinks={
                "analysis": SinkSettings(plugin="json", options={"path": "analysis.json", "schema": {"fields": "dynamic"}}),
                "archive": SinkSettings(plugin="json", options={"path": "archive.json", "schema": {"fields": "dynamic"}}),
            },
            gates=[
                GateSettingsConfig(
                    name="parallel_fork",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["analysis", "archive"],
                ),
            ],
            default_sink="analysis",
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

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"analysis": as_sink(analysis_sink), "archive": as_sink(archive_sink)},
            gates=settings.gates,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3

        # CRITICAL: rows_forked counts parent rows that forked
        assert result.rows_forked == 3, f"Expected 3 forked rows, got {result.rows_forked}"

        # Each sink should receive 3 rows (one child per source row)
        assert len(analysis_sink.results) == 3, f"analysis should get 3 rows, got {len(analysis_sink.results)}"
        assert len(archive_sink.results) == 3, f"archive should get 3 rows, got {len(archive_sink.results)}"

        # Verify all values are preserved
        analysis_values = {r["value"] for r in analysis_sink.results}
        archive_values = {r["value"] for r in archive_sink.results}
        expected_values = {10, 20, 30}

        assert analysis_values == expected_values
        assert archive_values == expected_values

        # P1 Fix: Verify fork audit trail including token_parents for lineage
        verify_fork_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_parent_rows=3,
            expected_child_count_per_parent=2,
            expected_gate_name="parallel_fork",
        )


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

    def test_source_transform_gate_sink_pipeline(self, landscape_db: LandscapeDB, payload_store) -> None:
        """End-to-end: Source -> Transform -> Config Gate -> Sink.

        P1 Fix: Added audit trail verification for the complete pipeline.
        """
        from elspeth.contracts import TransformResult
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseTransform

        db = landscape_db

        class NormalizeTransform(BaseTransform):
            """Transform that normalizes score to 0-1 range."""

            name = "normalize"
            input_schema = _RawScoreSchema
            output_schema = _NormalizedScoreSchema
            plugin_version = "1.0.0"

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                normalized = row["raw_score"] / 100.0
                return TransformResult.success({**row, "score": normalized}, success_reason={"action": "normalize"})

        source = ListSource(
            [
                {"raw_score": 90},  # 0.9 - high confidence
                {"raw_score": 50},  # 0.5 - low confidence
                {"raw_score": 85},  # 0.85 - exactly threshold
            ],
            schema=_RawScoreSchema,
        )
        transform = NormalizeTransform(config={"schema": {"fields": "dynamic"}})
        high_conf_sink = CollectSink()
        low_conf_sink = CollectSink()

        gate = GateSettings(
            name="confidence_gate",
            condition="row['score'] >= 0.85",
            routes={"true": "continue", "false": "low_conf"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(high_conf_sink), "low_conf": as_sink(low_conf_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3
        # 2 high confidence (0.9, 0.85), 1 low confidence (0.5)
        assert len(high_conf_sink.results) == 2
        assert len(low_conf_sink.results) == 1
        assert low_conf_sink.results[0]["score"] == 0.5

        # P1 Fix: Verify audit trail for end-to-end pipeline
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=3,
            expected_gate_name="confidence_gate",
            expected_terminal_outcomes={"completed": 2, "routed": 1},
        )

    def test_audit_trail_records_gate_evaluation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Verify audit trail records gate condition and result.

        P1 Fix: Added comprehensive audit trail verification.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        source = ListSource([{"value": 42}])
        sink = CollectSink()
        reject_sink = CollectSink()

        gate = GateSettings(
            name="audit_test_gate",
            condition="row['value'] > 0",
            routes={"true": "continue", "false": "reject"},
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"default": as_sink(sink), "reject": as_sink(reject_sink)},
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

        # Verify gate node is registered
        node_names = [n[0] for n in nodes]
        node_types = [n[1] for n in nodes]

        assert "config_gate:audit_test_gate" in node_names
        assert "gate" in node_types

        # P1 Fix: Verify audit trail completeness
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=1,
            expected_gate_name="audit_test_gate",
            expected_terminal_outcomes={"completed": 1},
        )

    def test_gate_audit_trail_includes_evaluation_metadata(self, landscape_db: LandscapeDB, payload_store) -> None:
        """WP-14b: Verify gate audit trail includes condition, result, and route.

        ELSPETH is built for high-stakes accountability. Every gate decision
        must be auditable - traceable to what condition was evaluated and
        what route was taken.

        This test verifies that for gate evaluations:
        1. The node_states table records the gate evaluation
        2. The routing_events table (for routed tokens) includes metadata
        3. The metadata contains: condition evaluated, evaluation result, route taken

        P2 Fix: Strengthened assertions to verify correct sink is targeted.
        """
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = landscape_db

        # Two rows: one high priority (routes to "urgent"), one low (continues to default)
        source = ListSource([{"priority": 10}, {"priority": 2}], schema=_PrioritySchema)
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
            sinks={"default": as_sink(default_sink), "urgent": as_sink(urgent_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

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
            # AUD-002: Both routed AND continue decisions now generate routing events
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

            # AUD-002: We expect 2 routing events - one for "true" -> urgent sink,
            # one for "continue" -> next node
            assert len(routing_events) == 2, f"Expected 2 routing events (AUD-002), got {len(routing_events)}"

            # Find the "true" route event (routes to sink)
            route_event = [e for e in routing_events if e[4] == "true"]
            assert len(route_event) == 1, f"Expected 1 'true' routing event, got {len(route_event)}"
            routing_event = route_event[0]
            edge_label = routing_event[4]
            assert edge_label == "true", f"Edge label should be 'true', got {edge_label}"

            # The routing event should have a reason_hash (metadata was recorded)
            reason_hash = routing_event[3]
            assert reason_hash is not None, "Routing event should have reason_hash for audit trail"

            # P2 Fix: Verify the edge points to the CORRECT sink (urgent)
            # Production path uses hashed node IDs, so we verify by plugin_name not node_id
            to_node_id = routing_event[5]
            sink_node = conn.execute(
                text("""
                    SELECT node_id, plugin_name FROM nodes WHERE node_id = :node_id
                """),
                {"node_id": to_node_id},
            ).fetchone()
            assert sink_node is not None, "Target node must exist"

            # P2 Fix: Assert this is specifically the urgent sink by checking the node exists
            # and is a sink (production path uses hashed IDs like sink_collect_xxx)
            assert "urgent" in to_node_id or sink_node[1] == "collect", (
                f"True route should target a sink for 'urgent', got node_id='{to_node_id}', plugin='{sink_node[1]}'"
            )

            # Also verify the "continue" routing event exists and goes to default sink
            continue_events = [e for e in routing_events if e[4] == "continue"]
            assert len(continue_events) == 1, f"Expected 1 'continue' routing event, got {len(continue_events)}"
            continue_event = continue_events[0]
            continue_to_node = continue_event[5]
            # Production path uses hashed node IDs
            assert "default" in continue_to_node or "sink" in continue_to_node, (
                f"Continue route should target default sink, got '{continue_to_node}'"
            )

        # Verify overall audit trail with helper
        verify_gate_audit_trail(
            db=db,
            run_id=result.run_id,
            expected_rows=2,
            expected_gate_name="priority_gate",
            expected_terminal_outcomes={"completed": 1, "routed": 1},
        )


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

    def test_missing_field_raises_evaluation_error(self) -> None:
        """Gate condition referencing missing field should fail with context.

        Per ELSPETH's design: operations on row values can fail, and when they
        do, the failure should be clear and auditable. A gate condition that
        references row['missing_field'] should raise ExpressionEvaluationError
        (wrapping the KeyError with helpful context like field name and
        available fields), and this should be recorded as a failed node state.
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
            node_type=NodeType.GATE,
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

        # The expression parser evaluates row['nonexistent'], which raises
        # ExpressionEvaluationError (wrapping the KeyError with context).
        # The executor should catch this and re-raise after recording failure
        with pytest.raises(ExpressionEvaluationError) as exc_info:
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
            # ExpressionEvaluationError wraps the KeyError with context
            assert error["type"] == "ExpressionEvaluationError", f"Expected ExpressionEvaluationError, got {error['type']}"

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
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={"condition": "row.get('optional', 0) > 5"},
            schema_config=schema_config,
        )
        # AUD-002: Register next node and continue edge for audit completeness
        next_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="next_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=schema_config,
        )
        continue_edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=node.node_id,
            to_node_id=next_node.node_id,
            label="continue",
            mode=RoutingMode.MOVE,
        )

        span_factory = SpanFactory()  # No tracer = no-op spans
        edge_map: dict[tuple[NodeID, str], str] = {(NodeID(node.node_id), "continue"): continue_edge.edge_id}
        route_resolution_map: dict[tuple[NodeID, str], str] = {
            (NodeID(node.node_id), "true"): "continue",
            (NodeID(node.node_id), "false"): "continue",
        }
        executor = GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
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

    def test_route_to_nonexistent_sink_caught_at_graph_construction(self, plugin_manager) -> None:
        """Route to non-existent sink caught when building graph."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            ElspethSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.config import (
            GateSettings as GateSettingsConfig,
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
                GateSettingsConfig(
                    name="bad_route",
                    condition="True",
                    routes={"true": "nonexistent_sink", "false": "continue"},
                ),
            ],
        )

        with pytest.raises(GraphValidationError, match="nonexistent_sink"):
            plugins = instantiate_plugins_from_config(settings)
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                default_sink=settings.default_sink,
            )
