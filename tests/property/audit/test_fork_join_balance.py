# tests/property/audit/test_fork_join_balance.py
"""Property-based tests for fork-join balance invariants.

FORK-JOIN BALANCE INVARIANT:
Every fork branch must have a destination, and every fork child must have
a parent link recorded in the audit trail.

This ensures:
1. No "orphan" branches that tokens disappear into
2. Complete lineage tracking for forked tokens
3. DAG construction rejects invalid fork configurations

Fork terminology:
- Fork gate: A gate that splits one token into multiple child tokens
- Branch: A named path from a fork (e.g., "path_a", "path_b")
- Coalesce: A merge point that joins tokens from multiple branches
- Parent token: The token that was forked
- Child tokens: The new tokens created, one per branch
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import bindparam, text

from elspeth.contracts import CoalesceName, GateName, RoutingAction, RoutingMode, SinkName
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, Determinism, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.run_result import RunResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.checkpoint.serialization import checkpoint_loads
from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import (
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.plugins import (
    CollectSink,
    ListSource,
    PassTransform,
)
from tests.fixtures.stores import MockPayloadStore

# =============================================================================
# Audit Verification Helpers
# =============================================================================


def count_fork_children_missing_parents(db: LandscapeDB, run_id: str) -> int:
    """Count fork children that lack parent links.

    This is a critical invariant: every fork child token must have a
    token_parents record linking it to the parent token.
    """
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                LEFT JOIN token_parents p ON p.token_id = t.token_id
                WHERE r.run_id = :run_id
                  AND t.fork_group_id IS NOT NULL
                  AND p.token_id IS NULL
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def count_forked_outcomes(db: LandscapeDB, run_id: str) -> int:
    """Count tokens with FORKED outcome (parent tokens that were split)."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                  AND o.outcome = 'transient'
                  AND o.path = 'fork_parent'
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def get_fork_group_stats(db: LandscapeDB, run_id: str) -> dict[str, int]:
    """Get statistics about fork groups.

    Returns dict with:
    - total_fork_groups: Number of unique fork groups
    - total_fork_children: Number of fork child tokens
    - children_with_parents: Fork children that have parent links
    """
    with db.connection() as conn:
        # Count unique fork groups
        total_groups = (
            conn.execute(
                text("""
                SELECT COUNT(DISTINCT t.fork_group_id)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                  AND t.fork_group_id IS NOT NULL
            """),
                {"run_id": run_id},
            ).scalar()
            or 0
        )

        # Count fork children
        total_children = (
            conn.execute(
                text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                  AND t.fork_group_id IS NOT NULL
            """),
                {"run_id": run_id},
            ).scalar()
            or 0
        )

        # Count children with parents
        with_parents = (
            conn.execute(
                text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                JOIN token_parents p ON p.token_id = t.token_id
                WHERE r.run_id = :run_id
                  AND t.fork_group_id IS NOT NULL
            """),
                {"run_id": run_id},
            ).scalar()
            or 0
        )

        return {
            "total_fork_groups": total_groups,
            "total_fork_children": total_children,
            "children_with_parents": with_parents,
        }


def orphan_leaf_token_ids(db: LandscapeDB, run_id: str) -> list[str]:
    """Return token_ids of non-delegation leaf tokens lacking a completed terminal outcome.

    A leaf token is any token whose own outcome is NOT a delegation marker
    (FORK_PARENT / EXPAND_PARENT). After a fully-resumed run, every such token
    must carry exactly one completed=1 outcome — an empty list means no orphans.
    Returning the ids (not just a count) keeps a future orphan-regression failure
    diagnosable: the assertion can name the offending tokens.
    """
    delegation = (TerminalPath.FORK_PARENT.value, TerminalPath.EXPAND_PARENT.value)
    with db.connection() as conn:
        rows = conn.execute(
            text("""
                SELECT t.token_id AS token_id
                FROM tokens t
                WHERE t.run_id = :run_id
                  AND t.token_id NOT IN (
                      SELECT o.token_id FROM token_outcomes o
                      WHERE o.run_id = :run_id AND o.path IN (:fp, :ep)
                  )
                  AND t.token_id NOT IN (
                      SELECT o.token_id FROM token_outcomes o
                      WHERE o.run_id = :run_id AND o.completed = 1
                            AND o.path NOT IN (:fp, :ep)
                  )
            """),
            {"run_id": run_id, "fp": delegation[0], "ep": delegation[1]},
        ).fetchall()
    return [row.token_id for row in rows]


def count_fork_groups_with_unexpected_children(db: LandscapeDB, run_id: str, expected_children: int) -> int:
    """Count fork groups that don't have the expected number of children."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM (
                    SELECT t.fork_group_id, COUNT(*) AS child_count
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.fork_group_id IS NOT NULL
                    GROUP BY t.fork_group_id
                    HAVING COUNT(*) != :expected_children
                ) bad_groups
            """),
            {"run_id": run_id, "expected_children": expected_children},
        ).scalar()
        return result or 0


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for row values
row_for_fork = st.fixed_dictionaries(
    {"value": st.integers(min_value=0, max_value=1000)},
)


# =============================================================================
# Helper: Build production graph from PipelineConfig
# =============================================================================


def _build_production_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build graph using production code path (from_plugin_instances).

    Replacement for v1 build_production_graph, inlined to avoid v1 imports.
    Auto-sets on_success on terminal transform for linear pipelines.
    """
    transforms = list(config.transforms)
    sink_name = next(iter(config.sinks))
    source_on_success = "source_out" if transforms else sink_name
    wired_transforms = (
        wire_transforms(
            transforms,
            source_connection=source_on_success,
            final_sink=sink_name,
        )
        if transforms
        else []
    )

    return ExecutionGraph.from_plugin_instances(
        source=config.source,
        source_settings=SourceSettings(plugin=config.source.name, on_success=source_on_success, options={}),
        transforms=wired_transforms,
        sinks=config.sinks,
        aggregations={},
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce_settings) if config.coalesce_settings else None,
    )


# =============================================================================
# Property Tests: DAG Construction Invariants
# =============================================================================


class TestDagForkBranchValidation:
    """Property tests for DAG-level fork branch validation.

    These test that ExecutionGraph.from_plugin_instances() correctly
    validates fork configurations at construction time.
    """

    def test_fork_to_unknown_destination_rejected(self) -> None:
        """Fork branch to non-existent destination is rejected at DAG construction.

        This is a critical safety check - typos in branch names would otherwise
        cause tokens to disappear silently.
        """
        # Create a gate config that forks to a branch that doesn't exist
        gate = GateSettings(
            name="bad_fork_gate",
            input="gate_in",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "default"},  # Route to fork action
            fork_to=["unknown_branch"],  # No coalesce or sink with this name
        )

        source = ListSource([{"value": 1}], on_success="default")
        sink = CollectSink()

        # This should fail at graph construction
        with pytest.raises(GraphValidationError, match="unknown_branch"):
            ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
                transforms=[],
                sinks={"default": as_sink(sink)},  # No "unknown_branch" sink
                gates=[gate],
                aggregations={},
                coalesce_settings=[],  # No coalesce with "unknown_branch"
            )

    def test_fork_to_sink_is_valid(self) -> None:
        """Fork branch targeting a sink is accepted."""
        gate = GateSettings(
            name="fork_to_sink_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        source = ListSource([{"value": 1}], on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # This should succeed - branches match sink names
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )

        gate_id = graph.get_config_gate_id_map()[GateName(gate.name)]
        sink_ids = graph.get_sink_id_map()
        edges = graph.get_edges()

        def has_fork_edge(branch: str, sink_name: str) -> bool:
            sink_id = sink_ids[SinkName(sink_name)]
            return any(
                edge.from_node == gate_id and edge.to_node == sink_id and edge.label == branch and edge.mode == RoutingMode.COPY
                for edge in edges
            )

        assert has_fork_edge("sink_a", "sink_a")
        assert has_fork_edge("sink_b", "sink_b")

    def test_fork_to_coalesce_is_valid(self) -> None:
        """Fork branch targeting a coalesce is accepted."""
        gate = GateSettings(
            name="fork_to_coalesce_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            on_success="default",
        )

        source = ListSource([{"value": 1}], on_success="default")
        sink = CollectSink()

        # This should succeed - branches match coalesce branches
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"default": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        branch_map = graph.get_branch_to_coalesce_map()
        assert branch_map == {"branch_a": "merge_point", "branch_b": "merge_point"}

        gate_id = graph.get_config_gate_id_map()[GateName(gate.name)]
        coalesce_id = graph.get_coalesce_id_map()[CoalesceName(coalesce.name)]
        edges = graph.get_edges()

        def has_fork_edge(branch: str) -> bool:
            return any(
                edge.from_node == gate_id and edge.to_node == coalesce_id and edge.label == branch and edge.mode == RoutingMode.COPY
                for edge in edges
            )

        assert has_fork_edge("branch_a")
        assert has_fork_edge("branch_b")

    def test_duplicate_fork_branches_rejected(self) -> None:
        """Fork with duplicate branch names is rejected."""
        gate = GateSettings(
            name="dup_fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["branch_a", "branch_a"],  # Duplicate!
        )

        source = ListSource([{"value": 1}], on_success="default")
        sink = CollectSink()

        # RoutingAction.fork_to_paths() validates uniqueness
        with pytest.raises((GraphValidationError, ValueError), match=r"[Dd]uplicate"):
            ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
                transforms=[],
                sinks={"default": as_sink(sink)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[],
            )

    def test_coalesce_branch_not_produced_rejected(self) -> None:
        """Coalesce expecting a branch that no gate produces is rejected."""
        # Gate only produces branch_a, but coalesce expects both
        gate = GateSettings(
            name="partial_fork",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "default"},
            fork_to=["branch_a"],  # Only one branch
        )

        coalesce = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],  # Expects branch_b too!
        )

        source = ListSource([{"value": 1}], on_success="default")
        sink = CollectSink()

        with pytest.raises(GraphValidationError, match="branch_b"):
            ExecutionGraph.from_plugin_instances(
                source=as_source(source),
                source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
                transforms=[],
                sinks={"default": as_sink(sink)},
                gates=[gate],
                aggregations={},
                coalesce_settings=[coalesce],
            )


class TestForkJoinRuntimeBalance:
    """Property tests for runtime fork-join balance.

    These test that when forks execute, the audit trail correctly
    records parent-child relationships.
    """

    @given(n_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_fork_to_sinks_all_children_have_parents(self, n_rows: int) -> None:
        """Property: When forking to sinks, all child tokens have parent links.

        This tests the simpler fork case (no coalesce) to verify parent
        link recording works correctly.
        """
        from elspeth.core.config import ElspethSettings

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        rows = [{"value": i} for i in range(n_rows)]
        source = ListSource(rows, on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # Gate that forks all rows to both sinks
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )

        # Settings needed for fork execution
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # Verify fork audit integrity
        missing_parents = count_fork_children_missing_parents(db, run.run_id)
        assert missing_parents == 0, (
            f"FORK AUDIT VIOLATION: {missing_parents} fork children missing parent links. Rows: {n_rows}. Fork lineage would be incomplete."
        )

        # Verify FORKED outcomes recorded for parent tokens
        forked_count = count_forked_outcomes(db, run.run_id)
        assert forked_count == n_rows, f"Expected {n_rows} FORKED outcomes (one per parent token), got {forked_count}"

        # Verify fork statistics
        stats = get_fork_group_stats(db, run.run_id)
        expected_children_per_group = len(gate.fork_to or [])
        expected_children_total = n_rows * expected_children_per_group
        assert stats["total_fork_children"] == stats["children_with_parents"], (
            f"Not all fork children have parents: {stats['children_with_parents']}/{stats['total_fork_children']}"
        )
        assert stats["total_fork_children"] == expected_children_total, (
            f"Expected {expected_children_total} fork children (rows={n_rows}, branches={expected_children_per_group}), "
            f"got {stats['total_fork_children']}."
        )
        assert stats["total_fork_groups"] == n_rows, (
            f"Expected {n_rows} fork groups (one per parent token), got {stats['total_fork_groups']}."
        )
        bad_groups = count_fork_groups_with_unexpected_children(db, run.run_id, expected_children=expected_children_per_group)
        assert bad_groups == 0, f"{bad_groups} fork groups have unexpected child counts."


class TestForkJoinEnumProperties:
    """Property tests for fork-related enums and outcomes."""

    def test_fork_parent_is_terminal_pair(self) -> None:
        """FORK_PARENT is a terminal pair (parent token's journey ends)."""
        assert (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT) in _LEGAL_TERMINAL_PAIRS

    def test_coalesced_is_terminal_pair(self) -> None:
        """COALESCED is a terminal pair (branch tokens merge)."""
        assert (TerminalOutcome.SUCCESS, TerminalPath.COALESCED) in _LEGAL_TERMINAL_PAIRS

    def test_routing_action_fork_requires_paths(self) -> None:
        """RoutingAction.fork_to_paths() requires at least one path."""
        with pytest.raises(ValueError, match="at least one"):
            RoutingAction.fork_to_paths([])

    def test_routing_action_fork_rejects_duplicates(self) -> None:
        """RoutingAction.fork_to_paths() rejects duplicate paths."""
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            RoutingAction.fork_to_paths(["a", "b", "a"])


class TestForkJoinEdgeCases:
    """Edge case tests for fork-join behavior."""

    def test_no_fork_no_fork_groups(self) -> None:
        """Pipeline without forks should have no fork groups."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = PassTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

        stats = get_fork_group_stats(db, run.run_id)
        assert stats["total_fork_groups"] == 0
        assert stats["total_fork_children"] == 0

    def test_empty_source_no_fork_issues(self) -> None:
        """Empty source with fork config should not cause issues."""
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([], on_success="sink_a")  # Empty
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )

        from elspeth.core.config import ElspethSettings

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # No rows means no forks
        stats = get_fork_group_stats(db, run.run_id)
        assert stats["total_fork_groups"] == 0
        missing = count_fork_children_missing_parents(db, run.run_id)
        assert missing == 0


class TestForkRecoveryInvariant:
    """Property tests for recovery invariant with forked tokens.

    These tests verify that the recovery system correctly detects partial
    fork completion. Bug P2-2026-01-29-recovery-skips-partial-forks showed
    that recovery could miss rows where only some fork children completed.
    """

    @given(n_rows=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_partial_fork_detected_by_recovery(self, n_rows: int) -> None:
        """Property: Recovery detects rows with incomplete forks.

        For any row that forks, if we simulate a crash after partial
        completion (by deleting one child's outcome), recovery must
        identify the row as unprocessed.

        This tests the fix for P2-2026-01-29-recovery-skips-partial-forks.
        """
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        rows = [{"value": i} for i in range(n_rows)]
        source = ListSource(rows, on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # Gate that forks all rows to both sinks
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)

        # Pipeline completed successfully - all rows processed
        # Now simulate partial failure by deleting ONE child outcome per row

        # Get tokens that went to sink_a (one branch of the fork)
        with db.engine.connect() as conn:
            sink_a_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id, t.row_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND o.sink_name = 'sink_a'
                """),
                {"run_id": run.run_id},
            ).fetchall()

            # Delete one branch's outcomes to simulate partial fork completion
            for outcome in sink_a_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        # Create a checkpoint (required for recovery to work)
        # Use actual token and sink node from the run
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            # Get an actual token from the run
            actual_token = conn.execute(
                text("""
                    SELECT t.token_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                    LIMIT 1
                """),
                {"run_id": run.run_id},
            ).fetchone()
            # Token must exist since run completed successfully
            assert actual_token is not None, "No tokens found for run"
            token_id = actual_token.token_id

        checkpoint_manager = CheckpointManager(db)
        checkpoint_manager.create_checkpoint(
            run_id=run.run_id,
            token_id=token_id,  # Use actual token from run
            node_id=sink_node_ids[0],  # Use actual sink node ID from graph
            sequence_number=1,
            graph=graph,
        )

        # Mark run as failed (required for recovery)
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run.run_id},
            )
            conn.commit()

        # Now test recovery - it should find all rows as unprocessed
        recovery_manager = RecoveryManager(db, checkpoint_manager)
        unprocessed = recovery_manager.get_unprocessed_rows(run.run_id)

        # PROPERTY: When fork is partial (one child outcome deleted),
        # ALL rows should appear in unprocessed list
        assert len(unprocessed) == n_rows, (
            f"RECOVERY INVARIANT VIOLATED: Expected {n_rows} unprocessed rows "
            f"(all have partial fork completion), got {len(unprocessed)}. "
            f"Recovery is incorrectly marking partially-completed forks as done."
        )

    def test_resume_does_not_reemit_completed_fork_branch(self) -> None:
        """Resuming a partial-fork row must not re-emit its completed branch.

        Reproduction of the resume double-emission defect (F1):
        a checkpoint-enabled fork pipeline is interrupted after one branch's
        sink write (and checkpoint) but before the sibling branch completes.
        Recovery correctly *detects* the row as unprocessed
        (``test_partial_fork_detected_by_recovery`` covers that), but
        ``process_existing_row`` then restarts the row from the source and
        re-forks to ALL branches — re-emitting the branch that already reached
        a terminal state. The audit trail (the legal record) then holds two
        terminal outcomes for the surviving branch under one ``row_id``, and the
        sink is written twice.

        INVARIANT: after resume, each (row_id, sink_name) has exactly ONE
        completed, non-delegation terminal outcome. The defect yields two for
        the branch that completed before the interruption.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings, ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        # Single row keeps the reproduction deterministic.
        source = ListSource([{"value": 1}], on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # Gate forks the row to both sinks → two terminal sink outcomes per row.
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # Baseline: the one row forked to both sinks — exactly one terminal
        # outcome per (row, sink).
        def _outcome_counts() -> dict[tuple[str, str], int]:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND o.completed = 1
                          AND o.sink_name IS NOT NULL
                        GROUP BY t.row_id, o.sink_name
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return {(row.row_id, row.sink_name): row.n for row in result}

        baseline = _outcome_counts()
        baseline_fork_stats = get_fork_group_stats(db, run_id)
        assert sorted(k[1] for k in baseline) == ["sink_a", "sink_b"], baseline
        assert all(n == 1 for n in baseline.values()), baseline

        # Interrupt the sink_a branch: delete its terminal outcome so the row
        # has one completed branch (sink_b) and one incomplete branch (sink_a).
        with db.engine.connect() as conn:
            sink_a_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.sink_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
            assert sink_a_outcomes, "expected a sink_a outcome to delete"
            for outcome in sink_a_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        # A checkpoint is the precondition for resume. Anchor it on a real token
        # and the sink node from the run.
        checkpoint_mgr = CheckpointManager(db)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )

        # Mark the run failed so it is resumable.
        with db.engine.connect() as conn:
            conn.execute(text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"), {"run_id": run_id})
            conn.commit()

        # Resume through the production path.
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # CONSERVATION LAW (spec ADDENDUM 1 test oracle):
        #  (a) multiset of completed (row_id, sink_name) outcomes == uninterrupted baseline
        #  (b) zero orphan leaf tokens (never-zero)
        #  (c) no (row_id, sink_name) carries two outcomes (never-two)
        #  (d) the resume result reports COMPLETED
        #  (e) the surviving branch's sink was physically written exactly once
        #  (f) fork-group shape is unchanged (guards against the Approach-2 double-parent)
        after = _outcome_counts()
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left {len(orphans)} non-delegation leaf token(s) with no terminal outcome (orphans): {orphans}"
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status
        assert len(sink_b.results) == 1, (
            f"sink_b (completed before interruption) was physically written {len(sink_b.results)} times; resume must not re-write it."
        )
        post_resume_stats = get_fork_group_stats(db, run_id)
        assert post_resume_stats["total_fork_groups"] == baseline_fork_stats["total_fork_groups"], (
            f"Resume changed fork-group shape: {baseline_fork_stats} -> {post_resume_stats}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # F1 Regression Cells (Task 8) — collision + provenance attributability
    # ─────────────────────────────────────────────────────────────────────

    def _setup_fork_and_interrupt(
        self,
    ) -> tuple[
        LandscapeDB,
        MockPayloadStore,
        PipelineConfig,
        ExecutionGraph,
        ElspethSettings,
        str,  # run_id
        str,  # incomplete_token_id (sink_a child)
    ]:
        """Shared setup: fork pipeline run → interrupt sink_a → return artifacts.

        Runs the fork pipeline to completion, then deletes the sink_a child's
        terminal OUTCOME (leaving its node_state row intact). Returns the
        artifacts needed by each regression cell.
        """
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 1}], on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # Locate the sink_a child token (the incomplete branch after interruption).
        with db.engine.connect() as conn:
            sink_a_tokens = conn.execute(
                text("""
                    SELECT t.token_id AS token_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.branch_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(sink_a_tokens) == 1, f"Expected exactly one sink_a fork-child token, got {len(sink_a_tokens)}"
        incomplete_token_id = sink_a_tokens[0].token_id

        # Interrupt: delete sink_a's terminal OUTCOME only. The node_state row for
        # the sink_a child at attempt=0 is NOT deleted — this is the run-1 record
        # that Cell 1 asserts is preserved after resume (append-only invariant).
        with db.engine.connect() as conn:
            sink_a_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.sink_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
            assert sink_a_outcomes, "expected a sink_a outcome to delete (setup precondition)"
            for outcome in sink_a_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        return db, payload_store, config, graph, settings_obj, run_id, incomplete_token_id

    def _resume_run(
        self,
        db: LandscapeDB,
        payload_store: MockPayloadStore,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings_obj: ElspethSettings,
        run_id: str,
    ) -> None:
        """Complete the resume path: checkpoint + mark-failed + resume."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings

        checkpoint_mgr = CheckpointManager(db)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

    def test_resume_fork_sink_coexists_with_run1_node_state(self) -> None:
        """Re-driving an incomplete fork->sink child must not collide on node_states.

        run-1 left a node_state at attempt=0 for the incomplete child (the
        sink ran; only its terminal OUTCOME was deleted to simulate the
        interruption — the node_state row remains). The re-drive must write
        at attempt=1, and the run-1 record must be preserved (append-only).

        RED (dispatch disabled): AssertionError — ``assert 1 in attempts``
        fails because process_existing_row mints a fresh token and never
        touches the original incomplete_token_id, so that token's node_states
        remain at ``[0]`` only; the re-drive record at attempt=1 is never
        written.
        """
        db, payload_store, config, graph, settings_obj, run_id, incomplete_token_id = self._setup_fork_and_interrupt()

        # Precondition: the run-1 attempt-0 node_state must already exist for
        # the incomplete child BEFORE resume. The setup deletes only the outcome,
        # not the node_state.
        with db.engine.connect() as conn:
            pre_resume_attempts = [
                r.attempt
                for r in conn.execute(
                    text("SELECT attempt FROM node_states WHERE token_id = :tid ORDER BY attempt"),
                    {"tid": incomplete_token_id},
                ).fetchall()
            ]
        assert 0 in pre_resume_attempts, (
            f"Setup precondition violated: run-1 attempt=0 node_state absent before resume "
            f"(attempts={pre_resume_attempts!r}). The outcome-delete must NOT remove node_states."
        )

        self._resume_run(db, payload_store, config, graph, settings_obj, run_id)

        with db.engine.connect() as conn:
            attempts = [
                r.attempt
                for r in conn.execute(
                    text("SELECT attempt FROM node_states WHERE token_id = :tid ORDER BY attempt"),
                    {"tid": incomplete_token_id},
                ).fetchall()
            ]
        assert 0 in attempts, "run-1 node_state (attempt=0) must be preserved after resume (append-only invariant)"
        assert 1 in attempts, (
            f"resume re-drive must record a new node_state at attempt=1 (max+1); "
            f"got attempts={attempts!r}. An IntegrityError here means the re-drive "
            f"attempted attempt=0 and collided — offset computation missed begin_node_state."
        )

    def test_resume_offset_is_max_plus_one_not_hardcoded(self) -> None:
        """If the incomplete child has TWO run-1 attempts (0 and 1, e.g. a tenacity
        retry before the crash), the re-drive must land at attempt=2 (max+1), not
        a hardcoded 1. Guards against offset = +1 / 'resume generation = 1'.

        RED (dispatch disabled): AssertionError — ``assert 2 in attempts`` (and
        ``assert max(attempts) == 2``) fail because process_existing_row mints a
        fresh token and never touches the original token's node_states, so the
        only attempt present on that token is the synthetic attempt=1 we inserted
        (i.e. attempts == [0, 1], not [0, 1, 2]).
        """

        db, payload_store, config, graph, settings_obj, run_id, incomplete_token_id = self._setup_fork_and_interrupt()

        # Insert a synthetic run-1 attempt-1 node_state for the incomplete child,
        # simulating a tenacity retry that happened during run-1 before the crash.
        # Clone all columns from the existing attempt-0 row via raw SQL INSERT/SELECT
        # so SQLite handles the datetime columns directly (avoiding SQLAlchemy's
        # DateTime type mapping, which rejects string values from fetchone().__mapping__).
        # resume_checkpoint_id stays NULL so this reads as a genuine run-1 row.
        import uuid

        synthetic_state_id = f"ns-synthetic-{uuid.uuid4().hex[:16]}"
        with db.engine.connect() as conn:
            # Verify the source row exists first.
            existing_count = conn.execute(
                text("SELECT COUNT(*) FROM node_states WHERE token_id = :tid AND attempt = 0"),
                {"tid": incomplete_token_id},
            ).scalar()
            assert existing_count == 1, "Expected an attempt=0 node_state for the incomplete sink_a child before insert"
            # Copy all columns from attempt=0, overriding only state_id, attempt, and
            # resume_checkpoint_id. Raw SQL INSERT...SELECT avoids the datetime coercion
            # issue that occurs when passing SQLite-stored strings through SQLAlchemy's
            # DateTime column type.
            conn.execute(
                text(
                    "INSERT INTO node_states "
                    "(state_id, token_id, run_id, node_id, step_index, attempt, status, "
                    " input_hash, output_hash, context_before_json, context_after_json, "
                    " duration_ms, error_json, success_reason_json, started_at, completed_at, "
                    " resume_checkpoint_id) "
                    "SELECT :new_state_id, token_id, run_id, node_id, step_index, 1, status, "
                    "       input_hash, output_hash, context_before_json, context_after_json, "
                    "       duration_ms, error_json, success_reason_json, started_at, completed_at, "
                    "       NULL "  # run-1 row — no provenance marker
                    "FROM node_states WHERE token_id = :tid AND attempt = 0"
                ),
                {"new_state_id": synthetic_state_id, "tid": incomplete_token_id},
            )
            conn.commit()

        # Precondition: attempts are now [0, 1] (max=1) so max+1 should land at 2.
        with db.engine.connect() as conn:
            pre_attempts = [
                r.attempt
                for r in conn.execute(
                    text("SELECT attempt FROM node_states WHERE token_id = :tid ORDER BY attempt"),
                    {"tid": incomplete_token_id},
                ).fetchall()
            ]
        assert pre_attempts == [0, 1], f"Pre-resume attempts must be [0, 1]; got {pre_attempts!r}"

        self._resume_run(db, payload_store, config, graph, settings_obj, run_id)

        with db.engine.connect() as conn:
            attempts = [
                r.attempt
                for r in conn.execute(
                    text("SELECT attempt FROM node_states WHERE token_id = :tid ORDER BY attempt"),
                    {"tid": incomplete_token_id},
                ).fetchall()
            ]
        assert 2 in attempts, (
            f"resume re-drive must land at attempt=max+1=2 (max was 1 from synthetic insert); "
            f"got attempts={attempts!r}. A hardcoded '1' would fail this assertion."
        )
        assert max(attempts) == 2, f"max(attempts) must be exactly 2 (max+1); got {max(attempts)!r} with attempts={attempts!r}"

    def test_resume_redrive_is_query_separable_from_retry(self) -> None:
        """A resume re-drive node_state carries resume_checkpoint_id; the run-1
        record at the same (token_id, node_id) does not. Proves explain() can
        separate them by NULL-ness alone.

        RED (dispatch disabled): AssertionError — ``assert run1 and redrive``
        fails because process_existing_row never touches the original token's
        node_states, so every node_state on that token has
        resume_checkpoint_id=NULL; the ``redrive`` list is empty.
        """
        db, payload_store, config, graph, settings_obj, run_id, incomplete_token_id = self._setup_fork_and_interrupt()

        self._resume_run(db, payload_store, config, graph, settings_obj, run_id)

        with db.engine.connect() as conn:
            rows = conn.execute(
                text("SELECT attempt, resume_checkpoint_id FROM node_states WHERE token_id = :tid ORDER BY attempt"),
                {"tid": incomplete_token_id},
            ).fetchall()
        run1 = [r for r in rows if r.resume_checkpoint_id is None]
        redrive = [r for r in rows if r.resume_checkpoint_id is not None]
        assert run1 and redrive, (
            f"Both a run-1 (resume_checkpoint_id=NULL) and a re-drive "
            f"(resume_checkpoint_id non-NULL) record must exist for the incomplete token; "
            f"run1={[(r.attempt,) for r in run1]!r} redrive={[(r.attempt,) for r in redrive]!r}"
        )
        assert all(r.attempt == 0 for r in run1), f"All run-1 records must be at attempt=0; got {[(r.attempt,) for r in run1]!r}"
        assert all(r.attempt >= 1 for r in redrive), f"All re-drive records must be at attempt>=1; got {[(r.attempt,) for r in redrive]!r}"

    def test_expand_token_persists_per_child_payload(self) -> None:
        """expand_token stores a {data, contract} envelope and writes tokens.token_data_ref.

        Verifies that:
        1. Each child has a non-null token_data_ref.
        2. Retrieving the ref bytes and loading via checkpoint_loads gives an envelope
           dict with keys "data" and "contract" — NOT a bare data dict (ADDENDUM 3).
        3. env["data"] round-trips the CORRECT child's payload, proving per-child
           value alignment and type fidelity (datetime survives as datetime, not string).
        4. SchemaContract.from_checkpoint(env["contract"]) restores the contract with
           field names and mode equal to the persisted contract (hash-validated by
           from_checkpoint itself — Tier-1 integrity).

        This is the Tier-1 audit invariant: every expand child is self-contained and
        reconstructable from its token_data_ref on resume without any nodes-table lookup.
        """
        from datetime import datetime

        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        _OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
        payload_store = MockPayloadStore()
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_OBSERVED_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"items": [1, 2]},
        )
        parent_token = factory.data_flow.create_token(row_id=row.row_id)

        # Build a real SchemaContract — the one the expand step would produce.
        output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(normalized_name="name", original_name="name", python_type=str, required=True, source="declared"),
                FieldContract(normalized_name="value", original_name="value", python_type=int, required=True, source="declared"),
                FieldContract(normalized_name="ts", original_name="ts", python_type=datetime, required=True, source="declared"),
            ),
            locked=True,
        )

        # Two DISTINCT payloads — one with a datetime (type-fidelity witness).
        # canonical_json would stringify the datetime; checkpoint_dumps preserves it.
        aware_dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        child_payloads = [
            {"name": "alpha", "value": 1, "ts": aware_dt},
            {"name": "beta", "value": 2, "ts": aware_dt},
        ]

        children, expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=parent_token.token_id, run_id=run.run_id),
            row_id=row.row_id,
            child_payloads=child_payloads,
            output_contract=output_contract,
            step_in_pipeline=1,
        )

        assert len(children) == 2
        assert expand_group_id is not None

        # Both children must have non-null token_data_ref
        for child in children:
            assert child.token_data_ref is not None, (
                f"expand_token must set token_data_ref on every child (epoch 11 invariant); child {child.token_id} has token_data_ref=None"
            )

        # Round-trip: retrieve bytes → checkpoint_loads → assert envelope shape + content.
        # Critically, verify EACH child has ITS OWN payload (not the sibling's).
        for i, (child, expected_payload) in enumerate(zip(children, child_payloads, strict=True)):
            raw_bytes = payload_store.retrieve(child.token_data_ref)
            env = checkpoint_loads(raw_bytes.decode("utf-8"))

            # Envelope shape — must be {data, contract}, NOT a bare data dict (ADDENDUM 3).
            assert isinstance(env, dict) and "data" in env and "contract" in env, (
                f"Child {i} token_data_ref payload is not a {{data, contract}} envelope; "
                f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
            )

            data = env["data"]

            # Value alignment: correct child, correct fields
            assert data["name"] == expected_payload["name"], (
                f"Child {i} (token_data_ref={child.token_data_ref!r}) stored wrong payload: "
                f"expected name={expected_payload['name']!r}, got {data['name']!r}"
            )
            assert data["value"] == expected_payload["value"], (
                f"Child {i} stored wrong value: expected {expected_payload['value']}, got {data['value']}"
            )

            # Type fidelity: datetime must come back as datetime, not a string.
            # This proves checkpoint_dumps was used (canonical_json would stringify datetime).
            assert isinstance(data["ts"], datetime), (
                f"Type fidelity failure: 'ts' field came back as {type(data['ts']).__name__!r}, "
                f"not datetime — checkpoint_dumps was not used (canonical_json stringifies datetime)"
            )
            assert data["ts"].tzinfo is not None, "Restored datetime must be timezone-aware"
            assert data["ts"] == aware_dt, f"datetime value mismatch: expected {aware_dt!r}, got {data['ts']!r}"

            # Contract round-trip: SchemaContract.from_checkpoint validates hash integrity
            # (Tier-1 — raises AuditIntegrityError on mismatch) and restores the contract.
            restored_contract = SchemaContract.from_checkpoint(env["contract"])
            assert restored_contract.mode == output_contract.mode, (
                f"Contract mode mismatch: expected {output_contract.mode!r}, got {restored_contract.mode!r}"
            )
            assert restored_contract.locked == output_contract.locked
            restored_names = {fc.normalized_name for fc in restored_contract.fields}
            expected_names = {fc.normalized_name for fc in output_contract.fields}
            assert restored_names == expected_names, f"Contract field names mismatch: expected {expected_names!r}, got {restored_names!r}"

    def test_coalesce_token_persists_merged_payload(self) -> None:
        """coalesce_tokens stores a {data, contract} envelope and writes tokens.token_data_ref.

        Verifies that:
        1. The merged token has a non-null token_data_ref.
        2. Retrieving the ref bytes and loading via checkpoint_loads gives an envelope
           dict with keys "data" and "contract" — NOT a bare data dict (ADDENDUM 3).
        3. env["data"] round-trips the merged payload with full type fidelity (datetime
           survives as datetime, not string).
        4. SchemaContract.from_checkpoint(env["contract"]) restores the contract with
           field names and mode equal to the persisted contract (hash-validated by
           from_checkpoint itself — Tier-1 integrity).

        This is the Tier-1 audit invariant: the merged token is reconstructable
        from its token_data_ref on resume without re-executing the merge strategy
        and without any nodes-table lookup (ADDENDUM 3).
        """
        from datetime import datetime

        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        _OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
        payload_store = MockPayloadStore()
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=_OBSERVED_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"key": "value"},
        )

        # Build a real SchemaContract — the one the coalesce step would produce.
        merged_contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(normalized_name="merged", original_name="merged", python_type=bool, required=True, source="declared"),
                FieldContract(normalized_name="count", original_name="count", python_type=int, required=True, source="declared"),
                FieldContract(
                    normalized_name="result_score", original_name="result_score", python_type=float, required=True, source="declared"
                ),
                FieldContract(
                    normalized_name="resolved_at", original_name="resolved_at", python_type=datetime, required=True, source="declared"
                ),
            ),
            locked=True,
        )

        # Create two branch tokens to coalesce
        token_a = factory.data_flow.create_token(row_id=row.row_id)
        token_b = factory.data_flow.create_token(row_id=row.row_id)

        # Merged payload includes a datetime to verify type fidelity.
        # canonical_json would stringify datetime; checkpoint_dumps preserves it.
        aware_dt = datetime(2025, 3, 10, 8, 30, 0, tzinfo=UTC)
        merged_payload = {
            "merged": True,
            "count": 2,
            "result_score": 0.87,
            "resolved_at": aware_dt,
        }

        merged_token = factory.data_flow.coalesce_tokens(
            parent_refs=[
                TokenRef(token_id=token_a.token_id, run_id=run.run_id),
                TokenRef(token_id=token_b.token_id, run_id=run.run_id),
            ],
            row_id=row.row_id,
            merged_payload=merged_payload,
            merged_contract=merged_contract,
            step_in_pipeline=2,
        )

        # Merged token must have non-null token_data_ref
        assert merged_token.token_data_ref is not None, "coalesce_tokens must set token_data_ref on the merged token (epoch 11 invariant)"
        assert merged_token.join_group_id is not None

        # Round-trip: retrieve bytes → checkpoint_loads → assert envelope shape + content.
        raw_bytes = payload_store.retrieve(merged_token.token_data_ref)
        env = checkpoint_loads(raw_bytes.decode("utf-8"))

        # Envelope shape — must be {data, contract}, NOT a bare data dict (ADDENDUM 3).
        assert isinstance(env, dict) and "data" in env and "contract" in env, (
            f"Merged token payload is not a {{data, contract}} envelope; "
            f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
        )

        data = env["data"]

        assert data["merged"] is True
        assert data["count"] == 2
        assert abs(data["result_score"] - 0.87) < 1e-9

        # Type fidelity: datetime must come back as datetime, not a string.
        # This proves checkpoint_dumps was used (canonical_json would stringify datetime).
        assert isinstance(data["resolved_at"], datetime), (
            f"Type fidelity failure: 'resolved_at' came back as {type(data['resolved_at']).__name__!r}, "
            f"not datetime — checkpoint_dumps was not used (canonical_json stringifies datetime)"
        )
        assert data["resolved_at"].tzinfo is not None, "Restored datetime must be timezone-aware"

        # Contract round-trip: SchemaContract.from_checkpoint validates hash integrity
        # (Tier-1 — raises AuditIntegrityError on mismatch) and restores the contract.
        restored_contract = SchemaContract.from_checkpoint(env["contract"])
        assert restored_contract.mode == merged_contract.mode, (
            f"Contract mode mismatch: expected {merged_contract.mode!r}, got {restored_contract.mode!r}"
        )
        assert restored_contract.locked == merged_contract.locked
        restored_names = {fc.normalized_name for fc in restored_contract.fields}
        expected_names = {fc.normalized_name for fc in merged_contract.fields}
        assert restored_names == expected_names, f"Contract field names mismatch: expected {expected_names!r}, got {restored_names!r}"

    def test_get_incomplete_tokens_by_row_returns_only_incomplete_leaf(self) -> None:
        """Recovery surfaces exactly the non-delegation tokens lacking a terminal outcome.

        Build the fork pipeline (gate forks each row to sink_a + sink_b), run to
        completion, delete sink_a's terminal outcome so its child is the sole incomplete
        leaf. get_incomplete_tokens_by_row must return exactly that child token and no
        others (not the FORK_PARENT delegation marker, not the completed sink_b child).
        """
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        # Single row keeps the test deterministic.
        source = ListSource([{"value": 1}], on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # Gate forks every row to both sinks.
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # Determine the sink_a child token ID independently of the method under test.
        # After a complete run we expect exactly one fork child per sink per row.
        with db.engine.connect() as conn:
            sink_a_tokens = conn.execute(
                text("""
                    SELECT t.token_id AS token_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.branch_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(sink_a_tokens) == 1, f"Expected exactly one sink_a fork-child token, got {len(sink_a_tokens)}"
        incomplete_token_id = sink_a_tokens[0].token_id

        # Interrupt: delete the sink_a child's terminal outcome to simulate a crash
        # after sink_b wrote but before sink_a wrote (or after sink_a wrote but before
        # the outcome was recorded).
        with db.engine.connect() as conn:
            sink_a_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.sink_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
            assert sink_a_outcomes, "expected a sink_a outcome to delete"
            for outcome in sink_a_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        # Exercise the method under test — no checkpoint or run-status required.
        checkpoint_mgr = CheckpointManager(db)
        recovery = RecoveryManager(db, checkpoint_mgr)
        by_row = recovery.get_incomplete_tokens_by_row(run_id)

        # Exactly one row_id group, exactly one spec in it.
        assert len(by_row) == 1, f"Expected 1 row group, got {len(by_row)}: {list(by_row.keys())}"
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 1, f"Expected exactly 1 incomplete token spec, got {len(all_specs)}: {[s.token_id for s in all_specs]}"

        spec = all_specs[0]
        assert spec.token_id == incomplete_token_id, f"Spec token_id {spec.token_id!r} != expected {incomplete_token_id!r}"
        assert spec.branch_name == "sink_a", f"Spec branch_name {spec.branch_name!r} != 'sink_a'"
        assert spec.fork_group_id is not None, "fork child must carry fork_group_id (set by the gate on fork)"
        assert spec.token_data_ref is None, "fork child shares the source payload (retrieval by row_id); token_data_ref must be NULL"
        # The fork child visited a sink node → node_states written → max_attempt should be 0.
        # If this fires at -1 it means fork children don't write node_states, which is a
        # real finding: the re-drive logic in a later task relies on max_attempt + 1.
        assert spec.max_attempt >= 0, (
            f"Fork child token {spec.token_id!r} has no node_state entry (max_attempt=-1); "
            f"the re-drive attempt-number anchor is missing — audit invariant violation."
        )

    def test_token_data_ref_read_paths_are_distinct(self) -> None:
        """Two read paths for token_data_ref have distinct responsibilities.

        After expand_token() persists an expand child with a non-null token_data_ref:
        (a) QueryRepository.get_token() (the lineage/audit read path) returns a Token
            with token_data_ref=None — TokenLoader.load() deliberately omits the column,
            as lineage queries never need to hydrate payloads.
        (b) get_incomplete_tokens_by_row() (the recovery read path) returns the real ref
            stored in the DB, because resume reconstruction needs to retrieve the payload.

        This test pins the boundary so neither path silently acquires the other's
        responsibility in future refactors.
        """
        from elspeth.contracts.audit import TokenRef
        from elspeth.contracts.enums import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.factory import RecorderFactory

        _OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
        payload_store = MockPayloadStore()
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_OBSERVED_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"items": [1, 2]},
        )
        parent_token = factory.data_flow.create_token(row_id=row.row_id)

        # Persist two expand children — both write token_data_ref (epoch 11 invariant).
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        _read_path_contract = SchemaContract(
            mode="OBSERVED",
            fields=(FieldContract(normalized_name="name", original_name="name", python_type=str, required=True, source="inferred"),),
            locked=True,
        )
        children, _expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=parent_token.token_id, run_id=run.run_id),
            row_id=row.row_id,
            child_payloads=[{"name": "alpha"}, {"name": "beta"}],
            output_contract=_read_path_contract,
            step_in_pipeline=1,
        )
        assert len(children) == 2
        child = children[0]

        # Precondition: the expand child genuinely has a non-null token_data_ref.
        assert child.token_data_ref is not None, "expand_token must set token_data_ref (epoch 11 invariant); test precondition failed"

        # (a) Lineage read path: QueryRepository.get_token() omits token_data_ref.
        # TokenLoader.load() doesn't pass the column, so the Token model gets its
        # default value (None) regardless of what the DB holds.
        token_via_loader = factory.query.get_token(child.token_id)
        assert token_via_loader is not None, "get_token returned None for a persisted expand child"
        assert token_via_loader.token_data_ref is None, (
            f"get_token (lineage path) must NOT hydrate token_data_ref; "
            f"got {token_via_loader.token_data_ref!r} — TokenLoader.load() gained a new responsibility"
        )

        # (b) Recovery read path: get_incomplete_tokens_by_row() surfaces the real ref.
        # The expand parent has an EXPAND_PARENT delegation outcome → filtered out.
        # The two children have no terminal outcome → both appear in incomplete specs.
        checkpoint_mgr = CheckpointManager(db)
        recovery = RecoveryManager(db, checkpoint_mgr)
        by_row = recovery.get_incomplete_tokens_by_row(run.run_id)

        # Find the spec for our specific child.
        child_specs = [s for specs in by_row.values() for s in specs if s.token_id == child.token_id]
        assert len(child_specs) == 1, f"Expected exactly one IncompleteTokenSpec for child token {child.token_id!r}; got {len(child_specs)}"
        child_spec = child_specs[0]
        assert child_spec.token_data_ref == child.token_data_ref, (
            f"Recovery read path must surface the real token_data_ref from the DB; "
            f"spec.token_data_ref={child_spec.token_data_ref!r} != "
            f"expand_token result.token_data_ref={child.token_data_ref!r}"
        )

    def test_reconstruct_token_row_fork_vs_envelope(self) -> None:
        """reconstruct_token_row covers both branches: fork (source_row) and envelope.

        Fork branch: spec.token_data_ref is None → source_row returned unchanged.
        Envelope branch: spec.token_data_ref is set (expand child) → payload retrieved
        from store, {data, contract} envelope decoded, PipelineRow(data, contract)
        returned. Datetime type fidelity and contract hash-validation are verified.

        This test pins the two-branch contract described in ADDENDUM 3 so neither
        branch silently breaks in future refactors.
        """
        from datetime import datetime

        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.recovery import IncompleteTokenSpec

        _OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
        payload_store = MockPayloadStore()
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_OBSERVED_SCHEMA,
        )

        # Build a source row and source_row PipelineRow to use as the fork reference.
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"source_val": 99},
        )
        source_schema = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(normalized_name="source_val", original_name="source_val", python_type=int, required=True, source="inferred"),
            ),
            locked=True,
        )
        from elspeth.contracts.schema_contract import PipelineRow as PLRow

        source_row = PLRow({"source_val": 99}, source_schema)

        # ── Fork branch: token_data_ref is None ──────────────────────────────────
        # Build a minimal IncompleteTokenSpec with token_data_ref=None (fork child).
        fork_spec = IncompleteTokenSpec(
            token_id="fork-child-token",
            row_id=row.row_id,
            branch_name="sink_a",
            fork_group_id="fg-1",
            join_group_id=None,
            expand_group_id=None,
            token_data_ref=None,
            step_in_pipeline=1,
            max_attempt=0,
        )

        checkpoint_mgr = CheckpointManager(db)
        recovery = RecoveryManager(db, checkpoint_mgr)

        fork_result = recovery.reconstruct_token_row(
            spec=fork_spec,
            run_id=run.run_id,
            source_row=source_row,
            payload_store=payload_store,
        )
        # Fork branch must return source_row identity — shared payload, no copy.
        assert fork_result is source_row, "Fork branch (token_data_ref=None) must return source_row unchanged (identity)"

        # ── Envelope branch: expand child with {data, contract} in token_data_ref ──
        # Build the output contract the expand step would produce.
        aware_dt = datetime(2024, 11, 5, 9, 15, 0, tzinfo=UTC)
        output_contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(normalized_name="item", original_name="item", python_type=str, required=True, source="declared"),
                FieldContract(normalized_name="score", original_name="score", python_type=float, required=True, source="declared"),
                FieldContract(
                    normalized_name="recorded_at", original_name="recorded_at", python_type=datetime, required=True, source="declared"
                ),
            ),
            locked=True,
        )

        parent_token = factory.data_flow.create_token(row_id=row.row_id)
        expand_payload = {"item": "widget", "score": 0.95, "recorded_at": aware_dt}
        children, _expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=parent_token.token_id, run_id=run.run_id),
            row_id=row.row_id,
            child_payloads=[expand_payload],
            output_contract=output_contract,
            step_in_pipeline=2,
        )
        assert len(children) == 1
        child = children[0]
        assert child.token_data_ref is not None, "expand_token must set token_data_ref (epoch 11 invariant)"

        # Build the IncompleteTokenSpec that recovery would produce for this child.
        envelope_spec = IncompleteTokenSpec(
            token_id=child.token_id,
            row_id=row.row_id,
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=child.expand_group_id,
            token_data_ref=child.token_data_ref,
            step_in_pipeline=2,
            max_attempt=-1,
        )

        envelope_result = recovery.reconstruct_token_row(
            spec=envelope_spec,
            run_id=run.run_id,
            source_row=source_row,
            payload_store=payload_store,
        )

        # Must NOT be source_row — it carries its own payload.
        assert envelope_result is not source_row

        # Data fidelity: correct values.
        assert envelope_result["item"] == "widget"
        assert abs(envelope_result["score"] - 0.95) < 1e-9

        # Type fidelity: datetime must survive as datetime, not a string.
        assert isinstance(envelope_result["recorded_at"], datetime), (
            f"Type fidelity failure: 'recorded_at' came back as {type(envelope_result['recorded_at']).__name__!r}, not datetime"
        )
        assert envelope_result["recorded_at"].tzinfo is not None
        assert envelope_result["recorded_at"] == aware_dt

        # Contract fidelity: the restored contract must match the persisted one.
        # SchemaContract.from_checkpoint validates the hash — if we reach here, Tier-1
        # integrity is confirmed.
        restored_contract = envelope_result.contract
        assert restored_contract.mode == output_contract.mode
        assert restored_contract.locked == output_contract.locked
        restored_names = {fc.normalized_name for fc in restored_contract.fields}
        expected_names = {fc.normalized_name for fc in output_contract.fields}
        assert restored_names == expected_names, f"Contract field names mismatch: expected {expected_names!r}, got {restored_names!r}"

    def test_reconstruct_token_row_rejects_malformed_envelope(self) -> None:
        """reconstruct_token_row raises AuditIntegrityError on a non-envelope payload.

        The envelope-shape guard (recovery.py) is the Tier-1 safety net against a
        corrupt or pre-ADDENDUM-3 (bare-dict) payload read from the audit store. It
        must crash loudly — no silent coercion — for BOTH malformed shapes:

        (a) non-dict payload (e.g. a bare string written by some other writer);
        (b) a dict that is missing the required "data"/"contract" keys (e.g. the
            old pre-envelope bare-data dict shape).

        We inject each malformed payload directly into the MockPayloadStore and build
        an IncompleteTokenSpec pointing at its ref — no full pipeline run is needed to
        exercise the guard.
        """
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.contracts.schema_contract import PipelineRow as PLRow
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
        from elspeth.core.checkpoint.serialization import checkpoint_dumps

        payload_store = MockPayloadStore()
        db = make_landscape_db()
        checkpoint_mgr = CheckpointManager(db)
        recovery = RecoveryManager(db, checkpoint_mgr)

        # A real source_row is only needed for the call signature; the guard fires
        # before it would ever be consumed (token_data_ref is non-None here).
        source_schema = SchemaContract(
            mode="OBSERVED",
            fields=(FieldContract(normalized_name="x", original_name="x", python_type=int, required=True, source="inferred"),),
            locked=True,
        )
        source_row = PLRow({"x": 1}, source_schema)

        def _spec_for(ref: str) -> IncompleteTokenSpec:
            return IncompleteTokenSpec(
                token_id="malformed-token",
                row_id="row-malformed",
                branch_name=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id="eg-malformed",
                token_data_ref=ref,
                step_in_pipeline=2,
                max_attempt=-1,
            )

        # (a) non-dict payload: a bare string round-tripped through checkpoint_dumps.
        non_dict_ref = payload_store.store(checkpoint_dumps("just a string").encode("utf-8"))
        with pytest.raises(AuditIntegrityError, match="not a"):
            recovery.reconstruct_token_row(
                spec=_spec_for(non_dict_ref),
                run_id="run-malformed",
                source_row=source_row,
                payload_store=payload_store,
            )

        # (b) dict-missing-keys payload: the pre-envelope bare-data dict shape.
        missing_keys_ref = payload_store.store(checkpoint_dumps({"old_key": 1}).encode("utf-8"))
        with pytest.raises(AuditIntegrityError, match="not a"):
            recovery.reconstruct_token_row(
                spec=_spec_for(missing_keys_ref),
                run_id="run-malformed",
                source_row=source_row,
                payload_store=payload_store,
            )

    # ─────────────────────────────────────────────────────────────────────
    # F1 Regression Cells (Task 9) — fork→coalesce + post-coalesce (B1)
    # ─────────────────────────────────────────────────────────────────────

    def _setup_coalesce_pipeline(
        self,
    ) -> tuple[
        LandscapeDB,
        MockPayloadStore,
        PipelineConfig,
        ExecutionGraph,
        ElspethSettings,
        str,  # run_id
        RunResult,  # the completed run-1 RunResult (live counters)
    ]:
        """Shared setup: build and run a fork→PassTransform→coalesce→sink pipeline.

        Topology: source → gate(fork_to=[path_a, path_b])
                    → path_a: PassTransform(pass_a) → coalesce 'merge'
                    → path_b: PassTransform(pass_b) → coalesce 'merge'
                    → sink 'output'

        PassTransforms on both branches ensure branch_first_node is a REAL
        processing node distinct from the coalesce barrier — this exercises
        resolve_branch_first_node() (not the coalesce node itself).

        CoalesceSettings.on_success='output' makes the coalesce TERMINAL
        (resolve_next_node(coalesce_node_id) is None in the traversal map —
        the sink is reached via resolve_coalesce_sink, not the next-node map).

        Returns (db, payload_store, config, graph, settings_obj, run_id, run)
        for the completed run.  ``run`` is the live RunResult of the
        uninterrupted run-1 (its counters come from the LIVE accumulator, a
        different code path from derive_resume_terminal_status_from_audit) —
        callers that need a same-topology uninterrupted oracle for
        field-for-field reconciliation against a RESUMED run should build a
        SEPARATE _setup_coalesce_pipeline() instance for run A and use this
        ``run`` as that oracle (do NOT interrupt run A).  The same
        payload_store MUST be used for both the initial run and the resume
        (merged token_data_ref lives there).
        """
        from elspeth.core.config import ElspethSettings

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 1}], on_success="gate_in")
        sink = CollectSink("output")

        pass_a = PassTransform(name="pass_a")
        pass_b = PassTransform(name="pass_b")

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )

        # branches dict maps branch-name → final-connection-into-coalesce.
        # wire_transforms wires: path_a → pass_a → done_a (consumed by coalesce 'merge').
        coalesce = CoalesceSettings(
            name="merge",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            on_success="output",
        )

        wired_a = wire_transforms([pass_a], source_connection="path_a", final_sink="done_a", names=["pass_a"])
        wired_b = wire_transforms([pass_b], source_connection="path_b", final_sink="done_b", names=["pass_b"])

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=wired_a + wired_b,
            sinks={"output": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(pass_a), as_transform(pass_b)],
            sinks={"output": as_sink(sink)},
            gates=[gate],
            coalesce_settings=[coalesce],
        )

        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "gate_in", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id
        return db, payload_store, config, graph, settings_obj, run_id, run

    def test_resume_fork_to_coalesce_before_barrier(self) -> None:
        """Re-driving branch tokens interrupted BEFORE the coalesce barrier must not
        double-emit: the barrier fires exactly once and conservation holds.

        Topology:
            source → gate(fork) → [path_a: PassTransform, path_b: PassTransform]
                   → coalesce('merge', on_success='output') → sink 'output'

        The PassTransform on each branch makes branch_first_node a REAL processing
        node distinct from the coalesce node, exercising resolve_branch_first_node().

        Interruption: after a complete run, undo the barrier entirely by deleting:
          - the merged token's terminal outcome, node_states, token_parents, and
            token row itself;
          - both branch children's COALESCED outcomes (which the barrier recorded).
        This leaves the two branch child tokens with no terminal outcome — faithfully
        mirroring a pre-barrier crash (the barrier code never ran).

        Resume-dispatch oracle (get_incomplete_tokens_by_row): must return exactly
        the two branch child tokens (branch_name set, join_group_id=None,
        token_data_ref=None) — no merged token.  This confirms dispatch Case 2
        (fork → coalesce, before-barrier) for both specs.

        RED (dispatch disabled, resume.py fork_expand_coalesce_specs branch commented):
        process_existing_row re-forks the row from the source node, creating a NEW
        fork parent and NEW branch children.  The interruption (step #6 above) deleted
        the coalesce node_states, so _check_landscape_for_completion returns False for
        the NEW children → they merge cleanly via the fresh CoalesceExecutor and a
        SECOND merged token is created.  However, the ORIGINAL branch tokens (path_a,
        path_b from run-1) remain in the DB with NO terminal outcomes (their COALESCED
        outcomes were deleted in step #5 and process_existing_row never re-drives them).
        At end-of-row, sweep_deferred_invariants_or_crash (ADR-019 I1a) scans fork
        parents: the ORIGINAL run-1 fork parent has FORK_PARENT outcome but its original
        children (path_a, path_b) have zero terminal outcomes → I1a fires.
        Observed: AuditIntegrityError — "fork/expand parent token(s) have no child
        token_outcomes rows at run-end." (1 token, path=fork_parent).

        GREEN (dispatch restored): both branch specs driven to the barrier; barrier
        fires exactly once on the second; merged token (re-)created; sink written once;
        full conservation holds.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import token_outcomes_table, token_parents_table, tokens_table

        db, payload_store, config, graph, settings_obj, run_id, _run1 = self._setup_coalesce_pipeline()

        # ── Baseline (run-1 completed) ──────────────────────────────────────────
        def _outcome_counts() -> dict[tuple[str, str], int]:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND o.completed = 1
                          AND o.sink_name IS NOT NULL
                        GROUP BY t.row_id, o.sink_name
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return {(row.row_id, row.sink_name): row.n for row in result}

        baseline = _outcome_counts()
        assert len(baseline) == 1, f"Expected exactly one (row_id, sink_name) completed outcome; got {baseline}"
        assert all(n == 1 for n in baseline.values()), baseline

        # ── Find the merged token (join_group_id set, branch_name NULL) ───────────
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.join_group_id AS join_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token (join_group_id set, branch/fork NULL); got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        join_group_id = merged_rows[0].join_group_id

        # ── Interrupt: undo the barrier entirely ───────────────────────────────────
        # A pre-barrier crash means: the barrier code never ran.  In production this
        # means no merged token exists, no branch COALESCED outcomes were recorded,
        # and the branch tokens' node_states at the coalesce node are NOT marked
        # completed (the CoalesceExecutor calls begin_node_state on arrival but only
        # calls complete_node_state once the barrier fires).
        #
        # We must therefore reverse everything the barrier wrote:
        #   1. merged token's terminal outcome (COMPLETED at sink)
        #   2. merged token's node_states
        #   3. token_parents where token_id = merged (merged→branch parent links)
        #   4. merged token row itself
        #   5. branch tokens' COALESCED outcomes (path='coalesced', jgid=join_group_id)
        #   6. branch tokens' COMPLETED node_states at the coalesce node —
        #      CoalesceExecutor._check_landscape_for_completion queries
        #      get_completed_row_ids_for_nodes which joins node_states→tokens and
        #      checks completed_at IS NOT NULL; if these remain, accept() sees
        #      "already completed" and records a spurious UNROUTED outcome instead
        #      of holding/merging the re-driven branch tokens.
        #
        # After this, the two branch child tokens have no terminal outcome and no
        # completed coalesce node_state — faithful pre-barrier crash state.

        # First, find the coalesce node_id so we can delete the branch tokens'
        # coalesce-node node_states by (token_id, node_id) rather than all states.
        coalesce_node_id_for_deletion = graph.get_coalesce_id_map()[CoalesceName("merge")]

        with db.engine.connect() as conn:
            # Temporarily disable FK enforcement to allow deletion of the merged token
            # and its dependents in a single connection without a strict topological order.
            # All rows being deleted are barrier artifacts from this specific run; the
            # connection is committed before FKs are re-enabled to keep the DB consistent.
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")

            # 1. merged token outcomes (COMPLETED at sink)
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.token_id == merged_token_id))
            # 2. merged token node_states (coalesce node may have routing_events referencing
            #    them; disable FKs makes this safe without a full cascade walk)
            conn.execute(
                text("DELETE FROM node_states WHERE token_id = :tid"),
                {"tid": merged_token_id},
            )
            # 3. token_parents for merged token (FK: token_id → token_parents.token_id)
            conn.execute(token_parents_table.delete().where(token_parents_table.c.token_id == merged_token_id))
            # 4. merged token row (FK deps removed above)
            conn.execute(tokens_table.delete().where(tokens_table.c.token_id == merged_token_id))
            # 5. branch COALESCED outcomes (recorded by the barrier, path='coalesced')
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.join_group_id == join_group_id))
            # 6. branch tokens' COMPLETED node_states at the coalesce node.
            #    CoalesceExecutor._check_landscape_for_completion (called by accept())
            #    queries completed_at IS NOT NULL for the coalesce node.  If these
            #    remain, the re-driven branch tokens are treated as late arrivals
            #    and get UNROUTED/FAILURE instead of being held/merged.
            #    We delete by (node_id=coalesce_node_id, run_id=run_id) — which covers
            #    both branch token_ids without needing to enumerate them.
            conn.execute(
                text("DELETE FROM node_states WHERE node_id = :nid AND run_id = :rid"),
                {"nid": str(coalesce_node_id_for_deletion), "rid": run_id},
            )
            conn.commit()
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")

        # ── Oracle: incomplete specs must be the TWO branch children (Case 2) ────
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)

        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 2, (
            f"Oracle must return exactly 2 incomplete specs (both branch children, "
            f"pre-barrier state); got {len(all_specs)}: {[s.token_id for s in all_specs]}"
        )
        for spec in all_specs:
            assert spec.branch_name is not None, (
                f"Before-barrier spec must have branch_name set (Case 2); got None for token {spec.token_id!r}"
            )
            assert spec.join_group_id is None, (
                f"Before-barrier spec must NOT have join_group_id (no merged token); got {spec.join_group_id!r} for token {spec.token_id!r}"
            )
            assert spec.token_data_ref is None, (
                f"Before-barrier spec (fork child) must NOT have token_data_ref (shares source payload); got {spec.token_data_ref!r}"
            )

        # ── Resume ────────────────────────────────────────────────────────────────
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Conservation law ──────────────────────────────────────────────────────
        after = _outcome_counts()
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left {len(orphans)} non-delegation leaf token(s) with no terminal outcome (orphans): {orphans}"
        assert all(n == 1 for n in after.values()), f"No (row_id, sink_name) may carry two outcomes after resume: {after}"
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        post_stats = get_fork_group_stats(db, run_id)
        assert post_stats["total_fork_groups"] == 1, f"Fork-group count must remain 1 (one row, one fork); got {post_stats}"

    def test_resume_coalesce_held_branch_not_redriven(self) -> None:
        """A branch HELD at a coalesce barrier (checkpointed into coalesce pending) must
        NOT be re-driven on resume — it is restored into the executor's _pending and
        flushes there. Re-driving it re-arrives at the barrier and crashes the
        duplicate-arrival guard. Only the genuinely-incomplete (not-yet-arrived) sibling
        re-drives.

        Topology (shared _setup_coalesce_pipeline):
            source → gate(fork) → [path_a: PassTransform, path_b: PassTransform]
                   → coalesce('merge', require_all, on_success='output') → sink 'output'

        Construction of the partial-fan-in interrupt state (genuine, via production run +
        targeted reversal — mirrors test_resume_fork_to_coalesce_before_barrier's
        barrier-undo, then re-creates the HELD state for ONE branch):
          - Run the pipeline to completion (both branches arrived, barrier merged).
          - Undo the barrier: delete the merged token (outcome, node_states, token_parents,
            row) and both branches' COALESCED outcomes — leaving both branch tokens with no
            terminal outcome (pre-merge state).
          - HELD branch: its coalesce-node node_state is reverted to the held/open state
            (status='open', completed_at = NULL — the state CoalesceExecutor.accept's
            begin_node_state writes on arrival) and its real state_id is captured. It is
            then placed into a genuine CoalesceCheckpointState (pending.branches) carrying
            its real row_data + contract + state_id, passed to
            create_checkpoint(coalesce_state=). On resume, restore_from_checkpoint
            repopulates _pending with this branch — it is "already arrived", awaiting the
            sibling. The HELD branch is chosen as the one with the smaller token_id so it
            is dispatched FIRST on resume (specs order by step_in_pipeline, token_id), which
            makes the duplicate-arrival the deterministic RED (it re-arrives while it is the
            only _pending entry, before any merge could complete).
          - SIBLING branch: its coalesce-node node_state is DELETED (genuinely
            not-yet-arrived) and it is NOT in the checkpoint pending. It must re-drive.

        After this, _get_buffered_checkpoint_token_ids returns the held branch token
        (asserted as a precondition: proof the coalesce-pending arm is live). The
        incomplete-spec oracle (get_incomplete_tokens_by_row) returns BOTH branch tokens
        BEFORE the fix (UNFILTERED) — including the held branch — which is the defect.

        RED (fix absent — get_incomplete_tokens_by_row does NOT exclude buffered tokens):
        resume dispatches BOTH branch tokens to resume_incomplete_token. The held branch
        (dispatched first) is re-driven from its branch_first_node with coalesce context,
        re-arrives at the barrier via _maybe_coalesce_token → CoalesceExecutor.accept. But
        it is ALREADY in _pending['merge', row] (restored from the checkpoint), so accept()
        hits the duplicate-arrival guard (coalesce_executor.py ~562-570):
        Observed: OrchestrationInvariantError — "Duplicate arrival for branch 'path_a' at
        coalesce 'merge'. Existing token: <id>, new token: <id>. This indicates a bug in
        fork, retry, or checkpoint/resume logic." (existing == new token id: the held
        branch re-arriving on itself; branch name is whichever token_id sorts first). This
        is a DISTINCT exception from the I1a orphan AuditIntegrityError that the
        before-barrier cell produces — confirming this cell pins the duplicate-arrival
        defect specifically.

        GREEN (fix present — buffered tokens excluded from the incomplete-spec oracle):
        only the sibling re-drives; it arrives at the barrier where the restored held branch
        already waits; require_all is satisfied → barrier fires exactly once → merged token
        created → sink written once. The restored held branch's open node_state is completed
        by _execute_merge (status 'open' is non-terminal). Conservation holds, zero orphans,
        one outcome per leaf.
        """
        from elspeth.contracts.coalesce_checkpoint import (
            CoalesceCheckpointState,
            CoalescePendingCheckpoint,
            CoalesceTokenCheckpoint,
        )
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import token_outcomes_table, token_parents_table, tokens_table
        from elspeth.engine.coalesce_executor import COALESCE_CHECKPOINT_VERSION

        db, payload_store, config, graph, settings_obj, run_id, _run1 = self._setup_coalesce_pipeline()

        # ── Baseline (run-1 completed) ──────────────────────────────────────────
        def _outcome_counts() -> dict[tuple[str, str], int]:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND o.completed = 1
                          AND o.sink_name IS NOT NULL
                        GROUP BY t.row_id, o.sink_name
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return {(row.row_id, row.sink_name): row.n for row in result}

        baseline = _outcome_counts()
        assert len(baseline) == 1, f"Expected exactly one (row_id, sink_name) completed outcome; got {baseline}"
        assert all(n == 1 for n in baseline.values()), baseline

        # ── Locate the merged token and the two branch tokens ─────────────────────
        coalesce_node_id = str(graph.get_coalesce_id_map()[CoalesceName("merge")])
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.join_group_id AS join_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
            branch_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.branch_name AS branch_name, t.row_id AS row_id,
                           t.fork_group_id AS fork_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.branch_name IN ('path_a', 'path_b')
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token; got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        join_group_id = merged_rows[0].join_group_id
        by_branch = {b.branch_name: b for b in branch_rows}
        assert set(by_branch) == {"path_a", "path_b"}, f"expected path_a + path_b; got {set(by_branch)}"
        # Hold the branch that dispatches FIRST so the held-branch re-drive is the FIRST
        # coalesce arrival on resume — it re-arrives while it is the only _pending entry
        # (sibling not yet re-driven, no merge yet, _check_landscape_for_completion False),
        # deterministically hitting the duplicate-arrival guard rather than the
        # late-arrival path. get_incomplete_tokens_by_row orders specs by
        # (step_in_pipeline, token_id); both branches share the coalesce step, so dispatch
        # order is ascending token_id → hold the lexicographically-smallest token_id.
        held_branch = min(branch_rows, key=lambda b: b.token_id)  # held at barrier (restored into _pending)
        incomplete_branch = next(b for b in branch_rows if b.token_id != held_branch.token_id)  # not-yet-arrived; re-drives
        held_branch_name = held_branch.branch_name
        row_id = held_branch.row_id

        # ── Capture the held branch's coalesce-node state_id (real node_state) ─────
        # restore_from_checkpoint requires a valid state_id for the held branch's pending
        # node_state — reuse the genuine run-1 coalesce-node node_state for path_a.
        with db.engine.connect() as conn:
            held_state_row = conn.execute(
                text("""
                    SELECT state_id FROM node_states
                    WHERE token_id = :tid AND node_id = :nid AND run_id = :rid
                    ORDER BY attempt DESC LIMIT 1
                """),
                {"tid": held_branch.token_id, "nid": coalesce_node_id, "rid": run_id},
            ).fetchone()
        assert held_state_row is not None, (
            "Setup precondition violated: path_a must have a coalesce-node node_state from run-1 "
            "(CoalesceExecutor.accept calls begin_node_state on arrival)."
        )
        held_state_id = held_state_row.state_id

        # ── The contract the row was produced under (for the held branch's checkpoint) ──
        checkpoint_mgr = CheckpointManager(db)
        recovery_for_contract = RecoveryManager(db, checkpoint_mgr)
        source_contract = recovery_for_contract.verify_contract_integrity(run_id)
        contract_dict = source_contract.to_checkpoint_format()

        # ── Interrupt: undo the barrier; revert path_a to PENDING, drop path_b's arrival ──
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            # Merged token artifacts (the barrier's output)
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.token_id == merged_token_id))
            conn.execute(text("DELETE FROM node_states WHERE token_id = :tid"), {"tid": merged_token_id})
            conn.execute(token_parents_table.delete().where(token_parents_table.c.token_id == merged_token_id))
            conn.execute(tokens_table.delete().where(tokens_table.c.token_id == merged_token_id))
            # Branch COALESCED outcomes (recorded by the barrier on both branches)
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.join_group_id == join_group_id))
            # HELD branch (path_a): revert its coalesce-node node_state to the held/open
            # state (status='open', completed_at NULL) — the genuine pre-barrier state that
            # CoalesceExecutor.accept's begin_node_state writes on arrival (the barrier never
            # completed it because the crash happened before path_b arrived). 'open' is
            # non-terminal, so on GREEN _execute_merge can complete this state by state_id
            # when the barrier finally fires; deleting it would crash the completion write,
            # and leaving it 'completed' would hit the immutable-terminal guard.
            conn.execute(
                text("UPDATE node_states SET status = 'open', completed_at = NULL, output_hash = NULL WHERE state_id = :sid"),
                {"sid": held_state_id},
            )
            # SIBLING branch (path_b): delete its coalesce-node node_state entirely — it is
            # genuinely not-yet-arrived (pre-barrier crash before path_b reached the barrier).
            conn.execute(
                text("DELETE FROM node_states WHERE token_id = :tid AND node_id = :nid AND run_id = :rid"),
                {"tid": incomplete_branch.token_id, "nid": coalesce_node_id, "rid": run_id},
            )
            conn.commit()
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")

        # ── Build a GENUINE CoalesceCheckpointState holding path_a at the barrier ──
        held_token_ckpt = CoalesceTokenCheckpoint(
            token_id=held_branch.token_id,
            row_id=row_id,
            branch_name=held_branch_name,
            fork_group_id=held_branch.fork_group_id,
            join_group_id=None,
            expand_group_id=None,
            row_data={"value": 1},
            contract=contract_dict,
            state_id=held_state_id,
            arrival_offset_seconds=0.0,
        )
        coalesce_state = CoalesceCheckpointState(
            version=COALESCE_CHECKPOINT_VERSION,
            pending=(
                CoalescePendingCheckpoint(
                    coalesce_name="merge",
                    row_id=row_id,
                    elapsed_age_seconds=0.5,
                    branches={held_branch_name: held_token_ckpt},
                    lost_branches={},
                ),
            ),
            completed_keys=(),
        )

        # ── Create the checkpoint WITH the genuine coalesce pending state ──
        sink_node_ids = graph.get_sinks()
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=held_branch.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
            coalesce_state=coalesce_state,
        )
        with db.engine.connect() as conn:
            conn.execute(text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"), {"run_id": run_id})
            conn.commit()

        recovery_mgr = RecoveryManager(db, checkpoint_mgr)

        # ── PRECONDITION: the coalesce-pending arm is genuinely live ──
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        buffered_ids = recovery_mgr._get_buffered_checkpoint_token_ids(checkpoint)
        assert held_branch.token_id in buffered_ids, (
            f"PRECONDITION FAILED: the held path_a token must be in the checkpoint coalesce-pending "
            f"buffered set (else the held-branch exclusion path is skipped). buffered_ids={buffered_ids}"
        )
        assert incomplete_branch.token_id not in buffered_ids, (
            f"path_b must NOT be buffered (it is the not-yet-arrived sibling); buffered_ids={buffered_ids}"
        )

        # ── Resume ────────────────────────────────────────────────────────────────
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Conservation law ──────────────────────────────────────────────────────
        after = _outcome_counts()
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left {len(orphans)} non-delegation leaf token(s) with no terminal outcome (orphans): {orphans}"
        assert all(n == 1 for n in after.values()), f"No (row_id, sink_name) may carry two outcomes after resume: {after}"
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        post_stats = get_fork_group_stats(db, run_id)
        assert post_stats["total_fork_groups"] == 1, f"Fork-group count must remain 1 (one row, one fork); got {post_stats}"

    def test_resume_post_coalesce_before_downstream(self) -> None:
        """Re-driving a branchless merged token interrupted AFTER the barrier must
        reconstruct from token_data_ref, NOT restart-and-re-fork (B1 review finding).

        Topology: same as test_resume_fork_to_coalesce_before_barrier
            source → gate(fork) → [path_a: PassTransform, path_b: PassTransform]
                   → coalesce('merge', on_success='output') → sink 'output'

        B1 sub-case exercised: TERMINAL coalesce.
        resolve_next_node(coalesce_node_id) is None because 'on_success' routes the
        merged token directly to a sink (the sink is reached via resolve_coalesce_sink,
        not via the next-node map).  This exercises the `_terminal_coalesce_row_result`
        reconstruction path in resume_incomplete_token (Case 4, terminal branch).
        The NON-terminal sub-case (coalesce → downstream transform → sink, where
        resolve_next_node returns a real node) is NOT covered here — it is left for
        Task 12's resume matrix.

        Interruption: after a complete run, delete ONLY the merged token's terminal
        COMPLETED outcome at the sink.  The merged token row, its token_data_ref,
        token_parents links, and the branch COALESCED outcomes are all preserved.

        Resume-dispatch oracle (get_incomplete_tokens_by_row): must return exactly the
        merged token (join_group_id set, fork_group_id=None, branch_name=None).
        The branch children must NOT appear (they keep their COALESCED outcomes).
        This confirms dispatch Case 4 (post-coalesce merged token).

        RED (dispatch disabled, resume.py fork_expand_coalesce_specs branch commented):
        The merged token has join_group_id set but no branch_name — it is not a
        fork→sink child and not an expand child.  process_existing_row re-drives the
        ORIGINAL row from the source node, minting a fresh fork parent and new branch
        children.  Unlike Cell 1, the interruption here preserves the run-1 coalesce
        node_states (completed_at IS NOT NULL).  The fresh children arrive at the
        coalesce and CoalesceExecutor._check_landscape_for_completion returns True
        (run-1 node_states still present) → late-arrival path → UNROUTED outcomes.
        The new fork parent has its FORK_PARENT outcome AND child tokens with UNROUTED
        outcomes → no I1a sweep violation.  Instead, no (row_id, 'output') completed
        outcome with sink_name IS NOT NULL is ever recorded (UNROUTED outcomes have
        NULL sink_name), so the conservation assertion fires directly.
        Observed: AssertionError — "Resume must conserve the terminal-outcome
        multiset. baseline={(..., 'output'): 1} after={}".
        The RED mechanism is DIFFERENT from Cell 1 (conservation assertion, not I1a);
        the ORACLE is also different (one Case-4 merged-token spec vs two Case-2 branch
        specs), confirming the cells exercise distinct dispatch paths.

        GREEN (dispatch restored): the merged token's token_data_ref is retrieved and
        decoded as a {data, contract} envelope; the merged payload is re-delivered to
        the sink; one COMPLETED outcome is recorded; conservation holds.  The
        token_data_ref round-trip assertion (below) proves the B1 envelope path was
        exercised — not merely that sink received some data.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.serialization import checkpoint_loads
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db, payload_store, config, graph, settings_obj, run_id, _run1 = self._setup_coalesce_pipeline()

        # ── Baseline ──────────────────────────────────────────────────────────────
        def _outcome_counts() -> dict[tuple[str, str], int]:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND o.completed = 1
                          AND o.sink_name IS NOT NULL
                        GROUP BY t.row_id, o.sink_name
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return {(row.row_id, row.sink_name): row.n for row in result}

        baseline = _outcome_counts()
        assert len(baseline) == 1, f"Expected exactly one (row_id, sink_name) completed outcome; got {baseline}"
        assert all(n == 1 for n in baseline.values()), baseline

        # ── Verify TERMINAL sub-case: resolve_next_node(coalesce_node) is None ────
        # The 'merge' coalesce has on_success='output' (a sink); the DAG has no
        # processing node after the coalesce node in the traversal map.
        # This is the B1 terminal path: _terminal_coalesce_row_result is used.
        coalesce_node_id = graph.get_coalesce_id_map()[CoalesceName("merge")]
        # The coalesce node is TERMINAL if its only successors are sinks (leaf nodes
        # with no outgoing edges in the traversal map).  The authoritative check is
        # performed at resume time by the processor via resolve_next_node, but we can
        # confirm via the graph that the only outgoing edge from coalesce goes to the
        # sink — which is NOT a processing node.
        sink_node_ids = set(graph.get_sinks())
        edges = graph.get_edges()
        coalesce_successors = {e.to_node for e in edges if e.from_node == coalesce_node_id}
        assert coalesce_successors.issubset(sink_node_ids), (
            f"TERMINAL sub-case precondition: all successors of the coalesce node must "
            f"be sinks (so resolve_next_node returns None in the traversal map). "
            f"Got non-sink successors: {coalesce_successors - sink_node_ids}. "
            f"If any non-sink successor is present this exercises the NON-TERMINAL path — "
            f"update the test or the topology."
        )

        # ── Find the merged token ──────────────────────────────────────────────────
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.token_data_ref AS token_data_ref
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token; got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        merged_data_ref = merged_rows[0].token_data_ref
        assert merged_data_ref is not None, (
            "Merged token must have token_data_ref set (epoch 11 invariant). "
            "Without it the B1 re-drive cannot reconstruct the merged payload."
        )

        # ── Interrupt: delete only the merged token's terminal COMPLETED outcome ──
        # The branch COALESCED outcomes, the merged token row, and its token_data_ref
        # are all preserved.  This is the "crashed between barrier-fire and sink-write
        # completion" scenario.
        with db.engine.connect() as conn:
            merged_outcomes = conn.execute(
                text("""
                    SELECT outcome_id FROM token_outcomes
                    WHERE token_id = :tid AND completed = 1
                """),
                {"tid": merged_token_id},
            ).fetchall()
            assert merged_outcomes, (
                f"Setup precondition: merged token {merged_token_id!r} must have a completed outcome to delete; none found."
            )
            for o in merged_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        # ── Oracle: exactly the merged token as Case-4 spec ───────────────────────
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)

        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 1, (
            f"Oracle must return exactly 1 incomplete spec (merged token, Case 4); got {len(all_specs)}: {[s.token_id for s in all_specs]}"
        )
        spec = all_specs[0]
        assert spec.token_id == merged_token_id, f"Spec must be the merged token; got {spec.token_id!r} != {merged_token_id!r}"
        assert spec.join_group_id is not None, "Post-coalesce spec must have join_group_id set (Case 4)"
        assert spec.fork_group_id is None, "Post-coalesce spec must NOT have fork_group_id"
        assert spec.branch_name is None, "Post-coalesce spec must NOT have branch_name"
        assert spec.token_data_ref == merged_data_ref, (
            f"Recovery must surface the token_data_ref from the DB; got {spec.token_data_ref!r} != {merged_data_ref!r}"
        )

        # ── Resume ────────────────────────────────────────────────────────────────
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=next(iter(sink_node_ids)),
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Conservation law ──────────────────────────────────────────────────────
        after = _outcome_counts()
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left {len(orphans)} non-delegation leaf token(s) with no terminal outcome (orphans): {orphans}"
        assert all(n == 1 for n in after.values()), f"No (row_id, sink_name) may carry two outcomes after resume: {after}"
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        # ── B1 token-identity proof ───────────────────────────────────────────────
        # The B1 Case-4 path drives the SAME merged token in place (reuses its
        # existing token_id / join_group_id).  A re-fork would create a SECOND merged
        # token → 2 join tokens in the DB, different token_id.  This check distinguishes
        # them: if dispatch was disabled and process_existing_row re-forked, a SECOND
        # merged token would appear (different token_id) and this assertion would fail.
        with db.engine.connect() as conn:
            merged_after = conn.execute(
                text("""
                    SELECT t.token_id AS token_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_after) == 1, (
            f"B1: exactly ONE merged token must exist after resume; "
            f"got {len(merged_after)} — a second means process_existing_row re-forked "
            f"instead of re-driving the original: {[t.token_id for t in merged_after]}"
        )
        assert merged_after[0].token_id == merged_token_id, (
            f"B1: resume must reuse the SAME merged token (token_id={merged_token_id!r}); "
            f"got {merged_after[0].token_id!r} — different token_id means it re-forked "
            f"and re-merged instead of driving the original branchless token in place."
        )
        # The same merged token now has its terminal outcome recorded.
        with db.engine.connect() as conn:
            term_outcomes = conn.execute(
                text("""
                    SELECT path, sink_name FROM token_outcomes
                    WHERE token_id = :tid AND completed = 1
                """),
                {"tid": merged_token_id},
            ).fetchall()
        assert term_outcomes, (
            f"B1: merged token {merged_token_id!r} has no terminal outcome after resume; "
            f"the Case-4 reconstruction path must record a COALESCED outcome."
        )

        # ── token_data_ref round-trip (B1 envelope payload verification) ─────────
        # This proves the {data, contract} envelope stored in run-1 is well-formed and
        # carries the merged payload — not that resume *read* it (the token-identity
        # check above already proves reconstruction ran).
        raw_bytes = payload_store.retrieve(merged_data_ref)
        env = checkpoint_loads(raw_bytes.decode("utf-8"))

        assert isinstance(env, dict) and "data" in env and "contract" in env, (
            f"Merged token payload is not a {{data, contract}} envelope; "
            f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
        )
        # Content round-trip (not just shape): the union-merge of
        # path_a({'value': 1}) and path_b({'value': 1}) — both PassTransform
        # identity-mapped over the single source row {'value': 1} — produces
        # exactly {'value': 1} (union over identical-keyed branches → one 'value'
        # key, no extras). Empirically confirmed against the real merged token's
        # token_data_ref envelope. Asserting the FULL dict (not "value in data")
        # catches an empty/garbage/extra-keyed payload that a membership check
        # would silently pass.
        data = env["data"]
        assert data == {"value": 1}, f"merged payload round-trip mismatch: expected {{'value': 1}}, got {data!r}"

        # Contract round-trip: SchemaContract.from_checkpoint validates the
        # version_hash (Tier-1 integrity) and must restore without raising.
        from elspeth.contracts.schema_contract import SchemaContract

        restored_contract = SchemaContract.from_checkpoint(env["contract"])
        assert {fc.normalized_name for fc in restored_contract.fields} == {"value"}, (
            f"restored merged contract fields mismatch: expected {{'value'}}, "
            f"got {{{', '.join(repr(fc.normalized_name) for fc in restored_contract.fields)}}}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # F1 Regression Cells (Task 10) — expand resume + value-fidelity +
    # re-driven external-call attributability
    # ─────────────────────────────────────────────────────────────────────────

    def test_resume_partial_expand_does_not_reemit(self) -> None:
        """A deaggregation row with one completed and one incomplete expanded child resumes
        the incomplete child only — no re-expansion, no double-emit.

        Topology:
            source → JSONExplode(array_field='items', output_field='ts')
                   → PassTransform('pass_after_explode') → sink 'output'

        The PassTransform is required: expand-child resume re-drives from the node
        AFTER the expand node (resolve_next_node(explode_node)).  Without a downstream
        transform, that node is None and process_token rejects it (no sink context).
        PassTransform is the re-drive target — the expand child is resumed from there.

        Source provides one row: {'score': 1, 'items': [dt0, dt1]}.
        JSONExplode emits two children:
          child0: {'score': 1, 'ts': dt0, 'item_index': 0}
          child1: {'score': 1, 'ts': dt1, 'item_index': 1}

        Each child carries its own DISTINCT datetime value in 'ts' — the datetime
        is the ITEM VALUE from the exploded array (output_field='ts'), giving each
        child a different token_data_ref envelope payload.

        VALUE FIDELITY: each child's exploded item is a dict carrying its OWN
        distinct datetime and Decimal — child0 {ts: dt0, amount: Decimal('1.5')},
        child1 {ts: dt1, amount: Decimal('99.25')}.  After interrupting child1 and
        resuming, retrieve child1's token_data_ref envelope and assert its item
        carries ts=dt1 as a datetime.datetime INSTANCE (not str) AND amount=Decimal
        ('99.25') as a decimal.Decimal INSTANCE (not str, not float, not the sibling's
        1.5).  This proves checkpoint_dumps/_loads round-trips BOTH non-JSON-native
        types through the token_data_ref envelope with full fidelity, and that child1
        carries ITS OWN values (not child0's).

        Decimal is the F1-regression witness: checkpoint_dumps previously crashed on
        Decimal (canonical_json accepted it lossily), so the F1 envelope-persistence
        path regressed Decimal-bearing rows from lossy-but-works to a happy-path
        crash.  Asserting Decimal fidelity here proves that regression is fixed and a
        Decimal genuinely survives the expand-child token_data_ref round-trip.

        RED (dispatch disabled, fork_expand_coalesce_specs branch commented):
        process_existing_row re-runs the entire source row from the beginning,
        which triggers JSONExplode again and creates TWO NEW expand children in
        addition to the existing (run-1) children (total 4 expand children in DB).
        The conservation assertion fires first (before the I1a sweep).
        Observed: AssertionError — 'Resume must NOT re-expand (no new children
        minted): expected 2 expand children, got 4.'
        """
        import datetime
        from decimal import Decimal

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.serialization import checkpoint_loads
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import token_outcomes_table
        from elspeth.plugins.transforms.json_explode import JSONExplode

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        dt0 = datetime.datetime(2021, 1, 1, tzinfo=datetime.UTC)
        dt1 = datetime.datetime(2022, 2, 2, tzinfo=datetime.UTC)
        amount0 = Decimal("1.5")
        amount1 = Decimal("99.25")

        # Source: one row whose 'items' contains two distinct dicts, each carrying
        # its OWN datetime AND Decimal.  JSONExplode explodes this into child0
        # (item={ts: dt0, amount: 1.5}, item_index=0) and child1 (item={ts: dt1,
        # amount: 99.25}, item_index=1).  Each child's 'item' IS the exploded dict —
        # distinct per child — proving value<->token alignment for BOTH types.
        #
        # TOPOLOGY: source → JSONExplode → PassTransform → sink
        # The PassTransform after JSONExplode is required for resume: the expand child's
        # resume_incomplete_token path calls process_token(current_node_id=after) where
        # `after = resolve_next_node(explode_node)`.  Without a downstream transform node,
        # `after` would be None, which process_token rejects (no sink context for branchless
        # tokens).  The PassTransform is the re-drive target — the child continues from there
        # rather than from the expand node itself.
        source = ListSource(
            [{"score": 1, "items": [{"ts": dt0, "amount": amount0}, {"ts": dt1, "amount": amount1}]}],
            on_success="explode_in",
        )
        sink = CollectSink("output")

        explode = JSONExplode(
            {
                "array_field": "items",
                "output_field": "item",  # each child gets its own {ts, amount} dict as 'item'
                "include_index": True,  # item_index 0/1 identifies which child
                "schema": {"mode": "observed"},
            }
        )
        # PassTransform downstream of explode — required for expand-child resume.
        pass_after = PassTransform(name="pass_after_explode")

        wired_transforms = wire_transforms(
            [explode, pass_after],
            source_connection="explode_in",
            final_sink="output",
            names=["explode", "pass_after"],
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="explode_in", options={}),
            transforms=wired_transforms,
            sinks={"output": as_sink(sink)},
            gates=[],
            aggregations={},
            coalesce_settings=[],
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(explode), as_transform(pass_after)],
            sinks={"output": as_sink(sink)},
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "explode_in", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # ── Baseline: both expand children completed (2 outcomes at 'output') ──
        with db.engine.connect() as conn:
            all_children = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.token_data_ref AS token_data_ref
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.expand_group_id IS NOT NULL
                    ORDER BY t.token_id
                """),
                {"run_id": run_id},
            ).fetchall()

        assert len(all_children) == 2, f"Expected exactly 2 expand children from a 2-element array; got {len(all_children)}"

        # Identify child1 (item_index=1) by reading its token_data_ref envelope.
        # We interrupt child1 (leave child0 complete) to simulate a mid-run crash.
        def _get_item_index(token_data_ref: str) -> int:
            raw = payload_store.retrieve(token_data_ref)
            env = checkpoint_loads(raw.decode("utf-8"))
            return env["data"]["item_index"]

        child_by_index: dict[int, str] = {_get_item_index(c.token_data_ref): c.token_id for c in all_children}
        assert set(child_by_index.keys()) == {0, 1}, f"Expected children with item_index 0 and 1; got indices {set(child_by_index.keys())}"
        incomplete_token_id = child_by_index[1]  # interrupt child1 (ts=dt1)

        # Pre-record child1's token_data_ref for value-fidelity assertion after resume.
        child1_token_data_ref = next(c.token_data_ref for c in all_children if c.token_id == incomplete_token_id)
        assert child1_token_data_ref is not None, "expand child1 must have token_data_ref before interruption (epoch 11 invariant)"

        # ── Interrupt: delete child1's terminal outcome only ──────────────────
        with db.engine.connect() as conn:
            child1_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    WHERE o.token_id = :tid AND o.completed = 1
                """),
                {"tid": incomplete_token_id},
            ).fetchall()
            assert child1_outcomes, "expand child1 must have a completed outcome to delete (setup precondition)"
            for outcome in child1_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        # ── Checkpoint + mark failed ──────────────────────────────────────────
        checkpoint_mgr = CheckpointManager(db)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            any_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert any_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=any_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        # ── Resume ───────────────────────────────────────────────────────────
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        # ── Conservation: expand child count unchanged (no re-expansion) ──────
        with db.engine.connect() as conn:
            post_children = conn.execute(
                text("""
                    SELECT t.token_id AS token_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.expand_group_id IS NOT NULL
                """),
                {"run_id": run_id},
            ).fetchall()

        assert len(post_children) == 2, (
            f"Resume must NOT re-expand (no new children minted): "
            f"expected 2 expand children, got {len(post_children)}. "
            f"If dispatch is disabled, process_existing_row re-runs JSONExplode and "
            f"creates 2 additional children → total 4."
        )

        # Each expand child must have exactly one completed outcome (no double-emit).
        with db.engine.connect() as conn:
            outcome_counts_by_token = {
                row.token_id: row.n
                for row in conn.execute(
                    text("""
                        SELECT o.token_id AS token_id, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND t.expand_group_id IS NOT NULL
                          AND o.completed = 1
                        GROUP BY o.token_id
                    """),
                    {"run_id": run_id},
                ).fetchall()
            }

        child_token_ids = {c.token_id for c in post_children}
        for token_id in child_token_ids:
            count = outcome_counts_by_token.get(token_id, 0)
            assert count == 1, (
                f"Expand child {token_id!r} has {count} completed outcomes after resume; "
                f"expected exactly 1 (conservation law — no double-emit). "
                f"incomplete_token_id={incomplete_token_id!r}"
            )

        # No orphan leaf tokens (all non-delegation tokens have a terminal outcome).
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left {len(orphans)} orphan leaf token(s): {orphans}"

        # ── VALUE FIDELITY: child1's envelope carries dt1 + Decimal('99.25') ──
        # Retrieve child1's token_data_ref envelope (run-1 artifact, unmodified by resume).
        # This is the payload the resume path's reconstruct_token_row() consumed to
        # re-drive child1.  Asserting type+value here proves:
        #   (a) checkpoint_dumps preserved the datetime type (not stringified it),
        #   (b) checkpoint_dumps preserved the Decimal type (the F1-regression witness:
        #       Decimal previously crashed the serializer on the happy path),
        #   (c) child1's envelope carries ITS OWN values (dt1/99.25), not the sibling's
        #       (dt0/1.5).
        raw = payload_store.retrieve(child1_token_data_ref)
        env = checkpoint_loads(raw.decode("utf-8"))

        assert isinstance(env, dict) and "data" in env and "contract" in env, (
            f"child1 token_data_ref is not a {{data, contract}} envelope; "
            f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
        )
        child1_data = env["data"]
        child1_item = child1_data["item"]

        # Type fidelity: 'ts' must come back as datetime.datetime, not str.
        assert isinstance(child1_item["ts"], datetime.datetime), (
            f"Type fidelity failure: 'ts' came back as {type(child1_item['ts']).__name__!r}, "
            f"not datetime — checkpoint_dumps was not used (canonical_json stringifies datetime)"
        )
        assert child1_item["ts"].tzinfo is not None, "Restored datetime must be timezone-aware"

        # Type fidelity: 'amount' must come back as decimal.Decimal, not str or float.
        # This is the F1-regression assertion — checkpoint_dumps previously crashed here.
        assert isinstance(child1_item["amount"], Decimal), (
            f"Type fidelity failure: 'amount' came back as {type(child1_item['amount']).__name__!r}, "
            f"not Decimal — checkpoint_dumps/_loads must round-trip Decimal through the "
            f"token_data_ref envelope (F1 envelope-fidelity regression guard)"
        )

        # Value alignment: child1 carries ITS OWN values (dt1/99.25), not the sibling's.
        assert child1_item["ts"] == dt1, (
            f"Value alignment failure: child1 carries ts={child1_item['ts']!r}, "
            f"expected dt1={dt1!r}. Got dt0={dt0!r} would mean zip(strict=True) alignment is broken."
        )
        assert child1_item["amount"] == amount1, (
            f"Value alignment failure: child1 carries amount={child1_item['amount']!r}, "
            f"expected amount1={amount1!r}. Got amount0={amount0!r} would mean alignment is broken."
        )
        # item_index alignment: child1 must have item_index=1.
        assert child1_data["item_index"] == 1, f"item_index alignment: expected 1 for child1, got {child1_data['item_index']!r}"

    def test_resume_redriven_transform_external_call_is_attributable(self) -> None:
        """A fork→coalesce branch whose transform makes a recorded state call is
        re-driven on resume; the call RE-FIRES (at-least-once) but the re-fired
        call's parent node_state carries resume_checkpoint_id, so an auditor can
        prove it came from the resume — the duplication is honest, not silent.

        Topology:
            source → gate(fork_to=[path_a, path_b])
                   → path_a: CallRecordingTransform(record_a) → coalesce 'merge'
                   → path_b: PassTransform(pass_b)            → coalesce 'merge'
                   → sink 'output'

        CallRecordingTransform records one HTTP state call (calls.state_id set)
        per invocation via ctx.record_call().  Being state-parented, the call
        inherits the resume_checkpoint_id of its parent node_state.

        Interruption: after a complete run, undo the barrier by deleting:
          - the merged token's outcome, node_states, token_parents, and token row;
          - both branch COALESCED outcomes;
          - branch tokens' completed coalesce node_states.
        This simulates a pre-barrier crash (both branches must re-drive).
        record_a's transform re-drives on resume → a SECOND state call is recorded.

        ATTRIBUTABILITY PROOF:
          - Run-1 node_state for record_a's transform: resume_checkpoint_id IS NULL
            (it was a first-pass execution, not a re-drive).
          - Resume node_state for record_a's transform (attempt = max+1):
            resume_checkpoint_id IS NOT NULL (set by the resume path, Task 5/7).
          - The re-fired call links to the resume node_state (calls.state_id).
          - Together these prove: the second call is provably attributable to the
            resume, not a silent duplicate appearing to be a run-1 call.

        Linkage used: calls.state_id → node_states.resume_checkpoint_id.
        operation_call (operations-table) is NOT used here — transforms produce
        state-parented calls; resume_checkpoint_id lives only on node_states.

        RED path A — dispatch disabled (fork_expand_coalesce_specs branch commented):
        process_existing_row re-forks the row from the source, creating NEW branch
        children.  The coalesce node_states from run-1 remain (completed_at IS NOT
        NULL).  The NEW branches arrive at the barrier, CoalesceExecutor.accept()
        sees them as late arrivals (already completed per run-1 node_states) and
        records UNROUTED outcomes.  The new fork parent carries FORK_PARENT outcome
        and its NEW children carry UNROUTED — no I1a sweep violation.  But the
        original run-1 branch tokens (which had their COALESCED outcomes deleted)
        are never re-driven by the disabled dispatch, so they remain in the DB
        with no terminal outcomes → I1a fires on those.
        Observed A: AuditIntegrityError — 'fork/expand parent token(s) have no child
        token_outcomes rows at run-end.'

        RED path B — stamp disabled (resume_checkpoint_id=None in resume_incomplete_token):
        Dispatch ON, re-drive ON; record_a's transform re-fires and its new node_state
        is written at attempt=max+1.  But with resume_checkpoint_id forced to None,
        the re-drive node_state is indistinguishable from a run-1 entry.
        Observed B: AssertionError — 'Re-drive node_state (attempt=1) must have
        resume_checkpoint_id set (attributability invariant, ADDENDUM 2.C); got None.'
        This confirms the attributability assertion has independent teeth beyond the
        dispatch-disable path.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import (
            token_outcomes_table,
            token_parents_table,
            tokens_table,
        )
        from tests.fixtures.plugins import CallRecordingTransform

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 1}], on_success="gate_in")
        sink = CollectSink("output")

        record_a = CallRecordingTransform(name="record_a")
        pass_b = PassTransform(name="pass_b")

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            on_success="output",
        )

        wired_a = wire_transforms([record_a], source_connection="path_a", final_sink="done_a", names=["record_a"])
        wired_b = wire_transforms([pass_b], source_connection="path_b", final_sink="done_b", names=["pass_b"])

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=wired_a + wired_b,
            sinks={"output": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(record_a), as_transform(pass_b)],
            sinks={"output": as_sink(sink)},
            gates=[gate],
            coalesce_settings=[coalesce],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "gate_in", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # ── Find the record_a branch token and its run-1 node_state ──────────
        # graph.get_transform_id_map() uses integer keys (sequential index), not names.
        # Find the node_id for record_a by searching graph.get_nodes() by plugin_name.
        record_a_node_id = next(n.node_id for n in graph.get_nodes() if n.plugin_name == "record_a")

        with db.engine.connect() as conn:
            record_a_run1_ns = conn.execute(
                text("""
                    SELECT ns.state_id AS state_id,
                           ns.attempt AS attempt,
                           ns.resume_checkpoint_id AS resume_checkpoint_id
                    FROM node_states ns
                    JOIN tokens t ON t.token_id = ns.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND ns.node_id = :node_id
                    ORDER BY ns.attempt
                """),
                {"run_id": run_id, "node_id": str(record_a_node_id)},
            ).fetchall()

        assert len(record_a_run1_ns) == 1, f"Expected 1 run-1 node_state for record_a transform; got {len(record_a_run1_ns)}"
        run1_state_id = record_a_run1_ns[0].state_id
        assert record_a_run1_ns[0].resume_checkpoint_id is None, (
            f"Run-1 node_state must have resume_checkpoint_id IS NULL (not a re-drive); got {record_a_run1_ns[0].resume_checkpoint_id!r}"
        )

        # ── Verify run-1 recorded exactly one state call ──────────────────────
        with db.engine.connect() as conn:
            run1_calls = conn.execute(
                text("SELECT call_id FROM calls WHERE state_id = :sid"),
                {"sid": run1_state_id},
            ).fetchall()

        assert len(run1_calls) == 1, f"Run-1 record_a transform must have recorded exactly 1 state call; got {len(run1_calls)}"

        # ── Find the merged token for barrier reversal ─────────────────────────
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.join_group_id AS join_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token; got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        join_group_id = merged_rows[0].join_group_id
        coalesce_node_id = graph.get_coalesce_id_map()[CoalesceName("merge")]

        # ── Interrupt: undo the barrier (same pattern as test_resume_fork_to_coalesce_before_barrier) ──
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            # 1. merged token outcomes
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.token_id == merged_token_id))
            # 2. merged token node_states
            conn.execute(
                text("DELETE FROM node_states WHERE token_id = :tid"),
                {"tid": merged_token_id},
            )
            # 3. token_parents for merged token
            conn.execute(token_parents_table.delete().where(token_parents_table.c.token_id == merged_token_id))
            # 4. merged token row
            conn.execute(tokens_table.delete().where(tokens_table.c.token_id == merged_token_id))
            # 5. branch COALESCED outcomes
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.join_group_id == join_group_id))
            # 6. branch tokens' completed coalesce node_states
            conn.execute(
                text("DELETE FROM node_states WHERE node_id = :nid AND run_id = :rid"),
                {"nid": str(coalesce_node_id), "rid": run_id},
            )
            conn.commit()
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")

        # ── Checkpoint + mark failed ──────────────────────────────────────────
        checkpoint_mgr = CheckpointManager(db)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            any_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert any_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=any_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        # ── Resume ───────────────────────────────────────────────────────────
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        # ── Attributability proof: re-driven node_state has resume_checkpoint_id ──
        # After resume, the record_a branch re-drove at attempt = max+1 (the resume
        # attempt).  Its node_state must carry resume_checkpoint_id IS NOT NULL.
        with db.engine.connect() as conn:
            record_a_all_ns = conn.execute(
                text("""
                    SELECT ns.state_id AS state_id,
                           ns.attempt AS attempt,
                           ns.resume_checkpoint_id AS resume_checkpoint_id
                    FROM node_states ns
                    JOIN tokens t ON t.token_id = ns.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND ns.node_id = :node_id
                    ORDER BY ns.attempt
                """),
                {"run_id": run_id, "node_id": str(record_a_node_id)},
            ).fetchall()

        assert len(record_a_all_ns) == 2, (
            f"After resume, record_a transform must have 2 node_states (run-1 + re-drive); "
            f"got {len(record_a_all_ns)} with attempts={[r.attempt for r in record_a_all_ns]!r}"
        )
        run1_ns = record_a_all_ns[0]
        redrive_ns = record_a_all_ns[1]

        # Run-1 node_state must still have resume_checkpoint_id IS NULL (append-only).
        assert run1_ns.state_id == run1_state_id, f"First node_state must be run-1 (state_id={run1_state_id!r}); got {run1_ns.state_id!r}"
        assert run1_ns.resume_checkpoint_id is None, (
            f"Run-1 node_state resume_checkpoint_id must remain NULL after resume; got {run1_ns.resume_checkpoint_id!r}"
        )

        # Re-drive node_state must have resume_checkpoint_id IS NOT NULL.
        assert redrive_ns.resume_checkpoint_id is not None, (
            f"Re-drive node_state (attempt={redrive_ns.attempt}) must have "
            f"resume_checkpoint_id set (attributability invariant, ADDENDUM 2.C); "
            f"got None.  The re-fired call cannot be attributed to the resume without this."
        )

        # ── Call count: the call RE-FIRED (at-least-once, bounded non-goal) ──
        # The re-driven node_state must have recorded a NEW state call.
        with db.engine.connect() as conn:
            redrive_calls = conn.execute(
                text("SELECT call_id FROM calls WHERE state_id = :sid"),
                {"sid": redrive_ns.state_id},
            ).fetchall()

        assert len(redrive_calls) == 1, (
            f"Re-drive node_state (state_id={redrive_ns.state_id!r}) must have exactly 1 re-fired state call; got {len(redrive_calls)}"
        )

        # Confirm total calls for record_a's transform: 1 (run-1) + 1 (re-drive) = 2.
        with db.engine.connect() as conn:
            all_record_a_calls = conn.execute(
                text("""
                    SELECT c.call_id AS call_id, c.state_id AS state_id
                    FROM calls c
                    WHERE c.state_id IN (
                        SELECT ns.state_id FROM node_states ns
                        JOIN tokens t ON t.token_id = ns.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id AND ns.node_id = :node_id
                    )
                """),
                {"run_id": run_id, "node_id": str(record_a_node_id)},
            ).fetchall()

        assert len(all_record_a_calls) == 2, (
            f"Total calls for record_a transform must be 2 (run-1 + re-drive = at-least-once); "
            f"got {len(all_record_a_calls)}.  "
            f"run1_state={run1_state_id!r}, redrive_state={redrive_ns.state_id!r}"
        )
        # Confirm one call per node_state (run-1 → run1_state_id, re-drive → redrive_ns.state_id).
        calls_by_state = {}
        for c in all_record_a_calls:
            calls_by_state.setdefault(c.state_id, []).append(c.call_id)
        assert run1_state_id in calls_by_state, "Run-1 state must have a call"
        assert redrive_ns.state_id in calls_by_state, "Re-drive state must have a call"

    # ─────────────────────────────────────────────────────────────────────
    # Task 11: F1/F2 counter-field reconciliation guard
    # ─────────────────────────────────────────────────────────────────────

    def test_resume_counters_reconcile_with_uninterrupted_run(self) -> None:
        """After (run1 + resume), EVERY RunResult counter equals a single
        uninterrupted run — field-for-field reconciliation (F1/F2).

        WHAT THIS TEST GUARDS:

        Full counter-field reconciliation — RESUME PATH == UNINTERRUPTED
        ----------------------------------------------------------------
        A resumed run (run1 interrupted mid-fork + resume re-drives the
        incomplete branch) must produce a RunResult whose counter fields are
        identical to an uninterrupted run of the same pipeline.  This exercises
        the "with-unprocessed-rows" (fork-re-drive) branch of ``resume()``.

        F2 fix (resume-fork-reemit) — UNIFY both resume branches on audit
        ------------------------------------------------------------------
        Pre-F2 the with-unprocessed-rows branch returned *resume-only* counters
        (only what THIS resume call reprocessed), while the no-unprocessed-rows
        branch reconstructed *cumulative* counters from ``token_outcomes`` via
        ``derive_resume_terminal_status_from_audit``.  A single RunResult type
        whose counter semantics depended on an invisible branch was a latent
        correctness trap: a resumed 1-source-row 2-branch fork reported
        ``rows_succeeded=1, rows_forked=0`` (only the re-driven sink_a leaf)
        instead of the cumulative ``2, 1``.  F2 made BOTH branches finalize the
        SAME audit-derived cumulative ``(status, counters)`` — the audit trail
        is the source of truth.

        rows_processed reconstruction — per SOURCE ROW, not per token
        --------------------------------------------------------------
        ``rows_processed`` is reconstructed as the count of DISTINCT source
        ``row_id`` reaching a terminal outcome (see
        ``QueryRepository.count_distinct_source_rows_with_terminal_outcome``),
        NOT a per-terminal-token tally.  This matches the live loops, which
        increment ``rows_processed`` once per source row regardless of fork
        fan-out, aggregation fan-in, or expand fan-out.  For this 1-source-row
        fork, both A and B report ``rows_processed == 1``.

        WHAT IS ASSERTED:
        - EVERY RunResult counter field (rows_processed, rows_succeeded,
          rows_failed, rows_routed_success, rows_routed_failure,
          rows_quarantined, rows_forked, rows_coalesced, rows_coalesce_failed,
          rows_expanded, rows_buffered, rows_diverted) is equal between the
          uninterrupted run A and the resumed run B — no field left resume-scoped
        - routed_destinations dict reconciles
        - Terminal-outcome multiset is conserved (regression guard, re-verified)
        - Resume result is COMPLETED (run is healthy)
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings, ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        # ── Run A (uninterrupted) ─────────────────────────────────────────
        db_a = make_landscape_db()
        payload_store_a = MockPayloadStore()

        source_a = ListSource([{"value": 1}], on_success="sink_a")
        sink_a_a = CollectSink("sink_a")
        sink_b_a = CollectSink("sink_b")
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )
        config_a = PipelineConfig(
            source=as_source(source_a),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a_a), "sink_b": as_sink(sink_b_a)},
            gates=[gate],
        )
        graph_a = ExecutionGraph.from_plugin_instances(
            source=as_source(source_a),
            source_settings=SourceSettings(plugin=source_a.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a_a), "sink_b": as_sink(sink_b_a)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_a = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orch_a = Orchestrator(db_a)
        run_a = orch_a.run(config_a, graph=graph_a, settings=settings_a, payload_store=payload_store_a)

        # Sanity: uninterrupted fork of 1 source row → rows_processed = 1
        assert run_a.rows_processed == 1, (
            f"Run A (uninterrupted, 1-row fork): expected rows_processed=1 (per source row), got {run_a.rows_processed}"
        )
        assert run_a.status == RunStatus.COMPLETED, run_a.status

        # ── Run B: run-1 then interrupt one fork branch, then resume ──────
        db_b = make_landscape_db()
        payload_store_b = MockPayloadStore()

        source_b = ListSource([{"value": 1}], on_success="sink_a")
        sink_a_b = CollectSink("sink_a")
        sink_b_b = CollectSink("sink_b")
        config_b = PipelineConfig(
            source=as_source(source_b),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a_b), "sink_b": as_sink(sink_b_b)},
            gates=[gate],
        )
        graph_b = ExecutionGraph.from_plugin_instances(
            source=as_source(source_b),
            source_settings=SourceSettings(plugin=source_b.name, on_success="gate_in", options={}),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a_b), "sink_b": as_sink(sink_b_b)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_b = ElspethSettings(
            source={"plugin": "test", "on_success": "sink_a", "options": {}},
            sinks={
                "sink_a": {"plugin": "test", "on_write_failure": "discard"},
                "sink_b": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[gate],
        )

        orch_b = Orchestrator(db_b)
        run_b1 = orch_b.run(config_b, graph=graph_b, settings=settings_b, payload_store=payload_store_b)
        run_id = run_b1.run_id

        # Interrupt: delete sink_a branch terminal outcome only (leaving node_states).
        with db_b.engine.connect() as conn:
            sink_a_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.sink_name = 'sink_a'
                """),
                {"run_id": run_id},
            ).fetchall()
            assert sink_a_outcomes, "expected a sink_a outcome to delete (setup precondition)"
            for outcome in sink_a_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

        # Create checkpoint + mark run failed (resume preconditions).
        checkpoint_mgr = CheckpointManager(db_b)
        sink_node_ids = graph_b.get_sinks()
        with db_b.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph_b,
        )
        with db_b.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        recovery_mgr = RecoveryManager(db_b, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph_b)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph_b)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orch = Orchestrator(db_b, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        run_b_resume = resume_orch.resume(resume_point, config_b, graph_b, payload_store=payload_store_b, settings=settings_b)

        # ── COUNTER-FIELD ASSERTIONS ──────────────────────────────────────

        # rows_processed: BOTH paths count per source row.
        # This is the primary F2 guard: the resume loop's ``rows_processed += 1``
        # is outside the per-spec dispatch loop, so even multi-spec rows
        # increment by exactly 1.  Uninterrupted A and resume B agree.
        assert run_b_resume.rows_processed == run_a.rows_processed, (
            f"rows_processed must equal uninterrupted run (both per source row): "
            f"A={run_a.rows_processed}, B={run_b_resume.rows_processed}. "
            f"If this fails, the resume loop is incrementing rows_processed per leaf "
            f"(per IncompleteTokenSpec) instead of per source row — an F2 regression."
        )

        # rows_processed must be 1 for our 1-source-row test pipeline (regression pin).
        assert run_b_resume.rows_processed == 1, (
            f"1 source row → rows_processed must be 1 (per-source-row invariant); got {run_b_resume.rows_processed}"
        )

        # Resume result must be COMPLETED (the fork row was fully resolved).
        assert run_b_resume.status == RunStatus.COMPLETED, (
            f"Resume of a 1-row fork pipeline must reach COMPLETED; got {run_b_resume.status}"
        )

        # Terminal-outcome multiset is conserved (conservation law, re-verified here
        # alongside counter assertions).
        def _outcome_counts(db: LandscapeDB, run_id: str) -> dict[tuple[str, str], int]:
            with db.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id
                          AND o.completed = 1
                          AND o.sink_name IS NOT NULL
                        GROUP BY t.row_id, o.sink_name
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return {(row.row_id, row.sink_name): row.n for row in result}

        a_outcomes = _outcome_counts(db_a, run_a.run_id)
        b_outcomes = _outcome_counts(db_b, run_id)
        # outcome multisets agree per-sink-count (ignoring different row_id UUIDs)
        a_sink_counts = sorted(v for v in a_outcomes.values())
        b_sink_counts = sorted(v for v in b_outcomes.values())
        assert a_sink_counts == b_sink_counts, (
            f"Terminal-outcome count per (row_id, sink_name) must match uninterrupted run. A={a_outcomes}, B={b_outcomes}"
        )
        assert all(n == 1 for n in b_outcomes.values()), (
            f"Every (row_id, sink_name) must have exactly 1 completed outcome after resume; got {b_outcomes}"
        )

        # F2 (resume-fork-reemit) — FULL COUNTER-FIELD RECONCILIATION.
        #
        # Both resume branches now finalize cumulative (status, counters) from
        # the audit trail via derive_resume_terminal_status_from_audit, so a
        # resumed run's RunResult matches an uninterrupted run FIELD-FOR-FIELD.
        # Before F2 this branch returned resume-only counters: a resumed 1-row
        # 2-branch fork reported rows_succeeded=1, rows_forked=0 (only the
        # re-driven sink_a leaf) instead of the cumulative 2, 1.  That divergence
        # is now eliminated — the audit trail is the source of truth, and a
        # single RunResult type no longer carries branch-dependent counter
        # semantics.
        #
        # Sanity-pin the cumulative truth for this 1-source-row 2-branch fork
        # (one success leaf per branch, one structural fork parent):
        assert run_a.rows_succeeded == 2, f"Uninterrupted 1-row 2-branch fork: rows_succeeded must be 2; got {run_a.rows_succeeded}"
        assert run_a.rows_forked == 1, f"Uninterrupted 1-row fork: rows_forked must be 1; got {run_a.rows_forked}"

        # Field-by-field equality across EVERY RunResult counter — no field is
        # left resume-scoped (derive reconstructs all of them faithfully:
        # rows_processed via distinct source row_id, the rest via per-terminal
        # tally over token_outcomes).
        #
        # SCOPE CAVEAT — rows_coalesce_failed (F2 review item A/D): this fixture
        # has rows_coalesce_failed == 0 on BOTH sides, so that field reconciles
        # vacuously here. The audit trail does NOT record a queryable signal for
        # coalesce-operation failures (telemetry-only roll-up), so derive() can
        # only ever return 0 for it; the with-rows branch grafts the live
        # re-drive counter back over that 0 (see resume() in core.py). The
        # reconciliation loop's rows_coalesce_failed assertion is therefore only
        # meaningful for coalesce failures that occur DURING THIS RESUME's
        # re-drive — run-1 (pre-interrupt) coalesce failures were live-counter-
        # only, are never re-driven, and are OUT of reconciliation scope (operator
        # follow-up: make rows_coalesce_failed a queryable terminal audit signal).
        counter_fields = (
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        )
        for field in counter_fields:
            a_val = getattr(run_a, field)
            b_val = getattr(run_b_resume, field)
            assert b_val == a_val, (
                f"F2 reconciliation failure on '{field}': resumed run (run1 + resume) must equal "
                f"the uninterrupted run field-for-field. uninterrupted={a_val}, resumed={b_val}. "
                f"Both resume branches finalize cumulative counters from the audit trail; a divergence "
                f"means either the with-rows branch regressed to resume-only counters or "
                f"derive_resume_terminal_status_from_audit miscounts this field."
            )

        # routed_destinations (the per-sink dict) must also reconcile.
        assert dict(run_b_resume.routed_destinations) == dict(run_a.routed_destinations), (
            f"F2 reconciliation failure on routed_destinations: "
            f"uninterrupted={dict(run_a.routed_destinations)}, resumed={dict(run_b_resume.routed_destinations)}"
        )

    def test_resume_coalesced_counter_reconciles_with_uninterrupted_run(self) -> None:
        """A resumed coalesce-SUCCESS run reconciles EVERY counter field —
        including rows_coalesced > 0 (non-vacuous) — with an uninterrupted run.

        WHY THIS CELL EXISTS (regression-detection gap closed):

        ``test_resume_counters_reconcile_with_uninterrupted_run`` reconciles all
        12 RunResult counter fields, but on a fork→sink topology where
        ``rows_coalesced == 0`` on BOTH sides — so that field reconciled only
        VACUOUSLY (0 == 0).  The integration-level
        ``tests/integration/test_adr_019_resume_counter_parity.py`` snapshot
        tests use non-coalesce topologies, so ``rows_coalesced`` was likewise
        0 == 0 everywhere.  Concrete uncovered regression: deleting
        ``counters.rows_coalesced += 1`` from the ``(SUCCESS, COALESCED)`` arm in
        ``src/elspeth/engine/orchestrator/run_status.py`` failed NO test — the
        co-incremented ``rows_succeeded`` keeps the run COMPLETED and nothing
        asserted ``rows_coalesced`` non-zero on a RESUMED run.  This cell closes
        that gap: it reconciles all 12 fields on a coalesce-SUCCESS topology
        where ``rows_coalesced >= 1``.

        TOPOLOGY (shared ``_setup_coalesce_pipeline``):
            source(1 row) → gate(fork) → [path_a: PassTransform,
                                          path_b: PassTransform]
                   → coalesce('merge', require_all, on_success='output')
                   → sink 'output'
        The coalesce SUCCEEDS (require_all quorum met by both branches) → its
        merged token reaches a ``(SUCCESS, COALESCED)`` terminal outcome, so
        ``rows_coalesced == 1`` (non-vacuous).

        RUN A (uninterrupted oracle): a SEPARATE ``_setup_coalesce_pipeline``
        instance, NOT interrupted.  Its counters come from the LIVE accumulator
        (a different code path from ``derive_resume_terminal_status_from_audit``).

        RUN B (run-1 + interrupt-before-barrier + resume): the SAME topology;
        run-1 completes, then the barrier is undone (merged token + branch
        COALESCED outcomes + coalesce node_states deleted — verbatim the
        before-barrier interrupt from ``test_resume_fork_to_coalesce_before_barrier``),
        leaving both branch children incomplete.  Resume exercises the
        WITH-UNPROCESSED-ROWS (fork-re-drive) branch: it re-drives both branch
        specs to the barrier, the barrier re-fires exactly once, a merged token
        is re-created with a ``(SUCCESS, COALESCED)`` outcome, and
        ``derive_resume_terminal_status_from_audit`` reconstructs the cumulative
        counters from the audit trail.

        ``rows_coalesced`` is reconstructed purely by ``derive`` (line 124 of
        run_status.py — it is NOT grafted from the live re-drive counter; only
        ``rows_coalesce_failed`` is grafted, see ``resume()`` in core.py).  So
        the line-124 lever bites the RESUMED run B (derive-reconstructed) while
        run A's value comes from the live accumulator.

        OBSERVED RED (lever: delete the ``counters.rows_coalesced += 1`` line
        from the ``(SUCCESS, COALESCED)`` arm in run_status.py, leaving the
        co-incremented ``counters.rows_succeeded += 1`` so status stays
        COMPLETED and the signal is isolated to rows_coalesced; run
        ``-k test_resume_coalesced_counter_reconciles_with_uninterrupted_run``):
            AssertionError: Resumed coalesce-success run must record at least one
            COALESCED terminal outcome (non-vacuous); got rows_coalesced=0.
            If 0, the barrier did not re-fire on resume or derive miscounts
            COALESCED.  assert 0 >= 1
        Run A (live accumulator) reports rows_coalesced=1; run B
        (derive-reconstructed, lever removed) reports 0 — a 1-vs-0 mismatch, not
        a both-zero vacuity (the non-vacuity guard fires first; the field-loop
        ``uninterrupted=1, resumed=0`` mismatch is the same signal).  Reverting
        the lever restores GREEN.

        NOTE (derive fix co-landed): GREEN-first on this cell uncovered a
        resume-independent bug in ``derive_resume_terminal_status_from_audit``:
        its ``(SUCCESS, COALESCED)`` arm counted EVERY COALESCED record
        (the merged output token AND each consumed branch input), reporting
        rows_coalesced == 3 AND rows_succeeded == 3 for this 2-branch
        coalesce-success while the LIVE RunResult reports 1 each.  The fix
        (same commit) makes derive count only the merged output token
        (``sink_name`` set), mirroring the live accumulator which never routes
        consumed inputs through ``accumulate_row_outcomes``.  See the inline
        comment on the COALESCED arm in run_status.py.

        SCOPE CAVEAT — rows_coalesce_failed: a coalesce-SUCCESS topology keeps
        ``rows_coalesce_failed == 0`` on both sides (reconciled vacuously here,
        which is correct — that field's non-vacuous resume coverage lives in
        ``test_adr_019_resume_counter_parity.py::
        test_resume_grafts_rows_coalesce_failed_from_timeout_redrive``).
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.landscape.schema import token_outcomes_table, token_parents_table, tokens_table

        # ── Run A (uninterrupted oracle) ──────────────────────────────────────
        _db_a, _ps_a, _config_a, _graph_a, _settings_a, _run_id_a, run_a = self._setup_coalesce_pipeline()
        assert run_a.status == RunStatus.COMPLETED, run_a.status
        # Non-vacuity precondition: the coalesce SUCCEEDED → rows_coalesced >= 1.
        assert run_a.rows_coalesced >= 1, (
            f"Run A (uninterrupted coalesce-success) must record at least one "
            f"COALESCED terminal outcome (non-vacuous precondition); got "
            f"rows_coalesced={run_a.rows_coalesced}"
        )

        # ── Run B (run-1 + interrupt-before-barrier + resume) ─────────────────
        db, payload_store, config, graph, settings_obj, run_id, _run_b1 = self._setup_coalesce_pipeline()

        # Find the merged token (join_group_id set, branch_name NULL).
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.join_group_id AS join_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token (join_group_id set, branch/fork NULL); got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        join_group_id = merged_rows[0].join_group_id

        # ── Interrupt: undo the barrier entirely (pre-barrier crash state) ────
        # Verbatim from test_resume_fork_to_coalesce_before_barrier: reverse
        # everything the barrier wrote so both branch children become incomplete
        # leaves with no completed coalesce node_state.  Resume then re-drives
        # both branches (the with-unprocessed-rows fork-re-drive branch).
        coalesce_node_id_for_deletion = graph.get_coalesce_id_map()[CoalesceName("merge")]

        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
            # 1. merged token outcomes (COMPLETED at sink)
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.token_id == merged_token_id))
            # 2. merged token node_states
            conn.execute(
                text("DELETE FROM node_states WHERE token_id = :tid"),
                {"tid": merged_token_id},
            )
            # 3. token_parents for merged token
            conn.execute(token_parents_table.delete().where(token_parents_table.c.token_id == merged_token_id))
            # 4. merged token row
            conn.execute(tokens_table.delete().where(tokens_table.c.token_id == merged_token_id))
            # 5. branch COALESCED outcomes (recorded by the barrier, path='coalesced')
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.join_group_id == join_group_id))
            # 6. branch tokens' COMPLETED node_states at the coalesce node
            conn.execute(
                text("DELETE FROM node_states WHERE node_id = :nid AND run_id = :rid"),
                {"nid": str(coalesce_node_id_for_deletion), "rid": run_id},
            )
            conn.commit()
            conn.exec_driver_sql("PRAGMA foreign_keys = ON")

        # ── Resume ────────────────────────────────────────────────────────────
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        run_b_resume = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Reconciliation: EVERY counter field, non-vacuous on rows_coalesced ─
        assert run_b_resume.status == RunStatus.COMPLETED, (
            f"Resume of the coalesce pipeline must reach COMPLETED; got {run_b_resume.status}"
        )

        # Non-vacuity: the RESUMED run must also record the COALESCED outcome.
        assert run_b_resume.rows_coalesced >= 1, (
            f"Resumed coalesce-success run must record at least one COALESCED "
            f"terminal outcome (non-vacuous); got rows_coalesced={run_b_resume.rows_coalesced}. "
            f"If 0, the barrier did not re-fire on resume or derive miscounts COALESCED."
        )
        assert run_b_resume.rows_coalesced == run_a.rows_coalesced, (
            f"rows_coalesced must equal the uninterrupted run: A={run_a.rows_coalesced}, "
            f"B={run_b_resume.rows_coalesced}. derive reconstructs this purely from the "
            f"(SUCCESS, COALESCED) arm in run_status.py (not grafted); a divergence means "
            f"that arm regressed or the barrier failed to re-fire on resume."
        )

        counter_fields = (
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        )
        for field in counter_fields:
            a_val = getattr(run_a, field)
            b_val = getattr(run_b_resume, field)
            assert b_val == a_val, (
                f"F2 reconciliation failure on '{field}': resumed run (run1 + resume) must equal "
                f"the uninterrupted run field-for-field. uninterrupted={a_val}, resumed={b_val}. "
                f"Both resume branches finalize cumulative counters from the audit trail; a divergence "
                f"means either the with-rows branch regressed to resume-only counters or "
                f"derive_resume_terminal_status_from_audit miscounts this field."
            )

        assert dict(run_b_resume.routed_destinations) == dict(run_a.routed_destinations), (
            f"F2 reconciliation failure on routed_destinations: "
            f"uninterrupted={dict(run_a.routed_destinations)}, resumed={dict(run_b_resume.routed_destinations)}"
        )

    @staticmethod
    def _build_end_of_source_flush_aggregation(
        n_source_rows: int,
        trigger_count: int | None = None,
    ) -> tuple[LandscapeDB, PipelineConfig, ExecutionGraph, ElspethSettings]:
        """Build a fresh source(N rows) → batch aggregation → sink pipeline.

        ``trigger_count`` is the aggregation's ``count`` trigger. Default
        (``None`` → ``N + 1``) is the canonical end-of-source-flush topology:
        the aggregation NEVER triggers mid-stream — all N input rows are BUFFERED
        and flushed together at end-of-source, and ``live == derive == N`` for
        ``rows_buffered``. Pass ``trigger_count=N`` to build the count==N
        mid-stream-trigger topology that the rows_buffered live/derive divergence
        pinning test (see ``test_count_equals_n_rows_buffered_divergence_is_pinned``)
        exercises.

        WHY count > N (NOT count == N): a count==N trigger FIRES on the Nth row
        mid-stream, and the live accumulator and derive() then DISAGREE on
        rows_buffered (live counts N-1, derive counts N from the persisted
        BATCH_CONSUMED→BUFFERED records — a separate divergence, out of scope for
        THIS reconciliation cell but now tracked (filigree elspeth-e1dd5e1303) and
        pinned by test_count_equals_n_rows_buffered_divergence_is_pinned).  The
        end-of-source-flush path (count > N) is the CANONICAL buffered path on which
        live == derive == N for every field, so it is the honest topology for a
        field-for-field live-vs-derive reconciliation cell.  (Mirrors
        tests/property/audit/test_terminal_states.py's count=9999 construction.)

        Returns (db, config, graph, settings) — caller runs / resumes it.
        """
        from elspeth.contracts.enums import Determinism, OutputMode
        from elspeth.contracts.schema_contract import PipelineRow
        from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult
        from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
        from tests.fixtures.plugins import CollectSink, ListSource

        class _SumAggregator(BaseTransform):
            name = "sum-aggregator"
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            source_file_hash = None
            input_schema = _TestSchema
            output_schema = _TestSchema
            is_batch_aware = True
            passes_through_input = False
            on_success = "output"
            on_error = "discard"

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, rows: list[PipelineRow], ctx: object) -> TransformResult:  # type: ignore[override]
                if not rows:
                    # Unreachable in practice (a batch flush always carries rows);
                    # "invalid_input" is a valid TransformResult.error reason literal.
                    return TransformResult.error({"reason": "invalid_input"}, retryable=False)
                total = sum(r.to_dict().get("value", 0) for r in rows)
                return TransformResult.success(PipelineRow({"sum": total}, rows[0].contract), success_reason={"action": "sum"})

        db = make_landscape_db()
        src = ListSource([{"value": i + 1} for i in range(n_source_rows)], name="list_source", on_success="agg_in")
        out = CollectSink("output")
        agg = _SumAggregator()
        agg_settings = AggregationSettings(
            name="sum_agg",
            plugin=agg.name,
            input="agg_in",
            on_success="output",
            on_error="discard",
            # Default count > N → never fires mid-stream → all N rows buffer to
            # end-of-source. trigger_count=N forces the mid-stream-trigger topology.
            trigger=TriggerConfig(count=trigger_count if trigger_count is not None else n_source_rows + 1, timeout_seconds=3600),
            output_mode=OutputMode.TRANSFORM,
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(src),
            source_settings=SourceSettings(plugin=src.name, on_success="agg_in", options={}),
            transforms=[],
            sinks={"output": as_sink(out)},
            aggregations={"sum_agg": (as_transform(agg), agg_settings)},
            gates=[],
        )
        agg_id_map = graph.get_aggregation_id_map()
        agg_node_id = agg_id_map[next(iter(agg_id_map))]
        agg.node_id = agg_node_id
        config = PipelineConfig(
            source=as_source(src),
            transforms=[as_transform(agg)],
            sinks={"output": as_sink(out)},
            aggregation_settings={agg_node_id: agg_settings},
        )
        settings = ElspethSettings(
            source={"plugin": src.name, "on_success": "agg_in", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
        )
        return db, config, graph, settings

    def test_resume_buffered_counter_reconciles_with_uninterrupted_run(self) -> None:
        """A resumed aggregation run reconciles EVERY counter field — including
        rows_buffered > 0 (non-vacuous) — with an uninterrupted run.

        WHY THIS CELL EXISTS (regression-detection gap closed):

        ``test_resume_counters_reconcile_with_uninterrupted_run`` (fork→sink) and
        the ``test_adr_019_resume_counter_parity.py`` snapshot tests all use
        topologies with ``rows_buffered == 0`` on both sides — so that field
        reconciled only VACUOUSLY (0 == 0).  ``rows_buffered`` is NOT always 0 at
        a COMPLETED run: a batch aggregation leaves one non-completed
        ``(None, BUFFERED)`` audit record per input row that derive() counts
        (run_status.py line ~113), so an N-row aggregation completes with
        ``rows_buffered == N`` (cf. test_terminal_states.py and
        test_t18_characterization.py, which both assert ``rows_buffered == N``
        on COMPLETED aggregation runs).  This cell reconciles all 12 fields on
        such a topology where ``rows_buffered >= 1`` (non-vacuous).

        TOPOLOGY (``_build_end_of_source_flush_aggregation``, N=3):
            source(3 rows) → batch aggregation(count=4 → NEVER fires mid-stream;
                              all 3 rows BUFFERED, flushed together at
                              end-of-source) → sink 'output'
        The 3 input rows each get a persisted ``(None, BUFFERED)`` audit record
        (TerminalPath.BUFFERED, non-completed) → ``rows_buffered == 3``
        (non-vacuous).  count > N is deliberate: see the helper docstring — a
        count==N mid-stream trigger makes the live accumulator and derive()
        DISAGREE on rows_buffered (tracked: elspeth-e1dd5e1303; pinned by
        test_count_equals_n_rows_buffered_divergence_is_pinned).

        RUN A (uninterrupted oracle): a SEPARATE
        ``_build_end_of_source_flush_aggregation`` instance, NOT interrupted.
        Its counters come from the LIVE accumulator.

        RUN B (run-1 + interrupt + resume): the SAME topology; run-1 completes
        with all 3 BUFFERED records + the aggregate result + sink write persisted
        intact.  The interrupt creates a checkpoint and marks the run failed but
        deletes NO token_outcomes — so on resume ``get_unprocessed_rows`` finds
        nothing to re-drive and resume takes the ALL-ROWS-ALREADY-PROCESSED
        branch (the branch Phase-2.2 introduced
        ``derive_resume_terminal_status_from_audit`` for).  derive() reconstructs
        the cumulative counters — including ``rows_buffered`` from the 3 intact
        BUFFERED records — and the run finalizes COMPLETED.

        ``rows_buffered`` is reconstructed purely by derive's ``(None, BUFFERED)``
        arm (line ~113 of run_status.py — NOT grafted; only ``rows_coalesce_failed``
        is grafted).  So the line-113 lever bites the RESUMED run B
        (derive-reconstructed) while run A's value comes from the live accumulator.

        OBSERVED RED (lever: delete ``counters.rows_buffered += 1`` from the
        ``(None, BUFFERED)`` arm in run_status.py — the non-completed branch near
        line 113, leaving the ``continue``; run
        ``-k test_resume_buffered_counter_reconciles_with_uninterrupted_run``):
            AssertionError: Resumed aggregation run must record at least one
            BUFFERED record (non-vacuous); got rows_buffered=0. ...
            assert 0 >= 1
        Run A (live accumulator) reports rows_buffered=3; run B
        (derive-reconstructed, lever removed) reports 0 — a 3-vs-0 mismatch, not
        a both-zero vacuity (the non-vacuity guard fires first; the field-loop
        ``uninterrupted=3, resumed=0`` mismatch is the same signal).  Reverting
        the lever restores GREEN.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings

        n = 3

        # ── Run A (uninterrupted oracle) ──────────────────────────────────────
        db_a, config_a, graph_a, settings_a = self._build_end_of_source_flush_aggregation(n)
        run_a = Orchestrator(db_a).run(config_a, graph=graph_a, settings=settings_a, payload_store=MockPayloadStore())
        assert run_a.status == RunStatus.COMPLETED, run_a.status
        # Non-vacuity precondition: all N rows buffered → rows_buffered == N >= 1.
        assert run_a.rows_buffered == n, (
            f"Run A (uninterrupted end-of-source-flush aggregation of {n} rows) must record "
            f"rows_buffered={n} (one BUFFERED record per input row); got {run_a.rows_buffered}"
        )
        assert run_a.rows_buffered >= 1, "non-vacuity precondition"

        # ── Run B (run-1 + interrupt + resume via the all-terminal branch) ────
        db, config, graph, settings_obj = self._build_end_of_source_flush_aggregation(n)
        payload_store = MockPayloadStore()
        run_b1 = Orchestrator(db).run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run_b1.run_id
        assert run_b1.rows_buffered == n, run_b1.rows_buffered

        # Interrupt: checkpoint + mark failed, deleting NO outcomes.  Every token
        # already has its terminal (or non-completed BUFFERED) record, so resume's
        # get_unprocessed_rows is empty → the all-rows-already-processed branch
        # reconstructs the cumulative counters from the intact audit trail.
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            conn.commit()

        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        run_b_resume = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Reconciliation: EVERY counter field, non-vacuous on rows_buffered ─
        assert run_b_resume.status == RunStatus.COMPLETED, (
            f"Resume of the aggregation pipeline must reach COMPLETED; got {run_b_resume.status}"
        )
        assert run_b_resume.rows_buffered >= 1, (
            f"Resumed aggregation run must record at least one BUFFERED record (non-vacuous); "
            f"got rows_buffered={run_b_resume.rows_buffered}. If 0, derive's (None, BUFFERED) arm "
            f"miscounts the persisted BUFFERED records."
        )
        assert run_b_resume.rows_buffered == run_a.rows_buffered, (
            f"rows_buffered must equal the uninterrupted run: A={run_a.rows_buffered}, "
            f"B={run_b_resume.rows_buffered}. derive reconstructs this purely from the "
            f"(None, BUFFERED) arm in run_status.py (not grafted); a divergence means that "
            f"arm regressed or the BUFFERED records were not preserved across resume."
        )

        counter_fields = (
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        )
        for field in counter_fields:
            a_val = getattr(run_a, field)
            b_val = getattr(run_b_resume, field)
            assert b_val == a_val, (
                f"F2 reconciliation failure on '{field}': resumed run (run1 + resume) must equal "
                f"the uninterrupted run field-for-field. uninterrupted={a_val}, resumed={b_val}. "
                f"derive_resume_terminal_status_from_audit must reconstruct this field from the "
                f"audit trail to match the live accumulator."
            )

        assert dict(run_b_resume.routed_destinations) == dict(run_a.routed_destinations), (
            f"F2 reconciliation failure on routed_destinations: "
            f"uninterrupted={dict(run_a.routed_destinations)}, resumed={dict(run_b_resume.routed_destinations)}"
        )

    def test_count_equals_n_rows_buffered_divergence_is_pinned(self) -> None:
        """CHARACTERIZATION PIN — known F2 parity break (filigree elspeth-e1dd5e1303).

        For a batch aggregation with a ``count == N`` trigger (fires mid-stream on
        the Nth row, NOT at end-of-source), the live accumulator and derive disagree
        on ``rows_buffered``:

            uninterrupted oracle (live):  rows_buffered == N - 1
            resumed (derive):             rows_buffered == N

        So a RESUMED run reports ``rows_buffered`` exactly one higher than the
        uninterrupted oracle.  Every OTHER counter field reconciles — this is the
        sole divergent field.  The canonical end-of-source-flush topology
        (``count > N``) does not exhibit it (live == derive == N), which is why
        ``test_resume_buffered_counter_reconciles_with_uninterrupted_run`` uses
        ``count > N`` and the F2 reconciliation suite sidesteps this case.

        WHY A CHARACTERIZATION PIN, NOT AN xfail: the follow-up note asked for an
        "xfail-style pinning test so the divergence can't silently widen."  A strict
        xfail asserting the ideal (resumed == oracle) catches an accidental *fix*
        (xpass → fail) but a *widening* divergence (gap → 2) still just fails → stays
        xfail → green, silently violating the stated goal.  This pin asserts the
        EXACT current values, so it goes RED on widening, narrowing, OR a fix in
        either direction — strictly stronger for "can't silently widen."  This is
        NOT an endorsement of either value: elspeth-e1dd5e1303 tracks the decision
        of which (N-1 or N) is authoritative; resolving it requires updating this
        pin to the chosen value.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings

        n = 3

        # ── Run A (uninterrupted oracle, count == N → mid-stream trigger) ──────
        db_a, config_a, graph_a, settings_a = self._build_end_of_source_flush_aggregation(n, trigger_count=n)
        run_a = Orchestrator(db_a).run(config_a, graph=graph_a, settings=settings_a, payload_store=MockPayloadStore())
        assert run_a.status == RunStatus.COMPLETED, run_a.status

        # ── Run B (run-1 + interrupt + resume via the all-terminal branch) ────
        db, config, graph, settings_obj = self._build_end_of_source_flush_aggregation(n, trigger_count=n)
        payload_store = MockPayloadStore()
        run_b1 = Orchestrator(db).run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run_b1.run_id
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id, token_id=actual_token.token_id, node_id=sink_node_ids[0], sequence_number=1, graph=graph
        )
        with db.engine.connect() as conn:
            conn.execute(text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"), {"run_id": run_id})
            conn.commit()
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        run_b_resume = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)
        assert run_b_resume.status == RunStatus.COMPLETED, run_b_resume.status

        # ── Pin the EXACT divergence (elspeth-e1dd5e1303) ─────────────────────
        assert run_a.rows_buffered == n - 1, (
            f"PIN: uninterrupted oracle (live accumulator) must report rows_buffered == N-1 == {n - 1} "
            f"on a count==N mid-stream trigger; got {run_a.rows_buffered}. If this changed, the live "
            f"buffered-accounting semantics moved — update elspeth-e1dd5e1303 and this pin."
        )
        assert run_b_resume.rows_buffered == n, (
            f"PIN: resumed run (derive) must report rows_buffered == N == {n} (one BUFFERED audit record "
            f"per input row); got {run_b_resume.rows_buffered}. If this changed, derive's (None, BUFFERED) "
            f"arm moved — update elspeth-e1dd5e1303 and this pin."
        )
        assert run_b_resume.rows_buffered == run_a.rows_buffered + 1, (
            f"PIN: the known F2 parity break is exactly +1 (resumed over oracle). "
            f"oracle={run_a.rows_buffered}, resumed={run_b_resume.rows_buffered}, "
            f"delta={run_b_resume.rows_buffered - run_a.rows_buffered}. A delta != 1 means the divergence "
            f"WIDENED or was fixed — neither may land silently. See elspeth-e1dd5e1303."
        )

        # ── rows_buffered must be the SOLE divergent field ────────────────────
        other_fields = (
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_diverted",
        )
        for field in other_fields:
            a_val = getattr(run_a, field)
            b_val = getattr(run_b_resume, field)
            assert b_val == a_val, (
                f"PIN: '{field}' must reconcile on the count==N topology (only rows_buffered is known to "
                f"diverge). uninterrupted={a_val}, resumed={b_val}. A NEW divergent field is a regression "
                f"beyond the tracked elspeth-e1dd5e1303 limitation."
            )

    # ─────────────────────────────────────────────────────────────────────
    # Task 12: Remaining risk-ordered resume-recovery matrix
    #
    # Each cell follows RED→GREEN→conservation-law and reuses the established
    # helpers.  The RED lever differs per cell and is named in each docstring;
    # most use the "dispatch-disable" technique (force resume.py's
    # fork_expand_coalesce_specs branch to [] so every row falls through to
    # process_existing_row, which re-forks/re-expands the whole row and
    # re-emits the already-completed branches — the F1 double-emission defect).
    # Cell #6-linear and #5 use different levers (named inline) because the
    # dispatch-disable lever does not exercise their specific invariant.
    # All RED reasons below were OBSERVED by hand-applying the named lever to
    # src/, running the single cell, capturing the failure, then reverting.
    # ─────────────────────────────────────────────────────────────────────

    def _run_nway_fork(
        self,
        *,
        n_source_rows: int,
        sink_names: list[str],
        sink_factories: dict[str, Any] | None = None,
    ) -> tuple[LandscapeDB, MockPayloadStore, PipelineConfig, ExecutionGraph, ElspethSettings, str, dict[str, Any]]:
        """Run a fork pipeline: each source row forks to every sink in ``sink_names``.

        Generalises _setup_fork_and_interrupt to N branches and N source rows
        without interrupting anything — callers choose which branch(es) to
        interrupt.  Uses CollectSink for every branch unless a custom sink
        instance is supplied via ``sink_factories`` (used by the failing-branch
        cell, which substitutes a DivertingSink).  All plugin instantiation and
        execution go through the production ExecutionGraph.from_plugin_instances
        + Orchestrator.run path (no bypass).

        Returns (db, payload_store, config, graph, settings_obj, run_id, sinks).
        """
        from elspeth.core.config import ElspethSettings

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        sinks_by_name = dict(sink_factories) if sink_factories else {}
        for name in sink_names:
            if name not in sinks_by_name:
                sinks_by_name[name] = CollectSink(name)

        first_sink = sink_names[0]
        rows = [{"value": i} for i in range(n_source_rows)]
        source = ListSource(rows, on_success=first_sink)

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": first_sink},
            fork_to=list(sink_names),
        )

        sink_map = {name: as_sink(sink) for name, sink in sinks_by_name.items()}
        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks=sink_map,
            gates=[gate],
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=[],
            sinks=sink_map,
            gates=[gate],
            aggregations={},
            coalesce_settings=[],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": first_sink, "options": {}},
            sinks={name: {"plugin": "test", "on_write_failure": "discard"} for name in sink_names},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        return db, payload_store, config, graph, settings_obj, run.run_id, sinks_by_name

    def _delete_branch_outcomes(self, db: LandscapeDB, run_id: str, branch_names: list[str]) -> None:
        """Delete the terminal OUTCOMES for the named fork branches (leaving node_states).

        Simulates a crash after those branches' sinks ran but before (or while)
        their terminal outcomes were recorded.  node_states are preserved so the
        re-drive attempt offset (max_attempt + 1) is anchored on real run-1 rows.
        """
        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.sink_name IN :branches
                """).bindparams(bindparam("branches", expanding=True)),
                {"run_id": run_id, "branches": branch_names},
            ).fetchall()
            assert outcomes, f"expected outcomes for branches {branch_names} to delete (setup precondition)"
            for outcome in outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == outcome.outcome_id))
            conn.commit()

    def _checkpoint_and_resume(
        self,
        db: LandscapeDB,
        payload_store: MockPayloadStore,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings_obj: ElspethSettings,
        run_id: str,
    ) -> RunResult:
        """Create a checkpoint, mark the run failed, and resume — returning the RunResult.

        Identical preconditions to _resume_run but returns the resume RunResult so
        cells can assert on the terminal status (some cells expect
        COMPLETED_WITH_FAILURES rather than COMPLETED).
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings

        checkpoint_mgr = CheckpointManager(db)
        sink_node_ids = graph.get_sinks()
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"), {"run_id": run_id})
            conn.commit()

        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        return resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

    def _completed_sink_outcome_counts(self, db: LandscapeDB, run_id: str) -> dict[tuple[str, str], int]:
        """Multiset of completed (row_id, sink_name) terminal outcomes for a run."""
        with db.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT t.row_id AS row_id, o.sink_name AS sink_name, COUNT(*) AS n
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND o.completed = 1
                      AND o.sink_name IS NOT NULL
                    GROUP BY t.row_id, o.sink_name
                """),
                {"run_id": run_id},
            ).fetchall()
        return {(row.row_id, row.sink_name): row.n for row in result}

    def test_resume_all_branches_incomplete(self) -> None:
        """matrix #2: a fork row where NO branch completed — every child re-drives.

        A single source row forks to sink_a + sink_b.  Both branches' terminal
        outcomes are deleted (the FORK_PARENT delegation outcome is preserved),
        so the row has two incomplete children and no completed leaf.  Resume
        must surface BOTH children and re-drive each to its sink exactly once.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        process_existing_row re-forks the whole row from the source, minting a
        NEW fork parent + two NEW children that write to both sinks.  The ORIGINAL
        run-1 children (whose outcomes were deleted) are never re-driven and remain
        with zero terminal outcomes.  At end-of-row sweep_deferred_invariants_or_crash
        (ADR-019 I1a) scans the ORIGINAL fork parent (still FORK_PARENT) and finds
        its children have no terminal outcomes.
        Observed: AuditIntegrityError — "fork/expand parent token(s) have no child
        token_outcomes rows at run-end." (path=fork_parent).
        """
        from elspeth.contracts.enums import RunStatus

        db, payload_store, config, graph, settings_obj, run_id, _sinks = self._run_nway_fork(
            n_source_rows=1, sink_names=["sink_a", "sink_b"]
        )
        baseline = self._completed_sink_outcome_counts(db, run_id)
        baseline_fork_stats = get_fork_group_stats(db, run_id)
        assert sorted(k[1] for k in baseline) == ["sink_a", "sink_b"], baseline
        assert all(n == 1 for n in baseline.values()), baseline

        # Oracle precondition: deleting BOTH branch outcomes leaves two incomplete children.
        self._delete_branch_outcomes(db, run_id, ["sink_a", "sink_b"])

        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 2, f"both branch children must be incomplete; got {[s.token_id for s in all_specs]}"
        assert sorted(s.branch_name for s in all_specs if s.branch_name is not None) == ["sink_a", "sink_b"], [
            s.branch_name for s in all_specs
        ]

        resume_result = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)

        # Conservation law.
        after = self._completed_sink_outcome_counts(db, run_id)
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left orphan leaf token(s): {orphans}"
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status
        # NOTE: both branches were interrupted by OUTCOME deletion only — their sinks
        # physically ran in run-1 and physically re-run on the re-drive (the CollectSink
        # `results` list legitimately grows).  The conserved quantity is the AUDIT trail
        # (one completed outcome per (row, sink)), asserted above — not the in-memory sink.
        post_stats = get_fork_group_stats(db, run_id)
        assert post_stats["total_fork_groups"] == baseline_fork_stats["total_fork_groups"], (
            f"Resume changed fork-group shape: {baseline_fork_stats} -> {post_stats}"
        )

    def test_resume_three_way_fork_two_incomplete(self) -> None:
        """matrix #3: a 3-way fork with 1 done + 2 incomplete; deterministic completion.

        A single source row forks to sink_a + sink_b + sink_c.  Two branches'
        outcomes are deleted (sink_a, sink_c), leaving sink_b complete.  Resume
        must re-drive exactly the two incomplete branches.  get_incomplete_tokens_by_row
        orders specs by (step_in_pipeline, token_id) — assert the dispatch order is
        deterministic, and conservation holds.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        process_existing_row re-forks the whole row, re-emitting the ALREADY-COMPLETE
        sink_b branch (double emission) while the original sink_a/sink_c children
        stay outcome-less.  The fresh re-forked children all reach terminal outcomes
        (so the NEW fork parent passes the I1a sweep), but the re-forked sink_b now
        carries a SECOND completed outcome → the conservation assertion catches it
        directly (it fires before any orphan/I1a concern, since the original orphaned
        children are abandoned without an I1a parent violation in this re-fork shape).
        Observed: AssertionError — "Resume must conserve the terminal-outcome multiset.
        baseline={(..., 'sink_b'): 1} after={(..., 'sink_b'): 2}" (sink_b double-emitted).
        """
        from elspeth.contracts.enums import RunStatus

        db, payload_store, config, graph, settings_obj, run_id, sinks = self._run_nway_fork(
            n_source_rows=1, sink_names=["sink_a", "sink_b", "sink_c"]
        )
        baseline = self._completed_sink_outcome_counts(db, run_id)
        assert sorted(k[1] for k in baseline) == ["sink_a", "sink_b", "sink_c"], baseline
        assert all(n == 1 for n in baseline.values()), baseline

        self._delete_branch_outcomes(db, run_id, ["sink_a", "sink_c"])

        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 2, f"two branches must be incomplete; got {[s.token_id for s in all_specs]}"
        assert sorted(s.branch_name for s in all_specs if s.branch_name is not None) == ["sink_a", "sink_c"], [
            s.branch_name for s in all_specs
        ]
        # Deterministic completion order: the recovery query orders by
        # (step_in_pipeline, token_id); the two fork children share step_in_pipeline
        # so token_id is the tiebreaker — assert the surfaced order is that sort.
        ordered_token_ids = [s.token_id for s in all_specs]
        assert ordered_token_ids == sorted(ordered_token_ids), (
            f"specs must be deterministically ordered by (step_in_pipeline, token_id); got {ordered_token_ids}"
        )

        resume_result = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)

        after = self._completed_sink_outcome_counts(db, run_id)
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left orphan leaf token(s): {orphans}"
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status
        # sink_b (already complete before interruption) must NOT be re-written.
        assert len(sinks["sink_b"].results) == 1, f"sink_b re-written: {len(sinks['sink_b'].results)} times (must stay 1)"

    def test_resume_failure_during_resumed_branch_yields_failed_not_orphan(self) -> None:
        """matrix #4: a resumed branch whose sink fails yields a FAILURE terminal, not an orphan.

        DEVIATION FROM PLAN NAME: the plan says "sink fails → FAILED terminal".
        A sink whose write() RAISES would crash the whole run (plugin-crash
        semantics, CLAUDE.md), not produce a bounded terminal.  Instead we use
        DivertingSink on the failing branch: it diverts every row, and the
        production SinkExecutor discard branch records a clean
        (FAILURE, SINK_DISCARDED) terminal outcome with completed=1 — the same
        invariant the plan intends (a recorded FAILURE-class terminal, zero
        orphan), achieved without crashing the run.

        Topology: one source row forks to sink_ok (CollectSink) + sink_bad
        (DivertingSink, diverts everything → FAILURE/SINK_DISCARDED).
        Interrupt the sink_bad branch (delete its FAILURE outcome), resume.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        process_existing_row re-forks the row; the ORIGINAL sink_bad branch (whose
        FAILURE outcome was deleted) is never re-driven.  sink_ok's outcome was NOT
        deleted, so the original fork parent still has ≥1 completed child → the I1a
        sweep does NOT fire; resume completes, but the original sink_bad token is left
        with no terminal outcome → an orphan.
        Observed: AssertionError — "Resume left orphan leaf token(s): [...]" (the
        not-orphan claim goes RED here).

        BOUNDEDNESS (asserted as a confirmed property, NOT a leverable RED): a
        FAILURE outcome records completed=1 (data_flow_repository.record_token_outcome:
        `completed = outcome is not None`).  get_incomplete_tokens_by_row's
        terminal_tokens subquery excludes any completed=1 token.  So the re-driven
        failed branch is NOT re-selected on a SUBSEQUENT resume.  We prove this by
        running a SECOND resume and asserting the failed token is absent from the
        incomplete specs and no new outcome appears.  There is no RED lever for this
        short of editing outcome-recording; it is a structural consequence of
        completed=1, demonstrated directly.
        """
        from elspeth.contracts.enums import RunStatus
        from tests.fixtures.plugins import DivertingSink

        sink_bad = DivertingSink(name="sink_bad")  # divert_count=None → diverts ALL rows
        db, payload_store, config, graph, settings_obj, run_id, sinks = self._run_nway_fork(
            n_source_rows=1,
            sink_names=["sink_ok", "sink_bad"],
            sink_factories={"sink_bad": sink_bad},
        )

        # ── Baseline: sink_ok has a completed sink write; sink_bad has a FAILURE terminal ──
        baseline_ok = self._completed_sink_outcome_counts(db, run_id)
        # sink_ok completed once; sink_bad diverted → its terminal is FAILURE/SINK_DISCARDED
        # (the discard-sentinel sink_name, not 'sink_bad'); not a CollectSink write.
        assert ("sink_ok" in {k[1] for k in baseline_ok}) or any(k[1] == "sink_ok" for k in baseline_ok), baseline_ok

        def _branch_failure_terminals() -> list[tuple[str, str | None]]:
            with db.engine.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT o.outcome AS outcome, o.path AS path
                        FROM token_outcomes o
                        JOIN tokens t ON t.token_id = o.token_id
                        JOIN rows r ON r.row_id = t.row_id
                        WHERE r.run_id = :run_id AND t.branch_name = 'sink_bad' AND o.completed = 1
                    """),
                    {"run_id": run_id},
                ).fetchall()
            return [(row.outcome, row.path) for row in rows]

        baseline_bad = _branch_failure_terminals()
        assert len(baseline_bad) == 1, f"sink_bad branch must have exactly one completed FAILURE terminal in run-1; got {baseline_bad}"
        assert baseline_bad[0][0] == "failure", f"sink_bad terminal outcome must be FAILURE; got {baseline_bad}"

        # Locate the sink_bad branch token (the one we interrupt).
        with db.engine.connect() as conn:
            bad_tokens = conn.execute(
                text("""
                    SELECT t.token_id AS token_id FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.branch_name = 'sink_bad'
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(bad_tokens) == 1, f"expected one sink_bad branch token; got {len(bad_tokens)}"
        bad_token_id = bad_tokens[0].token_id

        # ── Interrupt: delete the sink_bad branch's FAILURE outcome (leave node_states) ──
        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            bad_outcomes = conn.execute(
                text("SELECT outcome_id FROM token_outcomes WHERE token_id = :tid AND completed = 1"),
                {"tid": bad_token_id},
            ).fetchall()
            assert bad_outcomes, "sink_bad branch must have a completed outcome to delete (precondition)"
            for o in bad_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 1 and all_specs[0].token_id == bad_token_id, (
            f"only the sink_bad branch must be incomplete; got {[(s.token_id, s.branch_name) for s in all_specs]}"
        )

        resume_result = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)

        # ── Resume produces a FAILURE-class terminal, zero orphan ──
        # The DivertingSink diverts the re-driven row again → (FAILURE, SINK_DISCARDED).
        # Status is COMPLETED_WITH_FAILURES (a leaf reached a FAILURE terminal), not COMPLETED.
        assert resume_result.status == RunStatus.COMPLETED_WITH_FAILURES, (
            f"a diverted/failed leaf must drive COMPLETED_WITH_FAILURES; got {resume_result.status}"
        )
        post_bad = _branch_failure_terminals()
        assert len(post_bad) == 1, f"sink_bad branch must have exactly one FAILURE terminal after resume; got {post_bad}"
        assert post_bad[0][0] == "failure", f"sink_bad terminal must be FAILURE; got {post_bad}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left orphan leaf token(s): {orphans}"
        # sink_ok was complete before the interruption and must not be re-written.
        assert len(sinks["sink_ok"].results) == 1, f"sink_ok re-written: {len(sinks['sink_ok'].results)} (must stay 1)"

        # ── BOUNDEDNESS: a SECOND resume must NOT re-select the failed branch ──
        # The FAILURE outcome carries completed=1, so it is excluded from the
        # incomplete-token query.  Capture the outcome multiset, run a second
        # resume, and assert nothing changed and the token is not re-surfaced.
        outcomes_before_second = self._completed_sink_outcome_counts(db, run_id)
        bad_terminals_before = _branch_failure_terminals()

        by_row_2 = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        specs_2 = [s for specs in by_row_2.values() for s in specs]
        assert bad_token_id not in {s.token_id for s in specs_2}, (
            f"BOUNDEDNESS violated: the failed branch {bad_token_id!r} is re-selected on a second resume "
            f"(its completed=1 FAILURE outcome must exclude it). specs={[s.token_id for s in specs_2]}"
        )

        # Mark failed again and resume a second time; nothing must change.
        second_resume = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)
        assert second_resume.status in (RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES), second_resume.status
        assert self._completed_sink_outcome_counts(db, run_id) == outcomes_before_second, (
            "second resume must not add/remove completed sink outcomes (bounded)"
        )
        assert _branch_failure_terminals() == bad_terminals_before, "second resume must not add a duplicate FAILURE terminal (bounded)"

    def test_resume_multi_row_partial_fork(self) -> None:
        """matrix #6 (multi-row): two source rows, one fully complete + one partial.

        Two source rows each fork to sink_a + sink_b.  We interrupt ONE branch
        (sink_a) of ONE row, leaving the other row fully complete.  This is the
        cell that genuinely exercises get_incomplete_tokens_by_row's BY-ROW
        grouping (the single-row cells cannot): the incomplete spec must be
        grouped under exactly the partial row's row_id, and the dispatch loop
        must re-drive only that row's incomplete branch while leaving the complete
        row untouched.  Conservation holds over BOTH rows.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        the partial row falls through to process_existing_row, which re-forks it and
        re-emits its already-complete sink_b branch (sink_b's outcome was NOT deleted,
        so the original fork parent keeps ≥1 completed child → no I1a sweep).  Resume
        completes, but the partial row's sink_b now carries a SECOND completed outcome.
        Observed: AssertionError — "Resume must conserve the terminal-outcome multiset
        over BOTH rows. ... after={..., (partial_row, 'sink_b'): 2}" (sink_b double-emitted
        on the partial row; the complete row is untouched).
        """
        from elspeth.contracts.enums import RunStatus

        db, payload_store, config, graph, settings_obj, run_id, _sinks = self._run_nway_fork(
            n_source_rows=2, sink_names=["sink_a", "sink_b"]
        )
        baseline = self._completed_sink_outcome_counts(db, run_id)
        # Two rows x two sinks = four completed sink outcomes.
        assert len(baseline) == 4, f"expected 4 (row,sink) completed outcomes; got {baseline}"
        assert all(n == 1 for n in baseline.values()), baseline

        # Interrupt sink_a on exactly ONE row.  Pick the row with the lowest row_index.
        with db.engine.connect() as conn:
            sink_a_rows = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id, t.row_id AS row_id, rr.row_index AS row_index
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows rr ON rr.row_id = t.row_id
                    WHERE rr.run_id = :run_id AND o.sink_name = 'sink_a'
                    ORDER BY rr.row_index
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(sink_a_rows) == 2, f"expected one sink_a outcome per row (2 total); got {len(sink_a_rows)}"
        interrupted_row_id = sink_a_rows[0].row_id

        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == sink_a_rows[0].outcome_id))
            conn.commit()

        # By-row grouping oracle: exactly ONE row group, exactly ONE incomplete spec,
        # grouped under the interrupted row's row_id.
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        assert list(by_row.keys()) == [interrupted_row_id], (
            f"by-row grouping must surface exactly the partial row {interrupted_row_id!r}; got {list(by_row.keys())}"
        )
        assert len(by_row[interrupted_row_id]) == 1, f"partial row must have exactly one incomplete spec; got {by_row[interrupted_row_id]}"
        assert by_row[interrupted_row_id][0].branch_name == "sink_a", by_row[interrupted_row_id][0].branch_name

        resume_result = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)

        after = self._completed_sink_outcome_counts(db, run_id)
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset over BOTH rows. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left orphan leaf token(s): {orphans}"
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status
        # The COMPLETE row was NOT re-driven (its sink_b CollectSink physical write
        # count is unchanged by resume); only the partial row's sink_a branch re-ran.
        # Audit conservation (asserted above) is the load-bearing invariant — the
        # in-memory CollectSink for the re-driven sink_a legitimately grows.

    def test_resume_linear_pipeline_regression_audit(self) -> None:
        """matrix #6 (linear regression): a linear (no-fork) resume uses process_existing_row.

        A linear pipeline (source → PassTransform → sink) interrupted mid-row must
        resume via the whole-row restart path (process_existing_row), NOT the
        fork/expand/coalesce dispatch.  Asserted on the AUDIT TRAIL (token outcomes
        in the DB), not merely the in-memory sink, confirming the no-incomplete-
        children dispatch branch.

        RED LEVER (NOT dispatch-disable — that lever is a no-op for a linear row,
        which has no fork/expand/coalesce specs).  The genuine RED is the INVERSE:
        weaken resume.py's lineage-field filter so the linear token (branch_name,
        fork_group_id, expand_group_id, join_group_id all None) is ALSO routed to
        resume_incomplete_token.  That method has no resume-start pattern for a
        bare linear token and raises.
        Observed (filter weakened to route the linear spec): OrchestrationInvariantError
        — "Incomplete token ... has branch_name=None, fork_group_id=None,
        join_group_id=None, expand_group_id=None — no resume-start node resolvable."
        This proves the linear row depends on the process_existing_row branch.
        """
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.config import ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 7}], on_success="source_out")
        transform = PassTransform(name="pass_linear")
        sink = CollectSink("output")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
        )
        graph = _build_production_graph(config)
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "source_out", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        # Baseline audit: exactly one completed (row, output) outcome, no fork groups.
        baseline = self._completed_sink_outcome_counts(db, run_id)
        assert len(baseline) == 1 and all(n == 1 for n in baseline.values()), baseline
        assert get_fork_group_stats(db, run_id)["total_fork_groups"] == 0, "linear pipeline must have no fork groups"

        # Interrupt: delete the linear token's terminal outcome.  The single token
        # is a linear token (no lineage fields) → recovery must NOT classify it as a
        # fork/expand/coalesce spec.
        with db.engine.connect() as conn:
            outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    JOIN tokens t ON t.token_id = o.token_id
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND o.completed = 1 AND o.sink_name IS NOT NULL
                """),
                {"run_id": run_id},
            ).fetchall()
            assert outcomes, "linear pipeline must have a completed outcome to delete"
            for o in outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        # The incomplete linear token, if classified as fork/expand/coalesce, would
        # carry NO lineage fields — the resume loop's filter must NOT route it to
        # resume_incomplete_token (it routes to process_existing_row instead).
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 1, f"linear interrupt must surface one incomplete token; got {len(all_specs)}"
        linear_spec = all_specs[0]
        assert linear_spec.branch_name is None and linear_spec.fork_group_id is None, (
            f"linear token must have NO branch/fork lineage; got branch={linear_spec.branch_name!r} fork={linear_spec.fork_group_id!r}"
        )
        assert linear_spec.expand_group_id is None and linear_spec.join_group_id is None, (
            f"linear token must have NO expand/join lineage; got expand={linear_spec.expand_group_id!r} join={linear_spec.join_group_id!r}"
        )

        # Count tokens before resume so we can prove a FRESH token was minted (the
        # signature of the process_existing_row whole-row restart, distinct from the
        # in-place re-drive used by fork/expand/coalesce).
        with db.engine.connect() as conn:
            tokens_before = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id"),
                    {"run_id": run_id},
                ).scalar()
                or 0
            )

        resume_result = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)

        # AUDIT-TRAIL assertion (not just in-memory sink): the row reaches a completed
        # terminal outcome again via the process_existing_row whole-row restart.
        after = self._completed_sink_outcome_counts(db, run_id)
        assert len(after) == 1 and all(n == 1 for n in after.values()), (
            f"linear resume must restore exactly one completed (row, output) outcome; got {after}"
        )
        # process_existing_row mints a FRESH token (create_token_for_existing_row) rather
        # than re-driving the original in place — confirming the no-incomplete-children
        # dispatch branch was taken.  A fork/expand/coalesce re-drive would reuse the
        # original token_id and NOT add a token.  The ORIGINAL linear token is
        # legitimately left outcome-less (superseded by the fresh token) — that is the
        # designed linear-restart behaviour, so orphan_leaf_token_ids is NOT applicable
        # here (it correctly flags abandoned tokens, which the linear restart creates by
        # design — distinct from a fork re-drive, which must leave none).
        with db.engine.connect() as conn:
            tokens_after = int(
                conn.execute(
                    text("SELECT COUNT(*) FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id"),
                    {"run_id": run_id},
                ).scalar()
                or 0
            )
        assert tokens_after == tokens_before + 1, (
            f"linear resume must mint exactly ONE fresh token via process_existing_row "
            f"(whole-row restart); tokens before={tokens_before} after={tokens_after}"
        )
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

    def test_resume_of_resume_converges(self) -> None:
        """matrix #7: resume, then resume again — incomplete-leaf stock is monotone to 0.

        A fork row interrupted on one branch.  After the FIRST resume the row is
        complete (incomplete-leaf stock 0).  A SECOND resume of the (now-completed,
        artificially re-failed) run must be idempotent: the incomplete-leaf stock
        is monotone non-increasing across resumes and reaches 0, with no new
        outcomes minted.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        the first resume re-forks instead of re-driving, minting a fresh fork parent +
        fresh children that complete; the ORIGINAL interrupted sink_a child is never
        re-driven and stays incomplete.  sink_b's outcome was not deleted, so the
        original fork parent keeps ≥1 completed child → no I1a sweep; resume completes,
        but the original sink_a token remains an incomplete leaf → the incomplete-leaf
        stock never reaches 0.
        Observed: AssertionError — "first resume must converge incomplete-leaf stock to 0;
        got 1" (the original interrupted leaf is never re-driven, so convergence fails).
        """
        from elspeth.contracts.enums import RunStatus

        db, payload_store, config, graph, settings_obj, run_id, sinks = self._run_nway_fork(
            n_source_rows=1, sink_names=["sink_a", "sink_b"]
        )

        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

        recovery_mgr = RecoveryManager(db, CheckpointManager(db))

        def _incomplete_leaf_count() -> int:
            by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
            return sum(len(specs) for specs in by_row.values())

        # Stock before interruption: 0 (run completed).
        assert _incomplete_leaf_count() == 0, "completed run must have zero incomplete leaves"

        # Interrupt one branch → stock becomes 1.
        self._delete_branch_outcomes(db, run_id, ["sink_a"])
        stock_after_interrupt = _incomplete_leaf_count()
        assert stock_after_interrupt == 1, f"one interrupted branch → stock 1; got {stock_after_interrupt}"

        # First resume → stock must drop to 0.
        first = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)
        assert first.status == RunStatus.COMPLETED, first.status
        stock_after_first = _incomplete_leaf_count()
        assert stock_after_first == 0, f"first resume must converge incomplete-leaf stock to 0; got {stock_after_first}"
        assert stock_after_first <= stock_after_interrupt, "stock must be monotone non-increasing"

        baseline = self._completed_sink_outcome_counts(db, run_id)
        # Capture physical sink write counts BEFORE the second resume: an idempotent
        # second resume (zero incomplete leaves) must re-drive nothing, so these stay put.
        writes_before_second = {name: len(sink.results) for name, sink in sinks.items()}

        # Second resume (idempotent): no incomplete leaves, so nothing re-drives.
        # Mark failed again and resume; the stock stays 0 and outcomes are unchanged.
        second = self._checkpoint_and_resume(db, payload_store, config, graph, settings_obj, run_id)
        assert second.status == RunStatus.COMPLETED, second.status
        stock_after_second = _incomplete_leaf_count()
        assert stock_after_second == 0, f"second resume must keep incomplete-leaf stock at 0 (idempotent); got {stock_after_second}"
        assert stock_after_second <= stock_after_first, "stock must remain monotone non-increasing"
        assert self._completed_sink_outcome_counts(db, run_id) == baseline, "idempotent second resume must not change outcomes"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"resume-of-resume left orphan leaf token(s): {orphans}"
        writes_after_second = {name: len(sink.results) for name, sink in sinks.items()}
        assert writes_after_second == writes_before_second, (
            f"idempotent second resume must perform NO further sink writes: before={writes_before_second} after={writes_after_second}"
        )

    def test_resume_post_coalesce_nonterminal_redrives(self) -> None:
        """Task 12 addition: NON-TERMINAL post-coalesce (B1) re-drives THROUGH a downstream
        transform to the sink — not the terminal shortcut.

        Complements the existing test_resume_post_coalesce_before_downstream, which covers
        the TERMINAL coalesce sub-case (resolve_next_node(coalesce)=None →
        _terminal_coalesce_row_result).  Here the coalesce is NON-TERMINAL: its on_success
        feeds a further PassTransform ('post_merge') which then routes to the sink, so
        resolve_next_node(coalesce_node) returns a REAL processing node.  resume_incomplete_token
        Case 4 must take the `after is not None` branch and call process_token(after) — driving
        the merged token THROUGH 'post_merge' to the sink (process_token), NOT the terminal
        shortcut.

        Topology:
            source → gate(fork) → [path_a: PassTransform, path_b: PassTransform]
                   → coalesce('merge', on_success='post_merge')
                   → PassTransform('post_merge') → sink 'output'

        Interruption: after a complete run, delete ONLY the merged token's terminal
        COMPLETED outcome at the sink (the branch COALESCED outcomes, the merged token row,
        and its token_data_ref are preserved) — the "crashed after barrier, before the
        downstream transform/sink completed" scenario.

        RED (dispatch disabled — resume.py fork_expand_coalesce_specs forced []):
        the merged token (join_group_id set, branch_name/fork_group_id None) falls to
        process_existing_row, which re-forks the row from the source.  The run-1 coalesce
        node_states remain (completed_at IS NOT NULL), so the fresh branches arrive at the
        barrier and CoalesceExecutor sees them as late arrivals → UNROUTED (NULL sink_name)
        outcomes; no (row, 'output') completed sink outcome is ever recorded.
        Observed: AssertionError — "Resume must conserve the terminal-outcome multiset.
        baseline={(..., 'output'): 1} after={}".

        GREEN (dispatch restored): the merged token's token_data_ref is reconstructed as a
        {data, contract} envelope; the merged token is driven through 'post_merge' to the
        sink via process_token (the non-terminal Case-4 branch); one COMPLETED outcome;
        conservation holds; the SAME merged token id is reused (no re-fork).  The
        token_data_ref round-trip below proves the reconstructed payload+contract are
        well-formed.
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.serialization import checkpoint_loads
        from elspeth.core.config import CheckpointSettings, ElspethSettings
        from elspeth.core.landscape.schema import token_outcomes_table

        # ── Build NON-TERMINAL coalesce pipeline (coalesce → post_merge transform → sink) ──
        db = make_landscape_db()
        payload_store = MockPayloadStore()

        source = ListSource([{"value": 1}], on_success="gate_in")
        sink = CollectSink("output")
        pass_a = PassTransform(name="pass_a")
        pass_b = PassTransform(name="pass_b")
        post_merge = PassTransform(name="post_merge")

        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        # NON-TERMINAL coalesce construction: the DAG builder MANDATES that a coalesce's
        # on_success point to a SINK (builder.py raises "Coalesce on_success must point to
        # a sink when configured").  To make the coalesce feed a downstream TRANSFORM
        # instead, leave on_success unset (None): the builder then registers the coalesce
        # as a PRODUCER of a connection named after the coalesce itself ('merge').  The
        # post_merge transform consumes that connection (source_connection='merge') and
        # routes to the sink.  resolve_next_node(coalesce_node) then returns the post_merge
        # node → the NON-TERMINAL Case-4 branch (after is not None) in resume_incomplete_token.
        coalesce = CoalesceSettings(
            name="merge",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            # on_success omitted → coalesce produces connection "merge" (non-terminal).
        )

        wired_a = wire_transforms([pass_a], source_connection="path_a", final_sink="done_a", names=["pass_a"])
        wired_b = wire_transforms([pass_b], source_connection="path_b", final_sink="done_b", names=["pass_b"])
        wired_post = wire_transforms([post_merge], source_connection="merge", final_sink="output", names=["post_merge"])

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=wired_a + wired_b + wired_post,
            sinks={"output": as_sink(sink)},
            gates=[gate],
            aggregations={},
            coalesce_settings=[coalesce],
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(pass_a), as_transform(pass_b), as_transform(post_merge)],
            sinks={"output": as_sink(sink)},
            gates=[gate],
            coalesce_settings=[coalesce],
        )
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "gate_in", "options": {}},
            sinks={"output": {"plugin": "test", "on_write_failure": "discard"}},
            gates=[gate],
            coalesce=[coalesce],
        )

        orchestrator = Orchestrator(db)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id

        baseline = self._completed_sink_outcome_counts(db, run_id)
        assert len(baseline) == 1 and all(n == 1 for n in baseline.values()), baseline

        # ── Verify NON-TERMINAL sub-case: coalesce has a non-sink successor ──
        coalesce_node_id = graph.get_coalesce_id_map()[CoalesceName("merge")]
        sink_node_ids = set(graph.get_sinks())
        edges = graph.get_edges()
        coalesce_successors = {e.to_node for e in edges if e.from_node == coalesce_node_id}
        assert coalesce_successors and not coalesce_successors.issubset(sink_node_ids), (
            f"NON-TERMINAL precondition: the coalesce node must have a NON-sink successor "
            f"(the post_merge transform) so resolve_next_node returns a real node. "
            f"successors={coalesce_successors}, sinks={sink_node_ids}"
        )

        # ── Find the merged token + its token_data_ref ──
        with db.engine.connect() as conn:
            merged_rows = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.token_data_ref AS token_data_ref
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL
                      AND t.branch_name IS NULL
                      AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_rows) == 1, f"Expected exactly one merged token; got {len(merged_rows)}"
        merged_token_id = merged_rows[0].token_id
        merged_data_ref = merged_rows[0].token_data_ref
        assert merged_data_ref is not None, "merged token must carry token_data_ref (epoch 11 invariant)"

        # ── Interrupt: delete ONLY the merged token's terminal sink outcome ──
        with db.engine.connect() as conn:
            merged_outcomes = conn.execute(
                text("SELECT outcome_id FROM token_outcomes WHERE token_id = :tid AND completed = 1"),
                {"tid": merged_token_id},
            ).fetchall()
            assert merged_outcomes, f"merged token {merged_token_id!r} must have a completed outcome to delete"
            for o in merged_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        # ── Oracle: exactly the merged token as Case-4 spec ──
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        all_specs = [s for specs in by_row.values() for s in specs]
        assert len(all_specs) == 1, f"only the merged token must be incomplete (Case 4); got {[s.token_id for s in all_specs]}"
        spec = all_specs[0]
        assert spec.token_id == merged_token_id
        assert spec.join_group_id is not None and spec.fork_group_id is None and spec.branch_name is None, (
            f"Case-4 spec lineage mismatch: join={spec.join_group_id!r} fork={spec.fork_group_id!r} branch={spec.branch_name!r}"
        )
        assert spec.token_data_ref == merged_data_ref, f"recovery must surface the merged token_data_ref; got {spec.token_data_ref!r}"

        # ── Resume ──
        with db.engine.connect() as conn:
            actual_token = conn.execute(
                text("SELECT t.token_id AS token_id FROM tokens t JOIN rows r ON r.row_id = t.row_id WHERE r.run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).fetchone()
        assert actual_token is not None
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=actual_token.token_id,
            node_id=next(iter(sink_node_ids)),
            sequence_number=1,
            graph=graph,
        )
        with db.engine.connect() as conn:
            conn.execute(text("UPDATE runs SET status = 'failed' WHERE run_id = :run_id"), {"run_id": run_id})
            conn.commit()
        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume, f"cannot resume: {check.reason}"
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
        resume_orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        resume_result = resume_orchestrator.resume(resume_point, config, graph, payload_store=payload_store, settings=settings_obj)

        # ── Conservation law ──
        after = self._completed_sink_outcome_counts(db, run_id)
        assert after == baseline, f"Resume must conserve the terminal-outcome multiset. baseline={baseline} after={after}"
        orphans = orphan_leaf_token_ids(db, run_id)
        assert not orphans, f"Resume left orphan leaf token(s): {orphans}"
        assert all(n == 1 for n in after.values()), after
        assert resume_result.status == RunStatus.COMPLETED, resume_result.status

        # ── Identity proof: the SAME merged token was driven through 'post_merge' ──
        # A re-fork would mint a SECOND merged token (different id).  The merged token
        # must now carry a terminal outcome (recorded after it passed through post_merge
        # to the sink — the non-terminal process_token(after) path, not the terminal
        # _terminal_coalesce_row_result shortcut).
        with db.engine.connect() as conn:
            merged_after = conn.execute(
                text("""
                    SELECT t.token_id AS token_id FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id
                      AND t.join_group_id IS NOT NULL AND t.branch_name IS NULL AND t.fork_group_id IS NULL
                """),
                {"run_id": run_id},
            ).fetchall()
        assert len(merged_after) == 1 and merged_after[0].token_id == merged_token_id, (
            f"B1 non-terminal: resume must reuse the SAME merged token (not re-fork); got {[t.token_id for t in merged_after]}"
        )
        # The merged token must have visited the post_merge node (a node_state there)
        # — proving it was driven THROUGH the downstream transform, not shortcut.
        post_merge_node_id = next(n.node_id for n in graph.get_nodes() if n.plugin_name == "post_merge")
        with db.engine.connect() as conn:
            post_merge_states = conn.execute(
                text("SELECT COUNT(*) FROM node_states WHERE token_id = :tid AND node_id = :nid"),
                {"tid": merged_token_id, "nid": str(post_merge_node_id)},
            ).scalar()
        assert post_merge_states and post_merge_states >= 1, (
            f"the merged token must have a node_state at the post_merge transform — proving the "
            f"NON-TERMINAL Case-4 process_token(after) path drove it through the downstream "
            f"transform, not the terminal shortcut; got {post_merge_states} states"
        )

        # ── token_data_ref envelope round-trip (B1 payload reconstruction) ──
        raw_bytes = payload_store.retrieve(merged_data_ref)
        env = checkpoint_loads(raw_bytes.decode("utf-8"))
        assert isinstance(env, dict) and "data" in env and "contract" in env, (
            f"merged token payload is not a {{data, contract}} envelope; "
            f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
        )
        # union-merge of path_a({'value':1}) and path_b({'value':1}) → {'value': 1}.
        assert env["data"] == {"value": 1}, f"merged payload round-trip mismatch: expected {{'value': 1}}, got {env['data']!r}"
        from elspeth.contracts.schema_contract import SchemaContract

        restored_contract = SchemaContract.from_checkpoint(env["contract"])
        assert {fc.normalized_name for fc in restored_contract.fields} == {"value"}, (
            f"restored merged contract fields mismatch (non-terminal B1): expected {{'value'}}, "
            f"got {{{', '.join(repr(fc.normalized_name) for fc in restored_contract.fields)}}}"
        )

    def test_resume_aggregation_buffer_with_partial_fork(self) -> None:
        """matrix #5: a row with MIXED buffered + incomplete fork children is NOT excluded.

        get_unprocessed_rows (recovery.py) excludes a row from reprocessing ONLY when
        ALL its incomplete leaf tokens are buffered in checkpoint aggregation state
        (lines 474-501).  The bug this guards is row-level exclusion: dropping a row
        because it has ANY buffered token, which would silently orphan a sibling
        non-buffered incomplete token.

        Construction (genuine buffered state, NOT hand-crafted JSON):
        - One source row forks to sink_a + sink_b (real fork run via production paths).
        - BOTH branches' terminal outcomes are deleted so both are INCOMPLETE leaves
          (a "buffered" token is by definition pending — it has no terminal outcome yet;
          leaving sink_a's outcome in place would make it terminal, not buffered, and the
          mixed-state arithmetic would not engage).
        - The (now-incomplete) sink_a fork-child is placed into a genuine
          AggregationCheckpointState (typed dataclass) carrying its REAL row_data +
          contract, passed to CheckpointManager.create_checkpoint — so
          _get_buffered_checkpoint_token_ids returns the sink_a token (asserted as a
          precondition: proof the mixed-state path is live, not skipped).
        - sink_b is left incomplete and NON-buffered.

        The row therefore has one buffered-incomplete (sink_a) + one non-buffered
        incomplete (sink_b) token.  row_incomplete={sink_a, sink_b}; buffered={sink_a}.
        The correct exclusion (issubset) keeps the row (not ALL incomplete tokens are
        buffered → sink_b still needs reprocessing); excluding it would orphan sink_b.

        RED LEVER (NOT dispatch-disable — the mixed-state exclusion runs in
        get_unprocessed_rows BEFORE dispatch, so disabling dispatch is the wrong lever).
        Weaken recovery.py's exclusion condition (line ~498) from the correct
        `row_incomplete.issubset(buffered_token_ids)` (exclude only when ALL incomplete
        tokens are buffered) to an intersection test (exclude when ANY incomplete token
        is buffered).  The row is then dropped from unprocessed despite its non-buffered
        sink_b leaf.
        Observed (exclusion weakened to `if row_incomplete & buffered_token_ids:`):
        AssertionError — "mixed-state row must NOT be excluded ...: unprocessed=[]"
        (the row is silently dropped → sink_b would be orphaned).
        """
        from elspeth.contracts.aggregation_checkpoint import (
            AggregationCheckpointState,
            AggregationNodeCheckpoint,
            AggregationTokenCheckpoint,
        )
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.landscape.schema import token_outcomes_table

        db, _payload_store, _config, graph, _settings_obj, run_id, _sinks = self._run_nway_fork(
            n_source_rows=1, sink_names=["sink_a", "sink_b"]
        )

        # ── Locate both fork-child tokens and the source row's contract ──
        with db.engine.connect() as conn:
            children = conn.execute(
                text("""
                    SELECT t.token_id AS token_id, t.branch_name AS branch_name, t.row_id AS row_id,
                           t.fork_group_id AS fork_group_id
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    WHERE r.run_id = :run_id AND t.branch_name IN ('sink_a', 'sink_b')
                """),
                {"run_id": run_id},
            ).fetchall()
        by_branch = {c.branch_name: c for c in children}
        assert set(by_branch) == {"sink_a", "sink_b"}, f"expected sink_a + sink_b fork children; got {by_branch}"
        buffered_child = by_branch["sink_a"]  # will be placed into agg checkpoint state
        incomplete_child = by_branch["sink_b"]  # will be left incomplete + NON-buffered
        row_id = buffered_child.row_id

        # The run's stored schema contract is what the row was produced under — reuse it
        # for the buffered token's checkpoint entry (genuine, not fabricated).
        checkpoint_mgr = CheckpointManager(db)
        recovery_for_contract = RecoveryManager(db, checkpoint_mgr)
        source_contract = recovery_for_contract.verify_contract_integrity(run_id)
        contract_dict = source_contract.to_checkpoint_format()

        # ── Build a GENUINE AggregationCheckpointState buffering the sink_a token ──
        # The node_id key is opaque to _get_buffered_checkpoint_token_ids (it only reads
        # nodes.values() → tokens → token_id), but create_checkpoint validates node_id is
        # in the graph, so key the buffered state on a real node id (the source node).
        agg_node_key = str(graph.get_source())
        buffered_token_ckpt = AggregationTokenCheckpoint(
            token_id=buffered_child.token_id,
            row_id=row_id,
            branch_name="sink_a",
            fork_group_id=buffered_child.fork_group_id,
            join_group_id=None,
            expand_group_id=None,
            row_data={"value": 0},
            contract_version=source_contract.version_hash(),
            contract=contract_dict,
        )
        agg_state = AggregationCheckpointState(
            version="5.0",
            nodes={
                agg_node_key: AggregationNodeCheckpoint(
                    tokens=(buffered_token_ckpt,),
                    batch_id="batch-mixed-state-test",
                    elapsed_age_seconds=0.5,
                    count_fire_offset=None,
                    condition_fire_offset=None,
                    accepted_count_total=1,
                    completed_flush_count=0,
                )
            },
        )

        # ── Interrupt: delete BOTH branches' terminal outcomes ──
        # sink_a → buffered-incomplete (restored from checkpoint state, not reprocessed);
        # sink_b → non-buffered incomplete (must be reprocessed).  A "buffered" token must
        # itself be incomplete (no terminal outcome) for the mixed-state arithmetic to
        # engage — so sink_a's outcome is deleted too.
        with db.engine.connect() as conn:
            both_outcomes = conn.execute(
                text("""
                    SELECT o.outcome_id AS outcome_id
                    FROM token_outcomes o
                    WHERE o.token_id IN (:a, :b) AND o.completed = 1
                """),
                {"a": buffered_child.token_id, "b": incomplete_child.token_id},
            ).fetchall()
            assert len(both_outcomes) == 2, (
                f"both fork children must have a completed outcome to delete (precondition); got {len(both_outcomes)}"
            )
            for o in both_outcomes:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        # ── Create the checkpoint WITH the genuine aggregation state ──
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=buffered_child.token_id,
            node_id=agg_node_key,
            sequence_number=1,
            graph=graph,
            aggregation_state=agg_state,
        )

        recovery_mgr = RecoveryManager(db, checkpoint_mgr)

        # ── PRECONDITION: the mixed-state path is genuinely live ──
        # _get_buffered_checkpoint_token_ids must return the sink_a token (proof the
        # exclusion logic at lines 474-501 actually executes — `if buffered_token_ids
        # and unprocessed:`).  Without this, the test would pass vacuously.
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        buffered_ids = recovery_mgr._get_buffered_checkpoint_token_ids(checkpoint)
        assert buffered_child.token_id in buffered_ids, (
            f"PRECONDITION FAILED: the sink_a token must be in the checkpoint buffered set "
            f"(else the mixed-state exclusion path is skipped and the test is vacuous). "
            f"buffered_ids={buffered_ids}"
        )
        assert incomplete_child.token_id not in buffered_ids, (
            f"sink_b token must NOT be buffered (it is the non-buffered incomplete leaf); buffered_ids={buffered_ids}"
        )

        # ── MIXED-STATE INVARIANT: the row is NOT excluded ──
        # The row has one buffered (sink_a) + one non-buffered incomplete (sink_b) leaf.
        # get_unprocessed_rows must still return it (sink_b needs reprocessing).
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert row_id in unprocessed, (
            f"mixed-state row must NOT be excluded (only ALL-buffered rows are excluded); unprocessed={unprocessed}. "
            f"The row has a non-buffered incomplete sink_b leaf that still needs reprocessing — "
            f"excluding it would silently orphan sink_b."
        )

        # The incomplete (non-buffered) sink_b child must surface as a resume spec; the
        # buffered (sink_a) child must NOT (it is restored-and-flushed from the aggregation
        # buffer, not re-driven — get_incomplete_tokens_by_row excludes buffered tokens at
        # the token level via _get_buffered_checkpoint_token_ids' aggregation_state arm).
        # This is the function-level guard for the aggregation arm; the full-resume
        # double-emit guard is test_resume_aggregation_buffered_fork_branch_not_redriven.
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        incomplete_specs = [s for specs in by_row.values() for s in specs]
        spec_token_ids = {s.token_id for s in incomplete_specs}
        assert incomplete_child.token_id in spec_token_ids, (
            f"sink_b (non-buffered incomplete) must surface as a resume spec; specs={spec_token_ids}"
        )
        assert buffered_child.token_id not in spec_token_ids, (
            f"sink_a (aggregation-buffered) must NOT surface as a resume spec — re-driving a buffered "
            f"token double-emits; it is restored-and-flushed from the buffer instead. specs={spec_token_ids}"
        )

    def test_resume_expand_aggregation_all_buffered_row_excluded(self) -> None:
        """An aggregation-buffered expand row is excluded from resume re-drive at the ROW
        level — its tokens are restored-and-flushed from the aggregation buffer, never
        dispatched to resume_incomplete_token.

        This is the aggregation analogue of the coalesce held-branch defect
        (test_resume_coalesce_held_branch_not_redriven), and it documents WHY the
        aggregation case differs structurally from the coalesce case:

        REACHABILITY (verified against dag/builder.py + resume.py):
        - The buggy re-drive in run_resume_processing_loop only visits rows returned by
          get_unprocessed_rows (resume.py loops `for row_id, ... in unprocessed_rows`), and
          only dispatches tokens carrying a lineage field to resume_incomplete_token.
        - A token buffered in an aggregation acquires a lineage field only via fork / expand
          / join. fork→aggregation is STRUCTURALLY IMPOSSIBLE: the DAG builder
          (dag/builder.py ~492-505) requires every fork branch to terminate at a sink (name
          match) or a coalesce branch, and a fork branch cannot carry intermediate
          transforms unless it routes to a coalesce — an aggregation input is neither.
          (Verified: ExecutionGraph.from_plugin_instances on a fork→aggregation topology
          raises GraphValidationError: "Gate '<g>' has fork branch '<b>' with no
          destination. Fork branches must either: 1. Be listed in a coalesce 'branches'
          dict/list, or 2. Match a sink name exactly".)
        - expand→aggregation IS constructible, but ALL expand children of a row land in the
          SAME aggregation batch → the row is ALL-buffered (no non-buffered sibling) →
          get_unprocessed_rows excludes it → it is never visited by the resume loop → its
          buffered tokens are never dispatched.
        Therefore there is no mixed-state (one buffered + one non-buffered incomplete)
        aggregation row analogous to the coalesce case: fork→coalesce is allowed but
        fork→aggregation is not. The buffered-token exclusion in get_incomplete_tokens_by_row
        is correct defense-in-depth mirroring get_unprocessed_rows; the reachable handling
        is the ROW-level exclusion asserted here.

        Topology (production ExecutionGraph.from_plugin_instances + Orchestrator.run):
            source(1 row, items=[a, b]) → JSONExplode → aggregation(count=3; never fires on
            the 2 exploded children → both BUFFERED in the same batch) → sink 'output'

        Construction: run to completion, delete BOTH expand-child leaves' terminal outcomes
        (both become incomplete), and place BOTH into a genuine AggregationCheckpointState
        keyed on the real aggregation node id (all-buffered). Assert:
        - _get_buffered_checkpoint_token_ids returns BOTH leaves (precondition: the
          aggregation arm is live).
        - get_unprocessed_rows EXCLUDES the row (all its incomplete leaves are buffered).
        - get_incomplete_tokens_by_row returns NOTHING for the row (token-level exclusion;
          mirrors the row-level exclusion so the resume loop never dispatches them).

        RED LEVER (parallel to matrix #5): weakening get_incomplete_tokens_by_row to NOT
        exclude buffered tokens makes both buffered leaves surface as resume specs — which,
        for an all-buffered row that get_unprocessed_rows already excludes, drifts the
        token-level oracle from the row-level one and breaks the incomplete_by_row ⊆
        unprocessed_rows invariant the F1 dispatch relies on. The fix keeps the two
        functions aligned.
        Observed (buffered-exclusion disabled in get_incomplete_tokens_by_row):
        AssertionError — "all-buffered aggregation row must contribute NO incomplete specs
        (buffered tokens are excluded at the token level, mirroring get_unprocessed_rows);
        got ['<leaf0>', '<leaf1>']" — both buffered expand leaves wrongly surface as
        re-drivable resume specs.
        """
        import datetime

        from elspeth.contracts.aggregation_checkpoint import (
            AggregationCheckpointState,
            AggregationNodeCheckpoint,
            AggregationTokenCheckpoint,
        )
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.contracts.enums import RunStatus
        from elspeth.contracts.schema_contract import SchemaContract as _SchemaContract
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import AggregationSettings, CheckpointSettings, SinkSettings, TriggerConfig
        from elspeth.core.landscape.schema import token_outcomes_table
        from elspeth.plugins.transforms.json_explode import JSONExplode
        from tests.integration.pipeline.test_aggregation_checkpoint_bug import BatchCollectorTransform

        db = make_landscape_db()
        payload_store = MockPayloadStore()

        dt0 = datetime.datetime(2021, 1, 1, tzinfo=datetime.UTC)
        dt1 = datetime.datetime(2022, 2, 2, tzinfo=datetime.UTC)

        source = ListSource(
            [{"id": 1, "value": 10, "items": [{"ts": dt0}, {"ts": dt1}]}],
            on_success="explode_in",
        )
        sink = CollectSink("output")
        explode = JSONExplode({"array_field": "items", "output_field": "item", "include_index": True, "schema": {"mode": "observed"}})
        agg = BatchCollectorTransform()
        agg_transform = as_transform(agg)
        wired = wire_transforms(
            [as_transform(explode), agg_transform],
            source_connection="explode_in",
            final_sink="output",
            names=["explode", "agg"],
        )
        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(source),
            source_settings=SourceSettings(plugin=source.name, on_success="explode_in", options={}),
            transforms=wired,
            sinks={"output": as_sink(sink)},
            gates=[],
            aggregations={},
            coalesce_settings=[],
        )
        agg_node_id = graph.get_transform_id_map()[1]  # explode=0, agg=1
        agg_settings = AggregationSettings(
            name="agg",
            plugin="batch_collector",
            input="agg_in",
            on_success="output",
            on_error="discard",
            trigger=TriggerConfig(count=3, timeout_seconds=3600),
            output_mode="transform",
        )
        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(explode), agg_transform],
            sinks={"output": as_sink(sink)},
            aggregation_settings={agg_node_id: agg_settings},
            coalesce_settings=[],
        )
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")
        settings_obj = ElspethSettings(
            source={"plugin": "test", "on_success": "explode_in", "options": {}},
            sinks={"output": SinkSettings(plugin="test", on_write_failure="discard", options={})},
            aggregations=[agg_settings],
            checkpoint=checkpoint_settings,
        )

        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)
        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        run = orchestrator.run(config, graph=graph, settings=settings_obj, payload_store=payload_store)
        run_id = run.run_id
        assert run.status == RunStatus.COMPLETED, f"run-1 must complete: {run.status}"

        # ── The two expand-child LEAVES (buffered into the aggregation batch) ──
        with db.engine.connect() as conn:
            children = conn.execute(
                text("""
                    SELECT DISTINCT t.token_id AS token_id, t.row_id AS row_id, t.expand_group_id AS expand_group_id,
                           t.token_data_ref AS token_data_ref
                    FROM tokens t
                    JOIN rows r ON r.row_id = t.row_id
                    JOIN token_outcomes o ON o.token_id = t.token_id
                    WHERE r.run_id = :run_id AND t.expand_group_id IS NOT NULL
                      AND o.path = :consumed AND o.completed = 1
                    ORDER BY t.token_id
                """),
                {"run_id": run_id, "consumed": TerminalPath.BATCH_CONSUMED.value},
            ).fetchall()
        assert len(children) == 2, f"Expected exactly 2 expand-child leaves consumed into the aggregation batch; got {len(children)}"
        row_id = children[0].row_id

        def _envelope(token_data_ref: str) -> dict:
            raw = payload_store.retrieve(token_data_ref)
            return checkpoint_loads(raw.decode("utf-8"))

        # ── Interrupt: delete BOTH leaves' terminal (batch_consumed) outcomes ──
        with db.engine.connect() as conn:
            both = conn.execute(
                text("SELECT o.outcome_id AS outcome_id FROM token_outcomes o WHERE o.token_id IN (:a, :b) AND o.completed = 1"),
                {"a": children[0].token_id, "b": children[1].token_id},
            ).fetchall()
            assert len(both) == 2, f"both expand leaves must have a completed outcome to delete (precondition); got {len(both)}"
            for o in both:
                conn.execute(token_outcomes_table.delete().where(token_outcomes_table.c.outcome_id == o.outcome_id))
            conn.commit()

        # ── Build a GENUINE all-buffered AggregationCheckpointState (both leaves) ──
        token_ckpts = []
        for c in children:
            env = _envelope(c.token_data_ref)
            token_ckpts.append(
                AggregationTokenCheckpoint(
                    token_id=c.token_id,
                    row_id=row_id,
                    branch_name=None,
                    fork_group_id=None,
                    join_group_id=None,
                    expand_group_id=c.expand_group_id,
                    row_data=env["data"],
                    contract_version=_SchemaContract.from_checkpoint(dict(env["contract"])).version_hash(),
                    contract=env["contract"],
                )
            )
        agg_state = AggregationCheckpointState(
            version="5.0",
            nodes={
                str(agg_node_id): AggregationNodeCheckpoint(
                    tokens=tuple(token_ckpts),
                    batch_id="batch-all-buffered-test",
                    elapsed_age_seconds=0.5,
                    count_fire_offset=None,
                    condition_fire_offset=None,
                    accepted_count_total=len(token_ckpts),
                    completed_flush_count=0,
                )
            },
        )

        sink_node_ids = graph.get_sinks()
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=children[0].token_id,
            node_id=sink_node_ids[0],
            sequence_number=1,
            graph=graph,
            aggregation_state=agg_state,
        )

        recovery_mgr = RecoveryManager(db, checkpoint_mgr)

        # ── PRECONDITION: the aggregation arm is live — both leaves are buffered ──
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        buffered_ids = recovery_mgr._get_buffered_checkpoint_token_ids(checkpoint)
        assert {children[0].token_id, children[1].token_id}.issubset(buffered_ids), (
            f"PRECONDITION FAILED: both expand leaves must be in the checkpoint aggregation buffered set; buffered_ids={buffered_ids}"
        )

        # ── ROW-level exclusion: the all-buffered row is NOT in unprocessed_rows ──
        # This is the reachable mechanism: the resume loop iterates unprocessed_rows, so an
        # excluded row is never visited and its buffered tokens are never dispatched.
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert row_id not in unprocessed, (
            f"all-buffered aggregation row must be excluded from unprocessed_rows (its tokens are restored-and-flushed, "
            f"not re-driven); unprocessed={unprocessed}"
        )

        # ── TOKEN-level exclusion: get_incomplete_tokens_by_row returns nothing for the row ──
        # Mirrors the row-level exclusion (incomplete_by_row ⊆ unprocessed_rows). Without the
        # fix, both buffered leaves would surface here, drifting from get_unprocessed_rows.
        by_row = recovery_mgr.get_incomplete_tokens_by_row(run_id)
        row_specs = by_row.get(row_id, [])
        assert not row_specs, (
            f"all-buffered aggregation row must contribute NO incomplete specs (buffered tokens are excluded at the "
            f"token level, mirroring get_unprocessed_rows); got {[s.token_id for s in row_specs]}"
        )

    def test_resume_expand_coalesce_full_type_domain_roundtrip(self) -> None:
        """Task 12 addition (ADDENDUM 6): the token_data_ref envelope round-trips the FULL
        row-payload type domain through expand AND coalesce, with resume reconstruction.

        INTENT: mechanically enforce envelope completeness so a future type added to
        canonical_json (the proven row-payload domain) but NOT to checkpoint
        serialization is caught here.  checkpoint_dumps is called on the row payload at
        expand/coalesce time during a NORMAL run (Task 3 token_data_ref envelope), so a
        missing type is a happy-path crash, not just a resume bug.

        TYPE DOMAIN exercised (every type canonical_json accepts):
          str, int, float, bool, None, Decimal, datetime (tz-aware), date, time, bytes,
          UUID, nested dict, list, tuple, numpy scalar (np.int64 / np.float64 / np.bool_).

        END-TO-END PATH (production token_data_ref write + resume read):
          1. expand_token(child_payloads=[full-domain payload]) → writes the child's
             token_data_ref via checkpoint_dumps (the real expand envelope path).
          2. coalesce_tokens(merged_payload=full-domain payload) → writes the merged
             token's token_data_ref via checkpoint_dumps (the real coalesce envelope path).
          3. RecoveryManager.reconstruct_token_row(spec) on each → the resume read path:
             checkpoint_loads + SchemaContract.from_checkpoint (hash-validated, Tier-1).
          4. Assert every reconstructed value is byte/type-faithful to the original.

        NORMALIZATION CAVEATS (asserted, per ADDENDUM 6 + serialization.py):
          - numpy scalars are NORMALIZED to Python primitives (np.int64→int,
            np.float64→float, np.bool_→bool).  numpy-ness is not semantic; we assert the
            VALUE and the PYTHON-PRIMITIVE type, NOT numpy identity.
          - tuple round-trips as tuple (envelope-tagged); list stays list.  A list nested
            inside the payload comes back as list; a tuple comes back as tuple.

        This is fully end-to-end through the envelope (the production write+read paths),
        so no serializer-level fallback is needed — every type traverses the real
        token_data_ref path.  (A bare checkpoint_dumps/_loads symmetry check is also
        performed as a defense-in-depth cross-check on the exact same payload.)

        RED LEVER (this is a completeness GATE, not a dispatch cell — its lever is a
        missing serializer type, the exact regression class ADDENDUM 6 closes): drop one
        envelope-tag branch from checkpoint_dumps (e.g. the `bytes` handler) so that type
        is no longer serializable.  The expand_token write path calls checkpoint_dumps on
        the payload during the NORMAL run → happy-path crash.
        Observed (bytes handler dropped): TypeError — "Cannot serialize value of type
        'bytes' into a checkpoint payload ...".  This confirms the gate mechanically
        catches a domain type the serializer fails to handle.
        """
        import datetime as _dt
        from decimal import Decimal
        from uuid import UUID

        import numpy as np

        from elspeth.contracts.audit import TokenRef
        from elspeth.contracts.enums import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract
        from elspeth.contracts.schema_contract import PipelineRow as PLRow
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
        from elspeth.core.checkpoint.serialization import checkpoint_dumps, checkpoint_loads

        # ── The full-domain payload.  Each key documents the type under test. ──
        aware_dt = _dt.datetime(2024, 6, 15, 12, 30, 45, tzinfo=UTC)
        a_date = _dt.date(2023, 1, 2)
        a_time = _dt.time(8, 9, 10)
        a_uuid = UUID("12345678-1234-5678-1234-567812345678")
        a_decimal = Decimal("12345.6789")
        a_bytes = b"\x00\x01\x02binary\xff"

        domain_payload: dict[str, object] = {
            "f_str": "hello",
            "f_int": 42,
            "f_float": 3.14159,
            "f_bool": True,
            "f_none": None,
            "f_decimal": a_decimal,
            "f_datetime": aware_dt,
            "f_date": a_date,
            "f_time": a_time,
            "f_bytes": a_bytes,
            "f_uuid": a_uuid,
            "f_nested_dict": {"inner_str": "deep", "inner_dt": aware_dt, "inner_dec": a_decimal},
            "f_list": [1, "two", a_decimal, aware_dt],
            "f_tuple": ("a", 2, a_date),
            "f_np_int": np.int64(7),
            "f_np_float": np.float64(2.5),
            "f_np_bool": np.bool_(True),
        }

        def _assert_domain_faithful(data: dict[str, Any], *, context: str, tuple_as_tuple: bool) -> None:
            """Assert each reconstructed value is byte/type-faithful to the original.

            ``tuple_as_tuple``: at the bare checkpoint_dumps/_loads serializer level a tuple
            round-trips AS a tuple (envelope-tagged).  But the resume reconstruction returns
            a PipelineRow, and PipelineRow.to_dict() runs deep_thaw, which normalizes tuples
            to lists (a row's tuple is not a first-class row type — consistent with
            canonical_json treating tuples as JSON arrays).  So through the PipelineRow path
            the tuple legitimately comes back as a list.  The SERIALIZER fidelity (tuple→tuple)
            is proven by the bare round-trip; the row-path normalization (tuple→list) is the
            documented PipelineRow behaviour, not a serializer gap."""
            assert data["f_str"] == "hello", context
            assert data["f_int"] == 42 and type(data["f_int"]) is int, context
            assert abs(data["f_float"] - 3.14159) < 1e-12, context
            assert data["f_bool"] is True, context
            assert data["f_none"] is None, context
            # Decimal: exact value + decimal.Decimal instance (not float/str).
            assert isinstance(data["f_decimal"], Decimal) and data["f_decimal"] == a_decimal, f"{context}: f_decimal={data['f_decimal']!r}"
            # datetime: tz-aware datetime instance (not str).
            assert isinstance(data["f_datetime"], _dt.datetime) and data["f_datetime"] == aware_dt, context
            assert data["f_datetime"].tzinfo is not None, context
            # date: date instance, NOT datetime (subclass-ordering load-bearing in serializer).
            assert type(data["f_date"]) is _dt.date and data["f_date"] == a_date, f"{context}: f_date type={type(data['f_date']).__name__}"
            # time: time instance.
            assert isinstance(data["f_time"], _dt.time) and data["f_time"] == a_time, context
            # bytes: exact bytes (base64 round-trip).
            assert isinstance(data["f_bytes"], bytes) and data["f_bytes"] == a_bytes, f"{context}: f_bytes={data['f_bytes']!r}"
            # UUID: UUID instance, exact value.
            assert isinstance(data["f_uuid"], UUID) and data["f_uuid"] == a_uuid, context
            # nested dict: recursively type-faithful.
            nested = data["f_nested_dict"]
            assert isinstance(nested, dict), context
            assert nested["inner_str"] == "deep", context
            assert isinstance(nested["inner_dt"], _dt.datetime) and nested["inner_dt"] == aware_dt, context
            assert isinstance(nested["inner_dec"], Decimal) and nested["inner_dec"] == a_decimal, context
            # list: stays a list; elements type-faithful.
            lst = data["f_list"]
            assert isinstance(lst, list) and lst[0] == 1 and lst[1] == "two", context
            assert isinstance(lst[2], Decimal) and lst[2] == a_decimal, context
            assert isinstance(lst[3], _dt.datetime) and lst[3] == aware_dt, context
            # tuple: at the serializer level it round-trips as a TUPLE (envelope-tagged);
            # through the PipelineRow path it normalizes to a list (deep_thaw).  Either way
            # the ELEMENTS must be type-faithful.
            tup = data["f_tuple"]
            if tuple_as_tuple:
                assert isinstance(tup, tuple), f"{context}: f_tuple came back as {type(tup).__name__}, expected tuple (serializer level)"
            else:
                assert isinstance(tup, list), (
                    f"{context}: f_tuple via PipelineRow must normalize to list (deep_thaw); got {type(tup).__name__}"
                )
            assert tup[0] == "a" and tup[1] == 2, context
            assert type(tup[2]) is _dt.date and tup[2] == a_date, context
            # numpy scalars: NORMALIZED to Python primitives (numpy-ness is not semantic).
            assert data["f_np_int"] == 7 and type(data["f_np_int"]) is int, f"{context}: f_np_int type={type(data['f_np_int']).__name__}"
            assert abs(data["f_np_float"] - 2.5) < 1e-12 and type(data["f_np_float"]) is float, context
            assert data["f_np_bool"] is True and type(data["f_np_bool"]) is bool, (
                f"{context}: f_np_bool type={type(data['f_np_bool']).__name__}"
            )

        # ── Defense-in-depth: bare checkpoint_dumps/_loads symmetry on the exact payload ──
        # (Catches a serializer asymmetry independent of the expand/coalesce envelope.)
        bare_roundtrip = checkpoint_loads(checkpoint_dumps(domain_payload))
        _assert_domain_faithful(bare_roundtrip, context="bare checkpoint_dumps/_loads symmetry", tuple_as_tuple=True)

        # ── Build the contract covering the full domain (python_type=object for the
        #    catch-all keys is what an OBSERVED/FLEXIBLE row would carry). ──
        full_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=tuple(
                FieldContract(normalized_name=k, original_name=k, python_type=object, required=False, source="inferred")
                for k in domain_payload
            ),
            locked=True,
        )

        _OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
        payload_store = MockPayloadStore()
        db = make_landscape_db()
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=_OBSERVED_SCHEMA,
        )
        row = factory.data_flow.create_row(run_id=run.run_id, source_node_id=node.node_id, row_index=0, data={"seed": 1})
        parent = factory.data_flow.create_token(row_id=row.row_id)

        # ── (1) EXPAND path: child token_data_ref envelope carries the full domain ──
        children, _expand_group_id = factory.data_flow.expand_token(
            parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
            row_id=row.row_id,
            child_payloads=[dict(domain_payload)],
            output_contract=full_contract,
            step_in_pipeline=1,
        )
        assert len(children) == 1
        expand_child = children[0]
        assert expand_child.token_data_ref is not None, "expand child must carry token_data_ref"

        # ── (2) COALESCE path: merged token_data_ref envelope carries the full domain ──
        token_x = factory.data_flow.create_token(row_id=row.row_id)
        token_y = factory.data_flow.create_token(row_id=row.row_id)
        merged = factory.data_flow.coalesce_tokens(
            parent_refs=[
                TokenRef(token_id=token_x.token_id, run_id=run.run_id),
                TokenRef(token_id=token_y.token_id, run_id=run.run_id),
            ],
            row_id=row.row_id,
            merged_payload=dict(domain_payload),
            merged_contract=full_contract,
            step_in_pipeline=2,
        )
        assert merged.token_data_ref is not None, "merged token must carry token_data_ref"

        # ── (3) RESUME READ PATH: reconstruct_token_row on each (checkpoint_loads +
        #    SchemaContract.from_checkpoint, hash-validated Tier-1) ──
        recovery = RecoveryManager(db, CheckpointManager(db))
        source_row = PLRow({"seed": 1}, full_contract)

        expand_spec = IncompleteTokenSpec(
            token_id=expand_child.token_id,
            row_id=row.row_id,
            branch_name=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=expand_child.expand_group_id,
            token_data_ref=expand_child.token_data_ref,
            step_in_pipeline=1,
            max_attempt=-1,
        )
        expand_reconstructed = recovery.reconstruct_token_row(
            spec=expand_spec, run_id=run.run_id, source_row=source_row, payload_store=payload_store
        )
        _assert_domain_faithful(dict(expand_reconstructed.to_dict()), context="expand child reconstruct_token_row", tuple_as_tuple=False)

        merged_spec = IncompleteTokenSpec(
            token_id=merged.token_id,
            row_id=row.row_id,
            branch_name=None,
            fork_group_id=None,
            join_group_id=merged.join_group_id,
            expand_group_id=None,
            token_data_ref=merged.token_data_ref,
            step_in_pipeline=2,
            max_attempt=-1,
        )
        merged_reconstructed = recovery.reconstruct_token_row(
            spec=merged_spec, run_id=run.run_id, source_row=source_row, payload_store=payload_store
        )
        _assert_domain_faithful(dict(merged_reconstructed.to_dict()), context="merged token reconstruct_token_row", tuple_as_tuple=False)

        # ── Contract integrity: both reconstructions carry the full field set ──
        # SchemaContract.from_checkpoint (inside reconstruct_token_row) hash-validates the
        # contract; reaching here proves Tier-1 integrity held for both envelopes.
        assert {fc.normalized_name for fc in expand_reconstructed.contract.fields} == set(domain_payload), (
            "expand reconstructed contract must carry the full domain field set"
        )
        assert {fc.normalized_name for fc in merged_reconstructed.contract.fields} == set(domain_payload), (
            "merged reconstructed contract must carry the full domain field set"
        )
