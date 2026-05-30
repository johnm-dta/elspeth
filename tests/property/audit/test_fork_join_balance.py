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

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts import CoalesceName, GateName, RoutingAction, RoutingMode, SinkName
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import _LEGAL_TERMINAL_PAIRS, Determinism, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.checkpoint.serialization import checkpoint_loads
from elspeth.core.config import CoalesceSettings, GateSettings, SourceSettings
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
