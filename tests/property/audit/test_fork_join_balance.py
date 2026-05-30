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

        Returns (db, payload_store, config, graph, settings_obj, run_id) for
        the completed run.  The same payload_store MUST be used for both
        the initial run and the resume (merged token_data_ref lives there).
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
        return db, payload_store, config, graph, settings_obj, run_id

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

        db, payload_store, config, graph, settings_obj, run_id = self._setup_coalesce_pipeline()

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

        db, payload_store, config, graph, settings_obj, run_id = self._setup_coalesce_pipeline()

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

        VALUE FIDELITY: dt0=datetime(2021,1,1) for child0, dt1=datetime(2022,2,2)
        for child1.  After interrupting child1 and resuming, retrieve child1's
        token_data_ref envelope and assert it carries ts=dt1 as a datetime.datetime
        INSTANCE (not str), proving checkpoint_dumps/_loads type fidelity AND that
        child1 carries ITS OWN value (not the sibling's dt0).

        Decimal was considered as a second type-fidelity witness but is not
        supported by checkpoint_dumps (serializer handles int/str/float/bool/
        datetime/tuple/None — not decimal.Decimal).  The task's "datetime + Decimal"
        framing is explicit about adapting to what the fixture emits.  datetime
        suffices: it is the only non-JSON-native type checkpoint_dumps preserves, and
        distinct per-child values are achievable by embedding the datetimes directly
        in the exploded array.

        RED (dispatch disabled, fork_expand_coalesce_specs branch commented):
        process_existing_row re-runs the entire source row from the beginning,
        which triggers JSONExplode again and creates TWO NEW expand children in
        addition to the existing (run-1) children (total 4 expand children in DB).
        The conservation assertion fires first (before the I1a sweep).
        Observed: AssertionError — 'Resume must NOT re-expand (no new children
        minted): expected 2 expand children, got 4.'
        """
        import datetime

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

        # Source: one row whose 'items' contains the two distinct datetime values.
        # JSONExplode explodes this into child0 (ts=dt0, item_index=0) and
        # child1 (ts=dt1, item_index=1).  Each child's 'ts' IS the exploded item
        # value — distinct per child — proving value<->token alignment.
        #
        # TOPOLOGY: source → JSONExplode → PassTransform → sink
        # The PassTransform after JSONExplode is required for resume: the expand child's
        # resume_incomplete_token path calls process_token(current_node_id=after) where
        # `after = resolve_next_node(explode_node)`.  Without a downstream transform node,
        # `after` would be None, which process_token rejects (no sink context for branchless
        # tokens).  The PassTransform is the re-drive target — the child continues from there
        # rather than from the expand node itself.
        source = ListSource([{"score": 1, "items": [dt0, dt1]}], on_success="explode_in")
        sink = CollectSink("output")

        explode = JSONExplode(
            {
                "array_field": "items",
                "output_field": "ts",  # each child gets its own datetime as 'ts'
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

        # ── VALUE FIDELITY: child1's envelope carries ts=dt1 as datetime ──────
        # Retrieve child1's token_data_ref envelope (run-1 artifact, unmodified by resume).
        # This is the payload the resume path's reconstruct_token_row() consumed to
        # re-drive child1.  Asserting type+value here proves:
        #   (a) checkpoint_dumps preserved the datetime type (not stringified it),
        #   (b) child1's envelope carries ITS OWN dt1, not the sibling's dt0.
        raw = payload_store.retrieve(child1_token_data_ref)
        env = checkpoint_loads(raw.decode("utf-8"))

        assert isinstance(env, dict) and "data" in env and "contract" in env, (
            f"child1 token_data_ref is not a {{data, contract}} envelope; "
            f"got keys={sorted(env.keys()) if isinstance(env, dict) else type(env).__name__!r}"
        )
        child1_data = env["data"]

        # Type fidelity: 'ts' field must come back as datetime.datetime, not str.
        assert isinstance(child1_data["ts"], datetime.datetime), (
            f"Type fidelity failure: 'ts' field came back as {type(child1_data['ts']).__name__!r}, "
            f"not datetime — checkpoint_dumps was not used (canonical_json stringifies datetime)"
        )
        assert child1_data["ts"].tzinfo is not None, "Restored datetime must be timezone-aware"

        # Value alignment: child1's 'ts' must be dt1 (NOT dt0 — the sibling's value).
        assert child1_data["ts"] == dt1, (
            f"Value alignment failure: child1 carries ts={child1_data['ts']!r}, "
            f"expected dt1={dt1!r}. Got dt0={dt0!r} would mean zip(strict=True) alignment is broken."
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
