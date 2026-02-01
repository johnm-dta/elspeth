"""Tests for recovery with fork partial completion scenarios.

BUG: P2-recovery-skips-forked-rows

When a row forks to multiple tokens and one child completes while another crashes,
the row was incorrectly marked as "completed" and skipped on resume.

Root cause: Previous approach found rows with ANY terminal token and excluded them.
This failed in fork scenarios: if child A completed but child B crashed,
the row was marked "done" because it had at least one terminal token.

Fix: Use LEFT JOIN to find rows where ANY token lacks terminal outcome.
A row is only "complete" when ALL its tokens have terminal outcomes.
"""

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Determinism, NodeType, RowOutcome, RunStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)


class TestForkPartialCompletion:
    """Tests for fork scenarios where some children complete, others crash."""

    @pytest.fixture
    def landscape_db(self, tmp_path) -> LandscapeDB:
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def _create_fork_graph(self) -> ExecutionGraph:
        """Create a graph with a fork gate splitting to two paths."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={})
        graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork", config={})
        graph.add_node("path_a", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={})
        graph.add_node("path_b", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={})
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="csv_sink", config={})
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="csv_sink", config={})

        graph.add_edge("source", "fork_gate", label="continue")
        graph.add_edge("fork_gate", "path_a", label="fork_a")
        graph.add_edge("fork_gate", "path_b", label="fork_b")
        graph.add_edge("path_a", "sink_a", label="continue")
        graph.add_edge("path_b", "sink_b", label="continue")

        return graph

    def _setup_fork_partial_scenario(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> str:
        """Create scenario where row forks and one child crashes.

        Row 0 forks to tokens tok-000-a (COMPLETED) and tok-000-b (NO OUTCOME - crashed).
        The row MUST appear in unprocessed because tok-000-b needs recovery.
        """
        run_id = "fork-partial-completion"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            # Create run in FAILED status
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            # Create nodes
            for node_id, node_type in [
                ("source", NodeType.SOURCE),
                ("fork_gate", NodeType.GATE),
                ("path_a", NodeType.TRANSFORM),
                ("path_b", NodeType.TRANSFORM),
                ("sink_a", NodeType.SINK),
                ("sink_b", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name="test",
                        node_type=node_type,
                        plugin_version="1.0",
                        determinism=Determinism.DETERMINISTIC,
                        config_hash="x",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create row 0
            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="source",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # Create parent token (before fork) with FORKED outcome
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    fork_group_id="fork-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-parent",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.FORKED.value,
                    is_terminal=1,  # FORKED is terminal for the token, but excluded from row completion check
                    recorded_at=now,
                    fork_group_id="fork-group-000",
                )
            )

            # Create child tokens (after fork)
            # tok-000-a: COMPLETED to sink_a
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000-a",
                    row_id="row-000",
                    fork_group_id="fork-group-000",
                    branch_name="fork_a",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-a",
                    run_id=run_id,
                    token_id="tok-000-a",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink_a",
                )
            )

            # tok-000-b: NO OUTCOME (crashed before completing)
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000-b",
                    row_id="row-000",
                    fork_group_id="fork-group-000",
                    branch_name="fork_b",
                    created_at=now,
                )
            )
            # NO token_outcomes entry for tok-000-b - simulates crash

            conn.commit()

        # Create checkpoint (at the completed token)
        graph = self._create_fork_graph()
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000-a",
            node_id="sink_a",
            sequence_number=1,
            graph=graph,
        )

        return run_id

    def test_fork_one_child_completes_one_crashes(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Fork: 2 children, one COMPLETED, one crashes (no outcome).

        Row 0 forks to tok-0-a (COMPLETED) and tok-0-b (no outcome).
        Row 0 MUST appear in unprocessed because tok-0-b needs recovery.

        This is the core P2 bug reproduction test.
        """
        run_id = self._setup_fork_partial_scenario(landscape_db, checkpoint_manager)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # BUG (before fix): Row excluded because ANY child had terminal outcome
        # FIX: Row included because not ALL children have terminal outcomes
        assert len(unprocessed) == 1, f"Expected 1 unprocessed row, got {len(unprocessed)}: {unprocessed}"
        assert "row-000" in unprocessed, f"Row 0 must be included - tok-000-b has no terminal outcome. Got: {unprocessed}"

    def test_fork_multiple_children_partial_completion(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Fork: 3 children, 2 complete, 1 crashes.

        Row 0 forks to tok-0-a, tok-0-b (both COMPLETED), tok-0-c (no outcome).
        Row 0 MUST appear in unprocessed.
        """
        run_id = "fork-3-children"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # 3 child tokens
            for suffix, has_outcome in [("a", True), ("b", True), ("c", False)]:
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-000-{suffix}",
                        row_id="row-000",
                        fork_group_id="fork-group-000",
                        branch_name=f"fork_{suffix}",
                        created_at=now,
                    )
                )
                if has_outcome:
                    conn.execute(
                        token_outcomes_table.insert().values(
                            outcome_id=f"out-000-{suffix}",
                            run_id=run_id,
                            token_id=f"tok-000-{suffix}",
                            outcome=RowOutcome.COMPLETED.value,
                            is_terminal=1,
                            recorded_at=now,
                            sink_name="sink",
                        )
                    )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000-a",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert len(unprocessed) == 1
        assert "row-000" in unprocessed

    def test_fork_all_children_complete_excluded(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Fork: All children complete -> row NOT in unprocessed.

        Row 0 forks to tok-0-a, tok-0-b (both COMPLETED).
        Row 0 should NOT appear in unprocessed.

        CRITICAL (per QA review): Verifies FORKED parent outcome doesn't cause
        false positive. Parent token has FORKED outcome (not in terminal list),
        but children have COMPLETED outcomes, so row should be excluded.
        """
        run_id = "fork-all-complete"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # Parent token with FORKED outcome (non-terminal)
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    fork_group_id="fork-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-parent",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.FORKED.value,
                    is_terminal=0,  # FORKED is NOT in terminal list
                    recorded_at=now,
                    fork_group_id="fork-group-000",
                )
            )

            # Both child tokens COMPLETED
            for suffix in ["a", "b"]:
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-000-{suffix}",
                        row_id="row-000",
                        fork_group_id="fork-group-000",
                        branch_name=f"fork_{suffix}",
                        created_at=now,
                    )
                )
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"out-000-{suffix}",
                        run_id=run_id,
                        token_id=f"tok-000-{suffix}",
                        outcome=RowOutcome.COMPLETED.value,
                        is_terminal=1,
                        recorded_at=now,
                        sink_name="sink",
                    )
                )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000-a",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # All children have terminal outcomes, so row is complete
        # The parent's FORKED outcome (non-terminal) should NOT cause false positive
        assert len(unprocessed) == 0, f"Expected 0 unprocessed rows (all children completed), got: {unprocessed}"

    def test_forked_parent_outcome_does_not_affect_completion(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Parent token's FORKED outcome doesn't affect row completion status.

        Per QA review: FORKED is NOT in the terminal outcomes list
        [COMPLETED, ROUTED, QUARANTINED, FAILED]. This test verifies that:
        - Parent token with FORKED outcome is correctly ignored
        - Only child token outcomes determine row completion
        """
        run_id = "forked-parent-ignored"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # ONLY parent token with FORKED outcome - no children exist yet
            # This simulates crash DURING the fork operation
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    fork_group_id="fork-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-parent",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.FORKED.value,
                    is_terminal=1,  # FORKED is terminal for the token, but excluded from row completion check
                    recorded_at=now,
                    fork_group_id="fork-group-000",
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Parent has FORKED (non-terminal), no children exist
        # Row MUST be in unprocessed because no terminal outcome exists
        assert len(unprocessed) == 1
        assert "row-000" in unprocessed

    def test_row_with_no_tokens_included(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Row with no tokens (never started) included in unprocessed."""
        run_id = "no-tokens"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: has token with COMPLETED outcome
            # Row 1: has no tokens at all (never started processing)
            for i in range(2):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="sink",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )

            # Only create token for row 0
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink",
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Row 0 is complete (has terminal outcome)
        # Row 1 has no tokens - must be in unprocessed (LEFT JOIN catches this)
        assert len(unprocessed) == 1
        assert "row-001" in unprocessed
        assert "row-000" not in unprocessed


class TestTerminalOutcomeValidation:
    """Tests validating terminal outcome filtering (per QA review)."""

    @pytest.fixture
    def landscape_db(self, tmp_path) -> LandscapeDB:
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def test_quarantined_rows_excluded_from_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """QUARANTINED rows are NOT re-emitted during recovery.

        QUARANTINED is terminal. Auto-retrying quarantined data
        would violate audit policy.
        """
        run_id = "quarantined-excluded"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: QUARANTINED
            # Row 1: No outcome
            for i in range(2):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="sink",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{i:03d}",
                        row_id=f"row-{i:03d}",
                        created_at=now,
                    )
                )

            # Only row 0 has QUARANTINED outcome
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.QUARANTINED.value,
                    is_terminal=1,
                    recorded_at=now,
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Row 0 is QUARANTINED (terminal) - should NOT be re-processed
        # Row 1 has no outcome - should be processed
        assert len(unprocessed) == 1
        assert "row-001" in unprocessed
        assert "row-000" not in unprocessed  # QUARANTINED is terminal

    def test_failed_rows_excluded_from_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """FAILED rows are NOT re-emitted during recovery.

        FAILED is terminal. These rows require manual intervention.
        """
        run_id = "failed-excluded"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: FAILED
            # Row 1: No outcome
            for i in range(2):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="sink",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{i:03d}",
                        row_id=f"row-{i:03d}",
                        created_at=now,
                    )
                )

            # Only row 0 has FAILED outcome
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.FAILED.value,
                    is_terminal=1,
                    recorded_at=now,
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Row 0 is FAILED (terminal) - should NOT be re-processed
        # Row 1 has no outcome - should be processed
        assert len(unprocessed) == 1
        assert "row-001" in unprocessed
        assert "row-000" not in unprocessed  # FAILED is terminal

    def test_buffered_tokens_trigger_recovery(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Tokens with BUFFERED outcome (coalesce barrier) trigger recovery.

        BUFFERED is non-terminal. Rows with only BUFFERED tokens must
        appear in unprocessed list.
        """
        run_id = "buffered-non-terminal"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: BUFFERED (waiting at coalesce barrier)
            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    join_group_id="join-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.BUFFERED.value,
                    is_terminal=0,  # BUFFERED is NOT terminal
                    recorded_at=now,
                    join_group_id="join-group-000",
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Row 0 has BUFFERED outcome (non-terminal) - needs reprocessing
        assert len(unprocessed) == 1
        assert "row-000" in unprocessed

    def test_consumed_in_batch_excluded_from_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """CONSUMED_IN_BATCH rows are NOT re-emitted during recovery.

        CONSUMED_IN_BATCH is terminal at row level - the row was absorbed
        into an aggregation. Batch-level recovery handles whether the
        batch output was produced.
        """
        run_id = "consumed-in-batch-excluded"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: CONSUMED_IN_BATCH
            # Row 1: No outcome
            for i in range(2):
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"row-{i:03d}",
                        run_id=run_id,
                        source_node_id="sink",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{i:03d}",
                        row_id=f"row-{i:03d}",
                        created_at=now,
                    )
                )

            # Only row 0 has CONSUMED_IN_BATCH outcome
            # Note: batch_id omitted to avoid FK constraint (testing recovery logic, not batch schema)
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.CONSUMED_IN_BATCH.value,
                    is_terminal=1,
                    recorded_at=now,
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Row 0 is CONSUMED_IN_BATCH (terminal) - should NOT be re-processed
        # Row 1 has no outcome - should be processed
        assert len(unprocessed) == 1
        assert "row-001" in unprocessed
        assert "row-000" not in unprocessed

    def test_coalesced_rows_excluded_from_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """COALESCED rows are NOT re-emitted during recovery.

        COALESCED is terminal - the token was merged in a join operation.
        The merged token (created by coalesce) carries forward.
        """
        run_id = "coalesced-excluded"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Row 0: forked to 2 children, both coalesced, merged token completed
            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # Two forked children that were coalesced
            for suffix in ["a", "b"]:
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-000-{suffix}",
                        row_id="row-000",
                        fork_group_id="fork-group-000",
                        join_group_id="join-group-000",
                        branch_name=f"path_{suffix}",
                        created_at=now,
                    )
                )
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"out-000-{suffix}",
                        run_id=run_id,
                        token_id=f"tok-000-{suffix}",
                        outcome=RowOutcome.COALESCED.value,
                        is_terminal=1,
                        recorded_at=now,
                        join_group_id="join-group-000",
                    )
                )

            # Merged token that completed to sink
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000-merged",
                    row_id="row-000",
                    join_group_id="join-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-merged",
                    run_id=run_id,
                    token_id="tok-000-merged",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink",
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000-merged",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # All tokens have terminal outcomes (COALESCED + COMPLETED)
        # Row should NOT be in unprocessed
        assert len(unprocessed) == 0, f"Expected 0 unprocessed rows (all coalesced + merged complete), got: {unprocessed}"

    def test_expanded_parent_triggers_recovery_if_children_incomplete(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """EXPANDED is a delegation marker like FORKED.

        When a batch is deaggregated (expanded), the parent token gets
        EXPANDED outcome and children carry forward. If children don't
        have terminal outcomes, row needs recovery.
        """
        run_id = "expanded-delegation"
        now = datetime.now(UTC)

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="test",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="sink",
                    row_index=0,
                    source_data_hash="hash0",
                    created_at=now,
                )
            )

            # Parent token with EXPANDED outcome (deaggregation parent)
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    expand_group_id="expand-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-parent",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.EXPANDED.value,
                    is_terminal=1,  # EXPANDED is terminal for the parent token
                    recorded_at=now,
                )
            )

            # Expanded child A: COMPLETED
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000-expanded-a",
                    row_id="row-000",
                    expand_group_id="expand-group-000",
                    created_at=now,
                )
            )
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000-a",
                    run_id=run_id,
                    token_id="tok-000-expanded-a",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink",
                )
            )

            # Expanded child B: NO OUTCOME (crashed)
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000-expanded-b",
                    row_id="row-000",
                    expand_group_id="expand-group-000",
                    created_at=now,
                )
            )
            # NO outcome for child B

            conn.commit()

        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-000-expanded-a",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # Child B has no outcome → row needs recovery
        # Parent's EXPANDED outcome is excluded (delegation marker)
        assert len(unprocessed) == 1
        assert "row-000" in unprocessed
