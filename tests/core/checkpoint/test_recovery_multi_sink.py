"""Tests for recovery in multi-sink scenarios.

BUG: P1-2026-01-22-recovery-skips-rows-multi-sink

When rows are routed to different sinks in interleaved order, recovery can skip
rows that never completed their sink write. The current implementation uses a
single row_index boundary from the latest checkpoint, which fails when sinks
process rows out of order.

Example:
    Row 0 → sink_a (completes, checkpointed)
    Row 1 → sink_b (fails before completing)
    Row 2 → sink_a (completes, checkpointed at row_index=2)

    Recovery sees checkpoint at row_index=2 and returns only rows with index > 2.
    Row 1 is silently skipped, losing data.

The correct fix is to query for tokens lacking terminal sink outcomes rather
than using row_index boundaries.
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


class TestMultiSinkRecovery:
    """Tests for recovery with multiple sinks and interleaved routing."""

    @pytest.fixture
    def landscape_db(self, tmp_path) -> LandscapeDB:
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        return RecoveryManager(landscape_db, checkpoint_manager)

    def _create_multi_sink_graph(self) -> ExecutionGraph:
        """Create a graph with a gate routing to two sinks."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={})
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="router", config={})
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="csv_sink", config={})
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="csv_sink", config={})

        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink_a", label="continue")  # Default path
        graph.add_edge("gate", "sink_b", label="route_b")  # Routed path

        return graph

    def _setup_interleaved_multi_sink_scenario(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> str:
        """Create scenario where rows interleave between sinks and one sink fails.

        Scenario:
            Row 0 → sink_a (COMPLETED, checkpointed)
            Row 1 → sink_b (FAILS - no terminal outcome)
            Row 2 → sink_a (COMPLETED, checkpointed at row_index=2)
            Row 3 → sink_a (not started - no token outcome)
            Row 4 → sink_b (not started - no token outcome)

        After failure, latest checkpoint is at row_index=2 (from row 2's sink_a write).
        Row 1 was routed to sink_b but failed before completing.

        Expected: get_unprocessed_rows should return rows 1, 3, 4
        Bug: Current implementation returns only rows 3, 4 (skips row 1)
        """
        run_id = "multi-sink-interleaved"
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
                ("gate", NodeType.GATE),
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

            # Create 5 rows (indices 0-4)
            for i in range(5):
                row_id = f"row-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id="source",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"tok-{i:03d}",
                        row_id=row_id,
                        created_at=now,
                    )
                )

            # Record token outcomes:
            # - Row 0 → sink_a: COMPLETED (terminal)
            # - Row 1 → sink_b: NO OUTCOME (failed before completing)
            # - Row 2 → sink_a: COMPLETED (terminal)
            # - Row 3, 4: NO OUTCOME (not started)

            # Row 0 completed to sink_a
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink_a",
                )
            )

            # Row 2 completed to sink_a (AFTER row 1 was routed to sink_b)
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-002",
                    run_id=run_id,
                    token_id="tok-002",
                    outcome=RowOutcome.COMPLETED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="sink_a",
                )
            )

            conn.commit()

        # Create checkpoint at row 2 (the last successfully completed sink write)
        # This simulates the checkpoint being at the highest row_index that reached a sink
        graph = self._create_multi_sink_graph()
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="sink_a",
            sequence_number=3,  # Sequence can differ from row_index
            graph=graph,
        )

        return run_id

    def test_interleaved_multi_sink_includes_failed_sink_rows(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """Recovery must include rows routed to failed sink, not just rows after checkpoint.

        This is the core bug reproduction test.

        Scenario:
            Row 0 → sink_a (COMPLETED)
            Row 1 → sink_b (FAILED - no terminal outcome)
            Row 2 → sink_a (COMPLETED, checkpoint at row_index=2)
            Row 3, 4: not started

        BUG: Current code returns rows with row_index > 2, which gives [row-003, row-004]
        FIX: Should return rows without terminal sink outcomes: [row-001, row-003, row-004]
        """
        run_id = self._setup_interleaved_multi_sink_scenario(landscape_db, checkpoint_manager)

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        # The bug: Current implementation only returns rows 3 and 4
        # (row_index > 2 from checkpoint)
        #
        # The fix: Should return rows 1, 3, and 4
        # (tokens without terminal sink outcomes)

        assert len(unprocessed) == 3, f"Expected 3 unprocessed rows, got {len(unprocessed)}: {unprocessed}"

        # Row 1 must be included - it was routed to sink_b but never completed
        assert "row-001" in unprocessed, f"Row 1 must be included - it was routed to sink_b but failed. Got: {unprocessed}"

        # Rows 3 and 4 must also be included - they never started
        assert "row-003" in unprocessed, f"Row 3 must be included. Got: {unprocessed}"
        assert "row-004" in unprocessed, f"Row 4 must be included. Got: {unprocessed}"

        # Rows 0 and 2 should NOT be included - they completed successfully
        assert "row-000" not in unprocessed, f"Row 0 should not be included - it completed to sink_a. Got: {unprocessed}"
        assert "row-002" not in unprocessed, f"Row 2 should not be included - it completed to sink_a. Got: {unprocessed}"

    def test_all_sinks_completed_returns_empty(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """When all rows have terminal sink outcomes, recovery returns empty list."""
        run_id = "all-completed"
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

            # Create 3 rows, all with terminal outcomes
            for i in range(3):
                row_id = f"row-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
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
                        row_id=row_id,
                        created_at=now,
                    )
                )
                conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"out-{i:03d}",
                        run_id=run_id,
                        token_id=f"tok-{i:03d}",
                        outcome=RowOutcome.COMPLETED.value,
                        is_terminal=1,
                        recorded_at=now,
                        sink_name="sink",
                    )
                )

            conn.commit()

        # Create checkpoint at last row
        graph = ExecutionGraph()
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test", config={})
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="sink",
            sequence_number=3,
            graph=graph,
        )

        unprocessed = recovery_manager.get_unprocessed_rows(run_id)

        assert unprocessed == [], f"Expected empty list, got: {unprocessed}"

    def test_routed_outcome_counts_as_terminal(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        recovery_manager: RecoveryManager,
    ) -> None:
        """ROUTED outcome (gate routing to named sink) also counts as terminal."""
        run_id = "with-routed"
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

            # Row 0: ROUTED (terminal)
            # Row 1: No outcome (incomplete)
            for i in range(2):
                row_id = f"row-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
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
                        row_id=row_id,
                        created_at=now,
                    )
                )

            # Only row 0 has ROUTED outcome
            conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id="out-000",
                    run_id=run_id,
                    token_id="tok-000",
                    outcome=RowOutcome.ROUTED.value,
                    is_terminal=1,
                    recorded_at=now,
                    sink_name="error_sink",
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

        # Row 0 is ROUTED (terminal) - should not be in unprocessed
        # Row 1 has no outcome - should be in unprocessed
        assert len(unprocessed) == 1
        assert "row-001" in unprocessed
        assert "row-000" not in unprocessed
