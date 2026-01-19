"""Integration test for aggregation crash recovery.

End-to-end test simulating:
1. Run starts, processes rows, creates batch
2. Checkpoint created with aggregation state
3. Crash during batch flush
4. Recovery: restore state, retry batch, continue
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.enums import BatchStatus
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class TestAggregationRecoveryIntegration:
    """End-to-end test for aggregation crash recovery."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up complete test environment."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        recorder = LandscapeRecorder(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "recorder": recorder,
        }

    def test_full_recovery_cycle(self, test_env: dict[str, Any]) -> None:
        """Simulate crash during flush and verify recovery works."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        recorder = test_env["recorder"]

        # === PHASE 1: Normal execution until crash ===

        run = recorder.begin_run(
            config={"aggregation": {"trigger": {"count": 3}}},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register nodes using raw SQL to avoid schema_config requirement
        # (matching pattern from test_checkpoint_recovery.py)
        self._register_nodes_raw(db, run.run_id)

        # Record source rows and create tokens
        tokens = []
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch and add members
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens):
            recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Simulate checkpoint before flush
        agg_state = {"buffer": [0, 100, 200], "sum": 300, "count": 3}
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="sum_aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
        )

        # Simulate crash during flush
        recorder.update_batch_status(batch.batch_id, "executing")
        recorder.complete_run(run.run_id, status="failed")

        # === PHASE 2: Verify recovery is possible ===

        check = recovery_mgr.can_resume(run.run_id)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        resume_point = recovery_mgr.get_resume_point(run.run_id)
        assert resume_point is not None
        assert resume_point.aggregation_state == agg_state

        # === PHASE 3: Execute recovery steps ===

        # Find incomplete batches
        incomplete = recorder.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].batch_id == batch.batch_id
        assert incomplete[0].status == BatchStatus.EXECUTING

        # Mark executing as failed (crash interrupted)
        recorder.update_batch_status(batch.batch_id, "failed")

        # Retry the batch
        retry_batch = recorder.retry_batch(batch.batch_id)
        assert retry_batch.attempt == 1
        assert retry_batch.status == BatchStatus.DRAFT

        # Verify members were copied
        retry_members = recorder.get_batch_members(retry_batch.batch_id)
        assert len(retry_members) == 3

        # === PHASE 4: Verify final state ===

        # Original batch is failed
        original_batch = recorder.get_batch(batch.batch_id)
        assert original_batch is not None
        assert original_batch.status == BatchStatus.FAILED

        # Retry batch exists
        all_batches = recorder.get_batches(run.run_id, node_id="sum_aggregator")
        assert len(all_batches) == 2  # Original + retry

        # Verify attempt progression
        attempts = sorted([b.attempt for b in all_batches])
        assert attempts == [0, 1]

    def test_recovery_with_multiple_aggregations(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify recovery handles multiple aggregation nodes independently."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        recorder = test_env["recorder"]

        run = recorder.begin_run(
            config={"aggregations": ["sum", "count"]},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register multiple aggregation nodes
        self._register_nodes_raw(
            db,
            run.run_id,
            extra_nodes=[
                ("count_aggregator", "count_agg", "aggregation"),
            ],
        )

        # Create rows and tokens
        tokens = []
        for i in range(4):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 10},
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch for sum_aggregator (completed successfully)
        sum_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens[:2]):
            recorder.add_batch_member(sum_batch.batch_id, token.token_id, ordinal=i)
        recorder.update_batch_status(sum_batch.batch_id, "completed")

        # Create batch for count_aggregator (crashed during execution)
        count_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="count_aggregator",
        )
        for i, token in enumerate(tokens[2:]):
            recorder.add_batch_member(count_batch.batch_id, token.token_id, ordinal=i)
        recorder.update_batch_status(count_batch.batch_id, "executing")

        # Checkpoint at last processed token
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[3].token_id,
            node_id="count_aggregator",
            sequence_number=3,
            aggregation_state={"count": 2},
        )

        recorder.complete_run(run.run_id, status="failed")

        # Verify recovery
        check = recovery_mgr.can_resume(run.run_id)
        assert check.can_resume is True

        # Only count_aggregator batch should be incomplete
        incomplete = recorder.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].aggregation_node_id == "count_aggregator"
        assert incomplete[0].status == BatchStatus.EXECUTING

    def test_recovery_preserves_batch_member_order(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify batch member ordinals are preserved through retry."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recorder = test_env["recorder"]

        run = recorder.begin_run(
            config={"test": "order_preservation"},
            canonical_version="sha256-rfc8785-v1",
        )

        self._register_nodes_raw(db, run.run_id)

        # Create 5 rows with specific order
        tokens = []
        for i in range(5):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"seq": i, "data": f"item_{i}"},
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch with specific member ordering
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        # Add in reverse order to test ordinal preservation
        for i, token in enumerate(reversed(tokens)):
            recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Mark as failed for retry
        recorder.update_batch_status(batch.batch_id, "failed")

        # Checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="sum_aggregator",
            sequence_number=4,
        )

        # Retry
        retry_batch = recorder.retry_batch(batch.batch_id)

        # Verify member order is preserved
        original_members = recorder.get_batch_members(batch.batch_id)
        retry_members = recorder.get_batch_members(retry_batch.batch_id)

        assert len(retry_members) == len(original_members)
        for orig, retry in zip(original_members, retry_members, strict=False):
            assert orig.token_id == retry.token_id
            assert orig.ordinal == retry.ordinal

    def test_recovery_cannot_retry_non_failed_batch(
        self, test_env: dict[str, Any]
    ) -> None:
        """Verify retry_batch only works on failed batches."""
        db = test_env["db"]
        recorder = test_env["recorder"]

        run = recorder.begin_run(
            config={"test": "retry_validation"},
            canonical_version="sha256-rfc8785-v1",
        )

        self._register_nodes_raw(db, run.run_id)

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"id": 0},
        )
        token = recorder.create_token(row_id=row.row_id)

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        recorder.add_batch_member(batch.batch_id, token.token_id, ordinal=0)

        # Test with draft status
        with pytest.raises(ValueError, match="Can only retry failed batches"):
            recorder.retry_batch(batch.batch_id)

        # Test with executing status
        recorder.update_batch_status(batch.batch_id, "executing")
        with pytest.raises(ValueError, match="Can only retry failed batches"):
            recorder.retry_batch(batch.batch_id)

        # Test with completed status
        recorder.update_batch_status(batch.batch_id, "completed")
        with pytest.raises(ValueError, match="Can only retry failed batches"):
            recorder.retry_batch(batch.batch_id)

    def _register_nodes_raw(
        self,
        db: LandscapeDB,
        run_id: str,
        *,
        extra_nodes: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Register nodes using raw SQL to avoid schema_config requirement.

        Args:
            db: LandscapeDB instance
            run_id: Run to register nodes for
            extra_nodes: Optional list of (node_id, plugin_name, node_type) tuples
        """
        from elspeth.core.landscape.schema import nodes_table

        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            # Source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type="source",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Sum aggregator node
            conn.execute(
                nodes_table.insert().values(
                    node_id="sum_aggregator",
                    run_id=run_id,
                    plugin_name="sum_agg",
                    node_type="aggregation",
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Extra nodes
            if extra_nodes:
                for node_id, plugin_name, node_type in extra_nodes:
                    conn.execute(
                        nodes_table.insert().values(
                            node_id=node_id,
                            run_id=run_id,
                            plugin_name=plugin_name,
                            node_type=node_type,
                            plugin_version="1.0",
                            determinism="deterministic",
                            config_hash="test",
                            config_json="{}",
                            registered_at=now,
                        )
                    )

            conn.commit()
