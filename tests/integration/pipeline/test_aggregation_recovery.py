# tests/integration/pipeline/test_aggregation_recovery.py
"""Integration test for aggregation crash recovery.

End-to-end test simulating:
1. Run starts, processes rows, creates batch
2. Checkpoint created with aggregation state
3. Crash during batch flush
4. Recovery: restore state, retry batch, continue

Migrated from tests/integration/test_aggregation_recovery.py.

NOTE: Manual ExecutionGraph construction is intentional here. These tests
verify checkpoint/recovery logic for aggregation nodes and need specific
node IDs matching stored checkpoint records. These IDs must be deterministic
and match what's in the database, which makes from_plugin_instances()
inappropriate (it generates hash-based IDs). See
tests/integration/checkpoint/conftest.py for the same rationale.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.aggregation_checkpoint import (
    AggregationCheckpointState,
    AggregationNodeCheckpoint,
    AggregationTokenCheckpoint,
)
from elspeth.contracts.enums import BatchStatus, Determinism, NodeType, RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.landscape import make_factory


def _create_test_schema_contract() -> SchemaContract:
    """Create a minimal schema contract for test run creation."""
    field_contracts = (
        FieldContract(
            normalized_name="test_field",
            original_name="test_field",
            python_type=str,
            required=True,
            source="declared",
        ),
    )
    return SchemaContract(fields=field_contracts, mode="FIXED", locked=True)


class TestAggregationRecoveryIntegration:
    """End-to-end test for aggregation crash recovery."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up complete test environment."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        factory = make_factory(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "factory": factory,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for aggregation recovery tests."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        agg_config = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"mode": "observed"}},
            "schema": {"mode": "observed"},
        }
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("sum_aggregator", node_type=NodeType.AGGREGATION, plugin_name="test", config=agg_config)
        graph.add_node("count_aggregator", node_type=NodeType.AGGREGATION, plugin_name="count_agg", config=agg_config)
        return graph

    def test_full_recovery_cycle(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Simulate crash during flush and verify recovery works."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        factory: RecorderFactory = test_env["factory"]

        # === PHASE 1: Normal execution until crash ===

        run = factory.run_lifecycle.begin_run(
            config={"aggregation": {"trigger": {"count": 3}}},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=_create_test_schema_contract(),
        )

        # Register nodes using raw SQL to avoid schema_config requirement
        self._register_nodes_raw(db, run.run_id)

        # Record source rows and create tokens
        tokens = []
        for i in range(3):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch and add members
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens):
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Simulate checkpoint before flush — construct typed DTO
        agg_state = AggregationCheckpointState(
            version="4.0",
            nodes={
                "sum_aggregator": AggregationNodeCheckpoint(
                    tokens=tuple(
                        AggregationTokenCheckpoint(
                            token_id=t.token_id,
                            row_id=t.row_id,
                            branch_name=None,
                            fork_group_id=None,
                            join_group_id=None,
                            expand_group_id=None,
                            row_data={"id": i, "value": i * 100},
                            contract_version="test",
                            contract={"mode": "OBSERVED", "locked": True, "version_hash": "test", "fields": []},
                        )
                        for i, t in enumerate(tokens)
                    ),
                    batch_id=batch.batch_id,
                    elapsed_age_seconds=0.0,
                    count_fire_offset=None,
                    condition_fire_offset=None,
                ),
            },
        )
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="sum_aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
            graph=mock_graph,
        )

        # Simulate crash during flush
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # === PHASE 2: Verify recovery is possible ===

        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        resume_point = recovery_mgr.get_resume_point(run.run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None
        assert "sum_aggregator" in resume_point.aggregation_state.nodes

        # === PHASE 3: Execute recovery steps ===

        # Find incomplete batches
        incomplete = factory.execution.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].batch_id == batch.batch_id
        assert incomplete[0].status == BatchStatus.EXECUTING

        # Mark executing as failed (crash interrupted)
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.FAILED)

        # Retry the batch
        retry_batch = factory.execution.retry_batch(batch.batch_id)
        assert retry_batch.attempt == 1
        assert retry_batch.status == BatchStatus.DRAFT

        # Verify members were copied
        retry_members = factory.execution.get_batch_members(retry_batch.batch_id)
        assert len(retry_members) == 3

        # === PHASE 4: Verify final state ===

        # Original batch is failed
        original_batch = factory.execution.get_batch(batch.batch_id)
        assert original_batch is not None
        assert original_batch.status == BatchStatus.FAILED

        # Retry batch exists
        all_batches = factory.execution.get_batches(run.run_id, node_id="sum_aggregator")
        assert len(all_batches) == 2  # Original + retry

        # Verify attempt progression
        attempts = sorted([b.attempt for b in all_batches])
        assert attempts == [0, 1]

    def test_recovery_with_multiple_aggregations(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify recovery handles multiple aggregation nodes independently."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"aggregations": ["sum", "count"]},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=_create_test_schema_contract(),
        )

        # Register multiple aggregation nodes
        self._register_nodes_raw(
            db,
            run.run_id,
            extra_nodes=[
                ("count_aggregator", "count_agg", NodeType.AGGREGATION),
            ],
        )

        # Create rows and tokens
        tokens = []
        for i in range(4):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 10},
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch for sum_aggregator (completed successfully)
        sum_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens[:2]):
            factory.execution.add_batch_member(sum_batch.batch_id, token.token_id, ordinal=i)
        factory.execution.update_batch_status(sum_batch.batch_id, BatchStatus.COMPLETED)

        # Create batch for count_aggregator (crashed during execution)
        count_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="count_aggregator",
        )
        for i, token in enumerate(tokens[2:]):
            factory.execution.add_batch_member(count_batch.batch_id, token.token_id, ordinal=i)
        factory.execution.update_batch_status(count_batch.batch_id, BatchStatus.EXECUTING)

        # Checkpoint at last processed token — typed DTO
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[3].token_id,
            node_id="count_aggregator",
            sequence_number=3,
            aggregation_state=AggregationCheckpointState(
                version="4.0",
                nodes={
                    "count_aggregator": AggregationNodeCheckpoint(
                        tokens=tuple(
                            AggregationTokenCheckpoint(
                                token_id=t.token_id,
                                row_id=t.row_id,
                                branch_name=None,
                                fork_group_id=None,
                                join_group_id=None,
                                expand_group_id=None,
                                row_data={},
                                contract_version="test",
                                contract={"mode": "OBSERVED", "locked": True, "version_hash": "test", "fields": []},
                            )
                            for t in tokens[2:]
                        ),
                        batch_id=count_batch.batch_id,
                        elapsed_age_seconds=0.0,
                        count_fire_offset=None,
                        condition_fire_offset=None,
                    ),
                },
            ),
            graph=mock_graph,
        )

        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # Verify recovery
        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True

        # Only count_aggregator batch should be incomplete
        incomplete = factory.execution.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].aggregation_node_id == "count_aggregator"
        assert incomplete[0].status == BatchStatus.EXECUTING

    def test_recovery_preserves_batch_member_order(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify batch member ordinals are preserved through retry."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"test": "order_preservation"},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=_create_test_schema_contract(),
        )

        self._register_nodes_raw(db, run.run_id)

        # Create 5 rows with specific order
        tokens = []
        for i in range(5):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"seq": i, "data": f"item_{i}"},
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch with specific member ordering
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        # Add in reverse order to test ordinal preservation
        for i, token in enumerate(reversed(tokens)):
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Mark as failed for retry
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.FAILED)

        # Checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="sum_aggregator",
            sequence_number=4,
            graph=mock_graph,
        )

        # Retry
        retry_batch = factory.execution.retry_batch(batch.batch_id)

        # Verify member order is preserved
        original_members = factory.execution.get_batch_members(batch.batch_id)
        retry_members = factory.execution.get_batch_members(retry_batch.batch_id)

        assert len(retry_members) == len(original_members)
        for orig, retry in zip(original_members, retry_members, strict=False):
            assert orig.token_id == retry.token_id
            assert orig.ordinal == retry.ordinal

    def test_recovery_cannot_retry_non_failed_batch(self, test_env: dict[str, Any]) -> None:
        """Verify retry_batch only works on failed batches."""
        db = test_env["db"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"test": "retry_validation"},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=_create_test_schema_contract(),
        )

        self._register_nodes_raw(db, run.run_id)

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"id": 0},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=0)

        # Test with draft status
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

        # Test with executing status
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

        # Test with completed status
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.COMPLETED)
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

    def _register_nodes_raw(
        self,
        db: LandscapeDB,
        run_id: str,
        *,
        extra_nodes: list[tuple[str, str, NodeType]] | None = None,
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
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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
                    node_type=NodeType.AGGREGATION,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
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
                            determinism=Determinism.DETERMINISTIC,
                            config_hash="test",
                            config_json="{}",
                            registered_at=now,
                        )
                    )

            conn.commit()

    def test_timeout_preservation_on_resume(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify aggregation timeout window doesn't reset on resume (Bug #6).

        Scenario:
        1. Create aggregation with 60s timeout
        2. Accept rows, simulate 30s elapsed
        3. Create checkpoint (should store elapsed_age_seconds=30.0)
        4. Crash and resume
        5. Verify timeout triggers after 30 more seconds (not 60s)

        This is Bug #6 fix: timeout windows must preserve SLA across resume.
        """
        import time

        from elspeth.core.config import TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.triggers import TriggerEvaluator

        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        factory: RecorderFactory = test_env["factory"]

        # === PHASE 1: Original run with timeout trigger ===

        run = factory.run_lifecycle.begin_run(
            config={"aggregation": {"trigger": {"timeout_seconds": 60}}},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=_create_test_schema_contract(),
        )

        self._register_nodes_raw(db, run.run_id)

        # Create trigger evaluator with 60s timeout
        trigger_config = TriggerConfig(timeout_seconds=60.0)
        evaluator = TriggerEvaluator(trigger_config)

        # Simulate accepting 3 rows over time
        tokens = []
        for i in range(3):
            row_obj = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
            )
            token = factory.data_flow.create_token(row_id=row_obj.row_id)
            tokens.append(token)
            evaluator.record_accept()

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for ordinal, token in enumerate(tokens):
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=ordinal)

        # Verify initial state: 3 rows accepted, no trigger yet
        assert evaluator.batch_count == 3
        assert evaluator.should_trigger() is False  # Only 0 seconds elapsed so far

        # Simulate 30 seconds passing by mocking time.monotonic()
        # Store the original first_accept_time, then adjust it backward
        original_monotonic = time.monotonic()
        elapsed_seconds = 30.0
        evaluator._first_accept_time = original_monotonic - elapsed_seconds

        # Verify elapsed time is now 30s
        assert 29.0 <= evaluator.batch_age_seconds <= 31.0  # Allow small timing variance

        # Should NOT trigger yet (need 60s total)
        assert evaluator.should_trigger() is False

        # Create checkpoint with aggregation state — typed DTO
        contract = create_observed_contract({"id": 0, "value": 0})
        contract_version = contract.version_hash()
        elapsed = evaluator.get_age_seconds()
        sum_agg_node = AggregationNodeCheckpoint(
            tokens=tuple(
                AggregationTokenCheckpoint(
                    token_id=t.token_id,
                    row_id=t.row_id,
                    branch_name=None,
                    fork_group_id=None,
                    join_group_id=None,
                    expand_group_id=None,
                    row_data={},
                    contract_version=contract_version,
                    contract=contract.to_checkpoint_format(),
                )
                for t in tokens
            ),
            batch_id=batch.batch_id,
            elapsed_age_seconds=elapsed,  # Bug #6 fix: store elapsed time
            count_fire_offset=evaluator.get_count_fire_offset(),  # P2-2026-02-01
            condition_fire_offset=evaluator.get_condition_fire_offset(),  # P2-2026-02-01
        )
        agg_state = AggregationCheckpointState(
            version="4.0",
            nodes={"sum_aggregator": sum_agg_node},
        )

        # Verify elapsed time is stored in checkpoint state
        assert 29.0 <= elapsed <= 31.0

        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="sum_aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
            graph=mock_graph,
        )

        # Simulate crash
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # === PHASE 2: Resume and verify timeout preservation ===

        # Create new aggregation executor to simulate resume
        from elspeth.core.config import AggregationSettings
        from elspeth.engine.spans import SpanFactory

        span_factory = SpanFactory()  # No tracer = no-op spans

        # Create aggregation settings with timeout trigger
        agg_settings = {
            NodeID("sum_aggregator"): AggregationSettings(
                name="sum_aggregator",
                plugin="test_aggregation",
                input="source_out",
                on_error="discard",
                trigger=trigger_config,
                output_mode="transform",
                options={},
            )
        }

        executor = AggregationExecutor(
            execution=factory.execution,
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id=run.run_id,
            aggregation_settings=agg_settings,
        )

        # Restore state from checkpoint
        executor.restore_from_checkpoint(agg_state)

        # Verify buffer was restored
        assert executor.get_buffer_count(NodeID("sum_aggregator")) == 3

        # Get the restored evaluator
        node_state = executor._nodes.get(NodeID("sum_aggregator"))
        assert node_state is not None
        restored_evaluator = node_state.trigger

        # Verify batch count was restored
        assert restored_evaluator.batch_count == 3

        # Verify timeout age was restored (Bug #6 fix)
        # The restored evaluator should think 30s have already elapsed
        restored_age = restored_evaluator.batch_age_seconds
        assert 29.0 <= restored_age <= 31.0, f"Expected ~30s, got {restored_age}s"

        # Should NOT trigger immediately (need 30 more seconds)
        assert restored_evaluator.should_trigger() is False

        # Simulate 30 more seconds passing (total 60s)
        restored_evaluator._first_accept_time = time.monotonic() - 60.0

        # NOW it should trigger (60s total elapsed)
        assert restored_evaluator.should_trigger() is True
        assert restored_evaluator.which_triggered() == "timeout"

        # Verify it triggers at ~60s, not at ~90s (which would be 30s stored + 60s new timeout)
        # If Bug #6 wasn't fixed, timeout would reset and need another 60s (90s total)
        final_age = restored_evaluator.batch_age_seconds
        assert 59.0 <= final_age <= 61.0, f"Timeout should trigger at ~60s, got {final_age}s"
