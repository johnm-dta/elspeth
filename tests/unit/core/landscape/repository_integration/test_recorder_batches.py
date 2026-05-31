"""Tests for RecorderFactory batch operations."""

from __future__ import annotations

from elspeth.contracts import BatchStatus, Determinism, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryBatches:
    """Batch management for aggregation."""

    def test_create_batch(self, landscape_db: LandscapeDB) -> None:
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        agg_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        assert batch.batch_id is not None
        assert batch.status == "draft"
        assert batch.attempt == 0

    def test_add_batch_member(self, landscape_db: LandscapeDB) -> None:
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        agg_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        member = factory.execution.add_batch_member(
            batch_id=batch.batch_id,
            token_id=token.token_id,
            ordinal=0,
        )

        assert member.batch_id == batch.batch_id
        assert member.token_id == token.token_id

        # Verify we can retrieve members
        members = factory.execution.get_batch_members(batch.batch_id)
        assert len(members) == 1
        assert members[0].token_id == token.token_id

    def test_complete_batch(self, landscape_db: LandscapeDB) -> None:
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        agg_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        completed = factory.execution.complete_batch(
            batch_id=batch.batch_id,
            status=BatchStatus.COMPLETED,
            trigger_reason="count=10",
        )

        assert completed.status == "completed"
        assert completed.trigger_reason == "count=10"
        assert completed.completed_at is not None

    def test_batch_lifecycle(self, landscape_db: LandscapeDB) -> None:
        """Test full batch lifecycle: draft -> executing -> completed."""
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        agg_node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create batch in draft
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )
        assert batch.status == "draft"

        # Add members
        for i in range(3):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data={"idx": i},
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            factory.execution.add_batch_member(
                batch_id=batch.batch_id,
                token_id=token.token_id,
                ordinal=i,
            )

        # Move to executing
        factory.execution.update_batch_status(
            batch_id=batch.batch_id,
            status=BatchStatus.EXECUTING,
        )
        executing = factory.execution.get_batch(batch.batch_id)
        assert executing is not None
        assert executing.status == "executing"

        # Complete with trigger_reason
        factory.execution.update_batch_status(
            batch_id=batch.batch_id,
            status=BatchStatus.COMPLETED,
            trigger_reason="count=3",
        )
        completed = factory.execution.get_batch(batch.batch_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.trigger_reason == "count=3"
        assert completed.completed_at is not None

    def test_get_batches_by_status(self, landscape_db: LandscapeDB) -> None:
        """For crash recovery - find incomplete batches."""
        factory = RecorderFactory(landscape_db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        agg = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch1 = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        batch2 = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        factory.execution.update_batch_status(batch2.batch_id, BatchStatus.COMPLETED)

        # Get only draft batches
        drafts = factory.execution.get_batches(run.run_id, status=BatchStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].batch_id == batch1.batch_id


class TestBatchRecoveryQueries:
    """Tests for batch recovery query methods."""

    def test_get_incomplete_batches_returns_draft_and_executing(self, landscape_db: LandscapeDB) -> None:
        """get_incomplete_batches() finds batches needing recovery."""
        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        # Register a node so batches can reference it
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create batches in various states
        draft_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        executing_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        factory.execution.update_batch_status(executing_batch.batch_id, BatchStatus.EXECUTING)

        completed_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        factory.execution.update_batch_status(completed_batch.batch_id, BatchStatus.EXECUTING)
        factory.execution.update_batch_status(completed_batch.batch_id, BatchStatus.COMPLETED, trigger_reason="count")

        # Act
        incomplete = factory.execution.get_incomplete_batches(run.run_id)

        # Assert: Only draft and executing returned
        batch_ids = {b.batch_id for b in incomplete}
        assert draft_batch.batch_id in batch_ids
        assert executing_batch.batch_id in batch_ids
        assert completed_batch.batch_id not in batch_ids

    def test_get_incomplete_batches_includes_failed_for_retry(self, landscape_db: LandscapeDB) -> None:
        """Failed batches are returned for potential retry."""
        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        failed_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        factory.execution.update_batch_status(failed_batch.batch_id, BatchStatus.EXECUTING)
        factory.execution.update_batch_status(failed_batch.batch_id, BatchStatus.FAILED)

        incomplete = factory.execution.get_incomplete_batches(run.run_id)

        batch_ids = {b.batch_id for b in incomplete}
        assert failed_batch.batch_id in batch_ids

    def test_get_incomplete_batches_ordered_by_created_at(self, landscape_db: LandscapeDB) -> None:
        """Batches returned in creation order for deterministic recovery."""
        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch1 = factory.execution.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")
        batch2 = factory.execution.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")
        batch3 = factory.execution.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")

        incomplete = factory.execution.get_incomplete_batches(run.run_id)

        assert len(incomplete) == 3
        assert incomplete[0].batch_id == batch1.batch_id
        assert incomplete[1].batch_id == batch2.batch_id
        assert incomplete[2].batch_id == batch3.batch_id


class TestBatchRetry:
    """Tests for batch retry functionality."""

    def test_retry_batch_increments_attempt_and_resets_status(self, landscape_db: LandscapeDB) -> None:
        """retry_batch() creates new attempt with draft status."""
        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create and fail a batch
        original = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        factory.execution.update_batch_status(original.batch_id, BatchStatus.EXECUTING)
        factory.execution.update_batch_status(original.batch_id, BatchStatus.FAILED)

        # Act: Retry the batch
        retried = factory.execution.retry_batch(original.batch_id)

        # Assert: New batch with incremented attempt
        assert retried.batch_id != original.batch_id  # New batch ID
        assert retried.attempt == original.attempt + 1
        assert retried.status == BatchStatus.DRAFT
        assert retried.aggregation_node_id == original.aggregation_node_id

    def test_retry_batch_preserves_members(self, landscape_db: LandscapeDB) -> None:
        """retry_batch() copies batch members to new batch."""
        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        original = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )

        # Create tokens for members
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"id": 1},
        )
        token1 = factory.data_flow.create_token(row_id=row.row_id)
        token2 = factory.data_flow.create_token(row_id=row.row_id)

        # Add members to original
        factory.execution.add_batch_member(original.batch_id, token1.token_id, ordinal=0)
        factory.execution.add_batch_member(original.batch_id, token2.token_id, ordinal=1)
        factory.execution.update_batch_status(original.batch_id, BatchStatus.EXECUTING)
        factory.execution.update_batch_status(original.batch_id, BatchStatus.FAILED)

        # Act
        retried = factory.execution.retry_batch(original.batch_id)

        # Assert: Members copied
        members = factory.execution.get_batch_members(retried.batch_id)
        assert len(members) == 2
        assert members[0].token_id == token1.token_id
        assert members[1].token_id == token2.token_id

    def test_retry_batch_raises_for_non_failed_batch(self, landscape_db: LandscapeDB) -> None:
        """Can only retry failed batches."""
        import pytest

        factory = RecorderFactory(landscape_db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        # Batch is in draft status

        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

    def test_retry_batch_raises_for_nonexistent_batch(self, landscape_db: LandscapeDB) -> None:
        """Raises for nonexistent batch ID."""
        import pytest

        factory = RecorderFactory(landscape_db)

        with pytest.raises(AuditIntegrityError, match="batch nonexistent-batch-id not found"):
            factory.execution.retry_batch("nonexistent-batch-id")
