"""Tests for LandscapeRecorder batch operations."""

from __future__ import annotations

from elspeth.contracts import BatchStatus, Determinism, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestLandscapeRecorderBatches:
    """Batch management for aggregation."""

    def test_create_batch(self, landscape_db: LandscapeDB) -> None:
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        assert batch.batch_id is not None
        assert batch.status == "draft"
        assert batch.attempt == 0

    def test_add_batch_member(self, landscape_db: LandscapeDB) -> None:
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        member = recorder.add_batch_member(
            batch_id=batch.batch_id,
            token_id=token.token_id,
            ordinal=0,
        )

        assert member.batch_id == batch.batch_id
        assert member.token_id == token.token_id

        # Verify we can retrieve members
        members = recorder.get_batch_members(batch.batch_id)
        assert len(members) == 1
        assert members[0].token_id == token.token_id

    def test_complete_batch(self, landscape_db: LandscapeDB) -> None:
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        completed = recorder.complete_batch(
            batch_id=batch.batch_id,
            status=BatchStatus.COMPLETED,
            trigger_reason="count=10",
        )

        assert completed.status == "completed"
        assert completed.trigger_reason == "count=10"
        assert completed.completed_at is not None

    def test_batch_lifecycle(self, landscape_db: LandscapeDB) -> None:
        """Test full batch lifecycle: draft -> executing -> completed."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create batch in draft
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )
        assert batch.status == "draft"

        # Add members
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data={"idx": i},
            )
            token = recorder.create_token(row_id=row.row_id)
            recorder.add_batch_member(
                batch_id=batch.batch_id,
                token_id=token.token_id,
                ordinal=i,
            )

        # Move to executing
        recorder.update_batch_status(
            batch_id=batch.batch_id,
            status=BatchStatus.EXECUTING,
        )
        executing = recorder.get_batch(batch.batch_id)
        assert executing is not None
        assert executing.status == "executing"

        # Complete with trigger_reason
        recorder.update_batch_status(
            batch_id=batch.batch_id,
            status=BatchStatus.COMPLETED,
            trigger_reason="count=3",
        )
        completed = recorder.get_batch(batch.batch_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.trigger_reason == "count=3"
        assert completed.completed_at is not None

    def test_get_batches_by_status(self, landscape_db: LandscapeDB) -> None:
        """For crash recovery - find incomplete batches."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch1 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        batch2 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.update_batch_status(batch2.batch_id, BatchStatus.COMPLETED)

        # Get only draft batches
        drafts = recorder.get_batches(run.run_id, status=BatchStatus.DRAFT)
        assert len(drafts) == 1
        assert drafts[0].batch_id == batch1.batch_id


class TestBatchRecoveryQueries:
    """Tests for batch recovery query methods."""

    def test_get_incomplete_batches_returns_draft_and_executing(self, landscape_db: LandscapeDB) -> None:
        """get_incomplete_batches() finds batches needing recovery."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        # Register a node so batches can reference it
        recorder.register_node(
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
        draft_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        executing_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(executing_batch.batch_id, BatchStatus.EXECUTING)

        completed_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(completed_batch.batch_id, BatchStatus.EXECUTING)
        recorder.update_batch_status(completed_batch.batch_id, BatchStatus.COMPLETED, trigger_reason="count")

        # Act
        incomplete = recorder.get_incomplete_batches(run.run_id)

        # Assert: Only draft and executing returned
        batch_ids = {b.batch_id for b in incomplete}
        assert draft_batch.batch_id in batch_ids
        assert executing_batch.batch_id in batch_ids
        assert completed_batch.batch_id not in batch_ids

    def test_get_incomplete_batches_includes_failed_for_retry(self, landscape_db: LandscapeDB) -> None:
        """Failed batches are returned for potential retry."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        failed_batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(failed_batch.batch_id, BatchStatus.EXECUTING)
        recorder.update_batch_status(failed_batch.batch_id, BatchStatus.FAILED)

        incomplete = recorder.get_incomplete_batches(run.run_id)

        batch_ids = {b.batch_id for b in incomplete}
        assert failed_batch.batch_id in batch_ids

    def test_get_incomplete_batches_ordered_by_created_at(self, landscape_db: LandscapeDB) -> None:
        """Batches returned in creation order for deterministic recovery."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch1 = recorder.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")
        batch2 = recorder.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")
        batch3 = recorder.create_batch(run_id=run.run_id, aggregation_node_id="agg_node")

        incomplete = recorder.get_incomplete_batches(run.run_id)

        assert len(incomplete) == 3
        assert incomplete[0].batch_id == batch1.batch_id
        assert incomplete[1].batch_id == batch2.batch_id
        assert incomplete[2].batch_id == batch3.batch_id


class TestBatchRetry:
    """Tests for batch retry functionality."""

    def test_retry_batch_increments_attempt_and_resets_status(self, landscape_db: LandscapeDB) -> None:
        """retry_batch() creates new attempt with draft status."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
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
        original = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        recorder.update_batch_status(original.batch_id, BatchStatus.EXECUTING)
        recorder.update_batch_status(original.batch_id, BatchStatus.FAILED)

        # Act: Retry the batch
        retried = recorder.retry_batch(original.batch_id)

        # Assert: New batch with incremented attempt
        assert retried.batch_id != original.batch_id  # New batch ID
        assert retried.attempt == original.attempt + 1
        assert retried.status == BatchStatus.DRAFT
        assert retried.aggregation_node_id == original.aggregation_node_id

    def test_retry_batch_preserves_members(self, landscape_db: LandscapeDB) -> None:
        """retry_batch() copies batch members to new batch."""
        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        original = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )

        # Create tokens for members
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"id": 1},
        )
        token1 = recorder.create_token(row_id=row.row_id)
        token2 = recorder.create_token(row_id=row.row_id)

        # Add members to original
        recorder.add_batch_member(original.batch_id, token1.token_id, ordinal=0)
        recorder.add_batch_member(original.batch_id, token2.token_id, ordinal=1)
        recorder.update_batch_status(original.batch_id, BatchStatus.EXECUTING)
        recorder.update_batch_status(original.batch_id, BatchStatus.FAILED)

        # Act
        retried = recorder.retry_batch(original.batch_id)

        # Assert: Members copied
        members = recorder.get_batch_members(retried.batch_id)
        assert len(members) == 2
        assert members[0].token_id == token1.token_id
        assert members[1].token_id == token2.token_id

    def test_retry_batch_raises_for_non_failed_batch(self, landscape_db: LandscapeDB) -> None:
        """Can only retry failed batches."""
        import pytest

        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )
        # Batch is in draft status

        with pytest.raises(ValueError, match="Can only retry failed batches"):
            recorder.retry_batch(batch.batch_id)

    def test_retry_batch_raises_for_nonexistent_batch(self, landscape_db: LandscapeDB) -> None:
        """Raises for nonexistent batch ID."""
        import pytest

        from elspeth.core.landscape.recorder import LandscapeRecorder

        recorder = LandscapeRecorder(landscape_db)

        with pytest.raises(ValueError, match="Batch not found"):
            recorder.retry_batch("nonexistent-batch-id")
