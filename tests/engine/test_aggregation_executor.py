"""Tests for aggregation executor."""

import uuid
from typing import Any

import pytest

from elspeth.contracts.enums import Determinism, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NodeID
from tests.conftest import as_transform

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


def unique_id(prefix: str = "") -> str:
    """Generate unique ID for test data to avoid collisions when sharing database."""
    return f"{prefix}{uuid.uuid4().hex[:8]}"


class TestAggregationExecutorOldInterfaceDeleted:
    """Verify old accept()/flush() executor interface is deleted.

    OLD: TestAggregationExecutor tested executor.accept() and executor.flush()
         with plugin-level aggregation interface.
    NEW: Aggregation is engine-controlled via buffer_row()/execute_flush()
         with batch-aware transforms (is_batch_aware=True).
         See TestAggregationExecutorBuffering for new tests.
    """

    def test_old_accept_method_deleted(self, real_landscape_db) -> None:
        """Old accept() method should be deleted from AggregationExecutor."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Old accept() method should be deleted
        assert not hasattr(executor, "accept"), "accept() method should be deleted - use buffer_row() instead"

    def test_old_flush_method_deleted(self, real_landscape_db) -> None:
        """Old flush() method should be deleted from AggregationExecutor."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Old flush() method should be deleted (execute_flush() is the production method)
        # Note: flush() was the old method that called plugin.flush()
        # execute_flush() is the production method with full audit recording
        # _get_buffered_data() is internal-only for testing
        assert hasattr(executor, "execute_flush"), "execute_flush() should exist for production flush with audit"
        assert hasattr(executor, "_get_buffered_data"), "_get_buffered_data() should exist for testing buffer contents"


class TestAggregationExecutorTriggersDeleted:
    """Verify BaseAggregation-based trigger tests are deleted.

    OLD: TestAggregationExecutorTriggers tested trigger evaluation with
         BaseAggregation plugins (accept/flush interface).
    NEW: Trigger evaluation still exists but operates on engine buffers,
         not plugin state. See TestAggregationExecutorBuffering.
    """

    def test_base_aggregation_deleted(self) -> None:
        """BaseAggregation should be deleted (aggregation is structural)."""
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"


class TestAggregationExecutorRestore:
    """Tests for aggregation state restoration."""

    def test_restore_state_sets_internal_state(self, real_landscape_db) -> None:
        """restore_state() stores state for plugin access."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        state = {"buffer": [1, 2, 3], "sum": 6, "count": 3}

        executor.restore_state(NodeID("agg_node"), state)

        assert executor.get_restored_state(NodeID("agg_node")) == state

    def test_restore_state_returns_none_for_unknown_node(self, real_landscape_db) -> None:
        """get_restored_state() returns None for nodes without restored state."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        assert executor.get_restored_state(NodeID("unknown_node")) is None

    def test_restore_batch_sets_current_batch(self, real_landscape_db) -> None:
        """restore_batch() makes batch the current batch for its node."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register aggregation node
        recorder.register_node(
            run_id=run.run_id,
            node_id=NodeID("agg_node"),
            plugin_name="test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a batch
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=NodeID("agg_node"),
        )

        # Create executor for this run
        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Act
        executor.restore_batch(batch.batch_id)

        # Assert
        assert executor.get_batch_id(NodeID("agg_node")) == batch.batch_id

    def test_restore_batch_not_found_raises_error(self, real_landscape_db) -> None:
        """restore_batch() raises ValueError for unknown batch_id."""
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        with pytest.raises(ValueError, match="Batch not found"):
            executor.restore_batch("nonexistent-batch-id")

    def test_restore_batch_restores_member_count_deleted(self, real_landscape_db) -> None:
        """Test deleted - used old accept() interface.

        OLD: Tested that restoring a batch lets you call accept() with correct ordinals.
        NEW: Restore functionality now uses buffer_row() interface instead.
             See TestAggregationExecutorCheckpoint for new restore tests.
        """
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Verify old accept() method is deleted
        assert not hasattr(executor, "accept"), "accept() method should be deleted - use buffer_row() instead"


class TestAggregationExecutorBuffering:
    """Tests for engine-level row buffering in AggregationExecutor."""

    def test_executor_buffers_rows_internally(self, real_landscape_db) -> None:
        """Executor buffers rows without calling plugin.accept()."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="buffer_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Buffer 3 rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        # Check buffer
        buffered = executor.get_buffered_rows(node_id)
        assert len(buffered) == 3
        assert [r["value"] for r in buffered] == [0, 1, 2]

    def test_get_buffered_data_does_not_clear_buffer(self, real_landscape_db) -> None:
        """_get_buffered_data() returns data without clearing (internal method)."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="buffer_flush_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Buffer rows - use unique IDs to avoid collisions
        test_prefix = unique_id("buf_no_clear_")
        for i in range(2):
            token = TokenInfo(
                row_id=f"{test_prefix}row-{i}",
                token_id=f"{test_prefix}token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        # _get_buffered_data() returns data WITHOUT clearing (internal method)
        buffered_rows, buffered_tokens = executor._get_buffered_data(node_id)
        assert len(buffered_rows) == 2
        assert len(buffered_tokens) == 2

        # Buffer should still contain data (not cleared by _get_buffered_data)
        assert executor.get_buffered_rows(node_id) == buffered_rows
        assert executor.get_buffered_tokens(node_id) == buffered_tokens

    def test_buffered_tokens_are_tracked(self, real_landscape_db) -> None:
        """Executor tracks TokenInfo objects alongside buffered rows."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="token_track_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Buffer 2 rows - use unique IDs to avoid collisions
        test_prefix = unique_id("tok_track_")
        for i in range(2):
            token = TokenInfo(
                row_id=f"{test_prefix}row-{i}",
                token_id=f"{test_prefix}token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        # Check buffered tokens
        tokens = executor.get_buffered_tokens(node_id)
        assert len(tokens) == 2
        assert tokens[0].token_id == f"{test_prefix}token-0"
        assert tokens[1].token_id == f"{test_prefix}token-1"

    def test_buffer_creates_batch_on_first_row(self, real_landscape_db) -> None:
        """buffer_row() creates a batch on first row just like accept()."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_create_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=5),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # No batch yet
        assert executor.get_batch_id(node_id) is None

        # Buffer first row - use unique IDs to avoid collisions
        test_prefix = unique_id("batch_create_")
        token = TokenInfo(
            row_id=f"{test_prefix}row-0",
            token_id=f"{test_prefix}token-0",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)
        executor.buffer_row(node_id, token)

        # Batch should now exist
        batch_id = executor.get_batch_id(node_id)
        assert batch_id is not None

        # Batch should be in landscape
        batch = recorder.get_batch(batch_id)
        assert batch is not None
        assert batch.aggregation_node_id == agg_node.node_id

    def test_buffer_updates_trigger_evaluator(self, real_landscape_db) -> None:
        """buffer_row() updates trigger evaluator count."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="trigger_update_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),  # Trigger at 3
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Buffer 2 rows - should not trigger - use unique IDs to avoid collisions
        test_prefix = unique_id("trig_upd_")
        for i in range(2):
            token = TokenInfo(
                row_id=f"{test_prefix}row-{i}",
                token_id=f"{test_prefix}token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        assert executor.should_flush(node_id) is False

        # Buffer 3rd row - should trigger
        token = TokenInfo(
            row_id=f"{test_prefix}row-2",
            token_id=f"{test_prefix}token-2",
            row_data={"value": 2},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=2,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)
        executor.buffer_row(node_id, token)

        assert executor.should_flush(node_id) is True

    def test_get_buffered_data_returns_both_rows_and_tokens(self, real_landscape_db) -> None:
        """_get_buffered_data() returns tuple of (rows, tokens) for passthrough mode."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="flush_returns_tokens_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Buffer 2 rows with distinct tokens - use unique IDs to avoid collisions
        test_prefix = unique_id("buf_data_")
        token1 = TokenInfo(
            row_id=f"{test_prefix}row-1",
            token_id=f"{test_prefix}token-1",
            row_data={"x": 1},
        )
        token2 = TokenInfo(
            row_id=f"{test_prefix}row-2",
            token_id=f"{test_prefix}token-2",
            row_data={"x": 2},
        )

        for i, token in enumerate([token1, token2]):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        # _get_buffered_data() returns tuple of (rows, tokens) without clearing
        rows, tokens = executor._get_buffered_data(node_id)

        # Verify rows
        assert len(rows) == 2
        assert rows[0] == {"x": 1}
        assert rows[1] == {"x": 2}

        # Verify tokens
        assert len(tokens) == 2
        assert tokens[0].token_id == f"{test_prefix}token-1"
        assert tokens[1].token_id == f"{test_prefix}token-2"

        # Buffer should NOT be cleared (_get_buffered_data is internal, doesn't clear)
        assert executor.get_buffered_rows(node_id) == rows
        assert executor.get_buffered_tokens(node_id) == tokens


class TestAggregationExecutorCheckpoint:
    """Tests for buffer serialization/deserialization for crash recovery."""

    def test_get_checkpoint_state_stores_full_token_info(self, real_landscape_db) -> None:
        """Checkpoint stores complete TokenInfo (not just token_ids) for restoration."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="checkpoint_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),  # High count so we don't trigger
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Add tokens with all metadata - use unique IDs to avoid collisions
        test_prefix = unique_id("ckpt_store_")
        token1 = TokenInfo(
            row_id=f"{test_prefix}row-1",
            token_id=f"{test_prefix}token-101",
            row_data={"name": "Alice", "score": 95},
            branch_name="high_score",
        )
        token2 = TokenInfo(
            row_id=f"{test_prefix}row-2",
            token_id=f"{test_prefix}token-102",
            row_data={"name": "Bob", "score": 42},
            branch_name=None,  # Optional field
        )

        # Create rows in landscape
        for i, token in enumerate([token1, token2]):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(node_id, token)

        # Get checkpoint
        checkpoint = executor.get_checkpoint_state()

        # VERIFY: Full TokenInfo stored, not just IDs
        assert "tokens" in checkpoint[node_id], "Should have 'tokens' key, not 'token_ids'"
        assert "token_ids" not in checkpoint[node_id], "Old 'token_ids' format should be gone"

        tokens_data = checkpoint[node_id]["tokens"]
        assert len(tokens_data) == 2

        # Verify first token has ALL fields
        assert tokens_data[0]["token_id"] == f"{test_prefix}token-101"
        assert tokens_data[0]["row_id"] == f"{test_prefix}row-1"
        assert tokens_data[0]["row_data"] == {"name": "Alice", "score": 95}
        assert tokens_data[0]["branch_name"] == "high_score"

        # Verify second token (with None branch_name)
        assert tokens_data[1]["token_id"] == f"{test_prefix}token-102"
        assert tokens_data[1]["row_id"] == f"{test_prefix}row-2"
        assert tokens_data[1]["row_data"] == {"name": "Bob", "score": 42}
        assert tokens_data[1]["branch_name"] is None

        # Verify batch_id present in checkpoint
        assert "batch_id" in checkpoint[node_id]

        # Verify checkpoint is JSON serializable (required for persistence)
        import json

        serialized = json.dumps(checkpoint)
        deserialized = json.loads(serialized)
        assert deserialized == checkpoint, "Checkpoint must round-trip through JSON"

    def test_get_checkpoint_state_excludes_empty_buffers(self, real_landscape_db) -> None:
        """get_checkpoint_state() only includes non-empty buffers."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="empty_buffer_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Don't buffer anything
        state = executor.get_checkpoint_state()
        assert state == {"_version": "1.1"}  # Only version field when no buffers

    def test_restore_from_checkpoint_reconstructs_full_token_info(self, real_landscape_db) -> None:
        """Restore reconstructs complete TokenInfo objects from checkpoint data."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="restore_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Checkpoint state with new format (full TokenInfo data)
        checkpoint_state = {
            "_version": "1.1",
            node_id: {
                "tokens": [
                    {
                        "token_id": "token-101",
                        "row_id": "row-1",
                        "row_data": {"name": "Alice", "score": 95},
                        "branch_name": "high_score",
                    },
                    {
                        "token_id": "token-102",
                        "row_id": "row-2",
                        "row_data": {"name": "Bob", "score": 42},
                        "branch_name": None,  # Optional field
                    },
                ],
                "batch_id": "batch-123",
                "elapsed_age_seconds": 0.0,  # Required in v1.1 format
                "count_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
                "condition_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
            },
        }

        # Restore from checkpoint
        executor.restore_from_checkpoint(checkpoint_state)

        # VERIFY: Full TokenInfo objects reconstructed
        assert len(executor._buffer_tokens[node_id]) == 2

        token1 = executor._buffer_tokens[node_id][0]
        assert isinstance(token1, TokenInfo)
        assert token1.token_id == "token-101"
        assert token1.row_id == "row-1"
        assert token1.row_data == {"name": "Alice", "score": 95}
        assert token1.branch_name == "high_score"

        token2 = executor._buffer_tokens[node_id][1]
        assert isinstance(token2, TokenInfo)
        assert token2.token_id == "token-102"
        assert token2.row_id == "row-2"
        assert token2.row_data == {"name": "Bob", "score": 42}
        assert token2.branch_name is None

        # VERIFY: _buffers also populated with row_data
        assert len(executor._buffers[node_id]) == 2
        assert executor._buffers[node_id][0] == {"name": "Alice", "score": 95}
        assert executor._buffers[node_id][1] == {"name": "Bob", "score": 42}

        # VERIFY: batch_id restored
        assert executor.get_batch_id(node_id) == "batch-123"

    def test_restore_from_checkpoint_restores_trigger_count(self, real_landscape_db) -> None:
        """restore_from_checkpoint() restores trigger evaluator count."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="trigger_restore_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=5),  # Trigger at 5
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Simulate checkpoint state with 4 rows buffered (new format)
        checkpoint_state = {
            "_version": "1.1",
            node_id: {
                "tokens": [
                    {
                        "token_id": f"token-{i}",
                        "row_id": f"row-{i}",
                        "row_data": {"value": i},
                        "branch_name": None,
                    }
                    for i in range(4)
                ],
                "batch_id": "batch-123",
                "elapsed_age_seconds": 0.0,  # Required in v1.1 format
                "count_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
                "condition_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
            },
        }

        executor.restore_from_checkpoint(checkpoint_state)

        # Trigger evaluator should reflect restored count (4 rows)
        # Should NOT trigger yet (need 5)
        assert executor.should_flush(node_id) is False

        # Trigger evaluator internal count should be 4
        evaluator = executor._trigger_evaluators[node_id]
        assert evaluator.batch_count == 4

    def test_checkpoint_roundtrip(self, real_landscape_db) -> None:
        """Buffer state survives checkpoint/restore cycle."""
        import json

        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="roundtrip_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        # First executor - buffer some rows
        executor1 = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Use unique IDs to avoid collisions
        test_prefix = unique_id("ckpt_rt_")
        for i in range(3):
            token = TokenInfo(
                row_id=f"{test_prefix}row-{i}",
                token_id=f"{test_prefix}token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor1.buffer_row(node_id, token)

        # Get checkpoint state and serialize (simulates crash)
        state = executor1.get_checkpoint_state()
        serialized = json.dumps(state)

        # Second executor - restore from checkpoint
        executor2 = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        restored_state = json.loads(serialized)
        executor2.restore_from_checkpoint(restored_state)

        # Verify buffer restored correctly
        buffered = executor2.get_buffered_rows(node_id)
        assert buffered == [{"value": 0}, {"value": 10}, {"value": 20}]

        # Verify trigger count restored
        evaluator = executor2._trigger_evaluators[node_id]
        assert evaluator.batch_count == 3

    def test_checkpoint_restore_then_flush_succeeds(self, real_landscape_db) -> None:
        """After restoring from checkpoint, aggregation can flush successfully.

        This is the critical test for P1-2026-01-21: the original bug was that
        restored aggregations would crash on flush with IndexError because
        _buffer_tokens was empty while _buffers had rows.
        """

        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TriggerType
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="flush_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),  # Trigger at 2 rows
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Create batch in landscape (must exist before flush)
        batch = recorder.create_batch(run_id=run.run_id, aggregation_node_id=agg_node.node_id)

        # Use unique IDs to avoid collisions
        test_prefix = unique_id("ckpt_flush_")

        # Simulate checkpoint with 2 buffered tokens (ready to flush)
        checkpoint_state: dict[str, Any] = {
            "_version": "1.1",
            node_id: {
                "tokens": [
                    {
                        "token_id": f"{test_prefix}token-101",
                        "row_id": f"{test_prefix}row-1",
                        "row_data": {"value": 10},
                        "branch_name": None,
                    },
                    {
                        "token_id": f"{test_prefix}token-102",
                        "row_id": f"{test_prefix}row-2",
                        "row_data": {"value": 20},
                        "branch_name": None,
                    },
                ],
                "batch_id": batch.batch_id,
                "elapsed_age_seconds": 0.0,  # Required in v1.1 format
                "count_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
                "condition_fire_offset": None,  # P2-2026-02-01: Required in v1.1 format
            },
        }

        # Create rows and tokens in landscape for the checkpoint
        for i, token_data in enumerate(checkpoint_state[node_id]["tokens"]):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node_id,
                row_index=i,
                data=token_data["row_data"],
                row_id=token_data["row_id"],
            )
            recorder.create_token(row_id=row.row_id, token_id=token_data["token_id"])

        # Restore from checkpoint
        executor.restore_from_checkpoint(checkpoint_state)

        # CRITICAL: Flush should succeed (this is what failed in the original bug)
        # The bug was: _buffers had 2 rows, but _buffer_tokens was empty
        # So flush tried to access tokens[0] and tokens[1] but list was empty = IndexError

        # Mock the batch-aware transform
        class MockBatchTransform:
            name = "batch_sum"
            mock_node_id = node_id

            def process(self, rows: list[dict[str, Any]], ctx: PluginContext) -> TransformResult:
                total = sum(r["value"] for r in rows)
                return TransformResult.success({"sum": total}, success_reason={"action": "sum_batch"})

        transform = MockBatchTransform()
        ctx = PluginContext(run_id=run.run_id, config={})

        # Execute flush - should NOT crash with IndexError
        result, consumed_tokens, _batch_id = executor.execute_flush(
            node_id=node_id,
            transform=as_transform(transform),
            ctx=ctx,
            step_in_pipeline=1,
            trigger_type=TriggerType.COUNT,
        )

        # VERIFY: Flush succeeded
        assert result is not None, "Flush should return a result"
        assert result.status == "success", "Flush should succeed"
        assert result.row == {"sum": 30}, "Transform should have computed sum"

        # VERIFY: Consumed tokens match the buffered tokens
        assert len(consumed_tokens) == 2
        assert consumed_tokens[0].token_id == f"{test_prefix}token-101"
        assert consumed_tokens[1].token_id == f"{test_prefix}token-102"

    def test_execute_flush_detects_incomplete_restoration(self, real_landscape_db) -> None:
        """Defensive guard catches buffer/token length mismatch with clear error."""

        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TriggerType
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mismatch_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Initialize batch_id (required for flush)
        executor._batch_ids[node_id] = "batch-123"

        # SIMULATE THE ORIGINAL BUG: buffer has rows but tokens is empty
        # (This should never happen after Tasks 1 & 2, but the guard should catch it)
        executor._buffers[node_id] = [{"value": 10}, {"value": 20}]
        executor._buffer_tokens[node_id] = []  # EMPTY - the bug state!

        # Mock transform
        class MockTransform:
            name = "test"
            mock_node_id = node_id

        transform = MockTransform()
        ctx = PluginContext(run_id=run.run_id, config={})

        # VERIFY: Flush crashes with clear RuntimeError (not IndexError)
        with pytest.raises(RuntimeError, match="Internal state corruption"):
            executor.execute_flush(
                node_id=node_id,
                transform=as_transform(transform),
                ctx=ctx,
                step_in_pipeline=1,
                trigger_type=TriggerType.COUNT,
            )

    def test_checkpoint_size_warning_at_1mb_threshold(self, real_landscape_db) -> None:
        """Checkpoint size validation logs warning when exceeding 1MB."""
        from unittest.mock import patch

        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="size_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10000),  # High trigger to prevent flush
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )
        executor._batch_ids[node_id] = "batch-123"

        # Create large row_data to exceed 1MB when serialized
        # A single row with ~1KB of data, repeated 1500 times = ~1.5MB checkpoint
        large_row_data = {"data": "x" * 1000, "index": 0}

        # Add 1500 tokens with large row_data
        from elspeth.engine.executors import TokenInfo

        tokens = []
        for i in range(1500):
            row_data = large_row_data.copy()
            row_data["index"] = i
            tokens.append(
                TokenInfo(
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    branch_name=None,
                )
            )

        executor._buffer_tokens[node_id] = tokens

        # Capture log output
        import logging

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value

            # Get checkpoint (should trigger warning)
            _ = executor.get_checkpoint_state()

            # VERIFY: Warning was logged
            assert mock_logger.warning.called, "Should log warning for large checkpoint"

            # VERIFY: Warning message contains size info
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Large checkpoint" in warning_call
            assert "MB" in warning_call
            assert "buffered rows" in warning_call

    def test_checkpoint_size_error_at_10mb_limit(self, real_landscape_db) -> None:
        """Checkpoint size validation raises RuntimeError when exceeding 10MB."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TokenInfo
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="limit_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=20000),  # High trigger to prevent flush
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )
        executor._batch_ids[node_id] = "batch-123"

        # Create very large row_data to exceed 10MB when serialized
        # A single row with ~2KB of data, repeated 6000 times = ~12MB checkpoint
        very_large_row_data = {"data": "x" * 2000, "index": 0}

        # Add 6000 tokens with very large row_data
        tokens = []
        for i in range(6000):
            row_data = very_large_row_data.copy()
            row_data["index"] = i
            tokens.append(
                TokenInfo(
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    branch_name=None,
                )
            )

        executor._buffer_tokens[node_id] = tokens

        # VERIFY: Getting checkpoint raises RuntimeError
        with pytest.raises(RuntimeError, match=r"Checkpoint size.*exceeds 10MB limit"):
            executor.get_checkpoint_state()

    def test_checkpoint_size_error_message_includes_solutions(self, real_landscape_db) -> None:
        """Checkpoint size error message includes actionable solutions."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TokenInfo
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="solution_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=20000),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )
        executor._batch_ids[node_id] = "batch-123"

        # Create checkpoint > 10MB
        very_large_row_data = {"data": "x" * 2000}
        tokens = [
            TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data=very_large_row_data,
                branch_name=None,
            )
            for i in range(6000)
        ]
        executor._buffer_tokens[node_id] = tokens

        # Capture error message
        with pytest.raises(RuntimeError) as exc_info:
            executor.get_checkpoint_state()

        error_message = str(exc_info.value)

        # VERIFY: Error message includes solutions
        assert "Solutions:" in error_message or "Reduce" in error_message
        assert "rows" in error_message  # Mentions row count
        assert "nodes" in error_message  # Mentions node count

    def test_checkpoint_size_no_warning_under_1mb(self, real_landscape_db) -> None:
        """Checkpoint size validation is silent when under 1MB threshold."""
        from unittest.mock import patch

        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TokenInfo
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="under_1mb_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=5000),  # High trigger to prevent flush
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )
        executor._batch_ids[node_id] = "batch-123"

        # Create checkpoint just under 1MB
        # Target: ~750KB (safely under 1MB to account for JSON overhead)
        # 750 rows x 1KB data each = ~750KB
        # Note: checkpoint includes lineage fields (fork/join/expand_group_id)
        medium_row_data = {"data": "x" * 1000, "index": 0}

        tokens = []
        for i in range(750):
            row_data = medium_row_data.copy()
            row_data["index"] = i
            tokens.append(
                TokenInfo(
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    branch_name=None,
                )
            )

        executor._buffer_tokens[node_id] = tokens

        # Capture log output
        import logging

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value

            # Get checkpoint (should NOT trigger warning)
            checkpoint = executor.get_checkpoint_state()

            # VERIFY: No warning was logged
            assert not mock_logger.warning.called, "Should NOT log warning for checkpoint under 1MB"

            # VERIFY: Checkpoint was created successfully
            assert checkpoint is not None
            assert node_id in checkpoint

    def test_checkpoint_size_warning_but_no_error_between_thresholds(self, real_landscape_db) -> None:
        """Checkpoint between 1MB and 10MB logs warning but does not raise error."""
        from unittest.mock import patch

        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor, TokenInfo
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="between_thresholds_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10000),  # High trigger to prevent flush
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )
        executor._batch_ids[node_id] = "batch-123"

        # Create checkpoint ~5MB (between 1MB and 10MB thresholds)
        # 2500 rows x 2KB data each = ~5MB
        large_row_data = {"data": "x" * 2000, "index": 0}

        tokens = []
        for i in range(2500):
            row_data = large_row_data.copy()
            row_data["index"] = i
            tokens.append(
                TokenInfo(
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    branch_name=None,
                )
            )

        executor._buffer_tokens[node_id] = tokens

        # Capture log output
        import logging

        with patch.object(logging, "getLogger") as mock_get_logger:
            mock_logger = mock_get_logger.return_value

            # Get checkpoint (should trigger warning but NOT error)
            checkpoint = executor.get_checkpoint_state()

            # VERIFY: Warning was logged
            assert mock_logger.warning.called, "Should log warning for 5MB checkpoint"

            # VERIFY: No exception raised (checkpoint created successfully)
            assert checkpoint is not None
            assert node_id in checkpoint

    def test_restore_from_checkpoint_handles_empty_state(self, real_landscape_db) -> None:
        """Restore from checkpoint handles empty state gracefully."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="empty_state_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Restore from empty checkpoint (only version, no aggregation buffers)
        executor.restore_from_checkpoint({"_version": "1.1"})

        # VERIFY: No errors, buffers remain empty (but initialized for the node)
        assert node_id in executor._buffers
        assert len(executor._buffers[node_id]) == 0
        assert node_id in executor._buffer_tokens
        assert len(executor._buffer_tokens[node_id]) == 0

    @pytest.mark.parametrize(
        ("test_id", "checkpoint_node_data", "error_pattern"),
        [
            pytest.param(
                "missing_tokens_key",
                {
                    "rows": [{"value": 10}],  # Old format - missing 'tokens' key
                    "token_ids": ["token-1"],  # Old format
                    "batch_id": None,
                },
                "missing 'tokens' key",
                id="missing_tokens_key",
            ),
            pytest.param(
                "invalid_tokens_type",
                {
                    "tokens": "not-a-list",  # Wrong type
                    "batch_id": None,
                },
                "'tokens' must be a list",
                id="invalid_tokens_type",
            ),
            pytest.param(
                "missing_token_fields",
                {
                    "tokens": [
                        {
                            "token_id": 101,
                            "row_id": 1,
                            # "row_data" is MISSING
                            "branch_name": None,
                        }
                    ],
                    "batch_id": None,
                },
                r"missing required fields.*row_data",
                id="missing_token_fields",
            ),
        ],
    )
    def test_restore_from_checkpoint_crashes_on_invalid_checkpoint(
        self,
        real_landscape_db,
        test_id: str,
        checkpoint_node_data: dict,
        error_pattern: str,
    ) -> None:
        """Restore crashes with clear error on invalid checkpoint formats.

        Covers:
        - missing_tokens_key: Old checkpoint format missing 'tokens' key
        - invalid_tokens_type: 'tokens' is not a list
        - missing_token_fields: Token missing required fields like row_data
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup
        recorder = LandscapeRecorder(real_landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name=f"{test_id}_test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(agg_node.node_id)

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={node_id: settings},
        )

        # Build invalid checkpoint with the parameterized node data
        invalid_checkpoint = {
            "_version": "1.1",
            node_id: checkpoint_node_data,
        }

        # VERIFY: Crashes with clear ValueError
        with pytest.raises(ValueError, match=error_pattern):
            executor.restore_from_checkpoint(invalid_checkpoint)
