# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

from datetime import UTC, datetime


class TestNodesDeterminismColumn:
    """Tests for determinism column in nodes table."""

    def test_node_model_has_determinism_field(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import Determinism, Node, NodeType

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,  # New field
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.determinism == Determinism.DETERMINISTIC

    def test_determinism_values(self) -> None:
        """Verify valid determinism values match Determinism enum."""
        from elspeth.contracts import Determinism

        valid_values = {d.value for d in Determinism}
        # All 6 values per architecture specification
        expected = {
            "deterministic",
            "seeded",
            "io_read",
            "io_write",
            "external_call",
            "non_deterministic",
        }
        assert valid_values == expected


class TestPhase5CheckpointSchema:
    """Tests for checkpoint table added in Phase 5."""

    def test_checkpoints_table_has_topology_validation_columns(self) -> None:
        """P1: Verify checkpoint topology hash columns exist and are non-nullable.

        These columns were added in Bug #7 fix for checkpoint validation.
        A schema regression that removes them would break checkpoint
        compatibility validation and undermine recovery integrity.
        """
        from elspeth.core.landscape.schema import checkpoints_table

        columns = {c.name: c for c in checkpoints_table.columns}

        # upstream_topology_hash must exist and be non-nullable
        assert "upstream_topology_hash" in columns, "Missing upstream_topology_hash column"
        assert columns["upstream_topology_hash"].nullable is False, "upstream_topology_hash must be non-nullable for checkpoint validation"

        # checkpoint_node_config_hash must exist and be non-nullable
        assert "checkpoint_node_config_hash" in columns, "Missing checkpoint_node_config_hash column"
        assert columns["checkpoint_node_config_hash"].nullable is False, (
            "checkpoint_node_config_hash must be non-nullable for checkpoint validation"
        )

    def test_checkpoint_model(self) -> None:
        from elspeth.contracts import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="cp-001",
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=42,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
        )
        assert checkpoint.sequence_number == 42

    def test_checkpoint_model_with_aggregation_state(self) -> None:
        """Verify Checkpoint model supports aggregation state."""
        from elspeth.contracts import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="cp-002",
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=100,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
            aggregation_state_json='{"buffer": [1, 2, 3]}',
        )
        assert checkpoint.aggregation_state_json == '{"buffer": [1, 2, 3]}'


class TestBatchStatusType:
    """Tests for Batch.status type being BatchStatus enum (WP-05 Task 4)."""

    def test_batch_status_is_typed(self) -> None:
        """Batch.status should accept BatchStatus enum."""
        from elspeth.contracts import Batch, BatchStatus

        batch = Batch(
            batch_id="b1",
            run_id="r1",
            aggregation_node_id="agg1",
            attempt=1,
            status=BatchStatus.DRAFT,  # Should work without type error
            created_at=datetime.now(UTC),
        )

        assert batch.status == BatchStatus.DRAFT
