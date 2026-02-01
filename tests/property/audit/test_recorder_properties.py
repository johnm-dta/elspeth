# tests/property/audit/test_recorder_properties.py
"""Property-based tests for LandscapeRecorder determinism and integrity.

THE AUDIT BACKBONE:
The LandscapeRecorder is the heart of ELSPETH's audit trail. Every decision,
every transformation, every routing choice flows through this component.
If it fails, audit integrity is compromised.

Properties verified:
1. Recording is deterministic (same inputs -> same audit structure)
2. Foreign key constraints are satisfied (referential integrity)
3. No silent data loss (everything recorded is retrievable)
4. Unique ID generation (no collisions)

These tests use Hypothesis to generate diverse inputs and verify that the
recorder behaves correctly under all conditions.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import text

from elspeth.contracts import (
    Determinism,
    NodeStateStatus,
    NodeType,
    RowOutcome,
    RunStatus,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from tests.property.conftest import row_data

# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Strategy for run configurations (minimal valid config)
run_configs = st.fixed_dictionaries(
    {
        "source": st.fixed_dictionaries({"plugin": st.text(min_size=1, max_size=20)}),
        "sinks": st.fixed_dictionaries({"default": st.fixed_dictionaries({"plugin": st.text(min_size=1, max_size=20)})}),
    }
)

# Strategy for node types
node_types = st.sampled_from(list(NodeType))

# Strategy for determinism levels
determinism_levels = st.sampled_from(list(Determinism))

# Strategy for valid row indices
row_indices = st.integers(min_value=0, max_value=1_000_000)

# Strategy for terminal outcomes (excluding BUFFERED which is non-terminal)
terminal_outcomes = st.sampled_from([o for o in RowOutcome if o.is_terminal and o != RowOutcome.FORKED])


# =============================================================================
# Test Helpers
# =============================================================================


def create_dynamic_schema() -> SchemaConfig:
    """Create a dynamic schema config for testing."""
    return SchemaConfig.from_dict({"fields": "dynamic"})


def verify_row_exists(db: LandscapeDB, row_id: str) -> bool:
    """Check if a row exists in the database."""
    with db.connection() as conn:
        result = conn.execute(
            text("SELECT 1 FROM rows WHERE row_id = :row_id"),
            {"row_id": row_id},
        ).fetchone()
        return result is not None


def verify_token_exists(db: LandscapeDB, token_id: str) -> bool:
    """Check if a token exists in the database."""
    with db.connection() as conn:
        result = conn.execute(
            text("SELECT 1 FROM tokens WHERE token_id = :token_id"),
            {"token_id": token_id},
        ).fetchone()
        return result is not None


def verify_node_exists(db: LandscapeDB, node_id: str, run_id: str) -> bool:
    """Check if a node exists in the database (composite PK)."""
    with db.connection() as conn:
        result = conn.execute(
            text("SELECT 1 FROM nodes WHERE node_id = :node_id AND run_id = :run_id"),
            {"node_id": node_id, "run_id": run_id},
        ).fetchone()
        return result is not None


def count_rows_for_run(db: LandscapeDB, run_id: str) -> int:
    """Count all rows recorded for a run."""
    with db.connection() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM rows WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def count_tokens_for_run(db: LandscapeDB, run_id: str) -> int:
    """Count all tokens recorded for a run."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def count_outcomes_for_run(db: LandscapeDB, run_id: str) -> int:
    """Count all token outcomes recorded for a run."""
    with db.connection() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM token_outcomes WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).scalar()
        return result or 0


# =============================================================================
# Property Tests: Run Recording
# =============================================================================


class TestRunRecordingProperties:
    """Property tests for run lifecycle recording."""

    @given(config=run_configs)
    @settings(max_examples=50, deadline=None)
    def test_begin_run_creates_running_status(self, config: dict[str, Any]) -> None:
        """Property: begin_run() creates a run record in RUNNING status."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")

        # Verify run is persisted
        retrieved = recorder.get_run(run.run_id)
        assert retrieved is not None, "Run was not persisted to database"
        assert retrieved.status == RunStatus.RUNNING, f"Expected RUNNING status, got {retrieved.status}"
        assert retrieved.run_id == run.run_id
        assert retrieved.config_hash == run.config_hash

    @given(config=run_configs)
    @settings(max_examples=50, deadline=None)
    def test_complete_run_updates_status(self, config: dict[str, Any]) -> None:
        """Property: complete_run() updates status to COMPLETED."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        completed = recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED
        assert completed.completed_at is not None, "completed_at should be set"

        # Verify persisted state
        retrieved = recorder.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.status == RunStatus.COMPLETED

    @given(config=run_configs, status=st.sampled_from([RunStatus.COMPLETED, RunStatus.FAILED]))
    @settings(max_examples=50, deadline=None)
    def test_run_status_transitions(self, config: dict[str, Any], status: RunStatus) -> None:
        """Property: Runs can transition to terminal states (COMPLETED/FAILED)."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        completed = recorder.complete_run(run.run_id, status=status)

        assert completed.status == status
        retrieved = recorder.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.status == status

    @given(config=run_configs)
    @settings(max_examples=30, deadline=None)
    def test_config_hash_is_deterministic(self, config: dict[str, Any]) -> None:
        """Property: Same config produces same config_hash."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create two runs with identical config
        run1 = recorder.begin_run(config=config, canonical_version="1.0")
        run2 = recorder.begin_run(config=config, canonical_version="1.0")

        assert run1.config_hash == run2.config_hash, (
            "Identical configs should produce identical hashes. This is essential for reproducibility checks."
        )

        # Different configs should (almost always) produce different hashes
        # We can't guarantee this for all inputs, but it should hold for non-trivial diffs
        modified_config = {**config, "extra_key": "extra_value"}
        run3 = recorder.begin_run(config=modified_config, canonical_version="1.0")
        assert run3.config_hash != run1.config_hash, "Different configs should produce different hashes"


# =============================================================================
# Property Tests: Node Recording
# =============================================================================


class TestNodeRecordingProperties:
    """Property tests for node (plugin instance) registration."""

    @given(
        config=run_configs,
        plugin_name=st.text(min_size=1, max_size=50),
        node_type=node_types,
        determinism=determinism_levels,
    )
    @settings(max_examples=50, deadline=None)
    def test_register_node_persists_correctly(
        self,
        config: dict[str, Any],
        plugin_name: str,
        node_type: NodeType,
        determinism: Determinism,
    ) -> None:
        """Property: register_node() creates node record with correct fields."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name=plugin_name,
            node_type=node_type,
            plugin_version="1.0.0",
            config={"test": True},
            determinism=determinism,
            schema_config=create_dynamic_schema(),
        )

        # Verify persisted state
        retrieved = recorder.get_node(node.node_id, run.run_id)
        assert retrieved is not None, "Node was not persisted"
        assert retrieved.plugin_name == plugin_name
        assert retrieved.node_type == node_type
        assert retrieved.determinism == determinism
        assert retrieved.run_id == run.run_id

    @given(config=run_configs, n_nodes=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_multiple_nodes_have_unique_ids(self, config: dict[str, Any], n_nodes: int) -> None:
        """Property: Multiple nodes can be registered with unique IDs."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")

        node_ids = set()
        for i in range(n_nodes):
            node = recorder.register_node(
                run_id=run.run_id,
                plugin_name=f"plugin_{i}",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0.0",
                config={"index": i},
                sequence=i,
                schema_config=create_dynamic_schema(),
            )
            node_ids.add(node.node_id)

        assert len(node_ids) == n_nodes, f"Expected {n_nodes} unique node IDs, got {len(node_ids)}. ID collision detected!"

        # Verify all nodes are persisted
        nodes = recorder.get_nodes(run.run_id)
        assert len(nodes) == n_nodes

    @given(config=run_configs)
    @settings(max_examples=30, deadline=None)
    def test_node_config_hash_is_deterministic(self, config: dict[str, Any]) -> None:
        """Property: Same node config produces same config_hash."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")

        node_config = {"field": "value", "number": 42}

        node1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config=node_config,
            schema_config=create_dynamic_schema(),
        )

        node2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_plugin",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config=node_config,
            schema_config=create_dynamic_schema(),
        )

        assert node1.config_hash == node2.config_hash, "Identical configs should produce identical hashes"


# =============================================================================
# Property Tests: Row Recording
# =============================================================================


class TestRowRecordingProperties:
    """Property tests for source row recording."""

    @given(config=run_configs, data=row_data, row_index=row_indices)
    @settings(max_examples=50, deadline=None)
    def test_create_row_persists_correctly(self, config: dict[str, Any], data: dict[str, Any], row_index: int) -> None:
        """Property: create_row() creates row record with correct fields."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=row_index,
            data=data,
        )

        # Verify row is persisted
        assert verify_row_exists(db, row.row_id), "Row was not persisted"
        assert row.row_index == row_index
        assert row.source_data_hash == stable_hash(data)

    @given(config=run_configs, n_rows=st.integers(min_value=1, max_value=50))
    @settings(max_examples=30, deadline=None)
    def test_row_indices_stored_correctly(self, config: dict[str, Any], n_rows: int) -> None:
        """Property: Row indices are stored correctly."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        rows = []
        for i in range(n_rows):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                data={"value": i},
            )
            rows.append(row)

        # Verify all indices are correct
        for i, row in enumerate(rows):
            assert row.row_index == i, f"Row {i} has wrong index: {row.row_index}"

        # Verify count
        assert count_rows_for_run(db, run.run_id) == n_rows

    @given(config=run_configs, n_rows=st.integers(min_value=1, max_value=30))
    @settings(max_examples=30, deadline=None)
    def test_row_ids_are_unique(self, config: dict[str, Any], n_rows: int) -> None:
        """Property: Row IDs are unique."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row_ids = set()
        for i in range(n_rows):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                data={"value": i},
            )
            row_ids.add(row.row_id)

        assert len(row_ids) == n_rows, f"Expected {n_rows} unique row IDs, got {len(row_ids)}. ID collision!"


# =============================================================================
# Property Tests: Token Recording
# =============================================================================


class TestTokenRecordingProperties:
    """Property tests for token recording (row instances in DAG paths)."""

    @given(config=run_configs, data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_create_token_links_to_row(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        """Property: create_token() creates token linked to its row."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )

        token = recorder.create_token(row_id=row.row_id)

        # Verify token is persisted and linked
        assert verify_token_exists(db, token.token_id), "Token was not persisted"
        assert token.row_id == row.row_id, "Token not linked to row"

    @given(config=run_configs, n_tokens=st.integers(min_value=1, max_value=30))
    @settings(max_examples=30, deadline=None)
    def test_token_ids_are_unique(self, config: dict[str, Any], n_tokens: int) -> None:
        """Property: Token IDs are unique."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"value": 1},
        )

        token_ids = set()
        for _ in range(n_tokens):
            token = recorder.create_token(row_id=row.row_id)
            token_ids.add(token.token_id)

        assert len(token_ids) == n_tokens, f"Expected {n_tokens} unique token IDs, got {len(token_ids)}. ID collision!"


# =============================================================================
# Property Tests: Token Outcomes
# =============================================================================


class TestTokenOutcomeProperties:
    """Property tests for token outcome recording."""

    @given(config=run_configs, data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_record_completed_outcome(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        """Property: COMPLETED outcome is persisted with required sink_name."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )
        token = recorder.create_token(row_id=row.row_id)

        # Record COMPLETED outcome (requires sink_name)
        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="default",
        )

        assert outcome_id is not None

        # Verify persisted
        outcome = recorder.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.COMPLETED.value
        assert outcome.is_terminal is True
        assert outcome.sink_name == "default"

    @given(config=run_configs, data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_record_quarantined_outcome(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        """Property: QUARANTINED outcome is persisted with error_hash."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )
        token = recorder.create_token(row_id=row.row_id)

        # Record QUARANTINED outcome (requires error_hash)
        error_hash = stable_hash({"reason": "validation_failed"})
        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.QUARANTINED,
            error_hash=error_hash,
        )

        assert outcome_id is not None

        # Verify persisted
        outcome = recorder.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome == RowOutcome.QUARANTINED.value
        assert outcome.is_terminal is True
        assert outcome.error_hash == error_hash

    @given(config=run_configs, n_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_outcome_count_matches_recorded(self, config: dict[str, Any], n_rows: int) -> None:
        """Property: All recorded outcomes are persisted (no silent loss)."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        for i in range(n_rows):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                data={"value": i},
            )
            token = recorder.create_token(row_id=row.row_id)
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="default",
            )

        # Verify count
        outcome_count = count_outcomes_for_run(db, run.run_id)
        assert outcome_count == n_rows, f"Expected {n_rows} outcomes, got {outcome_count}. Data was lost!"

    @pytest.mark.parametrize(
        ("outcome", "required_field", "kwargs"),
        [
            (RowOutcome.COMPLETED, "sink_name", {}),
            (RowOutcome.ROUTED, "sink_name", {}),
            (RowOutcome.FORKED, "fork_group_id", {}),
            (RowOutcome.FAILED, "error_hash", {}),
            (RowOutcome.QUARANTINED, "error_hash", {}),
            (RowOutcome.CONSUMED_IN_BATCH, "batch_id", {}),
            (RowOutcome.COALESCED, "join_group_id", {}),
            (RowOutcome.EXPANDED, "expand_group_id", {}),
            (RowOutcome.BUFFERED, "batch_id", {}),
        ],
    )
    def test_record_outcome_requires_fields(
        self,
        outcome: RowOutcome,
        required_field: str,
        kwargs: dict[str, Any],
    ) -> None:
        """Required fields are enforced for each outcome type."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": {"plugin": "test"}}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"value": 1},
        )
        token = recorder.create_token(row_id=row.row_id)

        with pytest.raises(ValueError, match=required_field):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=outcome,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("outcome", "kwargs"),
        [
            (RowOutcome.COMPLETED, {"sink_name": "default"}),
            (RowOutcome.ROUTED, {"sink_name": "error_sink"}),
            (RowOutcome.FORKED, {"fork_group_id": "fork_group_1"}),
            (RowOutcome.FAILED, {"error_hash": stable_hash({"reason": "failure"})}),
            (RowOutcome.QUARANTINED, {"error_hash": stable_hash({"reason": "validation"})}),
            (RowOutcome.CONSUMED_IN_BATCH, {"batch_id": "batch_1"}),
            (RowOutcome.COALESCED, {"join_group_id": "join_group_1"}),
            (RowOutcome.EXPANDED, {"expand_group_id": "expand_group_1"}),
            (RowOutcome.BUFFERED, {"batch_id": "batch_2"}),
        ],
    )
    def test_record_outcome_accepts_required_fields(self, outcome: RowOutcome, kwargs: dict[str, Any]) -> None:
        """Outcomes with required fields are recorded and retrievable."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": {"plugin": "test"}}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"value": 1},
        )
        token = recorder.create_token(row_id=row.row_id)

        if outcome in {RowOutcome.CONSUMED_IN_BATCH, RowOutcome.BUFFERED}:
            aggregation_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_aggregation",
                node_type=NodeType.AGGREGATION,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )
            batch_id = kwargs["batch_id"]
            recorder.create_batch(
                run_id=run.run_id,
                aggregation_node_id=aggregation_node.node_id,
                batch_id=batch_id,
            )

        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=outcome,
            **kwargs,
        )
        assert outcome_id is not None

        persisted = recorder.get_token_outcome(token.token_id)
        assert persisted is not None
        assert persisted.outcome == outcome.value
        assert persisted.is_terminal == outcome.is_terminal


# =============================================================================
# Property Tests: Node State Recording
# =============================================================================


class TestNodeStateProperties:
    """Property tests for node state (processing record) recording."""

    @given(config=run_configs, data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_begin_node_state_creates_open_status(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        """Property: begin_node_state() creates state with OPEN status."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )
        token = recorder.create_token(row_id=row.row_id)

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform_node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=data,
        )

        assert state.status == NodeStateStatus.OPEN
        assert state.token_id == token.token_id
        assert state.node_id == transform_node.node_id

        # Verify persisted
        retrieved = recorder.get_node_state(state.state_id)
        assert retrieved is not None
        assert retrieved.status == NodeStateStatus.OPEN

    @given(config=run_configs, data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_complete_node_state_updates_status(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        """Property: complete_node_state() updates state to terminal status."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )
        token = recorder.create_token(row_id=row.row_id)

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform_node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=data,
        )

        output_data = {**data, "processed": True}
        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data=output_data,
            duration_ms=10.5,
        )

        assert completed.status == NodeStateStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.output_hash == stable_hash(output_data)


# =============================================================================
# Property Tests: Foreign Key Integrity
# =============================================================================


class TestForeignKeyIntegrity:
    """Property tests for referential integrity of audit records."""

    @given(config=run_configs, n_rows=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_rows_reference_valid_run(self, config: dict[str, Any], n_rows: int) -> None:
        """Property: All rows reference a valid run_id."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        for i in range(n_rows):
            recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                data={"value": i},
            )

        # Verify all rows reference valid run
        with db.connection() as conn:
            orphan_count = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM rows r
                    LEFT JOIN runs ru ON ru.run_id = r.run_id
                    WHERE ru.run_id IS NULL
                """)
            ).scalar()

        assert orphan_count == 0, f"Found {orphan_count} rows with invalid run_id"

    @given(config=run_configs, n_tokens=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_tokens_reference_valid_row(self, config: dict[str, Any], n_tokens: int) -> None:
        """Property: All tokens reference a valid row_id."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"value": 1},
        )

        for _ in range(n_tokens):
            recorder.create_token(row_id=row.row_id)

        # Verify all tokens reference valid row
        with db.connection() as conn:
            orphan_count = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM tokens t
                    LEFT JOIN rows r ON r.row_id = t.row_id
                    WHERE r.row_id IS NULL
                """)
            ).scalar()

        assert orphan_count == 0, f"Found {orphan_count} tokens with invalid row_id"

    @given(config=run_configs, n_outcomes=st.integers(min_value=1, max_value=20))
    @settings(max_examples=30, deadline=None)
    def test_outcomes_reference_valid_token(self, config: dict[str, Any], n_outcomes: int) -> None:
        """Property: All outcomes reference a valid token_id."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config=config, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        for i in range(n_outcomes):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=i,
                data={"value": i},
            )
            token = recorder.create_token(row_id=row.row_id)
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="default",
            )

        # Verify all outcomes reference valid token
        with db.connection() as conn:
            orphan_count = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM token_outcomes o
                    LEFT JOIN tokens t ON t.token_id = o.token_id
                    WHERE t.token_id IS NULL
                """)
            ).scalar()

        assert orphan_count == 0, f"Found {orphan_count} outcomes with invalid token_id"


# =============================================================================
# Property Tests: Data Hash Determinism
# =============================================================================


class TestHashDeterminism:
    """Property tests for hash consistency in audit records."""

    @given(data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_row_hash_is_deterministic(self, data: dict[str, Any]) -> None:
        """Property: Same row data produces same source_data_hash."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": {"plugin": "test"}}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        row1 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )

        row2 = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=1,
            data=data,
        )

        assert row1.source_data_hash == row2.source_data_hash, (
            "Identical row data should produce identical hashes. Hash determinism is essential for audit integrity."
        )

    @given(data=row_data)
    @settings(max_examples=50, deadline=None)
    def test_input_hash_is_deterministic(self, data: dict[str, Any]) -> None:
        """Property: Same input data produces same input_hash in node states."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": {"plugin": "test"}}, canonical_version="1.0")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=create_dynamic_schema(),
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=data,
        )
        token1 = recorder.create_token(row_id=row.row_id)
        token2 = recorder.create_token(row_id=row.row_id)

        state1 = recorder.begin_node_state(
            token_id=token1.token_id,
            node_id=transform_node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=data,
        )

        state2 = recorder.begin_node_state(
            token_id=token2.token_id,
            node_id=transform_node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data=data,
        )

        assert state1.input_hash == state2.input_hash, "Identical input data should produce identical input hashes"
