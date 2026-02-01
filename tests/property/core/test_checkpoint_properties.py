# tests/property/core/test_checkpoint_properties.py
"""Property-based tests for checkpoint recovery system.

These tests verify the fundamental invariants of ELSPETH's checkpoint system:

Recovery Properties:
- Aggregation state round-trips through JSON serialization deterministically
- Format version validation rejects incompatible versions
- Topology hash detects ANY graph change (not just upstream)

Ordering Properties:
- Sequence numbers are monotonically ordered on retrieval
- Latest checkpoint is always the highest sequence number

Validation Properties:
- Checkpoint creation requires node to exist in graph
- Config hash is stable for identical configurations

These properties are CRITICAL for crash recovery - if violated, resumed
pipelines could produce corrupt audit trails or lose aggregation data.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts import Checkpoint, Determinism, NodeType, RunStatus
from elspeth.core.canonical import stable_hash
from elspeth.core.checkpoint import CheckpointCompatibilityValidator, CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table, tokens_table
from tests.property.conftest import json_primitives, json_values

# =============================================================================
# Strategies for checkpoint testing
# =============================================================================

# Aggregation state: dict with string keys
aggregation_states = st.dictionaries(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
    json_values,
    min_size=0,
    max_size=5,
)

# Valid run IDs
run_ids = st.text(
    min_size=5,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
).filter(lambda s: s[0].isalpha())

# Valid token IDs
token_ids = st.text(
    min_size=5,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-"),
).filter(lambda s: s[0].isalpha())

# Sequence numbers (positive integers)
sequence_numbers = st.integers(min_value=1, max_value=1_000_000)


def create_test_graph(num_transforms: int = 2) -> ExecutionGraph:
    """Create a simple linear test graph."""
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="test_source")

    prev = "source"
    for i in range(num_transforms):
        node_id = f"transform_{i}"
        graph.add_node(node_id, node_type="transform", plugin_name="test_transform")
        graph.add_edge(prev, node_id, label="continue")
        prev = node_id

    graph.add_node("sink", node_type="sink", plugin_name="test_sink")
    graph.add_edge(prev, "sink", label="continue")

    return graph


def create_test_db() -> tuple[LandscapeDB, Path]:
    """Create a temporary test database.

    LandscapeDB auto-initializes tables in __init__, so no explicit
    initialize() call is needed.
    """
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_audit.db"
    db = LandscapeDB(f"sqlite:///{db_path}")
    return db, db_path


def setup_checkpoint_prerequisites(
    db: LandscapeDB,
    run_id: str,
    token_id: str = "token-001",
    node_id: str = "transform_0",
) -> None:
    """Set up all prerequisites for checkpoint creation.

    Checkpoints have multiple foreign key constraints:
    - run_id → runs
    - token_id → tokens → rows → nodes
    - (node_id, run_id) → nodes

    This helper creates the minimum required records.
    """
    now = datetime.now(UTC)

    with db.engine.begin() as conn:
        # 1. Create run
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test-config-hash",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.RUNNING,
            )
        )

        # 2. Create source node (needed for rows FK)
        conn.execute(
            nodes_table.insert().values(
                node_id="source",
                run_id=run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="source-hash",
                config_json="{}",
                registered_at=now,
            )
        )

        # 3. Create checkpoint node (for FK constraint)
        conn.execute(
            nodes_table.insert().values(
                node_id=node_id,
                run_id=run_id,
                plugin_name="test_transform",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash="transform-hash",
                config_json="{}",
                registered_at=now,
            )
        )

        # 4. Create row (needed for tokens FK)
        conn.execute(
            rows_table.insert().values(
                row_id="row-001",
                run_id=run_id,
                source_node_id="source",
                row_index=0,
                source_data_hash="data-hash",
                created_at=now,
            )
        )

        # 5. Create token (needed for checkpoints FK)
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id="row-001",
                created_at=now,
            )
        )


# =============================================================================
# Aggregation State Round-Trip Properties
# =============================================================================


class TestAggregationStateRoundTripProperties:
    """Property tests for aggregation state serialization."""

    @given(state=aggregation_states)
    @settings(max_examples=200)
    def test_aggregation_state_roundtrip_deterministic(self, state: dict[str, Any]) -> None:
        """Property: Aggregation state survives JSON round-trip identically.

        This is CRITICAL for crash recovery - if aggregation buffers are
        corrupted during checkpoint save/load, partial data could be lost
        or duplicated when pipeline resumes.
        """
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            # Create run first (foreign key constraint)
            setup_checkpoint_prerequisites(db, "test-run-001")

            # Create checkpoint with aggregation state
            checkpoint = manager.create_checkpoint(
                run_id="test-run-001",
                token_id="token-001",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
                aggregation_state=state,
            )

            assert checkpoint.aggregation_state_json is not None
            restored = json.loads(checkpoint.aggregation_state_json)
            assert restored == state, f"Aggregation state corrupted during round-trip!\nOriginal: {state}\nRestored: {restored}"
        finally:
            db.close()

    @given(state=aggregation_states)
    @settings(max_examples=100)
    def test_aggregation_state_hash_deterministic(self, state: dict[str, Any]) -> None:
        """Property: Same aggregation state produces same JSON representation.

        This uses the actual checkpoint serialization path to avoid
        diverging from production behavior.
        """
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-run-json", token_id="token-json")

            checkpoint1 = manager.create_checkpoint(
                run_id="test-run-json",
                token_id="token-json",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
                aggregation_state=state,
            )
            checkpoint2 = manager.create_checkpoint(
                run_id="test-run-json",
                token_id="token-json",
                node_id="transform_0",
                sequence_number=2,
                graph=graph,
                aggregation_state=state,
            )

            assert checkpoint1.aggregation_state_json is not None
            assert checkpoint1.aggregation_state_json == checkpoint2.aggregation_state_json
        finally:
            db.close()

    def test_none_aggregation_state_stored_as_null(self) -> None:
        """Property: None aggregation state is stored as SQL NULL."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-run-002", token_id="token-002")

            checkpoint = manager.create_checkpoint(
                run_id="test-run-002",
                token_id="token-002",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
                aggregation_state=None,
            )

            assert checkpoint.aggregation_state_json is None
        finally:
            db.close()

    def test_checkpoint_rejects_nan_in_aggregation_state(self) -> None:
        """Property: NaN in aggregation state raises ValueError.

        Per CLAUDE.md, NaN and Infinity are strictly rejected for audit integrity.
        Checkpoints use json.dumps(allow_nan=False) which enforces this policy.
        """
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-nan", token_id="token-nan")

            with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
                manager.create_checkpoint(
                    run_id="test-nan",
                    token_id="token-nan",
                    node_id="transform_0",
                    sequence_number=1,
                    graph=graph,
                    aggregation_state={"count": 0, "avg": float("nan")},
                )
        finally:
            db.close()

    def test_checkpoint_rejects_infinity_in_aggregation_state(self) -> None:
        """Property: Infinity in aggregation state raises ValueError.

        Per CLAUDE.md, NaN and Infinity are strictly rejected for audit integrity.
        Checkpoints use json.dumps(allow_nan=False) which enforces this policy.
        """
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-inf", token_id="token-inf")

            with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
                manager.create_checkpoint(
                    run_id="test-inf",
                    token_id="token-inf",
                    node_id="transform_0",
                    sequence_number=1,
                    graph=graph,
                    aggregation_state={"value": float("inf")},
                )
        finally:
            db.close()


# =============================================================================
# Format Version Validation Properties
# =============================================================================


class TestFormatVersionProperties:
    """Property tests for checkpoint format version validation."""

    @given(version=st.integers(min_value=0, max_value=100))
    @settings(max_examples=50)
    def test_incompatible_format_versions_rejected(self, version: int) -> None:
        """Property: Non-current format versions are rejected.

        Cross-version resume is explicitly forbidden - both older AND newer
        versions must be rejected to prevent silent corruption.
        """
        from elspeth.core.checkpoint.manager import IncompatibleCheckpointError

        # Skip current version (it's valid)
        assume(version != Checkpoint.CURRENT_FORMAT_VERSION)

        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-run-version", token_id="token-version")

            # Create a valid checkpoint first
            manager.create_checkpoint(
                run_id="test-run-version",
                token_id="token-version",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
            )

            # Manually update the format version to test rejection
            # (simulates loading a checkpoint from different version)
            from sqlalchemy import update

            from elspeth.core.landscape.schema import checkpoints_table

            with db.engine.begin() as conn:
                conn.execute(
                    update(checkpoints_table).where(checkpoints_table.c.run_id == "test-run-version").values(format_version=version)
                )

            # Now try to load - should raise
            with pytest.raises(IncompatibleCheckpointError):
                manager.get_latest_checkpoint("test-run-version")

        finally:
            db.close()

    def test_current_format_version_accepted(self) -> None:
        """Property: Current format version is accepted."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-run-current", token_id="token-current")

            checkpoint = manager.create_checkpoint(
                run_id="test-run-current",
                token_id="token-current",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
            )

            # Should have current version
            assert checkpoint.format_version == Checkpoint.CURRENT_FORMAT_VERSION

            # Should load without error
            loaded = manager.get_latest_checkpoint("test-run-current")
            assert loaded is not None
            assert loaded.format_version == Checkpoint.CURRENT_FORMAT_VERSION
        finally:
            db.close()


# =============================================================================
# Topology Hash Detection Properties
# =============================================================================


class TestTopologyHashProperties:
    """Property tests for topology change detection."""

    def test_identical_graph_produces_identical_hash(self) -> None:
        """Property: Same graph structure produces same topology hash."""
        graph1 = create_test_graph(num_transforms=3)
        graph2 = create_test_graph(num_transforms=3)

        validator = CheckpointCompatibilityValidator()
        hash1 = validator.compute_full_topology_hash(graph1)
        hash2 = validator.compute_full_topology_hash(graph2)

        assert hash1 == hash2, "Identical graphs produced different hashes"

    @given(num_transforms=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_different_transform_count_detected(self, num_transforms: int) -> None:
        """Property: Different number of transforms produces different hash."""
        graph_small = create_test_graph(num_transforms=num_transforms)
        graph_large = create_test_graph(num_transforms=num_transforms + 1)

        validator = CheckpointCompatibilityValidator()
        hash_small = validator.compute_full_topology_hash(graph_small)
        hash_large = validator.compute_full_topology_hash(graph_large)

        assert hash_small != hash_large, f"Different graph sizes produced same hash! ({num_transforms} vs {num_transforms + 1} transforms)"

    def test_sibling_branch_change_detected(self) -> None:
        """Property: Changes to sibling branches (not upstream) are detected.

        BUG-COMPAT-01: Previously only upstream changes were detected,
        allowing sibling sink branches to change silently.
        """
        # Create diamond graph with two sinks
        graph1 = ExecutionGraph()
        graph1.add_node("source", node_type="source", plugin_name="test_source")
        graph1.add_node("transform", node_type="transform", plugin_name="test_transform")
        graph1.add_node("sink_a", node_type="sink", plugin_name="test_sink_a")
        graph1.add_node("sink_b", node_type="sink", plugin_name="test_sink_b")
        graph1.add_edge("source", "transform", label="continue")
        graph1.add_edge("transform", "sink_a", label="route_a")
        graph1.add_edge("transform", "sink_b", label="route_b")

        # Create same graph but change sink_b's plugin
        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type="source", plugin_name="test_source")
        graph2.add_node("transform", node_type="transform", plugin_name="test_transform")
        graph2.add_node("sink_a", node_type="sink", plugin_name="test_sink_a")
        graph2.add_node("sink_b", node_type="sink", plugin_name="different_sink")  # Changed!
        graph2.add_edge("source", "transform", label="continue")
        graph2.add_edge("transform", "sink_a", label="route_a")
        graph2.add_edge("transform", "sink_b", label="route_b")

        validator = CheckpointCompatibilityValidator()
        hash1 = validator.compute_full_topology_hash(graph1)
        hash2 = validator.compute_full_topology_hash(graph2)

        assert hash1 != hash2, "Sibling branch change NOT detected! BUG-COMPAT-01 regression - downstream/sibling changes must be detected."

    def test_topology_hash_is_deterministic(self) -> None:
        """Property: Topology hash computation is deterministic."""
        graph = create_test_graph(num_transforms=3)
        validator = CheckpointCompatibilityValidator()

        hash1 = validator.compute_full_topology_hash(graph)
        hash2 = validator.compute_full_topology_hash(graph)
        hash3 = validator.compute_full_topology_hash(graph)

        assert hash1 == hash2 == hash3, "Topology hash is not deterministic"


# =============================================================================
# Compatibility Validation Properties
# =============================================================================


class TestCompatibilityValidationProperties:
    """Property tests for checkpoint-to-graph compatibility validation."""

    def test_unchanged_graph_validates(self) -> None:
        """Property: Checkpoint validates against unchanged graph."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-unchanged", token_id="token-unchanged")

            checkpoint = manager.create_checkpoint(
                run_id="test-unchanged",
                token_id="token-unchanged",
                node_id="transform_0",
                sequence_number=1,
                graph=graph,
            )

            validator = CheckpointCompatibilityValidator()
            result = validator.validate(checkpoint, graph)

            assert result.can_resume is True, f"Unchanged graph should validate: {result.reason}"
        finally:
            db.close()

    def test_missing_node_fails_validation(self) -> None:
        """Property: Checkpoint fails validation if its node is removed."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph_original = create_test_graph(num_transforms=3)

            setup_checkpoint_prerequisites(db, "test-missing-node", token_id="token-missing", node_id="transform_1")

            checkpoint = manager.create_checkpoint(
                run_id="test-missing-node",
                token_id="token-missing",
                node_id="transform_1",  # Will be missing in modified graph
                sequence_number=1,
                graph=graph_original,
            )

            # Create graph without transform_1
            graph_modified = create_test_graph(num_transforms=1)  # Only transform_0

            validator = CheckpointCompatibilityValidator()
            result = validator.validate(checkpoint, graph_modified)

            assert result.can_resume is False
            assert "no longer exists" in result.reason
        finally:
            db.close()

    def test_changed_config_fails_validation(self) -> None:
        """Property: Checkpoint fails validation if node config changed."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)

            # Graph with specific config
            graph1 = ExecutionGraph()
            graph1.add_node("source", node_type="source", plugin_name="test_source")
            graph1.add_node(
                "transform_0",
                node_type="transform",
                plugin_name="test_transform",
                config={"param": "value_a"},
            )
            graph1.add_node("sink", node_type="sink", plugin_name="test_sink")
            graph1.add_edge("source", "transform_0", label="continue")
            graph1.add_edge("transform_0", "sink", label="continue")

            setup_checkpoint_prerequisites(db, "test-config-change", token_id="token-config")

            checkpoint = manager.create_checkpoint(
                run_id="test-config-change",
                token_id="token-config",
                node_id="transform_0",
                sequence_number=1,
                graph=graph1,
            )

            # Same structure but different config
            graph2 = ExecutionGraph()
            graph2.add_node("source", node_type="source", plugin_name="test_source")
            graph2.add_node(
                "transform_0",
                node_type="transform",
                plugin_name="test_transform",
                config={"param": "value_b"},  # Changed!
            )
            graph2.add_node("sink", node_type="sink", plugin_name="test_sink")
            graph2.add_edge("source", "transform_0", label="continue")
            graph2.add_edge("transform_0", "sink", label="continue")

            validator = CheckpointCompatibilityValidator()
            result = validator.validate(checkpoint, graph2)

            assert result.can_resume is False
            assert "configuration has changed" in result.reason
        finally:
            db.close()


# =============================================================================
# Sequence Number Ordering Properties
# =============================================================================


class TestSequenceNumberProperties:
    """Property tests for checkpoint sequence ordering."""

    @given(seq_numbers=st.lists(sequence_numbers, min_size=2, max_size=10, unique=True))
    @settings(max_examples=50)
    def test_checkpoints_retrieved_in_sequence_order(self, seq_numbers: list[int]) -> None:
        """Property: get_checkpoints() returns checkpoints ordered by sequence."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-ordering")

            # Create checkpoints in reversed order
            for seq in reversed(seq_numbers):
                manager.create_checkpoint(
                    run_id="test-ordering",
                    token_id="token-001",  # Reuse same token - sequence_number is unique
                    node_id="transform_0",
                    sequence_number=seq,
                    graph=graph,
                )

            # Retrieve all checkpoints
            checkpoints = manager.get_checkpoints("test-ordering")

            # Verify ordered by sequence_number
            sequences = [cp.sequence_number for cp in checkpoints]
            assert sequences == sorted(sequences), f"Checkpoints not in sequence order: {sequences}"
        finally:
            db.close()

    @given(seq_numbers=st.lists(sequence_numbers, min_size=2, max_size=10, unique=True))
    @settings(max_examples=50)
    def test_latest_checkpoint_is_highest_sequence(self, seq_numbers: list[int]) -> None:
        """Property: get_latest_checkpoint() returns highest sequence number."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-latest")

            for seq in seq_numbers:
                manager.create_checkpoint(
                    run_id="test-latest",
                    token_id="token-001",  # Reuse same token - sequence_number is unique
                    node_id="transform_0",
                    sequence_number=seq,
                    graph=graph,
                )

            latest = manager.get_latest_checkpoint("test-latest")

            assert latest is not None
            assert latest.sequence_number == max(seq_numbers), (
                f"Latest checkpoint has sequence {latest.sequence_number}, expected {max(seq_numbers)}"
            )
        finally:
            db.close()


# =============================================================================
# Checkpoint Creation Validation Properties
# =============================================================================


class TestCheckpointCreationProperties:
    """Property tests for checkpoint creation validation."""

    def test_create_checkpoint_requires_valid_node(self) -> None:
        """Property: Checkpoint creation fails if node doesn't exist in graph."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)
            graph = create_test_graph()

            setup_checkpoint_prerequisites(db, "test-invalid-node")

            with pytest.raises(ValueError, match="does not exist"):
                manager.create_checkpoint(
                    run_id="test-invalid-node",
                    token_id="token-invalid",
                    node_id="nonexistent_node",
                    sequence_number=1,
                    graph=graph,
                )
        finally:
            db.close()

    def test_create_checkpoint_requires_graph(self) -> None:
        """Property: Checkpoint creation fails if graph is None."""
        db, _ = create_test_db()
        try:
            manager = CheckpointManager(db)

            with pytest.raises(ValueError, match="graph parameter is required"):
                manager.create_checkpoint(
                    run_id="test-no-graph",
                    token_id="token-no-graph",
                    node_id="transform_0",
                    sequence_number=1,
                    graph=None,  # type: ignore
                )
        finally:
            db.close()

    @given(config=st.dictionaries(st.text(min_size=1, max_size=10), json_primitives, max_size=5))
    @settings(max_examples=50)
    def test_config_hash_is_stable(self, config: dict[str, Any]) -> None:
        """Property: Same config always produces same hash."""
        hash1 = stable_hash(config)
        hash2 = stable_hash(config)

        assert hash1 == hash2, "Config hash is not stable"

    @given(
        config1=st.dictionaries(st.text(min_size=1, max_size=10), json_primitives, min_size=1, max_size=3),
        config2=st.dictionaries(st.text(min_size=1, max_size=10), json_primitives, min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_different_configs_different_hashes(self, config1: dict[str, Any], config2: dict[str, Any]) -> None:
        """Property: Different configs produce different hashes (collision resistance)."""
        assume(config1 != config2)

        hash1 = stable_hash(config1)
        hash2 = stable_hash(config2)

        assert hash1 != hash2, f"Hash collision detected!\nConfig1: {config1}\nConfig2: {config2}\nHash: {hash1}"
