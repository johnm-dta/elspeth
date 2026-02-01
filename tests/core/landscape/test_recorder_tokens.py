"""Tests for LandscapeRecorder token operations."""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import Determinism, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestLandscapeRecorderTokens:
    """Row and token management."""

    def test_create_row(self) -> None:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )

        assert row.row_id is not None
        assert row.row_index == 0
        assert row.source_data_hash is not None

    def test_create_row_hash_correctness(self) -> None:
        """P1: Verify row hash matches stable_hash of data (not just non-NULL).

        Hash correctness is the audit integrity anchor. A regression in
        canonicalization or hash computation would silently corrupt the
        audit trail while weak tests still pass.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        test_data = {"name": "Alice", "value": 42}
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=test_data,
        )

        # Verify hash correctness (not just existence)
        expected_hash = stable_hash(test_data)
        assert row.source_data_hash == expected_hash, f"row hash mismatch: expected {expected_hash}, got {row.source_data_hash}"

    def test_create_initial_token(self) -> None:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )

        token = recorder.create_token(row_id=row.row_id)

        assert token.token_id is not None
        assert token.row_id == row.row_id
        assert token.fork_group_id is None  # Initial token

    def test_fork_token(self) -> None:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to two branches
        child_tokens, _fork_group_id = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
            run_id=run.run_id,
        )

        assert len(child_tokens) == 2
        assert child_tokens[0].branch_name == "stats"
        assert child_tokens[1].branch_name == "classifier"
        # All children share same fork_group_id
        assert child_tokens[0].fork_group_id == child_tokens[1].fork_group_id

    def test_fork_token_parent_lineage_verified(self) -> None:
        """P1: Verify fork creates parent relationships in token_parents table.

        Fork lineage is foundational to audit explainability. A regression
        that drops or corrupts fork lineage would break explain() queries
        for forks, yet without this test the weak fork_group_id/branch_name
        assertions would still pass.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to two branches
        child_tokens, _fork_group_id = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
            run_id=run.run_id,
        )

        # P1: Verify token_parents entries for each child
        parents_0 = recorder.get_token_parents(child_tokens[0].token_id)
        assert len(parents_0) == 1, f"Expected 1 parent for child 0, got {len(parents_0)}"
        assert parents_0[0].parent_token_id == parent_token.token_id
        assert parents_0[0].ordinal == 0

        parents_1 = recorder.get_token_parents(child_tokens[1].token_id)
        assert len(parents_1) == 1, f"Expected 1 parent for child 1, got {len(parents_1)}"
        assert parents_1[0].parent_token_id == parent_token.token_id
        assert parents_1[0].ordinal == 1

    def test_coalesce_tokens(self) -> None:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)
        children, _fork_group_id = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
            run_id=run.run_id,
        )

        # Coalesce back together
        merged = recorder.coalesce_tokens(
            parent_token_ids=[c.token_id for c in children],
            row_id=row.row_id,
        )

        assert merged.token_id is not None
        assert merged.join_group_id is not None

    def test_fork_token_with_step_in_pipeline(self) -> None:
        """Fork stores step_in_pipeline in tokens table."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork with step_in_pipeline
        child_tokens, _fork_group_id = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
            run_id=run.run_id,
            step_in_pipeline=2,
        )

        # Verify step_in_pipeline is stored
        assert len(child_tokens) == 2
        assert child_tokens[0].step_in_pipeline == 2
        assert child_tokens[1].step_in_pipeline == 2

        # Verify retrieval via get_token
        retrieved = recorder.get_token(child_tokens[0].token_id)
        assert retrieved is not None
        assert retrieved.step_in_pipeline == 2

    def test_fork_token_rejects_empty_branches(self) -> None:
        """fork_token must have at least one branch (defense-in-depth).

        Per CLAUDE.md "no silent drops" invariant, empty forks would cause
        tokens to disappear without audit trail. Even if RoutingAction validates
        upstream, recorder MUST also validate as defense-in-depth.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Empty branches should be rejected
        with pytest.raises(ValueError, match="at least one branch"):
            recorder.fork_token(
                parent_token_id=parent_token.token_id,
                row_id=row.row_id,
                branches=[],  # Empty!
                run_id=run.run_id,
            )

    def test_coalesce_tokens_with_step_in_pipeline(self) -> None:
        """Coalesce stores step_in_pipeline in tokens table."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)
        children, _fork_group_id = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
            run_id=run.run_id,
            step_in_pipeline=1,
        )

        # Coalesce with step_in_pipeline
        merged = recorder.coalesce_tokens(
            parent_token_ids=[c.token_id for c in children],
            row_id=row.row_id,
            step_in_pipeline=3,
        )

        # Verify step_in_pipeline is stored
        assert merged.step_in_pipeline == 3

        # Verify retrieval via get_token
        retrieved = recorder.get_token(merged.token_id)
        assert retrieved is not None
        assert retrieved.step_in_pipeline == 3


class TestExpandToken:
    """Tests for expand_token (deaggregation audit trail)."""

    def test_expand_token_creates_children_with_parent_relationship(self) -> None:
        """expand_token creates child tokens linked to parent via token_parents."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup: create run, node, row, and parent token
        run = recorder.begin_run(config={"test": True}, canonical_version="1.0")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="json_explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"items": [1, 2, 3]},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Act: expand parent into 3 children
        children, _expand_group_id = recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            count=3,
            run_id=run.run_id,
            step_in_pipeline=2,
        )

        # Assert: 3 children created
        assert len(children) == 3

        # All children share same row_id (same source row)
        for child in children:
            assert child.row_id == row.row_id
            assert child.token_id != parent_token.token_id

        # All children share same expand_group_id
        expand_group_ids = {c.expand_group_id for c in children}
        assert len(expand_group_ids) == 1
        assert None not in expand_group_ids

        # Verify parent relationships recorded
        for i, child in enumerate(children):
            parents = recorder.get_token_parents(child.token_id)
            assert len(parents) == 1
            assert parents[0].parent_token_id == parent_token.token_id
            assert parents[0].ordinal == i

    def test_expand_token_with_zero_count_raises(self) -> None:
        """expand_token raises ValueError for count=0."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.IO_READ,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        with pytest.raises(ValueError, match="at least 1"):
            recorder.expand_token(
                parent_token_id=token.token_id,
                row_id=row.row_id,
                count=0,
                run_id=run.run_id,
                step_in_pipeline=1,
            )

    def test_expand_token_stores_step_in_pipeline(self) -> None:
        """expand_token stores step_in_pipeline on child tokens."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"list": [1, 2]},
        )
        parent = recorder.create_token(row_id=row.row_id)

        children, _expand_group_id = recorder.expand_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            count=2,
            run_id=run.run_id,
            step_in_pipeline=5,
        )

        # Verify step_in_pipeline stored
        for child in children:
            assert child.step_in_pipeline == 5
            # Verify retrieval via get_token
            retrieved = recorder.get_token(child.token_id)
            assert retrieved is not None
            assert retrieved.step_in_pipeline == 5

    def test_expand_token_with_single_child(self) -> None:
        """expand_token works with count=1."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="singleton",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)

        children, expand_group_id = recorder.expand_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            count=1,
            run_id=run.run_id,
            step_in_pipeline=1,
        )

        assert len(children) == 1
        assert children[0].expand_group_id is not None
        assert children[0].expand_group_id == expand_group_id

        parents = recorder.get_token_parents(children[0].token_id)
        assert len(parents) == 1
        assert parents[0].parent_token_id == parent.token_id
        assert parents[0].ordinal == 0

    def test_expand_token_preserves_expand_group_id_through_retrieval(self) -> None:
        """expand_group_id is preserved when retrieving tokens via get_token."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="1.0")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)

        children, _expand_group_id = recorder.expand_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            count=2,
            run_id=run.run_id,
            step_in_pipeline=3,
        )

        # Retrieve each child and verify expand_group_id matches
        for child in children:
            retrieved = recorder.get_token(child.token_id)
            assert retrieved is not None
            assert retrieved.expand_group_id == child.expand_group_id


class TestAtomicTokenOperations:
    """Tests verifying atomic behavior of fork_token and expand_token.

    These operations atomically create children AND record parent outcomes
    in a single transaction to eliminate crash windows.
    """

    def test_fork_token_records_parent_forked_outcome(self) -> None:
        """fork_token atomically records FORKED outcome on parent token.

        This is critical for crash recovery - if children are created but
        the parent outcome isn't recorded, recovery can't identify the fork.
        """

        from elspeth.contracts.enums import RowOutcome

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to two branches
        _children, fork_group_id = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
            run_id=run.run_id,
        )

        # Verify parent has FORKED outcome recorded atomically
        outcome = recorder.get_token_outcome(parent_token.token_id)
        assert outcome is not None, "Parent token should have FORKED outcome"
        assert outcome.outcome == RowOutcome.FORKED.value
        assert outcome.fork_group_id == fork_group_id
        assert outcome.is_terminal == 1

    def test_fork_token_stores_expected_branches_contract(self) -> None:
        """fork_token stores branch names in expected_branches_json for contract validation."""
        import json

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to three branches
        recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["alpha", "beta", "gamma"],
            run_id=run.run_id,
        )

        # Verify expected_branches_json is stored correctly
        outcome = recorder.get_token_outcome(parent_token.token_id)
        assert outcome is not None
        assert outcome.expected_branches_json is not None
        expected = json.loads(outcome.expected_branches_json)
        assert expected == ["alpha", "beta", "gamma"]

    def test_expand_token_records_parent_expanded_outcome(self) -> None:
        """expand_token atomically records EXPANDED outcome on parent token.

        By default, expand_token records the parent EXPANDED outcome in the
        same transaction as creating children, eliminating the crash window.
        """

        from elspeth.contracts.enums import RowOutcome

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Expand to 3 children (default: record_parent_outcome=True)
        _children, expand_group_id = recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            count=3,
            run_id=run.run_id,
            step_in_pipeline=2,
        )

        # Verify parent has EXPANDED outcome recorded atomically
        outcome = recorder.get_token_outcome(parent_token.token_id)
        assert outcome is not None, "Parent token should have EXPANDED outcome"
        assert outcome.outcome == RowOutcome.EXPANDED.value
        assert outcome.expand_group_id == expand_group_id
        assert outcome.is_terminal == 1

    def test_expand_token_stores_expected_count_contract(self) -> None:
        """expand_token stores count in expected_branches_json for contract validation."""
        import json

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="explode",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Expand to 5 children
        recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            count=5,
            run_id=run.run_id,
            step_in_pipeline=2,
        )

        # Verify expected_branches_json stores count
        outcome = recorder.get_token_outcome(parent_token.token_id)
        assert outcome is not None
        assert outcome.expected_branches_json is not None
        expected = json.loads(outcome.expected_branches_json)
        assert expected == {"count": 5}

    def test_expand_token_skips_parent_outcome_for_batch_aggregation(self) -> None:
        """expand_token with record_parent_outcome=False skips EXPANDED recording.

        Batch aggregation uses expand_token to create children but records
        CONSUMED_IN_BATCH separately for the parent (different semantics).
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="aggregator",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Expand with record_parent_outcome=False (batch aggregation pattern)
        children, _expand_group_id = recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            count=2,
            run_id=run.run_id,
            step_in_pipeline=2,
            record_parent_outcome=False,  # Don't record EXPANDED
        )

        # Children should be created
        assert len(children) == 2

        # But parent should NOT have an outcome yet
        outcome = recorder.get_token_outcome(parent_token.token_id)
        assert outcome is None, "Parent should not have outcome when record_parent_outcome=False"
