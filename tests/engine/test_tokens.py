# tests/engine/test_tokens.py
"""Tests for TokenManager."""

from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTokenManager:
    """High-level token management."""

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None
        assert token_info.row_data == {"value": 42}

    def test_fork_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # step_in_pipeline is required - Orchestrator/RowProcessor is the authority
        children = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            step_in_pipeline=1,  # Fork happens at step 1
        )

        assert len(children) == 2
        assert children[0].branch_name == "stats"
        assert children[1].branch_name == "classifier"
        # Children inherit row_data
        assert children[0].row_data == {"value": 42}

    def test_update_row_data(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"x": 1},
        )

        updated = manager.update_row_data(
            token_info,
            new_data={"x": 1, "y": 2},
        )

        assert updated.row_data == {"x": 1, "y": 2}
        assert updated.token_id == token_info.token_id  # Same token


class TestTokenManagerCoalesce:
    """Test token coalescing (join operations)."""

    def test_coalesce_tokens(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        # Create initial token and fork it
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        children = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            step_in_pipeline=1,
        )

        # Update children with branch-specific data
        stats_token = manager.update_row_data(
            children[0],
            new_data={"value": 42, "mean": 10.5},
        )
        classifier_token = manager.update_row_data(
            children[1],
            new_data={"value": 42, "label": "A"},
        )

        # Coalesce the branches
        merged = manager.coalesce_tokens(
            parents=[stats_token, classifier_token],
            merged_data={"value": 42, "mean": 10.5, "label": "A"},
            step_in_pipeline=3,
        )

        assert merged.token_id is not None
        assert merged.row_id == initial.row_id
        assert merged.row_data == {"value": 42, "mean": 10.5, "label": "A"}


class TestTokenManagerForkIsolation:
    """Test that forked tokens have isolated row_data (no shared mutable objects)."""

    def test_fork_nested_data_isolation(self) -> None:
        """Forked children must not share nested mutable objects.

        Bug: P2-2026-01-20-forked-token-row-data-shallow-copy-leaks-nested-mutations
        When row_data contains nested dicts/lists, shallow copy causes siblings
        to share nested objects. Mutating one affects all.
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        # Create token with NESTED data (common from JSON sources)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"payload": {"x": 1, "y": 2}, "items": [1, 2, 3]},
        )

        # Fork to two branches
        children = manager.fork_token(
            parent_token=initial,
            branches=["branch_a", "branch_b"],
            step_in_pipeline=1,
        )

        child_a = children[0]
        child_b = children[1]

        # Mutate nested data in child_a
        child_a.row_data["payload"]["x"] = 999
        child_a.row_data["items"].append(4)

        # Bug: child_b should NOT be affected, but shallow copy means it is
        assert child_b.row_data["payload"]["x"] == 1, "Nested dict mutation leaked to sibling!"
        assert child_b.row_data["items"] == [1, 2, 3], "Nested list mutation leaked to sibling!"

    def test_fork_with_custom_nested_data_isolation(self) -> None:
        """Custom row_data in fork should also be deep copied."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 1},
        )

        # Fork with custom nested row_data
        custom_data = {"nested": {"key": "original"}}
        children = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            step_in_pipeline=1,
            row_data=custom_data,
        )

        # Mutate one child's nested data
        children[0].row_data["nested"]["key"] = "modified"

        # Siblings should be isolated
        assert children[1].row_data["nested"]["key"] == "original"


class TestTokenManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_fork_with_custom_row_data(self) -> None:
        """Fork can override parent row_data."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # Fork with custom row_data
        children = manager.fork_token(
            parent_token=initial,
            branches=["branch_a"],
            step_in_pipeline=1,
            row_data={"value": 42, "forked": True},
        )

        assert children[0].row_data == {"value": 42, "forked": True}

    def test_update_preserves_branch_name(self) -> None:
        """update_row_data preserves branch_name."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"x": 1},
        )

        children = manager.fork_token(
            parent_token=initial,
            branches=["my_branch"],
            step_in_pipeline=1,
        )

        updated = manager.update_row_data(
            children[0],
            new_data={"x": 1, "y": 2},
        )

        assert updated.branch_name == "my_branch"

    def test_multiple_rows_different_tokens(self) -> None:
        """Each source row gets its own row_id and token_id."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        token1 = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"id": 1},
        )
        token2 = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=1,
            row_data={"id": 2},
        )

        assert token1.row_id != token2.row_id
        assert token1.token_id != token2.token_id


class TestTokenManagerStepInPipeline:
    """Test that step_in_pipeline flows through to audit trail."""

    def test_fork_stores_step_in_pipeline(self) -> None:
        """TokenManager.fork_token passes step_in_pipeline to recorder."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # Fork with step_in_pipeline=2
        children = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            step_in_pipeline=2,
        )

        # Verify step_in_pipeline is stored in audit trail
        token_a = recorder.get_token(children[0].token_id)
        token_b = recorder.get_token(children[1].token_id)
        assert token_a is not None
        assert token_b is not None
        assert token_a.step_in_pipeline == 2
        assert token_b.step_in_pipeline == 2

    def test_coalesce_stores_step_in_pipeline(self) -> None:
        """TokenManager.coalesce_tokens passes step_in_pipeline to recorder."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # Fork and then coalesce
        children = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            step_in_pipeline=1,
        )

        merged = manager.coalesce_tokens(
            parents=children,
            merged_data={"value": 42, "merged": True},
            step_in_pipeline=3,
        )

        # Verify step_in_pipeline is stored in audit trail
        merged_token = recorder.get_token(merged.token_id)
        assert merged_token is not None
        assert merged_token.step_in_pipeline == 3


class TestTokenManagerExpand:
    """Test token expansion (deaggregation: 1 input -> N outputs)."""

    def test_expand_token_creates_children(self) -> None:
        """expand_token creates child tokens for each expanded row."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        # Create initial token
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"original": "data"},
        )

        expanded_rows = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ]

        # Act
        children = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            step_in_pipeline=2,
        )

        # Assert: correct number of children
        assert len(children) == 3

        # All children share same row_id (same source row)
        for child in children:
            assert child.row_id == parent.row_id
            assert child.token_id != parent.token_id

        # Each child has its expanded row data
        assert children[0].row_data == {"id": 1, "value": "a"}
        assert children[1].row_data == {"id": 2, "value": "b"}
        assert children[2].row_data == {"id": 3, "value": "c"}

        # Verify parent relationships in database
        for i, child in enumerate(children):
            parents = recorder.get_token_parents(child.token_id)
            assert len(parents) == 1
            assert parents[0].parent_token_id == parent.token_id
            assert parents[0].ordinal == i

    def test_expand_token_inherits_branch_name(self) -> None:
        """expand_token children inherit parent's branch_name."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)

        # Create initial token and fork to get a branch_name
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={},
        )

        # Fork to get a token with branch_name
        forked = manager.fork_token(
            parent_token=initial,
            branches=["stats_branch"],
            step_in_pipeline=1,
        )
        parent = forked[0]  # Has branch_name="stats_branch"

        children = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            step_in_pipeline=2,
        )

        # Children inherit branch_name
        assert all(c.branch_name == "stats_branch" for c in children)

    def test_expand_token_stores_step_in_pipeline(self) -> None:
        """expand_token passes step_in_pipeline to recorder."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder)
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"x": 1},
        )

        children = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            step_in_pipeline=5,
        )

        # Verify step_in_pipeline is stored in audit trail
        for child in children:
            db_token = recorder.get_token(child.token_id)
            assert db_token is not None
            assert db_token.step_in_pipeline == 5
