# tests/engine/test_tokens.py
"""Tests for TokenManager."""


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
