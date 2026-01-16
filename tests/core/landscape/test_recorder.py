# tests/core/landscape/test_recorder.py
"""Tests for LandscapeRecorder."""

from elspeth.contracts import RoutingMode


class TestLandscapeRecorderRuns:
    """Run lifecycle management."""

    def test_begin_run(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={"source": "test.csv"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert run.run_id is not None
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None

    def test_complete_run_success(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="completed")

        assert completed.status == RunStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_run_failed(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="failed")

        assert completed.status == RunStatus.FAILED

    def test_get_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"key": "value"}, canonical_version="v1")
        retrieved = recorder.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id


class TestLandscapeRecorderRunStatusValidation:
    """Run status validation against RunStatus enum."""

    def test_begin_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={},
            canonical_version="v1",
            status=RunStatus.RUNNING,
        )

        assert run.status == RunStatus.RUNNING

    def test_begin_run_with_valid_string_status(self) -> None:
        """Test that valid string status is accepted and coerced."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={},
            canonical_version="v1",
            status="running",
        )

        assert run.status == RunStatus.RUNNING

    def test_begin_run_with_invalid_status_raises(self) -> None:
        """Test that invalid status string raises ValueError."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        with pytest.raises(ValueError, match="runnign"):  # Note typo
            recorder.begin_run(
                config={},
                canonical_version="v1",
                status="runnign",  # Typo! Should fail fast
            )

    def test_complete_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted for complete_run."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED

    def test_complete_run_with_valid_string_status(self) -> None:
        """Test that valid string status is accepted for complete_run."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="failed")

        assert completed.status == RunStatus.FAILED

    def test_complete_run_with_invalid_status_raises(self) -> None:
        """Test that invalid status string raises ValueError for complete_run."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        with pytest.raises(ValueError, match="compleed"):  # Note typo
            recorder.complete_run(run.run_id, status="compleed")  # Typo!

    def test_list_runs_with_enum_status_filter(self) -> None:
        """Test that RunStatus enum is accepted for list_runs filter."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create runs with different statuses
        run1 = recorder.begin_run(config={"n": 1}, canonical_version="v1")
        run2 = recorder.begin_run(config={"n": 2}, canonical_version="v1")
        recorder.complete_run(run2.run_id, status=RunStatus.COMPLETED)

        # Filter by enum
        running_runs = recorder.list_runs(status=RunStatus.RUNNING)
        assert len(running_runs) == 1
        assert running_runs[0].run_id == run1.run_id

    def test_list_runs_with_valid_string_status_filter(self) -> None:
        """Test that valid string status is accepted for list_runs filter."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create runs
        recorder.begin_run(config={"n": 1}, canonical_version="v1")  # running
        run2 = recorder.begin_run(config={"n": 2}, canonical_version="v1")
        recorder.complete_run(run2.run_id, status="completed")

        # Filter by string
        completed_runs = recorder.list_runs(status="completed")
        assert len(completed_runs) == 1
        assert completed_runs[0].run_id == run2.run_id

    def test_list_runs_with_invalid_status_filter_raises(self) -> None:
        """Test that invalid status string raises ValueError for list_runs."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        with pytest.raises(ValueError, match="completd"):  # Note typo
            recorder.list_runs(status="completd")  # Typo!


class TestLandscapeRecorderNodes:
    """Node and edge registration."""

    def test_register_node(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            sequence=0,
        )

        assert node.node_id is not None
        assert node.plugin_name == "csv_source"
        assert node.node_type == "source"

    def test_register_node_with_enum(self) -> None:
        """Test that NodeType enum is accepted and coerced."""
        from elspeth.contracts import NodeType
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Both enum and string should work
        node_from_enum = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform1",
            node_type=NodeType.TRANSFORM,  # Enum
            plugin_version="1.0.0",
            config={},
        )
        node_from_str = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform2",
            node_type="transform",  # String
            plugin_version="1.0.0",
            config={},
        )

        # Both should store the same string value
        assert node_from_enum.node_type == "transform"
        assert node_from_str.node_type == "transform"

    def test_register_node_invalid_type_raises(self) -> None:
        """Test that invalid node_type string raises ValueError."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        with pytest.raises(ValueError, match="transfom"):  # Note typo
            recorder.register_node(
                run_id=run.run_id,
                plugin_name="bad",
                node_type="transfom",  # Typo! Should fail fast
                plugin_version="1.0.0",
                config={},
            )

    def test_register_edge(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=source.node_id,
            to_node_id=transform.node_id,
            label="continue",
            mode=RoutingMode.MOVE,
        )

        assert edge.edge_id is not None
        assert edge.label == "continue"

    def test_get_nodes_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        nodes = recorder.get_nodes(run.run_id)
        assert len(nodes) == 2


class TestLandscapeRecorderTokens:
    """Row and token management."""

    def test_create_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )

        assert row.row_id is not None
        assert row.row_index == 0
        assert row.source_data_hash is not None

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to two branches
        child_tokens = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
        )

        assert len(child_tokens) == 2
        assert child_tokens[0].branch_name == "stats"
        assert child_tokens[1].branch_name == "classifier"
        # All children share same fork_group_id
        assert child_tokens[0].fork_group_id == child_tokens[1].fork_group_id

    def test_coalesce_tokens(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)
        children = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
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
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork with step_in_pipeline
        child_tokens = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
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

    def test_coalesce_tokens_with_step_in_pipeline(self) -> None:
        """Coalesce stores step_in_pipeline in tokens table."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)
        children = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
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


class TestLandscapeRecorderNodeStates:
    """Node state recording (what happened at each node)."""

    def test_begin_node_state(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=source.node_id,
            step_index=0,
            input_data={"value": 42},
        )

        assert state.state_id is not None
        assert state.status == "open"
        assert state.input_hash is not None

    def test_complete_node_state_success(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"x": 1},
        )

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data={"x": 1, "y": 2},
            duration_ms=10.5,
        )

        assert completed.status == "completed"
        assert completed.output_hash is not None
        assert completed.duration_ms == 10.5
        assert completed.completed_at is not None

    def test_complete_node_state_failed(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
        )

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status="failed",
            error={"message": "Validation failed", "code": "E001"},
            duration_ms=5.0,
        )

        assert completed.status == "failed"
        assert completed.error_json is not None
        assert "Validation failed" in completed.error_json

    def test_retry_increments_attempt(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # First attempt fails
        state1 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
            attempt=0,
        )
        recorder.complete_node_state(
            state1.state_id, status="failed", error={}, duration_ms=1.0
        )

        # Second attempt
        state2 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
            attempt=1,
        )

        assert state2.attempt == 1


class TestLandscapeRecorderRouting:
    """Routing event recording (gate decisions)."""

    def test_record_routing_event(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="high_value",
            mode=RoutingMode.MOVE,
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )

        event = recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode=RoutingMode.MOVE,
            reason={"rule": "value > 1000", "result": True},
        )

        assert event.event_id is not None
        assert event.routing_group_id is not None  # Auto-generated
        assert event.edge_id == edge.edge_id
        assert event.mode == "move"

    def test_record_multiple_routing_events(self) -> None:
        """Test recording fork to multiple destinations."""
        from elspeth.contracts import RoutingMode, RoutingSpec
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink_a",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        sink_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink_b",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_a.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_b.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )

        # Fork to both paths using batch method
        events = recorder.record_routing_events(
            state_id=state.state_id,
            routes=[
                RoutingSpec(edge_id=edge_a.edge_id, mode=RoutingMode.COPY),
                RoutingSpec(edge_id=edge_b.edge_id, mode=RoutingMode.COPY),
            ],
            reason={"action": "fork"},
        )

        assert len(events) == 2
        # All events share the same routing_group_id
        assert events[0].routing_group_id == events[1].routing_group_id
        assert events[0].ordinal == 0
        assert events[1].ordinal == 1


class TestLandscapeRecorderBatches:
    """Batch management for aggregation."""

    def test_create_batch(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        assert batch.batch_id is not None
        assert batch.status == "draft"
        assert batch.attempt == 0

    def test_add_batch_member(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
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

    def test_complete_batch(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        completed = recorder.complete_batch(
            batch_id=batch.batch_id,
            status="completed",
            trigger_reason="count=10",
        )

        assert completed.status == "completed"
        assert completed.trigger_reason == "count=10"
        assert completed.completed_at is not None

    def test_batch_lifecycle(self) -> None:
        """Test full batch lifecycle: draft -> executing -> completed."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
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
            status="executing",
        )
        executing = recorder.get_batch(batch.batch_id)
        assert executing is not None
        assert executing.status == "executing"

        # Complete with trigger_reason
        recorder.update_batch_status(
            batch_id=batch.batch_id,
            status="completed",
            trigger_reason="count=3",
        )
        completed = recorder.get_batch(batch.batch_id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.trigger_reason == "count=3"
        assert completed.completed_at is not None

    def test_get_batches_by_status(self) -> None:
        """For crash recovery - find incomplete batches."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        batch1 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        batch2 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.update_batch_status(batch2.batch_id, "completed")

        # Get only draft batches
        drafts = recorder.get_batches(run.run_id, status="draft")
        assert len(drafts) == 1
        assert drafts[0].batch_id == batch1.batch_id


class TestLandscapeRecorderArtifacts:
    """Artifact registration and queries."""

    def test_register_artifact(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            step_index=0,
            input_data={},
        )

        artifact = recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert artifact.artifact_id is not None
        assert artifact.path_or_uri == "/output/result.csv"

    def test_get_artifacts_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            step_index=0,
            input_data={},
        )

        recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="hash1",
            size_bytes=100,
        )
        recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/b.csv",
            content_hash="hash2",
            size_bytes=200,
        )

        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 2

    def test_get_rows_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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

        for i in range(3):
            recorder.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=i,
                data={"idx": i},
            )

        rows = recorder.get_rows(run.run_id)
        assert len(rows) == 3
        assert rows[0].row_index == 0
        assert rows[2].row_index == 2

    def test_get_tokens_for_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

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
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )

        # Create initial token and fork
        parent = recorder.create_token(row_id=row.row_id)
        _children = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
        )

        tokens = recorder.get_tokens(row.row_id)
        # Should have parent + 2 children
        assert len(tokens) == 3

    def test_get_node_states_for_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        node2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node1.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create states at two nodes
        recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            step_index=0,
            input_data={},
        )
        recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node2.node_id,
            step_index=1,
            input_data={},
        )

        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 2
        assert states[0].step_index == 0
        assert states[1].step_index == 1


class TestLandscapeRecorderEdges:
    """Edge query methods."""

    def test_get_edges_returns_all_edges_for_run(self) -> None:
        """get_edges should return all edges registered for a run."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink_1",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )

        # Register edge
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source_1",
            to_node_id="sink_1",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Query edges
        edges = recorder.get_edges(run.run_id)

        assert len(edges) == 1
        assert edges[0].edge_id == edge.edge_id
        assert edges[0].from_node_id == "source_1"
        assert edges[0].to_node_id == "sink_1"
        assert edges[0].default_mode == "move"

    def test_get_edges_returns_empty_list_for_run_with_no_edges(self) -> None:
        """get_edges should return empty list when no edges exist."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        edges = recorder.get_edges(run.run_id)

        assert edges == []

    def test_get_edges_returns_multiple_edges(self) -> None:
        """get_edges should return all edges when multiple exist."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="gate",
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink_high",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink_low",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )

        # Register edges
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source",
            to_node_id="gate",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink_high",
            label="high",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink_low",
            label="low",
            mode=RoutingMode.MOVE,
        )

        # Query edges
        edges = recorder.get_edges(run.run_id)

        assert len(edges) == 3


class TestLandscapeRecorderQueryMethods:
    """Additional query methods added in Task 9."""

    def test_get_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Retrieve by ID
        retrieved = recorder.get_row(row.row_id)
        assert retrieved is not None
        assert retrieved.row_id == row.row_id
        assert retrieved.row_index == 0

    def test_get_row_not_found(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = recorder.get_row("nonexistent")
        assert result is None

    def test_get_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Retrieve by ID
        retrieved = recorder.get_token(token.token_id)
        assert retrieved is not None
        assert retrieved.token_id == token.token_id

    def test_get_token_not_found(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = recorder.get_token("nonexistent")
        assert result is None

    def test_get_token_parents_for_coalesced(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )

        # Create parent token and fork
        parent = recorder.create_token(row_id=row.row_id)
        children = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
        )

        # Coalesce the children
        coalesced = recorder.coalesce_tokens(
            parent_token_ids=[c.token_id for c in children],
            row_id=row.row_id,
        )

        # Get parents of coalesced token
        parents = recorder.get_token_parents(coalesced.token_id)
        assert len(parents) == 2
        assert parents[0].ordinal == 0
        assert parents[1].ordinal == 1

    def test_get_routing_events(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="output",
            mode=RoutingMode.MOVE,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )

        # Record routing event (using new API with auto-generated routing_group_id)
        recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode=RoutingMode.MOVE,
        )

        # Query routing events
        events = recorder.get_routing_events(state.state_id)
        assert len(events) == 1
        assert events[0].mode == "move"
        assert events[0].edge_id == edge.edge_id

    def test_get_row_data_without_payload_ref(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.row_data import RowDataState

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload store
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Without payload_ref, should return NEVER_STORED
        result = recorder.get_row_data(row.row_id)
        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None


class TestExplainGracefulDegradation:
    """Tests for explain_row() when payloads are unavailable."""

    def test_explain_with_missing_row_payload(self, tmp_path) -> None:
        """explain_row() succeeds even when row payload is purged."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store row data in payload store
        row_data = {"name": "test", "value": 42}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Purge the payload (simulate retention policy)
        payload_store.delete(payload_ref)

        # explain_row should still work
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Payload unavailable
        assert lineage.payload_available is False

    def test_explain_reports_payload_status(self, tmp_path) -> None:
        """explain_row() explicitly reports payload availability."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store row data in payload store
        row_data = {"name": "test"}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Purge the payload
        payload_store.delete(payload_ref)

        # Check payload_available attribute
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.payload_available is False

    def test_explain_with_available_payload(self, tmp_path) -> None:
        """explain_row() returns payload when available."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store row data in payload store
        row_data = {"name": "test", "value": 123}
        payload_ref = payload_store.store(json.dumps(row_data).encode())

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Payload NOT purged
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data is not None  # Payload available
        assert lineage.source_data == row_data
        assert lineage.payload_available is True

    def test_explain_row_not_found(self) -> None:
        """explain_row() returns None when row doesn't exist."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id="nonexistent",
        )

        assert lineage is None

    def test_explain_row_without_payload_store(self) -> None:
        """explain_row() works when no payload store is configured."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload store

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload store
        assert lineage.payload_available is False

    def test_explain_row_with_no_payload_ref(self, tmp_path) -> None:
        """explain_row() handles rows created without payload_ref."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Create row without payload_ref
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
            # No payload_ref provided
        )

        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload_ref
        assert lineage.payload_available is False

    def test_explain_row_with_corrupted_payload(self, tmp_path) -> None:
        """explain_row() handles corrupted payload (invalid JSON) gracefully."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Store corrupted (non-JSON) data directly to payload store
        corrupted_data = b"this is not valid json {{{{"
        payload_ref = payload_store.store(corrupted_data)

        # Create row with the corrupted payload ref
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},  # Valid data for hash
            payload_ref=payload_ref,
        )

        # explain_row should handle JSONDecodeError gracefully
        lineage = recorder.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Corrupted payload not returned
        assert lineage.payload_available is False  # Reports as unavailable

    def test_explain_row_rejects_run_id_mismatch(self, tmp_path) -> None:
        """explain_row() returns None when row belongs to different run."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=payload_store)

        # Create two runs
        run1 = recorder.begin_run(config={}, canonical_version="v1")
        run2 = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run1.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        # Create row in run1
        row_data = {"name": "test"}
        payload_ref = payload_store.store(json.dumps(row_data).encode())
        row = recorder.create_row(
            run_id=run1.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
            payload_ref=payload_ref,
        )

        # Try to explain using run2's ID - should return None
        lineage = recorder.explain_row(
            run_id=run2.run_id,  # Wrong run!
            row_id=row.row_id,
        )

        assert lineage is None

        # Same row with correct run_id should work
        lineage_correct = recorder.explain_row(
            run_id=run1.run_id,
            row_id=row.row_id,
        )

        assert lineage_correct is not None
        assert lineage_correct.row_id == row.row_id


class TestExportStatusEnum:
    """Verify set_export_status uses enum validation."""

    def test_set_export_status_accepts_enum(self) -> None:
        """set_export_status accepts ExportStatus enum."""

        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Should accept enum without error
        recorder.set_export_status(run.run_id, ExportStatus.COMPLETED)

        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        # Value stored correctly
        assert (
            updated_run.export_status == "completed"
            or updated_run.export_status == ExportStatus.COMPLETED
        )

    def test_set_export_status_rejects_invalid_string(self) -> None:
        """set_export_status rejects invalid status strings."""
        import pytest

        from elspeth.core.landscape import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Should raise ValueError for invalid status
        with pytest.raises(ValueError, match="not a valid ExportStatus"):
            recorder.set_export_status(run.run_id, "invalid_status")


class TestReproducibilityGradeComputation:
    """Tests for reproducibility grade computation based on node determinism values."""

    def test_pure_pipeline_gets_full_reproducible(self) -> None:
        """Pipeline with only deterministic/seeded nodes gets FULL_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # All deterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type="transform",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="seeded_sampler",
            node_type="transform",
            plugin_version="1.0",
            config={},
            determinism=Determinism.SEEDED,  # seeded counts as reproducible
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_external_calls_gets_replay_reproducible(self) -> None:
        """Pipeline with nondeterministic nodes gets REPLAY_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Mix of deterministic and nondeterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_classifier",
            node_type="transform",
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,  # LLM call
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_finalize_run_sets_grade(self) -> None:
        """finalize_run() computes grade and completes the run."""
        from elspeth.contracts import Determinism, RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register deterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )

        completed_run = recorder.finalize_run(run.run_id, status="completed")

        assert completed_run.status == RunStatus.COMPLETED
        assert completed_run.completed_at is not None
        assert (
            completed_run.reproducibility_grade
            == ReproducibilityGrade.FULL_REPRODUCIBLE.value
        )

    def test_grade_degrades_after_purge(self) -> None:
        """REPLAY_REPRODUCIBLE degrades to ATTRIBUTABLE_ONLY after purge."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
        )

        # Finalize with REPLAY_REPRODUCIBLE grade
        completed_run = recorder.finalize_run(run.run_id, status="completed")
        assert (
            completed_run.reproducibility_grade
            == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value
        )

        # Simulate purge - grade should degrade
        update_grade_after_purge(db, run.run_id)

        # Check grade was degraded
        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        assert (
            updated_run.reproducibility_grade
            == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
        )

    def test_full_reproducible_unchanged_after_purge(self) -> None:
        """FULL_REPRODUCIBLE remains unchanged after purge (payloads not needed for replay)."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Deterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
        )

        # Finalize with FULL_REPRODUCIBLE grade
        completed_run = recorder.finalize_run(run.run_id, status="completed")
        assert (
            completed_run.reproducibility_grade
            == ReproducibilityGrade.FULL_REPRODUCIBLE.value
        )

        # Simulate purge - grade should NOT degrade
        update_grade_after_purge(db, run.run_id)

        # Check grade unchanged
        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        assert (
            updated_run.reproducibility_grade
            == ReproducibilityGrade.FULL_REPRODUCIBLE.value
        )

    def test_compute_grade_empty_pipeline(self) -> None:
        """Empty pipeline (no nodes) gets FULL_REPRODUCIBLE."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # No nodes registered
        grade = recorder.compute_reproducibility_grade(run.run_id)

        # Empty pipeline is trivially reproducible
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_update_grade_after_purge_nonexistent_run(self) -> None:
        """update_grade_after_purge() silently handles nonexistent run."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.reproducibility import update_grade_after_purge

        db = LandscapeDB.in_memory()

        # Should not raise - silently returns for nonexistent run
        update_grade_after_purge(db, "nonexistent_run_id")

    def test_attributable_only_unchanged_after_purge(self) -> None:
        """ATTRIBUTABLE_ONLY remains unchanged after purge (already at lowest grade)."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
        )

        # Finalize and degrade to ATTRIBUTABLE_ONLY
        recorder.finalize_run(run.run_id, status="completed")
        update_grade_after_purge(db, run.run_id)

        # Verify it's ATTRIBUTABLE_ONLY
        run_after_first_purge = recorder.get_run(run.run_id)
        assert run_after_first_purge is not None
        assert (
            run_after_first_purge.reproducibility_grade
            == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
        )

        # Call purge again - should remain ATTRIBUTABLE_ONLY
        update_grade_after_purge(db, run.run_id)

        run_after_second_purge = recorder.get_run(run.run_id)
        assert run_after_second_purge is not None
        assert (
            run_after_second_purge.reproducibility_grade
            == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
        )

    def test_default_determinism_counts_as_deterministic(self) -> None:
        """Nodes registered without explicit determinism default to DETERMINISTIC."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes WITHOUT specifying determinism - should default to DETERMINISTIC
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0",
            config={},
            # determinism not specified - should default to DETERMINISTIC
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type="transform",
            plugin_version="1.0",
            config={},
            # determinism not specified - should default to DETERMINISTIC
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        # Since defaults are DETERMINISTIC, should get FULL_REPRODUCIBLE
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE
