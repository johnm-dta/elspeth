# tests/core/landscape/test_recorder.py
"""Tests for LandscapeRecorder."""


class TestLandscapeRecorderRuns:
    """Run lifecycle management."""

    def test_begin_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={"source": "test.csv"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert run.run_id is not None
        assert run.status == "running"
        assert run.started_at is not None

    def test_complete_run_success(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="completed")

        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_complete_run_failed(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="failed")

        assert completed.status == "failed"

    def test_get_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"key": "value"}, canonical_version="v1")
        retrieved = recorder.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id


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
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.plugins.enums import NodeType

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
            mode="move",
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
        recorder.complete_node_state(state1.state_id, status="failed", error={})

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
            mode="move",
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
            routing_group_id="group1",
            ordinal=0,
            mode="move",
            reason={"rule": "value > 1000", "result": True},
        )

        assert event.event_id is not None
        assert event.edge_id == edge.edge_id
        assert event.mode == "move"

    def test_record_multiple_routing_events(self) -> None:
        """Test recording fork to multiple destinations."""
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
            mode="copy",
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_b.node_id,
            label="path_b",
            mode="copy",
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

        # Fork to both paths - same routing_group_id
        group_id = "fork_group_123"
        event_a = recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge_a.edge_id,
            routing_group_id=group_id,
            ordinal=0,
            mode="copy",
        )
        event_b = recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge_b.edge_id,
            routing_group_id=group_id,
            ordinal=1,
            mode="copy",
        )

        assert event_a.routing_group_id == event_b.routing_group_id
        assert event_a.ordinal == 0
        assert event_b.ordinal == 1


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
        executing = recorder.update_batch_status(
            batch_id=batch.batch_id,
            status="executing",
        )
        assert executing.status == "executing"

        # Complete
        completed = recorder.complete_batch(
            batch_id=batch.batch_id,
            status="completed",
            trigger_reason="count=3",
        )
        assert completed.status == "completed"


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
