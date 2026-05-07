# tests/integration/audit/test_recorder_artifacts.py
"""Tests for RecorderFactory artifact operations."""

from __future__ import annotations

from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryArtifacts:
    """Artifact registration and queries."""

    def test_register_artifact(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        artifact = factory.execution.register_artifact(
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
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        factory.execution.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="hash1",
            size_bytes=100,
        )
        factory.execution.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/b.csv",
            content_hash="hash2",
            size_bytes=200,
        )

        artifacts = factory.execution.get_artifacts(run.run_id)
        assert len(artifacts) == 2

    def test_register_artifact_with_idempotency_key(self) -> None:
        """register_artifact should accept and persist idempotency_key.

        Regression test for:
        docs/bugs/open/P2-2026-01-20-artifact-idempotency-key-column-ignored.md

        The idempotency_key allows retry deduplication - sinks can use a stable
        key (e.g., run_id:row_id:sink_name) to detect if an artifact was already
        written, enabling safe retries without duplicate outputs.
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Register artifact with idempotency key
        idem_key = f"{run.run_id}:{row.row_id}:csv_sink"
        artifact = factory.execution.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="abc123",
            size_bytes=1024,
            idempotency_key=idem_key,
        )

        assert artifact.idempotency_key == idem_key, (
            "register_artifact should return Artifact with idempotency_key set. "
            "See P2-2026-01-20-artifact-idempotency-key-column-ignored.md"
        )

        # Verify it's persisted and returned by get_artifacts
        artifacts = factory.execution.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].idempotency_key == idem_key, (
            "get_artifacts should return Artifact with idempotency_key populated. "
            "See P2-2026-01-20-artifact-idempotency-key-column-ignored.md"
        )

    def test_register_artifact_without_idempotency_key_returns_none(self) -> None:
        """register_artifact without idempotency_key should return None for that field.

        The idempotency_key is optional - not all sinks need deduplication support.
        When not provided, the field should be None (not missing or error).
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Register artifact WITHOUT idempotency key
        artifact = factory.execution.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert artifact.idempotency_key is None, "register_artifact without idempotency_key should return None for that field"

    def test_get_rows_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        for i in range(3):
            factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=i,
                data={"idx": i},
            )

        rows = factory.query.get_rows(run.run_id)
        assert len(rows) == 3
        assert rows[0].row_index == 0
        assert rows[2].row_index == 2

    def test_get_tokens_for_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )

        # Create initial token and fork
        parent = factory.data_flow.create_token(row_id=row.row_id)
        _children, _fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
            row_id=row.row_id,
            branches=["a", "b"],
        )

        tokens = factory.query.get_tokens(row.row_id)
        # Should have parent + 2 children
        assert len(tokens) == 3

    def test_get_node_states_for_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        node1 = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node2 = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=node1.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        # Create states at two nodes
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node1.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=node2.node_id,
            run_id=run.run_id,
            step_index=1,
            input_data={},
        )

        states = factory.query.get_node_states_for_token(token.token_id)
        assert len(states) == 2
        assert states[0].step_index == 0
        assert states[1].step_index == 1
