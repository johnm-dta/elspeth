"""Tests for RecorderFactory query methods."""

from __future__ import annotations

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryQueryMethods:
    """Additional query methods added in Task 9."""

    def test_get_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Retrieve by ID
        retrieved = factory.query.get_row(row.row_id)
        assert retrieved is not None
        assert retrieved.row_id == row.row_id
        assert retrieved.row_index == 0

    def test_get_row_not_found(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        result = factory.query.get_row("nonexistent")
        assert result is None

    def test_get_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
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
        token = factory.data_flow.create_token(row_id=row.row_id)

        # Retrieve by ID
        retrieved = factory.query.get_token(token.token_id)
        assert retrieved is not None
        assert retrieved.token_id == token.token_id

    def test_get_token_not_found(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        result = factory.query.get_token("nonexistent")
        assert result is None

    def test_get_token_parents_for_coalesced(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
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

        # Create parent token and fork
        parent = factory.data_flow.create_token(row_id=row.row_id)
        children, _fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
            row_id=row.row_id,
            branches=["a", "b"],
        )

        # Coalesce the children
        coalesced = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=c.token_id, run_id=run.run_id) for c in children],
            row_id=row.row_id,
        )

        # Get parents of coalesced token
        parents = factory.query.get_token_parents(coalesced.token_id)
        assert len(parents) == 2
        assert parents[0].ordinal == 0
        assert parents[1].ordinal == 1

    def test_get_token_children_for_consumed_parent(self) -> None:
        """Test forward lineage: find what a parent token merged into."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
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

        # Create parent token and fork into branches
        parent = factory.data_flow.create_token(row_id=row.row_id)
        forked_children, _fork_group_id = factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=parent.token_id, run_id=run.run_id),
            row_id=row.row_id,
            branches=["a", "b"],
        )

        # Coalesce the forked children back together
        coalesced = factory.data_flow.coalesce_tokens(
            parent_refs=[TokenRef(token_id=c.token_id, run_id=run.run_id) for c in forked_children],
            row_id=row.row_id,
        )

        # Forward lineage: given a consumed parent, find what it merged into
        # forked_children[0] was consumed in the coalesce → should find coalesced token
        children_of_first = factory.query.get_token_children(forked_children[0].token_id)
        assert len(children_of_first) == 1
        assert children_of_first[0].token_id == coalesced.token_id
        assert children_of_first[0].parent_token_id == forked_children[0].token_id

        # The second forked child should also point to the same coalesced token
        children_of_second = factory.query.get_token_children(forked_children[1].token_id)
        assert len(children_of_second) == 1
        assert children_of_second[0].token_id == coalesced.token_id

    def test_get_token_children_empty_for_terminal(self) -> None:
        """Tokens that never became parents return empty list."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
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
        token = factory.data_flow.create_token(row_id=row.row_id)

        # A fresh token has no children
        children = factory.query.get_token_children(token.token_id)
        assert children == []

    def test_get_routing_events(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        gate = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        edge = factory.data_flow.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="output",
            mode=RoutingMode.MOVE,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = factory.data_flow.create_token(row_id=row.row_id)
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )

        # Record routing event (using new API with auto-generated routing_group_id)
        factory.execution.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode=RoutingMode.MOVE,
        )

        # Query routing events
        events = factory.query.get_routing_events(state.state_id)
        assert len(events) == 1
        assert events[0].mode == "move"
        assert events[0].edge_id == edge.edge_id

    def test_get_row_data_without_payload_ref(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.row_data import RowDataState

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)  # No payload store
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Without payload_ref, should return NEVER_STORED
        result = factory.query.get_row_data(row.row_id)
        assert result.state == RowDataState.NEVER_STORED
        assert result.data is None
