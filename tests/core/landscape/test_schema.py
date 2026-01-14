# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

from pathlib import Path

from sqlalchemy import inspect


class TestSchemaDefinition:
    """SQLAlchemy table definitions."""

    def test_runs_table_exists(self) -> None:
        from elspeth.core.landscape.schema import runs_table

        assert runs_table.name == "runs"
        assert "run_id" in [c.name for c in runs_table.columns]

    def test_nodes_table_exists(self) -> None:
        from elspeth.core.landscape.schema import nodes_table

        assert nodes_table.name == "nodes"
        assert "node_id" in [c.name for c in nodes_table.columns]

    def test_rows_table_exists(self) -> None:
        from elspeth.core.landscape.schema import rows_table

        assert rows_table.name == "rows"
        assert "row_id" in [c.name for c in rows_table.columns]

    def test_tokens_table_exists(self) -> None:
        from elspeth.core.landscape.schema import tokens_table

        assert tokens_table.name == "tokens"

    def test_node_states_table_exists(self) -> None:
        from elspeth.core.landscape.schema import node_states_table

        assert node_states_table.name == "node_states"


class TestSchemaCreation:
    """Creating tables in a database."""

    def test_create_all_tables(self, tmp_path: Path) -> None:
        from sqlalchemy import create_engine

        from elspeth.core.landscape.schema import metadata

        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")

        metadata.create_all(engine)

        # Verify tables exist
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "runs" in tables
        assert "nodes" in tables
        assert "rows" in tables
        assert "tokens" in tables
        assert "node_states" in tables


class TestPhase3ASchemaAdditions:
    """Tests for tables added in Phase 3A."""

    def test_routing_events_table_exists(self) -> None:
        from elspeth.core.landscape.schema import routing_events_table

        assert routing_events_table.name == "routing_events"
        columns = {c.name for c in routing_events_table.columns}
        assert "event_id" in columns
        assert "state_id" in columns
        assert "edge_id" in columns
        assert "routing_group_id" in columns

    def test_batches_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batches_table

        assert batches_table.name == "batches"
        columns = {c.name for c in batches_table.columns}
        assert "batch_id" in columns
        assert "aggregation_node_id" in columns
        assert "status" in columns

    def test_batch_members_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batch_members_table

        assert batch_members_table.name == "batch_members"

    def test_batch_outputs_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batch_outputs_table

        assert batch_outputs_table.name == "batch_outputs"

    def test_all_13_tables_exist(self) -> None:
        from elspeth.core.landscape.schema import metadata

        table_names = set(metadata.tables.keys())
        expected = {
            "runs", "nodes", "edges", "rows", "tokens", "token_parents",
            "node_states", "routing_events", "calls", "batches",
            "batch_members", "batch_outputs", "artifacts",
        }
        assert expected.issubset(table_names), f"Missing: {expected - table_names}"


class TestPhase3AModels:
    """Tests for model classes added in Phase 3A."""

    def test_routing_event_model(self) -> None:
        from elspeth.core.landscape.models import RoutingEvent

        event = RoutingEvent(
            event_id="evt1",
            state_id="state1",
            edge_id="edge1",
            routing_group_id="grp1",
            ordinal=0,
            mode="move",
            created_at=None,  # type: ignore[arg-type]  # Will be set in real use
        )
        assert event.event_id == "evt1"

    def test_batch_model(self) -> None:
        from elspeth.core.landscape.models import Batch

        batch = Batch(
            batch_id="batch1",
            run_id="run1",
            aggregation_node_id="node1",
            attempt=0,
            status="draft",
            created_at=None,  # type: ignore[arg-type]
        )
        assert batch.status == "draft"


class TestNodesDeterminismColumn:
    """Tests for determinism column in nodes table."""

    def test_nodes_table_has_determinism_column(self) -> None:
        from elspeth.core.landscape.schema import nodes_table

        columns = {c.name for c in nodes_table.columns}
        assert "determinism" in columns

    def test_node_model_has_determinism_field(self) -> None:
        from datetime import UTC, datetime

        from elspeth.core.landscape.models import Node

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="test_plugin",
            node_type="transform",
            plugin_version="1.0.0",
            determinism="deterministic",  # New field
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
        assert node.determinism == "deterministic"

    def test_determinism_values(self) -> None:
        """Verify valid determinism values match Determinism enum."""
        from elspeth.plugins.enums import Determinism

        valid_values = {d.value for d in Determinism}
        # Current enum values (not the granular architecture spec values)
        expected = {"deterministic", "seeded", "nondeterministic"}
        assert valid_values == expected
