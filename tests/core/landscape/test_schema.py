# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

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

    def test_create_all_tables(self, tmp_path) -> None:
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
