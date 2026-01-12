"""Tests for database sink plugin."""

from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine, select

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestDatabaseSink:
    """Tests for DatabaseSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def db_url(self, tmp_path: Path) -> str:
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_implements_protocol(self) -> None:
        """DatabaseSink implements SinkProtocol."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """DatabaseSink has name and input_schema."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        assert DatabaseSink.name == "database"
        assert hasattr(DatabaseSink, "input_schema")

    def test_write_creates_table(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        # Verify data was written
        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 2
        # SQLite returns tuples; check by position or use dict access
        assert rows[0][1] == "alice"  # name column

    def test_batch_insert(self, db_url: str, ctx: PluginContext) -> None:
        """Rows are batched for efficiency."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink(
            {"url": db_url, "table": "output", "batch_size": 2}
        )

        # Write 5 rows with batch size 2
        for i in range(5):
            sink.write({"id": i, "value": f"val{i}"}, ctx)

        sink.flush()  # Flush remaining
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 5

    def test_close_is_idempotent(self, db_url: str, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        sink.write({"id": 1}, ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})

        sink.write({"col": "value"}, ctx)
        sink.flush()
        # Can't verify in-memory after close, but should not raise
        sink.close()
