"""Tests for DatabaseSink resume capability."""

import pytest

from elspeth.plugins.sinks.database_sink import DatabaseSink

# Strict schema for tests - DatabaseSink requires fixed columns
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int"]}


@pytest.fixture(autouse=True)
def allow_raw_secrets(monkeypatch):
    """Allow raw secrets for testing."""
    monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")


def test_database_sink_supports_resume():
    """DatabaseSink should declare supports_resume=True."""
    assert DatabaseSink.supports_resume is True


class TestDatabaseSinkResumeEndToEnd:
    """End-to-end tests for DatabaseSink resume capability.

    These tests verify actual database persistence across resume operations,
    not just internal state changes.
    """

    @pytest.fixture
    def db_url(self, tmp_path):
        """Create a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'resume_test.db'}"

    @pytest.fixture
    def ctx(self):
        """Create a plugin context with real landscape and operation records."""
        from tests.fixtures.factories import make_operation_context

        return make_operation_context(
            node_id="sink",
            plugin_name="database_sink",
            node_type="SINK",
            operation_type="sink_write",
        )

    def _get_all_rows(self, db_url: str, table_name: str) -> list[dict[str, object]]:
        """Helper to retrieve all rows from a table."""
        from sqlalchemy import MetaData, Table, create_engine, select

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))
        engine.dispose()
        # Convert to list of dicts for easier assertions
        return [dict(row._mapping) for row in rows]
