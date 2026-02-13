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
        """Create a minimal plugin context."""
        from elspeth.contracts.plugin_context import PluginContext

        return PluginContext(run_id="test-run", config={})

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

    def test_resume_appends_to_existing_table(self, db_url: str, ctx) -> None:
        """Resume mode appends to existing table instead of replacing it.

        Scenario:
        1. Create sink with replace mode, write initial rows
        2. Close sink
        3. Create new sink with configure_for_resume(), write more rows
        4. Verify ALL rows present in database
        """
        # Initial run: replace mode writes first batch
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink1.write([{"id": 1}, {"id": 2}], ctx)
        sink1.close()

        # Verify first batch written
        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {1, 2}

        # Resume run: configure for resume and write more rows
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",  # Will be changed by configure_for_resume
            }
        )
        sink2.configure_for_resume()  # Switch to append mode
        sink2.write([{"id": 3}, {"id": 4}], ctx)
        sink2.close()

        # Verify ALL rows present (old + new)
        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 4
        assert {r["id"] for r in rows} == {1, 2, 3, 4}

    def test_resume_without_configure_replaces_table(self, db_url: str, ctx) -> None:
        """Without configure_for_resume, replace mode drops existing data.

        This verifies that configure_for_resume is necessary for resume behavior.
        """
        # Initial run
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink1.write([{"id": 1}, {"id": 2}], ctx)
        sink1.close()

        # Second run WITHOUT configure_for_resume - should replace
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        # NO configure_for_resume call
        sink2.write([{"id": 3}], ctx)
        sink2.close()

        # Only new rows should exist (old rows replaced)
        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 1
        assert rows[0]["id"] == 3

    def test_resume_multiple_batches(self, db_url: str, ctx) -> None:
        """Resume mode can write multiple batches across multiple sink instances."""
        # Initial run
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.close()

        # Resume run 1
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink2.configure_for_resume()
        sink2.write([{"id": 2}], ctx)
        sink2.write([{"id": 3}], ctx)  # Multiple writes in same instance
        sink2.close()

        # Resume run 2
        sink3 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink3.configure_for_resume()
        sink3.write([{"id": 4}], ctx)
        sink3.close()

        # All rows should be present
        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 4
        assert {r["id"] for r in rows} == {1, 2, 3, 4}

    def test_resume_with_append_mode_default(self, db_url: str, ctx) -> None:
        """Resume mode works when initial sink uses append mode (default)."""
        # Initial run with append mode (default behavior)
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "append",
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.close()

        # Resume run - configure_for_resume is idempotent when already append
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "append",
            }
        )
        sink2.configure_for_resume()  # Should be no-op since already append
        sink2.write([{"id": 2}], ctx)
        sink2.close()

        # Both rows should be present
        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {1, 2}

    def test_validate_output_target_then_write_in_resume_mode(self, db_url: str, ctx) -> None:
        """validate_output_target() should not break first write initialization."""
        # Initial run creates the table.
        sink1 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "append",
            }
        )
        sink1.write([{"id": 1}], ctx)
        sink1.close()

        # Resume sink validates first, then writes.
        sink2 = DatabaseSink(
            {
                "url": db_url,
                "table": "output",
                "schema": STRICT_SCHEMA,
                "if_exists": "replace",
            }
        )
        sink2.configure_for_resume()
        validation = sink2.validate_output_target()
        assert validation.valid is True

        # Regression check: this write used to raise RuntimeError.
        sink2.write([{"id": 2}], ctx)
        sink2.close()

        rows = self._get_all_rows(db_url, "output")
        assert len(rows) == 2
        assert {r["id"] for r in rows} == {1, 2}
