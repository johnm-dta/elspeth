# tests/property/sinks/test_database_sink_properties.py
"""Property-based tests for database sink behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.plugins.sinks.database_sink import DatabaseSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_operation_context
from tests.strategies.settings import SLOW_SETTINGS

# =============================================================================
# Strategies
# =============================================================================

row_strategy = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),
        "name": st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
        "score": st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.none()),
    }
)
rows_strategy = st.lists(row_strategy, min_size=1, max_size=5)


def _table_name_strategy() -> st.SearchStrategy[str]:
    return st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz").map(lambda s: f"t_{s}")


# =============================================================================
# Property Tests
# =============================================================================


class TestDatabaseSinkProperties:
    """Property tests for database sink output."""

    @given(rows=rows_strategy, table_name=_table_name_strategy(), data=st.data())
    @SLOW_SETTINGS
    def test_database_sink_hash_and_row_count(self, rows: list[dict[str, object]], table_name: str, data: st.DataObject) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_id = data.draw(st.uuids()).hex
            db_path = Path(tmp_dir) / f"{db_id}.db"
            url = f"sqlite:///{db_path}"

            sink = inject_write_failure(
                DatabaseSink(
                    {
                        "url": url,
                        "table": table_name,
                        "schema": {"mode": "fixed", "fields": ["id: int", "name: str", "score: float?"]},
                    }
                )
            )
            ctx = make_operation_context(
                node_id="sink",
                plugin_name="database_sink",
                node_type="SINK",
                operation_type="sink_write",
            )

            result = sink.write(rows, ctx)
            sink.close()

            expected_hash = stable_hash(rows)
            expected_size = len(canonical_json(rows).encode("utf-8"))

            assert result.artifact.content_hash == expected_hash
            assert result.artifact.size_bytes == expected_size
            assert result.artifact.metadata is not None
            assert result.artifact.metadata["row_count"] == len(rows)

            engine = create_engine(url)
            with engine.connect() as conn:
                db_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = db_result.scalar_one()
            engine.dispose()

            assert count == len(rows)
