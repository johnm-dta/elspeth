# tests/property/sinks/test_database_sink_properties.py
"""Property-based tests for database sink behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from sqlalchemy import create_engine, text

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.database_sink import DatabaseSink
from tests.property.settings import SLOW_SETTINGS

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


class _NullLandscape:
    def record_operation_call(self, *args, **kwargs):
        return None


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

            sink = DatabaseSink(
                {
                    "url": url,
                    "table": table_name,
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str", "score: float?"]},
                }
            )
            ctx = PluginContext(run_id="test-run", config={})
            ctx.operation_id = "op-test"
            ctx.landscape = _NullLandscape()

            descriptor = sink.write(rows, ctx)
            sink.close()

            expected_hash = stable_hash(rows)
            expected_size = len(canonical_json(rows).encode("utf-8"))

            assert descriptor.content_hash == expected_hash
            assert descriptor.size_bytes == expected_size
            assert descriptor.metadata is not None
            assert descriptor.metadata["row_count"] == len(rows)

            engine = create_engine(url)
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar_one()
            engine.dispose()

            assert count == len(rows)
