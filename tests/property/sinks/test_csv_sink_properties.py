# tests/property/sinks/test_csv_sink_properties.py
"""Property-based tests for CSV sink behavior."""

from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db
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

# CSV schema fields must be valid identifiers to pass schema parsing.
identifier_headers = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]*", fullmatch=True)


# =============================================================================
# Helpers
# =============================================================================


def _compute_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# =============================================================================
# Property Tests
# =============================================================================


class TestCSVSinkProperties:
    """Property tests for CSV sink output."""

    @given(rows=rows_strategy, data=st.data())
    @SLOW_SETTINGS
    def test_csv_sink_hash_matches_file(self, rows: list[dict[str, object]], data: st.DataObject) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_id = data.draw(st.uuids()).hex
            path = Path(tmp_dir) / f"{file_id}.csv"

            sink = inject_write_failure(
                CSVSink(
                    {
                        "path": str(path),
                        "schema": {"mode": "fixed", "fields": ["id: int", "name: str", "score: float?"]},
                    }
                )
            )
            db = make_landscape_db()
            factory = make_factory(db)
            ctx = make_context(landscape=factory.plugin_audit_writer())

            result = sink.write(rows, ctx)
            sink.close()

            assert result.artifact.content_hash == _compute_sha256(path)
            assert result.artifact.size_bytes == path.stat().st_size

    @given(fieldnames=st.lists(identifier_headers, min_size=2, max_size=5, unique=True), data=st.data())
    @SLOW_SETTINGS
    def test_csv_sink_header_matches_schema_order(self, fieldnames: list[str], data: st.DataObject) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_id = data.draw(st.uuids()).hex
            path = Path(tmp_dir) / f"{file_id}.csv"

            schema_fields = [f"{name}: str" for name in fieldnames]
            permuted = data.draw(st.permutations(fieldnames))
            values = data.draw(
                st.lists(
                    st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
                    min_size=len(fieldnames),
                    max_size=len(fieldnames),
                )
            )
            row = dict(zip(permuted, values, strict=True))

            sink = inject_write_failure(
                CSVSink(
                    {
                        "path": str(path),
                        "schema": {"mode": "fixed", "fields": schema_fields},
                    }
                )
            )
            db = make_landscape_db()
            factory = make_factory(db)
            ctx = make_context(landscape=factory.plugin_audit_writer())

            sink.write([row], ctx)
            sink.close()

            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

            assert header == fieldnames
