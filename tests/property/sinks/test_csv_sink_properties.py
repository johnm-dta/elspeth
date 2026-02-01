# tests/property/sinks/test_csv_sink_properties.py
"""Property-based tests for CSV sink behavior."""

from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink
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

            sink = CSVSink(
                {
                    "path": str(path),
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str", "score: float?"]},
                }
            )
            ctx = PluginContext(run_id="test-run", config={})

            descriptor = sink.write(rows, ctx)
            sink.close()

            assert descriptor.content_hash == _compute_sha256(path)
            assert descriptor.size_bytes == path.stat().st_size

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

            sink = CSVSink(
                {
                    "path": str(path),
                    "schema": {"mode": "strict", "fields": schema_fields},
                }
            )
            ctx = PluginContext(run_id="test-run", config={})

            sink.write([row], ctx)
            sink.close()

            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)

            assert header == fieldnames

    def test_csv_sink_validate_input_rejects_wrong_types(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"

        sink = CSVSink(
            {
                "path": str(path),
                "schema": {"mode": "strict", "fields": ["value: int"]},
                "validate_input": True,
            }
        )
        ctx = PluginContext(run_id="test-run", config={})

        with pytest.raises(ValidationError):
            sink.write([{"value": "not-an-int"}], ctx)
