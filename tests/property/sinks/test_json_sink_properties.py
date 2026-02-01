# tests/property/sinks/test_json_sink_properties.py
"""Property-based tests for JSON sink behavior."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.json_sink import JSONSink
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


class TestJSONSinkProperties:
    """Property tests for JSON sink output."""

    @given(rows=rows_strategy, data=st.data())
    @SLOW_SETTINGS
    def test_jsonl_sink_hash_matches_file(self, rows: list[dict[str, object]], data: st.DataObject) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_id = data.draw(st.uuids()).hex
            path = Path(tmp_dir) / f"{file_id}.jsonl"

            sink = JSONSink(
                {
                    "path": str(path),
                    "format": "jsonl",
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str", "score: float?"]},
                }
            )
            ctx = PluginContext(run_id="test-run", config={})

            descriptor = sink.write(rows, ctx)
            sink.close()

            assert descriptor.content_hash == _compute_sha256(path)
            assert descriptor.size_bytes == path.stat().st_size

    @given(rows=rows_strategy, data=st.data())
    @SLOW_SETTINGS
    def test_json_array_sink_hash_matches_file(self, rows: list[dict[str, object]], data: st.DataObject) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_id = data.draw(st.uuids()).hex
            path = Path(tmp_dir) / f"{file_id}.json"

            sink = JSONSink(
                {
                    "path": str(path),
                    "format": "json",
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str", "score: float?"]},
                }
            )
            ctx = PluginContext(run_id="test-run", config={})

            descriptor = sink.write(rows, ctx)
            sink.close()

            assert descriptor.content_hash == _compute_sha256(path)
            assert descriptor.size_bytes == path.stat().st_size

    def test_json_sink_validate_input_rejects_wrong_types(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"

        sink = JSONSink(
            {
                "path": str(path),
                "format": "jsonl",
                "schema": {"mode": "strict", "fields": ["value: int"]},
                "validate_input": True,
            }
        )
        ctx = PluginContext(run_id="test-run", config={})

        with pytest.raises(ValidationError):
            sink.write([{"value": "not-an-int"}], ctx)
