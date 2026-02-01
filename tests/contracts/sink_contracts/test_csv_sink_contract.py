# tests/contracts/sink_contracts/test_csv_sink_contract.py
"""Contract tests for CSVSink plugin.

Verifies CSVSink honors the SinkProtocol contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.plugins.context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink

from .test_sink_protocol import SinkContractTestBase, SinkDeterminismContractTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SinkProtocol


class TestCSVSinkContract(SinkContractTestBase):
    """Contract tests for CSVSink."""

    @pytest.fixture
    def sink_factory(self, tmp_path: Path) -> Callable[[], SinkProtocol]:
        """Create a factory for CSVSink instances."""
        counter = [0]

        def factory() -> SinkProtocol:
            counter[0] += 1
            return CSVSink(
                {
                    "path": str(tmp_path / f"output_{counter[0]}.csv"),
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str", "score: float"]},
                }
            )

        return factory

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        """Provide sample rows to write."""
        return [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.0},
            {"id": 3, "name": "Charlie", "score": 91.2},
        ]


class TestCSVSinkDeterminism(SinkDeterminismContractTestBase):
    """Determinism contract tests for CSVSink."""

    @pytest.fixture
    def sink_factory(self, tmp_path: Path) -> Callable[[], SinkProtocol]:
        """Create a factory for CSVSink instances."""
        counter = [0]

        def factory() -> SinkProtocol:
            counter[0] += 1
            return CSVSink(
                {
                    "path": str(tmp_path / f"output_{counter[0]}.csv"),
                    "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                }
            )

        return factory

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        """Provide sample rows to write."""
        return [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]


class TestCSVSinkHashVerification:
    """Tests that verify content_hash and size_bytes match actual file content."""

    def test_content_hash_matches_file_content(self, tmp_path: Path) -> None:
        """Contract: content_hash MUST match SHA-256 of actual file bytes."""
        csv_path = tmp_path / "hash_verify.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "name: str"]}})
        result = sink.write(rows, ctx)
        sink.close()

        expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert result.content_hash == expected_hash, (
            f"content_hash does not match file content! reported={result.content_hash}, actual={expected_hash}"
        )

    def test_size_bytes_matches_file_size(self, tmp_path: Path) -> None:
        """Contract: size_bytes MUST match actual file size."""
        csv_path = tmp_path / "size_verify.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "name: str"]}})
        result = sink.write(rows, ctx)
        sink.close()

        expected_size = csv_path.stat().st_size
        assert result.size_bytes == expected_size, (
            f"size_bytes does not match file size! reported={result.size_bytes}, actual={expected_size}"
        )


class TestCSVSinkAppendMode:
    """Contract tests for CSVSink append mode."""

    def test_append_mode_adds_rows(self, tmp_path: Path) -> None:
        """Append mode MUST add rows to existing file."""
        csv_path = tmp_path / "append_test.csv"
        ctx = PluginContext(run_id="test", config={})

        sink1 = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "mode": "write",
            }
        )
        sink1.write([{"id": 1, "name": "Alice"}], ctx)
        sink1.close()

        sink2 = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "mode": "append",
            }
        )
        sink2.write([{"id": 2, "name": "Bob"}], ctx)
        sink2.close()

        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3
        assert "Alice" in content
        assert "Bob" in content

    def test_append_to_nonexistent_creates_file(self, tmp_path: Path) -> None:
        """Append mode on non-existent file MUST create it with header."""
        csv_path = tmp_path / "new_file.csv"
        ctx = PluginContext(run_id="test", config={})

        assert not csv_path.exists()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "mode": "append",
            }
        )
        sink.write([{"id": 1, "name": "Alice"}], ctx)
        sink.close()

        assert csv_path.exists()
        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2


class TestCSVSinkPropertyBased:
    """Property-based tests for CSVSink."""

    @given(
        rows=st.lists(
            st.fixed_dictionaries(
                {
                    "id": st.integers(min_value=1, max_value=1000),
                    "name": st.text(min_size=1, max_size=20).filter(lambda s: "\n" not in s and "," not in s and '"' not in s),
                    "value": st.integers(min_value=-(2**53 - 1), max_value=2**53 - 1),
                }
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_csv_sink_handles_arbitrary_rows(self, tmp_path: Path, rows: list[dict[str, Any]]) -> None:
        """Property: CSVSink handles any valid row data."""
        import uuid

        from elspeth.contracts import ArtifactDescriptor

        csv_path = tmp_path / f"test_{uuid.uuid4().hex[:8]}.csv"

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str", "value: int"]},
            }
        )
        ctx = PluginContext(run_id="test", config={})

        result = sink.write(rows, ctx)
        sink.close()

        assert isinstance(result, ArtifactDescriptor)
        assert len(result.content_hash) == 64
        assert result.size_bytes > 0

    @given(
        rows=st.lists(
            st.fixed_dictionaries(
                {
                    "id": st.integers(min_value=1, max_value=100),
                    "data": st.text(min_size=1, max_size=10).filter(lambda s: "\n" not in s and "," not in s and '"' not in s),
                }
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_csv_sink_hash_determinism_property(self, tmp_path: Path, rows: list[dict[str, Any]]) -> None:
        """Property: Same rows always produce same hash."""
        import uuid

        ctx = PluginContext(run_id="test", config={})

        path1 = tmp_path / f"test1_{uuid.uuid4().hex[:8]}.csv"
        path2 = tmp_path / f"test2_{uuid.uuid4().hex[:8]}.csv"

        sink1 = CSVSink({"path": str(path1), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        result1 = sink1.write(rows, ctx)
        sink1.close()

        sink2 = CSVSink({"path": str(path2), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.content_hash == result2.content_hash


class TestCSVSinkQuotingCharacters:
    """Tests for CSV quoting with special characters (commas, quotes, newlines)."""

    def test_csv_quoting_with_commas(self, tmp_path: Path) -> None:
        """CSVSink MUST properly quote fields containing commas."""
        import csv

        csv_path = tmp_path / "quoting_commas.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "data": "value with, comma"}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        sink.write(rows, ctx)
        sink.close()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 1
        assert read_rows[0]["data"] == "value with, comma"

    def test_csv_quoting_with_double_quotes(self, tmp_path: Path) -> None:
        """CSVSink MUST properly escape fields containing double quotes."""
        import csv

        csv_path = tmp_path / "quoting_quotes.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "data": 'value with "quotes"'}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        sink.write(rows, ctx)
        sink.close()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 1
        assert read_rows[0]["data"] == 'value with "quotes"'

    def test_csv_quoting_with_newlines(self, tmp_path: Path) -> None:
        """CSVSink MUST properly quote fields containing newlines."""
        import csv

        csv_path = tmp_path / "quoting_newlines.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "data": "value with\nnewline"}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        sink.write(rows, ctx)
        sink.close()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 1
        assert read_rows[0]["data"] == "value with\nnewline"

    def test_csv_quoting_all_special_characters(self, tmp_path: Path) -> None:
        """CSVSink MUST handle fields with all CSV special characters combined."""
        import csv

        csv_path = tmp_path / "quoting_all.csv"
        ctx = PluginContext(run_id="test", config={})

        rows = [{"id": 1, "data": 'value with "quotes" and, commas\nand newlines'}]
        sink = CSVSink({"path": str(csv_path), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        sink.write(rows, ctx)
        sink.close()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 1
        assert read_rows[0]["data"] == 'value with "quotes" and, commas\nand newlines'

    def test_csv_quoting_roundtrip_determinism(self, tmp_path: Path) -> None:
        """CSVSink MUST produce deterministic output with special characters."""
        ctx = PluginContext(run_id="test", config={})

        rows = [
            {"id": 1, "data": 'complex "value", with\nspecial chars'},
            {"id": 2, "data": "another\nvalue"},
        ]

        path1 = tmp_path / "roundtrip1.csv"
        path2 = tmp_path / "roundtrip2.csv"

        sink1 = CSVSink({"path": str(path1), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        result1 = sink1.write(rows, ctx)
        sink1.close()

        sink2 = CSVSink({"path": str(path2), "schema": {"mode": "strict", "fields": ["id: int", "data: str"]}})
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.content_hash == result2.content_hash
        assert path1.read_bytes() == path2.read_bytes()


class TestCSVSinkValidation:
    """Contract tests for CSVSink input validation."""

    def test_strict_schema_crashes_on_wrong_type(self, tmp_path: Path) -> None:
        """Strict schema MUST crash on wrong type (upstream bug!)."""
        from pydantic import ValidationError

        from elspeth.plugins.context import PluginContext

        sink = CSVSink(
            {
                "path": str(tmp_path / "strict.csv"),
                "schema": {
                    "mode": "strict",
                    "fields": ["id: int", "name: str"],
                },
                "validate_input": True,
            }
        )
        ctx = PluginContext(run_id="test", config={})

        # Wrong type for 'id' field
        bad_rows = [{"id": "not_an_int", "name": "Alice"}]

        # Per Three-Tier Trust Model: wrong types = crash
        with pytest.raises(ValidationError):
            sink.write(bad_rows, ctx)
