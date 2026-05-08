# tests/unit/contracts/sink_contracts/test_csv_sink_contract.py
"""Contract tests for CSVSink plugin.

Verifies CSVSink honors the SinkProtocol contract.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db

from .test_sink_protocol import SinkDeterminismContractTestBase

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol


_CSV_TEXT = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters=("\x00",),
    ),
    min_size=1,
    max_size=20,
)


class _PartiallyFailingFile:
    """File wrapper that simulates a disk-full failure after partial write."""

    def __init__(self, wrapped: Any, partial_chars: int) -> None:
        self._wrapped = wrapped
        self._partial_chars = partial_chars

    def write(self, data: str) -> int:
        self._wrapped.write(data[: self._partial_chars])
        raise OSError("simulated disk full during CSV write")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


class TestCSVSinkContract(SinkDeterminismContractTestBase):
    """Contract + determinism tests for CSVSink.

    Determinism base is a strict superset of the contract base, so a single
    class covers both surfaces.
    """

    @pytest.fixture
    def sink_factory(self, tmp_path: Path) -> Callable[[], SinkProtocol]:
        """Create a factory for CSVSink instances."""
        counter = [0]

        def factory() -> SinkProtocol:
            counter[0] += 1
            return inject_write_failure(
                CSVSink(
                    {
                        "path": str(tmp_path / f"output_{counter[0]}.csv"),
                        "schema": {"mode": "fixed", "fields": ["id: int", "name: str", "score: float"]},
                    }
                )
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


class TestCSVSinkHashVerification:
    """Tests that verify content_hash and size_bytes match actual file content."""

    def test_content_hash_matches_file_content(self, tmp_path: Path) -> None:
        """Contract: content_hash MUST match SHA-256 of actual file bytes."""
        csv_path = tmp_path / "hash_verify.csv"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]}}))
        result = sink.write(rows, ctx)
        sink.close()

        expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        assert result.artifact.content_hash == expected_hash, (
            f"content_hash does not match file content! reported={result.artifact.content_hash}, actual={expected_hash}"
        )

    def test_size_bytes_matches_file_size(self, tmp_path: Path) -> None:
        """Contract: size_bytes MUST match actual file size."""
        csv_path = tmp_path / "size_verify.csv"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]}}))
        result = sink.write(rows, ctx)
        sink.close()

        expected_size = csv_path.stat().st_size
        assert result.artifact.size_bytes == expected_size, (
            f"size_bytes does not match file size! reported={result.artifact.size_bytes}, actual={expected_size}"
        )


class TestCSVSinkAppendMode:
    """Contract tests for CSVSink append mode."""

    def test_append_mode_adds_rows(self, tmp_path: Path) -> None:
        """Append mode MUST add rows to existing file."""
        csv_path = tmp_path / "append_test.csv"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        sink1 = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                    "mode": "write",
                }
            )
        )
        sink1.write([{"id": 1, "name": "Alice"}], ctx)
        sink1.close()

        sink2 = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                    "mode": "append",
                }
            )
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
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        assert not csv_path.exists()

        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                    "mode": "append",
                }
            )
        )
        sink.write([{"id": 1, "name": "Alice"}], ctx)
        sink.close()

        assert csv_path.exists()
        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2

    def test_append_write_failure_rolls_back_partial_bytes(self, tmp_path: Path) -> None:
        """Append mode MUST not leave unaudited partial bytes after write failure."""
        csv_path = tmp_path / "append_partial_failure.csv"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                    "mode": "append",
                }
            )
        )
        sink.write([{"id": 1, "name": "Alice"}], ctx)
        sink.flush()
        content_before_failure = csv_path.read_bytes()

        assert sink._file is not None
        sink._file = _PartiallyFailingFile(sink._file, partial_chars=3)

        try:
            with pytest.raises(OSError, match="simulated disk full"):
                sink.write([{"id": 2, "name": "Bob"}], ctx)
            assert csv_path.read_bytes() == content_before_failure
        finally:
            sink.close()


class TestCSVSinkPropertyBased:
    """Property-based tests for CSVSink."""

    @given(
        rows=st.lists(
            st.fixed_dictionaries(
                {
                    "id": st.integers(min_value=1, max_value=1000),
                    "name": _CSV_TEXT,
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

        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str", "value: int"]},
                }
            )
        )
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        result = sink.write(rows, ctx)
        sink.close()

        assert isinstance(result.artifact, ArtifactDescriptor)
        assert len(result.artifact.content_hash) == 64
        assert result.artifact.size_bytes > 0
        with open(csv_path, encoding="utf-8", newline="") as f:
            read_rows = list(csv.DictReader(f))
        assert read_rows == [{"id": str(row["id"]), "name": row["name"], "value": str(row["value"])} for row in rows]

    @given(
        rows=st.lists(
            st.fixed_dictionaries(
                {
                    "id": st.integers(min_value=1, max_value=100),
                    "data": _CSV_TEXT,
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

        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        path1 = tmp_path / f"test1_{uuid.uuid4().hex[:8]}.csv"
        path2 = tmp_path / f"test2_{uuid.uuid4().hex[:8]}.csv"

        sink1 = inject_write_failure(CSVSink({"path": str(path1), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
        result1 = sink1.write(rows, ctx)
        sink1.close()

        sink2 = inject_write_failure(CSVSink({"path": str(path2), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.artifact.content_hash == result2.artifact.content_hash


class TestCSVSinkQuotingCharacters:
    """Tests for CSV quoting with special characters (commas, quotes, newlines)."""

    def test_csv_quoting_with_commas(self, tmp_path: Path) -> None:
        """CSVSink MUST properly quote fields containing commas."""
        import csv

        csv_path = tmp_path / "quoting_commas.csv"
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "data": "value with, comma"}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
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
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "data": 'value with "quotes"'}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
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
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "data": "value with\nnewline"}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
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
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [{"id": 1, "data": 'value with "quotes" and, commas\nand newlines'}]
        sink = inject_write_failure(CSVSink({"path": str(csv_path), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
        sink.write(rows, ctx)
        sink.close()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 1
        assert read_rows[0]["data"] == 'value with "quotes" and, commas\nand newlines'

    def test_csv_quoting_roundtrip_determinism(self, tmp_path: Path) -> None:
        """CSVSink MUST produce deterministic output with special characters."""
        db = make_landscape_db()
        factory = make_factory(db)
        ctx = make_context(landscape=factory.plugin_audit_writer())

        rows = [
            {"id": 1, "data": 'complex "value", with\nspecial chars'},
            {"id": 2, "data": "another\nvalue"},
        ]

        path1 = tmp_path / "roundtrip1.csv"
        path2 = tmp_path / "roundtrip2.csv"

        sink1 = inject_write_failure(CSVSink({"path": str(path1), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
        result1 = sink1.write(rows, ctx)
        sink1.close()

        sink2 = inject_write_failure(CSVSink({"path": str(path2), "schema": {"mode": "fixed", "fields": ["id: int", "data: str"]}}))
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.artifact.content_hash == result2.artifact.content_hash
        assert path1.read_bytes() == path2.read_bytes()
