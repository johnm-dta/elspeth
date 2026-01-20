# tests/contracts/sink_contracts/test_csv_sink_contract.py
"""Contract tests for CSVSink plugin.

Verifies CSVSink honors the SinkProtocol contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.plugins.sinks.csv_sink import CSVSink

from .test_sink_protocol import SinkContractTestBase, SinkDeterminismContractTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SinkProtocol


class TestCSVSinkContract(SinkContractTestBase):
    """Contract tests for CSVSink."""

    @pytest.fixture
    def sink(self, tmp_path: Path) -> SinkProtocol:
        """Create a CSVSink instance."""
        return CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": {"fields": "dynamic"},
            }
        )

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
    def sink(self, tmp_path: Path) -> SinkProtocol:
        """Create a CSVSink instance."""
        return CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        """Provide sample rows to write."""
        return [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

    def test_csv_sink_content_hash_is_deterministic(self, tmp_path: Path) -> None:
        """CSVSink: Same rows MUST produce same content_hash."""
        from elspeth.plugins.context import PluginContext

        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        ctx = PluginContext(run_id="test", config={})

        # Write to first file
        sink1 = CSVSink(
            {
                "path": str(tmp_path / "output1.csv"),
                "schema": {"fields": "dynamic"},
            }
        )
        result1 = sink1.write(rows, ctx)
        sink1.close()

        # Write to second file
        sink2 = CSVSink(
            {
                "path": str(tmp_path / "output2.csv"),
                "schema": {"fields": "dynamic"},
            }
        )
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.content_hash == result2.content_hash, "Same data produced different hashes - audit integrity compromised!"


class TestCSVSinkAppendMode:
    """Contract tests for CSVSink append mode."""

    def test_append_mode_adds_rows(self, tmp_path: Path) -> None:
        """Append mode MUST add rows to existing file."""
        from elspeth.plugins.context import PluginContext

        csv_path = tmp_path / "append_test.csv"
        ctx = PluginContext(run_id="test", config={})

        # First write
        sink1 = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"fields": "dynamic"},
                "mode": "write",
            }
        )
        sink1.write([{"id": 1, "name": "Alice"}], ctx)
        sink1.close()

        # Append write
        sink2 = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"fields": "dynamic"},
                "mode": "append",
            }
        )
        sink2.write([{"id": 2, "name": "Bob"}], ctx)
        sink2.close()

        # Verify file has both rows
        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 3  # Header + 2 data rows
        assert "Alice" in content
        assert "Bob" in content

    def test_append_to_nonexistent_creates_file(self, tmp_path: Path) -> None:
        """Append mode on non-existent file MUST create it with header."""
        from elspeth.plugins.context import PluginContext

        csv_path = tmp_path / "new_file.csv"
        ctx = PluginContext(run_id="test", config={})

        assert not csv_path.exists()

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"fields": "dynamic"},
                "mode": "append",
            }
        )
        sink.write([{"id": 1, "name": "Alice"}], ctx)
        sink.close()

        assert csv_path.exists()
        content = csv_path.read_text()
        lines = content.strip().split("\n")

        assert len(lines) == 2  # Header + 1 data row


class TestCSVSinkPropertyBased:
    """Property-based tests for CSVSink."""

    # RFC 8785 safe integer bounds
    _MAX_SAFE_INT = 2**53 - 1
    _MIN_SAFE_INT = -(2**53 - 1)

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
        # Use unique path for each test run
        import uuid

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.context import PluginContext

        csv_path = tmp_path / f"test_{uuid.uuid4().hex[:8]}.csv"

        sink = CSVSink(
            {
                "path": str(csv_path),
                "schema": {"fields": "dynamic"},
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

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})

        # Write twice to different files
        path1 = tmp_path / f"test1_{uuid.uuid4().hex[:8]}.csv"
        path2 = tmp_path / f"test2_{uuid.uuid4().hex[:8]}.csv"

        sink1 = CSVSink({"path": str(path1), "schema": {"fields": "dynamic"}})
        result1 = sink1.write(rows, ctx)
        sink1.close()

        sink2 = CSVSink({"path": str(path2), "schema": {"fields": "dynamic"}})
        result2 = sink2.write(rows, ctx)
        sink2.close()

        assert result1.content_hash == result2.content_hash


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
