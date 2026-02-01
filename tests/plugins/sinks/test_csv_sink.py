"""Tests for CSV sink plugin.

NOTE: Protocol compliance tests (test_implements_protocol, test_has_required_attributes)
are in conftest.py as parametrized tests covering all sink plugins.
"""

import csv
import hashlib
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext

# Strict schema config for tests - PathConfig now requires schema
# CSVSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "strict", "fields": ["id: str", "name: str"]}


class TestCSVSink:
    """Tests for CSVSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_write_creates_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() creates CSV file with headers."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.flush()
        sink.close()

        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "alice"

    def test_write_multiple_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Multiple writes append to CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.write([{"id": "2", "name": "bob"}], ctx)
        sink.write([{"id": "3", "name": "carol"}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_custom_delimiter(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support custom delimiter."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "delimiter": ";", "schema": STRICT_SCHEMA})

        sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert ";" in content
        assert "," not in content.replace(",", "")  # No commas except possibly in data

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        sink.write([{"id": "1"}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_flush_before_close(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Data is available after flush, before close."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        sink.write([{"id": "1"}], ctx)
        sink.flush()

        # File should have content before close
        content = output_file.read_text()
        assert "id" in content
        assert "1" in content

        sink.close()

    def test_batch_write_returns_artifact_descriptor(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        artifact = sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "file"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_sha256(self, tmp_path: Path, ctx: PluginContext) -> None:
        """content_hash is SHA-256 of file contents."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        artifact = sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()

        # Compute expected hash from file
        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_multiple_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Batch write handles multiple rows."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        rows = [
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "bob"},
            {"id": "3", "name": "carol"},
        ]
        artifact = sink.write(rows, ctx)
        sink.close()

        assert artifact.size_bytes > 0

        # Verify all rows written
        with open(output_file) as f:
            reader = csv.DictReader(f)
            written_rows = list(reader)
        assert len(written_rows) == 3

    def test_batch_write_empty_list(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        # Empty write still has a hash (of empty content)
        assert artifact.content_hash == hashlib.sha256(b"").hexdigest()

    def test_has_plugin_version(self) -> None:
        """CSVSink has plugin_version attribute."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv", "schema": STRICT_SCHEMA})
        assert hasattr(sink, "plugin_version")
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """CSVSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv", "schema": STRICT_SCHEMA})
        assert sink.determinism == Determinism.IO_WRITE

    def test_cumulative_hash_after_multiple_writes(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Each write() returns hash of cumulative file contents, not just new rows.

        For audit integrity, the ArtifactDescriptor returned from each write()
        must accurately reflect the complete file state at that moment.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        # First write
        artifact1 = sink.write([{"id": "1", "name": "alice"}], ctx)
        hash_after_first = artifact1.content_hash

        # Verify first hash matches file contents at this point
        file_content_after_first = output_file.read_bytes()
        expected_hash_after_first = hashlib.sha256(file_content_after_first).hexdigest()
        assert hash_after_first == expected_hash_after_first

        # Second write
        artifact2 = sink.write([{"id": "2", "name": "bob"}], ctx)
        hash_after_second = artifact2.content_hash

        # Verify second hash matches cumulative file contents (not just new rows)
        file_content_after_second = output_file.read_bytes()
        expected_hash_after_second = hashlib.sha256(file_content_after_second).hexdigest()
        assert hash_after_second == expected_hash_after_second

        # Third write
        artifact3 = sink.write([{"id": "3", "name": "carol"}], ctx)
        hash_after_third = artifact3.content_hash

        # Verify third hash matches full cumulative contents
        file_content_after_third = output_file.read_bytes()
        expected_hash_after_third = hashlib.sha256(file_content_after_third).hexdigest()
        assert hash_after_third == expected_hash_after_third

        sink.close()

        # Sanity check: each hash should be different (file grew each time)
        assert hash_after_first != hash_after_second
        assert hash_after_second != hash_after_third

    def test_explicit_schema_creates_all_headers_including_optional(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Headers should include all schema fields, not just first row keys.

        Bug: P2-2026-01-19-csvsink-fieldnames-inferred-from-first-row
        When schema is explicit, headers should come from schema config,
        not from first row keys. This ensures optional fields are present.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        # Strict schema with optional field 'score'
        explicit_schema = {
            "mode": "strict",
            "fields": ["id: int", "score: float?"],
        }

        output_file = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_file),
                "schema": explicit_schema,
            }
        )

        # First batch WITHOUT optional field
        sink.write([{"id": 1}], ctx)

        # Second batch WITH optional field - should NOT fail
        # Bug: This fails with ValueError because 'score' is not in fieldnames
        sink.write([{"id": 2, "score": 1.5}], ctx)

        sink.close()

        # Verify CSV has correct headers (including optional field)
        with open(output_file) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Headers should include 'score' even though first row didn't have it
        assert fieldnames is not None
        assert "score" in fieldnames

        assert len(rows) == 2

    def test_invalid_mode_rejected(self, tmp_path: Path) -> None:
        """Invalid mode values should be rejected at config time.

        This prevents typos like 'apend' from silently truncating files.
        """
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

        with pytest.raises(PluginConfigError, match=r"'write'.*'append'"):
            CSVSinkConfig.from_dict(
                {
                    "path": str(tmp_path / "output.csv"),
                    "schema": STRICT_SCHEMA,
                    "mode": "apend",  # Typo - should be "append"
                }
            )

    def test_valid_modes_accepted(self, tmp_path: Path) -> None:
        """Valid mode values 'write' and 'append' should be accepted."""
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

        # Both valid values should work without error
        write_config = CSVSinkConfig.from_dict(
            {
                "path": str(tmp_path / "write_output.csv"),
                "schema": STRICT_SCHEMA,
                "mode": "write",
            }
        )
        assert write_config.mode == "write"

        append_config = CSVSinkConfig.from_dict(
            {
                "path": str(tmp_path / "append_output.csv"),
                "schema": STRICT_SCHEMA,
                "mode": "append",
            }
        )
        assert append_config.mode == "append"


class TestCSVSinkSchemaValidation:
    """Tests for CSVSink schema compatibility validation.

    CSVSink requires fixed-column structure. Schemas that allow extra fields
    (free mode, dynamic mode) are incompatible because:
    - CSV headers are fixed at file creation
    - Extra fields would either be silently dropped (audit violation) or cause errors
    """

    def test_rejects_free_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink should reject free mode schemas at initialization.

        Free mode allows extra fields, but CSV requires fixed columns.
        This would cause silent data loss or runtime errors.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        free_schema = {"mode": "free", "fields": ["id: int"]}

        with pytest.raises(ValueError, match="allows_extra_fields"):
            CSVSink({"path": str(tmp_path / "output.csv"), "schema": free_schema})

    def test_rejects_dynamic_schema(self, tmp_path: Path) -> None:
        """CSVSink should reject dynamic schemas at initialization.

        Dynamic schemas allow any fields, but CSV requires fixed columns.
        This would cause silent data loss or runtime errors.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        dynamic_schema = {"fields": "dynamic"}

        with pytest.raises(ValueError, match="allows_extra_fields"):
            CSVSink({"path": str(tmp_path / "output.csv"), "schema": dynamic_schema})

    def test_accepts_strict_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink should accept strict mode schemas.

        Strict mode has fixed fields - compatible with CSV structure.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        strict_schema = {"mode": "strict", "fields": ["id: int", "name: str"]}

        # Should not raise
        sink = CSVSink({"path": str(tmp_path / "output.csv"), "schema": strict_schema})
        assert sink is not None
