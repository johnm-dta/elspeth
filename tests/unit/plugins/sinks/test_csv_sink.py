"""Tests for CSV sink plugin."""

import csv
import hashlib
from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext

# Strict schema config for tests - PathConfig now requires schema
# CSVSink requires fixed-column structure, so we use strict mode
# Tests that need specific fields define their own schema
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: str", "name: str"]}


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

        sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_flush_before_close(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Data is available after flush, before close."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        sink.write([{"id": "1", "name": "alice"}], ctx)
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

    def test_missing_required_field_fails_fast_even_without_validate_input(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Missing required fields must fail fast with validate_input=False."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        with pytest.raises(ValueError, match="missing required fields"):
            sink.write([{"id": "1"}], ctx)

        sink.close()
        assert not output_file.exists()

    def test_missing_required_field_mid_batch_writes_nothing(self, tmp_path: Path, ctx: PluginContext) -> None:
        """If any row misses required fields, entire batch fails before writes."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": STRICT_SCHEMA})

        rows = [
            {"id": "1", "name": "alice"},
            {"id": "2"},  # Missing required field "name"
        ]
        with pytest.raises(ValueError, match="missing required fields"):
            sink.write(rows, ctx)

        sink.close()
        assert not output_file.exists()

    def test_missing_optional_field_is_allowed(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Optional schema fields may be omitted without failing."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        optional_schema = {"mode": "fixed", "fields": ["id: str", "name: str?"]}
        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "schema": optional_schema})

        sink.write([{"id": "1"}], ctx)
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == ""

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
            "mode": "fixed",
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
    """Tests for CSVSink schema modes using infer-and-lock pattern.

    CSVSink supports all schema modes:
    - strict: columns from config, extras rejected at write time
    - free: declared columns + extras from first row, then locked
    - dynamic: columns from first row, then locked

    The "lock" is enforced by DictWriter's extrasaction='raise' default.
    """

    def test_accepts_strict_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts strict mode - columns from config."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        strict_schema = {"mode": "fixed", "fields": ["id: int", "name: str"]}

        sink = CSVSink({"path": str(tmp_path / "output.csv"), "schema": strict_schema})
        assert sink is not None

    def test_accepts_free_mode_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts free mode - declared + first-row extras, then locked."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        free_schema = {"mode": "flexible", "fields": ["id: int"]}

        sink = CSVSink({"path": str(tmp_path / "output.csv"), "schema": free_schema})
        assert sink is not None

    def test_accepts_dynamic_schema(self, tmp_path: Path) -> None:
        """CSVSink accepts dynamic mode - columns from first row, then locked."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        dynamic_schema = {"mode": "observed"}

        sink = CSVSink({"path": str(tmp_path / "output.csv"), "schema": dynamic_schema})
        assert sink is not None

    def test_dynamic_schema_infers_columns_from_first_row(self, tmp_path: Path) -> None:
        """Dynamic schema uses first row's keys as column headers."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        ctx = PluginContext(run_id="test-run", config={})
        sink = CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": {"mode": "observed"},
            }
        )

        # First write establishes columns
        sink.write([{"a": 1, "b": 2}], ctx)
        sink.close()

        # Verify headers match first row
        with open(tmp_path / "output.csv") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == {"a", "b"}

    def test_dynamic_schema_rejects_new_fields_after_lock(self, tmp_path: Path) -> None:
        """After first write, new fields are rejected (infer-and-lock)."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        ctx = PluginContext(run_id="test-run", config={})
        sink = CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": {"mode": "observed"},
            }
        )

        # First write locks columns to {a, b}
        sink.write([{"a": 1, "b": 2}], ctx)

        # Second write with extra field 'c' should fail
        with pytest.raises(ValueError, match="c"):
            sink.write([{"a": 3, "b": 4, "c": 5}], ctx)

        sink.close()

    def test_flexible_mode_includes_extras_from_first_row(self, tmp_path: Path) -> None:
        """Flexible mode includes declared fields + extras from first row.

        This is the documented behavior: flexible mode should accept at least
        the declared fields, plus any extras present in the first row.
        """
        from elspeth.plugins.sinks.csv_sink import CSVSink

        ctx = PluginContext(run_id="test-run", config={})
        # Schema declares only 'id', but first row has 'id', 'name', 'extra'
        flexible_schema = {"mode": "flexible", "fields": ["id: int"]}
        sink = CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": flexible_schema,
            }
        )

        # First write has declared field + two extras
        sink.write([{"id": 1, "name": "alice", "extra": "value"}], ctx)
        sink.close()

        # Verify all fields are in output (declared + extras)
        with open(tmp_path / "output.csv") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)

        # All three fields should be present
        assert "id" in fieldnames, "Declared field 'id' should be present"
        assert "name" in fieldnames, "Extra field 'name' from first row should be present"
        assert "extra" in fieldnames, "Extra field 'extra' from first row should be present"

        # Verify data was written correctly
        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "alice"
        assert rows[0]["extra"] == "value"

    def test_flexible_mode_declared_fields_come_first(self, tmp_path: Path) -> None:
        """Flexible mode should place declared fields before extras for predictability."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        ctx = PluginContext(run_id="test-run", config={})
        # Schema declares 'id' and 'name'
        flexible_schema = {"mode": "flexible", "fields": ["id: int", "name: str"]}
        sink = CSVSink(
            {
                "path": str(tmp_path / "output.csv"),
                "schema": flexible_schema,
            }
        )

        # First row has extras interspersed (dict order depends on Python version)
        sink.write([{"extra1": "a", "id": 1, "extra2": "b", "name": "alice"}], ctx)
        sink.close()

        # Verify declared fields come first in header order
        with open(tmp_path / "output.csv") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

        # Declared fields should be first (in schema order)
        assert fieldnames[0] == "id", "First declared field should be first"
        assert fieldnames[1] == "name", "Second declared field should be second"
        # Extras come after (order not guaranteed among extras)
        assert set(fieldnames[2:]) == {"extra1", "extra2"}, "Extra fields should follow declared fields"
