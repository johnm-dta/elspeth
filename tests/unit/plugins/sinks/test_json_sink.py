"""Tests for JSON sink plugin."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.config_base import PluginConfigError

# Dynamic schema config for tests - PathConfig now requires schema
DYNAMIC_SCHEMA = {"mode": "observed"}


class TestJSONSink:
    """Tests for JSONSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_write_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSON array."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.write([{"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "alice"

    def test_write_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSONL (one per line)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "format": "jsonl", "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.write([{"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "alice"
        assert json.loads(lines[1])["name"] == "bob"

    def test_auto_detect_format_from_extension(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        # .jsonl extension should default to jsonl format
        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        # Should be JSONL format (one object per line, not array)
        content = output_file.read_text().strip()
        data = json.loads(content)
        assert data == {"id": 1}  # Single object, not array

    def test_json_extension_defaults_to_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Auto-detect JSON array format from .json extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert data == [{"id": 1}]

    def test_json_array_append_mode_is_rejected(self, tmp_path: Path) -> None:
        """JSON array format must reject append mode to prevent silent overwrite."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        with pytest.raises(PluginConfigError, match="does not support mode='append'"):
            JSONSink(
                {
                    "path": str(output_file),
                    "format": "json",
                    "mode": "append",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_json_extension_append_mode_is_rejected(self, tmp_path: Path) -> None:
        """Auto-detected JSON array format must also reject append mode."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        with pytest.raises(PluginConfigError, match="does not support mode='append'"):
            JSONSink(
                {
                    "path": str(output_file),
                    "mode": "append",
                    "schema": DYNAMIC_SCHEMA,
                }
            )

    def test_jsonl_append_mode_preserves_existing_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL append mode should remain supported."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        output_file.write_text(json.dumps({"id": 1}) + "\n")

        sink = JSONSink(
            {
                "path": str(output_file),
                "format": "jsonl",
                "mode": "append",
                "schema": DYNAMIC_SCHEMA,
            }
        )
        sink.write([{"id": 2}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()  # Should not raise

    def test_pretty_print_option(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support pretty-printed JSON output."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink(
            {
                "path": str(output_file),
                "format": "json",
                "indent": 2,
                "schema": DYNAMIC_SCHEMA,
            }
        )

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert "\n" in content  # Pretty-printed has newlines
        assert "  " in content  # Has indentation

    def test_batch_write_returns_artifact_descriptor(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "file"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_sha256(self, tmp_path: Path, ctx: PluginContext) -> None:
        """content_hash is SHA-256 of file contents."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_jsonl_content_hash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL format also returns correct content hash."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([{"id": 1}, {"id": 2}], ctx)
        sink.close()

        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_empty_list(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        assert artifact.content_hash == hashlib.sha256(b"").hexdigest()

    def test_has_plugin_version(self) -> None:
        """JSONSink has plugin_version attribute."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """JSONSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json", "schema": DYNAMIC_SCHEMA})
        assert sink.determinism == Determinism.IO_WRITE

    def test_cumulative_hash_after_multiple_writes(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Each write() returns hash of cumulative file contents."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})

        # First write
        artifact1 = sink.write([{"id": 1}], ctx)
        expected_hash1 = hashlib.sha256(output_file.read_bytes()).hexdigest()
        assert artifact1.content_hash == expected_hash1

        # Second write - hash should reflect cumulative contents
        artifact2 = sink.write([{"id": 2}], ctx)
        expected_hash2 = hashlib.sha256(output_file.read_bytes()).hexdigest()
        assert artifact2.content_hash == expected_hash2

        # Hashes should differ (file grew)
        assert artifact1.content_hash != artifact2.content_hash

        sink.close()

    def test_json_array_atomic_write_preserves_data_on_crash(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSON array mode uses atomic write - crash during write preserves prior data."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})

        # First write succeeds - establishes file with data
        sink.write([{"id": 1, "name": "alice"}], ctx)
        first_content = output_file.read_text()
        assert json.loads(first_content) == [{"id": 1, "name": "alice"}]

        # Simulate crash during second write: os.replace raises after temp file written
        def crash_on_replace(src: str, dst: str) -> None:
            # Remove temp file (simulating cleanup after crash) and raise
            os.unlink(src)
            raise OSError("Simulated crash during atomic replace")

        with (
            patch("os.replace", side_effect=crash_on_replace),
            pytest.raises(OSError, match="Simulated crash"),
        ):
            sink.write([{"id": 2, "name": "bob"}], ctx)

        # Original file is PRESERVED - this is the key assertion
        surviving_content = output_file.read_text()
        assert json.loads(surviving_content) == [{"id": 1, "name": "alice"}]
        sink.close()

    def test_json_array_no_temp_file_left_after_write(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Atomic write cleans up temp file after successful write."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}], ctx)
        sink.close()

        # No .tmp file left behind
        temp_file = output_file.with_suffix(".json.tmp")
        assert not temp_file.exists()

    def test_json_array_write_raises_if_append_mode_is_forced(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Defense-in-depth: JSON array writer fails fast if mode becomes append."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})
        sink._mode = "append"

        with pytest.raises(ValueError, match="does not support mode='append'"):
            sink.write([{"id": 1}], ctx)

        sink.close()

    def test_json_array_close_releases_buffered_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() releases buffered rows even without a persistent file handle."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "schema": DYNAMIC_SCHEMA})

        sink.write([{"id": 1}, {"id": 2}, {"id": 3}], ctx)
        assert len(sink._rows) == 3

        sink.close()
        assert len(sink._rows) == 0
