"""Tests for engine adapters."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.plugins.context import PluginContext

if TYPE_CHECKING:
    from elspeth.contracts import ArtifactDescriptor


class MockSink:
    """Mock sink for testing adapter."""

    name = "mock"

    def __init__(self) -> None:
        self.rows_written: list[dict[str, Any]] = []
        self._closed = False

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        self.rows_written.append(row)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class BatchMockSink:
    """Mock sink with batch signature for testing adapter delegation."""

    name = "batch_mock"

    def __init__(self) -> None:
        self.rows_written: list[dict[str, Any]] = []
        self._closed = False
        self._artifact_path = "/tmp/batch_mock_output.csv"

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> "ArtifactDescriptor":
        from elspeth.contracts import ArtifactDescriptor

        self.rows_written.extend(rows)
        return ArtifactDescriptor.for_file(
            path=self._artifact_path,
            content_hash=f"hash_{len(self.rows_written)}",
            size_bytes=len(str(rows)),
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class TestSinkAdapter:
    """Tests for SinkAdapter."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_sink_adapter_exists(self) -> None:
        """SinkAdapter can be imported."""
        from elspeth.engine.adapters import SinkAdapter

        assert SinkAdapter is not None

    def test_sink_adapter_has_name_and_node_id(self) -> None:
        """SinkAdapter exposes name and node_id."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        assert adapter.name == "output"
        assert hasattr(adapter, "node_id")

    def test_sink_adapter_write_batch(self, ctx: PluginContext) -> None:
        """write() passes rows to underlying sink."""
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.engine.artifacts import ArtifactDescriptor

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = adapter.write(rows, ctx)

        assert len(mock.rows_written) == 3
        assert mock.rows_written[0] == {"id": 1}
        assert isinstance(result, ArtifactDescriptor)

    def test_sink_adapter_returns_artifact_info(self, ctx: PluginContext) -> None:
        """write() returns ArtifactDescriptor with required fields."""
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.engine.artifacts import ArtifactDescriptor

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/data/out.csv"},
        )

        result = adapter.write([{"id": 1}], ctx)

        # Should return ArtifactDescriptor with all required fields
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"
        assert result.path_or_uri is not None
        assert result.content_hash is not None
        assert result.size_bytes is not None

    def test_sink_adapter_tracks_rows_written(self, ctx: PluginContext) -> None:
        """Adapter tracks total rows written."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.write([{"id": 1}, {"id": 2}], ctx)
        adapter.write([{"id": 3}], ctx)

        assert adapter.rows_written == 3

    def test_sink_adapter_close(self, ctx: PluginContext) -> None:
        """close() calls underlying sink's close()."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.close()

        assert mock._closed

    def test_sink_adapter_artifact_kind_file(self) -> None:
        """artifact_kind returns 'file' for file-based sinks."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/output.csv"},
        )

        assert adapter.artifact_kind == "file"

    def test_sink_adapter_artifact_kind_database(self) -> None:
        """artifact_kind returns 'database' for database sinks."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="database",
            sink_name="output",
            artifact_descriptor={
                "kind": "database",
                "url": "sqlite:///test.db",
                "table": "results",
            },
        )

        assert adapter.artifact_kind == "database"

    def test_sink_adapter_artifact_path_file(self) -> None:
        """artifact_path returns path for file-based sinks."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/data/output.csv"},
        )

        assert adapter.artifact_path == "/data/output.csv"

    def test_sink_adapter_artifact_path_database(self) -> None:
        """artifact_path returns None for database sinks."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="database",
            sink_name="output",
            artifact_descriptor={
                "kind": "database",
                "url": "sqlite:///test.db",
                "table": "results",
            },
        )

        assert adapter.artifact_path is None

    def test_adapter_delegates_to_batch_sink(self, ctx: PluginContext) -> None:
        """Adapter delegates directly to batch sink without per-row loop."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = adapter.write(rows, ctx)

        # Batch sink receives all rows at once
        assert len(batch_sink.rows_written) == 3
        # Result is from sink, not computed by adapter
        assert isinstance(result, ArtifactDescriptor)
        assert result.content_hash == "hash_3"  # From BatchMockSink

    def test_adapter_uses_sink_artifact_for_batch(self, ctx: PluginContext) -> None:
        """Adapter returns sink's ArtifactDescriptor for batch sinks."""
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        batch_sink._artifact_path = "/custom/path.csv"

        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            # This artifact_descriptor should be IGNORED for batch sinks
            artifact_descriptor={"kind": "file", "path": "/different/path.csv"},
        )

        result = adapter.write([{"id": 1}], ctx)

        # Should use sink's path, not adapter's artifact_descriptor
        assert "/custom/path.csv" in result.path_or_uri

    def test_adapter_still_loops_for_row_wise_sink(self, ctx: PluginContext) -> None:
        """Adapter uses per-row loop for RowWiseSinkProtocol sinks."""
        from elspeth.engine.adapters import SinkAdapter

        row_sink = MockSink()  # Uses row-wise write(row, ctx) -> None
        adapter = SinkAdapter(
            row_sink,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        adapter.write(rows, ctx)

        # Row-wise sink still receives rows individually
        assert len(row_sink.rows_written) == 3

    def test_rows_written_accurate_for_batch_sink(self, ctx: PluginContext) -> None:
        """rows_written is accurate after batch writes."""
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.write([{"id": 1}, {"id": 2}], ctx)
        adapter.write([{"id": 3}], ctx)

        assert adapter.rows_written == 3


class TestArtifactDescriptor:
    """Tests for ArtifactDescriptor dataclass."""

    def test_artifact_descriptor_for_file(self) -> None:
        """ArtifactDescriptor.for_file() creates file descriptor."""
        from elspeth.engine.artifacts import ArtifactDescriptor

        desc = ArtifactDescriptor.for_file(
            path="/data/output.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert desc.artifact_type == "file"
        assert desc.path_or_uri == "file:///data/output.csv"
        assert desc.content_hash == "abc123"
        assert desc.size_bytes == 1024
        assert desc.metadata is None

    def test_artifact_descriptor_for_database(self) -> None:
        """ArtifactDescriptor.for_database() creates database descriptor."""
        from elspeth.engine.artifacts import ArtifactDescriptor

        desc = ArtifactDescriptor.for_database(
            url="postgresql://localhost/db",
            table="results",
            content_hash="def456",
            payload_size=2048,
            row_count=100,
        )

        assert desc.artifact_type == "database"
        assert desc.path_or_uri == "db://results@postgresql://localhost/db"
        assert desc.content_hash == "def456"
        assert desc.size_bytes == 2048
        assert desc.metadata == {"table": "results", "row_count": 100}

    def test_artifact_descriptor_for_webhook(self) -> None:
        """ArtifactDescriptor.for_webhook() creates webhook descriptor."""
        from elspeth.engine.artifacts import ArtifactDescriptor

        desc = ArtifactDescriptor.for_webhook(
            url="https://api.example.com/webhook",
            content_hash="ghi789",
            request_size=512,
            response_code=200,
        )

        assert desc.artifact_type == "webhook"
        assert desc.path_or_uri == "webhook://https://api.example.com/webhook"
        assert desc.content_hash == "ghi789"
        assert desc.size_bytes == 512
        assert desc.metadata == {"response_code": 200}

    def test_artifact_descriptor_is_frozen(self) -> None:
        """ArtifactDescriptor should be immutable."""
        from elspeth.engine.artifacts import ArtifactDescriptor

        desc = ArtifactDescriptor.for_file(
            path="/data/output.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        import pytest

        with pytest.raises((AttributeError, TypeError)):
            desc.content_hash = "modified"  # type: ignore[misc]


class TestSinkAdapterWithArtifactDescriptor:
    """Tests for SinkAdapter returning ArtifactDescriptor."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_database_sink_artifact_registration(
        self, ctx: PluginContext, tmp_path: Path
    ) -> None:
        """Database sink should return ArtifactDescriptor without crashing.

        This test reproduces the bug: database sinks crash with KeyError
        when the caller expects 'path', 'content_hash', and 'size_bytes' keys
        but database artifacts have 'url' and 'table' instead.
        """
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.engine.artifacts import ArtifactDescriptor

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="database",
            sink_name="db_output",
            artifact_descriptor={
                "kind": "database",
                "url": "sqlite:///test.db",
                "table": "results",
            },
        )

        rows = [{"id": 1, "value": "test"}]
        result = adapter.write(rows, ctx)

        # Result must be an ArtifactDescriptor with all required fields
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type is not None
        assert result.path_or_uri is not None
        assert result.content_hash is not None
        assert result.size_bytes is not None

    def test_file_sink_artifact_still_works(
        self, ctx: PluginContext, tmp_path: Path
    ) -> None:
        """File sink should continue to work with ArtifactDescriptor."""
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.engine.artifacts import ArtifactDescriptor

        # Create a test file
        test_file = tmp_path / "output.csv"
        test_file.write_text("a,b,c\n1,2,3\n")

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="csv",
            sink_name="csv_output",
            artifact_descriptor={"kind": "file", "path": str(test_file)},
        )

        rows = [{"a": 1, "b": 2, "c": 3}]
        result = adapter.write(rows, ctx)

        # Verify result is ArtifactDescriptor with all required fields
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"
        assert "file://" in result.path_or_uri
        assert result.content_hash is not None
        assert result.size_bytes > 0

    def test_webhook_sink_artifact_registration(self, ctx: PluginContext) -> None:
        """Webhook sink should return ArtifactDescriptor without crashing."""
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.engine.artifacts import ArtifactDescriptor

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="webhook",
            sink_name="webhook_output",
            artifact_descriptor={
                "kind": "webhook",
                "url": "https://api.example.com/events",
            },
        )

        rows = [{"event": "test"}]
        result = adapter.write(rows, ctx)

        # Result must be an ArtifactDescriptor
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "webhook"
        assert "webhook://" in result.path_or_uri


class TestSinkTypeDetection:
    """Tests for batch vs row-wise sink detection."""

    def test_is_batch_sink_exists(self) -> None:
        """is_batch_sink can be imported."""
        from elspeth.engine.adapters import is_batch_sink

        assert is_batch_sink is not None

    def test_detects_batch_sink(self) -> None:
        """Detects sink with write(rows: list) signature."""
        from typing import Any

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import is_batch_sink
        from elspeth.plugins.context import PluginContext

        class BatchMockSink:
            """Mock sink with batch signature."""

            name = "batch_mock"

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(
                    path="/tmp/test.csv",
                    content_hash="abc123",
                    size_bytes=100,
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = BatchMockSink()
        assert is_batch_sink(sink) is True

    def test_detects_row_wise_sink(self) -> None:
        """Detects sink with write(row: dict) signature."""
        from typing import Any

        from elspeth.engine.adapters import is_batch_sink
        from elspeth.plugins.context import PluginContext

        class RowWiseMockSink:
            """Mock sink with row-wise signature."""

            name = "row_mock"

            def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
                pass

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = RowWiseMockSink()
        assert is_batch_sink(sink) is False

    def test_real_csv_sink_is_batch(self) -> None:
        """CSVSink (after WP-03) is detected as batch sink."""
        from elspeth.engine.adapters import is_batch_sink
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert is_batch_sink(sink) is True

    def test_broken_plugin_crashes(self) -> None:
        """Broken plugin (no write method) raises AttributeError, not silent fallback.

        ELSPETH trust model: plugins are part of the system, not user data.
        A sink without write() is a broken plugin that must crash loudly.
        """
        import pytest

        from elspeth.engine.adapters import is_batch_sink

        class BrokenSink:
            """Broken sink - missing write method."""

            name = "broken"

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        broken = BrokenSink()
        with pytest.raises(AttributeError):
            is_batch_sink(broken)


class TestSinkAdapterIntegration:
    """Integration tests for SinkAdapter with real sink plugins."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_adapter_with_csv_sink(self, ctx: PluginContext, tmp_path: Path) -> None:
        """SinkAdapter correctly delegates to CSVSink."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        adapter = SinkAdapter(
            sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": str(output_file)},
        )

        rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = adapter.write(rows, ctx)

        # Verify delegation worked
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"
        assert result.size_bytes > 0
        assert len(result.content_hash) == 64  # SHA-256 hex

        # Verify file was written
        assert output_file.exists()
        content = output_file.read_text()
        assert "Alice" in content
        assert "Bob" in content

        sink.close()

    def test_adapter_with_json_sink(self, ctx: PluginContext, tmp_path: Path) -> None:
        """SinkAdapter correctly delegates to JSONSink."""
        import json

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        adapter = SinkAdapter(
            sink,
            plugin_name="json",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": str(output_file)},
        )

        rows = [{"id": 1, "value": "test"}]
        result = adapter.write(rows, ctx)

        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"

        # Verify JSON content
        data = json.loads(output_file.read_text())
        assert data == rows

        sink.close()
