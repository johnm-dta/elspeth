"""Tests for engine adapters."""

from pathlib import Path
from typing import Any

import pytest

from elspeth.plugins.context import PluginContext


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
