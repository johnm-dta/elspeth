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
            artifact_descriptor={"kind": "test"},
        )

        assert adapter.name == "output"
        assert hasattr(adapter, "node_id")

    def test_sink_adapter_write_batch(self, ctx: PluginContext) -> None:
        """write() passes rows to underlying sink."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "test"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = adapter.write(rows, ctx)

        assert len(mock.rows_written) == 3
        assert mock.rows_written[0] == {"id": 1}
        assert isinstance(result, dict)

    def test_sink_adapter_returns_artifact_info(self, ctx: PluginContext) -> None:
        """write() returns artifact descriptor info."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/data/out.csv"},
        )

        result = adapter.write([{"id": 1}], ctx)

        # Should include artifact info
        assert "path" in result or "kind" in result

    def test_sink_adapter_tracks_rows_written(self, ctx: PluginContext) -> None:
        """Adapter tracks total rows written."""
        from elspeth.engine.adapters import SinkAdapter

        mock = MockSink()
        adapter = SinkAdapter(
            mock,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "test"},
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
            artifact_descriptor={"kind": "test"},
        )

        adapter.close()

        assert mock._closed
