"""Tests for PassThrough transform."""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol


class TestPassThrough:
    """Tests for PassThrough transform plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """PassThrough implements TransformProtocol."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        assert isinstance(transform, TransformProtocol)

    def test_has_required_attributes(self) -> None:
        """PassThrough has name and schemas."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        assert PassThrough.name == "passthrough"
        assert hasattr(PassThrough, "input_schema")
        assert hasattr(PassThrough, "output_schema")

    def test_process_returns_unchanged_row(self, ctx: PluginContext) -> None:
        """process() returns row data unchanged."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row = {"id": 1, "name": "alice", "value": 100}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        assert result.row is not row  # Should be a copy, not the same object

    def test_process_with_nested_data(self, ctx: PluginContext) -> None:
        """Handles nested structures correctly."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row = {"id": 1, "meta": {"source": "test", "tags": ["a", "b"]}}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row
        # Nested structures should be deep copied
        assert result.row["meta"] is not row["meta"]
        assert result.row["meta"]["tags"] is not row["meta"]["tags"]

    def test_process_with_empty_row(self, ctx: PluginContext) -> None:
        """Handles empty row."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        row: dict = {}

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == {}

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        transform.close()
        transform.close()  # Should not raise
