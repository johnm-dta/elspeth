# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from typing import Iterator

import pytest


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from typing import runtime_checkable

        from elspeth.plugins.protocols import SourceProtocol

        # Should be a Protocol
        assert hasattr(SourceProtocol, "__protocol_attrs__")

    def test_source_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            value: int

        class MySource:
            """Example source implementation."""

            name = "my_source"
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict]:
                for i in range(3):
                    yield {"value": i}

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

        source = MySource({"path": "test.csv"})

        # IMPORTANT: Verify protocol conformance at runtime
        # This is why we use @runtime_checkable
        assert isinstance(source, SourceProtocol), "Source must conform to SourceProtocol"

        ctx = PluginContext(run_id="test", config={})

        rows = list(source.load(ctx))
        assert len(rows) == 3
        assert rows[0] == {"value": 0}

    def test_source_has_lifecycle_hooks(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Check protocol has expected methods
        assert hasattr(SourceProtocol, "load")
        assert hasattr(SourceProtocol, "close")
