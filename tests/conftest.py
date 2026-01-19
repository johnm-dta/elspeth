# tests/conftest.py
"""Shared test fixtures and helpers.

This module provides reusable test utilities for creating sources that
properly yield SourceRow objects as required by the source protocol.
"""

from collections.abc import Iterator
from typing import Any

from elspeth.contracts import PluginSchema, SourceRow


class _TestSourceBase:
    """Base class for test sources that properly yields SourceRow objects.

    Usage:
        class MyTestSource(_TestSourceBase):
            name = "my_source"
            output_schema = MySchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield from self.wrap_rows(self._data)
    """

    name: str
    output_schema: type[PluginSchema]

    def wrap_rows(self, rows: list[dict[str, Any]]) -> Iterator[SourceRow]:
        """Wrap plain dicts in SourceRow.valid() as required by source protocol."""
        for row in rows:
            yield SourceRow.valid(row)

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass


# Re-export for convenient import
__all__ = ["_TestSourceBase"]
