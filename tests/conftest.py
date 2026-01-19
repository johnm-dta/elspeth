# tests/conftest.py
"""Shared test fixtures and helpers.

This module provides reusable test utilities for creating sources that
properly yield SourceRow objects as required by the source protocol.

Hypothesis Configuration:
- "ci" profile: Fast tests for CI (100 examples) - default
- "nightly" profile: Thorough tests (1000 examples)
- "debug" profile: Minimal tests with verbose output (10 examples)

Set profile via environment variable:
    HYPOTHESIS_PROFILE=nightly pytest tests/property/
"""

import os
from collections.abc import Iterator
from typing import Any

from hypothesis import Phase, Verbosity, settings

from elspeth.contracts import PluginSchema, SourceRow

# =============================================================================
# Hypothesis Configuration
# =============================================================================

# CI profile: Fast tests for continuous integration
settings.register_profile(
    "ci",
    max_examples=100,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,  # Disable deadline for CI (timing varies)
)

# Nightly profile: Thorough testing for scheduled runs
settings.register_profile(
    "nightly",
    max_examples=1000,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,
)

# Debug profile: Minimal examples with verbose output for debugging
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    deadline=None,
)

# Load profile from environment, default to "ci"
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))


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
