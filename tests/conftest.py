# tests/conftest.py
"""Shared test fixtures and helpers.

This module provides reusable test utilities for creating test plugins that
properly implement the Protocol interfaces required by ELSPETH.

Test Base Classes:
- _TestSchema: Minimal PluginSchema for test fixtures
- _TestSourceBase: Base class for SourceProtocol test implementations
- _TestSinkBase: Base class for SinkProtocol test implementations
- _TestTransformBase: Base class for TransformProtocol test implementations

Hypothesis Configuration:
- "ci" profile: Fast tests for CI (100 examples) - default
- "nightly" profile: Thorough tests (1000 examples)
- "debug" profile: Minimal tests with verbose output (10 examples)

Set profile via environment variable:
    HYPOTHESIS_PROFILE=nightly pytest tests/property/
"""

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from hypothesis import Phase, Verbosity, settings

from elspeth.contracts import Determinism, PluginSchema, SourceRow

if TYPE_CHECKING:
    from elspeth.plugins.protocols import (
        GateProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
    )

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


# =============================================================================
# Shared Test Base Classes
# =============================================================================
# These provide all required Protocol attributes and methods so test classes
# only need to override what's specific to the test.


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures.

    Use this when tests don't need specific fields - just pass it as
    input_schema or output_schema to satisfy Protocol requirements.
    """

    pass


class _TestSourceBase:
    """Base class for test sources that implements SourceProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - output_schema: type[PluginSchema]
    - load(ctx) -> Iterator[SourceRow]

    Usage:
        class MyTestSource(_TestSourceBase):
            name = "my_source"
            output_schema = MySchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield from self.wrap_rows(self._data)
    """

    # Required by SourceProtocol - child classes must override
    name: str
    output_schema: type[PluginSchema]

    # Protocol-required attributes with defaults
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

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


class _TestSinkBase:
    """Base class for test sinks that implements SinkProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - write(rows, ctx) -> ArtifactDescriptor

    Usage:
        class MyTestSink(_TestSinkBase):
            name = "my_sink"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )
    """

    # Required by SinkProtocol - child classes must override
    name: str

    # Protocol-required attributes with defaults
    input_schema: type[PluginSchema] = _TestSchema
    idempotent: bool = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def flush(self) -> None:
        """Flush buffered data - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass


class _TestTransformBase:
    """Base class for test transforms that implements TransformProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - process(row, ctx) -> TransformResult

    Usage:
        class MyTestTransform(_TestTransformBase):
            name = "my_transform"

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success(row)
    """

    # Required by TransformProtocol - child classes must override
    name: str

    # Protocol-required attributes with defaults
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    is_batch_aware: bool = False
    creates_tokens: bool = False
    _on_error: str | None = None

    def on_start(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def on_complete(self, ctx: Any) -> None:
        """Lifecycle hook - no-op for tests."""
        pass

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass


# =============================================================================
# Type Cast Helpers for Tests
# =============================================================================
# These helpers provide type-safe ways to create test fixtures that satisfy
# the strict Protocol requirements. They use cast() internally to tell mypy
# that our test classes satisfy the protocols, even though they have minor
# differences (like more specific output_schema types).


def as_source(source: Any) -> "SourceProtocol":
    """Cast a test source to SourceProtocol.

    Use this when passing test source instances to functions expecting
    SourceProtocol. The cast tells mypy to trust that the test source
    satisfies the protocol.
    """
    return cast("SourceProtocol", source)


def as_transform(transform: Any) -> "TransformProtocol":
    """Cast a test transform to TransformProtocol."""
    return cast("TransformProtocol", transform)


def as_sink(sink: Any) -> "SinkProtocol":
    """Cast a test sink to SinkProtocol."""
    return cast("SinkProtocol", sink)


def as_gate(gate: Any) -> "GateProtocol":
    """Cast a test gate to GateProtocol."""
    return cast("GateProtocol", gate)


# Re-export for convenient import
__all__ = [
    "_TestSchema",
    "_TestSinkBase",
    "_TestSourceBase",
    "_TestTransformBase",
    "as_gate",
    "as_sink",
    "as_source",
    "as_transform",
]
