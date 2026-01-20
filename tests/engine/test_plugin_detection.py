# tests/engine/test_plugin_detection.py
"""Tests for type-safe plugin detection in processor.

These tests verify that isinstance-based plugin detection works correctly
with the base class hierarchy (BaseTransform, BaseGate).

NOTE: BaseAggregation tests were DELETED in aggregation structural cleanup.
Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
"""

from typing import Any

from elspeth.plugins.base import BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    GateResult,
    RoutingAction,
    TransformResult,
)


class TestPluginTypeDetection:
    """Tests for isinstance-based plugin detection."""

    def test_transform_is_base_transform(self) -> None:
        """Transforms should be instances of BaseTransform."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": {"fields": "dynamic"}})
        assert isinstance(transform, BaseTransform)

    def test_unknown_type_is_not_recognized(self) -> None:
        """Unknown plugin types should not match any base class."""

        class UnknownPlugin:
            """A class that is not a proper plugin."""

            pass

        unknown = UnknownPlugin()
        assert not isinstance(unknown, BaseTransform)
        assert not isinstance(unknown, BaseGate)

    def test_duck_typed_transform_not_recognized(self) -> None:
        """Duck-typed transforms without inheritance should NOT be recognized.

        This is the key behavior change - hasattr checks would have accepted
        this class, but isinstance checks correctly reject it.
        """

        class DuckTypedTransform:
            """Looks like a transform but doesn't inherit from BaseTransform."""

            name = "duck"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        duck = DuckTypedTransform()
        # Has the method but NOT an instance of BaseTransform
        assert hasattr(duck, "process")
        assert not isinstance(duck, BaseTransform)

    def test_duck_typed_gate_not_recognized(self) -> None:
        """Duck-typed gates without inheritance should NOT be recognized.

        This is the key behavior change - hasattr checks would have accepted
        this class, but isinstance checks correctly reject it.
        """

        class DuckTypedGate:
            """Looks like a gate but doesn't inherit from BaseGate."""

            name = "duck"

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        duck = DuckTypedGate()
        # Has the method but NOT an instance of BaseGate
        assert hasattr(duck, "evaluate")
        assert not isinstance(duck, BaseGate)


class TestPluginInheritanceHierarchy:
    """Tests verifying proper inheritance hierarchy."""

    def test_transform_not_gate(self) -> None:
        """Transforms should NOT be instances of BaseGate."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": {"fields": "dynamic"}})
        # mypy knows these are incompatible hierarchies - that's what we're verifying
        assert not isinstance(transform, BaseGate)  # type: ignore[unreachable]
