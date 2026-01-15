# tests/engine/test_plugin_detection.py
"""Tests for type-safe plugin detection in processor.

These tests verify that isinstance-based plugin detection works correctly
with the base class hierarchy (BaseTransform, BaseGate, BaseAggregation).
"""

from typing import Any

from elspeth.plugins.base import BaseAggregation, BaseGate, BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    RoutingAction,
    TransformResult,
)
from elspeth.plugins.schemas import PluginSchema


class TestPluginTypeDetection:
    """Tests for isinstance-based plugin detection."""

    def test_gate_is_base_gate(self) -> None:
        """Gates should be instances of BaseGate."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({"field": "score", "threshold": 0.5})
        assert isinstance(gate, BaseGate)

    def test_transform_is_base_transform(self) -> None:
        """Transforms should be instances of BaseTransform."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        assert isinstance(transform, BaseTransform)

    def test_aggregation_is_base_aggregation(self) -> None:
        """Aggregations should be instances of BaseAggregation."""
        # Create a minimal concrete aggregation for testing
        agg = _TestAggregation({"batch_size": 10})
        assert isinstance(agg, BaseAggregation)

    def test_unknown_type_is_not_recognized(self) -> None:
        """Unknown plugin types should not match any base class."""

        class UnknownPlugin:
            """A class that is not a proper plugin."""

            pass

        unknown = UnknownPlugin()
        assert not isinstance(unknown, BaseTransform)
        assert not isinstance(unknown, BaseGate)
        assert not isinstance(unknown, BaseAggregation)

    def test_duck_typed_transform_not_recognized(self) -> None:
        """Duck-typed transforms without inheritance should NOT be recognized.

        This is the key behavior change - hasattr checks would have accepted
        this class, but isinstance checks correctly reject it.
        """

        class DuckTypedTransform:
            """Looks like a transform but doesn't inherit from BaseTransform."""

            name = "duck"

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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

    def test_duck_typed_aggregation_not_recognized(self) -> None:
        """Duck-typed aggregations without inheritance should NOT be recognized.

        This is the key behavior change - hasattr checks would have accepted
        this class, but isinstance checks correctly reject it.
        """

        class DuckTypedAggregation:
            """Looks like an aggregation but doesn't inherit from BaseAggregation."""

            name = "duck"

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                return AcceptResult(accepted=True, trigger=False)

        duck = DuckTypedAggregation()
        # Has the method but NOT an instance of BaseAggregation
        assert hasattr(duck, "accept")
        assert not isinstance(duck, BaseAggregation)


class TestPluginInheritanceHierarchy:
    """Tests verifying proper inheritance hierarchy."""

    def test_gate_not_transform(self) -> None:
        """Gates should NOT be instances of BaseTransform."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({"field": "score", "threshold": 0.5})
        assert not isinstance(gate, BaseTransform)

    def test_transform_not_gate(self) -> None:
        """Transforms should NOT be instances of BaseGate."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({})
        assert not isinstance(transform, BaseGate)

    def test_aggregation_not_transform_or_gate(self) -> None:
        """Aggregations should NOT be instances of BaseTransform or BaseGate."""
        agg = _TestAggregation({"batch_size": 10})
        assert not isinstance(agg, BaseTransform)
        assert not isinstance(agg, BaseGate)


# Test-only aggregation implementation (no concrete aggregations exist yet)
class _TestSchema(PluginSchema):
    """Schema for test aggregation."""

    model_config = {"extra": "allow"}  # noqa: RUF012


class _TestAggregation(BaseAggregation):
    """Minimal aggregation for testing isinstance checks.

    This exists because no concrete aggregations are implemented yet.
    Once a real aggregation exists, this can be replaced.
    """

    name = "test_aggregation"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._batch: list[dict[str, Any]] = []
        self._batch_size = config.get("batch_size", 10)

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
        self._batch.append(row)
        return AcceptResult(accepted=True, trigger=len(self._batch) >= self._batch_size)

    def should_trigger(self) -> bool:
        return len(self._batch) >= self._batch_size

    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        result = self._batch.copy()
        self._batch = []
        return result

    def reset(self) -> None:
        self._batch = []
