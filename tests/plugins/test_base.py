# tests/plugins/test_base.py
"""Tests for plugin base classes."""

import pytest


class TestBaseTransform:
    """Base class for transforms."""

    def test_base_transform_abstract(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # Should not be instantiable directly
        with pytest.raises(TypeError):
            BaseTransform({})  # type: ignore[abstract]

    def test_subclass_implementation(self) -> None:
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            doubled: int

        class DoubleTransform(BaseTransform):
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "x": row["x"],
                    "doubled": row["x"] * 2,
                })

        transform = DoubleTransform({"some": "config"})
        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"x": 21}, ctx)
        assert result.row == {"x": 21, "doubled": 42}

    def test_lifecycle_hooks_exist(self) -> None:
        from elspeth.plugins.base import BaseTransform

        # These should exist as no-op methods
        assert hasattr(BaseTransform, "on_register")
        assert hasattr(BaseTransform, "on_start")
        assert hasattr(BaseTransform, "on_complete")


class TestBaseGate:
    """Base class for gates."""

    def test_base_gate_implementation(self) -> None:
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate(BaseGate):
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                threshold = self.config["threshold"]
                if row["value"] > threshold:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = ThresholdGate({"threshold": 50})
        ctx = PluginContext(run_id="test", config={})

        result = gate.evaluate({"value": 100}, ctx)
        assert result.action.kind == "route_to_sink"


class TestBaseAggregation:
    """Base class for aggregations."""

    def test_base_aggregation_implementation(self) -> None:
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int

        class SumAggregation(BaseAggregation):
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                super().__init__(config)
                self._values: list[int] = []

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(
                    accepted=True,
                    trigger=len(self._values) >= self.config["batch_size"],
                )

            def should_trigger(self) -> bool:
                return len(self._values) >= self.config["batch_size"]

            def flush(self, ctx: PluginContext) -> list[dict]:
                result = {"total": sum(self._values)}
                self._values = []
                return [result]

        agg = SumAggregation({"batch_size": 2})
        ctx = PluginContext(run_id="test", config={})

        agg.accept({"value": 10}, ctx)
        result = agg.accept({"value": 20}, ctx)
        assert result.trigger is True

        outputs = agg.flush(ctx)
        assert outputs == [{"total": 30}]


class TestBaseSink:
    """Base class for sinks."""

    def test_base_sink_implementation(self) -> None:
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = InputSchema
            idempotent = True

            def __init__(self, config: dict) -> None:
                super().__init__(config)
                self.rows: list[dict] = []

            def write(self, row: dict, ctx: PluginContext) -> None:
                self.rows.append(row)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = MemorySink({})
        ctx = PluginContext(run_id="test", config={})

        sink.write({"value": 1}, ctx)
        sink.write({"value": 2}, ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
