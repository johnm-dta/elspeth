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


class TestTransformProtocol:
    """Transform plugin protocol (stateless row processing)."""

    def test_transform_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import TransformProtocol
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                self.config = config

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        transform = DoubleTransform({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(transform, TransformProtocol), "Must conform to TransformProtocol"

        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"value": 21}, ctx)
        assert result.status == "success"
        assert result.row == {"value": 21, "doubled": 42}


class TestGateProtocol:
    """Gate plugin protocol (routing decisions)."""

    def test_gate_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import GateProtocol
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self, config: dict) -> None:
                self.threshold = config.get("threshold", 10)

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["value"] > self.threshold:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink(
                            "high_values",
                            reason={"value": row["value"], "threshold": self.threshold},
                        ),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        gate = ThresholdGate({"threshold": 50})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(gate, GateProtocol), "Must conform to GateProtocol"

        ctx = PluginContext(run_id="test", config={})

        # Below threshold - continue
        result = gate.evaluate({"value": 30}, ctx)
        assert result.action.kind == "continue"

        # Above threshold - route
        result = gate.evaluate({"value": 100}, ctx)
        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("high_values",)
