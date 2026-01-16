# tests/plugins/test_protocol_lifecycle.py
"""Tests for plugin lifecycle methods in protocols."""

from elspeth.contracts import Determinism
from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol, TransformProtocol
from elspeth.plugins.results import GateResult, RoutingAction, TransformResult
from elspeth.plugins.schemas import PluginSchema


class TestTransformProtocolLifecycle:
    """Tests for TransformProtocol.close()."""

    def test_protocol_has_close_method(self) -> None:
        """TransformProtocol should define close()."""
        assert hasattr(TransformProtocol, "close")

    def test_transform_with_close_satisfies_protocol(self) -> None:
        """A class with close() should satisfy the protocol."""

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int

        class MyTransform:
            name = "test"
            input_schema = InputSchema
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                self.closed = False

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

            def close(self) -> None:
                self.closed = True

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        # Should not raise - satisfies protocol
        transform: TransformProtocol = MyTransform({})
        assert isinstance(transform, TransformProtocol)

        # close() should be callable
        transform.close()
        assert transform.closed is True


class TestGateProtocolLifecycle:
    """Tests for GateProtocol.close()."""

    def test_protocol_has_close_method(self) -> None:
        """GateProtocol should define close()."""
        assert hasattr(GateProtocol, "close")

    def test_gate_with_close_satisfies_protocol(self) -> None:
        """A class with close() should satisfy the protocol."""

        class RowSchema(PluginSchema):
            value: int

        class MyGate:
            name = "test_gate"
            input_schema = RowSchema
            output_schema = RowSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                self.closed = False

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

            def close(self) -> None:
                self.closed = True

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        # Should not raise - satisfies protocol
        gate: GateProtocol = MyGate({})
        assert isinstance(gate, GateProtocol)

        # close() should be callable
        gate.close()
        assert gate.closed is True
