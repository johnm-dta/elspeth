"""Tests for node_id in plugin protocols.

node_id is set by the Orchestrator after plugin registration with Landscape.
This test verifies it's part of the plugin contract (protocols and base classes).
"""


class TestNodeIdProtocol:
    """Verify node_id is part of plugin contract."""

    def test_source_protocol_has_node_id(self) -> None:
        """SourceProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import SourceProtocol

        # Use __annotations__ to check protocol attributes (avoids forward ref issues)
        annotations = SourceProtocol.__annotations__
        assert "node_id" in annotations, "SourceProtocol should define node_id"
        # Type annotation should be str | None (as string due to forward refs)
        assert annotations["node_id"] == str | None

    def test_base_source_has_node_id(self) -> None:
        """BaseSource has node_id attribute with default None."""
        from collections.abc import Iterator
        from typing import Any

        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseSource
        from elspeth.plugins.context import PluginContext

        class TestSchema(PluginSchema):
            pass

        class TestSource(BaseSource):
            name = "test"
            output_schema = TestSchema

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:  # type: ignore[override]
                yield {}

            def close(self) -> None:
                pass

        source = TestSource({})
        assert source.node_id is None  # Default

        source.node_id = "node-123"
        assert source.node_id == "node-123"

    def test_transform_protocol_has_node_id(self) -> None:
        """TransformProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import TransformProtocol

        annotations = TransformProtocol.__annotations__
        assert "node_id" in annotations, "TransformProtocol should define node_id"
        assert annotations["node_id"] == str | None

    def test_base_transform_has_node_id(self) -> None:
        """BaseTransform has node_id attribute with default None."""
        from typing import Any

        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class TestSchema(PluginSchema):
            pass

        class TestTransform(BaseTransform):
            name = "test"
            input_schema = TestSchema
            output_schema = TestSchema

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        transform = TestTransform({})
        assert transform.node_id is None

        transform.node_id = "transform-456"
        assert transform.node_id == "transform-456"

    def test_gate_protocol_has_node_id(self) -> None:
        """GateProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import GateProtocol

        annotations = GateProtocol.__annotations__
        assert "node_id" in annotations, "GateProtocol should define node_id"
        assert annotations["node_id"] == str | None

    def test_base_gate_has_node_id(self) -> None:
        """BaseGate has node_id attribute with default None."""
        from typing import Any

        from elspeth.contracts import PluginSchema, RoutingAction
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult

        class TestSchema(PluginSchema):
            pass

        class TestGate(BaseGate):
            name = "test"
            input_schema = TestSchema
            output_schema = TestSchema

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = TestGate({})
        assert gate.node_id is None

        gate.node_id = "gate-789"
        assert gate.node_id == "gate-789"

    def test_aggregation_protocol_deleted(self) -> None:
        """AggregationProtocol should be deleted (aggregation is structural)."""
        import elspeth.plugins.protocols as protocols

        assert not hasattr(protocols, "AggregationProtocol"), "AggregationProtocol should be deleted - aggregation is structural"

    def test_base_aggregation_deleted(self) -> None:
        """BaseAggregation should be deleted (aggregation is structural)."""
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"

    def test_coalesce_protocol_has_node_id(self) -> None:
        """CoalesceProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import CoalesceProtocol

        annotations = CoalesceProtocol.__annotations__
        assert "node_id" in annotations, "CoalesceProtocol should define node_id"
        assert annotations["node_id"] == str | None

    def test_sink_protocol_has_node_id(self) -> None:
        """SinkProtocol defines node_id attribute."""
        from elspeth.plugins.protocols import SinkProtocol

        annotations = SinkProtocol.__annotations__
        assert "node_id" in annotations, "SinkProtocol should define node_id"
        assert annotations["node_id"] == str | None

    def test_base_sink_has_node_id(self) -> None:
        """BaseSink has node_id attribute with default None."""
        from typing import Any

        from elspeth.contracts import PluginSchema
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext

        class TestSchema(PluginSchema):
            pass

        class TestSink(BaseSink):
            name = "test"
            input_schema = TestSchema

            def write(self, row: dict[str, Any], ctx: PluginContext) -> None:  # type: ignore[override]
                pass

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = TestSink({})
        assert sink.node_id is None

        sink.node_id = "sink-202"
        assert sink.node_id == "sink-202"
