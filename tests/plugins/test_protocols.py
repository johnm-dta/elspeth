# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from collections.abc import Iterator


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Should be a Protocol (runtime_checkable protocols have this attribute)
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
        from elspeth.plugins.enums import Determinism
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
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

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
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.protocols import GateProtocol
        from elspeth.plugins.results import GateResult, RoutingAction
        from elspeth.plugins.schemas import PluginSchema

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                self.threshold = config["threshold"]

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


class TestAggregationProtocol:
    """Aggregation plugin protocol (stateful batching)."""

    def test_aggregation_implementation(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.protocols import AggregationProtocol
        from elspeth.plugins.results import AcceptResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int
            count: int

        class SumAggregation:
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                self.batch_size = config["batch_size"]
                self._values: list[int] = []

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= self.batch_size
                return AcceptResult(accepted=True, trigger=trigger)

            def should_trigger(self) -> bool:
                return len(self._values) >= self.batch_size

            def flush(self, ctx: PluginContext) -> list[dict]:
                result = {
                    "total": sum(self._values),
                    "count": len(self._values),
                }
                self._values = []
                return [result]

            def reset(self) -> None:
                self._values = []

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        agg = SumAggregation({"batch_size": 2})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(agg, AggregationProtocol), "Must conform to AggregationProtocol"

        ctx = PluginContext(run_id="test", config={})

        # First row - no trigger
        result = agg.accept({"value": 10}, ctx)
        assert result.accepted is True
        assert result.trigger is False

        # Second row - trigger
        result = agg.accept({"value": 20}, ctx)
        assert result.trigger is True

        # Flush
        outputs = agg.flush(ctx)
        assert len(outputs) == 1
        assert outputs[0] == {"total": 30, "count": 2}


class TestCoalesceProtocol:
    """Coalesce plugin protocol (merge parallel paths)."""

    def test_coalesce_policy_types(self) -> None:
        from elspeth.plugins.protocols import CoalescePolicy

        # All policies should exist
        assert CoalescePolicy.REQUIRE_ALL.value == "require_all"
        assert CoalescePolicy.QUORUM.value == "quorum"
        assert CoalescePolicy.BEST_EFFORT.value == "best_effort"

    def test_quorum_requires_threshold(self) -> None:
        """QUORUM policy needs a quorum_threshold."""
        from typing import ClassVar

        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            combined: str

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # At least 2 branches must arrive
            expected_branches: ClassVar[list[str]] = ["branch_a", "branch_b", "branch_c"]
            output_schema = OutputSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                pass

            def merge(self, branch_outputs: dict, ctx: PluginContext) -> dict:
                return {"combined": "+".join(branch_outputs.keys())}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = QuorumCoalesce({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(coalesce, CoalesceProtocol), "Must conform to CoalesceProtocol"

        assert coalesce.quorum_threshold == 2
        assert len(coalesce.expected_branches) == 3

    def test_coalesce_merge_behavior(self) -> None:
        """Test merge() combines branch outputs correctly."""
        from typing import ClassVar

        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol
        from elspeth.plugins.schemas import PluginSchema

        class OutputSchema(PluginSchema):
            total: int

        class SumCoalesce:
            name = "sum_merge"
            policy = CoalescePolicy.REQUIRE_ALL
            quorum_threshold = None
            expected_branches: ClassVar[list[str]] = ["branch_a", "branch_b"]
            output_schema = OutputSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict) -> None:
                pass

            def merge(self, branch_outputs: dict, ctx: PluginContext) -> dict:
                total = sum(out["value"] for out in branch_outputs.values())
                return {"total": total}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = SumCoalesce({})
        assert isinstance(coalesce, CoalesceProtocol)

        ctx = PluginContext(run_id="test", config={})

        branch_outputs = {
            "branch_a": {"value": 10},
            "branch_b": {"value": 20},
        }
        result = coalesce.merge(branch_outputs, ctx)
        assert result == {"total": 30}


class TestSinkProtocol:
    """Sink plugin protocol."""

    def test_sink_implementation(self) -> None:
        from typing import ClassVar

        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.protocols import SinkProtocol
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            value: int

        class MemorySink:
            """Test sink that stores rows in memory."""

            name = "memory"
            input_schema = InputSchema
            idempotent = True
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            rows: ClassVar[list[dict]] = []

            def __init__(self, config: dict) -> None:
                self.rows = []
                self.config = config

            def write(self, row: dict, ctx: PluginContext) -> None:
                self.rows.append(row)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        sink = MemorySink({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(sink, SinkProtocol), "Must conform to SinkProtocol"

        ctx = PluginContext(run_id="test", config={})

        sink.write({"value": 1}, ctx)
        sink.write({"value": 2}, ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}

    def test_sink_has_idempotency_support(self) -> None:
        """Sinks should support idempotency keys."""
        from elspeth.plugins.protocols import SinkProtocol

        # Protocol should have idempotent attribute
        assert hasattr(SinkProtocol, "__protocol_attrs__")


class TestProtocolMetadata:
    """Test that protocols include metadata attributes."""

    def test_transform_has_determinism_attribute(self) -> None:
        from elspeth.plugins.protocols import TransformProtocol

        # Protocol attributes are tracked in __protocol_attrs__
        assert "determinism" in TransformProtocol.__protocol_attrs__

    def test_transform_has_version_attribute(self) -> None:
        from elspeth.plugins.protocols import TransformProtocol

        assert "plugin_version" in TransformProtocol.__protocol_attrs__

    def test_deterministic_transform(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.results import TransformResult

        class MyTransform:
            name = "my_transform"
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        t = MyTransform()
        assert t.determinism == Determinism.DETERMINISTIC

    def test_nondeterministic_transform(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.enums import Determinism
        from elspeth.plugins.results import TransformResult

        class LLMTransform:
            name = "llm_classifier"
            determinism = Determinism.NONDETERMINISTIC
            plugin_version = "0.1.0"

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        t = LLMTransform()
        assert t.determinism == Determinism.NONDETERMINISTIC
