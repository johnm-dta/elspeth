# tests/plugins/test_protocols.py
"""Tests for plugin protocols."""

from collections.abc import Iterator
from typing import Any


class TestSourceProtocol:
    """Source plugin protocol."""

    def test_source_protocol_definition(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        # Should be a Protocol (runtime_checkable protocols have this attribute)
        assert hasattr(SourceProtocol, "__protocol_attrs__")

    def test_source_implementation(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol

        class OutputSchema(PluginSchema):
            value: int

        class MySource:
            """Example source implementation."""

            name = "my_source"
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                for i in range(3):
                    yield {"value": i}

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        source = MySource({"path": "test.csv"})

        # IMPORTANT: Verify protocol conformance at runtime
        # This is why we use @runtime_checkable
        assert isinstance(
            source, SourceProtocol
        ), "Source must conform to SourceProtocol"

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
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import TransformProtocol
        from elspeth.plugins.results import TransformResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(
                    {
                        "value": row["value"],
                        "doubled": row["value"] * 2,
                    }
                )

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        transform = DoubleTransform({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(
            transform, TransformProtocol
        ), "Must conform to TransformProtocol"

        ctx = PluginContext(run_id="test", config={})

        result = transform.process({"value": 21}, ctx)
        assert result.status == "success"
        assert result.row == {"value": 21, "doubled": 42}


class TestGateProtocol:
    """Gate plugin protocol (routing decisions)."""

    def test_gate_implementation(self) -> None:
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import GateProtocol
        from elspeth.plugins.results import GateResult, RoutingAction

        class RowSchema(PluginSchema):
            value: int

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.threshold = config["threshold"]

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                if row["value"] > self.threshold:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route(
                            "above",  # Route label, not sink name
                            reason={"value": row["value"], "threshold": self.threshold},
                        ),
                    )
                return GateResult(row=row, action=RoutingAction.route("below"))

            def close(self) -> None:
                pass

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

        # Below threshold - route to "below" label
        result = gate.evaluate({"value": 30}, ctx)
        assert result.action.kind == "route"
        assert result.action.destinations == ("below",)

        # Above threshold - route to "above" label
        result = gate.evaluate({"value": 100}, ctx)
        assert result.action.kind == "route"
        assert result.action.destinations == ("above",)


class TestAggregationProtocol:
    """Aggregation plugin protocol (stateful batching)."""

    def test_aggregation_implementation(self) -> None:
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import AggregationProtocol
        from elspeth.plugins.results import AcceptResult

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            total: int
            count: int

        class SumAggregation:
            name = "sum"
            input_schema = InputSchema
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.batch_size: int = config["batch_size"]
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= self.batch_size
                return AcceptResult(accepted=True, trigger=trigger)

            def should_trigger(self) -> bool:
                return len(self._values) >= self.batch_size

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
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
        assert isinstance(
            agg, AggregationProtocol
        ), "Must conform to AggregationProtocol"

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
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol

        class OutputSchema(PluginSchema):
            combined: str

        class QuorumCoalesce:
            name = "quorum_merge"
            policy = CoalescePolicy.QUORUM
            quorum_threshold = 2  # At least 2 branches must arrive
            expected_branches: list[str] = [
                "branch_a",
                "branch_b",
                "branch_c",
            ]
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def merge(
                self, branch_outputs: dict[str, dict[str, Any]], ctx: PluginContext
            ) -> dict[str, Any]:
                return {"combined": "+".join(branch_outputs.keys())}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = QuorumCoalesce({})

        # IMPORTANT: Verify protocol conformance at runtime
        # mypy may report this as unreachable due to structural subtyping analysis
        # but runtime_checkable protocols DO work at runtime
        assert isinstance(
            coalesce, CoalesceProtocol
        ), "Must conform to CoalesceProtocol"  # type: ignore[unreachable]

        assert coalesce.quorum_threshold == 2  # type: ignore[unreachable]
        assert len(coalesce.expected_branches) == 3  # type: ignore[unreachable]

    def test_coalesce_merge_behavior(self) -> None:
        """Test merge() combines branch outputs correctly."""
        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import CoalescePolicy, CoalesceProtocol

        class OutputSchema(PluginSchema):
            total: int

        class SumCoalesce:
            name = "sum_merge"
            policy = CoalescePolicy.REQUIRE_ALL
            quorum_threshold = None
            expected_branches: list[str] = ["branch_a", "branch_b"]
            output_schema = OutputSchema
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def merge(
                self, branch_outputs: dict[str, dict[str, Any]], ctx: PluginContext
            ) -> dict[str, Any]:
                total = sum(out["value"] for out in branch_outputs.values())
                return {"total": total}

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        coalesce = SumCoalesce({})
        # mypy may report this as unreachable due to structural subtyping analysis
        # but runtime_checkable protocols DO work at runtime
        assert isinstance(coalesce, CoalesceProtocol)  # type: ignore[unreachable]

        ctx = PluginContext(run_id="test", config={})  # type: ignore[unreachable]

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

        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol

        class InputSchema(PluginSchema):
            value: int

        class MemorySink:
            """Test sink that stores rows in memory."""

            name = "memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None  # Set by orchestrator
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            rows: ClassVar[list[dict[str, Any]]] = []

            def __init__(self, config: dict[str, Any]) -> None:
                self.instance_rows: list[dict[str, Any]] = []
                self.config = config

            def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
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

        # Protocol attributes are tracked in __protocol_attrs__ (runtime Protocol internals)
        assert "determinism" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_transform_has_version_attribute(self) -> None:
        from elspeth.plugins.protocols import TransformProtocol

        # __protocol_attrs__ is a runtime attribute on @runtime_checkable Protocols
        assert "plugin_version" in TransformProtocol.__protocol_attrs__  # type: ignore[attr-defined]

    def test_deterministic_transform(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class MyTransform:
            name = "my_transform"
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(row)

        t = MyTransform()
        assert t.determinism == Determinism.DETERMINISTIC

    def test_nondeterministic_transform(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        class LLMTransform:
            name = "llm_classifier"
            determinism = Determinism.EXTERNAL_CALL
            plugin_version = "0.1.0"

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                return TransformResult.success(row)

        t = LLMTransform()
        assert t.determinism == Determinism.EXTERNAL_CALL
