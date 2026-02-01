# tests/engine/test_plugin_detection.py
"""Tests for type-safe plugin detection in processor.

These tests verify that isinstance-based plugin detection works correctly
with the base class hierarchy (BaseTransform, BaseGate).

NOTE: BaseAggregation tests were DELETED in aggregation structural cleanup.
Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
"""

from typing import Any

from elspeth.contracts import NodeID, NodeType
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
                return TransformResult.success(row, success_reason={"action": "test"})

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


class TestProcessorRejectsDuckTypedPlugins:
    """Tests verifying RowProcessor rejects duck-typed plugins that don't inherit from base classes.

    This is the P1 fix for test_plugin_detection.py: verify that the processor
    actually raises TypeError when given duck-typed objects, not just that
    isinstance() returns False.
    """

    def test_processor_rejects_duck_typed_transform(self) -> None:
        """RowProcessor should raise TypeError for duck-typed transforms.

        Duck-typed transforms have the right methods (process, name, node_id)
        but don't inherit from BaseTransform. The processor must reject them
        to enforce the plugin contract.
        """
        import pytest

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        class DuckTypedTransform:
            """Looks like a transform but doesn't inherit from BaseTransform."""

            name = "duck"
            node_id = "fake_node_id"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # The duck-typed object has the method, but processor should reject it
        duck = DuckTypedTransform()
        assert hasattr(duck, "process"), "Duck type has process method"
        # Runtime check that duck is not a BaseTransform - mypy knows these are incompatible
        assert not isinstance(duck, BaseTransform), "But is not a BaseTransform"  # type: ignore[unreachable]

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor.process_row(
                row_index=0,
                row_data={"value": 1},
                transforms=[duck],
                ctx=ctx,
            )

    def test_processor_rejects_duck_typed_gate(self) -> None:
        """RowProcessor should raise TypeError for duck-typed gates.

        Duck-typed gates have the right methods (evaluate, name, node_id)
        but don't inherit from BaseGate. The processor must reject them.
        """
        import pytest

        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        class DuckTypedGate:
            """Looks like a gate but doesn't inherit from BaseGate."""

            name = "duck_gate"
            node_id = "fake_gate_id"

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(row=row, action=RoutingAction.continue_())

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # The duck-typed object has the method, but processor should reject it
        duck = DuckTypedGate()
        assert hasattr(duck, "evaluate"), "Duck type has evaluate method"
        # Runtime check that duck is not a BaseGate - mypy knows these are incompatible
        assert not isinstance(duck, BaseGate), "But is not a BaseGate"  # type: ignore[unreachable]

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor.process_row(
                row_index=0,
                row_data={"value": 1},
                transforms=[duck],
                ctx=ctx,
            )
