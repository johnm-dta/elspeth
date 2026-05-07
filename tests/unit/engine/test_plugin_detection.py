# tests/unit/engine/test_plugin_detection.py
"""Tests for type-safe plugin detection in processor.

These tests verify that isinstance-based plugin detection works correctly
with the base class hierarchy (BaseTransform).
"""

from typing import Any

from elspeth.contracts import SourceRow
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.types import NodeID
from elspeth.engine.processor import DAGTraversalContext
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.factories import make_context


def _single_node_traversal(node_id: NodeID, plugin: Any) -> DAGTraversalContext:
    """Build explicit traversal context for a one-node pipeline."""
    return DAGTraversalContext(
        node_step_map={node_id: 1},
        node_to_plugin={node_id: plugin},
        first_transform_node_id=node_id,
        node_to_next={node_id: None},
        coalesce_node_map={},
    )


class TestPluginTypeDetection:
    """Tests for isinstance-based plugin detection."""

    def test_transform_is_base_transform(self) -> None:
        """Transforms should be instances of BaseTransform."""
        from elspeth.plugins.transforms.passthrough import PassThrough

        transform = PassThrough({"schema": {"mode": "observed"}})
        assert isinstance(transform, BaseTransform)

    def test_unknown_type_is_not_recognized(self) -> None:
        """Unknown plugin types should not match any base class."""

        class UnknownPlugin:
            """A class that is not a proper plugin."""

            pass

        unknown = UnknownPlugin()
        assert not isinstance(unknown, BaseTransform)

    def test_duck_typed_transform_not_recognized(self) -> None:
        """Duck-typed transforms without inheritance should NOT be recognized.

        This is the key behavior change - hasattr checks would have accepted
        this class, but isinstance checks correctly reject it.
        """

        class DuckTypedTransform:
            """Looks like a transform but doesn't inherit from BaseTransform."""

            name = "duck"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row.to_dict(), success_reason={"action": "test"})  # type: ignore[attr-defined]

        duck = DuckTypedTransform()
        # Has the method but NOT an instance of BaseTransform
        assert hasattr(duck, "process")
        assert not isinstance(duck, BaseTransform)  # type: ignore[unreachable]


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

        from elspeth.contracts.types import NodeID
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from tests.fixtures.landscape import make_recorder_with_run

        class DuckTypedTransform:
            """Looks like a transform but doesn't inherit from BaseTransform."""

            name = "duck"
            node_id = "fake_node_id"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row.to_dict(), success_reason={"action": "test"})  # type: ignore[attr-defined]

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        # The duck-typed object has the method, but processor should reject it
        duck = DuckTypedTransform()
        assert hasattr(duck, "process"), "Duck type has process method"
        # Runtime check that duck is not a BaseTransform - mypy knows these are incompatible
        assert not isinstance(duck, BaseTransform), "But is not a BaseTransform"  # type: ignore[unreachable]
        duck_node_id = NodeID(duck.node_id)

        processor = RowProcessor(
            setup.execution,
            setup.data_flow,
            span_factory=SpanFactory(),
            run_id=run_id,
            source_node_id=NodeID(source_node_id),
            source_on_success="default",
            traversal=_single_node_traversal(duck_node_id, duck),
        )

        ctx = make_context(run_id=run_id, landscape=factory.plugin_audit_writer())

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor.process_row(
                row_index=0,
                source_row=SourceRow.valid({"value": 1}, contract=create_observed_contract({"value": 1})),
                transforms=[duck],
                ctx=ctx,
            )
