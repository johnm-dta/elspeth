"""Property tests for coalesce trigger semantics in RowProcessor.

Verifies the node-based trigger (`current_node_id == coalesce_node_id`) is
equivalent to legacy step-based triggering for reachable states.
"""

from __future__ import annotations

from unittest.mock import Mock

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts import NodeType, TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.engine.processor import DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.testing import make_row
from tests.fixtures.landscape import make_factory, make_landscape_db


def _make_processor(
    *,
    step_count: int,
    coalesce_name: CoalesceName | None,
    coalesce_node_id: NodeID | None,
    coalesce_executor: Mock | None,
) -> tuple[RowProcessor, Mock]:
    """Construct a minimal processor with deterministic step/node mappings."""
    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "test-run"
    source_node_id = NodeID("source-0")

    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="test-source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id=source_node_id,
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )

    node_ids = [NodeID(f"node-{i}") for i in range(1, step_count + 1)]
    node_step_map = {source_node_id: 0}
    for idx, node_id in enumerate(node_ids, start=1):
        node_step_map[node_id] = idx

    node_to_next = {node_id: (node_ids[i + 1] if i + 1 < len(node_ids) else None) for i, node_id in enumerate(node_ids)}

    traversal = DAGTraversalContext(
        node_step_map=node_step_map,
        node_to_plugin={},
        first_transform_node_id=node_ids[0],
        node_to_next=node_to_next,
        coalesce_node_map={coalesce_name: coalesce_node_id} if coalesce_name is not None and coalesce_node_id is not None else {},
    )

    processor = RowProcessor(
        execution=factory.execution,
        data_flow=factory.data_flow,
        span_factory=SpanFactory(),
        run_id=run_id,
        source_node_id=source_node_id,
        source_on_success="default",
        traversal=traversal,
        coalesce_executor=coalesce_executor,
    )

    return processor, coalesce_executor if coalesce_executor is not None else Mock()


class TestCoalesceTriggerEquivalence:
    """Compare old step-based predicate vs new node-based predicate."""

    @given(
        step_count=st.integers(min_value=1, max_value=10),
        coalesce_step=st.integers(min_value=1, max_value=10),
        current_step=st.integers(min_value=1, max_value=10),
        has_executor=st.booleans(),
        has_branch=st.booleans(),
        has_coalesce_name=st.booleans(),
    )
    @settings(max_examples=250, deadline=None)
    def test_maybe_coalesce_matches_legacy_step_semantics_for_reachable_states(
        self,
        step_count: int,
        coalesce_step: int,
        current_step: int,
        has_executor: bool,
        has_branch: bool,
        has_coalesce_name: bool,
    ) -> None:
        """For reachable states (current_step <= coalesce_step), predicates match."""
        assume(coalesce_step <= step_count)
        assume(current_step <= coalesce_step)

        coalesce_name = CoalesceName("merge") if has_coalesce_name else None
        coalesce_node_id = NodeID(f"node-{coalesce_step}") if has_coalesce_name else None
        current_node_id = NodeID(f"node-{current_step}")

        executor = Mock()
        executor.accept.return_value = Mock(held=True, merged_token=None)
        processor, coalesce_executor = _make_processor(
            step_count=step_count,
            coalesce_name=coalesce_name,
            coalesce_node_id=coalesce_node_id,
            coalesce_executor=executor if has_executor else None,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data=make_row({"v": 1}),
            branch_name="path_a" if has_branch else None,
        )

        handled, result = processor._maybe_coalesce_token(
            token,
            current_node_id=current_node_id,
            coalesce_node_id=coalesce_node_id,
            coalesce_name=coalesce_name,
            child_items=[],
        )

        legacy_should_handle = has_executor and has_branch and has_coalesce_name and current_step >= coalesce_step
        node_should_handle = has_executor and has_branch and has_coalesce_name and current_step == coalesce_step
        assert legacy_should_handle == node_should_handle
        assert handled is node_should_handle

        if node_should_handle:
            coalesce_executor.accept.assert_called_once()
            assert result is None
        else:
            coalesce_executor.accept.assert_not_called()
            assert result is None
