"""Tests for ``_record_flush_violation`` failure semantics — ADR-009 §Clause 2.

If ``record_token_outcome`` raises mid-loop while recording FAILED entries
for a batch-flush PassThroughContractViolation, ``_record_flush_violation``
must raise ``AuditIntegrityError`` so the audit-write failure surfaces
loudly rather than silently corrupting the trail. The primary violation is
preserved via Python's implicit ``__context__`` (we're raising inside an
``except PassThroughContractViolation`` block in the caller).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import TokenInfo, TransformProtocol
from elspeth.contracts.enums import OutputMode
from elspeth.contracts.errors import AuditIntegrityError, PassThroughContractViolation
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.engine.processor import _FlushContext
from elspeth.testing import make_contract, make_token_info


def _make_fctx(transform: Any, tokens: list[TokenInfo]) -> _FlushContext:
    settings = AggregationSettings(
        name="agg",
        plugin="batch_transform",
        input="source-0",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=len(tokens)),
        output_mode=OutputMode.PASSTHROUGH,
    )
    return _FlushContext(
        node_id=NodeID("agg-node"),
        transform=transform,
        settings=settings,
        buffered_tokens=tuple(tokens),
        batch_id="batch-1",
        error_msg="batch failed",
        expand_parent_token=tokens[0],
        triggering_token=tokens[-1],
        coalesce_node_id=None,
        coalesce_name=None,
    )


def _make_violation(transform_name: str, token: TokenInfo) -> PassThroughContractViolation:
    return PassThroughContractViolation(
        transform=transform_name,
        transform_node_id="agg-node",
        run_id="test-run",
        row_id=token.row_id,
        token_id=token.token_id,
        static_contract=frozenset({"x"}),
        runtime_observed=frozenset(),
        divergence_set=frozenset({"x"}),
        message=f"Transform {transform_name!r} dropped fields ['x'] from row {token.row_id!r}.",
    )


def test_recorder_failure_mid_loop_raises_audit_integrity_error() -> None:
    """When record_token_outcome raises mid-loop, AuditIntegrityError is raised.

    The primary violation is surfaced in the error message AND preserved via
    the exception chain so operators can see both signals.
    """
    from elspeth.engine.processor import DAGTraversalContext, RowProcessor
    from elspeth.engine.spans import SpanFactory
    from tests.fixtures.landscape import make_recorder_with_run

    setup = make_recorder_with_run(
        run_id="test-run",
        source_node_id="source-0",
        source_plugin_name="test-source",
    )
    traversal = DAGTraversalContext(
        node_step_map={NodeID("source-0"): 0, NodeID("agg-node"): 1},
        node_to_plugin={},
        first_transform_node_id=None,
        node_to_next={NodeID("source-0"): None, NodeID("agg-node"): None},
        coalesce_node_map={},
    )
    processor = RowProcessor(
        execution=setup.factory.execution,
        data_flow=setup.factory.data_flow,
        span_factory=SpanFactory(),
        run_id="test-run",
        source_node_id=NodeID("source-0"),
        source_on_success="default",
        traversal=traversal,
    )
    # Swap the recorder for one that fails.
    original_record = processor._data_flow.record_token_outcome

    def _faulty_recorder(*args: Any, **kwargs: Any) -> Any:
        raise LandscapeRecordError("simulated audit DB outage")

    processor._data_flow.record_token_outcome = _faulty_recorder  # type: ignore[assignment]

    transform = Mock(spec=TransformProtocol)
    transform.node_id = "agg-node"
    transform.name = "faulty-test"
    transform.on_error = "discard"
    transform.on_success = None
    transform.is_batch_aware = True
    transform.creates_tokens = False
    transform.declared_output_fields = frozenset()
    transform.declared_input_fields = frozenset()
    transform.passes_through_input = True
    transform.can_drop_rows = False
    transform._output_schema_config = None

    contract = make_contract(fields={"x": int}, mode="OBSERVED")
    tokens = [make_token_info(token_id=f"t{i}", row_id=f"row-t{i}").with_updated_data(PipelineRow({"x": i}, contract)) for i in range(3)]
    fctx = _make_fctx(transform=transform, tokens=tokens)
    violation = _make_violation("faulty-test", tokens[-1])

    with pytest.raises(AuditIntegrityError) as exc_info:
        processor._record_flush_violation(fctx, violation)

    # The __cause__ chain points back to the recorder failure.
    cause = exc_info.value.__cause__
    assert isinstance(cause, LandscapeRecordError)
    assert "simulated audit DB outage" in str(cause)

    # The error message includes the original violation summary.
    assert "faulty-test" in str(exc_info.value)
    assert "INCOMPLETE" in str(exc_info.value)

    # Restore the recorder.
    processor._data_flow.record_token_outcome = original_record  # type: ignore[assignment]


def test_non_landscape_recorder_bug_mid_loop_propagates_plainly() -> None:
    """Non-recorder bugs must keep their original type during flush violation recording."""
    from elspeth.engine.processor import DAGTraversalContext, RowProcessor
    from elspeth.engine.spans import SpanFactory
    from tests.fixtures.landscape import make_recorder_with_run

    setup = make_recorder_with_run(
        run_id="test-run",
        source_node_id="source-0",
        source_plugin_name="test-source",
    )
    traversal = DAGTraversalContext(
        node_step_map={NodeID("source-0"): 0, NodeID("agg-node"): 1},
        node_to_plugin={},
        first_transform_node_id=None,
        node_to_next={NodeID("source-0"): None, NodeID("agg-node"): None},
        coalesce_node_map={},
    )
    processor = RowProcessor(
        execution=setup.factory.execution,
        data_flow=setup.factory.data_flow,
        span_factory=SpanFactory(),
        run_id="test-run",
        source_node_id=NodeID("source-0"),
        source_on_success="default",
        traversal=traversal,
    )
    original_record = processor._data_flow.record_token_outcome

    def _faulty_recorder(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("recorder validation bug")

    processor._data_flow.record_token_outcome = _faulty_recorder  # type: ignore[assignment]

    transform = Mock(spec=TransformProtocol)
    transform.node_id = "agg-node"
    transform.name = "faulty-test"
    transform.on_error = "discard"
    transform.on_success = None
    transform.is_batch_aware = True
    transform.creates_tokens = False
    transform.declared_output_fields = frozenset()
    transform.declared_input_fields = frozenset()
    transform.passes_through_input = True
    transform.can_drop_rows = False
    transform._output_schema_config = None

    contract = make_contract(fields={"x": int}, mode="OBSERVED")
    tokens = [make_token_info(token_id=f"t{i}", row_id=f"row-t{i}").with_updated_data(PipelineRow({"x": i}, contract)) for i in range(3)]
    fctx = _make_fctx(transform=transform, tokens=tokens)
    violation = _make_violation("faulty-test", tokens[-1])

    with pytest.raises(ValueError, match="recorder validation bug"):
        processor._record_flush_violation(fctx, violation)

    processor._data_flow.record_token_outcome = original_record  # type: ignore[assignment]
