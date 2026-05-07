"""Batch-flush dispatcher routing regression (filigree issue elspeth-ef8d5d92ff).

Post-H2 (ADR-010 §Semantics amendment 2026-04-20): ``_cross_check_flush_output``
calls ``run_batch_flush_checks`` on its own dispatch site (not the post-
emission site). The counting contract below claims ``batch_flush_check``
so every flush path fires it.

Under the H2 bundle redesign, the 2A ``override_input_fields`` sentinel is
gone — ``effective_input_fields`` is always set by the caller. The
TRANSFORM-flush path computes the batch-homogeneous intersection once and
passes it in the bundle; the PASSTHROUGH-flush path passes each token's
own field set. Both paths satisfy "the dispatcher receives a well-formed
caller-derived field set."
"""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict
from unittest.mock import Mock

import pytest

from elspeth.contracts import TokenInfo, TransformProtocol, TransformResult
from elspeth.contracts.declaration_contracts import (
    BatchFlushInputs,
    BatchFlushOutputs,
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.enums import OutputMode
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.engine.processor import _FlushContext
from elspeth.testing import make_contract, make_token_info
from tests.fixtures.landscape import make_recorder_with_run

# ---------------------------------------------------------------------------
# Counting contract: records each batch-flush dispatcher invocation's
# (token_id, effective_input_fields). Inherits the nominal ABC and decorates
# the batch_flush_check method with the dispatch-site marker.
# ---------------------------------------------------------------------------


class _CountingPayload(TypedDict):
    token_id: str


class _CountingContract(DeclarationContract):
    name = "counting_test_contract"
    payload_schema: type = _CountingPayload
    invocations: ClassVar[list[tuple[str, frozenset[str]]]] = []

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        _CountingContract.invocations.append((inputs.token_id, inputs.effective_input_fields))

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        inputs = BatchFlushInputs(
            plugin=object(),
            node_id="n",
            run_id="r",
            row_id="rw",
            token_id="t",
            buffered_tokens=(object(),),
            static_contract=frozenset(),
            effective_input_fields=frozenset(),
        )
        outputs = BatchFlushOutputs(emitted_rows=(object(),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        # Same bundle; registered only inside the isolation fixture.
        return cls.negative_example()


# ---------------------------------------------------------------------------
# Processor / flush-context builders
# ---------------------------------------------------------------------------


def _make_contract(fields: dict[str, type]) -> SchemaContract:
    return make_contract(fields=fields, mode="OBSERVED")


def _make_token(
    token_id: str,
    data: dict[str, Any],
    contract: SchemaContract,
    *,
    row_id: str | None = None,
) -> TokenInfo:
    row = PipelineRow(data, contract)
    token = make_token_info(token_id=token_id, row_id=row_id or f"row-{token_id}")
    return token.with_updated_data(row)


def _make_flush_transform(*, passes_through_input: bool = True, can_drop_rows: bool = False) -> Mock:
    transform = Mock(spec=TransformProtocol)
    transform.node_id = "agg-node"
    transform.name = "test-transform"
    transform.on_error = "discard"
    transform.on_success = None
    transform.is_batch_aware = True
    transform.creates_tokens = False
    transform.declared_output_fields = frozenset()
    transform.declared_input_fields = frozenset()
    transform.passes_through_input = passes_through_input
    transform.can_drop_rows = can_drop_rows
    transform._output_schema_config = None
    transform.effective_static_contract.return_value = frozenset()
    return transform


def _make_fctx(
    *,
    transform: Any,
    tokens: list[TokenInfo],
    output_mode: OutputMode,
) -> _FlushContext:
    settings = AggregationSettings(
        name="agg",
        plugin="batch_transform",
        input="source-0",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=len(tokens)),
        output_mode=output_mode,
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


def _make_processor() -> Any:
    """Minimal processor capable of driving ``_cross_check_flush_output``."""
    from elspeth.engine.processor import DAGTraversalContext, RowProcessor
    from elspeth.engine.spans import SpanFactory

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
    return RowProcessor(
        execution=setup.factory.execution,
        data_flow=setup.factory.data_flow,
        span_factory=SpanFactory(),
        run_id="test-run",
        source_node_id=NodeID("source-0"),
        source_on_success="default",
        traversal=traversal,
    )


@pytest.fixture(autouse=True)
def _isolate_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    _CountingContract.invocations = []
    register_declaration_contract(_CountingContract())
    yield
    _restore_registry_snapshot_for_tests(snapshot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchFlushDispatcherRouting:
    """All batch-flush scenarios must invoke the registered dispatcher on the
    BATCH_FLUSH site."""

    def test_passthrough_mode_fires_dispatcher_per_token(self) -> None:
        """PASSTHROUGH flush routes per-token through run_batch_flush_checks."""
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(3)]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        rows = [PipelineRow({"x": i}, contract) for i in range(3)]
        result = TransformResult.success_multi(rows, success_reason={"action": "pt"})

        processor._cross_check_flush_output(fctx, result)

        assert len(_CountingContract.invocations) == 3
        seen_token_ids = {inv[0] for inv in _CountingContract.invocations}
        assert seen_token_ids == {"t0", "t1", "t2"}
        # PASSTHROUGH passes each token's own field set — every invocation
        # sees frozenset({"x"}).
        for _token_id, effective_input_fields in _CountingContract.invocations:
            assert effective_input_fields == frozenset({"x"})

    def test_transform_mode_fires_dispatcher_once(self) -> None:
        """TRANSFORM-flush must route through run_batch_flush_checks (the C1 gap
        closed by the prior fix; the dispatch site rename did not regress it).
        """
        processor = _make_processor()
        contract = _make_contract({"x": int, "y": int})
        tokens = [_make_token(f"t{i}", {"x": i, "y": i * 10}, contract) for i in range(3)]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        rows = [PipelineRow({"x": i, "y": i * 10}, contract) for i in range(6)]
        result = TransformResult.success_multi(rows, success_reason={"action": "expand"})

        processor._cross_check_flush_output(fctx, result)

        assert len(_CountingContract.invocations) >= 1, "TRANSFORM-flush bypassed dispatcher"

    def test_transform_mode_passes_batch_intersection_as_effective_input_fields(self) -> None:
        """Batch intersection reaches the dispatcher via ``effective_input_fields``.

        Heterogeneous batch: token t0 has {x, y}, token t1 has {x, z}.
        Intersection is {x}; under the H2 bundle design this IS
        ``effective_input_fields`` (the 2A ``override_input_fields`` sentinel
        is gone per panel F1 resolution).
        """
        processor = _make_processor()
        contract_xy = _make_contract({"x": int, "y": int})
        contract_xz = _make_contract({"x": int, "z": int})
        tokens = [
            _make_token("t0", {"x": 1, "y": 10}, contract_xy),
            _make_token("t1", {"x": 2, "z": 20}, contract_xz),
        ]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        reduced = _make_contract({"x": int})
        rows = [PipelineRow({"x": 1}, reduced), PipelineRow({"x": 2}, reduced)]
        result = TransformResult.success_multi(rows, success_reason={"action": "intersect"})

        processor._cross_check_flush_output(fctx, result)

        assert len(_CountingContract.invocations) >= 1, "TRANSFORM-flush bypassed dispatcher"
        for token_id, effective_input_fields in _CountingContract.invocations:
            assert effective_input_fields == frozenset({"x"}), (
                f"Expected effective_input_fields=frozenset({{'x'}}) for {token_id!r}, got {effective_input_fields!r}"
            )

    def test_passes_through_false_still_dispatches_non_pass_through_contracts(self) -> None:
        """Non-pass-through transforms still route through the batch dispatcher.

        ``passes_through_input`` only controls whether the pass-through
        contract applies. Other batch-flush contracts, such as
        ``declared_output_fields`` and ``schema_config_mode``, still rely on
        ``run_batch_flush_checks`` for non-pass-through transforms.
        """
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(2)]
        transform = _make_flush_transform(passes_through_input=False)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        rows = [PipelineRow({}, _make_contract({}))] * 2
        result = TransformResult.success_multi(rows, success_reason={"action": "noop"})

        processor._cross_check_flush_output(fctx, result)
        assert len(_CountingContract.invocations) == 1
        token_id, effective_input_fields = _CountingContract.invocations[0]
        assert token_id == "t1"
        assert effective_input_fields == frozenset({"x"})

    def test_passthrough_zero_emission_still_hits_dispatcher(self) -> None:
        """Zero-emission passthrough has no 1:1 pairing, but governance still dispatches."""
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(2)]
        transform = _make_flush_transform(passes_through_input=True, can_drop_rows=True)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        result = TransformResult.success_empty(success_reason={"action": "filtered"})

        processor._cross_check_flush_output(fctx, result)

        assert len(_CountingContract.invocations) == 1
        token_id, effective_input_fields = _CountingContract.invocations[0]
        assert token_id == "t1"
        assert effective_input_fields == frozenset({"x"})
