"""Tests for ``RowProcessor._cross_check_flush_output`` — ADR-009 §Clause 2.

Targets the batch-aware flush path that previously trusted static
``passes_through_input`` annotations. Every scenario uses real
``_FlushContext`` instances and calls the method on a real ``RowProcessor``
— no mocks of the method under test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import TokenInfo, TransformProtocol, TransformResult
from elspeth.contracts.declaration_contracts import AggregateDeclarationContractViolation
from elspeth.contracts.enums import OutputMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    PassThroughContractViolation,
    UnexpectedEmptyEmissionViolation,
    ZeroEmissionSuccessContractViolation,
)
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.engine.processor import _FlushContext
from elspeth.testing import make_contract, make_token_info
from tests.fixtures.landscape import make_recorder_with_run


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


def _make_flush_transform(
    *,
    name: str = "test-transform",
    passes_through_input: bool = True,
    can_drop_rows: bool = False,
    output_schema_config: Any = None,
) -> Mock:
    transform = Mock(spec=TransformProtocol)
    transform.node_id = "agg-node"
    transform.name = name
    transform.on_error = "discard"
    transform.on_success = None
    transform.is_batch_aware = True
    transform.creates_tokens = False
    transform.declared_output_fields = frozenset()
    transform.passes_through_input = passes_through_input
    transform.can_drop_rows = can_drop_rows
    transform._output_schema_config = output_schema_config
    transform.effective_static_contract.return_value = (
        output_schema_config.get_effective_guaranteed_fields() if output_schema_config is not None else frozenset()
    )
    return transform


def _make_fctx(
    *,
    transform: Any,
    tokens: list[TokenInfo],
    output_mode: OutputMode,
    triggering_token: TokenInfo | None = None,
    node_id: str = "agg-node",
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
        node_id=NodeID(node_id),
        transform=transform,
        settings=settings,
        buffered_tokens=tuple(tokens),
        batch_id="batch-1",
        error_msg="batch failed",
        expand_parent_token=tokens[0],
        triggering_token=triggering_token or tokens[-1],
        coalesce_node_id=None,
        coalesce_name=None,
    )


def _make_processor() -> Any:
    """Build a minimal processor that can drive _cross_check_flush_output."""
    from elspeth.contracts.types import BranchName, CoalesceName, GateName, SinkName  # noqa: F401
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


def _register_tokens(processor: Any, tokens: list[TokenInfo]) -> None:
    """Register rows and tokens in the audit DB so FAILED recording has FKs.

    Production code records BUFFERED first (the BUFFERED → terminal contract
    for aggregation tokens) but that requires a batch record; tests that
    only want to exercise the cross-check + failure-recording can skip the
    BUFFERED step without tripping the validator, which only checks that
    error_hash is present for FAILED.
    """
    for idx, token in enumerate(tokens):
        processor._data_flow.create_row(
            run_id="test-run",
            source_node_id="source-0",
            row_index=idx,
            data=token.row_data.to_dict(),
            row_id=token.row_id,
            source_row_index=idx,
            ingest_sequence=idx,
        )
        processor._data_flow.create_token(
            row_id=token.row_id,
            token_id=token.token_id,
        )


class TestPassThroughFalseDoesNotFirePassThroughViolation:
    def test_passes_through_false_does_not_raise_pass_through_violation(self) -> None:
        """The pass-through contract stays quiet when the declaration is False."""
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(3)]
        transform = _make_flush_transform(passes_through_input=False)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        # The processor still dispatches batch-flush contracts, but the
        # pass-through contract itself does not apply when the declaration is
        # false.
        result = TransformResult.success_multi(
            [PipelineRow({}, _make_contract({}))] * 3,
            success_reason={"action": "test"},
        )
        processor._cross_check_flush_output(fctx, result)


class TestPassthroughModePairwise:
    """Passthrough mode: 1:1 pairing. Each (input_token, output_row) checked independently."""

    def test_honest_passthrough_no_violation(self) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(3)]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        # Passthrough mode: same rows back, same contract.
        rows = [PipelineRow({"x": i}, contract) for i in range(3)]
        result = TransformResult.success_multi(rows, success_reason={"action": "passthrough"})
        # No exception — cross-check passes.
        processor._cross_check_flush_output(fctx, result)

    def test_mis_annotated_drops_field_violation(self) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int, "y": int})
        tokens = [_make_token(f"t{i}", {"x": i, "y": i * 10}, contract) for i in range(3)]
        _register_tokens(processor, tokens)
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        # Output drops 'y' — violation.
        reduced_contract = _make_contract({"x": int})
        rows = [PipelineRow({"x": i}, reduced_contract) for i in range(3)]
        result = TransformResult.success_multi(rows, success_reason={"action": "bad"})
        with pytest.raises(PassThroughContractViolation) as exc_info:
            processor._cross_check_flush_output(fctx, result)
        assert "y" in exc_info.value.divergence_set


class TestTransformModeIntersection:
    """Transform mode: batch-homogeneous intersection of input contracts."""

    def test_honest_transform_no_violation(self) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int, "y": int})
        tokens = [_make_token(f"t{i}", {"x": i, "y": i * 10}, contract) for i in range(3)]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        # Output preserves both fields.
        rows = [PipelineRow({"x": i, "y": i * 10}, contract) for i in range(6)]
        result = TransformResult.success_multi(rows, success_reason={"action": "expand"})
        processor._cross_check_flush_output(fctx, result)

    def test_heterogeneous_intersection_permissive_non_shared(self) -> None:
        """Heterogeneous batch: intersection permits drop of non-shared field."""
        processor = _make_processor()
        # Token 1 has {x, y}, token 2 has {x, z}. Intersection = {x}.
        contract_xy = _make_contract({"x": int, "y": int})
        contract_xz = _make_contract({"x": int, "z": int})
        tokens = [
            _make_token("t0", {"x": 1, "y": 10}, contract_xy),
            _make_token("t1", {"x": 2, "z": 20}, contract_xz),
        ]
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        # Output carries only {x} — drops y and z, but intersection was {x}, so OK.
        reduced = _make_contract({"x": int})
        rows = [PipelineRow({"x": 1}, reduced), PipelineRow({"x": 2}, reduced)]
        result = TransformResult.success_multi(rows, success_reason={"action": "intersect"})
        processor._cross_check_flush_output(fctx, result)

    def test_heterogeneous_intersection_rejects_intersection_drop(self) -> None:
        """Heterogeneous batch: dropping intersection field fails."""
        processor = _make_processor()
        contract_xy = _make_contract({"x": int, "y": int})
        contract_xz = _make_contract({"x": int, "z": int})
        tokens = [
            _make_token("t0", {"x": 1, "y": 10}, contract_xy),
            _make_token("t1", {"x": 2, "z": 20}, contract_xz),
        ]
        _register_tokens(processor, tokens)
        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.TRANSFORM)
        # Output drops 'x' which is in the intersection — violation.
        empty = _make_contract({})
        rows = [PipelineRow({}, empty), PipelineRow({}, empty)]
        result = TransformResult.success_multi(rows, success_reason={"action": "drop-intersection"})
        with pytest.raises(PassThroughContractViolation) as exc_info:
            processor._cross_check_flush_output(fctx, result)
        assert "x" in exc_info.value.divergence_set


class TestEmptyEmissionGovernance:
    @pytest.mark.parametrize(
        ("output_mode", "can_drop_rows"),
        [
            (OutputMode.PASSTHROUGH, False),
            (OutputMode.PASSTHROUGH, True),
            (OutputMode.TRANSFORM, False),
            (OutputMode.TRANSFORM, True),
        ],
    )
    def test_zero_emission_non_pass_through_raises_plugin_violation(
        self,
        output_mode: OutputMode,
        can_drop_rows: bool,
    ) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(2)]
        _register_tokens(processor, tokens)
        transform = _make_flush_transform(passes_through_input=False, can_drop_rows=can_drop_rows)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=output_mode)
        result = TransformResult.success_empty(success_reason={"action": "filtered"})

        with pytest.raises(ZeroEmissionSuccessContractViolation) as exc_info:
            processor._cross_check_flush_output(fctx, result)

        assert exc_info.value.passes_through_input is False
        assert exc_info.value.can_drop_rows is can_drop_rows
        assert exc_info.value.emitted_count == 0

    def test_zero_emission_passthrough_raises_aggregate_when_can_drop_rows_false(self) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(2)]
        _register_tokens(processor, tokens)
        transform = _make_flush_transform(passes_through_input=True, can_drop_rows=False)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        result = TransformResult.success_empty(success_reason={"action": "filtered"})

        with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
            processor._cross_check_flush_output(fctx, result)

        child_types = {type(child).__name__ for child in exc_info.value.violations}
        assert child_types == {"UnexpectedEmptyEmissionViolation", "PassThroughContractViolation"}
        empty_child = next(child for child in exc_info.value.violations if isinstance(child, UnexpectedEmptyEmissionViolation))
        assert empty_child.payload["emitted_count"] == 0
        pass_through_child = next(child for child in exc_info.value.violations if isinstance(child, PassThroughContractViolation))
        assert pass_through_child.divergence_set == frozenset({"x"})

    def test_zero_emission_passthrough_is_allowed_when_can_drop_rows_true(self) -> None:
        processor = _make_processor()
        contract = _make_contract({"x": int})
        tokens = [_make_token(f"t{i}", {"x": i}, contract) for i in range(2)]
        transform = _make_flush_transform(passes_through_input=True, can_drop_rows=True)
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        result = TransformResult.success_empty(success_reason={"action": "filtered"})

        processor._cross_check_flush_output(fctx, result)


class TestRecordFlushViolation:
    """Per-token FAILED audit entries with correct per-token context."""

    def test_records_failed_per_token_with_own_token_id_in_context(self) -> None:
        """$.context.token_id in audit records matches each row's own token,
        not the triggering token."""
        processor = _make_processor()
        contract = _make_contract({"x": int, "y": int})
        tokens = [_make_token(f"t{i}", {"x": i, "y": i * 10}, contract) for i in range(3)]
        _register_tokens(processor, tokens)

        transform = _make_flush_transform()
        fctx = _make_fctx(transform=transform, tokens=tokens, output_mode=OutputMode.PASSTHROUGH)
        # Mis-annotated: output drops y.
        reduced = _make_contract({"x": int})
        rows = [PipelineRow({"x": i}, reduced) for i in range(3)]
        result = TransformResult.success_multi(rows, success_reason={"action": "bad"})

        with pytest.raises(PassThroughContractViolation):
            processor._cross_check_flush_output(fctx, result)

        # Verify: every buffered token now has FAILED recorded, and each record
        # references its OWN token_id in the context payload.
        import json as _json

        import sqlalchemy as sa

        for token in tokens:
            query = sa.text(
                "SELECT outcome, path, context_json FROM token_outcomes WHERE token_id = :tid AND outcome = :outcome AND path = :path"
            ).bindparams(
                tid=token.token_id,
                outcome=TerminalOutcome.FAILURE.value,
                path=TerminalPath.UNROUTED.value,
            )
            row = processor._data_flow._ops.execute_fetchone(query)
            assert row is not None, f"Token {token.token_id} has no FAILED record"
            ctx = _json.loads(row.context_json)
            assert ctx["token_id"] == token.token_id, (
                f"context.token_id must match row's own token, not triggering token — got {ctx['token_id']!r} for row {token.token_id!r}"
            )


def _token_ref(token_id: str) -> Any:
    from elspeth.contracts.audit import TokenRef

    return TokenRef(token_id=token_id, run_id="test-run")
