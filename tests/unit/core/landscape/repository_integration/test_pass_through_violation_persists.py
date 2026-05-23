"""Audit-DB round-trip for PassThroughContractViolation — ADR-009 §Clause 2.

Closes the ADR-008 queryability promise: when a pass-through violation
occurs in the batch-aware flush path, every buffered token's context
payload must be recoverable via SQL ``json_extract`` and must identify its
own token (not the triggering token).

Uses the same ``_FlushContext``-driven integration pattern as
``tests/unit/engine/test_cross_check_flush_output.py``, but asserts against
SQLite ``json_extract`` rather than Python-side ``json.loads`` so SQLite-
specific encoding quirks surface.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import sqlalchemy as sa
from tests.fixtures.landscape import make_recorder_with_run

from elspeth.contracts import TokenInfo, TransformProtocol, TransformResult
from elspeth.contracts.enums import OutputMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import PassThroughContractViolation
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.engine.processor import DAGTraversalContext, RowProcessor, _FlushContext
from elspeth.engine.spans import SpanFactory
from elspeth.testing import make_contract, make_token_info


def _make_processor() -> RowProcessor:
    setup = make_recorder_with_run(
        run_id="test-run",
        source_node_id="source-0",
        source_plugin_name="test-source",
    )
    traversal = DAGTraversalContext(
        node_step_map={NodeID("source-0"): 0, NodeID("agg-node"): 1},
        node_to_plugin={},
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


def _register(processor: RowProcessor, tokens: list[TokenInfo]) -> None:
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
        processor._data_flow.create_token(row_id=token.row_id, token_id=token.token_id)


def _mk_transform() -> Mock:
    t = Mock(spec=TransformProtocol)
    t.node_id = "agg-node"
    t.name = "mis-annotated"
    t.on_error = "discard"
    t.on_success = None
    t.is_batch_aware = True
    t.creates_tokens = False
    t.declared_output_fields = frozenset()
    t.declared_input_fields = frozenset()
    t.passes_through_input = True
    t.can_drop_rows = False
    t._output_schema_config = None
    t.effective_static_contract.return_value = frozenset()
    return t


def _mk_fctx(transform: Mock, tokens: list[TokenInfo]) -> _FlushContext:
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


class TestAuditRoundTrip:
    """json_extract over SQLite: every key in to_audit_dict() survives the round-trip."""

    def test_json_extract_returns_per_token_identifiers(self) -> None:
        """For every buffered token, json_extract of context_json returns
        THAT token's identifiers — not the triggering token's."""
        processor = _make_processor()
        contract = make_contract(fields={"x": int, "y": int}, mode="OBSERVED")
        tokens = [
            make_token_info(token_id=f"t{i}", row_id=f"row-t{i}").with_updated_data(PipelineRow({"x": i, "y": i * 10}, contract))
            for i in range(3)
        ]
        _register(processor, tokens)

        # Mis-annotated passthrough drops 'y'.
        transform = _mk_transform()
        fctx = _mk_fctx(transform, tokens)
        reduced = make_contract(fields={"x": int}, mode="OBSERVED")
        result = TransformResult.success_multi(
            [PipelineRow({"x": i}, reduced) for i in range(3)],
            success_reason={"action": "bad"},
        )

        with pytest.raises(PassThroughContractViolation):
            processor._cross_check_flush_output(fctx, result)

        # Run SQL against the real SQLite DB. json_extract must return the
        # row's own token_id, not the triggering token's. Also verify the
        # other context keys survive the round-trip.
        for token in tokens:
            query = sa.text(
                """
                SELECT
                    outcome,
                    path,
                    error_hash,
                    json_extract(context_json, '$.token_id') AS ctx_token_id,
                    json_extract(context_json, '$.row_id') AS ctx_row_id,
                    json_extract(context_json, '$.triggering_token_id') AS ctx_triggering_token_id,
                    json_extract(context_json, '$.exception_type') AS ctx_exception_type,
                    json_extract(context_json, '$.transform') AS ctx_transform,
                    json_extract(context_json, '$.divergence_set') AS ctx_divergence_set,
                    json_extract(context_json, '$.runtime_observed') AS ctx_runtime_observed,
                    json_extract(context_json, '$.static_contract') AS ctx_static_contract,
                    json_extract(context_json, '$.message') AS ctx_message
                FROM token_outcomes
                WHERE token_id = :tid
                  AND outcome = :outcome
                  AND path = :path
                """
            ).bindparams(
                tid=token.token_id,
                outcome=TerminalOutcome.FAILURE.value,
                path=TerminalPath.UNROUTED.value,
            )
            row = processor._data_flow._ops.execute_fetchone(query)

            assert row is not None, f"Token {token.token_id} has no FAILED record"

            # Per-token context identifies THIS token, not the triggering one.
            assert row.ctx_token_id == token.token_id
            assert row.ctx_row_id == token.row_id
            # Triggering token is the last buffered token.
            assert row.ctx_triggering_token_id == tokens[-1].token_id

            # Exception metadata
            assert row.ctx_exception_type == "PassThroughContractViolation"
            assert row.ctx_transform == "mis-annotated"
            assert row.ctx_message is not None
            assert "mis-annotated" in row.ctx_message
            assert "'y'" in row.ctx_message

            # Divergence set serialises as a JSON array (sorted) — json_extract
            # returns the raw JSON, which we parse back to compare.
            import json as _json

            divergence = _json.loads(row.ctx_divergence_set)
            assert divergence == ["y"]

            runtime_observed = _json.loads(row.ctx_runtime_observed)
            assert runtime_observed == ["x"]

            static_contract = _json.loads(row.ctx_static_contract)
            assert static_contract == []  # _output_schema_config is None in the mock

            # Error_hash is present (Tier 1 contract for FAILED outcomes).
            assert row.error_hash is not None
            assert len(row.error_hash) == 16  # hexdigest()[:16]

    def test_negative_control_nonexistent_path_is_null(self) -> None:
        """Sanity check: json_extract of a missing key returns NULL, proving
        the successful lookups above aren't vacuously resolving."""
        processor = _make_processor()
        contract = make_contract(fields={"x": int}, mode="OBSERVED")
        tokens = [make_token_info(token_id="t0", row_id="row-t0").with_updated_data(PipelineRow({"x": 0}, contract))]
        _register(processor, tokens)

        transform = _mk_transform()
        fctx = _mk_fctx(transform, tokens)
        empty_contract = make_contract(fields={}, mode="OBSERVED")
        result = TransformResult.success_multi(
            [PipelineRow({}, empty_contract)],
            success_reason={"action": "bad"},
        )

        with pytest.raises(PassThroughContractViolation):
            processor._cross_check_flush_output(fctx, result)

        query = sa.text(
            "SELECT json_extract(context_json, '$.definitely_not_a_real_key') AS missing "
            "FROM token_outcomes WHERE token_id = :tid AND outcome = :outcome AND path = :path"
        ).bindparams(tid="t0", outcome=TerminalOutcome.FAILURE.value, path=TerminalPath.UNROUTED.value)
        row = processor._data_flow._ops.execute_fetchone(query)
        assert row is not None
        assert row.missing is None  # NULL — the key doesn't exist in the payload.

    def test_failed_records_count_matches_buffered_tokens(self) -> None:
        """Every buffered token has exactly one FAILED record — no duplicates,
        no misses. The violation is batch-level but the audit trail is
        per-token (ADR-009 §Clause 2)."""
        processor = _make_processor()
        contract = make_contract(fields={"x": int, "y": int}, mode="OBSERVED")
        tokens = [
            make_token_info(token_id=f"t{i}", row_id=f"row-t{i}").with_updated_data(PipelineRow({"x": i, "y": i * 10}, contract))
            for i in range(5)
        ]
        _register(processor, tokens)

        transform = _mk_transform()
        fctx = _mk_fctx(transform, tokens)
        reduced = make_contract(fields={"x": int}, mode="OBSERVED")
        result = TransformResult.success_multi(
            [PipelineRow({"x": i}, reduced) for i in range(5)],
            success_reason={"action": "bad"},
        )

        with pytest.raises(PassThroughContractViolation):
            processor._cross_check_flush_output(fctx, result)

        query = sa.text(
            "SELECT COUNT(*) AS cnt FROM token_outcomes "
            "WHERE outcome = :outcome "
            "AND path = :path "
            "AND json_extract(context_json, '$.exception_type') = 'PassThroughContractViolation'"
        ).bindparams(outcome=TerminalOutcome.FAILURE.value, path=TerminalPath.UNROUTED.value)
        row = processor._data_flow._ops.execute_fetchone(query)
        assert row is not None
        assert row.cnt == 5
