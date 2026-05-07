"""Unit tests for ``verify_pass_through`` — ADR-009 §Clause 2 primitive.

Exercises the semantic of the cross-check directly, decoupled from the two
call sites (single-token executor, batch-flush processor). The key axes:

- Happy path: input_fields ⊆ runtime_observed → no-op.
- Empty emitted_rows obey `can_drop_rows` governance.
- Divergence: contract-only, payload-only, and both.
- Framework invariant: emitted row with no contract → FrameworkBugError.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import FrameworkBugError, PassThroughContractViolation
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.engine.executors.pass_through import verify_pass_through
from elspeth.testing import make_contract


def _row(data: dict[str, object], fields: dict[str, type]) -> PipelineRow:
    contract = make_contract(fields=fields, mode="OBSERVED")
    return PipelineRow(data, contract)


class TestHappyPath:
    def test_all_fields_preserved_no_violation(self) -> None:
        row = _row({"x": 1, "y": 2}, {"x": int, "y": int})
        verify_pass_through(
            input_fields=frozenset({"x", "y"}),
            emitted_rows=[row],
            can_drop_rows=False,
            static_contract=frozenset({"x", "y"}),
            transform_name="t",
            transform_node_id="n",
            run_id="r",
            row_id="row-1",
            token_id="tok-1",
        )


class TestEmptyEmissionGovernance:
    def test_empty_emitted_rows_raise_when_can_drop_rows_false(self) -> None:
        with pytest.raises(PassThroughContractViolation) as exc_info:
            verify_pass_through(
                input_fields=frozenset({"x"}),
                emitted_rows=[],
                can_drop_rows=False,
                static_contract=frozenset(),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )
        assert exc_info.value.divergence_set == frozenset({"x"})

    def test_empty_emitted_rows_no_op_when_can_drop_rows_true(self) -> None:
        verify_pass_through(
            input_fields=frozenset({"x"}),
            emitted_rows=[],
            can_drop_rows=True,
            static_contract=frozenset(),
            transform_name="t",
            transform_node_id="n",
            run_id="r",
            row_id="row-1",
            token_id="tok-1",
        )


class TestDivergence:
    def test_contract_only_divergence_raises(self) -> None:
        # Contract drops 'y', payload keeps it — intersection excludes 'y'.
        row = _row({"x": 1, "y": 2}, {"x": int})
        with pytest.raises(PassThroughContractViolation) as exc_info:
            verify_pass_through(
                input_fields=frozenset({"x", "y"}),
                emitted_rows=[row],
                can_drop_rows=False,
                static_contract=frozenset({"x", "y"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )
        assert "y" in exc_info.value.divergence_set

    def test_payload_only_divergence_raises(self) -> None:
        # Payload drops 'y', contract keeps it — intersection excludes 'y'.
        row = _row({"x": 1}, {"x": int, "y": int})
        with pytest.raises(PassThroughContractViolation) as exc_info:
            verify_pass_through(
                input_fields=frozenset({"x", "y"}),
                emitted_rows=[row],
                can_drop_rows=False,
                static_contract=frozenset({"x", "y"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )
        assert "y" in exc_info.value.divergence_set

    def test_both_sides_drop_raises(self) -> None:
        row = _row({"x": 1}, {"x": int})
        with pytest.raises(PassThroughContractViolation) as exc_info:
            verify_pass_through(
                input_fields=frozenset({"x", "y"}),
                emitted_rows=[row],
                can_drop_rows=False,
                static_contract=frozenset({"x", "y"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )
        assert "y" in exc_info.value.divergence_set

    def test_raises_on_first_offending_row(self) -> None:
        """Multiple rows, one with drop: raise immediately, don't process remaining."""
        ok = _row({"x": 1}, {"x": int})
        bad = _row({}, {})
        with pytest.raises(PassThroughContractViolation):
            verify_pass_through(
                input_fields=frozenset({"x"}),
                emitted_rows=[ok, bad, ok],
                can_drop_rows=False,
                static_contract=frozenset({"x"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )


class TestFrameworkInvariants:
    def test_emitted_row_with_no_contract_is_framework_bug(self) -> None:
        # PipelineRow.__init__ always sets a contract, so we fake a
        # contract-less emitted row via a duck-typed stub to exercise the
        # framework-invariant check in verify_pass_through.
        class _ContractNoneRow:
            def __init__(self) -> None:
                self.contract = None

            def keys(self) -> frozenset[str]:
                return frozenset({"x"})

        with pytest.raises(FrameworkBugError):
            verify_pass_through(
                input_fields=frozenset({"x"}),
                emitted_rows=[_ContractNoneRow()],  # type: ignore[list-item]
                can_drop_rows=False,
                static_contract=frozenset({"x"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )


class TestViolationPayload:
    def test_to_audit_dict_has_nine_keys(self) -> None:
        row = _row({"x": 1}, {"x": int})
        try:
            verify_pass_through(
                input_fields=frozenset({"x", "y"}),
                emitted_rows=[row],
                can_drop_rows=False,
                static_contract=frozenset({"x", "y"}),
                transform_name="t",
                transform_node_id="n",
                run_id="r",
                row_id="row-1",
                token_id="tok-1",
            )
            pytest.fail("expected violation")
        except PassThroughContractViolation as exc:
            audit = exc.to_audit_dict()
            expected_keys = {
                "exception_type",
                "message",
                "transform",
                "transform_node_id",
                "run_id",
                "row_id",
                "token_id",
                "static_contract",
                "runtime_observed",
                "divergence_set",
            }
            assert set(audit.keys()) == expected_keys
            # Frozensets must serialize as sorted lists for canonical JSON determinism.
            assert audit["static_contract"] == sorted(["x", "y"])
            assert audit["divergence_set"] == sorted(["y"])
