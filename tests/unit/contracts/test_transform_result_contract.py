"""Tests for TransformResult results that carry PipelineRow contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest

from elspeth.contracts.errors import (
    PluginContractViolation,
    TransformErrorReason,
    TransformSuccessReason,
)
from elspeth.contracts.results import TransformResult
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.testing import make_field


@pytest.fixture
def output_contract() -> SchemaContract:
    """Locked output contract with distinct original and normalized names."""
    return SchemaContract(
        mode="FIXED",
        fields=(
            make_field("record_id", int, original_name="Record ID", required=True, source="declared"),
            make_field("result", str, original_name="Result Label", required=True, source="declared"),
        ),
        locked=True,
    )


def _output_row(record_id: int, result: str, contract: SchemaContract) -> PipelineRow:
    return PipelineRow({"record_id": record_id, "result": result}, contract)


class TestTransformResultWithPipelineRow:
    """TransformResult must preserve schema-bearing PipelineRow output."""

    def test_success_preserves_single_pipeline_row_contract_and_reason(
        self,
        output_contract: SchemaContract,
    ) -> None:
        pipeline_row = _output_row(1, "ok", output_contract)
        success_reason: TransformSuccessReason = {"action": "processed", "fields_added": ["result"]}

        result = TransformResult.success(
            row=pipeline_row,
            success_reason=success_reason,
        )

        assert result.status == "success"
        assert result.row is pipeline_row
        assert result.rows is None
        assert result.reason is None
        assert result.retryable is False
        assert result.success_reason == success_reason
        assert result.is_multi_row is False
        assert result.has_output_data is True
        assert result.row.contract is output_contract
        assert result.row["Record ID"] == 1
        assert result.row["Result Label"] == "ok"

    def test_success_rejects_plain_dict_rows(self) -> None:
        with pytest.raises(PluginContractViolation, match="PipelineRow"):
            TransformResult.success(
                row=cast(Any, {"record_id": 1, "result": "ok"}),
                success_reason={"action": "processed"},
            )

    def test_direct_success_construction_rejects_plain_dict_rows(self) -> None:
        with pytest.raises(PluginContractViolation, match="PipelineRow"):
            TransformResult(
                status="success",
                row=cast(Any, {"record_id": 1, "result": "ok"}),
                reason=None,
                success_reason={"action": "processed"},
            )

    @pytest.mark.parametrize("status", ("partial", 1))
    def test_direct_construction_rejects_unknown_status(self, status: object) -> None:
        with pytest.raises(ValueError, match=r"status.*success.*error"):
            TransformResult(
                status=cast(Any, status),
                row=None,
                reason=None,
            )

    def test_success_multi_preserves_tuple_of_pipeline_rows_with_shared_contract(
        self,
        output_contract: SchemaContract,
    ) -> None:
        rows = [
            _output_row(1, "alpha", output_contract),
            _output_row(2, "beta", output_contract),
        ]
        success_reason: TransformSuccessReason = {"action": "split", "metadata": {"strategy": "fanout"}}

        result = TransformResult.success_multi(
            rows=rows,
            success_reason=success_reason,
        )
        rows.append(_output_row(3, "late mutation", output_contract))

        assert result.status == "success"
        assert result.row is None
        assert result.rows == (rows[0], rows[1])
        assert isinstance(result.rows, tuple)
        assert result.success_reason == success_reason
        assert result.is_multi_row is True
        assert result.has_output_data is True
        assert [row["Record ID"] for row in result.rows] == [1, 2]
        assert all(row.contract is output_contract for row in result.rows)

    def test_success_multi_rejects_mixed_contract_instances_even_when_structurally_equal(
        self,
        output_contract: SchemaContract,
    ) -> None:
        structurally_equal_contract = SchemaContract(
            mode=output_contract.mode,
            fields=output_contract.fields,
            locked=output_contract.locked,
        )
        assert structurally_equal_contract == output_contract
        assert structurally_equal_contract is not output_contract

        with pytest.raises(PluginContractViolation, match="same contract instance"):
            TransformResult.success_multi(
                rows=[
                    _output_row(1, "alpha", output_contract),
                    _output_row(2, "beta", structurally_equal_contract),
                ],
                success_reason={"action": "split"},
            )

    def test_direct_multi_row_construction_rejects_mixed_contract_instances(
        self,
        output_contract: SchemaContract,
    ) -> None:
        structurally_equal_contract = SchemaContract(
            mode=output_contract.mode,
            fields=output_contract.fields,
            locked=output_contract.locked,
        )

        with pytest.raises(PluginContractViolation, match="same contract instance"):
            TransformResult(
                status="success",
                row=None,
                rows=(
                    _output_row(1, "alpha", output_contract),
                    _output_row(2, "beta", structurally_equal_contract),
                ),
                reason=None,
                success_reason={"action": "split"},
            )

    def test_success_multi_rejects_non_pipeline_row_elements(
        self,
        output_contract: SchemaContract,
    ) -> None:
        with pytest.raises(PluginContractViolation, match=r"rows\[1\].*PipelineRow"):
            TransformResult.success_multi(
                rows=[
                    _output_row(1, "alpha", output_contract),
                    cast(Any, {"record_id": 2, "result": "beta"}),
                ],
                success_reason={"action": "split"},
            )

    def test_success_empty_is_explicit_zero_row_success(self) -> None:
        success_reason: TransformSuccessReason = {"action": "filtered", "metadata": {"predicate": "keyword_filter"}}

        result = TransformResult.success_empty(success_reason=success_reason)

        assert result.status == "success"
        assert result.row is None
        assert result.rows == ()
        assert result.reason is None
        assert result.success_reason == success_reason
        assert result.is_multi_row is True
        assert result.has_output_data is True

    def test_error_result_carries_only_structured_reason_and_retryability(self) -> None:
        reason: TransformErrorReason = {"reason": "api_error", "error": "upstream unavailable"}

        result = TransformResult.error(reason, retryable=True)

        assert result.status == "error"
        assert result.row is None
        assert result.rows is None
        assert result.reason == reason
        assert result.retryable is True
        assert result.success_reason is None
        assert result.is_multi_row is False
        assert result.has_output_data is False
