"""Tests for SourceRow with SchemaContract integration."""

import inspect

import pytest

from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.testing import make_field


class TestSourceRowContractInvariant:
    """Valid SourceRow must have a contract — catches bugs at construction, not tokenization."""

    @pytest.fixture
    def sample_contract(self) -> SchemaContract:
        """Sample locked contract."""
        return SchemaContract(
            mode="FIXED",
            fields=(
                make_field("id", int, original_name="ID", required=True, source="declared"),
                make_field("name", str, original_name="Name", required=True, source="declared"),
            ),
            locked=True,
        )

    def test_valid_signature_requires_contract_without_default(self) -> None:
        """SourceRow.valid() should mechanically require contract at the API boundary."""
        contract_param = inspect.signature(SourceRow.valid).parameters["contract"]

        assert contract_param.kind is inspect.Parameter.KEYWORD_ONLY
        assert contract_param.default is inspect.Parameter.empty

    def test_valid_without_contract_raises(self) -> None:
        """SourceRow.valid() without contract raises TypeError.

        Bug fix: elspeth-a27e71979f. Previously, contract=None was accepted
        for valid rows, causing a crash later at tokenization. The invariant
        is now enforced in __post_init__ instead of failing later at
        to_pipeline_row().
        """
        with pytest.raises(TypeError, match="contract"):
            SourceRow.valid({"id": 1})

    def test_valid_with_contract(self, sample_contract: SchemaContract) -> None:
        """SourceRow.valid() with contract succeeds and carries the contract reference."""
        row_data = {"id": 1, "name": "Alice"}
        source_row = SourceRow.valid(row_data, contract=sample_contract)

        assert source_row.is_quarantined is False
        assert source_row.contract is sample_contract

    def test_quarantined_no_contract(self) -> None:
        """Quarantined rows don't carry contracts (they failed validation)."""
        source_row = SourceRow.quarantined(
            row={"bad": "data"},
            error="validation failed",
            destination="quarantine",
        )

        assert source_row.is_quarantined
        assert source_row.contract is None

    def test_to_pipeline_row(self, sample_contract: SchemaContract) -> None:
        """SourceRow can convert to PipelineRow."""
        row_data = {"id": 1, "name": "Alice"}
        source_row = SourceRow.valid(row_data, contract=sample_contract)

        pipeline_row = source_row.to_pipeline_row()

        assert isinstance(pipeline_row, PipelineRow)
        assert pipeline_row["id"] == 1
        assert pipeline_row["name"] == "Alice"

    def test_to_pipeline_row_raises_if_quarantined(self) -> None:
        """to_pipeline_row() raises for quarantined rows."""
        source_row = SourceRow.quarantined(
            row={"bad": "data"},
            error="failed",
            destination="quarantine",
        )

        with pytest.raises(ValueError, match="quarantined"):
            source_row.to_pipeline_row()
