# tests/integration/plugins/transforms/test_contract.py
"""Integration tests for schema contract propagation through transforms.

These tests verify the end-to-end integration of:
1. Source to sink contract flow with original header restoration
2. Contract preservation when transforms add fields
3. PipelineRow dual-name access (normalized and original)

Per CLAUDE.md Test Path Integrity: These tests use production code paths
(CSVSource, CSVSink, SchemaContract, PipelineRow) rather than manual construction.

Migrated from tests/integration/test_transform_contract_integration.py
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import (
    FieldContract,
    PipelineRow,
    SchemaContract,
    propagate_contract,
)
from elspeth.contracts.errors import ContractMergeError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.sinks.csv_sink import CSVSink

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import ValidationErrorToken
from elspeth.plugins.sources.csv_source import CSVSource


class _TestablePluginContext(PluginContext):
    """PluginContext subclass with validation error tracking for tests."""

    def __init__(self) -> None:
        super().__init__(
            run_id="test-run-001",
            config={},
        )
        self.validation_errors: list[dict[str, object]] = []

    def record_validation_error(
        self,
        row: object,
        error: str,
        schema_mode: str,
        destination: str,
    ) -> "ValidationErrorToken":
        """Override to track validation errors for test assertions."""
        from elspeth.contracts.plugin_context import ValidationErrorToken

        self.validation_errors.append(
            {
                "row": row,
                "error": error,
                "schema_mode": schema_mode,
                "destination": destination,
            }
        )
        # Return a mock token - tests don't have landscape
        return ValidationErrorToken(
            row_id="test-row",
            node_id=self.node_id or "test-node",
            destination=destination,
        )


def make_test_context() -> _TestablePluginContext:
    """Create a test context for integration tests."""
    return _TestablePluginContext()


class TestSourceToSinkContractFlow:
    """Test contract propagation from source to sink."""

    def test_original_headers_restored_in_output(self, tmp_path: Path) -> None:
        """Source with normalize_fields creates contract that sink uses for original headers."""
        # Create input CSV with messy headers
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("First Name!,Last Name@,Email Address\nJohn,Doe,john@example.com\n")

        # Configure source with normalize_fields
        source = CSVSource(
            {
                "path": str(input_csv),
                "normalize_fields": True,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        # Load rows and get contract
        ctx = make_test_context()
        rows = list(source.load(ctx))
        assert len(rows) == 1

        source_row = rows[0]
        assert not source_row.is_quarantined
        contract = source_row.contract
        assert contract is not None

        # Verify contract has both normalized and original names
        first_name_field = contract.get_field("first_name")
        assert first_name_field is not None
        assert first_name_field.original_name == "First Name!"

        last_name_field = contract.get_field("last_name")
        assert last_name_field is not None
        assert last_name_field.original_name == "Last Name@"

        email_field = contract.get_field("email_address")
        assert email_field is not None
        assert email_field.original_name == "Email Address"

        # Configure sink with original headers mode
        output_csv = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_csv),
                "headers": "original",
                "schema": {"mode": "fixed", "fields": ["first_name: str", "last_name: str", "email_address: str"]},
            }
        )

        # Set the contract on the sink
        sink.set_output_contract(contract)

        # Write through sink
        sink.write([source_row.row], ctx)
        sink.flush()
        sink.close()

        # Verify output has original headers
        output_content = output_csv.read_text()
        assert "First Name!" in output_content
        assert "Last Name@" in output_content
        assert "Email Address" in output_content
        # Verify data is present
        assert "John" in output_content
        assert "Doe" in output_content
        assert "john@example.com" in output_content

    def test_pipeline_row_dual_name_access(self, tmp_path: Path) -> None:
        """PipelineRow allows access by both original and normalized names."""
        # Create input with messy header
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Amount (USD),Currency Code\n100.50,USD\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "normalize_fields": True,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        contract = source_row.contract
        assert contract is not None

        # Create PipelineRow
        pipeline_row = PipelineRow(source_row.row, contract)

        # Access by normalized name
        assert pipeline_row["amount_usd"] == "100.50"

        # Access by original name
        assert pipeline_row["Amount (USD)"] == "100.50"

        # Attribute access (normalized only)
        assert pipeline_row.amount_usd == "100.50"

        # Similarly for currency code
        assert pipeline_row["currency_code"] == "USD"
        assert pipeline_row["Currency Code"] == "USD"
        assert pipeline_row.currency_code == "USD"

    def test_source_row_to_pipeline_row_conversion(self, tmp_path: Path) -> None:
        """SourceRow.to_pipeline_row() creates PipelineRow with contract."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Product Name,Unit Price\nWidget,9.99\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "normalize_fields": True,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]

        # Convert via to_pipeline_row
        pipeline_row = source_row.to_pipeline_row()

        # Verify dual access works
        assert pipeline_row["product_name"] == "Widget"
        assert pipeline_row["Product Name"] == "Widget"
        assert pipeline_row["unit_price"] == "9.99"
        assert pipeline_row["Unit Price"] == "9.99"


class TestContractPreservationThroughTransforms:
    """Test that contracts are preserved when transforms modify rows."""

    def test_contract_with_added_field(self, tmp_path: Path) -> None:
        """Contract is updated when transform adds a field."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,value\nTest,123\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        input_contract = source_row.contract
        assert input_contract is not None

        # Simulate transform adding a field
        output_row = {**source_row.row, "computed": 456}

        # Propagate contract
        output_contract = propagate_contract(input_contract, output_row)

        # Original fields preserved (field names match CSV headers exactly when not normalized)
        assert output_contract.get_field("name") is not None
        assert output_contract.get_field("value") is not None

        # New field added with inferred type
        computed_field = output_contract.get_field("computed")
        assert computed_field is not None
        assert computed_field.python_type is int
        assert computed_field.source == "inferred"

    def test_contract_with_added_dict_field_uses_object_type(self, tmp_path: Path) -> None:
        """Contract propagation preserves added dict fields as object type."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("name,value\nTest,123\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        input_contract = source_row.contract
        assert input_contract is not None

        output_row = {**source_row.row, "metadata": {"source": "pipeline", "valid": True}}

        output_contract = propagate_contract(input_contract, output_row)

        metadata_field = output_contract.get_field("metadata")
        assert metadata_field is not None
        assert metadata_field.python_type is object
        assert metadata_field.source == "inferred"
        assert metadata_field.required is False

    def test_propagate_contract_preserves_original_names(self, tmp_path: Path) -> None:
        """propagate_contract preserves original names from source contract."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("Customer ID,Order Total\n1001,250.00\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "normalize_fields": True,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        input_contract = source_row.contract
        assert input_contract is not None

        # Simulate transform adding status field
        output_row = {**source_row.row, "processed": True}

        # Propagate contract
        output_contract = propagate_contract(input_contract, output_row)

        # Original names preserved
        customer_id_field = output_contract.get_field("customer_id")
        assert customer_id_field is not None
        assert customer_id_field.original_name == "Customer ID"

        order_total_field = output_contract.get_field("order_total")
        assert order_total_field is not None
        assert order_total_field.original_name == "Order Total"

        # New field has same original and normalized name
        processed_field = output_contract.get_field("processed")
        assert processed_field is not None
        assert processed_field.original_name == "processed"  # No original for transform-created fields

    def test_passthrough_transform_preserves_contract(self, tmp_path: Path) -> None:
        """Passthrough transform returns same contract instance."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("ID,Status\n1,active\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        input_contract = source_row.contract
        assert input_contract is not None

        # Passthrough - output row is same as input
        output_row = source_row.row

        # Propagate with transform_adds_fields=False (passthrough)
        output_contract = propagate_contract(input_contract, output_row, transform_adds_fields=False)

        # Same contract instance returned
        assert output_contract is input_contract


class TestContractWithSinkHeaderModes:
    """Test different header modes in sink output."""

    def test_normalized_headers_mode(self, tmp_path: Path) -> None:
        """headers: normalized uses Python identifier names."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("First Name,Last Name\nAlice,Smith\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "normalize_fields": True,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        contract = source_row.contract
        assert contract is not None

        # Sink with normalized headers (default)
        output_csv = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_csv),
                "headers": "normalized",
                "schema": {"mode": "fixed", "fields": ["first_name: str", "last_name: str"]},
            }
        )

        sink.set_output_contract(contract)
        sink.write([source_row.row], ctx)
        sink.flush()
        sink.close()

        output_content = output_csv.read_text()
        # Headers are normalized Python identifiers
        assert "first_name" in output_content
        assert "last_name" in output_content
        # Original headers NOT present
        assert "First Name" not in output_content

    def test_custom_headers_mapping(self, tmp_path: Path) -> None:
        """headers with custom mapping uses explicit names."""
        input_csv = tmp_path / "input.csv"
        input_csv.write_text("amount,currency\n100,USD\n")

        source = CSVSource(
            {
                "path": str(input_csv),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )

        ctx = make_test_context()
        rows = list(source.load(ctx))
        source_row = rows[0]
        contract = source_row.contract
        assert contract is not None

        # Sink with custom header mapping
        output_csv = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_csv),
                "headers": {"amount": "TRANSACTION_AMOUNT", "currency": "CURRENCY_CODE"},
                "schema": {"mode": "fixed", "fields": ["amount: str", "currency: str"]},
            }
        )

        sink.set_output_contract(contract)
        sink.write([source_row.row], ctx)
        sink.flush()
        sink.close()

        output_content = output_csv.read_text()
        # Custom headers used
        assert "TRANSACTION_AMOUNT" in output_content
        assert "CURRENCY_CODE" in output_content
        # Original lowercase headers NOT present as column headers
        lines = output_content.strip().split("\n")
        header_line = lines[0]
        assert "amount" not in header_line or "TRANSACTION_AMOUNT" in header_line


class TestContractMergeAtCoalesce:
    """Test contract merge behavior for coalesce points."""

    def test_contract_merge_compatible_types(self) -> None:
        """Contracts with same fields and types merge successfully."""
        contract_a = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract("id", "ID", int, True, "declared"),
                FieldContract("name", "Name", str, False, "inferred"),
            ),
            locked=True,
        )

        contract_b = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract("id", "ID", int, True, "declared"),
                FieldContract("status", "Status", str, False, "inferred"),
            ),
            locked=True,
        )

        merged = contract_a.merge(contract_b)

        # All fields from both contracts
        assert merged.get_field("id") is not None
        assert merged.get_field("name") is not None
        assert merged.get_field("status") is not None

        # Merged is locked
        assert merged.locked is True

    def test_contract_merge_type_conflict_raises(self) -> None:
        """Contracts with conflicting field types raise ContractMergeError."""
        contract_a = SchemaContract(
            mode="OBSERVED",
            fields=(FieldContract("amount", "Amount", int, True, "declared"),),
            locked=True,
        )

        contract_b = SchemaContract(
            mode="OBSERVED",
            fields=(FieldContract("amount", "Amount", str, True, "declared"),),
            locked=True,
        )

        with pytest.raises(ContractMergeError) as exc_info:
            contract_a.merge(contract_b)

        assert "amount" in str(exc_info.value)
