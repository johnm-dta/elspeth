# tests/integration/plugins/sources/test_contract.py
"""Integration tests for source -> contract -> pipeline flow.

These tests verify the end-to-end integration of:
1. Source loading with schema validation
2. Contract creation and locking
3. PipelineRow dual-name access
4. Contract checkpoint serialization

Per CLAUDE.md Test Path Integrity: These tests use production code paths
(CSVSource, SchemaContract, PipelineRow) rather than manual construction.
"""

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.plugins.sources.csv_source import CSVSource

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import ValidationErrorToken


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


class TestSourceContractIntegration:
    """End-to-end tests for source contract integration."""

    def test_dynamic_schema_infer_and_lock(self, tmp_path: Path) -> None:
        """Dynamic schema infers types from first row and locks."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,amount,status
            1,100,active
            2,200,inactive
            3,300,active
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))

        # All rows valid
        assert len(rows) == 3
        assert all(not r.is_quarantined for r in rows)

        # Contract locked after first row
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
        assert contract.mode == "OBSERVED"

        # All rows have same contract
        for row in rows:
            assert row.contract is contract

    def test_dual_name_access(self, tmp_path: Path) -> None:
        """PipelineRow supports dual-name access (original and normalized)."""
        csv_file = tmp_path / "data.csv"
        # Use messy headers that will be normalized
        csv_file.write_text(
            dedent("""\
            Amount USD,Customer ID
            100,C001
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
                "normalize_fields": True,
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert not rows[0].is_quarantined

        pipeline_row = rows[0].to_pipeline_row()

        # Access by normalized name (dict-style)
        assert pipeline_row["amount_usd"] == "100"
        # Access by normalized name (attribute-style)
        assert pipeline_row.amount_usd == "100"

        # Access by original name
        assert pipeline_row["Amount USD"] == "100"

        # Similarly for customer_id
        assert pipeline_row["customer_id"] == "C001"
        assert pipeline_row.customer_id == "C001"
        assert pipeline_row["Customer ID"] == "C001"

    def test_strict_schema_validation(self, tmp_path: Path) -> None:
        """Strict schema validates and quarantines bad rows."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,amount
            1,100
            two,200
            3,300
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "amount: int"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))

        # Row 2 quarantined (id "two" not int)
        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        assert len(valid_rows) == 2
        assert len(quarantined_rows) == 1

        # Quarantined row has no contract
        assert quarantined_rows[0].contract is None

        # Valid rows have contracts
        for row in valid_rows:
            assert row.contract is not None

        # Validation error was recorded
        assert len(ctx.validation_errors) == 1
        assert "two" in str(ctx.validation_errors[0])

    def test_contract_survives_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """Contract can serialize for checkpoints."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,name
            1,Alice
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = make_test_context()

        # Consume the source to trigger contract creation
        list(source.load(ctx))
        contract = source.get_schema_contract()

        # Serialize and restore
        assert contract is not None
        checkpoint_data = contract.to_checkpoint_format()
        restored = SchemaContract.from_checkpoint(checkpoint_data)

        # Verify integrity
        assert restored.mode == contract.mode
        assert restored.locked == contract.locked
        assert len(restored.fields) == len(contract.fields)

        # Field contracts match
        for orig_field, restored_field in zip(
            sorted(contract.fields, key=lambda f: f.normalized_name),
            sorted(restored.fields, key=lambda f: f.normalized_name),
            strict=True,
        ):
            assert orig_field.normalized_name == restored_field.normalized_name
            assert orig_field.original_name == restored_field.original_name
            assert orig_field.python_type == restored_field.python_type
            assert orig_field.required == restored_field.required
            assert orig_field.source == restored_field.source

    def test_pipeline_row_checkpoint_round_trip(self, tmp_path: Path) -> None:
        """PipelineRow can serialize and restore via contract registry."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,value
            1,test_value
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1

        original_row = rows[0].to_pipeline_row()
        contract = source.get_schema_contract()
        assert contract is not None

        # Serialize pipeline row
        checkpoint_data = original_row.to_checkpoint_format()

        # Build contract registry (simulates checkpoint restore)
        contract_registry = {contract.version_hash(): contract}

        # Restore pipeline row
        restored_row = PipelineRow.from_checkpoint(checkpoint_data, contract_registry)

        # Verify data integrity
        assert restored_row["id"] == original_row["id"]
        assert restored_row["value"] == original_row["value"]
        assert restored_row.contract is contract

    def test_quarantined_row_cannot_convert_to_pipeline_row(self, tmp_path: Path) -> None:
        """Quarantined rows cannot be converted to PipelineRow."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,amount
            one,100
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "amount: int"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert rows[0].is_quarantined

        with pytest.raises(ValueError, match="Cannot convert quarantined row"):
            rows[0].to_pipeline_row()

    def test_contract_field_containment_check(self, tmp_path: Path) -> None:
        """PipelineRow supports 'in' operator for field checks."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            field_a,field_b
            val1,val2
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))
        pipeline_row = rows[0].to_pipeline_row()

        # Fields that exist
        assert "field_a" in pipeline_row
        assert "field_b" in pipeline_row

        # Fields that don't exist
        assert "field_c" not in pipeline_row
        assert "nonexistent" not in pipeline_row

    def test_contract_mode_flexible_with_declared_fields(self, tmp_path: Path) -> None:
        """FLEXIBLE mode infers first-row extras into the locked contract."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,name,extra_field
            1,Alice,bonus_data
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int"],  # Only id declared
                },
                "on_validation_failure": "discard",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert not rows[0].is_quarantined

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.mode == "FLEXIBLE"
        assert contract.locked is True

        # Contract contains declared and inferred first-row fields
        field_names = {field.normalized_name for field in contract.fields}
        assert field_names == {"id", "name", "extra_field"}

        # Declared field accessible via PipelineRow
        pipeline_row = rows[0].to_pipeline_row()
        assert pipeline_row["id"] == 1

        # FLEXIBLE mode allows access to extra fields and keeps them in contract
        assert "name" in pipeline_row
        assert pipeline_row["name"] == "Alice"
        assert "extra_field" in pipeline_row
        assert pipeline_row["extra_field"] == "bonus_data"

        # The underlying data contains all fields (same as to_dict)
        raw_data = pipeline_row.to_dict()
        assert raw_data["name"] == "Alice"
        assert raw_data["extra_field"] == "bonus_data"

    def test_empty_source_locks_contract(self, tmp_path: Path) -> None:
        """Contract is locked even when source has no valid rows."""
        csv_file = tmp_path / "data.csv"
        # All rows will be quarantined
        csv_file.write_text(
            dedent("""\
            id,amount
            one,100
            two,200
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "amount: int"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = make_test_context()

        rows = list(source.load(ctx))

        # All quarantined
        assert len(rows) == 2
        assert all(r.is_quarantined for r in rows)

        # Contract still exists and is locked
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
