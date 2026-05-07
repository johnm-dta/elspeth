# tests/integration/audit/test_contract_audit.py
"""End-to-end integration tests for contract audit trail.

These tests verify the full integration of:
1. Schema contract recording in the audit trail during pipeline execution
2. Validation errors including contract violation details
3. Contract round-trip through audit trail with full fidelity

Per CLAUDE.md Test Path Integrity: These tests use production code paths
(CSVSource, RecorderFactory, SchemaContract, PipelineRow) rather than
manual construction.

Migrated from tests/integration/test_contract_audit_integration.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select

from elspeth.contracts import (
    ContractAuditRecord,
    FieldContract,
    NodeType,
    PipelineRow,
    SchemaContract,
    TypeMismatchViolation,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.schema import validation_errors_table
from elspeth.plugins.sources.csv_source import CSVSource
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext


class MockContext:
    """Minimal context for integration testing.

    Implements the PluginContext interface methods used by sources.
    Note: Cast to PluginContext when passing to source.load() for type safety.
    """

    def __init__(self) -> None:
        self.validation_errors: list[dict[str, Any]] = []

    def record_validation_error(self, **kwargs: object) -> None:
        self.validation_errors.append(dict(kwargs))


class TestFullAuditTrailWithContracts:
    """Test complete pipeline execution records contracts in audit trail."""

    def test_full_audit_trail_with_contracts(self, tmp_path: Path) -> None:
        """Complete pipeline records contract with mode=OBSERVED, locked=True.

        Creates CSV with messy headers, runs with dynamic schema and field normalization,
        then verifies the contract is properly stored in the audit trail.
        """
        # Create CSV with messy headers that will be normalized
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            'Amount USD',Customer ID
            100,C001
            200,C002
            300,C003
        """)
        )

        # Create source with dynamic schema and field normalization
        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = MockContext()

        # Load rows to trigger contract creation
        rows = list(source.load(cast("PluginContext", ctx)))

        # Verify all rows loaded successfully
        assert len(rows) == 3
        assert all(not r.is_quarantined for r in rows)

        # Get the contract from the source
        contract = source.get_schema_contract()
        assert contract is not None

        # Verify contract properties
        assert contract.mode == "OBSERVED"
        assert contract.locked is True

        # Verify field resolution preserved (original_name vs normalized_name)
        amount_field = contract.get_field("amount_usd")
        assert amount_field is not None
        assert amount_field.normalized_name == "amount_usd"
        assert amount_field.original_name == "'Amount USD'"
        assert amount_field.source == "inferred"

        customer_field = contract.get_field("customer_id")
        assert customer_field is not None
        assert customer_field.normalized_name == "customer_id"
        assert customer_field.original_name == "Customer ID"
        assert customer_field.source == "inferred"

        # Store contract in audit trail via RecorderFactory
        db = make_landscape_db()
        factory = make_factory(db)

        # Begin run with contract
        run = factory.run_lifecycle.begin_run(
            config={"test": "config"},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=contract,
        )

        # Verify contract stored in run
        stored_contract = factory.run_lifecycle.get_run_contract(run.run_id)
        assert stored_contract is not None
        assert stored_contract.mode == "OBSERVED"
        assert stored_contract.locked is True

        # Verify field resolution survives round-trip
        stored_amount = stored_contract.get_field("amount_usd")
        assert stored_amount is not None
        assert stored_amount.original_name == "'Amount USD'"

        stored_customer = stored_contract.get_field("customer_id")
        assert stored_customer is not None
        assert stored_customer.original_name == "Customer ID"

    def test_dynamic_contract_update_after_begin(self, tmp_path: Path) -> None:
        """Contract can be updated after run begins (for dynamic schemas).

        For dynamic schemas, the contract is inferred from the first row
        and then updated in the audit trail via update_run_contract().
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            field_a,field_b
            value1,123
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = MockContext()

        # Begin run without contract (simulates dynamic schema)
        db = make_landscape_db()
        factory = make_factory(db)

        run = factory.run_lifecycle.begin_run(
            config={"test": "config"},
            canonical_version="sha256-rfc8785-v1",
            # No schema_contract initially
        )

        # Verify no contract yet
        assert factory.run_lifecycle.get_run_contract(run.run_id) is None

        # Load rows to infer contract
        rows = list(source.load(cast("PluginContext", ctx)))
        assert len(rows) == 1

        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True

        # Update run with inferred contract
        factory.run_lifecycle.update_run_contract(run.run_id, contract)

        # Verify contract now stored
        stored_contract = factory.run_lifecycle.get_run_contract(run.run_id)
        assert stored_contract is not None
        assert stored_contract.mode == "OBSERVED"
        assert len(stored_contract.fields) == 2


class TestValidationErrorWithContractDetails:
    """Test validation errors include contract violation details."""

    def test_validation_error_with_contract_details(self, tmp_path: Path) -> None:
        """Validation errors include violation_type and field details.

        Creates CSV with type mismatch, uses strict schema, verifies
        the validation error has structured contract violation data.
        """
        # Create CSV with type mismatch (string "not_int" in int field)
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,amount
            1,100
            2,not_int
            3,300
        """)
        )

        # Create source with strict schema expecting ints
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
        ctx = MockContext()

        # Load rows - one should be quarantined
        rows = list(source.load(cast("PluginContext", ctx)))

        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        assert len(valid_rows) == 2
        assert len(quarantined_rows) == 1

        # Verify validation error was recorded in context
        assert len(ctx.validation_errors) == 1
        error = ctx.validation_errors[0]
        assert "not_int" in str(error) or "amount" in str(error)

        # Now test storing validation error with contract violation in audit trail
        setup = make_recorder_with_run(canonical_version="sha256-rfc8785-v1")
        db, run_id = setup.db, setup.run_id
        source_node_id = setup.source_node_id

        # Create a contract violation for testing
        violation = TypeMismatchViolation(
            normalized_name="amount",
            original_name="amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_int",
        )

        # Record validation error with contract violation
        error_id = setup.data_flow.record_validation_error(
            run_id=run_id,
            node_id=source_node_id,
            row_data={"id": 2, "amount": "not_int"},
            error="Type mismatch: expected int, got str for field 'amount'",
            schema_mode="fixed",
            destination="quarantine",
            contract_violation=violation,
        )

        assert error_id is not None

        # Query validation_errors_table directly to verify contract details
        query = select(validation_errors_table).where(validation_errors_table.c.error_id == error_id)
        with db.engine.connect() as conn:
            result = conn.execute(query).fetchone()

        assert result is not None
        assert result.violation_type == "type_mismatch"
        assert result.normalized_field_name == "amount"
        assert result.original_field_name == "amount"
        assert result.expected_type == "int"
        assert result.actual_type == "str"

    def test_validation_error_without_contract_violation(self) -> None:
        """Validation errors work without contract violation."""
        db = make_landscape_db()
        factory = make_factory(db)

        run = factory.run_lifecycle.begin_run(
            config={"test": "config"},
            canonical_version="sha256-rfc8785-v1",
        )

        # Record validation error without contract_violation
        error_id = factory.data_flow.record_validation_error(
            run_id=run.run_id,
            node_id=None,
            row_data={"field": "value"},
            error="Generic validation error",
            schema_mode="fixed",
            destination="discard",
            # No contract_violation
        )

        assert error_id is not None

        # Verify error recorded with NULL contract fields
        query = select(validation_errors_table).where(validation_errors_table.c.error_id == error_id)
        with db.engine.connect() as conn:
            result = conn.execute(query).fetchone()

        assert result is not None
        assert result.violation_type is None
        assert result.normalized_field_name is None
        assert result.expected_type is None
        assert result.actual_type is None


class TestContractSurvivesAuditRoundTrip:
    """Test contract can be restored from audit trail with full fidelity."""

    def test_contract_survives_audit_round_trip(self, tmp_path: Path) -> None:
        """Contract can be restored from audit trail with full fidelity.

        Execute pipeline, get contract from audit trail, verify hash matches,
        restore contract, verify PipelineRow dual-name access works.
        """
        # Create source with messy headers
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            'Amount USD',Customer ID,Order Date
            100,C001,2024-01-15
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = MockContext()

        # Load rows to create contract
        rows = list(source.load(cast("PluginContext", ctx)))
        assert len(rows) == 1
        assert not rows[0].is_quarantined

        original_contract = source.get_schema_contract()
        assert original_contract is not None

        # Store version_hash for integrity verification
        original_hash = original_contract.version_hash()

        # Store contract in audit trail
        db = make_landscape_db()
        factory = make_factory(db)

        run = factory.run_lifecycle.begin_run(
            config={"test": "config"},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=original_contract,
        )

        # Restore contract from audit trail
        restored_contract = factory.run_lifecycle.get_run_contract(run.run_id)
        assert restored_contract is not None

        # Verify hash matches (integrity verification)
        restored_hash = restored_contract.version_hash()
        assert restored_hash == original_hash

        # Verify all fields match
        assert len(restored_contract.fields) == len(original_contract.fields)
        assert restored_contract.mode == original_contract.mode
        assert restored_contract.locked == original_contract.locked

        # Verify field details match
        for orig_field in original_contract.fields:
            restored_field = restored_contract.get_field(orig_field.normalized_name)
            assert restored_field is not None
            assert restored_field.original_name == orig_field.original_name
            assert restored_field.python_type == orig_field.python_type
            assert restored_field.required == orig_field.required
            assert restored_field.source == orig_field.source

        # Verify PipelineRow dual-name access works with restored contract
        row_data = rows[0].row  # Get the normalized data
        pipeline_row = PipelineRow(row_data, restored_contract)

        # Access by normalized name
        assert pipeline_row["amount_usd"] == "100"
        assert pipeline_row.amount_usd == "100"

        # Access by original name
        assert pipeline_row["'Amount USD'"] == "100"

        # Access other fields
        assert pipeline_row["customer_id"] == "C001"
        assert pipeline_row["Customer ID"] == "C001"

    def test_contract_audit_record_round_trip(self) -> None:
        """ContractAuditRecord round-trip preserves all fields."""
        # Create a contract with multiple field types
        original_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="id",
                    original_name="ID",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
                FieldContract(
                    normalized_name="amount_usd",
                    original_name="'Amount USD'",
                    python_type=float,
                    required=True,
                    source="inferred",
                ),
                FieldContract(
                    normalized_name="is_active",
                    original_name="Is Active?",
                    python_type=bool,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Convert to audit record
        audit_record = ContractAuditRecord.from_contract(original_contract)

        # Serialize to JSON
        json_str = audit_record.to_json()

        # Restore from JSON
        restored_record = ContractAuditRecord.from_json(json_str)

        # Convert back to SchemaContract
        restored_contract = restored_record.to_schema_contract()

        # Verify hash integrity
        assert restored_contract.version_hash() == original_contract.version_hash()

        # Verify all fields
        assert len(restored_contract.fields) == 3

        id_field = restored_contract.get_field("id")
        assert id_field is not None
        assert id_field.python_type is int
        assert id_field.required is True
        assert id_field.source == "declared"

        amount_field = restored_contract.get_field("amount_usd")
        assert amount_field is not None
        assert amount_field.original_name == "'Amount USD'"
        assert amount_field.python_type is float

        active_field = restored_contract.get_field("is_active")
        assert active_field is not None
        assert active_field.original_name == "Is Active?"
        assert active_field.python_type is bool

    def test_node_contracts_round_trip(self) -> None:
        """Node input/output contracts survive audit trail round-trip."""
        db = make_landscape_db()
        factory = make_factory(db)

        run = factory.run_lifecycle.begin_run(
            config={"test": "config"},
            canonical_version="sha256-rfc8785-v1",
        )

        # Create input and output contracts
        input_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="raw_data",
                    original_name="Raw Data",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

        output_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="raw_data",
                    original_name="Raw Data",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
                FieldContract(
                    normalized_name="processed_result",
                    original_name="processed_result",
                    python_type=str,
                    required=True,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Register node with contracts
        schema_config = SchemaConfig(mode="flexible", fields=None)
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=schema_config,
            input_contract=input_contract,
            output_contract=output_contract,
        )

        # Retrieve contracts
        retrieved_input, retrieved_output = factory.data_flow.get_node_contracts(run.run_id, node.node_id)

        # Verify input contract
        assert retrieved_input is not None
        assert retrieved_input.mode == "FLEXIBLE"
        assert len(retrieved_input.fields) == 1
        assert retrieved_input.get_field("raw_data") is not None

        # Verify output contract
        assert retrieved_output is not None
        assert retrieved_output.mode == "OBSERVED"
        assert len(retrieved_output.fields) == 2
        assert retrieved_output.get_field("raw_data") is not None
        assert retrieved_output.get_field("processed_result") is not None


class TestContractWithCheckpointRegistry:
    """Test PipelineRow restoration using contract registry pattern."""

    def test_pipeline_row_from_checkpoint_with_registry(self, tmp_path: Path) -> None:
        """PipelineRow can be restored from checkpoint using contract registry."""
        # Create source with contract
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            Product Name,Unit Price
            Widget,9.99
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            }
        )
        ctx = MockContext()

        rows = list(source.load(cast("PluginContext", ctx)))
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
        assert restored_row["product_name"] == original_row["product_name"]
        assert restored_row["unit_price"] == original_row["unit_price"]

        # Verify dual-name access works
        assert restored_row["Product Name"] == "Widget"
        assert restored_row.product_name == "Widget"

        # Verify contract reference
        assert restored_row.contract is contract

    def test_contract_registry_from_audit_trail(self, tmp_path: Path) -> None:
        """Contract registry can be built from audit trail for checkpoint restore."""
        # Create and store multiple contracts
        db = make_landscape_db()
        factory = make_factory(db)

        contract1 = SchemaContract(
            mode="OBSERVED",
            fields=(FieldContract("field_a", "Field A", str, True, "declared"),),
            locked=True,
        )

        contract2 = SchemaContract(
            mode="FLEXIBLE",
            fields=(FieldContract("field_b", "Field B", int, False, "inferred"),),
            locked=True,
        )

        # Store contracts in audit trail (via runs)
        run1 = factory.run_lifecycle.begin_run(
            config={"run": 1},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=contract1,
        )

        run2 = factory.run_lifecycle.begin_run(
            config={"run": 2},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=contract2,
        )

        # Build registry from audit trail
        registry: dict[str, SchemaContract] = {}

        restored1 = factory.run_lifecycle.get_run_contract(run1.run_id)
        if restored1:
            registry[restored1.version_hash()] = restored1

        restored2 = factory.run_lifecycle.get_run_contract(run2.run_id)
        if restored2:
            registry[restored2.version_hash()] = restored2

        # Verify registry can look up by hash
        assert contract1.version_hash() in registry
        assert contract2.version_hash() in registry

        # Verify contracts are correct
        assert registry[contract1.version_hash()].mode == "OBSERVED"
        assert registry[contract2.version_hash()].mode == "FLEXIBLE"
