# tests/core/landscape/test_recorder_contracts.py
"""Tests for RecorderFactory schema contract methods.

Phase 5 Unified Schema Contracts: Task 3 - RecorderFactory contract methods
for storing and retrieving schema contracts in the audit trail.
"""

from __future__ import annotations

from elspeth.contracts import (
    ContractAuditRecord,
    FieldContract,
    MissingFieldViolation,
    NodeType,
    SchemaContract,
    TypeMismatchViolation,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import nodes_table, runs_table, validation_errors_table

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestBeginRunWithSchemaContract:
    """Tests for begin_run() with schema_contract parameter."""

    def test_begin_run_stores_schema_contract_json(self) -> None:
        """begin_run() stores schema_contract_json when provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Create a schema contract
        contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="customer_id",
                    original_name="Customer ID",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
                FieldContract(
                    normalized_name="amount",
                    original_name="Amount",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

        run = factory.run_lifecycle.begin_run(
            config={"source": "test.csv"},
            canonical_version="v1",
            schema_contract=contract,
        )

        # Verify stored in database
        with db.connection() as conn:
            result = conn.execute(
                select(runs_table.c.schema_contract_json, runs_table.c.schema_contract_hash).where(runs_table.c.run_id == run.run_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.schema_contract_json is not None
        assert row.schema_contract_hash is not None
        assert row.schema_contract_hash == contract.version_hash()

        # Verify JSON can be restored
        restored = ContractAuditRecord.from_json(row.schema_contract_json)
        assert restored.mode == "FIXED"
        assert len(restored.fields) == 2

    def test_begin_run_without_schema_contract(self) -> None:
        """begin_run() leaves contract columns NULL when not provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(
            config={"source": "test.csv"},
            canonical_version="v1",
        )

        with db.connection() as conn:
            result = conn.execute(
                select(runs_table.c.schema_contract_json, runs_table.c.schema_contract_hash).where(runs_table.c.run_id == run.run_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.schema_contract_json is None
        assert row.schema_contract_hash is None


class TestUpdateRunContract:
    """Tests for update_run_contract() method."""

    def test_update_run_contract_stores_contract(self) -> None:
        """update_run_contract() stores contract after first-row inference."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Start run without contract
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # After processing first row, source infers schema
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="inferred_field",
                    original_name="Inferred Field",
                    python_type=str,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        factory.run_lifecycle.update_run_contract(run.run_id, contract)

        # Verify stored
        with db.connection() as conn:
            result = conn.execute(
                select(runs_table.c.schema_contract_json, runs_table.c.schema_contract_hash).where(runs_table.c.run_id == run.run_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.schema_contract_json is not None
        assert row.schema_contract_hash == contract.version_hash()


class TestUpdateNodeOutputContract:
    """Tests for update_node_output_contract() method.

    BUG FIX: c1v5 - Source output_contract recorded before it is built.
    This method allows updating a node's output_contract after first-row inference.
    """

    def test_update_node_output_contract_stores_contract(self) -> None:
        """update_node_output_contract() updates source node after inference."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Start run without contract
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register source node without output_contract (dynamic source)
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={"path": "test.csv"},
            schema_config=DYNAMIC_SCHEMA,
            output_contract=None,  # Not known at registration time
        )

        # After processing first valid row, source infers schema
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="customer_id",
                    original_name="Customer ID",
                    python_type=str,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Update the node's output_contract
        factory.data_flow.update_node_output_contract(run.run_id, "source_1", contract)

        # Verify stored
        with db.connection() as conn:
            result = conn.execute(
                select(nodes_table.c.output_contract_json).where(
                    (nodes_table.c.run_id == run.run_id) & (nodes_table.c.node_id == "source_1")
                )
            )
            row = result.fetchone()

        assert row is not None
        assert row.output_contract_json is not None

        # Verify can be restored
        audit_record = ContractAuditRecord.from_json(row.output_contract_json)
        assert audit_record.mode == "OBSERVED"
        assert len(audit_record.fields) == 1
        assert audit_record.fields[0].normalized_name == "customer_id"


class TestGetRunContract:
    """Tests for get_run_contract() method."""

    def test_get_run_contract_returns_stored_contract(self) -> None:
        """get_run_contract() returns the stored schema contract."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Store contract via begin_run
        original_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="id",
                    original_name="ID",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            schema_contract=original_contract,
        )

        # Retrieve it
        retrieved = factory.run_lifecycle.get_run_contract(run.run_id)

        assert retrieved is not None
        assert retrieved.mode == "FLEXIBLE"
        assert len(retrieved.fields) == 1
        assert retrieved.fields[0].normalized_name == "id"
        assert retrieved.fields[0].python_type is int
        assert retrieved.locked is True

    def test_get_run_contract_returns_none_when_not_stored(self) -> None:
        """get_run_contract() returns None when no contract is stored."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        result = factory.run_lifecycle.get_run_contract(run.run_id)

        assert result is None


class TestRegisterNodeWithContracts:
    """Tests for register_node() with input/output contracts."""

    def test_register_node_stores_input_contract(self) -> None:
        """register_node() stores input_contract_json when provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        input_contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="input_field",
                    original_name="Input Field",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
            input_contract=input_contract,
        )

        # Verify stored in database
        with db.connection() as conn:
            result = conn.execute(
                select(nodes_table.c.input_contract_json).where(
                    (nodes_table.c.node_id == node.node_id) & (nodes_table.c.run_id == run.run_id)
                )
            )
            row = result.fetchone()

        assert row is not None
        assert row.input_contract_json is not None

        # Verify can be restored
        restored = ContractAuditRecord.from_json(row.input_contract_json)
        assert restored.mode == "FIXED"
        assert len(restored.fields) == 1
        assert restored.fields[0].normalized_name == "input_field"

    def test_register_node_stores_output_contract(self) -> None:
        """register_node() stores output_contract_json when provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="output_field",
                    original_name="Output Field",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
            output_contract=output_contract,
        )

        # Verify stored
        with db.connection() as conn:
            result = conn.execute(
                select(nodes_table.c.output_contract_json).where(
                    (nodes_table.c.node_id == node.node_id) & (nodes_table.c.run_id == run.run_id)
                )
            )
            row = result.fetchone()

        assert row is not None
        assert row.output_contract_json is not None

    def test_register_node_stores_both_contracts(self) -> None:
        """register_node() can store both input and output contracts."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        input_contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="input",
                    original_name="Input",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
        )

        output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="output",
                    original_name="Output",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
            input_contract=input_contract,
            output_contract=output_contract,
        )

        # Verify both stored
        with db.connection() as conn:
            result = conn.execute(
                select(nodes_table.c.input_contract_json, nodes_table.c.output_contract_json).where(
                    (nodes_table.c.node_id == node.node_id) & (nodes_table.c.run_id == run.run_id)
                )
            )
            row = result.fetchone()

        assert row is not None
        assert row.input_contract_json is not None
        assert row.output_contract_json is not None

    def test_register_node_without_contracts(self) -> None:
        """register_node() leaves contract columns NULL when not provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
        )

        with db.connection() as conn:
            result = conn.execute(
                select(nodes_table.c.input_contract_json, nodes_table.c.output_contract_json).where(
                    (nodes_table.c.node_id == node.node_id) & (nodes_table.c.run_id == run.run_id)
                )
            )
            row = result.fetchone()

        assert row is not None
        assert row.input_contract_json is None
        assert row.output_contract_json is None


class TestGetNodeContracts:
    """Tests for get_node_contracts() method."""

    def test_get_node_contracts_returns_both_contracts(self) -> None:
        """get_node_contracts() returns (input, output) tuple."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        input_contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="in_field",
                    original_name="In Field",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
        )

        output_contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="out_field",
                    original_name="Out Field",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
            input_contract=input_contract,
            output_contract=output_contract,
        )

        input_result, output_result = factory.data_flow.get_node_contracts(run.run_id, node.node_id)

        assert input_result is not None
        assert input_result.mode == "FIXED"
        assert input_result.fields[0].normalized_name == "in_field"

        assert output_result is not None
        assert output_result.mode == "FLEXIBLE"
        assert output_result.fields[0].normalized_name == "out_field"

    def test_get_node_contracts_returns_none_tuple_when_not_stored(self) -> None:
        """get_node_contracts() returns (None, None) when no contracts stored."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
        )

        input_result, output_result = factory.data_flow.get_node_contracts(run.run_id, node.node_id)

        assert input_result is None
        assert output_result is None

    def test_get_node_contracts_partial_contracts(self) -> None:
        """get_node_contracts() handles partial contracts (only input or output)."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        output_only = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="result",
                    original_name="Result",
                    python_type=float,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_1",
            sequence=0,
            output_contract=output_only,
        )

        input_result, output_result = factory.data_flow.get_node_contracts(run.run_id, node.node_id)

        assert input_result is None
        assert output_result is not None
        assert output_result.mode == "OBSERVED"


class TestRecordValidationErrorWithContract:
    """Tests for record_validation_error() with contract_violation parameter."""

    def test_record_validation_error_stores_type_mismatch_violation(self) -> None:
        """record_validation_error() stores type mismatch violation details."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Create source node for FK constraint
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_1",
            sequence=0,
        )

        # Create a type mismatch violation
        violation = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount USD",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )

        error_id = factory.data_flow.record_validation_error(
            run_id=run.run_id,
            node_id="source_1",
            row_data={"amount": "not_a_number"},
            error="Type mismatch: expected int, got str",
            schema_mode="fixed",
            destination="quarantine",
            contract_violation=violation,
        )

        # Verify stored
        with db.connection() as conn:
            result = conn.execute(
                select(
                    validation_errors_table.c.violation_type,
                    validation_errors_table.c.normalized_field_name,
                    validation_errors_table.c.original_field_name,
                    validation_errors_table.c.expected_type,
                    validation_errors_table.c.actual_type,
                ).where(validation_errors_table.c.error_id == error_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.violation_type == "type_mismatch"
        assert row.normalized_field_name == "amount"
        assert row.original_field_name == "Amount USD"
        assert row.expected_type == "int"
        assert row.actual_type == "str"

    def test_record_validation_error_stores_missing_field_violation(self) -> None:
        """record_validation_error() stores missing field violation details."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_1",
            sequence=0,
        )

        violation = MissingFieldViolation(
            normalized_name="customer_id",
            original_name="Customer ID",
        )

        error_id = factory.data_flow.record_validation_error(
            run_id=run.run_id,
            node_id="source_1",
            row_data={},
            error="Missing required field: Customer ID",
            schema_mode="fixed",
            destination="discard",
            contract_violation=violation,
        )

        with db.connection() as conn:
            result = conn.execute(
                select(
                    validation_errors_table.c.violation_type,
                    validation_errors_table.c.normalized_field_name,
                    validation_errors_table.c.original_field_name,
                    validation_errors_table.c.expected_type,
                    validation_errors_table.c.actual_type,
                ).where(validation_errors_table.c.error_id == error_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.violation_type == "missing_field"
        assert row.normalized_field_name == "customer_id"
        assert row.original_field_name == "Customer ID"
        assert row.expected_type is None
        assert row.actual_type is None

    def test_record_validation_error_without_contract_violation(self) -> None:
        """record_validation_error() leaves contract columns NULL when no violation provided."""
        from sqlalchemy import select

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_1",
            sequence=0,
        )

        error_id = factory.data_flow.record_validation_error(
            run_id=run.run_id,
            node_id="source_1",
            row_data={"bad": "data"},
            error="Generic validation error",
            schema_mode="flexible",
            destination="discard",
        )

        with db.connection() as conn:
            result = conn.execute(
                select(
                    validation_errors_table.c.violation_type,
                    validation_errors_table.c.normalized_field_name,
                    validation_errors_table.c.original_field_name,
                    validation_errors_table.c.expected_type,
                    validation_errors_table.c.actual_type,
                ).where(validation_errors_table.c.error_id == error_id)
            )
            row = result.fetchone()

        assert row is not None
        assert row.violation_type is None
        assert row.normalized_field_name is None
        assert row.original_field_name is None
        assert row.expected_type is None
        assert row.actual_type is None


class TestContractIntegrityVerification:
    """Tests for contract integrity verification on retrieval."""

    def test_get_run_contract_verifies_hash(self) -> None:
        """get_run_contract() verifies hash integrity on retrieval."""
        from sqlalchemy import select

        from elspeth.core.landscape.schema import runs_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        contract = SchemaContract(
            mode="FIXED",
            fields=(
                FieldContract(
                    normalized_name="verified_field",
                    original_name="Verified Field",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            schema_contract=contract,
        )

        # Query database DIRECTLY to get the stored hash
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.schema_contract_hash).where(runs_table.c.run_id == run.run_id)).fetchone()
            assert result is not None, "Run not found in database"
            stored_hash = result[0]

        # Retrieve contract via factory
        retrieved = factory.run_lifecycle.get_run_contract(run.run_id)

        # Verify hash integrity: stored DB hash must match recomputed hash
        assert retrieved is not None
        computed_hash = retrieved.version_hash()
        assert stored_hash == computed_hash, f"Hash integrity violation: stored={stored_hash}, computed={computed_hash}"
        # Also verify against original (should match if storage is correct)
        assert computed_hash == contract.version_hash()

    def test_get_node_contracts_verifies_hash(self) -> None:
        """get_node_contracts() verifies hash integrity on retrieval."""
        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        contract = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract(
                    normalized_name="node_field",
                    original_name="Node Field",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
            ),
        )

        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="transform_1",
            sequence=1,
            input_contract=contract,
        )

        input_result, _ = factory.data_flow.get_node_contracts(run.run_id, node.node_id)

        # Hash should match
        assert input_result is not None
        assert input_result.version_hash() == contract.version_hash()
