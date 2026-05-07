from __future__ import annotations

import json

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory, make_landscape_db, make_recorder_with_run, register_test_node

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, RecorderFactory]:
    setup = make_recorder_with_run(run_id=run_id, source_node_id="source-0", source_plugin_name="csv")
    return setup.db, setup.factory


def _setup_with_token(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, RecorderFactory]:
    db, factory = _setup(run_id=run_id)
    register_test_node(factory.data_flow, run_id, "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
    factory.data_flow.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1")
    factory.data_flow.create_token("row-1", token_id="tok-1")
    return db, factory


class TestRecordValidationError:
    """Tests for DataFlowRepository.record_validation_error."""

    def test_returns_error_id_with_verr_prefix(self):
        _db, factory = _setup()
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"name": "alice", "age": 30},
            error="Field 'age' expected str, got int",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")

    def test_roundtrip_via_get_validation_errors_for_row(self):
        _db, factory = _setup()
        row_data = {"name": "alice", "age": 30}
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Field 'age' expected str, got int",
            schema_mode="strict",
            destination="quarantine",
        )
        row_hash = stable_hash(row_data)
        errors = factory.data_flow.get_validation_errors_for_row("run-1", row_hash)
        assert len(errors) == 1
        record = errors[0]
        assert record.error_id == error_id
        assert record.run_id == "run-1"
        assert record.node_id == "source-0"
        assert record.row_hash == row_hash
        assert record.error == "Field 'age' expected str, got int"
        assert record.schema_mode == "strict"
        assert record.destination == "quarantine"
        assert record.created_at is not None

    def test_stores_row_data_as_json(self):
        _db, factory = _setup()
        row_data = {"x": 1, "y": "hello"}
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="bad field",
            schema_mode="observed",
            destination="quarantine",
        )
        row_hash = stable_hash(row_data)
        errors = factory.data_flow.get_validation_errors_for_row("run-1", row_hash)
        assert len(errors) == 1
        parsed = json.loads(errors[0].row_data_json)
        assert parsed["x"] == 1
        assert parsed["y"] == "hello"

    def test_multiple_errors_for_same_row(self):
        _db, factory = _setup()
        row_data = {"name": "bob"}
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="error one",
            schema_mode="strict",
            destination="quarantine",
        )
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="error two",
            schema_mode="strict",
            destination="quarantine",
        )
        row_hash = stable_hash(row_data)
        errors = factory.data_flow.get_validation_errors_for_row("run-1", row_hash)
        assert len(errors) == 2
        error_messages = {e.error for e in errors}
        assert error_messages == {"error one", "error two"}

    def test_unique_error_ids_per_call(self):
        _db, factory = _setup()
        row_data = {"k": "v"}
        id1 = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="err a",
            schema_mode="observed",
            destination="quarantine",
        )
        id2 = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="err b",
            schema_mode="observed",
            destination="quarantine",
        )
        assert id1 != id2
        assert id1.startswith("verr_")
        assert id2.startswith("verr_")

    def test_with_contract_violation_stores_in_db(self):
        from sqlalchemy import select

        from elspeth.contracts.errors import TypeMismatchViolation
        from elspeth.core.landscape.schema import validation_errors_table

        db, factory = _setup()
        row_data = {"amount": "not_a_number"}
        violation = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Type mismatch on 'amount'",
            schema_mode="strict",
            destination="quarantine",
            contract_violation=violation,
        )
        assert error_id.startswith("verr_")
        # Verify contract violation fields are stored in DB
        with db.engine.connect() as conn:
            row = conn.execute(select(validation_errors_table).where(validation_errors_table.c.error_id == error_id)).one()
        assert row.violation_type == "type_mismatch"
        assert row.normalized_field_name == "amount"
        assert row.original_field_name == "Amount"
        assert row.expected_type == "int"
        assert row.actual_type == "str"

    def test_without_contract_violation_db_fields_are_null(self):
        from sqlalchemy import select

        from elspeth.core.landscape.schema import validation_errors_table

        db, factory = _setup()
        row_data = {"a": 1}
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="some error",
            schema_mode="observed",
            destination="quarantine",
        )
        with db.engine.connect() as conn:
            row = conn.execute(select(validation_errors_table).where(validation_errors_table.c.error_id == error_id)).one()
        assert row.violation_type is None
        assert row.normalized_field_name is None
        assert row.original_field_name is None
        assert row.expected_type is None
        assert row.actual_type is None


class TestRecordValidationErrorNonCanonicalData:
    """Tests for record_validation_error with non-canonical row data (repr fallback)."""

    def test_nan_in_row_data(self):
        _db, factory = _setup()
        row_data = {"value": float("nan")}
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="NaN not allowed",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        # The row should still be recorded (via repr fallback) — query by run
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1
        assert errors[0].error_id == error_id
        # row_data_json should contain something (repr fallback)
        assert errors[0].row_data_json is not None
        assert len(errors[0].row_data_json) > 0

    def test_infinity_in_row_data(self):
        _db, factory = _setup()
        row_data = {"value": float("inf")}
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Infinity not allowed",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1
        assert errors[0].row_data_json is not None

    def test_negative_infinity_in_row_data(self):
        _db, factory = _setup()
        row_data = {"value": float("-inf")}
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Negative infinity not allowed",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1

    def test_list_as_row_data(self):
        _db, factory = _setup()
        row_data = [1, 2, 3]
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Expected dict, got list",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1
        assert errors[0].row_data_json is not None

    def test_string_as_row_data(self):
        _db, factory = _setup()
        row_data = "not a dict at all"
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="Expected dict, got str",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1

    def test_none_as_row_data(self):
        _db, factory = _setup()
        error_id = factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=None,
            error="Row data was None",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 1


class TestRecordTransformError:
    """Tests for DataFlowRepository.record_transform_error."""

    def test_returns_error_id_with_terr_prefix(self):
        _db, factory = _setup_with_token()
        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "validation_failed", "field": "amount", "error": "ZeroDivisionError"},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

    def test_roundtrip_via_get_transform_errors_for_token(self):
        _db, factory = _setup_with_token()
        row_data = {"name": "test"}
        error_details = {
            "reason": "validation_failed",
            "field": "amount",
            "error": "ZeroDivisionError: division by zero",
        }
        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data=row_data,
            error_details=error_details,
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert len(errors) == 1
        record = errors[0]
        assert record.error_id == error_id
        assert record.run_id == "run-1"
        assert record.token_id == "tok-1"
        assert record.transform_id == "transform-1"
        assert record.destination == "quarantine"
        assert record.created_at is not None
        parsed_details = json.loads(record.error_details_json)
        assert parsed_details["reason"] == "validation_failed"
        assert parsed_details["field"] == "amount"

    def test_stores_row_hash(self):
        _db, factory = _setup_with_token()
        row_data = {"name": "test"}
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data=row_data,
            error_details={"reason": "invalid_input", "field": "date", "error": "ValueError"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert len(errors) == 1
        expected_hash = stable_hash(row_data)
        assert errors[0].row_hash == expected_hash

    def test_stores_row_data_json(self):
        _db, factory = _setup_with_token()
        row_data = {"name": "test", "value": 42}
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data=row_data,
            error_details={"reason": "type_mismatch", "field": "value", "error": "OverflowError"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert len(errors) == 1
        parsed = json.loads(errors[0].row_data_json)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_multiple_errors_for_same_token(self):
        _db, factory = _setup_with_token()
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "api_error", "field": "f1", "error": "Error A"},
            destination="quarantine",
        )
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "network_error", "field": "f2", "error": "Error B"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert len(errors) == 2
        reasons = {json.loads(e.error_details_json)["reason"] for e in errors}
        assert reasons == {"api_error", "network_error"}

    def test_unique_error_ids_per_call(self):
        _db, factory = _setup_with_token()
        id1 = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "missing_field", "field": "x", "error": "A"},
            destination="quarantine",
        )
        id2 = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "permanent_error", "field": "y", "error": "B"},
            destination="quarantine",
        )
        assert id1 != id2
        assert id1.startswith("terr_")
        assert id2.startswith("terr_")


class TestGetValidationErrorsForRow:
    """Tests for DataFlowRepository.get_validation_errors_for_row."""

    def test_returns_errors_matching_row_hash(self):
        _db, factory = _setup()
        row_data = {"id": 1, "name": "alice"}
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="bad field",
            schema_mode="strict",
            destination="quarantine",
        )
        row_hash = stable_hash(row_data)
        errors = factory.data_flow.get_validation_errors_for_row("run-1", row_hash)
        assert len(errors) == 1
        assert errors[0].row_hash == row_hash

    def test_empty_for_unknown_hash(self):
        _db, factory = _setup()
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"exists": True},
            error="some error",
            schema_mode="observed",
            destination="quarantine",
        )
        errors = factory.data_flow.get_validation_errors_for_row("run-1", "nonexistent_hash_value")
        assert errors == []

    def test_does_not_cross_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)
        # Set up run-1
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        # Set up run-2
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        row_data = {"shared": "data"}
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data=row_data,
            error="run 1 error",
            schema_mode="strict",
            destination="quarantine",
        )
        factory.data_flow.record_validation_error(
            run_id="run-2",
            node_id="source-0",
            row_data=row_data,
            error="run 2 error",
            schema_mode="strict",
            destination="quarantine",
        )
        row_hash = stable_hash(row_data)
        run1_errors = factory.data_flow.get_validation_errors_for_row("run-1", row_hash)
        run2_errors = factory.data_flow.get_validation_errors_for_row("run-2", row_hash)
        assert len(run1_errors) == 1
        assert run1_errors[0].error == "run 1 error"
        assert len(run2_errors) == 1
        assert run2_errors[0].error == "run 2 error"

    def test_empty_when_no_errors_recorded(self):
        _db, factory = _setup()
        errors = factory.data_flow.get_validation_errors_for_row("run-1", "any_hash")
        assert errors == []


class TestGetValidationErrorsForRun:
    """Tests for DataFlowRepository.get_validation_errors_for_run."""

    def test_returns_all_errors_for_run(self):
        _db, factory = _setup()
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"a": 1},
            error="error one",
            schema_mode="strict",
            destination="quarantine",
        )
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"b": 2},
            error="error two",
            schema_mode="strict",
            destination="quarantine",
        )
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"c": 3},
            error="error three",
            schema_mode="observed",
            destination="quarantine",
        )
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 3
        error_messages = {e.error for e in errors}
        assert error_messages == {"error one", "error two", "error three"}

    def test_empty_when_no_errors_exist(self):
        _db, factory = _setup()
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert errors == []

    def test_does_not_cross_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        factory.data_flow.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"x": 1},
            error="run-1 error",
            schema_mode="strict",
            destination="quarantine",
        )
        factory.data_flow.record_validation_error(
            run_id="run-2",
            node_id="source-0",
            row_data={"y": 2},
            error="run-2 error",
            schema_mode="strict",
            destination="quarantine",
        )
        run1_errors = factory.data_flow.get_validation_errors_for_run("run-1")
        run2_errors = factory.data_flow.get_validation_errors_for_run("run-2")
        assert len(run1_errors) == 1
        assert run1_errors[0].error == "run-1 error"
        assert len(run2_errors) == 1
        assert run2_errors[0].error == "run-2 error"

    def test_ordered_by_created_at(self):
        _db, factory = _setup()
        for i in range(5):
            factory.data_flow.record_validation_error(
                run_id="run-1",
                node_id="source-0",
                row_data={f"field_{i}": i},
                error=f"error {i}",
                schema_mode="observed",
                destination="quarantine",
            )
        errors = factory.data_flow.get_validation_errors_for_run("run-1")
        assert len(errors) == 5
        timestamps = [e.created_at for e in errors]
        assert timestamps == sorted(timestamps)


class TestGetTransformErrorsForToken:
    """Tests for DataFlowRepository.get_transform_errors_for_token."""

    def test_returns_errors_for_token(self):
        _db, factory = _setup_with_token()
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "invalid_input", "field": "date", "error": "ValueError"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert len(errors) == 1
        assert errors[0].token_id == "tok-1"

    def test_empty_for_unknown_token(self):
        _db, factory = _setup_with_token()
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "test_error", "field": "f", "error": "E"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_token("tok-nonexistent")
        assert errors == []

    def test_does_not_return_other_tokens_errors(self):
        _db, factory = _setup_with_token()
        # Create a second token
        factory.data_flow.create_row("run-1", "source-0", 1, {"name": "other"}, row_id="row-2")
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "api_call_failed", "field": "f", "error": "E1"},
            destination="quarantine",
        )
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-2", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "other"},
            error_details={"reason": "llm_call_failed", "field": "f", "error": "E2"},
            destination="quarantine",
        )
        tok1_errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        tok2_errors = factory.data_flow.get_transform_errors_for_token("tok-2")
        assert len(tok1_errors) == 1
        assert json.loads(tok1_errors[0].error_details_json)["reason"] == "api_call_failed"
        assert len(tok2_errors) == 1
        assert json.loads(tok2_errors[0].error_details_json)["reason"] == "llm_call_failed"

    def test_empty_when_no_errors_recorded(self):
        _db, factory = _setup_with_token()
        errors = factory.data_flow.get_transform_errors_for_token("tok-1")
        assert errors == []


class TestGetTransformErrorsForRun:
    """Tests for DataFlowRepository.get_transform_errors_for_run."""

    def test_returns_all_errors_for_run(self):
        _db, factory = _setup_with_token()
        factory.data_flow.create_row("run-1", "source-0", 1, {"name": "other"}, row_id="row-2")
        factory.data_flow.create_token("row-2", token_id="tok-2")
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "api_error", "field": "f1", "error": "A"},
            destination="quarantine",
        )
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-2", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "other"},
            error_details={"reason": "network_error", "field": "f2", "error": "B"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_run("run-1")
        assert len(errors) == 2
        reasons = {json.loads(e.error_details_json)["reason"] for e in errors}
        assert reasons == {"api_error", "network_error"}

    def test_empty_when_no_errors_exist(self):
        _db, factory = _setup_with_token()
        errors = factory.data_flow.get_transform_errors_for_run("run-1")
        assert errors == []

    def test_does_not_cross_runs(self):
        db = make_landscape_db()
        factory = make_factory(db)
        # Set up run-1
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
        factory.data_flow.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        register_test_node(factory.data_flow, "run-1", "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
        factory.data_flow.create_row("run-1", "source-0", 0, {"n": "a"}, row_id="row-r1")
        factory.data_flow.create_token("row-r1", token_id="tok-r1")
        # Set up run-2
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        factory.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        register_test_node(factory.data_flow, "run-2", "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
        factory.data_flow.create_row("run-2", "source-0", 0, {"n": "b"}, row_id="row-r2")
        factory.data_flow.create_token("row-r2", token_id="tok-r2")
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-r1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"n": "a"},
            error_details={"reason": "api_error", "field": "f", "error": "E"},
            destination="quarantine",
        )
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-r2", run_id="run-2"),
            transform_id="transform-1",
            row_data={"n": "b"},
            error_details={"reason": "network_error", "field": "f", "error": "E"},
            destination="quarantine",
        )
        run1_errors = factory.data_flow.get_transform_errors_for_run("run-1")
        run2_errors = factory.data_flow.get_transform_errors_for_run("run-2")
        assert len(run1_errors) == 1
        assert json.loads(run1_errors[0].error_details_json)["reason"] == "api_error"
        assert len(run2_errors) == 1
        assert json.loads(run2_errors[0].error_details_json)["reason"] == "network_error"

    def test_ordered_by_created_at(self):
        _db, factory = _setup_with_token()
        valid_reasons = ["api_error", "network_error", "missing_field", "type_mismatch", "validation_failed"]
        for i in range(5):
            factory.data_flow.record_transform_error(
                ref=TokenRef(token_id="tok-1", run_id="run-1"),
                transform_id="transform-1",
                row_data={"idx": i},
                error_details={"reason": valid_reasons[i], "field": "f", "error": "E"},
                destination="quarantine",
            )
        errors = factory.data_flow.get_transform_errors_for_run("run-1")
        assert len(errors) == 5
        timestamps = [e.created_at for e in errors]
        assert timestamps == sorted(timestamps)


# ===========================================================================
# Bug 7.4: record_transform_error NaN fallback
# ===========================================================================


class TestRecordTransformErrorNaNFallback:
    """Bug 7.4: NaN in error_details must not crash record_transform_error()."""

    def test_nan_in_error_details_does_not_crash(self):
        """error_details containing NaN should use repr-based fallback."""
        _db, factory = _setup_with_token()
        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "validation_failed", "field": "ratio", "error": "nan_result", "value": float("nan")},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

        # Verify the error was stored
        errors = factory.data_flow.get_transform_errors_for_run("run-1")
        assert len(errors) == 1
        # The error_details_json should contain the fallback metadata
        details = json.loads(errors[0].error_details_json)
        assert details["__non_canonical__"] is True
        assert "repr" in details

    def test_infinity_in_error_details_does_not_crash(self):
        """error_details containing Infinity should use repr-based fallback."""
        _db, factory = _setup_with_token()
        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "type_mismatch", "field": "big", "error": "inf", "value": float("inf")},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

    def test_normal_error_details_still_uses_canonical_json(self):
        """Normal error_details should still use canonical JSON (no fallback)."""
        _db, factory = _setup_with_token()
        factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "invalid_input", "field": "date", "error": "invalid format"},
            destination="quarantine",
        )
        errors = factory.data_flow.get_transform_errors_for_run("run-1")
        assert len(errors) == 1
        details = json.loads(errors[0].error_details_json)
        # Normal JSON - no fallback metadata
        assert "__non_canonical__" not in details
        assert details["reason"] == "invalid_input"


# ===========================================================================
# Regression tests: P1-2026-02-14 record_transform_error cross-run prevention
# ===========================================================================


def _setup_two_runs_with_transform() -> tuple[LandscapeDB, RecorderFactory]:
    """Set up a shared database with two runs, each with source + transform nodes."""
    db = make_landscape_db()
    factory = make_factory(db)
    for run_id in ("run-A", "run-B"):
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        register_test_node(factory.data_flow, run_id, "transform-1", node_type=NodeType.TRANSFORM, plugin_name="transform")
    return db, factory


class TestRecordTransformErrorCrossRunPrevention:
    """P1-2026-02-14: record_transform_error must validate token/run ownership.

    These tests verify that recording a transform error under the wrong run_id
    raises AuditIntegrityError immediately, rather than silently corrupting
    the audit trail.
    """

    def test_rejects_wrong_run_id(self):
        """record_transform_error must crash if token belongs to a different run."""
        _db, factory = _setup_two_runs_with_transform()

        # Create row and token in run-A
        factory.data_flow.create_row("run-A", "source-0", 0, {"name": "test"}, row_id="row-A")
        factory.data_flow.create_token("row-A", token_id="tok-A")

        # Attempt to record error under run-B -- must crash
        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            factory.data_flow.record_transform_error(
                ref=TokenRef(token_id="tok-A", run_id="run-B"),
                transform_id="transform-1",
                row_data={"name": "test"},
                error_details={"reason": "test_error", "field": "f", "error": "E"},
                destination="quarantine",
            )

    def test_accepts_correct_run_id(self):
        """record_transform_error must succeed when run_id matches token ownership."""
        _db, factory = _setup_with_token(run_id="run-1")

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "test_error", "field": "f", "error": "E"},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

    def test_rejects_nonexistent_token(self):
        """record_transform_error must crash if token does not exist."""
        _db, factory = _setup_with_token(run_id="run-1")

        with pytest.raises(AuditIntegrityError, match="does not exist"):
            factory.data_flow.record_transform_error(
                ref=TokenRef(token_id="nonexistent-token", run_id="run-1"),
                transform_id="transform-1",
                row_data={"name": "test"},
                error_details={"reason": "test_error", "field": "f", "error": "E"},
                destination="quarantine",
            )

    def test_schema_composite_fk_prevents_cross_run_error(self):
        """Schema composite FK on transform_errors must reject mismatched (token_id, run_id).

        Even if the application-level check were bypassed, the database constraint
        should reject the insert.
        """
        from sqlalchemy.exc import IntegrityError as SAIntegrityError

        from elspeth.core.landscape._helpers import generate_id, now
        from elspeth.core.landscape.schema import transform_errors_table

        _db, factory = _setup_two_runs_with_transform()

        # Create row and token in run-A
        factory.data_flow.create_row("run-A", "source-0", 0, {"name": "test"}, row_id="row-A")
        factory.data_flow.create_token("row-A", token_id="tok-A")

        # Try to insert directly into transform_errors with mismatched (token_id, run_id)
        # tok-A belongs to run-A, but we try to record under run-B
        # The composite FK should reject this
        row_data = {"name": "test"}
        with pytest.raises(SAIntegrityError), _db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id=f"terr_{generate_id()[:12]}",
                    run_id="run-B",
                    token_id="tok-A",
                    transform_id="transform-1",
                    row_hash=stable_hash(row_data),
                    row_data_json="{}",
                    error_details_json="{}",
                    destination="quarantine",
                    created_at=now(),
                )
            )
