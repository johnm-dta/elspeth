# tests/integration/plugins/sources/test_trust_boundary.py
"""Integration tests for the Tier 3 -> Tier 2 trust boundary at Sources.

Sources are the ONLY place in ELSPETH where type coercion is allowed.
External data (Tier 3) is zero-trust: Sources must either coerce it to valid
types or quarantine it. These tests verify:

1. Coercion works correctly (str -> int, str -> float, str -> bool)
2. Non-coercible data is quarantined, not crashed
3. Malformed input is handled gracefully
4. The Landscape audit trail records quarantine events
5. Source output satisfies its schema guarantees

Per CLAUDE.md Three-Tier Trust Model and the Coercion Rules table:
  Source -> allow_coercion=True (normalize external data at ingestion boundary)
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts.enums import NodeType
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import CANONICAL_VERSION
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.json_source import JSONSource

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import ValidationErrorToken


# ---------------------------------------------------------------------------
# Test-only helpers
# ---------------------------------------------------------------------------


class _TestablePluginContext(PluginContext):
    """PluginContext subclass with validation error tracking for tests.

    Used for tests that do not need real Landscape recording.
    For audit trail tests, use a real PluginContext wired to a recorder.
    """

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
        **kwargs: object,
    ) -> ValidationErrorToken:
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
        return ValidationErrorToken(
            row_id="test-row",
            node_id=self.node_id or "test-node",
            destination=destination,
        )


def _make_test_context() -> _TestablePluginContext:
    """Create a testable context without Landscape backing."""
    return _TestablePluginContext()


def _make_audited_context(
    factory: RecorderFactory,
    run_id: str,
    node_id: str | None = None,
) -> PluginContext:
    """Create a real PluginContext wired to a RecorderFactory.

    Delegates to the shared make_context() factory.

    This enables validation errors to be recorded to the Landscape
    audit database for full audit trail verification.
    """
    from tests.fixtures.factories import make_context

    return make_context(
        run_id=run_id,
        landscape=factory.plugin_audit_writer(),
        node_id=node_id,
    )


def _setup_run_and_node(
    factory: RecorderFactory,
) -> tuple[str, str]:
    """Create a run and source node in Landscape for FK integrity.

    The validation_errors table has a composite FK to nodes (node_id, run_id),
    so we must register both before recording validation errors.

    Returns:
        (run_id, node_id) tuple
    """
    run = factory.run_lifecycle.begin_run(
        config={"source": "test"},
        canonical_version=CANONICAL_VERSION,
    )
    node = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={"path": "test.csv"},
        schema_config=SchemaConfig(mode="observed", fields=None),
    )
    return run.run_id, node.node_id


# ---------------------------------------------------------------------------
# TestCSVSourceCoercion
# ---------------------------------------------------------------------------


class TestCSVSourceCoercion:
    """Tests that type coercion works at the source boundary.

    Trust boundary invariant: Sources coerce external string data to declared
    Python types. CSV files contain only strings; the schema declaration
    drives coercion (e.g., "42" -> int(42)).
    """

    def test_string_to_int_coercion(self, tmp_path: Path) -> None:
        """CSV string "42" with schema `id: int` produces Python int 42.

        Trust boundary: Tier 3 string data coerced to Tier 2 int at source.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\n42,Alice\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert not rows[0].is_quarantined

        pipeline_row = rows[0].to_pipeline_row()
        assert pipeline_row["id"] == 42
        assert isinstance(pipeline_row["id"], int)

    def test_string_to_float_coercion(self, tmp_path: Path) -> None:
        """CSV string "3.14" with schema `price: float` produces Python float 3.14.

        Trust boundary: Tier 3 string data coerced to Tier 2 float at source.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("item,price\nwidget,3.14\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["item: str", "price: float"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert not rows[0].is_quarantined

        pipeline_row = rows[0].to_pipeline_row()
        assert pipeline_row["price"] == pytest.approx(3.14)
        assert isinstance(pipeline_row["price"], float)

    def test_string_to_bool_coercion(self, tmp_path: Path) -> None:
        """CSV strings "true"/"false" with bool schema produce Python True/False.

        Trust boundary: Tier 3 string booleans coerced to Tier 2 bool at source.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,active\nAlice,true\nBob,false\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["name: str", "active: bool"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)

        alice = rows[0].to_pipeline_row()
        bob = rows[1].to_pipeline_row()
        assert alice["active"] is True
        assert bob["active"] is False
        assert isinstance(alice["active"], bool)
        assert isinstance(bob["active"], bool)

    def test_non_coercible_value_quarantined(self, tmp_path: Path) -> None:
        """CSV string "abc" for int field is quarantined, not crashed.

        Trust boundary: Non-coercible Tier 3 data must be quarantined,
        not cause a pipeline crash. The pipeline continues processing
        remaining rows.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name\nabc,Alice\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_error is not None
        # Validation error was recorded
        assert len(ctx.validation_errors) == 1

    def test_coerced_types_are_exact(self, tmp_path: Path) -> None:
        """After coercion, isinstance checks confirm exact Python types.

        Trust boundary: Post-coercion data is Tier 2 — types must be
        exactly what the schema declared, not strings.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,price,active\n7,9.99,true\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "price: float", "active: bool"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert not rows[0].is_quarantined

        pipeline_row = rows[0].to_pipeline_row()
        assert isinstance(pipeline_row["id"], int), f"Expected int, got {type(pipeline_row['id'])}"
        assert isinstance(pipeline_row["price"], float), f"Expected float, got {type(pipeline_row['price'])}"
        assert isinstance(pipeline_row["active"], bool), f"Expected bool, got {type(pipeline_row['active'])}"

        # Verify values are correct
        assert pipeline_row["id"] == 7
        assert pipeline_row["price"] == pytest.approx(9.99)
        assert pipeline_row["active"] is True


# ---------------------------------------------------------------------------
# TestCSVSourceMalformedInput
# ---------------------------------------------------------------------------


class TestCSVSourceMalformedInput:
    """Tests that malformed CSV data is quarantined, not crashed.

    Trust boundary invariant: External data (Tier 3) can be literal trash.
    Malformed rows must be quarantined for investigation, not crash the run.
    """

    def test_wrong_column_count_handled(self, tmp_path: Path) -> None:
        """Row with wrong column count is quarantined, not crashed.

        Trust boundary: CSV rows with wrong field count are external
        data format errors — quarantine and continue.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,status\n1,Alice\n")  # 2 fields, expected 3

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        # Row should be quarantined due to field count mismatch
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_error is not None
        assert "expected 3 fields" in rows[0].quarantine_error
        assert len(ctx.validation_errors) == 1

    def test_empty_value_in_required_field(self, tmp_path: Path) -> None:
        """Empty string in required int field is quarantined.

        Trust boundary: Empty strings cannot be coerced to int.
        The source must quarantine, not crash.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,amount\n,100\n")  # Empty id

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
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        # Empty string cannot coerce to int — must be quarantined
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert len(ctx.validation_errors) == 1

    def test_empty_csv_produces_no_rows(self, tmp_path: Path) -> None:
        """CSV with headers only produces 0 rows, no crash.

        Trust boundary: An empty source is a valid external input.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,status\n")  # Headers only

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 0
        assert len(ctx.validation_errors) == 0

    def test_valid_rows_unaffected_by_bad_rows(self, tmp_path: Path) -> None:
        """10 rows where 1 is bad produces 9 valid rows correctly.

        Trust boundary: A single bad row in Tier 3 data must not contaminate
        the other valid rows. Each row is independently validated.
        """
        lines = ["id,amount"]
        for i in range(1, 11):
            if i == 5:
                lines.append("bad,500")  # Row 5 has non-int id
            else:
                lines.append(f"{i},{i * 100}")
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("\n".join(lines) + "\n")

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
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        assert len(valid_rows) == 9
        assert len(quarantined_rows) == 1
        assert len(ctx.validation_errors) == 1

        # Verify valid rows have correct data
        for row in valid_rows:
            pr = row.to_pipeline_row()
            assert isinstance(pr["id"], int)
            assert isinstance(pr["amount"], int)


# ---------------------------------------------------------------------------
# TestJSONSourceBoundary
# ---------------------------------------------------------------------------


class TestJSONSourceBoundary:
    """Tests JSON source trust boundary.

    Trust boundary invariant: JSON external data is Tier 3. Non-standard
    constants (NaN, Infinity) violate RFC 8259 and canonical JSON policy.
    They must be rejected at parse time.
    """

    def test_nan_rejected_at_parse_jsonl(self, tmp_path: Path) -> None:
        """JSONL containing NaN is rejected at parse time.

        Trust boundary: NaN is not valid JSON per RFC 8259. The source
        must reject it (quarantine the line), not let it flow into Tier 2.
        """
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "value": NaN}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        # NaN line must be quarantined
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_error is not None
        assert "NaN" in rows[0].quarantine_error
        assert len(ctx.validation_errors) == 1

    def test_infinity_rejected_at_parse_jsonl(self, tmp_path: Path) -> None:
        """JSONL containing Infinity is rejected at parse time.

        Trust boundary: Infinity is not valid JSON per RFC 8259. The source
        must reject it at the parse boundary.
        """
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "value": Infinity}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_error is not None
        assert "Infinity" in rows[0].quarantine_error
        assert len(ctx.validation_errors) == 1

    def test_nan_rejected_at_parse_json_array(self, tmp_path: Path) -> None:
        """JSON array file containing NaN is rejected at parse time.

        Trust boundary: NaN in a JSON array format file must also be
        caught at the parse boundary, not just in JSONL.
        """
        json_file = tmp_path / "data.json"
        json_file.write_text('[{"id": 1, "value": NaN}]')

        source = JSONSource(
            {
                "path": str(json_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        # Entire file parse fails — quarantined as file-level error
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert len(ctx.validation_errors) == 1

    def test_valid_json_array_of_objects_succeeds(self, tmp_path: Path) -> None:
        """Normal JSONL with valid data produces all valid rows.

        Trust boundary: Well-formed external data passes the Tier 3
        boundary and becomes Tier 2 pipeline data.
        """
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "name": "Alice"}\n{"id": 2, "name": "Bob"}\n{"id": 3, "name": "Carol"}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert all(not r.is_quarantined for r in rows)
        assert len(ctx.validation_errors) == 0

        # Verify data integrity
        pr0 = rows[0].to_pipeline_row()
        assert pr0["id"] == 1
        assert pr0["name"] == "Alice"

    def test_negative_infinity_rejected_at_parse_jsonl(self, tmp_path: Path) -> None:
        """JSONL containing -Infinity is rejected at parse time.

        Trust boundary: -Infinity is a non-standard JSON constant.
        Must be caught at the parse boundary.
        """
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1, "value": -Infinity}\n')

        source = JSONSource(
            {
                "path": str(jsonl_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_error is not None
        assert "Infinity" in rows[0].quarantine_error


# ---------------------------------------------------------------------------
# TestSourceQuarantineAuditTrail
# ---------------------------------------------------------------------------


class TestSourceQuarantineAuditTrail:
    """Tests that quarantine events are recorded in the Landscape audit trail.

    These tests use a real RecorderFactory (not _TestablePluginContext)
    to verify that quarantine events persist to the audit database and can
    be queried for post-run investigation.

    Trust boundary invariant: Every quarantine event must be recorded for
    complete audit coverage. "I don't know what happened" is never acceptable.
    """

    def test_quarantine_recorded_in_landscape(self, tmp_path: Path, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """Quarantined row is recorded in Landscape validation_errors table.

        Trust boundary: The audit trail must record that a row was
        quarantined, even though it never enters the pipeline.
        """
        run_id, node_id = _setup_run_and_node(landscape_factory)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,amount\nabc,100\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["id: int", "amount: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_audited_context(landscape_factory, run_id, node_id)
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined

        # Query the Landscape for validation errors
        errors = landscape_factory.data_flow.get_validation_errors_for_run(run_id)
        assert len(errors) == 1
        assert errors[0].run_id == run_id
        assert errors[0].schema_mode == "fixed"
        assert errors[0].destination == "quarantine"

    def test_quarantine_includes_error_reason(self, tmp_path: Path, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """Audit record has a descriptive reason string.

        Trust boundary: The audit trail must explain WHY a row was
        quarantined, not just that it was.
        """
        run_id, node_id = _setup_run_and_node(landscape_factory)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,amount\nnot_a_number,100\n")

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["id: int", "amount: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_audited_context(landscape_factory, run_id, node_id)
        list(source.load(ctx))

        errors = landscape_factory.data_flow.get_validation_errors_for_run(run_id)
        assert len(errors) == 1
        # Error reason should mention the problematic value or field
        assert errors[0].error is not None
        assert len(errors[0].error) > 0
        # The row data should be preserved in the audit trail
        assert errors[0].row_data_json is not None

    def test_valid_and_quarantined_rows_both_recorded(
        self, tmp_path: Path, landscape_db: LandscapeDB, landscape_factory: RecorderFactory
    ) -> None:
        """Mix of valid/invalid rows: quarantined rows are recorded in Landscape.

        Trust boundary: The audit trail must show the full picture —
        both successful processing and quarantine events.
        """
        run_id, node_id = _setup_run_and_node(landscape_factory)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,amount
            1,100
            bad,200
            3,300
            also_bad,400
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "fixed", "fields": ["id: int", "amount: int"]},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_audited_context(landscape_factory, run_id, node_id)
        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]

        # 2 valid, 2 quarantined
        assert len(valid_rows) == 2
        assert len(quarantined_rows) == 2

        # Both quarantine events recorded in Landscape
        errors = landscape_factory.data_flow.get_validation_errors_for_run(run_id)
        assert len(errors) == 2

        # Each error has run_id and destination
        for error in errors:
            assert error.run_id == run_id
            assert error.destination == "quarantine"

    def test_column_count_mismatch_recorded_in_landscape(
        self, tmp_path: Path, landscape_db: LandscapeDB, landscape_factory: RecorderFactory
    ) -> None:
        """Column count mismatch quarantine is recorded as a parse-level error.

        Trust boundary: Structural malformation in external data is a
        distinct error category from schema validation failures.
        """
        run_id, node_id = _setup_run_and_node(landscape_factory)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,status\n1,Alice\n")  # 2 fields, expected 3

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_audited_context(landscape_factory, run_id, node_id)
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0].is_quarantined

        errors = landscape_factory.data_flow.get_validation_errors_for_run(run_id)
        assert len(errors) == 1
        # Column count errors are recorded as "parse" schema_mode
        assert errors[0].schema_mode == "parse"


# ---------------------------------------------------------------------------
# TestSourceSchemaGuarantees
# ---------------------------------------------------------------------------


class TestSourceSchemaGuarantees:
    """Tests that source output satisfies its schema guarantees.

    Trust boundary invariant: After crossing the Tier 3 -> Tier 2 boundary,
    every valid row must satisfy the declared schema contract.
    """

    def test_guaranteed_fields_present_in_all_valid_rows(self, tmp_path: Path) -> None:
        """Every non-quarantined row has all guaranteed_fields from config.

        Trust boundary: Guaranteed fields are the source's promise to
        downstream consumers. Every valid row must deliver on this promise.
        """
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
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "amount: int", "status: str"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        valid_rows = [r for r in rows if not r.is_quarantined]
        assert len(valid_rows) == 3

        for row in valid_rows:
            pipeline_row = row.to_pipeline_row()
            # All declared fields must be present
            assert "id" in pipeline_row
            assert "amount" in pipeline_row
            assert "status" in pipeline_row

    def test_guaranteed_field_types_match_schema(self, tmp_path: Path) -> None:
        """Types match what the schema declared after coercion.

        Trust boundary: The schema contract is a type guarantee. After
        coercion, runtime types must match declared types exactly.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,price,name
            1,19.99,Widget
            2,29.99,Gadget
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "price: float", "name: str"],
                },
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()
        rows = list(source.load(ctx))

        for row in rows:
            assert not row.is_quarantined
            pipeline_row = row.to_pipeline_row()

            # Check contract field types match runtime types
            contract = row.contract
            assert contract is not None
            for fc in contract.fields:
                value = pipeline_row[fc.normalized_name]
                # object type (from 'any') accepts anything
                if fc.python_type is not object:
                    assert isinstance(value, fc.python_type), (
                        f"Field '{fc.normalized_name}': expected {fc.python_type.__name__}, got {type(value).__name__} (value={value!r})"
                    )

    def test_observed_schema_locks_after_first_row(self, tmp_path: Path) -> None:
        """Dynamic/observed mode: schema locks after first valid row.

        Trust boundary: In observed mode, the source infers types from
        the first valid row and then locks the contract. Subsequent rows
        are validated against the locked schema. This prevents type drift
        within a single source load.
        """
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            dedent("""\
            id,name,score
            1,Alice,95
            2,Bob,87
            3,Carol,92
        """)
        )

        source = CSVSource(
            {
                "path": str(csv_file),
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        ctx = _make_test_context()

        rows = list(source.load(ctx))

        # All rows valid
        assert len(rows) == 3
        assert all(not r.is_quarantined for r in rows)

        # Contract must be locked after processing
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked is True
        assert contract.mode == "OBSERVED"

        # Contract has inferred fields from first row
        field_names = {fc.normalized_name for fc in contract.fields}
        assert "id" in field_names
        assert "name" in field_names
        assert "score" in field_names

        # All fields should be marked as inferred (not declared)
        for fc in contract.fields:
            assert fc.source == "inferred"

        # All rows share the same locked contract
        contracts = {id(r.contract) for r in rows}
        assert len(contracts) == 1, "All rows should share the same contract instance"
