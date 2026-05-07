"""Integration test for BatchReplicate contract provision during token expansion.

Verifies fix for bug where batch_replicate.process() returned TransformResult.success_multi()
without a contract, causing ValueError at processor.py:1826-1830 during token expansion.

Also verifies the union-of-keys contract fix: contract fields are built from ALL valid
output rows, not just the first one (first-element bias fix).
"""

from __future__ import annotations

from elspeth.contracts import PipelineRow
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.transforms.batch_replicate import BatchReplicate
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context


def test_batch_replicate_returns_contract_with_multi_row_output():
    """BatchReplicate provides contract when returning multi-row output.

    The processor requires multi-row transforms to provide contracts (processor.py:1826-1830).
    Without this, pipelines using batch_replicate with output_mode="transform" would crash
    with: ValueError: Transform batch_replicate produced multi-row output but returned no contract.

    Contract fields are built from the union of ALL valid output row keys.
    """
    # Create transform
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    # Create input as list (batch-aware mode)
    rows = [
        {"id": 1, "copies": 2},
        {"id": 2, "copies": 3},
    ]

    # Create PipelineRow instances with minimal contract
    pipeline_rows = []
    for row in rows:
        fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row)
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        pipeline_rows.append(make_row(row, contract=contract))

    # Process batch
    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    # Verify multi-row result
    assert result.is_multi_row, "Should return multi-row result"
    assert result.rows is not None, "Rows should not be None"
    assert len(result.rows) == 5, f"Expected 5 output rows (2+3), got {len(result.rows)}"

    # CRITICAL: Verify contract is provided (now inside PipelineRow)
    # Without this, processor.py:1826-1830 would raise ValueError
    assert isinstance(result.rows[0], PipelineRow), (
        "BatchReplicate MUST provide PipelineRow with contract for multi-row output. Missing contract causes ValueError during token expansion in processor."
    )

    # Verify contract is OBSERVED mode (correct for inferred fields)
    assert result.rows[0].contract.mode == "OBSERVED", "Contract should be OBSERVED mode"

    # Verify contract includes all output fields
    # BatchReplicate adds copy_index by default
    expected_fields = {"id", "copies", "copy_index"}
    contract_field_names = {fc.normalized_name for fc in result.rows[0].contract.fields}
    assert contract_field_names == expected_fields, (
        f"Contract should include all output fields. Expected {expected_fields}, got {contract_field_names}"
    )

    # Verify field metadata is preserved from input contracts where possible.
    # copy_index is a new field added by the transform, so it gets int type.
    # Input fields preserve their original contract metadata.
    for field in result.rows[0].contract.fields:
        if field.normalized_name == "copy_index":
            assert field.python_type is int, "copy_index should have int type"
            assert field.source == "inferred", "copy_index should be inferred"
        else:
            assert field.source == "inferred", f"Field {field.normalized_name} should be inferred"


def test_batch_replicate_contract_covers_all_output_shapes():
    """Contract uses union of ALL valid output row keys, not just the first row.

    When input rows have different field sets (e.g. one has 'copies' field,
    another uses default), the contract must cover the superset of all keys
    that appear in any valid output row.
    """
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
            "default_copies": 1,
        }
    )

    # Row 1: has 'extra' field but no 'copies' field (uses default)
    # Row 2: has 'copies' field but no 'extra' field
    row1_data = {"id": 1, "extra": "data"}
    row2_data = {"id": 2, "copies": 2}

    fields1 = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row1_data)
    fields2 = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row2_data)

    pipeline_rows = [
        make_row(row1_data, contract=SchemaContract(mode="OBSERVED", fields=fields1, locked=True)),
        make_row(row2_data, contract=SchemaContract(mode="OBSERVED", fields=fields2, locked=True)),
    ]

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.is_multi_row
    assert result.rows is not None
    # 1 (default) + 2 copies = 3 output rows
    assert len(result.rows) == 3

    # Contract must include union of ALL keys: id, extra, copies, copy_index
    contract_field_names = {fc.normalized_name for fc in result.rows[0].contract.fields}
    assert contract_field_names == {"id", "extra", "copies", "copy_index"}


def test_batch_replicate_contract_empty_output():
    """BatchReplicate returns error for empty batch — not fabricated data."""
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
            "default_copies": 1,
        }
    )

    # Empty batch
    ctx = make_context()
    result = transform.process([], ctx)

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "empty_batch"
    assert not result.retryable


def test_batch_replicate_all_invalid_returns_error():
    """When all rows have invalid copies, return error (not success with error rows).

    Error rows must NOT flow through success_multi() as child tokens,
    which would corrupt the token expansion audit trail.
    """
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    rows_data = [
        {"id": 1, "copies": 0},
        {"id": 2, "copies": -5},
    ]

    pipeline_rows = []
    for row in rows_data:
        fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row)
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        pipeline_rows.append(make_row(row, contract=contract))

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.status == "error"
    assert result.reason["reason"] == "all_rows_failed"
    assert "2 rows quarantined" in result.reason["error"]
    assert len(result.reason["row_errors"]) == 2


def test_batch_replicate_mixed_valid_invalid_excludes_quarantined():
    """Mixed batch: only valid rows in success_multi(), quarantined in metadata.

    Quarantined rows appear in success_reason for audit visibility,
    but do NOT become child tokens in the expansion.
    """
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    rows_data = [
        {"id": 1, "copies": -1},  # quarantined
        {"id": 2, "copies": 2},  # valid: 2 copies
        {"id": 3, "copies": 0},  # quarantined
        {"id": 4, "copies": 1},  # valid: 1 copy
    ]

    pipeline_rows = []
    for row in rows_data:
        fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row)
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        pipeline_rows.append(make_row(row, contract=contract))

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.status == "success"
    assert result.rows is not None
    # Only valid copies: 2 (from id=2) + 1 (from id=4) = 3
    assert len(result.rows) == 3
    assert result.rows[0]["id"] == 2
    assert result.rows[1]["id"] == 2
    assert result.rows[2]["id"] == 4

    # No _replicate_error field in any output row
    for row in result.rows:
        assert "_replicate_error" not in row

    # Quarantine info in success_reason.metadata
    assert result.success_reason["metadata"]["quarantined_count"] == 2
    assert result.success_reason["metadata"]["quarantined"][0]["row_data"]["id"] == 1
    assert result.success_reason["metadata"]["quarantined"][1]["row_data"]["id"] == 3


def test_batch_replicate_contract_excludes_quarantined_only_fields():
    """Successful output contracts describe emitted rows only, not quarantined rows."""
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    invalid_row = {"id": 1, "copies": 0, "quarantined_only_marker": "invalid"}
    valid_row = {"id": 2, "copies": 1}

    invalid_fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in invalid_row)
    valid_fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in valid_row)

    pipeline_rows = [
        make_row(invalid_row, contract=SchemaContract(mode="OBSERVED", fields=invalid_fields, locked=True)),
        make_row(valid_row, contract=SchemaContract(mode="OBSERVED", fields=valid_fields, locked=True)),
    ]

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.status == "success"
    assert result.rows is not None
    assert len(result.rows) == 1
    assert "quarantined_only_marker" not in result.rows[0]

    contract_field_names = {field.normalized_name for field in result.rows[0].contract.fields}
    assert contract_field_names == {"id", "copies", "copy_index"}


def test_batch_replicate_quarantined_indices_in_success_reason():
    """T26: Quarantined row indices must be in success_reason for audit trail.

    The processor needs quarantined_indices to record QUARANTINED terminal
    state (instead of CONSUMED_IN_BATCH) for quarantined tokens. Without
    this, quarantined rows silently get CONSUMED_IN_BATCH, misleading
    auditors into thinking the data was processed.
    """
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    rows_data = [
        {"id": 1, "copies": -1},  # quarantined (index 0)
        {"id": 2, "copies": 2},  # valid (index 1)
        {"id": 3, "copies": 0},  # quarantined (index 2)
        {"id": 4, "copies": 1},  # valid (index 3)
    ]

    pipeline_rows = []
    for row in rows_data:
        fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row)
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        pipeline_rows.append(make_row(row, contract=contract))

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.status == "success"
    # Must include quarantined_indices for processor to record QUARANTINED outcomes
    assert "quarantined_indices" in result.success_reason["metadata"], (
        "success_reason must include quarantined_indices for audit trail recording"
    )
    assert result.success_reason["metadata"]["quarantined_indices"] == [0, 2]


def test_batch_replicate_no_quarantine_no_indices():
    """When no rows are quarantined, quarantined_indices should be absent or empty."""
    transform = BatchReplicate(
        {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
        }
    )

    rows_data = [
        {"id": 1, "copies": 2},
        {"id": 2, "copies": 1},
    ]

    pipeline_rows = []
    for row in rows_data:
        fields = tuple(make_field(key, object, original_name=key, required=False, source="inferred") for key in row)
        contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        pipeline_rows.append(make_row(row, contract=contract))

    ctx = make_context()
    result = transform.process(pipeline_rows, ctx)

    assert result.status == "success"
    # No quarantine metadata at all when no rows are quarantined —
    # BatchReplicate only adds "metadata" key when quarantined is non-empty.
    assert "metadata" not in result.success_reason
