"""Integration test for BatchStats contract provision in transform mode.

Verifies fix for bug where batch_stats.process() returned TransformResult.success()
without a contract, causing ValueError at processor.py:712 in transform mode aggregations.
"""

from __future__ import annotations

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.transforms.batch_stats import BatchStats


def test_batch_stats_returns_contract_in_transform_mode():
    """BatchStats provides contract for transform mode aggregations.

    In transform mode, aggregations create new tokens via expand_token (processor.py:718).
    Even though BatchStats returns a single aggregated row (not multi-row), the processor
    wraps it in a list and requires a contract at line 712.

    Without the contract, pipelines using batch_stats with output_mode="transform"
    would crash with: ValueError: Batch transform batch_stats produced multi-row output
    but returned no contract.

    This test verifies the fix at batch_stats.py:178-192.
    """
    # Create transform
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "group_by": "category",
            "compute_mean": True,
        }
    )

    # Create input batch (list of dicts in batch-aware mode)
    rows = [
        {"category": "A", "amount": 100},
        {"category": "A", "amount": 200},
        {"category": "A", "amount": 300},
    ]

    # Process batch
    ctx = PluginContext(run_id="test-run", config={})
    result = transform.process(rows, ctx)

    # Verify single-row aggregated result
    assert not result.is_multi_row, "BatchStats returns single aggregated row"
    assert result.row is not None, "Result should have aggregated row"
    assert result.row["count"] == 3
    assert result.row["sum"] == 600
    assert result.row["mean"] == 200

    # CRITICAL: Verify contract is provided (now inside PipelineRow)
    # Without this, processor.py:712 would raise ValueError in transform mode
    assert isinstance(result.row, PipelineRow), (
        "BatchStats MUST provide PipelineRow with contract for transform mode. Missing contract causes ValueError when aggregation creates new tokens."
    )

    # Verify contract is OBSERVED mode (correct for aggregation output)
    assert result.row.contract.mode == "OBSERVED", "Contract should be OBSERVED mode"

    # Verify contract includes all output fields
    expected_fields = {"count", "sum", "batch_size", "mean", "category"}
    contract_field_names = {fc.normalized_name for fc in result.row.contract.fields}
    assert contract_field_names == expected_fields, (
        f"Contract should include all output fields. Expected {expected_fields}, got {contract_field_names}"
    )

    # Verify all fields are marked as inferred (OBSERVED mode pattern)
    for field in result.row.contract.fields:
        assert field.source == "inferred", f"Field {field.normalized_name} should be inferred"
        assert field.python_type is object, f"Field {field.normalized_name} should have object type"


def test_batch_stats_contract_empty_batch():
    """BatchStats provides contract even for empty batch."""
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "compute_mean": True,
        }
    )

    # Empty batch
    ctx = PluginContext(run_id="test-run", config={})
    result = transform.process([], ctx)

    # Should return aggregated result with zeros
    assert not result.is_multi_row
    assert result.row is not None
    assert result.row["count"] == 0
    assert result.row["batch_empty"] is True

    # CRITICAL: Contract must be provided even for empty batch (inside PipelineRow)
    assert isinstance(result.row, PipelineRow), "Should provide PipelineRow with contract for empty batch"
    assert result.row.contract.mode == "OBSERVED"

    # Verify empty batch contract includes marker fields
    expected_fields = {"count", "sum", "mean", "batch_empty"}
    contract_field_names = {fc.normalized_name for fc in result.row.contract.fields}
    assert contract_field_names == expected_fields


def test_batch_stats_contract_without_mean():
    """BatchStats contract adapts to compute_mean setting."""
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "compute_mean": False,  # Disable mean computation
        }
    )

    rows = [{"amount": 100}, {"amount": 200}]
    ctx = PluginContext(run_id="test-run", config={})
    result = transform.process(rows, ctx)

    # Verify mean is not in output
    assert "mean" not in result.row

    # Verify contract reflects actual output fields
    assert isinstance(result.row, PipelineRow)
    contract_field_names = {fc.normalized_name for fc in result.row.contract.fields}
    assert "mean" not in contract_field_names, "Contract should not include mean when disabled"
    assert contract_field_names == {"count", "sum", "batch_size"}


def test_batch_stats_contract_empty_batch_without_mean():
    """Empty batch contract also omits mean when compute_mean=False."""
    transform = BatchStats(
        {
            "schema": {"mode": "observed"},
            "value_field": "amount",
            "compute_mean": False,
        }
    )

    ctx = PluginContext(run_id="test-run", config={})
    result = transform.process([], ctx)

    assert not result.is_multi_row
    assert result.row is not None
    assert "mean" not in result.row
    assert result.row["batch_empty"] is True

    assert isinstance(result.row, PipelineRow)
    contract_field_names = {fc.normalized_name for fc in result.row.contract.fields}
    assert "mean" not in contract_field_names
    assert contract_field_names == {"count", "sum", "batch_empty"}
