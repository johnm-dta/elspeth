"""Tests for BatchStats aggregation transform.

BatchStats computes aggregate statistics (count, sum, mean) over batches of rows.
It is batch-aware, meaning it receives lists of rows when aggregation triggers fire.

OUTPUT SCHEMA BUG (P1-2026-01-19-shape-changing-transforms-output-schema-mismatch):
BatchStats outputs a completely different shape ({count, sum, mean, batch_size})
than its input, but incorrectly sets output_schema = input_schema.
"""

import sys
from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field, make_row

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_row(data: dict[str, Any]):
    """Create a PipelineRow with OBSERVED contract for testing."""
    fields = tuple(
        make_field(key, type(value) if value is not None else object, original_name=key, required=False, source="inferred")
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


class TestBatchStatsHappyPath:
    """Happy path tests for BatchStats transform."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_has_required_attributes(self) -> None:
        """BatchStats has name and is_batch_aware."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        assert BatchStats.name == "batch_stats"
        assert BatchStats.is_batch_aware is True

    def test_computes_count_sum_mean(self, ctx: PluginContext) -> None:
        """BatchStats computes count, sum, and mean from batch."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": 20.0}),
            _make_row({"id": 3, "amount": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 3
        assert result.row["sum"] == 60.0
        assert result.row["mean"] == 20.0
        assert result.row["batch_size"] == 3

    def test_includes_group_by_field(self, ctx: PluginContext) -> None:
        """BatchStats includes group_by field from first row."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
                "group_by": "category",
            }
        )

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0, "category": "sales"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["category"] == "sales"

    def test_empty_batch_returns_zeros(self, ctx: PluginContext) -> None:
        """Empty batch returns zero count/sum and None mean."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )

        result = transform.process([], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 0
        assert result.row["sum"] == 0
        assert result.row["mean"] is None
        assert result.row["batch_empty"] is True

    def test_empty_batch_without_mean_omits_mean_field(self, ctx: PluginContext) -> None:
        """Empty batch respects compute_mean=False in output and success metadata."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
                "compute_mean": False,
            }
        )

        result = transform.process([], ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 0
        assert result.row["sum"] == 0
        assert result.row["batch_empty"] is True
        assert "mean" not in result.row
        assert result.success_reason is not None
        assert result.success_reason["fields_added"] == ["count", "sum", "batch_empty"]

    def test_non_numeric_values_raise_type_error(self, ctx: PluginContext) -> None:
        """BatchStats raises TypeError on non-numeric values (no coercion).

        Per CLAUDE.md Tier 2 trust model: transforms receive pipeline data that
        should already be type-validated. Wrong types indicate upstream bugs
        and must raise TypeError, not be silently skipped.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": "not_a_number"}),  # This must raise, not skip
            _make_row({"id": 3, "amount": 30.0}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)


class TestBatchStatsOutputSchema:
    """Tests for output schema behavior of shape-changing transforms.

    Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch:
    BatchStats outputs {count, sum, mean, batch_size, group_by?} regardless of input.
    The output shape is completely different from input, so output_schema must be dynamic.
    """

    def test_output_schema_is_observed(self) -> None:
        """BatchStats uses dynamic output_schema.

        BatchStats always outputs {count, sum, mean, batch_size, ...} regardless
        of input schema. The output shape depends on the transform's logic, not input.
        Therefore output_schema must be dynamic.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        # Explicit schema: expects id, amount, category
        transform = BatchStats(
            {
                "schema": {"mode": "fixed", "fields": ["id: int", "amount: float", "category: str"]},
                "value_field": "amount",
                "group_by": "category",
            }
        )

        # Output schema should be dynamic (no required fields, extra="allow")
        # Output has: count, sum, mean, batch_size, category (NOT id, amount)
        # Currently fails because output_schema = input_schema (has id, amount, category)
        output_fields = transform.output_schema.model_fields

        assert len(output_fields) == 0, f"Expected dynamic schema with no required fields, got: {list(output_fields.keys())}"

        config = transform.output_schema.model_config
        assert config.get("extra") == "allow", "Output schema should allow extra fields (dynamic)"


class TestBatchStatsFloatOverflow:
    """Tests for float overflow and NaN/Inf handling.

    IEEE 754 NaN and Infinity pass isinstance(x, float) but produce garbage
    in arithmetic and crash downstream canonical JSON (RFC 8785 rejects
    non-finite values). These must be detected and handled.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_nan_input_skipped_from_computation(self, ctx: PluginContext) -> None:
        """NaN values are skipped from sum/mean, tracked in skipped_non_finite."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": float("nan")}),
            _make_row({"id": 3, "amount": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 2  # Only finite values counted
        assert result.row["sum"] == 40.0
        assert result.row["mean"] == 20.0
        assert result.row["skipped_non_finite"] == 1

    def test_inf_input_skipped_from_computation(self, ctx: PluginContext) -> None:
        """Infinity values are skipped from sum/mean, tracked in skipped_non_finite."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": float("inf")}),
            _make_row({"id": 3, "amount": float("-inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 1
        assert result.row["sum"] == 10.0
        assert result.row["skipped_non_finite"] == 2

    def test_all_non_finite_produces_zero_count(self, ctx: PluginContext) -> None:
        """Batch with only NaN/Inf values produces count=0, sum=0."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": float("nan")}),
            _make_row({"id": 2, "amount": float("inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 0
        assert result.row["sum"] == 0.0
        assert result.row["mean"] is None
        assert result.row["skipped_non_finite"] == 2

    def test_sum_overflow_returns_error(self, ctx: PluginContext) -> None:
        """Summing large valid floats that overflow to inf returns error."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": sys.float_info.max}),
            _make_row({"id": 2, "amount": sys.float_info.max}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "float_overflow"

    def test_no_skipped_field_when_all_finite(self, ctx: PluginContext) -> None:
        """skipped_non_finite field is absent when all values are finite."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": 20.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert "skipped_non_finite" not in result.row.to_dict()


class TestBatchStatsGroupByHomogeneity:
    """Tests for group_by value validation across batch rows.

    When group_by is configured, all rows in the batch must have the same
    value for that field. Mixed values indicate a trigger/topology bug.
    """

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_homogeneous_group_by_included(self, ctx: PluginContext) -> None:
        """All rows same group_by value — included in output."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0, "category": "sales"}),
            _make_row({"id": 3, "amount": 30.0, "category": "sales"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["category"] == "sales"

    def test_heterogeneous_group_by_raises(self, ctx: PluginContext) -> None:
        """Mixed group_by values raise ValueError — topology/config bug."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0, "category": "returns"}),
        ]

        with pytest.raises(ValueError, match="Heterogeneous"):
            transform.process(rows, ctx)

    def test_group_by_field_missing_from_all_rows_raises(self, ctx: PluginContext) -> None:
        """Configured group_by missing from all rows should fail fast."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": 20.0}),
        ]

        with pytest.raises(KeyError):
            transform.process(rows, ctx)

    def test_group_by_field_missing_from_later_row_raises(self, ctx: PluginContext) -> None:
        """Configured group_by missing from any row should fail fast."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0}),
        ]

        with pytest.raises(KeyError):
            transform.process(rows, ctx)
