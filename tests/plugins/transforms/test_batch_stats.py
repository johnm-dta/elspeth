"""Tests for BatchStats aggregation transform.

BatchStats computes aggregate statistics (count, sum, mean) over batches of rows.
It is batch-aware, meaning it receives lists of rows when aggregation triggers fire.

OUTPUT SCHEMA BUG (P1-2026-01-19-shape-changing-transforms-output-schema-mismatch):
BatchStats outputs a completely different shape ({count, sum, mean, batch_size})
than its input, but incorrectly sets output_schema = input_schema.
"""

from typing import Any

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import TransformProtocol

# Common schema config for dynamic field handling (accepts any fields)
DYNAMIC_SCHEMA = {"fields": "dynamic"}


class TestBatchStatsHappyPath:
    """Happy path tests for BatchStats transform."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """BatchStats implements TransformProtocol.

        Note: BatchStats is batch-aware and has a different process() signature
        (list[dict] instead of dict). At runtime, isinstance() passes because
        runtime_checkable only checks method presence. Mypy correctly identifies
        the signature incompatibility, so we type-ignore this specific check.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )
        # BatchStats.process() takes list[dict], not dict, so the protocol
        # signatures are incompatible at the type level. Runtime check passes.
        assert isinstance(transform, TransformProtocol)  # type: ignore[unreachable]

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
            {"id": 1, "amount": 10.0},
            {"id": 2, "amount": 20.0},
            {"id": 3, "amount": 30.0},
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
            {"id": 1, "amount": 10.0, "category": "sales"},
            {"id": 2, "amount": 20.0, "category": "sales"},
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

        rows: list[dict[str, Any]] = [
            {"id": 1, "amount": 10.0},
            {"id": 2, "amount": "not_a_number"},  # This must raise, not skip
            {"id": 3, "amount": 30.0},
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)


class TestBatchStatsOutputSchema:
    """Tests for output schema behavior of shape-changing transforms.

    Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch:
    BatchStats outputs {count, sum, mean, batch_size, group_by?} regardless of input.
    The output shape is completely different from input, so output_schema must be dynamic.
    """

    def test_output_schema_is_dynamic(self) -> None:
        """BatchStats uses dynamic output_schema.

        BatchStats always outputs {count, sum, mean, batch_size, ...} regardless
        of input schema. The output shape depends on the transform's logic, not input.
        Therefore output_schema must be dynamic.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        # Explicit schema: expects id, amount, category
        transform = BatchStats(
            {
                "schema": {"mode": "strict", "fields": ["id: int", "amount: float", "category: str"]},
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
