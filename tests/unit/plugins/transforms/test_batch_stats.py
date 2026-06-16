"""Tests for BatchStats aggregation transform.

BatchStats computes aggregate statistics (count, sum, mean) over batches of rows.
It is batch-aware, meaning it receives lists of rows when aggregation triggers fire.

OUTPUT SCHEMA BUG (P1-2026-01-19-shape-changing-transforms-output-schema-mismatch):
BatchStats outputs a completely different shape ({count, sum, mean, batch_size})
than its input, but incorrectly sets output_schema = input_schema.
"""

import sys
from decimal import Decimal
from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context

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
        return make_context()

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

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        """Empty batch returns error — not fabricated statistics."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable

    def test_empty_batch_without_mean_also_errors(self, ctx: PluginContext) -> None:
        """Empty batch errors regardless of compute_mean setting."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
                "compute_mean": False,
            }
        )

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"

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

    def test_backward_probe_rows_drop_original_fields_for_invariant_harness(self, ctx: PluginContext) -> None:
        """Backward invariant probe exercises the aggregate output path, not passthrough."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(BatchStats.probe_config())
        probe = _make_row({"baseline": "kept"})

        result = transform.execute_backward_invariant_probe(
            transform.backward_invariant_probe_rows(probe),
            ctx,
        )

        assert result.status == "success"
        assert result.row is not None
        assert "baseline" not in result.row
        assert result.row["count"] == 1
        assert result.row["sum"] == 1.0


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
        return make_context()

    def test_nan_input_skipped_from_computation(self, ctx: PluginContext) -> None:
        """NaN values are skipped from sum/mean, tracked in skipped_non_finite with indices."""
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
        assert result.row["skipped_non_finite_indices"] == (1,)

    def test_inf_input_skipped_from_computation(self, ctx: PluginContext) -> None:
        """Infinity values are skipped from sum/mean, tracked in skipped_non_finite with indices."""
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
        assert result.row["skipped_non_finite_indices"] == (1, 2)

    def test_all_non_finite_returns_error(self, ctx: PluginContext) -> None:
        """Batch with only NaN/Inf values returns error — not fabricated sum=0."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": float("nan")}),
            _make_row({"id": 2, "amount": float("inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "all_non_finite"
        assert result.reason["batch_size"] == 2
        assert result.reason["skipped_non_finite"] == 2
        assert result.reason["skipped_non_finite_indices"] == [0, 1]

    def test_none_value_skipped_and_reported(self, ctx: PluginContext) -> None:
        """None in value_field is a missing value: skipped-and-reported, never a crash.

        Per CLAUDE.md Tier 2/3 doctrine and every sibling batch transform
        (batch_distribution_profile etc.), None is missing data — it is skipped
        from the computation and reported in skipped_missing, NOT raised as a
        TypeError that aborts the whole run (plugins review C2).
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": None}),
            _make_row({"id": 3, "amount": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 2  # Only present values counted
        assert result.row["sum"] == 40.0
        assert result.row["mean"] == 20.0
        assert result.row["skipped_missing"] == 1
        assert result.row["skipped_missing_indices"] == (1,)

    def test_all_none_batch_returns_error_not_crash(self, ctx: PluginContext) -> None:
        """A group with only None values returns an audited error, never count=0/sum=0.

        Without the missing-value branch this hits sum([])=0 then mean=0/0
        (ZeroDivisionError), crashing the run — and fabricating count=0/sum=0
        would be a phantom statistic. The honest outcome is a validation_failed
        error naming each missing row (plugins review C2).
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        rows = [
            _make_row({"id": 1, "amount": None}),
            _make_row({"id": 2, "amount": None}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["batch_size"] == 2
        assert result.reason["skipped_count"] == 2
        assert result.reason["valid_count"] == 0
        row_error_reasons = {entry["reason"] for entry in result.reason["row_errors"]}
        assert row_error_reasons == {"missing_value"}

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

    def test_mean_overflow_returns_error(self, ctx: PluginContext) -> None:
        """Huge integer totals that overflow during mean computation return an error."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount"})

        result = transform.process([_make_row({"id": 1, "amount": 10**1000})], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "float_overflow"
        assert result.reason["operation"] == "mean"

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
        assert "skipped_non_finite_indices" not in result.row.to_dict()


class TestBatchStatsGroupByRollups:
    """Tests for group_by rollup behavior across batch rows."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    @pytest.mark.parametrize("blank_group_by", ["", "   "])
    def test_blank_group_by_rejected_at_config_boundary(self, blank_group_by: str) -> None:
        """Blank group_by cannot enter declared input/output contracts."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        with pytest.raises(PluginConfigError, match="group_by must not be empty"):
            BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": blank_group_by})

    @pytest.mark.parametrize("group_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_group_key_returns_error_before_success(self, ctx: PluginContext, group_value: float) -> None:
        """Non-finite group_by key must error before producing any output (B4.5-d)."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})
        rows = [
            _make_row({"amount": 10.0, "category": "A"}),
            _make_row({"amount": 20.0, "category": group_value}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_group_key"
        assert result.reason["field"] == "category"
        assert not result.retryable

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

    def test_heterogeneous_group_by_emits_one_rollup_per_group(self, ctx: PluginContext) -> None:
        """Mixed group_by values produce per-group aggregate rows."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0, "category": "returns"}),
            _make_row({"id": 3, "amount": 30.0, "category": "sales"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None

        rollups = {row["category"]: row for row in result.rows}
        assert set(rollups) == {"sales", "returns"}
        assert rollups["sales"]["count"] == 2
        assert rollups["sales"]["sum"] == 40.0
        assert rollups["sales"]["mean"] == 20.0
        assert rollups["sales"]["batch_size"] == 2
        assert rollups["returns"]["count"] == 1
        assert rollups["returns"]["sum"] == 20.0
        assert rollups["returns"]["mean"] == 20.0
        assert rollups["returns"]["batch_size"] == 1

    def test_distinct_group_keys_do_not_scan_every_prior_group(self, ctx: PluginContext) -> None:
        """Distinct group_by values are partitioned without quadratic prior-group scans."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        class CountingGroupKey:
            comparisons = 0

            def __init__(self, value: int) -> None:
                self.value = value

            def __eq__(self, other: object) -> bool:
                CountingGroupKey.comparisons += 1
                return isinstance(other, CountingGroupKey) and self.value == other.value

            def __hash__(self) -> int:
                return hash(self.value)

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})
        row_count = 64
        group_values = [CountingGroupKey(row_index) for row_index in range(row_count)]
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field("id", int, original_name="id", required=False, source="inferred"),
                make_field("amount", float, original_name="amount", required=False, source="inferred"),
                make_field("category", object, original_name="category", required=False, source="inferred"),
            ),
            locked=True,
        )
        rows = [
            make_row({"id": row_index, "amount": 1.0, "category": group_values[row_index]}, contract=contract)
            for row_index in range(row_count)
        ]

        grouped = transform._group_rows(rows)

        assert [group_value for group_value, _ in grouped] == group_values
        assert CountingGroupKey.comparisons <= row_count

    def test_two_group_batch_emits_two_rollups(self, ctx: PluginContext) -> None:
        """Mixed group_by values emit one aggregate row per group."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": 10.0, "category": "sales"}),
            _make_row({"id": 2, "amount": 20.0, "category": "returns"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["category"] for row in result.rows] == ["sales", "returns"]

    def test_all_non_finite_group_returns_group_error(self, ctx: PluginContext) -> None:
        """A group with no finite values errors without fabricating statistics."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "group_by": "category"})

        rows = [
            _make_row({"id": 1, "amount": float("nan"), "category": "sales"}),
            _make_row({"id": 2, "amount": float("inf"), "category": "returns"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "all_non_finite"
        assert result.reason["group_by"] == "category"
        assert result.reason["group_value"] == "sales"
        assert result.reason["skipped_non_finite_indices"] == [0]

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


class TestOutputSchemaConfig:
    def test_guaranteed_fields_with_mean_and_group_by(self):
        """group_by IS guaranteed on every successful result when configured."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {"mode": "observed"},
                "value_field": "amount",
                "compute_mean": True,
                "group_by": "category",
            }
        )
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset({"count", "sum", "batch_size", "mean", "category"})

    def test_guaranteed_fields_minimal(self):
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {"mode": "observed"},
                "value_field": "amount",
                "compute_mean": False,
            }
        )
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset({"count", "sum", "batch_size"})

    def test_declared_output_fields_includes_group_by(self):
        """group_by is guaranteed on every successful result when configured."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {"mode": "observed"},
                "value_field": "amount",
                "group_by": "region",
            }
        )
        assert transform.declared_output_fields == frozenset({"count", "sum", "batch_size", "mean", "region"})
        assert "region" in transform.declared_output_fields


class TestOutputSchemaConfigDoesNotPropagateInputContract:
    """Regression tests for elspeth-f5f798f797.

    The user-supplied ``schema:`` block describes the INPUT contract for an
    aggregation. ``_output_schema_config`` describes what the aggregation
    EMITS, which for batch_stats is independent of input shape (output is
    {count, sum, batch_size, mean?, group_by?} — never the input fields
    themselves). Therefore input ``fields``/``required_fields``/
    ``guaranteed_fields`` from the user's schema must not leak into
    ``_output_schema_config``; doing so causes the runtime
    SchemaConfigModeViolation check to require fields the aggregation never
    emits, and to expect declared field types that the OBSERVED output
    contract does not carry.
    """

    def test_output_config_drops_input_fields_for_flexible_mode(self) -> None:
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        # Input fields/required_fields describe input contract, not output.
        assert cfg.fields is None, f"Output config must not echo input fields, got {cfg.fields!r}"
        assert cfg.required_fields is None, f"Output config must not echo input required_fields, got {cfg.required_fields!r}"
        # guaranteed_fields reflect the actual aggregation output.
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset({"count", "sum", "batch_size", "customer_tier"})

    def test_output_config_drops_input_fields_for_fixed_mode(self) -> None:
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "amount: float"],
                },
                "value_field": "amount",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert cfg.required_fields is None

    def test_output_config_drops_input_guaranteed_fields(self) -> None:
        """Input-side ``guaranteed_fields`` describe upstream guarantees, not aggregation output."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {
                    "mode": "observed",
                    "guaranteed_fields": ["upstream_only_field"],
                },
                "value_field": "amount",
                "compute_mean": False,
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        # upstream_only_field is an INPUT guarantee — must not appear as an OUTPUT guarantee.
        assert "upstream_only_field" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset({"count", "sum", "batch_size"})

    def test_runtime_verify_passes_for_failing_eval_reproducer(self) -> None:
        """End-to-end: the eval-S2-v2 failing config must clear ``verify_schema_config_mode``.

        Reproduced from docs/composer/evidence/composer-llm-eval-2026-05-01.md S2 v2: the
        composer accepted the config but runtime rejected with
        SchemaConfigModeViolation. After the fix, the runtime check must
        pass for the same simulated emission.
        """
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.contracts.schema_contract import PipelineRow
        from elspeth.contracts.schema_contract_factory import create_contract_from_config
        from elspeth.engine.executors.schema_config_mode import verify_schema_config_mode
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            }
        )

        input_contract = create_contract_from_config(
            SchemaConfig.from_dict({"mode": "flexible", "fields": ["customer_tier: str", "amount: float"]})
        )
        rows = [
            PipelineRow({"customer_tier": "enterprise", "amount": 100.0}, input_contract),
            PipelineRow({"customer_tier": "enterprise", "amount": 150.0}, input_contract),
            PipelineRow({"customer_tier": "pro", "amount": 50.0}, input_contract),
        ]

        results = []
        for group_value, grouped in transform._group_rows(rows):
            aggregate, error = transform._aggregate_group(grouped, group_value)
            assert error is None
            results.append(aggregate)

        contract = transform._output_contract_for(results)
        emitted_rows = [PipelineRow(r, contract) for r in results]

        # Must NOT raise SchemaConfigModeViolation.
        verify_schema_config_mode(
            output_schema_config=transform._output_schema_config,
            emitted_rows=emitted_rows,
            plugin_name="batch_stats",
            node_id="agg",
            run_id="r",
            row_id="row1",
            token_id="t1",
        )


# =============================================================================
# Bug fix tests: batch_stats.py bug cluster
# =============================================================================


class TestOutputFieldCompleteness:
    """Tests for elspeth-8051921704: conditional output fields must be tracked.

    declared_output_fields contains guaranteed fields only (for DAG contract propagation).
    _all_possible_output_keys tracks all fields that can appear in output, including
    conditional ones like skipped_non_finite. This set is used for collision detection.
    """

    def test_all_possible_output_keys_includes_skipped_non_finite(self):
        """_all_possible_output_keys must include skipped_non_finite fields.

        These fields appear in output when non-finite values are present.
        The collision check must know about them to prevent group_by overwrites.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": {"mode": "observed"}, "value_field": "amount"})
        assert "skipped_non_finite" in transform._all_possible_output_keys
        assert "skipped_non_finite_indices" in transform._all_possible_output_keys

    def test_declared_output_fields_excludes_conditional(self):
        """declared_output_fields should NOT include conditional fields.

        skipped_non_finite is conditional (only present when non-finite values
        exist), so it must not be in guaranteed_fields via declared_output_fields.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": {"mode": "observed"}, "value_field": "amount"})
        assert "skipped_non_finite" not in transform.declared_output_fields
        assert "skipped_non_finite_indices" not in transform.declared_output_fields


class TestFieldsAddedResultSync:
    """Tests for elspeth-fabb35458b: fields_added must match result dict keys."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_fields_added_matches_result_keys_with_skipped(self, ctx: PluginContext) -> None:
        """When non-finite values are skipped, fields_added must include the skip fields."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "compute_mean": True})
        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": float("nan")}),
            _make_row({"id": 3, "amount": 30.0}),
        ]
        result = transform.process(rows, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.success_reason is not None

        fields_added = set(result.success_reason["fields_added"])
        result_keys = set(result.row.to_dict().keys())
        assert fields_added == result_keys, f"fields_added {fields_added} must match result keys {result_keys}"

    def test_fields_added_matches_result_keys_without_skipped(self, ctx: PluginContext) -> None:
        """When all values are finite, fields_added must match result keys exactly."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats({"schema": DYNAMIC_SCHEMA, "value_field": "amount", "compute_mean": True})
        rows = [
            _make_row({"id": 1, "amount": 10.0}),
            _make_row({"id": 2, "amount": 20.0}),
        ]
        result = transform.process(rows, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.success_reason is not None

        fields_added = set(result.success_reason["fields_added"])
        result_keys = set(result.row.to_dict().keys())
        assert fields_added == result_keys


class TestGroupByCollisionAtInit:
    """Tests for elspeth-d375bde404: group_by collision check must cover all possible output keys."""

    def test_group_by_skipped_non_finite_collides(self) -> None:
        """group_by='skipped_non_finite' should be rejected at init time.

        Previously, the collision check only ran at process() time and only
        against keys present in the current batch's result dict. A batch
        without non-finite values would miss the collision.
        """
        from elspeth.plugins.transforms.batch_stats import BatchStats

        with pytest.raises(PluginConfigError, match="collides"):
            BatchStats(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "value_field": "amount",
                    "group_by": "skipped_non_finite",
                }
            )

    def test_group_by_count_collides_at_init(self) -> None:
        """group_by='count' should be rejected at init, not at process time."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        with pytest.raises(PluginConfigError, match="collides"):
            BatchStats(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "value_field": "amount",
                    "group_by": "count",
                }
            )


class TestBatchStatsGroupByInOutputContract:
    """Tests for group_by appearing in declared_output_fields when configured.

    Bug fix: group_by was consumed at runtime but never declared in output
    guarantees. This meant the DAG builder couldn't validate that downstream
    transforms requiring group_by would receive it.
    """

    def test_group_by_in_declared_output_fields(self):
        """group_by appears in declared_output_fields when configured."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
                "group_by": "category",
            }
        )
        assert "category" in transform.declared_output_fields

    def test_group_by_in_guaranteed_fields(self):
        """group_by appears in _output_schema_config.guaranteed_fields when configured."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
                "group_by": "region",
            }
        )
        assert transform._output_schema_config is not None
        assert "region" in transform._output_schema_config.guaranteed_fields

    def test_no_group_by_not_in_declared_fields(self):
        """Without group_by, only stat fields are in declared_output_fields."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        transform = BatchStats(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "amount",
            }
        )
        # Only stat fields, no group_by
        assert transform.declared_output_fields == frozenset({"count", "sum", "batch_size", "mean"})

    def test_group_by_collision_still_detected(self):
        """group_by collision with stat fields is still detected at init."""
        from elspeth.plugins.transforms.batch_stats import BatchStats

        with pytest.raises(PluginConfigError, match="collides"):
            BatchStats(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "value_field": "amount",
                    "group_by": "sum",
                }
            )


def test_non_finite_decimal_key_guarded() -> None:
    """B4.5-d: a non-finite Decimal key is caught by the static guard (parity with batch_effect_size).

    Decimal is not an allowed FieldContract type, so a Decimal key can only reach a
    transform through an object-typed field; the guard must still reject it. Exercised at
    the helper because the end-to-end path requires an object-typed key column.
    """
    from elspeth.plugins.transforms.batch_stats import BatchStats

    guard = BatchStats._is_non_finite_group_key
    assert guard(Decimal("nan")) is True
    assert guard(Decimal("inf")) is True
    assert guard(Decimal("-inf")) is True
    assert guard(Decimal("1")) is False
