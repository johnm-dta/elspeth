"""Tests for BatchDistributionProfile aggregation transform."""

from decimal import Decimal
from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_row(data: dict[str, Any]):
    """Create a PipelineRow with OBSERVED contract for testing."""
    fields = tuple(
        make_field(key, type(value) if value is not None else object, original_name=key, required=False, source="inferred")
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


class TestBatchDistributionProfile:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        assert BatchDistributionProfile.name == "batch_distribution_profile"
        assert BatchDistributionProfile.is_batch_aware is True

    def test_returns_assistance_for_numeric_value_field_issue(self) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        assistance = BatchDistributionProfile.get_agent_assistance(
            issue_code="batch_distribution_profile.value_field.numeric",
        )

        assert assistance is not None
        assert assistance.plugin_name == "batch_distribution_profile"
        assert assistance.issue_code == "batch_distribution_profile.value_field.numeric"
        assert "numeric" in assistance.summary
        assert any("batch_top_k" in fix for fix in assistance.suggested_fixes)
        assert any("theme frequency" in hint for hint in assistance.composer_hints)

    def test_computes_distribution_summary_for_single_field(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 1.0}),
            _make_row({"id": 2, "score": 2.0}),
            _make_row({"id": 3, "score": 3.0}),
            _make_row({"id": 4, "score": 4.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["field"] == "score"
        assert result.row["count"] == 4
        assert result.row["batch_size"] == 4
        assert result.row["missing_count"] == 0
        assert result.row["non_finite_count"] == 0
        assert result.row["min"] == 1.0
        assert result.row["max"] == 4.0
        assert result.row["mean"] == 2.5
        assert result.row["median"] == 2.5
        assert result.row["p25"] == 1.75
        assert result.row["p75"] == 3.25
        assert result.row["stdev"] == pytest.approx(1.2909944487358056)
        assert result.row["summary"] == (
            "Distribution profile for score: 4 rows, 4 finite values, 0 missing, "
            "0 non-finite, mean 2.500, median 2.500, range 1.000 to 4.000."
        )

    def test_missing_values_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": None}),
            _make_row({"id": 3, "score": 20.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 2
        assert result.row["batch_size"] == 3
        assert result.row["missing_count"] == 1
        assert result.row["missing_indices"] == (1,)
        assert result.row["mean"] == 15.0

    def test_non_finite_values_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": float("nan")}),
            _make_row({"id": 3, "score": float("inf")}),
            _make_row({"id": 4, "score": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 2
        assert result.row["batch_size"] == 4
        assert result.row["non_finite_count"] == 2
        assert result.row["non_finite_indices"] == (1, 2)
        assert result.row["mean"] == 20.0

    def test_all_non_finite_or_missing_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": None}),
            _make_row({"id": 2, "score": float("nan")}),
            _make_row({"id": 3, "score": float("-inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "no_finite_values"
        assert result.reason["batch_size"] == 3
        assert result.reason["valid_count"] == 0
        assert result.reason["skipped_count"] == 3
        assert result.reason["row_errors"] == [
            {"row_index": 0, "reason": "missing_value"},
            {"row_index": 1, "reason": "non_finite_value"},
            {"row_index": 2, "reason": "non_finite_value"},
        ]

    def test_non_numeric_values_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": "not_a_number"}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)

    @pytest.mark.parametrize("group_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_group_key_returns_error_before_success(self, ctx: PluginContext, group_value: float) -> None:
        """Non-finite group_by key must error before producing any output (B4.5-d)."""
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": "variant"})
        rows = [
            _make_row({"variant": "A", "score": 1.0}),
            _make_row({"variant": group_value, "score": 2.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_group_key"
        assert result.reason["field"] == "variant"
        assert not result.retryable

    def test_group_by_emits_one_profile_per_group(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": "variant"})

        rows = [
            _make_row({"id": 1, "variant": "A", "score": 1.0}),
            _make_row({"id": 2, "variant": "B", "score": 10.0}),
            _make_row({"id": 3, "variant": "A", "score": 3.0}),
            _make_row({"id": 4, "variant": "B", "score": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["variant"] for row in result.rows] == ["A", "B"]

        profiles = {row["variant"]: row for row in result.rows}
        assert profiles["A"]["count"] == 2
        assert profiles["A"]["mean"] == 2.0
        assert profiles["A"]["median"] == 2.0
        assert profiles["A"]["summary"] == (
            "Distribution profile for score grouped by variant=A: 2 rows, 2 finite values, "
            "0 missing, 0 non-finite, mean 2.000, median 2.000, range 1.000 to 3.000."
        )
        assert profiles["B"]["count"] == 2
        assert profiles["B"]["mean"] == 20.0
        assert profiles["B"]["median"] == 20.0
        assert profiles["B"]["summary"] == (
            "Distribution profile for score grouped by variant=B: 2 rows, 2 finite values, "
            "0 missing, 0 non-finite, mean 20.000, median 20.000, range 10.000 to 30.000."
        )

    def test_group_by_preserves_bool_and_int_buckets(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": "variant"})
        rows = [
            _make_row({"variant": True, "score": 1.0}),
            _make_row({"variant": 1, "score": 10.0}),
            _make_row({"variant": True, "score": 3.0}),
            _make_row({"variant": 1, "score": 30.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [(type(row["variant"]).__name__, row["variant"]) for row in result.rows] == [("bool", True), ("int", 1)]
        assert [row["mean"] for row in result.rows] == [2.0, 20.0]

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable

    def test_single_value_reports_none_stdev(self, ctx: PluginContext) -> None:
        """n=1 stdev is undefined -- must emit None, never 0.0 (B4.5-a-distribution_profile)."""
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})
        rows = [_make_row({"score": 42.0})]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["count"] == 1
        # stdev undefined at n=1 -- honest None, never 0.0
        assert result.row["stdev"] is None


class TestBatchDistributionProfileConfig:
    @pytest.mark.parametrize("blank_value_field", ["", "   "])
    def test_blank_value_field_rejected_at_config_boundary(self, blank_value_field: str) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        with pytest.raises(PluginConfigError, match="value_field must not be empty"):
            BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": blank_value_field})

    @pytest.mark.parametrize("blank_group_by", ["", "   "])
    def test_blank_group_by_rejected_at_config_boundary(self, blank_group_by: str) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        with pytest.raises(PluginConfigError, match="group_by must not be empty"):
            BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": blank_group_by})

    @pytest.mark.parametrize("colliding_group_by", ["field", "count", "mean", "p25", "missing_count"])
    def test_group_by_collisions_rejected_at_config_boundary(self, colliding_group_by: str) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        with pytest.raises(PluginConfigError, match="collides with profile output key"):
            BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": colliding_group_by})

    def test_value_field_required_in_input_schema_without_group_by(self) -> None:
        # Misspelled value_field in a pipeline YAML must fail at config / DAG
        # validation, not at first batch flush. Mirrors the cohort behavior of
        # batch_top_k, which adds cfg.field to required_fields unconditionally.
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        required = transform._schema_config.required_fields or ()
        assert "score" in required

    def test_value_field_required_in_input_schema_with_group_by(self) -> None:
        # When group_by is configured, BOTH value_field and group_by must be
        # required in the input schema. Without the fix, only group_by is added.
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile({"schema": DYNAMIC_SCHEMA, "value_field": "score", "group_by": "variant"})

        required = transform._schema_config.required_fields or ()
        assert "score" in required
        assert "variant" in required

    def test_user_supplied_required_fields_preserved_when_value_field_added(self) -> None:
        # The fix must add value_field to required_fields without dropping the
        # user's existing required_fields entries.
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int", "score: float"],
                    "required_fields": ["id"],
                },
                "value_field": "score",
            }
        )

        required = set(transform._schema_config.required_fields or ())
        assert required == {"id", "score"}

    def test_output_schema_config_guarantees_profile_fields_and_group_by(self) -> None:
        from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

        transform = BatchDistributionProfile(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int", "variant: str", "score: float", "upstream_only: str"],
                    "required_fields": ["score"],
                    "guaranteed_fields": ["upstream_only"],
                },
                "value_field": "score",
                "group_by": "variant",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert cfg.required_fields is None
        assert "upstream_only" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "batch_size",
                "count",
                "field",
                "max",
                "mean",
                "median",
                "min",
                "missing_count",
                "non_finite_count",
                "p25",
                "p75",
                "stdev",
                "summary",
                "variant",
            }
        )


def test_non_finite_decimal_key_guarded() -> None:
    """B4.5-d: a non-finite Decimal key is caught by the static guard (parity with batch_effect_size).

    Decimal is not an allowed FieldContract type, so a Decimal key can only reach a
    transform through an object-typed field; the guard must still reject it. Exercised at
    the helper because the end-to-end path requires an object-typed key column.
    """
    from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

    guard = BatchDistributionProfile._is_non_finite_group_key
    assert guard(Decimal("nan")) is True
    assert guard(Decimal("inf")) is True
    assert guard(Decimal("-inf")) is True
    assert guard(Decimal("1")) is False
