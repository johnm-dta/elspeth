"""Tests for BatchDataQualityReport aggregation transform."""

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


class TestBatchDataQualityReport:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        assert BatchDataQualityReport.name == "batch_data_quality_report"
        assert BatchDataQualityReport.is_batch_aware is True

    def test_composer_hints_reference_real_output_fields(self) -> None:
        """Discovery guidance must stay aligned with emitted report schema."""
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        transform = BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": ["score"]})
        assistance = BatchDataQualityReport.get_agent_assistance(issue_code=None)

        assert assistance is not None
        hints = "\n".join(assistance.composer_hints)
        assert "quality_report_*" not in hints
        for field_name in ("field", "missing_count", "valid_rate", "batch_size"):
            assert field_name in transform.declared_output_fields
            assert field_name in hints

    def test_emits_one_quality_row_per_inspected_field(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        transform = BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": ["score", "label"]})
        rows = [
            _make_row({"score": 1.0, "label": "ok"}),
            _make_row({"score": None, "label": ""}),
            _make_row({"score": float("nan"), "label": "ok"}),
            _make_row({"score": 2.0, "label": "fail"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        reports = {row["field"]: row for row in result.rows}

        score = reports["score"]
        assert score["batch_size"] == 4
        assert score["missing_count"] == 1
        assert score["blank_string_count"] == 0
        assert score["non_finite_count"] == 1
        assert score["valid_count"] == 2
        assert score["distinct_count"] == 2
        assert score["duplicate_count"] == 0
        assert score["observed_type_counts"] == {"float": 3}

        label = reports["label"]
        assert label["missing_count"] == 0
        assert label["blank_string_count"] == 1
        assert label["valid_count"] == 3
        assert label["distinct_count"] == 2
        assert label["duplicate_count"] == 1
        assert label["observed_type_counts"] == {"str": 4}

    def test_distinct_count_preserves_bool_and_int_buckets(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        transform = BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": ["flag"]})
        rows = [
            _make_row({"flag": True}),
            _make_row({"flag": 1}),
            _make_row({"flag": True}),
            _make_row({"flag": 1}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["valid_count"] == 4
        assert result.row["distinct_count"] == 2
        assert result.row["duplicate_count"] == 2
        assert result.row["observed_type_counts"] == {"bool": 2, "int": 2}

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        transform = BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": ["score"]})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable


class TestBatchDataQualityReportConfig:
    @pytest.mark.parametrize("inspect_fields", [[], ["score", "score"], ["score", " "]])
    def test_invalid_inspect_fields_rejected_at_config_boundary(self, inspect_fields: list[str]) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        with pytest.raises(PluginConfigError):
            BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": inspect_fields})

    def test_excessive_inspect_fields_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        inspect_fields = [f"field_{index}" for index in range(129)]

        with pytest.raises(PluginConfigError, match="inspect_fields must contain at most 128 fields"):
            BatchDataQualityReport({"schema": DYNAMIC_SCHEMA, "inspect_fields": inspect_fields})

    def test_output_schema_config_guarantees_report_fields(self) -> None:
        from elspeth.plugins.transforms.batch_data_quality_report import BatchDataQualityReport

        transform = BatchDataQualityReport(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["score: float", "input_only: str"],
                    "guaranteed_fields": ["input_only"],
                },
                "inspect_fields": ["score"],
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert "input_only" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "batch_size",
                "blank_string_count",
                "blank_string_rate",
                "distinct_count",
                "duplicate_count",
                "field",
                "missing_count",
                "missing_rate",
                "non_finite_count",
                "non_scalar_count",
                "observed_count",
                "observed_type_counts",
                "valid_count",
                "valid_rate",
            }
        )
