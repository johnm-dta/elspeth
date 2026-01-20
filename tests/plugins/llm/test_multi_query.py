"""Tests for multi-query LLM support."""

import pytest

from elspeth.plugins.llm.multi_query import QuerySpec


class TestQuerySpec:
    """Tests for QuerySpec dataclass."""

    def test_query_spec_creation(self) -> None:
        """QuerySpec holds case study and criterion info."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "cs1_sym", "cs1_hist"],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG", "subcriteria": ["accuracy"]},
        )
        assert spec.case_study_name == "cs1"
        assert spec.criterion_name == "diagnosis"
        assert spec.output_prefix == "cs1_diagnosis"

    def test_query_spec_build_template_context(self) -> None:
        """QuerySpec builds template context with positional inputs and criterion."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "cs1_sym", "cs1_hist"],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG", "subcriteria": ["accuracy"]},
        )
        row = {
            "cs1_bg": "45yo male",
            "cs1_sym": "chest pain",
            "cs1_hist": "family history",
            "other_field": "ignored",
        }
        context = spec.build_template_context(row)

        assert context["input_1"] == "45yo male"
        assert context["input_2"] == "chest pain"
        assert context["input_3"] == "family history"
        assert context["criterion"]["code"] == "DIAG"
        assert context["row"] == row  # Full row for row-based lookups

    def test_build_template_context_raises_on_missing_field(self) -> None:
        """Missing input field raises KeyError with informative message."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "missing_field"],
            output_prefix="cs1_diagnosis",
            criterion_data={},
        )
        row = {"cs1_bg": "data"}

        with pytest.raises(KeyError) as exc_info:
            spec.build_template_context(row)

        assert "missing_field" in str(exc_info.value)
        assert "cs1_diagnosis" in str(exc_info.value)

    def test_build_template_context_empty_input_fields(self) -> None:
        """Empty input_fields produces context with only criterion and row."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=[],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG"},
        )
        row = {"some_field": "value"}
        context = spec.build_template_context(row)

        # No input_N fields
        assert "input_1" not in context
        # But criterion and row are present
        assert context["criterion"] == {"code": "DIAG"}
        assert context["row"] == row
