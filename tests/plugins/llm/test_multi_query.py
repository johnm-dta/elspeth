"""Tests for multi-query LLM support."""

import pytest

from elspeth.plugins.config_base import PluginConfigError
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


class TestCaseStudyConfig:
    """Tests for CaseStudyConfig validation."""

    def test_case_study_requires_name(self) -> None:
        """CaseStudyConfig requires name."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        with pytest.raises(PluginConfigError):
            CaseStudyConfig.from_dict({"input_fields": ["a", "b"]})

    def test_case_study_requires_input_fields(self) -> None:
        """CaseStudyConfig requires input_fields."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        with pytest.raises(PluginConfigError):
            CaseStudyConfig.from_dict({"name": "cs1"})

    def test_valid_case_study(self) -> None:
        """Valid CaseStudyConfig passes validation."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        config = CaseStudyConfig.from_dict(
            {
                "name": "cs1",
                "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"],
            }
        )
        assert config.name == "cs1"
        assert config.input_fields == ["cs1_bg", "cs1_sym", "cs1_hist"]

    def test_case_study_rejects_empty_input_fields(self) -> None:
        """CaseStudyConfig rejects empty input_fields list."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        with pytest.raises(PluginConfigError):
            CaseStudyConfig.from_dict({"name": "cs1", "input_fields": []})


class TestCriterionConfig:
    """Tests for CriterionConfig validation."""

    def test_criterion_requires_name(self) -> None:
        """CriterionConfig requires name."""
        from elspeth.plugins.llm.multi_query import CriterionConfig

        with pytest.raises(PluginConfigError):
            CriterionConfig.from_dict({"code": "DIAG"})

    def test_valid_criterion_minimal(self) -> None:
        """CriterionConfig works with just name."""
        from elspeth.plugins.llm.multi_query import CriterionConfig

        config = CriterionConfig.from_dict({"name": "diagnosis"})
        assert config.name == "diagnosis"
        assert config.code is None
        assert config.subcriteria == []

    def test_valid_criterion_full(self) -> None:
        """CriterionConfig accepts all fields."""
        from elspeth.plugins.llm.multi_query import CriterionConfig

        config = CriterionConfig.from_dict(
            {
                "name": "diagnosis",
                "code": "DIAG",
                "description": "Assess diagnostic accuracy",
                "subcriteria": ["accuracy", "evidence"],
            }
        )
        assert config.name == "diagnosis"
        assert config.code == "DIAG"
        assert config.description == "Assess diagnostic accuracy"
        assert config.subcriteria == ["accuracy", "evidence"]

    def test_criterion_to_template_data(self) -> None:
        """to_template_data returns dict suitable for template injection."""
        from elspeth.plugins.llm.multi_query import CriterionConfig

        config = CriterionConfig.from_dict(
            {
                "name": "diagnosis",
                "code": "DIAG",
                "description": "Assess accuracy",
                "subcriteria": ["a", "b"],
            }
        )
        data = config.to_template_data()

        assert data == {
            "name": "diagnosis",
            "code": "DIAG",
            "description": "Assess accuracy",
            "subcriteria": ["a", "b"],
        }


class TestMultiQueryConfig:
    """Tests for MultiQueryConfig validation."""

    def test_config_requires_case_studies(self) -> None:
        """MultiQueryConfig requires case_studies."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "criteria": [{"name": "diagnosis"}],
                    "response_format": "json",
                    "output_mapping": {"score": "score"},
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_config_requires_criteria(self) -> None:
        """MultiQueryConfig requires criteria."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                    "response_format": "json",
                    "output_mapping": {"score": "score"},
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_config_requires_output_mapping(self) -> None:
        """MultiQueryConfig requires output_mapping."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                    "criteria": [{"name": "diagnosis"}],
                    "response_format": "json",
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_valid_config(self) -> None:
        """Valid MultiQueryConfig passes validation."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }} {{ row.criterion.name }}",
                "system_prompt": "You are an AI.",
                "case_studies": [
                    {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym"]},
                    {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym"]},
                ],
                "criteria": [
                    {"name": "diagnosis", "code": "DIAG"},
                    {"name": "treatment", "code": "TREAT"},
                ],
                "response_format": "json",
                "output_mapping": {"score": "score", "rationale": "rationale"},
                "schema": {"fields": "dynamic"},
            }
        )
        assert len(config.case_studies) == 2
        assert len(config.criteria) == 2
        assert config.output_mapping == {"score": "score", "rationale": "rationale"}

    def test_config_rejects_empty_output_mapping(self) -> None:
        """MultiQueryConfig rejects empty output_mapping dict."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                    "criteria": [{"name": "diagnosis"}],
                    "output_mapping": {},  # Empty!
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_expand_queries_creates_cross_product(self) -> None:
        """expand_queries creates QuerySpec for each (case_study, criterion) pair."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [
                    {"name": "cs1", "input_fields": ["cs1_a"]},
                    {"name": "cs2", "input_fields": ["cs2_a"]},
                ],
                "criteria": [
                    {"name": "diagnosis", "code": "DIAG"},
                    {"name": "treatment", "code": "TREAT"},
                ],
                "response_format": "json",
                "output_mapping": {"score": "score"},
                "schema": {"fields": "dynamic"},
            }
        )

        specs = config.expand_queries()

        # 2 case studies x 2 criteria = 4 queries
        assert len(specs) == 4

        # Check cross-product
        prefixes = [s.output_prefix for s in specs]
        assert "cs1_diagnosis" in prefixes
        assert "cs1_treatment" in prefixes
        assert "cs2_diagnosis" in prefixes
        assert "cs2_treatment" in prefixes
