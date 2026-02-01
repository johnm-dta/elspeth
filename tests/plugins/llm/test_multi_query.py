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
            case_study_data={"name": "cs1", "context": "Capability"},
        )
        assert spec.case_study_name == "cs1"
        assert spec.criterion_name == "diagnosis"
        assert spec.output_prefix == "cs1_diagnosis"

    def test_query_spec_build_template_context(self) -> None:
        """QuerySpec builds template context with positional inputs, criterion, and case_study."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "cs1_sym", "cs1_hist"],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG", "subcriteria": ["accuracy"]},
            case_study_data={"name": "cs1", "context": "Capability", "description": "Technical capability"},
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
        assert context["case_study"]["name"] == "cs1"
        assert context["case_study"]["context"] == "Capability"
        assert context["row"] == row  # Full row for row-based lookups

    def test_build_template_context_raises_on_missing_field(self) -> None:
        """Missing input field raises KeyError with informative message."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "missing_field"],
            output_prefix="cs1_diagnosis",
            criterion_data={},
            case_study_data={"name": "cs1"},
        )
        row = {"cs1_bg": "data"}

        with pytest.raises(KeyError) as exc_info:
            spec.build_template_context(row)

        assert "missing_field" in str(exc_info.value)
        assert "cs1_diagnosis" in str(exc_info.value)

    def test_build_template_context_empty_input_fields(self) -> None:
        """Empty input_fields produces context with criterion, case_study, and row."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=[],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG"},
            case_study_data={"name": "cs1", "context": "Capability"},
        )
        row = {"some_field": "value"}
        context = spec.build_template_context(row)

        # No input_N fields
        assert "input_1" not in context
        # But criterion, case_study, and row are present
        assert context["criterion"] == {"code": "DIAG"}
        assert context["case_study"] == {"name": "cs1", "context": "Capability"}
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

    def test_case_study_with_context_and_description(self) -> None:
        """CaseStudyConfig accepts optional context and description."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        config = CaseStudyConfig.from_dict(
            {
                "name": "cs1",
                "input_fields": ["field_a"],
                "context": "Capability",
                "description": "Technical capability demonstration",
            }
        )
        assert config.name == "cs1"
        assert config.context == "Capability"
        assert config.description == "Technical capability demonstration"

    def test_case_study_to_template_data(self) -> None:
        """to_template_data returns dict suitable for template injection."""
        from elspeth.plugins.llm.multi_query import CaseStudyConfig

        config = CaseStudyConfig.from_dict(
            {
                "name": "cs1",
                "input_fields": ["field_a", "field_b"],
                "context": "Capability",
                "description": "Technical capability",
            }
        )
        data = config.to_template_data()

        assert data == {
            "name": "cs1",
            "context": "Capability",
            "description": "Technical capability",
            "input_fields": ["field_a", "field_b"],
        }


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
                    "response_format": "standard",
                    "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
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
                    "response_format": "standard",
                    "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
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
                    "response_format": "standard",
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
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
                "response_format": "standard",
                "output_mapping": {
                    "score": {"suffix": "score", "type": "integer"},
                    "rationale": {"suffix": "rationale", "type": "string"},
                },
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )
        assert len(config.case_studies) == 2
        assert len(config.criteria) == 2
        assert "score" in config.output_mapping
        assert "rationale" in config.output_mapping
        assert config.output_mapping["score"].suffix == "score"
        assert config.output_mapping["rationale"].suffix == "rationale"

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
                    "required_input_fields": [],  # Explicit opt-out for this test
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
                "response_format": "standard",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
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

    def test_duplicate_case_study_criterion_pairs_rejected(self) -> None:
        """Duplicate (case_study.name, criterion.name) pairs are rejected.

        Regression test for P1-2026-01-31-multi-query-output-key-collisions.
        """
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError) as exc_info:
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [
                        {"name": "cs1", "input_fields": ["a"]},
                        {"name": "cs1", "input_fields": ["b"]},  # Duplicate name!
                    ],
                    "criteria": [{"name": "diagnosis"}],
                    "response_format": "standard",
                    "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],
                }
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_duplicate_criterion_names_rejected(self) -> None:
        """Duplicate criterion names are rejected.

        Regression test for P1-2026-01-31-multi-query-output-key-collisions.
        """
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError) as exc_info:
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                    "criteria": [
                        {"name": "diagnosis"},
                        {"name": "diagnosis"},  # Duplicate name!
                    ],
                    "response_format": "standard",
                    "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],
                }
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_output_suffix_collision_with_reserved_suffix_rejected(self) -> None:
        """Output mapping suffix that collides with reserved suffixes is rejected.

        Reserved suffixes include _usage, _model, _template_hash, etc.
        Regression test for P1-2026-01-31-multi-query-output-key-collisions.
        """
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        with pytest.raises(PluginConfigError) as exc_info:
            MultiQueryConfig.from_dict(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "key",
                    "template": "{{ row.input_1 }}",
                    "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                    "criteria": [{"name": "diagnosis"}],
                    "response_format": "standard",
                    "output_mapping": {
                        "score": {"suffix": "score", "type": "integer"},
                        "usage": {"suffix": "usage", "type": "string"},  # Collides with _usage!
                    },
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],
                }
            )
        assert "reserved" in str(exc_info.value).lower() or "usage" in str(exc_info.value).lower()


class TestOutputFieldConfig:
    """Tests for OutputFieldConfig and JSON schema generation."""

    def test_string_type_to_json_schema(self) -> None:
        """String type generates correct JSON schema."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "rationale", "type": "string"})
        schema = config.to_json_schema()

        assert schema == {"type": "string"}

    def test_integer_type_to_json_schema(self) -> None:
        """Integer type generates correct JSON schema."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "score", "type": "integer"})
        schema = config.to_json_schema()

        assert schema == {"type": "integer"}

    def test_number_type_to_json_schema(self) -> None:
        """Number type generates correct JSON schema."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "probability", "type": "number"})
        schema = config.to_json_schema()

        assert schema == {"type": "number"}

    def test_boolean_type_to_json_schema(self) -> None:
        """Boolean type generates correct JSON schema."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "is_valid", "type": "boolean"})
        schema = config.to_json_schema()

        assert schema == {"type": "boolean"}

    def test_enum_type_to_json_schema(self) -> None:
        """Enum type generates JSON schema with allowed values."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict(
            {
                "suffix": "confidence",
                "type": "enum",
                "values": ["low", "medium", "high"],
            }
        )
        schema = config.to_json_schema()

        assert schema == {"type": "string", "enum": ["low", "medium", "high"]}

    def test_enum_requires_values(self) -> None:
        """Enum type without values raises validation error."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict({"suffix": "level", "type": "enum"})

    def test_enum_requires_non_empty_values(self) -> None:
        """Enum type with empty values list raises validation error."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict({"suffix": "level", "type": "enum", "values": []})

    def test_non_enum_rejects_values(self) -> None:
        """Non-enum types reject values parameter."""
        from elspeth.plugins.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict(
                {
                    "suffix": "score",
                    "type": "integer",
                    "values": ["a", "b"],  # Invalid for non-enum
                }
            )


class TestResponseFormatBuilding:
    """Tests for build_json_schema and build_response_format methods."""

    def test_build_json_schema_single_field(self) -> None:
        """build_json_schema generates valid schema for single field."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "structured",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        schema = config.build_json_schema()

        assert schema == {
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"],
            "additionalProperties": False,
        }

    def test_build_json_schema_multiple_fields(self) -> None:
        """build_json_schema generates valid schema for multiple fields."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "structured",
                "output_mapping": {
                    "score": {"suffix": "score", "type": "integer"},
                    "rationale": {"suffix": "rationale", "type": "string"},
                    "confidence": {"suffix": "confidence", "type": "enum", "values": ["low", "medium", "high"]},
                },
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        schema = config.build_json_schema()

        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"score", "rationale", "confidence"}
        assert schema["properties"]["score"] == {"type": "integer"}
        assert schema["properties"]["rationale"] == {"type": "string"}
        assert schema["properties"]["confidence"] == {"type": "string", "enum": ["low", "medium", "high"]}

    def test_build_response_format_standard_mode(self) -> None:
        """build_response_format returns json_object for standard mode."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "standard",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        response_format = config.build_response_format()

        assert response_format == {"type": "json_object"}

    def test_build_response_format_structured_mode(self) -> None:
        """build_response_format returns json_schema for structured mode."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "structured",
                "output_mapping": {
                    "score": {"suffix": "score", "type": "integer"},
                    "rationale": {"suffix": "rationale", "type": "string"},
                },
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        response_format = config.build_response_format()

        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "query_response"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["type"] == "object"
        assert "score" in response_format["json_schema"]["schema"]["properties"]
        assert "rationale" in response_format["json_schema"]["schema"]["properties"]

    def test_response_format_default_is_standard(self) -> None:
        """response_format defaults to standard when not specified."""
        from elspeth.plugins.llm.multi_query import MultiQueryConfig, ResponseFormat

        config = MultiQueryConfig.from_dict(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                # No response_format specified
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert config.response_format == ResponseFormat.STANDARD
        assert config.build_response_format() == {"type": "json_object"}
