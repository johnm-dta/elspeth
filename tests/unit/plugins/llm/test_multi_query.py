"""Tests for multi-query LLM support.

Tests for QuerySpec, resolve_queries, OutputFieldConfig, ResponseFormat,
and multi-query transform instantiation via unified LLMTransform.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.llm.multi_query import QueryDefinition
from elspeth.plugins.transforms.llm.transform import LLMTransform

# Re-export chaosllm_server fixture for field collision tests
from tests.fixtures.chaosllm import chaosllm_server  # noqa: F401


def _mapping_defs(defs: dict[str, dict[str, Any]]) -> dict[str, QueryDefinition]:
    """Build the typed mapping form of ``queries`` (value carries no ``name``)."""
    return {name: QueryDefinition(**spec) for name, spec in defs.items()}


def _list_defs(items: list[dict[str, Any]]) -> list[QueryDefinition]:
    """Build the typed list form of ``queries`` (each entry carries ``name``)."""
    return [QueryDefinition(**item) for item in items]


# ---------------------------------------------------------------------------
# Config helpers (inline, using the new unified format)
# ---------------------------------------------------------------------------

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_llm_config(**overrides: Any) -> dict[str, Any]:
    """Create valid LLMTransform multi-query config with optional overrides."""
    config: dict[str, Any] = {
        "provider": "azure",
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "prompt_template": "Evaluate: {{ row.text_content }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "schema": DYNAMIC_SCHEMA,
        "required_input_fields": [],
        "pool_size": 1,
        "queries": {
            "cs1_diag": {
                "input_fields": {"text_content": "cs1_bg"},
                "output_fields": [
                    {"suffix": "score", "type": "integer"},
                    {"suffix": "rationale", "type": "string"},
                ],
            },
        },
    }
    config.update(overrides)
    return config


class TestOutputFieldConfig:
    """Tests for OutputFieldConfig and JSON schema generation."""

    def test_string_type_to_json_schema(self) -> None:
        """String type generates correct JSON schema."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "rationale", "type": "string"})
        schema = config.to_json_schema()

        assert schema == {"type": "string"}

    def test_integer_type_to_json_schema(self) -> None:
        """Integer type generates correct JSON schema."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "score", "type": "integer"})
        schema = config.to_json_schema()

        assert schema == {"type": "integer"}

    def test_number_type_to_json_schema(self) -> None:
        """Number type generates correct JSON schema."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "probability", "type": "number"})
        schema = config.to_json_schema()

        assert schema == {"type": "number"}

    def test_boolean_type_to_json_schema(self) -> None:
        """Boolean type generates correct JSON schema."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        config = OutputFieldConfig.from_dict({"suffix": "is_valid", "type": "boolean"})
        schema = config.to_json_schema()

        assert schema == {"type": "boolean"}

    def test_enum_type_to_json_schema(self) -> None:
        """Enum type generates JSON schema with allowed values."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

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
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict({"suffix": "level", "type": "enum"})

    def test_enum_requires_non_empty_values(self) -> None:
        """Enum type with empty values list raises validation error."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict({"suffix": "level", "type": "enum", "values": []})

    def test_non_enum_rejects_values(self) -> None:
        """Non-enum types reject values parameter."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig

        with pytest.raises(PluginConfigError):
            OutputFieldConfig.from_dict(
                {
                    "suffix": "score",
                    "type": "integer",
                    "values": ["a", "b"],  # Invalid for non-enum
                }
            )


class TestMultiQueryDeclaredOutputFields:
    """Tests for declared_output_fields on unified LLMTransform.

    Field collision detection is enforced centrally by TransformExecutor
    (see TestTransformExecutor in test_executors.py). These tests verify
    that LLMTransform correctly declares its output fields so the
    executor can perform pre-execution collision checks.
    """

    def test_declared_output_fields_contains_prefixed_response_field(self) -> None:
        """Multi-query declared_output_fields includes query-prefixed fields."""
        transform = LLMTransform(_make_llm_config())
        # Multi-query declares prefixed fields matching actual output
        assert "cs1_diag_llm_response" in transform.declared_output_fields

    def test_declared_output_fields_excludes_audit_fields(self) -> None:
        """Multi-query declared_output_fields excludes audit fields.

        Audit fields (template_hash, variables_hash, etc.) now travel via
        success_reason["metadata"], not the output row. Only operational
        fields (usage, model) remain in declared_output_fields.
        """
        transform = LLMTransform(_make_llm_config())

        # Guaranteed operational fields (prefixed with query name)
        assert "cs1_diag_llm_response_usage" in transform.declared_output_fields
        assert "cs1_diag_llm_response_model" in transform.declared_output_fields
        # Audit fields are NOT in declared_output_fields
        assert "cs1_diag_llm_response_template_hash" not in transform.declared_output_fields

    def test_declared_output_fields_with_multiple_queries(self) -> None:
        """Multi-query declared_output_fields covers all query prefixes."""
        config = _make_llm_config(
            queries={
                "cs1_diagnosis": {
                    "input_fields": {"text_content": "cs1_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
                "cs2_diagnosis": {
                    "input_fields": {"text_content": "cs2_bg"},
                    "output_fields": [
                        {"suffix": "score", "type": "integer"},
                        {"suffix": "rationale", "type": "string"},
                    ],
                },
            },
        )

        transform = LLMTransform(config)

        # Both query prefixes must be declared
        assert "cs1_diagnosis_llm_response" in transform.declared_output_fields
        assert "cs2_diagnosis_llm_response" in transform.declared_output_fields
        # Extracted fields too
        assert "cs1_diagnosis_score" in transform.declared_output_fields
        assert "cs2_diagnosis_rationale" in transform.declared_output_fields

    def test_declared_output_fields_is_nonempty(self) -> None:
        """declared_output_fields is populated for schema evolution recording."""
        transform = LLMTransform(_make_llm_config())
        assert transform.declared_output_fields


class TestResolveQueriesDuplicateNames:
    """Tests for duplicate query name rejection in resolve_queries().

    Bug: list-form configs don't enforce unique spec.name values. If two
    queries share a name, they emit the same prefixed output keys (e.g.,
    "{name}_response", "{name}_metadata"), and later dict.update() merges
    silently overwrite earlier query results, losing data.

    Dict-form configs are naturally protected (Python dict keys are unique),
    but list-form configs can have duplicate "name" fields.
    """

    def test_duplicate_names_in_list_form_rejected(self) -> None:
        """List-form configs with duplicate query names raise ValueError."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="Duplicate query name"):
            resolve_queries(
                _list_defs(
                    [
                        {
                            "name": "diagnosis",
                            "input_fields": {"text": "col_a"},
                        },
                        {
                            "name": "diagnosis",
                            "input_fields": {"text": "col_b"},
                        },
                    ]
                )
            )

    def test_duplicate_names_in_query_spec_list_rejected(self) -> None:
        """QuerySpec list with duplicate names raises ValueError."""
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec, resolve_queries

        with pytest.raises(ValueError, match="Duplicate query name"):
            resolve_queries(
                [
                    QuerySpec(name="scoring", input_fields=MappingProxyType({"x": "a"})),
                    QuerySpec(name="scoring", input_fields=MappingProxyType({"x": "b"})),
                ]
            )

    def test_unique_names_in_list_form_accepted(self) -> None:
        """List-form configs with unique query names work fine."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        specs = resolve_queries(
            _list_defs(
                [
                    {
                        "name": "diagnosis_1",
                        "input_fields": {"text": "col_a"},
                    },
                    {
                        "name": "diagnosis_2",
                        "input_fields": {"text": "col_b"},
                    },
                ]
            )
        )
        assert len(specs) == 2
        assert specs[0].name == "diagnosis_1"
        assert specs[1].name == "diagnosis_2"

    def test_dict_form_naturally_unique(self) -> None:
        """Dict-form configs have naturally unique names (sanity check)."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        # Python dicts can't have duplicate keys, so this is always safe
        specs = resolve_queries(
            _mapping_defs(
                {
                    "query_a": {"input_fields": {"text": "col_a"}},
                    "query_b": {"input_fields": {"text": "col_b"}},
                }
            )
        )
        assert len(specs) == 2


class TestResolveQueriesReservedSuffixes:
    """Regression: reserved suffix collision was warning-only.

    Output field suffixes like 'usage', 'model', 'error' collide with
    system-reserved LLM suffixes. The full output key '{name}_{suffix}'
    would silently overwrite structured LLM metadata. Must raise ValueError,
    not just log a warning.
    """

    def test_reserved_suffix_usage_raises_error(self) -> None:
        """Suffix 'usage' collides with LLM usage metadata — must error."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="reserved LLM suffix"):
            resolve_queries(
                _mapping_defs(
                    {
                        "query_a": {
                            "input_fields": {"text": "col_a"},
                            "output_fields": [
                                {"suffix": "usage", "type": "string"},
                            ],
                        },
                    }
                )
            )

    def test_reserved_suffix_model_raises_error(self) -> None:
        """Suffix 'model' collides with LLM model metadata — must error."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="reserved LLM suffix"):
            resolve_queries(
                _mapping_defs(
                    {
                        "query_a": {
                            "input_fields": {"text": "col_a"},
                            "output_fields": [
                                {"suffix": "model", "type": "string"},
                            ],
                        },
                    }
                )
            )

    def test_reserved_suffix_error_raises_error(self) -> None:
        """Suffix 'error' collides with multi-query error handling — must error."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="reserved LLM suffix"):
            resolve_queries(
                _mapping_defs(
                    {
                        "query_a": {
                            "input_fields": {"text": "col_a"},
                            "output_fields": [
                                {"suffix": "error", "type": "string"},
                            ],
                        },
                    }
                )
            )

    def test_audit_only_suffixes_are_accepted(self) -> None:
        """Audit-only suffixes do not collide with multi-query output rows."""
        from elspeth.plugins.transforms.llm import LLM_AUDIT_SUFFIXES
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        for suffix in (s.lstrip("_") for s in LLM_AUDIT_SUFFIXES):
            specs = resolve_queries(
                _mapping_defs(
                    {
                        f"query_{suffix}": {
                            "input_fields": {"text": "col_a"},
                            "output_fields": [
                                {"suffix": suffix, "type": "string"},
                            ],
                        },
                    }
                )
            )

            assert specs[0].output_fields is not None
            assert specs[0].output_fields[0].suffix == suffix

    def test_non_reserved_suffix_accepted(self) -> None:
        """Non-reserved suffixes must be accepted without error."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        specs = resolve_queries(
            _mapping_defs(
                {
                    "query_a": {
                        "input_fields": {"text": "col_a"},
                        "output_fields": [
                            {"suffix": "score", "type": "integer"},
                            {"suffix": "rationale", "type": "string"},
                        ],
                    },
                }
            )
        )
        assert len(specs) == 1


class TestQueryDefinitionDualForm:
    """Both accepted authoring forms normalize to identical QuerySpec lists.

    Mapping form keys each query by name (value omits ``name``); list form
    carries ``name`` on each entry. Under ``extra=forbid`` the asymmetry is real
    — a mapping value must not require ``name`` while a list entry must — but
    ``resolve_queries`` erases it, producing byte-identical runtime specs.
    """

    def test_mapping_and_list_forms_produce_identical_query_specs(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        shared_spec = {
            "input_fields": {"text": "content"},
            "response_format": "structured",
            "output_fields": [
                {"suffix": "score", "type": "integer"},
                {"suffix": "band", "type": "enum", "values": ["low", "high"]},
            ],
            "template": "Rate {{ text }}",
            "max_tokens": 128,
        }

        mapping_form = _mapping_defs({"clarity": dict(shared_spec)})
        list_form = _list_defs([{"name": "clarity", **shared_spec}])

        mapping_specs = resolve_queries(mapping_form)
        list_specs = resolve_queries(list_form)

        # Byte-identical runtime specs: same order, names, input_fields,
        # response_format, output_fields, template, and max_tokens.
        assert mapping_specs == list_specs
        assert [spec.name for spec in mapping_specs] == ["clarity"]
        assert mapping_specs[0].output_fields == list_specs[0].output_fields

    def test_multi_query_dual_form_order_and_equality_preserved(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        mapping_form = _mapping_defs(
            {
                "clarity": {"input_fields": {"text": "a"}},
                "depth": {"input_fields": {"text": "b"}},
            }
        )
        list_form = _list_defs(
            [
                {"name": "clarity", "input_fields": {"text": "a"}},
                {"name": "depth", "input_fields": {"text": "b"}},
            ]
        )

        assert resolve_queries(mapping_form) == resolve_queries(list_form)

    def test_list_entry_without_name_is_rejected_safely(self) -> None:
        """A list-form entry that omits ``name`` fails closed with a ValueError."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="must include a 'name'"):
            resolve_queries([QueryDefinition(input_fields={"text": "a"})])

    def test_mapping_value_forbids_name_via_extra_forbid(self) -> None:
        """A raw mapping value that carries its own ``name`` key would be extra.

        The mapping form injects the key as ``name`` at the ``LLMConfig`` layer;
        constructing ``QueryDefinition`` from a mapping value that itself declares
        ``name`` is legal (name is a real field), but a genuinely unexpected key
        is rejected under ``extra=forbid`` — proving the model is closed.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryDefinition.model_validate({"input_fields": {"text": "a"}, "unexpected_key": 1})


class TestQueryTemplateCompileParity:
    """queries.*.template must fail at CONFIG time exactly like prompt_template.

    Pack pressure-suite run 4 (P4, compose-vs-execute parity class): the
    candidate validator Jinja-compiles options.prompt_template (LLMConfig
    field validator) but never queries.*.template — three "passing" pipelines
    carried {{interpretation:...}} tokens inside per-query templates that
    crash ENGINE BUILD (QueryExecutor pre-compiles overrides at init) after
    operator review resolution. Compose-ready passed what execute refuses.
    """

    def test_interpretation_token_in_query_template_is_rejected_with_the_platform_gap_named(self) -> None:
        import pytest

        from elspeth.plugins.transforms.llm.multi_query import QueryDefinition

        with pytest.raises(ValueError, match="interpretation"):
            QueryDefinition(
                input_fields={"field_a": "field_a"},
                template="Rate {{interpretation:cool}} for {{ field_a }}",
            )

    def test_jinja_invalid_query_template_is_rejected_at_config_time(self) -> None:
        import pytest

        from elspeth.plugins.transforms.llm.multi_query import QueryDefinition

        with pytest.raises(ValueError, match=r"[Tt]emplate"):
            QueryDefinition(
                input_fields={"field_a": "field_a"},
                template="Broken {% if unclosed",
            )

    def test_valid_query_template_still_accepted(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QueryDefinition

        q = QueryDefinition(input_fields={"field_a": "field_a"}, template="Rate {{ field_a }}")
        assert q.template == "Rate {{ field_a }}"
