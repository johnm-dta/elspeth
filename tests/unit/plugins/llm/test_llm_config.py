# tests/unit/plugins/llm/test_llm_config.py
"""Tests for unified LLM config models (Task 8).

Tests the new provider-dispatched LLMConfig, domain-agnostic QuerySpec,
resolve_queries() normalization, and provider-specific config classes.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from pydantic import ValidationError

from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.transforms.llm.base import LLMConfig

# Shared observed schema for test convenience
_OBSERVED_SCHEMA = SchemaConfig(mode="observed", fields=None)

# A valid OpenRouter catalog model id (the retired anthropic/claude-3-opus was
# dropped from the litellm-derived catalog; OpenRouterConfig now rejects models
# absent from it). Mirrors test_openrouter.py.
_OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"


# ---------------------------------------------------------------------------
# LLMConfig base changes
# ---------------------------------------------------------------------------


class TestLLMConfigBase:
    """Tests for LLMConfig base class changes."""

    def test_model_optional_defaults_to_none(self) -> None:
        """model field is optional and defaults to None."""
        config = LLMConfig(
            provider="azure",
            prompt_template="Classify: {{ row.text }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["text"],
        )
        assert config.model is None

    def test_model_accepts_explicit_value(self) -> None:
        config = LLMConfig(
            provider="azure",
            model="gpt-4o",
            prompt_template="Classify: {{ row.text }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["text"],
        )
        assert config.model == "gpt-4o"

    def test_provider_field_required(self) -> None:
        """provider field is required — Literal["azure", "openrouter"]."""
        with pytest.raises(ValidationError):
            LLMConfig(
                provider="invalid_provider",
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
            )

    def test_provider_azure_accepted(self) -> None:
        config = LLMConfig(
            provider="azure",
            prompt_template="hello {{ row.text }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["text"],
        )
        assert config.provider == "azure"

    def test_provider_openrouter_accepted(self) -> None:
        config = LLMConfig(
            provider="openrouter",
            prompt_template="hello {{ row.text }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["text"],
        )
        assert config.provider == "openrouter"

    def test_queries_field_none_by_default(self) -> None:
        """queries is None when not provided (single-query mode)."""
        config = LLMConfig(
            provider="azure",
            prompt_template="hello {{ row.text }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["text"],
        )
        assert config.queries is None

    def test_resolved_prompt_template_hash_must_match_prompt_template(self) -> None:
        """Phase 5b runtime anchor refuses prompt/hash drift at config load."""
        resolved_template = "Rate how innovative this is."
        config = LLMConfig(
            provider="azure",
            prompt_template=resolved_template,
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
            resolved_prompt_template_hash=stable_hash(resolved_template),
        )
        assert config.resolved_prompt_template_hash == stable_hash(resolved_template)

        with pytest.raises(ValidationError, match="resolved_prompt_template_hash"):
            LLMConfig(
                provider="azure",
                prompt_template="Rate how boring this is.",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                resolved_prompt_template_hash=stable_hash(resolved_template),
            )

    def test_missing_required_input_fields_error_names_composer_options_repair(self) -> None:
        """Runtime preflight errors must name the composer patch location."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="URL: {{ row['url'] }}\nContent: {{ row['content'] }}",
                schema_config=_OBSERVED_SCHEMA,
            )

        message = str(exc_info.value)
        assert "options.required_input_fields" in message
        assert "patch_node_options" in message
        assert '"patch": {"required_input_fields": ["content", "url"]}' in message

    def test_dynamic_row_item_access_requires_explicit_opt_out(self) -> None:
        """Dynamic row[expr] access must not look like a no-row-field prompt."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template='{% set k = "ssn" %}Secret: {{ row[k] }}',
                schema_config=_OBSERVED_SCHEMA,
            )

        message = str(exc_info.value)
        assert "dynamic row field access" in message
        assert "row[expr]" in message
        assert "options.required_input_fields: []" in message

    def test_dynamic_row_get_access_requires_explicit_opt_out(self) -> None:
        """Dynamic row.get(expr) access must fail closed by default."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="Secret: {{ row.get(k) }}",
                schema_config=_OBSERVED_SCHEMA,
            )

        message = str(exc_info.value)
        assert "dynamic row field access" in message
        assert "row.get(expr)" in message

    def test_dynamic_row_attr_filter_requires_explicit_opt_out(self) -> None:
        """Dynamic row|attr(expr) access must fail closed by default."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="{{ row | attr(row.selector) }}",
                schema_config=_OBSERVED_SCHEMA,
            )

        message = str(exc_info.value)
        assert "dynamic row field access" in message
        assert "row|attr(expr)" in message

    def test_dynamic_row_map_attribute_filter_requires_explicit_opt_out(self) -> None:
        """Dynamic row|map(attribute=expr) access must fail closed by default."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="{{ row | map(attribute=field_name) | list }}",
                schema_config=_OBSERVED_SCHEMA,
            )

        message = str(exc_info.value)
        assert "dynamic row field access" in message
        assert "map(attribute=expr)" in message

    def test_dynamic_row_access_rejected_even_with_declared_fields(self) -> None:
        """A declared field list is not a dynamic-key allowlist."""
        with pytest.raises(ValidationError, match="dynamic row field access"):
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template='{% set k = "ssn" %}Secret: {{ row[k] }}',
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=["ssn"],
            )

    def test_dynamic_row_access_accepts_empty_required_fields_opt_out(self) -> None:
        """The documented empty-list opt-out remains explicit and accepted."""
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template='{% set k = "ssn" %}Secret: {{ row[k] }}',
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
        )

        assert config.required_input_fields == []


class TestRequiredInputFieldsAppearInTemplate:
    """Dual of `_validate_required_input_fields_declared`: catches the inverse
    asymmetry where `required_input_fields` is declared but the prompt template
    interpolates zero `row.*` fields, so every row is sent the same static prompt.
    """

    def test_declared_fields_without_row_interpolation_rejected(self) -> None:
        """A non-empty required_input_fields with a static prompt body is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="Identify primary colours used on the page. Return JSON.",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=["url", "content"],
            )

        message = str(exc_info.value)
        assert "does not interpolate any row.* fields" in message
        assert "['content', 'url']" in message
        assert "{{ row.url }}" in message
        assert "{{ row.content }}" in message

    def test_declared_fields_with_matching_row_interpolation_accepted(self) -> None:
        """Canonical case: every declared field appears as a row.* reference."""
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template="URL: {{ row.url }}\nContent: {{ row.content }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["url", "content"],
        )
        assert sorted(config.required_input_fields or []) == ["content", "url"]

    def test_declared_fields_with_partial_row_interpolation_accepted(self) -> None:
        """Validator fires only on the empty-row-refs case, not on partial overlap.

        Partial mismatch (declared fields not all interpolated, or extra row refs
        not declared) is a softer signal; rejecting it would break legitimate
        cases like "declared a field for downstream cleanup but not used in this
        specific prompt body." The reciprocity guidance lives in the skill prompt.
        """
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template="URL: {{ row.url }}",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["url", "content"],
        )
        assert config.prompt_template == "URL: {{ row.url }}"

    def test_explicit_opt_out_empty_list_accepted_with_static_prompt(self) -> None:
        """`required_input_fields: []` is the documented opt-out and must pass."""
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template="Return a fixed JSON greeting.",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
        )
        assert config.required_input_fields == []

    def test_required_input_fields_none_does_not_fire_this_check(self) -> None:
        """When `required_input_fields` is undeclared, the dual validator must not fire.

        That case is owned by `_validate_required_input_fields_declared`. With a
        static prompt that references no row.* fields, neither validator fires,
        and the config is accepted (audit-philosophy opt-out by omission).
        """
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template="Return a fixed JSON greeting.",
            schema_config=_OBSERVED_SCHEMA,
        )
        assert config.required_input_fields is None

    def test_multi_query_mode_is_out_of_scope_for_this_check(self) -> None:
        """Multi-query mode flows row data via per-query input_fields mappings.

        A top-level template without `row.*` references is therefore not by itself
        diagnostic in multi-query mode; the dual validator restricts itself to
        single-query mode where the absence is unambiguous.
        """
        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-sonnet-4.6",
            prompt_template="Assess each case.",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=["case_text"],
            queries={
                "diagnosis": {
                    "input_fields": {"input_1": "case_text"},
                    "template": "Diagnose: {{ row.input_1 }}",
                }
            },
        )
        assert config.queries is not None

    def test_error_message_names_composer_repair_path(self) -> None:
        """The error must point the composer at patch_node_options to repair."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                provider="openrouter",
                model="anthropic/claude-sonnet-4.6",
                prompt_template="Static prompt without interpolation.",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=["page_body"],
            )

        message = str(exc_info.value)
        assert "patch_node_options" in message
        assert '"prompt_template"' in message
        assert "{{ row.page_body }}" in message


class TestLLMConfigResponseFieldValidation:
    """Verify LLMConfig rejects invalid response_field names.

    Bug: elspeth-23d1bcff6b. LLMConfig accepts invalid response_field names
    even though downstream schema builders require a non-empty Python identifier.
    Bad config survives model validation and only explodes later.
    """

    def test_empty_response_field_rejected(self) -> None:
        """Empty string response_field is rejected."""
        with pytest.raises(ValidationError, match="response_field"):
            LLMConfig(
                provider="azure",
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                response_field="",
            )

    def test_whitespace_response_field_rejected(self) -> None:
        """Whitespace-only response_field is rejected."""
        with pytest.raises(ValidationError, match="response_field"):
            LLMConfig(
                provider="azure",
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                response_field="   ",
            )

    def test_non_identifier_response_field_rejected(self) -> None:
        """Non-Python-identifier response_field is rejected (e.g., 'my-field')."""
        with pytest.raises(ValidationError, match="response_field"):
            LLMConfig(
                provider="azure",
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                response_field="my-field",
            )

    def test_valid_identifier_response_field_accepted(self) -> None:
        """Valid Python identifier response_field is accepted."""
        config = LLMConfig(
            provider="azure",
            prompt_template="hello",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
            response_field="llm_output",
        )
        assert config.response_field == "llm_output"


# ---------------------------------------------------------------------------
# Provider-specific configs
# ---------------------------------------------------------------------------


class TestAzureOpenAIConfig:
    """Tests for Azure-specific config class."""

    def test_requires_deployment_name(self) -> None:
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        with pytest.raises((ValidationError, ValueError)):
            AzureOpenAIConfig(  # type: ignore[call-arg]  # intentionally missing required args
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                # Missing deployment_name, endpoint, api_key
            )

    def test_model_defaults_to_deployment_name(self) -> None:
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        config = AzureOpenAIConfig(
            deployment_name="gpt-4o-deploy",
            endpoint="https://test.openai.azure.com/",
            api_key="key",
            prompt_template="hello",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
        )
        # Azure sets model = deployment_name when model is empty/None
        assert config.model == "gpt-4o-deploy"

    def test_tracing_field_on_azure(self) -> None:
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        config = AzureOpenAIConfig(
            deployment_name="gpt-4o",
            endpoint="https://test.openai.azure.com/",
            api_key="key",
            prompt_template="hello",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
            tracing={"provider": "langfuse", "public_key": "pk"},
        )
        assert config.tracing is not None


class TestAzureOpenAIConfigTracing:
    """Tests for tracing configuration in AzureOpenAIConfig (from_dict path)."""

    def _make_azure_base_config(self) -> dict[str, Any]:
        """Create base config with all required fields for Azure."""
        return {
            "provider": "azure",
            "deployment_name": "gpt-4",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "prompt_template": "Hello {{ row.name }}",
            "schema": {"mode": "observed"},
            "required_input_fields": [],
        }

    def test_tracing_field_accepts_none(self) -> None:
        """Tracing field defaults to None (no tracing)."""
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        config = AzureOpenAIConfig.from_dict(self._make_azure_base_config())
        assert config.tracing is None

    def test_tracing_field_accepts_azure_ai_config(self) -> None:
        """Tracing field accepts Azure AI configuration dict."""
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        cfg = self._make_azure_base_config()
        cfg["tracing"] = {
            "provider": "azure_ai",
            "connection_string": "InstrumentationKey=xxx",
            "enable_content_recording": True,
        }
        config = AzureOpenAIConfig.from_dict(cfg)
        assert config.tracing is not None
        assert config.tracing["provider"] == "azure_ai"

    def test_tracing_field_accepts_langfuse_config(self) -> None:
        """Tracing field accepts Langfuse configuration dict."""
        from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig

        cfg = self._make_azure_base_config()
        cfg["tracing"] = {
            "provider": "langfuse",
            "public_key": "pk-xxx",
            "secret_key": "sk-xxx",
        }
        config = AzureOpenAIConfig.from_dict(cfg)
        assert config.tracing is not None
        assert config.tracing["provider"] == "langfuse"


class TestOpenRouterConfigTracing:
    """Tests for tracing configuration in OpenRouterConfig (from_dict path)."""

    def _make_openrouter_base_config(self) -> dict[str, Any]:
        """Create base config with all required fields for OpenRouter."""
        return {
            "provider": "openrouter",
            "model": _OPENROUTER_MODEL,
            "api_key": "test-key",
            "prompt_template": "Hello {{ row.name }}",
            "schema": {"mode": "observed"},
            "required_input_fields": [],
        }

    def test_tracing_field_accepts_none(self) -> None:
        """Tracing field defaults to None (no tracing)."""
        from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig

        config = OpenRouterConfig.from_dict(self._make_openrouter_base_config())
        assert config.tracing is None

    def test_tracing_field_accepts_langfuse_config(self) -> None:
        """Tracing field accepts Langfuse configuration dict."""
        from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig

        cfg = self._make_openrouter_base_config()
        cfg["tracing"] = {
            "provider": "langfuse",
            "public_key": "pk-xxx",
            "secret_key": "sk-xxx",
        }
        config = OpenRouterConfig.from_dict(cfg)
        assert config.tracing is not None
        assert config.tracing["provider"] == "langfuse"


class TestOpenRouterConfig:
    """Tests for OpenRouter-specific config class."""

    def test_requires_model(self) -> None:
        """OpenRouter requires model to be non-None."""
        from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig

        # model=None should fail validation
        with pytest.raises((ValidationError, ValueError)):
            OpenRouterConfig(  # type: ignore[call-arg]  # intentionally missing model
                api_key="key",
                prompt_template="hello",
                schema_config=_OBSERVED_SCHEMA,
                required_input_fields=[],
                # model not provided — should fail because OpenRouter needs it
            )

    def test_accepts_explicit_model(self) -> None:
        from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig

        config = OpenRouterConfig(
            model="openai/gpt-4o",
            api_key="key",
            prompt_template="hello",
            schema_config=_OBSERVED_SCHEMA,
            required_input_fields=[],
        )
        assert config.model == "openai/gpt-4o"


# ---------------------------------------------------------------------------
# Domain-agnostic QuerySpec
# ---------------------------------------------------------------------------


class TestQuerySpec:
    """Tests for the new domain-agnostic QuerySpec."""

    def test_post_init_rejects_empty_name(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        with pytest.raises(ValueError, match="name must be non-empty"):
            QuerySpec(name="", input_fields=MappingProxyType({"text": "text"}))

    def test_post_init_rejects_empty_input_fields(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        with pytest.raises(ValueError, match="input_fields must be non-empty"):
            QuerySpec(name="q1", input_fields=MappingProxyType({}))

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        spec = QuerySpec(name="q1", input_fields=MappingProxyType({"text": "text_col"}))
        with pytest.raises(FrozenInstanceError):
            spec.name = "modified"  # type: ignore[misc]

    def test_defaults(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec, ResponseFormat

        spec = QuerySpec(name="q1", input_fields=MappingProxyType({"text": "text_col"}))
        assert spec.response_format == ResponseFormat.STANDARD
        assert spec.output_fields is None
        assert spec.template is None
        assert spec.max_tokens is None

    def test_build_template_context_named_variables(self) -> None:
        """Named input_fields map to template variables directly."""
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        spec = QuerySpec(
            name="q1",
            input_fields=MappingProxyType({"text_content": "text", "category_name": "category"}),
        )
        row = {"text": "hello world", "category": "science", "extra": "ignored"}
        ctx = spec.build_template_context(row)

        assert ctx["text_content"] == "hello world"
        assert ctx["category_name"] == "science"
        assert ctx["source_row"] is row

    def test_build_template_context_missing_field_raises(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        spec = QuerySpec(
            name="q1",
            input_fields=MappingProxyType({"text_content": "text"}),
        )
        with pytest.raises(KeyError, match="text"):
            spec.build_template_context({"other": "value"})

    def test_input_fields_is_deeply_immutable(self) -> None:
        """input_fields dict must be truly immutable — shared across rows."""
        from types import MappingProxyType

        from elspeth.plugins.transforms.llm.multi_query import QuerySpec

        original = {"text": "text_col", "cat": "category_col"}
        spec = QuerySpec(name="q1", input_fields=MappingProxyType(original))

        assert isinstance(spec.input_fields, MappingProxyType)
        with pytest.raises(TypeError):
            spec.input_fields["injected"] = "evil"  # type: ignore[index]

        # Caller's original dict must be decoupled
        original["injected"] = "evil"
        assert "injected" not in spec.input_fields

    def test_output_fields_is_tuple(self) -> None:
        """output_fields list must be stored as tuple when provided."""
        from elspeth.plugins.transforms.llm.multi_query import OutputFieldConfig, OutputFieldType, QuerySpec

        fields = [OutputFieldConfig(suffix="label", type=OutputFieldType.STRING)]
        spec = QuerySpec(name="q1", input_fields=MappingProxyType({"text": "col"}), output_fields=tuple(fields))

        assert isinstance(spec.output_fields, tuple)
        # Caller's original list must be decoupled
        fields.append(OutputFieldConfig(suffix="extra", type=OutputFieldType.STRING))
        assert len(spec.output_fields) == 1


# ---------------------------------------------------------------------------
# resolve_queries()
# ---------------------------------------------------------------------------


class TestResolveQueries:
    """Tests for resolve_queries() normalization."""

    def test_empty_list_raises(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="no queries configured"):
            resolve_queries([])

    def test_empty_dict_raises(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="no queries configured"):
            resolve_queries({})

    def test_dict_to_list_normalization(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        result = resolve_queries(
            {
                "q1": {
                    "input_fields": {"text": "text_col"},
                },
                "q2": {
                    "input_fields": {"category": "cat_col"},
                },
            }
        )
        assert len(result) == 2
        names = {q.name for q in result}
        assert names == {"q1", "q2"}

    def test_list_normalization(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import QuerySpec, resolve_queries

        specs = [
            QuerySpec(name="q1", input_fields=MappingProxyType({"text": "text_col"})),
        ]
        result = resolve_queries(specs)
        assert len(result) == 1
        assert result[0].name == "q1"

    def test_key_collision_raises(self) -> None:
        """Two queries whose name+suffix combination produces the same full output key.

        Query "q1_extra" with suffix "score" -> key "q1_extra_score"
        Query "q1" with suffix "extra_score" -> key "q1_extra_score"

        Both produce the identical full output key, so resolve_queries must raise.
        """
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="collision"):
            resolve_queries(
                {
                    "q1_extra": {
                        "input_fields": {"text": "text_col"},
                        "output_fields": [{"suffix": "score", "type": "integer"}],
                    },
                    "q1": {
                        "input_fields": {"text": "text_col"},
                        "output_fields": [{"suffix": "extra_score", "type": "integer"}],
                    },
                }
            )

    def test_reserved_suffix_raises_error(self) -> None:
        """Output field with reserved _error suffix raises ValueError."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="reserved LLM suffix"):
            resolve_queries(
                {
                    "q1": {
                        "input_fields": {"text": "text_col"},
                        "output_fields": [{"suffix": "error", "type": "string"}],
                    },
                }
            )

    def test_reserved_suffix_from_constants_raises_error(self) -> None:
        """Output field with suffix derived from LLM_GUARANTEED_SUFFIXES (e.g., 'usage') raises ValueError."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="reserved LLM suffix"):
            resolve_queries(
                {
                    "q1": {
                        "input_fields": {"text": "text_col"},
                        "output_fields": [{"suffix": "usage", "type": "string"}],
                    },
                }
            )

    def test_single_query_returns_one_element_list(self) -> None:
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        result = resolve_queries(
            {
                "only_one": {"input_fields": {"text": "text_col"}},
            }
        )
        assert len(result) == 1

    def test_rejects_positional_template_variables(self) -> None:
        """Templates with {{ input_1 }} pattern raise with migration guidance."""
        from elspeth.plugins.transforms.llm.multi_query import resolve_queries

        with pytest.raises(ValueError, match="positional variables"):
            resolve_queries(
                {
                    "q1": {
                        "input_fields": {"text": "text_col"},
                        "template": "Evaluate {{ input_1 }} quality",
                    },
                }
            )


class TestMultiQueryInputFieldsValidation:
    """Regression: _validate_required_input_fields_declared must check multi-query input_fields."""

    def test_multi_query_dict_form_requires_declaration(self) -> None:
        """Multi-query with input_fields must require required_input_fields declaration."""
        with pytest.raises(ValidationError, match="required_input_fields"):
            LLMConfig(
                provider="openrouter",
                model="test-model",
                prompt_template="Static template",
                schema_config=_OBSERVED_SCHEMA,
                queries={
                    "q1": {
                        "input_fields": {"text": "customer_text"},
                    },
                },
            )

    def test_multi_query_list_form_requires_declaration(self) -> None:
        """Multi-query list form also triggers required_input_fields check."""
        with pytest.raises(ValidationError, match="required_input_fields"):
            LLMConfig(
                provider="openrouter",
                model="test-model",
                prompt_template="Static template",
                schema_config=_OBSERVED_SCHEMA,
                queries=[
                    {
                        "name": "q1",
                        "input_fields": {"text": "customer_text"},
                    },
                ],
            )

    def test_multi_query_with_explicit_fields_passes(self) -> None:
        """Multi-query with required_input_fields declared passes validation."""
        config = LLMConfig(
            provider="openrouter",
            model="test-model",
            prompt_template="Static template",
            schema_config=_OBSERVED_SCHEMA,
            queries={
                "q1": {
                    "input_fields": {"text": "customer_text"},
                },
            },
            required_input_fields=["customer_text"],
        )
        assert config.required_input_fields == ["customer_text"]

    def test_multi_query_opt_out_passes(self) -> None:
        """Multi-query with empty required_input_fields (opt-out) passes."""
        config = LLMConfig(
            provider="openrouter",
            model="test-model",
            prompt_template="Static template",
            schema_config=_OBSERVED_SCHEMA,
            queries={
                "q1": {
                    "input_fields": {"text": "customer_text"},
                },
            },
            required_input_fields=[],
        )
        assert config.required_input_fields == []
