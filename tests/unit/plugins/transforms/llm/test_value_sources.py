"""Provider VALUE_SOURCES declarations and the model_catalog reader."""

from __future__ import annotations

import pytest

from elspeth.contracts.value_source import (
    CatalogValueSource,
    DerivedFromSiblingValueSource,
    UnknownCatalogIdError,
    get_catalog_values,
)
from elspeth.plugins.transforms.llm.azure_batch import (
    AzureBatchConfig,
    AzureBatchLLMTransform,
)
from elspeth.plugins.transforms.llm.model_catalog import (
    MODEL_CATALOG_OPENROUTER,
    read_litellm_model_list,
)

# Importing the batch transforms triggers their module-level
# ``register_value_source_plugin`` calls. Ordering matters: the import
# must occur before any walker-driven test that depends on the
# registration.
from elspeth.plugins.transforms.llm.openrouter_batch import (
    OpenRouterBatchConfig,
    OpenRouterBatchLLMTransform,
)
from elspeth.plugins.transforms.llm.providers.azure import AzureOpenAIConfig
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig


class TestProviderDeclarations:
    def test_openrouter_declares_catalog_source_for_model(self) -> None:
        decls = OpenRouterConfig.VALUE_SOURCES
        assert len(decls) == 1
        decl = decls[0]
        assert isinstance(decl, CatalogValueSource)
        assert decl.field_name == "model"
        assert decl.catalog_id == MODEL_CATALOG_OPENROUTER

    def test_azure_declares_derived_source_for_model(self) -> None:
        decls = AzureOpenAIConfig.VALUE_SOURCES
        assert len(decls) == 1
        decl = decls[0]
        assert isinstance(decl, DerivedFromSiblingValueSource)
        assert decl.field_name == "model"
        assert decl.sibling_field == "deployment_name"
        assert decl.allow_empty_default is True

    def test_value_sources_not_in_pydantic_model_fields(self) -> None:
        """ClassVar must NOT appear as a Pydantic field — would expose
        VALUE_SOURCES in serialized config and break the contract."""
        assert "VALUE_SOURCES" not in OpenRouterConfig.model_fields
        assert "VALUE_SOURCES" not in AzureOpenAIConfig.model_fields

    def test_openrouter_batch_declares_catalog_source_for_model(self) -> None:
        """Batch parity (pr-test-analyzer Observation X): the batch
        transform's config must enforce the same catalog membership as
        the non-batch transform; otherwise the "fail-fast on hallucinated
        model" guarantee leaks for any pipeline using ``openrouter_batch_llm``.
        """
        decls = OpenRouterBatchConfig.VALUE_SOURCES
        assert len(decls) == 1
        decl = decls[0]
        assert isinstance(decl, CatalogValueSource)
        assert decl.field_name == "model"
        assert decl.catalog_id == MODEL_CATALOG_OPENROUTER
        # ``applies_when`` predicate must match the non-batch declaration —
        # chaos-test pipelines that override base_url to a chaosllm /
        # errorworks endpoint must skip the catalog check identically.
        assert decl.applies_when == (("base_url", "https://openrouter.ai/api/v1"),)

    def test_openrouter_batch_value_sources_not_in_pydantic_model_fields(self) -> None:
        assert "VALUE_SOURCES" not in OpenRouterBatchConfig.model_fields

    def test_azure_batch_declares_derived_source_for_model(self) -> None:
        """Batch parity: ``AzureBatchConfig`` mirrors the non-batch
        :class:`AzureOpenAIConfig` declaration so operator ergonomics
        are identical (set ``model:`` in YAML and the walker enforces
        ``model == deployment_name``; omit it and the validator fills
        from ``deployment_name``).
        """
        decls = AzureBatchConfig.VALUE_SOURCES
        assert len(decls) == 1
        decl = decls[0]
        assert isinstance(decl, DerivedFromSiblingValueSource)
        assert decl.field_name == "model"
        assert decl.sibling_field == "deployment_name"
        assert decl.allow_empty_default is True

    def test_azure_batch_validator_fills_model_from_deployment_name(self) -> None:
        """The non-batch :meth:`AzureOpenAIConfig._set_model_from_deployment`
        validator pattern must be replicated on the batch config so an
        operator who omits ``model:`` in YAML still gets the canonical
        Azure deployment identifier in ``model`` post-validation.
        """
        # Use ``from_dict`` so the test goes through the same Pydantic
        # path the runtime takes — the ``schema:`` block needs the
        # config-base coercion to a ``SchemaConfig`` instance.
        cfg = AzureBatchConfig.from_dict(
            {
                "deployment_name": "my-deploy",
                "endpoint": "https://example.openai.azure.com",
                "api_key": "placeholder",
                "template": "hello",
                "schema": {"mode": "observed"},
            },
            plugin_name="azure_batch_llm",
        )
        assert cfg.model == "my-deploy"

    def test_azure_batch_value_sources_not_in_pydantic_model_fields(self) -> None:
        assert "VALUE_SOURCES" not in AzureBatchConfig.model_fields


class TestModelCatalogReader:
    def test_openrouter_catalog_registered_at_import_time(self) -> None:
        # Module import side effect: registration of the OpenRouter reader.
        # The exact catalog content depends on litellm version; we only assert
        # that the reader is registered (returns a frozenset, possibly empty).
        result = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        assert isinstance(result, frozenset)

    def test_unknown_catalog_id_raises(self) -> None:
        with pytest.raises(UnknownCatalogIdError):
            get_catalog_values("nonsense-catalog-id")

    def test_read_litellm_model_list_returns_tuple(self) -> None:
        result = read_litellm_model_list()
        assert isinstance(result, tuple)
        # Either non-empty (litellm installed and populated) or empty
        # (litellm absent / model_list missing). Both are valid.
        assert all(isinstance(m, str) for m in result)

    def test_openrouter_catalog_returns_unprefixed_slugs(self) -> None:
        """Catalog entries must round-trip back to ``litellm.model_list``
        when re-prefixed with the routing prefix — the invariant that
        makes validate-time and runtime agree.

        Note: some OpenRouter slugs legitimately *contain* the substring
        ``openrouter/`` (e.g. ``openrouter/auto``, OpenRouter's
        auto-routing endpoint). The catalog strips exactly one
        ``openrouter/`` prefix, so we don't assert "no entry starts with
        openrouter/" — instead we assert the round-trip:
        ``f"openrouter/{entry}"`` must exist in ``litellm.model_list``.
        That's the contract that prevents false negatives at validate
        time and the per-row HTTP-404 class of bug at runtime.
        """
        full = read_litellm_model_list()
        if not full:
            pytest.skip("litellm.model_list is empty; cannot verify prefix filtering")
        catalog = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        full_set = set(full)
        for entry in catalog:
            prefixed = f"openrouter/{entry}"
            assert prefixed in full_set, f"{prefixed!r} not in litellm.model_list — catalog out of sync"

    def test_openrouter_catalog_strips_one_prefix_level(self) -> None:
        """At least one canonical OpenRouter slug appears un-prefixed in
        the catalog. Proves the strip happened — guards against a
        regression where the prefix-stripping was disabled and entries
        carried the litellm routing prefix (the original P1 defect).

        We pick ``anthropic/claude-3.5-sonnet`` as the canary because it
        is the model identifier the original bug report cited as failing
        at HTTP-call time.
        """
        full = read_litellm_model_list()
        if not full:
            pytest.skip("litellm.model_list is empty")
        catalog = get_catalog_values(MODEL_CATALOG_OPENROUTER)
        canary = "anthropic/claude-3.5-sonnet"
        litellm_form = f"openrouter/{canary}"
        if litellm_form not in set(full):
            pytest.skip(f"{litellm_form!r} not in this litellm version's model_list")
        assert canary in catalog, (
            f"expected un-prefixed slug {canary!r} in catalog; if only {litellm_form!r} is present, the stripping regressed"
        )


class TestBatchTransformRegistration:
    """Pin that the batch transforms are wired into the L0 plugin
    opt-in registry. Without registration, the walker's
    ``find_value_source_config`` returns ``None`` for the batch
    transform — the original parity gap (pr-test-analyzer Observation X)
    where "fail-fast on hallucinated model" only held for the non-batch
    path.
    """

    def test_openrouter_batch_transform_exposes_provider_config(self) -> None:
        from elspeth.contracts.value_source import find_value_source_config

        plugin = OpenRouterBatchLLMTransform(
            {
                "api_key": "placeholder",
                "model": "openrouter/auto",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        config = find_value_source_config(plugin)
        # Same identity as ``provider_config`` — proves the registry
        # found the registration and resolved the public attribute.
        assert config is plugin.provider_config
        assert isinstance(config, OpenRouterBatchConfig)

    def test_azure_batch_transform_exposes_provider_config(self) -> None:
        from elspeth.contracts.value_source import find_value_source_config

        plugin = AzureBatchLLMTransform(
            {
                "deployment_name": "my-deploy",
                "endpoint": "https://example.openai.azure.com",
                "api_key": "placeholder",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        config = find_value_source_config(plugin)
        assert config is plugin.provider_config
        assert isinstance(config, AzureBatchConfig)
        # Validator filled ``model`` from ``deployment_name`` — the
        # walker can therefore confirm equality without per-row work.
        assert config.model == "my-deploy"


class TestBatchTransformWalkerBehaviour:
    """End-to-end through ``validate_value_source_compliance`` —
    proves the batch transforms now share the non-batch enforcement
    surface for hallucinated model identifiers.
    """

    def test_openrouter_llm_rejects_hallucinated_model_with_trailing_slash_base_url(self) -> None:
        """Canonical OpenRouter URL with a trailing slash still uses the catalog.

        Runtime HTTP joins strip the trailing slash before posting to
        OpenRouter. The value-source predicate must treat that URL as the
        same endpoint so a hallucinated model fails before the HTTP call.
        """
        from unittest.mock import MagicMock, patch

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance
        from elspeth.engine.orchestrator.types import ValueSourceValidationError
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        plugin = LLMTransform(
            {
                "provider": "openrouter",
                "api_key": "placeholder",
                "model": "anthropic/claude-3.5-sonnet",
                "base_url": "https://openrouter.ai/api/v1/",
                "template": "Hello",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "openrouter_node_1"
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openai/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])

        finding = exc_info.value.findings[0]
        assert finding.component_id == "openrouter_node_1"
        assert finding.field_name == "model"
        assert "anthropic/claude-3.5-sonnet" in finding.reason

    def test_openrouter_batch_rejects_hallucinated_model_at_canonical_endpoint(self) -> None:
        from unittest.mock import MagicMock, patch

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance
        from elspeth.engine.orchestrator.types import ValueSourceValidationError

        plugin = OpenRouterBatchLLMTransform(
            {
                "api_key": "placeholder",
                "model": "anthropic/claude-3.5-sonnet",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "openrouter_batch_node_1"
        # Catalog patched to a known set that does NOT include the
        # hallucinated identifier — proves the walker rejects.
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openai/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])
        finding = exc_info.value.findings[0]
        assert finding.component_id == "openrouter_batch_node_1"
        assert finding.field_name == "model"
        assert "anthropic/claude-3.5-sonnet" in finding.reason

    def test_openrouter_batch_rejects_hallucinated_model_with_trailing_slash_base_url(self) -> None:
        """Batch OpenRouter must mirror the non-batch canonical URL predicate."""
        from unittest.mock import MagicMock, patch

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance
        from elspeth.engine.orchestrator.types import ValueSourceValidationError

        plugin = OpenRouterBatchLLMTransform(
            {
                "api_key": "placeholder",
                "model": "anthropic/claude-3.5-sonnet",
                "base_url": "https://openrouter.ai/api/v1/",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "openrouter_batch_node_1"
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.get_catalog_values",
                return_value=frozenset({"openai/gpt-4o"}),
            ),
            pytest.raises(ValueSourceValidationError) as exc_info,
        ):
            validate_value_source_compliance([wired])

        finding = exc_info.value.findings[0]
        assert finding.component_id == "openrouter_batch_node_1"
        assert finding.field_name == "model"
        assert "anthropic/claude-3.5-sonnet" in finding.reason

    def test_openrouter_batch_skips_catalog_when_base_url_overridden(self) -> None:
        """``applies_when`` predicate parity: chaos-test pipelines pointing
        at errorworks/chaosllm endpoints with synthetic model identifiers
        (e.g. ``chaosllm/fake-gpt-4``) must be accepted by the walker —
        the canonical OpenRouter catalog isn't authoritative for that
        endpoint. This is the fix path for elspeth-ea207837d9 carried
        forward to the batch sibling.
        """
        from unittest.mock import MagicMock, patch

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance

        plugin = OpenRouterBatchLLMTransform(
            {
                "api_key": "placeholder",
                "model": "chaosllm/fake-gpt-4",
                "base_url": "http://127.0.0.1:8199/v1",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "chaos_batch_node_1"
        with patch(
            "elspeth.engine.orchestrator.preflight.get_catalog_values",
            return_value=frozenset({"openai/gpt-4o"}),
        ):
            validate_value_source_compliance([wired])  # passes (predicate skipped)

    def test_azure_batch_rejects_divergent_model_override(self) -> None:
        from unittest.mock import MagicMock

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance
        from elspeth.engine.orchestrator.types import ValueSourceValidationError

        plugin = AzureBatchLLMTransform(
            {
                "deployment_name": "right-deploy",
                "model": "wrong-deploy",  # diverges from deployment_name
                "endpoint": "https://example.openai.azure.com",
                "api_key": "placeholder",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "azure_batch_node_1"
        with pytest.raises(ValueSourceValidationError) as exc_info:
            validate_value_source_compliance([wired])
        finding = exc_info.value.findings[0]
        assert finding.component_id == "azure_batch_node_1"
        assert finding.field_name == "model"
        assert "wrong-deploy" in finding.reason
        assert "right-deploy" in finding.reason

    def test_azure_batch_passes_when_model_omitted(self) -> None:
        """Operator omits ``model:`` in YAML; validator fills from
        ``deployment_name``; walker confirms equality. End-to-end happy
        path matching the non-batch ergonomics.
        """
        from unittest.mock import MagicMock

        from elspeth.engine.orchestrator.preflight import validate_value_source_compliance

        plugin = AzureBatchLLMTransform(
            {
                "deployment_name": "my-deploy",
                "endpoint": "https://example.openai.azure.com",
                "api_key": "placeholder",
                "template": "Hello",
                "schema": {"mode": "observed"},
            }
        )
        wired = MagicMock()
        wired.plugin = plugin
        wired.settings = MagicMock()
        wired.settings.name = "azure_batch_node_2"
        validate_value_source_compliance([wired])  # passes
