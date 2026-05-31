"""Tests for CatalogServiceImpl with real PluginManager."""

from __future__ import annotations

import pytest

from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.catalog.service import CatalogServiceImpl


@pytest.fixture(scope="module")
def plugin_manager() -> PluginManager:
    """Shared PluginManager with builtins registered."""
    pm = PluginManager()
    pm.register_builtin_plugins()
    return pm


@pytest.fixture(scope="module")
def catalog(plugin_manager: PluginManager) -> CatalogServiceImpl:
    return CatalogServiceImpl(plugin_manager)


class TestCatalogService:
    """CatalogServiceImpl satisfies the CatalogService protocol."""

    def test_implements_protocol(self, catalog: CatalogServiceImpl) -> None:
        assert isinstance(catalog, CatalogService)


class TestListSources:
    """list_sources() returns all registered source plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        assert len(sources) > 0

    def test_csv_source_present(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        names = [s.name for s in sources]
        assert "csv" in names

    def test_text_source_present(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        names = [s.name for s in sources]
        assert "text" in names

    def test_all_entries_are_plugin_summaries(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        for s in sources:
            assert isinstance(s, PluginSummary)
            assert s.plugin_type == "source"
            assert s.name
            assert s.description

    def test_config_fields_populated_for_csv(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        csv_source = next(s for s in sources if s.name == "csv")
        field_names = [f.name for f in csv_source.config_fields]
        assert "path" in field_names

    def test_config_field_has_type_and_required(self, catalog: CatalogServiceImpl) -> None:
        sources = catalog.list_sources()
        csv_source = next(s for s in sources if s.name == "csv")
        for field in csv_source.config_fields:
            assert field.type
            assert isinstance(field.required, bool)

    def test_matches_plugin_manager_count(self, catalog: CatalogServiceImpl, plugin_manager: PluginManager) -> None:
        sources = catalog.list_sources()
        assert len(sources) == len(plugin_manager.get_sources())


class TestListTransforms:
    """list_transforms() returns all registered transform plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        assert len(transforms) > 0

    def test_passthrough_present(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        names = [t.name for t in transforms]
        assert "passthrough" in names

    def test_all_entries_have_transform_type(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        assert len(transforms) > 0, "catalog returned no transforms; type assertion would be vacuously true"
        for t in transforms:
            assert t.plugin_type == "transform"

    def test_matches_plugin_manager_count(self, catalog: CatalogServiceImpl, plugin_manager: PluginManager) -> None:
        transforms = catalog.list_transforms()
        assert len(transforms) == len(plugin_manager.get_transforms())

    def test_no_gates_in_transforms(self, catalog: CatalogServiceImpl) -> None:
        """Gates are config-driven system operations, not plugins (AC6)."""
        transforms = catalog.list_transforms()
        names = {t.name for t in transforms}
        # Gates are not registered in PluginManager.get_transforms(), so they
        # should never appear here. Known gate names for a sanity check:
        gate_names = {"threshold_gate", "routing_gate", "classification_gate"}
        assert names.isdisjoint(gate_names), f"Gates found in transforms: {names & gate_names}"

    def test_batch_aware_transform_summary_hides_required_input_fields(self, catalog: CatalogServiceImpl) -> None:
        """ADR-013: batch-aware transforms cannot advertise declared input fields."""
        transforms = catalog.list_transforms()
        batch_stats = next(t for t in transforms if t.name == "batch_stats")

        field_names = {field.name for field in batch_stats.config_fields}

        assert "required_input_fields" not in field_names


class TestListSinks:
    """list_sinks() returns all registered sink plugins."""

    def test_returns_non_empty_list(self, catalog: CatalogServiceImpl) -> None:
        sinks = catalog.list_sinks()
        assert len(sinks) > 0

    def test_csv_sink_present(self, catalog: CatalogServiceImpl) -> None:
        sinks = catalog.list_sinks()
        names = [s.name for s in sinks]
        assert "csv" in names

    def test_all_entries_have_sink_type(self, catalog: CatalogServiceImpl) -> None:
        sinks = catalog.list_sinks()
        for s in sinks:
            assert s.plugin_type == "sink"

    def test_matches_plugin_manager_count(self, catalog: CatalogServiceImpl, plugin_manager: PluginManager) -> None:
        sinks = catalog.list_sinks()
        assert len(sinks) == len(plugin_manager.get_sinks())


class TestGetSchema:
    """get_schema() returns full JSON schema for a plugin's config."""

    def test_csv_source_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("source", "csv")
        assert isinstance(info, PluginSchemaInfo)
        assert info.name == "csv"
        assert info.plugin_type == "source"
        assert info.description
        assert isinstance(info.json_schema, dict)
        assert "properties" in info.json_schema
        assert info.json_schema["type"] == "object"

    def test_csv_source_schema_matches_model_json_schema(self, catalog: CatalogServiceImpl) -> None:
        """AC2: json_schema output matches model_json_schema() directly."""
        from elspeth.plugins.sources.csv_source import CSVSourceConfig

        info = catalog.get_schema("source", "csv")
        assert info.json_schema == CSVSourceConfig.model_json_schema()

    def test_text_source_schema_matches_model_json_schema(self, catalog: CatalogServiceImpl) -> None:
        from elspeth.plugins.sources.text_source import TextSourceConfig

        info = catalog.get_schema("source", "text")
        assert info.json_schema == TextSourceConfig.model_json_schema()

    def test_passthrough_transform_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("transform", "passthrough")
        assert info.name == "passthrough"
        assert info.plugin_type == "transform"
        assert isinstance(info.json_schema, dict)

    def test_batch_aware_transform_schema_hides_required_input_fields(self, catalog: CatalogServiceImpl) -> None:
        """Full schema must not offer fields the batch-aware runtime rejects."""
        info = catalog.get_schema("transform", "batch_stats")

        assert "required_input_fields" not in info.json_schema["properties"]

    def test_batch_stats_group_by_schema_describes_per_group_rollups(self, catalog: CatalogServiceImpl) -> None:
        """Composer-visible schema must describe group_by as a rollup partition."""
        info = catalog.get_schema("transform", "batch_stats")

        description = info.json_schema["properties"]["group_by"]["description"]
        assert "one aggregate row per distinct value" in description

    def test_csv_sink_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("sink", "csv")
        assert info.name == "csv"
        assert info.plugin_type == "sink"
        assert isinstance(info.json_schema, dict)

    def test_null_source_returns_empty_schema(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("source", "null")
        assert info.name == "null"
        assert info.json_schema == {}

    def test_llm_transform_emits_discriminated_schema(self, catalog: CatalogServiceImpl) -> None:
        """Regression: bug elspeth-dcf12c061b.

        The LLM transform dispatches config on ``provider`` at runtime. The
        catalog must publish a Pydantic discriminated union with $defs per
        provider — not just the thin base LLMConfig whose required set is
        missing the Azure (deployment_name/endpoint/api_key) and OpenRouter
        (model/api_key) mandatory fields.

        Skipped when the ``[llm]`` extra is not installed: ``PluginManager``
        discovery silently omits plugins whose module-level imports fail
        (see ``discover_plugins_in_directory``), so the ``llm`` transform
        will not be registered on a bare dev install. The test speaks to the
        discriminated-union contract, not to plugin availability, so gating
        on the extra is the correct scoping.
        """
        pytest.importorskip(
            "litellm",
            reason="LLM transform requires the [llm] extra; catalog discovery skips it otherwise.",
        )
        info = catalog.get_schema("transform", "llm")
        schema = info.json_schema
        assert "oneOf" in schema
        assert len(schema["oneOf"]) == 2
        assert schema["discriminator"]["propertyName"] == "provider"
        assert set(schema["discriminator"]["mapping"].keys()) == {"azure", "openrouter"}
        defs = schema["$defs"]
        assert "AzureOpenAIConfig" in defs
        assert "OpenRouterConfig" in defs
        assert set(defs["AzureOpenAIConfig"]["required"]) >= {
            "deployment_name",
            "endpoint",
            "api_key",
            "prompt_template",
        }
        assert set(defs["OpenRouterConfig"]["required"]) >= {"model", "api_key", "prompt_template"}

    def test_llm_transform_summary_includes_provider_fields(self, catalog: CatalogServiceImpl) -> None:
        """Regression: bug elspeth-dcf12c061b.

        list_transforms()[llm].config_fields must surface provider-specific
        fields (deployment_name, endpoint, api_key, base_url, timeout_seconds,
        model) — not just the base LLMConfig fields. Required-in-all-variants
        is the honest summary rule: api_key and template appear in every
        provider's required set; provider-specific required fields are marked
        required=False because they are conditional on the discriminator
        value, and the full schema encodes that conditionality.

        Gated on the ``[llm]`` extra for the same reason as the schema test
        above — without litellm installed, the LLM transform module fails
        to import and PluginManager skips it.
        """
        pytest.importorskip(
            "litellm",
            reason="LLM transform requires the [llm] extra; catalog discovery skips it otherwise.",
        )
        transforms = catalog.list_transforms()
        llm = next(t for t in transforms if t.name == "llm")
        field_names = {f.name for f in llm.config_fields}
        assert field_names >= {
            "provider",
            "prompt_template",
            "api_key",
            "deployment_name",
            "endpoint",
            "model",
            "base_url",
            "timeout_seconds",
        }
        required = {f.name for f in llm.config_fields if f.required}
        # Fields that are required in EVERY provider variant — honest intersection.
        assert "api_key" in required
        assert "prompt_template" in required
        # Fields required only for some providers must not claim universal requiredness.
        assert "deployment_name" not in required
        assert "endpoint" not in required
        assert "model" not in required

    def test_unknown_type_raises_value_error(self, catalog: CatalogServiceImpl) -> None:
        with pytest.raises(ValueError, match="Unknown plugin type"):
            catalog.get_schema("widgets", "csv")

    def test_unknown_name_raises_value_error(self, catalog: CatalogServiceImpl) -> None:
        with pytest.raises(ValueError, match="Unknown source plugin"):
            catalog.get_schema("source", "nonexistent_plugin_xyz")

    def test_unknown_name_includes_available(self, catalog: CatalogServiceImpl) -> None:
        with pytest.raises(ValueError, match="Available:"):
            catalog.get_schema("source", "nonexistent_plugin_xyz")


class TestTransformConfigVisibility:
    """Regression: transforms with config models must expose config fields.

    Prior bug: _resolve_config_model() caught UnknownPluginTypeError and
    returned None, silently downgrading transforms like type_coerce and
    value_transform to "no config" in the catalog.
    """

    def test_type_coerce_has_config_fields(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        entry = next((t for t in transforms if t.name == "type_coerce"), None)
        assert entry is not None, "type_coerce not found in registered transforms"
        assert len(entry.config_fields) > 0

    def test_value_transform_has_config_fields(self, catalog: CatalogServiceImpl) -> None:
        transforms = catalog.list_transforms()
        entry = next((t for t in transforms if t.name == "value_transform"), None)
        assert entry is not None, "value_transform not found in registered transforms"
        assert len(entry.config_fields) > 0

    def test_type_coerce_schema_has_properties(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("transform", "type_coerce")
        assert "properties" in info.json_schema

    def test_value_transform_schema_has_properties(self, catalog: CatalogServiceImpl) -> None:
        info = catalog.get_schema("transform", "value_transform")
        assert "properties" in info.json_schema


class TestGetConfigSchemaDefault:
    """Single-model plugins inherit the default get_config_schema().

    Pins the default implementation so that single-model plugins continue
    to return exactly ``config_model.model_json_schema()`` — not some
    subset or wrapper — and NullSource (no config_model) returns ``{}``.

    The default is duplicated on BaseSource, BaseTransform, and BaseSink
    (three near-identical copies). Each copy is regression-pinned
    independently here so silent drift in any one of them fails the suite
    instead of hiding until a plugin of that kind is catalog-inspected.
    """

    def test_default_matches_model_json_schema_for_csv_source(self, catalog: CatalogServiceImpl) -> None:
        from elspeth.plugins.sources.csv_source import CSVSource, CSVSourceConfig

        assert CSVSource.get_config_schema() == CSVSourceConfig.model_json_schema()
        info = catalog.get_schema("source", "csv")
        assert info.json_schema == CSVSourceConfig.model_json_schema()

    def test_default_matches_model_json_schema_for_passthrough_transform(self, catalog: CatalogServiceImpl) -> None:
        """BaseTransform's default must return exactly the Pydantic schema."""
        from elspeth.plugins.transforms.passthrough import PassThrough, PassThroughConfig

        assert PassThrough.get_config_schema() == PassThroughConfig.model_json_schema()
        info = catalog.get_schema("transform", "passthrough")
        assert info.json_schema == PassThroughConfig.model_json_schema()

    def test_default_matches_model_json_schema_for_csv_sink(self, catalog: CatalogServiceImpl) -> None:
        """BaseSink's default must return exactly the Pydantic schema."""
        from elspeth.plugins.sinks.csv_sink import CSVSink, CSVSinkConfig

        assert CSVSink.get_config_schema() == CSVSinkConfig.model_json_schema()
        info = catalog.get_schema("sink", "csv")
        assert info.json_schema == CSVSinkConfig.model_json_schema()

    def test_null_source_returns_empty_schema_default(self, catalog: CatalogServiceImpl) -> None:
        """Plugins without a config model return ``{}`` — not an error."""
        info = catalog.get_schema("source", "null")
        assert info.json_schema == {}


class TestFieldsFromDiscriminatedMalformed:
    """Exercise the boundary branches allowlisted in enforce_tier_model/web.yaml.

    Each branch corresponds to one allowlist entry whose ``safety`` field
    asserts a specific behavior when the incoming JSON schema is structurally
    unusual. These tests pin those behaviors so the allowlist rationale and
    the code stay in lockstep — a future refactor that silently changes
    "returns empty field list" into "raises KeyError" fails here, not in
    production.
    """

    @staticmethod
    def _service() -> CatalogServiceImpl:
        """Build a CatalogServiceImpl with builtins for driving internals."""
        pm = PluginManager()
        pm.register_builtin_plugins()
        return CatalogServiceImpl(pm)

    def test_oneof_entry_without_ref_is_skipped(self) -> None:
        """Inline-schema oneOf entries (no ``$ref``) are skipped, not crashed on.

        JSON Schema permits oneOf entries to be either ``{"$ref": ...}`` or
        an inline object schema. Pydantic's discriminated-union emission uses
        ``$ref``, but a mixed shape from a custom override must degrade
        gracefully to "contribute no fields."
        """
        svc = self._service()
        schema = {
            "oneOf": [
                {"type": "object", "properties": {"inline_field": {"type": "string"}}},
                {"$ref": "#/$defs/RealVariant"},
            ],
            "$defs": {
                "RealVariant": {
                    "properties": {"real_field": {"type": "string"}},
                    "required": ["real_field"],
                }
            },
        }
        fields = svc._fields_from_discriminated(schema)
        assert [f.name for f in fields] == ["real_field"]

    def test_non_defs_ref_prefix_is_skipped(self) -> None:
        """``$ref`` that does not point into local ``$defs`` is skipped.

        e.g. ``#/components/schemas/X`` — a remote OpenAPI shape, not the
        local-$defs shape Pydantic emits. Skipped rather than raising so the
        flattener stays robust to schemas we don't fully own.
        """
        svc = self._service()
        schema = {
            "oneOf": [
                {"$ref": "#/components/schemas/External"},
                {"$ref": "#/$defs/Local"},
            ],
            "$defs": {
                "Local": {
                    "properties": {"local_field": {"type": "string"}},
                    "required": ["local_field"],
                }
            },
        }
        fields = svc._fields_from_discriminated(schema)
        assert [f.name for f in fields] == ["local_field"]

    def test_variant_without_properties_contributes_no_fields(self) -> None:
        """A variant declaring no ``properties`` contributes zero fields.

        Valid JSON Schema ("empty config is allowed") — not an error.
        """
        svc = self._service()
        schema = {
            "oneOf": [
                {"$ref": "#/$defs/Empty"},
                {"$ref": "#/$defs/WithField"},
            ],
            "$defs": {
                "Empty": {},  # no properties, no required
                "WithField": {
                    "properties": {"only_field": {"type": "string"}},
                    "required": ["only_field"],
                },
            },
        }
        fields = svc._fields_from_discriminated(schema)
        assert [f.name for f in fields] == ["only_field"]
        # Field appears in only one variant => not universally required.
        assert fields[0].required is False

    def test_variant_without_required_marks_field_not_required(self) -> None:
        """Universal requiredness requires presence in EVERY variant.

        A field present in two variants but required-listed in only one must
        come back ``required=False`` — the intersection rule for honest
        summary reporting.
        """
        svc = self._service()
        schema = {
            "oneOf": [
                {"$ref": "#/$defs/A"},
                {"$ref": "#/$defs/B"},
            ],
            "$defs": {
                "A": {
                    "properties": {"shared": {"type": "string"}},
                    "required": ["shared"],
                },
                "B": {
                    "properties": {"shared": {"type": "string"}},
                    # no "required" key at all
                },
            },
        }
        fields = svc._fields_from_discriminated(schema)
        shared = next(f for f in fields if f.name == "shared")
        assert shared.required is False

    def test_dangling_ref_raises(self) -> None:
        """Dangling ``$ref`` surfaces as KeyError — treated as a Pydantic bug.

        If Pydantic emits a ``$ref`` into ``$defs`` but the corresponding
        entry is missing, silently returning an empty summary would hide
        the schema-generation bug. Surfacing KeyError makes it an auditable
        failure rather than invisible data loss.
        """
        svc = self._service()
        schema = {
            "oneOf": [{"$ref": "#/$defs/Missing"}],
            "$defs": {},
        }
        with pytest.raises(KeyError):
            svc._fields_from_discriminated(schema)


class TestDiscriminatedFieldOrdering:
    """Field insertion order across discriminated-union variants is preserved.

    The flattener walks variants in ``oneOf`` order and appends each
    field name the first time it is seen. This test pins that invariant so
    a refactor replacing the ordered-dedup with ``set(...)`` (which would
    lose ordering) fails loudly.
    """

    def test_llm_fields_ordered_common_then_azure_then_openrouter(self, catalog: CatalogServiceImpl) -> None:
        pytest.importorskip(
            "litellm",
            reason="LLM transform requires the [llm] extra; catalog discovery skips it otherwise.",
        )
        transforms = catalog.list_transforms()
        llm = next(t for t in transforms if t.name == "llm")
        field_order = [f.name for f in llm.config_fields]
        # ``provider`` + ``template`` are common to both variants (Azure first
        # in _PROVIDERS) so they must precede the Azure-only and
        # OpenRouter-only fields.
        assert field_order.index("provider") < field_order.index("deployment_name")
        assert field_order.index("provider") < field_order.index("base_url")
        # Azure-specific fields must come before OpenRouter-only fields
        # because Azure is registered first in _PROVIDERS.
        assert field_order.index("deployment_name") < field_order.index("base_url")
        assert field_order.index("endpoint") < field_order.index("base_url")


class TestOneOfRoutingPredicate:
    """``_extract_config_fields`` routes on BOTH ``oneOf`` and ``$defs``.

    The routing predicate requires both keys before delegating to
    ``_fields_from_discriminated``. A schema with just ``oneOf`` (and no
    ``$defs``) falls through to the flat-schema path rather than crashing.
    """

    @staticmethod
    def _service() -> CatalogServiceImpl:
        pm = PluginManager()
        pm.register_builtin_plugins()
        return CatalogServiceImpl(pm)

    def test_oneof_without_defs_treated_as_flat(self) -> None:
        svc = self._service()
        # No ``$defs`` => flat-schema path; returns whatever ``properties`` holds.
        schema = {
            "oneOf": [{"type": "object"}],
            "properties": {"top_field": {"type": "string"}},
            "required": ["top_field"],
        }
        fields = svc._extract_config_fields(schema)
        assert [f.name for f in fields] == ["top_field"]
        assert fields[0].required is True
