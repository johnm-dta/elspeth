"""CatalogServiceImpl — wraps PluginManager for catalog browsing."""

from __future__ import annotations

from typing import Any, cast

from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.plugins.infrastructure.discovery import get_plugin_description
from elspeth.plugins.infrastructure.manager import PluginManager, PluginNotFoundError
from elspeth.web.catalog.schemas import (
    ConfigFieldSummary,
    PluginKind,
    PluginSchemaInfo,
    PluginSummary,
)

# A plugin class the catalog can introspect. Narrower than bare ``type`` so
# the attribute access on ``plugin_cls.name`` / ``plugin_cls.get_config_schema()``
# is type-checked instead of silenced with ``# type: ignore[attr-defined]``.
PluginClass = type[SourceProtocol] | type[TransformProtocol] | type[SinkProtocol]

# Valid singular plugin type identifiers
_VALID_TYPES = frozenset({"source", "transform", "sink"})

# JSON-Schema $ref prefix for local $defs used by Pydantic discriminated unions.
_DEFS_REF_PREFIX = "#/$defs/"

# ADR-013 declared-input-field checks currently dispatch only for non-batch
# transforms. Batch-aware transform schemas must not advertise this option
# until a batch pre-emission dispatch site exists.
_DECLARED_INPUT_FIELDS_OPTION = "required_input_fields"


class CatalogServiceImpl:
    """Read-only catalog backed by PluginManager.

    Receives an already-initialized PluginManager via constructor injection.
    Does NOT call register_builtin_plugins() — the shared singleton factory
    (get_shared_plugin_manager) handles initialization before injection.

    Caches plugin class lists once at construction. The plugin set is
    fixed for the lifetime of the process.

    Schema emission is delegated to ``plugin_cls.get_config_schema()`` so
    that plugins whose config is a discriminated union (e.g. ``LLMTransform``
    over ``provider``) can publish a full ``oneOf`` contract instead of a
    truncated base-class schema.
    """

    def __init__(self, plugin_manager: PluginManager) -> None:
        self._pm = plugin_manager
        # Cache plugin classes once
        self._source_classes = plugin_manager.get_sources()
        self._transform_classes = plugin_manager.get_transforms()
        self._sink_classes = plugin_manager.get_sinks()

    def list_sources(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "source") for cls in self._source_classes]

    def list_transforms(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "transform") for cls in self._transform_classes]

    def list_sinks(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "sink") for cls in self._sink_classes]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        plugin_cls = self._get_plugin_class(plugin_type, name)

        # Plugins own schema emission — single-model plugins use the default
        # on the plugin base, discriminated-union plugins override.
        json_schema = self._catalog_schema(plugin_cls, plugin_type)

        # Full docstring for schema view (not just first line)
        description = (plugin_cls.__doc__ or "").strip()
        if not description:
            description = get_plugin_description(plugin_cls)

        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=description,
            json_schema=json_schema,
        )

    # -- Private helpers --

    def _get_plugin_class(self, plugin_type: PluginKind, name: str) -> PluginClass:
        """Look up a plugin class by (type, name) with a descriptive error.

        Raises ``ValueError`` when the plugin name is not registered. The
        narrower ``PluginNotFoundError`` subtype from ``PluginManager`` is
        caught and rewrapped so callers see a single consistent exception
        type regardless of which lookup method rejected the name.

        Defense-in-depth guard on ``plugin_type``: the parameter is typed
        ``PluginKind`` so structurally reachable only with "source" /
        "transform" / "sink", but composer tools dispatch via untyped
        ``arguments["plugin_type"]`` reads — a bypass at that boundary
        should crash here with a descriptive error, not silently fall
        through to ``get_sink_by_name`` and surface as a misleading
        "Unknown sink plugin" message.
        """
        if plugin_type not in _VALID_TYPES:
            raise ValueError(f"Unknown plugin type: {plugin_type}. Must be one of: {sorted(_VALID_TYPES)}")

        try:
            if plugin_type == "source":
                return self._pm.get_source_by_name(name)
            if plugin_type == "transform":
                return self._pm.get_transform_by_name(name)
            return self._pm.get_sink_by_name(name)
        except PluginNotFoundError as exc:
            available = self._available_names(plugin_type)
            raise ValueError(f"Unknown {plugin_type} plugin: {name}. Available: {available}") from exc

    def _to_summary(self, plugin_cls: PluginClass, plugin_type: PluginKind) -> PluginSummary:
        """Convert a plugin class to a PluginSummary."""
        name: str = plugin_cls.name
        description = get_plugin_description(plugin_cls)
        schema = self._catalog_schema(plugin_cls, plugin_type)
        config_fields = self._extract_config_fields(schema)
        return PluginSummary(
            name=name,
            description=description,
            plugin_type=plugin_type,
            config_fields=config_fields,
        )

    def _catalog_schema(self, plugin_cls: PluginClass, plugin_type: PluginKind) -> dict[str, Any]:
        """Return the composer-visible schema for a plugin class."""
        schema: dict[str, Any] = plugin_cls.get_config_schema()
        if plugin_type != "transform":
            return schema

        transform_cls = cast(type[TransformProtocol], plugin_cls)
        if not transform_cls.is_batch_aware:
            return schema

        return self._without_declared_input_fields(schema)

    def _without_declared_input_fields(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Remove ADR-013 non-batch-only fields from a JSON schema object."""
        sanitized = dict(schema)
        if "properties" in schema:
            sanitized["properties"] = {
                field_name: field_schema
                for field_name, field_schema in schema["properties"].items()
                if field_name != _DECLARED_INPUT_FIELDS_OPTION
            }
        if "required" in schema:
            sanitized["required"] = [field_name for field_name in schema["required"] if field_name != _DECLARED_INPUT_FIELDS_OPTION]
        if "$defs" in schema:
            sanitized["$defs"] = {
                definition_name: self._without_declared_input_fields(definition_schema)
                for definition_name, definition_schema in schema["$defs"].items()
            }
        return sanitized

    def _extract_config_fields(self, schema: dict[str, Any]) -> list[ConfigFieldSummary]:
        """Flatten a plugin's JSON schema into ConfigFieldSummary entries.

        Discriminated unions (``oneOf`` over ``$defs``) are flattened into
        the union of every variant's fields. A field is marked ``required``
        only when it is required in **every** variant — that is the only
        defensible summary answer when requiredness varies by discriminator
        (the full schema on ``PluginSchemaInfo.json_schema`` preserves the
        per-variant truth).
        """
        if "oneOf" in schema and "$defs" in schema:
            return self._fields_from_discriminated(schema)

        # Pydantic's model_json_schema() produces a JSON Schema dict.
        # Keys like "type", "anyOf", "description", "default" are conditionally
        # present per JSON Schema spec — .get() is correct here (not defensive
        # programming, but standard JSON Schema traversal).
        properties: dict[str, Any] = schema.get("properties", {})
        required_fields: set[str] = set(schema.get("required", []))
        return [
            self._field_summary(field_name, field_schema, field_name in required_fields) for field_name, field_schema in properties.items()
        ]

    def _fields_from_discriminated(self, schema: dict[str, Any]) -> list[ConfigFieldSummary]:
        """Union fields across ``$defs`` variants referenced by ``oneOf``.

        Precondition: caller (``_extract_config_fields``) has already verified
        that ``"oneOf" in schema and "$defs" in schema`` — so both keys are
        required here, not optional. Direct subscript is therefore correct;
        KeyError would indicate a caller bug, not a JSON-Schema edge case.

        Required iff required in every variant that references this schema's
        ``oneOf``. When the same field appears in multiple variants, its
        per-variant ``type``/``description``/``default`` may diverge (e.g.
        ``api_key`` in Azure vs OpenRouter). The summary reports the **first**
        variant's metadata as a deliberate lossy projection — the authoritative
        per-variant contract lives in the caller-visible ``$defs`` on the full
        JSON schema. Consumers needing per-variant truth MUST read
        ``PluginSchemaInfo.json_schema`` directly.
        """
        defs: dict[str, dict[str, Any]] = schema["$defs"]
        variant_props: list[dict[str, Any]] = []
        variant_required: list[set[str]] = []
        for entry in schema["oneOf"]:
            # A ``oneOf`` entry may legitimately be an inline schema rather
            # than a ``$ref`` — JSON Schema permits both shapes. ``.get()``
            # with a default lets us skip the inline case (contributing no
            # variant fields) rather than crashing on a valid-but-unusual
            # Pydantic output.
            ref = entry.get("$ref", "")
            if not ref.startswith(_DEFS_REF_PREFIX):
                continue
            # Dangling ``$ref`` (target missing from ``$defs``) is treated
            # as a Pydantic-schema bug — direct subscript lets the KeyError
            # propagate instead of silently producing a truncated summary.
            variant = defs[ref[len(_DEFS_REF_PREFIX) :]]
            variant_props.append(variant.get("properties", {}))
            variant_required.append(set(variant.get("required", [])))

        # Preserve insertion order: walk variants in oneOf order, append new names.
        ordered_fields: list[str] = []
        seen: set[str] = set()
        for props in variant_props:
            for field_name in props:
                if field_name not in seen:
                    seen.add(field_name)
                    ordered_fields.append(field_name)

        fields: list[ConfigFieldSummary] = []
        for field_name in ordered_fields:
            # First variant that carries this field defines its surface metadata.
            field_schema: dict[str, Any] = next(
                (props[field_name] for props in variant_props if field_name in props),
                {},
            )
            required = all(field_name in props and field_name in req for props, req in zip(variant_props, variant_required, strict=True))
            fields.append(self._field_summary(field_name, field_schema, required))
        return fields

    @staticmethod
    def _field_summary(name: str, field_schema: dict[str, Any], required: bool) -> ConfigFieldSummary:
        """Build a ConfigFieldSummary from one JSON-Schema property entry."""
        json_type = field_schema.get("type", "object")
        # anyOf produces no top-level type — pick first non-null branch type
        if "anyOf" in field_schema and not field_schema.get("type"):
            for branch in field_schema["anyOf"]:
                if branch.get("type") != "null":
                    json_type = branch.get("type", "object")
                    break
        return ConfigFieldSummary(
            name=name,
            type=json_type,
            required=required,
            description=field_schema.get("description"),
            default=field_schema.get("default"),
        )

    def _available_names(self, plugin_type: PluginKind) -> list[str]:
        """Get sorted list of available plugin names for a type."""
        if plugin_type == "source":
            classes: list[PluginClass] = list(self._source_classes)
        elif plugin_type == "transform":
            classes = list(self._transform_classes)
        else:
            classes = list(self._sink_classes)
        return sorted(cls.name for cls in classes)
