"""CatalogServiceImpl — wraps PluginManager for catalog browsing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from elspeth.contracts.enums import AuditCharacteristic, DerivedAuditCharacteristics, Determinism
from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.plugins.infrastructure.discovery import get_plugin_description
from elspeth.plugins.infrastructure.manager import PluginManager, PluginNotFoundError
from elspeth.web.catalog.knob_schema import (
    KnobSchema,
    lower_discriminated_to_knob_schema,
    lower_model_to_knob_schema,
    validate_knob_schema,
)
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


# Typed views over the JSON-Schema documents that Pydantic's
# ``model_json_schema()`` emits for plugin config models. The *values* in
# these documents are first-party (our own plugin config models produced
# them — system code), but the *presence* of individual keys is governed by
# the JSON Schema specification, which we do not author: ``required`` is
# omitted when no field is mandatory, ``default`` is omitted when a field has
# none, top-level ``type`` is absent for ``anyOf`` properties, ``$ref`` is
# absent for inline ``oneOf`` entries. Parsing each fragment into one of
# these permissive models makes the spec-optional keys explicit typed fields
# with honest defaults (absent ``required`` -> empty list, absent ``default``
# -> ``None``) so the traversal accesses typed attributes directly instead of
# guessing with ``.get(key, default)``. A ``ValidationError`` here would mean
# our own schema generation produced a structurally impossible document — a
# first-party bug — so it is intentionally left to propagate (crash), never
# swallowed.
class _SchemaProperty(BaseModel):
    """One entry under a JSON-Schema ``properties`` map."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str | None = None
    description: str | None = None
    default: Any = None
    any_of: list[_SchemaProperty] = Field(default_factory=list, alias="anyOf")


class _SchemaObject(BaseModel):
    """A JSON-Schema object document (top-level model or ``$defs`` variant)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    properties: dict[str, _SchemaProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class _OneOfEntry(BaseModel):
    """One entry of a discriminated-union ``oneOf`` list.

    JSON Schema permits each entry to be either a ``$ref`` into ``$defs`` or
    an inline object schema; an inline entry simply omits ``$ref``, so the
    field defaults to the empty string and the caller skips it.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ref: str = Field(default="", alias="$ref")


# Map Determinism enum values to the AuditCharacteristic flag they
# imply. The catalog surfaces these as visual cues on the plugin card so
# a compliance-focused user (Linda persona) can see at a glance which
# audit traits apply without reading the technical description.
#
# Subscript access ([determinism]) is deliberate: if a future seventh
# Determinism value is added to contracts/enums.py without updating this
# table, the KeyError surfaces immediately at catalog-build time rather
# than silently returning None and dropping the inferred flag.
#
# The closed vocabulary of audit-characteristic strings is the
# AuditCharacteristic enum itself (defined in contracts/enums.py).  A typo
# at a plugin's `audit_characteristics` declaration site (e.g. attempting
# `frozenset({"io-read"})` as a bare string) fails mypy rather than
# silently disappearing from the rendered catalog card.
_DETERMINISM_TO_AUDIT_FLAG: dict[Determinism, AuditCharacteristic] = {
    Determinism.IO_READ: AuditCharacteristic.IO_READ,
    Determinism.IO_WRITE: AuditCharacteristic.IO_WRITE,
    Determinism.EXTERNAL_CALL: AuditCharacteristic.EXTERNAL_CALL,
    Determinism.DETERMINISTIC: AuditCharacteristic.DETERMINISTIC,
    Determinism.SEEDED: AuditCharacteristic.SEEDED,
    Determinism.NON_DETERMINISTIC: AuditCharacteristic.NON_DETERMINISTIC,
}

# Default Determinism value per plugin kind — mirrors the class-level
# `determinism = Determinism.<X>` declared on BaseSource / BaseTransform /
# BaseSink (`plugins/infrastructure/base.py`).
#
# Used by `_derive_audit_characteristics` to suppress emission of the
# determinism-derived AuditCharacteristic flag when the plugin's declared
# value EQUALS the kind default. The predicate is value-equality, not
# inheritance-detection — every concrete plugin redeclares `determinism`
# explicitly under the ADR-010 declaration-trust framework's
# `__init_subclass__` guard, so "inherited the default" never literally
# occurs at runtime. The suppression is therefore about display
# redundancy, not authorial intent:
#
#   "Every Source has Determinism.IO_READ" is an architectural fact,
#   not a per-plugin signal. Showing the io_read chip on every Source
#   card adds visual noise without distinguishing CSV from Dataverse.
#   A non-default value (e.g. NullSource → DETERMINISTIC, DataverseSource
#   → EXTERNAL_CALL) does distinguish — the chip's presence then
#   communicates "this Source deviates from the kind norm in a
#   reproducibility-relevant way."
#
# A plugin author who explicitly redeclares the kind default still gets
# suppression — the chip would teach the user nothing they couldn't
# infer from the kind. Surfacing authorial intent (the *act* of
# declaring) belongs in source-code review, not on the catalog card.
#
# Subscript access is deliberate: a future PluginKind addition or
# Determinism rebase on the base classes that drifts from this table
# fails fast at catalog-build time instead of silently mis-suppressing.
_KIND_DEFAULT_DETERMINISM: dict[PluginKind, Determinism] = {
    "source": Determinism.IO_READ,
    "transform": Determinism.DETERMINISTIC,
    "sink": Determinism.IO_WRITE,
}


def _derive_audit_characteristics(plugin_cls: PluginClass, *, plugin_kind: PluginKind) -> DerivedAuditCharacteristics:
    """Compose declared + inferred audit characteristics for a plugin.

    The declared set comes from the plugin class's `audit_characteristics`
    attribute (defaulting to `frozenset()` on the base). The inferred set
    is derived from `determinism` *only when the subclass overrode the
    kind default*. A plugin that inherits its kind's default determinism
    contributes no inferred flag, because surfacing it on every card of
    that kind teaches the user nothing per-plugin — the architectural
    fact (every Source reads I/O, every Sink writes I/O, every Transform
    defaults to deterministic) belongs in category-level documentation,
    not in a flag repeated 6+ times.

    Quarantine behaviour is **author-declared, not inferred.** The
    source-quarantine signal lives in `_on_validation_failure`, which is
    set per-instance in `__init__` from runtime config — it does not
    exist on the class object. Reading `plugin_cls._on_validation_failure`
    here would AttributeError at catalog-build time. Sources whose
    runtime configuration supports non-discard quarantine routing
    declare `"quarantine"` in their `audit_characteristics` frozenset
    (the CSV canonical example does this).

    Direct attribute access (`plugin_cls.audit_characteristics`,
    `plugin_cls.determinism`) is correct here: the bases and protocols
    declare these with sensible defaults, so every plugin reachable via
    the catalog has them. A plugin without these attributes would be a
    malformed system plugin (Tier 1 bug); crash via AttributeError is
    the correct response, not defensive fallback.

    Cross-reference (do not unify):
    ``elspeth.web.audit_readiness.service._build_plugin_trust_row`` also
    reads ``determinism`` from every plugin in a composition, but for a
    *different* purpose: it classifies each plugin as boundary-vs-internal
    for the readiness panel's plugin-trust row. The two surfaces
    deliberately diverge on kind-default determinism:

      - **This function** suppresses the determinism-derived audit-
        characteristic flag when a plugin uses its kind's default,
        because surfacing "every Source reads I/O" on every Source card
        teaches the user nothing per-plugin (architectural facts
        belong in category-level documentation, not in a repeated chip).

      - **``_build_plugin_trust_row``** does NOT suppress: every Source
        and every Sink is unconditionally boundary, because writing
        data out of the pipeline (or reading external data in) crosses
        a Tier-3 trust boundary regardless of whether the destination
        is a local file or a remote service. The readiness panel's
        compliance question ("which components cross a trust boundary
        on this run?") is structurally different from the catalog
        card's UX question ("what does this plugin teach the operator
        that they don't already know from its kind?").

    The deliberate widening of sink classification (csv/json sinks are
    now boundary, where the deleted ``trust.py`` excluded them) is
    captured in ADR-021. Extracting a shared
    ``BoundaryDerivation`` helper would conflate "compose display
    chips" with "classify trust crossings" — two operations that
    happen to read the same input but answer different questions.
    """
    declared: frozenset[AuditCharacteristic] = plugin_cls.audit_characteristics
    determinism = plugin_cls.determinism

    inferred: frozenset[AuditCharacteristic] = frozenset()
    if determinism is not _KIND_DEFAULT_DETERMINISM[plugin_kind]:
        # Author overrode the kind default — emit the corresponding flag.
        # Subscript raises KeyError if a future Determinism value is added
        # to contracts/enums.py without updating _DETERMINISM_TO_AUDIT_FLAG;
        # that crash is correct (silent None would drop the flag with no
        # test failure and no audit-trail signal).
        inferred = frozenset({_DETERMINISM_TO_AUDIT_FLAG[determinism]})

    # Sort for stable wire-format ordering; the response model exposes
    # this as a tuple[AuditCharacteristic, ...] which serialises to a
    # flat list of flag strings on the wire (StrEnum members serialise
    # as their str value).
    return tuple(sorted(declared | inferred))


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
        self._schema_cache: dict[tuple[PluginKind, str], PluginSchemaInfo] = {}
        self._populate_schema_cache("source", self._source_classes)
        self._populate_schema_cache("transform", self._transform_classes)
        self._populate_schema_cache("sink", self._sink_classes)

    def list_sources(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "source") for cls in self._source_classes]

    def list_transforms(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "transform") for cls in self._transform_classes]

    def list_sinks(self) -> list[PluginSummary]:
        return [self._to_summary(cls, "sink") for cls in self._sink_classes]

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        if plugin_type not in _VALID_TYPES:
            raise ValueError(f"Unknown plugin type: {plugin_type}. Must be one of: {sorted(_VALID_TYPES)}")

        key = (plugin_type, name)
        if key in self._schema_cache:
            return self._schema_cache[key]

        available = self._available_names(plugin_type)
        raise ValueError(f"Unknown {plugin_type} plugin: {name}. Available: {available}")

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        """Dispatch postscript-hint resolution to the plugin's classmethod.

        ``_get_plugin_class`` raises ``ValueError`` for unknown plugins;
        we let that propagate so callers see the same error shape as
        ``get_schema``.
        """
        plugin_cls = self._get_plugin_class(plugin_type, plugin_name)
        return plugin_cls.get_post_call_hints(
            tool_name=tool_name,
            config_snapshot=config_snapshot,
        )

    # -- Private helpers --

    def _populate_schema_cache(self, plugin_type: PluginKind, classes: Sequence[PluginClass]) -> None:
        for plugin_cls in classes:
            name: str = plugin_cls.name
            self._schema_cache[(plugin_type, name)] = self._build_schema_info(plugin_type, name, plugin_cls)

    def _build_schema_info(
        self,
        plugin_type: PluginKind,
        name: str,
        plugin_cls: PluginClass,
    ) -> PluginSchemaInfo:
        # Plugins own schema emission — single-model plugins use the default
        # on the plugin base, discriminated-union plugins override.
        json_schema = self._catalog_schema(plugin_cls, plugin_type)
        knob_schema = self._knob_schema(plugin_cls, plugin_type=plugin_type, name=name)
        validate_knob_schema(knob_schema, plugin_kind=plugin_type, plugin_name=name)

        # Full docstring for schema view (not just first line)
        description = (plugin_cls.__doc__ or "").strip()
        if not description:
            description = get_plugin_description(plugin_cls)

        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=description,
            json_schema=json_schema,
            knob_schema=cast(dict[str, Any], knob_schema),
            composer_hints=self._discovery_composer_hints(plugin_cls),
        )

    def _discovery_composer_hints(self, plugin_cls: PluginClass) -> tuple[str, ...]:
        """Pull discovery-time composer_hints from a plugin's assistance hook.

        Calls ``plugin_cls.get_agent_assistance(issue_code=None)`` and
        returns ``assistance.composer_hints`` when populated, else an
        empty tuple. The hook is part of the plugin contract (defined on
        BaseTransform, BaseSink, BaseSource — see
        ``plugins/infrastructure/base.py``); any plugin that doesn't
        override it returns ``None`` here, which the catalog renders as
        empty hints. Advisory coaching only — not part of any audit
        hash; see ``contracts/plugin_assistance.py`` for the discipline.
        """
        assistance = plugin_cls.get_agent_assistance(issue_code=None)
        if assistance is None:
            return ()
        return assistance.composer_hints

    def _knob_schema(self, plugin_cls: PluginClass, *, plugin_type: PluginKind, name: str) -> KnobSchema:
        try:
            discriminated_variants = cast(Any, plugin_cls).discriminated_variants
        except AttributeError:
            config_model = plugin_cls.get_config_model()
            if config_model is None:
                return {"fields": []}
            return lower_model_to_knob_schema(
                cast(type[BaseModel], config_model),
                plugin_kind=plugin_type,
                plugin_name=name,
            )
        if not callable(discriminated_variants):
            return lower_discriminated_to_knob_schema(
                plugin_cls,
                plugin_kind=plugin_type,
                plugin_name=name,
            )
        return lower_discriminated_to_knob_schema(
            plugin_cls,
            plugin_kind=plugin_type,
            plugin_name=name,
        )

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
        """Convert a plugin class to a PluginSummary.

        Phase 7A: also emits reference-content fields. Audit
        characteristics are the *derived* set: declared chars from
        `audit_characteristics` composed with the flag derived from
        `determinism`. The frontend reads audit_characteristics as a
        flat list of flag strings.
        """
        name: str = plugin_cls.name
        description = get_plugin_description(plugin_cls)
        schema = self._catalog_schema(plugin_cls, plugin_type)
        config_fields = self._extract_config_fields(schema)

        # Direct attribute access: the bases and protocols declare every
        # Phase-7A field with a default. A plugin missing them would be
        # a malformed system plugin (Tier 1 bug); crash is correct.
        usage_when_to_use = plugin_cls.usage_when_to_use
        usage_when_not_to_use = plugin_cls.usage_when_not_to_use
        example_use = plugin_cls.example_use
        capability_tags = plugin_cls.capability_tags

        audit_characteristics = _derive_audit_characteristics(plugin_cls, plugin_kind=plugin_type)

        return PluginSummary(
            name=name,
            description=description,
            plugin_type=plugin_type,
            config_fields=config_fields,
            usage_when_to_use=usage_when_to_use,
            usage_when_not_to_use=usage_when_not_to_use,
            example_use=example_use,
            capability_tags=capability_tags,
            audit_characteristics=audit_characteristics,
            composer_hints=self._discovery_composer_hints(plugin_cls),
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

        # Pydantic's model_json_schema() produces a JSON Schema document whose
        # spec-optional keys ("properties", "required", "type", ...) are made
        # explicit typed fields by _SchemaObject (absent "required" -> empty
        # list). Parsing here surfaces a structurally impossible document as a
        # first-party ValidationError (crash) rather than guessing with .get().
        parsed = _SchemaObject.model_validate(schema)
        required_fields = set(parsed.required)
        return [
            self._field_summary(field_name, field_schema, field_name in required_fields)
            for field_name, field_schema in parsed.properties.items()
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
        variant_props: list[dict[str, _SchemaProperty]] = []
        variant_required: list[set[str]] = []
        for raw_entry in schema["oneOf"]:
            # A ``oneOf`` entry may legitimately be an inline schema rather
            # than a ``$ref`` — JSON Schema permits both shapes. _OneOfEntry
            # makes the optional ``$ref`` an explicit field (default ""), so
            # the inline case is skipped explicitly, not crashed on.
            entry = _OneOfEntry.model_validate(raw_entry)
            if not entry.ref.startswith(_DEFS_REF_PREFIX):
                continue
            # Dangling ``$ref`` (target missing from ``$defs``) is treated
            # as a Pydantic-schema bug — direct subscript lets the KeyError
            # propagate instead of silently producing a truncated summary.
            variant = _SchemaObject.model_validate(defs[entry.ref[len(_DEFS_REF_PREFIX) :]])
            variant_props.append(variant.properties)
            variant_required.append(set(variant.required))

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
            field_schema: _SchemaProperty = next(
                (props[field_name] for props in variant_props if field_name in props),
                _SchemaProperty(),
            )
            required = all(field_name in props and field_name in req for props, req in zip(variant_props, variant_required, strict=True))
            fields.append(self._field_summary(field_name, field_schema, required))
        return fields

    @staticmethod
    def _field_summary(name: str, field_schema: _SchemaProperty, required: bool) -> ConfigFieldSummary:
        """Build a ConfigFieldSummary from one JSON-Schema property entry.

        ``field_schema`` is a typed view of a Pydantic-emitted property whose
        spec-optional keys ("type", "anyOf", "default", "description") are
        explicit fields with honest absence defaults (``None`` / empty list),
        so requiredness/default absence is preserved without ``.get()``.
        """
        # Type precedence: (1) explicit top-level ``type``; else (2) the first
        # non-null ``anyOf`` branch type (Pydantic emits no top-level ``type``
        # for ``X | None`` fields); else (3) ``"object"`` as the catch-all.
        json_type = field_schema.type or "object"
        if field_schema.any_of and not field_schema.type:
            for branch in field_schema.any_of:
                if branch.type != "null":
                    json_type = branch.type or "object"
                    break
        return ConfigFieldSummary(
            name=name,
            type=json_type,
            required=required,
            description=field_schema.description,
            default=field_schema.default,
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
