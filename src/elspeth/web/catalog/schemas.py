"""Catalog API response models.

All schemas in this module are Tier 1 responses describing system-owned
plugin metadata.  They inherit from ``_StrictResponse`` so that any
backend emission of a wrong type (or an extra field a frontend feature
flag would later quietly read) crashes at construction time instead of
reaching the UI.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from elspeth.contracts.enums import DerivedAuditCharacteristics
from elspeth.contracts.plugin_capabilities import CapabilityDeclaration, WebConfigAuthority

PluginKind = Literal["source", "transform", "sink"]


class _StrictResponse(BaseModel):
    """Tier 1 base for catalog responses — no coercion, no extras."""

    model_config = ConfigDict(strict=True, extra="forbid")


class ConfigFieldSummary(_StrictResponse):
    """Summary of a single field in a plugin's config model."""

    name: str
    type: str
    required: bool
    description: str | None = None
    default: Any | None = None


class PluginSecretRequirement(_StrictResponse):
    """Secret inventory requirement for composer plugin discovery.

    ``candidates`` names browser-safe secret refs, never secret values. Empty
    means the plugin requires the field to be wired to some available secret,
    but does not declare a canonical inventory name.
    """

    field: str
    candidates: tuple[str, ...] = ()


class PluginSummary(_StrictResponse):
    """Lightweight plugin info for catalog browsing.

    Phase 7A adds reference-content fields (when-to-use prose, capability
    tags, audit-characteristic flags) so the catalog drawer can render
    persona-facing reference cards instead of a bare name+description.
    All new fields are optional and default to None / empty for plugins
    that haven't been authored yet; the frontend renders a fallback
    message rather than blocking display.

    `audit_characteristics` is the catalog service's *derived* set:
    declared characteristics from the plugin class composed with the
    characteristic inferred from `determinism`, then sorted into a
    deterministic tuple for stable wire-format ordering. Quarantine
    behaviour is author-declared (the `_on_validation_failure` signal
    is a per-instance attribute and cannot be inferred from the class
    object). See `CatalogServiceImpl._derive_audit_characteristics`.
    """

    name: str
    description: str
    plugin_type: PluginKind
    config_fields: list[ConfigFieldSummary]

    # Phase 7A reference-content fields
    usage_when_to_use: str | None = None
    usage_when_not_to_use: str | None = None
    example_use: str | None = None
    capability_tags: tuple[str, ...] = ()
    web_config_authority: WebConfigAuthority = WebConfigAuthority.USER_CONFIGURABLE
    policy_capabilities: tuple[CapabilityDeclaration, ...] = ()
    audit_characteristics: DerivedAuditCharacteristics = ()

    # JIT-hints Phase 1: discovery-time composer hints. Populated from
    # ``plugin_cls.get_agent_assistance(issue_code=None).composer_hints``
    # when the plugin overrides the hook; ``()`` otherwise. Advisory
    # coaching only — not contract, not audit-hashed.
    composer_hints: tuple[str, ...] = ()

    # Secret availability gate for composer discovery. The composer filters
    # plugins with unsatisfied requirements before exposing them to the LLM.
    secret_requirements: tuple[PluginSecretRequirement, ...] = ()


class PluginSchemaInfo(_StrictResponse):
    """Full plugin schema detail for the composer.

    ``json_schema`` contains the raw output of ``ConfigModel.model_json_schema()``.
    It is ``{}`` (empty dict) when the plugin has no configuration model
    (e.g., the ``null`` source).
    """

    name: str
    plugin_type: PluginKind
    description: str
    json_schema: dict[str, Any]
    knob_schema: dict[str, Any]
    """Lowered composer knob schema, computed once at catalog load."""

    # JIT-hints Phase 1: same semantics as PluginSummary.composer_hints.
    # Carried on the full-schema response so an LLM that asked for the
    # schema (e.g. to introspect config fields before set_source) sees
    # the same hints it would have seen from list_*. Avoids requiring
    # a follow-up get_plugin_assistance call to surface the hints.
    composer_hints: tuple[str, ...] = ()

    # Mirrors PluginSummary.secret_requirements on the full-schema surface so
    # get_plugin_schema and get_plugin_assistance can enforce the same gate.
    secret_requirements: tuple[PluginSecretRequirement, ...] = ()
    web_config_authority: WebConfigAuthority = WebConfigAuthority.USER_CONFIGURABLE
    policy_capabilities: tuple[CapabilityDeclaration, ...] = ()
