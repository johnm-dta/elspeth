"""Frozen, audit-safe evidence for one web plugin-policy decision."""

from __future__ import annotations

import re
from dataclasses import dataclass

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLUGIN_ID = re.compile(r"(?:source|transform|sink):[a-z][a-z0-9_]*\Z")
_PROFILE_ALIAS = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\Z")
_SOURCE_HASH = re.compile(r"sha256:[0-9a-f]{16}\Z")
_DECISION_CODE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")


def _require_sha256(name: str, value: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _require_plugin_id(value: str) -> None:
    if _PLUGIN_ID.fullmatch(value) is None:
        raise ValueError("plugin evidence contains an invalid kind-qualified plugin id")


def _require_canonical(name: str, values: tuple[object, ...]) -> None:
    if values != tuple(sorted(values, key=repr)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a canonical sorted unique tuple")


@dataclass(frozen=True, slots=True)
class WebPluginPolicyEvidence:
    """Sanitized policy/snapshot facts persisted with a web run.

    This L0 value carries only stable identifiers and bounded decision codes.
    Principal identity, secret references, profile bindings and remote payloads
    have no representational surface here.
    """

    schema_version: int
    policy_hash: str
    snapshot_hash: str
    authorized_plugin_ids: tuple[str, ...]
    available_plugin_ids: tuple[str, ...]
    control_modes: tuple[tuple[str, str], ...]
    selected_implementations: tuple[tuple[str, str | None], ...]
    selected_profile_aliases: tuple[tuple[str, str | None], ...]
    plugin_code_identities: tuple[tuple[str, str, str], ...]
    binding_generation_fingerprint: str
    decision_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        _require_sha256("policy_hash", self.policy_hash)
        _require_sha256("snapshot_hash", self.snapshot_hash)
        _require_sha256("binding_generation_fingerprint", self.binding_generation_fingerprint)

        _require_canonical("authorized_plugin_ids", self.authorized_plugin_ids)
        _require_canonical("available_plugin_ids", self.available_plugin_ids)
        for plugin_id in self.authorized_plugin_ids + self.available_plugin_ids:
            _require_plugin_id(plugin_id)
        if not set(self.available_plugin_ids) <= set(self.authorized_plugin_ids):
            raise ValueError("available_plugin_ids must be a subset of authorized_plugin_ids")

        _require_canonical("control_modes", self.control_modes)
        for capability, mode in self.control_modes:
            PluginCapability(capability)
            ControlMode(mode)

        _require_canonical("selected_implementations", self.selected_implementations)
        selected_capabilities: set[str] = set()
        for capability, selected_plugin_id in self.selected_implementations:
            PluginCapability(capability)
            if capability in selected_capabilities:
                raise ValueError("selected_implementations must contain one row per capability")
            selected_capabilities.add(capability)
            if selected_plugin_id is not None:
                _require_plugin_id(selected_plugin_id)
                if selected_plugin_id not in self.available_plugin_ids:
                    raise ValueError("selected implementation must be available")

        _require_canonical("selected_profile_aliases", self.selected_profile_aliases)
        for plugin_id, alias in self.selected_profile_aliases:
            _require_plugin_id(plugin_id)
            if plugin_id not in self.authorized_plugin_ids:
                raise ValueError("selected profile plugin must be authorized")
            if alias is not None and _PROFILE_ALIAS.fullmatch(alias) is None:
                raise ValueError("selected profile alias must be a safe lowercase opaque profile alias")

        _require_canonical("plugin_code_identities", self.plugin_code_identities)
        identity_plugins: set[str] = set()
        for plugin_id, version, source_hash in self.plugin_code_identities:
            _require_plugin_id(plugin_id)
            if plugin_id not in self.authorized_plugin_ids:
                raise ValueError("plugin code identity must be authorized")
            if plugin_id in identity_plugins:
                raise ValueError("plugin_code_identities must contain one row per plugin")
            identity_plugins.add(plugin_id)
            if _VERSION.fullmatch(version) is None:
                raise ValueError("plugin code identity contains an invalid version")
            if _SOURCE_HASH.fullmatch(source_hash) is None:
                raise ValueError("plugin code identity contains an invalid source hash")

        _require_canonical("decision_codes", self.decision_codes)
        if not self.decision_codes:
            raise ValueError("decision_codes must contain at least one bounded decision")
        if any(_DECISION_CODE.fullmatch(code) is None or len(code) > 64 for code in self.decision_codes):
            raise ValueError("decision_codes must be bounded lowercase identifiers")
