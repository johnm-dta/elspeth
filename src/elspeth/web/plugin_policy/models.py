"""Immutable plugin-policy and request-snapshot values."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability

PluginKind = Literal["source", "transform", "sink"]
_PLUGIN_ID = re.compile(r"(source|transform|sink):([a-z][a-z0-9_]*)\Z")


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class PluginId:
    kind: PluginKind
    name: str

    def __post_init__(self) -> None:
        if _PLUGIN_ID.fullmatch(f"{self.kind}:{self.name}") is None:
            raise ValueError("invalid kind-qualified plugin id")

    @classmethod
    def parse(cls, raw: str) -> PluginId:
        match = _PLUGIN_ID.fullmatch(raw)
        if match is None:
            raise ValueError("invalid kind-qualified plugin id")
        return cls(cast(PluginKind, match.group(1)), match.group(2))

    def __str__(self) -> str:
        return f"{self.kind}:{self.name}"


class PluginUnavailableReason(StrEnum):
    NOT_AUTHORIZED = "plugin_not_enabled"
    NOT_INSTALLED = "plugin_not_installed"
    LOCAL_REQUIREMENT_MISSING = "plugin_unavailable"
    CREDENTIAL_MISSING = "credential_unavailable"
    PROFILE_UNAVAILABLE = "profile_unavailable"


@dataclass(frozen=True, slots=True, order=True)
class PluginAvailability:
    plugin_id: PluginId
    reason: PluginUnavailableReason


@dataclass(frozen=True, slots=True)
class WebPluginPolicy:
    schema_version: int
    required: frozenset[PluginId]
    configured_optional: frozenset[PluginId]
    authorized: frozenset[PluginId]
    preferences: tuple[tuple[PluginCapability, tuple[PluginId, ...]], ...]
    control_modes: tuple[tuple[PluginCapability, ControlMode], ...]
    plugin_code_identities: tuple[tuple[PluginId, str, str], ...]
    policy_hash: str

    @classmethod
    def create(
        cls,
        *,
        required: frozenset[PluginId],
        configured_optional: frozenset[PluginId],
        preferences: tuple[tuple[PluginCapability, tuple[PluginId, ...]], ...],
        control_modes: tuple[tuple[PluginCapability, ControlMode], ...],
        plugin_code_identities: tuple[tuple[PluginId, str, str], ...],
        schema_version: int = 1,
    ) -> WebPluginPolicy:
        authorized = required | configured_optional
        canonical = {
            "schema_version": schema_version,
            "required": sorted(map(str, required)),
            "configured_optional": sorted(map(str, configured_optional)),
            "authorized": sorted(map(str, authorized)),
            "preferences": [(capability.value, [str(plugin_id) for plugin_id in ordered]) for capability, ordered in preferences],
            "control_modes": [(capability.value, mode.value) for capability, mode in control_modes],
            "plugin_code_identities": [
                (str(plugin_id), version, source_hash)
                for plugin_id, version, source_hash in sorted(plugin_code_identities, key=lambda item: item[0])
            ],
        }
        return cls(
            schema_version=schema_version,
            required=required,
            configured_optional=configured_optional,
            authorized=authorized,
            preferences=preferences,
            control_modes=control_modes,
            plugin_code_identities=tuple(sorted(plugin_code_identities, key=lambda item: item[0])),
            policy_hash=_canonical_hash(canonical),
        )


@dataclass(frozen=True, slots=True)
class PluginAvailabilitySnapshot:
    policy_hash: str
    principal_scope: str
    available: frozenset[PluginId]
    unavailable: tuple[PluginAvailability, ...]
    selected: tuple[tuple[PluginCapability, PluginId | None], ...]
    usable_profile_aliases: tuple[tuple[PluginId, tuple[str, ...]], ...]
    selected_profile_aliases: tuple[tuple[PluginId, str | None], ...]
    binding_generation_fingerprint: str
    snapshot_hash: str

    @classmethod
    def create(
        cls,
        *,
        policy_hash: str,
        principal_scope: str,
        available: frozenset[PluginId],
        unavailable: tuple[PluginAvailability, ...],
        selected: tuple[tuple[PluginCapability, PluginId | None], ...],
        usable_profile_aliases: tuple[tuple[PluginId, tuple[str, ...]], ...],
        selected_profile_aliases: tuple[tuple[PluginId, str | None], ...],
        binding_generation_fingerprint: str,
    ) -> PluginAvailabilitySnapshot:
        canonical = {
            "policy_hash": policy_hash,
            "principal_scope": principal_scope,
            "available": sorted(map(str, available)),
            "unavailable": [(str(item.plugin_id), item.reason.value) for item in sorted(unavailable)],
            "selected": [(capability.value, None if plugin_id is None else str(plugin_id)) for capability, plugin_id in selected],
            "usable_profile_aliases": [(str(plugin_id), list(aliases)) for plugin_id, aliases in usable_profile_aliases],
            "selected_profile_aliases": [(str(plugin_id), alias) for plugin_id, alias in selected_profile_aliases],
            "binding_generation_fingerprint": binding_generation_fingerprint,
        }
        return cls(
            policy_hash=policy_hash,
            principal_scope=principal_scope,
            available=available,
            unavailable=unavailable,
            selected=selected,
            usable_profile_aliases=usable_profile_aliases,
            selected_profile_aliases=selected_profile_aliases,
            binding_generation_fingerprint=binding_generation_fingerprint,
            snapshot_hash=_canonical_hash(canonical),
        )
