"""Pure construction of one principal-scoped plugin availability snapshot."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Protocol

from elspeth.contracts.plugin_capabilities import PluginCapability, WebConfigAuthority
from elspeth.contracts.secrets import SecretsError
from elspeth.core.security.secret_loader import SecretNotFoundError
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.plugin_policy.models import (
    PluginAvailability,
    PluginAvailabilitySnapshot,
    PluginId,
    PluginUnavailableReason,
    WebPluginPolicy,
)
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, ProfileCredentialInventory

if TYPE_CHECKING:
    from elspeth.contracts.auth import AuthProviderType
    from elspeth.web.auth.models import UserIdentity
    from elspeth.web.secrets.server_store import ServerSecretStore
    from elspeth.web.secrets.service import WebSecretService
    from elspeth.web.secrets.user_store import UserSecretStore


class SecretInventory(ProfileCredentialInventory, Protocol):
    def has_ref(self, principal: str, name: str) -> bool: ...


def _catalog_items(catalog: CatalogService) -> dict[PluginId, PluginSummary]:
    return {
        **{PluginId("source", item.name): item for item in catalog.list_sources()},
        **{PluginId("transform", item.name): item for item in catalog.list_transforms()},
        **{PluginId("sink", item.name): item for item in catalog.list_sinks()},
    }


def _schema_secret_ready(schema: PluginSchemaInfo, *, principal: str, inventory: SecretInventory) -> bool:
    for requirement in schema.secret_requirements:
        if requirement.candidates:
            if not any(inventory.has_ref(principal, candidate) for candidate in requirement.candidates):
                return False
        else:
            # A required credential without a declared candidate cannot be
            # proven from the sanitized inventory and is therefore unavailable.
            return False
    return True


def _binding_fingerprint(
    *,
    generation_key: bytes,
    principal_scope: str,
    aliases: tuple[tuple[PluginId, tuple[str, ...]], ...],
    profile_bindings: tuple[tuple[PluginId, str, str, str], ...],
    available: frozenset[PluginId],
) -> str:
    payload = json.dumps(
        {
            "principal_scope": principal_scope,
            "aliases": [(str(plugin_id), list(values)) for plugin_id, values in aliases],
            "profile_bindings": [(str(plugin_id), alias, scope, generation) for plugin_id, alias, scope, generation in profile_bindings],
            "available": sorted(map(str, available)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hmac.new(generation_key, payload, hashlib.sha256).hexdigest()


def build_plugin_snapshot(
    *,
    policy: WebPluginPolicy,
    catalog: CatalogService,
    profiles: OperatorProfileRegistry,
    principal_scope: str,
    secret_inventory: SecretInventory,
    generation_key: bytes,
) -> PluginAvailabilitySnapshot:
    """Combine frozen policy with local, principal-specific availability facts."""
    catalog_items = _catalog_items(catalog)
    available: set[PluginId] = set()
    unavailable: list[PluginAvailability] = []
    usable_profile_aliases: list[tuple[PluginId, tuple[str, ...]]] = []
    selected_profile_aliases: list[tuple[PluginId, str | None]] = []
    profile_bindings: list[tuple[PluginId, str, str, str]] = []

    for plugin_id in sorted(policy.authorized):
        summary = catalog_items.get(plugin_id)
        if summary is None:
            unavailable.append(PluginAvailability(plugin_id, PluginUnavailableReason.NOT_INSTALLED))
            continue
        if summary.web_config_authority is WebConfigAuthority.OPERATOR_PROFILED:
            profile_states = profiles.profile_availability(
                plugin_id,
                principal=principal_scope,
                inventory=secret_inventory,
            )
            aliases = tuple(
                item.alias for item in profile_states if item.usable and profiles.check_local_requirements(plugin_id, item.alias).available
            )
            for profile_state in profile_states:
                if not profile_state.usable or profile_state.credential_scope is None or profile_state.generation is None:
                    continue
                generation_token = hmac.new(
                    generation_key,
                    json.dumps(
                        {
                            "scope": profile_state.credential_scope,
                            "alias": profile_state.alias,
                            "generation": profile_state.generation,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                    hashlib.sha256,
                ).hexdigest()
                profile_bindings.append((plugin_id, profile_state.alias, profile_state.credential_scope, generation_token))
            usable_profile_aliases.append((plugin_id, aliases))
            selected_profile_aliases.append((plugin_id, aliases[0] if aliases else None))
            if not aliases:
                unavailable.append(PluginAvailability(plugin_id, PluginUnavailableReason.PROFILE_UNAVAILABLE))
                continue
        schema = catalog.get_schema(plugin_id.kind, plugin_id.name)
        if not _schema_secret_ready(schema, principal=principal_scope, inventory=secret_inventory):
            unavailable.append(PluginAvailability(plugin_id, PluginUnavailableReason.CREDENTIAL_MISSING))
            continue
        available.add(plugin_id)

    declared_by_capability: dict[PluginCapability, set[PluginId]] = {capability: set() for capability in PluginCapability}
    for plugin_id in policy.authorized:
        item = catalog_items[plugin_id]
        for declaration in item.policy_capabilities:
            declared_by_capability[declaration.capability].add(plugin_id)
    preferences = dict(policy.preferences)
    selected: list[tuple[PluginCapability, PluginId | None]] = []
    for capability in PluginCapability:
        ordered = preferences.get(capability)
        if ordered is None:
            ordered = tuple(sorted(declared_by_capability[capability]))
        selected.append((capability, next((plugin_id for plugin_id in ordered if plugin_id in available), None)))

    alias_tuple = tuple(usable_profile_aliases)
    available_frozen = frozenset(available)
    fingerprint = _binding_fingerprint(
        generation_key=generation_key,
        principal_scope=principal_scope,
        aliases=alias_tuple,
        profile_bindings=tuple(sorted(profile_bindings)),
        available=available_frozen,
    )
    return PluginAvailabilitySnapshot.create(
        policy_hash=policy.policy_hash,
        principal_scope=principal_scope,
        available=available_frozen,
        unavailable=tuple(unavailable),
        selected=tuple(selected),
        usable_profile_aliases=alias_tuple,
        selected_profile_aliases=tuple(selected_profile_aliases),
        binding_generation_fingerprint=fingerprint,
        control_modes=policy.control_modes,
    )


class _BoundSecretInventory:
    def __init__(
        self,
        *,
        user_id: str,
        auth_provider: AuthProviderType,
        service: WebSecretService,
        server_store: ServerSecretStore,
        user_store: UserSecretStore,
    ) -> None:
        self._user_id = user_id
        self._auth_provider = auth_provider
        self._service = service
        self._server_store = server_store
        self._user_store = user_store

    def has_server_ref(self, name: str) -> bool:
        return self._server_store.has_secret(name)

    def has_user_ref(self, principal: str, name: str) -> bool:
        return self._user_store.has_secret(name, user_id=self._user_id, auth_provider_type=self._auth_provider)

    def has_ref(self, principal: str, name: str) -> bool:
        return self._service.has_ref(self._user_id, name, auth_provider_type=self._auth_provider)

    def server_generation(self, name: str) -> str | None:
        if not self._server_store.has_secret(name):
            return None
        try:
            _value, ref = self._server_store.get_secret(name)
        except (SecretsError, SecretNotFoundError):
            return None
        return ref.fingerprint

    def user_generation(self, principal: str, name: str) -> str | None:
        del principal
        if not self._user_store.has_secret(name, user_id=self._user_id, auth_provider_type=self._auth_provider):
            return None
        try:
            _value, ref = self._user_store.get_secret(name, user_id=self._user_id, auth_provider_type=self._auth_provider)
        except (SecretsError, SecretNotFoundError):
            return None
        return ref.fingerprint


class RequestPluginSnapshotFactory:
    """Settings-aligned request factory; never caches across principals."""

    def __init__(
        self,
        *,
        policy: WebPluginPolicy,
        catalog: CatalogService,
        profiles: OperatorProfileRegistry,
        auth_provider: AuthProviderType,
        secret_service: WebSecretService,
        server_store: ServerSecretStore,
        user_store: UserSecretStore,
        generation_key: bytes,
    ) -> None:
        self._policy = policy
        self._catalog = catalog
        self._profiles = profiles
        self._auth_provider = auth_provider
        self._secret_service = secret_service
        self._server_store = server_store
        self._user_store = user_store
        self._generation_key = generation_key

    def __call__(self, user: UserIdentity) -> PluginAvailabilitySnapshot:
        return self.for_user_id(user.user_id)

    def for_user_id(self, user_id: str) -> PluginAvailabilitySnapshot:
        """Build a snapshot for an already-authenticated web principal ID."""
        principal_scope = f"{self._auth_provider}:{user_id}"
        return build_plugin_snapshot(
            policy=self._policy,
            catalog=self._catalog,
            profiles=self._profiles,
            principal_scope=principal_scope,
            secret_inventory=_BoundSecretInventory(
                user_id=user_id,
                auth_provider=self._auth_provider,
                service=self._secret_service,
                server_store=self._server_store,
                user_store=self._user_store,
            ),
            generation_key=self._generation_key,
        )
