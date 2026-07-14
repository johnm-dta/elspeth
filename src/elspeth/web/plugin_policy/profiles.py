"""Typed operator profile settings and frozen runtime conversion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability
from elspeth.plugins.transforms.aws.guardrail_profiles import BedrockGuardrailProfileSettings

if TYPE_CHECKING:
    from elspeth.web.catalog.schemas import PluginSchemaInfo
    from elspeth.web.config import WebSettings
    from elspeth.web.plugin_policy.models import PluginId, WebPluginPolicy

CredentialScope = Literal["server", "user"]
_ALIAS = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_SECRET_REF = re.compile(r"[A-Z][A-Z0-9_]{0,255}\Z")


def validate_profile_alias(alias: str) -> str:
    if _ALIAS.fullmatch(alias) is None:
        raise ValueError("profile alias must be a lowercase opaque identifier")
    return alias


class WebLLMProfileSettings(BaseModel):
    """Operator-owned provider binding; private fields stay out of reprs."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    provider: str = Field(repr=False)
    model: str = Field(min_length=1, max_length=512, repr=False)
    credential_scope: CredentialScope | None = Field(default=None, repr=False)
    credential_ref: str | None = Field(default=None, repr=False)
    region_name: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", repr=False)
    endpoint: str | None = Field(default=None, repr=False)
    deployment_name: str | None = Field(default=None, min_length=1, max_length=256, repr=False)
    api_version: str | None = Field(default=None, min_length=1, max_length=64, repr=False)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300, repr=False)
    max_tokens: int | None = Field(default=None, gt=0, le=131072, repr=False)

    @model_validator(mode="after")
    def _validate_provider_binding(self) -> WebLLMProfileSettings:
        # Plan 09 owns this registry.  Profile validation consumes it rather
        # than maintaining a second provider allowlist.
        from elspeth.plugins.transforms.llm.transform import LLMTransform

        providers = LLMTransform.discriminated_variants()[1]
        if self.provider not in providers:
            raise ValueError("profile provider is not registered")
        if self.provider == "bedrock":
            if self.credential_scope is not None or self.credential_ref is not None:
                raise ValueError("Bedrock profiles use the keyless AWS credential chain")
            if self.endpoint is not None or self.deployment_name is not None or self.api_version is not None:
                raise ValueError("Bedrock profile contains fields owned by another provider")
            # Reuse Plan 09's provider model validation for model/region shape.
            providers[self.provider](
                provider="bedrock",
                model=self.model,
                region_name=self.region_name,
                schema={"mode": "observed"},
                prompt_template="{{ row }}",
            )
        else:
            if self.credential_scope is None or self.credential_ref is None:
                raise ValueError("credentialed profile requires explicit scope and reference")
            if _SECRET_REF.fullmatch(self.credential_ref) is None:
                raise ValueError("credential reference has invalid syntax")
            if self.provider == "openrouter" and any(
                value is not None for value in (self.region_name, self.endpoint, self.deployment_name, self.api_version)
            ):
                raise ValueError("OpenRouter profile contains unsupported provider fields")
            if self.provider == "azure":
                if self.endpoint is None or self.deployment_name is None:
                    raise ValueError("Azure profile requires operator endpoint and deployment")
                from elspeth.plugins.infrastructure.url_validation import validate_credential_safe_https_url

                validate_credential_safe_https_url(self.endpoint, field_name="endpoint")
        return self


@dataclass(frozen=True, slots=True)
class RuntimeWebLLMProfile:
    alias: str
    provider: str = field(repr=False)
    model: str = field(repr=False)
    credential_scope: CredentialScope | None = field(default=None, repr=False)
    credential_ref: str | None = field(default=None, repr=False)
    provider_options: tuple[tuple[str, object], ...] = field(default=(), repr=False)

    @classmethod
    def from_settings(cls, alias: str, settings: WebLLMProfileSettings) -> RuntimeWebLLMProfile:
        validate_profile_alias(alias)
        provider_fields = {
            "bedrock": (("region_name", settings.region_name),),
            "azure": (
                ("endpoint", settings.endpoint),
                ("deployment_name", settings.deployment_name),
                ("api_version", settings.api_version),
            ),
            "openrouter": (("timeout_seconds", settings.timeout_seconds),),
        }
        options = tuple(
            (name, value) for name, value in (*provider_fields[settings.provider], ("max_tokens", settings.max_tokens)) if value is not None
        )
        return cls(
            alias=alias,
            provider=settings.provider,
            model=settings.model,
            credential_scope=settings.credential_scope,
            credential_ref=settings.credential_ref,
            provider_options=options,
        )


@dataclass(frozen=True, slots=True)
class RuntimeWebPluginConfig:
    plugin_allowlist: tuple[str, ...]
    plugin_preferences: tuple[tuple[PluginCapability, tuple[str, ...]], ...]
    plugin_control_modes: tuple[tuple[PluginCapability, ControlMode], ...]
    llm_profiles: tuple[tuple[str, RuntimeWebLLMProfile], ...] = field(repr=False)
    tutorial_llm_profile: str | None
    bedrock_guardrail_profiles: tuple[BedrockGuardrailProfileSettings, ...] = field(repr=False)
    bedrock_guardrail_default_profiles: tuple[tuple[str, str], ...]

    @classmethod
    def from_settings(cls, settings: WebSettings) -> RuntimeWebPluginConfig:
        return cls(
            plugin_allowlist=tuple(settings.plugin_allowlist),
            plugin_preferences=tuple(
                (capability, tuple(plugin_ids))
                for capability, plugin_ids in sorted(settings.plugin_preferences.items(), key=lambda item: item[0].value)
            ),
            plugin_control_modes=tuple(sorted(settings.plugin_control_modes.items(), key=lambda item: item[0].value)),
            llm_profiles=tuple(
                (alias, RuntimeWebLLMProfile.from_settings(alias, profile)) for alias, profile in sorted(settings.llm_profiles.items())
            ),
            tutorial_llm_profile=settings.tutorial_llm_profile,
            bedrock_guardrail_profiles=tuple(sorted(settings.bedrock_guardrail_profiles, key=lambda profile: profile.alias)),
            bedrock_guardrail_default_profiles=tuple(sorted(settings.bedrock_guardrail_default_profiles.items())),
        )


class ProfileUnavailableReason(StrEnum):
    CREDENTIAL_MISSING = "credential_unavailable"
    LOCAL_REQUIREMENT_MISSING = "local_requirement_unavailable"


@dataclass(frozen=True, slots=True, order=True)
class ProfileAvailability:
    alias: str
    credential_scope: CredentialScope | None
    usable: bool
    reason: ProfileUnavailableReason | None = None
    generation: str | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class LocalRequirementResult:
    available: bool
    reason: ProfileUnavailableReason | None = None


@dataclass(frozen=True, slots=True)
class LoweredPluginConfig:
    executable_options: Mapping[str, object] = field(repr=False)
    audit_safe_options: Mapping[str, object]


class ProfileCredentialInventory(Protocol):
    def has_server_ref(self, name: str) -> bool: ...

    def has_user_ref(self, principal: str, name: str) -> bool: ...

    def server_generation(self, name: str) -> str | None: ...

    def user_generation(self, principal: str, name: str) -> str | None: ...


class OperatorProfileResolver(Protocol):
    def public_schema(self, full_schema: PluginSchemaInfo, available_aliases: tuple[str, ...]) -> PluginSchemaInfo: ...

    def lower_options(self, alias: str, safe_options: dict[str, object]) -> LoweredPluginConfig: ...

    def profile_availability(
        self,
        principal: str,
        inventory: ProfileCredentialInventory,
    ) -> tuple[ProfileAvailability, ...]: ...

    def check_local_requirements(self, alias: str) -> LocalRequirementResult: ...


_LLM_PRIVATE_OPTIONS = frozenset(
    {
        "provider",
        "model",
        "api_key",
        "api_key_secret",
        "base_url",
        "endpoint",
        "deployment_name",
        "region_name",
        "api_version",
        "credential_ref",
        "credential_scope",
        "tracing",
        "timeout_seconds",
        "max_tokens",
        "pool_size",
        "min_dispatch_delay_ms",
        "max_dispatch_delay_ms",
        "backoff_multiplier",
        "recovery_step_ms",
        "max_capacity_retry_seconds",
        "prompt_template_source",
        "lookup_source",
        "system_prompt_source",
        "resolved_prompt_template_hash",
    }
)


class _LLMProfileResolver:
    def __init__(
        self,
        profiles: tuple[tuple[str, RuntimeWebLLMProfile], ...],
        *,
        preferred_alias: str | None,
    ) -> None:
        self._profiles = dict(profiles)
        aliases = tuple(self._profiles)
        self._ordered_aliases = (
            (preferred_alias, *(alias for alias in aliases if alias != preferred_alias)) if preferred_alias in self._profiles else aliases
        )

    def public_schema(self, full_schema: PluginSchemaInfo, available_aliases: tuple[str, ...]) -> PluginSchemaInfo:
        from elspeth.web.catalog.schemas import PluginSchemaInfo

        safe_properties: dict[str, Any] = {}
        definitions = full_schema.json_schema.get("$defs", {})
        if isinstance(definitions, dict):
            from elspeth.plugins.transforms.llm.transform import LLMTransform

            provider_definition_names = {config_cls.__name__ for config_cls in LLMTransform.discriminated_variants()[1].values()}
            for definition_name in provider_definition_names:
                definition = definitions.get(definition_name)
                if not isinstance(definition, dict):
                    continue
                properties = definition.get("properties", {})
                if not isinstance(properties, dict):
                    continue
                for name, property_schema in properties.items():
                    if name not in _LLM_PRIVATE_OPTIONS and isinstance(property_schema, dict):
                        safe_properties.setdefault(name, deepcopy(property_schema))
        safe_properties = {
            "profile": {
                "type": "string",
                "enum": list(available_aliases),
                "description": "Operator-approved LLM profile alias",
            },
            **safe_properties,
        }
        public_json_schema: dict[str, Any] = {
            "type": "object",
            "properties": safe_properties,
            "required": ["profile", "prompt_template", "schema"],
            "additionalProperties": False,
        }
        referenced_definitions: dict[str, Any] = {}
        pending = _schema_refs(public_json_schema)
        while pending:
            definition_name = pending.pop()
            if definition_name in referenced_definitions:
                continue
            definition = definitions.get(definition_name) if isinstance(definitions, dict) else None
            if isinstance(definition, dict):
                referenced_definitions[definition_name] = deepcopy(definition)
                pending.update(_schema_refs(definition))
        if referenced_definitions:
            public_json_schema["$defs"] = referenced_definitions
        fields = [
            {
                "name": name,
                "type": "string" if name == "profile" else str(schema.get("type", "object")),
                "required": name in public_json_schema["required"],
                "description": schema.get("description"),
                **({"choices": list(available_aliases)} if name == "profile" else {}),
            }
            for name, schema in safe_properties.items()
        ]
        return PluginSchemaInfo(
            name=full_schema.name,
            plugin_type=full_schema.plugin_type,
            description=full_schema.description,
            json_schema=public_json_schema,
            knob_schema={"fields": fields},
            composer_hints=full_schema.composer_hints,
            secret_requirements=(),
            web_config_authority=full_schema.web_config_authority,
            policy_capabilities=full_schema.policy_capabilities,
        )

    def lower_options(self, alias: str, safe_options: dict[str, object]) -> LoweredPluginConfig:
        if set(safe_options) & _LLM_PRIVATE_OPTIONS:
            raise ValueError("private_profile_option")
        try:
            profile = self._profiles[alias]
        except KeyError:
            raise ValueError("profile_unavailable") from None
        executable = dict(safe_options)
        executable["provider"] = profile.provider
        executable["model"] = profile.model
        executable.update(profile.provider_options)
        if profile.credential_ref is not None:
            assert profile.credential_scope is not None
            executable["api_key"] = {
                "secret_ref": profile.credential_ref,
                "secret_scope": profile.credential_scope,
            }
        audit_safe = {"profile": alias, **safe_options}
        return LoweredPluginConfig(
            executable_options=MappingProxyType(executable),
            audit_safe_options=MappingProxyType(audit_safe),
        )

    def profile_availability(
        self,
        principal: str,
        inventory: ProfileCredentialInventory,
    ) -> tuple[ProfileAvailability, ...]:
        result: list[ProfileAvailability] = []
        for alias in self._ordered_aliases:
            profile = self._profiles[alias]
            if profile.credential_scope is None:
                result.append(ProfileAvailability(alias=alias, credential_scope=None, usable=True))
                continue
            assert profile.credential_ref is not None
            generation = (
                inventory.server_generation(profile.credential_ref)
                if profile.credential_scope == "server"
                else inventory.user_generation(principal, profile.credential_ref)
            )
            usable = generation is not None
            result.append(
                ProfileAvailability(
                    alias=alias,
                    credential_scope=profile.credential_scope,
                    usable=usable,
                    reason=None if usable else ProfileUnavailableReason.CREDENTIAL_MISSING,
                    generation=generation,
                )
            )
        return tuple(result)

    def check_local_requirements(self, alias: str) -> LocalRequirementResult:
        return LocalRequirementResult(available=alias in self._profiles)


_BEDROCK_PRIVATE_OPTIONS = frozenset(
    {
        "guardrail_identifier",
        "guardrail_version",
        "region",
        "endpoint",
        "endpoint_url",
        "credential",
        "credentials",
        "access_key",
        "secret_key",
        "session_token",
        "environment",
        "environment_marker",
    }
)


class _BedrockGuardrailProfileResolver:
    def __init__(self, profiles: tuple[BedrockGuardrailProfileSettings, ...], *, default_alias: str | None) -> None:
        self._profiles = {profile.alias: profile for profile in profiles}
        aliases = tuple(self._profiles)
        self._ordered_aliases = (
            (default_alias, *(alias for alias in aliases if alias != default_alias)) if default_alias in self._profiles else aliases
        )

    def public_schema(self, full_schema: PluginSchemaInfo, available_aliases: tuple[str, ...]) -> PluginSchemaInfo:
        from elspeth.web.catalog.schemas import PluginSchemaInfo

        safe_names = ("fields", "schema") if full_schema.name == "aws_bedrock_prompt_shield" else ("fields", "schema", "source")
        full_properties = full_schema.json_schema.get("properties", {})
        safe_properties: dict[str, Any] = {
            "profile": {
                "type": "string",
                "enum": list(available_aliases),
                "description": "Operator-approved Bedrock Guardrail profile alias",
            }
        }
        if isinstance(full_properties, dict):
            for name in safe_names:
                value = full_properties.get(name)
                if isinstance(value, dict):
                    safe_properties[name] = deepcopy(value)
        required = ["profile", "fields", "schema"]
        public_json_schema: dict[str, Any] = {
            "type": "object",
            "properties": safe_properties,
            "required": required,
            "additionalProperties": False,
        }
        fields = [
            {
                "name": name,
                "type": "string" if name == "profile" else str(schema.get("type", "object")),
                "required": name in required,
                "description": schema.get("description"),
                **({"choices": list(available_aliases)} if name == "profile" else {}),
            }
            for name, schema in safe_properties.items()
        ]
        return PluginSchemaInfo(
            name=full_schema.name,
            plugin_type=full_schema.plugin_type,
            description=full_schema.description,
            json_schema=public_json_schema,
            knob_schema={"fields": fields},
            composer_hints=full_schema.composer_hints,
            secret_requirements=(),
            web_config_authority=full_schema.web_config_authority,
            policy_capabilities=full_schema.policy_capabilities,
        )

    def lower_options(self, alias: str, safe_options: dict[str, object]) -> LoweredPluginConfig:
        if set(safe_options) & _BEDROCK_PRIVATE_OPTIONS:
            raise ValueError("private_profile_option")
        try:
            profile = self._profiles[alias]
        except KeyError:
            raise ValueError("profile_unavailable") from None
        allowed = {"fields", "schema"}
        if profile.plugin == "aws_bedrock_content_safety":
            allowed.add("source")
        if set(safe_options) - allowed:
            raise ValueError("private_profile_option")
        executable = dict(safe_options)
        executable.update(
            {
                "guardrail_identifier": profile.guardrail_identifier,
                "guardrail_version": profile.guardrail_version,
                "region": profile.region,
            }
        )
        return LoweredPluginConfig(
            executable_options=MappingProxyType(executable),
            audit_safe_options=MappingProxyType({"profile": alias, **safe_options}),
        )

    def profile_availability(
        self,
        principal: str,
        inventory: ProfileCredentialInventory,
    ) -> tuple[ProfileAvailability, ...]:
        del principal, inventory
        return tuple(ProfileAvailability(alias=alias, credential_scope=None, usable=True) for alias in self._ordered_aliases)

    def check_local_requirements(self, alias: str) -> LocalRequirementResult:
        profile = self._profiles.get(alias)
        if profile is None or not profile.check_local_requirements().available:
            return LocalRequirementResult(available=False, reason=ProfileUnavailableReason.LOCAL_REQUIREMENT_MISSING)
        return LocalRequirementResult(available=True)


def _schema_refs(value: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            refs.add(ref.removeprefix("#/$defs/"))
        for nested in value.values():
            refs.update(_schema_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.update(_schema_refs(nested))
    return refs


class OperatorProfileRegistry:
    """Resolver registry bound to one frozen process policy/config."""

    def __init__(self, *, policy: WebPluginPolicy, settings: RuntimeWebPluginConfig) -> None:
        from elspeth.web.plugin_policy.models import PluginId

        self._policy = policy
        self._resolvers: dict[PluginId, OperatorProfileResolver] = {
            PluginId("transform", "llm"): _LLMProfileResolver(
                settings.llm_profiles,
                preferred_alias=settings.tutorial_llm_profile,
            )
        }
        defaults = dict(settings.bedrock_guardrail_default_profiles)
        for plugin_name in ("aws_bedrock_prompt_shield", "aws_bedrock_content_safety"):
            plugin_profiles = tuple(profile for profile in settings.bedrock_guardrail_profiles if profile.plugin == plugin_name)
            if plugin_profiles:
                self._resolvers[PluginId("transform", plugin_name)] = _BedrockGuardrailProfileResolver(
                    plugin_profiles,
                    default_alias=defaults.get(plugin_name),
                )

    def public_schema(
        self,
        plugin_id: PluginId,
        full_schema: PluginSchemaInfo,
        *,
        available_aliases: tuple[str, ...],
    ) -> PluginSchemaInfo:
        resolver = self._resolvers.get(plugin_id)
        if resolver is None:
            return full_schema
        return resolver.public_schema(full_schema, available_aliases)

    def lower_options(
        self,
        plugin_id: PluginId,
        *,
        alias: str,
        safe_options: dict[str, object],
    ) -> LoweredPluginConfig:
        try:
            resolver = self._resolvers[plugin_id]
        except KeyError:
            raise ValueError("plugin_has_no_operator_profile") from None
        return resolver.lower_options(alias, safe_options)

    def profile_availability(
        self,
        plugin_id: PluginId,
        *,
        principal: str,
        inventory: ProfileCredentialInventory,
    ) -> tuple[ProfileAvailability, ...]:
        try:
            resolver = self._resolvers[plugin_id]
        except KeyError:
            return ()
        return resolver.profile_availability(principal, inventory)

    def check_local_requirements(self, plugin_id: PluginId, alias: str) -> LocalRequirementResult:
        try:
            resolver = self._resolvers[plugin_id]
        except KeyError:
            return LocalRequirementResult(available=False, reason=ProfileUnavailableReason.LOCAL_REQUIREMENT_MISSING)
        return resolver.check_local_requirements(alias)
