"""Typed operator profile settings and frozen runtime conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from elspeth.contracts.plugin_capabilities import ControlMode, PluginCapability

if TYPE_CHECKING:
    from elspeth.web.config import WebSettings

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

    provider: str
    model: str = Field(min_length=1, max_length=512, repr=False)
    credential_scope: CredentialScope | None = Field(default=None, repr=False)
    credential_ref: str | None = Field(default=None, repr=False)
    region_name: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    endpoint: str | None = Field(default=None, repr=False)
    deployment_name: str | None = Field(default=None, min_length=1, max_length=256, repr=False)
    api_version: str | None = Field(default=None, min_length=1, max_length=64)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    max_tokens: int | None = Field(default=None, gt=0, le=131072)

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
    provider: str
    model: str = field(repr=False)
    credential_scope: CredentialScope | None = field(default=None, repr=False)
    credential_ref: str | None = field(default=None, repr=False)
    provider_options: tuple[tuple[str, object], ...] = field(default=(), repr=False)

    @classmethod
    def from_settings(cls, alias: str, settings: WebLLMProfileSettings) -> RuntimeWebLLMProfile:
        validate_profile_alias(alias)
        options = tuple(
            (name, value)
            for name, value in (
                ("region_name", settings.region_name),
                ("endpoint", settings.endpoint),
                ("deployment_name", settings.deployment_name),
                ("api_version", settings.api_version),
                ("timeout_seconds", settings.timeout_seconds),
                ("max_tokens", settings.max_tokens),
            )
            if value is not None
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
        )
