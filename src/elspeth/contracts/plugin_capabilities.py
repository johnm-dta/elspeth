"""Closed, stdlib-only plugin policy declarations.

These declarations describe policy meaning.  They deliberately remain separate
from the open-ended ``capability_tags`` used for catalog presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PluginCapability(StrEnum):
    LLM = "llm"
    PROMPT_SHIELD = "prompt_shield"
    CONTENT_SAFETY = "content_safety"


class ControlMode(StrEnum):
    RECOMMEND = "recommend"
    REQUIRED = "required"


class WebConfigAuthority(StrEnum):
    USER_CONFIGURABLE = "user_configurable"
    USER_CONFIGURABLE_WITH_POLICY = "user_configurable_with_policy"
    OPERATOR_PROFILED = "operator_profiled"


class ControlRole(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ContentTrust(StrEnum):
    TRUSTED_INTERNAL = "trusted_internal"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True, order=True)
class CapabilityDeclaration:
    capability: PluginCapability
    control_role: ControlRole | None = None
    blocks_positive_detection: bool = False

    def __post_init__(self) -> None:
        expected_role = {
            PluginCapability.LLM: None,
            PluginCapability.PROMPT_SHIELD: ControlRole.INPUT,
            PluginCapability.CONTENT_SAFETY: ControlRole.OUTPUT,
        }[self.capability]
        if self.control_role is not expected_role:
            raise ValueError("capability declaration has an invalid control role")
        if self.capability is PluginCapability.LLM and self.blocks_positive_detection:
            raise ValueError("LLM capability cannot claim control blocking")
