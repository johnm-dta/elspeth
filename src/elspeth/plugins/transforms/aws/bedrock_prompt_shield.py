"""Bedrock Guardrail prompt-attack shield."""

from __future__ import annotations

from typing import Any

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.plugin_capabilities import CapabilityDeclaration, ControlRole, PluginCapability, WebConfigAuthority
from elspeth.plugins.transforms.aws._guardrail_transform import BedrockGuardrailTransformBase, BedrockGuardrailTransformConfig
from elspeth.plugins.transforms.aws.guardrails_client import PROMPT_ATTACK_FILTERS, GuardrailSource


class AWSBedrockPromptShieldConfig(BedrockGuardrailTransformConfig):
    """Explicit CLI configuration for the prompt shield."""


class AWSBedrockPromptShield(BedrockGuardrailTransformBase):
    """Block prompt attacks identified by an operator-owned Guardrail."""

    name = "aws_bedrock_prompt_shield"
    determinism = Determinism.EXTERNAL_CALL
    web_config_authority = WebConfigAuthority.OPERATOR_PROFILED
    policy_capabilities = frozenset(
        {
            CapabilityDeclaration(
                PluginCapability.PROMPT_SHIELD,
                ControlRole.INPUT,
                blocks_positive_detection=True,
            )
        }
    )
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:1aaf47d6ca39af81"
    config_model = AWSBedrockPromptShieldConfig
    _required_filters = PROMPT_ATTACK_FILTERS
    _detected_reason = "prompt_injection_detected"
    _probe_field = "bedrock_prompt_shield_probe_text"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = AWSBedrockPromptShieldConfig.from_dict(config, plugin_name=self.name)
        super().__init__(config, cfg, "AWSBedrockPromptShield")

    @property
    def _guardrail_source(self) -> GuardrailSource:
        return "INPUT"

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Block prompt attacks using an operator-approved Bedrock Guardrail profile.",
                composer_hints=(
                    "Select only an opaque operator profile in web-authored pipelines.",
                    "List every untrusted prompt field that must be checked before an LLM call.",
                    "Both detect-only positives and Guardrail interventions are blocked.",
                ),
            )
        return None
