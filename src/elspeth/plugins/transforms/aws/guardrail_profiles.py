"""Operator-owned Bedrock Guardrail profile contracts."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BedrockGuardrailPlugin = Literal[
    "aws_bedrock_prompt_shield",
    "aws_bedrock_content_safety",
]

BEDROCK_GUARDRAIL_PLUGIN_IDS: tuple[BedrockGuardrailPlugin, ...] = (
    "aws_bedrock_prompt_shield",
    "aws_bedrock_content_safety",
)

_ALIAS = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_GUARDRAIL_ID = re.compile(r"(?:[a-z0-9]+|arn:(?:aws|aws-cn|aws-us-gov):bedrock:[a-z0-9-]+:[0-9]{12}:guardrail/[a-z0-9]+)\Z")
_REGION = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)+-[0-9]\Z")
_NUMERIC_VERSION = re.compile(r"[1-9][0-9]*\Z")


@dataclass(frozen=True, slots=True)
class BedrockLocalRequirementResult:
    available: bool


class BedrockGuardrailProfileSettings(BaseModel):
    """Frozen private binding selected through an opaque public alias."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    alias: str
    plugin: BedrockGuardrailPlugin
    guardrail_identifier: str = Field(min_length=1, max_length=2048, repr=False)
    guardrail_version: str = Field(min_length=1, max_length=32, repr=False)
    region: str = Field(min_length=1, max_length=64, repr=False)

    @field_validator("alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        if _ALIAS.fullmatch(value) is None:
            raise ValueError("profile alias must be a lowercase opaque identifier")
        return value

    @field_validator("guardrail_identifier")
    @classmethod
    def _validate_guardrail_identifier(cls, value: str) -> str:
        if _GUARDRAIL_ID.fullmatch(value) is None:
            raise ValueError("guardrail identifier has invalid syntax")
        return value

    @field_validator("guardrail_version")
    @classmethod
    def _validate_guardrail_version(cls, value: str) -> str:
        if _NUMERIC_VERSION.fullmatch(value) is None:
            raise ValueError("guardrail version must be an immutable positive numeric version")
        return value

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: str) -> str:
        if _REGION.fullmatch(value) is None:
            raise ValueError("AWS region has invalid syntax")
        return value

    def check_local_requirements(self) -> BedrockLocalRequirementResult:
        """Check optional SDK availability/version without making a network call."""
        try:
            boto3 = importlib.import_module("boto3")
            botocore = importlib.import_module("botocore")
        except ImportError:
            return BedrockLocalRequirementResult(available=False)

        for module in (boto3, botocore):
            version = getattr(module, "__version__", "")
            try:
                major, minor, *_rest = (int(part) for part in version.split("."))
            except (TypeError, ValueError):
                return BedrockLocalRequirementResult(available=False)
            if major != 1 or minor < 40:
                return BedrockLocalRequirementResult(available=False)
        return BedrockLocalRequirementResult(available=True)
