"""Operator-owned Bedrock Guardrail profile contracts."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BedrockGuardrailPlugin = Literal[
    "aws_bedrock_prompt_shield",
    "aws_bedrock_content_safety",
]

BEDROCK_GUARDRAIL_PLUGIN_IDS: tuple[BedrockGuardrailPlugin, ...] = (
    "aws_bedrock_prompt_shield",
    "aws_bedrock_content_safety",
)

_ALIAS = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_GUARDRAIL_ID = re.compile(r"(?:[a-z0-9]+|arn:aws(?:-[^:]+)?:bedrock:[a-z0-9-]{1,20}:[0-9]{12}:guardrail/[a-z0-9]+)\Z")
_GUARDRAIL_ARN = re.compile(
    r"arn:(?P<partition>aws(?:-[^:]+)?):bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:guardrail/[a-z0-9]+\Z"
)
_NUMERIC_VERSION = re.compile(r"[1-9][0-9]{0,7}\Z")

# Pinned from boto3/botocore 1.43.46's offline ``endpoints.json`` for the
# commercial ``aws`` partition and service ``bedrock``.  Keep this static so
# config validation neither imports the optional SDK nor performs network I/O.
BEDROCK_GUARDRAIL_REGIONS = frozenset(
    {
        "af-south-1",
        "ap-east-2",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-northeast-3",
        "ap-south-1",
        "ap-south-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-southeast-3",
        "ap-southeast-4",
        "ap-southeast-5",
        "ap-southeast-6",
        "ap-southeast-7",
        "ca-central-1",
        "ca-west-1",
        "eu-central-1",
        "eu-central-2",
        "eu-north-1",
        "eu-south-1",
        "eu-south-2",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "il-central-1",
        "me-central-1",
        "me-south-1",
        "mx-central-1",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    }
)


@dataclass(frozen=True, slots=True)
class BedrockLocalRequirementResult:
    available: bool


def check_bedrock_local_requirements() -> BedrockLocalRequirementResult:
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


def validate_guardrail_identifier(value: str) -> str:
    if _GUARDRAIL_ID.fullmatch(value) is None:
        raise ValueError("guardrail identifier has invalid syntax")
    return value


def validate_guardrail_version(value: str) -> str:
    if _NUMERIC_VERSION.fullmatch(value) is None:
        raise ValueError("guardrail version must be an immutable positive numeric version")
    return value


def validate_guardrail_region(value: str) -> str:
    if value not in BEDROCK_GUARDRAIL_REGIONS:
        raise ValueError("AWS region is not in the supported Bedrock vocabulary")
    return value


class BedrockGuardrailProfileSettings(BaseModel):
    """Frozen private binding selected through an opaque public alias."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    alias: str
    plugin: BedrockGuardrailPlugin
    guardrail_identifier: str = Field(min_length=1, max_length=2048, repr=False)
    guardrail_version: str = Field(min_length=1, max_length=8, repr=False)
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
        return validate_guardrail_identifier(value)

    @field_validator("guardrail_version")
    @classmethod
    def _validate_guardrail_version(cls, value: str) -> str:
        return validate_guardrail_version(value)

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: str) -> str:
        return validate_guardrail_region(value)

    @model_validator(mode="after")
    def _validate_arn_region(self) -> BedrockGuardrailProfileSettings:
        match = _GUARDRAIL_ARN.fullmatch(self.guardrail_identifier)
        if match is not None and (match.group("partition") != "aws" or match.group("region") != self.region):
            raise ValueError("guardrail ARN partition and region must match the configured commercial region")
        return self

    def check_local_requirements(self) -> BedrockLocalRequirementResult:
        """Check optional SDK availability/version without making a network call."""
        return check_bedrock_local_requirements()
