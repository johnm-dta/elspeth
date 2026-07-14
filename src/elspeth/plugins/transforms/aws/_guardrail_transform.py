"""Shared fail-closed runtime for Bedrock Guardrail transforms."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, field_validator

from elspeth.contracts import Determinism
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.errors import FrameworkBugError, TransformErrorCategory
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms.aws.guardrail_profiles import (
    check_bedrock_local_requirements,
    validate_guardrail_identifier,
    validate_guardrail_region,
    validate_guardrail_version,
)
from elspeth.plugins.transforms.aws.guardrails_client import (
    BedrockGuardrailsClient,
    GuardrailResponseError,
    GuardrailServiceError,
    GuardrailSource,
    build_bedrock_runtime_client,
)
from elspeth.plugins.transforms.safety_utils import validate_fields_not_empty


class BedrockGuardrailTransformConfig(TransformDataConfig):
    """Explicit trained-operator binding shared by both transforms."""

    _component_type_exempt: ClassVar[bool] = True
    model_config: ClassVar[ConfigDict] = ConfigDict(**TransformDataConfig.model_config, hide_input_in_errors=True)

    guardrail_identifier: str = Field(min_length=1, max_length=2048, repr=False)
    guardrail_version: str = Field(min_length=1, max_length=32, repr=False)
    region: str = Field(min_length=1, max_length=64, repr=False)
    fields: list[str] = Field(min_length=1, max_length=32)

    @field_validator("guardrail_identifier")
    @classmethod
    def _identifier(cls, value: str) -> str:
        return validate_guardrail_identifier(value)

    @field_validator("guardrail_version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_guardrail_version(value)

    @field_validator("region")
    @classmethod
    def _region(cls, value: str) -> str:
        return validate_guardrail_region(value)

    @field_validator("fields")
    @classmethod
    def _fields(cls, value: list[str]) -> list[str]:
        validated = validate_fields_not_empty(value)
        assert isinstance(validated, list)
        if len(set(validated)) != len(validated):
            raise ValueError("fields must not contain duplicates")
        if any(len(field) > 256 for field in validated):
            raise ValueError("field names must contain at most 256 characters")
        return validated


class BedrockGuardrailTransformBase(BaseTransform, ABC):
    """Common row scanning, audited call construction, and fail-closed aggregation."""

    determinism = Determinism.EXTERNAL_CALL
    passes_through_input = True
    plugin_version = "1.0.0"

    _required_filters: ClassVar[tuple[str, ...]]
    _detected_reason: ClassVar[TransformErrorCategory]
    _probe_field: ClassVar[str]

    @classmethod
    def check_web_local_requirements(cls) -> bool:
        return check_bedrock_local_requirements().available

    def __init__(self, config: dict[str, Any], cfg: BedrockGuardrailTransformConfig, schema_name: str) -> None:
        super().__init__(config)
        self._initialize_declared_input_fields(cfg)
        self._guardrail_identifier = cfg.guardrail_identifier
        self._guardrail_version = cfg.guardrail_version
        self._region = cfg.region
        self._fields = tuple(cfg.fields)
        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
        self.input_schema, self.output_schema = self._create_schemas(cfg.schema_config, schema_name)
        self._recorder: PluginAuditWriter | None = None
        self._run_id = ""
        self._telemetry_emit: Callable[[Any], None] = lambda _event: None
        self._sdk_client: Any | None = None

    @property
    @abstractmethod
    def _guardrail_source(self) -> GuardrailSource: ...

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        if ctx.landscape is None:
            raise FrameworkBugError("Bedrock Guardrail transforms require Landscape audit recording")
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        if self._sdk_client is None:
            self._sdk_client = build_bedrock_runtime_client(self._region)

    def close(self) -> None:
        close = getattr(self._sdk_client, "close", None)
        if callable(close):
            close()
        self._sdk_client = None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "guardrail_identifier": "probeguardrail",
            "guardrail_version": "1",
            "region": "us-east-1",
            "fields": [cls._probe_field],
            "schema": {"mode": "observed"},
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [self._augment_invariant_probe_row(probe, field_name=self._probe_field, value="safe content")]

    def execute_forward_invariant_probe(self, probe_rows: list[PipelineRow], ctx: Any) -> TransformResult:
        if len(probe_rows) != 1:
            raise FrameworkBugError("Bedrock Guardrail invariant probe requires exactly one row")

        class _SafeProbeSDK:
            def __init__(self, required_filters: tuple[str, ...]) -> None:
                self._required_filters = required_filters

            def apply_guardrail(self, **_kwargs: object) -> dict[str, object]:
                usage = {
                    "contentPolicyUnits": 1,
                    "contextualGroundingPolicyUnits": 0,
                    "sensitiveInformationPolicyFreeUnits": 0,
                    "sensitiveInformationPolicyUnits": 0,
                    "topicPolicyUnits": 0,
                    "wordPolicyUnits": 0,
                }
                filters = [{"type": name, "confidence": "NONE", "action": "NONE", "detected": False} for name in self._required_filters]
                return {
                    "usage": usage,
                    "action": "NONE",
                    "outputs": [],
                    "assessments": [{"contentPolicy": {"filters": filters}}],
                }

            def close(self) -> None:
                return None

        prior_recorder = self._recorder
        prior_run_id = self._run_id
        prior_telemetry = self._telemetry_emit
        prior_sdk = self._sdk_client
        try:
            self._recorder = ctx.landscape
            self._run_id = ctx.run_id
            self._telemetry_emit = ctx.telemetry_emit
            self._sdk_client = _SafeProbeSDK(self._required_filters)
            return self.process(probe_rows[0], ctx)
        finally:
            self._recorder = prior_recorder
            self._run_id = prior_run_id
            self._telemetry_emit = prior_telemetry
            self._sdk_client = prior_sdk

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        if self._recorder is None or self._sdk_client is None or not self._run_id:
            raise FrameworkBugError("Bedrock Guardrail transform used before on_start")
        if ctx.state_id is None:
            raise FrameworkBugError("Bedrock Guardrail transform requires a state_id")
        token_id = ctx.token.token_id if ctx.token is not None else None
        client = BedrockGuardrailsClient(
            execution=self._recorder,
            state_id=ctx.state_id,
            run_id=self._run_id,
            telemetry_emit=self._telemetry_emit,
            guardrail_identifier=self._guardrail_identifier,
            guardrail_version=self._guardrail_version,
            region=self._region,
            audit_salt=hashlib.sha256(f"elspeth-bedrock-guardrail:{self._run_id}".encode()).digest(),
            sdk_client=self._sdk_client,
            token_id=token_id,
        )

        for field_name in self._fields:
            if field_name not in row:
                return TransformResult.error({"reason": "missing_field", "field": field_name}, retryable=False)
            value = row[field_name]
            if type(value) is not str:
                return TransformResult.error(
                    {"reason": "non_string_field", "field": field_name, "actual_type": type(value).__name__},
                    retryable=False,
                )
            try:
                decision = client.apply_guardrail(
                    text=value,
                    source=self._guardrail_source,
                    required_filters=self._required_filters,
                )
            except (GuardrailServiceError, GuardrailResponseError, ValueError):
                return TransformResult.error(
                    {
                        "reason": "api_call_failed",
                        "field": field_name,
                        "error_type": "guardrail_service_error",
                    },
                    retryable=False,
                )
            if decision.detected:
                return TransformResult.error(
                    {
                        "reason": self._detected_reason,
                        "field": field_name,
                        "categories": list(decision.matched_filters),
                        "error_type": "intervened" if decision.intervened else "detect_only",
                    },
                    retryable=False,
                )

        return TransformResult.success(
            self._align_output_row_contract(row),
            success_reason={"action": "validated", "metadata": {"fields_checked": len(self._fields)}},
        )
