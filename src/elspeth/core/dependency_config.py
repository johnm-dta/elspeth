"""Configuration models for pipeline dependencies and commencement gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from elspeth.contracts.freeze import deep_freeze, deep_thaw, freeze_fields, require_int


def _validate_non_blank(value: str, field_name: str) -> str:
    """Strip and reject blank-only strings at the config boundary."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be blank (whitespace-only)")
    return stripped


class DependencyConfig(BaseModel):
    """Declares a pipeline that must run before this one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Unique label for this dependency")
    settings: str = Field(min_length=1, description="Path to dependency pipeline settings file")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return _validate_non_blank(v, "name")

    @field_validator("settings")
    @classmethod
    def _strip_settings(cls, v: str) -> str:
        return _validate_non_blank(v, "settings")


class CommencementGateConfig(BaseModel):
    """Declares a go/no-go condition evaluated before the pipeline starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Unique label for this gate")
    condition: str = Field(min_length=1, description="Expression evaluated against pre-flight context")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return _validate_non_blank(v, "name")

    @field_validator("condition")
    @classmethod
    def _validate_condition(cls, v: str) -> str:
        """Parse the condition expression at config time.

        ExpressionParser raises ExpressionSyntaxError or ExpressionSecurityError
        on invalid input — surfaced immediately instead of at gate evaluation time.
        """
        stripped = _validate_non_blank(v, "condition")
        from elspeth.core.expression_parser import ExpressionParser

        # Commencement gates evaluate against collections/dependency_runs/env context
        ExpressionParser(stripped, allowed_names=["collections", "dependency_runs", "env"])
        return stripped


# Supported probe providers — intentionally small, system-owned registry.
SupportedProbeProvider = Literal["chroma"]


class CollectionProbeConfig(BaseModel):
    """Declares a vector store collection to probe before gate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection: str = Field(min_length=1, description="Collection name to probe")
    provider: SupportedProbeProvider = Field(description="Provider type (e.g., 'chroma')")
    provider_config: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific connection config",
    )

    @field_validator("collection")
    @classmethod
    def _strip_collection(cls, v: str) -> str:
        return _validate_non_blank(v, "collection")

    @model_validator(mode="after")
    def _freeze_provider_config(self) -> CollectionProbeConfig:
        """Deep-freeze provider_config contents — Pydantic frozen=True prevents reassignment but not mutation of mutable containers."""
        object.__setattr__(self, "provider_config", deep_freeze(self.provider_config))
        return self

    @field_serializer("provider_config")
    @classmethod
    def _serialize_provider_config(cls, value: Mapping[str, Any]) -> dict[str, Any]:
        """Thaw MappingProxyType back to dict for Pydantic JSON serialization."""
        result = deep_thaw(value)
        if type(result) is not dict:
            raise TypeError(
                f"deep_thaw(provider_config) returned {type(result).__name__}, expected dict. Input type was {type(value).__name__}."
            )
        return result


@dataclass(frozen=True, slots=True)
class DependencyRunResult:
    """Result of a successful dependency pipeline run."""

    name: str
    run_id: str
    settings_hash: str
    duration_ms: int
    indexed_at: str  # ISO 8601 timestamp

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not self.settings_hash:
            raise ValueError("settings_hash must not be empty")
        if not self.indexed_at:
            raise ValueError("indexed_at must not be empty")
        require_int(self.duration_ms, "duration_ms", min_value=0)


@dataclass(frozen=True, slots=True)
class CommencementGateResult:
    """Result of a successful commencement gate evaluation.

    The ``result`` field is always ``True`` — gate failures raise
    ``CommencementGateFailedError`` instead of returning ``result=False``.
    The field exists so the audit trail records an explicit pass verdict,
    not just the absence of a failure.
    """

    name: str
    condition: str
    result: bool
    context_snapshot: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.condition:
            raise ValueError("condition must not be empty")
        if not self.result:
            raise ValueError(
                "CommencementGateResult.result must be True — gate failures raise CommencementGateFailedError, not result=False"
            )
        freeze_fields(self, "context_snapshot")


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Combined pre-flight results for audit recording.

    Produced by ``resolve_preflight()`` and carried through the orchestrator
    to the Landscape recorder. Recording is deferred until orchestrator.run()
    begins, so the audit trail captures preflight results alongside the run
    record. Both the CLI path and ``bootstrap_and_run()`` (sub-pipeline
    execution) produce this via the shared ``resolve_preflight()``.
    """

    dependency_runs: tuple[DependencyRunResult, ...]
    gate_results: tuple[CommencementGateResult, ...]

    def __post_init__(self) -> None:
        freeze_fields(self, "dependency_runs", "gate_results")
