"""Runtime preflight result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields, require_int


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

    The ``result`` field is always ``True`` - gate failures raise
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
                "CommencementGateResult.result must be True - gate failures raise CommencementGateFailedError, not result=False"
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


__all__ = [
    "CommencementGateResult",
    "DependencyRunResult",
    "PreflightResult",
]
