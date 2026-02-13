# src/elspeth/testing/__init__.py
"""Test infrastructure for ELSPETH pipelines.

Factories for constructing production types with sensible defaults.
When a backbone type's constructor changes, update the factory here.
Tests and benchmarks that use factories need ZERO changes.

This package also contains:
- chaosengine: Shared utilities for chaos testing (injection engine, metrics store, latency)
- chaosllm: Fake LLM server for load testing and fault injection
- chaosweb: Fake web server for scraping pipeline resilience testing
- chaosllm_mcp: MCP server for analyzing ChaosLLM test results

Usage:
    from elspeth.testing import make_row, make_source_row, make_contract
    from elspeth.testing import make_success, make_error, make_gate_continue
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from elspeth.contracts import (
    ArtifactDescriptor,
    PipelineRow,
    SourceRow,
)
from elspeth.contracts.schema_contract import FieldContract, SchemaContract

if TYPE_CHECKING:
    from elspeth.contracts.contract_records import ContractAuditRecord
    from elspeth.contracts.enums import (
        CallStatus,
        CallType,
        NodeStateStatus,
        RowOutcome,
        RunStatus,
    )
    from elspeth.contracts.errors import TransformErrorReason, TransformSuccessReason
    from elspeth.contracts.events import (
        ExternalCallCompleted,
        PhaseCompleted,
        PhaseStarted,
        PipelinePhase,
        RunCompletionStatus,
        RunSummary,
        TokenCompleted,
        TransformCompleted,
    )
    from elspeth.contracts.plugin_context import (
        TransformErrorToken,
        ValidationErrorToken,
    )
    from elspeth.contracts.results import (
        FailureInfo,
        GateResult,
        RowResult,
    )
    from elspeth.engine.batch_adapter import ExceptionResult
    from elspeth.engine.orchestrator.types import (
        AggregationFlushResult,
        ExecutionCounters,
        PipelineConfig,
        RunResult,
    )
    from elspeth.engine.tokens import TokenInfo
    from elspeth.plugins.results import TransformResult


# =============================================================================
# Schema Contracts — The #1 source of test rewrite pain
# =============================================================================


def make_contract(
    data: dict[str, Any] | None = None,
    *,
    fields: dict[str, type] | None = None,
    mode: Literal["FIXED", "FLEXIBLE", "OBSERVED"] = "OBSERVED",
    locked: bool = True,
) -> SchemaContract:
    """Build a SchemaContract from data or explicit field types.

    Usage:
        contract = make_contract({"id": 1, "name": "Alice"})       # Infer from data
        contract = make_contract(fields={"id": int, "name": str})   # Explicit types
        contract = make_contract()                                   # Bare contract
    """
    if fields is not None:
        field_contracts = tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=python_type,
                required=True,
                source="declared",
            )
            for name, python_type in fields.items()
        )
    elif data is not None:
        field_contracts = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=type(value) if value is not None else object,
                required=False,
                source="inferred",
            )
            for key, value in data.items()
        )
    else:
        field_contracts = ()

    return SchemaContract(mode=mode, fields=field_contracts, locked=locked)


def make_field(
    name: str,
    python_type: type = object,
    *,
    original_name: str | None = None,
    required: bool = False,
    source: Literal["declared", "inferred"] = "inferred",
) -> FieldContract:
    """Build a single FieldContract."""
    return FieldContract(
        normalized_name=name,
        original_name=original_name or name,
        python_type=python_type,
        required=required,
        source=source,
    )


# =============================================================================
# PipelineRow / SourceRow
# =============================================================================


def make_row(
    data: dict[str, Any] | None = None,
    *,
    contract: SchemaContract | None = None,
    **kwargs: Any,
) -> PipelineRow:
    """Build a PipelineRow from a dict.

    Usage:
        row = make_row({"id": 1, "name": "Alice"})
        row = make_row(id=1, name="Alice")                 # kwargs shorthand
        row = make_row({"id": 1}, contract=my_contract)     # explicit contract
    """
    if data is None:
        data = kwargs
    if contract is None:
        contract = make_contract(data)
    return PipelineRow(data, contract)


def make_pipeline_row(data: dict[str, Any]) -> PipelineRow:
    """Create a PipelineRow with an OBSERVED schema contract for testing.

    Builds a contract where every key in ``data`` becomes an inferred,
    optional field typed as ``object``.  This is the standard test helper
    used across the entire test suite.

    Preserved for backward compatibility with existing 66 call sites.
    Prefer ``make_row()`` for new tests.

    Args:
        data: Row data as a plain dict.

    Returns:
        PipelineRow wrapping *data* with a locked OBSERVED contract.
    """
    fields = tuple(
        FieldContract(
            normalized_name=k,
            original_name=k,
            python_type=object,
            required=False,
            source="inferred",
        )
        for k in data
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return PipelineRow(data=data, contract=contract)


def make_source_row(
    data: dict[str, Any] | None = None,
    *,
    contract: SchemaContract | None = None,
    **kwargs: Any,
) -> SourceRow:
    """Build a valid SourceRow from a dict."""
    if data is None:
        data = kwargs
    if contract is None:
        contract = make_contract(data)
    return SourceRow.valid(data, contract=contract)


def make_source_row_quarantined(
    data: dict[str, Any],
    error: str = "validation_failed",
    destination: str = "quarantine",
) -> SourceRow:
    """Build a quarantined SourceRow."""
    return SourceRow.quarantined(row=data, error=error, destination=destination)


# =============================================================================
# TransformResult — 424 call sites, must build PipelineRow first today
# =============================================================================


def make_success(
    data: dict[str, Any] | PipelineRow | None = None,
    *,
    reason: TransformSuccessReason | None = None,
    context_after: dict[str, Any] | None = None,
    **kwargs: Any,
) -> TransformResult:
    """Build a TransformResult.success().

    Usage:
        result = make_success({"id": 1, "score": 0.9})
        result = make_success({"id": 1}, reason={"action": "classified"})
        result = make_success(row)  # From existing PipelineRow
        result = make_success({"id": 1}, context_after={"model": "gpt-4"})
    """
    from elspeth.plugins.results import TransformResult

    if data is None:
        data = kwargs or {"_empty": True}

    if isinstance(data, dict):
        data = make_row(data)

    extra: dict[str, Any] = {}
    if context_after is not None:
        extra["context_after"] = context_after

    return TransformResult.success(
        data,
        success_reason=reason or {"action": "test"},
        **extra,
    )


def make_success_multi(
    rows: list[dict[str, Any] | PipelineRow],
    *,
    reason: TransformSuccessReason | None = None,
    contract: SchemaContract | None = None,
) -> TransformResult:
    """Build a TransformResult.success_multi() from multiple rows.

    All rows share a single contract instance (required by success_multi).
    When *rows* contains dicts, a shared contract is built from the union
    of all keys.  Pre-built PipelineRow instances must already share
    the same contract identity.
    """
    from elspeth.plugins.results import TransformResult

    # Build a shared contract from the union of all dict keys if needed.
    if contract is None:
        all_keys: dict[str, type] = {}
        for r in rows:
            if isinstance(r, dict):
                all_keys.update(dict.fromkeys(r, object))
        contract = make_contract(all_keys) if all_keys else make_contract({})

    pipeline_rows = [make_row(r, contract=contract) if isinstance(r, dict) else r for r in rows]
    return TransformResult.success_multi(
        pipeline_rows,
        success_reason=reason or {"action": "test"},
    )


def make_error(
    reason: TransformErrorReason | str | None = None,
    *,
    retryable: bool = False,
) -> TransformResult:
    """Build a TransformResult.error().

    Usage:
        result = make_error("llm_timeout")
        result = make_error({"reason": "bad_json", "raw": "..."}, retryable=True)
    """
    from elspeth.plugins.results import TransformResult

    if isinstance(reason, str):
        error_reason: TransformErrorReason = {"reason": reason}  # type: ignore[typeddict-item]
    elif reason is None:
        error_reason = {"reason": "test_error"}
    else:
        error_reason = reason
    return TransformResult.error(
        error_reason,
        retryable=retryable,
    )


# =============================================================================
# GateResult
# =============================================================================


def make_gate_continue(
    data: dict[str, Any] | PipelineRow,
    *,
    contract: SchemaContract | None = None,
) -> GateResult:
    """Build a GateResult that continues the pipeline."""
    from elspeth.contracts.results import GateResult
    from elspeth.contracts.routing import RoutingAction

    if isinstance(data, PipelineRow):
        row_dict = data.to_dict()
        contract = contract or data.contract
    else:
        row = make_row(data, contract=contract)
        row_dict = row.to_dict()
        contract = row.contract
    return GateResult(row=row_dict, action=RoutingAction.continue_(), contract=contract)


def make_gate_route(
    data: dict[str, Any] | PipelineRow,
    sink: str,
    *,
    contract: SchemaContract | None = None,
) -> GateResult:
    """Build a GateResult that routes to a named sink."""
    from elspeth.contracts.results import GateResult
    from elspeth.contracts.routing import RoutingAction

    if isinstance(data, PipelineRow):
        row_dict = data.to_dict()
        contract = contract or data.contract
    else:
        row = make_row(data, contract=contract)
        row_dict = row.to_dict()
        contract = row.contract
    return GateResult(row=row_dict, action=RoutingAction.route(sink), contract=contract)


def make_gate_fork(
    data: dict[str, Any] | PipelineRow,
    paths: list[str],
    *,
    contract: SchemaContract | None = None,
) -> GateResult:
    """Build a GateResult that forks to multiple paths."""
    from elspeth.contracts.results import GateResult
    from elspeth.contracts.routing import RoutingAction

    if isinstance(data, PipelineRow):
        row_dict = data.to_dict()
        contract = contract or data.contract
    else:
        row = make_row(data, contract=contract)
        row_dict = row.to_dict()
        contract = row.contract
    return GateResult(row=row_dict, action=RoutingAction.fork_to_paths(paths), contract=contract)


# =============================================================================
# ArtifactDescriptor
# =============================================================================


def make_artifact(
    path: str = "memory://test",
    *,
    size_bytes: int = 0,
    content_hash: str = "test_hash",
) -> ArtifactDescriptor:
    """Build an ArtifactDescriptor for tests."""
    return ArtifactDescriptor.for_file(
        path=path,
        content_hash=content_hash,
        size_bytes=size_bytes,
    )


# =============================================================================
# TokenInfo
# =============================================================================


def make_token_info(
    row_id: str = "row-1",
    token_id: str | None = None,
    data: dict[str, Any] | None = None,
    branch_name: str | None = None,
) -> TokenInfo:
    """Build a TokenInfo for plugin context."""
    from elspeth.engine.tokens import TokenInfo

    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_row(data or {}),
        branch_name=branch_name,
    )


# =============================================================================
# Engine / Orchestrator Result Types
# =============================================================================


def make_run_result(
    *,
    run_id: str = "test-run",
    status: RunStatus | None = None,
    rows_processed: int = 10,
    rows_succeeded: int = 10,
    rows_failed: int = 0,
    rows_routed: int = 0,
    rows_quarantined: int = 0,
    rows_forked: int = 0,
    rows_coalesced: int = 0,
    rows_coalesce_failed: int = 0,
    rows_expanded: int = 0,
    rows_buffered: int = 0,
    routed_destinations: dict[str, int] | None = None,
) -> RunResult:
    """Build a RunResult with sensible defaults."""
    from elspeth.contracts.enums import RunStatus
    from elspeth.engine.orchestrator.types import RunResult

    return RunResult(
        run_id=run_id,
        status=status or RunStatus.COMPLETED,
        rows_processed=rows_processed,
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed=rows_routed,
        rows_quarantined=rows_quarantined,
        rows_forked=rows_forked,
        rows_coalesced=rows_coalesced,
        rows_coalesce_failed=rows_coalesce_failed,
        rows_expanded=rows_expanded,
        rows_buffered=rows_buffered,
        routed_destinations=routed_destinations or {},
    )


def make_flush_result(
    *,
    rows_succeeded: int = 5,
    rows_failed: int = 0,
    rows_routed: int = 0,
    rows_quarantined: int = 0,
    rows_coalesced: int = 0,
    rows_forked: int = 0,
    rows_expanded: int = 0,
    rows_buffered: int = 0,
    routed_destinations: dict[str, int] | None = None,
) -> AggregationFlushResult:
    """Build an AggregationFlushResult."""
    from elspeth.engine.orchestrator.types import AggregationFlushResult

    return AggregationFlushResult(
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed=rows_routed,
        rows_quarantined=rows_quarantined,
        rows_coalesced=rows_coalesced,
        rows_forked=rows_forked,
        rows_expanded=rows_expanded,
        rows_buffered=rows_buffered,
        routed_destinations=routed_destinations or {},
    )


def make_execution_counters(**overrides: int) -> ExecutionCounters:
    """Build ExecutionCounters with optional overrides."""
    from dataclasses import fields

    from elspeth.engine.orchestrator.types import ExecutionCounters

    valid_fields = {f.name for f in fields(ExecutionCounters)}
    invalid = set(overrides) - valid_fields
    if invalid:
        raise TypeError(f"Invalid ExecutionCounters fields: {sorted(invalid)}. Valid fields: {sorted(valid_fields)}")
    counters = ExecutionCounters()
    for key, value in overrides.items():
        setattr(counters, key, value)
    return counters


def make_row_result(
    data: dict[str, Any] | None = None,
    *,
    outcome: RowOutcome | None = None,
    sink_name: str | None = None,
    error: Any | None = None,
) -> RowResult:
    """Build a RowResult (final row outcome).

    COMPLETED, ROUTED, and COALESCED outcomes require sink_name
    (enforced by RowResult.__post_init__).
    Defaults to "default" when not explicitly provided for these outcomes.
    """
    from elspeth.contracts.enums import RowOutcome
    from elspeth.contracts.results import RowResult

    resolved_outcome = outcome or RowOutcome.COMPLETED
    # Sink-targeting outcomes require sink_name — default for test convenience
    _SINK_OUTCOMES = {RowOutcome.COMPLETED, RowOutcome.ROUTED, RowOutcome.COALESCED}
    resolved_sink_name = sink_name
    if resolved_outcome in _SINK_OUTCOMES and resolved_sink_name is None:
        resolved_sink_name = "default"

    token = make_token_info()
    return RowResult(
        token=token,
        final_data=make_pipeline_row(data) if data is not None else make_pipeline_row({"_result": True}),
        outcome=resolved_outcome,
        sink_name=resolved_sink_name,
        error=error,
    )


def make_failure_info(
    exception_type: str = "ValueError",
    message: str = "test failure",
    *,
    attempts: int = 1,
    last_error: str | None = None,
) -> FailureInfo:
    """Build a FailureInfo for error scenarios."""
    from elspeth.contracts.results import FailureInfo

    return FailureInfo(
        exception_type=exception_type,
        message=message,
        attempts=attempts,
        last_error=last_error or message,
    )


def make_exception_result(
    exc: BaseException | None = None,
    tb: str = "Traceback (test)",
) -> ExceptionResult:
    """Build ExceptionResult (wraps exceptions from worker threads)."""
    from elspeth.engine.batch_adapter import ExceptionResult

    return ExceptionResult(
        exception=exc or ValueError("test exception"),
        traceback=tb,
    )


def make_pipeline_config(
    source: Any = None,
    transforms: list[Any] | None = None,
    sinks: dict[str, Any] | None = None,
    **overrides: Any,
) -> PipelineConfig:
    """Build PipelineConfig with sensible defaults."""
    from elspeth.engine.orchestrator.types import PipelineConfig

    kwargs: dict[str, Any] = {
        "source": source,
        "transforms": transforms or [],
        "sinks": sinks or {},
    }
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


# =============================================================================
# Telemetry Events
# =============================================================================


def make_phase_started(
    phase: PipelinePhase | None = None,
    action: str | None = None,
    *,
    target: str | None = None,
) -> PhaseStarted:
    """Build PhaseStarted event."""
    from elspeth.contracts.events import PhaseAction, PhaseStarted, PipelinePhase

    return PhaseStarted(
        phase=phase or PipelinePhase.PROCESS,
        action=PhaseAction(action) if action else PhaseAction.PROCESSING,
        target=target,
    )


def make_phase_completed(
    phase: PipelinePhase | None = None,
    duration_seconds: float = 1.5,
) -> PhaseCompleted:
    """Build PhaseCompleted event."""
    from elspeth.contracts.events import PhaseCompleted, PipelinePhase

    return PhaseCompleted(
        phase=phase or PipelinePhase.PROCESS,
        duration_seconds=duration_seconds,
    )


def make_run_summary(
    *,
    run_id: str = "test-run",
    status: RunCompletionStatus | None = None,
    total_rows: int = 10,
    succeeded: int = 10,
    failed: int = 0,
    quarantined: int = 0,
    duration_seconds: float = 1.5,
    exit_code: int = 0,
    routed: int = 0,
    routed_destinations: tuple[tuple[str, int], ...] = (),
) -> RunSummary:
    """Build RunSummary event."""
    from elspeth.contracts.events import RunCompletionStatus, RunSummary

    return RunSummary(
        run_id=run_id,
        status=status or RunCompletionStatus.COMPLETED,
        total_rows=total_rows,
        succeeded=succeeded,
        failed=failed,
        quarantined=quarantined,
        duration_seconds=duration_seconds,
        exit_code=exit_code,
        routed=routed,
        routed_destinations=routed_destinations,
    )


def make_external_call_completed(
    *,
    call_type: CallType | None = None,
    provider: str = "azure",
    status: CallStatus | None = None,
    latency_ms: float = 150.0,
    run_id: str = "test-run",
    state_id: str | None = "state-123",
    operation_id: str | None = None,
) -> ExternalCallCompleted:
    """Build ExternalCallCompleted event."""
    from datetime import UTC, datetime

    from elspeth.contracts.enums import CallStatus, CallType
    from elspeth.contracts.events import ExternalCallCompleted

    return ExternalCallCompleted(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        call_type=call_type or CallType.LLM,
        provider=provider,
        status=status or CallStatus.SUCCESS,
        latency_ms=latency_ms,
        state_id=state_id,
        operation_id=operation_id,
    )


def make_transform_completed(
    *,
    row_id: str = "row-1",
    token_id: str = "tok-1",
    node_id: str = "transform-1",
    plugin_name: str = "test-transform",
    status: NodeStateStatus | None = None,
    duration_ms: float = 10.0,
    run_id: str = "test-run",
    input_hash: str = "hash_in",
    output_hash: str = "hash_out",
) -> TransformCompleted:
    """Build TransformCompleted event."""
    from datetime import UTC, datetime

    from elspeth.contracts.enums import NodeStateStatus
    from elspeth.contracts.events import TransformCompleted

    return TransformCompleted(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        node_id=node_id,
        plugin_name=plugin_name,
        status=status or NodeStateStatus.COMPLETED,
        duration_ms=duration_ms,
        input_hash=input_hash,
        output_hash=output_hash,
    )


def make_token_completed(
    *,
    row_id: str = "row-1",
    token_id: str = "tok-1",
    outcome: RowOutcome | None = None,
    sink_name: str | None = "default",
    run_id: str = "test-run",
) -> TokenCompleted:
    """Build TokenCompleted event."""
    from datetime import UTC, datetime

    from elspeth.contracts.enums import RowOutcome
    from elspeth.contracts.events import TokenCompleted

    return TokenCompleted(
        timestamp=datetime.now(UTC),
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        outcome=outcome or RowOutcome.COMPLETED,
        sink_name=sink_name,
    )


# =============================================================================
# Structural Dict Factories
# =============================================================================


def make_success_reason(
    action: str = "processed",
    *,
    fields_modified: list[str] | None = None,
    fields_added: list[str] | None = None,
    fields_removed: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a success_reason dict matching TransformSuccessReason shape."""
    reason: dict[str, Any] = {"action": action}
    if fields_modified:
        reason["fields_modified"] = fields_modified
    if fields_added:
        reason["fields_added"] = fields_added
    if fields_removed:
        reason["fields_removed"] = fields_removed
    if metadata:
        reason["metadata"] = metadata
    return reason


def make_error_reason(
    reason: str = "test_error",
    *,
    error: str | None = None,
    field: str | None = None,
    retryable: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build an error_reason dict matching TransformErrorReason shape."""
    result: dict[str, Any] = {"reason": reason}
    if error:
        result["error"] = error
    if field:
        result["field"] = field
    result.update(extra)
    return result


# =============================================================================
# Audit Trail Record Types
# =============================================================================


def make_contract_audit_record(
    data: dict[str, Any] | None = None,
    *,
    mode: Literal["FIXED", "FLEXIBLE", "OBSERVED"] = "OBSERVED",
) -> ContractAuditRecord:
    """Build ContractAuditRecord for contract serialization testing."""
    from elspeth.contracts.contract_records import ContractAuditRecord, FieldAuditRecord

    if data is not None:
        fields = tuple(
            FieldAuditRecord(
                normalized_name=k,
                original_name=k,
                python_type=type(v).__name__,
                required=False,
                source="inferred",
            )
            for k, v in data.items()
        )
    else:
        fields = ()
    return ContractAuditRecord(
        mode=mode,
        locked=True,
        version_hash="test-hash",
        fields=fields,
    )


def make_validation_error_token(
    row_id: str = "row-1",
    node_id: str = "source-node",
    error_id: str = "err-1",
    destination: str = "quarantine",
) -> ValidationErrorToken:
    """Build ValidationErrorToken."""
    from elspeth.contracts.plugin_context import ValidationErrorToken

    return ValidationErrorToken(
        row_id=row_id,
        node_id=node_id,
        error_id=error_id,
        destination=destination,
    )


def make_transform_error_token(
    token_id: str = "tok-1",
    transform_id: str = "transform-1",
    error_id: str = "err-1",
    destination: str = "quarantine",
) -> TransformErrorToken:
    """Build TransformErrorToken."""
    from elspeth.contracts.plugin_context import TransformErrorToken

    return TransformErrorToken(
        token_id=token_id,
        transform_id=transform_id,
        error_id=error_id,
        destination=destination,
    )
