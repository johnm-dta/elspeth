# tests/fixtures/plugins.py
"""Consolidated test plugins — one canonical definition per plugin.

Eliminates the 3 duplicate ListSource/CollectSink definitions from v1.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from hashlib import sha256
from typing import Any, ClassVar

from pydantic import ConfigDict

from elspeth.contracts import (
    SINK_EFFECT_PROTOCOL_VERSION,
    ArtifactDescriptor,
    CallType,
    Determinism,
    PluginSchema,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
    SourceRow,
)
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, _TestSinkBase, _TestSourceBase


class _EngineTestSchema(PluginSchema):
    """Dynamic schema for engine-level test plugins."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class ListSource(_TestSourceBase):
    """Source that yields rows from a list.

    Usage:
        source = ListSource([{"value": 1}, {"value": 2}])
        source = ListSource([{"value": 1}], name="my_source")
    """

    determinism = Determinism.IO_READ
    output_schema = _EngineTestSchema

    def __init__(self, data: list[dict[str, Any]], name: str = "list_source", on_success: str = "default") -> None:
        super().__init__()
        self._data = data
        self.name = name
        self.on_success = on_success

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        yield from self.wrap_rows(self._data)

    def close(self) -> None:
        pass


class CollectSink(_TestSinkBase):
    """Sink that collects results into a list.

    Usage:
        sink = CollectSink()
        sink = CollectSink("output_sink")
        sink = CollectSink("output", node_id="sink_node_123")
    """

    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.FILESYSTEM
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})

    def __init__(
        self,
        name: str = "collect",
        *,
        node_id: str | None = None,
        divert_ordinals: frozenset[int] = frozenset(),
    ) -> None:
        super().__init__()
        self.name = name
        self.node_id = node_id
        self.results: list[dict[str, Any]] = []
        self._artifact_counter = 0
        self._configured_divert_ordinals = divert_ordinals
        self._effect_plans: dict[str, SinkEffectPlan] = {}
        self._effect_rows: dict[str, tuple[dict[str, Any], ...]] = {}
        self._effect_ordinals: dict[str, tuple[int, ...]] = {}
        self._effect_diverted_ordinals: dict[str, tuple[int, ...]] = {}
        self._effect_commits: dict[str, SinkEffectCommitResult] = {}

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode:
        del cls, config, purpose
        return ResolvedSinkEffectMode("write")

    @property
    def rows_written(self) -> list[dict[str, Any]]:
        """Alias for results — some tests use this name."""
        return self.results

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        self.results.extend(rows)
        self._artifact_counter += 1
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(rows)),
                content_hash=f"{self._artifact_counter:064x}",
            )
        )

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del ctx
        committed = self._effect_commits.get(request.effect_id)
        evidence: dict[str, object] = {"state": "not_applied"}
        if committed is not None:
            evidence = {"state": "committed", "content_hash": committed.descriptor.content_hash}
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.INSPECTED,
            reference=request.target,
            evidence=evidence,
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if not isinstance(request.effect_input, SinkEffectPipelineMembersInput):
            raise TypeError("CollectSink only supports pipeline member effects")

        rows = tuple(deep_thaw(member.row) for member in request.effect_input.members)
        if any(not isinstance(row, dict) for row in rows):  # pragma: no cover - contract guarantees mappings
            raise TypeError("CollectSink effect rows must thaw to dictionaries")
        ordinals = tuple(member.ordinal for member in request.effect_input.members)
        diverted = tuple(ordinal for ordinal in ordinals if ordinal in self._configured_divert_ordinals)
        if not self._configured_divert_ordinals.issubset(ordinals):
            raise ValueError("CollectSink configured diversion ordinal is outside the effect batch")
        accepted = tuple(ordinal for ordinal in ordinals if ordinal not in self._configured_divert_ordinals)
        row_by_ordinal = dict(zip(ordinals, rows, strict=True))
        accepted_rows = tuple(row_by_ordinal[ordinal] for ordinal in accepted)
        diversion_reason = "collect sink rejected configured row"
        self._diversion_log = [
            RowDiversion(row_index=ordinal, reason=diversion_reason, row_data=dict(row_by_ordinal[ordinal])) for ordinal in diverted
        ]
        payload = canonical_json(accepted_rows).encode("utf-8")
        payload_hash = sha256(payload).hexdigest()
        descriptor = ArtifactDescriptor.for_file(
            path=request.inspection.reference,
            size_bytes=len(payload),
            content_hash=payload_hash,
        )
        plan = SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=request.inspection.reference,
            plan_hash=stable_hash(
                {
                    "effect_id": request.effect_id,
                    "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
                    "payload_hash": payload_hash,
                    "target": request.inspection.reference,
                }
            ),
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence={
                "accepted_ordinals": list(accepted),
                "diversion_attribution": [
                    {
                        "error_hash": sha256(diversion_reason.encode("utf-8")).hexdigest()[:16],
                        "ordinal": ordinal,
                        "reason_hash": stable_hash({"diversion_reason": diversion_reason}),
                    }
                    for ordinal in diverted
                ],
                "diverted_ordinals": list(diverted),
                "sink_kind": "collect",
            },
        )
        request.validate_plan(plan)

        existing = self._effect_plans.get(request.effect_id)
        if existing is not None and existing != plan:
            raise ValueError("CollectSink effect_id was prepared with a different plan")
        self._effect_plans[request.effect_id] = plan
        self._effect_rows[request.effect_id] = accepted_rows
        self._effect_ordinals[request.effect_id] = accepted
        self._effect_diverted_ordinals[request.effect_id] = diverted
        return plan

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        return tuple(self._diversion_log)

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        del ctx
        prepared = self._effect_plans.get(plan.effect_id)
        if prepared is None or prepared != plan:
            raise ValueError("CollectSink can only commit the exact prepared plan")
        committed = self._effect_commits.get(plan.effect_id)
        if committed is not None:
            return committed
        if plan.expected_descriptor is None:  # pragma: no cover - PRECOMPUTED contract guarantees this
            raise ValueError("CollectSink prepared plan is missing its descriptor")

        self.results.extend(self._effect_rows[plan.effect_id])
        self._artifact_counter += 1
        result = SinkEffectCommitResult(
            descriptor=plan.expected_descriptor,
            evidence={"state": "committed"},
            accepted_ordinals=self._effect_ordinals[plan.effect_id],
            diverted_ordinals=self._effect_diverted_ordinals[plan.effect_id],
        )
        self._effect_commits[plan.effect_id] = result
        return result

    def reconcile_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        del ctx
        prepared = self._effect_plans.get(plan.effect_id)
        if prepared is not None and prepared != plan:
            raise ValueError("CollectSink can only reconcile the exact prepared plan")
        committed = self._effect_commits.get(plan.effect_id)
        if committed is None:
            return SinkEffectReconcileResult.not_applied(evidence={"state": "not_applied"})
        return SinkEffectReconcileResult.applied(
            committed.descriptor,
            evidence={"state": "committed"},
        )

    def close(self) -> None:
        pass


class FailingSink(CollectSink):
    """Effect-capable sink whose publication always raises RuntimeError.

    For testing error handling in orchestrator, executors, and outcome recording.

    Usage:
        sink = FailingSink()
        sink = FailingSink("broken_sink")
        sink = FailingSink(error_message="Custom error")
    """

    def __init__(
        self,
        name: str = "failing_sink",
        *,
        node_id: str | None = None,
        error_message: str = "Sink write failed",
    ) -> None:
        super().__init__(name, node_id=node_id)
        self._error_message = error_message

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        raise RuntimeError(self._error_message)

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        del plan, ctx
        raise RuntimeError(self._error_message)

    def close(self) -> None:
        pass


class DivertingSink(CollectSink):
    """Sink that diverts the first configured rows and writes the rest.

    This exercises the production ``SinkExecutor`` discard branch: when no
    failsink is configured for this sink, every returned ``RowDiversion`` is
    recorded as ``(FAILURE, SINK_DISCARDED)`` with the discard sentinel sink.
    """

    name = "diverting_sink"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        divert_count: int | None = None,
    ) -> None:
        options = dict(config or {})
        configured_name = name if name is not None else options.get("name", self.name)
        super().__init__(str(configured_name))
        configured_divert_count = divert_count if divert_count is not None else options.get("divert_count")
        if configured_divert_count is None:
            self._divert_count: int | None = None
        else:
            self._divert_count = int(configured_divert_count)

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        if not isinstance(request.effect_input, SinkEffectPipelineMembersInput):
            raise TypeError("DivertingSink only supports pipeline member effects")
        ordinals = tuple(member.ordinal for member in request.effect_input.members)
        limit = self._divert_count if self._divert_count is not None else len(ordinals)
        self._configured_divert_ordinals = frozenset(ordinals[:limit])
        return super().prepare_effect(request, ctx)

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        del ctx
        row_list = list(rows)
        limit = self._divert_count if self._divert_count is not None else len(row_list)
        diversions = tuple(
            RowDiversion(
                row_index=idx,
                reason="diverting_sink: forced divert for ADR-019 test",
                row_data=dict(row),
            )
            for idx, row in enumerate(row_list)
            if idx < limit
        )
        primary_rows = [dict(row) for idx, row in enumerate(row_list) if idx >= limit]
        self.results.extend(primary_rows)
        self._artifact_counter += 1
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(primary_rows)),
                content_hash=f"{self._artifact_counter:064x}",
            ),
            diversions=diversions,
        )

    def close(self) -> None:
        pass


class FailingSource(ListSource):
    """Source whose load() always raises RuntimeError.

    For testing error handling during source loading (orchestrator cleanup,
    export partial semantics, etc.).

    Usage:
        source = FailingSource()
        source = FailingSource(error_message="Custom load failure")
    """

    determinism = Determinism.IO_READ

    def __init__(
        self,
        *,
        name: str = "failing_source",
        error_message: str = "Source failed intentionally",
    ) -> None:
        super().__init__(data=[], name=name)
        self._error_message = error_message

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        raise RuntimeError(self._error_message)


class PassTransform(BaseTransform):
    """Identity transform — passes rows through unchanged."""

    name = "pass_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})


class FailTransform(BaseTransform):
    """Transform that always returns an error result."""

    name = "fail_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        error_reason: str = "always_fail",
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_error is not None:
            self.on_error = on_error
        self._error_reason = error_reason

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.error({"reason": "deliberate_failure", "error": self._error_reason})


class ConditionalErrorTransform(BaseTransform):
    """Transform that errors on rows where 'fail' key is truthy."""

    name = "conditional_error"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        if row["fail"]:
            return TransformResult.error({"reason": "test_error"})
        return TransformResult.success(row, success_reason={"action": "test"})


class CountingTransform(BaseTransform):
    """Transform that counts invocations (for retry testing)."""

    name = "counting_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self.call_count = 0

    def process(self, row: Any, ctx: Any) -> TransformResult:
        self.call_count += 1
        return TransformResult.success(row, success_reason={"action": "counted"})


class SlowTransform(BaseTransform):
    """Transform with configurable delay (for timeout testing)."""

    name = "slow_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        delay_seconds: float = 0.1,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self._delay = delay_seconds

    def process(self, row: Any, ctx: Any) -> TransformResult:
        import time

        time.sleep(self._delay)
        return TransformResult.success(row, success_reason={"action": "delayed"})


class ErrorOnNthTransform(BaseTransform):
    """Transform that errors on the Nth invocation (for retry integration)."""

    name = "error_on_nth"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    on_error = "discard"

    def __init__(
        self,
        error_on: int = 1,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error
        self._error_on = error_on
        self._call_count = 0

    def process(self, row: Any, ctx: Any) -> TransformResult:
        self._call_count += 1
        if self._call_count == self._error_on:
            return TransformResult.error({"reason": "simulated_failure", "error": f"nth_error_{self._error_on}"}, retryable=True)
        return TransformResult.success(row, success_reason={"action": "passed"})


class CallRecordingTransform(BaseTransform):
    """Transform that records a single HTTP operation_call via ctx.record_call().

    Used by F1 Cell-2 regression: proves re-driven transforms RE-FIRE their
    recorded calls (at-least-once) and that the re-fired call's node_state
    carries resume_checkpoint_id (attributability invariant, ADDENDUM 2.C).

    Each invocation of process() records one HTTP call with a synthetic
    request payload.  The call is state-parented (calls.state_id → node_states)
    so it inherits the resume_checkpoint_id from its parent node_state.
    """

    name = "call_recording_transform"
    determinism = Determinism.NON_DETERMINISTIC  # has external call side-effects
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        *,
        name: str | None = None,
        input_connection: str | None = None,
        on_success: str | None = None,
        on_error: str | None = None,
    ) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        if name is not None:
            self.name = name
        if input_connection is not None:
            self.input = input_connection
        if on_success is not None:
            self.on_success = on_success
        if on_error is not None:
            self.on_error = on_error

    def process(self, row: Any, ctx: Any) -> TransformResult:
        from elspeth.contracts.enums import CallStatus, CallType

        # Record a synthetic HTTP call so the audit trail contains a calls row
        # linked to this node_state (state-parented: calls.state_id is set).
        # On resume, the re-drive writes a new node_state at attempt=max+1 with
        # resume_checkpoint_id set, and records another calls row linked there.
        ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://test.internal/probe", "row_value": row["value"] if "value" in row else 0},
            response_data={"status": 200},
        )
        return TransformResult.success(row, success_reason={"action": "call_recorded"})
