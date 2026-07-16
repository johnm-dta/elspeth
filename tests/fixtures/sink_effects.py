"""Effect-capable test doubles for caller-level recovery proofs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)


@dataclass(slots=True)
class DuplicateObservableTarget:
    publication_count: int = 0
    effect_id: str | None = None
    descriptor: ArtifactDescriptor | None = None


class DuplicateObservableSink:
    """Naive publisher whose coordinator must prevent duplicate calls."""

    name = "duplicate-observable"
    declared_required_fields = frozenset()
    _on_write_failure = None

    class input_schema:
        @classmethod
        def model_validate(cls, value: object) -> object:
            return value

    def __init__(self, target: DuplicateObservableTarget) -> None:
        self.external_target = target
        self.node_id: str | None = None
        self.config = {"path": "duplicate-observable.jsonl"}

    def _reset_diversion_log(self) -> None:
        return None

    def _get_diversions(self) -> tuple[object, ...]:
        return ()

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del request, ctx
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference="no-inspection-required:v1",
            evidence=MappingProxyType({}),
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        rows = [deep_thaw(member.row) for member in request.effect_input.members]  # type: ignore[union-attr]
        payload_hash = stable_hash(rows)
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///tmp/duplicate-observable.jsonl",
            content_hash=payload_hash,
            size_bytes=len(rows),
        )
        plan_hash = stable_hash(
            {
                "descriptor": {
                    "content_hash": descriptor.content_hash,
                    "path": descriptor.path_or_uri,
                    "size": descriptor.size_bytes,
                },
                "effect_id": request.effect_id,
                "schema": "duplicate-observable-plan-v1",
            }
        )
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=request.input_kind,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=plan_hash,
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": request.inspection.reference},
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        assert plan.expected_descriptor is not None
        self.external_target.publication_count += 1
        self.external_target.effect_id = plan.effect_id
        self.external_target.descriptor = plan.expected_descriptor
        return SinkEffectCommitResult(
            descriptor=plan.expected_descriptor,
            evidence={"effect_id": plan.effect_id},
            accepted_ordinals=(0,),
            diverted_ordinals=(),
        )

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        if self.external_target.effect_id is None:
            return SinkEffectReconcileResult.not_applied(evidence={"target": "absent"})
        if self.external_target.effect_id == plan.effect_id and self.external_target.descriptor == plan.expected_descriptor:
            assert self.external_target.descriptor is not None
            return SinkEffectReconcileResult.applied(
                self.external_target.descriptor,
                evidence={"effect_id": plan.effect_id},
            )
        return SinkEffectReconcileResult.unknown(evidence={"target": "divergent"})


__all__ = ["DuplicateObservableSink", "DuplicateObservableTarget"]
