"""Effect-capable test doubles for caller-level recovery proofs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

from elspeth.contracts.diversion import RowDiversion
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
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


class PartitioningObservableSink(DuplicateObservableSink):
    """Effect double that diverts rows carrying ``divert=True`` during prepare."""

    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})

    def __init__(
        self,
        target: DuplicateObservableTarget,
        *,
        name: str,
        fail_prepare_once: bool = False,
        divert_rows: bool = True,
    ) -> None:
        super().__init__(target)
        self.name = name
        self._fail_prepare_once = fail_prepare_once
        self._divert_rows = divert_rows
        self._diversions: tuple[RowDiversion, ...] = ()
        self.config = {"path": f"{name}.jsonl"}

    def _reset_diversion_log(self) -> None:
        self._diversions = ()

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        return self._diversions

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if self._fail_prepare_once:
            self._fail_prepare_once = False
            raise RuntimeError("injected failure between primary and failsink publication")
        effect_input = request.effect_input
        assert effect_input.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS
        rows = [deep_thaw(member.row) for member in effect_input.members]  # type: ignore[union-attr]
        diverted = tuple(index for index, row in enumerate(rows) if self._divert_rows and bool(row.get("divert", False)))
        accepted = tuple(index for index in range(len(rows)) if index not in diverted)
        self._diversions = tuple(
            RowDiversion(row_index=index, reason="injected diversion", row_data=dict(rows[index])) for index in diverted
        )
        accepted_rows = [rows[index] for index in accepted]
        if accepted:
            payload_hash = stable_hash(accepted_rows)
            descriptor_mode = SinkEffectDescriptorMode.PRECOMPUTED
            publication_kind = "returned"
            size_bytes = len(accepted_rows)
        else:
            payload_hash = sha256(b"").hexdigest()
            descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
            publication_kind = "virtual"
            size_bytes = 0
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri=f"file:///tmp/{self.name}.jsonl",
            content_hash=payload_hash,
            size_bytes=size_bytes,
        )
        safe_evidence = {
            "accepted_ordinals": list(accepted),
            "diversion_attribution": [
                {
                    "error_hash": sha256(b"injected diversion").hexdigest()[:16],
                    "ordinal": ordinal,
                    "reason_hash": stable_hash({"diversion_reason": "injected diversion"}),
                }
                for ordinal in diverted
            ],
            "diverted_ordinals": list(diverted),
            "publication_kind": publication_kind,
        }
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=request.input_kind,
            descriptor_mode=descriptor_mode,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=stable_hash(
                {
                    "descriptor": descriptor.content_hash,
                    "effect_id": request.effect_id,
                    "evidence": safe_evidence,
                    "schema": "partitioning-observable-plan-v1",
                }
            ),
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence=safe_evidence,
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        assert plan.expected_descriptor is not None
        accepted = tuple(int(value) for value in plan.safe_evidence["accepted_ordinals"])  # type: ignore[union-attr]
        diverted = tuple(int(value) for value in plan.safe_evidence["diverted_ordinals"])  # type: ignore[union-attr]
        self.external_target.publication_count += 1
        self.external_target.effect_id = plan.effect_id
        self.external_target.descriptor = plan.expected_descriptor
        return SinkEffectCommitResult(
            descriptor=plan.expected_descriptor,
            evidence={"effect_id": plan.effect_id},
            accepted_ordinals=accepted,
            diverted_ordinals=diverted,
        )


__all__ = ["DuplicateObservableSink", "DuplicateObservableTarget", "PartitioningObservableSink"]
