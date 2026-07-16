"""Caller-level coordinator for durable, recoverable sink effects.

The coordinator is the only production boundary allowed to invoke an
effect-capable sink's inspection, reconciliation, or commit methods.  Every
possibly observable call is bracketed by the durable sink-effect ledger and
final publication is committed atomically with the pipeline audit outcome.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol, cast

from elspeth.contracts.audit import SinkEffect, SinkEffectAttempt
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectAttemptAction,
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
    SinkEffectAttemptState,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectFinalizationMember,
    SinkEffectFinalizationResult,
    SinkEffectFinalizeRequest,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectLease,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
    SinkEffectReconcileResult,
    SinkEffectReservationRequest,
    SinkEffectState,
)
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.sink_effect_attempt_results import (
    SinkEffectReturnedResult,
    decode_sink_effect_returned_result,
    encode_sink_effect_returned_result,
)
from elspeth.core.landscape.factory import RecorderFactory


class SinkEffectExecutionSeam(StrEnum):
    """Deterministic crash seams around the externally observable commit."""

    BEFORE_EFFECT = "before_effect"
    AFTER_EFFECT_BEFORE_RETURN = "after_effect_before_return"
    AFTER_RETURN_BEFORE_FINALIZE = "after_return_before_finalize"
    AFTER_FINALIZE_BEFORE_RESPONSE = "after_finalize_before_response"


class SinkEffectInjectedFault(RuntimeError):
    """Test-only crash signal raised by a configured execution seam."""

    def __init__(self, seam: SinkEffectExecutionSeam) -> None:
        self.seam = seam
        super().__init__(f"injected sink-effect fault at {seam.value}")


class SinkEffectUnknownError(RuntimeError):
    """Raised when reconciliation cannot prove whether publication happened."""

    def __init__(self, effect_id: str) -> None:
        self.effect_id = effect_id
        super().__init__(f"sink effect {effect_id} reconciliation is UNKNOWN; refusing to commit")


class SinkEffectPredecessorPending(RuntimeError):
    """Raised when a replacing-target predecessor has not finalized yet."""


class _SinkEffectAdapter(Protocol):
    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection: ...

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan: ...

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult: ...

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult: ...


@dataclass(frozen=True, slots=True)
class SinkEffectExecutionRequest:
    """Complete caller-owned input needed to execute and finalize one effect."""

    reservation: SinkEffectReservationRequest
    effect_input: object
    finalization_members: Sequence[SinkEffectFinalizationMember]

    def __post_init__(self) -> None:
        from elspeth.contracts.sink_effects import SinkEffectAuditExportSnapshotInput, SinkEffectPipelineMembersInput

        if type(self.reservation) is not SinkEffectReservationRequest:
            raise TypeError("reservation must be exact SinkEffectReservationRequest")
        if not isinstance(self.effect_input, (SinkEffectPipelineMembersInput, SinkEffectAuditExportSnapshotInput)):
            raise TypeError("effect_input must be a closed sink-effect input value")
        if self.effect_input.input_kind is not self.reservation.input_kind:
            raise ValueError("effect_input kind must equal reservation input kind")
        members = tuple(self.finalization_members)
        if any(type(member) is not SinkEffectFinalizationMember for member in members):
            raise TypeError("finalization_members must contain exact SinkEffectFinalizationMember values")
        if tuple(member.ordinal for member in members) != tuple(sorted({member.ordinal for member in members})):
            raise ValueError("finalization member ordinals must be unique and ascending")
        object.__setattr__(self, "finalization_members", members)


class SinkEffectCoordinator:
    """Drive reservation, intent, external I/O, recovery, and finalization."""

    def __init__(
        self,
        *,
        factory: RecorderFactory,
        worker_id: str,
        lease_ttl: timedelta = timedelta(minutes=5),
        fault_hook: Callable[[SinkEffectExecutionSeam], None] | None = None,
    ) -> None:
        if not isinstance(factory, RecorderFactory):
            raise TypeError("factory must be RecorderFactory")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        if type(lease_ttl) is not timedelta or lease_ttl <= timedelta(0):
            raise ValueError("lease_ttl must be a positive timedelta")
        self._factory = factory
        self._effects = factory.execution.sink_effects
        self._worker_id = worker_id
        self._lease_ttl = lease_ttl
        self._fault_hook = fault_hook

    def execute(
        self,
        request: SinkEffectExecutionRequest,
        sink: _SinkEffectAdapter,
    ) -> SinkEffectFinalizationResult:
        if type(request) is not SinkEffectExecutionRequest:
            raise TypeError("request must be exact SinkEffectExecutionRequest")

        self._persist_pipeline_member_payloads(request.effect_input)
        reservation = self._effects.reserve(request.reservation)
        effect_ids = (*reservation.finalized_effect_ids, *reservation.open_effect_ids)
        if reservation.new_effect is not None:
            effect_ids = (*effect_ids, reservation.new_effect.effect_id)
        effect_ids = tuple(dict.fromkeys(effect_ids))
        if not effect_ids:
            raise LandscapeRecordError("sink effect reservation returned no effect partition")
        effects = sorted(
            (self._require_effect(effect_id) for effect_id in effect_ids),
            key=lambda effect: (
                effect.stream_id is None,
                -1 if effect.stream_sequence is None else effect.stream_sequence,
                effect.effect_id,
            ),
        )
        results: list[SinkEffectFinalizationResult] = []
        for effect in effects:
            refreshed = self._require_effect(effect.effect_id)
            if refreshed.state is SinkEffectState.FINALIZED:
                results.append(self._load_finalized(refreshed.effect_id))
                continue
            partition_request = self._request_for_effect(refreshed, request)
            results.append(self._execute_effect(refreshed, partition_request, sink))
        return results[-1]

    def _execute_effect(
        self,
        effect: SinkEffect,
        request: SinkEffectExecutionRequest,
        sink: _SinkEffectAdapter,
    ) -> SinkEffectFinalizationResult:
        self._require_predecessor(effect)
        ctx = self._context(effect)
        plan = self._prepare(effect, request, sink, ctx)
        effect = self._require_effect(effect.effect_id)

        if plan.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION:
            result = self._finalize_no_publication(effect, plan, request)
            self._fault(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE)
            return result

        lease = self._lease(effect)
        self._close_abandoned_attempts(
            effect.effect_id,
            actions=(SinkEffectAttemptAction.COMMIT, SinkEffectAttemptAction.RECONCILE),
            recovery_lease=lease,
        )
        returned_commit = self._returned_attempt(
            effect.effect_id,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash=self._commit_request_hash(plan),
        )
        if returned_commit is not None:
            commit_result, commit_attempt = returned_commit
            if not isinstance(commit_result, SinkEffectCommitResult):
                raise LandscapeRecordError("durable commit attempt decoded to the wrong result type")
            result = self._finalize(
                effect_id=effect.effect_id,
                request=request,
                lease=lease,
                descriptor=commit_result.descriptor,
                evidence=commit_result.evidence,
                accepted_ordinals=tuple(commit_result.accepted_ordinals),
                diverted_ordinals=tuple(commit_result.diverted_ordinals),
                attempt_id=commit_attempt.attempt_id,
                evidence_kind="returned",
                reconcile_kind=None,
            )
            self._fault(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE)
            return result
        reconciliation, reconcile_attempt_id = self._reconcile(plan, sink, ctx, lease)
        if reconciliation.kind is SinkEffectReconcileKind.UNKNOWN:
            raise SinkEffectUnknownError(effect.effect_id)
        if reconciliation.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR:
            assert reconciliation.descriptor is not None
            result = self._finalize(
                effect_id=effect.effect_id,
                request=request,
                lease=lease,
                descriptor=reconciliation.descriptor,
                evidence=reconciliation.evidence,
                accepted_ordinals=tuple(member.ordinal for member in request.finalization_members),
                diverted_ordinals=(),
                attempt_id=reconcile_attempt_id,
                evidence_kind="reconciled",
                reconcile_kind=reconciliation.kind,
            )
            self._fault(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE)
            return result

        commit, commit_attempt_id = self._commit(plan, sink, ctx, lease)
        self._fault(SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE)
        result = self._finalize(
            effect_id=effect.effect_id,
            request=request,
            lease=lease,
            descriptor=commit.descriptor,
            evidence=commit.evidence,
            accepted_ordinals=tuple(commit.accepted_ordinals),
            diverted_ordinals=tuple(commit.diverted_ordinals),
            attempt_id=commit_attempt_id,
            evidence_kind="returned",
            reconcile_kind=None,
        )
        self._fault(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE)
        return result

    def _request_for_effect(
        self,
        effect: SinkEffect,
        request: SinkEffectExecutionRequest,
    ) -> SinkEffectExecutionRequest:
        if not isinstance(request.effect_input, SinkEffectPipelineMembersInput):
            return request
        durable_members = self._effects.get_members(effect.effect_id)
        caller_by_token = {member.token_id: member for member in request.effect_input.members}
        finalization_by_token = {
            member.token_id: finalization
            for member, finalization in zip(request.effect_input.members, request.finalization_members, strict=True)
        }
        current_members: list[SinkEffectMember] = []
        current_finalization: list[SinkEffectFinalizationMember] = []
        for durable in durable_members:
            caller = caller_by_token.get(durable.token_id)
            finalization = finalization_by_token.get(durable.token_id)
            if caller is None or finalization is None:
                raise LandscapeRecordError(f"open sink effect {effect.effect_id} cannot be recovered from a partial caller partition")
            current_members.append(
                replace(
                    caller,
                    ordinal=durable.ordinal,
                    member_effect_id=durable.member_effect_id,
                )
            )
            current_finalization.append(replace(finalization, ordinal=durable.ordinal))
        members = tuple(current_members)
        return SinkEffectExecutionRequest(
            reservation=request.reservation,
            effect_input=SinkEffectPipelineMembersInput(
                members=members,
                target_snapshot_members=self._target_snapshot_members(
                    effect,
                    members,
                    known_members=caller_by_token,
                ),
            ),
            finalization_members=tuple(current_finalization),
        )

    def _target_snapshot_members(
        self,
        effect: SinkEffect,
        current_members: tuple[SinkEffectMember, ...],
        *,
        known_members: Mapping[str, SinkEffectMember],
    ) -> tuple[SinkEffectMember, ...]:
        chain: list[SinkEffect] = []
        predecessor_id = effect.predecessor_effect_id
        seen = {effect.effect_id}
        while predecessor_id is not None:
            if predecessor_id in seen or len(chain) >= 256:
                raise LandscapeRecordError("sink effect predecessor chain is cyclic or exceeds 256 effects")
            seen.add(predecessor_id)
            predecessor = self._require_effect(predecessor_id)
            if predecessor.state is not SinkEffectState.FINALIZED:
                raise SinkEffectPredecessorPending(f"sink effect {effect.effect_id} is waiting for predecessor {predecessor.effect_id}")
            chain.append(predecessor)
            predecessor_id = predecessor.predecessor_effect_id
        chain.reverse()

        snapshot: list[SinkEffectMember] = []
        for predecessor in chain:
            for durable in self._effects.get_members(predecessor.effect_id):
                known = known_members.get(durable.token_id)
                member = replace(known, member_effect_id=durable.member_effect_id) if known is not None else self._hydrate_member(durable)
                snapshot.append(replace(member, ordinal=len(snapshot)))
        current_start = len(snapshot)
        snapshot.extend(replace(member, ordinal=current_start + ordinal) for ordinal, member in enumerate(current_members))
        return tuple(snapshot)

    def _persist_pipeline_member_payloads(self, effect_input: object) -> None:
        if not isinstance(effect_input, SinkEffectPipelineMembersInput):
            return
        store = self._factory.payload_store
        if store is None:
            return
        for member in effect_input.members:
            content = canonical_json(deep_thaw(member.row)).encode("utf-8")
            content_hash = store.store(content)
            if content_hash != member.payload_hash:
                raise LandscapeRecordError("sink effect payload store returned a divergent member content hash")

    def _hydrate_member(self, durable: object) -> SinkEffectMember:
        from elspeth.contracts.audit import SinkEffectMemberRecord

        if not isinstance(durable, SinkEffectMemberRecord):
            raise TypeError("durable must be SinkEffectMemberRecord")
        store = self._factory.payload_store
        if store is None:
            raise LandscapeRecordError("replacing sink effect recovery requires the configured payload store")
        content = store.retrieve(durable.payload_hash)
        try:
            row = json.loads(content)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LandscapeRecordError("sink effect member payload is not canonical JSON") from exc
        if type(row) is not dict or canonical_json(row).encode("utf-8") != content:
            raise LandscapeRecordError("sink effect member payload is not an exact canonical object")
        return SinkEffectMember(
            ordinal=durable.ordinal,
            token_id=durable.token_id,
            row_id=durable.row_id,
            ingest_sequence=durable.ingest_sequence,
            lineage_json=durable.lineage_json,
            lineage_hash=durable.lineage_hash,
            payload_hash=durable.payload_hash,
            row=row,
            member_effect_id=durable.member_effect_id,
        )

    def _prepare(
        self,
        effect: SinkEffect,
        request: SinkEffectExecutionRequest,
        sink: _SinkEffectAdapter,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        if effect.state is not SinkEffectState.RESERVED:
            return self._load_plan(effect)
        predecessor = self._predecessor_descriptor(effect)
        inspection_request = SinkEffectInspectionRequest(
            effect_id=effect.effect_id,
            target=effect.target_json,
            predecessor_descriptor=predecessor,
        )
        request_hash = stable_hash(
            {
                "effect_id": effect.effect_id,
                "predecessor": None if predecessor is None else predecessor.content_hash,
                "schema": "sink-effect-inspection-request-v1",
                "target": effect.target_json,
            }
        )
        self._close_abandoned_attempts(effect.effect_id, actions=(SinkEffectAttemptAction.INSPECT,))
        returned_inspection = self._returned_attempt(
            effect.effect_id,
            action=SinkEffectAttemptAction.INSPECT,
            request_hash=request_hash,
        )
        if returned_inspection is not None:
            inspection_result, _attempt = returned_inspection
            if not isinstance(inspection_result, SinkEffectInspection):
                raise LandscapeRecordError("durable inspect attempt decoded to the wrong result type")
            inspection = inspection_result
        else:
            inspection_attempt = self._effects.begin_attempt(
                SinkEffectAttemptRequest(
                    effect_id=effect.effect_id,
                    member_ordinal=None,
                    generation=effect.generation,
                    action=SinkEffectAttemptAction.INSPECT,
                    request_hash=request_hash,
                )
            )
            started = time.monotonic()
            try:
                inspection = sink.inspect_effect(inspection_request, ctx)
            except BaseException:
                self._effects.mark_response_lost(inspection_attempt.attempt_id)
                raise
            self._effects.record_attempt_result(
                SinkEffectAttemptResult(
                    attempt_id=inspection_attempt.attempt_id,
                    evidence=encode_sink_effect_returned_result(inspection),
                    latency_ms=(time.monotonic() - started) * 1_000,
                )
            )
        prepare_request = SinkEffectPrepareRequest(
            effect_id=effect.effect_id,
            effect_input=request.effect_input,  # type: ignore[arg-type]
            inspection=inspection,
        )
        plan = sink.prepare_effect(prepare_request, ctx)
        prepare_request.validate_plan(plan)
        self._effects.complete_plan(effect.effect_id, plan)
        return plan

    @staticmethod
    def _load_plan(effect: SinkEffect) -> SinkEffectPlan:
        if effect.plan_json is None:
            raise LandscapeRecordError("prepared sink effect is missing its durable plan")
        try:
            payload = json.loads(effect.plan_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LandscapeRecordError("prepared sink effect has invalid durable plan JSON") from exc
        if type(payload) is not dict:
            raise LandscapeRecordError("prepared sink effect durable plan must be an object")
        descriptor_payload = payload.get("expected_descriptor")
        descriptor: ArtifactDescriptor | None
        if descriptor_payload is None:
            descriptor = None
        elif type(descriptor_payload) is dict:
            descriptor = ArtifactDescriptor(
                artifact_type=descriptor_payload["artifact_type"],
                path_or_uri=descriptor_payload["path_or_uri"],
                content_hash=descriptor_payload["content_hash"],
                size_bytes=descriptor_payload["size_bytes"],
                metadata=descriptor_payload.get("metadata"),
            )
        else:
            raise LandscapeRecordError("prepared sink effect durable descriptor is invalid")
        try:
            return SinkEffectPlan(
                effect_id=payload["effect_id"],
                protocol_version=payload["protocol_version"],
                input_kind=SinkEffectInputKind(payload["input_kind"]),
                descriptor_mode=SinkEffectDescriptorMode(payload["descriptor_mode"]),
                inspection_mode=SinkEffectInspectionMode(payload["inspection_mode"]),
                target=payload["target"],
                plan_hash=payload["plan_hash"],
                payload_hash=payload["payload_hash"],
                expected_descriptor=descriptor,
                safe_evidence=payload["safe_evidence"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LandscapeRecordError("prepared sink effect durable plan is incomplete or divergent") from exc

    def _lease(self, effect: SinkEffect) -> SinkEffectLease:
        if effect.state is SinkEffectState.PREPARED:
            return self._effects.acquire_lease(effect.effect_id, owner=self._worker_id, ttl=self._lease_ttl)
        if effect.state is not SinkEffectState.IN_FLIGHT or effect.lease_expires_at is None:
            raise LandscapeRecordError(f"sink effect cannot execute from state {effect.state.value!r}")
        expires_at = (
            effect.lease_expires_at.replace(tzinfo=UTC)
            if effect.lease_expires_at.tzinfo is None
            else effect.lease_expires_at.astimezone(UTC)
        )
        if effect.lease_owner == self._worker_id and expires_at >= datetime.now(UTC):
            return self._effects.acquire_lease(effect.effect_id, owner=self._worker_id, ttl=self._lease_ttl)
        if expires_at < datetime.now(UTC):
            return self._effects.takeover_expired(effect.effect_id, owner=self._worker_id, ttl=self._lease_ttl)
        raise SinkEffectPredecessorPending(f"sink effect {effect.effect_id} has a live lease owned by another worker")

    def _reconcile(
        self,
        plan: SinkEffectPlan,
        sink: _SinkEffectAdapter,
        ctx: RestrictedSinkEffectContext,
        lease: SinkEffectLease,
    ) -> tuple[SinkEffectReconcileResult, str]:
        request_hash = self._reconcile_request_hash(plan)
        latest_lost_commit = self._latest_attempt(
            plan.effect_id,
            action=SinkEffectAttemptAction.COMMIT,
            state=SinkEffectAttemptState.RESPONSE_LOST,
        )
        returned = self._returned_attempt(
            plan.effect_id,
            action=SinkEffectAttemptAction.RECONCILE,
            request_hash=request_hash,
            started_after=None if latest_lost_commit is None else latest_lost_commit.started_at,
        )
        if returned is not None:
            result, attempt = returned
            if not isinstance(result, SinkEffectReconcileResult):
                raise LandscapeRecordError("durable reconcile attempt decoded to the wrong result type")
            return result, attempt.attempt_id
        attempt = self._effects.begin_attempt(
            SinkEffectAttemptRequest(
                effect_id=plan.effect_id,
                member_ordinal=None,
                generation=lease.generation,
                action=SinkEffectAttemptAction.RECONCILE,
                request_hash=request_hash,
            )
        )
        started = time.monotonic()
        try:
            result = sink.reconcile_effect(plan, ctx)
        except BaseException:
            self._effects.mark_response_lost(attempt.attempt_id)
            raise
        self._effects.record_attempt_result(
            SinkEffectAttemptResult(
                attempt_id=attempt.attempt_id,
                evidence=encode_sink_effect_returned_result(result),
                latency_ms=(time.monotonic() - started) * 1_000,
            )
        )
        return result, attempt.attempt_id

    def _commit(
        self,
        plan: SinkEffectPlan,
        sink: _SinkEffectAdapter,
        ctx: RestrictedSinkEffectContext,
        lease: SinkEffectLease,
    ) -> tuple[SinkEffectCommitResult, str]:
        request_hash = self._commit_request_hash(plan)
        attempt = self._effects.begin_attempt(
            SinkEffectAttemptRequest(
                effect_id=plan.effect_id,
                member_ordinal=None,
                generation=lease.generation,
                action=SinkEffectAttemptAction.COMMIT,
                request_hash=request_hash,
            )
        )
        self._fault(SinkEffectExecutionSeam.BEFORE_EFFECT)
        started = time.monotonic()
        try:
            result = sink.commit_effect(plan, ctx)
            self._fault(SinkEffectExecutionSeam.AFTER_EFFECT_BEFORE_RETURN)
        except BaseException:
            self._effects.mark_response_lost(attempt.attempt_id)
            raise
        self._effects.record_attempt_result(
            SinkEffectAttemptResult(
                attempt_id=attempt.attempt_id,
                evidence=encode_sink_effect_returned_result(result),
                latency_ms=(time.monotonic() - started) * 1_000,
            )
        )
        return result, attempt.attempt_id

    @staticmethod
    def _reconcile_request_hash(plan: SinkEffectPlan) -> str:
        return stable_hash({"effect_id": plan.effect_id, "plan_hash": plan.plan_hash, "schema": "sink-effect-reconcile-request-v1"})

    @staticmethod
    def _commit_request_hash(plan: SinkEffectPlan) -> str:
        return stable_hash({"effect_id": plan.effect_id, "plan_hash": plan.plan_hash, "schema": "sink-effect-commit-request-v1"})

    def _close_abandoned_attempts(
        self,
        effect_id: str,
        *,
        actions: tuple[SinkEffectAttemptAction, ...],
        recovery_lease: SinkEffectLease | None = None,
    ) -> None:
        for attempt in self._effects.get_attempts(effect_id):
            if attempt.action in actions and attempt.state is SinkEffectAttemptState.INTENT:
                self._effects.mark_response_lost(attempt.attempt_id, recovery_lease=recovery_lease)

    def _returned_attempt(
        self,
        effect_id: str,
        *,
        action: SinkEffectAttemptAction,
        request_hash: str,
        started_after: datetime | None = None,
    ) -> tuple[SinkEffectReturnedResult, SinkEffectAttempt] | None:
        winners = [
            attempt
            for attempt in self._effects.get_attempts(effect_id)
            if attempt.action is action
            and attempt.request_hash == request_hash
            and attempt.state is SinkEffectAttemptState.RETURNED
            and (started_after is None or self._utc(attempt.started_at) > self._utc(started_after))
        ]
        if not winners:
            return None
        winner = winners[-1]
        if winner.evidence_json is None:
            raise LandscapeRecordError("returned sink effect attempt is missing its durable result")
        result = decode_sink_effect_returned_result(action, winner.evidence_json)
        for attempt in winners[:-1]:
            if attempt.evidence_json is None or decode_sink_effect_returned_result(action, attempt.evidence_json) != result:
                raise LandscapeRecordError("same sink effect request has divergent durable returned results")
        return result, winner

    def _latest_attempt(
        self,
        effect_id: str,
        *,
        action: SinkEffectAttemptAction,
        state: SinkEffectAttemptState,
    ) -> SinkEffectAttempt | None:
        attempts = [attempt for attempt in self._effects.get_attempts(effect_id) if attempt.action is action and attempt.state is state]
        return attempts[-1] if attempts else None

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    def _finalize(
        self,
        *,
        effect_id: str,
        request: SinkEffectExecutionRequest,
        lease: SinkEffectLease,
        descriptor: ArtifactDescriptor,
        evidence: Mapping[str, object],
        accepted_ordinals: tuple[int, ...],
        diverted_ordinals: tuple[int, ...],
        attempt_id: str,
        evidence_kind: str,
        reconcile_kind: SinkEffectReconcileKind | None,
    ) -> SinkEffectFinalizationResult:
        by_ordinal = {member.ordinal: member for member in request.finalization_members}
        try:
            members = tuple(by_ordinal[ordinal] for ordinal in accepted_ordinals)
        except KeyError as exc:
            raise LandscapeRecordError(f"sink result accepted unknown member ordinal {exc.args[0]}") from exc
        return self._effects.finalize(
            SinkEffectFinalizeRequest(
                effect_id=effect_id,
                lease_owner=lease.owner,
                generation=lease.generation,
                descriptor=descriptor,
                publication_performed=True,
                publication_evidence_kind=evidence_kind,  # type: ignore[arg-type]
                accepted_ordinals=accepted_ordinals,
                diverted_ordinals=diverted_ordinals,
                evidence=evidence,
                members=members,
                attempt_id=attempt_id,
                reconcile_kind=reconcile_kind,
            )
        )

    def _finalize_no_publication(
        self,
        effect: SinkEffect,
        plan: SinkEffectPlan,
        request: SinkEffectExecutionRequest,
    ) -> SinkEffectFinalizationResult:
        if plan.expected_descriptor is None:
            raise LandscapeRecordError("no-publication effect requires its precomputed descriptor")
        publication_kind = plan.safe_evidence.get("publication_kind")
        if publication_kind == "inherited":
            evidence_kind: Literal["inherited", "virtual"] = "inherited"
        elif publication_kind == "virtual":
            evidence_kind = "virtual"
        else:
            raise LandscapeRecordError("no-publication effect requires inherited or virtual publication evidence")
        accepted = tuple(member.ordinal for member in request.finalization_members)
        return self._effects.finalize(
            SinkEffectFinalizeRequest(
                effect_id=effect.effect_id,
                lease_owner=None,
                generation=effect.generation,
                descriptor=plan.expected_descriptor,
                publication_performed=False,
                publication_evidence_kind=evidence_kind,
                accepted_ordinals=accepted,
                diverted_ordinals=(),
                evidence=plan.safe_evidence,
                members=request.finalization_members,
            )
        )

    def _context(self, effect: SinkEffect) -> RestrictedSinkEffectContext:
        run = self._factory.run_lifecycle.get_run(effect.run_id)
        if run is None:
            raise LandscapeRecordError(f"sink effect run {effect.run_id!r} does not exist")
        operation = next(
            (item for item in self._factory.execution.get_operations_for_run(effect.run_id) if item.sink_effect_id == effect.effect_id),
            None,
        )
        if operation is None:
            raise LandscapeRecordError("sink effect stable operation does not exist")
        return RestrictedSinkEffectContext(
            run_id=effect.run_id,
            run_started_at=run.started_at,
            operation_id=operation.operation_id,
            sink_node_id=effect.sink_node_id,
        )

    def _require_predecessor(self, effect: SinkEffect) -> None:
        if effect.predecessor_effect_id is None:
            return
        predecessor = self._effects.get_effect(effect.predecessor_effect_id)
        if predecessor is None or predecessor.state is not SinkEffectState.FINALIZED:
            raise SinkEffectPredecessorPending(f"sink effect {effect.effect_id} is waiting for predecessor {effect.predecessor_effect_id}")

    def _predecessor_descriptor(self, effect: SinkEffect) -> ArtifactDescriptor | None:
        if effect.predecessor_effect_id is None:
            return None
        predecessor = self._effects.get_effect(effect.predecessor_effect_id)
        if predecessor is None:
            raise LandscapeRecordError("sink effect predecessor disappeared")
        artifact = next(
            (
                item
                for item in self._factory.execution.get_artifacts(effect.run_id, sink_node_id=effect.sink_node_id)
                if item.sink_effect_id == predecessor.effect_id
            ),
            None,
        )
        if artifact is None:
            raise LandscapeRecordError("finalized sink effect predecessor is missing its artifact")
        if artifact.artifact_type not in {"file", "database", "webhook"}:
            raise LandscapeRecordError("finalized sink effect predecessor has an invalid artifact type")
        return ArtifactDescriptor(
            artifact_type=cast(Literal["file", "database", "webhook"], artifact.artifact_type),
            path_or_uri=artifact.path_or_uri,
            content_hash=artifact.content_hash,
            size_bytes=artifact.size_bytes,
        )

    def _load_finalized(self, effect_id: str) -> SinkEffectFinalizationResult:
        effect = self._require_effect(effect_id)
        if effect.state is not SinkEffectState.FINALIZED:
            raise LandscapeRecordError("reservation classified a non-finalized effect as finalized")
        artifacts = self._factory.execution.get_artifacts(effect.run_id, sink_node_id=effect.sink_node_id)
        artifact = next((item for item in artifacts if item.sink_effect_id == effect.effect_id), None)
        if artifact is None:
            raise LandscapeRecordError("finalized sink effect is missing its artifact")
        return SinkEffectFinalizationResult(effect=effect, artifact=artifact, state_ids=(), outcome_ids=())

    def _require_effect(self, effect_id: str) -> SinkEffect:
        effect = self._effects.get_effect(effect_id)
        if effect is None:
            raise LandscapeRecordError(f"sink effect {effect_id!r} does not exist")
        return effect

    def _fault(self, seam: SinkEffectExecutionSeam) -> None:
        if self._fault_hook is not None:
            self._fault_hook(seam)


__all__ = [
    "SinkEffectCoordinator",
    "SinkEffectExecutionRequest",
    "SinkEffectExecutionSeam",
    "SinkEffectInjectedFault",
    "SinkEffectPredecessorPending",
    "SinkEffectUnknownError",
]
