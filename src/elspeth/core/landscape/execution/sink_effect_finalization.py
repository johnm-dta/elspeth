"""Atomic, globally ordered sink-effect finalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Final

from sqlalchemy import Row, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import NodeStateStatus
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SinkEffectAttemptAction,
    SinkEffectAttemptState,
    SinkEffectDescriptorMode,
    SinkEffectFinalizationMember,
    SinkEffectFinalizationResult,
    SinkEffectFinalizeRequest,
    SinkEffectReconcileKind,
    SinkEffectState,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.data_flow.outcomes import TokenOutcomeRepository
from elspeth.core.landscape.data_flow.ownership import RowTokenOwnership
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.artifacts import ArtifactRepository
from elspeth.core.landscape.execution.node_states import NodeStateRepository
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    NodeStateLoader,
    RoutingEventLoader,
    SinkEffectLoader,
    TokenOutcomeLoader,
)
from elspeth.core.landscape.schema import (
    artifacts_table,
    node_states_table,
    operations_table,
    sink_effect_attempts_table,
    sink_effect_members_table,
    sink_effect_streams_table,
    sink_effects_table,
    token_outcomes_table,
)

_MAX_WITNESS_RESTARTS: Final = 3


def _descriptor_payload(descriptor: ArtifactDescriptor) -> dict[str, object]:
    return {
        "artifact_type": descriptor.artifact_type,
        "content_hash": descriptor.content_hash,
        "metadata": None if descriptor.metadata is None else deep_thaw(descriptor.metadata),
        "path_or_uri": descriptor.path_or_uri,
        "size_bytes": descriptor.size_bytes,
    }


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _OptimisticWitness:
    effect_id: str
    token_ids: tuple[str, ...]
    state_ids_by_ordinal: tuple[tuple[int, str], ...]
    linked_effect_ids: tuple[str, ...]


class _WitnessChanged(Exception):
    pass


class SinkEffectFinalization:
    """Own the single effect/artifact/state/outcome commit boundary."""

    def __init__(self, db: LandscapeDB, ops: DatabaseOps, *, effect_loader: SinkEffectLoader) -> None:
        self._db = db
        self._effect_loader = effect_loader
        self._artifact_loader = ArtifactLoader()
        self._node_states = NodeStateRepository(
            db,
            ops,
            node_state_loader=NodeStateLoader(),
            routing_event_loader=RoutingEventLoader(),
        )
        ownership = RowTokenOwnership(ops)
        self._outcomes = TokenOutcomeRepository(
            db,
            ops,
            token_outcome_loader=TokenOutcomeLoader(),
            ownership=ownership,
        )
        self._artifacts = ArtifactRepository(ops, artifact_loader=self._artifact_loader)

    def finalize(self, request: SinkEffectFinalizeRequest) -> SinkEffectFinalizationResult:
        if type(request) is not SinkEffectFinalizeRequest:
            raise TypeError("request must be exact SinkEffectFinalizeRequest")
        self._validate_outcome_shapes(request)
        for restart in range(_MAX_WITNESS_RESTARTS):
            optimistic = self._resolve_optimistic_witness(request)
            try:
                with self._db.write_connection() as conn:
                    result = self._finalize_on(conn, request, optimistic)
            except _WitnessChanged:
                if restart + 1 == _MAX_WITNESS_RESTARTS:
                    raise LandscapeRecordError(
                        "sink effect current state witness changed repeatedly; refusing unbounded lock-order restart"
                    ) from None
                continue
            except LandscapeRecordError:
                raise
            except SQLAlchemyError as exc:
                raise LandscapeRecordError(
                    f"sink effect finalization failed — database rejected atomic audit write: {type(exc).__name__}"
                ) from exc
            self._after_commit(request.effect_id)
            return result
        raise AssertionError("unreachable witness restart state")

    def _validate_outcome_shapes(self, request: SinkEffectFinalizeRequest) -> None:
        for member in request.members:
            self._outcomes._validate_outcome_fields(
                member.outcome,
                member.path,
                sink_name=member.sink_name,
                batch_id=member.batch_id,
                fork_group_id=member.fork_group_id,
                join_group_id=member.join_group_id,
                expand_group_id=member.expand_group_id,
                error_hash=member.error_hash,
            )

    def _resolve_optimistic_witness(self, request: SinkEffectFinalizeRequest) -> _OptimisticWitness:
        with self._db.read_only_connection() as conn:
            effect = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == request.effect_id)).fetchone()
            if effect is None:
                raise LandscapeRecordError(f"sink effect {request.effect_id!r} does not exist")
            members = self._load_members(conn, request.effect_id)
            self._validate_partition(request, members)
            state_ids = (
                self._resolve_current_states(conn, request, effect, members) if effect.state != SinkEffectState.FINALIZED.value else ()
            )
            linked = conn.execute(
                select(sink_effects_table.c.effect_id).where(
                    (sink_effects_table.c.effect_id == request.effect_id)
                    | (sink_effects_table.c.primary_effect_id == request.effect_id)
                    | (sink_effects_table.c.effect_id == effect.primary_effect_id)
                    | (sink_effects_table.c.effect_id == effect.predecessor_effect_id)
                )
            ).fetchall()
        return _OptimisticWitness(
            effect_id=request.effect_id,
            token_ids=tuple(sorted(str(member.token_id) for member in members)),
            state_ids_by_ordinal=state_ids,
            linked_effect_ids=tuple(sorted(str(row.effect_id) for row in linked)),
        )

    def _finalize_on(
        self,
        conn: Connection,
        request: SinkEffectFinalizeRequest,
        optimistic: _OptimisticWitness,
    ) -> SinkEffectFinalizationResult:
        optimistic_effect = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == request.effect_id)).fetchone()
        if optimistic_effect is None:
            raise LandscapeRecordError("sink effect disappeared before finalization")
        if optimistic_effect.state == SinkEffectState.FINALIZED.value:
            locked = self._lock_stream_and_effects(conn, optimistic_effect, optimistic.linked_effect_ids)
            effect = locked[request.effect_id]
            return self._load_finalized_winner(conn, request, effect)

        refs = tuple(TokenRef(token_id=token_id, run_id=str(optimistic_effect.run_id)) for token_id in optimistic.token_ids)
        self._outcomes.lock_token_outcome_dependencies(refs, conn=conn)
        self._after_token_locks(self._backend_pid(conn), optimistic.token_ids)

        members = self._load_members(conn, request.effect_id)
        self._validate_partition(request, members)
        current_state_ids = self._resolve_current_states(conn, request, optimistic_effect, members)
        if current_state_ids != optimistic.state_ids_by_ordinal:
            raise _WitnessChanged

        # Task 1a is the one shared state prelock/completion primitive. It
        # deduplicates and locks state IDs ascending before its first read.
        completions = tuple(
            (
                dict(current_state_ids)[member.ordinal],
                deep_thaw(member.output_data),
                member.duration_ms,
            )
            for member in request.members
        )
        self._node_states.complete_node_states_completed_many(completions, conn=conn)
        self._after_state_locks(self._backend_pid(conn), tuple(sorted(state_id for _ordinal, state_id in current_state_ids)))

        locked = self._lock_stream_and_effects(conn, optimistic_effect, optimistic.linked_effect_ids)
        locked_effect = locked.get(request.effect_id)
        if locked_effect is None:
            raise LandscapeRecordError("sink effect disappeared while acquiring finalization locks")
        effect = locked_effect
        if effect.state == SinkEffectState.FINALIZED.value:
            raise _WitnessChanged
        self._validate_effect_authority(conn, request, effect, locked)
        descriptor_hash = self._validate_plan_and_descriptor(request, effect)
        evidence_hash = sha256(canonical_json(deep_thaw(request.evidence)).encode("utf-8")).hexdigest()

        artifact = self._artifacts.register_artifact(
            run_id=str(effect.run_id),
            sink_node_id=str(effect.sink_node_id),
            artifact_type=request.descriptor.artifact_type,
            path=request.descriptor.path_or_uri,
            content_hash=request.descriptor.content_hash,
            size_bytes=request.descriptor.size_bytes,
            sink_effect_id=request.effect_id,
            artifact_id=str(effect.artifact_id),
            idempotency_key=str(effect.artifact_idempotency_key),
            publication_performed=request.publication_performed,
            publication_evidence_kind=request.publication_evidence_kind,
            conn=conn,
        )

        operation = conn.execute(
            select(operations_table).where(operations_table.c.sink_effect_id == request.effect_id).with_for_update(of=operations_table)
        ).fetchone()
        if operation is None:
            raise LandscapeRecordError("sink effect stable operation is missing")
        self._validate_attempt(conn, request, effect, operation)
        if operation.status != "open":
            raise LandscapeRecordError(f"sink effect operation is not open during first finalization: {operation.status!r}")
        completed_at = now()
        operation_result = conn.execute(
            operations_table.update()
            .where(operations_table.c.operation_id == operation.operation_id, operations_table.c.status == "open")
            .values(
                status="completed",
                completed_at=completed_at,
                duration_ms=request.operation_duration_ms,
                error_message=None,
                output_data_hash=stable_hash(
                    {
                        "accepted_ordinals": list(request.accepted_ordinals),
                        "artifact_id": artifact.artifact_id,
                        "descriptor_hash": descriptor_hash,
                        "diverted_ordinals": list(request.diverted_ordinals),
                        "effect_id": request.effect_id,
                    }
                ),
            )
        )
        if operation_result.rowcount != 1:
            raise LandscapeRecordError("sink effect operation completion CAS lost")

        outcome_ids: list[str] = []
        member_by_ordinal = {int(member.ordinal): member for member in members}
        state_ids = dict(current_state_ids)
        for finalization_member in request.members:
            durable_member = member_by_ordinal[finalization_member.ordinal]
            outcome_ids.append(
                self._outcomes.record_token_outcome(
                    TokenRef(token_id=str(durable_member.token_id), run_id=str(effect.run_id)),
                    finalization_member.outcome,
                    finalization_member.path,
                    sink_name=finalization_member.sink_name,
                    sink_node_id=str(effect.sink_node_id),
                    artifact_id=artifact.artifact_id,
                    batch_id=finalization_member.batch_id,
                    fork_group_id=finalization_member.fork_group_id,
                    join_group_id=finalization_member.join_group_id,
                    expand_group_id=finalization_member.expand_group_id,
                    error_hash=finalization_member.error_hash,
                    context=finalization_member.context,
                    conn=conn,
                    dependencies_prelocked=True,
                )
            )

        for ordinal in request.accepted_ordinals:
            conn.execute(
                sink_effect_members_table.update()
                .where(sink_effect_members_table.c.effect_id == request.effect_id, sink_effect_members_table.c.ordinal == ordinal)
                .values(
                    prepared_disposition="accepted",
                    member_state=SinkEffectState.FINALIZED.value,
                    descriptor_hash=descriptor_hash,
                    evidence_hash=evidence_hash,
                )
            )
        for ordinal in request.diverted_ordinals:
            conn.execute(
                sink_effect_members_table.update()
                .where(sink_effect_members_table.c.effect_id == request.effect_id, sink_effect_members_table.c.ordinal == ordinal)
                .values(
                    prepared_disposition="diverted",
                    member_state=SinkEffectState.FINALIZED.value,
                    descriptor_hash=descriptor_hash,
                    evidence_hash=evidence_hash,
                )
            )

        self._advance_stream_head(conn, effect, descriptor_hash)
        reconcile_kind = request.reconcile_kind.value if request.reconcile_kind is not None else None
        updated = conn.execute(
            sink_effects_table.update()
            .where(
                sink_effects_table.c.effect_id == request.effect_id,
                sink_effects_table.c.state == effect.state,
                sink_effects_table.c.generation == request.generation,
            )
            .values(
                state=SinkEffectState.FINALIZED.value,
                reconcile_kind=reconcile_kind,
                reconcile_evidence_hash=evidence_hash if reconcile_kind is not None else None,
                result_descriptor_hash=descriptor_hash,
                publication_performed=request.publication_performed,
                publication_evidence_kind=request.publication_evidence_kind,
                lease_owner=None,
                lease_expires_at=None,
                lease_heartbeat_at=None,
                updated_at=completed_at,
                finalized_at=completed_at,
            )
        )
        if updated.rowcount != 1:
            raise LandscapeRecordError("sink effect finalization CAS lost")
        winner = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == request.effect_id)).one()
        return SinkEffectFinalizationResult(
            effect=self._effect_loader.load(winner),
            artifact=artifact,
            state_ids=tuple(state_ids[member.ordinal] for member in request.members),
            outcome_ids=tuple(outcome_ids),
        )

    @staticmethod
    def _load_members(conn: Connection, effect_id: str) -> list[Row[Any]]:
        return list(
            conn.execute(
                select(sink_effect_members_table)
                .where(sink_effect_members_table.c.effect_id == effect_id)
                .order_by(sink_effect_members_table.c.ordinal)
            ).fetchall()
        )

    @staticmethod
    def _validate_partition(request: SinkEffectFinalizeRequest, members: Sequence[Row[Any]]) -> None:
        durable_ordinals = tuple(int(member.ordinal) for member in members)
        requested_ordinals = tuple(request.accepted_ordinals) + tuple(request.diverted_ordinals)
        if requested_ordinals != tuple(sorted(durable_ordinals)):
            raise LandscapeRecordError("finalization accepted/diverted ordinals must exactly partition durable membership")
        expected = list(range(len(members)))
        if list(durable_ordinals) != expected:
            raise LandscapeRecordError("sink effect durable member ordinals are not dense")

    @staticmethod
    def _resolve_current_states(
        conn: Connection,
        request: SinkEffectFinalizeRequest,
        effect: Row[Any],
        members: Sequence[Row[Any]],
    ) -> tuple[tuple[int, str], ...]:
        member_by_ordinal = {int(member.ordinal): member for member in members}
        resolved: list[tuple[int, str]] = []
        for finalization_member in request.members:
            member = member_by_ordinal[finalization_member.ordinal]
            rows = conn.execute(
                select(node_states_table)
                .where(
                    node_states_table.c.run_id == effect.run_id,
                    node_states_table.c.node_id == effect.sink_node_id,
                    node_states_table.c.token_id == member.token_id,
                )
                .order_by(node_states_table.c.attempt.desc(), node_states_table.c.state_id.desc())
                .limit(2)
            ).fetchall()
            if not rows or rows[0].status != NodeStateStatus.OPEN.value:
                raise LandscapeRecordError(f"sink effect member ordinal {finalization_member.ordinal} has no current open state witness")
            current = rows[0]
            if current.input_hash != member.payload_hash:
                raise LandscapeRecordError("sink effect current state witness input does not match reserved member payload")
            resolved.append((finalization_member.ordinal, str(current.state_id)))
        return tuple(resolved)

    def _lock_stream_and_effects(
        self,
        conn: Connection,
        optimistic_effect: Row[Any],
        linked_effect_ids: tuple[str, ...],
    ) -> dict[str, Row[Any]]:
        if optimistic_effect.stream_id is not None:
            stream = conn.execute(
                select(sink_effect_streams_table)
                .where(sink_effect_streams_table.c.stream_id == optimistic_effect.stream_id)
                .with_for_update(of=sink_effect_streams_table)
            ).fetchone()
            if stream is None:
                raise LandscapeRecordError("sink effect stream disappeared during finalization")
        effect_ids = tuple(sorted(set(linked_effect_ids) | {str(optimistic_effect.effect_id)}))
        rows = conn.execute(
            select(sink_effects_table)
            .where(sink_effects_table.c.effect_id.in_(effect_ids))
            .order_by(sink_effects_table.c.effect_id)
            .with_for_update(of=sink_effects_table)
        ).fetchall()
        by_id = {str(row.effect_id): row for row in rows}
        if set(by_id) != set(effect_ids):
            raise LandscapeRecordError("linked sink effect set changed or disappeared during finalization")
        self._after_effect_locks(self._backend_pid(conn), effect_ids)
        return by_id

    def _validate_effect_authority(
        self,
        conn: Connection,
        request: SinkEffectFinalizeRequest,
        effect: Row[Any],
        linked: Mapping[str, Row[Any]],
    ) -> None:
        if effect.descriptor_mode == SinkEffectDescriptorMode.NO_PUBLICATION.value:
            if effect.state != SinkEffectState.PREPARED.value:
                raise LandscapeRecordError("no-publication effect must finalize directly from prepared state")
            if request.lease_owner is not None or request.generation != effect.generation:
                raise LandscapeRecordError("no-publication finalization must not claim lease ownership")
            if request.attempt_id is not None:
                raise LandscapeRecordError("no-publication finalization forbids an external attempt")
        else:
            if effect.state != SinkEffectState.IN_FLIGHT.value:
                raise LandscapeRecordError(f"sink effect cannot finalize from state {effect.state!r}")
            if effect.lease_owner != request.lease_owner:
                raise LandscapeRecordError("sink effect finalization has stale lease owner")
            if effect.generation != request.generation:
                raise LandscapeRecordError("sink effect finalization has stale generation")
            if effect.lease_expires_at is None or _utc(effect.lease_expires_at) < now():
                raise LandscapeRecordError("sink effect finalization lease has expired")
        if effect.primary_effect_id is not None:
            primary = linked.get(str(effect.primary_effect_id))
            if primary is None or primary.run_id != effect.run_id or primary.state != SinkEffectState.FINALIZED.value:
                raise LandscapeRecordError("failsink effect requires its same-run primary effect to be finalized")
        if effect.predecessor_effect_id is not None:
            predecessor = linked.get(str(effect.predecessor_effect_id))
            if predecessor is None or predecessor.state != SinkEffectState.FINALIZED.value:
                raise LandscapeRecordError("stream predecessor must be finalized before successor finalization")
        current = conn.execute(select(sink_effects_table.c.effect_id).where(sink_effects_table.c.effect_id == request.effect_id)).fetchone()
        if current is None:  # pragma: no cover - protected by row lock
            raise LandscapeRecordError("sink effect ownership witness disappeared")

    @staticmethod
    def _validate_plan_and_descriptor(request: SinkEffectFinalizeRequest, effect: Row[Any]) -> str:
        if effect.plan_json is None or effect.plan_hash is None or effect.descriptor_mode is None:
            raise LandscapeRecordError("sink effect finalization requires one complete immutable plan")
        try:
            plan = json.loads(effect.plan_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LandscapeRecordError("sink effect durable plan is not valid JSON") from exc
        if not isinstance(plan, dict):
            raise LandscapeRecordError("sink effect durable plan must be an object")
        exact_plan_fields = {
            "effect_id": effect.effect_id,
            "input_kind": effect.input_kind,
            "plan_hash": effect.plan_hash,
            "descriptor_mode": effect.descriptor_mode,
        }
        mismatches = [field for field, expected in exact_plan_fields.items() if plan.get(field) != expected]
        if mismatches:
            raise LandscapeRecordError("sink effect durable plan disagrees with ledger fields: " + ", ".join(mismatches))
        descriptor_payload = _descriptor_payload(request.descriptor)
        descriptor_hash = stable_hash(descriptor_payload)
        mode = SinkEffectDescriptorMode(effect.descriptor_mode)
        if mode in {SinkEffectDescriptorMode.PRECOMPUTED, SinkEffectDescriptorMode.NO_PUBLICATION}:
            if effect.expected_descriptor_hash != descriptor_hash or plan.get("expected_descriptor") != descriptor_payload:
                raise LandscapeRecordError("sink effect finalization descriptor differs from immutable plan")
        elif mode is SinkEffectDescriptorMode.RESULT_DERIVED:
            evidence = deep_thaw(request.evidence)
            expected_evidence = {
                "accepted_ordinals": list(request.accepted_ordinals),
                "descriptor": descriptor_payload,
                "diverted_ordinals": list(request.diverted_ordinals),
            }
            if evidence != expected_evidence:
                raise LandscapeRecordError("result-derived evidence is not the exact authoritative descriptor and member partition")
        if mode is SinkEffectDescriptorMode.NO_PUBLICATION:
            if request.publication_performed or request.publication_evidence_kind not in {"inherited", "virtual"}:
                raise LandscapeRecordError("no-publication finalization requires inherited or virtual non-publication evidence")
        elif not request.publication_performed or request.publication_evidence_kind not in {"returned", "reconciled"}:
            raise LandscapeRecordError("published effect requires returned or reconciled publication evidence")
        return descriptor_hash

    @staticmethod
    def _validate_attempt(
        conn: Connection,
        request: SinkEffectFinalizeRequest,
        effect: Row[Any],
        operation: Row[Any],
    ) -> None:
        if effect.descriptor_mode == SinkEffectDescriptorMode.NO_PUBLICATION.value:
            existing_external = conn.scalar(
                select(func.count())
                .select_from(sink_effect_attempts_table)
                .where(
                    sink_effect_attempts_table.c.effect_id == request.effect_id,
                    sink_effect_attempts_table.c.action.in_(
                        (SinkEffectAttemptAction.COMMIT.value, SinkEffectAttemptAction.RECONCILE.value)
                    ),
                )
            )
            if existing_external:
                raise LandscapeRecordError("no-publication effect has forbidden external attempt history")
            return
        if request.attempt_id is None:
            raise LandscapeRecordError("published effect finalization requires an exact returned attempt")
        attempt = conn.execute(
            select(sink_effect_attempts_table)
            .where(sink_effect_attempts_table.c.attempt_id == request.attempt_id)
            .with_for_update(of=sink_effect_attempts_table)
        ).fetchone()
        if attempt is None or attempt.effect_id != request.effect_id:
            raise LandscapeRecordError("finalization attempt does not belong to the sink effect")
        if attempt.state != SinkEffectAttemptState.RETURNED.value or attempt.generation != request.generation:
            raise LandscapeRecordError("finalization attempt is not an exact returned result for this generation")
        expected_action = (
            SinkEffectAttemptAction.RECONCILE.value
            if request.publication_evidence_kind == "reconciled"
            else SinkEffectAttemptAction.COMMIT.value
        )
        if attempt.action != expected_action:
            raise LandscapeRecordError("finalization attempt action disagrees with publication evidence kind")
        evidence_json = canonical_json(deep_thaw(request.evidence))
        if attempt.evidence_json != evidence_json or attempt.evidence_hash != sha256(evidence_json.encode("utf-8")).hexdigest():
            raise LandscapeRecordError("finalization evidence differs from the returned attempt winner")
        if request.publication_evidence_kind == "reconciled":
            if request.reconcile_kind is not SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR:
                raise LandscapeRecordError("reconciled finalization requires APPLIED_WITH_EXACT_DESCRIPTOR")
        elif request.reconcile_kind is not None:
            raise LandscapeRecordError("returned commit finalization must not claim a reconcile result")
        assert operation.operation_id is not None

    @staticmethod
    def _advance_stream_head(conn: Connection, effect: Row[Any], descriptor_hash: str) -> None:
        if effect.stream_id is None:
            return
        predicates = [sink_effect_streams_table.c.stream_id == effect.stream_id]
        if effect.stream_sequence == 0:
            predicates.append(sink_effect_streams_table.c.head_effect_id.is_(None))
        else:
            predicates.append(sink_effect_streams_table.c.head_effect_id == effect.predecessor_effect_id)
        result = conn.execute(
            sink_effect_streams_table.update()
            .where(*predicates)
            .values(head_effect_id=effect.effect_id, head_descriptor_hash=descriptor_hash)
        )
        if result.rowcount != 1:
            raise LandscapeRecordError("sink effect stream head CAS refused a skipped or divergent predecessor")

    def _load_finalized_winner(
        self,
        conn: Connection,
        request: SinkEffectFinalizeRequest,
        effect: Row[Any],
    ) -> SinkEffectFinalizationResult:
        if effect.generation != request.generation:
            raise LandscapeRecordError("finalized sink effect winner has a different generation")
        descriptor_hash = self._validate_plan_and_descriptor(request, effect)
        if effect.result_descriptor_hash != descriptor_hash:
            raise LandscapeRecordError("finalized sink effect winner has a divergent descriptor")
        if (
            effect.publication_performed != request.publication_performed
            or effect.publication_evidence_kind != request.publication_evidence_kind
        ):
            raise LandscapeRecordError("finalized sink effect winner has divergent publication evidence")
        artifact_row = conn.execute(
            select(artifacts_table).where(artifacts_table.c.artifact_id == effect.artifact_id).with_for_update(of=artifacts_table)
        ).fetchone()
        if artifact_row is None:
            raise LandscapeRecordError("finalized sink effect artifact winner is missing")
        artifact = self._artifact_loader.load(artifact_row)
        if (
            artifact.sink_effect_id != request.effect_id
            or artifact.idempotency_key != effect.artifact_idempotency_key
            or artifact.artifact_type != request.descriptor.artifact_type
            or artifact.path_or_uri != request.descriptor.path_or_uri
            or artifact.content_hash != request.descriptor.content_hash
            or artifact.size_bytes != request.descriptor.size_bytes
        ):
            raise LandscapeRecordError("finalized sink effect artifact winner is divergent")
        self._validate_attempt_retry(conn, request, effect)
        members = self._load_members(conn, request.effect_id)
        self._validate_partition(request, members)
        state_ids: list[str] = []
        outcome_ids: list[str] = []
        by_ordinal = {int(member.ordinal): member for member in members}
        for requested_member in request.members:
            durable = by_ordinal[requested_member.ordinal]
            state = conn.execute(
                select(node_states_table.c.state_id)
                .where(
                    node_states_table.c.run_id == effect.run_id,
                    node_states_table.c.node_id == effect.sink_node_id,
                    node_states_table.c.token_id == durable.token_id,
                    node_states_table.c.status == NodeStateStatus.COMPLETED.value,
                )
                .order_by(node_states_table.c.attempt.desc(), node_states_table.c.state_id.desc())
                .limit(1)
            ).fetchone()
            outcome = conn.execute(
                select(token_outcomes_table.c.outcome_id)
                .where(
                    token_outcomes_table.c.run_id == effect.run_id,
                    token_outcomes_table.c.token_id == durable.token_id,
                    token_outcomes_table.c.completed == 1,
                )
                .order_by(token_outcomes_table.c.recorded_at.desc(), token_outcomes_table.c.outcome_id.desc())
                .limit(1)
            ).fetchone()
            if state is None or outcome is None:
                raise LandscapeRecordError("finalized sink effect winner is missing member state/outcome evidence")
            state_ids.append(str(state.state_id))
            outcome_ids.append(str(outcome.outcome_id))
        return SinkEffectFinalizationResult(
            effect=self._effect_loader.load(effect),
            artifact=artifact,
            state_ids=tuple(state_ids),
            outcome_ids=tuple(outcome_ids),
        )

    @staticmethod
    def _validate_attempt_retry(conn: Connection, request: SinkEffectFinalizeRequest, effect: Row[Any]) -> None:
        if effect.descriptor_mode == SinkEffectDescriptorMode.NO_PUBLICATION.value:
            if request.attempt_id is not None:
                raise LandscapeRecordError("no-publication winner cannot carry an attempt")
            return
        if request.attempt_id is None:
            raise LandscapeRecordError("finalized retry requires the original returned attempt")
        attempt = conn.execute(
            select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == request.attempt_id)
        ).fetchone()
        evidence_json = canonical_json(deep_thaw(request.evidence))
        if (
            attempt is None
            or attempt.effect_id != request.effect_id
            or attempt.generation != request.generation
            or attempt.state != SinkEffectAttemptState.RETURNED.value
            or attempt.evidence_json != evidence_json
        ):
            raise LandscapeRecordError("finalized retry attempt/evidence differs from the durable winner")

    @staticmethod
    def _backend_pid(conn: Connection) -> int:
        if conn.dialect.name == "postgresql":
            return int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        return id(conn.connection)

    def _after_token_locks(self, _backend_pid: int, _token_ids: tuple[str, ...]) -> None:
        """Deterministic real-backend contention seam."""

    def _after_state_locks(self, _backend_pid: int, _state_ids: tuple[str, ...]) -> None:
        """Deterministic real-backend contention seam."""

    def _after_effect_locks(self, _backend_pid: int, _effect_ids: tuple[str, ...]) -> None:
        """Deterministic real-backend contention seam."""

    def _after_commit(self, _effect_id: str) -> None:
        """Response-loss seam: called only after the atomic commit succeeds."""


__all__ = [
    "SinkEffectFinalization",
    "SinkEffectFinalizationMember",
    "SinkEffectFinalizationResult",
    "SinkEffectFinalizeRequest",
]
