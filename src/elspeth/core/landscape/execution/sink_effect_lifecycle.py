"""Short, generation-fenced transactions for the sink-effect lifecycle."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, Final

from sqlalchemy import Row, func, select
from sqlalchemy.engine import Connection

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.audit import SinkEffect, SinkEffectAttempt
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.sink_effects import (
    SinkEffectAttemptAction,
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
    SinkEffectAttemptState,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectLease,
    SinkEffectPlan,
    SinkEffectState,
)
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.model_loaders import SinkEffectAttemptLoader, SinkEffectLoader
from elspeth.core.landscape.schema import (
    calls_table,
    operations_table,
    sink_effect_attempts_table,
    sink_effect_members_table,
    sink_effect_streams_table,
    sink_effects_table,
)

_LOWER_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")


def _require_hash(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_positive_ttl(ttl: timedelta) -> None:
    if type(ttl) is not timedelta or ttl <= timedelta(0):
        raise ValueError("lease ttl must be a positive timedelta")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _descriptor_payload(plan: SinkEffectPlan) -> object:
    descriptor = plan.expected_descriptor
    if descriptor is None:
        return None
    return {
        "artifact_type": descriptor.artifact_type,
        "content_hash": descriptor.content_hash,
        "metadata": None if descriptor.metadata is None else deep_thaw(descriptor.metadata),
        "path_or_uri": descriptor.path_or_uri,
        "size_bytes": descriptor.size_bytes,
    }


def _plan_json(plan: SinkEffectPlan) -> str:
    return canonical_json(
        {
            "descriptor_mode": plan.descriptor_mode.value,
            "effect_id": plan.effect_id,
            "expected_descriptor": _descriptor_payload(plan),
            "input_kind": plan.input_kind.value,
            "inspection_mode": plan.inspection_mode.value,
            "payload_hash": plan.payload_hash,
            "plan_hash": plan.plan_hash,
            "protocol_version": plan.protocol_version,
            "safe_evidence": deep_thaw(plan.safe_evidence),
            "target": plan.target,
        }
    )


def _attempt_id(request: SinkEffectAttemptRequest, *, ordinal: int) -> str:
    return stable_hash(
        {
            "action": request.action.value,
            "effect_id": request.effect_id,
            "generation": request.generation,
            "member_ordinal": request.member_ordinal,
            "ordinal": ordinal,
            "request_hash": request.request_hash,
            "schema": "sink-effect-attempt-v1",
        }
    )


class SinkEffectLifecycle:
    """Own lifecycle mutations without holding locks across external I/O."""

    def __init__(self, db: LandscapeDB, *, effect_loader: SinkEffectLoader) -> None:
        self._db = db
        self._effect_loader = effect_loader
        self._attempt_loader = SinkEffectAttemptLoader()

    def complete_plan(self, effect_id: str, plan: SinkEffectPlan) -> SinkEffect:
        _require_hash(effect_id, "effect_id")
        if type(plan) is not SinkEffectPlan:
            raise TypeError("plan must be exact SinkEffectPlan")
        if plan.effect_id != effect_id:
            raise ValueError("plan effect_id does not match the requested effect")
        encoded_plan = _plan_json(plan)
        expected_descriptor_hash = None if plan.expected_descriptor is None else stable_hash(_descriptor_payload(plan))

        with self._db.write_connection() as conn:
            row = self._lock_effect(conn, effect_id, include_stream=True)
            if row.input_kind != plan.input_kind.value:
                raise LandscapeRecordError("sink effect plan input kind is divergent")
            if row.state != SinkEffectState.RESERVED.value:
                if row.plan_json == encoded_plan and row.plan_hash == plan.plan_hash:
                    return self._effect_loader.load(row)
                raise LandscapeRecordError("sink effect has a divergent plan")
            self._require_predecessor_finalized(conn, row)

            inspection_attempt_id: str
            if plan.inspection_mode is SinkEffectInspectionMode.NO_INSPECTION_REQUIRED:
                inspection_attempt_id = stable_hash(
                    {"effect_id": effect_id, "reference": "no-inspection-required:v1", "schema": "sink-effect-inspection-sentinel-v1"}
                )
            else:
                attempt = conn.execute(
                    select(sink_effect_attempts_table)
                    .where(
                        sink_effect_attempts_table.c.effect_id == effect_id,
                        sink_effect_attempts_table.c.action == SinkEffectAttemptAction.INSPECT.value,
                        sink_effect_attempts_table.c.state == SinkEffectAttemptState.RETURNED.value,
                    )
                    .order_by(sink_effect_attempts_table.c.completed_at.desc(), sink_effect_attempts_table.c.attempt_id.desc())
                    .limit(1)
                ).fetchone()
                if attempt is None:
                    raise LandscapeRecordError("inspected sink effect plan requires a returned inspect attempt")
                inspection_attempt_id = str(attempt.attempt_id)

            timestamp = now()
            updated = conn.execute(
                sink_effects_table.update()
                .where(
                    sink_effects_table.c.effect_id == effect_id,
                    sink_effects_table.c.state == SinkEffectState.RESERVED.value,
                    sink_effects_table.c.plan_hash.is_(None),
                )
                .values(
                    state=SinkEffectState.PREPARED.value,
                    target_json=canonical_json({"resolved_target": plan.target}),
                    inspection_mode=plan.inspection_mode.value,
                    inspection_attempt_id=inspection_attempt_id,
                    plan_json=encoded_plan,
                    plan_hash=plan.plan_hash,
                    descriptor_mode=plan.descriptor_mode.value,
                    expected_descriptor_hash=expected_descriptor_hash,
                    precondition_hash=stable_hash(
                        {"inspection_attempt_id": inspection_attempt_id, "safe_evidence": deep_thaw(plan.safe_evidence)}
                    ),
                    prepared_at=timestamp,
                    updated_at=timestamp,
                )
            )
            if updated.rowcount != 1:
                raise LandscapeRecordError("sink effect plan CAS lost unexpectedly")
            if row.input_kind == SinkEffectInputKind.PIPELINE_MEMBERS.value:
                conn.execute(
                    sink_effect_members_table.update()
                    .where(sink_effect_members_table.c.effect_id == effect_id)
                    .values(prepared_disposition="accepted", member_state=SinkEffectState.PREPARED.value)
                )
            winner = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == effect_id)).fetchone()
            if winner is None:  # pragma: no cover - same transaction
                raise LandscapeRecordError("prepared sink effect disappeared")
            return self._effect_loader.load(winner)

    def acquire_lease(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        self._validate_owner(owner)
        _require_positive_ttl(ttl)
        with self._db.write_connection() as conn:
            row = self._lock_effect(conn, effect_id, include_stream=True)
            if row.state == SinkEffectState.RESERVED.value:
                raise LandscapeRecordError("sink effect must be prepared before lease acquisition")
            if row.state != SinkEffectState.PREPARED.value:
                if row.state == SinkEffectState.IN_FLIGHT.value and row.lease_owner == owner and _utc(row.lease_expires_at) >= now():
                    return SinkEffectLease(row.effect_id, row.lease_owner, row.generation, _utc(row.lease_expires_at))
                raise LandscapeRecordError(f"sink effect cannot acquire lease from state {row.state!r}")
            self._require_predecessor_finalized(conn, row)
            timestamp = now()
            expires_at = timestamp + ttl
            generation = int(row.generation) + 1
            conn.execute(
                sink_effects_table.update()
                .where(sink_effects_table.c.effect_id == effect_id, sink_effects_table.c.state == SinkEffectState.PREPARED.value)
                .values(
                    state=SinkEffectState.IN_FLIGHT.value,
                    lease_owner=owner,
                    generation=generation,
                    lease_heartbeat_at=timestamp,
                    lease_expires_at=expires_at,
                    updated_at=timestamp,
                )
            )
            return SinkEffectLease(effect_id, owner, generation, expires_at)

    def heartbeat_lease(
        self,
        effect_id: str,
        *,
        owner: str,
        generation: int,
        ttl: timedelta,
    ) -> SinkEffectLease:
        self._validate_owner(owner)
        _require_positive_ttl(ttl)
        with self._db.write_connection() as conn:
            row = self._lock_effect(conn, effect_id, include_stream=False)
            timestamp = now()
            if (
                row.state != SinkEffectState.IN_FLIGHT.value
                or row.lease_owner != owner
                or row.generation != generation
                or _utc(row.lease_expires_at) < timestamp
            ):
                raise LandscapeRecordError("sink effect lease heartbeat has stale owner or generation")
            expires_at = timestamp + ttl
            conn.execute(
                sink_effects_table.update()
                .where(sink_effects_table.c.effect_id == effect_id)
                .values(lease_heartbeat_at=timestamp, lease_expires_at=expires_at, updated_at=timestamp)
            )
            return SinkEffectLease(effect_id, owner, generation, expires_at)

    def takeover_expired(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        self._validate_owner(owner)
        _require_positive_ttl(ttl)
        with self._db.write_connection() as conn:
            row = self._lock_effect(conn, effect_id, include_stream=False)
            timestamp = now()
            if row.state == SinkEffectState.FINALIZED.value:
                raise LandscapeRecordError("finalized sink effect cannot be taken over")
            if row.state != SinkEffectState.IN_FLIGHT.value or _utc(row.lease_expires_at) >= timestamp:
                raise LandscapeRecordError("sink effect lease has not expired")
            generation = int(row.generation) + 1
            expires_at = timestamp + ttl
            conn.execute(
                sink_effects_table.update()
                .where(
                    sink_effects_table.c.effect_id == effect_id,
                    sink_effects_table.c.state == SinkEffectState.IN_FLIGHT.value,
                    sink_effects_table.c.generation == row.generation,
                )
                .values(
                    lease_owner=owner,
                    generation=generation,
                    lease_heartbeat_at=timestamp,
                    lease_expires_at=expires_at,
                    updated_at=timestamp,
                )
            )
            return SinkEffectLease(effect_id, owner, generation, expires_at)

    def begin_attempt(self, request: SinkEffectAttemptRequest) -> SinkEffectAttempt:
        if type(request) is not SinkEffectAttemptRequest:
            raise TypeError("request must be exact SinkEffectAttemptRequest")
        with self._db.write_connection() as conn:
            effect = self._lock_effect(conn, request.effect_id, include_stream=True)
            self._validate_attempt_authority(effect, request)
            if request.member_ordinal is not None:
                member = conn.execute(
                    select(sink_effect_members_table)
                    .where(
                        sink_effect_members_table.c.effect_id == request.effect_id,
                        sink_effect_members_table.c.ordinal == request.member_ordinal,
                    )
                    .with_for_update()
                ).fetchone()
                if member is None:
                    raise LandscapeRecordError("sink effect attempt references a missing member")
            operation = self._lock_operation(conn, request.effect_id)
            member_predicate = (
                sink_effect_attempts_table.c.member_ordinal.is_(None)
                if request.member_ordinal is None
                else sink_effect_attempts_table.c.member_ordinal == request.member_ordinal
            )
            existing_intent = conn.execute(
                select(sink_effect_attempts_table).where(
                    sink_effect_attempts_table.c.effect_id == request.effect_id,
                    member_predicate,
                    sink_effect_attempts_table.c.generation == request.generation,
                    sink_effect_attempts_table.c.action == request.action.value,
                    sink_effect_attempts_table.c.request_hash == request.request_hash,
                    sink_effect_attempts_table.c.state == SinkEffectAttemptState.INTENT.value,
                )
            ).fetchone()
            if existing_intent is not None:
                return self._attempt_loader.load(existing_intent)
            ordinal = int(
                conn.scalar(
                    select(func.count())
                    .select_from(sink_effect_attempts_table)
                    .where(sink_effect_attempts_table.c.effect_id == request.effect_id)
                )
                or 0
            )
            attempt_id = _attempt_id(request, ordinal=ordinal)
            timestamp = now()
            conn.execute(
                sink_effect_attempts_table.insert().values(
                    attempt_id=attempt_id,
                    effect_id=request.effect_id,
                    member_ordinal=request.member_ordinal,
                    generation=request.generation,
                    action=request.action.value,
                    call_kind=request.action.value,
                    request_hash=request.request_hash,
                    state=SinkEffectAttemptState.INTENT.value,
                    evidence_json=None,
                    evidence_hash=None,
                    started_at=timestamp,
                    completed_at=None,
                    latency_ms=None,
                )
            )
            row = conn.execute(select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == attempt_id)).one()
            assert operation is not None
            return self._attempt_loader.load(row)

    def record_attempt_result(self, result: SinkEffectAttemptResult) -> SinkEffectAttempt:
        if type(result) is not SinkEffectAttemptResult:
            raise TypeError("result must be exact SinkEffectAttemptResult")
        evidence_json = canonical_json(deep_thaw(result.evidence))
        evidence_hash = sha256(evidence_json.encode("utf-8")).hexdigest()
        with self._db.write_connection() as conn:
            optimistic = conn.execute(
                select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == result.attempt_id)
            ).fetchone()
            if optimistic is None:
                raise LandscapeRecordError(f"sink effect attempt {result.attempt_id!r} does not exist")
            effect = self._lock_effect(conn, optimistic.effect_id, include_stream=True)
            operation = self._lock_operation(conn, optimistic.effect_id)
            attempt = self._lock_attempt(conn, result.attempt_id)
            if attempt.state == SinkEffectAttemptState.RETURNED.value:
                if (
                    attempt.evidence_json == evidence_json
                    and attempt.evidence_hash == evidence_hash
                    and attempt.latency_ms == result.latency_ms
                ):
                    return self._attempt_loader.load(attempt)
                raise LandscapeRecordError("attempt result is divergent from the durable winner")
            if attempt.state != SinkEffectAttemptState.INTENT.value:
                raise LandscapeRecordError(f"attempt cannot return from state {attempt.state!r}")
            if effect.generation != attempt.generation:
                raise LandscapeRecordError("attempt result has stale generation")
            timestamp = now()
            conn.execute(
                sink_effect_attempts_table.update()
                .where(
                    sink_effect_attempts_table.c.attempt_id == result.attempt_id,
                    sink_effect_attempts_table.c.state == SinkEffectAttemptState.INTENT.value,
                )
                .values(
                    state=SinkEffectAttemptState.RETURNED.value,
                    evidence_json=evidence_json,
                    evidence_hash=evidence_hash,
                    completed_at=timestamp,
                    latency_ms=result.latency_ms,
                )
            )
            self._insert_call(
                conn,
                operation_id=operation.operation_id,
                call_type=self._call_type(attempt.action),
                status=CallStatus.SUCCESS,
                request_hash=attempt.request_hash,
                response_hash=evidence_hash,
                error_json=None,
                latency_ms=result.latency_ms,
            )
            winner = conn.execute(
                select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == result.attempt_id)
            ).one()
            return self._attempt_loader.load(winner)

    def mark_response_lost(self, attempt_id: str) -> SinkEffectAttempt:
        _require_hash(attempt_id, "attempt_id")
        evidence = {"classification": "response_lost"}
        evidence_json = canonical_json(evidence)
        evidence_hash = sha256(evidence_json.encode("utf-8")).hexdigest()
        with self._db.write_connection() as conn:
            optimistic = conn.execute(
                select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == attempt_id)
            ).fetchone()
            if optimistic is None:
                raise LandscapeRecordError(f"sink effect attempt {attempt_id!r} does not exist")
            effect = self._lock_effect(conn, optimistic.effect_id, include_stream=True)
            operation = self._lock_operation(conn, optimistic.effect_id)
            attempt = self._lock_attempt(conn, attempt_id)
            if attempt.state == SinkEffectAttemptState.RESPONSE_LOST.value:
                return self._attempt_loader.load(attempt)
            if attempt.state != SinkEffectAttemptState.INTENT.value:
                raise LandscapeRecordError(f"attempt cannot become response-lost from state {attempt.state!r}")
            if effect.generation != attempt.generation:
                raise LandscapeRecordError("response-lost classification has stale generation")
            timestamp = now()
            latency_ms = max(0.0, (timestamp - _utc(attempt.started_at)).total_seconds() * 1_000)
            conn.execute(
                sink_effect_attempts_table.update()
                .where(
                    sink_effect_attempts_table.c.attempt_id == attempt_id,
                    sink_effect_attempts_table.c.state == SinkEffectAttemptState.INTENT.value,
                )
                .values(
                    state=SinkEffectAttemptState.RESPONSE_LOST.value,
                    evidence_json=evidence_json,
                    evidence_hash=evidence_hash,
                    completed_at=timestamp,
                    latency_ms=latency_ms,
                )
            )
            self._insert_call(
                conn,
                operation_id=operation.operation_id,
                call_type=self._call_type(attempt.action),
                status=CallStatus.ERROR,
                request_hash=attempt.request_hash,
                response_hash=None,
                error_json=evidence_json,
                latency_ms=latency_ms,
            )
            winner = conn.execute(select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == attempt_id)).one()
            return self._attempt_loader.load(winner)

    def _lock_effect(self, conn: Connection, effect_id: str, *, include_stream: bool) -> Row[Any]:
        _require_hash(effect_id, "effect_id")
        optimistic = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == effect_id)).fetchone()
        if optimistic is None:
            raise LandscapeRecordError(f"sink effect {effect_id!r} does not exist")
        if include_stream and optimistic.stream_id is not None:
            stream = conn.execute(
                select(sink_effect_streams_table).where(sink_effect_streams_table.c.stream_id == optimistic.stream_id).with_for_update()
            ).fetchone()
            if stream is None:
                raise LandscapeRecordError("sink effect stream disappeared")
        effect_ids = sorted({effect_id, optimistic.predecessor_effect_id} - {None})
        rows = conn.execute(
            select(sink_effects_table)
            .where(sink_effects_table.c.effect_id.in_(effect_ids))
            .order_by(sink_effects_table.c.effect_id)
            .with_for_update()
        ).fetchall()
        by_id = {row.effect_id: row for row in rows}
        row = by_id.get(effect_id)
        if row is None:
            raise LandscapeRecordError("sink effect disappeared while locking")
        self._after_effect_lock(self._backend_pid(conn), effect_id)
        return row

    @staticmethod
    def _require_predecessor_finalized(conn: Connection, effect: Row[Any]) -> None:
        if effect.predecessor_effect_id is None:
            return
        predecessor = conn.execute(
            select(sink_effects_table.c.state).where(sink_effects_table.c.effect_id == effect.predecessor_effect_id)
        ).fetchone()
        if predecessor is None or predecessor.state != SinkEffectState.FINALIZED.value:
            raise LandscapeRecordError("sink effect predecessor must be finalized")

    @staticmethod
    def _lock_operation(conn: Connection, effect_id: str) -> Row[Any]:
        row = conn.execute(select(operations_table).where(operations_table.c.sink_effect_id == effect_id).with_for_update()).fetchone()
        if row is None:
            raise LandscapeRecordError("sink effect operation is missing")
        return row

    @staticmethod
    def _lock_attempt(conn: Connection, attempt_id: str) -> Row[Any]:
        row = conn.execute(
            select(sink_effect_attempts_table).where(sink_effect_attempts_table.c.attempt_id == attempt_id).with_for_update()
        ).fetchone()
        if row is None:
            raise LandscapeRecordError(f"sink effect attempt {attempt_id!r} does not exist")
        return row

    @staticmethod
    def _validate_attempt_authority(effect: Row[Any], request: SinkEffectAttemptRequest) -> None:
        if effect.generation != request.generation:
            raise LandscapeRecordError("sink effect attempt has stale generation")
        if request.action is SinkEffectAttemptAction.INSPECT:
            if effect.state not in {SinkEffectState.RESERVED.value, SinkEffectState.PREPARED.value}:
                raise LandscapeRecordError("inspect attempt requires reserved or prepared effect")
        elif effect.state != SinkEffectState.IN_FLIGHT.value:
            raise LandscapeRecordError("commit/reconcile attempt requires an in-flight effect")
        if effect.descriptor_mode == SinkEffectDescriptorMode.NO_PUBLICATION.value and request.action is SinkEffectAttemptAction.COMMIT:
            raise LandscapeRecordError("no-publication effect cannot begin a commit attempt")

    @staticmethod
    def _insert_call(
        conn: Connection,
        *,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_hash: str,
        response_hash: str | None,
        error_json: str | None,
        latency_ms: float,
    ) -> None:
        current = conn.scalar(select(func.max(calls_table.c.call_index)).where(calls_table.c.operation_id == operation_id))
        call_index = 0 if current is None else int(current) + 1
        call_id = stable_hash({"call_index": call_index, "operation_id": operation_id, "schema": "sink-effect-call-v1"})
        conn.execute(
            calls_table.insert().values(
                call_id=call_id,
                state_id=None,
                operation_id=operation_id,
                call_index=call_index,
                call_type=call_type.value,
                status=status.value,
                request_hash=request_hash,
                request_ref=None,
                response_hash=response_hash,
                response_ref=None,
                resolved_prompt_template_hash=None,
                error_json=error_json,
                latency_ms=latency_ms,
                created_at=now(),
            )
        )

    @staticmethod
    def _call_type(action: str) -> CallType:
        return (
            CallType.HTTP
            if action in {SinkEffectAttemptAction.INSPECT.value, SinkEffectAttemptAction.RECONCILE.value}
            else CallType.FILESYSTEM
        )

    @staticmethod
    def _validate_owner(owner: str) -> None:
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("lease owner must be non-empty")
        if len(owner) > 128:
            raise ValueError("lease owner exceeds 128 characters")

    @staticmethod
    def _backend_pid(conn: Connection) -> int:
        if conn.dialect.name == "postgresql":
            return int(conn.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
        return id(conn.connection)

    def _after_effect_lock(self, _backend_pid: int, _effect_id: str) -> None:
        """Deterministic contention seam used only by real-backend tests."""


__all__ = [
    "SinkEffectAttemptRequest",
    "SinkEffectAttemptResult",
    "SinkEffectLease",
    "SinkEffectLifecycle",
]
