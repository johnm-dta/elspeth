"""Public repository for durable sink-effect reservation and reads."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from sqlalchemy import select

from elspeth.contracts.audit import SinkEffect, SinkEffectAttempt, SinkEffectMemberRecord, SinkEffectStream
from elspeth.contracts.sink_effects import (
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
    SinkEffectCommitResult,
    SinkEffectFinalizationResult,
    SinkEffectFinalizeRequest,
    SinkEffectInputKind,
    SinkEffectLease,
    SinkEffectMember,
    SinkEffectPlan,
    SinkEffectReconcileResult,
    SinkEffectReservationRequest,
    SinkEffectRole,
)
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_finalization import (
    SinkEffectFinalization,
)
from elspeth.core.landscape.execution.sink_effect_lifecycle import (
    SinkEffectLifecycle,
)
from elspeth.core.landscape.execution.sink_effect_reservation import (
    SinkEffectReservation,
    SinkEffectReservationResult,
)
from elspeth.core.landscape.model_loaders import (
    SinkEffectAttemptLoader,
    SinkEffectLoader,
    SinkEffectMemberLoader,
    SinkEffectStreamLoader,
)
from elspeth.core.landscape.schema import (
    sink_effect_attempts_table,
    sink_effect_members_table,
    sink_effect_streams_table,
    sink_effects_table,
)


class SinkEffectRepository:
    """Typed persistence surface for the sink-effect aggregate."""

    def __init__(
        self,
        db: LandscapeDB,
        ops: DatabaseOps,
        *,
        effect_loader: SinkEffectLoader,
        member_loader: SinkEffectMemberLoader,
        stream_loader: SinkEffectStreamLoader,
    ) -> None:
        self._ops = ops
        self._effect_loader = effect_loader
        self._member_loader = member_loader
        self._stream_loader = stream_loader
        self._attempt_loader = SinkEffectAttemptLoader()
        self._reservation = SinkEffectReservation(db, effect_loader=effect_loader)
        self._lifecycle = SinkEffectLifecycle(db, effect_loader=effect_loader)
        self._finalization = SinkEffectFinalization(db, ops, effect_loader=effect_loader)

    def reserve(
        self,
        request: SinkEffectReservationRequest | None = None,
        *,
        run_id: str | None = None,
        sink_node_id: str | None = None,
        role: SinkEffectRole | None = None,
        input_kind: SinkEffectInputKind | None = None,
        requested_target_hash: str | None = None,
        members: Sequence[SinkEffectMember] = (),
        audit_export_snapshot_id: str | None = None,
        config_hash: str | None = None,
        replacing_target: bool = False,
        primary_effect_id: str | None = None,
    ) -> SinkEffectReservationResult:
        """Reserve from an exact request or the equivalent explicit fields."""
        if request is not None:
            if (
                any(
                    value is not None
                    for value in (run_id, sink_node_id, role, input_kind, requested_target_hash, audit_export_snapshot_id, config_hash)
                )
                or members
                or replacing_target
                or primary_effect_id is not None
            ):
                raise TypeError("request cannot be combined with reservation keyword fields")
            return self._reservation.reserve(request)
        if None in (run_id, sink_node_id, role, input_kind, requested_target_hash, config_hash):
            raise TypeError("explicit reservation requires run, node, role, input kind, target hash, and config hash")
        assert run_id is not None
        assert sink_node_id is not None
        assert role is not None
        assert input_kind is not None
        assert requested_target_hash is not None
        assert config_hash is not None
        built = SinkEffectReservationRequest(
            run_id=run_id,
            sink_node_id=sink_node_id,
            role=role,
            input_kind=input_kind,
            requested_target_hash=requested_target_hash,
            members=tuple(members),
            audit_export_snapshot_id=audit_export_snapshot_id,
            config_hash=config_hash,
            replacing_target=replacing_target,
            primary_effect_id=primary_effect_id,
        )
        return self._reservation.reserve(built)

    def get_effect(self, effect_id: str) -> SinkEffect | None:
        row = self._ops.execute_fetchone(select(sink_effects_table).where(sink_effects_table.c.effect_id == effect_id))
        return None if row is None else self._effect_loader.load(row)

    def claim_preparation(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        return self._lifecycle.claim_preparation(effect_id, owner=owner, ttl=ttl)

    def complete_plan(self, effect_id: str, plan: SinkEffectPlan, *, claim: SinkEffectLease) -> SinkEffect:
        return self._lifecycle.complete_plan(effect_id, plan, claim=claim)

    def acquire_lease(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        return self._lifecycle.acquire_lease(effect_id, owner=owner, ttl=ttl)

    def heartbeat_lease(
        self,
        effect_id: str,
        *,
        owner: str,
        generation: int,
        ttl: timedelta,
    ) -> SinkEffectLease:
        return self._lifecycle.heartbeat_lease(effect_id, owner=owner, generation=generation, ttl=ttl)

    def takeover_expired(self, effect_id: str, *, owner: str, ttl: timedelta) -> SinkEffectLease:
        return self._lifecycle.takeover_expired(effect_id, owner=owner, ttl=ttl)

    def begin_attempt(self, request: SinkEffectAttemptRequest) -> SinkEffectAttempt:
        return self._lifecycle.begin_attempt(request)

    def get_attempts(self, effect_id: str) -> tuple[SinkEffectAttempt, ...]:
        return self._lifecycle.get_attempts(effect_id)

    def record_attempt_result(self, result: SinkEffectAttemptResult) -> SinkEffectAttempt:
        return self._lifecycle.record_attempt_result(result)

    def complete_member_result(
        self,
        attempt_id: str,
        result: SinkEffectCommitResult | SinkEffectReconcileResult,
        *,
        lease: SinkEffectLease,
    ) -> None:
        self._lifecycle.complete_member_result(attempt_id, result, lease=lease)

    def mark_response_lost(
        self,
        attempt_id: str,
        *,
        recovery_lease: SinkEffectLease | None = None,
    ) -> SinkEffectAttempt:
        return self._lifecycle.mark_response_lost(attempt_id, recovery_lease=recovery_lease)

    def finalize(self, request: SinkEffectFinalizeRequest) -> SinkEffectFinalizationResult:
        """Finalize one exact effect winner and all dependent audit state."""
        return self._finalization.finalize(request)

    def get_members(self, effect_id: str) -> tuple[SinkEffectMemberRecord, ...]:
        rows = self._ops.execute_fetchall(
            select(sink_effect_members_table)
            .where(sink_effect_members_table.c.effect_id == effect_id)
            .order_by(sink_effect_members_table.c.ordinal)
        )
        return tuple(self._member_loader.load(row) for row in rows)

    def get_members_for_tokens(
        self,
        *,
        run_id: str,
        sink_node_id: str,
        role: SinkEffectRole,
        token_ids: Sequence[str],
    ) -> tuple[SinkEffectMemberRecord, ...]:
        if type(role) is not SinkEffectRole:
            raise TypeError("role must be exact SinkEffectRole")
        requested = tuple(token_ids)
        if not requested:
            return ()
        rows = self._ops.execute_fetchall(
            select(sink_effect_members_table)
            .where(
                sink_effect_members_table.c.run_id == run_id,
                sink_effect_members_table.c.sink_node_id == sink_node_id,
                sink_effect_members_table.c.role == role.value,
                sink_effect_members_table.c.token_id.in_(requested),
            )
            .order_by(sink_effect_members_table.c.effect_id, sink_effect_members_table.c.ordinal)
        )
        return tuple(self._member_loader.load(row) for row in rows)

    def get_stream(self, stream_id: str | None) -> SinkEffectStream | None:
        if stream_id is None:
            return None
        row = self._ops.execute_fetchone(select(sink_effect_streams_table).where(sink_effect_streams_table.c.stream_id == stream_id))
        return None if row is None else self._stream_loader.load(row)

    def get_streams_for_run(self, run_id: str) -> tuple[SinkEffectStream, ...]:
        """Return target streams in stable stream-ID order."""
        rows = self._ops.execute_fetchall(
            select(sink_effect_streams_table)
            .where(sink_effect_streams_table.c.run_id == run_id)
            .order_by(sink_effect_streams_table.c.stream_id)
        )
        return tuple(self._stream_loader.load(row) for row in rows)

    def get_effects_for_run(self, run_id: str) -> tuple[SinkEffect, ...]:
        """Return effects in stable stream/sequence/effect-ID order."""
        rows = self._ops.execute_fetchall(
            select(sink_effects_table)
            .where(sink_effects_table.c.run_id == run_id)
            .order_by(
                sink_effects_table.c.stream_id,
                sink_effects_table.c.stream_sequence,
                sink_effects_table.c.effect_id,
            )
        )
        return tuple(self._effect_loader.load(row) for row in rows)

    def get_members_for_run(self, run_id: str) -> tuple[SinkEffectMemberRecord, ...]:
        """Return all pipeline members in stable effect/ordinal order."""
        rows = self._ops.execute_fetchall(
            select(sink_effect_members_table)
            .where(sink_effect_members_table.c.run_id == run_id)
            .order_by(sink_effect_members_table.c.effect_id, sink_effect_members_table.c.ordinal)
        )
        return tuple(self._member_loader.load(row) for row in rows)

    def get_attempts_for_run(self, run_id: str) -> tuple[SinkEffectAttempt, ...]:
        """Return every call witness in stable per-effect call order."""
        rows = self._ops.execute_fetchall(
            select(sink_effect_attempts_table)
            .join(sink_effects_table, sink_effect_attempts_table.c.effect_id == sink_effects_table.c.effect_id)
            .where(sink_effects_table.c.run_id == run_id)
            .order_by(
                sink_effect_attempts_table.c.effect_id,
                sink_effect_attempts_table.c.started_at,
                sink_effect_attempts_table.c.attempt_id,
            )
        )
        return tuple(self._attempt_loader.load(row) for row in rows)


__all__ = ["SinkEffectRepository"]
