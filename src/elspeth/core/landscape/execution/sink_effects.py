"""Public repository for durable sink-effect reservation and reads."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from elspeth.contracts.audit import SinkEffect, SinkEffectMemberRecord, SinkEffectStream
from elspeth.contracts.sink_effects import SinkEffectInputKind, SinkEffectMember, SinkEffectRole
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_reservation import (
    SinkEffectReservation,
    SinkEffectReservationRequest,
    SinkEffectReservationResult,
)
from elspeth.core.landscape.model_loaders import SinkEffectLoader, SinkEffectMemberLoader, SinkEffectStreamLoader
from elspeth.core.landscape.schema import sink_effect_members_table, sink_effect_streams_table, sink_effects_table


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
        self._reservation = SinkEffectReservation(db, effect_loader=effect_loader)

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

    def get_members(self, effect_id: str) -> tuple[SinkEffectMemberRecord, ...]:
        rows = self._ops.execute_fetchall(
            select(sink_effect_members_table)
            .where(sink_effect_members_table.c.effect_id == effect_id)
            .order_by(sink_effect_members_table.c.ordinal)
        )
        return tuple(self._member_loader.load(row) for row in rows)

    def get_stream(self, stream_id: str | None) -> SinkEffectStream | None:
        if stream_id is None:
            return None
        row = self._ops.execute_fetchone(select(sink_effect_streams_table).where(sink_effect_streams_table.c.stream_id == stream_id))
        return None if row is None else self._stream_loader.load(row)


__all__ = ["SinkEffectRepository"]
