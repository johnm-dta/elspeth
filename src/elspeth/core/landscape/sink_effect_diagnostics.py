"""Credential-safe read model for sink-effect recovery diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sqlalchemy import select

from elspeth.contracts.export_records import (
    SinkEffectAttemptExportRecord,
    SinkEffectExportRecord,
    SinkEffectMemberExportRecord,
)
from elspeth.contracts.freeze import freeze_fields
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.export_mappers import (
    sink_effect_attempt_to_export_record,
    sink_effect_member_to_export_record,
    sink_effect_to_export_record,
)
from elspeth.core.landscape.model_loaders import SinkEffectAttemptLoader, SinkEffectLoader, SinkEffectMemberLoader
from elspeth.core.landscape.schema import sink_effect_attempts_table, sink_effect_members_table, sink_effects_table


@dataclass(frozen=True, slots=True)
class SinkEffectRecoveryHistory:
    """Complete safe history for one effect and its external calls."""

    effect: SinkEffectExportRecord
    members: tuple[SinkEffectMemberExportRecord, ...]
    attempts: tuple[SinkEffectAttemptExportRecord, ...]
    member_progress: Mapping[str, int]
    response_lost_attempts: int
    operator_guidance: str

    def __post_init__(self) -> None:
        freeze_fields(self, "member_progress")


def load_sink_effect_recovery_history(db: LandscapeDB, effect_id: str) -> SinkEffectRecoveryHistory | None:
    """Load one effect without returning target, plan, or provider evidence bodies."""
    with db.read_only_connection() as conn:
        effect_row = conn.execute(select(sink_effects_table).where(sink_effects_table.c.effect_id == effect_id)).fetchone()
        if effect_row is None:
            return None
        member_rows = conn.execute(
            select(sink_effect_members_table)
            .where(sink_effect_members_table.c.effect_id == effect_id)
            .order_by(sink_effect_members_table.c.ordinal)
        ).fetchall()
        attempt_rows = conn.execute(
            select(sink_effect_attempts_table)
            .where(sink_effect_attempts_table.c.effect_id == effect_id)
            .order_by(sink_effect_attempts_table.c.started_at, sink_effect_attempts_table.c.attempt_id)
        ).fetchall()

    effect = SinkEffectLoader().load(effect_row)
    members = tuple(SinkEffectMemberLoader().load(row) for row in member_rows)
    attempts = tuple(SinkEffectAttemptLoader().load(row) for row in attempt_rows)
    member_progress: dict[str, int] = {}
    for member in members:
        state = "unprepared" if member.member_state is None else member.member_state.value
        member_progress[state] = member_progress.get(state, 0) + 1
    response_lost_attempts = sum(attempt.state.value == "response_lost" for attempt in attempts)

    if effect.reconcile_kind is not None and effect.reconcile_kind.value == "unknown":
        guidance = (
            "Do not retry publication speculatively. Inspect the target out of band, then reconcile only with exact descriptor evidence."
        )
    elif response_lost_attempts or effect.state.value == "in_flight":
        guidance = (
            "Do not retry publication speculatively. Acquire recovery authority and reconcile the durable plan against the target first."
        )
    elif effect.state.value == "finalized":
        guidance = "The effect is finalized; use the exact artifact and publication evidence recorded here."
    else:
        guidance = "Continue the reserved effect through inspection, immutable planning, and fenced commit."

    return SinkEffectRecoveryHistory(
        effect=sink_effect_to_export_record(effect),
        members=tuple(sink_effect_member_to_export_record(member) for member in members),
        attempts=tuple(
            sink_effect_attempt_to_export_record(effect.run_id, attempt, attempt_index=index) for index, attempt in enumerate(attempts)
        ),
        member_progress=MappingProxyType(member_progress),
        response_lost_attempts=response_lost_attempts,
        operator_guidance=guidance,
    )


__all__ = ["SinkEffectRecoveryHistory", "load_sink_effect_recovery_history"]
