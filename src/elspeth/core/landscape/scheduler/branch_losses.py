"""Coalesce branch-loss ledger (design SE.5 durable hand-off ledger).

The append-only ``coalesce_branch_losses`` table: in-transaction recording
(riding the caller's disposition transaction), the intake/takeover reads,
and the fenced replay-cursor adoption mark. Extracted from
``TokenSchedulerRepository`` (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.ids import generate_id
from elspeth.core.landscape.database import Tier1Engine
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.schema import coalesce_branch_losses_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoalesceBranchLoss:
    """One ``coalesce_branch_losses`` row (§E.5 durable hand-off ledger)."""

    loss_id: str
    run_id: str
    coalesce_name: str
    row_id: str
    branch_name: str
    token_id: str
    reason: str
    recorded_by: str
    recorded_at: datetime
    adopted_epoch: int | None


def record_coalesce_branch_loss(
    conn: Connection,
    *,
    run_id: str,
    coalesce_name: str,
    row_id: str,
    branch_name: str,
    token_id: str,
    reason: str,
    recorded_by: str,
    now: datetime,
) -> bool:
    """Append one branch-loss row in the CALLER's transaction (§E.5).

    Rides the caller's lease-fenced item-disposition transaction (design
    §C.4 row 1) — it deliberately does NOT open its own transaction, so the
    loss record commits iff the disposition commits. Idempotent on the
    natural key ``(run_id, coalesce_name, row_id, branch_name)``
    (``uq_coalesce_branch_losses_natural``): a conflicting re-record returns
    ``False``. A natural-key hit with a DIFFERENT ``token_id`` is lineage
    corruption (two distinct tokens claiming one branch of one row) and
    raises Tier-1; a different ``reason`` is tolerated and logged — re-drives
    can legitimately fail for a different reason, and the first durable
    record wins (it may already have fired a must-fail policy).

    Returns ``True`` if this call inserted the row, ``False`` if the row
    pre-existed.
    """
    loss_id = f"loss_{generate_id()[:12]}"
    values = {
        "loss_id": loss_id,
        "run_id": run_id,
        "coalesce_name": coalesce_name,
        "row_id": row_id,
        "branch_name": branch_name,
        "token_id": token_id,
        "reason": reason,
        "recorded_by": recorded_by,
        "recorded_at": now,
        "adopted_epoch": None,
    }
    dialect = conn.dialect.name
    stmt: Any
    if dialect == "sqlite":
        stmt = sqlite_insert(coalesce_branch_losses_table).values(**values)
    elif dialect == "postgresql":
        stmt = postgresql_insert(coalesce_branch_losses_table).values(**values)
    else:
        raise NotImplementedError(
            "coalesce branch-loss recording requires an atomic insert-or-ignore for landscape database "
            f"dialect {dialect!r}; supported dialects: sqlite, postgresql"
        )
    inserted_loss_id = conn.execute(
        stmt.on_conflict_do_nothing(index_elements=["run_id", "coalesce_name", "row_id", "branch_name"]).returning(
            coalesce_branch_losses_table.c.loss_id
        )
    ).scalar_one_or_none()
    if inserted_loss_id is not None:
        if inserted_loss_id != loss_id:
            raise AuditIntegrityError(
                f"Coalesce branch-loss insert returned unexpected loss_id={inserted_loss_id!r}; expected {loss_id!r}."
            )
        return True
    existing = (
        conn.execute(
            select(coalesce_branch_losses_table)
            .where(coalesce_branch_losses_table.c.run_id == run_id)
            .where(coalesce_branch_losses_table.c.coalesce_name == coalesce_name)
            .where(coalesce_branch_losses_table.c.row_id == row_id)
            .where(coalesce_branch_losses_table.c.branch_name == branch_name)
        )
        .mappings()
        .one()
    )
    if existing["token_id"] != token_id:
        raise AuditIntegrityError(
            f"Coalesce branch-loss record for run_id={run_id!r} coalesce_name={coalesce_name!r} "
            f"row_id={row_id!r} branch_name={branch_name!r} already exists with token_id={existing['token_id']!r}, "
            f"but this call claims token_id={token_id!r}; two distinct tokens cannot lose the same branch of one row "
            "— token lineage corruption."
        )
    if existing["reason"] != reason:
        logger.warning(
            "coalesce branch-loss re-record for run %r coalesce %r row %r branch %r tolerated a reason change "
            "(durable %r, offered %r); the first durable record wins",
            run_id,
            coalesce_name,
            row_id,
            branch_name,
            existing["reason"],
            reason,
        )
    return False


def _loss_from_mapping(row: RowMapping) -> CoalesceBranchLoss:
    recorded_at = row["recorded_at"]
    if type(recorded_at) is datetime and recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=UTC)
    return CoalesceBranchLoss(
        loss_id=row["loss_id"],
        run_id=row["run_id"],
        coalesce_name=row["coalesce_name"],
        row_id=row["row_id"],
        branch_name=row["branch_name"],
        token_id=row["token_id"],
        reason=row["reason"],
        recorded_by=row["recorded_by"],
        recorded_at=recorded_at,
        adopted_epoch=row["adopted_epoch"],
    )


class CoalesceBranchLossRepository:
    """Read/adopt surface over the append-only branch-loss ledger."""

    def __init__(self, engine: Tier1Engine) -> None:
        self._engine = engine

    def list_unadopted_coalesce_branch_losses(self, *, run_id: str) -> list[CoalesceBranchLoss]:
        """Branch-loss rows not yet replayed into leader memory (§E.5 intake read).

        The leader replays these through ``notify_branch_lost`` BEFORE each
        drain iteration's trigger evaluation, then marks them via
        :meth:`adopt_coalesce_branch_losses` (journal-first: mark durably
        FIRST, then replay — a crash between mark and replay loses nothing
        because takeover restore derives from the FULL table). Read-only; no
        event is recorded.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(coalesce_branch_losses_table)
                    .where(coalesce_branch_losses_table.c.run_id == run_id)
                    .where(coalesce_branch_losses_table.c.adopted_epoch.is_(None))
                    .order_by(coalesce_branch_losses_table.c.recorded_at, coalesce_branch_losses_table.c.loss_id)
                )
                .mappings()
                .all()
            )
        return [_loss_from_mapping(row) for row in rows]

    def list_coalesce_branch_losses(self, *, run_id: str) -> list[CoalesceBranchLoss]:
        """ALL branch-loss rows, adopted or not (§E.4 takeover restore read).

        The new leader rebuilds ``lost_branches`` for still-pending coalesce
        groups from the full table (the D3 checkpoint scalar is retained as a
        cross-check only) and seeds executor memory directly; unadopted rows
        are then re-marked under its own epoch. Append-only ledger: rows are
        never deleted or consumed destructively. Read-only; no event.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(coalesce_branch_losses_table)
                    .where(coalesce_branch_losses_table.c.run_id == run_id)
                    .order_by(coalesce_branch_losses_table.c.recorded_at, coalesce_branch_losses_table.c.loss_id)
                )
                .mappings()
                .all()
            )
        return [_loss_from_mapping(row) for row in rows]

    def adopt_coalesce_branch_losses(
        self,
        *,
        run_id: str,
        loss_ids: Sequence[str],
        now: datetime,
        coordination_token: CoordinationToken,
    ) -> int:
        """Fenced replay-cursor mark: ``adopted_epoch NULL → epoch`` (§E.5).

        Same CAS-marker pattern as row adoption; ``coordination_token`` is
        REQUIRED (new-verb doctrine). Returns the number of rows marked —
        fewer than ``len(loss_ids)`` is FINE (already-adopted rows are the
        idempotent skip; in-memory replay dedup is structural because
        ``notify_branch_lost`` is keyed-dict assignment). A stale leader gets
        :class:`RunLeadershipLostError` before any mark.
        """
        if not loss_ids:
            return 0
        with fenced_leader_transaction(
            self._engine,
            token=coordination_token,
            now=now,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="adopt_coalesce_branch_losses",
        ) as conn:
            result = conn.execute(
                update(coalesce_branch_losses_table)
                .where(coalesce_branch_losses_table.c.run_id == run_id)
                .where(coalesce_branch_losses_table.c.loss_id.in_(tuple(loss_ids)))
                .where(coalesce_branch_losses_table.c.adopted_epoch.is_(None))
                .values(adopted_epoch=coordination_token.leader_epoch)
            )
            marked = result.rowcount
        return 0 if marked is None else int(marked)
