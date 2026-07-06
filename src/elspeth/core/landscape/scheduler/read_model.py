"""Read models over scheduler work items: counts, summaries, quiescence.

Read-only aggregation queries plus the shared quiescence/unresolved
predicates the engine's invariant checks are built on. No writes, no
events. Extracted from ``TokenSchedulerRepository``
(filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import ColumnElement, and_, func, or_, select

from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.core.landscape.database import Tier1Engine
from elspeth.core.landscape.schema import token_work_items_table


def unresolved_work_predicate() -> ColumnElement[bool]:
    """Predicate for work that has not reached a durable sink handoff.

    PENDING_SINK rows — and LEASED rows carrying a ``pending_sink_name``
    (a pending sink re-claimed during resume recovery) — are excluded:
    their producer work is durably complete and they are terminalized
    only after sink durability via ``mark_pending_sink_terminal``.
    """
    return or_(
        token_work_items_table.c.status.in_(
            (
                TokenWorkStatus.READY.value,
                TokenWorkStatus.BLOCKED.value,
            )
        ),
        and_(
            token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
            token_work_items_table.c.pending_sink_name.is_(None),
        ),
    )


def unquiesced_work_predicate() -> ColumnElement[bool]:
    """Predicate for §D step-2 journal quiescence (ADR-030, slice 3).

    The EOF flush may run only when no work item can still produce a NEW
    barrier arrival: READY rows and LEASED rows are in flight. Excluded:

    - BLOCKED rows — barrier holds are exactly what the EOF flush
      resolves (and queue holds are released by the flush outputs);
    - PENDING_SINK rows and LEASED rows carrying ``pending_sink_name``
      (resume-recovered pending sinks stay LEASED through this point) —
      their producer work is durably complete; they are terminalized only
      after the LATER sink write. Counting them would wedge every resume
      with recovered pending sinks (mirror of ``_unresolved_work_predicate``
      minus BLOCKED).
    """
    return or_(
        token_work_items_table.c.status == TokenWorkStatus.READY.value,
        and_(
            token_work_items_table.c.status == TokenWorkStatus.LEASED.value,
            token_work_items_table.c.pending_sink_name.is_(None),
        ),
    )


class SchedulerReadModel:
    """Read-only counts and summaries over ``token_work_items``."""

    def __init__(self, engine: Tier1Engine) -> None:
        self._engine = engine

    def count_ready_in_set(self, *, run_id: str, work_item_ids: Sequence[str]) -> int:
        """Count how many of the given work item IDs are in READY status.

        Returns the number of the supplied ``work_item_ids`` whose
        ``token_work_items`` row is currently ``READY``. Scoped to ``run_id``
        like every sibling verb; an ID belonging to another run does not count.
        A non-READY (or missing) row is simply not counted — this verb proves
        only "how many are still READY", NOT why the others are not (that could
        be LEASED, PENDING_SINK, TERMINAL, FAILED, or absent). An empty input
        returns 0.
        """
        if not work_item_ids:
            return 0
        # Bound each ``.in_()`` so a large explode fan-out (>999 siblings)
        # cannot exceed SQLITE_MAX_VARIABLE_NUMBER. One extra bind slot is used
        # by the run_id parameter, so keep the chunk comfortably under the
        # historical 999 limit.
        chunk_size = 900
        ids = list(work_item_ids)
        total = 0
        with self._engine.connect() as conn:
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                result = conn.execute(
                    select(func.count())
                    .select_from(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.work_item_id.in_(chunk))
                    .where(token_work_items_table.c.status == TokenWorkStatus.READY.value)
                ).scalar_one()
                total += int(result)
        return total

    def count_failed_in_set(self, *, run_id: str, work_item_ids: Sequence[str]) -> int:
        """Count how many of the given work item IDs are in FAILED status.

        Companion to :meth:`count_ready_in_set` for the ADR-030 M1 relinquish
        discriminator. FAILED is the ONLY ``token_work_items`` status absent from
        BOTH the run-level backstop (:meth:`count_active_work` →
        ``has_unresolved_scheduler_work``) AND ``complete_run``'s quiescence CAS
        (which both cover READY/LEASED/BLOCKED/PENDING_SINK). So a leader that
        relinquished a self-FAILED pending continuation would lose it with no
        backstop. The leader uses this verb to REFUSE to relinquish whenever ANY
        pending row is FAILED — keeping a self-FAILED stray loud. Scoped to
        ``run_id`` like every sibling verb; chunked for
        ``SQLITE_MAX_VARIABLE_NUMBER``. An empty input returns 0.
        """
        if not work_item_ids:
            return 0
        chunk_size = 900
        ids = list(work_item_ids)
        total = 0
        with self._engine.connect() as conn:
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                result = conn.execute(
                    select(func.count())
                    .select_from(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.work_item_id.in_(chunk))
                    .where(token_work_items_table.c.status == TokenWorkStatus.FAILED.value)
                ).scalar_one()
                total += int(result)
        return total

    def has_peer_owned_work(self, *, run_id: str, caller_owner: str) -> bool:
        """Return True if any non-terminal row is owned by a DIFFERENT worker.

        ADR-030 M1 relinquish discriminator, arm (3): proves a PEER (some
        ``lease_owner`` other than ``caller_owner``) is/was carrying work on this
        run. A row counts if it is LEASED or PENDING_SINK with a non-NULL
        ``lease_owner`` that differs from ``caller_owner``. PENDING_SINK is
        included deliberately: ``mark_pending_sink`` KEEPS the claimant's
        ``lease_owner`` (unlike mark_terminal/mark_failed/mark_blocked, which NULL
        it), so a follower that has already parked all its claims as PENDING_SINK
        and dropped its active LEASES is STILL detectable here — the in-claim drain
        must see that the work was legitimately taken by a peer even after the
        peer's leases lapse. An N=1 leader's own rows carry the leader's owner (and
        BLOCKED rows carry no owner), so this returns False for a solo leader,
        keeping its self-stranded continuations loud. Scoped to ``run_id`` like
        every sibling verb.
        """
        with self._engine.connect() as conn:
            found = conn.execute(
                select(token_work_items_table.c.work_item_id)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status.in_((TokenWorkStatus.LEASED.value, TokenWorkStatus.PENDING_SINK.value)))
                .where(token_work_items_table.c.lease_owner.is_not(None))
                .where(token_work_items_table.c.lease_owner != caller_owner)
                .limit(1)
            ).first()
        return found is not None

    def count_active_work(self, *, run_id: str) -> int:
        """Count non-terminal scheduler work for a run."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
        )
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.status.in_(active_statuses))
            ).scalar_one()
        return int(result)

    def active_row_ids(self, *, run_id: str) -> frozenset[str]:
        """Return row IDs represented by non-terminal scheduler work."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
        )
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table.c.row_id)
                    .distinct()
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status.in_(active_statuses))
                )
                .scalars()
                .all()
            )
        return frozenset(rows)

    def count_unquiesced_work(self, *, run_id: str) -> int:
        """Count work items still able to deposit new barrier arrivals (§D step 2)."""
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(unquiesced_work_predicate())
            ).scalar_one()
        return int(result)

    def summarize_unquiesced_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize §D step-2 unquiesced work for invariant diagnostics."""
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        token_work_items_table.c.status,
                        token_work_items_table.c.node_id,
                        token_work_items_table.c.lease_owner,
                        func.count(),
                    )
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(unquiesced_work_predicate())
                    .group_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.node_id,
                        token_work_items_table.c.lease_owner,
                    )
                    .order_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.node_id,
                        token_work_items_table.c.lease_owner,
                    )
                )
                .tuples()
                .all()
            )
        return tuple(
            f"status={status}, node={node_id}, lease_owner={lease_owner}, count={count}" for status, node_id, lease_owner, count in rows
        )

    def count_unresolved_work(self, *, run_id: str) -> int:
        """Count scheduler work not yet resolved into a durable sink handoff."""
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(unresolved_work_predicate())
            ).scalar_one()
        return int(result)

    def summarize_unresolved_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize unresolved work grouped by status and blocking keys."""
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                        func.count(),
                    )
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(unresolved_work_predicate())
                    .group_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                    .order_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                )
                .tuples()
                .all()
            )
        return tuple(
            f"status={status}, queue={queue_key}, barrier={barrier_key}, count={count}" for status, queue_key, barrier_key, count in rows
        )

    def summarize_active_work(self, *, run_id: str) -> tuple[str, ...]:
        """Summarize active work grouped by status and blocking keys."""
        active_statuses = (
            TokenWorkStatus.READY.value,
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
        )
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                        func.count(),
                    )
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status.in_(active_statuses))
                    .group_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                    .order_by(
                        token_work_items_table.c.status,
                        token_work_items_table.c.queue_key,
                        token_work_items_table.c.barrier_key,
                    )
                )
                .tuples()
                .all()
            )
        return tuple(
            f"status={status}, queue={queue_key}, barrier={barrier_key}, count={count}" for status, queue_key, barrier_key, count in rows
        )
