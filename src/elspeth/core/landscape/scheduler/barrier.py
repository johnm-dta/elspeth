"""Barrier journal: atomic completion, adoption, and barrier-hold reads.

The F1 atomic barrier completion (consume BLOCKED inputs, emit
pending-sink/ready outputs in ONE journal transaction), its legacy
partial-release wrappers, the SE.2 fenced journal-first adoption verbs,
and the barrier-hold read surface. Extracted from
``TokenSchedulerRepository`` (filigree elspeth-ef9c36d767).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, insert, select, update
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.enums import TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import (
    BarrierEmission,
    BatchMembershipSpec,
    BlockedPendingSinkHandoff,
    BranchLossSpec,
    BufferedOutcomeSpec,
    SchedulerEventType,
    TokenWorkItem,
    TokenWorkStatus,
)
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.database import Tier1Engine, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.scheduler.branch_losses import record_coalesce_branch_loss
from elspeth.core.landscape.scheduler.events import SchedulerEventStore
from elspeth.core.landscape.scheduler.fencing import fenced_or_plain_write
from elspeth.core.landscape.scheduler.payload_codec import scrubbed_row_payload_json
from elspeth.core.landscape.scheduler.work_items import (
    insert_work_item,
    item_from_mapping,
    ready_work_item_values,
    validate_work_item_references,
)
from elspeth.core.landscape.scheduler.work_items import (
    work_item_id as make_work_item_id,
)
from elspeth.core.landscape.schema import (
    batch_members_table,
    batches_table,
    blocked_barrier_hold_clause,
    token_outcomes_table,
    token_work_items_table,
)


@dataclass(frozen=True)
class BarrierAdoptionResult:
    """Outcome of :meth:`TokenSchedulerRepository.adopt_blocked_barrier_item`.

    ``adopted=False`` is the idempotent success-SKIP (the row already carries
    a non-NULL ``barrier_adopted_epoch``): the caller MUST NOT re-feed
    executor memory or re-record audit rows on that arm — skipping the
    inserts there is exactly what makes double-BUFFERED structurally
    impossible (design §C.4 row 6a).
    """

    adopted: bool
    barrier_adopted_epoch: int
    outcome_id: str | None


class BarrierJournalRepository:
    """Barrier journal completion, adoption and read surface."""

    def __init__(self, engine: Tier1Engine, *, events: SchedulerEventStore) -> None:
        self._engine = engine
        self._events = events

    def complete_barrier(
        self,
        *,
        run_id: str,
        barrier_key: str,
        consumed_token_ids: Sequence[str],
        emitted_pending_sink: Sequence[BarrierEmission],
        emitted_ready: Sequence[BarrierEmission],
        now: datetime,
        require_exhaustive_release: bool = True,
        scope_row_id: str | None = None,
        intake_snapshot_token_ids: frozenset[str] | None = None,
        release_context: Mapping[str, object] | None = None,
        coordination_token: CoordinationToken,
        pending_sink_lease_owner: str | None = None,
        branch_losses: Sequence[BranchLossSpec] = (),
    ) -> int:
        """Complete a barrier atomically: consume BLOCKED inputs and emit outputs.

        ``coordination_token`` (ADR-030 §C.4 row 6, slice-4 ratchet:
        REQUIRED): the verify-and-extend epoch fence runs as the FIRST
        statement of the journal transaction — a deposed leader's completion
        is refused before any journal mutation. Fenced on BOTH arms (the
        legacy partial-release wrappers are leader verbs too).
        ``pending_sink_lease_owner`` is the attributed-park stamp for the
        pending-sink emission arms (passthrough transition + fresh insert):
        strict post-sink terminalization
        (:meth:`mark_pending_sink_terminal[_many]`) CASes on it. The engine
        passes its scheduler lease owner (== the §A.1 worker identity); None
        preserves the legacy NULL park for direct repository callers.

        ``intake_snapshot_token_ids`` (ADR-030 §E.3, slice 3): the token_ids
        the leader has adopted into THIS firing group — never whole executor
        memory. When supplied (strict arm only), it narrows the
        exhaustiveness universe to ``durable ∩ snapshot``: durable BLOCKED
        rows OUTSIDE the snapshot are late arrivals that legitimately stay
        BLOCKED (recorded as ``late_arrival_token_ids`` in every emission
        event's context so the completion is reconstructable from
        scheduler_events alone). The snapshot is also a defence surface —
        consumed/handed-off tokens outside it, snapshot tokens the journal
        does not hold, and snapshot tokens whose durable row belongs to a
        DIFFERENT row group than ``scope_row_id`` all raise Tier-1.
        ``None`` preserves the durable-universe exhaustiveness exactly (the
        N=1 semantics before §E.2's journal-first intake).

        ``release_context`` (§E.3a): extra forensic keys merged into the
        per-row ``MARK_BLOCKED_BARRIER_TERMINAL`` event context
        (``{"barrier_key": ..., **release_context}``) — used by the
        late-arrival journal-release call sites. ``None`` preserves the
        pinned ``{"barrier_key"}``-only legacy context.

        ``branch_losses`` (§E.5): durable branch-loss records riding THIS
        completion — a flush whose empty emission lossy-disposes fork-lineage
        branches feeding a coalesce records each loss in the SAME transaction
        as the consumption that disposes the branch tokens (record-then-notify
        uniformity rule; idempotent on the natural key).

        ONE journal transaction (F1, elspeth-ae5183307b) performs:

        - validation of the consumed set against the durable BLOCKED set under
          ``(run_id, barrier_key)`` — both directions: every consumed/handed-off
          token must hold a BLOCKED row, and every BLOCKED row in the
          exhaustiveness universe must be consumed or handed off;

          ``scope_row_id`` narrows that BLOCKED universe — both the
          missing-token cross-check and the exhaustiveness check — to one
          pending group. COALESCE barriers share ONE ``barrier_key`` (the
          coalesce_name) across ALL row_ids pending at the coalesce, and
          fork/expand children inherit their parent's ``row_id``, so a firing
          group's identity is exactly ``(run_id, barrier_key, row_id)``.
          Coalesce call sites MUST pass the firing group's ``row_id`` —
          otherwise every other row's held branches would be spuriously
          reported as orphaned. Aggregation call sites omit it
          (``barrier_key`` = the node id; one batch consumes ALL BLOCKED rows
          of the node, so whole-key exhaustiveness is correct);
        - consumed rows -> TERMINAL with payload scrub and per-row
          ``MARK_BLOCKED_BARRIER_TERMINAL`` events (``{"barrier_key"}`` context,
          the existing event shape);
        - ``emitted_pending_sink``: buffered passthrough tokens (those with a
          BLOCKED row under this barrier) transition BLOCKED -> PENDING_SINK in
          place with the handoff bundle; others INSERT a fresh PENDING_SINK row
          on the node_id-NULL terminal lane (deterministic work_item_id);
        - ``emitted_ready``: INSERT READY continuations with ``ENQUEUE`` events;
        - every emission event's context carries
          ``{"barrier_key": ..., "consumed_count": N}`` so the atomic completion
          is reconstructable from scheduler_events alone.

        ``require_exhaustive_release=False`` is the legacy wrapper arm used by
        ``mark_blocked_barrier_terminal`` / ``mark_blocked_barrier_pending_sink_many``
        (their partial-release semantics and ``{"barrier_key"}``-only emission
        context are pinned by existing tests and the lifecycle state machine):
        BLOCKED rows not named stay BLOCKED, every pending-sink emission must be
        a passthrough (fresh inserts are refused), and emission events keep the
        legacy context. New callers must not pass it; the processor's flush
        sites were migrated onto the strict default (Task 3.4, landed).

        Returns the number of consumed rows terminalized.
        """
        if scope_row_id is not None and not require_exhaustive_release:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"scope_row_id={scope_row_id!r} with require_exhaustive_release=False; row scoping narrows the "
                "exhaustiveness universe and is meaningless on the legacy partial-release arm."
            )
        if intake_snapshot_token_ids is not None and not require_exhaustive_release:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"intake_snapshot_token_ids with require_exhaustive_release=False; the snapshot narrows the "
                "exhaustiveness universe and is meaningless on the legacy partial-release arm."
            )
        consumed = tuple(consumed_token_ids)
        consumed_set = frozenset(consumed)
        if len(consumed_set) != len(consumed):
            duplicates = sorted(token_id for token_id in consumed_set if consumed.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization received duplicate live token_ids for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        pending_token_ids = tuple(emission.token_id for emission in emitted_pending_sink)
        if len(frozenset(pending_token_ids)) != len(pending_token_ids):
            duplicates = sorted(token_id for token_id in frozenset(pending_token_ids) if pending_token_ids.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier completion received duplicate pending-sink emissions for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        ready_token_ids = tuple(emission.token_id for emission in emitted_ready)
        if len(frozenset(ready_token_ids)) != len(ready_token_ids):
            duplicates = sorted(token_id for token_id in frozenset(ready_token_ids) if ready_token_ids.count(token_id) > 1)
            raise AuditIntegrityError(
                f"Scheduler barrier completion received duplicate ready emissions for run_id={run_id!r} "
                f"barrier_key={barrier_key!r}: {duplicates!r}"
            )
        pending_overlap = consumed_set & frozenset(pending_token_ids)
        if pending_overlap:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                f"token_ids both consumed and emitted to pending-sink: {sorted(pending_overlap)!r}; a BLOCKED row "
                "cannot be terminalized and handed off in the same completion."
            )
        for emission in emitted_pending_sink:
            if emission.sink_name is None or emission.outcome is None or emission.path is None:
                raise AuditIntegrityError(
                    f"Scheduler barrier completion pending-sink emission for run_id={run_id!r} "
                    f"barrier_key={barrier_key!r} token_id={emission.token_id!r} requires sink_name, outcome and path."
                )
        if intake_snapshot_token_ids is not None:
            consumed_outside_snapshot = consumed_set - intake_snapshot_token_ids
            if consumed_outside_snapshot:
                raise AuditIntegrityError(
                    f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} consumed "
                    f"token(s) outside its own intake snapshot: {sorted(consumed_outside_snapshot)!r}; the flush "
                    "caller may only consume tokens it durably adopted into this firing group (ADR-030 §E.3)."
                )
        if scope_row_id is not None:
            for emission in (*emitted_pending_sink, *emitted_ready):
                if emission.row_id is not None and emission.row_id != scope_row_id:
                    raise AuditIntegrityError(
                        f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                        f"emission token_id={emission.token_id!r} with row_id={emission.row_id!r} outside the scoped "
                        f"pending group scope_row_id={scope_row_id!r}; a scoped completion must not emit into "
                        "another row's group."
                    )

        emission_context: dict[str, object] = {"barrier_key": barrier_key}
        if require_exhaustive_release:
            emission_context["consumed_count"] = len(consumed_set)

        # Narrowed column set: no arm reads the blocked rows' row_payload_json
        # (terminalization scrubs it, passthrough overwrites it with the
        # handoff payload), so fetching it would drag every held row's full
        # payload into memory on large barrier sets for nothing.
        blocked_select = (
            select(
                token_work_items_table.c.work_item_id,
                token_work_items_table.c.token_id,
                token_work_items_table.c.node_id,
                token_work_items_table.c.attempt,
                token_work_items_table.c.lease_owner,
                token_work_items_table.c.lease_expires_at,
            )
            .where(token_work_items_table.c.run_id == run_id)
            .where(token_work_items_table.c.barrier_key == barrier_key)
            .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
        )
        if scope_row_id is not None:
            blocked_select = blocked_select.where(token_work_items_table.c.row_id == scope_row_id)

        with fenced_or_plain_write(self._engine, coordination_token=coordination_token, now=now, verb="complete_barrier") as conn:
            blocked_rows = (
                conn.execute(
                    blocked_select.order_by(
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.step_index,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
            blocked_by_token: dict[str, list[RowMapping]] = {}
            for row in blocked_rows:
                blocked_by_token.setdefault(row["token_id"], []).append(row)
            durable_token_ids = frozenset(blocked_by_token)

            scope_note = "" if scope_row_id is None else f" scope_row_id={scope_row_id!r}."
            missing_token_ids = consumed_set - durable_token_ids
            if missing_token_ids:
                matching_count = len(consumed_set & durable_token_ids)
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                    f"live consumed {len(consumed_set)} token(s), but durable BLOCKED rows contained "
                    f"{matching_count} matching token(s) out of {len(durable_token_ids)} blocked token(s). "
                    f"missing token_ids={sorted(missing_token_ids)!r}; durable token_ids={sorted(durable_token_ids)!r}."
                    f"{scope_note}"
                )

            if intake_snapshot_token_ids is not None:
                unknown_snapshot_token_ids = intake_snapshot_token_ids - durable_token_ids
                if unknown_snapshot_token_ids:
                    if scope_row_id is not None:
                        # Scope validation (§E.3 defence): a snapshot token that IS
                        # durably BLOCKED under this barrier_key but in a DIFFERENT
                        # row group is a cross-group mis-inclusion by the flush
                        # caller — name it precisely, never silently intersect.
                        cross_group_rows = conn.execute(
                            select(token_work_items_table.c.token_id, token_work_items_table.c.row_id)
                            .where(token_work_items_table.c.run_id == run_id)
                            .where(token_work_items_table.c.barrier_key == barrier_key)
                            .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                            .where(token_work_items_table.c.token_id.in_(sorted(unknown_snapshot_token_ids)))
                        ).all()
                        cross_group = {row.token_id: row.row_id for row in cross_group_rows if row.row_id != scope_row_id}
                        if cross_group:
                            raise AuditIntegrityError(
                                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                                f"scope_row_id={scope_row_id!r} received intake snapshot token(s) whose durable "
                                f"BLOCKED rows belong to a DIFFERENT row group: "
                                f"{dict(sorted(cross_group.items()))!r}; the "
                                "flush caller built its firing-group snapshot across row groups (ADR-030 §E.3 "
                                "scope validation)."
                            )
                    raise AuditIntegrityError(
                        f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} received "
                        f"intake snapshot token(s) with no durable BLOCKED row under the barrier: "
                        f"{sorted(unknown_snapshot_token_ids)!r}; the leader believes in a token the journal does "
                        f"not hold.{scope_note}"
                    )

            passthrough_emissions: list[BarrierEmission] = []
            fresh_emissions: list[BarrierEmission] = []
            for emission in emitted_pending_sink:
                matching_rows = blocked_by_token[emission.token_id] if emission.token_id in blocked_by_token else []
                if not matching_rows:
                    # Strict/legacy partition: on the strict arm (the F1 verb)
                    # an unbuffered sink-bound emission is a fresh terminal-lane
                    # INSERT; on the legacy wrapper arm it is a refused partial
                    # handoff (mark_blocked_barrier_pending_sink_many's pinned
                    # behavior).
                    if require_exhaustive_release:
                        fresh_emissions.append(emission)
                        continue
                    raise AuditIntegrityError(
                        f"Scheduler barrier pending-sink handoff for run_id={run_id!r} barrier_key={barrier_key!r} "
                        f"is missing token_id={emission.token_id!r}; refusing partial sink handoff."
                    )
                if len(matching_rows) != 1:
                    raise AuditIntegrityError(
                        f"Scheduler barrier pending-sink handoff for run_id={run_id!r} barrier_key={barrier_key!r} "
                        f"token_id={emission.token_id!r} found {len(matching_rows)} matching rows; expected exactly one."
                    )
                passthrough_emissions.append(emission)

            if require_exhaustive_release:
                handed_off_token_ids = frozenset(emission.token_id for emission in passthrough_emissions)
                if intake_snapshot_token_ids is not None:
                    handed_off_outside_snapshot = handed_off_token_ids - intake_snapshot_token_ids
                    if handed_off_outside_snapshot:
                        raise AuditIntegrityError(
                            f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} handed "
                            f"off buffered token(s) outside its own intake snapshot: "
                            f"{sorted(handed_off_outside_snapshot)!r}; the flush caller may only hand off tokens it "
                            "durably adopted into this firing group (ADR-030 §E.3)."
                        )
                    # §E.3: the exhaustiveness universe is the firing group the
                    # leader adopted. Durable BLOCKED rows outside the snapshot
                    # are late arrivals — they legitimately stay BLOCKED and are
                    # recorded on every emission event for forensics.
                    required_token_ids = durable_token_ids & intake_snapshot_token_ids
                    late_arrival_token_ids = durable_token_ids - intake_snapshot_token_ids
                    if late_arrival_token_ids:
                        emission_context["late_arrival_token_ids"] = sorted(late_arrival_token_ids)
                else:
                    required_token_ids = durable_token_ids
                uncovered_token_ids = required_token_ids - consumed_set - handed_off_token_ids
                if uncovered_token_ids:
                    raise AuditIntegrityError(
                        f"Scheduler barrier completion mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                        f"durable BLOCKED rows hold {len(uncovered_token_ids)} token(s) neither consumed nor handed "
                        f"off; the completion would orphan them. uncovered token_ids={sorted(uncovered_token_ids)!r}; "
                        f"consumed token_ids={sorted(consumed_set)!r}; handoff token_ids={sorted(handed_off_token_ids)!r}."
                        f"{scope_note}"
                    )

            terminalized = self._terminalize_consumed_barrier_rows(
                conn,
                run_id=run_id,
                barrier_key=barrier_key,
                consumed=consumed,
                blocked_by_token=blocked_by_token,
                now=now,
                release_context=release_context,
            )
            self._transition_passthrough_pending_sink(
                conn,
                run_id=run_id,
                barrier_key=barrier_key,
                blocked_rows=blocked_rows,
                passthrough_emissions=passthrough_emissions,
                emission_context=emission_context,
                now=now,
                parked_lease_owner=pending_sink_lease_owner,
            )
            for emission in fresh_emissions:
                self._insert_fresh_pending_sink_emission(
                    conn,
                    run_id=run_id,
                    barrier_key=barrier_key,
                    emission=emission,
                    emission_context=emission_context,
                    now=now,
                    parked_lease_owner=pending_sink_lease_owner,
                )
            for emission in emitted_ready:
                self._insert_ready_emission(
                    conn,
                    run_id=run_id,
                    barrier_key=barrier_key,
                    emission=emission,
                    emission_context=emission_context,
                    now=now,
                )
            for loss in branch_losses:
                # §E.5 record-then-notify: the durable loss record commits iff
                # this barrier completion (the branch's disposition) commits.
                record_coalesce_branch_loss(
                    conn,
                    run_id=run_id,
                    coalesce_name=loss.coalesce_name,
                    row_id=loss.row_id,
                    branch_name=loss.branch_name,
                    token_id=loss.token_id,
                    reason=loss.reason,
                    recorded_by=loss.recorded_by,
                    now=now,
                )
        return terminalized

    def _terminalize_consumed_barrier_rows(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        consumed: tuple[str, ...],
        blocked_by_token: Mapping[str, list[RowMapping]],
        now: datetime,
        release_context: Mapping[str, object] | None = None,
    ) -> int:
        """BLOCKED -> TERMINAL for the consumed set (legacy terminalization arm).

        ``release_context`` (§E.3a) is merged into each per-row
        ``MARK_BLOCKED_BARRIER_TERMINAL`` event context; ``None`` keeps the
        pinned ``{"barrier_key"}``-only legacy shape.
        """
        terminal_event_context: dict[str, object] = {"barrier_key": barrier_key}
        if release_context is not None:
            terminal_event_context.update(release_context)
        consumed_set = frozenset(consumed)
        candidate_rows = sorted(
            (row for token_id in consumed_set for row in blocked_by_token[token_id]),
            key=lambda row: (row["token_id"], row["work_item_id"]),
        )
        terminalized = 0
        for row in candidate_rows:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.token_id.in_(consumed))
                .values(
                    status=TokenWorkStatus.TERMINAL.value,
                    row_payload_json=scrubbed_row_payload_json(barrier_key),
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            if result.rowcount == 1:
                self._events.record(
                    conn,
                    event_type=SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL,
                    run_id=run_id,
                    token_id=row["token_id"],
                    work_item_id=row["work_item_id"],
                    node_id=row["node_id"],
                    from_status=TokenWorkStatus.BLOCKED,
                    to_status=TokenWorkStatus.TERMINAL,
                    from_lease_owner=row["lease_owner"],
                    to_lease_owner=None,
                    from_attempt=row["attempt"],
                    to_attempt=row["attempt"],
                    recorded_at=now,
                    from_lease_expires_at=row["lease_expires_at"],
                    to_lease_expires_at=None,
                    context=terminal_event_context,
                )
                terminalized += 1
            elif result.rowcount not in (0, None):
                raise AuditIntegrityError(
                    f"Scheduler barrier terminalization affected {result.rowcount} rows for "
                    f"run_id={run_id!r} barrier_key={barrier_key!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                )
        if consumed_set and terminalized != len(consumed_set):
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                f"live consumed {len(consumed_set)} token(s), but durable scheduler terminalized {terminalized}."
            )
        return terminalized

    def _transition_passthrough_pending_sink(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        blocked_rows: Sequence[RowMapping],
        passthrough_emissions: Sequence[BarrierEmission],
        emission_context: Mapping[str, object],
        now: datetime,
        parked_lease_owner: str | None = None,
    ) -> None:
        """BLOCKED -> PENDING_SINK in place for buffered passthrough tokens (legacy handoff arm).

        ``parked_lease_owner`` is the attributed-park stamp (ADR-030): the
        handed-off row is parked owner-attributed but not leased
        (``lease_expires_at`` stays NULL) so the strict post-sink owner CAS
        can terminalize it.
        """
        emission_by_token = {emission.token_id: emission for emission in passthrough_emissions}
        if not emission_by_token:
            return
        transitioned = 0
        for row in blocked_rows:
            emission = emission_by_token.get(row["token_id"])
            if emission is None:
                continue
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == row["work_item_id"])
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.token_id == row["token_id"])
                .values(
                    status=TokenWorkStatus.PENDING_SINK.value,
                    row_payload_json=emission.row_payload_json,
                    pending_sink_name=emission.sink_name,
                    pending_outcome=emission.outcome,
                    pending_path=emission.path,
                    pending_error_hash=emission.error_hash,
                    pending_error_message=emission.error_message,
                    lease_owner=parked_lease_owner,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            if result.rowcount == 1:
                self._events.record(
                    conn,
                    event_type=SchedulerEventType.MARK_PENDING_SINK,
                    run_id=run_id,
                    token_id=row["token_id"],
                    work_item_id=row["work_item_id"],
                    node_id=row["node_id"],
                    from_status=TokenWorkStatus.BLOCKED,
                    to_status=TokenWorkStatus.PENDING_SINK,
                    from_lease_owner=row["lease_owner"],
                    to_lease_owner=parked_lease_owner,
                    from_attempt=row["attempt"],
                    to_attempt=row["attempt"],
                    recorded_at=now,
                    from_lease_expires_at=row["lease_expires_at"],
                    to_lease_expires_at=None,
                    context=emission_context,
                )
                transitioned += 1
            elif result.rowcount not in (0, None):
                raise AuditIntegrityError(
                    f"Scheduler barrier pending-sink handoff affected {result.rowcount} rows for "
                    f"run_id={run_id!r} barrier_key={barrier_key!r} work_item_id={row['work_item_id']!r}; expected 0 or 1."
                )
        if transitioned != len(emission_by_token):
            raise AuditIntegrityError(
                f"Scheduler barrier pending-sink handoff mismatch for run_id={run_id!r} barrier_key={barrier_key!r}: "
                f"requested {len(emission_by_token)} token(s), transitioned {transitioned}."
            )

    def _insert_fresh_pending_sink_emission(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        emission: BarrierEmission,
        emission_context: Mapping[str, object],
        now: datetime,
        parked_lease_owner: str | None = None,
    ) -> None:
        """INSERT a fresh PENDING_SINK row on the node_id-NULL terminal lane.

        ``parked_lease_owner``: attributed-park stamp (ADR-030); see
        :meth:`_transition_passthrough_pending_sink`.
        """
        if emission.node_id is not None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"received fresh pending-sink emission token_id={emission.token_id!r} with "
                f"node_id={emission.node_id!r}; fresh sink-bound emissions live on the node_id-NULL terminal lane."
            )
        if emission.row_id is None or emission.step_index is None or emission.ingest_sequence is None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"fresh pending-sink emission token_id={emission.token_id!r} requires row_id, step_index "
                "and ingest_sequence; the inserted journal row must be a complete resume cursor."
            )
        validate_work_item_references(
            conn,
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            ingest_sequence=emission.ingest_sequence,
            node_id=None,
            coalesce_node_id=emission.coalesce_node_id,
        )
        work_item_id = make_work_item_id(run_id, emission.token_id, None, emission.attempt)
        values: dict[str, object] = {
            "work_item_id": work_item_id,
            "run_id": run_id,
            "token_id": emission.token_id,
            "row_id": emission.row_id,
            "node_id": None,
            "step_index": emission.step_index,
            "ingest_sequence": emission.ingest_sequence,
            "row_payload_json": emission.row_payload_json,
            "status": TokenWorkStatus.PENDING_SINK.value,
            "queue_key": emission.queue_key,
            "barrier_key": emission.barrier_key,
            "on_success_sink": emission.on_success_sink,
            "pending_sink_name": emission.sink_name,
            "pending_outcome": emission.outcome,
            "pending_path": emission.path,
            "pending_error_hash": emission.error_hash,
            "pending_error_message": emission.error_message,
            "branch_name": emission.branch_name,
            "fork_group_id": emission.fork_group_id,
            "join_group_id": emission.join_group_id,
            "expand_group_id": emission.expand_group_id,
            "coalesce_node_id": emission.coalesce_node_id,
            "coalesce_name": emission.coalesce_name,
            "attempt": emission.attempt,
            "lease_owner": parked_lease_owner,
            "lease_expires_at": None,
            "available_at": now,
            "created_at": now,
            "updated_at": now,
        }
        insert_work_item(conn, values=values, operation="barrier-completion PENDING_SINK emission")
        self._events.record(
            conn,
            event_type=SchedulerEventType.MARK_PENDING_SINK,
            run_id=run_id,
            token_id=emission.token_id,
            work_item_id=work_item_id,
            node_id=None,
            from_status=None,
            to_status=TokenWorkStatus.PENDING_SINK,
            from_lease_owner=None,
            to_lease_owner=parked_lease_owner,
            from_attempt=None,
            to_attempt=emission.attempt,
            recorded_at=now,
            context=emission_context,
        )

    def _insert_ready_emission(
        self,
        conn: Connection,
        *,
        run_id: str,
        barrier_key: str,
        emission: BarrierEmission,
        emission_context: Mapping[str, object],
        now: datetime,
    ) -> None:
        """INSERT a READY continuation emitted by a barrier completion."""
        if emission.row_id is None or emission.step_index is None or emission.ingest_sequence is None:
            raise AuditIntegrityError(
                f"Scheduler barrier completion for run_id={run_id!r} barrier_key={barrier_key!r} "
                f"ready emission token_id={emission.token_id!r} requires row_id, step_index "
                "and ingest_sequence; the inserted journal row must be a complete resume cursor."
            )
        validate_work_item_references(
            conn,
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            ingest_sequence=emission.ingest_sequence,
            node_id=emission.node_id,
            coalesce_node_id=emission.coalesce_node_id,
        )
        values = ready_work_item_values(
            run_id=run_id,
            token_id=emission.token_id,
            row_id=emission.row_id,
            node_id=emission.node_id,
            step_index=emission.step_index,
            ingest_sequence=emission.ingest_sequence,
            row_payload_json=emission.row_payload_json,
            available_at=now,
            attempt=emission.attempt,
            queue_key=emission.queue_key,
            barrier_key=emission.barrier_key,
            on_success_sink=emission.on_success_sink,
            branch_name=emission.branch_name,
            fork_group_id=emission.fork_group_id,
            join_group_id=emission.join_group_id,
            expand_group_id=emission.expand_group_id,
            coalesce_node_id=emission.coalesce_node_id,
            coalesce_name=emission.coalesce_name,
        )
        insert_work_item(conn, values=values, operation="barrier-completion READY emission")
        self._events.record(
            conn,
            event_type=SchedulerEventType.ENQUEUE,
            run_id=run_id,
            token_id=emission.token_id,
            work_item_id=str(values["work_item_id"]),
            node_id=emission.node_id,
            from_status=None,
            to_status=TokenWorkStatus.READY,
            from_lease_owner=None,
            to_lease_owner=None,
            from_attempt=None,
            to_attempt=emission.attempt,
            recorded_at=now,
            context=emission_context,
        )

    def mark_blocked_barrier_pending_sink_many(
        self,
        *,
        run_id: str,
        barrier_key: str,
        handoffs: Mapping[str, BlockedPendingSinkHandoff],
        now: datetime,
        coordination_token: CoordinationToken | None = None,
        pending_sink_lease_owner: str | None = None,
    ) -> int:
        """Move BLOCKED barrier work to PENDING_SINK before external sink writes.

        Delegating wrapper over :meth:`complete_barrier` (F1 Task 2.3). The
        legacy arm (``require_exhaustive_release=False``) preserves the pinned
        semantics: BLOCKED rows not named in ``handoffs`` stay BLOCKED, a
        handoff token without a BLOCKED row is refused (no fresh inserts), and
        handoff events keep the ``{"barrier_key"}``-only context.
        """
        requested_token_ids = tuple(handoffs.keys())
        if not requested_token_ids:
            return 0
        emissions = tuple(
            BarrierEmission(
                token_id=token_id,
                row_payload_json=handoff.row_payload_json,
                sink_name=handoff.sink_name,
                outcome=handoff.outcome,
                path=handoff.path,
                error_hash=handoff.error_hash,
                error_message=handoff.error_message,
            )
            for token_id, handoff in handoffs.items()
        )
        self.complete_barrier(
            run_id=run_id,
            barrier_key=barrier_key,
            consumed_token_ids=(),
            emitted_pending_sink=emissions,
            emitted_ready=(),
            now=now,
            require_exhaustive_release=False,
            coordination_token=coordination_token,  # type: ignore[arg-type]  # legacy wrapper: stays Optional (slice-4 deferred)
            pending_sink_lease_owner=pending_sink_lease_owner,
        )
        # complete_barrier raised unless every requested handoff transitioned.
        return len(requested_token_ids)

    def mark_blocked_barrier_terminal(
        self,
        *,
        run_id: str,
        barrier_key: str,
        token_ids: tuple[str, ...],
        now: datetime,
        coordination_token: CoordinationToken | None = None,
        release_context: Mapping[str, object] | None = None,
    ) -> int:
        """Mark BLOCKED work consumed by a resolved barrier as terminal.

        Delegating wrapper over :meth:`complete_barrier` (F1 Task 2.3). The
        legacy arm (``require_exhaustive_release=False``) preserves the pinned
        partial-release semantics: BLOCKED rows under the barrier that are not
        named in ``token_ids`` stay BLOCKED.

        ``release_context`` (ADR-030 §E.3a, additive): extra forensic keys
        merged into each per-row ``MARK_BLOCKED_BARRIER_TERMINAL`` event
        context — the late-arrival journal-release call sites pass
        ``{"late_arrival": True, "reason": ..., "released_by": ...,
        "scope_row_id": ...}``. ``None`` (every pre-§E.3a caller) preserves
        the pinned ``{"barrier_key"}``-only legacy context exactly.
        """
        if not token_ids:
            raise AuditIntegrityError(
                f"Scheduler barrier terminalization for run_id={run_id!r} barrier_key={barrier_key!r} "
                "requires at least one live token_id; refusing to terminalize durable BLOCKED rows without "
                "live barrier evidence."
            )
        return self.complete_barrier(
            run_id=run_id,
            barrier_key=barrier_key,
            consumed_token_ids=token_ids,
            emitted_pending_sink=(),
            emitted_ready=(),
            now=now,
            require_exhaustive_release=False,
            coordination_token=coordination_token,  # type: ignore[arg-type]  # legacy wrapper: stays Optional (slice-4 deferred)
            release_context=release_context,
        )

    def adopt_blocked_barrier_item(
        self,
        *,
        run_id: str,
        work_item_id: str,
        token_id: str,
        barrier_key: str,
        membership: BatchMembershipSpec | None,
        buffered_outcome: BufferedOutcomeSpec | None,
        now: datetime,
        coordination_token: CoordinationToken,
    ) -> BarrierAdoptionResult:
        """Fenced, backdated adoption of one durable BLOCKED barrier hold (§E.2).

        The journal-first intake verb: the leader adopts a follower-deposited
        (or its own) BLOCKED row into executor memory by first making the
        adoption durable — epoch fence, then the
        ``barrier_adopted_epoch NULL → epoch`` CAS, then (aggregation arm)
        the ``batch_members`` row and the BUFFERED ``token_outcomes`` row —
        all in ONE ``BEGIN IMMEDIATE`` transaction. ``coordination_token`` is
        REQUIRED (new-verb doctrine; no unfenced arm).

        ``token_outcomes`` has NO non-terminal uniqueness, so the adoption
        CAS is the ONLY structural guard against double-BUFFERED (design
        §C.4 row 6a): rowcount 0 with the marker already non-NULL is the
        idempotent success-SKIP (``adopted=False``) and the caller MUST skip
        both the memory accept and the audit writes on that arm. Any
        non-NULL epoch — this one or an earlier one — counts as adopted:
        §E.4 treats adopted-at-any-epoch rows as restorable members and the
        intake filter is ``barrier_adopted_epoch IS NULL``.

        Backdated accept timing (§E.2): the BUFFERED outcome's
        ``recorded_at`` is the row's durable ``barrier_blocked_at`` arrival
        stamp, NOT ``now`` — the audit's accept instant is invariant under
        leader takeover. Honest provenance rides ``context_json``
        (``adopted_epoch`` / ``adopted_at``).

        Aggregation callers pass BOTH specs; coalesce callers pass
        ``None``/``None`` (their held-arrival durable bookkeeping is a
        node_states row; the adoption payload is the CAS marker alone).
        Mixed specs raise Tier-1. Note the FIRST member's ``batches`` row is
        created by the caller in its own transaction before this verb: a
        deposed leader can durably create one orphan DRAFT batches row —
        accepted residue (no members or outcomes reference it; restore
        derives the batch from BUFFERED outcomes, not DRAFT batches rows).

        A crash mid-transaction rolls everything back and the row stays
        intake-pending (``barrier_adopted_epoch IS NULL``) — the legitimate
        restore-reconcile disposition (design §E.2; crash-walk "leader crash
        mid-adoption").
        """
        if (membership is None) != (buffered_outcome is None):
            raise AuditIntegrityError(
                f"Scheduler barrier adoption for run_id={run_id!r} work_item_id={work_item_id!r} "
                f"token_id={token_id!r} received membership={'set' if membership is not None else 'None'} with "
                f"buffered_outcome={'set' if buffered_outcome is not None else 'None'}; the aggregation arm "
                "requires BOTH specs and the coalesce arm requires NEITHER — a mixed adoption would write a "
                "batch membership without its BUFFERED witness (or vice versa)."
            )
        if membership is not None and buffered_outcome is not None and membership.batch_id != buffered_outcome.batch_id:
            raise AuditIntegrityError(
                f"Scheduler barrier adoption for run_id={run_id!r} token_id={token_id!r} received "
                f"membership.batch_id={membership.batch_id!r} but buffered_outcome.batch_id="
                f"{buffered_outcome.batch_id!r}; one adoption belongs to exactly one batch."
            )
        epoch = coordination_token.leader_epoch
        with fenced_leader_transaction(
            self._engine,
            token=coordination_token,
            now=now,
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="adopt_blocked_barrier_item",
        ) as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.work_item_id == work_item_id)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.token_id == token_id)
                .where(token_work_items_table.c.barrier_key == barrier_key)
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .where(token_work_items_table.c.barrier_adopted_epoch.is_(None))
                .values(barrier_adopted_epoch=epoch, updated_at=now)
            )
            if result.rowcount != 1:
                row = (
                    conn.execute(
                        select(
                            token_work_items_table.c.status,
                            token_work_items_table.c.token_id,
                            token_work_items_table.c.barrier_key,
                            token_work_items_table.c.barrier_adopted_epoch,
                        )
                        .where(token_work_items_table.c.work_item_id == work_item_id)
                        .where(token_work_items_table.c.run_id == run_id)
                    )
                    .mappings()
                    .one_or_none()
                )
                if (
                    row is not None
                    and row["status"] == TokenWorkStatus.BLOCKED.value
                    and row["token_id"] == token_id
                    and row["barrier_key"] == barrier_key
                    and row["barrier_adopted_epoch"] is not None
                ):
                    # Idempotent success-SKIP: already adopted (this epoch or an
                    # earlier one). The caller skips memory and audit writes.
                    return BarrierAdoptionResult(
                        adopted=False,
                        barrier_adopted_epoch=int(row["barrier_adopted_epoch"]),
                        outcome_id=None,
                    )
                observed = (
                    "missing"
                    if row is None
                    else (
                        f"status={row['status']!r}, token_id={row['token_id']!r}, "
                        f"barrier_key={row['barrier_key']!r}, barrier_adopted_epoch={row['barrier_adopted_epoch']!r}"
                    )
                )
                raise AuditIntegrityError(
                    f"Scheduler barrier adoption CAS for run_id={run_id!r} work_item_id={work_item_id!r} expected a "
                    f"BLOCKED hold with token_id={token_id!r} under barrier_key={barrier_key!r}, but the journal row "
                    f"is {observed}. The barrier plane is leader-owned between intake listing and adoption and "
                    "followers can only ADD blocked rows — a vanished or mutated row is journal corruption."
                )
            outcome_id: str | None = None
            if membership is not None:
                batch_run_id = conn.execute(
                    select(batches_table.c.run_id).where(batches_table.c.batch_id == membership.batch_id)
                ).scalar_one_or_none()
                if batch_run_id is None:
                    raise AuditIntegrityError(
                        f"Scheduler barrier adoption for run_id={run_id!r} token_id={token_id!r} cannot add batch "
                        f"member: batch {membership.batch_id!r} not found."
                    )
                if batch_run_id != run_id:
                    raise AuditIntegrityError(
                        f"Scheduler barrier adoption for run_id={run_id!r} token_id={token_id!r} cannot add batch "
                        f"member: batch {membership.batch_id!r} belongs to run {batch_run_id!r}."
                    )
                conn.execute(
                    insert(batch_members_table).values(
                        batch_id=membership.batch_id,
                        run_id=run_id,
                        token_id=token_id,
                        ordinal=membership.ordinal,
                    )
                )
            if buffered_outcome is not None:
                if not buffered_outcome.batch_id:
                    # ADR-019 BUFFERED rule replicated: batch_id REQUIRED for path='buffered'.
                    raise AuditIntegrityError(
                        f"Scheduler barrier adoption for run_id={run_id!r} token_id={token_id!r} requires a "
                        "non-empty buffered_outcome.batch_id (ADR-019: (NULL, BUFFERED) requires batch_id)."
                    )
                barrier_blocked_at = conn.execute(
                    select(token_work_items_table.c.barrier_blocked_at).where(token_work_items_table.c.work_item_id == work_item_id)
                ).scalar_one()
                if barrier_blocked_at is None:
                    raise AuditIntegrityError(
                        f"Scheduler barrier adoption for run_id={run_id!r} work_item_id={work_item_id!r} found a "
                        "BLOCKED barrier hold with no barrier_blocked_at arrival stamp; the backdated BUFFERED "
                        "accept instant cannot be derived — journal corruption (mark_blocked stamps every hold)."
                    )
                if barrier_blocked_at.tzinfo is None:
                    barrier_blocked_at = barrier_blocked_at.replace(tzinfo=UTC)
                caller_context = {} if buffered_outcome.context is None else dict(buffered_outcome.context)
                outcome_id = f"out_{generate_id()[:12]}"
                conn.execute(
                    insert(token_outcomes_table).values(
                        outcome_id=outcome_id,
                        run_id=run_id,
                        token_id=token_id,
                        outcome=None,
                        path=TerminalPath.BUFFERED.value,
                        completed=0,
                        recorded_at=barrier_blocked_at,
                        batch_id=buffered_outcome.batch_id,
                        context_json=canonical_json(
                            {
                                **caller_context,
                                # Honest provenance — caller context cannot mask it.
                                "adopted_epoch": epoch,
                                "adopted_at": now.isoformat(),
                            }
                        ),
                    )
                )
        return BarrierAdoptionResult(adopted=True, barrier_adopted_epoch=epoch, outcome_id=outcome_id)

    def list_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        """Return BLOCKED barrier holds for a run in deterministic order.

        The dual-use BLOCKED partition (D1) lives in the shared
        ``blocked_barrier_hold_clause`` predicate — see its docstring.

        Iteration order is ``(barrier_key, ingest_sequence, work_item_id)``
        for determinism only; buffer ORDER at restore comes from
        ``batch_members.ordinal``, not from this verb. Read-only: no
        scheduler_event is recorded.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(blocked_barrier_hold_clause())
                    .order_by(
                        token_work_items_table.c.barrier_key,
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
        return [item_from_mapping(row) for row in rows]

    def blocked_barrier_token_ids(self, *, run_id: str) -> frozenset[str]:
        """Return token IDs currently held by journal BLOCKED barrier rows."""
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table.c.token_id)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(blocked_barrier_hold_clause())
                )
                .scalars()
                .all()
            )
        return frozenset(rows)

    def count_blocked_barrier_items(self, *, run_id: str) -> int:
        """Count journal BLOCKED barrier holds for a run."""
        with self._engine.connect() as conn:
            result = conn.execute(
                select(func.count())
                .select_from(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(blocked_barrier_hold_clause())
            ).scalar_one()
        return int(result)

    def list_pending_blocked_barrier_items(self, *, run_id: str) -> list[TokenWorkItem]:
        """Return intake-pending BLOCKED barrier holds (``barrier_adopted_epoch IS NULL``).

        ADR-030 §E.2 intake scan shape: the per-iteration journal-first intake
        ONLY needs rows that have not yet been adopted by any leader epoch.
        Already-adopted rows (``barrier_adopted_epoch IS NOT NULL``) are already
        in executor memory and returning them here would cause O(N²/2)
        full-row hydrations as a batch fills — N=1000 → ~500 k payload-bearing
        fetches per drain iteration.

        The ``has_blocked_barrier_work`` EOF gate uses the non-filtered
        ``list_blocked_barrier_items`` (all epochs) because the quiescence
        predicate must wait until ALL BLOCKED rows are terminalized, not just
        the pending-epoch ones.
        """
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    select(token_work_items_table)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(blocked_barrier_hold_clause())
                    .where(token_work_items_table.c.barrier_adopted_epoch.is_(None))
                    .order_by(
                        token_work_items_table.c.barrier_key,
                        token_work_items_table.c.ingest_sequence,
                        token_work_items_table.c.work_item_id,
                    )
                )
                .mappings()
                .all()
            )
        return [item_from_mapping(row) for row in rows]

    def reset_adoption_marker_to_pending(
        self,
        *,
        work_item_ids: Sequence[str],
        run_id: str,
    ) -> int:
        """Reset ``barrier_adopted_epoch`` to NULL for crash-window BLOCKED rows.

        ADR-030 §E.3/§E.4 crash-window recovery at restore: a BLOCKED coalesce
        row whose adoption CAS committed but whose accept() never wrote the
        PENDING hold node_state must be re-classified as intake-pending so the
        new leader's first journal-first intake adopts it properly and runs the
        full accept-then-trigger path (including merge, failure and late-arrival
        handling) — outcomes the post-restore phase cannot safely produce.

        Resets ONLY BLOCKED rows (completed/terminal rows are left alone).
        Returns the count of rows actually reset (should equal len(work_item_ids)
        unless a row transitioned out of BLOCKED between the holdless-detection
        read and this call — a non-matching count is logged but not fatal because
        the next intake pass will re-classify the row correctly).

        Called from ``BarrierRecoveryCoordinator.restore_from_journal`` for holdless non-completed
        rows (before ``restore_from_journal`` runs, so no executor state is
        touched).  The operation is epoch-fence-free (it runs before the new
        leader's first fenced verb and its safety derives from the takeover CAS
        already having committed — any concurrent old-leader adoption attempt
        would fail the CAS, and the new leader is the only actor with write
        access at this point).
        """
        if not work_item_ids:
            return 0
        with begin_write(self._engine) as conn:
            result = conn.execute(
                update(token_work_items_table)
                .where(token_work_items_table.c.run_id == run_id)
                .where(token_work_items_table.c.work_item_id.in_(list(work_item_ids)))
                .where(token_work_items_table.c.status == TokenWorkStatus.BLOCKED.value)
                .values(barrier_adopted_epoch=None)
            )
        return result.rowcount
