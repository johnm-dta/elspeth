# tests/property/engine/test_scheduler_work_item_lifecycle_state_machine.py
"""Property-based stateful tests for the durable token-scheduler work-item lifecycle.

SCHEDULER WORK-ITEM STATE MACHINE (``TokenWorkStatus``, ADR-026):

    READY --claim_ready--> LEASED --mark_{terminal,failed}--> TERMINAL | FAILED
                            |  \\--mark_pending_sink--> PENDING_SINK
                            \\---mark_blocked(barrier)--> BLOCKED
    PENDING_SINK --claim_pending_sink--> LEASED (pending_sink_name preserved)
    PENDING_SINK / LEASED+pending_sink --mark_pending_sink_terminal--> TERMINAL
    BLOCKED --barrier release--> TERMINAL | PENDING_SINK (only via its own barrier_key)
    LEASED (expired, peer sweep) --recover_expired_leases--> READY (attempt+1,
        work_item_id rotated) | PENDING_SINK (attempt and work_item_id PRESERVED)

Each rule drives the REAL ``TokenSchedulerRepository`` over a real SQLite
engine and mirrors the call in a Python model; invariants then prove model/DB
agreement plus the global lifecycle properties:

1. Conservation — no work item is lost or duplicated (exactly one durable row
   per scheduled token, exactly the tokens the model scheduled).
2. No claim under a live lease — ``claim_ready``/``claim_pending_sink`` only
   ever return the deterministically-ordered next unleased candidate; an
   unexpired LEASED row is never handed to a second owner.
3. READY never jumps directly to TERMINAL — every terminalization event in
   the durable scheduler_events journal departs from LEASED, BLOCKED, or
   PENDING_SINK, and ``mark_terminal`` on a READY row raises.
4. ``lease_owner``/``lease_expires_at`` are non-NULL iff status is LEASED.
5. Expired-lease recovery bumps ``attempt`` and rotates ``work_item_id``,
   EXCEPT the PENDING_SINK recovery path, which preserves both (the sink
   handoff is already durable; no transform work is replayed).
6. BLOCKED is released only via its own barrier_key; a release naming a
   different barrier raises and changes nothing.

WAITING is NOT modelled: the lane was deleted outright (F4, 2026-06-10)
after verification that ``mark_waiting`` had no production caller and the
``release_waiting`` maintenance sweep serviced an unreachable state.
The single-row-token lifecycle (fork/coalesce lineage) is covered separately
by test_token_lifecycle_state_machine.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule
from sqlalchemy import create_engine, insert, select

from elspeth.contracts import NodeType
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunLeadershipLostError
from elspeth.contracts.scheduler import BlockedPendingSinkHandoff, TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_coordination_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-rc6-scheduler-property"
NODE_ID = "normalize"
SOURCE_NODE_ID = "source-a"
SINK_NAME = "sink-a"
LEASE_SECONDS = 60

WORKERS = ("worker-a", "worker-b")
BARRIERS = ("barrier-east", "barrier-west")
# ADR-030 slice 3: the leader identity whose seat the fenced adoption verbs
# verify (epoch 1, minted with the run row below).
LEADER_WORKER_ID = "machine-leader"
LEADER_TOKEN = CoordinationToken(run_id=RUN_ID, worker_id=LEADER_WORKER_ID, leader_epoch=1)
STALE_LEADER_TOKEN = CoordinationToken(run_id=RUN_ID, worker_id=LEADER_WORKER_ID, leader_epoch=99)

# Statuses from which a work item may never transition again.
FINAL_STATUSES = {TokenWorkStatus.TERMINAL, TokenWorkStatus.FAILED}


def _make_scheduler_engine() -> Tier1Engine:
    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _row_payload_json() -> str:
    return TokenSchedulerRepository.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))


def _insert_run_and_nodes(engine: Tier1Engine, *, now: datetime) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=now,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        # ADR-030 epoch-21 seat: the fenced adoption verbs (slice 3) verify
        # identity+epoch against this row.
        conn.execute(
            insert(run_coordination_table).values(
                run_id=RUN_ID,
                leader_worker_id=LEADER_WORKER_ID,
                leader_epoch=1,
                leader_heartbeat_expires_at=now + timedelta(days=1),
                updated_at=now,
            )
        )
        for node_id, node_type, plugin in (
            (SOURCE_NODE_ID, NodeType.SOURCE, "csv"),
            (NODE_ID, NodeType.TRANSFORM, "identity"),
        ):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin,
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=now,
                )
            )


def _insert_row_and_token(engine: Tier1Engine, *, sequence: int, now: datetime) -> tuple[str, str]:
    token_id = f"token-{sequence}"
    row_id = f"row-{sequence}"
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id=row_id,
                run_id=RUN_ID,
                source_node_id=SOURCE_NODE_ID,
                row_index=sequence,
                source_row_index=sequence,
                ingest_sequence=sequence,
                source_data_hash=f"hash-{row_id}",
                created_at=now,
            )
        )
        conn.execute(insert(tokens_table).values(token_id=token_id, row_id=row_id, run_id=RUN_ID, created_at=now))
    return token_id, row_id


@dataclass
class ModelItem:
    """Model mirror of one durable token_work_items row."""

    token_id: str
    work_item_id: str
    ingest_sequence: int
    status: TokenWorkStatus
    attempt: int = 1
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    pending_sink_name: str | None = None
    barrier_key: str | None = None
    # ADR-030 §E.2 adoption marker: NULL until the leader's journal-first
    # intake adopts the BLOCKED barrier hold; persists across the release.
    barrier_adopted_epoch: int | None = None


class SchedulerWorkItemLifecycleStateMachine(RuleBasedStateMachine):
    """Stateful property tests driving the real TokenSchedulerRepository."""

    tokens = Bundle("tokens")

    def __init__(self) -> None:
        super().__init__()
        self.engine = _make_scheduler_engine()
        self.repo = TokenSchedulerRepository(self.engine)
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.payload = _row_payload_json()
        self.model: dict[str, ModelItem] = {}
        self.next_sequence = 0
        _insert_run_and_nodes(self.engine, now=self.now)

    def teardown(self) -> None:
        self.engine.dispose()

    # -------------------------------------------------------------------------
    # Setup / helpers
    # -------------------------------------------------------------------------

    def _tick(self) -> datetime:
        """Advance the logical clock; every rule acts at a distinct instant."""
        self.now += timedelta(seconds=1)
        return self.now

    def _db_rows(self) -> dict[str, dict[str, object]]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(token_work_items_table).where(token_work_items_table.c.run_id == RUN_ID)).mappings().all()
        by_token: dict[str, dict[str, object]] = {}
        for row in rows:
            assert row["token_id"] not in by_token, f"Duplicate work item rows for token {row['token_id']}"
            by_token[row["token_id"]] = dict(row)
        return by_token

    def _next_claimable(self, status: TokenWorkStatus) -> ModelItem | None:
        """The deterministic next claim target: lowest ingest_sequence among ``status`` items.

        Production orders by (ingest_sequence, step_index, created_at,
        work_item_id); every modelled token has a unique ingest_sequence and a
        constant step_index, so ingest_sequence alone is disambiguating here
        (the tie legs are pinned by test_rc6_scheduler_ordering_characterization).
        """
        candidates = [item for item in self.model.values() if item.status is status]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.ingest_sequence)

    def _blocked_at(self, barrier_key: str) -> list[ModelItem]:
        return [item for item in self.model.values() if item.status is TokenWorkStatus.BLOCKED and item.barrier_key == barrier_key]

    # -------------------------------------------------------------------------
    # Rules: enqueue
    # -------------------------------------------------------------------------

    @rule(target=tokens)
    def enqueue(self) -> str:
        """Persist a new row/token pair and enqueue its READY continuation."""
        now = self._tick()
        sequence = self.next_sequence
        self.next_sequence += 1
        token_id, row_id = _insert_row_and_token(self.engine, sequence=sequence, now=now)
        item = self.repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=row_id,
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=sequence,
            available_at=now,
            row_payload_json=self.payload,
        )
        assert item.status is TokenWorkStatus.READY
        assert item.attempt == 1
        assert item.lease_owner is None
        self.model[token_id] = ModelItem(
            token_id=token_id,
            work_item_id=item.work_item_id,
            ingest_sequence=sequence,
            status=TokenWorkStatus.READY,
        )
        return token_id

    @rule(token_id=tokens)
    def reenqueue_is_idempotent(self, token_id: str) -> None:
        """Crash-replay re-enqueue of an unclaimed continuation must not duplicate it."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.READY or model.attempt != 1:
            return
        now = self._tick()
        item = self.repo.enqueue_ready(
            run_id=RUN_ID,
            token_id=token_id,
            row_id=f"row-{model.ingest_sequence}",
            node_id=NODE_ID,
            step_index=1,
            ingest_sequence=model.ingest_sequence,
            available_at=now,
            row_payload_json=self.payload,
        )
        assert item.work_item_id == model.work_item_id
        assert item.status is TokenWorkStatus.READY
        assert item.attempt == 1

    # -------------------------------------------------------------------------
    # Rules: claims
    # -------------------------------------------------------------------------

    @rule(owner=st.sampled_from(WORKERS))
    def claim_ready(self, owner: str) -> None:
        """claim_ready returns exactly the next READY item — never a leased one."""
        now = self._tick()
        expected = self._next_claimable(TokenWorkStatus.READY)
        item = self.repo.claim_ready(run_id=RUN_ID, lease_owner=owner, lease_seconds=LEASE_SECONDS, now=now)
        if expected is None:
            assert item is None
            return
        assert item is not None
        assert item.token_id == expected.token_id
        assert item.work_item_id == expected.work_item_id
        assert item.status is TokenWorkStatus.LEASED
        assert item.lease_owner == owner
        assert item.attempt == expected.attempt
        expected.status = TokenWorkStatus.LEASED
        expected.lease_owner = owner
        expected.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)

    @rule(owner=st.sampled_from(WORKERS))
    def claim_pending_sink(self, owner: str) -> None:
        """claim_pending_sink re-leases the next sink handoff, preserving identity."""
        now = self._tick()
        expected = self._next_claimable(TokenWorkStatus.PENDING_SINK)
        item = self.repo.claim_pending_sink(run_id=RUN_ID, lease_owner=owner, lease_seconds=LEASE_SECONDS, now=now)
        if expected is None:
            assert item is None
            return
        assert item is not None
        assert item.token_id == expected.token_id
        assert item.work_item_id == expected.work_item_id
        assert item.status is TokenWorkStatus.LEASED
        assert item.lease_owner == owner
        assert item.attempt == expected.attempt
        assert item.pending_sink_name == expected.pending_sink_name
        expected.status = TokenWorkStatus.LEASED
        expected.lease_owner = owner
        expected.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)

    # -------------------------------------------------------------------------
    # Rules: leased-item transitions
    # -------------------------------------------------------------------------

    @rule(token_id=tokens)
    def mark_pending_sink(self, token_id: str) -> None:
        """LEASED transform work hands off durably to PENDING_SINK.

        Attributed park (ADR-030): the parked row KEEPS its owner with a
        NULL ``lease_expires_at`` — "parked, owner-attributed, not leased" —
        so the strict post-sink owner CAS can terminalize it.
        """
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.LEASED or model.pending_sink_name is not None:
            return
        now = self._tick()
        assert model.lease_owner is not None
        item = self.repo.mark_pending_sink(
            work_item_id=model.work_item_id,
            row_payload_json=self.payload,
            sink_name=SINK_NAME,
            outcome="success",
            path="completed",
            error_hash=None,
            error_message=None,
            now=now,
            expected_lease_owner=model.lease_owner,
        )
        assert item.status is TokenWorkStatus.PENDING_SINK
        assert item.lease_owner == model.lease_owner
        assert item.lease_expires_at is None
        assert item.pending_sink_name == SINK_NAME
        model.status = TokenWorkStatus.PENDING_SINK
        model.lease_expires_at = None
        model.pending_sink_name = SINK_NAME

    @rule(token_id=tokens, barrier_key=st.sampled_from(BARRIERS))
    def mark_blocked(self, token_id: str, barrier_key: str) -> None:
        """LEASED transform work parks durably at a barrier, releasing the lease."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.LEASED or model.pending_sink_name is not None:
            return
        now = self._tick()
        assert model.lease_owner is not None
        item = self.repo.mark_blocked(
            work_item_id=model.work_item_id,
            queue_key=None,
            barrier_key=barrier_key,
            now=now,
            expected_lease_owner=model.lease_owner,
        )
        assert item.status is TokenWorkStatus.BLOCKED
        assert item.lease_owner is None
        assert item.barrier_key == barrier_key
        model.status = TokenWorkStatus.BLOCKED
        model.lease_owner = None
        model.lease_expires_at = None
        model.barrier_key = barrier_key

    @rule(token_id=tokens)
    def mark_terminal(self, token_id: str) -> None:
        """LEASED transform work terminalizes under its lease owner."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.LEASED or model.pending_sink_name is not None:
            return
        now = self._tick()
        assert model.lease_owner is not None
        item = self.repo.mark_terminal(work_item_id=model.work_item_id, now=now, expected_lease_owner=model.lease_owner)
        assert item.status is TokenWorkStatus.TERMINAL
        assert item.lease_owner is None
        model.status = TokenWorkStatus.TERMINAL
        model.lease_owner = None
        model.lease_expires_at = None

    @rule(token_id=tokens)
    def mark_failed(self, token_id: str) -> None:
        """LEASED transform work fails under its lease owner after exhausted retries."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.LEASED or model.pending_sink_name is not None:
            return
        now = self._tick()
        assert model.lease_owner is not None
        item = self.repo.mark_failed(work_item_id=model.work_item_id, now=now, expected_lease_owner=model.lease_owner)
        assert item.status is TokenWorkStatus.FAILED
        assert item.lease_owner is None
        model.status = TokenWorkStatus.FAILED
        model.lease_owner = None
        model.lease_expires_at = None

    @rule(token_id=tokens)
    def mark_pending_sink_terminal(self, token_id: str) -> None:
        """A durable sink handoff (parked or re-claimed) terminalizes after sink durability.

        Strict owner CAS (ADR-030): only the attributed owner terminalizes.
        A NULL-owner handoff (the reap arm's park) is refused — it must be
        re-claimed via ``claim_pending_sink`` (which restores attribution)
        before terminalization.
        """
        model = self.model[token_id]
        is_parked_handoff = model.status is TokenWorkStatus.PENDING_SINK
        is_claimed_handoff = model.status is TokenWorkStatus.LEASED and model.pending_sink_name is not None
        if not (is_parked_handoff or is_claimed_handoff):
            return
        now = self._tick()
        if model.lease_owner is None:
            # Reap-parked handoff: strict CAS refusal, zero mutation.
            terminalized = self.repo.mark_pending_sink_terminal(
                run_id=RUN_ID,
                token_id=token_id,
                now=now,
                expected_lease_owner=WORKERS[0],
            )
            assert terminalized == 0
            return
        terminalized = self.repo.mark_pending_sink_terminal(
            run_id=RUN_ID,
            token_id=token_id,
            now=now,
            expected_lease_owner=model.lease_owner,
        )
        assert terminalized == 1
        model.status = TokenWorkStatus.TERMINAL
        model.lease_owner = None
        model.lease_expires_at = None

    # -------------------------------------------------------------------------
    # Rules: barrier release (the ONLY exits from BLOCKED)
    # -------------------------------------------------------------------------

    @rule(barrier_key=st.sampled_from(BARRIERS))
    def release_blocked_via_barrier_terminal(self, barrier_key: str) -> None:
        """A barrier flush consumes its BLOCKED members directly to TERMINAL."""
        blocked = self._blocked_at(barrier_key)
        if not blocked:
            return
        now = self._tick()
        token_ids = tuple(item.token_id for item in blocked)
        terminalized = self.repo.mark_blocked_barrier_terminal(run_id=RUN_ID, barrier_key=barrier_key, token_ids=token_ids, now=now)
        assert terminalized == len(blocked)
        for item in blocked:
            item.status = TokenWorkStatus.TERMINAL

    @rule(barrier_key=st.sampled_from(BARRIERS), owner=st.sampled_from(WORKERS))
    def release_blocked_via_barrier_pending_sink(self, barrier_key: str, owner: str) -> None:
        """A barrier flush hands its BLOCKED members off to PENDING_SINK pre-sink-write.

        Production parity (ADR-030 attributed park): the engine always stamps
        ``pending_sink_lease_owner`` so the strict post-sink owner CAS can
        terminalize the handoff.
        """
        blocked = self._blocked_at(barrier_key)
        if not blocked:
            return
        now = self._tick()
        handoffs = {
            item.token_id: BlockedPendingSinkHandoff(
                row_payload_json=self.payload,
                sink_name=SINK_NAME,
                outcome="success",
                path="completed",
                error_hash=None,
                error_message=None,
            )
            for item in blocked
        }
        transitioned = self.repo.mark_blocked_barrier_pending_sink_many(
            run_id=RUN_ID, barrier_key=barrier_key, handoffs=handoffs, now=now, pending_sink_lease_owner=owner
        )
        assert transitioned == len(blocked)
        for item in blocked:
            item.status = TokenWorkStatus.PENDING_SINK
            item.pending_sink_name = SINK_NAME
            item.lease_owner = owner

    @rule(token_id=tokens)
    def blocked_is_immune_to_foreign_barrier(self, token_id: str) -> None:
        """Releasing a BLOCKED item through a different barrier raises and changes nothing."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.BLOCKED:
            return
        assert model.barrier_key is not None
        foreign_barrier = next(barrier for barrier in BARRIERS if barrier != model.barrier_key)
        now = self._tick()
        with pytest.raises(AuditIntegrityError):
            self.repo.mark_blocked_barrier_terminal(run_id=RUN_ID, barrier_key=foreign_barrier, token_ids=(token_id,), now=now)

    # -------------------------------------------------------------------------
    # Rules: journal-first adoption (ADR-030 §E.2, slice 3)
    # -------------------------------------------------------------------------

    @rule(token_id=tokens)
    def adopt_blocked_barrier_item(self, token_id: str) -> None:
        """The leader adopts a BLOCKED barrier hold: NULL→epoch CAS, idempotent SKIP after."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.BLOCKED:
            return
        assert model.barrier_key is not None
        now = self._tick()
        result = self.repo.adopt_blocked_barrier_item(
            run_id=RUN_ID,
            work_item_id=model.work_item_id,
            token_id=token_id,
            barrier_key=model.barrier_key,
            membership=None,
            buffered_outcome=None,
            now=now,
            coordination_token=LEADER_TOKEN,
        )
        if model.barrier_adopted_epoch is None:
            assert result.adopted is True
            assert result.barrier_adopted_epoch == 1
            model.barrier_adopted_epoch = 1
        else:
            # Idempotent success-SKIP: any non-NULL epoch counts as adopted.
            assert result.adopted is False
            assert result.barrier_adopted_epoch == model.barrier_adopted_epoch

    @rule(token_id=tokens)
    def adoption_requires_a_blocked_hold(self, token_id: str) -> None:
        """Adopting a non-BLOCKED row is journal corruption: Tier-1, zero mutation."""
        model = self.model[token_id]
        if model.status is TokenWorkStatus.BLOCKED:
            return
        now = self._tick()
        with pytest.raises(AuditIntegrityError):
            self.repo.adopt_blocked_barrier_item(
                run_id=RUN_ID,
                work_item_id=model.work_item_id,
                token_id=token_id,
                barrier_key=BARRIERS[0],
                membership=None,
                buffered_outcome=None,
                now=now,
                coordination_token=LEADER_TOKEN,
            )

    @rule(token_id=tokens)
    def stale_leader_cannot_adopt(self, token_id: str) -> None:
        """A deposed leader's adoption is fence-refused before any mutation."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.BLOCKED:
            return
        assert model.barrier_key is not None
        now = self._tick()
        with pytest.raises(RunLeadershipLostError):
            self.repo.adopt_blocked_barrier_item(
                run_id=RUN_ID,
                work_item_id=model.work_item_id,
                token_id=token_id,
                barrier_key=model.barrier_key,
                membership=None,
                buffered_outcome=None,
                now=now,
                coordination_token=STALE_LEADER_TOKEN,
            )

    # -------------------------------------------------------------------------
    # Rules: lease expiry and recovery
    # -------------------------------------------------------------------------

    @rule()
    def let_leases_expire(self) -> None:
        """Advance time past every live lease window (a crashed/stalled worker)."""
        self.now += timedelta(seconds=LEASE_SECONDS + 1)

    @rule(caller_owner=st.sampled_from(WORKERS))
    def recover_expired_leases(self, caller_owner: str) -> None:
        """Recovery reaps exactly the expired peer leases: attempt bump + id rotation
        for transform work, attempt/work_item_id preservation for sink handoffs;
        live leases and the caller's own leases (G1 self-steal guard) are untouched."""
        now = self._tick()
        expected = [
            item
            for item in self.model.values()
            if item.status is TokenWorkStatus.LEASED
            and item.lease_expires_at is not None
            and item.lease_expires_at < now
            and item.lease_owner != caller_owner
        ]
        recovered = self.repo.recover_expired_leases(run_id=RUN_ID, now=now, caller_owner=caller_owner)
        assert recovered == len(expected)
        if not expected:
            return
        db_rows = self._db_rows()
        for item in expected:
            row = db_rows[item.token_id]
            if item.pending_sink_name is not None:
                # PENDING_SINK recovery path: durable handoff preserved in place.
                assert row["status"] == TokenWorkStatus.PENDING_SINK.value
                assert row["work_item_id"] == item.work_item_id
                assert row["attempt"] == item.attempt
                item.status = TokenWorkStatus.PENDING_SINK
            else:
                assert row["status"] == TokenWorkStatus.READY.value
                assert row["work_item_id"] != item.work_item_id
                assert row["attempt"] == item.attempt + 1
                item.status = TokenWorkStatus.READY
                item.work_item_id = str(row["work_item_id"])
                item.attempt += 1
            item.lease_owner = None
            item.lease_expires_at = None

    # -------------------------------------------------------------------------
    # Rules: forbidden transitions raise
    # -------------------------------------------------------------------------

    @rule(token_id=tokens)
    def ready_cannot_jump_to_terminal(self, token_id: str) -> None:
        """READY work cannot be terminalized without first being claimed."""
        model = self.model[token_id]
        if model.status is not TokenWorkStatus.READY:
            return
        now = self._tick()
        with pytest.raises(AuditIntegrityError):
            self.repo.mark_terminal(work_item_id=model.work_item_id, now=now, expected_lease_owner=WORKERS[0])

    @rule(token_id=tokens)
    def final_states_are_final(self, token_id: str) -> None:
        """TERMINAL/FAILED work admits no further transitions."""
        model = self.model[token_id]
        if model.status not in FINAL_STATUSES:
            return
        now = self._tick()
        with pytest.raises(AuditIntegrityError):
            self.repo.mark_terminal(work_item_id=model.work_item_id, now=now, expected_lease_owner=WORKERS[0])
        with pytest.raises(AuditIntegrityError):
            self.repo.mark_pending_sink(
                work_item_id=model.work_item_id,
                row_payload_json=self.payload,
                sink_name=SINK_NAME,
                outcome="success",
                path="completed",
                error_hash=None,
                error_message=None,
                now=now,
                expected_lease_owner=WORKERS[0],
            )

    # -------------------------------------------------------------------------
    # Invariants
    # -------------------------------------------------------------------------

    @invariant()
    def work_items_are_conserved(self) -> None:
        """Exactly one durable work item per scheduled token; none lost, none invented."""
        db_rows = self._db_rows()  # asserts per-token uniqueness internally
        assert set(db_rows) == set(self.model), f"Conservation violated: db={sorted(db_rows)} model={sorted(self.model)}"

    @invariant()
    def database_matches_model(self) -> None:
        """Durable status/attempt/lease/sink/barrier state agrees with the model."""
        db_rows = self._db_rows()
        for token_id, item in self.model.items():
            row = db_rows[token_id]
            assert row["status"] == item.status.value, f"{token_id}: db status {row['status']} != model {item.status.value}"
            assert row["attempt"] == item.attempt
            assert row["work_item_id"] == item.work_item_id
            assert row["lease_owner"] == item.lease_owner
            assert row["pending_sink_name"] == item.pending_sink_name
            if item.status is TokenWorkStatus.BLOCKED:
                assert row["barrier_key"] == item.barrier_key
            # ADR-030 §E.2: the adoption marker agrees with the model on every
            # row that ever blocked (it persists across the barrier release).
            assert row["barrier_adopted_epoch"] == item.barrier_adopted_epoch, (
                f"{token_id}: db barrier_adopted_epoch {row['barrier_adopted_epoch']} != model {item.barrier_adopted_epoch}"
            )

    @invariant()
    def lease_fields_set_iff_leased(self) -> None:
        """lease_expires_at is non-NULL exactly when LEASED; lease_owner is
        non-NULL when LEASED, MAY be non-NULL on PENDING_SINK (attributed
        park, ADR-030), and is NULL everywhere else."""
        for token_id, row in self._db_rows().items():
            is_leased = row["status"] == TokenWorkStatus.LEASED.value
            is_pending_sink = row["status"] == TokenWorkStatus.PENDING_SINK.value
            assert (row["lease_expires_at"] is not None) == is_leased, f"{token_id}: lease_expires_at/{row['status']} mismatch"
            if is_leased:
                assert row["lease_owner"] is not None, f"{token_id}: LEASED without lease_owner"
            elif not is_pending_sink:
                assert row["lease_owner"] is None, f"{token_id}: lease_owner set on {row['status']}"

    @invariant()
    def terminalization_never_departs_ready(self) -> None:
        """The durable journal shows no READY→TERMINAL edge; every terminalization
        departs from LEASED, BLOCKED, or PENDING_SINK."""
        allowed_terminal_origins = {
            TokenWorkStatus.LEASED.value,
            TokenWorkStatus.BLOCKED.value,
            TokenWorkStatus.PENDING_SINK.value,
        }
        with self.engine.connect() as conn:
            terminal_events = (
                conn.execute(
                    select(scheduler_events_table.c.event_type, scheduler_events_table.c.from_status, scheduler_events_table.c.token_id)
                    .where(scheduler_events_table.c.run_id == RUN_ID)
                    .where(scheduler_events_table.c.to_status == TokenWorkStatus.TERMINAL.value)
                )
                .mappings()
                .all()
            )
        for event in terminal_events:
            assert event["from_status"] in allowed_terminal_origins, (
                f"Token {event['token_id']} terminalized from {event['from_status']!r} via {event['event_type']!r}"
            )

    @invariant()
    def unresolved_and_active_counts_match_model(self) -> None:
        """count_unresolved_work and count_active_work agree with the model's view:
        PENDING_SINK and LEASED-with-pending_sink_name are active but resolved."""
        expected_unresolved = sum(
            1
            for item in self.model.values()
            if item.status in (TokenWorkStatus.READY, TokenWorkStatus.BLOCKED)
            or (item.status is TokenWorkStatus.LEASED and item.pending_sink_name is None)
        )
        expected_active = sum(1 for item in self.model.values() if item.status not in FINAL_STATUSES)
        assert self.repo.count_unresolved_work(run_id=RUN_ID) == expected_unresolved
        assert self.repo.count_active_work(run_id=RUN_ID) == expected_active


TestSchedulerWorkItemLifecycleStateMachine = SchedulerWorkItemLifecycleStateMachine.TestCase
TestSchedulerWorkItemLifecycleStateMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)


def test_expired_pending_sink_lease_recovers_in_place_preserving_attempt_and_work_item_id() -> None:
    """Deterministic pin of invariant 5's EXCEPT branch (Phase-1 trace): a
    re-claimed sink handoff (LEASED with pending_sink_name) whose lease expires
    is recovered back to PENDING_SINK by a peer sweep with attempt and
    work_item_id PRESERVED — the handoff is already durable, so no transform
    work is replayed under a new audit identity. The stateful machine can reach
    this path only through a four-rule sequence, so it is pinned explicitly.
    """
    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    payload = _row_payload_json()
    _insert_run_and_nodes(engine, now=now)
    token_id, row_id = _insert_row_and_token(engine, sequence=0, now=now)
    repo.enqueue_ready(
        run_id=RUN_ID,
        token_id=token_id,
        row_id=row_id,
        node_id=NODE_ID,
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=LEASE_SECONDS, now=now)
    assert claimed is not None
    repo.mark_pending_sink(
        work_item_id=claimed.work_item_id,
        row_payload_json=payload,
        sink_name=SINK_NAME,
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=1),
        expected_lease_owner="worker-a",
    )
    reclaimed = repo.claim_pending_sink(run_id=RUN_ID, lease_owner="worker-a", lease_seconds=LEASE_SECONDS, now=now + timedelta(seconds=2))
    assert reclaimed is not None
    assert reclaimed.work_item_id == claimed.work_item_id
    assert reclaimed.pending_sink_name == SINK_NAME
    assert reclaimed.attempt == 1

    expired_at = now + timedelta(seconds=LEASE_SECONDS + 3)
    assert repo.recover_expired_leases(run_id=RUN_ID, now=expired_at, caller_owner="worker-b") == 1

    with engine.connect() as conn:
        row = (
            conn.execute(
                select(token_work_items_table)
                .where(token_work_items_table.c.run_id == RUN_ID)
                .where(token_work_items_table.c.token_id == token_id)
            )
            .mappings()
            .one()
        )
    assert row["status"] == TokenWorkStatus.PENDING_SINK.value
    assert row["work_item_id"] == claimed.work_item_id
    assert row["attempt"] == 1
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None

    # The preserved handoff is claimable and terminalizable exactly as before.
    drained = repo.claim_pending_sink(run_id=RUN_ID, lease_owner="worker-b", lease_seconds=LEASE_SECONDS, now=expired_at)
    assert drained is not None
    assert drained.work_item_id == claimed.work_item_id
    assert (
        repo.mark_pending_sink_terminal(
            run_id=RUN_ID,
            token_id=token_id,
            now=expired_at + timedelta(seconds=1),
            expected_lease_owner="worker-b",
        )
        == 1
    )
    engine.dispose()
