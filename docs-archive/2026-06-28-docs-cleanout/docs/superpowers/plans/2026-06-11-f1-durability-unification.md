# F1 Durability Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The scheduler journal (`token_work_items`) becomes the single source of barrier-buffer truth on resume; the `AggregationCheckpointState`/`CoalesceCheckpointState` blob layer is deleted; barrier flush becomes one atomic journal transition; the checkpoint row shrinks to `(run_id, sequence_number, upstream_topology_hash)` plus a small scalar-metadata JSON.

**Architecture:** Today a buffered token already gets a durable BLOCKED journal row at buffer time, but on resume the executors rebuild their in-memory buffers from checkpoint blobs and the journal is reconciled *to* the blob (`ensure_blocked_barrier_work_item`, default `attempt=1`). F1 inverts the read direction: resume rebuilds executor buffers *from* BLOCKED rows, derives barrier metadata from durable audit where possible (batch_members, token_outcomes, row timestamps), keeps only the truly underivable scalars (trigger fire latches, completed_flush_count, lost_branches) on the checkpoint row, and makes barrier completion (consume inputs + emit outputs) one transaction so no flush output ever exists only in memory.

**Tech Stack:** Python 3.12, SQLAlchemy Core against SQLite (Tier-1 `LandscapeDB`), pytest (`env -u VIRTUAL_ENV .venv/bin/pytest`, plain `pytest tests/` — NEVER `-o addopts=""`), mypy. Work in the worktree `/home/john/elspeth/.worktrees/multi-source-token-scheduler` on branch `feat/multi-source-token-scheduler` (local, unpushed — do NOT push).

**Filigree:** implements elspeth-4d5cbf2fcf (P0); dissolves elspeth-ae5183307b, elspeth-262911c26b; forces/settles elspeth-e1dd5e1303, elspeth-7294de558e; unblocks elspeth-1396d3f790. Actor suggestion: `claude-f1-durability`.

---

# Part I — Design (the design doc elspeth-4d5cbf2fcf requires)

## I.1 Current model (verified facts, all citations at branch HEAD)

1. **Live buffering already writes the journal.** Aggregation: `buffer_row` mutates memory + writes `batch_members`/BUFFERED `token_outcomes` (processor.py:1589-1603), then the drain transitions the claimed row LEASED→BLOCKED via `mark_blocked` (processor.py:2957-2964 → scheduler_repository.py:1204-1228, in-place, attempt untouched). Coalesce: held branch → same BLOCKED transition (processor.py:2973-2974); barrier_key = node_id for aggregation, coalesce_name for coalesce (processor.py:3460-3466).
2. **Flush is already (mostly) a journal transition.** Consumed BLOCKED rows → TERMINAL via `mark_blocked_barrier_terminal` (scheduler_repository.py:1648-1752, with live-vs-durable set-equality `AuditIntegrityError` cross-checks); passthrough sink-bound flush → BLOCKED→PENDING_SINK via `mark_blocked_barrier_pending_sink_many` (:1349-1452); merged coalesce continuation → fresh `enqueue_ready` (processor.py:3350-3414). **But** consume and emit are separate transactions, and out-of-claim flush outputs (timeout/EOF) travel in `pending_tokens` memory with *no* journal row until sink write (source_iteration.py:607-616) — the unresumable window of elspeth-ae5183307b.
3. **Resume distrusts all of it.** Executor buffers are rebuilt only from the blob (processor.py:503-504, run_core.py:358); the journal is reconciled to the blob via `ensure_blocked_barrier_work_item` (idempotent insert at hardcoded default `attempt=1`, scheduler_repository.py:308-451; callers pass no attempt — processor.py:514-554, 556-586); `RecoveryManager.can_resume` hard-requires a checkpoint row ("No checkpoint found for recovery"); `get_unprocessed_rows` excludes rows whose incomplete tokens are buffered *in the blob* (recovery.py:574-578, 668-679, 797-812).
4. **Blob-only contents** (the migration list — assessment said 2 items; recon found 6):
   - per aggregation node: `elapsed_age_seconds`, `count_fire_offset`, `condition_fire_offset` (trigger timing); `accepted_count_total`, `completed_flush_count` (pagination); `batch_id`;
   - per coalesce pending key: `elapsed_age_seconds`, `lost_branches`;
   - per coalesce token: `state_id` (node_states pointer), `arrival_offset_seconds`;
   - plus the agg-only `contract_version` integrity hash and the `"5.0"`/`"1.0"` version gates.
5. **Schema/epoch:** `SQLITE_SCHEMA_EPOCH = 19` (schema.py:98); pre-1.0 policy = bump + operator deletes the DB; no migration runner. `checkpoints` post-F2 = checkpoint_id, run_id, sequence_number, aggregation_state_json, coalesce_state_json, created_at, upstream_topology_hash, format_version (schema.py:903-924; `CURRENT_FORMAT_VERSION = 4`).

## I.2 End-state decisions

### D1 — Buffered-token membership and payload truth = BLOCKED journal rows
On resume, a new read verb `list_blocked_barrier_items(run_id)` feeds `restore_from_journal(...)` on both executors. Row payloads come from `deserialize_row_payload(row_payload_json)` (the same round-trip already used for PENDING_SINK re-drive, processor.py:3098-3121). The two `_restore_scheduler_blocks_from_*` methods, `ensure_blocked_barrier_work_item`, and the `RESTORE_BLOCKED` event type are deleted.

*Integrity argument:* the journal row was written by the live run at buffer time. (`ensure_blocked_barrier_work_item`'s byte-identity validation passing today is only weak evidence — it fires solely on `work_item_id` hash collision, i.e. attempt-1 cursor-matching rows — so the typed-payload round-trip fixture in Known-risks item 2 is MANDATORY, not optional.) The aggregation-only `contract_version` "checkpoint may be corrupted" check (processor.py:524-529) retires with the blob — the journal payload embeds the contract and is not a re-serialization.

*Dissolves by construction:* the attempt-1 manufacture seam (a live BLOCKED row at attempt>1 can never collide with a manufactured attempt-1 twin again), and the latent blob-vs-LEASED `AuditIntegrityError` (a post-sink checkpoint can capture a memory-buffered token whose journal row is still LEASED; restore would then `ensure_blocked` against a LEASED row and hard-fail — that class is unreachable once nothing manufactures rows).

*Barrier discrimination rule (keyed on `barrier_key`, NEVER `node_id`):* a BLOCKED row's `node_id` is the **enqueue-time cursor**, not the barrier node (`mark_blocked` and `_transition` never touch node_id — scheduler_repository.py:1204, :1939). `barrier_key` IS the barrier identity by construction (processor.py:3460-3466: coalesce → coalesce_name, aggregation → str(aggregation node_id)). Also: **BLOCKED is dual-use** — ADR-028 queue-holds are BLOCKED rows with `queue_key` set and `barrier_key` NULL (`mark_blocked` accepts either key). So: barrier restore and the resume buffered-exclusion both select `status='blocked' AND barrier_key IS NOT NULL`; queue-holds are untouched (their resume path is unchanged by F1). Partition: `barrier_key ∈` coalesce names (keys of `_coalesce_node_ids`, which is keyed by `CoalesceName` — processor.py:408) → coalesce; `barrier_key ∈` str-keyed aggregation node ids (from the aggregation settings map) → aggregation; neither → `AuditIntegrityError`. Aggregation rows may carry non-NULL `coalesce_name` *lineage* (processor.py:531-537), so never discriminate on `coalesce_name IS NOT NULL`.

### D2 — Absolute timestamps replace offset arithmetic
New nullable column `token_work_items.barrier_blocked_at` (DateTime tz), written by `mark_blocked` only. (It is stamped at the drain's BLOCKED transition — a few ms after the in-memory accept — so restored ages are conservatively slightly *younger*; harmless, and stamped on queue-holds too, where nothing reads it.) On restore:
- coalesce per-branch `arrival_time` ← row's `barrier_blocked_at`; `first_arrival` ← min over branches; `elapsed_age_seconds` ← now − first_arrival. The blob's `arrival_offset_seconds`/`elapsed_age_seconds` (offset reconstruction, stale by last-checkpoint-age) are deleted with *better* fidelity, not worse.
- aggregation `first_accept_time` ← min `barrier_blocked_at` of the node's BLOCKED rows (the first accepted row of a batch always blocks; a count-1 trigger fires in-claim and has no timing to restore). `TriggerEvaluator.restore_from_checkpoint(batch_count, elapsed_age_seconds, count_fire_offset, condition_fire_offset)` keeps its signature — only the *source* of elapsed changes.

We do not reuse `updated_at` (any future transition write would silently corrupt arrival times); a dedicated column is honest and the epoch bumps anyway.

### D3 — Derive what audit already proves; checkpoint keeps only true scalars
- `accepted_count_total` ← `COUNT(batch_members)` over all batches of the node in the run (each accept writes batch_members durably in its own txn *before* the BLOCKED transition — executors/aggregation.py:194-202). Not stored.
- `batch_id` per restored token ← the token's BUFFERED `token_outcomes.batch_id` (existing derivation pattern, processor.py:3201-3234), passed through `handle_incomplete_batches`' dead-batch remap. Not stored.
- coalesce `state_id` ← query node_states for the token's PENDING hold at the coalesce node (precedent: `_completed_keys` is already Landscape-reconstructed, coalesce_executor.py:328-385). Not stored, no new column.
- `completed_flush_count` ← `COUNT(batches WHERE status='completed' AND aggregation node = ?)` — the only COMPLETED-status `complete_batch` call is the flush-success path (executors/aggregation.py:491) where the counter increments (:501); failed flushes complete FAILED (:516/:532). Derivation is *fresher* than any stored scalar (no lost-increment crash window between flush success and the next checkpoint). Not stored.
- **Stored** (the truly underivable scalars), in one new `checkpoints.barrier_scalars_json` Text column replacing the two blob columns: per aggregation node `{count_fire_offset, condition_fire_offset}` (the two trigger latches); per coalesce pending key `{lost_branches}`. That is the whole inventory. New small contracts module `contracts/barrier_scalars.py` replaces both blob families. `format_version` 4→5.

*Staleness audit (stored scalars are stale by last-checkpoint-age — same as the blob today, no regression):* missing count latch is healed by the existing conservative re-latch when `batch_count >= config.count` (engine/triggers.py:300-304); a lost condition latch delays firing to the next accept/EOF flush (same membership, possibly different `which_triggered` attribution — the resumed run records what actually happened); `lost_branches` lost in the window degrades require_all/best_effort evaluation to the timeout path, identical to today's blob loss window. Counters have NO staleness at all post-F1 (both derived from per-mutation audit).

### D4 — A checkpoint row always exists: sequence-0 at run start
`run()` writes an initial checkpoint (sequence_number 0, `upstream_topology_hash`, empty scalars) before the first source row. Consequences: topology-compatibility validation stays unconditional; `can_resume`'s "No checkpoint found" arm becomes a genuine refusal only for runs that predate the run-start write or ran with checkpointing disabled (reason rewords to journal-flavoured text — test_resume_rejection.py:168-170 deliberately does not pin the wording); `rebase_sequence` is unchanged (resume rebases to the latest sequence; it never writes a fresh 0 — the unique `(run_id, sequence_number)` index would refuse it anyway). Checkpoint deletion on success removes the sequence-0 row with the rest (lifecycle unchanged). Tests that pin per-run checkpoint COUNTS shift by +1 — see the "D4 count-shift family" row in Part III. Restored `TokenInfo` gets `resume_checkpoint_id` from this always-present resume point, satisfying the `resume_attempt_offset > 0 ⟹ resume_checkpoint_id is not None` invariant (test_identity.py:186-225).

### D5 — Attempt discipline from the journal
Restored barrier tokens — **both aggregation and coalesce** — rebuild `TokenInfo` with `resume_attempt_offset` derived from node_states `max_attempt + 1` (the processor.py:2323 discipline), exactly as `get_incomplete_tokens_by_row` already computes (recovery.py:722-728), and `resume_checkpoint_id` from the resume point. This fixes elspeth-262911c26b by construction — `AggregationExecutor.restore_from_checkpoint`'s default-0 rebuild (aggregation.py:729-739) is deleted, and both replacements carry the offset.

### D6 — Flush is ONE atomic journal transition
New repo verb `complete_barrier(...)` performs, in a single transaction: validate the consumed set against the durable BLOCKED set (both directions, preserving the :1683-1690/:1747-1751 cross-checks), consumed → TERMINAL (payload scrub), sink-bound emissions → PENDING_SINK (transition for buffered passthrough tokens, *insert* on the node_id-NULL terminal lane for merged/aggregate outputs — the `uq_token_work_items_terminal_identity` partial unique index is that lane), continuation emissions → READY inserts. `mark_blocked_barrier_terminal` and `mark_blocked_barrier_pending_sink_many` become internal arms of it. This closes the out-of-claim flush window: a timeout/EOF flush output is journal-durable (PENDING_SINK) the moment its inputs are consumed, and a crash before sink write recovers via the existing PENDING_SINK re-drive.

In-claim flush keeps the LEASED triggering-token exclusion exactly as today (the claim's own result rides the normal drain arms).

### D7 — Explicitly out of scope
- The memory-first window between in-memory buffer/accept and the drain's `mark_blocked`. A crash there leaves a `batch_members` row with no BLOCKED journal row; on resume the membership reconcile refuses with `AuditIntegrityError` (a re-drive would PK-violate `(batch_id, token_id)`). That refusal is **pre-existing semantics, unchanged by F1** — do NOT "fix" it ad hoc in Task 3.4 (Known-risks item 7).
- Cross-process worker coordination (option c, elspeth-1396d3f790) — F1 unblocks it, does not start it.
- Any shared Barrier abstraction (F3-long-term) — twin implementations stay twins; every change here is made to both sides per the docs/architecture/barrier-machinery.md checklist.
- Checkpoint cadence/config semantics (`RuntimeCheckpointConfig.frequency`) — untouched.

## I.3 Safety-net contract (must pass; from recon, verified per-file)
- **Zero edits:** tests/integration/engine/test_multi_source_chaos.py (5), tests/unit/core/landscape/test_scheduler_lease_recovery_races.py (4 fn/5 collected), the 9 RC6 proofs (test_rc6_multisource_provenance_proof.py 3, test_rc6_concurrent_pumping_proof.py 2, tests/unit/engine/test_rc6_scheduler_ordering_characterization.py 4). All grep-clean of checkpoint symbols.
- **Assertions unchanged, named harness seams edited:** tests/e2e/recovery/test_resume_rejection.py (`_recovery_manager` :53-62), tests/e2e/recovery/test_concurrent_resume.py (`_run_to_interrupted_checkpoint` :270, `_CrashedRun.resume_orchestrator` :240-246, `_recovery_manager` :249-257, `_resume_point` :260-267). Both files document this contract in their docstrings.
- **Deliberate flips (loud, by design):** test_rc6_eof_resume_proof.py:298 (`test_real_eof_flush_crash_is_not_yet_resumable` → positive roundtrip) and test_fork_join_balance.py:4186 (`test_count_equals_n_rows_buffered_divergence_is_pinned` → unified value N, plus its production step — Task 4.3).

## I.4 Observable behavior changes (call out in the final report)
1. Virgin EOF-flush crashes become resumable (elspeth-ae5183307b dissolved).
2. Tokens buffered *after* the last checkpoint are restored into buffers instead of re-driven from source — strictly fewer duplicate executions on resume.
3. `rows_buffered` live-vs-derive unifies on **N** (audit value: every accepted member gets a BUFFERED token_outcome, including the in-claim triggering token; the live counter today only counts buffered *drain results* = N−1 — outcomes.py:385). Requires a production step, not just a pin flip — see Task 4.3 Step 2 (elspeth-e1dd5e1303 decision forced).
4. `can_resume` reason strings become journal-flavoured.
5. `which_triggered` attribution after a crash-in-the-condition-latch-window may report the resumed run's actual trigger rather than the pre-crash latch (honest, documented).
6. DB epoch 19→20; operator deletes Landscape DBs (delete-the-DB policy; SESSION_SCHEMA_EPOCH untouched).

---

# Part II — Tasks

Conventions for every task: run tests as `cd /home/john/elspeth/.worktrees/multi-source-token-scheduler && env -u VIRTUAL_ENV .venv/bin/pytest <selector> -x -q`; commit with conventional style and trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; `git stash` is hook-blocked (use WIP commit + `git reset --soft`); never re-sign trust-tier entries (report displacements instead); if a *session-DB* schema change ever looks necessary, stop and report (Landscape epoch bump is planned here, that one is expected).

## Phase 0 — ADR + schema (one epoch bump covers everything)

### Task 0.1: ADR-029

**Files:**
- Create: `docs/architecture/adr/029-journal-is-barrier-buffer-truth.md`
- Modify: `docs/architecture/adr/README.md` (index row), `docs/README.md` if it indexes ADRs

- [ ] **Step 1: Write the ADR** — sections: Context (three durability layers, assessment citation), Decision (D1–D6 verbatim from Part I), Consequences (I.4 list), Alternatives rejected (scalars on a new Tier-1 `barrier_state` table — rejected as gold-plating until option c needs live cross-worker metadata; scalars denormalized onto BLOCKED rows — rejected because counter-only/post-flush state has zero rows to carry it; reusing `updated_at` for arrival times — rejected as fragile coupling).
- [ ] **Step 2: Add the README index rows; cite ADR-026 (4-key ORDER BY) and barrier-machinery.md as related.**
- [ ] **Step 3: Commit** — `docs(adr): ADR-029 — scheduler journal is the single source of barrier-buffer truth (F1)`

### Task 0.2: Schema epoch 20 — ADDITIVE ONLY (keeps every commit green)

The blob-column DROPs and the `RESTORE_BLOCKED` removal happen in Task 4.1, AFTER all readers are migrated. Epoch 20 is unreleased, so both the additions here and the deletions there share the one bump.

**Files:**
- Create: `notes/f1-pretask-fullsuite-baseline.txt` (persist the inherited-red baseline OUT of /tmp before any code change: `env -u VIRTUAL_ENV .venv/bin/pytest tests/ -q 2>&1 | grep -E "^(FAILED|ERROR)" > notes/f1-pretask-fullsuite-baseline.txt`, plus the tail summary line; commit it with this task)
- Modify: `src/elspeth/core/landscape/schema.py` (token_work_items ~:396-449; checkpoints ~:903-924; epoch constant+history :35-98)
- Modify: `src/elspeth/core/landscape/database.py` (`_REQUIRED_COLUMNS`/`_REQUIRED_INDEXES` lists)
- Test: `tests/unit/core/landscape/test_schema_epoch_and_required_columns.py`

- [ ] **Step 0: Capture and commit the baseline file** (above) — Task 5.3's regression gate diffs against it.
- [ ] **Step 1: Write the failing tests** (rename `test_epoch_is_nineteen`):

```python
def test_epoch_is_twenty() -> None:
    assert SQLITE_SCHEMA_EPOCH == 20

def test_token_work_items_has_barrier_blocked_at() -> None:
    assert "barrier_blocked_at" in token_work_items_table.c

def test_checkpoints_have_barrier_scalars_column() -> None:
    assert "barrier_scalars_json" in checkpoints_table.c
```

- [ ] **Step 2: Run, verify the three fail** (`pytest tests/unit/core/landscape/test_schema_epoch_and_required_columns.py -x -q`).
- [ ] **Step 3: Implement (additions only)** — in `schema.py`: `SQLITE_SCHEMA_EPOCH = 20` plus history entry `# 20: F1 durability unification: token_work_items.barrier_blocked_at added; checkpoints aggregation_state_json/coalesce_state_json replaced by barrier_scalars_json; restore_blocked event type removed from scheduler_events CHECK.`; add to token_work_items `Column("barrier_blocked_at", DateTime(timezone=True), nullable=True)`; add to checkpoints `Column("barrier_scalars_json", Text, nullable=True)` (blob columns stay until Task 4.1). Update `database.py` required-columns lists per the existing pattern (test_schema_epoch_and_required_columns.py:31-38).
- [ ] **Step 4: Run the schema tests, verify pass; full landscape unit dir green** (`pytest tests/unit/core/landscape -q`).
- [ ] **Step 5: Commit** — `feat(schema): epoch 20 — barrier_blocked_at on token_work_items; checkpoints gain barrier_scalars_json (F1, additive half)`

## Phase 1 — Checkpoint manager + scalars contracts (tree compiles again)

### Task 1.1: contracts/barrier_scalars.py

**Files:**
- Create: `src/elspeth/contracts/barrier_scalars.py`
- Modify: `src/elspeth/contracts/__init__.py` (re-exports)
- Test: `tests/unit/contracts/test_barrier_scalars.py` (new)

- [ ] **Step 1: Write failing tests** — round-trip, validation, empty-is-falsy:

```python
from elspeth.contracts.barrier_scalars import (
    AggregationNodeScalars, BarrierScalars, CoalescePendingScalars,
)

def test_round_trip() -> None:
    s = BarrierScalars(
        aggregation={"agg-1": AggregationNodeScalars(
            count_fire_offset=1.5, condition_fire_offset=None)},
        coalesce={("merge", "row-7"): CoalescePendingScalars(
            lost_branches={"b2": "branch lost: transform failed"})},
    )
    assert BarrierScalars.from_dict(s.to_dict()) == s

def test_empty_state_is_falsy_and_serializes_minimal() -> None:
    empty = BarrierScalars(aggregation={}, coalesce={})
    assert not empty.has_state
    assert BarrierScalars.from_dict(empty.to_dict()) == empty

def test_coalesce_key_with_hostile_characters_round_trips() -> None:
    # coalesce names have no charset constraint (core/config.py:717) and row_id is
    # operator-influenced — keys must survive arbitrary strings
    s = BarrierScalars(aggregation={}, coalesce={
        ("we::ird", 'row"7'): CoalescePendingScalars(lost_branches={})})
    assert BarrierScalars.from_dict(s.to_dict()) == s

def test_negative_offset_rejected() -> None:
    with pytest.raises(ValueError):
        AggregationNodeScalars(count_fire_offset=-0.1, condition_fire_offset=None)
```

- [ ] **Step 2: Run, verify fail (module missing).**
- [ ] **Step 3: Implement** — frozen dataclasses in the house style of the retired `contracts/aggregation_checkpoint.py` (slots, `__post_init__` validation, `to_dict`/`from_dict` with a `_version` key, `BARRIER_SCALARS_VERSION = "1.0"`, unknown-key rejection). `AggregationNodeScalars` fields: `count_fire_offset: float | None`, `condition_fire_offset: float | None` — counters are NOT stored (D3: both derived from audit). Coalesce pending keys serialize as **2-element JSON arrays** `[coalesce_name, row_id]` (NOT a delimiter-joined string — names/row_ids have no charset constraint).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(contracts): BarrierScalars — the post-F1 checkpoint barrier metadata (replaces blob state contracts)`

### Task 1.2: CheckpointManager + Checkpoint contract on the new column

**Files:**
- Modify: `src/elspeth/core/checkpoint/manager.py` (`create_checkpoint` :99-198, size guards :46-78, `CURRENT_FORMAT_VERSION` consumer)
- Modify: `src/elspeth/contracts/audit.py` (`Checkpoint` :513-555 — fields and `CURRENT_FORMAT_VERSION = 5`)
- Modify: `src/elspeth/contracts/checkpoint.py` (`ResumePoint` :38-74 — `aggregation_state`/`coalesce_state` → `barrier_scalars: BarrierScalars | None`)
- Modify: `src/elspeth/cli.py` :2011-2032 — the `resume` command's inspection output reads `resume_point.aggregation_state`/`.coalesce_state`; replace with `has_barrier_scalars` plus a journal BLOCKED-barrier-row count (the user-facing "what will be restored" answer now lives in the journal)
- Test: `tests/unit/core/checkpoint/test_manager.py`

- [ ] **Step 1: Write the failing test** for the new signature:

```python
def test_create_checkpoint_persists_barrier_scalars(db, graph) -> None:
    mgr = CheckpointManager(db)
    scalars = BarrierScalars(
        aggregation={"agg-1": AggregationNodeScalars(count_fire_offset=2.25,
                                                     condition_fire_offset=None)},
        coalesce={})
    cp = mgr.create_checkpoint(run_id=run_id, sequence_number=1,
                               barrier_scalars=scalars, graph=graph)
    row = _select_checkpoint(db, cp.checkpoint_id)
    assert json.loads(row.barrier_scalars_json)["aggregation"]["agg-1"]["count_fire_offset"] == 2.25
    assert cp.format_version == 5
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — `create_checkpoint(*, run_id, sequence_number, barrier_scalars: BarrierScalars | None, graph)`: same single-transaction shape (duplicate-sequence guard, topology hash, INSERT), serializing `checkpoint_dumps(barrier_scalars.to_dict()) if barrier_scalars and barrier_scalars.has_state else None`. Keep `checkpoint_dumps` (float-fidelity rationale at manager.py:147-148 still applies to offsets). Size guard: keep the hard-fail, threshold can stay (scalars are tiny). `Checkpoint` dataclass: replace the two `*_state_json` fields with `barrier_scalars_json: str | None`; `CURRENT_FORMAT_VERSION = 5` with docstring updated ("buffered tokens live in token_work_items; checkpoint carries only scalar barrier metadata"). `ResumePoint`: `barrier_scalars: BarrierScalars | None`, keep the `sequence_number == checkpoint.sequence_number` post-init.
- [ ] **Step 4: Update the directly-broken callers to compile** (mechanical, no behavior): `recovery.py` `_restore_checkpoint_states` → deserialize `barrier_scalars_json` (rename `_RestoredCheckpointStates` accordingly); `checkpointing.py` passes scalars (Task 2.4 finishes this); `cli.py` per the Files note above; `orchestrator/aggregation.py` `rebind_checkpoint_batch_ids` loses its blob input — remove its import/call sites in `resume.py:594-597` (Task 3.2 replaces the mechanism).
- [ ] **Step 5: Run** `pytest tests/unit/core/checkpoint/test_manager.py tests/unit/contracts/ -q` — manager+contracts green (recovery/orchestrator tests still red until Phases 2–3; that's expected mid-chain).
- [ ] **Step 6: Commit** — `feat(checkpoint): checkpoint row carries BarrierScalars only; format_version 5 (F1)`

### Task 1.3: `mark_blocked` writes `barrier_blocked_at`; journal read verb

**Files:**
- Modify: `src/elspeth/core/landscape/scheduler_repository.py` (`mark_blocked` :1204-1228; `_transition` :1939-2028; new `list_blocked_barrier_items`)
- Test: `tests/unit/core/landscape/test_scheduler_repository.py` (or the existing repo-verb test home — locate with `grep -rl "mark_blocked" tests/unit/core/landscape/`)

- [ ] **Step 1: Write failing tests:**

```python
def test_mark_blocked_stamps_barrier_blocked_at(repo, seeded_leased_item) -> None:
    now = datetime(2026, 6, 11, 3, 0, tzinfo=UTC)
    repo.mark_blocked(work_item_id=seeded_leased_item.work_item_id,
                      queue_key=None, barrier_key="agg-1",
                      now=now, expected_lease_owner="w1")
    row = _fetch(repo, seeded_leased_item.work_item_id)
    assert row.barrier_blocked_at == now
    assert row.status == "blocked"

def test_list_blocked_barrier_items_returns_only_barrier_blocked_for_run(repo) -> None:
    # seed: one barrier-BLOCKED in run A, one READY in run A, one BLOCKED in run B,
    # and one ADR-028 QUEUE-hold in run A (status=blocked, queue_key set, barrier_key NULL)
    items = repo.list_blocked_barrier_items(run_id="run-A")
    assert [i.status for i in items] == [TokenWorkStatus.BLOCKED]
    assert items[0].barrier_key is not None      # queue-hold NOT swept in
    assert items[0].barrier_blocked_at is not None
    # deterministic iteration order: (barrier_key, ingest_sequence, work_item_id);
    # buffer ORDER comes from batch_members.ordinal at restore (Task 2.1), not from here
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — `mark_blocked` adds `barrier_blocked_at=now` to its UPDATE values (only this verb writes it; queue-holds get stamped too — harmless, nothing reads it on that arm). `list_blocked_barrier_items(run_id) -> list[TokenWorkItem]`: own read transaction, `WHERE run_id = :run_id AND status = 'blocked' AND barrier_key IS NOT NULL` (BLOCKED is dual-use — the barrier_key filter is what keeps ADR-028 queue-holds out, per the D1 discrimination rule), ORDER BY `(barrier_key, ingest_sequence, work_item_id)`; extend the `TokenWorkItem` row-mapping dataclass with `barrier_blocked_at: datetime | None`. No event (read-only).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(scheduler): mark_blocked stamps barrier_blocked_at; list_blocked_barrier_items read verb (F1)`

### Task 1.4: Phase-0/1 checkpoint gate

- [ ] **Step 1:** `pytest tests/unit/contracts tests/unit/core/landscape tests/unit/core/checkpoint/test_manager.py -q` — green. Record the full-suite failure count for reference (`pytest tests/ -q | tail -3`) but do NOT expect baseline yet; recovery/orchestrator/e2e reds are the Phase 2–4 worklist. `mypy` on touched files green (`.venv/bin/mypy src/elspeth/contracts src/elspeth/core/checkpoint src/elspeth/core/landscape`).

## Phase 2 — Executors restore from the journal; live scalars capture

### Task 2.1: AggregationExecutor.restore_from_journal

**Files:**
- Modify: `src/elspeth/engine/executors/aggregation.py` (replace `restore_from_checkpoint` :684-796 and `get_checkpoint_state` :607-682)
- Test: `tests/unit/engine/test_executors.py` (rewrite group-B tests in place)

- [ ] **Step 1: Write the failing test** (model on the retired `test_restore_from_checkpoint_creates_pipeline_row` :2886):

```python
def test_restore_from_journal_rebuilds_buffer_and_attempt_offset(agg_executor) -> None:
    # payloads carry a typed value (datetime) — proves journal round-trip fidelity
    # (Known-risks item 2, MANDATORY)
    items = [_blocked_item(token_id="t1", row_id="r1",
                           payload=_payload({"v": 1, "at": datetime(2026, 6, 1, tzinfo=UTC)}),
                           blocked_at=T0),
             _blocked_item(token_id="t2", row_id="r2",
                           payload=_payload({"v": 2, "at": datetime(2026, 6, 2, tzinfo=UTC)}),
                           blocked_at=T0 + 5s)]
    agg_executor.restore_from_journal(
        node_id=NodeID("agg-1"),
        items=items,
        member_order=["t2", "t1"],               # batch_members.ordinal order (Task 3.1)
        batch_id="batch-9",                      # derived by caller (Task 3.1)
        accepted_count_total=2,                  # derived: COUNT(DISTINCT token_id)
        completed_flush_count=0,                 # derived: COUNT(batches completed)
        scalars=AggregationNodeScalars(None, None),
        attempt_offsets={"t1": 1, "t2": 1},      # max_attempt+1 discipline
        resume_checkpoint_id="cp-0",
        now=T0 + 60s,
    )
    node = agg_executor._nodes[NodeID("agg-1")]
    assert [t.token_id for t in node.tokens] == ["t2", "t1"]   # ordinal order, not item order
    assert node.buffers[0]["at"] == datetime(2026, 6, 2, tzinfo=UTC)
    assert node.batch_id == "batch-9"
    assert node.accepted_count_total == 2
    assert all(t.resume_attempt_offset == 1 and t.resume_checkpoint_id == "cp-0"
               for t in node.tokens)
    # trigger age restored from absolute blocked_at, not an offset blob
    assert node.trigger.get_age_seconds() == pytest.approx(60.0)

def test_restore_from_journal_counter_only_node(agg_executor) -> None:
    agg_executor.restore_from_journal(
        node_id=NodeID("agg-1"), items=[], member_order=[], batch_id=None,
        accepted_count_total=5, completed_flush_count=2,
        scalars=AggregationNodeScalars(None, None),
        attempt_offsets={}, resume_checkpoint_id="cp-0", now=T0)
    node = agg_executor._nodes[NodeID("agg-1")]
    assert node.completed_flush_count == 2 and node.accepted_count_total == 5
    assert node.batch_id is None and node.tokens == []

def test_restore_from_journal_null_blocked_at_is_corruption(agg_executor) -> None:
    # post-epoch-20 every BLOCKED row was stamped; NULL = corruption (Known-risks 8)
    with pytest.raises(AuditIntegrityError):
        agg_executor.restore_from_journal(
            node_id=NodeID("agg-1"),
            items=[_blocked_item(token_id="t1", row_id="r1",
                                 payload=_payload({"v": 1}), blocked_at=None)],
            member_order=["t1"], batch_id="b", accepted_count_total=1,
            completed_flush_count=0, scalars=AggregationNodeScalars(None, None),
            attempt_offsets={"t1": 1}, resume_checkpoint_id="cp-0", now=T0)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `restore_from_journal(*, node_id, items, member_order, batch_id, accepted_count_total, completed_flush_count, scalars, attempt_offsets, resume_checkpoint_id, now)`:
  - per item: raise `AuditIntegrityError` if `item.barrier_blocked_at is None` (corruption — every post-epoch-20 BLOCKED row is stamped); `row = deserialize_row_payload(item.row_payload_json)` (import from scheduler repo module or move the helper to a shared home); rebuild `TokenInfo(token_id=item.token_id, row_id=item.row_id, row_data=row, branch_name=item.branch_name, fork_group_id=..., join_group_id=..., expand_group_id=..., resume_attempt_offset=attempt_offsets[item.token_id], resume_checkpoint_id=resume_checkpoint_id)`;
  - **buffer ORDER comes from `member_order`** (the caller's `batch_members.ordinal` ordering — the authoritative accept order; journal iteration order ties on ingest_sequence for fork/expand siblings and work_item_id hashes don't sort by acceptance): `node.tokens`/`node.buffers` rebuilt in `member_order`; `node.batch_id = batch_id`; `node.member_count = len(items)`; the membership reconcile against persisted `batch_members` (old `_reconcile_checkpoint_batch_members` :798-835) degenerates to **set-equality** (order is now sourced FROM batch_members, so ordered-tuple comparison would be circular) — keep the helper, simplify it, document why;
  - counters: `node.accepted_count_total = accepted_count_total`, `node.completed_flush_count = completed_flush_count` (both audit-derived by the caller — D3);
  - trigger: `elapsed = max(0.0, (now - min(item.barrier_blocked_at for item in items)).total_seconds()) if items else 0.0`; call `node.trigger.restore_from_checkpoint(batch_count=len(items), elapsed_age_seconds=elapsed, count_fire_offset=scalars.count_fire_offset, condition_fire_offset=scalars.condition_fire_offset)` (signature unchanged, engine/triggers.py:256-317 untouched);
  - empty-items arm: batch_count=0, elapsed 0.0, offsets passed through (the old restore passes offsets at batch_count=0 — aggregation.py:770-783; mirror it).
  - Delete `get_checkpoint_state`/`restore_from_checkpoint`/`AGGREGATION_CHECKPOINT_VERSION` and the `restored_state`/`restore_state`/`get_restored_state` plugin stash (recon: zero production callers; only tests/unit/engine/test_executors.py references — rewrite those alongside). Add `get_barrier_scalars() -> dict[NodeID, AggregationNodeScalars]` reading the live `node.trigger.get_count_fire_offset()`/`get_condition_fire_offset()`/`completed_flush_count` (the trigger getters at triggers.py:234-254 stay).
- [ ] **Step 4: Run the rewritten executor tests, verify pass.**
- [ ] **Step 5: Commit** — `feat(engine): aggregation buffers restore from BLOCKED journal rows; blob restore deleted (F1)`

### Task 2.2: CoalesceExecutor.restore_from_journal

**Files:**
- Modify: `src/elspeth/engine/coalesce_executor.py` (replace `restore_from_checkpoint` :299-359 and `get_checkpoint_state` :257-297; keep `_reconstruct_completed_keys_from_landscape` :361-385)
- Test: `tests/unit/engine/test_coalesce_executor.py`

- [ ] **Step 1: Write the failing test:**

```python
def test_restore_from_journal_rebuilds_pending_with_absolute_arrivals(coalesce_executor) -> None:
    items = [_blocked_item(token_id="tA", row_id="r1", branch_name="left",
                           node_id="co-1", coalesce_name="merge", blocked_at=T0),
             _blocked_item(token_id="tB", row_id="r1", branch_name="right",
                           node_id="co-1", coalesce_name="merge", blocked_at=T0 + 3s)]
    coalesce_executor.restore_from_journal(
        items=items,
        scalars={("merge", "r1"): CoalescePendingScalars(lost_branches={"mid": "lost"})},
        state_ids={"tA": "st-1", "tB": "st-2"},   # derived from node_states (Task 3.1)
        attempt_offsets={"tA": 1, "tB": 1},       # D5 applies to coalesce too
        resume_checkpoint_id="cp-0",
        now=T0 + 10s,
    )
    pending = coalesce_executor._pending[("merge", "r1")]
    assert set(pending.branches) == {"left", "right"}
    assert pending.lost_branches == {"mid": "lost"}
    assert pending.branches["left"].state_id == "st-1"
    # first_arrival is the earliest blocked_at, expressed on the executor clock
    assert pending.branches["right"].arrival_time - pending.first_arrival == pytest.approx(3.0)
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `restore_from_journal(*, items, scalars, state_ids, attempt_offsets, resume_checkpoint_id, now)`:
  - group items by `(item.coalesce_name, item.row_id)`; per group build `_PendingCoalesce` with `first_arrival = clock.monotonic() - max(0.0, (now - min(blocked_at)).total_seconds())` (same `max(0.0, …)` clamp as Task 2.1 — a wall-clock backward step must not put first_arrival in the monotonic future) and each `_BranchEntry(token, arrival_time=first_arrival + (blocked_at - min_blocked_at).total_seconds(), state_id=state_ids[token_id])` — same monotonic-clock anchoring trick the old restore used (:335-353), sourced from absolutes; rebuilt `TokenInfo` carries `resume_attempt_offset=attempt_offsets[token_id]` and `resume_checkpoint_id` (D5);
  - `lost_branches` from `scalars.get(key)`, default `{}` (missing entry = none recorded — document the staleness window per D3);
  - branch_name required non-empty and `barrier_blocked_at` required non-NULL (raise `AuditIntegrityError` otherwise — that row is corrupt; add a test for each);
  - `_completed_keys` reconstruction from Landscape unchanged;
  - delete `get_checkpoint_state`/`restore_from_checkpoint`/`COALESCE_CHECKPOINT_VERSION`; add `get_barrier_scalars() -> dict[tuple[str, str], CoalescePendingScalars]` (live `lost_branches` per pending key).
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(engine): coalesce pending state restores from BLOCKED journal rows; blob restore deleted (F1)`

### Task 2.3: complete_barrier — atomic consume+emit

**Files:**
- Modify: `src/elspeth/core/landscape/scheduler_repository.py` (new verb; `mark_blocked_barrier_terminal` :1648-1752 and `mark_blocked_barrier_pending_sink_many` :1349-1452 become its arms)
- Test: `tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py` (new)

- [ ] **Step 1: Write failing tests:**

```python
def test_complete_barrier_consumes_and_emits_atomically(repo) -> None:
    # seed 3 BLOCKED rows under barrier_key="agg-1" for run R
    n = repo.complete_barrier(
        run_id="R", barrier_key="agg-1",
        consumed_token_ids=["t1", "t2", "t3"],
        emitted_pending_sink=[BarrierEmission(
            token_id="t-agg-out", row_id="r-agg", row_payload_json=PAYLOAD,
            sink_name="out", outcome="success", path="aggregated",
            step_index=4, ingest_sequence=1)],
        emitted_ready=[], now=NOW)
    assert n == 3
    assert {r.status for r in _rows(repo, "R", ["t1", "t2", "t3"])} == {"terminal"}
    out = _row_for_token(repo, "R", "t-agg-out")
    assert out.status == "pending_sink" and out.node_id is None  # terminal lane

def test_complete_barrier_refuses_partial_consumed_set(repo) -> None:
    # 3 BLOCKED rows, only 2 consumed -> AuditIntegrityError, nothing changed
    with pytest.raises(AuditIntegrityError):
        repo.complete_barrier(run_id="R", barrier_key="agg-1",
                              consumed_token_ids=["t1", "t2"],
                              emitted_pending_sink=[], emitted_ready=[], now=NOW)
    assert {r.status for r in _rows(repo, "R", ["t1", "t2", "t3"])} == {"blocked"}

def test_complete_barrier_crash_atomicity(repo) -> None:
    # natural failure INSIDE the txn: emit an item whose work_item_id collides with an
    # existing row (the deterministic sha256 identity) -> IntegrityError inside the txn
    # -> the consumed transitions MUST roll back with it
    with pytest.raises(Exception):
        repo.complete_barrier(run_id="R", barrier_key="agg-1",
                              consumed_token_ids=["t1", "t2", "t3"],
                              emitted_pending_sink=[_emission_colliding_with("t1")],
                              emitted_ready=[], now=NOW)
    assert {r.status for r in _rows(repo, "R", ["t1", "t2", "t3"])} == {"blocked"}
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `complete_barrier(*, run_id, barrier_key, consumed_token_ids, emitted_pending_sink, emitted_ready, now, leased_exclusion_token_id=None) -> int`:
  - one `self._engine.begin()`; SELECT BLOCKED rows under `(run_id, barrier_key)`; both-direction set-equality vs `consumed_token_ids ∪ {handoff transitions}` preserving the exact `AuditIntegrityError` semantics of :1683-1690/:1747-1751 (the leased triggering token is excluded via `leased_exclusion_token_id`, matching today's `leased_token_id` exclusions);
  - consumed → TERMINAL with payload scrub (reuse `_scrub..."purged"` helper :2075-2078); per-row `MARK_BLOCKED_BARRIER_TERMINAL` events with `{"barrier_key": ...}` context;
  - `emitted_pending_sink`: if the token already has a BLOCKED row under this barrier (passthrough), transition it BLOCKED→PENDING_SINK with the handoff bundle (absorbing `mark_blocked_barrier_pending_sink_many`'s body); else INSERT a fresh PENDING_SINK row on the node_id-NULL terminal lane (deterministic work_item_id, attempt=1, `MARK_PENDING_SINK` event);
  - `emitted_ready`: INSERT READY rows (absorb the `enqueue_ready` insert body on the same conn; `ENQUEUE` events);
  - every emission event's `context_json` carries `{"barrier_key": ..., "consumed_count": N}` so the atomic completion is reconstructable from scheduler_events alone (today's barrier events already carry barrier_key context — extend the convention to emissions);
  - introduce a small frozen `BarrierEmission` dataclass in `contracts/scheduler.py` for the emission payloads (fields per the test above plus the lineage/coalesce columns);
  - rewrite `mark_blocked_barrier_terminal` and `mark_blocked_barrier_pending_sink_many` as delegating wrappers (`complete_barrier(consumed_token_ids=..., emitted_pending_sink=[], ...)` etc.), keeping their public signatures. **End state: both wrappers remain public post-F1** — the RC6 pumping proof and the lifecycle state machine reference them, so "inline and retire" is not on the table; Task 3.4 only migrates the processor's *flush* call sites.
- [ ] **Step 4: Run, verify pass; also run the lease-race safety net** `pytest tests/unit/core/landscape/ -q` — all green.
- [ ] **Step 5: Commit** — `feat(scheduler): complete_barrier — atomic barrier consume+emit transition (F1)`

### Task 2.4: Checkpoint callback carries scalars

**Files:**
- Modify: `src/elspeth/engine/orchestrator/checkpointing.py` (`BatchCheckpointCallback.__call__` :121-146, `CheckpointCoordinator.maybe_checkpoint` :50-105, `checkpoint_interrupted_progress` :156-192)
- Modify: `src/elspeth/engine/processor.py` (`get_aggregation_checkpoint_state` :808-818 / `get_coalesce_checkpoint_state` :820-824 → one `get_barrier_scalars()`)
- Test: `tests/unit/engine/orchestrator/test_graceful_shutdown.py`, plus the existing checkpointing unit home

- [ ] **Step 1: Write/adjust the failing test** — `test_creates_checkpoint_even_with_no_buffered_state` (:150) becomes: shutdown checkpoint is written with `barrier_scalars_json IS NULL` when nothing has state; an aggregation whose count trigger has latched writes `count_fire_offset` into the scalars (counters are NOT in the scalars — they derive from audit, D3).
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — processor exposes `get_barrier_scalars() -> BarrierScalars` (composing the two executors' `get_barrier_scalars()`); both checkpoint writers pass it to the Task-1.2 `create_checkpoint` signature. The blob-era asymmetry (sink path passed empty agg state, shutdown nulled it — recon contradiction 3) unifies: `None` when `not scalars.has_state`.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(engine): checkpoint writers carry BarrierScalars from live executors (F1)`

## Phase 3 — Resume reads the journal; run-start checkpoint; flush call sites

### Task 3.1: Processor restore rewiring (the inversion itself)

**Files:**
- Modify: `src/elspeth/engine/processor.py` (delete `_restore_scheduler_blocks_from_aggregation_checkpoint` :514-554 and `_restore_scheduler_blocks_from_coalesce_checkpoint` :556-586; rework the `__init__` restore block :498-507)
- Modify: `src/elspeth/engine/orchestrator/run_core.py` (`build_processor` coalesce restore :357-358)
- Test: `tests/unit/engine/test_processor.py` (rewrite group-C pins)

- [ ] **Step 1: Write the failing tests** (replacing `test_restores_aggregation_state` :459 / `test_resume_materializes_scheduler_blocks_from_coalesce_checkpoint` :4431 / `test_coalesce_checkpoint_restore_rejects_stale_scheduler_metadata` :4528):

```python
def test_resume_restores_aggregation_buffers_from_blocked_rows(...) -> None:
    # seed journal: 2 BLOCKED rows barrier_key="agg-1" with payloads + barrier_blocked_at,
    # seed audit: batch_members for batch "b1", BUFFERED token_outcomes with batch_id "b1",
    # node_states at attempt 0 for both tokens.
    processor = _build_resuming_processor(resume_point=_resume_point(scalars=...))
    node = processor._aggregation_executor._nodes[NodeID("agg-1")]
    assert [t.token_id for t in node.tokens] == ["t1", "t2"]
    assert node.batch_id == "b1"
    assert all(t.resume_attempt_offset == 1 for t in node.tokens)  # max_attempt(0)+1
    # and NO new journal rows were created: the BLOCKED rows are reused as-is
    assert _journal_row_count(db, run_id) == _count_before

def test_resume_restores_coalesce_pending_from_blocked_rows(...) -> None:
    ...  # mirror for a held branch; state_id resolved from the PENDING node_state
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** the new restore block in `RowProcessor.__init__` (replacing :498-507):
  - `items = self._scheduler.list_blocked_barrier_items(run_id)` (only when resuming — gate on the resume signal that previously gated blob restore);
  - **partition on `barrier_key`** (D1 rule — a row's `node_id` is the enqueue cursor, NOT the barrier): `barrier_key ∈ self._coalesce_node_ids.keys()` (keyed by `CoalesceName`, processor.py:408) → coalesce; `barrier_key ∈` the str-keyed aggregation node ids (from the aggregation settings map) → aggregation; neither → `AuditIntegrityError` naming the orphan barrier_key;
  - aggregation, per **barrier_key** group: `batch_id` ← BUFFERED `token_outcomes.batch_id` of the group's tokens (reuse the `_barrier_key_for_buffered_scheduler_result` derivation, :3201-3234), passed through the `handle_incomplete_batches` remap (Task 3.2 hands it in); cross-check all tokens in a group agree on batch_id (`AuditIntegrityError` if split); `member_order` + `accepted_count_total` ← `batch_members` query ordered by `ordinal`, counting `COUNT(DISTINCT token_id)` (NOT raw COUNT — `retry_batch` copies members to the retry batch, execution_repository.py:1847, and `handle_incomplete_batches` runs at resume.py:585 *before* this derivation, so raw COUNT double-counts); `completed_flush_count` ← `COUNT(batches WHERE status='completed')` for the node (D3); `attempt_offsets` ← from `recovery.get_incomplete_tokens_by_row`'s max_attempt map (already computed on the resume path — plumb it through rather than re-query); call `restore_from_journal(...)` (Task 2.1);
  - coalesce: `state_ids` ← node_states PENDING holds per token at the coalesce node (one query); `attempt_offsets` as above; call Task-2.2 `restore_from_journal`;
  - counter-only nodes (zero BLOCKED rows but audit shows completed batches, or a scalars entry exists): call `restore_from_journal(items=[], ...)` with the derived counters — preserving post-flush pagination across resume (the old counter-only snapshots, executors/aggregation.py:618-622);
  - add one pin test: an ADR-028 queue-hold row (status=blocked, queue_key set, barrier_key NULL) is neither restored as a barrier member nor (Task 3.2) excluded from re-drive accounting — the queue resume path is untouched by F1;
  - delete both `_restore_scheduler_blocks_from_*` methods.
- [ ] **Step 4: Run the rewritten processor tests, verify pass.**
- [ ] **Step 5: Commit** — `feat(engine): resume rebuilds barrier buffers FROM the journal; blob->journal materialization deleted (F1)`

### Task 3.2: ResumeCoordinator + recovery on journal truth

**Files:**
- Modify: `src/elspeth/core/checkpoint/recovery.py` (`can_resume`; `_get_buffered_checkpoint_token_ids` :797-812 → journal query; `get_unprocessed_rows` :549-666 / `get_incomplete_tokens_by_row` :699-765 blob-exclusion arms)
- Modify: `src/elspeth/engine/orchestrator/resume.py` (`reconstruct_resume_state` :478-611; the no-work arm :719-764; delete `rebind_checkpoint_batch_ids` usage :594-597)
- Delete: `src/elspeth/engine/orchestrator/aggregation.py` `rebind_checkpoint_batch_ids` :121-158 (its remap moves: `handle_incomplete_batches`' old→new batch mapping now feeds Task 3.1's batch_id derivation directly)
- Test: `tests/unit/core/checkpoint/test_recovery.py`, `tests/unit/engine/orchestrator/test_resume_failure.py`

- [ ] **Step 1: Write the failing tests:**

```python
def test_buffered_exclusion_reads_journal_not_blob(db, recovery) -> None:
    # row r1: token t1 BLOCKED in journal, incomplete in token_outcomes
    # -> r1 is NOT in get_unprocessed_rows (buffered = will be restored, not re-driven)
    assert "r1" not in {r.row_id for r in recovery.get_unprocessed_rows(run_id, ...)}

def test_post_checkpoint_buffered_token_is_restored_not_redriven(db, recovery) -> None:
    # BLOCKED row exists; checkpoint (scalars) predates it -> still excluded from re-drive
    ...

def test_can_resume_reason_for_missing_baseline_is_journal_flavoured(...) -> None:
    check = recovery.can_resume(run_id_with_no_checkpoint, graph)
    assert not check.can_resume
    assert "checkpoint" not in check.reason.lower() or "baseline" in check.reason.lower()
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement:**
  - `_get_buffered_checkpoint_token_ids` → `_get_buffered_journal_token_ids(run_id)`: `SELECT token_id FROM token_work_items WHERE run_id=? AND status='blocked' AND barrier_key IS NOT NULL` (the `barrier_key` filter keeps ADR-028 queue-holds out of the buffered-exclusion — a queue-held token is NOT "waiting at a barrier" and must stay in the re-drive work-set); both exclusion arms (:574-578, :750-751) repoint;
  - `can_resume`: keep the status gate (`check_run_status_resumable`, post-F3-near) and the topology validation against the latest checkpoint; the "No checkpoint found for recovery" reason becomes "Run has no resume baseline (run predates run-start checkpointing or checkpointing was disabled)" — D4 makes a checkpoint row exist for every checkpointing-enabled run;
  - `reconstruct_resume_state`: the `restored_state`/`restored_coalesce_state` returns shrink to a `has_restored_barrier_work: bool` derived from `scheduler.list_blocked_barrier_items(run_id)` being non-empty (or a cheap count); `rebind_checkpoint_batch_ids` call (:594-597) deleted — instead `handle_incomplete_batches`' remap dict is returned/threaded into processor construction (Task 3.1 consumes it); the no-work arm condition becomes `not unprocessed_rows and not has_restored_barrier_work`;
  - `ResumePoint.barrier_scalars` threads through to processor construction (`build_processor`).
- [ ] **Step 4: Run, verify pass:** `pytest tests/unit/core/checkpoint tests/unit/engine/orchestrator/test_resume_failure.py -q`.
- [ ] **Step 5: Commit** — `feat(orchestrator): resume work-set and buffered-exclusion derive from journal; batch rebind folds into restore (F1)`

### Task 3.3: Run-start checkpoint (D4)

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py` (run() startup, near the :1058 checkpoint-factory wiring)
- Test: `tests/unit/engine/orchestrator/` (new test beside the checkpointing tests)

- [ ] **Step 1: Write the failing test:**

```python
def test_run_writes_sequence_zero_checkpoint_at_start(orchestrator_harness) -> None:
    # run a 1-row pipeline with checkpointing enabled
    rows = _select_checkpoints(db, run_id)
    assert rows[0].sequence_number == 0
    assert rows[0].barrier_scalars_json is None
    assert rows[0].upstream_topology_hash == compute_full_topology_hash(graph)

def test_resume_does_not_rewrite_sequence_zero(...) -> None:
    # crash after start, resume; exactly one sequence-0 row exists
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — in `run()` immediately after the checkpoint coordinator is constructed and before source iteration: `if checkpoint_config.enabled and manager is not None: manager.create_checkpoint(run_id=..., sequence_number=0, barrier_scalars=None, graph=...)`. `CheckpointCoordinator._sequence_number` starts at 0 and pre-increments (:78), so post-sink checkpoints begin at 1 — no collision; the duplicate-sequence guard (manager.py:128-137) is the backstop. Resume path: `rebase_sequence(resume_point.sequence_number)` already skips 0 forward; do NOT add a run-start write to resume.
- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `feat(orchestrator): sequence-0 run-start checkpoint — topology baseline always exists (F1/D4)`

### Task 3.4: Flush call sites adopt complete_barrier (out-of-claim window closed)

**Files:**
- Modify: `src/elspeth/engine/processor.py` (`_mark_blocked_sink_results_pending` :2727-2775; `_mark_buffered_scheduler_work_terminal` :2707-2725; coalesce consume :2688-2705 + merged-child enqueue :2437-2442/:2954-2955; timeout/EOF flush arms)
- Modify: `src/elspeth/engine/orchestrator/source_iteration.py` (EOF flush :607-616) and `outcomes.py` (`_mark_barrier_tokens_terminal` :92-111) as call-site survey dictates
- Test: `tests/integration/pipeline/test_aggregation_recovery.py` (rewrite), new crash-window test

- [ ] **Step 1: Write the failing test** (the window elspeth-ae5183307b's family leaves open even after Phase 3.1–3.3):

```python
def test_timeout_flush_output_is_journal_durable_before_sink_write(...) -> None:
    # drive a timeout flush whose sink write crashes; assert the emitted aggregate
    # token has a PENDING_SINK row (node_id NULL lane) and the consumed tokens are
    # TERMINAL — atomically (no state where consumed are TERMINAL and output absent).
    # Harness: reuse the crash-injection pattern of test_aggregation_recovery.py /
    # the _FailOnceEOFBatchTransform-style fixture in test_rc6_eof_resume_proof.py
    # (raise on first sink write, not a mock of the repo).
```

- [ ] **Step 2: Run, verify fail** (today the output token has no row).
- [ ] **Step 3: Implement** — migrate each flush call site to ONE `complete_barrier` call per barrier completion:
  - in-claim count-trigger passthrough (:1649-1659): consumed + handoffs in one call (`leased_exclusion_token_id=current_token.token_id`);
  - in-claim TRANSFORM mode (:1664): consumed only (the emitted aggregate rides the LEASED claim as today);
  - out-of-claim (timeout/EOF) flush: consumed + emitted_pending_sink for sink-bound outputs (this is the new durability — the emission insert replaces the memory-only `pending_tokens` handoff for journal purposes; `pending_tokens` keeps feeding the in-process sink write, and the post-sink callback's `mark_sink_bound_scheduler_terminal_many` (:3084-3096) terminalizes the inserted rows — verify the token_ids align);
  - coalesce fire (:2418-2443): consumed siblings + (non-terminal) merged child as `emitted_ready` in the same call, deleting the separate `enqueue_ready` hop;
  - the wrapper verbs stay public permanently (Task 2.3 end-state note) — this task migrates only the processor flush sites listed above.
- [ ] **Step 4: Run, verify pass** plus the chaos battery: `pytest tests/integration/engine/test_multi_source_chaos.py tests/integration/pipeline/test_aggregation_recovery.py -q`.
- [ ] **Step 5: Commit** — `feat(engine): barrier flush is one atomic journal transition; out-of-claim flush outputs journal-durable (F1/D6)`

## Phase 4 — Delete the blob layer; flip the pinned gaps

### Task 4.1: Delete blob contracts, blob columns, RESTORE_BLOCKED (the subtractive half of epoch 20)

**Files:**
- Delete: `src/elspeth/contracts/aggregation_checkpoint.py`, `src/elspeth/contracts/coalesce_checkpoint.py`
- Modify: `src/elspeth/core/landscape/schema.py` — DROP `checkpoints.aggregation_state_json`/`coalesce_state_json`; remove `'restore_blocked'` from the scheduler_events event_type CHECK (epoch stays 20 — unreleased, one bump covers both halves; history entry from Task 0.2 already names this)
- Modify: `src/elspeth/contracts/__init__.py` (drop re-exports :24, :73, :426, :429), `src/elspeth/core/checkpoint/recovery.py` (`_RestoredCheckpointStates` remnants), `src/elspeth/core/landscape/scheduler_repository.py` (delete `ensure_blocked_barrier_work_item` :308-451), `src/elspeth/contracts/scheduler.py` (remove `RESTORE_BLOCKED`, :27), `src/elspeth/core/landscape/database.py` (required-columns lists), `src/elspeth/engine/orchestrator/types.py` (:33-35, :127-130, :585-586 blob type references)
- Tests: retire group A (test_coalesce_checkpoint.py, test_checkpoint_post_init.py, blob halves of test_checkpoint.py, test_version_validation.py, blob round-trip halves of test_checkpoint_properties.py — topology-hash properties KEEP and repoint); delete `tests/integration/pipeline/test_aggregation_checkpoint_bug.py` (defends the deleted column); add to `test_schema_epoch_and_required_columns.py`:

```python
def test_checkpoint_blob_columns_are_gone() -> None:
    assert "aggregation_state_json" not in checkpoints_table.c
    assert "coalesce_state_json" not in checkpoints_table.c

def test_restore_blocked_event_type_is_gone() -> None:
    assert "restore_blocked" not in {e.value for e in SchedulerEventType}
```

- [ ] **Step 1:** `grep -rn "AggregationCheckpointState\|CoalesceCheckpointState\|ensure_blocked_barrier_work_item\|aggregation_state_json\|coalesce_state_json\|RESTORE_BLOCKED" src/` — work the hit list to zero (the recon §6 reader inventory is the checklist).
- [ ] **Step 2:** retire/rewrite the group-A test files per the matrix in Part III, plus the deleted-surface pins outside group A: `tests/unit/contracts/test_audit.py:501-515` (Checkpoint blob fields), `tests/unit/core/landscape/test_schema.py:86-88` (column enumeration), `tests/unit/core/landscape/test_scheduler_events.py:738/:807` (RESTORE_BLOCKED), ResumePoint-constructor users `tests/unit/engine/orchestrator/test_resume_guardrails.py` + `test_orchestrator_registry_bootstrap.py`. Run `pytest tests/unit/contracts tests/unit/core/landscape tests/property/core -q` green.
- [ ] **Step 3:** `grep -rn` again over `src/ tests/` — zero hits.
- [ ] **Step 4: Commit** — `refactor(contracts)!: delete checkpoint blob layer + blob columns — journal is the only barrier-buffer truth (F1)`

### Task 4.2: Flip the EOF characterization (elspeth-ae5183307b)

**Files:**
- Modify: `tests/integration/pipeline/test_rc6_eof_resume_proof.py` (flip target `test_real_eof_flush_crash_is_not_yet_resumable` = :298-367; durable-state proof :337-362; refusal assert :364-367)

- [ ] **Step 1: Flip per the test's own docstring** (:309-311): rename to `test_real_eof_flush_crash_resumes_from_journal`; keep the durable-state proof section (:337-362 — exhausted lifecycle, 3 batch_members, 3 BLOCKED rows) and replace the refusal assert (:364-367) with the positive roundtrip modeled on the sibling (:190): `check.can_resume is True`; `Orchestrator.resume(...)` completes; `output_sink.results == [{"value": 60, "count": 3}]`; `transform.batch_calls == 2` post-resume (the crashing `process` increments BEFORE raising, :117-120 — 1 crashed + 1 resumed); source `load_invocations` unchanged (no replay); all token_work_items terminal; run COMPLETED; **no node_states UNIQUE violation and both attempts visible in node_states** (elspeth-262911c26b acceptance). **Do NOT touch `test_interrupted_source_still_fails_resume_safely` (:369)** — that one pins a fail-safe refusal that must survive F1.
- [ ] **Step 2: Run, verify pass.** If it fails on the attempt seam, the bug is in Task 3.1's attempt_offsets derivation — fix there, not in the test.
- [ ] **Step 3:** The sibling `test_exhausted_source_eof_aggregation_resumes_without_source_replay` (:190) asserts `aggregation_state_json is not None` / `resume_point.aggregation_state is not None` — rewrite those asserts to journal surfaces (BLOCKED rows + `barrier_scalars`); it stays as the graceful-shutdown-variant proof (Part III group E covers it).
- [ ] **Step 4: Commit** — `test(resume): EOF-flush crash now resumable from journal — characterization flipped (closes elspeth-ae5183307b path)`

### Task 4.3: Sweep the remaining blob-coupled tests (groups B–G)

**Files:** per the Part III matrix.

- [ ] **Step 1:** work the matrix file-by-file (each file = one focused edit + targeted run). The two e2e harness files edit ONLY their named seams; assertions byte-identical.
- [ ] **Step 2:** settle elspeth-e1dd5e1303 — this needs a PRODUCTION step, not just a pin flip. Decision: unified `rows_buffered` = **N**, the audit value (every accepted member has a BUFFERED token_outcome, including the in-claim triggering token; the journal's BLOCKED rows are N−1 because the Nth rides the LEASED claim). Production step: the live counter today only counts buffered *drain results* at outcomes.py:385 — move the increment to where the BUFFERED token_outcome is recorded (processor.py:1598-1603 region) so live == audit by construction, or derive the live figure from the BUFFERED-outcome count at summary time (pick whichever the counter plumbing makes smaller; state which in the commit). Then flip the pin (`test_fork_join_balance.py::test_count_equals_n_rows_buffered_divergence_is_pinned`, :4186) to assert live == derive == N and rename `test_rows_buffered_live_equals_derive_after_unification`. Record the decision in the elspeth-e1dd5e1303 close comment.
- [ ] **Step 3:** `pytest tests/unit tests/integration tests/e2e tests/property -q` — failure set ⊆ inherited baseline (see Gates).
- [ ] **Step 4: Commit** — `test: blob-coupled suites rewritten to journal-truth surfaces (F1 sweep)`

## Phase 5 — Counters family + docs + gates + tickets

### Task 5.1: rows_coalesce_failed audit arm (elspeth-7294de558e)

**Files:**
- Modify: the resume counter derivation in `src/elspeth/engine/orchestrator/resume.py` (:823-887 graft region)
- Test: `tests/integration/test_adr_019_resume_counter_parity.py::test_resume_grafts_rows_coalesce_failed_from_timeout_redrive`

- [ ] **Step 1:** the durable evidence for failed coalesces already exists pre-D6 (`_fail_pending` writes FAILED node_states + token_outcomes); the gap is the *derive* arm. Build it with all three dimensions specified:
  - **granularity:** the counter is per pending-KEY (one failed coalesce = one row), but token_outcomes are per BRANCH token — a naive outcome count over-reports; count DISTINCT (coalesce node, row_id) pairs;
  - **attribution:** token_outcomes carry no node_id — anchor the query on FAILED `node_states` at the run's coalesce node_ids (or the `failure_reason` family `_fail_pending` writes, e.g. timeout/late-arrival reasons — pick whichever is already indexed and pin it with a unit test);
  - **cumulativity:** the derived value must cover run-1 failures AND resumed-run failures (the parity pin :644-664 asserts run_A==3 vs run_B==1 today precisely because the resume-only graft forgets run-1) — derive over the whole run_id history, then flip the pin to equality.
  NOTE: this test is currently in the inherited-red set — landing this task REMOVES one inherited failure; record that in the report.
- [ ] **Step 2: Run, verify the parity test passes.**
- [ ] **Step 3: Commit** — `fix(orchestrator): rows_coalesce_failed derives from durable audit on resume (closes elspeth-7294de558e)`

### Task 5.2: Docs

**Files:**
- Modify: `docs/architecture/barrier-machinery.md` (paired-surfaces rows naming get_checkpoint_state/restore_from_checkpoint/_restore_scheduler_blocks_* → the journal-restore surfaces; the F1 forward note becomes "landed" with this plan's commit SHAs; checklist item 5's attempt discipline now cites the journal column)
- Modify: `docs/contracts/system-operations.md` if it names the blob classes (grep)
- Verify: ADR-029 cross-references final symbol names

- [ ] **Step 1: Update; grep docs/ for the deleted symbol names — zero stale references.**
- [ ] **Step 2: Commit** — `docs: barrier machinery + ADR-029 reflect journal-as-truth (F1 landed)`

### Task 5.3: Gates

- [ ] **Step 1:** Full suite: `env -u VIRTUAL_ENV .venv/bin/pytest tests/ -q | tail -5`. Compare the failure SET (not count) against `notes/f1-pretask-fullsuite-baseline.txt` (committed in Task 0.2 Step 0 — do NOT rely on anything in /tmp). The adr_019 coalesce-parity red retires via Task 5.1. For baseline-red tests in subsystems F1 touches (engine, checkpoint, landscape), also eyeball that the failure MODE is unchanged, not just the test ID. Any new failure = fix before commit.
- [ ] **Step 2:** `mypy`: `.venv/bin/mypy src/elspeth` — clean (495 files at baseline).
- [ ] **Step 3:** Safety-net explicit pass: chaos (5) + lease races (5) + concurrent-resume/rejection (7) + RC6 proofs (9) + the flipped EOF roundtrip — all green; verify the three zero-edit files are byte-identical to HEAD~ (`git diff --stat` shows no hits).
- [ ] **Step 4:** Trust-tier: displacements are EXPECTED on this change set — scheduler_repository.py and recovery.py carry signed allowlist fingerprints, deleting `ensure_blocked_barrier_work_item` strands its per-file entries, and the new frozen `barrier_scalars.py` may trip immutability rules. Report every displacement in the final report; never re-sign (no HMAC); `SKIP=elspeth-lints-trust-tier` ONLY for provably pre-existing failures (state which). Plugin hashes: none of the touched files are plugins (verify: `git diff --name-only <base>.. | grep plugins/`).

### Task 5.4: Tickets + memory

- [ ] **Step 1:** filigree (CLI, `--actor claude-f1-durability`; `add-comment ISSUE_ID TEXT` is positional):
  - close elspeth-4d5cbf2fcf with commit SHAs + one-paragraph design summary;
  - close elspeth-ae5183307b ("dissolved: journal is the truth; no flush-boundary checkpoint writer needed; flipped test = <name>");
  - close elspeth-262911c26b ("restore_from_checkpoint deleted; replacement carries max_attempt+1 via journal; covered by flipped EOF roundtrip");
  - close elspeth-e1dd5e1303 ("decision: rows_buffered = journal value N; pin flipped to equality");
  - close elspeth-7294de558e (Task 5.1 SHA);
  - comment elspeth-ce3adfb7b7 (verify whether journal-derived cumulative counters now reach the FAILED ceremony; close if yes, else state what remains);
  - comment elspeth-1396d3f790 ("F1 landed — blocker clears; the two concurrent-resume characterizations are your flip targets") and remove the dependency edge.
- [ ] **Step 2:** memory: update `project_rc6_canonical_merge_2026-06-10.md` (F1 LANDED section: SHAs, epoch 20, DB-delete owed, next = option c design) + the MEMORY.md index line.

---

# Part III — Test-sweep matrix (from recon, grouped)

| Group | Files | Disposition |
|---|---|---|
| A blob-DTO contracts | test_coalesce_checkpoint.py; test_checkpoint_post_init.py; blob halves of test_checkpoint.py, test_version_validation.py, test_checkpoint_properties.py | RETIRE (topology-hash properties KEEP, repoint to new create_checkpoint signature) |
| B executor restore | test_executors.py (12 restore refs); test_coalesce_executor.py (15); test_triggers.py (23) + test_trigger_properties.py (3); test_post_init_validations.py (11) | REWRITE carrier to restore_from_journal/scalars; trigger-restore semantics tests KEEP (TriggerEvaluator API unchanged); landscape-reconstruction tests (:2526, :2584) KEEP |
| C processor re-derivation pins | test_processor.py :459, :4354, :4431, :4528, :6145 | RETIRE direction-pins; replace per Task 3.1 tests |
| D manager/recovery | test_recovery.py (unit+integration), test_manager.py, test_compatibility.py, test_topology_validation.py | REWRITE signatures; topology/compat semantics KEEP |
| E orchestrator flows | test_aggregation_checkpoint_bug.py DELETE; test_aggregation_recovery.py, test_rc6_checkpoint_interrupted_edge.py, test_aggregation.py, test_resume_failure.py, both graceful-shutdown files, test_orchestrator_checkpointing.py, test_crash_and_resume.py | REWRITE to journal truth; harness `create_checkpoint` users (test_resume_comprehensive.py et al.) get mechanical signature edits |
| F attempt-offset family | test_identity.py, test_resume_offset_propagation.py, epoch/columns test | KEEP (invariant survives; D5 strengthens it); epoch test edited in Tasks 0.2 + 4.1 |
| G fork/join property suite | test_fork_join_balance.py | invariants KEEP; checkpoint-construction harness REWRITE; :4186 pin FLIP (Task 4.3) |
| H deleted-surface pins outside A–G | test_audit.py:501-515 (Checkpoint blob fields); landscape test_schema.py:86-88 (column enumeration); test_scheduler_events.py:738/:807 (RESTORE_BLOCKED); ResumePoint ctors in test_resume_guardrails.py + test_orchestrator_registry_bootstrap.py | REWRITE in Task 4.1 |
| I D4 count-shift family (run-start checkpoint adds +1 per run) | test_explicit_sink_routing.py:387 (3→4); test_resume_comprehensive.py:522/:766 (1→2); test_orchestrator_checkpointing.py:97/:139 (+1); sweep `grep -rn "checkpoint" tests/ | grep -iE "len\(|count"` for stragglers | UPDATE counts in Task 3.3 (same commit as the feature, so the suite never sees the shift un-pinned) |
| J EOF proof | test_rc6_eof_resume_proof.py | :298 FLIP (Task 4.2); :190 sibling REWRITE to journal surfaces (Task 4.2 Step 3); :369 fail-safe DO NOT TOUCH |

---

# Commit/PR shape

~15 conventional commits as tasked (0.1 docs+ADR, 0.2 baseline+additive schema, 1.x contracts/manager/repo, 2.x executors + complete_barrier + callback, 3.x resume inversion + run-start checkpoint + flush sites, 4.x subtractive schema + deletions + flips + sweep, 5.x counters + docs). No push. Mid-chain note: full-suite green is only expected at phase boundaries 0–1 (targeted dirs), end of Phase 3 (most of the suite), and Task 5.3 (baseline-equal); the recovery/orchestrator window between Tasks 1.2 and 3.2 is deliberately red on resume surfaces — each task still keeps its OWN targeted selectors green before committing. DB epoch bump means: after merge, operator deletes Landscape DBs (staging included); SESSION DB untouched.

# Known risks the implementer must re-verify in code (do not trust this plan blindly)

1. **`handle_incomplete_batches` remap semantics** (Task 3.2/3.1): confirm what it returns and that derived batch_ids must route through it; if it mutates audit in place and returns nothing, derive batch_id AFTER it runs. Remember `retry_batch` COPIES members to the retry batch (execution_repository.py:1847) — hence `COUNT(DISTINCT token_id)` everywhere.
2. **`serialize_row_payload` round-trip fidelity** for restored buffers (MANDATORY, not optional): the Task 2.1 Step 1 fixture includes a typed value (datetime; add Decimal if the contract system passes them) — the old byte-identity validation never exercised this path for cursor≠barrier or attempt>1 rows, so this is the only proof.
3. **Out-of-claim emission token_id alignment** (Task 3.4): the post-sink terminalize callback keys on token_ids — the inserted PENDING_SINK rows must use the emitted tokens' real token_ids.
4. **Trigger restore when the only batch member is the in-claim triggering token** (count==1): no BLOCKED rows, no timing — confirm the trigger needs no restore in that arm (it fired in-claim; nothing pending).
5. **`leased_exclusion_token_id`** must thread through every in-claim complete_barrier call or the set-equality check will fire spuriously (today's `leased_token_id` exclusions show every needed site).
6. **`barrier_blocked_at` NULL on legacy paths:** rows blocked before this code never exist post-epoch-bump (DB deleted), so a NULL barrier_blocked_at on a BLOCKED row is corruption — restore raises `AuditIntegrityError`, never defaults (pinned by tests in Tasks 2.1/2.2).
7. **Do not "fix" the accept-then-crash-before-mark_blocked window** (D7): resume refuses it via the membership reconcile (`AuditIntegrityError`) — pre-existing, deliberate, out of scope. If Task 3.4's tests trip it, the test harness is crashing in the wrong window.
8. **Dual-use BLOCKED everywhere:** any new query over `status='blocked'` MUST carry the `barrier_key IS NOT NULL` (barrier) or `queue_key IS NOT NULL` (queue) qualifier. The Task 3.1 queue-hold pin test is the canary.

# Review provenance

Recon: 4-agent parallel fact-pack sweep (blob write path / resume read path / schema+repo / tests+tickets), workflow wf_fefd36fb-32b. Review: 4-lens panel (reality, architecture, quality, systems) + synthesis, workflow wf_11dfe537-e62 — verdict GO-WITH-FIXES; all 12 required fixes and the minor set applied in this revision (notably: barrier_key-keyed discrimination replacing the node-id rule the panel proved wrong; counters fully derived from audit; additive→subtractive schema phasing; rows_buffered production step).
