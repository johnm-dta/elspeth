# ADR-029: Scheduler Journal Is the Single Source of Barrier-Buffer Truth

**Date:** 2026-06-11
**Status:** Accepted (amended 2026-06-13 — journal-first acceptance,
snapshot-scoped exhaustiveness, branch-loss records; see *Amendment* note
below and *Amendment detail* at the end)
**Deciders:** John Morrissey, Claude Fable 5
**Tags:** scheduler, checkpoint, resume, barrier, aggregation, coalesce,
          durability, f1, multi-source-token-scheduler

## Amendment (2026-06-13, ADR-030 slice 3)

ADR-030 (one-host WAL pack, multi-worker coordination) slice 3 amends this
ADR in place. Original text is preserved with dated inline annotations
(*"Amended 2026-06-13 (ADR-030 slice 3): ..."*); the full rationale lives in
the *Amendment detail* section at the end of this document. The five changes:

- **The in-claim flush arm is removed** (§E.2) — the owned N=1 behavior
  change: a barrier flush now happens at drain-iteration intake
  (journal-first) instead of inside the triggering token's claim.
- **The D7 memory-first window is structurally closed** (polarity flip):
  `mark_blocked` now *precedes* the in-memory accept on every path; the
  invariant inverts to journal-BLOCKED ⊇ `batch_members`.
- **`complete_barrier` exhaustiveness narrows to durable ∩ intake-snapshot,
  per firing group** (§E.3) — the one invariant relaxation, with structural
  compensations (per-iteration intake, late-arrival release, EOF quiescence
  gating, finalize refusing COMPLETED while BLOCKED rows survive).
- **Late arrivals are journal-released** via
  `mark_blocked_barrier_terminal(late_arrival)` with an evented release
  context (§E.3a).
- **`coalesce_branch_losses` becomes the durable loss truth** (§E.5); the
  D3 checkpoint scalar is demoted to a cross-check.

Authority: ADR-030
([030-multi-worker-deployment-shape.md](030-multi-worker-deployment-shape.md))
and the design note
`notes/option-c-multi-worker-coordination-design-2026-06-11.md`
(§D, §E.1–E.5, §G, §H).

## Context

The fork/coalesce architecture assessment
(`notes/fork-coalesce-architecture-assessment-2026-06-10.md`) identified a
structural durability gap in how barrier-buffered tokens survive crash and
resume. ELSPETH's token execution model has three durability layers:

1. **The scheduler journal** (`token_work_items`): every schedulable
   continuation has a row before a worker touches it. A buffered token is
   already written to the journal as a `BLOCKED` row at buffer time — the
   `mark_blocked` transition is called by the drain loop immediately after the
   in-memory accept. *Amended 2026-06-13 (ADR-030 slice 3): the order is
   inverted — `mark_blocked` now precedes the in-memory accept on every path
   (journal-first arrival); the accept happens at the leader's fenced
   adoption (`adopt_blocked_barrier_item`) during per-iteration intake.*
2. **The checkpoint blob** (`checkpoints.aggregation_state_json` /
   `coalesce_state_json`): periodically serialized executor in-memory state.
   Contains membership lists, timing offsets, trigger latches, and
   pagination counters for every live barrier.
3. **The Landscape audit trail** (`token_outcomes`, `node_states`, `rows`,
   `tokens`): the immutable, per-mutation legal record. Aggregation batch
   membership (`BUFFERED` token_outcomes) and coalesce node_states are written
   per-accept, before the BLOCKED transition. *Amended 2026-06-13 (ADR-030
   slice 3): membership and the BUFFERED outcome are now written atomically
   WITH the adoption marker (`barrier_adopted_epoch` NULL→epoch CAS), AFTER
   the BLOCKED row exists — one `BEGIN IMMEDIATE` transaction.*

The gap: on resume today, executor buffers are rebuilt *only* from the blob.
The journal is reconciled *to* the blob via `ensure_blocked_barrier_work_item`
(an idempotent insert manufacturing BLOCKED rows at a hardcoded default
`attempt=1`). This means:

- The journal's BLOCKED rows are treated as derivative of the blob, not
  authoritative.
- A crash between flush output emission and the next checkpoint write leaves
  flush outputs in `pending_tokens` memory with no journal row — an
  unresumable window (elspeth-ae5183307b).
- Blob-vs-journal skew is possible: a post-sink checkpoint can capture a
  memory-buffered token whose journal row is still `LEASED`; a subsequent
  `ensure_blocked` against a `LEASED` row hard-fails with `AuditIntegrityError`.
- A live `BLOCKED` row at `attempt > 1` can collide with a manufactured
  `attempt = 1` twin.

The blob carries six categories of state beyond what the journal and audit
trail can supply: per aggregation node `elapsed_age_seconds`,
`count_fire_offset`, `condition_fire_offset`, `accepted_count_total`,
`completed_flush_count`, `batch_id`; per coalesce pending key
`elapsed_age_seconds`, `lost_branches`; per coalesce token `state_id`,
`arrival_offset_seconds`; plus the `contract_version` integrity hash and
version gates.

The assessment concluded that most of this is derivable from durable facts
already in the journal and audit trail. Only trigger latches and
`lost_branches` are genuinely underivable and need to be stored. The
architectural fix is to invert the read direction: the journal becomes the
authoritative barrier-buffer truth; the blob layer is deleted; the checkpoint
row shrinks to a small scalar-metadata JSON for the truly underivable scalars.

## Decision

The scheduler journal (`token_work_items` BLOCKED rows) is the single source
of truth for barrier-buffer membership and payload on resume. The blob layer
(`AggregationCheckpointState` / `CoalesceCheckpointState` JSON families) is
deleted. The checkpoint row is retained as a durable scalar store for the
small set of state that cannot be derived from the journal or audit trail.
Flush becomes one atomic journal transition.

### D1 — Buffered-token membership and payload truth = BLOCKED journal rows

On resume, a new read verb `list_blocked_barrier_items(run_id)` feeds
`restore_from_journal(...)` on both executors. Row payloads come from
`deserialize_row_payload(row_payload_json)` — the same round-trip already
used for `PENDING_SINK` re-drive (`processor.py:3098-3121`). The two
`_restore_scheduler_blocks_from_*` methods, `ensure_blocked_barrier_work_item`,
and the `RESTORE_BLOCKED` event type are deleted.

*Barrier discrimination rule:* a BLOCKED row's `node_id` is the
enqueue-time cursor, not the barrier node (`mark_blocked` and `_transition`
never touch `node_id` — `scheduler_repository.py:1204`, `:1939`). `barrier_key`
is the barrier identity by construction (`processor.py:3460-3466`: coalesce →
`coalesce_name`, aggregation → `str(aggregation node_id)`). BLOCKED is dual-use
— queue-holds are BLOCKED rows with `queue_key` set and `barrier_key` NULL.
Barrier restore and the resume buffered-exclusion both select
`status='blocked' AND barrier_key IS NOT NULL`; queue-holds are untouched.
Partition: `barrier_key` in coalesce names (keys of `_coalesce_node_ids`,
which is keyed by `CoalesceName` — `processor.py:408`) → coalesce;
`barrier_key` in str-keyed aggregation node ids (from the aggregation settings
map) → aggregation; neither → `AuditIntegrityError`. Aggregation rows may
carry non-NULL `coalesce_name` *lineage* (`processor.py:531-537`), so never
discriminate on `coalesce_name IS NOT NULL`.

*Integrity argument:* the journal row was written by the live run at buffer
time. The `ensure_blocked_barrier_work_item` byte-identity validation passing
today is only weak evidence (it fires solely on `work_item_id` hash collision
— attempt-1 cursor-matching rows); a typed-payload round-trip fixture is
mandatory. The aggregation-only `contract_version` "checkpoint may be
corrupted" check (`processor.py:524-529`) retires with the blob — the journal
payload embeds the contract and is not a re-serialization.

### D2 — Absolute timestamps replace offset arithmetic

A new nullable column `token_work_items.barrier_blocked_at` (DateTime tz) is
written by `mark_blocked` only. (It is stamped at the drain's BLOCKED
transition — a few ms after the in-memory accept — so restored ages are
conservatively slightly *younger*; harmless, and stamped on queue-holds too,
where nothing reads it.) *Amended 2026-06-13 (ADR-030 slice 3):
`barrier_blocked_at` now precedes the accept; the fenced adoption is
backdated to it, so the live frame and the restore frame share one durable
timing anchor — a batch's fire time is a pure function of durable state +
config, invariant under leader takeover (the §H pinned doctrine). Same
conservative direction this decision already accepted.* On restore:

- Coalesce per-branch `arrival_time` ← row's `barrier_blocked_at`;
  `first_arrival` ← min over branches; `elapsed_age_seconds` ← now −
  `first_arrival`. The blob's `arrival_offset_seconds` / `elapsed_age_seconds`
  (offset reconstruction, stale by last-checkpoint-age) are deleted with
  *better* fidelity, not worse.
- Aggregation `first_accept_time` ← min `barrier_blocked_at` of the node's
  BLOCKED rows (the first accepted row of a batch always blocks; a count-1
  trigger fires in-claim and has no timing to restore). *Amended 2026-06-13
  (ADR-030 slice 3): the in-claim parenthetical is superseded — no in-claim
  firing exists; every accepted token, including a count-1 trigger member,
  has a BLOCKED row and durable timing.*
  `TriggerEvaluator.restore_from_checkpoint(batch_count, elapsed_age_seconds, count_fire_offset, condition_fire_offset)`
  keeps its signature — only the source of elapsed changes.

Reusing `updated_at` for arrival times is rejected (see *Alternatives
Considered*). A dedicated column is honest and the epoch bumps anyway.

### D3 — Derive what audit already proves; checkpoint keeps only true scalars

The following quantities are derivable from durable audit records and are not
stored:

- `accepted_count_total` ← `COUNT(batch_members)` over all batches of the
  node in the run (each accept writes `batch_members` durably in its own
  transaction before the BLOCKED transition — `executors/aggregation.py:194-202`).
- `batch_id` per restored token ← the token's `BUFFERED` `token_outcomes.batch_id`
  (existing derivation pattern, `processor.py:3201-3234`), passed through
  `handle_incomplete_batches`' dead-batch remap.
- Coalesce `state_id` ← query `node_states` for the token's PENDING hold at
  the coalesce node (precedent: `_completed_keys` is already
  Landscape-reconstructed, `coalesce_executor.py:328-385`). Not stored, no
  new column.
- `completed_flush_count` ←
  `COUNT(batches WHERE status='completed' AND aggregation node = ?)` —
  the only COMPLETED-status `complete_batch` call is
  the flush-success path (`executors/aggregation.py:491`) where the counter
  increments (`:501`); failed flushes complete FAILED (`:516`/`:532`).
  Derivation is *fresher* than any stored scalar (no lost-increment crash
  window between flush success and the next checkpoint).

**Stored** (the truly underivable scalars), in one new
`checkpoints.barrier_scalars_json` Text column replacing the two blob columns:
per aggregation node `{count_fire_offset, condition_fire_offset}` (the two
trigger latches); per coalesce pending key `{lost_branches}`. That is the
whole inventory. A new small contracts module `contracts/barrier_scalars.py`
replaces both blob families. `format_version` 4→5. *Amended 2026-06-13
(ADR-030 slice 3): the durable truth for `lost_branches` moves to the
append-only `coalesce_branch_losses` table (§E.5) — losses are recorded in
the same lease-fenced transaction as the lossy disposition, re-derived from
the full ledger on restore, and replayed through `notify_branch_lost` before
the next trigger evaluation. The checkpoint scalar is retained as a
cross-check only; the ledger wins.*

*Staleness audit:* stored scalars are stale by last-checkpoint-age — same as
the blob today, no regression. A missing count latch is healed by the existing
conservative re-latch when `batch_count >= config.count`
(`engine/triggers.py:300-304`). A lost condition latch delays firing to the
next accept/EOF flush (same membership, possibly different `which_triggered`
attribution — the resumed run records what actually happened). `lost_branches`
lost in the window degrades `require_all`/`best_effort` evaluation to the
timeout path, identical to today's blob loss window. *Amended 2026-06-13
(ADR-030 slice 3): the `lost_branches` loss window is closed — the
`coalesce_branch_losses` record rides the lease-fenced disposition
transaction, so there is no checkpoint-age staleness for branch losses.*
Counters (`accepted_count_total`,
`completed_flush_count`) have no staleness at all post-F1 — both are derived
from per-mutation audit.

### D4 — A checkpoint row always exists: sequence-0 at run start

`run()` writes an initial checkpoint (sequence_number 0,
`upstream_topology_hash`, empty scalars) before the first source row.
Consequences:

- Topology-compatibility validation stays unconditional.
- `can_resume`'s "No checkpoint found" arm becomes a genuine refusal only for
  runs that predate the run-start write or ran with checkpointing disabled
  (reason text becomes journal-flavoured).
- `rebase_sequence` is unchanged (resume rebases to the latest sequence; it
  never writes a fresh 0 — the unique `(run_id, sequence_number)` index would
  refuse it anyway).
- Checkpoint deletion on success removes the sequence-0 row with the rest
  (lifecycle unchanged).
- Restored `TokenInfo` gets `resume_checkpoint_id` from this always-present
  resume point, satisfying the
  `resume_attempt_offset > 0 ⟹ resume_checkpoint_id is not None` invariant.

### D5 — Attempt discipline from the journal

Restored barrier tokens — both aggregation and coalesce — rebuild `TokenInfo`
with `resume_attempt_offset` derived from `node_states` `max_attempt + 1` (the
`processor.py:2323` discipline), exactly as `get_incomplete_tokens_by_row`
already computes (`recovery.py:722-728`), and `resume_checkpoint_id` from the
resume point. This fixes elspeth-262911c26b by construction —
`AggregationExecutor.restore_from_checkpoint`'s default-0 rebuild
(`aggregation.py:729-739`) is deleted, and both replacements carry the offset.

### D6 — Flush is one atomic journal transition

A new repository verb `complete_barrier(...)` performs, in a single
transaction:

1. Validate the consumed set against the durable BLOCKED set (both directions,
   preserving the `:1683-1690` / `:1747-1751` cross-checks in
   `scheduler_repository.py`). *Amended 2026-06-13 (ADR-030 slice 3): the
   validation universe narrows from the whole durable BLOCKED set to the
   per-firing-group intake snapshot (§E.3). The verb takes
   `intake_snapshot_token_ids` — exactly the tokens durably adopted into THIS
   firing group — plus `scope_row_id` validation, and enforces a three-arm
   algebra: snapshot − durable ⇒ Tier-1; (durable ∩ snapshot) − consumed −
   handed_off ⇒ Tier-1; durable − snapshot ⇒ late arrivals — legitimate,
   left BLOCKED, recorded as `late_arrival_token_ids` in every emission event
   context. `intake_snapshot_token_ids=None` preserves the durable-universe
   semantics for direct repository callers. Run-boundary orphan detection is
   preserved by the finalize quiescence predicate: a run refuses to finalize
   COMPLETED while BLOCKED rows survive.*
2. Consumed → `TERMINAL` (payload scrub).
3. Sink-bound emissions → `PENDING_SINK` (transition for buffered passthrough
   tokens; *insert* on the `node_id`-NULL terminal lane for merged/aggregate
   outputs — the `uq_token_work_items_terminal_identity` partial unique index
   is that lane).
4. Continuation emissions → `READY` inserts.

`mark_blocked_barrier_terminal` and `mark_blocked_barrier_pending_sink_many`
become delegating wrappers over `complete_barrier`. This closes the out-of-claim flush
window: a timeout/EOF flush output is journal-durable (`PENDING_SINK`) the
moment its inputs are consumed. A crash before sink write recovers via the
existing `PENDING_SINK` re-drive. ~~In-claim flush keeps the `LEASED`
triggering-token exclusion exactly as today (the claim's own result rides the
normal drain arms).~~ *Amended 2026-06-13 (ADR-030 slice 3): superseded — the
in-claim arm is removed (§E.2) and `leased_exclusion_token_id` is deleted
from `complete_barrier`. The triggering token is journal-BLOCKED like any
arrival, adopted at the next intake, and flushed out-of-claim in the same
drain iteration — an owned N=1 timing change (§H timing-invariance doctrine).*

### Scope exclusions (D7)

The following are explicitly out of scope for this change:

- **The memory-first window** between in-memory buffer/accept and the drain's
  `mark_blocked`. A crash there leaves a `batch_members` row with no BLOCKED
  journal row; on resume the membership reconcile refuses with
  `AuditIntegrityError` (a re-drive would PK-violate `(batch_id, token_id)`).
  That refusal is pre-existing semantics, unchanged by F1. *Amended
  2026-06-13 (ADR-030 slice 3): the window no longer exists — the polarity
  flips. The invariant inverts to journal-BLOCKED ⊇ `batch_members`:
  membership-without-BLOCKED is structurally unreachable at epoch 21 (the
  membership row, the adoption marker and the BUFFERED outcome commit in one
  fenced transaction against an already-BLOCKED row) and REMAINS Tier-1 if
  ever observed. BLOCKED-without-membership is now a legitimate
  intake-pending state (`barrier_adopted_epoch IS NULL`) — a restore-reconcile
  disposition adopted by the next leader's intake, NOT corruption.*
- **Cross-process worker coordination** (option c, elspeth-1396d3f790) — F1
  unblocks it, does not start it. *Amended 2026-06-13: started — ADR-030.*
- **Any shared Barrier abstraction** (F3-long-term) — twin implementations
  stay twins; every change here is made to both sides per the
  `docs/architecture/barrier-machinery.md` checklist.
- **Checkpoint cadence/config semantics** (`RuntimeCheckpointConfig.frequency`)
  — untouched.

## Consequences

The following observable behavior changes result from implementing D1–D6:

1. **Virgin EOF-flush crashes become resumable** (elspeth-ae5183307b
   dissolved). Flush outputs are journal-durable the moment inputs are
   consumed; the crash window that previously left them only in `pending_tokens`
   memory is closed.
2. **Tokens buffered after the last checkpoint are restored into buffers
   instead of re-driven from source** — strictly fewer duplicate executions on
   resume. This is a correctness improvement: pre-F1, tokens accepted between
   the last checkpoint and the crash were not in the blob and so were treated
   as unprocessed rows and re-driven; post-F1, their BLOCKED journal rows
   restore them directly.
3. **`rows_buffered` live-vs-derive unifies on N** (audit value: every accepted
   member gets a `BUFFERED` `token_outcomes` row, including the in-claim
   triggering token; the live counter today only counts buffered *drain results*
   = N−1 — `outcomes.py:385`). This requires a production step, not just a pin
   flip (elspeth-e1dd5e1303 decision forced). *Amended 2026-06-13 (ADR-030
   slice 3): the synthetic in-claim BUFFERED tally mechanism is deleted; the
   N parity now holds structurally — every member, the trigger arrival
   included, is an ordinary blocked-arrival result.*
4. **`can_resume` reason strings become journal-flavoured** — "No checkpoint
   found for recovery" rewords to reflect the new checkpoint-always-exists
   invariant; only runs predating D4 or running with checkpointing disabled
   trigger the refusal.
5. **`which_triggered` attribution** after a crash in the condition-latch window
   may report the resumed run's actual trigger rather than the pre-crash latch.
   This is honest: the resumed run records what actually happened rather than
   replaying a stale inference.
6. **DB epoch 19→20.** Operators must delete their Landscape DBs before the
   first run on the new code (delete-the-DB policy; SESSION_SCHEMA_EPOCH
   untouched). *Amended 2026-06-13: advanced again, 20→21, under ADR-030
   slice 2 (`barrier_adopted_epoch` column + coordination tables).*

*Consequences 7–9 added by the 2026-06-13 amendment (ADR-030 slice 3):*

7. **Count/condition flush timing moves from in-claim to same-iteration
   intake** (§E.2) — the triggering token gains one extra BLOCKED→consumed
   journal transition and the flush fires at the next intake step of the same
   drain iteration (~ms latency at N=1). Batch composition and audit identity
   are unchanged; only the timing moves, and the move is what makes fire
   timing invariant under leader takeover (§H).
8. **Late arrivals are journal-released** (§E.3a): a BLOCKED row arriving
   after its group completed is released via
   `mark_blocked_barrier_terminal` with a `late_arrival` release context —
   evented, never silently dropped, never wedging a COMPLETED finalize.
9. **The end-of-input flush is quiescence-gated** (§D steps 2–3): the EOF
   flush waits for zero READY/LEASED journal rows, then loops
   intake → evaluate → flush until no BLOCKED rows remain — so a slow peer's
   member joins the one EOF batch instead of stranding or splitting it.

## Alternatives Considered

### Alternative 1: Scalars on a new Tier-1 `barrier_state` table

**Description:** Instead of slimming the checkpoint row, introduce a separate
`barrier_state` table in the Landscape database to store per-barrier scalars
(trigger latches, `lost_branches`, membership counts) as live operational rows.
Workers would update the table on every accept and flush, making the state
queryable without a checkpoint read.

**Rejected because:** Gold-plating for the current single-worker deployment
shape. The scalar set is small and write-rare (written once per flush, read
only on resume). A dedicated table would add cross-table transactional
complexity — every accept would need to maintain two rows atomically — without
unlocking any capability the checkpoint-row scalar store does not provide at
N=1. The case where live cross-worker metadata *is* needed (option c,
elspeth-1396d3f790) requires a separate cross-process coordination design (D7
scope exclusion); that design can adopt a `barrier_state` table as part of its
own ADR when the deployment shape is settled. Building the table now
is premature generalization.

*Amended 2026-06-13 (ADR-030 slice 3): ADR-030 decided NOT to adopt a
`barrier_state` table — the barrier plane is leader-owned, fed by
journal-arrival hand-off (§E.1), so no live cross-worker barrier metadata
table is needed. The table remains available to a future distributed-barriers
ADR.*

### Alternative 2: Scalars denormalized onto BLOCKED rows

**Description:** Store per-barrier scalars (trigger latches, `lost_branches`,
flush counters) directly on the `token_work_items` BLOCKED rows — one row
carries the barrier's aggregate metadata alongside its token payload.

**Rejected because:** Counter-only and post-flush state has zero rows to carry
it. The scalar quantities (`count_fire_offset`, `condition_fire_offset`,
`completed_flush_count`) belong to the barrier node, not to any individual
token. A flush empties all BLOCKED rows of a barrier; post-flush, there are no
BLOCKED rows left to carry the flush counter or the trigger-latch state. A
denormalization approach would require keeping at least one sentinel BLOCKED
row alive past terminalization — a lifecycle violation that conflicts with D6's
atomic consume-and-terminate model. The dedicated `barrier_scalars_json`
column on the checkpoint row is a cleaner separation: one durable row per
barrier-bearing run carries the underivable scalars; every BLOCKED row carries
its token.

### Alternative 3: Reuse `updated_at` for arrival timestamps

**Description:** Instead of adding `barrier_blocked_at` (D2), derive
coalesce arrival and aggregation first-accept times from the existing
`token_work_items.updated_at` column, which is written on every state
transition including the `mark_blocked` write.

**Rejected because:** Fragile coupling. `updated_at` is written by every state
transition, not just `mark_blocked`. Any future transition write on a BLOCKED
row — a lease touch, a heartbeat update, an operator repair tool — would
silently corrupt the arrival time. The column's semantics would become load-
bearing for a purpose it was not designed for, with no schema-level enforcement
preventing drift. A dedicated `barrier_blocked_at` column is an honest, single-
write column: it is written by `mark_blocked` and never again. The schema epoch
bumps for D2 anyway; adding one nullable column is negligible cost.

## Related Decisions

- **ADR-030** ([multi-worker-deployment-shape](030-multi-worker-deployment-shape.md))
  — the one-host WAL pack (option c, elspeth-1396d3f790) and the authority
  for the 2026-06-13 amendment to this ADR. Slice 3 of its landing plan
  lands journal-first barrier acceptance: the §E.2 in-claim-arm removal, the
  §E.3 snapshot-scoped exhaustiveness, the §E.3a late-arrival release, and
  the §E.5 branch-loss ledger amended into D2/D3/D6/D7 above.
- **ADR-026** ([durable-token-scheduler](026-durable-token-scheduler.md)) —
  the scheduler primitive that F1 promotes to barrier-buffer truth. The 4-key
  claim order (`ingest_sequence, step_index, created_at, work_item_id`), the
  CAS discipline, and the `BLOCKED` lifecycle state that F1 builds on are all
  defined and justified there.
- **ADR-028** ([queue-vs-coalesce-not-duplicates](028-queue-vs-coalesce-not-duplicates.md))
  — establishes that BLOCKED is dual-use (queue-holds have `queue_key` set and
  `barrier_key` NULL; barrier-holds have `barrier_key` set). D1's
  `barrier_key IS NOT NULL` filter is the direct consequence of ADR-028's
  dual-use ruling.
- **[barrier-machinery.md](../barrier-machinery.md)** — the twin-implementation
  checklist and structural documentation for aggregation + coalesce. D1–D6
  changes apply to both sides per that checklist.
- **ADR-024** (delivery governance for single-maintainer mode) — preserved.
  The checkpoint row and journal remain embedded SQLite state; no external
  service or separate store is added.
- **ADR-019** (two-axis terminal model) — preserved. The `TERMINAL` / `FAILED`
  axis distinction on scheduler rows is maintained through the atomic flush
  transition in D6.

## References

### Architecture assessment

- `notes/fork-coalesce-architecture-assessment-2026-06-10.md` — the durability
  gap analysis that motivated this ADR; findings F1–F6, particularly the
  blob-vs-journal skew risk and the unresumable flush-output window.
- `notes/option-c-multi-worker-coordination-design-2026-06-11.md` — the
  multi-worker coordination design behind the 2026-06-13 amendment (§D
  finalization, §E.1 journal-arrival hand-off, §E.2 in-claim arm removal,
  §E.3/§E.3a snapshot exhaustiveness + late-arrival release, §E.5 branch-loss
  hand-off, §H pinned timing-invariance doctrine).

### Plan

- Historical F1 durability-unification implementation plan — preserved in git
  history or maintainer-local archives. Part I (I.1–I.4) is recorded here as
  the architecture decision; Part II contained the task-level implementation
  guide.

### Code (pre-F1 state, cited as reference for what changes)

- `src/elspeth/core/landscape/scheduler_repository.py` —
  `mark_blocked` (:1204-1228), `ensure_blocked_barrier_work_item`
  (:308-451), `mark_blocked_barrier_terminal` (:1648-1752),
  `mark_blocked_barrier_pending_sink_many` (:1349-1452). The last two become
  internal arms of `complete_barrier` under D6.
- `src/elspeth/core/landscape/schema.py` — `checkpoints` table
  (:903-924); `SQLITE_SCHEMA_EPOCH = 19` (:98); `token_work_items_table`
  (:396-449). Epoch advances to 20; `barrier_blocked_at` column added to
  `token_work_items`; `barrier_scalars_json` column added to `checkpoints`;
  blob columns removed.
- `src/elspeth/engine/processor.py` — resume blob-read path (:503-504),
  `ensure_blocked_barrier_work_item` callers (:514-554, :556-586),
  `PENDING_SINK` re-drive (:3098-3121), barrier discriminator
  (:3460-3466).
- `src/elspeth/core/checkpoint/recovery.py` —
  `get_incomplete_tokens_by_row` attempt derivation (:722-728),
  buffered-row exclusion (:574-578, :668-679, :797-812).
- `src/elspeth/engine/executors/aggregation.py` —
  `restore_from_checkpoint` (:729-739, deleted under D5).
- `src/elspeth/engine/coalesce_executor.py` —
  `_completed_keys` Landscape reconstruction (:328-385).
- `src/elspeth/engine/triggers.py` — conservative count re-latch
  (:300-304).
- `src/elspeth/engine/orchestrator/outcomes.py:385` — live `rows_buffered` counter
  (N−1 pre-F1; unified to N post-F1 per D3/I.4 item 3).
- `src/elspeth/contracts/barrier_scalars.py` — new module replacing both
  blob families under D3.

### Code (2026-06-13 amendment surfaces, ADR-030 slice 3)

- `src/elspeth/core/landscape/scheduler_repository.py` —
  `adopt_blocked_barrier_item` (the fenced backdated-adoption verb:
  epoch fence + `barrier_adopted_epoch` NULL→epoch CAS + `batch_members` +
  BUFFERED `token_outcomes` in one `BEGIN IMMEDIATE` transaction; the
  adoption CAS is the only double-BUFFERED guard), `complete_barrier`'s
  `intake_snapshot_token_ids` three-arm validation (`leased_exclusion_token_id`
  deleted), `record_coalesce_branch_loss` /
  `list_unadopted_coalesce_branch_losses` / `adopt_coalesce_branch_losses`
  (§E.5 ledger verbs), and the `late_arrival` release context on
  `mark_blocked_barrier_terminal`.
- `src/elspeth/engine/processor.py` — per-iteration journal-first intake;
  `_complete_coalesce_fire` / the aggregation flush completion passing the
  per-firing-group snapshot; branch-loss replay before trigger evaluation.

### Tickets

- **elspeth-4d5cbf2fcf** — the F1 durability unification epic this ADR
  implements.
- **elspeth-ae5183307b** — unresumable EOF-flush window, dissolved by D6.
- **elspeth-262911c26b** — aggregation resume default-0 attempt offset bug,
  fixed by D5.
- **elspeth-e1dd5e1303** — `rows_buffered` live-vs-derive discrepancy,
  forced by D3/I.4 item 3.
- **elspeth-1396d3f790** — cross-process worker coordination (option c),
  unblocked by F1 (D7 scope exclusion).

## Amendment detail (2026-06-13): ADR-030 slice 3 — journal-first barrier acceptance

ADR-030 (One-Host WAL Pack, option-c multi-worker coordination) slice 3
supersedes two clauses of this ADR; the inline annotations above record each
supersession at its site, and this section carries the consolidated
rationale. The journal remains the single source of
barrier-buffer truth; this amendment *strengthens* that doctrine by making the
journal write the FIRST act of barrier acceptance.

### D7 polarity flip — the memory-first window is closed

D7 excluded "the memory-first window between in-memory buffer/accept and the
drain's `mark_blocked`". That window no longer exists, because the order is
inverted (§E.2): a claimed token arriving at a barrier records NOTHING — no
executor memory, no `batch_members`, no BUFFERED `token_outcomes` — and simply
returns its `(None, BUFFERED)` result so the drain marks the journal row
BLOCKED. Executor memory is fed ONLY by the per-drain-iteration journal-first
intake, which adopts each intake-pending row (`barrier_adopted_epoch IS NULL`)
via the leader-fenced `adopt_blocked_barrier_item` verb: the epoch CAS, the
`batch_members` row and the BUFFERED outcome commit in ONE `BEGIN IMMEDIATE`
transaction, then memory is fed.

Consequences of the flip:

- **journal-BLOCKED is a superset of `batch_members`**: a BLOCKED row may
  briefly exist without membership (intake-pending — the legitimate
  crash-recovery disposition, adopted by the next leader's intake), but
  membership-without-BLOCKED stays Tier-1 *and is now structurally
  unreachable* (the only `batch_members` writer for barrier arrivals commits
  atomically with the adoption marker on an already-BLOCKED row).
- **Backdated accept timing**: the BUFFERED outcome's `recorded_at` and every
  trigger latch anchor at the row's durable `barrier_blocked_at` (the D2
  stamp), converted onto the monotonic scale with the same clamped transform
  the restore path uses. A batch's timeout fire time is therefore a pure
  function of durable state + config — invariant under leader takeover (the
  §H pinned doctrine). This is the same conservative direction D2 already
  accepted.
- **The in-claim count/condition flush arm is deleted** (the D6 sentence
  "in-claim flush keeps the LEASED triggering-token exclusion exactly as
  today" is superseded): count/condition triggers fire from the intake step
  of the same drain — one extra BLOCKED→consumed transition per trigger
  member, ~ms latency at N=1 — and every flush runs out-of-claim through
  `complete_barrier`. `complete_barrier`'s `leased_exclusion_token_id` arm is
  deleted. Intake-fired TERMINAL coalesce merges emit their COALESCED output
  as a fresh PENDING_SINK row in the same completion (the historical in-claim
  ride left the merged output memory-only between consumption and the claim
  disposition).

### §E.3 — snapshot-scoped exhaustiveness (the one invariant relaxation)

D6's whole-universe exhaustiveness ("every BLOCKED row under the barrier must
be consumed or handed off") narrows to the per-firing-group intake snapshot:
`complete_barrier` takes `intake_snapshot_token_ids` — exactly the token_ids
the leader durably adopted into THIS firing group — and the exhaustiveness
universe becomes `durable ∩ snapshot`. Durable BLOCKED rows OUTSIDE the
snapshot are late arrivals: they legitimately stay BLOCKED, are recorded as
`late_arrival_token_ids` in every emission event context, and join the next
batch at the next intake. The snapshot is also a defence surface:
consumed/handed-off tokens outside it, snapshot tokens the journal does not
hold, and snapshot tokens in a different row group all raise Tier-1.
`None` preserves the pre-§E.2 durable-universe semantics for direct
repository callers.

### §E.3a — late-arrival journal release

A coalesce branch whose group already completed is released by the intake in
the SAME drain iteration via `mark_blocked_barrier_terminal` with a
`late_arrival` release context (`{"late_arrival": true, "reason": ...,
"released_by": ..., "scope_row_id": ...}` merged into the per-row release
event), so a COMPLETED finalize is never wedged by a straggler row.

### §E.5 — durable branch-loss hand-off

`coalesce_branch_losses` is an append-only ledger: every lossy disposition of
a fork-lineage branch records its loss in the SAME transaction as the
disposition (`mark_failed` / `mark_pending_sink` / `mark_terminal` /
`complete_barrier` all take loss specs). The leader replays unadopted losses
(`adopted_epoch IS NULL`) at intake — journal-first, mark before replay — and
the takeover restore seeds `lost_branches` from the FULL ledger (the D3
checkpoint scalar is retained as a cross-check only; the ledger wins).

### Supersession map

- D6 closing in-claim sentence ("In-claim flush keeps the LEASED
  triggering-token exclusion exactly as today") — superseded by §E.2
  (this amendment); annotated inline at the site.
- D7 memory-first exclusion (first scope-exclusion bullet) — dissolved;
  polarity flipped; annotated inline at the site.
- D3 lost-branch scalars — demoted to cross-check; `coalesce_branch_losses`
  is the restore truth (§E.5).
- Residual (tracked, unchanged): the `batches`-COMPLETED → `complete_barrier`
  ordering window inside `execute_flush` (elspeth-3977d8ab60) and the first
  member's DRAFT `batches` row created outside the adoption transaction
  (accepted residue, documented on `adopt_blocked_barrier_item`).
