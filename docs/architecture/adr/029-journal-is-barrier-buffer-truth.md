# ADR-029: Scheduler Journal Is the Single Source of Barrier-Buffer Truth

**Date:** 2026-06-11
**Status:** Accepted
**Deciders:** John Morrissey, Claude Fable 5
**Tags:** scheduler, checkpoint, resume, barrier, aggregation, coalesce,
          durability, f1, multi-source-token-scheduler

## Context

The fork/coalesce architecture assessment
(`notes/fork-coalesce-architecture-assessment-2026-06-10.md`) identified a
structural durability gap in how barrier-buffered tokens survive crash and
resume. ELSPETH's token execution model has three durability layers:

1. **The scheduler journal** (`token_work_items`): every schedulable
   continuation has a row before a worker touches it. A buffered token is
   already written to the journal as a `BLOCKED` row at buffer time — the
   `mark_blocked` transition is called by the drain loop immediately after the
   in-memory accept.
2. **The checkpoint blob** (`checkpoints.aggregation_state_json` /
   `coalesce_state_json`): periodically serialized executor in-memory state.
   Contains membership lists, timing offsets, trigger latches, and
   pagination counters for every live barrier.
3. **The Landscape audit trail** (`token_outcomes`, `node_states`, `rows`,
   `tokens`): the immutable, per-mutation legal record. Aggregation batch
   membership (`BUFFERED` token_outcomes) and coalesce node_states are written
   per-accept, before the BLOCKED transition.

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
where nothing reads it.) On restore:

- Coalesce per-branch `arrival_time` ← row's `barrier_blocked_at`;
  `first_arrival` ← min over branches; `elapsed_age_seconds` ← now −
  `first_arrival`. The blob's `arrival_offset_seconds` / `elapsed_age_seconds`
  (offset reconstruction, stale by last-checkpoint-age) are deleted with
  *better* fidelity, not worse.
- Aggregation `first_accept_time` ← min `barrier_blocked_at` of the node's
  BLOCKED rows (the first accepted row of a batch always blocks; a count-1
  trigger fires in-claim and has no timing to restore).
  `TriggerEvaluator.restore_from_checkpoint(batch_count,
  elapsed_age_seconds, count_fire_offset, condition_fire_offset)` keeps its
  signature — only the source of elapsed changes.

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
- `completed_flush_count` ← `COUNT(batches WHERE status='completed' AND
  aggregation node = ?)` — the only COMPLETED-status `complete_batch` call is
  the flush-success path (`executors/aggregation.py:491`) where the counter
  increments (`:501`); failed flushes complete FAILED (`:516`/`:532`).
  Derivation is *fresher* than any stored scalar (no lost-increment crash
  window between flush success and the next checkpoint).

**Stored** (the truly underivable scalars), in one new
`checkpoints.barrier_scalars_json` Text column replacing the two blob columns:
per aggregation node `{count_fire_offset, condition_fire_offset}` (the two
trigger latches); per coalesce pending key `{lost_branches}`. That is the
whole inventory. A new small contracts module `contracts/barrier_scalars.py`
replaces both blob families. `format_version` 4→5.

*Staleness audit:* stored scalars are stale by last-checkpoint-age — same as
the blob today, no regression. A missing count latch is healed by the existing
conservative re-latch when `batch_count >= config.count`
(`engine/triggers.py:300-304`). A lost condition latch delays firing to the
next accept/EOF flush (same membership, possibly different `which_triggered`
attribution — the resumed run records what actually happened). `lost_branches`
lost in the window degrades `require_all`/`best_effort` evaluation to the
timeout path, identical to today's blob loss window. Counters (`accepted_count_total`,
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
  resume point, satisfying the `resume_attempt_offset > 0 ⟹
  resume_checkpoint_id is not None` invariant.

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
   `scheduler_repository.py`).
2. Consumed → `TERMINAL` (payload scrub).
3. Sink-bound emissions → `PENDING_SINK` (transition for buffered passthrough
   tokens; *insert* on the `node_id`-NULL terminal lane for merged/aggregate
   outputs — the `uq_token_work_items_terminal_identity` partial unique index
   is that lane).
4. Continuation emissions → `READY` inserts.

`mark_blocked_barrier_terminal` and `mark_blocked_barrier_pending_sink_many`
become internal arms of `complete_barrier`. This closes the out-of-claim flush
window: a timeout/EOF flush output is journal-durable (`PENDING_SINK`) the
moment its inputs are consumed. A crash before sink write recovers via the
existing `PENDING_SINK` re-drive. In-claim flush keeps the `LEASED`
triggering-token exclusion exactly as today (the claim's own result rides the
normal drain arms).

### Scope exclusions (D7)

The following are explicitly out of scope for this change:

- **The memory-first window** between in-memory buffer/accept and the drain's
  `mark_blocked`. A crash there leaves a `batch_members` row with no BLOCKED
  journal row; on resume the membership reconcile refuses with
  `AuditIntegrityError` (a re-drive would PK-violate `(batch_id, token_id)`).
  That refusal is pre-existing semantics, unchanged by F1.
- **Cross-process worker coordination** (option c, elspeth-1396d3f790) — F1
  unblocks it, does not start it.
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
   flip (elspeth-e1dd5e1303 decision forced).
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
   untouched).

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

### Plan

- `docs/superpowers/plans/2026-06-11-f1-durability-unification.md` —
  implementation plan whose Part I (I.1–I.4) this ADR records as an
  architecture decision. Part II contains the task-level implementation guide.

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
- `src/elspeth/engine/orchestrator/recovery.py` —
  `get_incomplete_tokens_by_row` attempt derivation (:722-728),
  buffered-row exclusion (:574-578, :668-679, :797-812).
- `src/elspeth/engine/executors/aggregation.py` —
  `restore_from_checkpoint` (:729-739, deleted under D5).
- `src/elspeth/engine/executors/coalesce_executor.py` —
  `_completed_keys` Landscape reconstruction (:328-385).
- `src/elspeth/engine/triggers.py` — conservative count re-latch
  (:300-304).
- `src/elspeth/engine/outcomes.py:385` — live `rows_buffered` counter
  (N−1 pre-F1; unified to N post-F1 per D3/I.4 item 3).
- `src/elspeth/contracts/barrier_scalars.py` — new module replacing both
  blob families under D3.

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
