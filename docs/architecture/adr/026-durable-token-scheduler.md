# ADR-026: Durable Token Scheduler

**Date:** 2026-05-23
**Status:** Accepted
**Deciders:** John Morrissey, Claude Opus
**Tags:** scheduler, checkpoint, resume, leases, cas, audit-integrity,
          embedded-database, rc6, multi-source-token-scheduler

## Context

Through RC5.2 the token execution model was *inline-with-source-
iteration*: the orchestrator pulled rows from a source one at a time,
processed each token through the DAG to a terminal state, recorded the
audit entries, and only then pulled the next row. In-flight token
state lived in process memory; crash recovery rebuilt the state by
replaying recorded `node_states` from the audit trail and joining
unprocessed rows back through the source plugin.

The RC5.2 model has three weaknesses that block the two RC6 outcomes
the `feat/multi-source-token-scheduler` branch ships: multi-source
ingestion and **multi-worker concurrent token execution**. Multi-worker
is not a future possibility the scheduler primitive permits; it is a
stated RC6 deliverable that the scheduler exists to make sound:

1. **No durable record of "what was the orchestrator about to do
   next?"** Recovery infers next-token selection from the absence of
   audit entries; if the crash happens mid-decision (e.g. between
   "claim row" and "begin token processing"), the inferred resume
   answer is ambiguous. With one source this is benign in practice —
   `row_index` orders inferentially. With N sources, cross-source
   ordering has no inferable answer.

2. **No identity for "this attempt is the second try at this token's
   sink write."** If a crash occurred between `sink.write()` succeeding
   and the audit row being committed, resume cannot tell whether to
   retry the sink (double-write risk) or skip it (lost-write risk).
   The single-source RC5.2 model accepts this small risk because most
   sinks are idempotent enough; the multi-source model with longer
   inter-row latencies makes the window wider and the answer more
   load-bearing.

3. **No primitive that survives multi-worker.** The RC5.2 model has no
   place to put a lease — token ownership is implicit in "the worker
   holding the row dict." RC6 lifts the single-worker ceiling: token
   execution is concurrent across multiple workers, and that only
   becomes a soundness property when ownership is durable. The
   scheduler row, not worker memory, is authoritative for lease
   identity, claim ordering, and resume.

The branch introduces a **durable token scheduler**: a SQLite-backed
table (`token_work_items`) that records every schedulable token
continuation as a row. The row is authoritative for resume — workers
rebuild a `WorkItem` from the row alone, without consulting in-memory
state. The row tracks lease ownership (CAS-gated), attempt count,
status (`READY → LEASED → … → TERMINAL | FAILED`), barrier and queue
keys, coalesce cursor, terminal-sink handoff context, and the row
payload (purged on terminalization).

The structural review (`notes/branch-review-multi-source-token-
scheduler-consolidation-2026-05-22.md`) judged the scheduler primitive
itself well-designed: *"The scheduler primitive itself is well-
designed. The upgrade around the scheduler is unfinished."* The dim1
engine audit found three P0 correctness bugs in the worker-loop / CAS
discipline that have since been fixed (see *Decision history* below).
What remained missing was the architectural-decision record. This ADR
discharges that obligation.

### What the scheduler primitive looks like

- **Table:** `token_work_items` (declared in
  `core/landscape/schema.py` as `token_work_items_table`).
- **Repository:** `TokenSchedulerRepository` at
  `src/elspeth/core/landscape/scheduler_repository.py` (1002 LOC).
  Persistence boundary; takes a SQLAlchemy `Engine`; every method is
  CAS-aware.
- **Worker loop:** `_drain_scheduler_claims` in `engine/processor.py`
  around lines 2540-2657. Per CLAUDE.md tier discipline, this is L2
  engine code consuming an L1 repository.
- **Contracts:** `TokenWorkItem` (frozen dataclass) and
  `TokenWorkStatus` (StrEnum) at `src/elspeth/contracts/scheduler.py`.

### Lifecycle

```
READY → LEASED → (WAITING | BLOCKED | PENDING_SINK) → TERMINAL | FAILED
```

- `READY`: enqueued, awaiting claim.
- `LEASED`: a worker holds the row for `lease_seconds`; CAS-gated by
  `lease_owner`.
- `WAITING`: held off until `available_at` (delayed retry).
- `BLOCKED`: durable barrier — a join/coalesce row that won't proceed
  until its sibling rows complete.
- `PENDING_SINK`: transform work durable; sink write is the remaining
  step. (This is a separately leasable state — recovered out of
  `LEASED` per attempt rather than rolled back to `READY`, so the
  sink-side work isn't re-executed under an incremented attempt.)
- `TERMINAL`: success path complete; `row_payload_json` is scrubbed.
- `FAILED`: terminal failure path; `row_payload_json` is scrubbed.

### Identity and ordering

- **`work_item_id = sha256(f"{run_id}:{token_id}:{node_id or
  '<terminal>'}:{attempt}")`** — deterministic. The same logical
  continuation always hashes to the same key, so `INSERT … ON
  CONFLICT DO NOTHING`-style idempotency works across crash boundaries.
  See `TokenSchedulerRepository._work_item_id`
  (`scheduler_repository.py:955-958`).
- **`ingest_sequence`** — global ordering primitive across all sources
  (NEW on this branch; replaces `row_index` as the resume sort key).
  Token construction is offensive: `_require_source_row_identity`
  crashes with `OrchestrationInvariantError` if either
  `source_row_index` or `ingest_sequence` is missing, with the
  institutional-memory message *"Do not fabricate source_row_index or
  ingest_sequence from row_index"* (G22 / elspeth-7f3ac1ac65 lifts
  that string into a doc + lint rule).
- **`step_index`** — secondary ordering for tie-breaking within a
  single token's lifetime.
- **Claim order:** `ORDER BY ingest_sequence, step_index, created_at`
  (see `claim_ready`, `scheduler_repository.py:433-437`, and the
  identical order in `claim_pending_sink`). This is the
  determinism-of-claim contract.

### CAS discipline (post-fix)

Every state transition that consumes a lease takes an
`expected_lease_owner: str` keyword argument and rejects the
transition if the row's `lease_owner` does not match. `mark_terminal`,
`mark_pending_sink`, `mark_blocked`, `mark_waiting`, and `mark_failed`
all enforce this. `recover_expired_leases` filters by `lease_owner !=
caller_owner` so a worker can never reap its own still-running lease.
The CAS pattern is what makes "two workers, same row" impossible to
silently corrupt — the loser of a race raises
`AuditIntegrityError`, not a stale write.

## Decision

ELSPETH owns a **durable token scheduler** as a first-class engine
primitive. The scheduler row is authoritative for resume; in-memory
work state is a cache of the durable row and must never diverge from
it.

### Concretely

1. **`TokenWorkItem` is the unit of work.** Every scheduled token
   continuation — initial entry into the pipeline, every downstream
   node hop, every barrier resolution, every pending-sink handoff —
   has a corresponding `token_work_items` row before the worker
   touches the in-memory `WorkItem`. The in-memory `pending_items`
   list is a cache of `READY` rows the worker plans to claim next; if
   `pending_items` is non-empty when the drain loop finds no
   `READY`/`PENDING_SINK` work, that is a *SCREAM* invariant violation
   (`processor.py` drain loop around line 2580): in-memory state
   must be backed by a durable row.

2. **`work_item_id` is deterministic.** The SHA-256 of
   `(run_id, token_id, node_id, attempt)` is the row's primary key.
   Two enqueues with identical inputs produce identical IDs; the
   insert path uses `_insert_work_item_idempotent` so a crash mid-
   enqueue is recoverable by re-running with the same inputs. The
   `attempt` component is what makes this deterministic-and-distinct
   across retries.

3. **`ingest_sequence` is the cross-source ordering primitive.** Not
   `row_index`. `row_index` (per-source position) and
   `source_row_index` (NEW; explicit per-source row index) remain on
   the row identity, but resume sort order and claim order use
   `ingest_sequence` exclusively. Without this, resume would replay
   rows in a different order than the original run, breaking the
   audit trail.

4. **Leases are CAS-gated and self-reap-safe.** Every state-changing
   call that consumes a lease takes `expected_lease_owner` and the
   underlying SQL UPDATE includes the matching predicate. The CAS
   loser raises `AuditIntegrityError` with a message naming the
   `work_item_id`, the expected owner, and the actual row count
   matched. `recover_expired_leases(run_id, now, caller_owner)`
   excludes `lease_owner = caller_owner` to prevent a worker from
   reaping its own still-running lease — the G1 fix, recorded below.

5. **Attempt rotation on lease expiry.** When
   `recover_expired_leases` reaps an expired lease that was *not* in
   `PENDING_SINK`, the row's `attempt` is incremented and
   `work_item_id` is recomputed
   (`scheduler_repository.py:584-608`). The downstream effect is
   that `(token_id, node_id, attempt)` audit identity is fresh on
   retry; the prior attempt's node_state rows remain in the audit
   trail under their original attempt, not overwritten. The
   institutional-memory comment is in
   `recover_expired_leases`'s docstring (`scheduler_repository.py:
   552-573`).

6. **`PENDING_SINK` recovers to `PENDING_SINK`, not to `READY`.** A
   sink-bound token whose transform work is durably recorded does
   *not* re-run the transform on lease expiry; it re-claims its
   pending-sink lease and retries the sink write. This is the G3
   fix, recorded below. The contract: `PENDING_SINK` means the
   audit-prior work is durable and only the terminal sink handoff
   is outstanding. Recovery preserves `attempt` and `work_item_id`
   in that case (`scheduler_repository.py:585-592`).

7. **Barrier reconciliation is offensive.** `mark_blocked_barrier_
   terminal` refuses to terminalize a barrier unless the supplied
   live `token_ids` exactly match the durable BLOCKED token set
   (`scheduler_repository.py:759-829`). Missing live tokens,
   duplicate live tokens, and mismatched rowcount are all
   `AuditIntegrityError` paths. The barrier is the only join/
   coalesce primitive that survives crash, so its terminalization
   has to prove the durable set and the live set agree.

8. **Drain loop drains all PENDING_SINK rows on resume.** The
   per-drain-call flag that limited resume to one pending-sink drain
   is deleted (G3 fix). Resume converges in one
   `_drain_scheduler_claims` invocation regardless of how many
   tokens crashed mid-sink.

9. **Row payload is scrubbed on terminalization.** Once a work item
   reaches `TERMINAL` or `FAILED`, its `row_payload_json` is
   replaced with `{"row_payload": "purged", "payload_hash": <sha256
   of anchor>}` (see `_scrubbed_row_payload_json`,
   `scheduler_repository.py:999-1002`). The audit trail retains the
   payload's identity via the hash; the bulk content moves to the
   payload store with its own retention policy. This is consistent
   with the payload-store separation principle in CLAUDE.md.

10. **Multi-worker is an RC6 deliverable.** RC6 ships concurrent
    token execution across N>1 workers as a stated outcome, on
    equal footing with multi-source ingestion. The scheduler
    primitive is the soundness layer that makes N>1 sound — its
    load-bearing invariants (CAS on every state-changing
    transition, deterministic `work_item_id`, deterministic claim
    ordering, `caller_owner`-aware lease recovery) are written
    for the multi-worker case; N=1 is a degenerate execution of
    the same contract, not the target. The scheduler column
    shape is final; what remains to deliver RC6 is enumerated
    under *RC6 Preconditions* below.

### Schema columns of note

`token_work_items` carries every datum needed to rebuild a
`WorkItem` without consulting in-memory state:

- Identity: `work_item_id` (PK), `run_id`, `token_id`, `row_id`,
  `node_id`, `attempt`.
- Ordering: `ingest_sequence`, `step_index`.
- Lifecycle: `status`, `available_at`, `created_at`, `updated_at`.
- Lease: `lease_owner`, `lease_expires_at`.
- Payload: `row_payload_json` (purged at terminal).
- Routing: `queue_key`, `barrier_key`, `on_success_sink`,
  `branch_name`.
- Pending sink: `pending_sink_name`, `pending_outcome`,
  `pending_path`, `pending_error_hash`, `pending_error_message`.
- Lineage: `fork_group_id`, `join_group_id`, `expand_group_id`,
  `coalesce_node_id`, `coalesce_name`.

### What this is NOT

- **Not a message broker.** The scheduler is an embedded SQLite
  table; ADR-024's single-maintainer governance and the embedded-
  database discipline (uv-installable, no Redis/RabbitMQ/Kafka
  runtime) are preserved. No external service is required to run
  ELSPETH.
- **Not a message broker for arbitrary cross-process queues.**
  The scheduler is the in-tree token-work queue for ELSPETH
  workers cooperating on the same run; it is not a general-purpose
  broker. Multi-worker is in scope for RC6 (preconditions are
  enumerated below), but the queue is bounded to the
  scheduler-row schema, the CAS discipline, and the embedded
  SQLite store — not to arbitrary external producers/consumers.
- **Not a replacement for the audit trail.** Scheduler rows are
  operational (lifecycle, lease, retry) and ephemeral (purged on
  terminalization). The audit trail (`node_states`, `rows`,
  `tokens`, `run_sources`) is the legal record and never purges.
  An auditor's question is answered against the audit trail, not
  the scheduler. G29 (scheduler state transitions absent from
  audit) is the next durability gap to close (see *Tickets*
  below).
- **Not an in-memory queue.** The `_drain_in_memory_work_queue`
  function on the present branch survives only for tests and is
  marked for deletion (G7 / elspeth-b680e81bce). The production
  code path is the scheduler; the in-memory path violates *never
  bypass production code paths in tests* (CLAUDE.md).

## Consequences

### Positive Consequences

- **Resume is fast and deterministic.** Reconstruction reads a
  bounded SELECT from `token_work_items` rather than replaying
  source iteration; ordering is `ingest_sequence`-driven and
  identical to the original run. The audit trail's `(token_id,
  node_id, attempt)` uniqueness constraint is preserved across
  crash boundaries.
- **Mid-token crashes have an answer.** Whether the worker
  crashed between source emission and transform start
  (`READY`), between transform start and sink handoff
  (`LEASED`), between sink handoff queue and sink write
  (`PENDING_SINK`), or between barrier resolution and barrier
  terminal (`BLOCKED`), there is exactly one row in
  `token_work_items` describing the durable state — no
  ambiguity, no inference.
- **Lease ownership is provable.** The CAS discipline turns a
  potential silent corruption (two workers writing
  conflicting outcomes) into a loud failure
  (`AuditIntegrityError` on the loser of the race), which is
  the project's preferred failure mode.
- **Multi-worker is sound on this scheduler shape.** The
  lifecycle, CAS semantics, and audit-identity contract are
  written for N>1 and final at the column level; the gates
  between the present branch and RC6 multi-worker
  publish-readiness are the *RC6 Preconditions* list below
  (G27 WAL CAS-race fix, G28 PRAGMA uniformity, G29 audited
  transitions, G25b/G25h isolation and chaos coverage, G19
  runbook). No invariant in this ADR changes when worker
  count increases — the preconditions are completeness
  gates, not redesigns.
- **The scheduler is also a debugging surface.**
  `summarize_active_work`, `count_active_work`, and
  `active_row_ids` give the operator (and the MCP failure-
  context tool) a precise picture of "what is the run
  currently doing?" that the inline-with-source model
  couldn't.

### Negative Consequences

- **`token_work_items` rows are operational state in the
  Landscape database.** A deliberate compromise: single-file
  deployability and FK integrity against `runs`, `tokens`,
  `rows`, `nodes` without a cross-database join. Retention
  (purge of terminalized scheduler rows) is RC6 follow-up;
  today they remain forever (payloads purged, rows present).
- **Scheduler state transitions emit no Landscape audit rows
  today.** A reviewer asking "when did this row's lease
  expire?" can read the final state of `token_work_items`
  but cannot reconstruct the timeline. This is G29 /
  elspeth-2b608abbd3, recorded below as a P1 audit-primacy
  gap. The intended remediation is a `scheduler_events`
  table (or equivalent) that the scheduler writes on every
  transition; the ADR commits to that as the answer, with
  schema design downstream.
- **`recover_expired_leases` requires a unique `caller_owner`
  per process.** Each `RowProcessor` generates a
  `row-processor:<run_id>:<uuid>` identity (docstring at
  `scheduler_repository.py:559-572`); operators cannot reuse
  owner strings across processes. G27.x (elspeth-34d83daedc)
  makes the contract type-mechanical.
- **PRAGMA discipline is load-bearing — more so under
  multi-worker.** The scheduler shares `db.engine` with the
  audit recorder; the WAL, busy_timeout, and foreign_keys
  PRAGMAs at `core/landscape/database.py:320-329` apply
  uniformly within a process. Under multi-worker, *every*
  scheduler-bearing process must apply the same PRAGMA set on
  every connection — a worker without `journal_mode=WAL` or
  with a different `busy_timeout` is a stealth-divergence
  risk. G28 is therefore an RC6 precondition (see below).
- **Schema epoch advance is required when the scheduler
  schema changes.** The current `token_work_items` schema is
  authoritative; future column additions or FK changes must
  advance the schema epoch (elspeth-af13a34ccd) so that
  operator-side *delete the old DB* discipline applies.

### Neutral Consequences

- The `step_index` field is per-token, not per-run. Two
  tokens' `step_index=0` rows are distinct rows because
  `(token_id, node_id, attempt)` differs.
- The `MAX_WORK_QUEUE_ITERATIONS` bound on the drain loop
  prevents pathological spin if a scheduler bug ever
  produces unreachable `READY` rows; the loop terminates
  with a `SCREAM` invariant violation rather than hanging.
- Two-phase claim (`SELECT … LIMIT 1` then `UPDATE … WHERE
  status = 'ready'`) is the SQLite-portable pattern, but
  under multi-worker the two-phase shape is a *correctness*
  concern, not just a trade-off: another worker can
  transition the row in the window between SELECT and
  UPDATE. G27 collapses the pair into a single CAS UPDATE.

### RC6 Preconditions

Multi-worker concurrent token execution is in scope for RC6.
The scheduler primitive's design is final; the items below
are the gates between the present `feat/multi-source-token-
scheduler` branch and RC6 publish-readiness, in the order a
reviewer should expect them resolved:

1. **Lease ownership semantics (G1 / elspeth-941f1508f5,
   commit `3025168b2`). Done.** `recover_expired_leases`
   filters `lease_owner != caller_owner` — makes
   `caller_owner` a true identity under N>1.

2. **PENDING_SINK drain isolation (G3 / elspeth-5c5e88b071,
   commit `3dcebe9ec`). Done.** Resume converges in one
   `_drain_scheduler_claims` invocation — a single
   surviving worker can take over after a fleet event.

3. **Per-source schema-contract resume (G2 /
   elspeth-01942858c3). Pending.** Lands with the ADR-025
   structural fix; required so any worker reading
   `run_sources` reconstructs the same schema_contract
   regardless of process or claim order.

4. **CAS race fix on `claim_ready` and `claim_pending_sink`
   (G27 / elspeth-4678a5aa73). Required for RC6.** The
   SELECT-then-UPDATE pattern across separate
   statements/transactions is racy under SQLite WAL — a
   correctness claim even at N=1 if the engine pool ever
   serves a second connection, and the primary load-bearing
   multi-worker gap. The fix folds the two statements into
   a single CAS UPDATE (UPDATE … RETURNING `work_item_id`
   with a single-row predicate, or the two-statement form
   inside `BEGIN IMMEDIATE`).

5. **PRAGMA discipline on scheduler-bearing connections
   (G28 / elspeth-8536552dcb). Required for RC6.** Probe-
   and-assert that every worker has `journal_mode=WAL`,
   the agreed `busy_timeout`, and `foreign_keys=ON` —
   crash if not (probe path elspeth-97f8509b35). Without
   uniformity, worker behaviour under contention diverges
   per worker.

6. **Scheduler state transitions in Landscape audit (G29 /
   elspeth-2b608abbd3). Required for RC6.** Multi-worker
   without audited lease-ownership transitions is
   forensically blind — the auditor's "which worker held
   this lease and when did it expire?" gets *current* row
   state, not the timeline. Audit primacy is broken in the
   multi-worker case without it. Remediation: a
   `scheduler_events` table (or equivalent) written on
   every transition.

7. **Multi-worker isolation tests (G25b /
   elspeth-6116873e3b) and chaos coverage (G25h /
   elspeth-7bb7124e8f). Required for RC6.** CAS-loser,
   lease-expiry, claim-ordering, and barrier-
   terminalization paths exercised under N>1 with ChaosLLM
   / ChaosWeb / ChaosEngine wired against the scheduler.
   Without these, the multi-worker claim is unproved.

8. **Runbook for lease recovery (G19 /
   elspeth-559bce3459). Required for RC6 publish.** First-
   operator-incident artifact: stuck lease, over-aggressive
   expiry, fleet event. Multi-worker amplifies the
   incident surface — "is this worker dead or just slow?"
   becomes a frequent question.

## Alternatives Considered

### Alternative 1: In-memory queue + on-disk checkpoint snapshot

**Description:** Keep token continuations in an in-process queue
(e.g. `collections.deque`); periodically snapshot the queue to
disk (along with aggregation/coalesce state). On crash, restore
the queue from the most recent snapshot. Identity and CAS are
unnecessary because there is only ever one worker reading the
in-memory queue.

**Rejected because:** Snapshot interval is the problem. Snapshot
frequently enough to bound data loss to a few seconds, and the
snapshot becomes the hot write path (every token's enqueue and
dequeue touch the disk). Snapshot less frequently and a crash
between snapshots loses scheduler intent — the orchestrator was
about to claim row X, the snapshot doesn't reflect it, recovery
re-claims row X *or* skips row X depending on whether the prior
node's audit row was committed in time. The fundamental issue is
that the queue is the state; making the queue's *snapshot* the
durable record is doing the durable-row work without the
discipline. The scheduler row primitive does the same work with
one UPDATE per state transition and a CAS predicate.

### Alternative 2: External broker (Redis Streams / RabbitMQ / Kafka)

**Description:** Replace the in-memory queue with a hosted
message broker. The broker provides queue semantics, lease
equivalents (visibility timeout), and multi-worker out of the
box. ELSPETH gains free horizontal scaling.

**Rejected because:** ADR-024 (delivery governance for single-
maintainer mode) constrains the operational surface to
mechanical, in-tree gates with uv-installable dependencies. A
broker is a separate runtime — operator must install, configure,
secure, back up, and monitor it; the embedded-database
discipline (one SQLite file is the system) is the present
deployability contract. The audit-trail integrity story also
becomes harder: the broker holds operational state, the SQLite
holds audit state, the FK between scheduler row and audit row
disappears (or moves to a cross-database integrity check the
broker doesn't offer). RC6's multi-worker requirement is served
by the in-tree scheduler with the CAS and PRAGMA preconditions
above. Further-out horizontal scaling (across machines, not
just processes on one host) is the case where a horizontally-
distributed SQLite pattern (per-run shards) or a Postgres
backend with the same scheduler schema becomes relevant — both
preserve the single-store audit-and-scheduler invariant.

### Alternative 3: Per-token audit row drives recovery directly

**Description:** Skip the dedicated scheduler table; rely on the
audit `node_states` table to record token progress and infer
"what's next" from "which `(token_id, node_id, attempt)`
combinations are absent." This is the RC5.2 model, generalized.

**Rejected because:** The inference is unsound at branch points.
A token that forks into two paths produces two distinct
`token_id` values; recovery has to know whether the fork already
emitted the second token (in which case both paths are in
flight) or only the first (in which case the second must be
created). The pre-RC5.2 single-token state machine could carry
this in process memory; the multi-source / concurrent model
cannot. The scheduler row makes the answer durable: the second
fork token has its own `work_item_id`, in `READY` (if claim
hasn't happened) or downstream (if it has).

Also: barrier states (BLOCKED) and pending-sink states have no
representation in `node_states` until they terminalize. The
scheduler row is the durable handle on "not-yet-done" state, the
audit trail is the durable handle on "what-happened" state, and
those are different questions.

## Decision history

The structural decision to ship a durable scheduler predates this
ADR; the implementation landed on `feat/multi-source-token-
scheduler` over 22 commits. Two P0 correctness bugs in the
scheduler discipline were discovered and fixed on 2026-05-22 and
2026-05-23 as part of the dim1 engine audit; this ADR records
the post-fix contract.

- **Commit `3025168b2`** — *fix(scheduler): worker no longer
  steals back its own in-flight lease.* G1 / elspeth-941f1508f5.
  Added `caller_owner` to `recover_expired_leases` plus the
  filter `lease_owner != caller_owner` (and `lease_owner IS
  NULL` write predicates). Pre-fix, a worker whose LLM call
  exceeded the 300s lease would have its own in-flight lease
  reset to READY by the next iteration of its own drain loop,
  causing a deterministic `AuditIntegrityError` on the
  subsequent `mark_terminal` / `mark_pending_sink` /
  `mark_blocked` write.

- **Commit `3dcebe9ec`** — *fix(scheduler): drain all pre-
  existing PENDING_SINK rows on resume.* G3 /
  elspeth-5c5e88b071. Removed the
  `created_pending_sink_this_drain` flag that limited resume
  to one pending-sink claim per `_drain_scheduler_claims`
  call — predated the lease-recovery model and made multi-
  token crash-mid-sink unrecoverable in a single resume
  invocation. Post-fix, recovery converges regardless of
  pending-sink crash count.

G2 / elspeth-01942858c3 (the arbitrary-source schema_contract
pick on multi-source resume) is the third P0 and is the subject
of ADR-025's *Decision* point (3) — the structural fix lands
with the dim4 test plan from elspeth-d5f0194fc8.

The recovery transition rule recorded in *Decision* point (6)
above — `LEASED → PENDING_SINK` on expiry for sink-pending work
rather than `LEASED → READY` — is the *post-G1+G3* contract.
Pre-fix code rolled all expired leases back to `READY`, which
would have re-executed the transform side of a sink-pending
token and broken audit identity.

## Tickets this ADR covers / unblocks

### Directly closes (after implementation lands)

- **G17 / elspeth-57d0031a14** — *No architectural doc explains
  the multi-source / scheduler design.* This ADR + ADR-025
  close it.

### P0 fixes already landed on this branch

- **G1 / elspeth-941f1508f5** — lease self-steal (commit
  `3025168b2`).
- **G3 / elspeth-5c5e88b071** — PENDING_SINK drain starvation
  (commit `3dcebe9ec`).

### RC6 preconditions covered by this ADR

These tickets are *gates*, not follow-ups: RC6 publish-readiness
depends on each. See *RC6 Preconditions* above for the ordered
rationale.

- **G27 / elspeth-4678a5aa73** — `claim_ready` /
  `claim_pending_sink` SELECT-then-UPDATE CAS-race fix. Multi-
  worker correctness blocker, and a correctness claim under
  WAL regardless of worker count.
- **G28 / elspeth-8536552dcb** — PRAGMA discipline on
  scheduler-bearing connections (probe-and-assert
  elspeth-97f8509b35, test coverage elspeth-addd3dc41f,
  type-enforce elspeth-34d83daedc). Under multi-worker,
  non-uniformity becomes a divergence risk.
- **G29 / elspeth-2b608abbd3** — scheduler state transitions
  into Landscape audit trail. Without this, multi-worker is
  forensically blind. The `scheduler_events` remediation
  referenced in *Negative Consequences*.
- **G25b / elspeth-6116873e3b** — multi-worker isolation
  tests (CAS-loser, lease-expiry, claim-ordering, barrier
  terminalization under N>1).
- **G25h / elspeth-7bb7124e8f** — chaos fixtures (ChaosLLM,
  ChaosWeb, ChaosEngine) wired against the scheduler.
- **G19 / elspeth-559bce3459** — runbook for lease recovery,
  authored against the multi-worker incident surface. (Also
  listed under *Documentation / governance* — same ticket.)

### Architecturally anchors (post-RC6 follow-up)

- **G25d / elspeth-0bae6d8a52** — `recover_expired_leases`
  multi-expiry test coverage (P2; not a correctness blocker).
- **G25c / elspeth-e8a1250782** — Hypothesis state machine
  still models OLD lifecycle; needs `READY → LEASED → …`.
- **elspeth-9c0a79ed26** — lease-retry attempt offset wired
  to wrong queue path (+ NameError).
- **elspeth-e7463a935b** — ready-claim ordering test-side
  proof of the determinism contract in *Identity and
  ordering*.
- **elspeth-af13a34ccd** — schema epoch advance required for
  scheduler row FK.
- **elspeth-1869c9ba64** — `ingest_sequence` set from
  counters AFTER quarantine, BEFORE check (ordering nuance
  in *Identity and ordering*).

### Documentation / governance

- **G18 / elspeth-06aecb78a0** — `docs/architecture/
  landscape.md` missing `token_work_items` schema.
- **G19 / elspeth-559bce3459** — no runbook covers
  `TokenWorkItem` lease recovery. The intended audience is
  the first operator hit by a lease-expiry incident;
  runbook is RC6 publish gate.
- **G22 / elspeth-7f3ac1ac65** — *"Do not fabricate
  source_row_index / ingest_sequence"* lives only in an
  exception string; lift to doc + lint rule (cross-cut
  with ADR-025).
- **G15 / elspeth-bc91898548** — `docs/release/
  guarantees.md §7.1 "single-threaded in RC-3"` contradicted
  by scheduler (cross-cut with ADR-025).

### Code health

- **G26 / elspeth-54e9c72f1b** — `processor.py` carries
  3620 LOC including the scheduler drain loop; extract
  `SchedulerDriver` (and downstream `MultiSourceCoordinator`,
  cross-cut with ADR-025).
- **G32 / elspeth-d869cc0113** — allowlist churn from
  multi-source code shifts; periodic
  `cicd-allowlist-audit` skill.

## Open questions / future work

- **Scheduler-event audit table.** G29's remediation is the
  next durability gap; the schema design (event table vs
  append-only column journal vs `node_states` extension) is
  intentionally not committed here. The decision is its own
  follow-up ADR-amend or a new ADR if the answer is
  structural.
- **Worker process lifecycle.** The *shape* of multi-worker
  deployment is not yet decided — co-tenants inside a
  single-binary `elspeth run`, separate `elspeth run
  --worker` invocations on the same host, or both. Each
  option implies a different spawn/supervise pattern and
  lease-owner-identity story. Likely a separate ADR — surface
  before the runbook (G19) lands.
- **Coordinated worker shutdown semantics.** On run
  cancellation or drain, do workers cooperate on graceful
  termination (stop claiming, finish in-flight leases,
  exit), or does each worker stop claiming and the run
  completes when leases expire? CAS supports either;
  policy is open.
- **Telemetry per worker_id.** Lease owner is
  `row-processor:<run_id>:<uuid>`. Is the uuid the identity
  Landscape should record for G29, or does multi-worker
  want a stable `worker_id` (e.g. a host-process-spawn-
  index) alongside? Stable IDs help operator dashboards;
  uuid is sufficient for correctness. Decide before G29's
  `scheduler_events` schema lands.
- **Still-active prior-worker lease semantics.** The G3
  fix's `processor.py` comment documents a "stranded prior
  worker lease" case. Under single-worker the strand is
  benign (prior worker is dead but not yet timed out);
  under multi-worker the prior worker may still be *alive*
  and the strand is correct — that distinction is
  load-bearing for the recovery contract. CAS already
  handles this; the open question is whether the comment
  needs updating for the multi-worker case. Re-read
  alongside G27.
- **`pending_items` cache scope under N>1.** The
  in-memory `pending_items` list (`processor.py` drain
  loop) is a per-`RowProcessor` cache of `READY` rows the
  *current* worker plans to claim next. Under multi-worker
  each worker carries its own `pending_items`; the durable
  `token_work_items` table is the only shared truth. The
  SCREAM invariant ("non-empty `pending_items` with no
  matching durable rows") is per-worker by construction.
  Open question: do we need a worker-local guard that
  rejects `pending_items` entries already leased by
  *another* worker between the worker's last drain and the
  current iteration (the CAS will reject, but the cache
  becomes a wasted-claim source), or is the wasted-claim
  cost acceptable? Decide alongside G27.
- **Retention of terminalized scheduler rows.** Today
  `TERMINAL` and `FAILED` rows remain forever (with
  scrubbed payloads). A purge cadence aligned with the
  existing `elspeth purge --retention-days N` CLI command
  is a follow-up; ticket not yet filed.
- **Coalesce cursor representation in `token_work_items`.**
  The branch carries `coalesce_node_id` and `coalesce_name`
  on every row; whether the coalesce *state* (which fork
  paths have arrived) belongs on the work-item row or in a
  separate barrier table is open. Current pattern uses
  BLOCKED rows + `mark_blocked_barrier_terminal`; this is
  serviceable and not under review.
- **Postgres backend.** If the project ever needs a non-
  embedded backend, the same scheduler schema, CAS
  discipline, and lease semantics port directly to
  Postgres. The decision to remain SQLite-only at the
  v1 layer (memory `project_phase9_sqlite_only`) is
  preserved here.

## Related Decisions

- **ADR-001** (plugin-level concurrency) — **amended by
  this ADR.** RC5.2 ran one worker per process; RC6 adds
  concurrent token execution across multiple workers, sound
  via the scheduler row being authoritative for lease
  ownership and claim ordering. ADR-001's determinism
  contract carried the implicit "one worker per run"
  assumption; the multi-worker contract is "claim order is
  `ingest_sequence, step_index, created_at` regardless of
  which worker wins the CAS." A formal ADR-001 amendment
  may follow once the deployment shape (see *Open
  questions*) is decided.
- **ADR-010** (declaration-trust framework) — preserved. The
  scheduler row carries no declaration-trust state; resume
  precondition (runtime VAL manifest match) is enforced at
  `orchestrator/core.py:3482-3491` before any scheduler
  rows are consulted.
- **ADR-019** (two-axis terminal model) — preserved. The
  durable terminal-axis distinction (`TERMINAL` vs
  `FAILED`) is preserved on the scheduler row alongside the
  audit row.
- **ADR-021** (sources and sinks uniformly boundary) —
  preserved. The scheduler's `PENDING_SINK` state holds the
  sink crossing distinct from the transform crossing; the
  Tier-3 boundary classification is unchanged.
- **ADR-023** (custom Python CI analyzer) — the
  `tier-model` rule under `elspeth-lints` will see
  fingerprint rotation as scheduler-related code lands
  (memory `feedback_ast_shift_fingerprint_rotation`); the
  enforcement contract is preserved.
- **ADR-024** (delivery governance for single-maintainer
  mode) — preserved and load-bearing. The embedded-database
  discipline, the no-external-broker discipline, and the
  mechanical-gate discipline are the rationale for
  Alternative 2 being rejected.
- **ADR-025** (multi-source ingestion) — companion. ADR-025
  records *what* the source surface looks like and *why*
  it's plural; ADR-026 records *how* tokens produced by
  that surface survive crash and resume.

## References

### Code

- `src/elspeth/contracts/scheduler.py` — `TokenWorkItem`,
  `TokenWorkStatus`.
- `src/elspeth/core/landscape/scheduler_repository.py` —
  `TokenSchedulerRepository` (full method index in
  *Decision* citations above; key spans: `claim_ready`
  416-483, `claim_pending_sink` 485-550,
  `recover_expired_leases` 552-609,
  `mark_blocked_barrier_terminal` 759-829, `_transition`
  909-952, `_work_item_id` 954-958).
- `src/elspeth/core/landscape/schema.py` —
  `token_work_items_table`.
- `src/elspeth/core/landscape/database.py:320-329` —
  PRAGMA `connect` listener (G28 surface).
- `src/elspeth/engine/processor.py` ~2540-2657 —
  `_drain_scheduler_claims`, the stranded-`pending_items`
  SCREAM invariant, `MAX_WORK_QUEUE_ITERATIONS` bound.
- `src/elspeth/engine/orchestrator/core.py:3427-3555` —
  `_reconstruct_resume_state` (scheduler-resume entry
  point; preserved after the ADR-025 structural fix).
- `src/elspeth/core/dag/graph.py:283-353` — graph
  validation including the QUEUE fan-in requirement.

### Commits

- `3025168b2` — G1 lease self-steal fix.
- `3dcebe9ec` — G3 PENDING_SINK drain starvation fix.

### Review notes

- `notes/branch-review-multi-source-token-scheduler-architecture-2026-05-22.md`
  — load-bearing primitives, the scheduler-primitive overview,
  the *post-fix* determinism story.
- `notes/branch-review-multi-source-token-scheduler-consolidation-2026-05-22.md`
  — the 32 canonical findings, tier-1 sequencing, the
  *"scheduler primitive itself is well-designed"* framing.
- `.worktrees/multi-source-token-scheduler/notes/multi-source-audit-dedup-map.md`
  — execution-detail dedup map.
- `notes/RC6-large-list.md` — canonical RC6 ticket
  enumeration.

### Project policy

- `CLAUDE.md` — *Auditability Standard*, *Three-Tier Trust
  Model* (Tier 1 audit-trail crash discipline that the
  `AuditIntegrityError` paths embody), *No Legacy Code
  Policy* (the `_drain_in_memory_work_queue` cleanup
  precondition).
- Memory `project_phase9_sqlite_only` — SQLite-only baseline
  (rationale for the embedded-database choice).
- Memory `project_db_migration_policy` — *delete the old DB*
  rather than migrate.
- Memory `project_multi_source_token_scheduler_rc6` — this
  branch targets RC6.

## Notes

The branch's three P0 correctness bugs (G1, G3, G2) demonstrate
that the scheduler primitive is *correct in design but easy to
get wrong in disciplinary detail*. The CAS discipline, the
`caller_owner` lease-recovery rule, the `PENDING_SINK` recovery
exemption, and the `expected_lease_owner` parameter on every
transition are non-negotiable invariants — future contributors
extending the scheduler must preserve them or extend this ADR.
The *RC6 Preconditions* list above is how the branch is brought
from "scheduler primitive shipped" to "RC6 multi-source +
multi-worker publish-ready."
