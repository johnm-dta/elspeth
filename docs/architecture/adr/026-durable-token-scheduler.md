# ADR-026: Durable Token Scheduler

**Date:** 2026-05-23
**Status:** Accepted with stated preconditions (see *RC6
Preconditions*; the separate deployment-shape ADR required by
Precondition #9 is now authored as ADR-030 — proposed at slice
0 of its landing plan, → Accepted at slice 5)
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
ingestion and **multi-worker concurrent token execution**. Multi-source
ingestion here means **sequential multi-source ingest**: source
iteration remains one source at a time within a run. YAML declaration
order is the determinism anchor for cross-source `ingest_sequence`
assignment. The scheduler's concurrency model is worker-token
concurrency, **not concurrent source iteration**. Multi-worker is not a
future possibility the scheduler primitive permits; it is a stated RC6
deliverable that the scheduler exists to make sound:

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
  `src/elspeth/core/landscape/scheduler_repository.py` (2163 LOC).
  Persistence boundary; takes a SQLAlchemy `Engine`; every method is
  CAS-aware.
- **Worker loop:** `_drain_scheduler_claims` in `engine/processor.py`
  around lines 2820-3015. Per CLAUDE.md tier discipline, this is L2
  engine code consuming an L1 repository.
- **Contracts:** `TokenWorkItem` (frozen dataclass) and
  `TokenWorkStatus` (StrEnum) at `src/elspeth/contracts/scheduler.py`.

### Lifecycle

```
READY → LEASED → (BLOCKED | PENDING_SINK) → TERMINAL | FAILED
```

- `READY`: enqueued, awaiting claim. Delayed retry is not a distinct
  status: a `READY` row with a future `available_at` is simply held
  off until that time. (A `WAITING` status existed in earlier
  revisions of this ADR; the lane was dead code and was deleted in
  the F4 dead-lanes cleanup, commit `80d8baff2` — it does not exist
  in `contracts/scheduler.py`.)
- `LEASED`: a worker holds the row for `lease_seconds`; CAS-gated by
  `lease_owner`.
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
  (`scheduler_repository.py:2116-2119`).
- **`ingest_sequence`** — global ordering primitive across all sources
  (NEW on this branch; replaces `row_index` as the resume sort key).
  The orchestrator assigns it monotonically while iterating sources in
  YAML declaration order. Token construction is type-enforced:
  `TokenManager.create_initial_token()` and
  `TokenManager.create_quarantine_token()` require
  `source_row_index` and `ingest_sequence` as keyword-only `int`
  parameters, so missing identity is a caller type error rather than a
  runtime defaulting path.
- **`step_index`** — secondary ordering for tie-breaking within a
  single token's lifetime.
- **Claim order:** `ORDER BY ingest_sequence, step_index, created_at,
  work_item_id` (see `claim_ready`, `scheduler_repository.py:724-733`,
  and the identical order in `claim_pending_sink`). The fourth key is
  a stable last-resort tiebreaker for cross-source same-tick
  collisions where the first three keys are not jointly
  disambiguating (filigree elspeth-6cb89db535, G3
  determinism-reviewer M1). This is the determinism-of-claim
  contract: no two work items can ever tie.

### CAS discipline (post-fix)

Every state transition that consumes a lease takes an
`expected_lease_owner: str` keyword argument and rejects the
transition if the row's `lease_owner` does not match. `mark_terminal`,
`mark_pending_sink`, `mark_blocked`, and `mark_failed`
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
   (`processor.py` drain loop around line 2893): in-memory state
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
   (`scheduler_repository.py:979-1008`). The downstream effect is
   that `(token_id, node_id, attempt)` audit identity is fresh on
   retry; the prior attempt's node_state rows remain in the audit
   trail under their original attempt, not overwritten. The
   institutional-memory comment is in
   `recover_expired_leases`'s docstring (`scheduler_repository.py:
   915-935`).

6. **`PENDING_SINK` recovers to `PENDING_SINK`, not to `READY`.** A
   sink-bound token whose transform work is durably recorded does
   *not* re-run the transform on lease expiry; it re-claims its
   pending-sink lease and retries the sink write. This is the G3
   fix, recorded below. The contract: `PENDING_SINK` means the
   audit-prior work is durable and only the terminal sink handoff
   is outstanding. Recovery preserves `attempt` and `work_item_id`
   in that case (`scheduler_repository.py:980-988`).

7. **Barrier reconciliation is offensive.** `mark_blocked_barrier_
   terminal` refuses to terminalize a barrier unless the supplied
   live `token_ids` exactly match the durable BLOCKED token set
   (`scheduler_repository.py:1729-1833`). Missing live tokens,
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

10. **The scheduler primitive is multi-worker-sound by
    construction; whether RC6 actually ships N>1 workers is a
    separate decision whose preconditions are enumerated below.**
    The load-bearing invariants (CAS on every state-changing
    transition, deterministic `work_item_id`, deterministic claim
    ordering, `caller_owner`-aware lease recovery) are written
    for the multi-worker case; N=1 is a degenerate execution of
    the same contract. The scheduler column shape is final.

    **Honest framing — scope history.** The original
    `feat/multi-source-token-scheduler` branch name does not
    mention multi-worker, and G27 (the primary multi-worker
    correctness blocker) was found during the branch review on
    2026-05-22, not during original implementation. Multi-worker
    capability was retrofitted onto a primitive designed for
    multi-source. This ADR records the post-review contract; it
    does not assert that multi-worker was the original target.
    What remains to deliver an RC6 with N>1 workers is split into
    two precondition lists below: *scheduler-correctness gates
    required at any N≥1* and *multi-worker-specific gates
    required only if N>1 ships*. The deployment-shape
    decision (worker process lifecycle) is itself a precondition,
    not an Open Question — see *Deployment-shape precondition*
    below.

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
  audit) is closed as of 2026-06: every scheduler transition writes
  an immutable `scheduler_events` row (schema epochs 16–17; see
  *Revision history*).
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
  today.** *Resolved 2026-06 (G29 / elspeth-2b608abbd3).* As
  originally accepted, a reviewer asking "when did this row's
  lease expire?" could read the final state of
  `token_work_items` but not reconstruct the timeline. The
  committed remediation — a `scheduler_events` table written
  on every transition — has since landed (schema epochs
  16–17): enqueue, claim, recovery, lease-loss, and
  terminalization transitions are immutable, exportable
  facts, including from/to `lease_expires_at` evidence for
  lease recovery and heartbeat loss.
- **`recover_expired_leases` requires a unique `caller_owner`
  per process.** Each `RowProcessor` generates a
  `row-processor:<run_id>:<uuid>` identity (docstring at
  `scheduler_repository.py:921-935`); operators cannot reuse
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
  UPDATE. G27 collapsed the pair into a single CAS UPDATE
  (closed, commit `f79332aa8`; see Precondition #4).

### RC6 Preconditions

The scheduler primitive's design is final. The items below
are the gates between the present `feat/multi-source-token-
scheduler` branch and RC6 publish-readiness. They are split
into two lists because conflating them inflates the apparent
cost of multi-worker and hides that several gates are
correctness requirements at *any* worker count.

#### A. Scheduler-correctness preconditions (required at any N≥1)

These gates apply regardless of whether RC6 ships N=1 or
N>1 workers. Several are correctness claims under SQLite
WAL even with a single worker (the connection pool may
serve more than one connection per process). A future
session that chooses to ship N=1 still owes every item in
this list.

1. **Lease ownership semantics (G1 / elspeth-941f1508f5,
   commit `3025168b2`). Done.** `recover_expired_leases`
   filters `lease_owner != caller_owner`. At N=1 this
   prevents a worker stealing its own in-flight lease back
   on the next drain iteration when an LLM call exceeds
   the lease window.

2. **PENDING_SINK drain isolation (G3 / elspeth-5c5e88b071,
   commit `3dcebe9ec`). Done.** Resume converges in one
   `_drain_scheduler_claims` invocation regardless of how
   many tokens crashed mid-sink. Required for crash
   recovery at any N.

3. **Per-source schema-contract resume (G2 /
   elspeth-01942858c3). Pending.** Lands with the ADR-025
   structural fix; required so resume reconstructs the
   same `schema_contract` regardless of which source's
   row is consulted first. Multi-source correctness at any
   worker count.

4. **CAS race fix on `claim_ready` and `claim_pending_sink`
   (G27 / elspeth-4678a5aa73). Done.** The
   SELECT-then-UPDATE pattern across separate
   statements/transactions is racy under SQLite WAL even
   at N=1 if the engine pool serves a second connection.
   The fix folds the two statements into a single CAS
   UPDATE (UPDATE … RETURNING `work_item_id` with a
   single-row predicate, or the two-statement form inside
   `BEGIN IMMEDIATE`). Genuinely sharper under N>1, but
   the *correctness* claim does not depend on N. Closed by
   commit `f79332aa8` (a lost claim race surfaces as "no
   work", never a stale write); the cross-process residual
   is closed by ADR-030's write-intent `BEGIN IMMEDIATE`
   discipline on every scheduler/coordination write path.

5. **PRAGMA discipline on scheduler-bearing connections
   (G28 / elspeth-8536552dcb). Required.** Probe-and-
   assert that every connection has `journal_mode=WAL`,
   the agreed `busy_timeout`, and `foreign_keys=ON` —
   crash if not (probe path elspeth-97f8509b35). At N=1
   uniformity is needed across connections within the
   process; at N>1 across processes too.

6. **Scheduler state transitions in Landscape audit (G29 /
   elspeth-2b608abbd3). Implemented (2026-06).** Audit-primacy
   gap at any N, now closed: the `scheduler_events` table
   (schema epochs 16–17) records every transition — enqueue,
   claim, recovery, lease-loss, terminalization — with from/to
   status, attempt, lease owner, and `lease_expires_at`
   evidence. The landed schema records worker identity as the
   opaque `lease_owner` / `caller_owner` strings
   (`row-processor:<run_id>:<uuid>`); ADR-030 (Precondition
   #9) settles the N>1 identity as `worker:<run_id>:<uuid>`
   owner strings — extending, not changing, those semantics.

7. **Runbook for lease recovery (G19 /
   elspeth-559bce3459). Authored for N=1; N>1 rewrite
   pending.** First-operator-incident artifact: stuck
   lease, over-aggressive expiry, crashed worker. Required
   at N=1 (every "did the worker crash or is the LLM slow?"
   question lands here). The runbook exists at
   `docs/runbooks/scheduler-lease-recovery.md`. The N>1
   incident surface is now fixed by ADR-030 (Precondition
   #9); the runbook's N>1 rewrite (including the
   kill-the-wedged-incumbent step) rides slice 6 of
   ADR-030's landing plan and remains the RC6 publish gate.

#### B. Multi-worker-specific preconditions (required only if N>1 ships)

These gates exist only because RC6 might ship N>1
workers. If a future session decides RC6 ships at N=1, the
items here become post-RC6 follow-ups rather than RC6
gates. Listing them honestly distinguishes scheduler
correctness from multi-worker capability.

8. **Multi-worker isolation tests (G25b /
   elspeth-6116873e3b) and chaos coverage (G25h /
   elspeth-7bb7124e8f). Required if N>1.** CAS-loser,
   lease-expiry, claim-ordering, and barrier-
   terminalization paths exercised under N>1 with
   ChaosLLM / ChaosWeb / ChaosEngine wired against the
   scheduler. Without these the multi-worker claim is
   unproved. At N=1 the same surfaces are covered by the
   existing single-worker suite.

9. **Deployment-shape decision (worker process
   lifecycle). Required if N>1; satisfied by ADR-030
   (proposed).** As originally accepted, the *shape* of
   multi-worker deployment was not decided — co-tenants
   inside a single-binary
   `elspeth run`, separate `elspeth run --worker`
   invocations on the same host, or both. Each option
   implies a different spawn/supervise pattern, a
   different lease-owner-identity story, and a different
   shutdown protocol. This is *not* deferrable like a
   typical Open Question because four downstream items
   depend on it being settled:

    - **G29's `scheduler_events` schema** (Precondition
      #6 above) originally needed to know whether the
      audited identity is `(host, pid, uuid)` or
      `(worker_id, run_id)` or some other shape. The
      table has since landed carrying the opaque owner
      strings; an N>1 identity decision here extends
      those semantics rather than reshaping the table.
    - **G19's runbook** (Precondition #7) cannot describe
      "is this worker dead or just slow?" without
      knowing how a worker is spawned and supervised.
    - **Coordinated worker shutdown semantics.** On run
      cancellation or drain, do workers cooperate
      (stop claiming, finish in-flight leases, exit) or
      does each worker stop claiming and the run
      completes when leases expire? CAS supports either
      shape, but the runbook and the operator
      cancellation UX both pin on one choice.
    - **`pending_items` cache scope under N>1.** The
      in-memory `pending_items` list is per-`RowProcessor`.
      Under N>1 a worker may claim entries already
      leased by another worker between drains; the CAS
      rejects (no corruption) but the wasted-claim cost
      depends on whether the worker pool is co-tenant
      (same process, easy cross-worker advisory locks)
      or separate processes (no shared memory, only the
      durable row mediates).

    **Required artifact:** a separate ADR (or formal
    amendment to this one) that fixes the deployment
    shape, defines worker identity, decides shutdown
    semantics, and resolves the `pending_items`
    cross-worker question. Per the *Honest framing —
    scope history* note above, the deployment shape was
    not part of the original scheduler design; promoting
    it to a precondition rather than an Open Question
    is the honest record of what this ADR's RC6
    multi-worker claim actually depends on.

    **That ADR now exists: ADR-030 (One-Host WAL Pack,
    proposed 2026-06-11).** It fixes the deployment shape
    (one elected leader plus claim-only followers on a
    single host over the WAL-mode audit DB), worker
    identity (`worker:<run_id>:<uuid>` registered in a
    worker registry and doubling as the scheduler
    `lease_owner`), shutdown semantics, and the
    `pending_items` cross-worker question (per-process
    caches always backed by durable rows — unchanged
    invariant). This precondition is discharged when
    ADR-030 is accepted (slice 5 of its landing plan); no
    invariant in this ADR changes as a result. RC6 still
    cannot ship N>1 before that acceptance.

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

These tickets are *gates*, not follow-ups. Grouping mirrors
the §A / §B split in *RC6 Preconditions* above.

**§A — scheduler correctness (required at any N≥1):**

- **G27 / elspeth-4678a5aa73** — `claim_ready` /
  `claim_pending_sink` SELECT-then-UPDATE CAS-race fix.
  Correctness claim under WAL regardless of worker count;
  sharper at N>1 but the gate is not multi-worker-specific.
  **Closed** (commit `f79332aa8`; cross-process residual
  closed by ADR-030's write-intent `BEGIN IMMEDIATE`
  discipline — see Precondition #4).
- **G28 / elspeth-8536552dcb** — PRAGMA discipline on
  scheduler-bearing connections (probe-and-assert
  elspeth-97f8509b35, test coverage elspeth-addd3dc41f,
  type-enforce elspeth-34d83daedc). Uniformity required
  within-process at N=1, cross-process at N>1.
- **G29 / elspeth-2b608abbd3** — scheduler state transitions
  into Landscape audit trail. **Implemented (2026-06)** as the
  `scheduler_events` table, schema epochs 16–17; see
  Precondition #6.
- **G19 / elspeth-559bce3459** — runbook for lease recovery,
  authored at `docs/runbooks/scheduler-lease-recovery.md`.
  Required at N=1; incident surface broadens at N>1 — the
  N>1 rewrite rides slice 6 of ADR-030's landing plan. (Also
  listed under *Documentation / governance* — same ticket.)

**§B — multi-worker-specific (required only if N>1 ships):**

- **G25b / elspeth-6116873e3b** — multi-worker isolation
  tests (CAS-loser, lease-expiry, claim-ordering, barrier
  terminalization under N>1).
- **G25h / elspeth-7bb7124e8f** — chaos fixtures (ChaosLLM,
  ChaosWeb, ChaosEngine) wired against the scheduler under N>1.
- **Deployment-shape ADR (Precondition #9) — authored as
  ADR-030 (elspeth-1396d3f790).** Required artifact before
  RC6 ships N>1. Defines worker process lifecycle, worker
  identity, shutdown semantics, and `pending_items`
  cross-worker behaviour. Proposed at slice 0 of its
  landing plan; → Accepted at slice 5.

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
- **G19 / elspeth-559bce3459** — `TokenWorkItem` lease
  recovery runbook, authored at
  `docs/runbooks/scheduler-lease-recovery.md`. The intended
  audience is the first operator hit by a lease-expiry
  incident; the N>1 rewrite (slice 6 of ADR-030's landing
  plan) is the RC6 publish gate.
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

This section lists genuinely deferrable decisions —
choices that can be made after RC6 publishes without
invalidating any of the ADR's correctness arguments. The
load-bearing items previously listed here (worker process
lifecycle, coordinated worker shutdown semantics,
telemetry per worker_id, `pending_items` cache scope
under N>1) were retrofitted-as-Open-Questions and have
been promoted to *RC6 Preconditions §B item 9 —
Deployment-shape decision* above. See *Revision history*
for the move.

- **Scheduler-event audit table schema shape.** *Resolved
  2026-06:* the dedicated `scheduler_events` table shape
  shipped (schema epochs 16–17) with worker identity as
  opaque owner strings. The residue is settled by ADR-030
  (Precondition #9): `worker:<run_id>:<uuid>` identities
  extend the owner-string semantics.
- **Still-active prior-worker lease semantics — comment
  audit.** The G3 fix's `processor.py` comment documents
  a "stranded prior worker lease" case framed for
  single-worker. CAS already handles the multi-worker
  case correctly; the question is comment hygiene, not
  correctness. Re-read alongside G27.
- **Retention of terminalized scheduler rows.** Today
  `TERMINAL` and `FAILED` rows remain forever (with
  scrubbed payloads). A purge cadence aligned with the
  existing `elspeth purge --retention-days N` CLI
  command is a follow-up; ticket not yet filed.
- **Coalesce cursor representation in `token_work_items`.**
  The branch carries `coalesce_node_id` and `coalesce_name`
  on every row; whether the coalesce *state* (which fork
  paths have arrived) belongs on the work-item row or in
  a separate barrier table is open. Current pattern uses
  BLOCKED rows + `mark_blocked_barrier_terminal`; this is
  serviceable and not under review.
- **Postgres backend.** If the project ever needs a non-
  embedded backend, the same scheduler schema, CAS
  discipline, and lease semantics port directly to
  Postgres. The decision to remain SQLite-only at the
  v1 layer (memory `project_phase9_sqlite_only`) is
  preserved here.

## Revision history

- **2026-05-23** — ADR accepted. Initial structure listed
  "Worker process lifecycle", "Coordinated worker shutdown
  semantics", "Telemetry per worker_id", and "`pending_items`
  cache scope under N>1" under *Open questions / future
  work*, while *Decision* point (10) framed multi-worker as
  "an RC6 deliverable" and the *RC6 Preconditions* list was
  presented as eight ordered, independent gates.
- **2026-05-23** — *Related Decisions* updated to resolve the
  ADR-001 amendment contradiction (commit `9c3bfe85d1`,
  elspeth-d678a718fd). Source-iteration axis preserved by
  ADR-025; worker-execution axis amended by this ADR; ADR-001
  *Amendments* section is canonical.
- **2026-05-24** — Restructure following sme-review
  elspeth-3997769b6b. Three changes recorded together to
  preserve the audit trail of why each entry moved:
    - *Decision* point (10) softened from "multi-worker is an
      RC6 deliverable" to "the scheduler primitive is
      multi-worker-sound by construction; whether RC6 actually
      ships N>1 is a separate decision." A *Honest framing —
      scope history* paragraph was added recording that
      multi-worker was retrofitted onto a branch named
      `feat/multi-source-token-scheduler` and that G27 was
      found during the 2026-05-22 branch review, not during
      original implementation. The original framing presented
      retrofitted scope as settled deliverable; the new
      framing records the actual scope history.
    - *RC6 Preconditions* split into §A (scheduler-correctness
      preconditions required at any N≥1) and §B
      (multi-worker-specific preconditions required only if
      N>1 ships). G1, G2, G3, G27, G28, G29, G19 moved to §A;
      G25b and G25h remained as multi-worker-specific in §B.
      Rationale: G27 is admitted as a correctness claim under
      WAL even at N=1; G28's PRAGMA discipline applies
      within-process; G29's audit-primacy gap is a Tier-1
      requirement at any N; G19 covers single-worker
      incidents too. Listing all eight as "multi-worker
      preconditions" inflated apparent cost and conflated
      correctness with capability.
    - *Open questions* entry "Worker process lifecycle"
      promoted to a new *RC6 Preconditions §B* item (#9 —
      *Deployment-shape decision*) with explicit dependency
      arrows to Preconditions #6 (G29 schema) and #7 (G19
      runbook). "Coordinated worker shutdown semantics",
      "Telemetry per worker_id" (renamed in-place to "worker
      identity column shape"), and "`pending_items` cache
      scope under N>1" were absorbed into the same item as
      sub-bullets. Rationale: each was framed as deferrable
      ("policy is open", "decide alongside G27") but four
      RC6-precondition tickets (G19, G29, G27 follow-ups)
      cannot be discharged without a settled answer. Calling
      a precondition an Open Question hides risk in the
      wrong section.
- A reader sweeping the post-revision text should find: no
  load-bearing decisions hidden under *Open questions*; a
  Status line that honestly states the precondition surface;
  and a *Decision* §10 that records the post-review framing
  rather than asserting retrofitted scope as original intent.
- **2026-06-10** — Reconciliation against the merged RC6.0 line
  (post release/0.5.5 merge and orchestrator decomposition),
  following the Phase 1 verification verdict
  (`notes/rc6-phase1-verdict-2026-06-10.md`). Three changes:
    - *Claim order* updated from the three-key
      `ingest_sequence, step_index, created_at` to the four-key
      order ending in the `work_item_id` last-resort tiebreaker
      (filigree elspeth-6cb89db535) — the fix had landed in
      `claim_ready` / `claim_pending_sink` /
      `recover_expired_leases` but the ADR text was one
      tiebreaker stale.
    - *G29 marked implemented:* the `scheduler_events` table
      shipped (schema epochs 16–17). Precondition #6, the
      *Negative Consequences* entry, the §A ticket entry, and
      the *Open questions* schema-shape entry updated from
      "gap/required" to "implemented", recording that the landed
      worker identity is the opaque `lease_owner`/`caller_owner`
      string and that a future N>1 identity decision
      (Precondition #9, still open) extends rather than reshapes
      it.
    - *Code citations refreshed* to the post-decomposition tree:
      `scheduler_repository.py` spans, the `processor.py` drain
      loop, the PRAGMA listener, and the resume entry point
      (now `ResumeCoordinator.reconstruct_resume_state`,
      `orchestrator/resume.py`).
- **2026-06-12** — Stale-text reconciliation against ADR-030
  (slice 0 of the One-Host WAL Pack landing plan,
  elspeth-1396d3f790). Four changes:
    - *WAITING / `mark_waiting` removed from prose.* The
      lifecycle diagram, state list, and CAS-discipline list
      named a `WAITING` status and a `mark_waiting` verb that
      no longer exist: the lane was dead code deleted in the
      F4 dead-lanes cleanup (commit `80d8baff2`) and is absent
      from `contracts/scheduler.py`. Delayed retry is a
      `READY` row with a future `available_at`, not a status.
    - *G27 (Precondition #4) marked Done.* The single-CAS
      claim fix landed as commit `f79332aa8` (a lost claim
      race surfaces as "no work"); the cross-process residual
      is closed by ADR-030's write-intent `BEGIN IMMEDIATE`
      discipline. The "Required" markers here and in the §A
      ticket list were stale.
    - *G19 runbook pointer added.* The lease-recovery runbook
      exists at `docs/runbooks/scheduler-lease-recovery.md`
      (authored for N=1); "authoring blocked on Precondition
      #9" was stale. The N>1 rewrite rides slice 6 of
      ADR-030's landing plan and remains the RC6 publish gate.
    - *Precondition #9 marked satisfied by ADR-030* (proposed
      at slice 0, → Accepted at slice 5): deployment shape,
      worker identity, shutdown semantics, and the
      `pending_items` cross-worker question are fixed there.
      Status line, §B item 9, the §B ticket entry ("ticket not
      yet filed" was stale), and the *Open questions* residue
      updated; *Related Decisions* gained the ADR-030 entry.

## Related Decisions

- **ADR-001** (plugin-level concurrency) — **amended by
  this ADR along the worker-execution axis**; ADR-001's
  *Amendments* section records the amendment inline as of
  2026-05-23. RC5.2 ran one worker per process; RC6 adds
  concurrent token execution across multiple workers, sound
  via the scheduler row being authoritative for lease
  ownership and claim ordering. ADR-001's determinism
  contract carried the implicit "one worker per run"
  assumption; the multi-worker contract is "claim order is
  `ingest_sequence, step_index, created_at, work_item_id`
  regardless of which worker wins the CAS." The orthogonal source-iteration
  axis of ADR-001 (orchestrator pulls from one source at a
  time within a run) is not amended here and is preserved
  by ADR-025. That preserved model is sequential multi-source
  ingest: YAML declaration order is the determinism anchor,
  and this ADR is not concurrent source iteration. Any future
  concurrent source-iteration design would require its own ADR.
- **ADR-010** (declaration-trust framework) — preserved. The
  scheduler row carries no declaration-trust state; resume
  precondition (runtime VAL manifest match) is enforced at
  `orchestrator/resume.py:521-530` before any scheduler
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
- **ADR-030** (multi-worker deployment shape — One-Host WAL
  Pack) — satisfies Precondition #9: deployment shape,
  worker identity (`worker:<run_id>:<uuid>`, extending the
  opaque `lease_owner` scheme this ADR requires), shutdown
  semantics, and the `pending_items` cross-worker question.
  Its write-intent `BEGIN IMMEDIATE` discipline also closes
  the cross-process half of G27 (Precondition #4). Proposed
  at slice 0 of its landing plan; this ADR's N>1 claim
  becomes shippable when ADR-030 is accepted (slice 5).

## References

### Code

- `src/elspeth/contracts/scheduler.py` — `TokenWorkItem`,
  `TokenWorkStatus`.
- `src/elspeth/core/landscape/scheduler_repository.py` —
  `TokenSchedulerRepository` (full method index in
  *Decision* citations above; key spans: `claim_ready`
  708-822, `claim_pending_sink` 824-912,
  `recover_expired_leases` 914-1031,
  `mark_blocked_barrier_terminal` 1729-1833, `_transition`
  2024-2113, `_work_item_id` 2116-2119).
- `src/elspeth/core/landscape/schema.py` —
  `token_work_items_table`, `scheduler_events_table`
  (G29 surface, schema epochs 16–17).
- `src/elspeth/core/landscape/database.py:438-455` —
  PRAGMA `connect` listener (G28 surface).
- `src/elspeth/engine/processor.py` ~2820-3015 —
  `_drain_scheduler_claims`, the stranded-`pending_items`
  SCREAM invariant, `MAX_WORK_QUEUE_ITERATIONS` bound.
- `src/elspeth/engine/orchestrator/resume.py:475` —
  `ResumeCoordinator.reconstruct_resume_state`
  (scheduler-resume entry point; relocated from the
  pre-decomposition `orchestrator/core.py`
  `_reconstruct_resume_state` and preserved after the
  ADR-025 structural fix).
- `src/elspeth/core/dag/graph.py:283-358` — graph
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
