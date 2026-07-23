# Runbook: Scheduler Lease Recovery

Diagnose and recover from durable-scheduler lease and run-coordination
incidents in the ELSPETH token scheduler (ADR-026 durable scheduler; ADR-030
multi-worker deployment shape).

Current as of 2026-06-13. This runbook assumes familiarity with the
[Landscape schema](../architecture/landscape.md) — in particular the
`token_work_items` table and, for N>1 deployments, the `run_coordination`,
`run_workers`, and `run_coordination_events` coordination tables — and with
[Resume Failed Run](resume-failed-run.md).

This runbook covers both deployment shapes. A single-worker (N=1) run is the
**degenerate case of the same protocol**: it still creates a `run_coordination`
seat at epoch 1 and registers its origin process as the leader, so the
diagnostics below apply unchanged. Steps 1–8 are the single-worker procedure;
the multi-worker layer (topology, dead-leader takeover, the wedged-incumbent
SIGKILL step, follower lifecycle) is added in the sections after Step 8, and
the operator-guidance section at the end gives the sizing and deployment rules
for N>1.

---

## When To Use

- A worker process crashed mid-run and you suspect work items are stuck `leased`.
- The drain loop logs `Scheduler has N in-memory continuations ... but no READY work item could be claimed` — a Tier-1 SCREAM invariant.
- The drain loop raises `Work queue exceeded 100000 iterations. Possible infinite loop in pipeline.`
- A CAS write raises `AuditIntegrityError` referencing `expected_lease_owner` mismatch.
- A run is `running` but no progress visible in `node_states` for several minutes.
- The audit trail shows token `attempt` counts climbing without corresponding application-level retries — you suspect lease expiry churn.

For N>1 deployments, additionally:

- A run is `running` but its leader seat has expired — the leader process died, was SIGSTOPped, OS-wedged, or locked out of the DB (the previously-wedged "RUNNING with a dead leader" state, now recoverable).
- `elspeth resume` raises `WriteLockHeldError` ("the audit DB write lock is held by a live or frozen process") — a process is frozen inside an open transaction and is stalling the whole pack, including the takeover itself.
- `elspeth join` raises `JoinRefusedError` — admission was refused (run terminal, config-hash mismatch, no live leader, or a filesystem-writability preflight failure).
- A follower exits with `RunWorkerEvictedError` or `FollowerSeatDeadError`.
- The coordination ledger shows `fence_refusal` / `leadership_lost` events, or `heartbeat_degraded` events.

---

## Mental Model

`token_work_items` is the durable unit of work; the scheduler row is the
authoritative resume target, not in-memory state. Each row goes through a
finite lifecycle:

```
ready → leased → (blocked | pending_sink) → terminal | failed
```

A worker takes a row out of `ready` by calling `claim_ready`, which atomically
sets `status='leased'`, `lease_owner=<worker:run_id:uuid>`, and
`lease_expires_at = now + lease_seconds`. The lease is the worker's exclusive
claim on the row. Every state-changing call carries `expected_lease_owner`;
CAS-mismatch raises `AuditIntegrityError` rather than overwriting another
worker's progress.

> **Note on `available_at` (not a `waiting` state).** Delayed retries are
> driven by `available_at` on the work-item row, not by a separate `waiting`
> status. There is no `waiting`/`mark_waiting` state in the scheduler
> contract (`contracts/scheduler.py`) — earlier ADR-026 prose and earlier
> versions of this runbook named one in error (reconciled by ADR-030).

### Two-level liveness (the N>1 layer)

Multi-worker recovery turns on **two independent clocks** with distinct
meanings:

- **Item lease** (`token_work_items.lease_expires_at`) — "is this *claim*
  making progress?" Expiry means the lease may be reapable.
- **Run heartbeat** (`run_coordination.leader_heartbeat_expires_at` for the
  leader; `run_workers.heartbeat_expires_at` for each worker) — "is this
  *process* alive?" Each worker runs a dedicated heartbeat thread on its own
  DB connection.

A dead worker is one whose **run heartbeat** has expired. A worker stuck in a
long LLM call is *slow*, not dead: its item lease may expire while its run
heartbeat stays live. `recover_expired_leases` therefore reaps an expired item
lease only when **both signals are stale** — the lease has expired **and** the
owner's `run_workers` row is absent, `evicted`, `departed`, or `active` but
heartbeat-expired past a grace window — **or** the lease has been expired
longer than the hard stall budget (`item_stall_budget_seconds`, default
`600s` = 2 × the 300s item lease). The stall-budget arm covers the
heartbeat-thread-alive-but-drain-wedged pathology and writes a `worker_stalled`
coordination event in the same transaction as the attempt rotation, so the
rotation is reconstructable from the ledger.

In N>1, `recover_expired_leases` is **leader-only and epoch-fenced**
(`scheduler_repository.py:1279`): only the elected leader reaps, which gives
every attempt rotation a single attributable author and avoids an
O(workers²) sweep storm.

#### Long plugin calls and the external replay boundary

The hard stall budget is an intentional takeover boundary, even when the old
worker's process heartbeat remains live. If a synchronous transform call is
still running when the leader rotates its item, a replacement worker may run
the same logical transform attempt. When the old call later returns or raises,
the drain performs a post-call item heartbeat: the missing/reassigned attempt
records `lease_lost` and the old worker abandons both its result and scheduler
disposition. Exactly one claimant can therefore commit the internal
`mark_*` transition.

That fence does not transact with work the plugin already performed outside
the Landscape database. `IO_WRITE` and `EXTERNAL_CALL` transforms are
**at-least-once across item takeover**: one rotation can produce one call from
the old claimant and one replay by its replacement, and repeated stalls can
produce further attempts. Such plugins must use a provider/application
idempotency key, reconcile an already-applied effect, or move publication to a
supported durable sink-effect boundary. `worker_stalled` plus `lease_lost`
explains the internal winner/loser sequence; neither event proves that an
external effect happened only once.

**Two design notes that bite operators in incidents:**

- `attempt` is bumped on lease expiry (ADR-026 §Decision 5). A row at
  `attempt = 4` does **not** mean four application-level retries — it
  could be one application call plus three lease-expiry rotations. The
  `node_states` table records each application attempt separately under
  its own `attempt`; cross-reference both tables before concluding the
  pipeline is unstable.
- `lease_owner IS NULL` while `status='leased'` is a wedge state the
  recovery sweep tolerates (so it can recover), but the schema's
  `ck_token_work_items_lease_owner_required_when_leased` CHECK
  constraint refuses new writes that create the wedge
  (`filigree elspeth-9990c81e14`, schema epoch 14). If you see a wedge
  row, it predates the constraint or arrived through a path that bypassed
  it — treat it as Tier-1 evidence corruption.

---

## Inputs

- Path to the Landscape SQLite database (typically `runs/audit.db`).
- The `run_id` of the affected run.
- The lease window the run was using (`scheduler.lease_seconds` in settings;
  default is in `engine/processor.py`).
- For N>1: the settings file the leader was started with (required by
  `elspeth join` to compute the matching `config_hash`).

For the queries below, replace `<RUN_ID>` and `<DB>` with concrete values.

---

## Step 1: Confirm the run is actually stuck

A "stuck" pipeline is sometimes just a slow LLM call inside the lease window.
Confirm the run has no live progress before invoking recovery.

```bash
sqlite3 <DB> "
  SELECT MAX(created_at) AS last_node_state
  FROM node_states
  WHERE run_id = '<RUN_ID>';
"

sqlite3 <DB> "
  SELECT MAX(updated_at) AS last_work_item_update
  FROM token_work_items
  WHERE run_id = '<RUN_ID>';
"
```

If `last_node_state` and `last_work_item_update` are within the last lease
window (e.g. 5 minutes for the default), the run is processing — do not
intervene.

> **N>1:** progress can also be confirmed at the process level via the
> coordination heartbeats (see the *Inspect coordination state* section
> below). A live leader seat plus live worker heartbeats means the pack is
> alive even if a single item lease has lapsed.

---

## Step 2: Inspect work-item state distribution

```bash
sqlite3 <DB> "
  SELECT status, COUNT(*) AS n
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
  GROUP BY status
  ORDER BY n DESC;
"
```

Expected distributions:

| State | Interpretation |
|-------|----------------|
| Mostly `terminal` / `failed` | Run is finishing or finished. Check `runs.status`. |
| Mix of `ready`, `leased`, `terminal` | Run is in flight. Check Step 3 for lease ages. |
| Many `blocked` | Barrier-join coalesce siblings outstanding (or aggregation holds). Check Step 7. |
| `leased` rows with old `updated_at` | Worker probably died. Continue to Step 3. |
| Rows with a future `available_at` | Delayed retries scheduled; not stuck. |

---

## Step 3: Identify expired leases

```bash
sqlite3 <DB> "
  SELECT
    work_item_id,
    token_id,
    node_id,
    attempt,
    lease_owner,
    lease_expires_at,
    updated_at
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
    AND status = 'leased'
    AND lease_expires_at < datetime('now')
  ORDER BY lease_expires_at;
"
```

Any row returned is a candidate for `recover_expired_leases`. Note the
`lease_owner` values — if all expired leases belong to a single
`worker:<run_id>:<uuid>`, that worker is dead. If they span multiple owners,
multiple workers died (or are partitioned from the database). **At N>1,
multiple `lease_owner` values are normal** (each follower has its own
identity); the question is whether each owner's *run heartbeat* is live — see
the next section.

**Wedge-state check** — `lease_owner IS NULL` while `status='leased'` is a
Tier-1 invariant violation:

```bash
sqlite3 <DB> "
  SELECT work_item_id, status, lease_owner, updated_at
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
    AND status = 'leased'
    AND (lease_owner IS NULL OR lease_owner = '');
"
```

Any row returned should be reported to the data-integrity team — the schema
check constraint is supposed to make this unreachable on writes. Capture the
row and `updated_at` before doing anything else; recovery rewrites these
fields.

---

## Step 4: Recover stuck leases by restarting the run

The supported recovery path is `elspeth resume`. The resume command starts
a fresh leader `RowProcessor` whose drain loop calls `recover_expired_leases`
on every iteration; expired leases owned by dead/evicted/departed workers (or
past the stall budget) are reaped automatically.

```bash
elspeth resume <RUN_ID>          # Dry run — check resume readiness
elspeth resume <RUN_ID> --execute
```

Do **not** manually `UPDATE token_work_items SET status='ready'` to clear
stuck leases. That bypasses the `attempt` rotation, which means the next
worker's `node_states` rows collide with the dead worker's prior entries
under the same `(token_id, node_id, attempt)` audit identity. The audit
trail loses the ability to distinguish the dead attempt from the recovered
one — Tier-1 evidence corruption.

If for some reason the resume command cannot be used (e.g. the host that owns
the audit DB is partitioned from the operator's environment), the only
defensible manual recovery is to wait for the lease window to elapse and let
the leader's next drain iteration sweep the row. The `attempt` rotation
happens inside `recover_expired_leases`; there is no manual equivalent that
preserves it.

---

## Step 5: SCREAM — "in-memory continuations but no READY work item"

The drain loop raises:

```
OrchestrationInvariantError: Scheduler has <N> in-memory continuations
for run <RUN_ID> but no READY work item could be claimed
```

This is a Tier-1 SCREAM (`engine/processor.py` drain loop). It means the
worker's in-memory `pending_items` cache references work the scheduler row
does not know about. Causes:

- A code path enqueued a `WorkItem` in memory without writing the durable
  row first (engine bug — file P0).
- The database was rewound or restored to a snapshot that predates the
  in-memory state.
- The in-memory state was carried across a database swap.

**Do not** restart the worker and hope. Capture:

```bash
sqlite3 <DB> "
  SELECT *
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
  ORDER BY ingest_sequence, step_index;
" > /tmp/scheduler-state-<RUN_ID>.txt
```

…the worker's stderr (the SCREAM message lists the `len(pending_items)` and
the offending run_id), and any recent `mark_*` errors in the audit DB:

```bash
sqlite3 <DB> "
  SELECT * FROM validation_errors WHERE run_id = '<RUN_ID>' ORDER BY created_at DESC LIMIT 20;
"
```

Escalate per your team's Tier-1 incident process. The run is unsafe to
resume until the divergence is explained.

---

## Step 6: Drain loop hit `MAX_WORK_QUEUE_ITERATIONS`

```
RuntimeError: Work queue exceeded 100000 iterations. Possible infinite loop in pipeline.
```

This is `engine/processor.py` (`MAX_WORK_QUEUE_ITERATIONS = 100_000`).
It indicates either:

- A pipeline that produces more work than it consumes (a transform that
  enqueues N+1 child items per input — almost always a plugin bug), or
- A barrier that fails to resolve and keeps the worker re-checking the
  same `blocked` rows.

Diagnosis queries:

```bash
sqlite3 <DB> "
  SELECT status, COUNT(*) AS n
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
    AND status = 'blocked'
  GROUP BY status;
"
```

If `blocked` rows dominate, the barrier-join coalesce is not reconciling —
capture the `barrier_key` and the sibling token set:

```bash
sqlite3 <DB> "
  SELECT work_item_id, token_id, node_id, barrier_key, attempt
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
    AND status = 'blocked'
  ORDER BY barrier_key, token_id;
"
```

Do not restart the run blindly. The bounded loop exists precisely to
prevent silent runaway resource consumption; treat the limit as evidence,
not as a transient to retry.

---

## Step 7: Distinguishing lease expiry from application retries

`token_work_items.attempt` advances on every lease-expiry recovery
(except `pending_sink`, which preserves it). `node_states.attempt`
advances on every application-level retry. They are not the same axis.

To see what the row's application history actually looked like:

```bash
sqlite3 <DB> "
  SELECT token_id, node_id, attempt, status, started_at, completed_at, error_json
  FROM node_states
  WHERE run_id = '<RUN_ID>'
    AND token_id = '<TOKEN_ID>'
  ORDER BY started_at;
"
```

If `node_states.attempt` is `[0, 1, 2]` but `token_work_items.attempt` is
`5`, three lease-expiries happened with no corresponding application call —
the worker died inside its lease window three times. Investigate worker
health (memory limits, oom-killer, deploy-time restart loops).

> **N>1:** an item rotated under the stall-budget or registry-dead arm writes
> a `worker_stalled` event naming the reaped owner — cross-reference the
> coordination ledger (next section) to attribute the rotation to a specific
> worker rather than guessing.

---

## Step 8: Escalation criteria

Escalate to the platform / data-integrity team if any of:

- Step 3 returned wedge rows (`lease_owner IS NULL` with `status='leased'`).
- Step 5 fired (SCREAM invariant).
- Step 6 fired and the cause is not an obvious plugin bug.
- `recover_expired_leases` returns 0 for a run that visibly has expired
  leases owned by a dead worker (suggests the recovery sweep itself is
  failing to write — CAS predicate or concurrent transaction issue).
- Any `AuditIntegrityError` raised by a CAS-gated state transition
  (`mark_terminal`, `mark_pending_sink`, `mark_blocked`, `mark_failed`) —
  this is Tier-1 evidence integrity and must be triaged before the next
  resume.

For N>1 deployments, additionally escalate if:

- `RunLeadershipLostError` / `fence_refusal` events appear in a *storm* (many
  in a short window) rather than as the expected one-or-two at a takeover —
  this suggests two processes both believe they lead, or a stale generation
  token is being driven repeatedly.
- `WriteLockHeldError` persists *after* a SIGKILL of the named PID — the lock
  did not release, which means either the wrong PID was killed, the holder is
  on a different host (an unsupported multi-host deployment), or a fresh
  process re-acquired the lock. Re-read the roster before retrying.
- More than one **live, non-terminal `BUFFERED`** `token_outcomes` row exists
  for a single token — restore raises a loud Tier-1 diagnostic on this. It
  indicates a double-adoption that the barrier-adoption CAS is supposed to
  make structurally impossible (ADR-030 §C.4 row 6a). Diagnostic:

  ```bash
  sqlite3 <DB> "
    SELECT token_id, COUNT(*) AS n
    FROM token_outcomes
    WHERE run_id = '<RUN_ID>'
      AND state = 'BUFFERED'
      AND completed = 0
    GROUP BY token_id
    HAVING n > 1;
  "
  ```

Capture for escalation: the run_id, the full output of Steps 2–4, the
worker's stderr from the time of the incident, the `run_coordination_events`
ledger for the run, and the audit DB at incident time (do not let it be
overwritten by a fresh resume).

---

## Multi-worker (N>1) recovery

The sections below are the N>1 layer. They build on Steps 1–8: confirm the
run is stuck and inspect the work-item distribution first, then use the
coordination tables to decide *which* recovery the situation calls for.

### Topology and identity

An ELSPETH multi-worker run is **one host, one WAL SQLite audit DB, and a
pack of cooperating OS processes** (ADR-030 D1/D2): exactly one
**epoch-fenced leader** plus N−1 **claim-only followers**. The leader owns
everything that lives in a single process's memory today — source ingest,
checkpoint sequencing, the barrier plane (executors, trigger evaluation,
`complete_barrier`), the end-of-input flush, the sink-write phase, PENDING_SINK
terminalization, the recovery sweep, and run finalization. Followers do
exactly the expensive parallelizable thing — claim READY items and run their
transforms (the LLM calls) — and nothing else. Followers never ingest, write
checkpoints, evaluate barriers, write sinks, or finalize.

Each worker's identity is `worker:{run_id}:{uuid4().hex}`, minted at
registration and used directly as the scheduler `lease_owner` string
(`follower.py`). Identities are **single-use**: a `run_workers` row that
reaches `departed` or `evicted` never returns to `active`; a returning
process mints a fresh uuid and re-admits via `elspeth join` or `elspeth
resume`. This makes eviction a permanent fence with no re-admission race.

`elspeth run` and `elspeth resume` produce the **leader**; the new `elspeth
join <run_id>` produces a **follower**. There is no supervisor daemon —
workers are operator-spawned.

### Inspect coordination state

Three tables carry the run's coordination state (epoch 21 schema):

**The leader seat (`run_coordination`)** — exactly one row per run. The seat
is **live** iff `leader_heartbeat_expires_at > now`:

```bash
sqlite3 <DB> "
  SELECT
    leader_worker_id,
    leader_epoch,
    leader_heartbeat_expires_at,
    (leader_heartbeat_expires_at > datetime('now')) AS seat_live,
    updated_at
  FROM run_coordination
  WHERE run_id = '<RUN_ID>';
"
```

`seat_live = 0` (or a NULL `leader_worker_id`) means the leader is dead,
SIGSTOPped, OS-wedged, or persistently locked out of the DB — **not merely
inside a long LLM call** (the heartbeat rides a dedicated thread that survives
slow plugin work). `leader_epoch` is the monotonic fencing token; it bumps on
every leadership acquisition.

**The worker roster (`run_workers`)** — one row per worker that ever joined:

```bash
sqlite3 <DB> "
  SELECT
    worker_id, role, status,
    heartbeat_expires_at,
    (heartbeat_expires_at > datetime('now')) AS worker_live,
    pid, hostname,
    evicted_by_worker_id, evicted_at, departed_at
  FROM run_workers
  WHERE run_id = '<RUN_ID>'
  ORDER BY role, registered_at;
"
```

`pid` and `hostname` are **forensics** — they are the columns the
wedged-incumbent SIGKILL step reads (see below). `status` is `active`,
`departed` (clean exit), or `evicted` (CAS-evicted, e.g. a deposed leader or a
swept-out dead follower).

**The coordination ledger (`run_coordination_events`)** — an append-only
history. **Order it by `seq`, not `recorded_at`**: `recorded_at` is a
process-stamped wall-clock that can invert commit order under `busy_timeout`
stalls; `seq` is the authoritative replay order.

```bash
sqlite3 <DB> "
  SELECT seq, event_type, worker_id, leader_epoch, recorded_at, context_json
  FROM run_coordination_events
  WHERE run_id = '<RUN_ID>'
  ORDER BY seq;
"
```

Event types: `worker_register`, `worker_depart`, `worker_evict`,
`worker_stalled`, `leader_acquire`, `leader_release`, `leadership_lost`,
`fence_refusal`, `heartbeat_degraded`, `finalize`. A clean takeover looks like
`leader_acquire` + `worker_evict` (of the deposed leader) at a bumped
`leader_epoch`. Successful heartbeats do **not** write events; only the
diagnostic ones (`heartbeat_degraded`, `worker_stalled`) and the state
transitions do.

### Dead-leader takeover (`elspeth resume`)

**Detection:** `run_coordination.leader_heartbeat_expires_at < now` (the
`seat_live = 0` query above), while `runs.status` is still `running`. This is
the previously-wedged state: before ADR-030 the resume entry guard refused
`RUNNING` unconditionally and the run was unrecoverable without manual
surgery. It is **now resumable** — the expired seat is the proof of lost
custody (`checkpoint/recovery.py:137-160`).

**Procedure:**

```bash
elspeth resume <RUN_ID>          # Dry run — confirms RUNNING + expired seat is admissible
elspeth resume <RUN_ID> --execute
```

The first durable act of `resume()` is the seat-acquisition CAS
(`acquire_run_leadership`, `resume.py:614-623`), in one `BEGIN IMMEDIATE`
transaction:

- the seat CAS bumps `leader_epoch` and installs the new leader;
- the **deposed incumbent is evicted by identity, unconditionally** — whatever
  identity sat in `leader_worker_id` is CAS-evicted (`evicted_by_worker_id` set,
  `worker_evict` event), regardless of its own worker-row heartbeat. The
  expired seat is sufficient proof; there is no heartbeat predicate that could
  miss it (ADR-030 D3, §B.4);
- **followers are NOT bulk-evicted** at takeover. The new leader's housekeeping
  sweep evicts dead followers individually, under a grace period and a
  no-unexpired-leases precondition (see *Follower death* below).

After a successful takeover the ledger shows `leader_acquire` + `worker_evict`
at the incremented epoch. The new leader then runs the existing resume
reconstruction (barrier restore from BLOCKED rows, checkpoint scalars, pending-sink
recovery, source-position recovery) and continues the run.

**Followers during a leader death** idle and then exit; they **never
auto-promote** (ADR-030 D3). A follower that observes a dead seat finishes or
abandons its current item, takes no new claims, departs cleanly, and exits
with `FollowerSeatDeadError` (CLI exit code 2 — "no live leader; use `elspeth
resume`", `follower.py:320-322`, `cli.py:2591-2612`). Takeover requires the
full resume reconstruction, which only `elspeth resume` performs.

**Racing resumes:** if two operators run `elspeth resume` against the same run
at once, exactly one wins the seat CAS. The loser gets
`NonResumableRunError("run leadership is held by …")` with **zero mutation** —
the rowcount-0 path commits nothing, closing the documented resume TOCTOU
(`resume.py`).

### Kill the wedged incumbent (`WriteLockHeldError` → SIGKILL → resume)

This is the central new step for N>1 and the reason a dead-leader run can
need operator intervention beyond a plain resume.

**The problem.** SQLite file locks live exactly as long as the holding
process — nothing times them out. A process frozen *inside* a `BEGIN
IMMEDIATE` transaction (SIGSTOPped, VM-paused, GC-wedged, debugger-attached)
holds the WAL write lock indefinitely and stalls **the entire pack** — every
peer heartbeat, every claim, **and the takeover CAS itself** (ADR-030 D5; the
design's §J risk 7). `elspeth resume` cannot simply take over, because its
first durable act is a write that the frozen holder is blocking.

**The signal.** When the takeover CAS times out on the write lock,
`acquire_run_leadership` distinguishes a busy timeout from a clean CAS loss and
raises **`WriteLockHeldError`**, not `NonResumableRunError`
(`run_coordination_repository.py`, error at `contracts/errors.py:880-908`). Its
message reads:

```
The audit DB write lock is held by a live or frozen process; the coordination
write for run '<RUN_ID>' timed out at BEGIN IMMEDIATE. Registered workers:
worker_id='…' role=… status=… pid=… hostname='…'; … If a worker is frozen
inside a transaction, SIGKILL it (SQLite locks release on process death) and retry.
```

The error carries the `run_workers` roster (pid/hostname/role/status) so you
know exactly what to inspect.

**Procedure:**

1. Read the roster from the error message, or directly:

   ```bash
   sqlite3 <DB> "
     SELECT worker_id, role, status, pid, hostname
     FROM run_workers
     WHERE run_id = '<RUN_ID>' AND status = 'active'
     ORDER BY role;
   "
   ```

2. **Confirm before you kill.** Verify the `hostname` is the local host and
   the named PID is genuinely wedged (e.g. process state `T`/`D`, no CPU
   progress, blocked on the DB) — not merely slow inside a long LLM call. A
   slow-but-live leader keeps its seat via the heartbeat thread and does not
   produce `WriteLockHeldError`.

3. SIGKILL the frozen holder. SQLite releases its locks on process death:

   ```bash
   kill -9 <PID>
   ```

4. Re-run the takeover — the write lock is now free:

   ```bash
   elspeth resume <RUN_ID> --execute
   ```

**SIGKILL goes before resume, always** (ADR-030 D5, Negative Consequences):
the lock must be released before the takeover CAS can even begin. If
`WriteLockHeldError` persists after the SIGKILL, escalate per Step 8 — the
wrong PID was killed, the holder is on another host (an unsupported deployment),
or a fresh process grabbed the lock.

### Follower death

**Detection:** a `run_workers` row with `role='follower'` and
`heartbeat_expires_at < now` (`worker_live = 0` in the roster query;
`checkpoint/recovery.py` two-level liveness). A follower death triggers **no
run-level ceremony** — no resume, no takeover.

**What the leader does automatically.** The leader's housekeeping sweep evicts
the dead follower individually, in one CAS transaction, under the §C.2
predicate: the target must hold **no unexpired item leases** and must have been
heartbeat-stale past a grace window. Only after eviction does the sweep reap
the follower's expired item leases (with attempt rotation), so eviction
precedes reap on this path. A follower that heartbeated in the meantime, or
still holds a live lease, is a benign skip (rowcount 0).

**Operator action: usually none.** The pack self-heals. You may optionally
re-attach capacity with `elspeth join` (a *new* worker under a fresh identity —
single-use identity means the dead row never returns).

**Distinguish the two follower exits.** When a follower process itself exits,
read its exit code and the ledger:

- **Finalize-departure** — the run reached a terminal state and the follower
  departed cleanly: `run_workers.status='departed'`, `worker_depart` event,
  CLI **exit 0** (`follower.py:281-307`). Expected; no action.
- **Eviction while still RUNNING** — the follower was CAS-evicted mid-run:
  `RunWorkerEvictedError`, CLI **exit 3** (`cli.py:2572-2590`). The follower
  abandoned in-flight work without emitting (the lease-lost discipline). Its
  identity is spent; re-admit a fresh one if you want the capacity back.

(For completeness: `FollowerSeatDeadError` is **exit 2** — leader seat died
mid-drain, run incomplete, resume required; SIGINT is **exit 3**; a Tier-1
audit-integrity / framework error is **exit 4**.)

### Attach a follower (`elspeth join`)

```bash
elspeth join <RUN_ID> --settings <SETTINGS_FILE>
```

The `--settings` file is **required**: the joiner must reproduce the same
`config_hash` as the leader (`cli.py:2287, 2330-2337`). A joiner with a
different pipeline would disagree about the graph and barrier keys, so a
mismatch is refused.

Admission is one atomic `BEGIN IMMEDIATE` transaction (ADR-030 D3, §B.1) that
checks **all three** preconditions or refuses:

1. `runs.status = 'RUNNING'` (else `JoinRefusedError` — terminal runs use
   `elspeth resume`);
2. the joiner's resolved settings hash equals `runs.config_hash` (else
   refused);
3. the leader seat is **live** (else "no live leader — use `elspeth resume` to
   take the seat"). A follower must never be the first process on an abandoned
   run — barrier and ingest state need a leader. **Join and takeover are
   disjoint verbs.**

A **filesystem-writability preflight runs first** (before the registry is
touched): the follower verifies write access to the DB file, its directory,
and any `-wal`/`-shm` sidecars. A failure raises `JoinRefusedError` naming the
path and the required permission (`contracts/errors.py:830-842`,
`cli.py:2422-2438`). See *Operator guidance §2.4* for the shared-group/umask
setup this depends on.

**Exit codes:** 0 = clean departure (run completed); 1 = admission refusal or
setup error (`JoinRefusedError`, bad settings/DB); 2 = `FollowerSeatDeadError`
(seat died mid-drain — resume required); 3 = `RunWorkerEvictedError` or SIGINT;
4 = Tier-1 audit-integrity / framework error (`cli.py:2314-2315, 2438, 2590,
2612, 2628, 2665`).

---

## Operator Guidance (multi-worker deployment)

These are the operator-facing requirements and sizing rules for N>1, matching
ADR-030 D6. They are **runbook-stated requirements, not probed invariants** —
the engine assumes them and will fail (sometimes loudly, sometimes as
`SQLITE_READONLY_CANTLOCK` mid-run) if they are not met.

### Lease sizing vs the heartbeat window

The run-liveness window must be sized against worst-case **write-lock
occupancy**, not against the longest LLM call (that is the whole point of the
separate heartbeat thread — a slow LLM call does not make a worker look dead).
The rule (`contracts/coordination.py:41-51`):

```
run_liveness_window  >=  4 × (run_heartbeat_seconds + busy_timeout)
                     =   4 × (15 s + 5 s)  =  80 s   (defaults)
```

- `run_heartbeat_seconds = 15` and `run_liveness_window = 80` are the defaults
  (both configurable).
- `busy_timeout` is the probe-pinned SQLite setting, **5000 ms**
  (`database.py:77, 86, 560`) — a polling retry with **no fairness guarantee**.
- A **dead leader is detected in ~80–95 s** (the window plus the grace), which
  drives the takeover decision above.

The **item lease** defaults are unchanged — **300 s lease / 60 s item
heartbeat**. Critically, under two-level liveness `scheduler_lease_seconds`
**stops being a correctness bound and becomes a hygiene bound**: a live worker
stuck in a long LLM call keeps its item via the registry even after the item
lease lapses, because its next `heartbeat_lease` CAS revives it. The hard
**stall budget** is `item_stall_budget_seconds = 600 s` (2 × the item lease;
`contracts/coordination.py:60`, `scheduler_repository.py:1287`) — past it, even
a registry-live worker's item is rotated with a `worker_stalled` event.

### Worker count: recommended 2–8 (the write-lock convoy)

**Recommended N is 2–8** (ADR-030 Supported list). The audit DB is a
single-writer WAL file; concurrent writers serialize behind `busy_timeout`
**polling retry, with no fairness** (ADR-030 D5). The workload is
LLM-dominated (seconds per row, a handful of sub-millisecond writes per item),
so even ten workers produce only tens of writes per second — two orders of
magnitude under WAL's single-writer capacity. The constraint is **lock
occupancy and the absence of fairness**, not raw throughput.

The slice-1 contention measurements (design §H, §I):

- N=2: worst-case `BEGIN IMMEDIATE` wait ≈ **1.3 s**.
- N=3: ≈ **2.7 s** — about **55 % of the 5 s `busy_timeout`**.

The headroom shrinks with N, which is why 8 is the recommended ceiling. Above
it, a write-lock convoy can starve heartbeats toward false evictions
(mitigated by the occupancy-sized window and BUSY-tolerant heartbeats, but not
eliminated). `complete_barrier` stays a single atomic transaction by design
(atomicity *is* the invariant); its hold time scales with the barrier group
size, so very large barrier groups are the other thing to watch — keep group
sizes bounded relative to the measured hold time.

### Per-worker JSONL journal is forensic-only at N>1

When `dump_to_jsonl` is enabled, each worker writes to its **own** file,
`<db>.journal.<uuid-hex>.jsonl`, where the hex is the worker_id's uuid suffix
(`cli.py:2516`, `follower.py:77-84`). This dissolves the file-corruption half
of a shared journal.

**It does not dissolve the ordering half.** Per-statement `timestamp` stamps
are *not* WAL commit order across processes, so at N>1 the JSONL journal is
**forensic-only — not a replayable backup** (ADR-030 Supported list). Restore
tooling **gates on worker-count = 1 provenance** and will refuse a multi-worker
journal. A true total order (an in-transaction `journal_seq` counter) is noted
as future work, not 0.6.0. At N=1 the journal remains a usable backup.

### Shared group + group-writable state dir + umask

**Every pack member needs write access to the DB file, its directory, and the
`-wal`/`-shm` sidecars** — the sidecars are created with the *creating
process's* umask (ADR-030 D6, §B.1 step 0). The common failure mode is the
web-hosted-leader + CLI-follower shape: the web service runs under a service
account while CLI followers run as the operator. Without a shared group and a
group-writable state directory (or followers running *as* the service user),
the follower **fails at first write — or worse, hits
`SQLITE_READONLY_CANTLOCK` on the `-shm` file mid-run**.

`elspeth join`'s filesystem preflight catches most of this at admission and
refuses with `JoinRefusedError` naming the path and permission
(`contracts/errors.py:830-842`). Practical setup:

```bash
chgrp <shared-group> <state-dir> <state-dir>/audit.db    # shared group
chmod g+ws <state-dir>                                    # group-writable + setgid
# both leader and follower processes run with:
umask 002                                                 # new files group-writable
```

The `setgid` bit (`g+s`) makes new sidecar files inherit the directory's
group; `umask 002` makes them group-writable.

### SQLCipher passphrase on every worker

SQLCipher is structurally SQLite (ADR-030; design §0). If the audit DB is
encrypted, **every follower must hold the passphrase**
(`cli.py:2374-2380`, ADR-030 D6). Without it the follower cannot open the DB
and fails immediately. Provision the passphrase to every pack member the same
way you provision it to the leader (e.g. the same Key Vault reference in each
worker's settings/environment).

### Shared clock

All liveness timestamps are **wall-clock UTC written by whichever process
acts** (ADR-030 D1; design §A.4). The one-host shape makes this sound: one host
⇒ one system clock ⇒ cross-worker comparisons are exact to within NTP slew,
orders of magnitude below the 80 s liveness window and the seconds-to-minutes
barrier-timeout granularity. Seat liveness, item-lease expiry, and
`barrier_blocked_at` timeout math all compare absolute timestamps.

**Containerized workers MUST share the host clock** (ADR-030 D6) — this is a
runbook-stated operator requirement, not something the engine probes. Do not
run pack members against independent container clocks; a skewed clock can cause
a false eviction or a mis-timed barrier flush.

### Quick reference: leader vs follower death

| Situation | Signal | Operator action |
|---|---|---|
| **Leader dies** (clean) | seat expired, `runs.status='running'`, followers idle/exit | `elspeth resume <RUN_ID> --execute` |
| **Leader frozen** (holds the write lock) | `WriteLockHeldError` + pid roster | `kill -9 <pid>` (confirm wedged + local first), **then** `elspeth resume --execute` |
| **Follower dies** | `run_workers` follower row heartbeat-expired | none — leader auto-evicts + reaps; optionally re-`join` |
| **Two racing resumes** | one `leader_acquire`; loser `NonResumableRunError` | none — one CAS winner, loser is side-effect-free |

---

## See Also

- [Resume Failed Run](resume-failed-run.md) — the supported recovery
  primitive; this runbook describes the diagnostics that precede it.
- [Investigate Routing](investigate-routing.md) — for explaining a
  specific token's terminal path after recovery completes.
- [Landscape architecture](../architecture/landscape.md) — `token_work_items`
  reference and the scheduler trust-model invariants.
- [ADR-026: Durable Token Scheduler](../architecture/adr/026-durable-token-scheduler.md) — design context and the rationale for CAS-gated leases.
- [ADR-030: Multi-Worker Deployment Shape](../architecture/adr/030-multi-worker-deployment-shape.md) — the one-host WAL pack: leader/follower topology, epoch fencing, the wedged-lock-holder remediation, and the operator requirements summarized above.
