# Runbook: Scheduler Lease Recovery

Diagnose and recover from durable-scheduler lease incidents in the ELSPETH
token scheduler (ADR-026).

Current as of 2026-05-24. This runbook assumes familiarity with the
[Landscape schema](../architecture/landscape.md) — in particular the
`token_work_items` table — and with [Resume Failed Run](resume-failed-run.md).

---

## When To Use

- A worker process crashed mid-run and you suspect work items are stuck `leased`.
- The drain loop logs `Scheduler has N in-memory continuations ... but no READY work item could be claimed` — a Tier-1 SCREAM invariant.
- The drain loop raises `Work queue exceeded 100000 iterations. Possible infinite loop in pipeline.`
- A CAS write raises `AuditIntegrityError` referencing `expected_lease_owner` mismatch.
- A run is `running` but no progress visible in `node_states` for several minutes.
- The audit trail shows token `attempt` counts climbing without corresponding application-level retries — you suspect lease expiry churn.

---

## Mental Model

`token_work_items` is the durable unit of work; the scheduler row is the
authoritative resume target, not in-memory state. Each row goes through a
finite lifecycle:

```
ready → leased → (waiting | blocked | pending_sink) → terminal | failed
```

A worker takes a row out of `ready` by calling `claim_ready`, which atomically
sets `status='leased'`, `lease_owner=<row-processor:run_id:uuid>`, and
`lease_expires_at = now + lease_seconds`. The lease is the worker's exclusive
claim on the row. Every state-changing call carries `expected_lease_owner`;
CAS-mismatch raises `AuditIntegrityError` rather than overwriting another
worker's progress.

`recover_expired_leases` runs at the top of every drain iteration. It returns
expired `leased` rows (`lease_expires_at < now`) owned by *other* callers
(`lease_owner IS NULL OR lease_owner != caller_owner`) back to `ready`,
incrementing `attempt` and rotating `work_item_id`. `pending_sink` rows are
the exception: they recover to `pending_sink`, preserving `attempt` and
`work_item_id`, because the transform work is already durable and only the
sink handoff is outstanding.

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
| Many `waiting` | Delayed retries or barrier coalesce. Check Step 6. |
| Many `blocked` | Barrier-join coalesce siblings outstanding. Check Step 7. |
| `leased` rows with old `updated_at` | Worker probably died. Continue to Step 3. |

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
`row-processor:<run_id>:<uuid>`, that worker is dead. If they span multiple
owners, multiple workers died (or are partitioned from the database).

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
a fresh `RowProcessor` whose drain loop calls `recover_expired_leases` on
every iteration; expired leases are reaped automatically.

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
the next live worker's drain iteration sweep the row. The `attempt` rotation
happens inside `recover_expired_leases`; there is no manual equivalent that
preserves it.

---

## Step 5: SCREAM — "in-memory continuations but no READY work item"

The drain loop raises:

```
OrchestrationInvariantError: Scheduler has <N> in-memory continuations
for run <RUN_ID> but no READY work item could be claimed
```

This is a Tier-1 SCREAM (`engine/processor.py` drain loop around line 2650).
It means the worker's in-memory `pending_items` cache references work the
scheduler row does not know about. Causes:

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
  same `waiting` / `blocked` rows.

Diagnosis queries:

```bash
sqlite3 <DB> "
  SELECT status, COUNT(*) AS n
  FROM token_work_items
  WHERE run_id = '<RUN_ID>'
    AND status IN ('waiting', 'blocked')
  GROUP BY status;
"
```

If `waiting` rows dominate, inspect `available_at` and whether the
condition the worker is polling for has fired. If `blocked` rows dominate,
the barrier-join coalesce is not reconciling — capture the `barrier_key`
and the sibling token set:

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

---

## Step 8: Escalation criteria

Escalate to the platform / data-integrity team if any of:

- Step 3 returned wedge rows (`lease_owner IS NULL` with `status='leased'`).
- Step 5 fired (SCREAM invariant).
- Step 6 fired and the cause is not an obvious plugin bug.
- `recover_expired_leases` returns 0 for a run that visibly has expired
  leases (suggests the recovery sweep itself is failing to write — CAS
  predicate or concurrent transaction issue).
- Multiple distinct `lease_owner` values appear in expired leases and
  the deployment is not configured for multi-worker (per ADR-026 RC6
  preconditions, multi-worker requires explicit opt-in).
- Any `AuditIntegrityError` raised by a CAS-gated state transition
  (`mark_terminal`, `mark_pending_sink`, `mark_blocked`, `mark_waiting`,
  `mark_failed`) — this is Tier-1 evidence integrity and must be triaged
  before the next resume.

Capture for escalation: the run_id, the full output of Steps 2–4, the
worker's stderr from the time of the incident, and the audit DB at
incident time (do not let it be overwritten by a fresh resume).

---

## See Also

- [Resume Failed Run](resume-failed-run.md) — the supported recovery
  primitive; this runbook describes the diagnostics that precede it.
- [Investigate Routing](investigate-routing.md) — for explaining a
  specific token's terminal path after recovery completes.
- [Landscape architecture](../architecture/landscape.md) — `token_work_items`
  reference and the scheduler trust-model invariants.
- [ADR-026: Durable Token Scheduler](../architecture/adr/026-durable-token-scheduler.md) — design context and the rationale for CAS-gated leases.
