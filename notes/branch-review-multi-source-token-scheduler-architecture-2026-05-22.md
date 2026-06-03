# Architecture orientation — feat/multi-source-token-scheduler

Date: 2026-05-22
Worktree: `.worktrees/multi-source-token-scheduler`
Base: `origin/RC5.2`
Scope: 249 files, +14642 / -1667 LOC, 22 commits

## What the branch delivers

Two outcomes:

1. **Multi-source** — pipelines were "exactly 1 source per run"; now they support
   N sources with explicit fan-in through a new first-class `QUEUE` graph node.
2. **Concurrent token execution** — tokens were processed one-at-a-time inline
   with source iteration; now they're persisted as durable work-items in a
   leased scheduler queue and processed when their dependencies are ready.

## Load-bearing primitives

### Token identity (3 distinct indices)

| field               | meaning                                               | who owns it |
| ------------------- | ----------------------------------------------------- | ----------- |
| `row_index`         | Position within a single source's emission stream     | per-source  |
| `source_row_index`  | NEW. Explicit per-source row index                    | per-source  |
| `ingest_sequence`   | NEW. Global multi-source ordering primitive           | global      |

Token construction is offensive: `_require_source_row_identity` crashes
with `OrchestrationInvariantError` if either new field is missing. Error
message: *"Do not fabricate source_row_index or ingest_sequence from
row_index."* That phrasing is institutional memory in an exception string.

### Durable scheduler (`contracts/scheduler.py`, `core/landscape/scheduler_repository.py`)

`TokenWorkItem` is the work-queue row. Lifecycle:

```
READY → LEASED → (WAITING | BLOCKED | PENDING_SINK) → TERMINAL | FAILED
```

- `work_item_id = f(run_id, token_id, node_id, attempt)` — deterministic,
  idempotent enqueue.
- `lease_owner` + `lease_expires_at` — lease pattern enables both crash
  recovery (`recover_expired_leases`) and future multi-worker (today: one
  worker per process).
- Every state transition uses `expected_lease_owner` (CAS pattern). Two
  workers can't both mark the same item terminal.
- `ensure_blocked_barrier_work_item` — joins / coalesces are durable rows,
  not in-memory state.
- `coalesce_cursor` lives on the scheduler row, so restart restores it
  without replaying upstream.

### DAG (`core/dag/builder.py`, `core/dag/graph.py`)

- New `NodeType.QUEUE`. Multi-source fan-in is **mandatory through a
  queue** — the graph refuses to validate if multiple producers MOVE-fan
  into one ordinary processing node (`graph.py:316–344`).
- `source_name` (the dict key) is injected into node config so two
  instances of the same plugin remain distinct DAG roots.
- Per-source `__quarantine__` edges — quarantines attribute to the source
  they came from.
- `get_first_pipeline_node` returns `None` for multi-source pipelines —
  any caller that previously required a non-None result is a latent bug.

### Recovery (`core/checkpoint/recovery.py`)

- `source_node_id` is the durable resume lineage anchor. Every row
  carries it.
- Resume ordering switched from `row_index` → `ingest_sequence`. Without
  this, resume would replay rows in a different order than the original
  run, breaking the audit trail.
- `get_unprocessed_row_data_by_source` validates each row with the
  *originating* source's schema class. Multi-source resume rejects the
  temptation to "just use a global row schema."
- Strictness: `isinstance(x, dict)` → `type(x) is dict` (exactly-builtin,
  no Mapping subclasses).

### Worker loop (`engine/processor.py:2540–2657`)

The drain loop is the heart:

```python
while True:
    release_waiting(...)            # WAITING → READY if dependencies cleared
    recover_expired_leases(...)     # Steal back leases that expired
    claimed = claim_ready(...)      # Try to lease the next READY item
    if claimed is None:
        if pending_items: SCREAM    # In-memory state with no scheduler row
        try claim_pending_sink(...) # Pre-leased sink-pending items
        break
    process_single_token(item)
    # Fan-out to state machine: mark_failed / mark_pending_sink /
    # mark_blocked / mark_terminal — all CAS-gated by lease_owner
```

- `MAX_WORK_QUEUE_ITERATIONS` bounded loop, can't spin forever.
- `pending_items` invariant: any in-memory continuation MUST be backed by
  a `READY` work item in the durable scheduler.

### Resume reconstruction (`orchestrator/core.py:3427–3555`)

- Pre-flight: `recorded_runtime_val_manifest` vs current → refuses resume
  if registries drifted. Tier-1 declaration-trust check.
- Multi-source branch: per-source schema reconstruction from
  `get_run_source_resume_records`.
- Single-source branch: legacy fallback path (separate from multi-source).
- **Smell**: `schema_contract = next(iter(...))` picks an arbitrary
  source's contract as "the" run contract. Either invariant or latent bug.

## Structural tensions worth surfacing

1. **Legacy single-source facade in `build_execution_graph`.** Accepts both
   `source=…` (old) and `sources=…` (new). When old is used, the source name
   is forced to `"source"` to preserve checkpoint identity. The comment is
   explicit: *"keeps the RC5.2 plugin-name/raw-config identity for
   checkpoint compatibility."* This contradicts the No Legacy Code Policy
   (CLAUDE.md). Either delete the facade (old checkpoints become unreadable;
   "delete the old DB" memory permits this) or amend the policy.

2. **`schema_contract = next(iter(...))` at orchestrator/core.py:3512.**
   Multi-source resume picks an arbitrary source's contract. Needs to be
   either an enforced invariant ("all sources must share contract") or
   the downstream consumer made source-aware.

3. **Per-call keyword-only `None` defaults on `create_initial_token` /
   `create_quarantine_token`.** Transitional shape — if every caller now
   passes them, drop the defaults and make the parameters truly required.

4. **Stale doc.** Repo-level CLAUDE.md still says *"Source: Load data —
   exactly 1 per run"*. Dim3 territory.

## Determinism story

- Cross-source ordering: `ingest_sequence` (set at ingest, preserved on
  scheduler row, used as resume sort key).
- Within-source ordering: `source_row_index` (kept distinct from
  `row_index`).
- Scheduler claim order: the scheduler's `claim_ready` ordering inside
  SQLite is what determines next-token selection. Need to confirm it's
  deterministic (ORDER BY ingest_sequence or work_item_id).
- Lease expiry → re-claim path bumps `attempt`; replay sees the original
  attempt count via `attempt_offset` in `_process_single_token`.

## What I did NOT exhaustively read

The four dim audits are designed to cover these:

- `engine/processor.py` (+743 LOC) — only structure + the drain loop read
- `engine/orchestrator/core.py` (+544 LOC) — only the resume reconstruction
  and signatures read
- The ~70 web/composer files exposing multi-source through UI/API
- All 249 file diff exhaustively

## Dim audit dispatch

The four prompts at `docs/prompts/multi-token/dim{1,2,3,4}.md` were
written for parallel fan-out. They share the same skeleton, differ only
by lens:

- **dim1** — engine static analysis (Python, systems, determinism+replay,
  quality, embedded DB). Hunts incorrect single-token / single-source
  assumptions, races, ordering bugs, replay risks.
- **dim2** — architecture / systems design (solution arch, systems,
  Python, security, embedded DB). Hunts stretched abstractions, tangled
  concerns, leaky boundaries, naming.
- **dim3** — policy / docs / CI / redaction / secret-refs (security,
  quality, systems, solution arch). Hunts stale single-source references
  in docs, CI, runbooks, redaction policy.
- **dim4** — tests / observability / release readiness (quality, Python,
  systems, determinism+replay, security, embedded DB). Hunts missing
  concurrent / multi-source coverage, weak assertions, observability gaps.

Each files Filigree tickets directly; consolidation later.
