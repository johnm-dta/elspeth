# ADR-030: Multi-Worker Deployment Shape — One-Host WAL Pack

**Date:** 2026-06-11
**Status:** Proposed (Draft at slice 0 of elspeth-1396d3f790; → Accepted at slice 5)
**Deciders:** John Morrissey, Claude Fable 5
**Tags:** scheduler, coordination, multi-worker, deployment, sqlite, wal,
          leader-election, fencing, multi-source-token-scheduler, adr-026,
          precondition-9

## Context

ADR-026 (durable token scheduler) Precondition #9 blocks N>1 worker
deployment until an ADR fixes three things: the deployment shape, worker
identity and lifecycle, and shutdown semantics. The refusal is hard-coded
today: `drain_scheduled_work` refuses to start when `peer_active_leases`
reports another live owner (`processor.py:2884-2911`), and `app.py` refuses
multi-worker web serving. The refusal protects nothing once a lease expires:
`tests/e2e/recovery/test_concurrent_resume.py` characterizes that a
competitor resume admitted mid-flight can drain a stalled worker's expired
leases and finalize the run out from under a suspended-but-alive winner.
Audit integrity held in every deterministically drivable interleaving; the
suspended winner's post-takeover continuation cannot be driven
deterministically in-process and must be closed by design.

F1 durability unification (ADR-029, epoch 20) removed the last durable-state
obstacle: a buffered barrier token IS a durable BLOCKED `token_work_items`
row; checkpoint blobs are deleted; checkpoints carry only barrier scalars.
Coordination state can therefore live entirely in the database.

The operator decision of 2026-06-10 designates branch
`feat/multi-source-token-scheduler` as the single→multi-worker transition and
the reason this ships as 0.6.0. Standing constraints: DB migration policy is
delete-the-old-DB (epoch bump, no Alembic); audit integrity is non-negotiable
(ADR-026 preconditions, Landscape audit DB); the workload is
LLM-call-dominated (seconds per row), so DB write throughput is rarely the
bottleneck; ELSPETH runs as CLI batch runs and behind a single-worker web
service.

Three deployment shapes were considered. The shape decision drives every
protocol choice downstream (topology, fencing-token storage, clock posture,
heartbeat transport), so it is fixed first and the alternatives are refused
explicitly rather than deferred silently.

## Decision

**ELSPETH multi-worker = a pack of cooperating OS processes on one host
sharing one WAL SQLite audit DB, coordinated as one epoch-fenced leader plus
claim-only followers.**

### D1 — Shape: N processes, one host, one WAL SQLite file

Rejected: in-process thread pools (the ticket's bug is cross-process by
nature, and `RowProcessor`'s claim/heartbeat state plane is documented
single-threaded, `processor.py:536-545`). Refused for 0.6.0: multi-host
(SQLite WAL coordination runs through the mmap-shared `-shm` file and is
documented-unsafe over network filesystems; Landscape-on-Postgres has zero
runtime test coverage; the epoch mechanism is `PRAGMA user_version`,
SQLite-only; the payload store is a local directory; multi-host reintroduces
clock skew into absolute-timestamp liveness and barrier-timeout math).

One host gives one clock (all liveness and `barrier_blocked_at` comparisons
are wall-clock UTC, exact to within NTP slew against an 80-second liveness
window) and one local payload store.

### D2 — Topology: one leader, N−1 claim-only followers

The five facts that live in exactly one process's memory today —
`ingest_sequence` assignment, checkpoint sequencing, barrier executor state
and trigger latches, the end-of-input flush decision, the sink-write phase —
are *roles*, not data to distribute. They belong to a single elected
**leader**, fenced by a monotonic `leader_epoch` in a new `run_coordination`
row (one per run, created by `begin_run`). Followers claim READY work,
process transforms, and dispose via the existing lease-fenced verbs; they
hold no run-long memory that gates completion, so run quiescence is fully
DB-derivable. Distributing ingest, barriers, or sinks would each need its own
ADR (ADR-026 already requires one for concurrent source iteration).

Uniformity rule: every run, including N=1, runs the full protocol (seat at
epoch 1, registered leader, fenced verbs), so the entire existing suite
exercises every fence on every run.

### D3 — Worker lifecycle and identity

Workers are operator-spawned CLI processes. `elspeth run` mints the run
leader (epoch 1, inside `begin_run`'s transaction). `elspeth resume` mints a
leader via the seat-takeover CAS (epoch+1) — its first durable act, which
closes the documented resume TOCTOU (`resume.py:668-674`); the same
transaction **evicts the deposed incumbent by identity** (the expired seat is
the proof of lost custody — no heartbeat predicate, so a deposed leader can
never linger `active` and pass membership fences). Followers are NOT evicted
at takeover; the new leader's housekeeping sweep evicts dead followers
individually under a grace period and a no-unexpired-leases precondition.
`elspeth join <run_id>` mints followers via one atomic `BEGIN IMMEDIATE`
admission (status RUNNING + live seat + matching `runs.config_hash`),
preceded by a filesystem-writability preflight. No supervisor daemon.

Identity is `worker:{run_id}:{uuid4().hex}`, registered in `run_workers` and
used as the scheduler `lease_owner` — extending, never reshaping, ADR-026's
opaque-owner-string scheme. Identities are single-use: `departed`/`evicted`
rows never return to `active`; a returning process mints a fresh uuid.

Shutdown: follower SIGINT = finish/abandon current claim, depart, exit 0;
leader SIGINT = `checkpoint_interrupted_progress`, fenced
`update_run_status(INTERRUPTED)`, `leader_release` with the seat zeroed.
Recovery of any leaderless run is `elspeth resume`. Followers never
auto-promote.

### D4 — Fencing: epoch verify-and-extend + membership, both in-statement

Every run-scoped write verb (finalize, run status, checkpoint,
`complete_barrier`, barrier adoption, the ingest transaction, the recovery
sweep, pending-sink terminalization) opens with a verify-and-extend CAS —
`UPDATE run_coordination SET leader_heartbeat_expires_at = … WHERE run_id …
AND leader_worker_id = :wid AND leader_epoch = :epoch` — inside the same
`BEGIN IMMEDIATE` transaction as the payload write; rowcount 0 rolls back
everything and raises `RunLeadershipLostError`. Every fenced verb thereby
doubles as a seat heartbeat. Claim and enqueue verbs carry a shared
membership EXISTS fence on `run_workers.status='active'`. Item-scoped writes
keep the existing item-lease CAS unchanged. Run-level liveness rides a
dedicated per-worker heartbeat thread (own connection, one transaction for
the leader's two rows, SQLITE_BUSY treated as liveness-unknown — never as
eviction — with `heartbeat_degraded` eventing).

### D5 — SQLite/WAL write discipline

All engine-side write transactions (scheduler, coordination, lifecycle,
checkpoint) use `BEGIN IMMEDIATE`, installed via a dialect-keyed `do_begin`
listener keyed on an **explicit per-connection write-intent execution
option** — never on "writer-mode engine", which is not observable at
`do_begin` time and which web read paths would match. **Web and dashboard
read surfaces convert to `LandscapeDB.from_url(read_only=True)`**
(`database.py:977-1031`) so reads provably never hold the write lock and
WAL's readers-don't-block-writers property actually holds. This eliminates
cross-process `SQLITE_BUSY_SNAPSHOT` aborts on SELECT-then-UPDATE shapes that
`busy_timeout` cannot retry, and closes the cross-process half of ADR-026
Precondition #4 (G27). `busy_timeout` (probe-pinned 5000 ms) is a polling
retry with no fairness; the run-liveness window is therefore sized against
worst-case write-lock occupancy (window ≥ 4 × (beat interval + busy_timeout))
and a two-process contention test with concurrent dashboard-style reads
measures max hold time and asserts heartbeat headroom before anything depends
on the discipline.

**Stated plainly:** a process frozen inside an IMMEDIATE transaction holds
the WAL write lock for as long as it exists and stalls the entire pack,
including the takeover CAS. `acquire_run_leadership` distinguishes
SQLITE_BUSY from CAS-loss and surfaces the registered workers'
`pid`/`hostname` forensics; the remediation is operator SIGKILL of the wedged
process (locks release on process death), documented in the G19 runbook.

### D6 — Operator requirements (one host, shared filesystem identity)

All pack members need write access to the DB file, its directory, and the
`-wal`/`-shm` sidecars (created with the creating process's umask). The
web-hosted-leader + CLI-follower shape requires a shared group with a
group-writable state directory, or followers running as the service user;
SQLCipher followers must hold the passphrase. `elspeth join` preflights
writability and refuses actionably. Containerized workers must share the host
clock. These are runbook-stated operator requirements, not probed invariants.

### D7 — Postgres port form (recorded now, enabled never in 0.6.0)

Every fence is a plain conditional UPDATE or a shared EXISTS fragment. The
load-bearing note for a future port: the epoch fence's dialect-portable form
is the verify-and-extend conditional UPDATE inside the payload transaction —
an UPDATE row-lock survives Postgres READ COMMITTED where a bare EXISTS
subquery's snapshot does not. Membership fences port as verify-UPDATEs;
`BEGIN IMMEDIATE` maps to `SELECT … FOR UPDATE` where a read precedes its
write; liveness comparisons must move to DB-server time. Enablement requires
its own ADR plus a runtime Landscape-on-Postgres test campaign.

### Supported / NOT supported

**Supported:** N processes (recommended 2–8), one host, one WAL audit DB; CLI
leader + CLI followers; web-hosted leader + CLI followers (subject to D6);
JSONL change journal at N>1 via per-worker files
`<db>.journal.<uuid-hex>.jsonl` — **forensic-only at N>1**: per-statement
timestamps cannot reconstruct cross-process commit order, so the journal is
not a replayable backup above one worker, and restore tooling gates on
worker-count=1 provenance.

**NOT supported (refused, not deferred silently):** multi-host anything;
network/NFS-mounted audit DBs; Postgres at runtime; uvicorn/gunicorn web
workers >1 (the `app.py` refusal stands); in-process row-level thread
workers; concurrent or sharded source ingestion (its own ADR per ADR-026);
follower auto-promotion.

### Reconciles stale text

- ADR-026 Precondition #4 (G27) "Required" marker — closed by f79332aa8 plus
  this ADR's write-intent IMMEDIATE discipline.
- The nonexistent `waiting`/`mark_waiting` state named in ADR-026 prose and
  the lease-recovery runbook (absent from `contracts/scheduler.py`;
  `available_at` is the delayed-retry mechanism).
- The G19 lease-recovery runbook is rewritten for N>1 (including the
  kill-the-wedged-incumbent step) in the design's slice 6.

## Consequences

### Positive Consequences

- ADR-026 Precondition #9 is satisfied: deployment shape, worker
  identity/lifecycle, and shutdown semantics are fixed by this ADR, and the
  `drain_scheduled_work` hard refusal is replaced by positive registry
  admission.
- The suspended-winner surface — the ticket's named, untestable-in-process
  gap — closes by construction: every run-scoped mutation is an epoch-fenced
  CAS against an injectable stale token, which also makes the scenario an
  ordinary deterministic test.
- Dead-vs-slow becomes solvable at both levels: run-level liveness rides a
  dedicated thread (long LLM calls stop being takeover bait), and item reaps
  require both signals stale or an explicit stall budget, with every rotation
  explained in the coordination ledger.
- N=1 runs the same protocol, so the entire existing suite exercises every
  fence on every run; the previously wedged RUNNING-with-dead-leader state
  becomes resumable.
- The protocol is Postgres-portable in atomicity form, not just SQL shape,
  without committing 0.6.0 to any Postgres work.

### Negative Consequences

- At-least-once external sink emission across leader suspension remains,
  bounded to one in-flight batch, ledger-refused and audit-attributed — the
  irreducible residue of fencing non-transactional side effects with a
  database token.
- Leader failover requires operator action (`elspeth resume`); a dead-leader
  run stalls until then, and a *frozen* lock-holder additionally requires an
  operator SIGKILL before recovery can proceed. A deliberate
  availability-for-explainability trade.
- Every leader-fenced verb adds one `run_coordination` UPDATE, and claims add
  one indexed membership probe — negligible against seconds-per-row LLM work,
  but measured in the contention tests rather than assumed.
- The JSONL change journal is demoted to forensic-only at N>1.
- One host is a hard ceiling on horizontal scale until a Postgres ADR.

## References

- Design document: cross-process multi-worker run coordination (option c),
  elspeth-1396d3f790 — schema (epoch 21), verbs, fence enumeration, barrier
  hand-off (arrivals AND branch losses), takeover walk, slice plan,
  adversarial-review dispositions.
- ADR-026 — durable token scheduler (Preconditions #4/#9, opaque owner
  strings, escape-hatch note).
- ADR-029 — journal is barrier-buffer truth (epoch 20 prerequisite; amended
  alongside slice 3 for journal-first acceptance, the per-firing-group intake
  snapshot, and late-arrival dispositions).
- `src/elspeth/core/landscape/database.py` — PRAGMA invariants (:55-72),
  read-only opens (:977-1031).
- `src/elspeth/core/landscape/scheduler_repository.py` — claim/lease CAS
  verbs, `complete_barrier` (:1255), `mark_blocked_barrier_terminal` (:2027),
  `list_blocked_barrier_items` (:2142).
- `src/elspeth/engine/processor.py` — single-threaded state-plane contract
  (:536-545), `peer_active_leases` refusal (:2884-2911), lost-branch
  notification (:2754).
- `tests/e2e/recovery/test_concurrent_resume.py` — the two pinned tests this
  design flips (:691, :757).

### Tickets

- **elspeth-1396d3f790** — cross-process multi-worker run coordination
  (option c), the feature this ADR shapes.
- **elspeth-2f23292372** — resume() entry guard (option b), landed; gains
  seat-liveness precision under this ADR.
- **elspeth-6116873e3b / elspeth-7bb7124e8f** — G25b isolation and G25h chaos
  campaigns (slice 5).
- **elspeth-3977d8ab60** — batches-row-COMPLETED-before-complete_barrier
  residual; unchanged and named.
- **elspeth-7294de558e** — single-bookkeeper unification, finished by
  audit-derived terminal status on all paths.
