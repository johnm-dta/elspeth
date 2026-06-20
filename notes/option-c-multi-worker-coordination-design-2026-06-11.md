# Cross-process multi-worker run coordination (option c) — FINAL design

**Ticket:** elspeth-1396d3f790 (P1 feature) · **Target:** 0.6.0, branch `feat/multi-source-token-scheduler` · **Schema:** epoch 20 → 21 (delete-the-DB)
**Prerequisite:** F1 durability unification (ADR-029 journal-as-truth) — LANDED at epoch 20; a buffered barrier token IS a durable BLOCKED `token_work_items` row; checkpoints carry only barrier scalars at format_version 5.
**Provenance:** judge-panel winner "One-Host WAL Pack" (deployment-shape-first leader/follower with epoch fencing), merged with the panel's graft list, then repaired against a 21-finding adversarial review (dispositions in §K). Every load-bearing code citation below was re-verified against the worktree at final-synthesis time.

---

## 0. The deployment-shape decision drives everything else

Three shapes were on the table. We commit to one and refuse the other two explicitly, because every protocol choice below (leader/follower topology, fencing-token storage, clock posture, heartbeat transport) is a consequence of the shape. The full ADR text is §F (Draft ADR-030).

### Shape 1 considered: in-process worker pool (N threads, one engine) — REJECTED
Tempting because the workload is LLM-call-dominated (I/O-bound; the GIL is irrelevant). Rejected:
1. **The ticket's bug is cross-process by nature.** The unprotected surface is a *second OS process* (a competing `elspeth resume`, or a web-spawned run racing a CLI run) finalizing a run under a suspended winner. Threads in one process do not close that race.
2. **The state plane is documented single-threaded.** `RowProcessor`'s claim/heartbeat instance state carries an explicit "single-threaded per row, so this instance state has no concurrent access" contract (verified, `processor.py:536-545`).
3. Intra-claim concurrency already exists where it pays (`ConcurrencySettings.max_workers` async batch-transform pool). Row-level thread parallelism is a different feature with a different ADR.

### Shape 3 considered: multi-host (Postgres or shared-filesystem SQLite) — REFUSED for 0.6.0
- SQLite WAL coordination runs through the `-shm` shared-memory file, which must be mmap-able by every connecting process; WAL over NFS/network filesystems is documented-unsafe. One host, full stop.
- Landscape-on-Postgres has zero runtime test coverage (DDL-compile checks only); the epoch mechanism is SQLite-only (`PRAGMA user_version`); SQLCipher deployments are structurally SQLite.
- The `FilesystemPayloadStore` is a local directory tree; multi-host would need a shared store.
- Multi-host reintroduces clock skew into a protocol whose liveness windows and `barrier_blocked_at` timeout math compare absolute timestamps (§A clock posture).

**However**: every coordination verb below is dialect-portable by construction — plain conditional UPDATEs and shared fence fragments, no SQLite-only SQL except a dialect-keyed `BEGIN IMMEDIATE` hook. ADR-026 names Postgres-with-the-same-CAS-semantics as the horizontal-scale escape hatch; §F documents the exact Postgres port form of each fence (the verify-UPDATE row-lock pattern) so "portable" is true in the *atomicity* dimension, not just the SQL-shape dimension.

### Shape CHOSEN: N cooperating processes, one host, one WAL SQLite audit DB
- **Workers are operator-spawned OS processes.** `elspeth run` / `elspeth resume` produce the run **leader**; the new `elspeth join <run_id>` produces a **follower**. No supervisor daemon in 0.6.0; SIGINT semantics per role in §B.
- **elspeth-web stays a single serving worker** (the hard refusal in `app.py` stands; ProgressBroadcaster is process-local). The web execution thread can *be* a leader exactly as today; operators attach CLI followers to it. Web-managed worker pools are a non-goal (§J). The cross-uid filesystem reality this shape implies is an operator requirement with a join-time preflight (§B.1 step 0).
- **Write contention is a non-issue at this workload's shape — provided reads stay reads.** Rows take seconds (LLM calls); journal writes are a handful per row. WAL gives readers-don't-block-writer *only if readers open read-only*; `busy_timeout=5000` (probe-enforced, `database.py:58-72`, verify-or-crash at `database.py:460-513`) queues — by polling retry, with no fairness guarantee — the single writer. Recommended N is small (2–8); §J names the convoy risk and §A.3 specifies how the heartbeat survives it.
- **The IMMEDIATE-begin discipline, mechanism specified.** SQLite transactions are currently DEFERRED everywhere — grep-verified: no `BEGIN IMMEDIATE` exists anywhere in `src/elspeth/core/landscape/`. Cross-process, a deferred read-then-write transaction (claim's SELECT→UPDATE inside one `engine.begin()`, `scheduler_repository.py:608-722`; `complete_barrier`'s validate-then-write, `:1255-1506`) can abort with snapshot-upgrade `SQLITE_BUSY` that `busy_timeout` does **not** retry. **Decision (corrected after review — the original "writer-mode engines" keying was unimplementable and would have turned web dashboard reads into write-lock holders):**
  1. `LandscapeDB` gains a **dedicated write-intent path**: scheduler, coordination, lifecycle, and checkpoint *write* verbs run their transactions on connections carrying an explicit `elspeth_write_intent` execution option (equivalently: a second Engine instance reserved for engine-side writers). A dialect-keyed `do_begin` event listener issues `BEGIN IMMEDIATE` **only** for connections carrying that option. Nothing is keyed on "writer-mode engine", which is not an observable property at `do_begin` time and which almost every web read path would have matched.
  2. **Web read paths are converted to true read-only opens.** Today the only `read_only=True` caller in the tree is `mcp/analyzer.py:64`; every web read surface (`accounting.py:38`, `diagnostics.py:117`, `outputs.py:175`, `discard_summary.py:43`, `failure_samples.py`, `auth/audit.py:174-246`, `tutorial_service.py:193/302`) opens writable via `LandscapeDB.from_url(..., create_tables=False)`. These move to `from_url(read_only=True)` — the facility exists and is WAL-aware (`database.py:977-1031`, `_sqlite_read_only_url` at `:950`) — so dashboard reads provably never enter the write-lock domain. This lands in slice 1, before the IMMEDIATE listener, and the slice-1 contention test runs dashboard-style reads concurrently with the two-process claim hammer and **asserts heartbeat latency stays inside the liveness window**.
  3. Scheduler verbs additionally wrap in one bounded retry-with-jitter on `OperationalError: database is locked` as a belt. This package formally closes ADR-026 Precondition #4 (G27)'s cross-process residual; ADR-030 reconciles the stale ADR-026 text that still marks G27 "Required".
- **PRAGMA uniformity (G28)** is already probe-enforced per process: `LandscapeDB` refuses to open on PRAGMA mismatch and `TokenSchedulerRepository` re-probes at construction (verified, `scheduler_repository.py:86-118`). Every joining process opens through `LandscapeDB`; ADR-030 states this as a cross-process invariant.
- **Shared-config admission check:** a joiner refuses to attach if its resolved settings hash differs from `runs.config_hash` (verified present, `schema.py:129`) — same DB file, same payload-store path, same plugin set, by construction.
- **A frozen lock-holder stalls the whole pack, including recovery.** SQLite file locks live as long as the holding process; nothing times them out. A leader SIGSTOPped *inside* an IMMEDIATE transaction blocks every peer write — heartbeats, claims, and the takeover CAS itself. This is stated plainly in ADR-030; the remediation is operator SIGKILL of the wedged process (locks release on process death), and `acquire_run_leadership` is specified to make that actionable (§B.4).

### The topology the shape implies: one leader + N−1 followers
Five things live in exactly one process's memory today, each protected by a unique constraint that makes a second writer crash loudly: `ingest_sequence` assignment (`source_iteration.py:799-800` + `rows` UNIQUE(run_id, ingest_sequence)), checkpoint sequencing (in-memory counter, verified `checkpointing.py:34` + `checkpoints` UNIQUE(run_id, sequence_number)), barrier executor state + trigger latches, the end-of-input flush decision, and the sink-write phase. The deployment-shape-first answer is to **not** distribute any of them: they are *roles*, and roles belong to a single elected **leader**. Followers are claim-only transform engines. Distributing ingest, barriers, or sinks would each need its own ADR (ADR-026 already says concurrent source iteration does).

| Role | May do | May NOT do |
|---|---|---|
| **Leader** (exactly one, epoch-fenced) | ingest source rows; claim READY + PENDING_SINK; own barrier executors, run barrier intake/adoption, evaluate triggers, `complete_barrier`; write checkpoints; run end-of-input flush; write sinks; terminalize PENDING_SINK; run the recovery sweep; finalize the run | — |
| **Follower** (0..N−1) | claim READY; process transforms (LLM calls); `mark_blocked` / `mark_pending_sink` / `mark_terminal` / `mark_failed`; record durable coalesce branch losses (§E.5); enqueue child continuations; item heartbeats | ingest, checkpoint, barrier adoption/evaluation/flush, sink writes, PENDING_SINK terminalization, recovery sweep, finalize, run-status writes |

Followers parallelize exactly the expensive thing (LLM-dominated transform work) and nothing else.

### Two design invariants stated up front

**Uniformity rule (graft, Design 1).** *Every* run — including an N=1 CLI run — creates its `run_coordination` row in `begin_run`, registers its origin worker in `run_workers` as leader at epoch 1, and threads the epoch token through every fenced verb. Single-worker execution is the N=1 degenerate case of the protocol, not a parallel code path. Consequence, enforced as test doctrine: every fence is exercised by the entire existing suite on every run, not only in N>1 chaos tests. Repository-level unit tests that drive scheduler verbs directly register a worker in their fixtures.

**Single-use identity (graft, Design 1).** A `run_workers` identity that reaches `departed` or `evicted` can never return to `active`. A returning process mints a fresh uuid and re-admits (join or resume). This one rule makes eviction a permanent fence with no re-admission race.

---

## A. Worker identity, registry, and run-level heartbeat

### A.1 Identity
`worker_id = f"worker:{run_id}:{uuid4().hex}"`, minted at registration and used **as the scheduler `lease_owner` string** (injected via the existing `scheduler_lease_owner` constructor parameter — verified injectable, `processor.py:525`, which today defaults to `row-processor:{run_id}:{uuid4().hex}`). This *extends* the opaque-owner-string semantics exactly as ADR-026 requires — `scheduler_events` columns are untouched. Role is a registry attribute, never parsed from the string. Pre-epoch-21 `row-processor:` owners cannot appear (epoch bump ⇒ DB deleted).

### A.2 Schema (epoch 21 — DDL sketch)

```sql
CREATE TABLE run_coordination (              -- exactly one row per run, created by begin_run
    run_id                      TEXT PRIMARY KEY REFERENCES runs(run_id),
    leader_worker_id            TEXT,                       -- NULL = vacant seat
    leader_epoch                INTEGER NOT NULL DEFAULT 0, -- THE fencing token; monotonic, bumps on every acquisition
    leader_heartbeat_expires_at TIMESTAMP,                  -- run-level leader liveness clock
    updated_at                  TIMESTAMP NOT NULL,
    CHECK ((leader_worker_id IS NULL) = (leader_heartbeat_expires_at IS NULL))
);

CREATE TABLE run_workers (
    worker_id            TEXT PRIMARY KEY,                  -- 'worker:{run_id}:{uuid}'
    run_id               TEXT NOT NULL REFERENCES runs(run_id),
    role                 TEXT NOT NULL CHECK (role IN ('leader','follower')),
    status               TEXT NOT NULL CHECK (status IN ('active','departed','evicted')),
    registered_at        TIMESTAMP NOT NULL,
    heartbeat_expires_at TIMESTAMP NOT NULL,                -- run-level worker liveness clock
    departed_at          TIMESTAMP,
    evicted_at           TIMESTAMP,                         -- forensics (graft, Design 1)
    evicted_by_worker_id TEXT,                              -- forensics: who ran the eviction CAS
    pid INTEGER, hostname TEXT, entry_point TEXT,           -- forensic only — EXCEPT pid, surfaced by
                                                            -- the BUSY-takeover diagnostic (§B.4)
    CHECK ((status = 'evicted') = (evicted_at IS NOT NULL))
);
CREATE INDEX ix_run_workers_liveness ON run_workers(run_id, status, heartbeat_expires_at);

CREATE TABLE run_coordination_events (       -- append-only ledger, same discipline as scheduler_events
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,         -- authoritative replay order (process-stamped
                                                            -- recorded_at can invert commit order under
                                                            -- busy_timeout stalls; seq cannot)
    event_id     TEXT NOT NULL UNIQUE,       -- sha256(canonical_json(identity)), same dedup recipe as
                                             -- scheduler events (scheduler_repository.py:434-455)
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    event_type   TEXT NOT NULL CHECK (event_type IN
        ('worker_register','worker_depart','worker_evict','worker_stalled',
         'leader_acquire','leader_release','leadership_lost',
         'fence_refusal','heartbeat_degraded','finalize')),
    worker_id    TEXT NOT NULL,
    leader_epoch INTEGER,
    recorded_at  TIMESTAMP NOT NULL,         -- forensic wall-clock; NOT the replay order
    context_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX ix_run_coordination_events_run ON run_coordination_events(run_id, seq);

CREATE TABLE coalesce_branch_losses (        -- §E.5: durable cross-worker branch-loss hand-off
    loss_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    coalesce_name TEXT NOT NULL,
    row_id        TEXT NOT NULL,
    branch_name   TEXT NOT NULL,
    token_id      TEXT NOT NULL,
    reason        TEXT NOT NULL,             -- failed / quarantined / error_routed / ...
    recorded_by   TEXT NOT NULL,             -- worker_id
    recorded_at   TIMESTAMP NOT NULL,
    adopted_epoch INTEGER,                   -- NULL = not yet replayed into leader memory
    UNIQUE (run_id, coalesce_name, row_id, branch_name)
);
```

Run-level events get their **own table** rather than widening `scheduler_events`: that table's CHECK pins exactly 11 item-centric event types (verified, `schema.py:544-551`) and its `token_id`/`work_item_id` are NOT NULL — run-scoped events do not fit, and item-event consumers should not have to skip them. Every coordination state transition writes its event row **in the same transaction** as the state change (the scheduler_events discipline). Successful heartbeats do NOT write events — matching `heartbeat_lease`, which only ledgers the LEASE_LOST miss (verified, `scheduler_repository.py:1040-1090`). The one exception to same-transaction eventing is `fence_refusal`: the payload transaction that tripped the fence rolls back, so the refusal event is written **on a fresh connection** immediately after rollback. This is **best-effort attribution, not a durability guarantee** — a crash between the rollback and the fresh-connection write loses the event, which is benign precisely because the refused transaction left no durable state needing explanation; the design originally overclaimed this and the claim is hereby corrected. The run's coordination history (seat custody, membership, refusals, finalization) is reconstructable from this ledger, ordered by `seq`; `recorded_at` is forensic color only, and the same caveat applies to the cross-file JSONL merge (§C.4 row 13).

### A.3 Who writes, cadence — the dedicated heartbeat thread
The per-item heartbeat fires only *between* plugin calls, so a single long LLM call makes an alive worker look dead (documented limitation, verified `processor.py:3680-3689`; the constructor enforces heartbeat < lease, verified `processor.py:527-535`). The run-level heartbeat therefore must NOT ride the drain loop. **Every worker runs a small daemon heartbeat thread** that owns its own DB connection and touches *only* its own `run_workers` row (and, for the leader, the `run_coordination` row — **both in ONE transaction**, so the two liveness clocks can never skew in the dangerous worker-fresher-than-seat direction) via single-row CAS UPDATEs. It shares zero Python state with the processor — it communicates through one atomic "coordination_lost" flag — so the single-threaded `RowProcessor` contract is preserved.

- `worker_heartbeat`: `UPDATE run_workers SET heartbeat_expires_at = :now + :window WHERE worker_id = :wid AND status = 'active'`. rowcount 0 ⇒ this worker is no longer active ⇒ read run status: run terminal + own row `departed` ⇒ clean exit path; otherwise set the flag; the drain loop raises `RunWorkerEvictedError` (Tier-2) at the next claim/node boundary.
- **Error semantics under contention (specified, not assumed):** a heartbeat UPDATE that fails with `OperationalError`/SQLITE_BUSY is **"liveness unknown", never "evicted"** — the thread retries immediately with short local backoff inside the same tick, never crashes (all exceptions are caught at the tick boundary; a dead heartbeat thread is a guaranteed false eviction), and never sets the coordination-lost flag on a DB error — only on rowcount-0. After `k` (default 3) consecutive busy failures the thread emits a `heartbeat_degraded` coordination event on a fresh connection, so a later eviction is diagnosable post-hoc as "could not reach the DB" rather than "process died". The liveness window is sized against worst-case write-lock occupancy, not just scheduling stall: **window ≥ 4 × (beat interval + busy_timeout)** = 4 × 20s = 80s at defaults, and the slice-1 contention test measures max write-lock hold time against it.
- **The heartbeat verb returns a `CoordinationSnapshot(leader_worker_id, leader_epoch, seat_live: bool)`** read in the same transaction (graft, Design 2) — followers learn of leader death or seat handover for free on their existing cadence, with no extra polling query. **A leader-mode process that observes a snapshot whose `leader_worker_id` is not itself treats that as fatal** (same path as the eviction flag): it has been deposed even if its registry row was not yet evicted. This is a second, independent latch behind the per-verb fences.
- The leader's seat heartbeat is largely subsumed by the verify-and-extend fence (§C.4): every fenced verb extends `leader_heartbeat_expires_at` as a side effect. The thread still beats the seat on its tick so an *idle* leader (waiting on a slow source, polling a barrier timeout) stays live.
- **Cadence:** `run_heartbeat_seconds = 15`, `run_liveness_window = 80` (per the sizing rule above). Both configurable; the window must exceed worst plausible same-host scheduling stall AND worst-case lock occupancy, NOT the longest LLM call — that is the point of the separate thread. Item-lease defaults (300s lease / 60s heartbeat, verified `processor.py:404-405`) are unchanged.

### A.4 Clock-skew posture
All liveness timestamps are wall-clock UTC written by whichever process acts — same as `barrier_blocked_at` (stamped by `mark_blocked`, its only writer; verified `scheduler_repository.py:1127-1131`) and `lease_expires_at`. **The chosen shape makes this sound:** one host ⇒ one system clock ⇒ cross-worker comparisons are exact to within NTP slew, orders of magnitude below the 80s liveness window and the seconds-to-minutes barrier-timeout granularity (restore computes `now − min(barrier_blocked_at)`). ADR-030 records the single-clock assumption as one reason multi-host is refused, and notes a future Postgres shape must switch liveness comparisons to DB-server time. Containerized workers must share the host clock — a runbook-stated operator requirement, not a probed invariant.

### A.5 Dead-vs-slow becomes solvable at the ITEM level too
With run-level liveness in hand, `recover_expired_leases` (verified, `scheduler_repository.py:814`) gains a liveness-aware predicate. Today's rule "item lease expired ⇒ reapable" cannot distinguish a dead worker from one stuck in a 10-minute LLM call. Crucially, `heartbeat_lease`'s CAS predicate is `(work_item_id, run_id, status=LEASED, lease_owner)` **without an expiry check** (verified, `scheduler_repository.py:1014-1028`) — an expired-but-unreaped lease can be revived by its still-alive owner. So:

> **Reap rule (new):** an expired item lease is reaped iff its `lease_owner` maps to a `run_workers` row that is (a) absent, (b) `status='evicted'` or `'departed'`, or (c) `active` with `heartbeat_expires_at < now − grace` — **OR** the lease has been expired longer than a hard stall budget (`item_stall_budget_seconds`, default `2 × scheduler_lease_seconds`), covering the heartbeat-thread-alive-but-drain-wedged pathology.

The two clocks have distinct meanings: item lease = "is this *claim* making progress"; run heartbeat = "is this *process* alive". Reaping requires both signals stale (or the budget blown). Consequences:
- **Slow is now safe**: a live-registered worker stuck in a 10-minute LLM call keeps its registry lease via the thread; its expired *item* lease is not reaped; its next `heartbeat_lease` CAS still matches and revives it. `scheduler_lease_seconds` stops being a correctness bound and becomes a hygiene bound.
- **Dead is detected in ~80–95s** (registry window + grace), regardless of item lease length.
- **Reap-while-registered is a legal, evented state** (corrected after review — the earlier draft promised "a worker can never observe its items were taken while still registered", which arms (c) and the stall budget contradict by construction, since the §C.2 no-unexpired-leases eviction precondition can make a stale-but-leased worker unevictable for up to a full item lease). The honest invariant is: **every attempt rotation against a non-evicted owner is explained in the coordination ledger** — a reap under arm (c) or the stall budget writes a `worker_stalled` event (same transaction as the rotation) naming the owner and the reaped item, so the rotation is reconstructable from the ledger alone. A worker whose items were reaped under arm (c) and which then revives simply continues as a member in good standing; its old claims rotated, attempt identity separates the executions, audit integrity holds.
- Attempt-rotation semantics on reap are unchanged (READY at attempt+1 with rotated `work_item_id`; PENDING_SINK preserved — `scheduler_repository.py:879-907`); the self-steal guard and symmetric SELECT/UPDATE predicates are preserved; the new predicate only *narrows* the reap set.
- The recovery sweep becomes **leader-only** (epoch-fenced, §C.4) — kills the anticipated O(workers²) sweep pattern and gives every attempt rotation a single attributable author.
- `peer_active_leases` and the drain-entry refusal it powers (verified, `processor.py:2884-2911` — today's only run-level exclusion, hard-coded to cite Precondition #9) are **retired**, replaced by positive registry admission; the verb is kept as a read-only diagnostic.

---

## B. Join protocol — worker #2 attaches to a RUNNING run

Join is a **new public entry point** (`Orchestrator.join_run(run_id)` / CLI `elspeth join <run_id>`), not a `resume()` variant. `resume()` keeps refusing RUNNING-with-live-leader; racing resume() remains the bug; join is the feature.

### B.1 Join sequence (follower)
0. **Filesystem preflight** (new after review): verify write access to the DB file, its directory, and any existing `-wal`/`-shm` sidecars before touching the registry; refuse with an operator-actionable `JoinRefusedError` naming the path and the required permission otherwise. The web-hosted-leader shape runs the service under a service account while CLI followers run as the operator; without a shared group + group-writable state directory the follower fails at first write (or worse, `SQLITE_READONLY_CANTLOCK` mid-run on the `-shm` file). SQLCipher deployments additionally require the follower to hold the passphrase. The shared-group/umask requirement is documented in the slice-6 runbook and stated in ADR-030 as an operator requirement alongside the shared-clock requirement.
1. Open the DB through `LandscapeDB` (PRAGMA probe + epoch check inherited — G28 cross-process uniformity by construction).
2. **Atomic admission — one `BEGIN IMMEDIATE` transaction** (graft, Design 2):
   - `SELECT runs.status, runs.config_hash` — status must be `RUNNING`, else Tier-2 `JoinRefusedError` ("run is terminal" / "FAILED or INTERRUPTED — use `elspeth resume`"); the joiner's resolved settings hash must equal `config_hash`, else refused (a joiner with a different pipeline would disagree about the graph and barrier keys).
   - `SELECT` `run_coordination` — the leader seat must be **live** (`leader_heartbeat_expires_at > :now`), else refused with "no live leader — use `elspeth resume` to take the seat" (a follower must never be the first process on an abandoned run; barrier/ingest state needs a leader). Join and takeover stay disjoint verbs.
   - `INSERT` the `run_workers` row (`role='follower'`, `status='active'`) + `worker_register` event. COMMIT.

   **Join-vs-finalize, the honest argument** (corrected after review — the earlier draft claimed the finalize statement's "membership predicates" would see the joiner; the §D statement had no such predicate): admission and finalization serialize on the WAL write lock because both are IMMEDIATE transactions, and the race is harmless in *both* orders for a structural reason, not a predicate: a joiner that registers before finalize **cannot create work** — it can only claim existing READY rows, and §D's quiescence predicate refuses COMPLETED while any claimable row exists; a joiner that registers after finalize reads terminal status at its first claim/idle check and departs. Additionally, the §D finalize transaction now flips any remaining `active` non-leader `run_workers` rows to `departed` (`context: run_finalized`, evented), so a follower that died after admission cannot leave a permanently-"active" row on a COMPLETED run misleading the forensic story; a live idle follower discovers the flip via its heartbeat rowcount-0 → reads terminal run status → exits 0 through the clean-departure path (§A.3).
3. Start the heartbeat thread. Construct a **follower-mode `RowProcessor`**: scheduler + payload store wired as usual, but NO source plugins, NO checkpoint coordinator, NO sink pipeline, and NO barrier executors. It needs the `ExecutionGraph` (from config) only to recognize barrier and sink nodes for hand-off routing. `lease_owner = worker_id`.
4. Drain loop: `claim_ready` only (never `claim_pending_sink` — sink work is the leader's), now carrying the membership fence (§C.4 row 2). Process the token node-by-node with the existing item heartbeat. Dispositions are today's verbs, all already CAS-fenced on `expected_lease_owner` (`_transition`, verified `scheduler_repository.py:2271`):
   - barrier node reached ⇒ `mark_blocked(barrier_key=…)` — durable hold, **no in-memory accept** (§E) — and the traversal ends;
   - lossy disposition of a fork-lineage token mapped to a coalesce ⇒ the durable branch-loss record is written **in the same transaction** as the `mark_failed`/divert (§E.5);
   - sink-bound ⇒ `mark_pending_sink` with the full handoff bundle — the leader picks it up; **followers never perform sink I/O**;
   - otherwise ⇒ `mark_terminal` / `mark_failed`. Child continuations use the existing idempotent `enqueue_ready` (deterministic IDs + reconciliation), now membership-fenced.
5. No claimable READY work ⇒ idle with backoff, re-checking run status and the heartbeat's coordination snapshot. Run terminal ⇒ `depart_worker` (CAS `active → departed` + `worker_depart` event, no-op if finalize already departed it) + exit 0. Seat dead ⇒ finish/abandon current item, take no new claims after a grace period, exit with "no live leader; run can be taken over via `elspeth resume`". **Followers never auto-promote** in 0.6.0: takeover needs the full resume reconstruction anyway, and implicit promotion hides an operator-relevant recovery event.
6. SIGINT: finish or abandon the current claim (abandon = let the lease lapse; the liveness-aware reaper hands it over cleanly), depart, exit.

### B.2 Does a joiner reuse `_restore_barriers_from_journal`? Who owns trigger evaluation?
**No, and it never needs to.** `_restore_barriers_from_journal` (`processor.py:580-813`) rebuilds *barrier executor memory*, and followers have none by construction. Barrier trigger evaluation (count/condition/timeout) is **leader-only** (§E). The restore path is reused unchanged — plus the reconcile dispositions in §E.4 — by exactly one caller class: a leader bootstrapping over a journal with pre-existing BLOCKED rows (fresh resume, or seat takeover). This is precisely what makes join cheap: no checkpoint read, no ResumePoint, no `update_run_status`, none of `reconstruct_resume_state`'s machinery; the resume coverage gate is irrelevant because a follower never replays sources.

### B.3 Interaction with the landed resume() entry guard (option b)
The guard (verified at `resume.py:674-680`, refusing via `check_run_status_resumable`) **keeps its polarity and gains precision** — "RUNNING" alone no longer decides; the check consults `run_coordination`:
- `RUNNING` + live seat ⇒ refuse: *"Run is in progress under live leader <worker_id> (seat expires <ts>) — use `elspeth join`"*. (The pinned test at :757 re-pins on this reason.)
- `RUNNING` + **expired** seat ⇒ resumable: the dead-leader takeover path, today a *wedged* state (status stuck RUNNING, guard refuses unconditionally, run unrecoverable without manual surgery). Check-then-act at the guard is acceptable because the leadership CAS (§B.4) is the arbiter.
- `FAILED` / `INTERRUPTED` ⇒ resumable iff the seat is vacant or expired.

### B.4 Closing the documented resume TOCTOU
The known residual (verified comment, `resume.py:668-674`: two resumes can both observe FAILED before either flips to RUNNING) closes because the **first durable act of resume() becomes the seat-acquisition CAS**, executed where `reconstruct_resume_state` today performs its first durable write:

```sql
BEGIN IMMEDIATE;
SELECT leader_worker_id FROM run_coordination WHERE run_id = :run;  -- capture :prior (may be NULL)
UPDATE run_coordination
   SET leader_worker_id = :wid,
       leader_epoch = leader_epoch + 1,
       leader_heartbeat_expires_at = :now + :window,
       updated_at = :now
 WHERE run_id = :run
   AND (leader_worker_id IS NULL OR leader_heartbeat_expires_at < :now);
-- rowcount must be 1, else ROLLBACK + NonResumableRunError("run leadership is held by <owner>") — zero mutation
UPDATE runs SET status='running', completed_at=NULL
 WHERE run_id = :run AND status IN ('failed','interrupted');   -- skipped on the dead-leader RUNNING takeover arm
UPDATE run_workers SET status='evicted', evicted_at=:now, evicted_by_worker_id=:wid
 WHERE worker_id = :prior AND status='active';                 -- evict the DEPOSED LEADER BY IDENTITY,
                                                               -- unconditionally: the expired seat IS the
                                                               -- proof of lost custody; no heartbeat predicate
INSERT INTO run_workers (worker_id=:wid, role='leader', status='active', ...);
INSERT INTO run_coordination_events ('leader_acquire', leader_epoch=:new_epoch, ...);  -- + worker_evict event
COMMIT;
```

Two corrections after review, both load-bearing:

1. **The incumbent is evicted by identity, never by liveness.** The earlier draft evicted `WHERE status='active' AND heartbeat_expires_at < :now`, which can MISS the deposed leader entirely when the worker-row clock is fresher than the seat clock (the §A.3 one-transaction rule removes the leader-side skew source, but identity-eviction makes the takeover correct independent of any clock). The seat row is read in the same transaction; whatever identity sat in `leader_worker_id` is unconditionally CAS-evicted. A deposed leader is therefore *always* evicted at takeover — single-use identity holds, and every membership fence (§C.4 rows 2–3) refuses it from the instant of takeover.
2. **No bulk follower eviction in the takeover transaction.** The earlier draft bulk-evicted every stale-heartbeat worker, with neither the grace period nor the no-unexpired-leases precondition that §C.2 declares mandatory — under a write-lock convoy (the exact circumstance surrounding a leader death) that would evict the entire healthy pack. Followers are left alone at takeover; the new leader's post-takeover housekeeping sweep evicts them under the full §C.2 predicate (grace + no unexpired leases), gracefully and individually.

`acquire_run_leadership` additionally **distinguishes SQLITE_BUSY from CAS-loss**: on a busy timeout it does not report "leadership held" (false) but raises an operator-actionable error — "the audit DB write lock is held by a live or frozen process; registered workers: <pid/hostname rows from run_workers>" — surfacing the forensic `pid`/`hostname` columns so the operator knows exactly what to SIGKILL. The kill-the-wedged-incumbent step is part of the G19 runbook rewrite (slice 6), and ADR-030 states plainly that a frozen lock-holder stalls the entire pack including recovery.

Exactly one of two racing resumes commits; the loser's rowcount-0 is side-effect-free, preserving the pinned refusal-before-mutation discipline (`test_concurrent_resume.py:740-754, 820-833`). The terminal-run arm is refused earlier at the entry guard, with the immutability guards (`run_lifecycle_repository.py:333-349, 938-950` — verified) retained beneath as the durable backstop. Fresh runs: `begin_run` creates the `run_coordination` row and acquires epoch 1 in the same transaction (uniformity rule).

---

## C. Dead-vs-slow takeover, eviction, and fencing the suspended winner

### C.1 Detection
- **Leader dead** ⇔ `run_coordination.leader_heartbeat_expires_at < now`. Because the heartbeat rides a dedicated thread (§A.3) with specified busy-retry semantics, expiry means the process is dead, SIGSTOPped, OS-wedged, or persistently locked out of the DB — NOT merely inside a long LLM call. 80s window vs 300s item leases.
- **Follower dead** ⇔ its `run_workers.heartbeat_expires_at < now`. Consequence: its expired item leases become reapable under §A.5. Nothing else — follower death never triggers run-level ceremony beyond eviction housekeeping.
- **Slow** ⇔ run heartbeat live. Its expired item leases are *not* reaped (until the stall budget) and `heartbeat_lease` revives them — the still-live-sink-writer doctrine (verified, `processor.py:3285-3293`: stranded-until-expiry is deliberate because "the prior worker may still be alive and finishing the sink write") generalized to all work.

### C.2 Eviction protocol
Two eviction paths, both CAS, both evented in the same transaction (`worker_evict` + forensics columns — graft, Design 1):

1. **Leader evicts a dead follower** (maintenance housekeeping): one `BEGIN IMMEDIATE` transaction —
   - verify, in-transaction, that the target holds **no unexpired item leases** (`token_work_items` LEASED rows with `lease_owner=:target AND lease_expires_at >= :now`) — the belt-and-braces precondition (graft, Design 1): registry eviction must never outrun a lease the item layer still considers possibly alive;
   - `UPDATE run_workers SET status='evicted', evicted_at=:now, evicted_by_worker_id=:leader WHERE worker_id=:target AND status='active' AND heartbeat_expires_at < :now − :grace` — rowcount 0 ⇒ the worker heartbeated (or still holds live leases) ⇒ benign skip;
   - `worker_evict` event. Eviction-before-reap ordering holds **on this path**: only after eviction does the sweep reap the target's expired item leases. (Arm-(c) and stall-budget reaps are the documented, `worker_stalled`-evented exception — §A.5.)
2. **Takeover evicts the deposed leader, by identity, unconditionally**: folded into the `acquire_run_leadership` transaction (§B.4). The epoch bump fences the old leader everywhere immediately *and* its registry row is evicted in the same commit, so the membership fences refuse it from the same instant; its item leases reap as they expire under §A.5.

The evicted-but-alive worker discovers eviction at its next heartbeat tick (CAS miss ⇒ flag ⇒ `RunWorkerEvictedError` at the next node boundary ⇒ abandon in-flight work without emitting, the existing lease-lost discipline) — and independently at every fenced verb below, so the flag is an optimization, not the fence.

### C.3 Leader failover = resume-as-takeover (no in-place promotion)
When the leader dies, followers drain what is claimable, then idle/exit per §B.1.5. Recovery is a new `elspeth resume`: seat CAS, then the **existing** resume reconstruction — `_restore_barriers_from_journal` over BLOCKED rows, trigger-latch/`lost_branches` scalars from the latest checkpoint under ADR-029 D3 staleness rules (with `lost_branches` now re-derived primarily from the durable loss table, §E.5), checkpoint `rebase_sequence(MAX(sequence_number))` read from the DB (replacing trust in the in-memory counter), pending-sink recovery, source-position recovery. Why not promote a follower in place: a follower has no source plugins, no ingest state, no checkpoint coordinator, no barrier memory; promotion would re-construct, inside a live process, everything resume already constructs correctly from durable state — double machinery, worse audit story. A follower *may* be the process that runs the takeover, but it does so by exiting its loop and re-entering through `resume()`.

### C.4 Fencing the suspended-but-alive evicted worker — closed by construction

This is the ticket's named unprotected surface. Principle: **two-level fencing matching two-level state.** Item-scoped writes keep item leases (already sound). Every RUN-scoped write verb carries `(leader_worker_id, leader_epoch)` and fences via a **verify-and-extend UPDATE CAS** inside the **same `BEGIN IMMEDIATE` transaction** as the payload write (graft, Design 2):

```sql
-- first statement of every leader-fenced transaction:
UPDATE run_coordination
   SET leader_heartbeat_expires_at = :now + :window, updated_at = :now
 WHERE run_id = :run
   AND leader_worker_id = :wid
   AND leader_epoch     = :epoch;
-- rowcount 1 required; 0 => ROLLBACK, write 'fence_refusal' event on a FRESH connection (best-effort,
-- §A.2), raise RunLeadershipLostError (Tier-2). Nothing else committed.
```

This form wins on three axes: (1) under `BEGIN IMMEDIATE` the verify and the payload are one atomic unit — no peer can swap the seat between them; (2) every fenced verb doubles as a seat heartbeat, so a leader doing a long burst of fenced writes never spuriously loses its seat between thread ticks; (3) it is the exact form that ports to Postgres — a conditional UPDATE takes a row lock that survives READ COMMITTED, where a bare EXISTS subquery's snapshot does not (§F).

The **follower/membership fence** is a shared predicate compiled into the same statement as the guarded write:

```python
def active_worker_fence_clause(worker_id_param) -> ColumnElement[bool]:
    """AND EXISTS (SELECT 1 FROM run_workers
                    WHERE worker_id = :caller AND run_id = :run AND status = 'active')"""
```

Implemented as **one shared SQLAlchemy construct in the schema module with its own dedicated unit test** — the `blocked_barrier_hold_clause` hygiene pattern (verified, `schema.py:498-517`) — because the fence appears in ~10 verbs and per-verb hand-rolled copies will drift (graft, Design 1). The leader-epoch verify-UPDATE is likewise emitted by one shared helper in the coordination repository.

**Enumeration of every write surface a suspended-then-evicted worker can touch, and its fence** (rows 9 and 10 corrected, and row 6a added, after review):

| # | Write surface | Fence | Mechanism |
|---|---|---|---|
| 1 | Item journal transitions: `heartbeat_lease`, `mark_*` via `_transition` | **item lease (existing, unchanged)** | reaped lease ⇒ CAS miss ⇒ `SchedulerLeaseLostError` / loud refusal; drain abandons without emitting (`processor.py:3361-3374` discipline). *Unreaped, unexpired* lease ⇒ the write **succeeds, and that is correct** — the new leader cannot have claimed that item (it is LEASED), so exactly-once holds and the audit stays coherent. Deliberately unchanged. The coalesce-loss record (§E.5) rides the same lease-fenced transaction. |
| 2 | **Fresh claims**: `claim_ready` (`:608`) / `claim_pending_sink` (`:724`) | **membership (NEW — graft, Design 1)** | `active_worker_fence_clause` compiled into the claim CAS UPDATE. An evicted-but-alive worker cannot claim new work even inside the ≤15s window before its heartbeat flag latches. Takeover now evicts the deposed leader by identity in the same commit as the epoch bump (§B.4), so this fence is live against the deposed leader from the instant of takeover — no skew window. |
| 3 | **Fresh enqueues**: `enqueue_ready` (`:120`) child continuations | **membership (NEW — graft, Design 1)** | INSERT…SELECT WHERE EXISTS membership; rowcount 0 ⇒ re-read membership ⇒ `RunWorkerEvictedError`. |
| 4 | `complete_run` / `update_run_status` / `finalize_run` (verified `run_lifecycle_repository.py:286-354, 901-950, 1195`) | **epoch** | verify-UPDATE joins the existing conditional-UPDATE transaction (which already refuses terminal overwrite). An evicted leader can neither finalize COMPLETED out from under the new one nor stamp FAILED over its progress (the failure ceremony, `resume.py:895-901`, is fenced too — a deposed leader skips the ceremony; the run is no longer its to fail). Immutability predicates stay beneath as the durable backstop. |
| 5 | Checkpoint writes (`CheckpointManager.save_checkpoint`) | **epoch** + existing UNIQUE(run_id, sequence_number) backstop (`manager.py:110-119`) | verify-UPDATE inside the checkpoint transaction; the deposed leader's INSERT is refused before the unique constraint is even reached; a deposed leader can never poison the scalars the next leader restores from. |
| 6 | `complete_barrier` (verified `:1255`) | **epoch** | verify-UPDATE joins its single atomic transaction, refusing before any journal mutation. The bidirectional exhaustiveness validation stays as the loud split-brain backstop; the intake-snapshot amendment (§E.3) governs late arrivals. |
| **6a** | **Barrier adoption (NEW VERB): batch_members + BUFFERED `token_outcomes` + executor accept bookkeeping** | **epoch + per-row adoption CAS** | The earlier draft left these as bare audit inserts under row 10 — wrongly: `token_outcomes` has NO uniqueness for non-terminal rows (the partial unique index covers `completed=1` only, verified `schema.py:392-399`; `record_token_outcome` is a plain INSERT, `data_flow_repository.py:1198-1220`), the BLOCKED token holds no lease, and a deposed leader's stale intake could write a second live BUFFERED outcome — over-counting `rows_buffered` in what §D makes the sole derivation, and poisoning `_derive_restored_batch_id` on the next restore. Adoption is therefore a **leader-fenced journal verb** `adopt_blocked_barrier_item`: one IMMEDIATE transaction = (1) verify-and-extend epoch fence, (2) CAS adoption marker on the BLOCKED row (`token_work_items.barrier_adopted_epoch NULL → :epoch`; rowcount 0 with the marker already at this epoch ⇒ idempotent skip), (3) batch_members + BUFFERED token_outcome inserts. A stale leader rolls back at (1); a re-entrant live leader no-ops at (2); double-BUFFERED is structurally impossible; and the old separate-transactions crash window (membership committed, BUFFERED not) **cannot occur at epoch 21** because the two inserts share one transaction. Defense-in-depth: restore raises a loud Tier-1 diagnostic if it ever observes >1 live non-terminal BUFFERED outcome for a token, instead of silently taking the most recent. |
| 7 | Sink writes (external side effects) + PENDING_SINK terminalization | **epoch at the ledger; bounded at the wire** | leader holds PENDING_SINK rows as LEASED via `claim_pending_sink` before writing; verify-UPDATE fence at sink-phase entry and per terminalization batch; `mark_pending_sink_terminal[_many]` **tightened to strict CAS** — `expected_lease_owner` becomes required and the NULL-owner acceptance is removed. Residual: a suspension landing *after* the fence check and *during* the external write can duplicate at most one in-flight batch — irreducible for non-transactional side effects; refused at the ledger, attributed via `fence_refusal` + LEASE_LOST events. Same class as the pre-existing N=1 crash-after-sink-write-before-terminalize window, now bounded and audited. (The bound holds because identity-eviction at takeover makes `claim_pending_sink`'s membership fence live immediately — the deposed leader cannot claim *further* PENDING_SINK rows into fresh batches.) |
| 8 | `terminalize_pending_sinks_with_terminal_outcomes` repair sweep (verified unfenced today, `:1953`) and `recover_expired_leases` | **epoch** | leader-only verbs; both gain the verify-UPDATE. A deposed leader cannot rotate attempts under the new one. |
| 9 | **Source ingest: rows insert + tokens insert + initial `enqueue_ready_claimed`** | **epoch, on the WHOLE ingest transaction (corrected)** | The earlier draft fenced only the enqueue and called the `rows` UNIQUE constraints "uncontested" — false: `create_row_with_token` commits in its OWN transaction (verified, `data_flow_repository.py:599-650`) before the enqueue, so a woken deposed leader could durably insert an orphan `rows` row at sequence N and the LIVE leader would then crash on UNIQUE(run_id, ingest_sequence) when its recovered iteration reached N — a repeatable wedge, with the orphan row (on this multi-source branch, potentially a *different* row than the legitimate lineage would have produced at N) then driven into the run by resume coverage. **Fix: the leader ingest step is ONE fenced `BEGIN IMMEDIATE` transaction** — verify-and-extend epoch fence first, then rows insert + tokens insert + initial `enqueue_ready_claimed` (a coordination-aware ingest verb composing the existing repository helpers on one connection). A deposed leader's ingest rolls back atomically before any `rows` row is durable; the UNIQUE constraints become true backstops. Any ingest-adjacent durable write that a takeover leader's source-position recovery reads (run_sources position state) rides this fenced transaction or the fenced checkpoint (row 5); run_sources registration rows are write-once at run start and re-verified, never re-written, on takeover. The woken-mid-ingest interleaving joins the §H deterministic suite. |
| 10 | Audit writes mid-token (node_states / token_outcomes / calls) **under a valid item lease** | **attempt identity (existing)** | deliberately permitted: they record execution that genuinely happened under the superseded attempt; re-drives run at attempt+1 after rotation, so UNIQUE(token_id, node_id, attempt) on node_states (`schema.py:632-633`) makes same-identity collisions impossible. Honest history, not corruption. **Scope corrected:** this row covers only writes made while traversing a validly-leased item. Leaseless audit writes from the barrier plane are row 6a; terminal outcomes remain gated behind fenced verbs. |
| 11 | Payload-store writes | **content addressing (existing)** | sha256-addressed files; duplicate writes are byte-identical no-ops. |
| 12 | Registry writes (own heartbeat) | **worker-row CAS** | first post-wake heartbeat misses ⇒ flag ⇒ clean abort; usually fires before rows 2–9 are attempted. |
| 13 | JSONL change journal (`dump_to_jsonl`, `database.py:372-374` → `journal.py`) | **per-worker paths (graft, Design 2)** | each worker appends to its own derived file `<db>.journal.<uuid-hex>.jsonl` (the worker_id's uuid suffix; full worker_id recorded in a header line). The file-corruption half of the shared-file problem dissolves. **The ordering half does not** (corrected): records carry per-statement `timestamp` stamps (verified, `journal.py:39,121`) buffered to commit, and statement-time order is not WAL commit order across processes — so at N>1 the JSONL journal is **forensic-only, not a replayable backup**; ADR-030 and the journal docs say so explicitly, and any restore tooling gates on worker-count=1 provenance. (A true total order via an in-transaction `journal_seq` counter is noted as future work, not 0.6.0.) |

The suspended winner therefore cannot make **any** durable state transition the new leader's epoch does not authorize — including barrier adoption (6a) and ingest (9), the two surfaces the adversarial review showed the earlier draft left open — and cannot claim or enqueue fresh work at all; its only residual capabilities are finishing item work it still validly leases (correct, row 1) and at most one in-flight external sink batch (bounded, ledger-refused, audit-attributed, row 7). Every fence is a DB CAS against an injectable stale token, so the scenario the ticket says "cannot be driven deterministically in-process" becomes an ordinary deterministic test (§H).

---

## D. Finalization arbitration

**Who:** the current leader, exclusively. **When:** the leader's end-of-run sequence, in order:

1. **Source EOF** (the leader owns ingest, so EOF is its fact).
2. **Upstream quiescence before any end-of-input flush** (corrected after review — the earlier draft never ordered flush after in-flight work, so an N=2 slow follower mid-LLM at EOF would have its arrival classified "late" by §E.3 with no next batch ever coming, wedging the run and nondeterministically splitting batches across a FAILED/resume cycle): the end-of-input flush may run **only when the journal shows zero READY and zero LEASED work items for the run** — DB-derivable precisely because followers hold no run-long memory; any in-flight follower work IS a journal row, and the leader waits for it (claiming what it can itself in the meantime).
3. **Flush loop:** intake/adoption → trigger evaluation → flush, repeated **until the BLOCKED set is empty** (late-branch rejections are journal-released inside the loop, §E.3a). Because step 2 guaranteed nothing can newly `mark_blocked`, the loop terminates.
4. **PENDING_SINK drained** (claim → sink write → strict terminalize).
5. **The terminal statement** — quiescence and fencing move INTO the terminal UPDATE, under `BEGIN IMMEDIATE` so no peer can claim, enqueue, or join between check and flip:

```sql
BEGIN IMMEDIATE;
-- (i) verify-and-extend seat fence (C.4) — rowcount must be 1
UPDATE runs SET status = :terminal, completed_at = :now,
       reproducibility_grade = COALESCE(:grade, reproducibility_grade)
 WHERE run_id = :run
   AND status NOT IN (<terminal statuses>)                       -- existing immutability arm (verified :333-349)
   AND NOT EXISTS (SELECT 1 FROM token_work_items t
                   WHERE t.run_id = :run
                     AND t.status IN ('ready','leased','blocked','pending_sink'));  -- quiescence
UPDATE run_workers SET status='departed', departed_at=:now
 WHERE run_id = :run AND status='active' AND role='follower';    -- leftover-member hygiene (§B.1; evented)
INSERT INTO run_coordination_events ('finalize', ...);
COMMIT;
```

rowcount-0 diagnosis order: already-terminal ⇒ `AuditIntegrityError` (immutability contract unchanged); fence mismatch ⇒ `RunLeadershipLostError` (+ `fence_refusal` event); residual work ⇒ `OrchestrationInvariantError` (the existing has-unresolved-work invariant, now enforced *inside* the arbitration statement). The quiescence predicate is deliberately **stricter than `count_unresolved_work`**, whose `_unresolved_work_predicate` excludes PENDING_SINK and LEASED-with-pending_sink_name (verified, `scheduler_repository.py:2171-2191`): at finalize time the sink phase is complete and surviving PENDING_SINK is a bug worth refusing on. The quiescence predicate applies to success statuses; FAILED/INTERRUPTED ceremony finalization checks only fence + immutability (the journal is left intact for resume). A truly orphaned BLOCKED row blocks COMPLETED finalization here — orphan detection moves from per-flush to finalize-time without weakening — and the §E.3a release rule ensures the design itself never *manufactures* such an orphan.

**Terminal-status derivation:** multi-worker runs always derive terminal status and counters from the audit trail (`derive_resume_terminal_status_from_audit` becomes the only derivation); the normal path's in-memory `ExecutionCounters` derivation is demoted to a cross-check assertion — under N>1 no single process's memory tallies the whole run. This finishes the single-bookkeeper unification (elspeth-7294de558e). (The audit-derived `rows_buffered` counter is safe against double-BUFFERED inflation because adoption is a CAS-fenced verb, §C.4 row 6a; the restore-time >1-live-BUFFERED diagnostic is the backstop.)

---

## E. Barrier ownership under multi-worker

### E.1 Decision: leader-owned barrier plane, journal-arrival hand-off, no barrier-group leases
`complete_barrier`'s exhaustiveness contract — every BLOCKED row under `(run_id, barrier_key[, scope_row_id])` consumed or handed off, validated in both directions in one transaction (verified, `scheduler_repository.py:1421-1430, 1456-1469`) — already forces a single flusher with the complete picture, and ADR-029 D3 deliberately keeps trigger latches as one worker's checkpointed scalars. Rather than lease barrier groups to arbitrary workers (shattering the single-owner story and multiplying the checkpoint-scalars problem by N), **all barrier memory, trigger evaluation, and flushing live in the leader**. The exhaustive-release CAS is retained as defense-in-depth; ownership is structural (only the leader constructs barrier executors) plus the epoch fence on `complete_barrier` and on the new adoption verb (§C.4 rows 6/6a), which is what protects against the split-brains structure cannot prevent — a deposed leader flushing, or adopting, late. ADR-029 Alternative 1's `barrier_state` table remains available to a future distributed-barriers ADR; we do not take it.

### E.2 Arrival hand-off, unified: journal-first for every worker, the in-claim arm REMOVED
New traversal rule at a barrier node **for every worker including the leader**: do not call `executor.accept`; transition the leased item `LEASED → BLOCKED` with the derived `barrier_key` (aggregation: `str(node_id)`; coalesce: `coalesce_name` — the existing dual-use predicate, `blocked_barrier_hold_clause`) and end the traversal. `mark_blocked` stamps `barrier_blocked_at` at durable arrival (verified, `:1127-1131`).

**The contradiction the review caught, resolved:** the earlier draft said "journal-first for every worker" while simultaneously preserving a leader-side synchronous in-claim flush arm (today's count-trigger fire inside the claim, `processor.py:1841-1918`, coalesce `:2796-2830, :3164-3181`) via `leased_exclusion_token_id`. Both could not be true — the retained arm would (a) trip §E.3's snapshot algebra with a designed-in Tier-1 on every in-claim flush (the leased trigger token is in memory but has no BLOCKED row) and (b) falsify the D7-closure claim (BUFFERED committed before a seconds-long flush transform = the exact membership-without-BLOCKED crash window D7 refuses, now hit routinely at takeover). **Decision: the in-claim arm is removed entirely; §E.2 is literally true.** The triggering token is marked BLOCKED like any other arrival; the same drain iteration's intake adopts it (one fenced verb, §C.4 row 6a) and trigger evaluation fires the flush out-of-claim through `complete_barrier`, with flush output enqueued via the barrier's continuation path. `leased_exclusion_token_id` is deleted from `complete_barrier` (it has no remaining caller). **This is an owned N=1 behavior change**, not "bit-identical" as previously claimed: batch composition and audit identity are unchanged, but the count-trigger flush moves from inside the claim to the same drain iteration immediately after it (one extra BLOCKED→consumed journal transition per triggering token, ~ms latency); the affected tests are re-pinned in slice 3. What we buy: the D7 crash window genuinely closes on every path — in-memory accept never precedes the durable BLOCKED row anywhere — and the snapshot algebra needs no leased-token exemption.

The leader runs the **intake/adoption step on every drain iteration, and ALWAYS immediately before any trigger evaluation or flush** (supersedes the winner's 64-iteration maintenance-sweep cadence): list BLOCKED barrier holds (`list_blocked_barrier_items`, read-only, verified `:2142`), and for each row not yet adopted (`barrier_adopted_epoch IS NULL`) run the fenced adoption verb (§C.4 row 6a): rehydrate via `token_from_journal_item` (verified, `:45`), perform the in-memory accept, and durably record batch membership + the BUFFERED token_outcome — all in one epoch-fenced transaction with a per-row adoption CAS. Trigger evaluation (count/condition/timeout) runs only over post-intake memory — condition triggers therefore evaluate real rehydrated token content, not SQL aggregates.

**Adoption is backdated (corrected after review — the earlier draft claimed "timeout math is immune to intake latency because it reads durable `barrier_blocked_at`", which is true only of the RESTORE path; the LIVE path anchors `TriggerEvaluator._first_accept_time` and coalesce `first_arrival` to `clock.monotonic()` at accept time, verified `triggers.py:110-118` and `coalesce_executor.py:703`, i.e. to INTAKE time under this design — so a takeover would re-anchor the same trigger to a different clock and change batch composition as a function of whether a crash happened).** The adoption path passes the row's durable `barrier_blocked_at` into the executor accept, converted wall→monotonic with the exact clamped transform the restore code already uses (verified: coalesce `coalesce_executor.py:425-450`, aggregation `executors/aggregation.py:765-780`). `_first_accept_time` / `first_arrival` and the count/condition fire latches are thereby anchored to durable arrival in BOTH the live and the restore frame; a batch's timeout fire time is invariant under leader takeover at any point between `mark_blocked` and flush, and a §H test pins that invariance. (N=1 note: backdating by the few-ms gap between accept and the old in-claim `mark_blocked` is the same conservatism ADR-029 D2 already accepted, in the same direction.)

**Journal-first acceptance closes the D7 window — now on every path.** Under arrival-first the invariant becomes `journal-BLOCKED ⊇ batch_members`: a BLOCKED row without membership is **intake-pending** (`barrier_adopted_epoch IS NULL` — legitimate; a restore-reconcile disposition); membership without a BLOCKED row remains Tier-1 corruption — and at epoch 21 it is structurally unreachable, because membership and the BLOCKED row's adoption marker commit in one transaction (§C.4 row 6a) and the in-claim arm that used to commit BUFFERED before a long flush no longer exists.

### E.3 The late-arrival race, fixed honestly (MANDATORY graft) — snapshot scoped per firing group
At N≥2 a follower's `mark_blocked` can commit between the leader's intake and its `complete_barrier`. The verified exhaustive-release arm (`:1456-1469`) computes `uncovered = durable_BLOCKED − consumed − handed_off` and raises Tier-1 `AuditIntegrityError` on any remainder — a **designed-in false alarm** at N=2, since the flush transform takes seconds. The fix is an explicit amendment to `complete_barrier`'s contract (shipped as an ADR-029 amendment alongside the D7 polarity flip):

```python
def complete_barrier(self, *, run_id, barrier_key, consumed_token_ids, emitted_pending_sink,
                     emitted_ready, now, scope_row_id=None,
                     require_exhaustive_release=True,
                     intake_snapshot_token_ids: frozenset[str] | None = None,   # NEW
                     leader_worker_id: str, leader_epoch: int) -> int:          # NEW (fence)
                     # leased_exclusion_token_id DELETED — no in-claim arm remains (§E.2)
```

**The snapshot is per-firing-group, by contract (corrected after review):** `intake_snapshot_token_ids` is the set of token_ids the leader has adopted into **the (barrier_key, scope_row_id) group being flushed** — NOT the executor's whole memory. A coalesce executor's `_pending` is keyed `(coalesce_name, row_id)` in one shared memory (verified, `coalesce_executor.py:209`), and `complete_barrier`'s durable universe is scoped by `scope_row_id` (verified, `:1400-1401`); passing whole-memory snapshots would put every *other* pending row group's tokens into `snapshot − durable_BLOCKED` and raise a spurious Tier-1 on every healthy multi-row coalesce flush. As defense against caller bugs, `complete_barrier` additionally validates that every snapshot token's durable row carries the flush's `scope_row_id` (mismatch ⇒ Tier-1 naming the caller, not silent intersection). The exhaustiveness universe becomes `durable_BLOCKED ∩ snapshot`:
- `snapshot − durable_BLOCKED` non-empty ⇒ Tier-1 (the existing missing-token arm, `:1421-1430`, generalizes — the leader believes in a token the journal does not hold);
- `(durable_BLOCKED ∩ snapshot) − consumed − handed_off` non-empty ⇒ Tier-1 (true orphaning within the snapshot, the original invariant);
- `durable_BLOCKED − snapshot` = **late arrivals** ⇒ legitimately stay BLOCKED, dispositioned by the next intake (§E.3a); recorded in the completion event context as `late_arrival_token_ids` for forensics.

This is set-identity-based, not timestamp-based, so it is deterministic and immune to commit-vs-stamp ordering ambiguity. Single-worker callers pass the full firing group (their memory is complete because intake precedes every evaluation), so the N=1 algebra never has a non-empty late set. Safety is not weakened: any BLOCKED row that never gets dispositioned blocks COMPLETED finalization via §D's quiescence predicate, so orphan detection is preserved at the run boundary.

### E.3a Late-arrival dispositions — every late row gets journal-released (NEW after review)
"Stays BLOCKED for the next batch" is only a complete answer for aggregation. The full disposition table, owned explicitly:
- **Aggregation:** the next intake adopts the late row; the next trigger evaluation fires it into the next batch. At end-of-input this is guaranteed to happen because §D step 2 forbids the EOF flush while any READY/LEASED work survives, and §D step 3 loops intake→evaluate→flush until BLOCKED is empty.
- **Coalesce, group still pending:** the next intake adopts it as an ordinary branch arrival.
- **Coalesce, group already completed (`(coalesce_name, row_id) ∈ _completed_keys`):** the in-memory rejection (verified, `coalesce_executor.py:209-217`) is **not enough** — it would strand a durable BLOCKED row that §D's quiescence predicate then refuses to finalize over, forever, on a routine slow-follower-past-timeout interleaving. **Rule: when intake adoption hits a completed key (or any executor-level rejection), the leader transitions that BLOCKED row to a terminal state in the same drain iteration via the existing partial-release arm `mark_blocked_barrier_terminal` (verified present, `scheduler_repository.py:2027`) with a `late_arrival` context**, recording the rejection in the audit trail so the journal converges before finalize. The same reconcile runs at takeover restore (§E.4) — a new leader can inherit an unreleased late-branch row. A slow-follower-past-timeout test pins this in the §H suite.

### E.4 When the evaluating worker dies
Exactly F1 resume semantics: the takeover leader rebuilds buffers from BLOCKED rows (`_restore_barriers_from_journal`) and restores trigger-latch scalars from the latest checkpoint under ADR-029 D3 staleness rules; `lost_branches` is re-derived from the durable loss table (§E.5), with the checkpoint scalar retained as a cross-check. The restore reconcile now has these dispositions: adopted rows (`barrier_adopted_epoch` set — membership + BUFFERED exist atomically, restorable as members), intake-pending rows (`barrier_adopted_epoch IS NULL` — re-adopted by the new leader's first intake), unreplayed loss records (`adopted_epoch IS NULL` in `coalesce_branch_losses` — replayed through `notify_branch_lost`), and inherited late-branch rows against completed keys (journal-released per §E.3a). Membership-without-BLOCKED remains Tier-1; >1 live non-terminal BUFFERED outcome per token is a new loud Tier-1 diagnostic (§C.4 row 6a). Checkpoint writes are seat-fenced (§C.4 row 5), so a deposed leader can never poison the scalars the next leader restores from — and adoption fencing (row 6a) now extends the same guarantee to the audit rows restore reads for batch derivation. The pre-existing batches-row-COMPLETED-before-complete_barrier residual (elspeth-3977d8ab60) keeps its existing loud-refusal semantics and is not widened.

### E.5 Branch-loss hand-off — the second arrival kind (NEW after review)
Coalesce trigger evaluation depends on TWO inputs, and the earlier draft handed off only one. Arrivals reach the leader as BLOCKED rows; **branch LOSSES did not reach the leader at all**: today the worker that fails/quarantines/error-routes a forked token synchronously calls `_notify_coalesce_of_lost_branch` (verified: definition `processor.py:2754`, call sites `:1451, :4034, :4082, :4187, :4218, :4293, :4327`), which may immediately fire the merge or fail the whole group (`failure_reason` arm `:2832-2837`). A follower has no coalesce executor, so under the unrepaired design a follower-failed branch would be invisible to the leader forever — must-fail groups silently surviving to a timeout merge, `branches_lost` audit records silently wrong, no-timeout groups stalling to EOF, and checkpointed `lost_branches` scalars structurally incomplete for every takeover. The §D claim "followers hold no run-long memory" was true but inverted — the *leader's* memory was not fully DB-derivable once losses happened on followers.

**Fix — durable loss records with the same journal-first discipline as arrivals:**
- When a fork-lineage token whose `branch_name` maps to a coalesce (`_branch_to_coalesce`) reaches a lossy disposition on **any** worker, a row is inserted into `coalesce_branch_losses` (schema §A.2) **in the same lease-fenced transaction** as the `mark_failed`/divert. Uniqueness on `(run_id, coalesce_name, row_id, branch_name)` makes re-drives idempotent.
- The leader's intake step adopts unreplayed loss records (`adopted_epoch IS NULL → :epoch`, same CAS-marker pattern as row adoption, inside the same fenced intake pass) alongside BLOCKED rows, replaying each through `CoalesceExecutor.notify_branch_lost` **before** trigger evaluation — so must-fail policies fire, best-effort merges carry correct `branches_lost`, and group-completion latency for a lost branch is one drain iteration, not a timeout.
- Takeover restore re-derives `lost_branches` from this table (cross-checked against the D3 checkpoint scalar) and replays unadopted records — losses survive leader death by construction.
- **Uniformity rule applies:** at N=1 the leader's own lossy dispositions also write the record, then notify synchronously in the same drain step as today — record-then-notify, so N=1 observable behavior is unchanged while the durable trail exists on every path.
- The rejected alternative — restricting fork-lineage claims to the leader — would gut follower parallelism for every forking pipeline; the durable-loss record is the right shape.

---

## F. Draft ADR-030 — Multi-worker deployment shape (satisfies ADR-026 Precondition #9)

> **Status:** Draft (slice 0) → Accepted at slice 5.
> **Decision: ELSPETH multi-worker = a pack of cooperating OS processes on one host sharing one WAL SQLite audit DB, coordinated as one epoch-fenced leader plus claim-only followers.**
>
> **Context.** ADR-026 Precondition #9 blocks N>1 deployment until an ADR fixes the deployment shape, worker identity/lifecycle, and shutdown semantics (the refusal is hard-coded in `drain_scheduled_work`). F1/ADR-029 (epoch 20) made the journal the barrier-buffer truth, removing the last durable-state obstacle.
>
> **Worker lifecycle.** Workers are operator-spawned CLI processes: `elspeth run` mints the run leader (epoch 1, in `begin_run`'s transaction); `elspeth resume` mints a leader via the seat-takeover CAS (epoch+1, identity-evicting the deposed incumbent in the same commit); `elspeth join <run_id>` mints followers via atomic IMMEDIATE admission after a filesystem-writability preflight. No supervisor daemon. Follower SIGINT: finish/abandon current claim, depart, exit 0. Leader SIGINT: `checkpoint_interrupted_progress`, fenced `update_run_status(INTERRUPTED)`, `leader_release` event with the seat zeroed; followers observe via their heartbeat snapshot and depart. Recovery of any leaderless run is `elspeth resume`. **A process frozen inside an IMMEDIATE transaction holds the WAL write lock indefinitely and stalls the entire pack including the takeover CAS; the remediation is operator SIGKILL (locks release on process death), and `acquire_run_leadership` reports BUSY distinctly from CAS-loss, surfacing registered pids.**
>
> **Worker identity.** `worker:{run_id}:{uuid4().hex}` registered in `run_workers`, doubling as the scheduler `lease_owner` — extending, never reshaping, the opaque-owner-string scheme ADR-026 requires. Identities are single-use: `departed`/`evicted` rows never return to `active`. `pending_items` remain per-process caches always backed by durable rows (unchanged invariant).
>
> **SQLite/WAL reasoning.** WAL's reader/writer coordination runs through a shared-memory `-shm` file that every connecting process must mmap — a one-host mechanism; SQLite documents WAL as unsafe over network filesystems. One host also gives one clock (liveness windows and `barrier_blocked_at` timeout math compare absolute wall-clock timestamps) and one local payload store. Every process opens through `LandscapeDB` and inherits the probe-enforced PRAGMAs (G28, now a stated cross-process invariant). Engine-side **write** transactions use `BEGIN IMMEDIATE` via a dialect-keyed `do_begin` listener keyed on an explicit per-connection write-intent execution option carried only by scheduler/coordination/lifecycle/checkpoint write verbs — never keyed on "writer-mode engine"; **web and dashboard read paths open `read_only=True`** so reads provably stay outside the write-lock domain and WAL's readers-don't-block-writers property actually holds. This eliminates cross-process `SQLITE_BUSY_SNAPSHOT` aborts on SELECT-then-UPDATE shapes that `busy_timeout` cannot retry and closes the cross-process half of ADR-026 Precondition #4 (G27); the stale "Required" marker in ADR-026 is reconciled by this ADR. Throughput sanity: the workload is LLM-dominated (seconds per row, ~5–10 sub-millisecond writes per item); ten workers ≈ tens of writes/second, two orders of magnitude under WAL's single-writer capacity — but `busy_timeout` is a polling retry with no fairness, so the run-liveness window is sized against worst-case write-lock occupancy (window ≥ 4 × (beat + busy_timeout)) and the slice-1 contention test measures max hold time, including concurrent dashboard-style reads.
>
> **Filesystem/permissions (operator requirement).** All pack members need write access to the DB file, its directory, and the `-wal`/`-shm` sidecars (created with the creating process's umask). The web-hosted-leader + CLI-follower shape requires a shared group with a group-writable state directory (or followers running as the service user); SQLCipher followers must hold the passphrase. `elspeth join` preflights writability and refuses actionably. Containerized workers must share the host clock.
>
> **Postgres port form (recorded now, enabled never-in-0.6.0).** Every fence is a plain conditional UPDATE or a shared EXISTS fragment, so the schema and verbs port unchanged. The load-bearing note: the dialect port of the epoch fence is the **verify-and-extend conditional UPDATE on `run_coordination` inside the same transaction as the payload write** — an UPDATE row-lock survives Postgres READ COMMITTED where a bare EXISTS subquery's snapshot does not. The membership EXISTS fences likewise port by re-expressing as verify-UPDATEs, and `BEGIN IMMEDIATE` maps to ordinary transactions with `SELECT … FOR UPDATE` where a read precedes its write. Liveness comparisons must move to DB-server time. Enablement requires its own ADR plus a runtime Landscape-on-Postgres test campaign (today: zero runtime coverage; epoch mechanism SQLite-only; SQLCipher deployments SQLite-locked).
>
> **Supported:** N processes (recommended 2–8), one host, one audit DB in WAL; CLI leader + CLI followers; web-hosted leader + CLI followers (subject to the filesystem requirement above); JSONL change journal at N>1 via per-worker journal files `<db>.journal.<uuid-hex>.jsonl` — **forensic-only at N>1** (per-statement timestamps cannot reconstruct cross-process commit order; restore tooling gates on worker-count=1 provenance).
> **NOT supported (refused, not deferred-silently):** multi-host anything; network/NFS-mounted audit DBs; Postgres at runtime; uvicorn/gunicorn web workers >1 (the `app.py` refusal stands); in-process row-level thread workers (`RowProcessor` state plane is documented single-threaded); concurrent or sharded source ingestion (needs its own ADR per ADR-026); follower auto-promotion.
> **Reconciles stale text:** ADR-026 G27 "Required" marker (closed by f79332aa8 + this ADR's IMMEDIATE discipline); the nonexistent `waiting`/`mark_waiting` state named in ADR-026 prose and the lease-recovery runbook (absent from `contracts/scheduler.py` — `available_at` is the delayed-retry mechanism); the G19 runbook is rewritten for N>1 in slice 6 (including the kill-the-wedged-incumbent step).

(The standalone ADR file ships as `docs/architecture/adr/030-multi-worker-deployment-shape.md`; full text in the companion artifact.)

---

## G. Schema changes (epoch 20 → 21) and verb-signature changes

Epoch bump `SQLITE_SCHEMA_EPOCH = 21` (verified currently 20 at `schema.py:104`); operators delete old DBs per standing policy — no migration. The `_REQUIRED_COLUMNS`/`_REQUIRED_*_KEYS`/`_REQUIRED_CHECK_CONSTRAINTS`/`_REQUIRED_INDEXES` compatibility maps in `database.py` are extended for the new tables and column. **Unchanged tables:** `scheduler_events` (item identity untouched; owner strings extend), `checkpoints` (format_version stays 5), `runs` (coordination state deliberately kept OUT of the audit-bearing legal record), `node_states`, `batches`, `batch_members`.

**Changed table:** `token_work_items` gains one nullable column, `barrier_adopted_epoch INTEGER` (the adoption CAS marker, §C.4 row 6a — written only by the fenced adoption verb; NULL means intake-pending). Item identity, lease columns, and event semantics untouched.

**New tables:** `run_coordination`, `run_workers`, `run_coordination_events` (with `seq` AUTOINCREMENT replay ordinal and the `worker_stalled`/`heartbeat_degraded` event types), `coalesce_branch_losses` (§A.2, §E.5).

**New shared constructs (one definition, one dedicated unit test each):** `active_worker_fence_clause()` (schema module, sibling of `blocked_barrier_hold_clause`); the leader verify-and-extend fence helper (coordination repository).

**New repository** `RunCoordinationRepository` (same engine, same constructor PRAGMA probe as the scheduler repository):

```python
def register_run_leader(self, *, run_id, worker_id, now, window_seconds) -> CoordinationToken   # begin_run, epoch 1
def acquire_run_leadership(self, *, run_id, worker_id, now, window_seconds) -> CoordinationToken # §B.4 CAS; identity-evicts
                                                                                                 # the incumbent; raises
                                                                                                 # NonResumableRunError on CAS-loss,
                                                                                                 # WriteLockHeldError on BUSY (with pids)
def release_seat(self, *, token, now) -> None                                                    # graceful leader shutdown
def admit_follower(self, *, run_id, worker_id, config_hash, now, window_seconds) -> None         # §B.1 atomic IMMEDIATE admission
                                                                                                 # (after filesystem preflight);
                                                                                                 # raises JoinRefusedError
def worker_heartbeat(self, *, worker_id, now, window_seconds) -> CoordinationSnapshot            # CAS + snapshot; leader beats BOTH
                                                                                                 # rows in one txn; BUSY = liveness
                                                                                                 # unknown, never eviction (§A.3)
def depart_worker(self, *, worker_id, now) -> None
def evict_worker(self, *, token, target_worker_id, now, grace_seconds) -> bool                   # §C.2 path 1: grace + no-unexpired-
                                                                                                 # leases precondition, same txn
def live_leader(self, *, run_id, now) -> LeaderInfo | None                                       # read-only; entry guard
def record_fence_refusal(self, *, run_id, worker_id, leader_epoch, verb, now) -> None            # fresh connection; best-effort
def record_heartbeat_degraded(self, *, run_id, worker_id, failures, now) -> None                 # fresh connection; best-effort
```

`CoordinationToken(run_id, worker_id, leader_epoch)` is a frozen dataclass threaded into every leader-fenced verb.

**Changed signatures (keyword-only additions; N=1 callers thread the token minted at startup, so single-worker behavior is identical except where §E.2 owns the change):**

| Verb | Change |
|---|---|
| `TokenSchedulerRepository.claim_ready` / `claim_pending_sink` | claim CAS UPDATE gains `active_worker_fence_clause` (caller = `lease_owner`) |
| `enqueue_ready` | INSERT…SELECT WHERE EXISTS membership |
| **leader ingest verb (NEW)** | one fenced IMMEDIATE transaction = epoch verify-UPDATE + rows insert + tokens insert + initial `enqueue_ready_claimed` (composes `create_row_with_token` + enqueue on one connection; §C.4 row 9) |
| **`adopt_blocked_barrier_item` (NEW)** | epoch verify-UPDATE + `barrier_adopted_epoch` CAS + batch_members + BUFFERED token_outcome, one IMMEDIATE transaction, idempotent (§C.4 row 6a); backdated accept timing (§E.2) |
| **`record_coalesce_branch_loss` (NEW)** | INSERT into `coalesce_branch_losses` inside the caller's lease-fenced disposition transaction; idempotent on the natural key (§E.5) |
| `complete_barrier` | + required `leader_worker_id`/`leader_epoch` (verify-UPDATE in-txn); + `intake_snapshot_token_ids: frozenset[str] \| None`, **per-firing-group with scope validation** (§E.3); **`leased_exclusion_token_id` DELETED** (in-claim arm removed, §E.2) |
| `recover_expired_leases` | + leader token; + `liveness_view` (registry-dead owner predicate via LEFT JOIN on `run_workers`) + stall budget; arm-(c)/stall reaps write `worker_stalled` events |
| `mark_pending_sink_terminal` / `_many` | `expected_lease_owner` becomes **required**; NULL-owner acceptance removed |
| `terminalize_pending_sinks_with_terminal_outcomes` | + required leader token |
| `CheckpointManager.save_checkpoint` | + required leader token |
| `RunLifecycleRepository.complete_run` / `update_run_status` / `finalize_run` | + leader token; `complete_run` gains the in-statement NOT EXISTS quiescence predicate + leftover-follower departure (§D) |
| `RowProcessor.__init__` | + `worker_role`, + `coordination_token` (leader) / registry handle (follower); barrier executors constructed only in leader mode |
| `RowProcessor.drain_scheduled_work` | `peer_active_leases` refusal deleted; replaced by registry admission (`peer_active_leases` kept as read-only diagnostic); in-claim barrier-flush arms deleted (§E.2); EOF flush gated per §D steps 2–3 |
| `LandscapeDB` | write-intent execution option + dialect-keyed `BEGIN IMMEDIATE` `do_begin` listener keyed on it; per-worker JSONL journal path derivation; web read surfaces converted to `from_url(read_only=True)` |

**New errors (Tier-2, abandon-cleanly — siblings of `SchedulerLeaseLostError`):** `RunLeadershipLostError`, `RunWorkerEvictedError`, `JoinRefusedError`, `WriteLockHeldError` (operator-actionable, carries registered pids); seat-held refusals fold into `NonResumableRunError` reasons. Tier discipline preserved: legitimate coordination outcomes are Tier-2; journal/audit disagreement stays Tier-1 `AuditIntegrityError` (including the new >1-live-BUFFERED restore diagnostic).

---

## H. Flipping the two pinned tests (`tests/e2e/recovery/test_concurrent_resume.py`)

(Verified at this HEAD: exactly one test still carries the `test_characterization_` prefix, at :691; the mid-flight test was promoted to `test_entry_guard_refuses_resume_while_run_status_running` at :757 when option b landed. "The two tests" = these two.)

1. **`test_characterization_two_resumes_same_run_id_loser_after_winner` (:691)** — characterization dies; replaced by designed contracts:
   - *Loser-during*: the second resume is refused at `acquire_run_leadership` (rowcount 0) with `NonResumableRunError("run leadership is held by …")` BEFORE any durable write — a designed coordination outcome, not an immutability-guard side effect. The side-effect-free assertions (empty sink, no duplicate outcomes, journal untouched) carry over verbatim.
   - *Loser-after* (winner COMPLETED): refused at the entry guard with "run is terminal"; the immutability guard (`run_lifecycle_repository.py:938-950`) retained and independently pinned as the durable backstop.
   - *New deterministic suspended-winner suite* (the surface the ticket said could not be driven in-process): drive a takeover by bumping `leader_epoch` directly (the in-DB image of takeover), then call every fenced verb with the stale generation token — `complete_run`, `update_run_status`, `save_checkpoint`, `complete_barrier`, **`adopt_blocked_barrier_item`**, **the fenced ingest verb (the woken-mid-ingest interleaving: stale ingest rolls back atomically; no orphan rows row exists)**, `recover_expired_leases`, strict `mark_pending_sink_terminal_many`, the fenced repair sweep — asserting `RunLeadershipLostError`, **a `fence_refusal` event row per refusal**, and zero payload mutation. Plus the membership half: flip a worker to `evicted` and assert `claim_ready`/`claim_pending_sink`/`enqueue_ready` refuse with `RunWorkerEvictedError` and zero rows. Plus: takeover identity-eviction pinned (deposed leader's row is `evicted` even when its worker-row heartbeat is FRESH).
2. **`test_entry_guard_refuses_resume_while_run_status_running` (:757)** — splits into three:
   - (a) RUNNING + live seat ⇒ `resume()` refused before any mutation, reason naming the leader worker_id and seat expiry and directing to `elspeth join` (refusal-before-mutation assertions kept verbatim);
   - (b) new companion `test_join_attaches_follower_to_running_run`: against the same RUNNING run, `join_run()` admits a follower atomically, the follower claims a READY item under its own worker_id and completes it (terminal outcome exactly once, attempt identity clean), the leader's records untouched, the run finalizes exactly once — join-not-resume is the asserted feature;
   - (c) new companion pinning the previously-wedged state: RUNNING + *expired* seat IS resumable; the takeover CAS bumps the epoch, identity-evicts the incumbent (with `evicted_by_worker_id` + `worker_evict` event asserted, and NO follower bulk-eviction), and the run completes with the recovered row at the sink exactly once.

**Additional pinned doctrine (new and carried):**
- Uniformity: an N=1 `elspeth run` asserts the `run_coordination` row exists at epoch 1, the origin worker is registered as leader, and a normal completion writes the `finalize` event — so every existing e2e run exercises the substrate.
- **Timing invariance under takeover (§E.2):** a barrier batch's timeout fire time is invariant under leader takeover at any point between `mark_blocked` and flush.
- **Per-firing-group snapshot (§E.3):** a coalesce with two pending row groups flushes one group cleanly while the other's tokens are absent from the snapshot — and the genuine-Tier-1 arms still fire when tokens are deliberately mis-included.
- **Late-branch release (§E.3a):** slow follower marks a branch BLOCKED after its group completed ⇒ intake journal-releases it via `mark_blocked_barrier_terminal(late_arrival)`; run finalizes COMPLETED.
- **Branch-loss hand-off (§E.5):** follower-failed branch of a must-fail coalesce policy fails the group within one leader drain iteration; `branches_lost` audit record correct on best-effort merges; loss survives leader takeover.
- **EOF gating (§D):** N=2, slow follower mid-LLM at source EOF ⇒ exactly ONE batch, not two; no FAILED/resume cycle.
- **Contention (slice 1):** two-process claim hammer + dashboard-style concurrent reads ⇒ heartbeat latency stays inside the liveness window; max write-lock hold time measured against `busy_timeout`.

---

## I. Incremental landing plan (each slice independently shippable; N=1 behavior intact throughout, except the owned §E.2 flush-timing change in slice 3)

1. **Slice 0 — ADR-030 authored** (§F) + ADR-026 stale-text reconciliation (G27, `waiting`/`mark_waiting`, runbook pointer). Docs only.
2. **Slice 1 — write-intent IMMEDIATE discipline + read-only web reads** + a real two-process contention test (two processes hammering `claim_ready`/`recover_expired_leases` on one WAL file, **with concurrent dashboard-style reads, asserting heartbeat-latency headroom and measuring max write-lock hold time**). No schema change; closes the SQLITE_BUSY_SNAPSHOT hazard and the read-as-writer hazard before anything depends on either.
3. **Slice 2 — epoch 21: coordination substrate + leadership CAS + fences.** New tables + `barrier_adopted_epoch` column; `RunCoordinationRepository`; the shared fence constructs with their dedicated unit tests; `begin_run` mints the seat (uniformity rule); resume()'s first durable act becomes the seat CAS with identity-eviction (TOCTOU closes); BUSY-vs-CAS-loss discrimination; verify-and-extend epoch fences on finalize/run-status/checkpoint/complete_barrier/repair-sweep/**ingest**; strict pending-sink terminalization; `fence_refusal` eventing; audit-derived terminal status on all paths. Test #1 flips here; the deterministic stale-token suite (including woken-mid-ingest) lands here. A single worker is simply leader-of-its-own-run.
4. **Slice 3 — journal-first barrier acceptance:** in-claim flush arms removed (owned N=1 change, tests re-pinned); fenced backdated adoption verb; per-iteration intake; per-firing-group intake-snapshot exhaustiveness; late-arrival journal-release; **`coalesce_branch_losses` + loss hand-off/replay/restore**; EOF flush gating. Ships as the ADR-029 amendment covering the D7 polarity flip, §E.3, §E.3a, and §E.5 — alone, behind the full recovery characterization suite plus the new timing-invariance, cross-group-snapshot, late-branch, branch-loss, and EOF-gating tests.
5. **Slice 4 — registry liveness:** heartbeat thread (one-transaction leader beat, BUSY-tolerant semantics, `heartbeat_degraded`) + coordination snapshot (foreign-leader fatal latch); eviction protocol with the no-unexpired-leases precondition; liveness-aware + stall-budget reap with `worker_stalled` eventing; membership fences on `claim_ready`/`claim_pending_sink`/`enqueue_ready`; `peer_active_leases` demoted; entry guard learns seat liveness (RUNNING+dead-seat becomes resumable). Dead-vs-slow benefits N=1 immediately (long LLM calls stop being takeover bait); test #2 contract (c) flips here.
6. **Slice 5 — followers:** follower-mode `RowProcessor`, filesystem preflight + atomic `join_run` admission, CLI `elspeth join`, per-worker JSONL paths (forensic-only doctrine documented); test #2 contract (b) lands; G25b isolation + G25h chaos campaigns (elspeth-6116873e3b, elspeth-7bb7124e8f); ADR-030 → Accepted.
7. **Slice 6 — docs:** lease-recovery runbook rewrite for N>1 (G19, including kill-the-wedged-incumbent), operator guidance (lease sizing, N≤8, per-worker journals, shared-group/umask, shared clock), 0.6.0 stamp.

**Scope cut line:** slices 2–4 deliver the entire safety story (TOCTOU closure, takeover, suspended-winner fencing, dead-vs-slow) with N=1 behavior intact (modulo the owned slice-3 flush-timing change). If 0.6.0 must cut scope, cut Slice 5 (followers), never the fences.

---

## J. Non-goals and honest risks

**Non-goals:** multi-host; Postgres runtime support (protocol is portable in shape *and* atomicity form per §F; enablement is its own ADR + test campaign); web-managed worker pools / web `--workers>1`; parallel or sharded source ingestion; barrier work-stealing or distributed trigger evaluation; sink idempotency keys (sink plugins are heterogeneous; at-least-once-with-attribution is the 0.6.0 contract); follower auto-promotion; scheduler-row retention (pre-existing follow-up); auto-supervision of workers; a totally-ordered replayable JSONL journal at N>1 (forensic-only, §C.4 row 13).

**Risks, stated plainly:**
1. **At-least-once external sink emission across leader suspension** — bounded to one in-flight batch (the bound now actually enforced by identity-eviction making the claim fence immediate, §B.4/§C.4 row 7), refused at the ledger, attributed via `fence_refusal`/LEASE_LOST events. The honest residue of fencing non-transactional side effects with a database token; today's recovery path has the identical residue with no bound at all.
2. **Write-lock convoy** if N is large or IMMEDIATE transactions are held long — mitigated by the LLM-dominated duty cycle, short per-verb transactions, read-only web reads (slice 1), the lock-occupancy-aware liveness window (§A.3), BUSY-tolerant heartbeats with `heartbeat_degraded` eventing, and the documented N≤8 recommendation; the slice-1 contention test measures max hold time before anyone depends on it. `complete_barrier` remains one atomic transaction by design (atomicity IS the invariant); its size scales with barrier group size, so ADR-030 documents a barrier-size ceiling guidance figure derived from the measured hold time rather than chunking the transaction.
3. **Heartbeat-thread-alive / drain-wedged pathology** makes a worker look alive while its claims rot — covered by the item stall budget (§A.5), surfaced as a `worker_stalled` event.
4. **The intake-snapshot exhaustiveness amendment (§E.3) relaxes a per-flush invariant** (durable-universe coverage → snapshot-universe coverage). Compensated structurally: per-iteration intake keeps the snapshot near-complete; late arrivals are evented and journal-dispositioned (§E.3a); EOF flush is quiescence-gated (§D); and §D's finalize predicate refuses COMPLETED while any BLOCKED row survives — orphan detection is preserved at the run boundary. Ships only with the ADR-029 amendment and the full recovery suite green.
5. **Membership fences add an EXISTS probe to the hottest verbs** (`claim_ready`) — one indexed point lookup per claim against seconds-per-row work; measured in the slice-1/slice-4 contention tests.
6. **Child-continuation re-creation on re-drive** (worker enqueues children, dies before parent terminal mark) is **inherited** single-worker recovery semantics, exercised by the existing mid-claim-crash proofs (`test_concurrent_resume.py:473`); followers reuse the identical path; neither fixed nor worsened, and said so.
7. **Leader failover requires operator action** (`elspeth resume`) — a run with a dead leader stalls until then (followers idle/exit). A deliberate availability-for-explainability trade, stated in the runbook. The frozen-lock-holder variant additionally requires an operator SIGKILL before resume can proceed (§B.4) — stated in ADR-030, not hidden.
8. **Branch-loss replay latency:** a follower-discovered coalesce loss reaches the leader's memory at the next drain-iteration intake, not synchronously — a must-fail group fails one iteration later than at N=1. Bounded, evented, and identical in audit content.
9. **Scope honesty:** Slice 5 (followers) is the bulk of the engine-side work; the cut line above protects the fences.

---

## Crash-window walk: suspended winner after takeover (the ticket's named surface)

T0: Worker A leads run R at epoch 5, holds item leases (300s), heartbeat thread beating 15s into an 80s window, mid-LLM-call on token X.
T1: A is SIGSTOPped (or VM-paused, or GC-wedged). Both its heartbeats stop (they shared one transaction, so they stop together).
T2 (+≤80s): the seat expires. X's item lease is still unexpired (+≤300s).
T3: operator runs `elspeth resume R`. Entry guard sees RUNNING + expired seat ⇒ admissible. `acquire_run_leadership` (one IMMEDIATE txn): seat CAS ⇒ B at epoch 6; A's registry row `active → evicted` **by identity, unconditionally** (`evicted_by_worker_id=B`, `worker_evict` event) — fresh worker-row heartbeats cannot save a deposed leader; followers are NOT bulk-evicted; `leader_acquire` event. A racing second resume loses the CAS — `NonResumableRunError`, zero mutation (the closed TOCTOU). Had A instead been frozen *inside* an IMMEDIATE transaction, B's CAS would surface `WriteLockHeldError` naming A's pid for operator SIGKILL — stall made actionable, not silent.
T4: B runs the existing resume reconstruction (barrier restore from BLOCKED rows + adoption markers + loss-table replay + checkpoint scalars under D3; sequence rebase from MAX(sequence_number); pending-sink recovery; late-branch reconcile). B does **not** reap X yet: the reap rule requires item-lease expiry even for evicted owners (still-live-sink-writer doctrine generalized), and B cannot claim X (it is LEASED) — no double-drive exists.
T5: A wakes. Every write it can attempt:
1. `heartbeat_lease(X)` — if X expired and was reaped: CAS miss ⇒ LEASE_LOST event ⇒ `SchedulerLeaseLostError` ⇒ abandon without emitting. If X is still validly leased: the heartbeat **succeeds and A may finish X and mark_terminal under its valid lease — correct, not a hole**: B provably never claimed X; exactly-once holds. (A lossy disposition of X writes its branch-loss record in the same lease-fenced transaction — durable, idempotent, adopted by B's next intake.)
2. Run-level heartbeat — registry CAS misses (evicted, by identity, at T3) ⇒ flag ⇒ `RunWorkerEvictedError` at the next boundary. The flag is an optimization; every fence below refuses independently.
3. Fresh `claim_ready`/`claim_pending_sink` — membership fence in the claim CAS ⇒ rowcount 0 ⇒ refused, **from the instant of T3** (identity-eviction closed the heartbeat-skew window). `enqueue_ready` (child continuations outside a valid lease) — INSERT…SELECT WHERE EXISTS ⇒ zero rows.
4. Checkpoint write — verify-UPDATE (epoch 5 ≠ 6) misses in the same IMMEDIATE txn ⇒ rollback, `fence_refusal` event on a fresh connection (best-effort), `RunLeadershipLostError`; the UNIQUE constraint is an unreachable backstop.
5. `complete_barrier` — verify-UPDATE misses before any journal mutation. **Barrier adoption** — the fenced adoption verb misses the same way; A cannot write a second BUFFERED outcome or foreign batch membership (§C.4 row 6a); restore's batch derivation stays clean.
6. `finalize_run` / `complete_run` / `update_run_status` / the FAILED ceremony — verify-UPDATE misses ⇒ refused; A can neither finalize COMPLETED out from under B nor stamp FAILED over B's progress; immutability predicates beneath, untouched. **The previously unprotected surface, closed.**
7. Sink phase — A must hold LEASED PENDING_SINK rows; the per-batch verify-UPDATE refuses at phase entry; the membership fence refuses fresh PENDING_SINK claims. Worst case: A was already inside an external write at T1 and the bytes land at T5 ⇒ at most one duplicate external batch; its strict-CAS terminalization fails loudly; B re-claims and re-writes; both attempts attributed. Bounded, audited, irreducible.
8. Source ingest — A's next ingest step is ONE fenced transaction (rows + tokens + initial enqueue); the epoch fence misses as its first statement ⇒ **the rows insert rolls back with everything else** — no orphan `rows` row exists, B's recovered iteration re-inserts nothing it didn't author, and the UNIQUE(run_id, ingest_sequence) constraint is genuinely uncontested (§C.4 row 9, corrected).
9. node_states/token_outcomes at the old attempt **under A's still-valid lease** — permitted, honest history; B's re-drives run at attempt+1; UNIQUE(token_id,node_id,attempt) cannot collide. Leaseless audit writes from the barrier plane are fenced (step 5).
10. Payload store — content-addressed, idempotent. 11. `recover_expired_leases` — fenced, refused. 12. JSONL journal — A appends only to its own per-worker file (forensic-only at N>1).

Net: A's only residual capabilities are finishing still-validly-leased item work (correct) and ≤1 in-flight external sink batch (bounded, refused at the ledger, attributed). Every refusal leaves a best-effort `fence_refusal`/LEASE_LOST trail; every state mutation a deposed worker could attempt is CAS-refused. Because every fence is a DB CAS against an injectable stale token, the entire scenario is deterministically testable in-process (§H).

**Other windows considered:** dead leader with no takeover — previously wedged (guard refused RUNNING unconditionally), now resumable; leader frozen holding the write lock — pack-wide stall, surfaced actionably by `WriteLockHeldError` + pid forensics, remediated by operator SIGKILL (runbook); crash between seat CAS and status flip — impossible, one transaction; two racing resumes — single CAS winner; follower crash — registry expiry ⇒ graceful evict-then-reap under the full §C.2 predicate, attempt rotation, exactly-once proofs hold; slow follower in a 10-minute LLM call with expired item lease — registry live ⇒ not reapable ⇒ lease revived (false-takeover structurally gone); arm-(c)/stall-budget reap of a registered-but-stale worker — legal, `worker_stalled`-evented, lease-CAS-safe (§A.5); join racing finalize — both IMMEDIATE, serialized, harmless in both orders for the structural reason in §B.1, leftover member rows departed at finalize; late `mark_blocked` racing a flush — stays BLOCKED outside the per-group snapshot, dispositioned next intake (§E.3/E.3a), no Tier-1 false alarm; late branch against a completed coalesce group — journal-released with `late_arrival` context, finalize unblocked (§E.3a); follower-discovered branch loss — durable record, replayed before the next trigger evaluation, restored across takeover (§E.5); slow follower at source EOF — EOF flush waits for journal quiescence, one batch not two (§D); leader crash between a follower's `mark_blocked` and intake — BLOCKED row is durable truth, next leader restores, intake-pending disposition; leader crash mid-adoption — one transaction, rolls back whole, row stays intake-pending; crash between batches-COMPLETED and `complete_barrier` — pre-existing residual (elspeth-3977d8ab60), unchanged and named; heartbeat thread dies while drain lives — impossible to leave unexplained: the thread never self-terminates on DB errors (§A.3), and if the process truly wedges the stall budget reaps with eventing; heartbeat starved by write convoy — BUSY = liveness-unknown + `heartbeat_degraded` eventing + occupancy-sized window; process killed mid-IMMEDIATE-transaction — SQLite rolls back on connection death, no partial fence state observable; eviction CAS racing a live heartbeat — §C.2 path 1's grace + lease preconditions make rowcount 0; takeover never evicts followers at all.

---

## K. Adversarial review — findings and dispositions

All 21 findings were verified against the worktree; none were factually wrong on the code. 19 are fixed by mechanism amendments above; 2 (F4, F19) are fixed by re-stating an invariant the design could not honestly keep, plus eventing. No finding is silently dropped.

| # | Sev | Finding (short) | Disposition |
|---|---|---|---|
| F1 | critical | Ingest rows-insert unfenced in its own transaction; deposed leader's orphan row crashes live leader on UNIQUE(run_id, ingest_sequence) and wedges resume | **FIXED** — §C.4 row 9: leader ingest is one fenced IMMEDIATE transaction (epoch fence + rows + tokens + initial enqueue); deposed ingest rolls back atomically; ingest-adjacent recovery-read writes ride the same fence or the fenced checkpoint; woken-mid-ingest in the §H suite. Verified: `create_row_with_token` commits separately today (`data_flow_repository.py:599-650`). |
| F2 | critical | Intake's BUFFERED/batch_members writes leaseless+unfenced; deposed leader writes second live BUFFERED ⇒ false Tier-1 at restore, run unrecoverable | **FIXED** — §C.4 row 6a: adoption promoted to a leader-fenced verb (epoch fence + `barrier_adopted_epoch` CAS + membership + BUFFERED in one transaction, idempotent); restore gains a loud >1-live-BUFFERED Tier-1 diagnostic instead of silent latest-wins. Verified: token_outcomes uniqueness is terminal-only (`schema.py:392-399`); `record_token_outcome` is a plain INSERT. |
| F3 | important | Takeover eviction heartbeat-predicated, can miss the incumbent; deposed leader stays 'active' and passes membership fences | **FIXED** — §B.4: incumbent evicted BY IDENTITY, unconditionally (expired seat = proof of lost custody); §A.3: leader heartbeat beats both rows in ONE transaction (removes the skew source); snapshot showing a foreign leader_worker_id is fatal for leader-mode processes. |
| F4 | minor | Eviction-before-reap invariant contradicted by reap arm (c) | **FIXED (invariant restated)** — §A.5/§C.2: polarity (a) adopted; "never observe taken-but-registered" withdrawn as absolute; arm-(c)/stall reaps are the documented, lease-CAS-safe exception, now `worker_stalled`-evented. |
| F5 | critical | Coalesce branch LOSS has no cross-worker hand-off; follower-failed branches invisible to leader ⇒ wrong merges, wrong `branches_lost`, stalls; checkpoint scalars structurally incomplete | **FIXED** — §E.5 + schema: durable `coalesce_branch_losses` table written in the same lease-fenced transaction as the lossy disposition; leader intake replays losses through `notify_branch_lost` before trigger evaluation; takeover restore re-derives `lost_branches` from the table; uniformity at N=1. Verified: synchronous `_notify_coalesce_of_lost_branch` call sites (`processor.py:2754` + 7 sites). |
| F6 | important | Live timers anchor to intake time, not `barrier_blocked_at`; "immune to intake latency" false; live vs restore diverge under takeover | **FIXED** — §E.2: adoption is backdated — accept passes durable `barrier_blocked_at` through the exact wall→monotonic clamped transform restore already uses (`coalesce_executor.py:425-450`, `aggregation.py:765-780`, both verified); §H pins fire-time invariance under takeover; the false claim is corrected in the ADR-029 amendment. |
| F7 | important | §E.2 universal journal-first contradicts retained in-claim flush arm; either reading falsifies a claim (snapshot Tier-1 false alarm, or D7 closure) | **FIXED (contradiction resolved)** — §E.2: in-claim arm REMOVED entirely; `leased_exclusion_token_id` deleted from `complete_barrier`; the N=1 flush-timing change is owned and re-pinned in slice 3; D7-closure claim now true on every path; "bit-identical" claim withdrawn. |
| F8 | important | Intake-snapshot algebra unspecified w.r.t. `scope_row_id`; every multi-group coalesce flush raises spurious Tier-1 | **FIXED** — §E.3: snapshot specified per-firing-group `(barrier_key, scope_row_id)` at the contract level; `complete_barrier` validates snapshot tokens' durable row_id against scope (loud, no silent intersection); cross-group flush test in §H. |
| F9 | important | Late coalesce branch under journal-first has no BLOCKED-row release; `_completed_keys` in-memory rejection strands a row §D then refuses to finalize over | **FIXED** — §E.3a: intake-side rejection journal-releases via `mark_blocked_barrier_terminal` (verified present, `:2027`) with `late_arrival` context, same drain iteration; added to the §E.4 takeover reconcile; slow-follower-past-timeout test in §H. |
| F10 | critical | "Writer-mode engine" IMMEDIATE-listener keying unimplementable; web reads open writable ⇒ dashboard reads become write-lock holders that starve heartbeats into false evictions | **FIXED** — §0: listener keyed on an explicit write-intent execution option carried only by engine-side write verbs; web read surfaces converted to `from_url(read_only=True)` (facility verified, `database.py:977-1031`); slice-1 contention test includes dashboard-style reads with a heartbeat-latency assertion; ADR-030 text corrected. Verified: every cited web path opens writable today; sole read_only caller is `mcp/analyzer.py:64`. |
| F11 | important | Heartbeat behavior under SQLITE_BUSY/starvation/thread-death unspecified; registry-level dead-vs-slow confusion reintroduced | **FIXED** — §A.3: BUSY = liveness-unknown (retry in-tick, never crash, never flag on DB error); `heartbeat_degraded` event after k busy failures; window sized ≥ 4×(beat+busy_timeout); `complete_barrier` kept atomic with a documented barrier-size ceiling instead of chunking; slice-1 measures max hold time. |
| F12 | important | Frozen lock-holder blocks the takeover CAS itself; `elspeth resume` fails database-is-locked against the leader it exists to replace | **FIXED** — §B.4/§F: `acquire_run_leadership` distinguishes BUSY from CAS-loss (`WriteLockHeldError` carrying run_workers pid/hostname forensics); ADR-030 states the pack-wide stall and the SIGKILL remediation; G19 runbook gains the kill step (slice 6); the overclaiming "rolls back on connection death" line scoped to actual death. |
| F13 | important | Takeover bulk eviction lacks §C.2's grace + no-unexpired-leases preconditions; one takeover can evict the healthy pack | **FIXED** — §B.4: takeover evicts ONLY the deposed leader (by identity); follower eviction removed from the takeover transaction, deferred to the new leader's housekeeping sweep under the full §C.2 predicate. |
| F14 | minor | Per-worker JSONL merge by timestamp ≠ commit order; "emergency backup" silently demoted at N>1 | **FIXED (documented)** — §C.4 row 13/§F: JSONL journal declared forensic-only at N>1 in ADR-030 and journal docs; restore tooling gates on worker-count=1 provenance; field-name (`timestamp`) corrected; in-transaction `journal_seq` total order noted as future work. |
| F15 | minor | Web-leader + CLI-follower cross-uid SQLite file access never addressed | **FIXED** — §B.1 step 0: join-time filesystem-writability preflight with actionable `JoinRefusedError`; shared-group/umask + SQLCipher-passphrase requirements in ADR-030 and the slice-6 runbook. |
| F16 | critical | Row 10 misclassifies barrier-intake audit writes (node_states constraint cited for token_outcomes); double-BUFFERED corrupts `rows_buffered` in the sole derivation; plus the membership-without-BUFFERED crash window | **FIXED** — merged with F2: §C.4 row 6a fenced atomic adoption makes double-BUFFERED and membership-without-BUFFERED structurally impossible at epoch 21; row 10 re-scoped to lease-held writes only; restore diagnostic added; `rows_buffered` derivation safe by construction with the diagnostic as backstop. |
| F17 | important | Ingest fence one committed transaction too late (companion to F1) | **FIXED** — merged with F1: same single fenced ingest transaction; §H gains the woken-mid-ingest deterministic case; the "uncontested by construction" claim is now true. |
| F18 | important | EOF flush vs in-flight follower arrivals: "late arrivals wait for the next batch" has no next batch at EOF; nondeterministic batch splitting via FAILED/resume | **FIXED** — §D steps 2–3: EOF flush gated on journal quiescence (zero READY/LEASED — DB-derivable because followers hold no run-long memory); leader loops intake→evaluate→flush until BLOCKED empty before the terminal statement; N=2 slow-follower-at-EOF test asserts one batch. |
| F19 | minor | Arm-(c)/stall reaps leave rotations the coordination ledger cannot explain; "ledger alone" claim violated | **FIXED (invariant restated + evented)** — §A.5: reaped-while-registered is legal and now `worker_stalled`-evented in the same transaction as the rotation; the ledger-reconstructability claim is made true by the event rather than withdrawn. |
| F20 | minor | Join-vs-finalize "no window by construction" cites membership predicates the §D statement does not contain; leftover active rows on COMPLETED runs | **FIXED** — §B.1: argument rewritten to the true structural one (a registered joiner cannot create work; the quiescence predicate covers all claimable work); §D finalize transaction departs remaining active follower rows (evented), and the heartbeat rowcount-0 path distinguishes terminal-run departure (clean exit) from eviction. |
| F21 | minor | `recorded_at` can invert commit order across workers; `fence_refusal` "must survive the refusal" is not actually guaranteed by a fresh-connection write | **FIXED** — §A.2: `run_coordination_events.seq` AUTOINCREMENT is the authoritative replay order (`recorded_at` forensic-only; same caveat documented for the JSONL merge); `fence_refusal` reworded to best-effort attribution with the benign-crash-window rationale stated. |
