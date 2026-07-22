# Token Scheduler State Engine Executed Evidence

**Assessment date:** 2026-07-15

**Baseline:** `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d`

## Evidence rule

The verdicts below apply the full proof standard from
[completeness criteria](../../completeness-criteria.md). A Gap may have strong
positive evidence; it means one or more required proof dimensions remain absent
or an implementation defect was reproduced.

## Wave 1 verdict

Wave 1 executed 83 passing specialist node invocations across three packages,
followed by a fresh 16-test cross-package run that included the direct
two-process claim hammer. Two direct production-repository probes reproduced the
subtype defects.

| Leg | Verdict | Positive evidence | Missing or failed proof |
| --- | --- | --- | --- |
| TS-00 | **Gap** | Real enqueue, exact replay, single event, membership behavior | Incompatible replay and reference refusals lack complete row/event rollback images. |
| TS-01 | **Gap** | New-row composition emits ordered enqueue and claim events | Existing-READY reconciliation, composed rollback, and standalone membership are incomplete. |
| TS-02 | **Gap** | Stale leader rolls back row, token, item, and leader seat | Successful full composition and source-COMPLETED crash seam remain unproved. |
| TS-03 | **Confirmed** | State, ordering, identity/attempt, event, rollback, membership refusal, two-process contention | Claim deliberately limited to direct repository/N=0 membership contention. |
| TS-04 | **Gap** | Claim/event and membership refusal execute | Malformed pending-sink row is claimable; bundle preservation and contention are incomplete. |
| TS-05 | **Confirmed** | Attempt bump, identity rotation, event, liveness arms, races, rollback, two-process recovery | Registered plugin/follower contention remains outside this leg's claim. |
| TS-06 | **Gap** | Return to PENDING_SINK, identity/attempt preservation, reclaim, terminalization, ABA protection | No complete bundle-plus-event proof or sink-redrive contention. |
| TS-07 | **Gap** | Real drain to BLOCKED, release keys, membership/owner guards, event existence | Sink-redrive refusal, event rollback, and one complete effects/event proof are absent. |
| TS-08 | **Gap** | Real terminal drain, membership/owner refusal, event | Scrub, branch loss, rollback, and sink-redrive refusal are not jointly proved. |
| TS-09 | **Gap** | Real failed drain, fencing, event, direct branch-loss composition | Real failing plugin does not jointly prove failed audit, TS-09, event, and effects. |
| TS-10 | **Gap** | Real sink handoff, bundle/owner effects, membership, branch loss, event existence | Stale-owner zero mutation, detailed event, rollback, and subtype refusal are incomplete. |
| AUX-01 | **Gap** | Heartbeat extends expiry while preserving identity and lease state | No successful no-scheduler-event assertion. |
| AUX-02 | **Gap** | CAS loss emits `LEASE_LOST`; drain abandons disposition | Complete row image, event, and production abandonment are not integrated. |
| F-07 | **Confirmed** | Fresh/self/live/dead/stalled arms, `worker_stalled`, compatibility behavior | Long-plugin stall window is tracked separately. |
| PB-01 | **Gap** | Real source loading, accepted, failure, quarantine, end-to-end execution | Initial events, no-row exclusions, and source-state crash seam are incomplete. |
| PB-02 | **Gap** | Real transform retry/audit and scheduler failure each execute | No single real-plugin proof spans both evidence planes. |
| PB-03 | **Gap** | Real gate evaluation and routing execute | Disposition state/event/payload/fencing/branch-loss effects are not jointly asserted. |
| PB-08 | **Gap** | Follower-shaped dispositions execute repository paths | Tests bypass real follower construction, plugin traversal, and drain. |

Wave 1 therefore contains **3 Confirmed** legs and **15 Gap** legs.

## Wave 1 positively confirmed candidates

| Candidate | Classification | Executed conclusion |
| --- | --- | --- |
| 3 | Implementation defect | A sink-redrive lease accepts normal disposition verbs and retains stale sink metadata. |
| 4 | Implementation defect | `claim_pending_sink` leases a PENDING_SINK row whose sink name is NULL. |
| 6 | Evidence gap | Event-insert rollback is directly injected for TS-03, not TS-07–TS-10. |
| 7 | Bounded concurrency | Direct READY claim/recovery has multiprocess proof; registered/plugin/follower/sink/barrier paths do not. |
| 10 | Evidence gap | Source rejection/quarantine tests do not all assert an empty scheduler table. |
| 11 | Composition gap | Real failing-transform audit and TS-09 are proved separately. |
| 14 | Crash-seam gap | Child enqueue before parent disposition has no interruption/replay test. |
| 17 | Implementation/fencing defect | Standalone initial claim can bypass the public membership fence. |
| 18 | Contract/concurrency gap | No plugin execution crosses both lease TTL and worker stall budget under contention. |
| 19 | Crash-seam gap | No process-death proof covers TS-02 before source COMPLETED durability. |

## Wave 2 reconnaissance verdict

Wave 2 traced the real caller chains and reviewed the strongest current tests.
It did not edit tests or promote provisional guarantees. No newly assessed Wave
2 leg yet meets every applicable proof dimension.

### Sink durability

| Leg | Strongest current evidence | Remaining proof |
| --- | --- | --- |
| TS-11/12/13 | Per-token events, batch guards, owner CAS, stale-token refusal, resumed sink-redrive | Real sink-to-audit-to-bulk-close composition, complete rollback images, event-failure rollback, and process contention. |
| TS-14 | Outcome-witness repair without reclaim/re-emission in processor tests | Real prior sink I/O, crash/resume, no-outcome zero mutation, event rollback, and strict fencing. |
| PB-06/07 | Real CSV artifact/audit evidence, primary audit atomicity, full diversion pipelines | One scheduler-backed production test spanning plugin I/O, outcome, callback interruption, repair, and emission count. |

Candidate 12 remains a production-composition gap. Candidate 13 remains an
exact-once diversion/crash gap. The pre-outcome sink-flush duplication window is
not yet characterized as an explicit contract.

### Barrier completion

| Leg | Strongest current evidence | Remaining proof |
| --- | --- | --- |
| TS-15–18 | Broad repository consume/emit, snapshot, collision, late-arrival, event, and crash-atomicity tests | Real plugin entry with complete effects/event/refusal image and process overlap. |
| AUX-03 | Adoption, idempotency, stale guard, and rollback-after-CAS | Explicit no-event contract, real contention, and orphan-DRAFT recovery. |
| AUX-04 | Real restore resets and re-adopts a holdless coalesce row | Direct guard/rollback/no-event/concurrency matrix. |
| AUX-05 | Idempotent branch-loss adoption and takeover replay | No-event, cursor rollback, and process-overlap proof. |
| PB-04 | Real aggregation failure/restart and failed-flush reconciliation | Successful transform crash before completion and non-sink crash after completion. |
| PB-05 | Real coalesce terminal/nonterminal paths, restore, loss replay, late release | Crash after durable merge decision but before scheduler completion. |
| F-09/10 | Extensive negative barrier matrix and stale-winner E2E images | Complete row/event/aux rollback for every refusal and caller-level missing-token proof. |

Candidate 5 is a reproduced implementation defect with existing ownership.
Candidates 9 and 15 are confirmed proof gaps. Candidates 16, 21, 22, and 23
remain crash-seam discriminators: source inspection shows a risky durable image,
but the restart outcome still requires execution.

### Fencing and read models

| Leg | Evidence state |
| --- | --- |
| RM-01 | Complete production-verb matrix for READY, transform LEASED, BLOCKED, PENDING_SINK, sink-redrive LEASED, TERMINAL, and FAILED. |
| RM-02 | Positive READY/transform-LEASED evidence; focused exclusions and summary grouping incomplete. |
| RM-03 | Active-count assertions are scattered; no single all-status/run-scope matrix. |
| RM-04 | Peer LEASED/PENDING_SINK evidence; ownerless, self-attributed PENDING_SINK, and other-status exclusions incomplete. |
| RM-05 | Strong READY/FAILED/solo/peer-LEASED consumer tests; peer PENDING_SINK relinquishment missing. |
| RM-06 | Peer-live and heartbeat extension evidence; exact expiry, caller, non-LEASED, deduplication, and ordering matrix missing. |
| AUX-06/F-06 | Strong membership refusals; scheduler-event preservation needs explicit assertions. |
| AUX-07/F-10 | Strong sequential stale-epoch zero-mutation E2E; missing-token and real simultaneous-process boundaries remain. |
| F-07 | Remains Wave 1 Confirmed; eviction-before-reap orchestration ordering deserves one production-coordinator test. |

Candidate 8 is a **confirmed evidence gap**: RM-02–RM-06 do not yet have the
required complete truth-table/consumer proof.

## Verification runs retained for this assessment

| Command or scope | Result | Establishes | Does not establish |
| --- | --- | --- | --- |
| Wave 1 specialist selections | 83 passing node invocations | Breadth across the three Wave 1 packages | Unique test count or uncovered production seams |
| Wave 1 root cross-package selection | 16 passed, 1 warning in 4.88s | Fresh representative evidence including direct two-process claim contention | Registered orchestration or Wave 2 paths |
| `.venv/bin/pytest -q tests/unit/core/test_count_ready_in_set.py tests/unit/core/test_multi_source_foundation.py -k 'unresolved_work_excludes_durable_sink_handoffs or count_ready or count_failed'` | 16 passed in 3.70s | Adjacent RM-01/RM-05 read helpers are healthy before Wave 2 edits | Missing RM-02–RM-06 matrix |
| Wave 2 read/fencing specialist bundle | 63 passed in 3.64s | Existing focused suites remain green | The exact shell command was not retained, so this is supplementary rather than audit-grade command provenance |

Detailed suite and source inventories are under [evidence](evidence/README.md).
