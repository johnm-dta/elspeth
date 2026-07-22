# State Engine Proof Matrix

This is the human-readable result for the current assessment at
`3c782ac3c7efb0550495be38f75800eddffa639a`. The machine authority is the
[v1 catalog](proof-catalog/v1/catalog.json) plus the current dated
[assessment manifest](assessments/2026-07-18-1820/assessment.json).

## Result

**Verdict: not complete.** No catalog leg currently has a complete package
across all ten mandatory dimensions. The 127 freshly executed checks below
establish important narrow properties, but they do not substitute for the
missing production-boundary, multiprocess, crash/restart, read-model, and
lifecycle evidence.

| Family | Legs | Confirmed | Gap | Unknown | Main unresolved proof |
| --- | ---: | ---: | ---: | ---: | --- |
| Token transitions | 19 | 0 | 9 | 10 | Cross-transaction continuation, multiprocess winners, restart at every seam |
| Auxiliary state | 7 | 0 | 4 | 3 | Long-call heartbeat/lease loss, membership orchestration, leader takeover |
| Run coordination | 7 | 0 | 7 | 0 | Leader-seat and worker-registry contention, atomicity, takeover, and teardown |
| Production boundaries | 9 | 0 | 8 | 1 | Real plugins, fresh-process restart, follower and lifecycle behavior |
| Read models | 13 | 0 | 13 | 0 | Complete scheduler, coordination, resume, redrive, and barrier-intake truth tables |
| Forbidden paths | 13 | 0 | 3 | 10 | Repository-wide refusal and zero-mutation coverage |
| **Total** | **68** | **0** | **44** | **24** | Every leg still has at least one unresolved mandatory cell |

`Gap` means a known architectural or evidence deficit names the leg. `Unknown`
means the package has not yet demonstrated a defect but still lacks mandatory
evidence. These counts are derived from the dated manifest; they are not a
maturity score.

## Strong current evidence

| Area | Fresh evidence | What it establishes | Important limit |
| --- | --- | --- | --- |
| TS-02 source completion and compatibility recovery | EV-004, 13 checks across `test_processor.py` and the public-resume E2E harness | Current ingress commits the source witness inside TS-02; selected row/token/source/work rollback planes pass; pre-fix repair validates exact attempt/event/identity/state evidence before plugin execution; a crash after repair resumes with one transform call | SQLite and deterministic crash injection with fresh resume objects; not every durable plane or guard, abrupt OS process death, independent-process contention, every source exclusion arm, or the remaining TS-02/PB-01 dimensions |
| Strict fencing and authority | EV-001, 29 checks in `test_scheduler_fencing.py` | `None` authority is rejected, stale tokens refuse without payload mutation, strict wrappers require authority, and production sources do not select the named legacy adapter | Mostly direct/source-contract evidence; not every live production call path or multiprocess takeover |
| Pending-sink admission | EV-001, 17 checks in `test_scheduler_pending_sink_claim.py` | Malformed bundles are refused without mutation; complete bundles claim as sink-redrive leases | SQLite repository scope; not expiry/reclaim or production redrive lifecycle |
| TS-07–TS-10 dispositions | EV-003, 30 checks in `test_scheduler_events.py` | Exact row/event/branch-loss success images, subtype/owner/membership refusal, and event/branch-loss rollback | Direct SQLite repository scope; not production plugin composition, contention, restart, or read-model truth tables |
| Barrier completion | EV-002, 29 checks in `test_scheduler_repository_complete_barrier.py` | Exact snapshot coverage, atomic consume/emission, refusal cases, and rollback on injected repository failure | Direct repository scope; not every aggregation/coalesce process-death seam |
| Built-in local sink recovery | EV-002, 9 checks in `test_builtin_sink_effect_recovery.py` | CSV/JSON response-loss reconciliation avoids duplicate publication; diversion and virtual-effect arms execute through the pipeline | Built-in local sinks only; injected response loss is not abrupt process death or long-lease takeover |

See [evidence.md](assessments/2026-07-18-1820/evidence.md) for exact command
vectors, collection counts, timings, and negative claims.

## Open proof themes

| Theme | Primary catalog legs | Live owner where known | Exit condition |
| --- | --- | --- | --- |
| Plugin call exceeds lease/stall budget | AUX-01, AUX-02, PB-02, PB-06 | `elspeth-51a4b5c771` | Bounded production call demonstrates heartbeat or safe lease-loss abandonment and takeover |
| Child enqueue before parent disposition | TS-00, PB-02, PB-03 | `elspeth-7cdc4da434` | Each crash point resumes without duplicate child, stranded parent, or conflicting audit image |
| Queue atomicity and replay | TS-00, TS-01 | `elspeth-c0d4a28e11` | Exact replay, incompatible replay, rollback, and independent-connection winner/loser cases pass |
| Source ingress and exclusion | PB-01, F-11 | `elspeth-9cd07962c7` | Real accepted, pre-row failure, quarantine, and process-death paths prove scheduler inclusion/exclusion |
| Registered multiprocess orchestration | TS-03–TS-06, AUX-06 | `elspeth-9a52eb80f9` | Independent processes prove one winner and correct redrive/membership behavior |
| Pending-sink bundle preservation | TS-04, TS-06 | `elspeth-76bb92bc7d` | Claim, expiry, reclaim, refusal, and restart preserve every bundle field |
| Heartbeat integration | AUX-01, AUX-02 | `elspeth-2aba594afb` | Production drain proves success, loss observation, and zero-mutation abandonment |
| Transform/gate composition | PB-02, PB-03 | `elspeth-2e66723070` | Representative plugins exercise every disposition and audit image |
| Follower traversal | PB-08, PB-09 | `elspeth-6f6bbbec00` | Real follower build, claim, traversal, disposition, and teardown pass |
| Aggregation continuation after TS-15 | TS-15, TS-18, PB-04 | Unowned in captured tracker snapshot | Durable continuation authority survives death between consume and child enqueue |
| Sink-effect long-call takeover | PB-06, PB-07 | Unowned in captured tracker snapshot | Deterministic independent-process test proves heartbeat/takeover and no duplicate external publication |
| Run coordination | RC-01–RC-07, RM-07–RM-08 | `elspeth-b8d0c9b40a` (RC-04), `elspeth-33c1793a26` (RC-07), plus existing multiprocess/lifecycle owners | Real register/takeover/heartbeat/release/admit/depart/evict paths and their read models pass independent-process and restart matrices; RC-04/RC-07 predicates are repaired |
| Read-model truth tables | RM-01–RM-13 | Unowned in captured tracker snapshot | Every status, subtype, seat, owner, run, grace/expiry, resume, redrive, and barrier-intake arm drives the correct production decision |

## Hard gates

All ten hard gates remain open. EV-001 through EV-004 partially support
malformed-bundle refusal, strict authority, zero-mutation refusal, atomic
source/barrier/disposition writes, source compatibility recovery, and
duplicate-publication recovery, but none closes its repository-wide gate.
`HG-09-mandatory-leg-unresolved` is independently open because every leg has an
unresolved required cell.
