# DAG gap analysis and remediation order

**Baseline:** `release/0.7.1` at `6e8a6bf5f2f8542bf5b95b1669ce3d3df68d93e3`
**Verdict:** **Not complete**
**Normalized maturity indicator:** **Not calculated** while mandatory
dimensions remain `U`.

## What materially improved since the seed assessment

- R1 normal dispositions now reject sink-redrive leases
  (`elspeth-f8f9272b68`, closed).
- R2 malformed pending-sink rows now fail closed at claim
  (`elspeth-d8e172676c`, closed).
- TS-07 through TS-10 have one maintained state/subtype/effect/rollback matrix
  (`elspeth-1076e2716a`, closed).
- Missing-token fencing, legacy barrier wrappers, and initial enqueue/claim
  membership gaps from the seed assessment are closed
  (`elspeth-e66c371acb`, `elspeth-b68bf5c161`, `elspeth-c25bcf5717`).
- Coalesce claim/CAS, artifact idempotency, call-index allocation, batch
  closure, token-outcome atomicity, routing-reason atomicity, and token
  ownership owners named in the seed assessment are closed.
- The exact nested `fork -> coalesce -> fork -> coalesce` production-builder
  reproducer passes; `elspeth-a6ca0bef77` is correctly closed.
- Queue import/export support is current and `elspeth-6421ffa028` is closed.
- Four seeded browser correctness files now exist, but their six cases remain
  describe-level skipped and therefore provide no acceptance evidence.

These changes raise confidence in selected local invariants. They do not close
a complete mandatory scenario row.

## Current remediation program

### Track 0 — Maintain one acceptance spine (P1, start immediately)

`elspeth-ef29ef6ba4` owns the 15-scenario production-path corpus. Add each proof
or fix below as a row in that corpus rather than creating another isolated test
island. A row is complete only when its applicable config, build, contract,
runtime, audit, restart, contention, authoring, round-trip, and scale cells have
exact evidence.

This track runs alongside the safety fixes; it does not delay them.

### Track 1 — Close reproduced correctness and security hard gates (P1)

1. Close expansion replay/identity (`elspeth-a25e9c009e`).
2. Serialize output-contract updates (`elspeth-3335de38c2`).
3. Prevent the sidecar journal from recording uncommitted transactions
   (`elspeth-d8d4d2849b`).
4. Complete verification/closure of the strict coordination-token work already
   in progress (`elspeth-97c7661957`).
5. Define one public/redacted/fingerprinted graph-config contract and apply it to
   node metadata, node identity, topology/checkpoint hashing, run settings,
   credential-bearing URLs, raw-secret development mode, and commencement-gate
   conditions. Current owners are `elspeth-c4080bfb06`, `elspeth-69c957ed96`,
   `elspeth-c8152fa4a8`, `elspeth-173d929d51`, `elspeth-f321e3ff21`,
   `elspeth-a71f3e49d0`, and `elspeth-d49417ab97`.

**Exit gate:** replay returns the same durable identity or a clean conflict;
state and evidence commit together; and common secret forms cannot enter any
graph identity, metadata, persistence, export, checkpoint, error, or diagnostic
surface.

### Track 2 — Complete deterministic crash and atomicity seams (P1)

Use one coherent close/commit cycle per owner:

1. TS-00/01 queue replay and atomicity (`elspeth-c0d4a28e11`).
2. TS-02/PB-01 ingress composition (`elspeth-9cd07962c7`).
3. TS-02 commit to source `COMPLETED` process-death seam
   (`elspeth-aafba3b298`).
4. Child enqueue to parent disposition process-death seam
   (`elspeth-7cdc4da434`).
5. TS-04/06 complete sink-bundle preservation (`elspeth-76bb92bc7d`).
6. Heartbeat success and real lease-loss abandonment
   (`elspeth-2aba594afb`).

Then fill the remaining aggregation/coalesce/whole-row fault points identified
in the state-engine plan. Every crash test must reopen the same database and
assert exact rows, tokens, identities, events, outcomes, effects, and branch
losses without timing-only sleeps.

### Track 3 — Prove registered contention and real plugin boundaries (P1)

1. Registered orchestration, pending-sink claim, sink-redrive recovery, barrier
   completion, and losing contenders (`elspeth-9a52eb80f9`).
2. Plugin execution beyond lease TTL/stall budget with reclaim and late return
   (`elspeth-51a4b5c771`).
3. Real transform/gate plugin disposition and audit (`elspeth-2e66723070`).
4. Production follower build/drain/traversal (`elspeth-6f6bbbec00`).

**Exit gate:** each claim epoch has one effective owner and disposition; stale
or losing processes cannot mutate, duplicate a sink effect, or remint child
work.

### Track 4 — Deliver authoring parity and browser acceptance (P2 after safety floor)

- Enable the six seeded Playwright cases tracked by `elspeth-7cf763da7c`.
- Implement the missing guided topology operations and the 27-case
  guided/freeform/import matrix.
- Compare canonical graphs and traversal plans, not YAML formatting.
- Resolve `elspeth-a5b86149d4` with an explicit row-union support or rejection
  contract; do not mislabel queue or coalesce as long-format row union.

**Exit gate:** every supported mandatory fixture is expressible and rejected
consistently across guided, freeform, import/export, and configured browser
flows.

### Track 5 — Repair and release-gate the contract (P2)

- `elspeth-be41d0ea25` updates the stale normative execution-graph contract and
  depends on the scenario corpus.
- `elspeth-cb1053fe46` defines the topology/runtime scale envelope, thresholds,
  failure behavior, owner, and required CI gate; it also depends on the corpus.
- Fix the signed fingerprint baseline (`elspeth-18fe6e759e`) and current
  contract-boundary gate debt so maintained-contract evidence can be green.

## Recommended next implementation package

Keep `elspeth-ef29ef6ba4` as the acceptance spine, then take the highest-risk
unassigned reproduced correctness defect: `elspeth-a25e9c009e`. After that,
close `elspeth-3335de38c2` and `elspeth-d8d4d2849b`, then execute the source
crash sequence beginning with TS-00/01 and TS-02 ingress before
`elspeth-aafba3b298`.

This order fixes known duplicate/atomicity failures before interpreting missing
crash evidence, while ensuring every closure lands in the maintained scenario
matrix.
