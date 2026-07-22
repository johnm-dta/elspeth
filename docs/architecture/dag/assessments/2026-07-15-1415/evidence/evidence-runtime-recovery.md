# Runtime and recovery evidence

**Assessment date:** 2026-07-15
**Assigned baseline:** `0dcd61acaa44082d93ec205683700e798748ee6d`
**Scope:** runtime scheduling, durable state transitions, sink redrive, crash recovery, and multi-worker behavior.
**Evidence basis:** current source and test inventory reconciled with the executed Wave 1 ledger in `docs/architecture/token-scheduler-state-engine.md`.

## Verdict

Runtime and recovery are **partially demonstrated, not complete**. The current suite provides useful happy-path and selected recovery coverage, especially around fork/coalesce accounting and multi-worker fencing. It does not yet establish the end-to-end safety claim needed for a production-complete DAG: no lost or duplicated work across every durable boundary, valid subtype state at every claim/disposition, and deterministic restart after crashes.

Two concrete invariant failures were reproduced in Wave 1. These are blockers, not evidence gaps:

1. A normal disposition can be applied to a sink-redrive subtype and leave a stale `pending_sink_name` behind.
2. `claim_pending_sink` can lease a malformed `PENDING_SINK` row whose `pending_sink_name` is `NULL`.

The remaining crash and concurrency seams below are **unknown until exercised**. They must not be described as passing merely because nearby unit or audit tests pass.

## Evidence ledger

| Capability | Current evidence | Result | Assessment |
|---|---|---:|---|
| Token creation and ordinary claim/disposition | Scheduler implementation plus ordinary execution tests; Wave 1 recorded `TS-03` as confirmed | Demonstrated | Core happy path is credible, but does not prove crash safety at transaction boundaries. |
| Sink-redrive lifecycle | Durable pending-sink state and redrive paths exist; Wave 1 recorded `TS-05` as confirmed | Partial | Nominal path exists, but subtype invariants are not enforced at all entry points; both reproduced defects are in this surface. |
| Multi-worker fencing | Existing multi-worker end-to-end tests cover join/follower/fence behavior; Wave 1 recorded `F-07` as confirmed | Partial | Selected contention behavior is demonstrated. Registered multi-process contention, initial-claim fencing, and long-plugin lease/stall behavior remain unproved. |
| Fork/coalesce recovery | Audit tests cover fork branch validation, fork/coalesce accounting, partial-fork recovery, post-coalesce resume, and expand/coalesce behavior | Demonstrated for covered scenarios | Strong scenario evidence, but not a complete scheduler crash matrix and not proof of event/state atomicity. |
| Event and state atomicity | No executed evidence identified for rollback across an event write plus scheduler/state mutation | Unknown | A crash or transaction failure could expose split-brain audit/state history until an explicit rollback test proves otherwise. |
| Source-boundary durability | No executed scheduler-row evidence identified at the source admission boundary | Unknown | Restart behavior before and after the first durable row/token boundary is not established. |
| Transform failure composition | Existing failure-path tests do not establish the scheduler state produced for every failing-transform plus route/disposition combination | Unknown | Must cover retry, quarantine, discard, error routing, and downstream branch loss under the same durable-state assertions. |
| Crash recovery at every durable seam | No complete crash-point/fault-injection matrix identified | Unknown | Recovery is scenario-based rather than a maintained state-machine contract. |
| Long-running plugin lease behavior | No executed evidence identified for lease expiry, heartbeat/renewal, process stall, and late completion | Unknown | A stalled or slow plugin can interact with reclaim/fencing in ways not covered by short-running tests. |
| Corrupt durable row handling | Wave 1 reproduced a malformed `PENDING_SINK` row being leased | Fail | Claim paths do not consistently fail closed on invalid subtype state. |

The executed ledger marks `TS-03`, `TS-05`, and `F-07` as confirmed. It leaves `TS-00`, `TS-01`, `TS-02`, `TS-04`, `TS-06`, `TS-07` through `TS-10`, `AUX-01`, `AUX-02`, `PB-01`, `PB-02`, `PB-03`, and `PB-08` as gaps. Those identifiers should remain linked to their definitions in the token-scheduler state-engine document; the completeness assessment should not silently reinterpret them as passes.

## Reproduced correctness defects

### R1. Normal dispositions accept a sink-redrive subtype

**Observed:** a row in the sink-redrive subtype can take a normal disposition while retaining `pending_sink_name`.

**Why it matters:** durable state no longer has one unambiguous meaning. A later claimant, recovery routine, audit consumer, or operator can see a normal lifecycle state carrying sink-redrive-only metadata. That weakens replay determinism and makes cleanup dependent on which transition path happened to run.

**Required invariant:** normal dispositions must either reject sink-redrive rows or atomically clear every sink-redrive-only field as part of an explicitly allowed state transition. The preferred contract is fail closed: subtype-specific transitions own subtype exit.

**Acceptance evidence:**

- Parameterized tests for every normal disposition against every incompatible subtype.
- Transaction assertions covering the row, lease/fence fields, pending sink metadata, and emitted audit event.
- A direct test that rejected transitions leave the entire durable record unchanged.

### R2. `claim_pending_sink` leases malformed pending-sink work

**Observed:** a `PENDING_SINK` row with `pending_sink_name = NULL` can be leased.

**Why it matters:** claim converts corrupt durable state into active work. Failure then occurs farther from the corruption boundary, potentially after lease/fence mutation, and the operator loses a precise invariant violation at the point of admission.

**Required invariant:** `claim_pending_sink` must atomically require the full subtype predicate, including a non-null, valid pending sink identity. Invalid rows must remain unleased and produce a clear fail-closed diagnostic or quarantine path.

**Acceptance evidence:**

- Negative claim tests for missing, unknown, and inconsistent pending sink identities.
- Assertions that invalid rows are not leased and no fence/attempt counters advance.
- Concurrent claimant test proving no worker can acquire a malformed row between validation and update.

## Evidence gaps to close

## Risk ranking and Filigree ownership

| Rank | Risk | Primary consequence | Filigree issue |
|---|---|---|---|
| P1 reproduced defect | Normal dispositions accept a sink-redrive subtype | Ambiguous durable state, unsafe recovery choice, and misleading audit history; can contribute to stuck or repeated sink work | `elspeth-f8f9272b68` |
| P1 reproduced defect | Malformed `PENDING_SINK` row can be claimed | Corrupt work becomes active, moves the failure away from its cause, and can remain stuck under a lease | `elspeth-d8e172676c` |
| P1 crash-proof gap | TS-02 commit before source `COMPLETED` durability | Restart may repeat admitted input or leave source progress inconsistent with durable scheduler work | `elspeth-aafba3b298` |
| P1 crash-proof gap | Child enqueue after commit but before parent disposition | Restart may duplicate a child or strand/redo the parent depending on the interruption point | `elspeth-7cdc4da434` |
| P1 stall-proof gap | Long plugin call crosses item TTL and stall budget | Reclaim plus late completion can duplicate effects; refusal without a viable owner can leave work stuck | `elspeth-51a4b5c771` |

The broader proof packages are already tracked as `elspeth-c0d4a28e11` (TS-00/01), `elspeth-9cd07962c7` (TS-02/PB-01), `elspeth-9a52eb80f9` (registered and multi-process contention), `elspeth-76bb92bc7d` (TS-04/06 bundle preservation), `elspeth-2aba594afb` (heartbeat/lease-loss integration), `elspeth-1076e2716a` (TS-07–10 effects/guards/rollback), `elspeth-2e66723070` (real transform/gate composition), and `elspeth-6f6bbbec00` (production follower traversal). Standalone initial-claim fencing remains owned by pre-existing `elspeth-c25bcf5717`.

### P1 — Restore durable-state fail-closed invariants

Fix and regression-test R1 and R2 before claiming runtime completeness. Centralize the subtype predicate so disposition, claim, resume, and repair paths cannot drift independently. Add a database constraint where the storage model can express it; retain application validation for actionable diagnostics.

**Exit gate:** every scheduler transition preserves the state/subtype truth table, malformed rows cannot be claimed, and failed transitions are transactionally inert.

### P1 — Build a crash-seam recovery matrix

Create fault-injection tests at each durable boundary:

1. before token/row creation;
2. after durable creation but before claim acknowledgement;
3. after claim/fence mutation but before plugin invocation;
4. during plugin invocation and after plugin success but before disposition;
5. between output persistence, audit event persistence, and scheduler disposition;
6. before and after fork child creation;
7. before and after coalesce readiness/claim;
8. before and after sink write and pending-sink redrive transition.

For each seam, restart from the same database and assert exact row counts, token states, fence ownership, audit events, sink effects, and absence of duplicate downstream work.

**Exit gate:** every seam has a deterministic expected state and a restart test; no scenario relies on timing-only sleeps.

### P1 — Prove transaction and audit atomicity

Inject database failures between scheduler mutation and event/audit persistence in both orders. Verify rollback leaves neither a state-only transition nor an event-only transition. Include fork/coalesce accounting and sink-redrive transitions, not only ordinary transforms.

**Exit gate:** the durable state and its audit explanation commit or roll back as one contract wherever the architecture claims atomicity.

### P1 — Exercise real multi-process contention

Extend beyond in-process or selected multi-worker scenarios:

- independent registered worker processes against the same database;
- simultaneous initial claims;
- simultaneous fork child and coalesce readiness claims;
- lease expiry followed by reclaim while the original worker returns late;
- pending-sink claims from competing workers;
- process termination rather than cooperative cancellation.

All assertions must use durable fence/lease values and final state, not only successful command exit.

**Exit gate:** exactly one effective owner/disposition per claim epoch, late workers are fenced, and retries do not duplicate sink or branch effects.

### P2 — Cover source and failure composition boundaries

Add scenarios that join source admission, transform failure, error policy, branch behavior, and restart. At minimum cover retry, quarantine, discard, routed error output, one failed branch before coalesce, and row expansion followed by failure/recovery.

**Exit gate:** each failure policy has a documented scheduler-state result and an executable restart assertion.

### P2 — Test long-running plugin and stalled-worker windows

Use deterministic clocks or controllable lease time to test heartbeat/renewal, expiry, reclaim, late success, and late failure. Include a plugin that exceeds a lease interval and a process that stops making progress without exiting.

**Exit gate:** the ownership model states whether leases are renewed or deliberately expire, and tests prove the corresponding fencing behavior.

## Required maintained test matrix

| Scenario | Single worker | Multi-process | Crash/restart | Audit/state atomicity | Expected minimum |
|---|---:|---:|---:|---:|---:|
| Source admission and first claim | Yes | Yes | Yes | Yes | Pass |
| Ordinary transform success | Yes | Yes | Yes | Yes | Pass |
| Transform retry/quarantine/discard/error route | Yes | Yes | Yes | Yes | Pass |
| Fork child creation and partial completion | Yes | Yes | Yes | Yes | Pass |
| Coalesce readiness, claim, and completion | Yes | Yes | Yes | Yes | Pass |
| Sequential fork/coalesce compositions | Yes | Yes | Yes | Yes | Pass |
| Row expansion and downstream recovery | Yes | Yes | Yes | Yes | Pass |
| Sink write and pending-sink redrive | Yes | Yes | Yes | Yes | Pass |
| Lease expiry, reclaim, and late worker | Yes | Yes | Yes | Yes | Pass |
| Malformed durable subtype | Yes | Yes | N/A | Yes | Fail closed |

Passing the single-worker column alone caps the runtime/recovery rating at **3/5 (happy-path supported)**. A production-supported **4/5** requires all mandatory multi-process, restart, and atomicity cells to pass, with R1 and R2 closed. A maintained-contract **5/5** additionally requires the state/subtype truth table and crash matrix to be versioned architecture contracts and mandatory CI gates.

## Focused test evidence and limitations

The existing repository inventory includes property coverage for deep and composed DAG topologies, audit coverage for fork/coalesce recovery scenarios, and multi-worker end-to-end coverage for join/follower/fence behavior. In the parent session, the DAG property suites passed **62 tests**, and the composer importer/generator suites passed **66 tests**. Those results support graph structure and authoring round-trip; they do **not** close the runtime/recovery gaps in this document.

Fresh focused verification for this assessment:

```text
uv run pytest tests/unit/core/landscape/test_scheduler_events.py \
  tests/integration/engine/test_two_process_scheduler_contention.py -q
26 passed, 13 warnings in 6.31s
```

This confirms the currently exercised scheduler-event paths and direct two-process contention slice. It does not exercise malformed persisted state, registered plugin/follower contention, sink-redrive contention, or the listed process-death seams. The reproduced R1/R2 results and `TS-*`/`F-*` ledger status come from the executed Wave 1 session already recorded in the architecture document. The shared execution host was intermittently FD-exhausted during this assessment; commands that did run are reported explicitly, and unexecuted seams remain `Unknown`.

## Assessment handoff

Treat runtime/recovery as **blocked from a completeness verdict** until both reproduced defects are fixed and the P1 evidence gates pass. The most efficient shore-up order is:

1. enforce the durable subtype truth table at transition and claim boundaries;
2. add regression tests for R1 and R2;
3. add transaction rollback tests;
4. implement the crash-seam matrix;
5. run real multi-process lease/fence contention;
6. fill source/failure and long-plugin composition coverage;
7. promote the matrix to mandatory CI and architecture documentation.

This ordering first restores fail-closed state semantics, then proves restart and concurrency behavior on top of a trustworthy state machine.
