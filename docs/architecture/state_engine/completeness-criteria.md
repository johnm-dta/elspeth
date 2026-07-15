# State Engine Completeness Criteria

## Purpose

This document defines what Elspeth means when it calls a state engine
**complete**, **confirmed**, or **production-supported**. It prevents broad test
counts, method names, or a successful happy path from standing in for durable
correctness.

The first assessed engine is the durable token scheduler. The criteria are
written so a later web-engine assessment can reuse the evidence discipline while
defining its own states and boundaries.

## Product boundary

For the token scheduler, the state-engine boundary starts when a source plugin
produces a row and ends only when every resulting scheduler obligation is
terminal and its plugin-side durable consequence is recorded.

The boundary includes:

- source acceptance, rejection, quarantine, and initial scheduling;
- deterministic work identity, claims, leases, heartbeat, and recovery;
- transform and gate dispositions;
- aggregation and coalesce barrier intake, adoption, completion, and replay;
- sink handoff, external write ordering, durable outcomes, redrive, and repair;
- worker membership, leader epochs, stale-owner refusal, and run completion;
- scheduler events, coordination events, branch losses, token outcomes, and
  every read model that authorizes orchestration progress;
- fresh execution, resume, follower execution, process death, and contention;
- production plugin lifecycle at every boundary.

The current assessment excludes the web session engine, UI-only state, and
third-party plugin internals beyond their declared protocol boundary.

## Evidence vocabulary

| Verdict | Meaning |
| --- | --- |
| **Mapped** | The production implementation, callers, guards, and durable effects were located. |
| **Candidate** | Source or a test suggests a behavior or gap, but the proof package has not been executed. |
| **Confirmed** | The required positive, negative, rollback, and applicable concurrency/plugin evidence executed against the recorded baseline. |
| **Gap** | Execution or exhaustive evidence review positively established a defect, an absent proof, or an unresolved policy boundary. |
| **Intentionally absent** | Production reachability was disproved or an accepted ADR explicitly excludes the path, with a regression preventing accidental use. |
| **Unknown** | Evidence is absent, skipped, stale, or not representative enough to classify. |

`Unknown` and `Candidate` never count as passing. A closed issue or implementation
plan is evidence of intent, not evidence of correctness.

## Unit of proof

Assess each `TS-*`, `AUX-*`, `PB-*`, `RM-*`, and `F-*` leg independently. A leg
is Confirmed only when its proof package covers every applicable dimension:

1. **Production entry:** call the real public repository, orchestrator, or plugin
   boundary used at runtime. A private helper test can supplement but not replace
   this evidence.
2. **Precondition image:** record the complete relevant row, event, outcome,
   branch-loss, coordination, and external-effect state before the operation.
3. **Success effects:** assert the state transition, identity, attempt, lease,
   payload, subtype bundle, auxiliary rows, and event attribution.
4. **Guard/refusal:** exercise wrong state, wrong owner, wrong run, missing
   authority, stale epoch, incomplete subtype, duplicate input, and incompatible
   replay wherever the contract applies.
5. **Zero-mutation rollback:** compare complete before/after images when a guard,
   compare-and-swap, event insert, auxiliary insert, or injected failure refuses
   the operation.
6. **Concurrency:** use independent database connections or operating-system
   processes for any claim about races, unique ownership, fencing, or
   exactly-once durable effects.
7. **Crash/restart:** interrupt every cross-transaction seam, restart from the
   same durable database, and prove the declared replay result.
8. **Plugin boundary:** use a real protocol implementation when the guarantee
   crosses source, transform, gate, aggregation, coalesce, or sink execution.
9. **Read-model truth table:** prove every included state/subtype and every
   deliberately excluded state/subtype, plus run and owner scoping.
10. **Maintenance:** bind the proof to a mandatory test suite and a named
    remediation or ownership surface.

## Hard gates

The state engine is not complete while any of these conditions holds:

- a legal production call can create or accept an invalid operational subtype;
- a missing or stale coordination token can downgrade a protected write;
- two workers can both win a single-owner transition or durable effect;
- a refused operation can mutate scheduler state, evidence, or plugin-visible
  effects;
- state can commit without its required event or evidence, or vice versa;
- restart can lose work, duplicate an internal effect, or replay a non-idempotent
  plugin effect outside a documented recovery window;
- a read model can permit flush, relinquish, resume, or finalization with an
  untested state/subtype arm;
- a plugin lifecycle path can leak resources or bypass required durability;
- a mandatory leg remains Candidate, Unknown, or Gap;
- the normative ADR contradicts live code or executed evidence.

## Mandatory proof families

| Family | Required result |
| --- | --- |
| Intake and identity | Deterministic enqueue/claim, exact replay, source exclusion, rollback, and real contention. |
| Leasing and recovery | Subtype-preserving claims, heartbeat, identity/attempt behavior, liveness, ABA protection, and competing recovery. |
| Claimed dispositions | Exact state/event/effect bundles, owner and membership guards, branch-loss atomicity, and real plugin paths. |
| Sink durability | External-write ordering, outcome witness, terminal callback, redrive/repair, diversion modes, and declared duplication window. |
| Barrier machinery | Adoption, reset, branch-loss replay, exhaustive consume/emit, late-arrival isolation, and every crash seam. |
| Fencing and reads | Membership, epoch, eviction-before-reap, role separation, and complete RM truth tables. |
| Plugin lifecycle | Fresh, resume, follower, partial-start failure, normal teardown, and exceptional teardown for each plugin kind. |
| Forbidden paths | Every illegal edge fails closed; every dormant edge is proved unreachable or removed. |

## Completion rule

Declare the assessed state engine complete only when:

- every mandatory leg is Confirmed or Intentionally absent;
- no state-engine hard gate is open;
- every confirmed gap has a deduplicated owner and its closing regression passes;
- every concurrency claim has real independent-connection or process evidence;
- every cross-transaction seam has a deterministic crash/restart result;
- the full proof matrix is mandatory in continuous integration;
- the dated assessment, current hub verdict, live tracker, and source agree; and
- the comprehensive state-engine ADR is accepted and explicitly reconciles the
  prior state-machine decisions.

Completion is binary. Counts and maturity summaries may show progress, but they
cannot average away a hard-gate failure.
