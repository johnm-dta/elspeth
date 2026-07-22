# Token Scheduler State Engine Gap Analysis

**Assessment date:** 2026-07-15

**Baseline:** `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d`

**Verdict:** **Not complete — hard-gate defects and mandatory proof gaps remain**

## Executive assessment

Elspeth already has a substantial durable scheduler: deterministic work
identity, explicit status transitions, transition events, write-intent
transactions, leader and membership fences, journal-first barrier recovery, and
outcome-witnessed sink repair. Its weakness is not the absence of a state
machine. The weakness is that operational subtypes, compatibility arms, plugin
transaction boundaries, and orchestration reads are not yet governed by one
fully executed contract.

The immediate objective is therefore:

> **Make every legal leg explicit and every illegal or interrupted leg fail
> closed, then turn that matrix into the normative ADR and a mandatory CI
> contract.**

## Hard-gate implementation defects

| Priority | Defect | Consequence | Existing Filigree owner |
| --- | --- | --- | --- |
| P1 | Normal dispositions accept sink-redrive LEASED rows | A redrive can take an invalid disposition and retain stale sink metadata. | `elspeth-f8f9272b68` |
| P1 | `claim_pending_sink` accepts an incomplete PENDING_SINK subtype | Malformed durable work becomes leased and ambiguous. | `elspeth-d8e172676c` |
| P1 | `fenced_or_plain_write(None)` downgrades to a plain write | Missing authority can mutate protected state. | `elspeth-e66c371acb` |
| P1 | Legacy barrier wrappers rely on the unfenced compatibility arm | Barrier rows can be released without current leader authority. | `elspeth-b68bf5c161` |
| P2 | Standalone enqueue-and-claim can bypass membership fencing | An absent/inactive worker can reach an internal claim path. | `elspeth-c25bcf5717` |
| P2 | TS-14 repair has a production-visible explicit-None fencing path | Outcome-backed repair can execute without the documented required fence. | `elspeth-97c7661957` |

The existing issues own these defects. The remediation campaign should add the
missing regressions but must not file duplicates.

## Confirmed evidence gaps

| Scope | Gap | Current owner |
| --- | --- | --- |
| TS-00/01 | Refusal, replay, complete atomicity | `elspeth-c0d4a28e11` |
| TS-02/PB-01 | Full ingress composition and scheduler exclusion | `elspeth-9cd07962c7` |
| TS-02 crash seam | Initial lease before source COMPLETED | `elspeth-aafba3b298` |
| Registered contention | Plugin/follower/sink/barrier multiprocess proof | `elspeth-9a52eb80f9` |
| TS-04/06 | Complete sink-bundle preservation | `elspeth-76bb92bc7d` |
| AUX-01/02 | Heartbeat/no-event and lease-loss integration | `elspeth-2aba594afb` |
| Long plugin execution | Lease TTL plus worker stall budget | `elspeth-51a4b5c771` |
| TS-07–10 | Effects, guards, subtype refusals, event rollback | `elspeth-1076e2716a` |
| PB-02/03 | Real transform/gate disposition composition | `elspeth-2e66723070` |
| Child enqueue seam | TS-00 children before parent disposition | `elspeth-7cdc4da434` |
| PB-08 | Real follower construction and plugin traversal | `elspeth-6f6bbbec00` |
| RM-02–RM-06 | Complete truth tables and peer PENDING_SINK consumer arm | No dedicated issue found; file only after the ledger records the executed gap |
| TS-11–14/PB-06/07 | Real sink durability-to-terminalization and diversion repair | No dedicated issue found for candidates 12/13 |
| AUX-03–05 | Explicit no-event, rollback, and concurrency contract | No dedicated issue found |
| Barrier caller authority | Missing token cannot leak to compatibility completion | Owned in part by the two P1 fencing defects; no separate proof task found |

## Crash seams requiring executable discrimination

These are not yet classified as implementation defects. Each needs a fault or
process-death test that captures the durable image before restart.

| Candidate | Seam | Discriminator |
| --- | --- | --- |
| 16 | Coalesce decision/audit before `complete_barrier` | Crash after real merge decision; restart must emit exactly one result. |
| 19 | TS-02 before source COMPLETED | Crash after initial lease; restart must reconcile one source row and one work identity. |
| 21 | Successful aggregation outcome before TS-15 | Crash after COMPLETED batch/outcomes while inputs remain BLOCKED; restart must finish or expose an explicit fail-closed recovery policy. |
| 22 | Aggregation TS-15 before later child TS-00 | Crash before downstream scheduling; restart must not lose the continuation. |
| 23 | DRAFT batch before AUX-03 adoption | Restart must ignore or safely reclaim orphan membership state. |
| 24 | Plugin return/effect before terminal node audit | Replay policy must prevent or explicitly bound repeated effects. |
| Sink pre-witness | External flush before durable outcome | Test and document the exact duplication window for non-idempotent sinks. |

## Scale and transaction-shape gaps

- TS-13 builds an unbounded token-ID `IN` batch: `elspeth-2861b3b0fa`.
- TS-14 repair can run as one unbounded transaction: `elspeth-edc2698211`.
- Current multiprocess proof is direct repository/N=0 membership and does not
  establish registered orchestration or plugin behavior.
- Most negative tests assert status rather than a complete row/event/auxiliary
  before-and-after image.

## Policy decisions reserved for the superseding ADR

1. Whether the scheduler event plane is intentionally transition-only or must
   cover successful AUX mutations.
2. Whether unfenced compatibility writes remain available only through an
   explicitly named test/legacy API or are removed entirely.
3. Whether queue-only BLOCKED work is a reserved subtype, reachable production
   behavior, or dead state to remove.
4. The declared external sink duplication boundary before the outcome witness.
5. The authoritative recovery policy for plugin effects that precede terminal
   audit evidence.
6. Which earlier ADRs are superseded versus amended or retained independently.

## Priority order

| Order | Objective | Why first |
| ---: | --- | --- |
| 1 | Complete RM-02–RM-06 truth tables | Small, bounded, and likely implementation-neutral; improves orchestration visibility. |
| 2 | Pin and fix subtype/fencing hard gates | Current code can accept invalid state or missing authority. |
| 3 | Prove primary sink outcome/callback/repair composition | Establishes the external-durability boundary before diversion complexity. |
| 4 | Execute aggregation/coalesce crash discriminators | Separates recoverable seams from implementation loss/duplication defects. |
| 5 | Extend to diversion, registered processes, and long plugin calls | Adds the real concurrency and external-effect proof currently missing. |
| 6 | Close plugin lifecycle and forbidden paths | Settles residual reachability and teardown contracts. |
| 7 | Publish comprehensive ADR and CI matrix | Architectural guarantees follow evidence; they do not precede it. |

The exact task sequence and commands are in the
[remediation plan](05-remediation-plan.md).
