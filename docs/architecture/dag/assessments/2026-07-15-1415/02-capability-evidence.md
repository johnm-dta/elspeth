# DAG Completeness Capability Evidence

**Assessment date:** 2026-07-15
**Baseline:** `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d`

This document consolidates the three evidence passes. Detailed evidence remains in:

- [`evidence/evidence-core-dag.md`](evidence/evidence-core-dag.md)
- [`evidence/evidence-runtime-recovery.md`](evidence/evidence-runtime-recovery.md)
- [`evidence/evidence-authoring-contracts.md`](evidence/evidence-authoring-contracts.md)

## Consolidated verdict

Elspeth has a mature graph model and a broad topology vocabulary. It is not yet an end-to-end complete DAG product. Its strongest layers are graph representation and structural validation. Its weakest layers are durable concurrency/recovery, secret-safe graph identity, guided authoring parity, and maintained cross-surface proof.

The current maturity score is **2.4/5**. This is a maturity indicator, not a percentage-complete estimate. The hard-gate verdict is **Not complete** because confirmed defects can violate durable subtype, fencing, idempotency, or secret-handling invariants.

Dimension scores are **layer-local**: a 4 for structural validation means that layer has production-grade local evidence, not that recovery, authoring, or security also score 4. The overall verdict is governed by the weakest mandatory lifecycle cells and the hard gates, not by averaging strong graph-model scores over safety failures.

| Dimension | Score | Current evidence |
| --- | ---: | --- |
| Topology expressiveness | 4 | Seven node types; plural sources; gates, queues, aggregations, coalesces, and multiple sinks. |
| Structural validation | 4 | Acyclicity, roots/sinks, reachability, explicit fan-in, unique labels, and route metadata fail closed. |
| Schema and cardinality contracts | 3 | Strong edge/coalesce validation, but schema escape hatches and row-union absence remain. |
| Compositional closure | 3 | Coalesce-to-downstream-gate builder/runtime regressions pass; exact sequential fork/coalesce compositions still lack one canonical production-path matrix. |
| Runtime happy path | 3 | Broad execution and audit scenarios exist, including multi-source queue and fork/coalesce flows. |
| Recovery and multi-worker safety | 1 | Two P1 subtype defects are reproduced; critical crash, stall, CAS, fencing, and idempotency gaps remain open. |
| Audit and explainability | 2 | Rich evidence exists, but several state/event/outcome writes are not proven atomic. |
| Authoring parity | 2 | Freeform/import/export is capable; guided mode remains linear-only and browser parity specs are pending. |
| Security of graph identity/metadata | 1 | Raw secret-bearing plugin configuration can enter topology hashing and DAG metadata. |
| Scale evidence | 1 | Focused validator performance exists, but no maintained topology/runtime scale envelope proves supported limits. |
| Maintained contracts | 2 | The scheduler ledger is strong; the execution-graph contract and parity suite lag live behavior. |

## Verified strengths

- `NodeType` defines source, queue, transform, gate, aggregation, coalesce, and sink.
- `build_execution_graph()` accepts plural sources and every canonical structural node type.
- `ExecutionGraph.validate()` enforces the main structural invariants.
- Queue fan-in is intentionally scheduling coordination, while coalesce is a sibling-row join.
- Coalesce supports four completion policies and three merge strategies.
- The current builder resolves schemas for gates downstream of coalesce in a second pass.
- Freeform YAML importer and generator support queue sections.
- Property, scheduler, Composer, and focused coalesce-to-gate regressions passed during this assessment.

## Hard-gate failures

| Gate | Evidence | Consequence |
| --- | --- | --- |
| Valid durable subtype transitions | `elspeth-f8f9272b68`, `elspeth-d8e172676c` | Ambiguous or malformed work can be leased or disposed. |
| Fenced multi-worker writes | `elspeth-e66c371acb`, `elspeth-b68bf5c161` | Missing coordination tokens can downgrade protected writes to plain writes. |
| Idempotent join/expansion effects | `elspeth-2172918fb7`, `elspeth-a25e9c009e` | Coalesce or expansion can duplicate durable work under races or replay. |
| Atomic state and audit evidence | `elspeth-4003f7993a`, `elspeth-322c417d23`, `elspeth-d8d4d2849b` | State, routing reason, outcome, or journal evidence can diverge. |
| Secret-safe graph representation | `elspeth-69c957ed96`, `elspeth-c4080bfb06` | Credentials can leak or become correlation oracles through graph metadata/hash surfaces. |

## Test evidence from this assessment

| Command/scope | Result | What it establishes |
| --- | --- | --- |
| DAG property suites | 62 passed in 3.78s | Structural algorithms, complex direct-graph shapes, and step maps. |
| Composer YAML importer/generator | 66 passed in 6.14s | Current queue-aware import/export helper behavior. |
| Scheduler events and direct two-process contention | 26 passed, 13 warnings in 6.31s | The covered event paths and a bounded direct contention slice. |
| Deferred coalesce schema and non-terminal coalesce runtime path | 2 passed in 2.48s | Gate-after-coalesce schema propagation and runtime continuation. |

These tests are positive evidence for their exact scopes. They do not close the open crash seams, subtype defects, fencing bypasses, secret exposure, guided parity, or full scenario matrix.

## Tracker reconciliation

- `elspeth-6421ffa028` still says queue YAML is dropped, but current queue import/export source and 66 focused tests contradict that description. Replay the original reproducer, then close or narrow the issue.
- `elspeth-a5b86149d4` retains a real row-union/append need. Its older queue-exposure premise is stale; a pass-through queue is not a long-format row union.
- `elspeth-a6ca0bef77` reports a possible coalesce-to-gate schema failure. Current builder/runtime regressions show the basic boundary works, but the exact fork→coalesce→fork→coalesce fixture is still needed before closing the issue.
