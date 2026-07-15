# DAG Completeness Gap Analysis

**Assessment date:** 2026-07-15
**Baseline:** `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d`
**Verdict:** **Not complete — 2.4/5 maturity with hard-gate failures**

## Executive assessment

Elspeth's DAG core is more mature than its end-to-end completeness posture suggests. The repository has a strong graph vocabulary, fail-closed structural validation, multi-source queue support, rich fork/coalesce semantics, and broad test coverage. The weakest part is not drawing or compiling a graph; it is proving that the graph remains correct under durable failure, replay, multi-process contention, secret handling, and every advertised authoring surface.

The main conclusion is therefore:

> **The DAG is structurally capable but not yet production-complete as a whole product.**

The immediate work should restore durable invariants and fencing before expanding the authoring surface. A parity suite and updated contract should then make completeness measurable rather than narrative.

## What is already strong

- Seven canonical node types: source, queue, transform, gate, aggregation, coalesce, and sink.
- Multiple source roots and explicit queue fan-in.
- Clear distinction between queue coordination and coalesce row joining.
- Structural checks for cycles, roots/sinks, reachability, ordinary-node fan-in, edge labels, and route metadata.
- Four coalesce policies and three merge strategies with schema-aware materialization.
- Production-path evidence for non-terminal coalesce continuing to a downstream gate.
- Queue-aware YAML import and export.
- Property, integration, audit, and bounded multi-worker test slices.

These strengths justify a 3–4 score for the graph/compiler layer. They do not offset hard failures in durable runtime and security boundaries.

## Current scorecard

| Area | Score | Verdict |
| --- | ---: | --- |
| Graph vocabulary and topology | 4 | Strong |
| Structural validation | 4 | Strong |
| Schema/cardinality | 3 | Capable with known contract gaps |
| Compositional closure | 3 | Basic coalesce→gate works; full production matrix incomplete |
| Runtime happy path | 3 | Broad but scenario-fragmented |
| Recovery/concurrency | 1 | Hard blockers |
| Audit consistency | 2 | Rich evidence, incomplete atomicity |
| Authoring parity | 2 | Freeform capable, guided incomplete |
| Security of graph config | 1 | Hard blocker |
| Scale/maintained contract | 1–2 | Missing complete envelope and CI matrix |

## Scenario gap matrix

Legend: **Pass** = current evidence supports production-path behavior; **Partial** = some layers pass but recovery/parity or a composition is missing; **Fail** = known missing capability or correctness defect; **Unknown** = no adequate evidence.

| Scenario | Build/contracts | Runtime | Recovery/concurrency | Authoring/round-trip | Overall |
| --- | --- | --- | --- | --- | --- |
| Linear transform | Pass | Partial | Partial | Partial | Partial |
| Multiple independent sources | Pass | Partial | Partial | Partial | Partial |
| Multi-source queue fan-in | Pass | Partial | Partial | Partial | Partial |
| Conditional routing | Pass | Partial | Partial | Partial | Partial |
| Fork to sinks | Pass | Partial | Partial | Partial | Partial |
| Fork + coalesce policies | Pass | Partial | Fail/Partial | Partial | Partial |
| Sequential fork→coalesce→fork | Partial | Unknown | Unknown | Fail/Unknown | Partial |
| Parallel coalesces | Partial | Partial | Unknown | Fail/Unknown | Partial |
| Aggregation | Partial | Partial | Fail/Partial | Fail/Unknown | Partial |
| Row expansion | Pass | Partial | Fail/Partial | Fail/Unknown | Partial |
| Cross-variant row union/interleave | Fail | Fail | N/A | Fail | Fail |
| Error/quarantine/retry/discard | Pass | Partial | Unknown | Partial | Partial |
| Sink write and redrive | Pass | Partial | Fail | Partial | Fail |
| Checkpoint/resume | Partial | Partial | Unknown | N/A | Partial |
| Multi-worker execution | Pass | Partial | Fail/Unknown | N/A | Fail |

No mandatory scenario currently earns a maintained-contract score of 5, and several fail the production-support threshold of 4.

`N/A` has three narrow meanings in this seed matrix:

- row union/interleave has no supported runtime construct, so recovery and
  concurrency do not apply after the scenario already fails build/runtime;
- checkpoint/resume is a runtime lifecycle behavior rather than an authored
  topology surface; and
- worker multiplicity is deployment/runtime configuration rather than DAG
  authoring or round-trip semantics.

**Framework adoption note (2026-07-15):** this seed snapshot predates the
normalized framework in `../../assessment-framework.md`. Some
`Build/contracts` Pass cells reflect current source and test inventory rather
than an exact executed production-path row. Under the permanent framework, those
cells are unproven until exact evidence is attached and must be recorded as
Partial/Unknown in the next assessment if they are not rerun. The legacy 2.4
indicator is not numerically comparable with later 15-dimension assessments.

## Priority gaps

### G1 — Restore the scheduler subtype truth table (P1, stop-ship)

**Owner surface:** State engine / Landscape scheduler
**Evidence:** `elspeth-f8f9272b68`, `elspeth-d8e172676c`

Wave 1 reproduced two invalid transitions:

- normal dispositions accept sink-redrive work and can retain stale sink metadata;
- `claim_pending_sink` leases a malformed row with no pending sink name.

**Shore up:** centralize subtype predicates; require the complete sink bundle during claim; reject incompatible disposition verbs; make rejected transitions transactionally inert; add database constraints where possible.

**Exit gate:** every claim and disposition matches the versioned state/subtype truth table, and negative tests prove zero row, lease, event, or branch-loss mutation.

### G2 — Make all multi-worker mutations fail closed (P1, stop-ship)

**Owner surface:** Scheduler fencing and worker coordination
**Evidence:** `elspeth-e66c371acb`, `elspeth-b68bf5c161`, `elspeth-c25bcf5717`

The common fencing helper and legacy barrier wrappers can treat a missing coordination token as permission to use a plain write transaction. Queue initial claim also has a membership-fence gap.

**Shore up:** split explicitly unfenced legacy/test writes from production fenced verbs; require a typed coordination token for protected mutations; make token absence an invariant error; prove stale leaders and losing workers cannot mutate.

**Exit gate:** no production scheduler/barrier mutation has an implicit unfenced path, and real processes prove late/stale workers are harmless.

### G3 — Add database-owned idempotency for joins, expansion, batches, and calls (P1)

**Owner surface:** Landscape durable data-flow repository
**Evidence:** `elspeth-2172918fb7`, `elspeth-a25e9c009e`, `elspeth-74a343d5ad`, `elspeth-1ec0772662`, `elspeth-8a540d3324`

Open verified defects show that:

- two writers can coalesce the same parents without a claim/CAS or unique parent-set key;
- `record_parent_outcome=False` can leave expanded children while the parent remains reprocessable;
- artifact idempotency keys are stored but not enforced;
- call-index allocation is process-local;
- batch membership can mutate after closure.

**Shore up:** add unique keys, conditional inserts, CAS/claim rows, and mutable-status predicates inside one write transaction. Reserve durable identities before external side effects.

**Exit gate:** replay and contention return the same durable identity or a clean conflict; they never mint duplicate effective work.

### G4 — Make state and its explanation one atomic contract (P1)

**Owner surface:** Landscape audit and execution persistence
**Evidence:** `elspeth-3335de38c2`, `elspeth-4003f7993a`, `elspeth-322c417d23`, `elspeth-d8d4d2849b`, `elspeth-3a8cb4a1b8`

Verified defects include last-writer-wins output contracts, non-atomic token-outcome validation, routing events committed before reason materialization, a sidecar journal that can get ahead of the database, and token ownership that trusts a denormalized run ID.

**Shore up:** move read-check-write invariants into one `BEGIN IMMEDIATE` transaction or conditional statement; add versions/hashes for inferred contracts; store routing reasons before their event or use a recoverable pending state; make journal restore verify committed batches; add composite run/row/token ownership constraints.

**Exit gate:** fault injection cannot produce state without evidence, evidence without state, or cross-run ownership.

### G5 — Close the crash, stall, and registered-process proof matrix (P1)

**Owner surface:** Runtime/recovery QA and state-engine contracts
**Evidence:** `elspeth-aafba3b298`, `elspeth-7cdc4da434`, `elspeth-51a4b5c771`, plus the Wave 1 proof-package issues

Three high-risk seams are positively identified but unproved:

- source work commits before the source `COMPLETED` audit state;
- children enqueue before the parent disposition;
- a plugin call outlives both the lease TTL and worker stall budget.

**Shore up:** add deterministic process-death/fault injection at each durable boundary; restart from the same database; assert exact rows, tokens, work items, claim epochs, events, plugin calls, sink effects, and branch losses. Extend current direct contention to registered workers and sink redrive.

**Exit gate:** every seam has a deterministic expected state and restart test, with no timing-only proof.

### G6 — Separate secret-bearing runtime config from graph identity (P1 hard gate)

**Owner surface:** Core DAG + security/audit
**Evidence:** `elspeth-69c957ed96`, `elspeth-c4080bfb06`

Topology hashing consumes raw node configuration, creating an offline oracle or correlation surface. Separately, DAG metadata retains raw plugin configuration and can directly disclose credentials through audit exports, debug dumps, dashboards, or serialization.

**Shore up:** define one canonical public/redacted configuration contract; replace secrets with handles or keyed fingerprints; use it consistently for node metadata, topology identity, checkpoints, exports, errors, and audit records; fail closed when fingerprinting is required but unavailable.

**Exit gate:** regression tests prove common secret forms never appear in persisted or emitted graph surfaces, topology hashing receives only the redacted/fingerprinted representation, and equivalent secret references retain stable identity.

### G7 — Build the canonical production-path scenario matrix (P1)

**Owner surface:** DAG/compiler + QA
**Evidence:** current tests are split across direct-graph properties, builder units, integration flows, audit tests, and scheduler tests.

The direct topology tests are valuable, but a manually assembled graph does not prove configuration parsing, plugin assembly, schema propagation, traversal compilation, runtime, audit, or recovery.

**Shore up:** create one table-driven corpus that runs every mandatory scenario through parsing, production build, validation, traversal-plan snapshot, execution, audit, restart, and contention as applicable.

**Exit gate:** every scenario in [`03-completeness-model.md`](03-completeness-model.md) has an evidence row; no mandatory cell is unknown or plan-only.

### G8 — Deliver guided/freeform parity and browser acceptance (P2 after safety floor)

**Owner surface:** Composer
**Evidence:** approved Composer parity design, `elspeth-7cf763da7c`

Guided mode still materializes a linear transform chain. It cannot author multiple sources, routes, forks, queues, coalesces, aggregations, explicit edges, or multiple outputs. The planned nine-fixture × three-surface matrix and four seeded Playwright correctness specs are not implemented.

**Shore up:** implement full guided node/edge operations; compile guided and freeform inputs through the same production boundary; compare canonical graphs, not YAML formatting; enable deterministic Playwright topology, required-field, schema-preview, and export/import tests.

**Exit gate:** all 27 positive parity cases plus negative variants pass in CI, and every advertised surface produces the same canonical graph.

### G9 — Decide the row-union product contract (P2 product decision)

**Owner surface:** Product + DAG semantics
**Evidence:** remaining scope of `elspeth-a5b86149d4`

Queue fan-in interleaves independently scheduled rows; coalesce joins sibling branches of the same row. Neither provides a long-format row union/append construct for cross-variant batch statistics.

**Shore up:** either design a first-class row-union primitive with schema, ordering, identity, audit, recovery, and authoring semantics, or document that users must reshape upstream/outside Elspeth. Do not overload queue or coalesce with incompatible meaning.

**Exit gate:** one explicit supported contract exists and the A/B statistics scenario either passes end to end or is consistently rejected with guidance.

### G10 — Repair the normative contract and stale tracker state (P2)

**Owner surface:** Documentation + tracker governance
**Evidence:** `docs/contracts/execution-graph.md`, `elspeth-6421ffa028`, `elspeth-a6ca0bef77`

The execution-graph contract still says six node types, exactly one source, a singular builder facade, and no queue type. Tracker descriptions also lag queue support and likely lag coalesce-to-gate schema fixes.

**Shore up:** update the contract after the scenario matrix pins current semantics; replay stale issue reproducers; close or narrow tracker items; bind every normative claim to a test or invariant.

**Exit gate:** contracts, examples, current source, and live tracker agree on supported behavior.

### G11 — Consolidate graph and traversal compilation (P3 architectural hardening)

**Owner surface:** Core DAG + engine architecture
**Evidence:** `elspeth-d4e15aee36`, `elspeth-a2905d4964`, `elspeth-a1d9c01bad`

The builder owns many compilation phases, while traversal metadata is recomputed across `ExecutionGraph`, graph wiring, and `DAGNavigator`. This raises drift risk and contributes to plugin/node identity gaps.

**Shore up:** behind the completed scenario harness, split the builder into typed phases and emit one immutable traversal plan with one-to-one plugin binding checks.

**Exit gate:** topology changes have one compilation authority and one plan contract consumed by runtime and resume.

## Recommended sequence

| Wave | Objective | Gaps | Completion evidence |
| --- | --- | --- | --- |
| 0 — Safety floor | Restore fail-closed durable semantics | G1–G4, G6 | Negative invariants, DB constraints/CAS, fault-injection rollback, secret scans. |
| 1 — Recovery proof | Demonstrate restart and process safety | G5, G7 runtime columns | Full crash matrix and registered multi-process suite. |
| 2 — Product parity | Make supported capability equal across surfaces | G8–G10 | 27-case parity, browser CI, row-union decision, current contract/tracker. |
| 3 — Maintainability and scale | Reduce drift and declare limits | G11 plus scale gates | Single traversal compiler, topology/runtime benchmarks, release thresholds. |

Do not start with the builder refactor. First pin the durable safety floor and canonical scenario matrix; then refactor behind those contracts.

## Release gates for a “complete” claim

All of the following must be true:

- no open defect can lose, duplicate, cross-run, or ambiguously subtype work;
- every protected mutation requires and validates current ownership/fencing;
- state, audit event, routing reason, outcome, and journal evidence are transactionally consistent;
- graph metadata and identity expose no raw secret;
- every mandatory scenario passes build, runtime, audit, recovery, and concurrency columns;
- guided and freeform authoring produce equivalent canonical graphs;
- import/export round-trips semantically;
- browser correctness specs are active CI tests;
- the normative contract matches live code;
- supported scale limits and owners are documented;
- no mandatory cell is unknown, skipped, or plan-only.

## Evidence and limitations

This assessment used a fresh Loomweave index for the recorded commit, live Filigree issue data, current source/contracts/plans, and the focused test runs listed in [`02-capability-evidence.md`](02-capability-evidence.md). It distinguishes reproduced defects from unproved seams and stale tracker descriptions.

It did not execute every repository test, every third-party plugin combination, or the missing crash/parity matrices. Those cells remain partial or unknown. The shared execution host intermittently exhausted file descriptors during parallel evidence collection; the analysis switched to sequential workers, and only commands with observed results are reported.
