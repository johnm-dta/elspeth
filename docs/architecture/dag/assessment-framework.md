# DAG Completeness Assessment Framework

**Status:** Evergreen assessment procedure
**Last reviewed:** 2026-07-15

## Purpose

Use this framework to produce repeatable, evidence-backed DAG assessments. It
turns the criteria in [`completeness-criteria.md`](completeness-criteria.md)
into a fixed workflow, comparable scorecards, and durable dated snapshots.

## Required assessment metadata

Every assessment starts with a frozen baseline:

```markdown
**Assessment date:** YYYY-MM-DD
**Branch:** `<branch>`
**Commit:** `<full SHA>`
**Index state:** `<fresh status and analysis run ID>`
**Tracker state captured:** `<timestamp and query scope>`
**Assessment owner:** `<person or team>`
**Verdict:** `<Complete | Not complete | Insufficient evidence>`
```

Do not combine evidence from different commits without identifying each source.
If the working tree changes during evidence collection, either restart from the
new baseline or mark the affected cells unknown.

## Evidence hierarchy

Use evidence in this order:

1. Exact tests executed against the recorded commit.
2. Current source and generated or hand-written contracts that match it.
3. Live tracker records, including original reproducers and current status.
4. Approved designs and plans as evidence of intent only.
5. Historical notes only after re-verifying them against current state.

Record inspected-but-not-run tests as inventory, not as passes. A stale tracker
description does not override current source and executed tests, but it remains
open until its original acceptance condition is replayed or narrowed.

## Assessment workflow

### 1. Freeze and verify the baseline

- Record branch, full commit, working-tree state, date, and tool versions.
- Check the Loomweave index before structural queries; refresh it when stale.
- Capture live Filigree ready, blocked, and relevant issue state.
- Record environmental limitations such as skipped tests or resource exhaustion.

### 2. Inventory the product chain

Map the implementation and evidence owners for:

- authoring and configuration;
- graph model, builder, and validation;
- traversal compilation and plugin binding;
- runtime scheduling and sink delivery;
- durable state, recovery, and contention;
- audit, identity, export, and security; and
- contracts, tests, scale gates, and ownership.

The inventory prevents a graph-model assessment from being mistaken for an
end-to-end product assessment.

### 3. Build the capability manifest

List every supported node type, edge form, route label, source/sink cardinality,
fan-in rule, join policy, schema mode, failure destination, and authoring
surface. For each entry, identify the production builder boundary and the
runtime consumer.

### 4. Run the mandatory scenario corpus

Exercise each scenario from
[`completeness-criteria.md`](completeness-criteria.md#mandatory-scenario-corpus)
through the production path. Use a shared fixture corpus wherever possible so
parsing, building, runtime, audit, recovery, authoring, and browser tests do not
silently describe different graphs.

### 5. Probe the hard gates

Run focused negative and fault-injection evidence for:

- invalid durable subtypes and transaction rollback;
- replay/idempotency at joins, expansion, batches, calls, and sinks;
- claim ownership, lease expiry, stale-worker fencing, and process death;
- state/event/reason/outcome/journal atomicity;
- secret-bearing configuration in every persisted or emitted graph surface; and
- divergence between guided, freeform, import/export, and production validation.

### 6. Reconcile live issues

For every failure or stale claim:

- cite the Filigree issue when one exists;
- distinguish reproduced defects from evidence gaps and product decisions;
- replay the original reproducer before closing or narrowing stale issues; and
- define one observable exit gate.

### 7. Score without averaging away risk

Score each dimension and scenario cell independently. Compute a numerical
maturity indicator only as a navigation aid. The product verdict follows the
weakest mandatory cell and the hard gates.

For assessments created under this framework, calculate the maturity indicator
as follows:

1. Use the 15 dimensions in the scorecard below with equal weight.
2. Assign one integer score from 0 through 5 to each dimension; ranges are not
   valid scores.
3. Derive a dimension's score from its lowest applicable mandatory-scenario
   evidence. For a cross-cutting dimension, use the lowest evidence-backed
   maturity reached across its mandatory scope.
4. Exclude a scenario cell marked `N/A` only when the cell includes a reason.
   A mandatory dimension itself cannot be `N/A`.
5. Do not calculate an aggregate while any dimension is `U`.
6. Take the unweighted arithmetic mean and round to one decimal place.

The number remains subordinate to the hard-gate verdict. It cannot turn a
failed mandatory cell into a complete result.

The 2026-07-15 seed assessment predates this normalized 15-dimension
calculation. Its **2.4/5** value is retained as a legacy maturity indicator and
must not be compared numerically with later framework-conformant assessments.

### 8. Validate independently

Have a reviewer check:

- every Pass has an exact executed-evidence reference;
- each consequence follows from the cited defect;
- unknowns were not converted to passes by inference;
- scores are layer-local and internally consistent;
- relative links and diagrams render; and
- the final verdict follows the declared completion rule.

Record review findings and their dispositions in the dated assessment.

### 9. Publish and update the hub

- Create a new directory under `assessments/YYYY-MM-DD-HHMM/`.
- Preserve numbered synthesis reports, raw evidence, validation, and provenance.
- Update [`README.md`](README.md) to identify the latest assessment and verdict.
- Do not rewrite an older snapshot to describe later code. Add an explicit
  erratum when correcting the snapshot itself.

## Result vocabulary

Use the following status terms in matrices:

| Status | Meaning |
| --- | --- |
| Pass | Exact current evidence meets the cell's production-support requirement. |
| Partial | Some required layers pass, but the cell lacks complete applicable evidence. |
| Fail | A reproduced defect, missing mandatory capability, or contradictory behavior exists. |
| Unknown | Adequate evidence was not executed or does not exist. |
| N/A | The dimension genuinely does not apply; include a reason. |

Do not use Pass for "a nearby test exists," "source looks correct," or "the
plan says this will work."

## Seed-assessment adoption note

The assessment at `assessments/2026-07-15-1415/` was completed before this
evergreen framework was finalized. It remains the current seed snapshot because
its hard-gate findings and Not complete verdict are current for its recorded
commit. However:

- its 2.4 value is the rounded mean of 11 layer-local scores in the capability
  evidence, while the final report consolidates them into 10 display areas and
  includes one score range;
- some `Build/contracts` Pass cells rely on inspected source or nearby test
  inventory rather than an exact executed scenario row; and
- future assessments must treat those cells as unproven until exact evidence is
  attached or downgrade them to Partial/Unknown.

This exception preserves historical evidence; it does not relax the framework
for future assessments.

## Dimension scorecard template

| Dimension | Score | Status | Evidence | Open gate or next proof |
| --- | ---: | --- | --- | --- |
| Topology expressiveness | U | Unknown | | |
| Canonical configuration | U | Unknown | | |
| Structural validation | U | Unknown | | |
| Schema contracts | U | Unknown | | |
| Cardinality and identity | U | Unknown | | |
| Compositional closure | U | Unknown | | |
| Runtime semantics | U | Unknown | | |
| Durable recovery | U | Unknown | | |
| Concurrency and fencing | U | Unknown | | |
| Atomic evidence | U | Unknown | | |
| Security | U | Unknown | | |
| Authoring parity | U | Unknown | | |
| Semantic round-trip | U | Unknown | | |
| Scale | U | Unknown | | |
| Maintained contract | U | Unknown | | |

## Scenario evidence template

| Scenario | Config | Build | Contracts | Runtime | Audit | Recovery | Concurrency | Freeform | Guided | Round-trip | Scale | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear source → transform → sink | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Multiple independent sources | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Multi-source queue fan-in | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Conditional routing | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Fork to multiple terminals | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Fork and coalesce | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Sequential or nested forks and coalesces | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Parallel coalesces | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Aggregation | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Row expansion | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Row union/interleave | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Retry, quarantine, discard, and routed errors | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Sink write/redrive | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Checkpoint/resume | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |
| Multi-worker execution | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown | Unknown |

Add rows for new supported constructs. Do not remove a row merely because it
currently fails.

## Finding template

```markdown
### DAG-<sequence> — <short finding title> (<priority>)

**Type:** Reproduced defect | Evidence gap | Product decision | Contract drift
**Owner surface:** <subsystem>
**Status:** <confirmed | unconfirmed | stale claim | resolved>
**Evidence:** <issue IDs, files, exact commands, observed result>

<One precise statement of the gap and why it matters.>

**Exit gate:** <observable condition and required evidence>
```

Keep consequence and evidence separate. For example, raw metadata may disclose
a credential directly, while hashing raw configuration creates an oracle or
correlation surface; a hash alone is not evidence of direct disclosure.

## Executed-evidence ledger template

| Command or probe | Result | Establishes | Does not establish |
| --- | --- | --- | --- |
| `<exact command>` | `<observed result and duration>` | `<narrow supported claim>` | `<adjacent untested claims>` |

Include failures and aborted runs. Environmental failures explain an Unknown;
they do not turn it into a Pass or Fail for product behavior.

## Verdict rules

Use exactly one product verdict:

- **Complete:** every applicable mandatory cell is at least score 4, all hard
  gates are closed, and the evidence matrix is required in CI.
- **Not complete:** a mandatory cell fails, a hard gate is open, or an
  advertised surface is materially incomplete.
- **Insufficient evidence:** no hard failure is established, but one or more
  mandatory cells remain unknown.

When reporting a maturity score, always pair it with the verdict and open hard
gates. Never report the score as a percentage complete.

## Reassessment triggers

Create a new dated assessment when any of these occurs:

- a release branch or release candidate is cut;
- a node, edge, join, cardinality, schema, or traversal contract changes;
- a recovery, fencing, idempotency, or audit hard gate closes or regresses;
- an authoring surface gains or loses topology capability;
- the execution-graph contract changes materially; or
- supported topology or runtime scale limits change.

Small evidence additions may be appended to the current assessment with a dated
note. Changes that alter the baseline, verdict, or score require a new snapshot.
