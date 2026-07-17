# DAG Information and Completeness

This directory is the permanent home for Elspeth's DAG completeness criteria,
assessment method, current verdict, and dated evidence. It is designed to
mature through repeated assessments without erasing earlier conclusions.

## Current status

| Field | Current value |
| --- | --- |
| Assessment | [2026-07-18 03:19 AEST](assessments/2026-07-18-0319/02-scorecard-and-scenario-delta.md) |
| Baseline | `codex/dag-scenario-corpus` at `0235739274b534bd9e4e2b859bdd94a0b6a09651`, integrating `release/0.7.1` at `a5ec6e3d5` |
| Verdict | **Not complete** |
| Normalized maturity indicator | **Not calculated**; mandatory dimensions remain `U`, so the permanent framework forbids an aggregate |
| Strongest layer | Structural validation |
| Weakest layers | Guided authoring, secret-safe graph identity/audit, scale, and maintained contracts |

The current assessment concludes that Elspeth's DAG is structurally capable but
not production-complete as a whole product. Expansion replay identity,
output-contract serialization, and committed sidecar publication are now
closed. Hard gates remain in source and parent/child process-death proof,
registered-process and long-plugin contention, secret-bearing graph surfaces,
guided/browser parity, scale, and the normative contract.

## Change since the seed assessment

The 2026-07-15 snapshot remains immutable. The current assessment re-executed
the following post-seed work at a new baseline:

| Date | Assessment gap | Current result | Maintained evidence |
|---|---|---|---|
| 2026-07-17 | R1: normal dispositions accepted a reclaimed sink-redrive lease (`elspeth-f8f9272b68`) | Fixed: TS-07–TS-10 require the transform-lease subtype and preserve the complete durable image on refusal. | [`token-scheduler-state-engine.md`](../token-scheduler-state-engine.md#durable-subtype-admission-truth-table-v1) and the disposition truth-table regression matrix |
| 2026-07-17 | R2: malformed `PENDING_SINK` rows were claimable (`elspeth-d8e172676c`) | Fixed: TS-04 uses the complete bundle predicate in both diagnosis and CAS admission. | Pending-sink incomplete/legal/atomic-race parameterizations |
| 2026-07-17 | TS-07–TS-10 effects, guards, and rollback proof (`elspeth-1076e2716a`) | Verified: exact state, payload, lease, event, branch-loss, owner, membership, subtype, and rollback cells are maintained as one matrix. TS-10 also refuses malformed bundles at the writer boundary. | [Disposition follow-up evidence](../token-scheduler-state-engine.md#disposition-follow-up-executed-evidence--2026-07-17) |
| 2026-07-18 | Expansion replay identity (`elspeth-a25e9c009e`) | Fixed: batch expansion consumes one durable claim, sequential replay is refused, and concurrent PostgreSQL attempts create one child set. | [Integration delta](assessments/2026-07-18-0319/02-scorecard-and-scenario-delta.md#closed-hard-gates) and live `cardinality-identity-09/10/11` evidence |
| 2026-07-18 | Output-contract serialization (`elspeth-3335de38c2`) | Fixed: compatible observations merge under hash CAS and incompatible state fails closed across SQLite and PostgreSQL. | [Executed evidence](assessments/2026-07-18-0319/01-executed-evidence.md) |
| 2026-07-18 | Sidecar journal commit ordering (`elspeth-d8d4d2849b`) | Fixed: a committed outbox publishes and recovers idempotently without recording failed transactions. | [Executed evidence](assessments/2026-07-18-0319/01-executed-evidence.md) |

The reassessment confirms these closures. It also confirms selected fencing,
idempotency, and atomicity improvements, but no mandatory scenario yet passes
the entire production-support lifecycle.

## Start here

- [Live scenario corpus](scenario-corpus/README.md) — the authoritative
  15-scenario inventory, common evidence contract, executable cases, owned
  gaps, and promotion workflow.
- [Completeness criteria](completeness-criteria.md) — the stable definition,
  mandatory dimensions, hard gates, and scenario corpus.
- [Assessment framework](assessment-framework.md) — the repeatable workflow,
  evidence rules, scorecards, and templates.
- [Current scorecard and scenario delta](assessments/2026-07-18-0319/02-scorecard-and-scenario-delta.md)
  — the full current dimension/scenario view and the narrow integration changes.
- [Current executed evidence](assessments/2026-07-18-0319/01-executed-evidence.md)
  — exact integration commands, observed results, and limitations.
- [2026-07-17 full gap analysis](assessments/2026-07-17-1739/03-gap-analysis-and-remediation.md)
  — the last full discovery assessment; the 2026-07-18 delta supersedes its
  three now-closed hard-gate entries without rewriting the snapshot.

## Current hard-gate themes

The current assessment groups the immediate blockers into five themes:

1. Close source and parent/child process-death seams plus the remaining
   state/evidence atomicity proof.
2. Keep raw secrets out of graph identity, metadata, persistence, exports, and diagnostics.
3. Execute the deterministic crash and registered-process contention matrix.
4. Deliver guided/freeform/browser parity and an explicit row-union contract.
5. Repair and CI-bind the normative contract, scenario corpus, and scale envelope.

The maintained 15-scenario corpus (`elspeth-ef29ef6ba4`) is the acceptance spine
for all five themes.

## Directory model

```text
docs/architecture/dag/
├── README.md
├── completeness-criteria.md
├── assessment-framework.md
├── scenario-corpus/
│   ├── README.md
│   └── v1/
│       └── manifest.yaml
└── assessments/
    └── YYYY-MM-DD-HHMM/
        ├── numbered assessment reports
        ├── evidence/
        └── optional provenance/
```

The evergreen files may evolve as the product's quality bar matures. Dated
assessments remain point-in-time records for their stated commit. Correct a
snapshot with an explicit erratum; do not silently rewrite it to describe later
code.

## Assessment history

| Date | Baseline | Verdict | Evidence |
| --- | --- | --- | --- |
| 2026-07-18 | `codex/dag-scenario-corpus` at `02357392` | Not complete — three hard gates closed; row-expansion Recovery and Concurrency are Partial; mandatory dimensions remain `U` | [Integration delta](assessments/2026-07-18-0319/02-scorecard-and-scenario-delta.md) |
| 2026-07-17 | `release/0.7.1` at `6e8a6bf5` | Not complete — no aggregate while mandatory dimensions remain `U`; hard-gate failures are open | [Assessment package](assessments/2026-07-17-1739/02-scorecard-and-scenario-matrix.md) |
| 2026-07-15 | `release/0.7.1` at `0dcd61ac` | Not complete — 2.4/5 with hard-gate failures | [Assessment package](assessments/2026-07-15-1415/04-dag-completeness-gap-analysis.md) |

The 2026-07-15 package retains all 13 files from the original same-day analysis.
Supporting evidence and independent validation live under `evidence/`; the
worker briefs remain under `provenance/` so the analysis can be reconstructed.
It is the seed assessment: its 2.4 score and broad `Build/contracts` Pass cells
predate the normalized framework and are not comparison baselines for later
assessments.

The 2026-07-17 package is the first assessment to use the normalized
15-dimension framework. It correctly withholds an aggregate because mandatory
dimensions remain `U`; the seed's legacy 2.4 is not a comparison baseline.

The 2026-07-18 package is an integration-triggered delta with a complete
current matrix. It removes three closed defect gates, promotes only two
row-expansion cells, and retains the same **Not complete** verdict and
no-aggregate posture.

## How to iterate

1. Re-read the [criteria](completeness-criteria.md) and change them only when the
   intended quality bar changes.
2. Follow the [assessment framework](assessment-framework.md) against one fixed
   commit.
3. Create a new dated directory rather than overwriting an older baseline.
4. Preserve exact commands, observed results, limitations, review findings, and
   issue reconciliation with the snapshot.
5. Update this page's current-status block and history table.
6. Promote stable semantic decisions into the execution-graph contract or an
   architecture decision record; keep assessment evidence here.

## Authority boundaries

- [`../../contracts/execution-graph.md`](../../contracts/execution-graph.md) is
  the normative execution-graph contract.
- [`../adr/README.md`](../adr/README.md) indexes accepted architecture decisions.
- [`../token-scheduler-state-engine.md`](../token-scheduler-state-engine.md)
  records the durable scheduler state/evidence model.
- [`completeness-criteria.md`](completeness-criteria.md) defines the bar for a
  completeness claim.
- [`scenario-corpus/v1/manifest.yaml`](scenario-corpus/v1/manifest.yaml) is the
  evergreen status, evidence, ownership, exit-gate, and executable-case
  inventory for that bar.
- Dated assessments report whether the current implementation meets that bar.

When these surfaces disagree, the assessment records the contradiction as a
gap. It does not quietly choose the most convenient source.
