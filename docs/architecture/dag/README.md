# DAG Information and Completeness

This directory is the permanent home for Elspeth's DAG completeness criteria,
assessment method, current verdict, and dated evidence. It is designed to
mature through repeated assessments without erasing earlier conclusions.

## Current status

| Field | Current value |
| --- | --- |
| Assessment | [2026-07-15 14:15 AEST](assessments/2026-07-15-1415/04-dag-completeness-gap-analysis.md) |
| Baseline | `release/0.7.1` at `0dcd61acaa44082d93ec205683700e798748ee6d` |
| Verdict | **Not complete** |
| Legacy maturity indicator | **2.4/5** — not a percentage-complete estimate or a future comparison baseline |
| Strongest layers | Graph vocabulary and structural validation |
| Weakest layers | Durable recovery/concurrency, secret-safe graph identity, authoring parity, and maintained cross-surface proof |

The current assessment concludes that Elspeth's DAG is structurally capable but
not production-complete as a whole product. Hard-gate failures remain in durable
subtype handling, fencing, idempotency, atomic evidence, and secret-bearing
configuration surfaces.

## Start here

- [Completeness criteria](completeness-criteria.md) — the stable definition,
  mandatory dimensions, hard gates, and scenario corpus.
- [Assessment framework](assessment-framework.md) — the repeatable workflow,
  evidence rules, scorecards, and templates.
- [Current gap analysis](assessments/2026-07-15-1415/04-dag-completeness-gap-analysis.md)
  — the latest verdict and prioritized shore-up sequence.
- [Current capability evidence](assessments/2026-07-15-1415/02-capability-evidence.md)
  — consolidated proof, limitations, and tracker reconciliation.
- [Current completeness model](assessments/2026-07-15-1415/03-completeness-model.md)
  — the assessment lifecycle, rating scale, and scenario definition used by the
  current snapshot.

## Current hard-gate themes

The current assessment groups the immediate blockers into five themes:

1. Restore valid durable subtype transitions and fail-closed claims.
2. Require fencing for every protected multi-worker mutation.
3. Add database-owned idempotency for joins, expansion, batches, and calls.
4. Commit state and its audit explanation atomically.
5. Keep raw secrets out of topology identity, metadata, exports, and diagnostics.

After that safety floor, the assessment calls for a production-path scenario
matrix, guided/freeform parity, browser acceptance, an explicit row-union
decision, and repair of the normative execution-graph contract.

## Directory model

```text
docs/architecture/dag/
├── README.md
├── completeness-criteria.md
├── assessment-framework.md
└── assessments/
    └── YYYY-MM-DD-HHMM/
        ├── 00-coordination.md
        ├── 01-discovery-findings.md
        ├── 02-capability-evidence.md
        ├── 03-completeness-model.md
        ├── 04-dag-completeness-gap-analysis.md
        ├── evidence/
        └── provenance/
```

The evergreen files may evolve as the product's quality bar matures. Dated
assessments remain point-in-time records for their stated commit. Correct a
snapshot with an explicit erratum; do not silently rewrite it to describe later
code.

## Assessment history

| Date | Baseline | Verdict | Evidence |
| --- | --- | --- | --- |
| 2026-07-15 | `release/0.7.1` at `0dcd61ac` | Not complete — 2.4/5 with hard-gate failures | [Assessment package](assessments/2026-07-15-1415/04-dag-completeness-gap-analysis.md) |

The 2026-07-15 package retains all 13 files from the original same-day analysis.
Supporting evidence and independent validation live under `evidence/`; the
worker briefs remain under `provenance/` so the analysis can be reconstructed.
It is the seed assessment: its 2.4 score and broad `Build/contracts` Pass cells
predate the normalized framework and are not comparison baselines for later
assessments.

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
- Dated assessments report whether the current implementation meets that bar.

When these surfaces disagree, the assessment records the contradiction as a
gap. It does not quietly choose the most convenient source.
