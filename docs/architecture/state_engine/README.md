# State Engine

This directory is the sole canonical entrypoint for Elspeth's durable state
engine architecture, completeness bar, proof inventory, assessment procedure,
current verdict, and historical assessments.

## Current verdict

| Field | Current value |
| --- | --- |
| Assessment | [2026-07-18 18:20 AEST](assessments/2026-07-18-1820/README.md) |
| Code baseline | `release/0.7.1` at `3c782ac3c7efb0550495be38f75800eddffa639a` |
| Catalog | `elspeth-state-engine-v1`, schema 1, 68 legs, 10 dimensions |
| Verdict | **Not complete** |
| Strongest evidence | TS-02 source-completion atomicity and strict legacy-image recovery; strict fencing; pending-sink admission; TS-07–TS-10 disposition images; barrier atomicity; built-in sink-effect recovery |
| Primary gaps | Long plugin/effect calls beyond leases; child-enqueue/parent-disposition seam; aggregation continuation; RC-04/RC-07 repository predicates; source exclusion breadth; production follower and read-model matrices |

The July 2026 implementation is materially stronger than the 2026-07-15
snapshot. The six concrete hard defects recorded there and the named
TS-02/source-completion seam are fixed at the current baseline. The engine
still fails the completion bar because mandatory crash/restart, concurrency,
plugin-boundary, read-model, and maintenance cells remain unresolved. Closing
one seam does not close the repository-wide atomicity or restart hard gates.

Do not reuse the historical `3 Confirmed / 15 Gap` count. The v1 catalog closes
68 legs across ten dimensions, so that older denominator is not comparable.

## Authority and precedence

Use this order when documents disagree:

1. Current source and freshly executed tests establish observed behavior.
2. [Completeness criteria](completeness-criteria.md) define the claim bar.
3. [Current architecture](architecture.md) describes the maintained model and
   known durable seams.
4. The [v1 proof catalog](proof-catalog/v1/catalog.json) defines the finite leg,
   dimension, case, and hard-gate universe.
5. The current dated `assessment.json` binds evidence and findings to one code
   baseline.
6. Filigree owns live work status, assignment, priority, and dependencies.
7. Older dated assessments preserve only their baseline-bound conclusions.

No other document is canonical for current state-engine status. In particular,
`docs/architecture/token-scheduler-state-engine.md` is a deprecated pointer.

## Start here

- [Architecture](architecture.md) — token scheduler, sink-effect, barrier,
  fencing, read-model, and transaction-boundary model.
- [Proof matrix](proof-matrix.md) — human-readable current result and open proof
  themes. The JSON catalog and dated manifest remain the machine authorities.
- [Completeness criteria](completeness-criteria.md) — status vocabulary,
  dimensions, hard gates, and completion tiers.
- [Assessment program](assessment-program.md) — exact reproducible procedure
  for full, delta, and historical rerun assessments.
- [Assessment framework](assessment-framework.md) — evidence and
  classification rules used while interpreting results.
- [Proof catalog](proof-catalog/README.md) — schema, stable IDs, and promotion
  rules.
- [Assessment history](assessments/README.md) — immutable baseline records and
  current pointer policy.

## Directory model

```text
docs/architecture/state_engine/
├── README.md
├── architecture.md
├── proof-matrix.md
├── completeness-criteria.md
├── assessment-program.md
├── assessment-framework.md
├── proof-catalog/
│   ├── README.md
│   └── v1/catalog.json
├── templates/
│   ├── assessment-readme.md
│   ├── verification-run.md
│   └── review-record.md
└── assessments/
    ├── README.md
    └── YYYY-MM-DD-HHMM/
        ├── README.md
        ├── assessment.json
        ├── evidence.md
        ├── review.md
        ├── nodes/              # retained exact node IDs when needed
        └── artifacts/          # retained JUnit/stdout/stderr when executed
```

Future assessments stay small. Add raw artifacts only when they preserve a
material fact that Git, a command vector, or an output digest cannot recover.
Do not add remediation checklists; create or update Filigree issues instead.

## When to reassess

Run a full assessment when any of these change:

- the state or subtype vocabulary;
- a scheduler, sink-effect, barrier, fencing, or read-model contract;
- transaction boundaries or restart choreography;
- supported database, worker, plugin, or deployment profiles;
- the proof catalog, hard gates, or completion semantics.

Run a delta assessment for a bounded implementation or evidence change. A
delta may update named cells but cannot declare the whole engine complete.

Update this page only after the new assessment passes its direct validation and
independent review. Never silently rewrite an older assessment to describe new
code.

## Assessment history

| Date | Baseline | Mode | Verdict | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-18 | `3c782ac3c` | Full contract refresh and source-ingress evidence | Not complete | Adds the exact source `COMPLETED` witness to TS-02/PB-01, records strict pre-fix reconciliation, and attaches 13 fresh focused checks without promoting the broader legs or hard gates. |
| 2026-07-18 | `422415009` | Full framework reset and conservative evidence import | Not complete | Introduces the 68-leg v1 catalog, explicit coordination state, sink-effect architecture, reproducibility contract, and reviewed authority model. |
| 2026-07-15 | `0dcd61ac` | Seed assessment | Not complete | Historical 18-leg Wave 1 result; useful evidence, obsolete denominator and blocker list. |
