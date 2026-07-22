# DAG integration delta baseline and method

**Assessment date:** 2026-07-18 03:19 AEST
**Branch:** `codex/dag-scenario-corpus`
**Commit:** `0235739274b534bd9e4e2b859bdd94a0b6a09651`
**Integrated parent:** `release/0.7.1` at `a5ec6e3d50688605bef7215cbe58a704d8afaedb`
**Manifest:** `docs/architecture/dag/scenario-corpus/v1/manifest.yaml`, schema version 1
**Tracker state captured:** 2026-07-18 03:19 AEST, focused DAG hard-gate owners
**Assessment owner:** Codex
**Verdict:** **Not complete**

## Scope

This package is a new point-in-time assessment triggered by recovery,
idempotency, and audit hard-gate closures on the release branch. It preserves
the full 2026-07-17 assessment and evaluates only the integration delta needed
to reconcile that snapshot with the maintained scenario corpus.

The assessed merge commit combines the completed corpus foundation at
`065105a33e1e5b29ceee200e8257390c55c285f7` with the release bugfix sequence at
`a5ec6e3d50688605bef7215cbe58a704d8afaedb`. Before the merge, both worktrees
were clean. `git merge-tree` reported no conflict messages or overlapping
paths, and produced tree `40071f019230564b00e7e9785aebcd380ea85cae`.

## Method

1. Simulate and then perform the three-way merge in the isolated corpus
   worktree; do not modify the release worktree.
2. Execute the corpus contract and production-path suites.
3. Execute every pytest locator registered by the pre-delta manifest.
4. Re-run the adjacent recovery suite and the unit surfaces changed by the
   release bugfixes.
5. Opt into the PostgreSQL testcontainer lane for the exact expansion,
   output-contract, and sidecar-journal regressions.
6. Reconcile the live Filigree owners and change only cells directly supported
   by current executable evidence.
7. Preserve every unrelated scenario status and the **Not complete** verdict.

## Tooling limitation

Wardline and Warpline were being upgraded externally for worktree support
during this assessment. Warpline's available snapshot was 36 commits stale and
did not index the corpus branch, so its zero affected-entity result was not
treated as evidence of no impact. Git's three-way merge, exact repository
diffs, current tests, and live Filigree state are authoritative for this delta.

## Authority boundary

- The [2026-07-17 assessment](../2026-07-17-1739/02-scorecard-and-scenario-matrix.md)
  remains an immutable record of its baseline.
- The [live scenario manifest](../../scenario-corpus/v1/manifest.yaml) carries
  the reconciled evergreen cells and owners.
- [Executed evidence](01-executed-evidence.md) records what was run at this
  baseline.
- [The delta scorecard](02-scorecard-and-scenario-delta.md) records the current
  full verdict and the narrow matrix changes.
