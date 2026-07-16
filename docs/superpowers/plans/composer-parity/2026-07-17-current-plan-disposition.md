# Composer Capability Parity — Current Plan Disposition

**Status:** Product gap remains; 2026-07-13 execution package retired
**Checked against:** `release/0.7.1` at `16017e9fe898702c8c6f23f013f266c31a4c14fb`
**Controlling issue:** `elspeth-7e2dd67275`

## Decision

Do not execute the seven-plan package dated 2026-07-13. Its review was valid
for an older code baseline, but its implementation mechanics are no longer
valid for the release branch. The package remains as historical design input;
it is not approved work.

The discarded `plans/composer-parity-review-fixes` worktree contained plan
edits only. It contained no implementation or test suite to merge.

## Why it was retired

- The reviewed code baseline was `a1b2b5a39`; this review found the release
  branch 140 commits beyond that baseline.
- The current constants are `SESSION_SCHEMA_EPOCH = 28`,
  `GUIDED_SESSION_SCHEMA_VERSION = 7`, and `SQLITE_SCHEMA_EPOCH = 27`. The old
  package assigns epoch numbers that have already been consumed.
- Durable sink-effect and coalesce-effect ledgers now own recovery and artifact
  identity. Plan 01's proposed parallel operation-parent lifecycle would
  duplicate or bypass that machinery.
- Proposal persistence, profile-aware splice behavior, and authoritative-review
  reconciliation have moved since the original review and must be reused.
- The old deployment plan folds independent release-programme machinery into
  the Composer feature and pins admission to obsolete candidate identities.

## Product requirement retained

Guided Composer still uses `ChainProposal`, `PROPOSE_CHAIN`, and a linear-chain
solver. It still cannot author every pipeline graph that freeform Composer can.
The controlling feature therefore remains open.

The following invariants survive re-planning:

- interaction style is the only intended mode distinction;
- guided and freeform use one canonical `set_pipeline` language;
- `PipelineProposal` is an approval/audit envelope around exact canonical
  arguments, not another topology model;
- one shared planner and audited commit seam serve every authoring surface;
- guided stores reviewed facts and deferred intent, not a partial DAG IR;
- the two-LLM colour pipeline remains a useful parity acceptance fixture;
- before 1.0, incompatible stores are uninstalled, discarded/recreated, and
  reinstalled; no in-place migration, compatibility reader, or backfill is
  built.

## Required re-plan

When the feature is started, write a new plan from the then-current release:

1. Re-characterize current proposal persistence, splice/reconciliation seams,
   policy/profile contracts, and the guided chain path.
2. Introduce the shared planner/commit seam by extending existing proposal
   persistence; first route a freeform vertical slice without a schema change.
3. Replace guided state, protocol, backend, and frontend atomically. Allocate a
   new session epoch only if the persisted schema changes; derive it from the
   live constant and recreate state rather than migrating it.
4. Add the real-path parity corpus, wrong-stage intent tests, tutorial identity
   tests, and a refreshed colour-pipeline proof.
5. Merge the green work back into the active release branch. Branch signing is
   the final release action, not a prerequisite for planning or implementation.

Coalesce failure routing, empty-output artifact evidence, and public typed LLM
query configuration remain useful findings, but they must be assessed against
current ownership boundaries and tracked separately when they are not required
by the smallest Composer parity slice.
