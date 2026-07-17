# Composer Capability Parity — Current Plan Disposition

**Status:** Product gap remains; 2026-07-13 plans archived and replaced
**Checked against:** `release/0.7.1` at `cc593f3a7ae29cc52d94bd82661fbdfb04e5fd81`
**Controlling issue:** `elspeth-7e2dd67275`
**Current plan:** [2026-07-17 implementation plan set](2026-07-17-composer-capability-parity-implementation-plan.md)

## Decision

Do not execute the seven plans dated 2026-07-13. They are preserved under
`retired-2026-07-13/` as historical input. The replacement plan set dated
2026-07-17 is current.

## Why it was retired

- The reviewed code baseline was `a1b2b5a39`; this review found the release
  branch 140 commits beyond that baseline.
- The current constants are `SESSION_SCHEMA_EPOCH = 28`,
  `GUIDED_SESSION_SCHEMA_VERSION = 7`, and `SQLITE_SCHEMA_EPOCH = 27`. The old
  plans assign epoch numbers that have already been consumed.
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

## Replacement plan

The current plan implements the required re-plan in this dependency order:

1. Re-characterize current proposal persistence, splice/reconciliation seams,
   policy/profile contracts, and the guided chain path.
2. Introduce the shared planner/commit seam by extending existing proposal
   persistence; first route a freeform vertical slice without a schema change.
3. Replace guided state, protocol, backend, and frontend atomically. Allocate a
   new session epoch only if the persisted schema changes; derive it from the
   live constant and recreate state rather than migrating it.
4. Add the real-path parity corpus, wrong-stage intent tests, tutorial identity
   tests, and a refreshed colour-pipeline proof.
5. Run deterministic, generated, frontend, and live staging proofs before
   closing the controlling issue.

The replacement deliberately reuses the current proposal lifecycle and omits
signed plan packages, plan hashes, review receipts, approval choreography, and
release-programme machinery. Runtime argument hashes, proposal/audit bindings,
schema sentinels, and test evidence remain because they protect code, data, and
execution integrity.

Coalesce failure routing, empty-output artifact evidence, and public typed LLM
query configuration remain useful findings, but they must be assessed against
current ownership boundaries and tracked separately when they are not required
by the smallest Composer parity slice.
