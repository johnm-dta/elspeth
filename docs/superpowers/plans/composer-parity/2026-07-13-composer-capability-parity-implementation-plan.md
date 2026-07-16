# Composer Guided/Freeform Capability Parity Implementation Plan Set

> **RETIRED EXECUTION PACKAGE (2026-07-17): DO NOT EXECUTE.** Preserve this as
> historical design input only; re-plan from the live release as directed by
> [the current disposition](2026-07-17-current-plan-disposition.md).

> **For implementers:** Use `superpowers:subagent-driven-development` or
> `superpowers:executing-plans`. Use `superpowers:test-driven-development` for
> behavior changes and `superpowers:verification-before-completion` before each
> slice is handed off.

**Goal:** Make freeform, guided-full, guided-staged, and tutorial-profile use one
complete pipeline language, one planner, and one audited commit path.

**Delivery posture:** ELSPETH is pre-1.0. This is a forward-only feature
replacement. Bump the session schema epoch, recreate stale session state, ship
one coherent implementation, and fix defects in that implementation. Do not
build a legacy reader, dual protocol, runtime architecture switch, or downgrade
path.

## Source of truth

- Implementation handoff: `docs/superpowers/plans/composer-parity/00-implementation-prompt.md`
- Design: `docs/superpowers/plans/composer-parity/2026-07-13-composer-guided-freeform-capability-parity-design.md`
- Live example design: `docs/superpowers/specs/2026-07-13-two-llm-colour-hybrid-pipeline-design.md`
- Live example run sheet: `docs/superpowers/plans/2026-07-13-two-llm-colour-hybrid-demo-run-sheet.md`
- Controlling Filigree issue: `elspeth-7e2dd67275`; its old instruction to
  expand `ChainProposal` is superseded by this canonical-proposal design.
- Related but separate row-union issue: `elspeth-a5b86149d4`.

```text
for every canonical set_pipeline payload p accepted by shared web policy:
  freeform_big_bang, guided_full, and guided_staged can independently derive,
  review, commit, and execute a graph-isomorphic pipeline for p.
```

Tutorial uses the guided-staged planner, schema, catalog, and discovery surface.
Its fixed lesson changes teaching flow only.

## Ordered run sheet

| Order | Plan | Exit result |
| --- | --- | --- |
| 01 | `2026-07-13-composer-capability-parity-plan-01-contract-foundation.md` | Canonical schema is complete and structurally locked; coalesce failure routing and secret-ref probes work |
| 02 | `2026-07-13-composer-capability-parity-plan-02-shared-planner.md` | One `plan_pipeline()` and audited commit seam serve the production freeform path |
| 03 | `2026-07-13-composer-capability-parity-plan-03-guided-replacement.md` | Epoch 28 and guided schema 8 replace the linear proposal, protocol, state, and renderer |
| 04 | `2026-07-13-composer-capability-parity-plan-04-guided-canonical-authoring.md` | Guided full/staged/tutorial entrypoints use the shared planner and handle wrong-stage intent safely |
| 05 | `2026-07-13-composer-capability-parity-plan-05-shared-capability-skills.md` | Big-bang and staged skill packs expose the same request-bound capabilities and typed LLM contract |
| 06 | `2026-07-13-composer-capability-parity-plan-06-parity-verification.md` | Real-path deterministic, property, mutation, repair, and tutorial identity gates pass |
| 07 | `2026-07-13-composer-capability-parity-plan-07-deploy-live-acceptance.md` | Staging state is recreated, one build is deployed, and all three ten-row colour proofs pass |

## Non-negotiable implementation rules

- The registered canonical `set_pipeline` declaration and
  `SetPipelineArgumentsModel` are the only topology schema authorities.
- Guided may store reviewed conversational facts; it may not store a second DAG
  model or translate a transform list into a graph.
- The production freeform route, authenticated guided-full seam, `/guided/start`,
  and tutorial all invoke the same `plan_pipeline()` implementation.
- Accepted proposal arguments reach the audited `set_pipeline` dispatch without
  topology translation. Invalid candidates never become current state.
- Inline content enters blob custody before proposal hashing. Checkpoints and
  audit projections contain no credential literals, resolved secrets, raw blob
  content, or raw provider/validation messages.
- A valid plugin requested too early is deferred with an explicit wait message
  and later consumed. A plugin for a completed stage triggers stable-id back/edit.
  Unavailable and ambiguous plugins remain distinct cases.
- Do not use freeform handoff, hidden YAML, recipes, or topology simplification
  to claim guided success.
- Remove the active `ChainProposal`/`PROPOSE_CHAIN`/`solve_chain` path. Do not
  retain it as compatibility code.
- Bump `SESSION_SCHEMA_EPOCH` 27 -> 28 with `GUIDED_SESSION_SCHEMA_VERSION` 7 ->
  8. Stale pre-release sessions are deleted/recreated per the existing runbook.
- Replace active runbook instructions that restore old source/data. Failure
  handling fixes the current implementation, recreates fresh state, redeploys,
  and reverifies; diagnostic archives are not service recovery points.
- Wardline is not a mandatory gate for this work under the operator's current
  direction. Preserve targeted boundary/security tests and record what ran.

## Slice protocol

For each plan:

1. claim or advance the controlling Filigree work atomically;
2. write the named failing test before each behavior change;
3. implement only the current slice and run its narrow tests;
4. run the slice regression command and `git diff --check`;
5. review the diff against the design and this run sheet;
6. commit the green slice with its evidence; and
7. update the evidence ledger before starting the next slice.

No issue closes until the final live evidence is green. A defect discovered in
scope is fixed in scope; it is not converted into an expiring observation.

## Evidence ledger

| Plan | Commit | Command/artifact | Result |
| --- | --- | --- | --- |
| 01 | — | contract, coalesce, secret-ref tests | pending |
| 02 | — | planner, custody, freeform entrypoint tests | pending |
| 03 | — | epoch/current-schema/backend/frontend replacement tests | pending |
| 04 | — | guided entrypoint, proposal lifecycle, deferral tests | pending |
| 05 | — | request-bound skill/catalog/schema identity tests | pending |
| 06 | — | 27-case matrix, generated DAGs, mutation and repair controls | pending |
| 07 | — | deployed revision, three live runs, Playwright, artifacts | pending |

## Acceptance traceability

| AC | Owning plan | Required evidence |
| --- | --- | --- |
| One active topology language and commit seam | 02-04 | AST/import guards, deep-equality commit capture |
| Complete guided topology capability | 04, 06 | nine fixture classes on three real adapters |
| Shared big-bang/staged capability surface | 05, 06 | planner, canonical/terminal schema, discovery tools, catalog assistance, and capability-core equality; each distinct message hash matches its own manifest/audit record |
| Tutorial is a profile, not a subset | 04-06 | planner/schema/discovery identity plus tutorial journey |
| Clean pre-release schema replacement | 03, 07 | epoch mismatch fail-close, runbook recreation, boot proof |
| Secret, blob, audit, and proposal custody | 01-03 | canaries, exact hash, rejection and restart tests |
| Wrong-stage plugin behavior | 04-06 | future deferral, restart, consume, back/edit, ambiguous/unavailable cases |
| Plain-English two-LLM split/merge | 07 | freeform, guided-full, guided-staged live proofs |

## Final completion gate

- [ ] All seven slices are committed and their evidence rows are green.
- [ ] No active or compatibility linear-chain authoring path remains.
- [ ] Epoch 28/current schema boots only with recreated session state.
- [ ] All three real authoring adapters pass all nine deterministic fixtures and
      generated valid DAGs.
- [ ] Tutorial planner/schema/catalog identity and its fixed journey pass.
- [ ] The exact ten-row two-LLM colour scenario passes freeform, guided-full, and
      guided-staged; guided-staged is executed with Playwright.
- [ ] The report distinguishes code-complete, deployed, and live-accepted state.
- [ ] The controlling Filigree issue closes only after every other criterion is
      satisfied; the temporarily removed Wardline gate is not used as a blocker.
