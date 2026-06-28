# P3 (interpretation surfacing) — go/no-go review

**Verdict: GO (with warnings).** Execution-ready against `release/0.7.0`.
**Reviewed:** 2026-06-24, plan == HEAD `4586a673c` (clean tree).
**Method:** 9-lens reality-weighted multi-agent review (6 reality/fidelity verifiers + architecture/quality/systems) + synthesis, run against the live tree.

This is a re-review of a hardened plan: P3 already absorbed a prior **NO_GO** round
(fixes committed `24895e692`) and the F1/F21/F22/F23 reconcile (`86424fee1`). All four
prior dispositions verified still-correct against the tree; none re-opened.

## What GO means here
The plan's anchors are **real** (every named symbol/signature/semantic resolves — line
numbers drifted but are advisory), the **writer-boundary per-kind semantics hold** for the
canonical guided flows, every embedded **test fixture constructs** against the on-disk
types, and the **strategy is complete** (surface-at-commit advisory layer + run-time hard
gate). It does **not** mean the tests will pass — this is a TDD plan, so pass/fail is an
execution-time outcome. One P3.5 reviewer ran the plan's exact backstop test into the tree
and got `1 passed`.

## Discriminating constraint
The verdict rests on the **writer-boundary linchpin**: `create_pending_interpretation_event`
is strict-per-kind for invented_source/prompt_template/pipeline_decision and falls through a
no-check else-branch for model_choice + bare vague_term — verified claim-by-claim. Had that
been wrong, the surfacer's per-kind dispatch breaks and this is a NO_GO. It held.

## No blocking issues.

## Warnings (apply during execution; none blocks starting)

- **W1 — writer-boundary, model-only node (substantive).** The surfacer's `LLM_MODEL_CHOICE`
  precondition is necessary-but-not-sufficient: the writer else-branch also requires a
  non-empty `prompt_template` (via `_find_llm_transform_node`) and raises an
  `InterpretationResolveError(ValueError)` otherwise. The surfacer checks `model` but not
  `prompt_template`, does **not** wrap the writer call in try/except, and the persist seam
  (`guided.py` ~`:1219`) has no outer `except` — so a model-set/empty-prompt_template node
  would 500 and, since state is already persisted, wedge the session on retry. NOT reachable
  via canonical guided flows (chain-accept/recipe-apply co-stage prompt_template+model; the
  wizard has no model-only edit) and untested → **latent, not live**.
  **Fix:** wrap the `create_pending_interpretation_event` call in the surfacer loop in
  `try/except ValueError` (or `except InterpretationResolveError`) → log + skip. Makes the
  docstring's promised advisory polarity real and closes model_choice + pipeline_decision +
  any future strict-kind gap at once. Add one test: model + staged `llm_model_choice` req +
  empty/absent `prompt_template` → surfacer does not raise, surfaces no model_choice event.

- **W2 — blast radius (P3.2 Step 0).** Mutating the BASE `composer_test_client` conftest
  fixture (`composer_service=None` → real impl) activates the surfacer across ~11 inheriting
  test files. The "no-op when no pending requirements" claim holds only for source/sink-only
  states; any sibling test driving a state with staged LLM requirements gains unexpected
  pending events. `test_progressive_disclosure` deliberately used a SEPARATE fixture to avoid
  this. **Fix:** prefer a dedicated fixture (mirror `composer_freeform_client`); if keeping
  the base mutation, treat P3.2 Step 4's `pytest -k 'guided or respond'` run as the gate and
  name the specific risk (surfacer creating unexpected events on staged-requirement states).

- **W3 — floating promises (P3.6 Step 1).** Making `refreshInterpretationEventsForSession`
  async leaves three existing freeform callers (`sessionStore.ts` ~`:645/:753/:1040`) as
  floating promises. Not gate-enforced today (no eslint in §9.2). **Fix:** add an explicit
  instruction to prefix those three call sites with `void`.

- **W4 — inverted red phase (P3.6 Step 5).** The ChatPanel disabled-button test is sequenced
  AFTER its Step 4 implementation, so "run to fail before Step 4" is counterfactual (test
  passes immediately). **Fix:** reorder the test ahead of Step 4, or relabel Step 5 as a
  pin-confirmation (the honest framing P3.5 already uses).

## Notes (cosmetic / optional)
- `test_surfacer_skips_bare_vague_term` comment "the writer boundary would reject a bare
  vague_term" is FALSE — the writer would ACCEPT it; the skip is correct because the surfacer
  returns None *before* calling the writer. Comment-rationale fix only.
- `compose.py:13` imports `ComposerService` via a `.._helpers` re-export, not directly from
  `protocol`; the plan's prescribed direct import in `guided.py` is valid and cycle-free.
- P3.5 PERMIT-failure diagnostic should also mention `evaluate_execution_fanout_guard` (runs
  before `create_run`), not only `validate_pipeline`.
- P3.7 correctly omits the slot-type cross-language gate (P3 changes no `guided.ts`/SlotType);
  add one "N/A for P3" sentence so the §9.2 scoping is explicit rather than a silent omission.
- Optional: factor the `pending && user_approved` + byCreatedAt projection into a shared
  selector reused by `TutorialTurn2bShowBuilt` and the new `GuidedInterpretationReviews`.

## Reviewer rollup
- reality/P3.1 anchors: GO — import-state claim exactly right.
- reality/writer-boundary (linchpin): GO_WITH_WARNINGS — semantics verified; W1.
- reality/P3.2 route+protocol+conftest: GO.
- reality/P3.5 run-tier backstop: GO — ran the test → 1 passed.
- reality/P3.6 frontend anchors: GO — 23-field InterpretationEvent matches fixture 1:1.
- reality/constructor fidelity: GO — every fixture constructs.
- architecture: GO_WITH_WARNINGS — single-seam sound, no one-way doors; W2, W3.
- systems: GO — stale-event benign (worst case = extra resolvable card, not deadlock);
  refresh-reject doesn't wedge submit; surfacer out-of-transaction + idempotent, fails safe.
- quality: GO_WITH_WARNINGS — pass-counts 6/7/9 correct; tier placement correct; W4 + gate note.

Full reviewer JSON: workflow `wf_bca1dd93-96f` (+ backfill `wf_185d6e61-f03`).
