# Phase 10 close — Composer Guided Mode (Gap 6 + docs + umbrella PR)

**Status: complete.** Phase 10 delivered the three things the dispatch brief required: (1) Gap 6 RecipeOfferTurn editable slots with the demo SLA E2E un-fixmed and passing, (2) user-facing documentation, and (3) the umbrella PR is open against `main`. No deferrals required; Tasks 9.2 / 9.3 / ChaosLLM Playwright fixture remain explicitly out of scope per Phase 9's close (carried forward in the PR description for the merge reviewer).

## TL;DR

**Demo SLA: GREEN.** `tests/e2e/composer-guided.spec.ts` passes at **9 clicks / 3.5s** (budget: ≤9 / <30s). The recipe-match happy path (CSV → classify-rows-llm-jsonl → JSON) is now end-to-end verifiable, deterministic (zero LLM calls), and assertable in CI.

**Phase 10 commits (4 substantive + this handover):**
| SHA | Title | Role |
|---|---|---|
| `c7cb7ee9` | feat(web/composer): RecipeOfferTurn editable slots; un-fixme demo SLA E2E | Gap 6 wire-shape change (Convention 14) + un-fixme |
| `f8ca63f9` | refactor(web/composer): tighten _RecipeSlotInput.slot_type to SlotType Literal | Code-review Important #1 — closes wire-schema/recipes.py drift gap |
| `d6ea2ca0` | docs(guides+changelog): composer guided mode | User-facing docs + CHANGELOG entry |
| _(this file)_ | docs(handover): Phase 10 complete | Phase 10 close |

**Gates at Phase 10 close:**
- pytest guided (`tests/unit/web/composer/guided/` + `tests/integration/web/composer/guided/`): **181 passed**
- mypy: **clean, 391 source files**
- enforce_tier_model.py check: **passed** (FPs rotated in `config/cicd/enforce_tier_model/web.yaml`)
- enforce_freeze_guards.py: **passed**
- vitest: **427 passed / 39 files**
- tsc -p tsconfig.app.json --noEmit: **clean**
- Playwright composer-guided: **4 passed / 9 skipped** (demo SLA test moved from skipped to passing)

## What landed (per task)

### Task 10.0 + 10.0b — Gap 6 + un-fixme demo SLA

Wire-shape change subject to Convention 14 (one commit, all sites). 15 files / +726 / -196 in `c7cb7ee9`, followed by the slot_type Literal tightening in `f8ca63f9`.

**Design:** Option (c) from the pre-dispatch advisor consult — `RecipeMatch` carries `unsatisfied_slots: Mapping[str, SlotSpec]`, populated inside `match_recipe` via `get_recipe(recipe_name).slots` minus the resolver's coverage. Emitter stays pure (no upward import to `recipes.py`).

**Touch list (17 sites, all visited):**

Backend:
1. `src/elspeth/web/composer/guided/recipe_match.py` — `RecipeMatch.unsatisfied_slots` field + `freeze_fields` extension; `match_recipe` derivation; offensive crash on missing recipe.
2. `src/elspeth/web/composer/guided/protocol.py` — `_RecipeSlotInput` TypedDict with `slot_type: SlotType` (after `f8ca63f9`); `RecipeOfferPayload.unsatisfied_slots`; `_REQUIRED_KEYS` updated.
3. `src/elspeth/web/composer/guided/emitters.py` — `build_step_2_5_recipe_offer_turn` projects `(name, SlotSpec)` → `_RecipeSlotInput` literal.
4. `src/elspeth/web/sessions/routes.py` — apply-time `_RecipeMatch(..., unsatisfied_slots={})` reconstruction (consumer reads only `recipe_name` / `slots`; comment-justified, observation filed — see below).

Frontend:
5. `src/elspeth/web/frontend/src/types/guided.ts` — `RecipeSlotInput` interface + `unsatisfied_slots` field on `RecipeOfferPayload`.
6. `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx` — controlled inputs, inline fieldset (no disclosure), `<label htmlFor>` accessibility, Apply disabled until every required slot is non-empty trimmed, merge on submit, **`type="text"` for `str` slots / `type="number"` for `int`/`float` — NEVER `type="password"`** (audit-trail integrity).
7. `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.test.tsx` — new tests: empty case, labelled inputs, disabled→enabled gate, merged submit, numeric type, **security pin (positive + negative assertion that `api_key_secret` renders `type="text"` not `type="password"`)**, hint rendering.

Tests:
8. `tests/unit/web/composer/guided/test_protocol.py` — literal updated + happy/missing-key validate_payload tests.
9. `tests/integration/web/composer/guided/test_step_handlers.py` — `RecipeMatch` literals updated (2 sites).
10. `tests/integration/web/composer/guided/test_respond.py` — `unsatisfied_slots` shape assertions for classify (3 entries) + split-threshold (2 entries).
11. `tests/integration/web/composer/guided/test_audit_emission.py` — no change required (its drive-helper already supplied all required slots).
12. `tests/unit/web/composer/guided/test_skill.py` — no change required (iterates TurnType values).
13. `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.test.tsx` — `RECIPE_OFFER_PAYLOAD` literal updated.
14. `src/elspeth/web/frontend/src/types/guided.test.ts` — no change required.
15. `tests/unit/web/composer/guided/test_recipe_match.py` — new `TestUnsatisfiedSlots` class (6 tests: classify 3 entries, split-threshold 2 entries, exclusion of resolver-filled + optional-with-defaults, SlotSpec metadata preserved, freeze contract).
16. `tests/unit/web/composer/guided/test_emitters.py` — NEW file, 7 tests pinning emitter projection.

E2E:
17. `src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts` — `test.fixme()` removed; three `.fill()` calls populate `classifier_template`, `model`, `api_key_secret`; header trimmed to "Gap 6 RESOLVED" summary.

**Sites verified NOT in the touch list:**
- `src/elspeth/web/composer/tools.py:_execute_explain_validation_error` — pattern-matches error text only; independent of `RecipeOfferPayload` shape.
- Audit schema — only `stable_hash(payload)` recorded; new payload field changes the hash but not the audit-record column shape. No DB migration in this change.

### Task 10.1 + 10.2 — User docs + CHANGELOG

Landed at `d6ea2ca0` (markdown-only commit, `--no-verify` per `feedback_doc_only_commits_no_ci`).

- `docs/guides/user-manual.md` (+148 / -1) — new "Web Composer: Guided Mode" section (~1100 words) covers: what guided mode is, when to use it vs. freeform, the three-step flow, the closed six-turn taxonomy table, recipe pre-match at Step 2.5, Step 3 chain proposer + auto-drop behaviour, completion / Save-and-exit / Drop-to-freeform-to-keep-editing controls, scope limits, and a See-also pointing at the design spec.
- `docs/guides/troubleshooting.md` (+94) — new "Web Composer — Guided Mode" subsection with three Cause/Solution entries: (a) auto-dropped to freeform → `solver_exhausted` diagnosis; (b) wizard disagreed with source schema → `inspect_and_confirm` edit; (c) recipe didn't appear → predicate-discriminator explanation + `list_recipes` workflow.
- `CHANGELOG.md` (+13) — new `[Unreleased]` section at the top (Keep-a-Changelog convention) with the verbatim plan-body text under `### Added`. The `[0.5.1]` section is preserved unchanged.

**Source-grounded technical writing discipline applied:** the docs-implementer grepped the actual frontend components (`ExitToFreeformButton.tsx`, `CompletionSummary.tsx`) for UI labels and the design spec for the six-turn taxonomy terms before writing prose. No invented features; no internal Phase numbers or filigree observation IDs leak into user-facing text.

### Task 10.3 — Umbrella PR open

PR #37 (https://github.com/johnm-dta/elspeth/pull/37) opened from `feat/composer-guided-mode`. Initially targeted `main`; retargeted to `RC5.2` at operator's request immediately after PR-open. Branch is 87 commits ahead of `RC5.2` / 0 commits behind, so retarget was clean (no merge resolution required). Phase-by-phase commit map (1–10) carried in the PR description for reviewer navigation; DB migration note (delete sessions DB), open observations, and deferred tasks (9.2 / 9.3 / ChaosLLM fixture) all surfaced.

No reviewers requested in the PR-open per dispatch-brief constraint — operator chooses.

## Observation housekeeping

| ID | Status | Action |
|---|---|---|
| `elspeth-obs-f626607b13` | RESOLVED by `c7cb7ee9` (Gap 6 wire shape) | Dismissed in this phase |
| `elspeth-obs-d3d0d7fa70` | RESOLVED by `9ae407a2` (sessionStore wiring) | Dismissed in this phase |
| `elspeth-obs-a8a9bc010a` | RESOLVED by `74ea68eb` (slot resolver blob_ref) | Dismissed in this phase |
| `elspeth-obs-5ea21f94af` | RESOLVED by `2b692cab` (focus on step advance) | Dismissed in this phase |
| `elspeth-obs-5e905f3c9d` | RESOLVED by `a5df0b6c` (Bug 2 sink intent persistence) | Verified closed in filigree |
| `elspeth-obs-06854f0842` | NEW (P3) — `RecipeMatch` lifecycle ambiguity at apply-time reconstruction | Filed in this phase; deferred to a future cleanup phase |
| `elspeth-obs-134474dfcb` | Open (P2) — InspectAndConfirmTurn unreachable on the live emission path | Carried forward; surfaced in PR description |
| `elspeth-obs-83d97315a7` | Open (P3) — forkFromMessage doesn't restart guided state for new session | Carried forward; surfaced in PR description |
| `elspeth-obs-a076365f64` | Open (P3) — test-file fixture extraction (Phase 8) | Carried forward; cosmetic |
| `elspeth-obs-510a4fbdeb` | Open — TurnPayload discriminated-union refactor | Carried forward; no urgency |
| `elspeth-obs-f9e991f517` | Open — useNonInitialEffect hook extraction | Carried forward; no urgency |
| `elspeth-611fc01d94` | Open — GuidedHistory rich step summaries (Phase 7) | Carried forward; cosmetic |
| `elspeth-2c08408170` | Open + Phase 9 deferral note | Carried forward; gates a future Task 9.2 |

## What's deferred (carried from Phase 9, not addressed in Phase 10)

Per Phase 9 close and the Phase 10 dispatch brief, the following are explicitly out of Phase 10 scope. The umbrella PR description surfaces this rationale for the merge reviewer:

- **Task 9.2 — Hand-built path (LLM-driven Step 3) E2E.** Hard-blocked on `elspeth-2c08408170` (Step-3 backend handler completion). Awaiting backend resolution.
- **Task 9.3 — Auto-drop path E2E.** Failure-path coverage, not demo-anchor. Depends on the LLM-stub Playwright fixture.
- **ChaosLLM Playwright fixture.** Required by 9.2 + 9.3. Not built — Phase 10 scope was strictly "demo SLA + umbrella PR."

These belong to a polish phase or a Phase 11 plan. Phase 10's outcome — "demo SLA assertion green + umbrella PR opens for review" — does not require them.

## Conventions reaffirmed in Phase 10

The fourteen prior-handover conventions all held. Two were exercised this phase:

- **Convention 13** (Playwright verification before declaring a feature complete): Task 10.0's verification gate explicitly required the un-fixmed `composer-guided.spec.ts` to pass before the implementer reported DONE. It did, at 9 clicks / 3.5s.
- **Convention 14** (wire-shape changes ship as one commit covering every touched site). Task 10.0 hit 14 of 17 sites in `c7cb7ee9`; sites 11, 12, 14 required no change (test fixtures that didn't construct the affected literal). The follow-up `f8ca63f9` is a separate forward-compat hardening commit, not a Convention-14 violation — it tightens a type annotation that doesn't change wire bytes.

No new conventions emerged in Phase 10. The phase was small and clean, as the Phase 9 close advisor predicted ("Phase 9's value is making Phase 10 small").

## Hand-off

The umbrella PR is open. Operator owns triage from here:

- **Code review** — operator nominates reviewers. Branch state: clean (no uncommitted work); 87 commits total against `RC5.2`.
- **Merge timing** — operator's call. PR is non-draft.
- **Post-merge cleanup** — sibling branch `feat/composer-progress-persistence-1a` (Phase 1B/1C/Phase-2-redaction work, per memory `project_phase1b_implementation_complete` / `project_phase1c_implementation_complete` / `project_phase2_implementation_complete`) is independent work not delivered by this PR.
- **Phase 11 / polish phase scope** — Tasks 9.2 / 9.3 / ChaosLLM fixture + the `elspeth-obs-06854f0842` `RecipeMatch` lifecycle refactor + the carried-forward observations. No commitments on timing.

The composer-guided-mode feature ships to `main` with this PR.
