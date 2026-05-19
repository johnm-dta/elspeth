# Dispatch brief — Composer Guided Mode Phase 9 (Playwright E2E + demo SLA)

**Companion doc.** This is a forward-looking brief for the Phase 9 implementer. The retrospective handover at `docs/superpowers/handovers/2026-05-12-phase-8-complete.md` captures what landed at Phase 8 close; this doc tells you what Phase 9 needs to deliver and how to sequence it.

Read both. Where they disagree, this brief wins (it's the more recent document).

## TL;DR

Phase 9 lands three Playwright E2E flows plus the demo-SLA timing assertion that anchors the whole guided-mode feature:

1. **Task 9.1 — Recipe-match happy path.** CSV → JSONL via recipe pre-match. ≤7 clicks, <30s. SLA assertion at the bottom of the test.
2. **Task 9.2 — Hand-built path.** LLM-driven Step 3 via `<ProposeChainTurn>`. **HARD-BLOCKED** by `elspeth-2c08408170` (backend Step-3 handler is partial); cannot land meaningfully until backend completes. See §"Sequencing decisions" below.
3. **Task 9.3 — Auto-drop path.** LLM stub returns invalid chains twice; assert wizard auto-drops to freeform with `compositionState` intact.

Before any Playwright work, **strongly consider landing the focus-on-step-advance fix** (`elspeth-obs-5ea21f94af`). The demo-SLA assertion is structurally fragile without it — Tab cycling between steps inflates both click count and timing.

Phase 10 closes documentation + CHANGELOG + (per `project_adr010_umbrella_branch` memory) umbrella PR open. Phase 9 does **NOT** open a PR.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode`
- **Top commit at start of Phase 9:** `06faa8e8` (Phase 8 handover doc) — or whatever has accumulated since.
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results. The worktree venv is Python 3.13 (per `project_tier_model_python_version` memory).
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). `vitest` is the unit/integration runner. `playwright` is the E2E runner (likely; verify before starting — see §"Playwright infrastructure verification").
- **Canonical frontend typecheck:** `npx tsc -p tsconfig.app.json --noEmit` — NOT `npx tsc --noEmit` at the root.
- **ESLint is broken** on this worktree (v9 flat-config migration debt). Don't try to fix it as part of Phase 9. Rely on vitest + tsc for unit/integration; Playwright's own type-check for E2E.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` — Phase 9 starts line ~4560.
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` — Phase 9 relevant: §1.3 (demo SLA), §10 "Ring 3 + demo SLA".

## State at Phase 9 start (post Phase 8)

- ChatPanel `mode discriminator` is wired (Task 8.1, commits `a46f05c9` + `35850ace`).
- `role="log" aria-live="polite"` region wraps `<GuidedTurn>` in the guided-active branch (Task 8.2, commits `ff388128` + `23e7df5c` + `cdee530f`).
- Vitest baseline at Phase 9 start: **412 / 412 pass**. `tsc -p tsconfig.app.json --noEmit`: clean.
- Backend: unchanged from Phase 5 close. Step-3 handler in `src/elspeth/web/sessions/routes.py:2030-2137` consumes only `chosen: ["accept"]` (success) and `chosen: ["reject"]` (501 stub).
- Three Phase-5 follow-ups durably tracked in Filigree: `elspeth-5e905f3c9d`, `elspeth-2c08408170`, `elspeth-611fc01d94`. None resolved.
- Two new Phase-8 observations: `elspeth-obs-5ea21f94af` (P2 focus-on-step-advance) and `elspeth-obs-a076365f64` (P3 test-file fixture extraction).
- `elspeth-obs-0a1002de6d` is now structurally enforced by `ChatPanel.test.tsx` regression-pin test 3 — can be dismissed.

## Playwright infrastructure verification (do this FIRST)

The Phase 9 plan (line 4567) assumes `tests/playwright/composer-guided.spec.ts` is a new file to create AND assumes an "existing Playwright LLM-stub fixture" is available for Task 9.2. **Both assumptions are unverified.** Before dispatching any Phase 9 implementer, verify:

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode/src/elspeth/web/frontend
ls tests/playwright/                                       # does the directory exist?
cat playwright.config.* 2>/dev/null                        # is there a Playwright config?
grep -rn "page.route\|page.setExtraHTTPHeaders\|LLM.*stub\|stub.*LLM" tests/playwright/ 2>/dev/null
npx playwright --version                                   # is it installed?
```

Possible states:
- **(a) Full Playwright infra exists + LLM-stub fixture exists:** proceed straight to Task 9.1.
- **(b) Playwright infra exists, no LLM-stub fixture:** Task 9.1 (recipe-match path, no LLM needed) can proceed; Task 9.2 needs the fixture built first.
- **(c) No Playwright infra at all:** Phase 9 has an implicit Task 9.0a (set up Playwright) that the plan does not call out. Scope this explicitly before starting; the plan body has shown drift in every prior phase.

If you find state (c), pause and brief the operator before scaffolding Playwright unilaterally — it's a non-trivial dev-dep addition and may interact with the worktree's broken ESLint v9 migration.

## Sequencing decisions

### Recommended Task 0 — Focus-on-step-advance (`elspeth-obs-5ea21f94af`)

The Phase 8 cross-task reviewer flagged this as the most actionable Phase-9-blocking deferral. Reasoning:

- Spec §7.4 line 484 says: *"Maintain focus on the first interactive element of the new turn after step advance (reuse existing `useFocusTrap` patterns)."*
- Today, when `respondGuided` resolves and a new turn arrives, focus stays wherever the user was — typically the just-clicked button (now unmounted). Focus then falls to `<body>`.
- Task 9.1's demo SLA assertion is `expect(clicks).toBeLessThanOrEqual(7); expect(Date.now() - start).toBeLessThan(30_000);`. Without focus management, the user has to Tab through the document body to reach each new turn's first interactive — Tab counts as a keypress, not a click, but the **timing** budget will absorb the user-thinking time of "where did focus go?"
- Three implementation options are documented in the observation. Recommended: option (c) — a `useEffect` in `ChatPanel` keyed on `guidedNextTurn.step_index` that queries the log region for the first focusable element. Centralized, decoupled, doesn't require new widget contracts.

If you decide to skip this and address it post-Phase-9, document the decision in the dispatch brief for the Task 9.1 implementer so they know to budget extra Tab time in the SLA assertion (or to set up keyboard-driven E2E navigation differently).

### Task 9.1 — Recipe-match happy path (UNBLOCKED — safe to dispatch)

The plan body at line 4564 sketches the test. The path:
1. Create new session
2. Attach CSV blob
3. **Step 1 (source):** pick CSV via `SingleSelectTurn`; confirm columns via `InspectAndConfirmTurn`
4. **Step 2 (sink):** pick JSONL via `SingleSelectTurn`; declare required field "category" via `MultiSelectWithCustomTurn`
5. **Step 2.5 (recipe pre-match):** recipe found → `RecipeOfferTurn` → Apply
6. **Termination:** `CompletionSummary` → Save and exit

Backend dependencies for 9.1:
- Step 1/2/2.5 handlers all complete at Phase 5 close. ✅
- Recipe pre-match logic complete at Phase 2.3. ✅
- `CompletionSummary` wired at Phase 7 + ChatPanel integration at Phase 8. ✅

**No blockers.** Task 9.1 is the obvious starting point.

### Task 9.2 — Hand-built path (HARD-BLOCKED)

The plan body at line 4612 says: *"Use the existing Playwright LLM-stub fixture. Force pre-match to fail (e.g., DB sink), drive Step 3 through the chain proposal, accept, complete."*

**Blocker:** `elspeth-2c08408170`. The backend Step-3 handler in `routes.py:2030-2137` consumes only `chosen: ["accept"]` (success path) and `chosen: ["reject"]` (501 stub). The per-step Edit and Ask-advisor buttons remain absent from `ProposeChainTurn` widget UI (Phase 7 deferred them). A Playwright test of the hand-built path can only exercise Accept-all today.

Three options:
1. **Land `elspeth-2c08408170` first** (preferred per project memory `feedback_default_is_fix_not_ticket`). This is several backend tasks; consult the spec §3 Step 3 transforms and `routes.py:2030-2137` for the missing handlers.
2. **Rescope Task 9.2 to Accept-only.** Document that Reject/Edit/Ask-advisor paths are deferred. The Playwright test exercises only the success path of Step 3.
3. **Defer Task 9.2 entirely.** Phase 9 ships 9.1 + 9.3; 9.2 lands once backend completes.

The right choice depends on demo timing. If the demo needs the hand-built path to demonstrate LLM-driven chain proposal, option 1 is required. If the demo only needs to show that hand-built exists as a fallback (without full button surface), option 2 is sufficient. **Surface this decision to the operator before starting 9.2** — it's a sequencing call, not a technical one.

### Task 9.3 — Auto-drop path (UNBLOCKED)

The plan body at line 4623 says: *"Force LLM stub to return invalid chains twice; assert wizard auto-drops and freeform `<ChatInput>` appears with the partial pipeline state in `compositionState`."*

The auto-drop logic was finalized in Phase 5 (per `project_phase5_implementation_complete` memory: "auto-drop + progressive disclosure + audit-emission tests"). The discriminator's fall-through-to-freeform branch is verified by `ChatPanel.test.tsx` test 4 (Phase 8).

**Blocker:** depends on LLM-stub fixture availability (see §"Playwright infrastructure verification" state b/c). If the fixture exists, 9.3 is unblocked. If not, building the fixture is shared work with Task 9.2.

## Per-task dispatch hints

### For Task 9.1 dispatch

**Files to read first:**
- `docs/superpowers/plans/2026-05-11-composer-guided-mode.md:4564-4610` — plan body
- `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md:1.3` — demo SLA spec (≤4 user actions for source/sink/required-fields/recipe in <30s; the plan's "≤7 clicks" budget adds session-creation + blob-attach overhead)
- `tests/playwright/composer-guided.spec.ts` if it exists (it shouldn't yet; Phase 9 creates it)
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` — the discriminator surfaces and class hooks (`chat-panel--guided`, `chat-panel--completed`, `chat-panel-guided-log`)

**Plan-vs-reality drift to override:**
- The plan's `page.click('button:has-text("CSV")')` selectors may break if the SingleSelectTurn widget's button labels differ from "CSV" / "JSONL" — verify against `list_sources` / `list_sinks` MCP tool output OR the test fixtures in `src/api/client.guided.test.ts` for actual option labels.
- The plan's `page.fill('[placeholder*="custom"]', "category")` selector assumes the MultiSelectWithCustomTurn's custom input has placeholder containing "custom" — verify against `src/components/chat/guided/MultiSelectWithCustomTurn.tsx` (look for `placeholder=`).
- The plan's `expect(page.locator("text=Save and exit")).toBeVisible()` assumes CompletionSummary's button label is "Save and exit" — confirmed correct (verified in Phase 7 `CompletionSummary.tsx`).

**Demo-SLA budget reasoning:**
- Plan budget: ≤7 clicks. Source pick (1) + columns confirm (1) + sink pick (1) + required-field type+add (counts as 1 user action with the input + add button, but Playwright counts the button click as 1 — plus the `.fill` doesn't count as a click) + continue (1) + apply recipe (1) + save and exit (1) = 7. New session + blob attach are the slack.
- Plan budget: <30s. End-to-end including network round-trips. If the worktree's backend is running locally with no LLM call on the recipe-match path (deterministic pre-match), 30s is loose. If anything in the chain hits an LLM, you'll blow the budget.
- If `elspeth-obs-5ea21f94af` is unresolved, add Tab-cycle time to your mental budget — each step advance forces the user to navigate to the new turn's first interactive.

### For Task 9.2 dispatch (only if unblocked)

**Pre-condition:** `elspeth-2c08408170` landed (option 1) OR the dispatch brief explicitly rescopes 9.2 to Accept-only (option 2).

**Files to read first:**
- `docs/superpowers/plans/2026-05-11-composer-guided-mode.md:4612-4620` — plan body
- `src/elspeth/web/sessions/routes.py:2030-2137` — Step-3 backend handler current state
- `src/elspeth/web/composer/llm_chain_solver.py` — LLM chain solver (Phase 4 deliverable)
- `src/elspeth/web/composer/guided/state_machine.py` — guided state machine
- `src/components/chat/guided/ProposeChainTurn.tsx` — widget; note 3 deferred buttons

**LLM stub fixture pattern (if it exists):**
Look for `MockLLM`, `stubLLM`, `page.route('**/api/**llm**', ...)`, or similar in `tests/playwright/` or `tests/` more broadly. If you find a pattern, use it; if not, building one is non-trivial and should be a separate task.

**Force pre-match to fail:** the plan suggests `DB sink` as an example. Verify what sinks are in the deterministic recipe table (`src/elspeth/web/composer/recipes.py` or similar) and pick something NOT in it. CSV → DB sink, or JSONL → S3 sink, are likely candidates.

### For Task 9.3 dispatch

**Files to read first:**
- `docs/superpowers/plans/2026-05-11-composer-guided-mode.md:4622-4628` — plan body
- `src/elspeth/web/composer/guided/auto_drop.py` (or wherever auto-drop logic lives — search for `auto_drop` or `solver_exhausted`)
- `tests/integration/web/composer/guided/test_*auto_drop*.py` — Phase 5 auto-drop tests for assertion shape
- `src/components/chat/ChatPanel.test.tsx` test 4 (freeform fall-through) — what the post-drop state should look like

**Force LLM stub to return invalid chains twice:**
The auto-drop trigger (per spec §3.5 and §9.4) is `solver_exhausted` after a configured retry budget. Verify the budget config (likely 2 retries → 3rd failure triggers drop) and stub the LLM to return chains that the chain validator rejects (e.g., missing required slots, wrong shape, hallucinated plugin names).

**Assert partial state in compositionState:**
After auto-drop, the freeform `<ChatInput>` should appear (verified by Phase 8 discriminator test 4), and `compositionState.guided_session` should carry the audit trail of what got accomplished pre-drop (per spec §8.2 progressive disclosure: "do not re-run any work it already accomplished"). The test should fetch `compositionState` via the API after drop and assert non-null `guided_session` with completed turns.

## Convention reminders (inherited from Phase 8)

All eight conventions from the Phase 7 handover §"Convention reminders inherited from Phase 7" remain in force. Phase 8 added four more (carried forward to Phase 9):

9. **Three-branch discriminator order is completed-first → active-guided → fall-through.** Phase 9 should NOT add a fourth branch unless absolutely necessary; if it does, more-specific-before-less-specific precedence must hold.
10. **`chat-panel-guided-log` wraps `<GuidedTurn>` only.** No nesting of additional live regions inside (the "don't nest live regions" convention).
11. **Symbol-anchored cross-references in comments, never `Filename.tsx:LL-LL`.** Phase 9 will add a Playwright spec file with likely cross-references to widgets and store actions; use class names and function names.
12. **Discriminator-anchored test assertions over widget-DOM assertions.** Playwright tests for Phase 9 may be tempted to assert on widget internals because they're easier to grab; prefer asserting on the discriminator's class hooks (`chat-panel--guided`, `chat-panel-guided-log`) where possible.

## Plan-vs-reality drift expected in Phase 9

Phase 9 is Playwright E2E. The drift patterns most likely:

- **LLM-stub fixture assumed but not verified.** Plan asserts "existing Playwright LLM-stub fixture" — see §"Playwright infrastructure verification".
- **Selector specificity assumed in plan body.** Plan uses `button:has-text("CSV")` etc. — actual button labels may differ; verify before locking in.
- **Demo SLA budget assumed achievable.** The plan's ≤7 clicks / <30s assumes focus management works (otherwise budget the Tab cycles). Surface to operator if focus-on-step-advance is unresolved at Phase 9 start.
- **`elspeth-2c08408170` blocker not flagged in plan.** Plan body for Task 9.2 assumes a complete Step-3 backend; reality at Phase 9 start is partial. See §"Sequencing decisions".
- **`elspeth-611fc01d94` (rich step summaries) cosmetic gap.** Plan body for Task 9.1 doesn't assert on GuidedHistory's display content because the wire is hash-only. Demo can describe but not show "Step 1: CSV with cols [price, qty]" until backend extends `TurnRecordResponse`.

## Active follow-ups at Phase 9 start

| Issue | Status | Phase-9 impact |
|---|---|---|
| `elspeth-5e905f3c9d` (MultiSelect escape button wire) | Open | Task 9.1 doesn't exercise; Task 9.3 may need workaround if auto-drop scenario depends on Step 2 escape. |
| `elspeth-2c08408170` (Step-3 backend handler) | Open | **Hard blocker for Task 9.2.** |
| `elspeth-611fc01d94` (GuidedHistory rich summaries) | Open | Cosmetic gap visible in 9.1's recorded demo; not a Playwright test blocker. |
| `elspeth-obs-5ea21f94af` (focus on step advance) | Open (P2) | **Soft blocker for Task 9.1 SLA assertion stability.** Strongly recommend landing first. |
| `elspeth-obs-a076365f64` (test-file fixture extraction) | Open (P3) | Affects `ChatPanel.test.tsx` only; Phase 9 adds Playwright tests in a different file, so not direct. Still worth landing before Phase 9 grows the unit-test file further. |
| `elspeth-obs-510a4fbdeb` (TurnPayload discriminated-union refactor) | Open | No Phase-9 impact. |
| `elspeth-obs-f9e991f517` (useNonInitialEffect hook extraction) | Open | No Phase-9 impact. |
| `elspeth-obs-0a1002de6d` (suppress ExitToFreeformButton on completed) | Resolved by Phase 8 test pin; can be dismissed. | None. |

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** Phase 9 likely won't touch the DB, but if it does (e.g., adding an E2E-only seed fixture), no Alembic / no migration scripts.
- **Default to worktree.** Stay in `/home/john/elspeth/.worktrees/composer-guided-mode`; do not work in main.
- **No git stash.** Commit work to a branch if preservation is needed.
- **No calendar shipping commitments.** ELSPETH ships work-until-done.
- **Correctness beats performance always.** Even if the demo-SLA assertion fails because of Tab-cycle timing, fix the focus management — don't lower the SLA budget to make a flaky test green.
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **`any` is forbidden in TypeScript.** Playwright specs are TypeScript; same rule.
- **No optimistic updates.** Server is authoritative.
- **snake_case wire field names; camelCase store-internal.**
- **ESLint broken; rely on vitest + tsc.** Playwright's own type-check covers E2E specs.
- **Symbol-anchored cross-references in comments.**
- **No PR-open at end of Phase 9.** Phase 10's Task 10.3 opens the umbrella PR.

## First action when you start Phase 9

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline acf712d2..HEAD | head -10                # confirm Phase 8 close commits + handovers
cd src/elspeth/web/frontend
npm test -- --run                                          # expect 412 / 412 pass
npx tsc -p tsconfig.app.json --noEmit                      # expect clean
ls tests/playwright/ 2>/dev/null || echo "no playwright dir"
test -f playwright.config.ts && echo "playwright config exists" || echo "no playwright config"
npx playwright --version 2>/dev/null || echo "playwright not installed"
```

If any baseline gate fails, **stop and investigate** before starting Phase 9.

Then make four decisions, in order:

1. **Playwright infra:** state (a), (b), or (c) per §"Playwright infrastructure verification". If (c), brief the operator before scaffolding.
2. **Focus management:** address `elspeth-obs-5ea21f94af` first, OR proceed without it (with explicit Tab-budget acknowledgement in the Task 9.1 dispatch brief).
3. **Task 9.2 sequencing:** land `elspeth-2c08408170` backend completion (option 1), rescope 9.2 to Accept-only (option 2), or defer 9.2 entirely (option 3). Surface to operator.
4. **Task ordering:** with the above decisions, the most likely ordering is: Task 0 (focus) → Task 9.1 (recipe path) → Task 9.3 (auto-drop) → Task 9.2 (hand-built, if option 1/2). Auto-drop before hand-built because 9.3 is unblocked and 9.2 might still be waiting on backend.

## How to dispatch Phase 9 implementers

Phase 9 differs from Phases 6–8 in that the deliverables are **integration tests against a running backend**, not standalone frontend components. The implementer needs:

- A running ELSPETH dev server (typically `elspeth-web` service or `npm run dev`).
- A way to seed/reset the audit DB between tests (per project memory `project_db_migration_policy`: delete the DB rather than migrate it; the test harness can do this via the API or via direct SQLite file replacement).
- LLM stub control (per Task 9.2/9.3 — see fixture verification above).

The dispatch brief for each Task 9.X should explicitly tell the implementer:
- The Playwright test file path (`tests/playwright/composer-guided.spec.ts`)
- The selector strategy (data-testid > role > text — in that order of preference)
- The SLA budget reasoning (for 9.1)
- The expected backend state (running on which port; auth or no auth)
- The audit-DB-reset mechanism

Phase 8 used the subagent-driven-development skill with general-purpose implementers. Phase 9 may benefit from the same pattern; consider whether you want a Playwright-specialist agent if one exists in the skill marketplace.
