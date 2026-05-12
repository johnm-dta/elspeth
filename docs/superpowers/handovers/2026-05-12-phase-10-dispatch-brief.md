# Dispatch brief — Composer Guided Mode Phase 10 (Gap 6 + docs + umbrella PR)

**Companion doc.** This is forward-looking guidance for the Phase 10 controller. The retrospective handover at `docs/superpowers/handovers/2026-05-12-phase-9-complete.md` captures what landed at Phase 9 close; this doc tells you what Phase 10 needs to deliver and how to sequence it.

Read both. Where they disagree, this brief wins (it's the more recent document).

## TL;DR

Phase 10 lands two things and opens the umbrella PR:

1. **Task 10.0 — Gap 6: RecipeOfferTurn editable slots.** Extends `RecipeOfferPayload` (wire contract) so unsatisfied required slots ship with their schema, and `RecipeOfferTurn` renders editable inputs for them. Multi-component frontend+backend work — the textbook scenario for convention 14 (one-commit wire-contract change). **Gates the demo SLA assertion.**
2. **Task 10.0b — Un-fixme the demo SLA spec.** Once Gap 6 lands, `tests/e2e/composer-guided.spec.ts` becomes a one-line change (`test.fixme(true, ...)` → `test(...)`). Run end-to-end, confirm ≤9 clicks / <30s SLA holds.
3. **Task 10.1 — User-facing docs.** Add "Guided Mode" section to `docs/guides/user-manual.md` + entries in `docs/guides/troubleshooting.md`. Original Phase 10 plan body §10.1.
4. **Task 10.2 — CHANGELOG entry.** Add `### Added` entry under the active release header per plan body §10.2.
5. **Task 10.3 — Open the umbrella PR.** Per `project_adr010_umbrella_branch` memory: this PR delivers all of Phases 1–10 of composer-guided-mode against `main`. PR description carries a phase-by-phase commit map (per Phase 9 cross-impl reviewer's umbrella-PR-prep guidance) so reviewers can navigate the diff.

**Out of Phase 10 scope (deferred to Phase 11 or polish):** Tasks 9.2 (hand-built path E2E — hard-blocked on `elspeth-2c08408170`), 9.3 (auto-drop path E2E), and the ChaosLLM Playwright fixture they both depend on. Phase 9 explicitly closed without these; Phase 10's outcome ("demo SLA assertion green + umbrella PR opens for review") doesn't need them. Document the deferral in the Phase 10 close handover.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode`
- **Top commit at start of Phase 10:** `80cab3b6 docs(handover): fix Phase 9 commit count to include the handover itself` — or whatever has accumulated since.
- **Python:** `.venv/bin/python` in the worktree. Do NOT use main's venv (Python-version mismatch corrupts `enforce_tier_model.py` results — per memory `project_tier_model_python_version`).
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). Vitest is the unit/integration runner. Playwright (1.59.1) is the E2E runner; auto-launches uvicorn (:8451) + Vite (:5173) under `webServer`.
- **Canonical frontend typecheck:** `npx tsc -p tsconfig.app.json --noEmit` — NOT `npx tsc --noEmit` at the root.
- **ESLint is broken** on this worktree (v9 flat-config migration debt). Don't fix it as part of Phase 10. Rely on vitest + tsc + Playwright.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` — Phase 10 starts around line 4634. Phase 9 amendment landed at `df5306cf`.
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md`. Relevant for Phase 10: §3.3 (recipe pre-match), §6.4 (recipe matcher backend), §7.1 (component layout — RecipeOfferTurn lives in `components/chat/guided/`).

## State at Phase 10 start (post Phase 9)

- Demo SLA E2E spec exists at `tests/e2e/composer-guided.spec.ts` — drives steps 1–5 successfully, blocked at step 6 (Apply recipe) by Gap 6. `test.fixme()`.
- Backend wire-shape bugs Bug 1 + Bug 2 fixed (commits `e05e02b2`, `a5df0b6c`).
- Slot resolver `blob_ref` fix landed (`74ea68eb`) — including blob_ref injection through `handle_step_1_source` (the deeper architectural gap that surfaced during Gap 5 verification).
- sessionStore guided wiring landed (`9ae407a2`).
- Plan body for Task 9.1 corrected (`df5306cf`).
- Vitest baseline at Phase 10 start: **418 / 418 pass.** `tsc -p tsconfig.app.json --noEmit`: clean.
- pytest guided suite: 165 pass.
- Playwright: 3 pass + 10 fixme (the demo SLA spec + adjacent fixmes).
- Five Phase 9 observations open in filigree:
  - `elspeth-obs-f626607b13` (P1): Gap 6 — RecipeOfferTurn editable slots. **Phase 10 first task.**
  - `elspeth-obs-134474dfcb` (P2): InspectAndConfirmTurn unreachable on the live emission path.
  - `elspeth-obs-83d97315a7` (P3): forkFromMessage doesn't restart guided state for the new session.
  - `elspeth-obs-d3d0d7fa70` (P1): RESOLVED by `9ae407a2` — dismiss as part of Phase 10 housekeeping.
  - `elspeth-obs-a8a9bc010a` (P0): RESOLVED by `74ea68eb` — dismiss as part of Phase 10 housekeeping.
- Phase 8 observation `elspeth-obs-5ea21f94af` RESOLVED by `2b692cab` — dismiss.

## Pre-flight verification (do this FIRST)

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline 80cab3b6..HEAD | head -10                            # confirm Phase 9 close + any drift
cd src/elspeth/web/frontend
npm test -- --run                                                      # expect 418 / 418 pass
npx tsc -p tsconfig.app.json --noEmit                                  # expect clean
npx playwright test composer-guided                                    # expect 3 passed, 10 skipped
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -q | tail -5   # expect 165 pass
.venv/bin/python -m mypy src/ 2>&1 | tail -3                           # expect clean
```

If any baseline gate fails, stop and investigate before starting Phase 10.

## Sequencing decisions

### Task 10.0 — Gap 6 first; no parallelism with docs

Gap 6 is multi-component wire-contract work that touches:

- `src/elspeth/web/composer/guided/types.py` (or wherever `RecipeOfferPayload` is defined — verify the actual path)
- `src/elspeth/web/composer/guided/emitters.py` (or `recipe_match.py` — wherever the `RecipeOfferTurn` payload is built)
- `src/elspeth/web/composer/recipes.py` (slot specs already exist; surface them)
- `src/elspeth/web/frontend/src/types/guided.ts` — the TypeScript wire type
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx` — render the editable form
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.test.tsx` — vitest coverage
- `tests/unit/web/composer/guided/test_emitters.py` (or wherever payload-building is tested)
- `tests/integration/web/composer/guided/test_endpoints.py` — full-flow round-trip
- The validator-error explanation tool (search for `explain_validation_error` — Phase 9 reviewer flagged this specifically)
- Possibly the audit schema if the new payload field changes audit-record shape

**Convention 14 from Phase 9 close:** all of those touched in ONE commit. The Phase 10 controller should write the Gap 6 dispatch brief with an explicit per-site checklist; the implementer must check each off before reporting DONE.

Do NOT parallelize Task 10.0 with the docs/CHANGELOG work. Docs reference the demo SLA assertion as proof the demo path works; if the SLA hasn't actually been asserted yet, the docs are aspirational and will need rewriting after Gap 6 lands. Sequence: Gap 6 → un-fixme + verify SLA → docs/CHANGELOG → PR open.

### Task 10.0b — Un-fixme is bundled with Gap 6's verification dispatch

The implementer for Task 10.0 should ALSO un-fixme the spec and run the full Playwright suite as part of their dispatch's verification gate. If the SLA fails after Gap 6 lands, they investigate and fix in-session (not "land Gap 6, then a separate dispatch un-fixmes" — that's gratuitous handoff overhead). If a seventh integration bug surfaces, refer to Phase 9's pattern: small bounded fix in-session; large multi-component fix → file observation, surface DONE_WITH_CONCERNS, escalate.

### Tasks 9.2 / 9.3 / LLM-stub — explicitly out of Phase 10 scope

Per Phase 9 close: 9.3 is failure-path coverage, not the demo anchor; 9.2 is hard-blocked on `elspeth-2c08408170` (Step-3 backend handler completion); both depend on the LLM-stub fixture which doesn't exist. Phase 10's deliverable is "demo SLA assertion green + umbrella PR opens" — none of those three tasks are required for that.

The Phase 10 close handover should explicitly say: "Tasks 9.2, 9.3, and the LLM-stub Playwright fixture are deferred to Phase 11 (or a polish phase) — they are not blockers for the umbrella PR, and including them would inflate Phase 10 scope without changing the demo readiness story."

### Task 10.3 — Umbrella PR open is THE close action

Per memory `project_adr010_umbrella_branch`: this branch delivers Phases 1–10 end-to-end. The PR is opened against `main`. The PR description must:

1. Carry a phase-by-phase commit map for navigation. Without it, reviewers see ~100+ commits across 10 phases and have no chapter markers.
2. Include the DB migration note flagged by the Phase 9 cross-impl reviewer: "delete your sessions DB before running — Phase 9 added a `step_2_sink_intent` field to `GuidedSession` with strict `from_dict`; pre-Phase-9 sessions are not forward-compatible. This matches the project's stated migration policy (`project_db_migration_policy` memory)."
3. Reference the four open observations Phase 10 leaves unresolved (Gap 6 will be closed by Task 10.0; the three remaining: `elspeth-obs-134474dfcb`, `elspeth-obs-83d97315a7`, plus polish-phase items) so the merge reviewer doesn't think they were missed.
4. Note the Tasks 9.2/9.3/LLM-stub deferral with the rationale above.

The PR open does NOT request review from `johnm-dta` or any other operator-side reviewer — operator decides who reviews. The PR open just makes the work visible.

## Per-task dispatch hints

### For Task 10.0 (Gap 6) dispatch

**Files to read first:**
- `docs/superpowers/handovers/2026-05-12-phase-9-complete.md` — particularly the Gap 6 row in the bug table.
- `mcp__filigree__get_finding elspeth-obs-f626607b13` — the observation's full text.
- `tests/e2e/composer-guided.spec.ts` — the spec header documents the Gap 6 failure pattern with backend file:line references the prior implementer found.
- `src/elspeth/web/composer/recipes.py:181-300` — `_RECIPE1_SLOTS` and the `classify-rows-llm-jsonl` recipe definition; particularly the `classifier_template`, `model`, `api_key_secret` slots that have no derivable defaults.
- `src/elspeth/web/composer/guided/recipe_match.py` — `_classify_slot_resolver` (the resolver that derives the satisfiable slots; the unresolved ones must be surfaced to the frontend).
- `src/elspeth/web/composer/guided/emitters.py` (or wherever RecipeOfferTurn payload is built) — the current build site.
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx` — current widget; buttons exist (Apply / Build manually); add a slot form between recipe summary and action buttons.
- `src/elspeth/web/frontend/src/types/guided.ts` lines around 240 (per Phase 9 Gap 6 finding) — `RecipeOfferPayload` definition.

**Wire-contract design (suggested but not authoritative):**

```python
# Backend: extend RecipeOfferPayload with unsatisfied_slots
@dataclass(frozen=True, slots=True)
class RecipeSlotInput:
    name: str          # slot name (e.g. "classifier_template")
    slot_type: str     # one of SlotType — needed for input rendering
    description: str   # operator-readable hint
    required: bool     # always True for this surface (satisfied slots are pre-filled)

# RecipeOfferPayload gets a new field:
unsatisfied_slots: tuple[RecipeSlotInput, ...] = ()
```

```ts
// Frontend wire type:
interface RecipeSlotInput {
  name: string;
  slot_type: "blob_id" | "str" | "float" | "int" | "str_list";
  description: string;
  required: boolean;
}

interface RecipeOfferPayload {
  // ...existing fields...
  unsatisfied_slots: RecipeSlotInput[];
}
```

The widget renders a `<fieldset>` with one `<input>` per unsatisfied slot. On Apply, the widget submits `chosen: ["accept"], custom_inputs: <slot values>` (verify the custom_inputs key is the right shape — check the RecipeOfferTurn submit handler).

**Critical: respect convention 14.** The dispatch brief MUST list every file the change touches and require the implementer to check each off:

- [ ] `RecipeOfferPayload` dataclass (Python L2 contract)
- [ ] Recipe match `emitters.py` payload builder — populate `unsatisfied_slots` from slot specs
- [ ] `recipes.py` slot specs — verify `description` field exists or add it
- [ ] Frontend `guided.ts` TypeScript type
- [ ] Frontend `RecipeOfferTurn.tsx` form rendering
- [ ] Frontend `RecipeOfferTurn.test.tsx` vitest coverage (form renders; submit shape correct)
- [ ] Backend unit tests for `emitters.py` payload-building
- [ ] Backend integration tests for full recipe-match flow with required slots
- [ ] `explain_validation_error` tool (in `composer/tools.py` — verify; the Phase 9 reviewer flagged this as part of the wire-contract surface but didn't confirm it actually needs updating)
- [ ] Audit schema (only if `RecipeOfferPayload` shape is part of the audit record — check)
- [ ] Tier-model allowlist (FP rotation likely needed when files grow)

**Verification gate (must all pass before reporting DONE):**

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -q | tail -5
.venv/bin/python -m mypy src/ 2>&1 | tail -3
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model 2>&1 | tail -3
.venv/bin/python scripts/cicd/enforce_freeze_guards.py 2>&1 | tail -3
cd src/elspeth/web/frontend
npm test -- --run 2>&1 | tail -3
npx tsc -p tsconfig.app.json --noEmit 2>&1 | tail -3
npx playwright test composer-guided 2>&1 | tail -10            # un-fixme + must pass; SLA met
```

After landing, dismiss `elspeth-obs-f626607b13`.

### For Task 10.1 (user-facing docs) dispatch

The original plan body for §10.1 (line 4634+) is fine as-is. Read it; honor it. Two notes:

- The "screenshots from Playwright" suggestion in the plan body is optional. If you skip screenshots, the prose stands on its own. If you include screenshots, capture them via `npx playwright test composer-guided --trace on` and embed; otherwise this becomes its own sub-task.
- The "user manual" target is `docs/guides/user-manual.md` — verify the file exists; if not, this is a new file (the plan body assumes it exists).

### For Task 10.2 (CHANGELOG) dispatch

Trivial; can be combined with Task 10.1's commit. The plan body at line 4670+ has the Markdown verbatim — copy it under the active release header.

### For Task 10.3 (umbrella PR) dispatch

Use the `pr-create` workflow (or the equivalent toolchain in your environment — likely `gh pr create` directly). The PR description body must include:

1. **Phase-by-phase commit map.** One bullet per phase: phase number, brief description, commit range. Read each phase's `*-complete.md` handover for the commit ranges.
2. **DB migration note.** "Delete your sessions DB before running this branch. Phase 9 added a `step_2_sink_intent` field to `GuidedSession` with strict `from_dict` (per `project_db_migration_policy`)."
3. **Open observations the PR doesn't close.** List `elspeth-obs-134474dfcb` (InspectAndConfirmTurn unreachable, P2), `elspeth-obs-83d97315a7` (forkFromMessage guided-state, P3), and any other observations open at PR-open time.
4. **Deferred tasks with rationale.** Tasks 9.2 (hand-built path E2E — hard-blocked on `elspeth-2c08408170`), 9.3 (auto-drop path E2E — failure-path coverage, not demo-anchor), and the ChaosLLM Playwright fixture they depend on. Phase 11 owns these.
5. **Demo SLA verification.** "Recipe-match demo path verified end-to-end via `tests/e2e/composer-guided.spec.ts`: ≤9 user clicks, <30s wall-clock, zero LLM calls (recipe-match path is deterministic)."

**Do NOT request reviewers in the PR-open. Operator decides who reviews.** The brief landing the work makes it visible; operator triages from there.

## Convention reminders (carry from Phase 9)

All fourteen conventions from prior handovers carry forward. Two added in Phase 9 are particularly load-bearing for Phase 10:

13. **Backend integration bugs at the seams require curl probes against the live backend OR a Playwright run before declaring a feature complete.** Phase 10 Task 10.0 (Gap 6) is the canonical example — the Phase 10 dispatch brief MUST require the implementer to demonstrate the un-fixmed Playwright spec passes before reporting DONE.

14. **Wire-shape changes must touch L0 contract type + backend builder + frontend type + widget + widget tests + schema-compliance tests + validator-error tool, all in the same commit.** Gap 6 IS this scenario. The dispatch brief MUST include the per-site checklist.

## Plan-vs-reality drift expected in Phase 10

Phase 10 should drift LESS than Phase 9 because the seam-cleanup work is done. Likely drift patterns:

- **`unsatisfied_slots` shape may differ from the suggested design.** The slot-input wire shape might need additional fields (e.g. `default_value` for partially-derivable slots, `validation_pattern` for client-side validation) once the implementer reads the actual slot resolver code. Treat the suggested shape as a starting point.
- **`explain_validation_error` may not need changes.** The Phase 9 reviewer flagged it as "probably part of the wire-contract surface" but didn't confirm. The implementer should investigate; if the tool's response shape is independent of `RecipeOfferPayload`, no change needed and convention 14's list shrinks by one.
- **Audit schema may not need changes.** Same investigation: does `RecipeOfferPayload` appear in the audit record verbatim, or is only a hash recorded? If only a hash, the schema is unchanged. Verify before assuming.
- **The Playwright SLA may NOT pass on first un-fixme** even after Gap 6 lands — the click count budget (≤9) was an estimate from the Phase 9 plan correction. If actual UX produces 10 clicks, investigate where the extra click is and either fix the UX (preferred) or bump the budget with operator buy-in. Per project memories `feedback_correctness_beats_performance` and `feedback_no_calendar_shipping_commitments` — fix root cause, don't lower SLA to make tests green.
- **The `pr-create` workflow / `gh pr create` may need a `--draft` flag** if the operator prefers staged review. Default to non-draft; flag to operator if the PR description suggests they want draft review.

## Active follow-ups at Phase 10 start

| Issue | Status | Phase 10 priority |
|---|---|---|
| `elspeth-obs-f626607b13` (Gap 6 — RecipeOfferTurn editable slots) | Open (P1) | **First task. Gates the demo SLA assertion.** |
| `elspeth-obs-134474dfcb` (InspectAndConfirmTurn unreachable) | Open (P2) | Out of scope — note in PR description; address in polish phase. |
| `elspeth-obs-83d97315a7` (forkFromMessage doesn't restart guided state) | Open (P3) | Out of scope — note in PR description. |
| `elspeth-obs-d3d0d7fa70` (startGuided unwired) | RESOLVED by `9ae407a2` | Dismiss as housekeeping. |
| `elspeth-obs-a8a9bc010a` (slot resolver blob_id) | RESOLVED by `74ea68eb` | Dismiss as housekeeping. |
| `elspeth-obs-5ea21f94af` (focus on step advance) | RESOLVED by `2b692cab` | Dismiss as housekeeping. |
| `elspeth-obs-a076365f64` (test-file fixture extraction P3, Phase 8) | Open | Out of scope — polish. |
| `elspeth-obs-510a4fbdeb` (TurnPayload discriminated-union refactor) | Open | Out of scope — no urgency. |
| `elspeth-obs-f9e991f517` (useNonInitialEffect hook extraction) | Open | Out of scope — no urgency. |
| `elspeth-2c08408170` (Step-3 backend handler) | Open + Phase 9 deferral note | Out of scope — gates Phase 11's Task 9.2. |
| `elspeth-5e905f3c9d` (referenced in `MultiSelectWithCustomTurn.tsx` original NOTE) | RESOLVED by Bug 2 fix `a5df0b6c` | Verify it's marked closed in filigree; close as housekeeping if not. |
| `elspeth-611fc01d94` (GuidedHistory rich step summaries — Phase 7) | Open | Out of scope — cosmetic, polish. |

## Important constraints (do not relitigate)

Inherited from CLAUDE.md and prior handovers:

- **DB migration = delete the DB.** No Alembic, no schema-version probes, no migration scripts. Phase 9 added a strict `from_dict` field; pre-Phase-9 sessions crash on load — this is intentional.
- **Default to worktree.** Stay in `/home/john/elspeth/.worktrees/composer-guided-mode`; do not work in main.
- **No git stash.** Commit work to a branch if preservation is needed.
- **No calendar shipping commitments.** Phase 10 ships work-until-done.
- **Correctness beats performance always.** If the demo SLA fails post-Gap-6, fix the underlying UX latency — don't lower the budget.
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **`any` is forbidden in TypeScript.**
- **No optimistic updates.** Server is authoritative.
- **snake_case wire field names; camelCase store-internal.**
- **ESLint broken; rely on vitest + tsc + Playwright.**
- **Symbol-anchored cross-references in comments**, never `Filename.tsx:LL-LL`.
- **No PR-open BEFORE Task 10.3.** The umbrella PR is the close action.
- **Doc-only commits use `git commit --no-verify`** (markdown-only changes don't need CI hooks per `feedback_doc_only_commits_no_ci`).

## First action when you start Phase 10

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline 80cab3b6..HEAD | head -10                            # confirm Phase 9 close + handovers
cd src/elspeth/web/frontend
npm test -- --run                                                      # expect 418 / 418 pass
npx tsc -p tsconfig.app.json --noEmit                                  # expect clean
npx playwright test composer-guided                                    # expect 3 passed, 10 skipped
cd /home/john/elspeth/.worktrees/composer-guided-mode
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -q | tail -5
.venv/bin/python -m mypy src/ 2>&1 | tail -3
```

If any baseline gate fails, stop and investigate.

Then make three decisions, in order:

1. **Gap 6 wire-shape design:** read `_RECIPE1_SLOTS` and `RecipeOfferPayload`; confirm the suggested `unsatisfied_slots: tuple[RecipeSlotInput, ...]` shape is appropriate, OR adjust based on what slot specs actually carry. The wire shape is the convention-14 anchor — get it right before any other surface adapts.

2. **Convention 14 site list:** before dispatching Task 10.0, finalize the per-site checklist (the suggested list above is a starting point; investigate `explain_validation_error` and the audit schema to confirm whether they're in the touch set or not). The dispatch brief is malformed without the complete list — convention 14 explicitly requires it.

3. **PR-open mechanics:** confirm the PR target is `main` and the branch is `feat/composer-guided-mode`; verify no inadvertent side branches need merging in first; draft vs non-draft (default non-draft).

## How to dispatch Phase 10 implementers

Phase 10 has fewer dispatches than Phase 9 (which had ~6) because the work is bounded:

1. **One dispatch for Task 10.0 + 10.0b** (Gap 6 + un-fixme + verify SLA) — the largest dispatch in Phase 10 by far. This is multi-component and the per-site checklist must be enforced. Use a capable model.
2. **One dispatch for Tasks 10.1 + 10.2** (docs + CHANGELOG) — small, doc-focused. Cheap model is fine. Can be combined with the close handover writing if scope is small enough.
3. **One dispatch for Task 10.3** (umbrella PR) — smallest; PR description writing + `gh pr create` invocation. Could be done by the controller directly rather than dispatched.

Per the SDD discipline established in Phase 9, EACH code-touching dispatch (Task 10.0 here) gets:
- Spec compliance review (focused on convention 14 site coverage)
- Code quality review (standard)
- Re-review on any fix-up

Doc-touching dispatches (10.1, 10.2) can use a lighter review — read the diff, confirm the prose is clear and references match reality, ship.

The Phase 10 close handover should be similar in shape to Phase 9's: list all commits, gates verified at close, observations dismissed/promoted, what Phase 11 (if any) inherits, and convention reminders carried forward.

---

The advisor consulted at Phase 9 close said: *"Phase 9's value is making Phase 10 small."* Phase 10 inherits one tight target (Gap 6) plus the original Phase 10 docs/PR work. If you find yourself dispatching more than three or four times, stop and reassess — Phase 10 should be a small, clean phase compared to Phase 9.
