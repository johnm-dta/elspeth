# Handover — Composer Guided Mode (Phase 8 complete; Phase 9 onward)

## TL;DR

Phase 8 is **complete and green**. The `ChatPanel.tsx` top-level mode discriminator is wired: a three-branch router (completed → `<CompletionSummary>`; active guided → `<GuidedHistory>` + `<GuidedTurn>` inside a `role="log" aria-live="polite"` region + `<ExitToFreeformButton>`; else → unmodified freeform body) plus a Phase 7 contract honoured (the parent-hosted aria-live region that `InspectAndConfirmTurn` documented as load-bearing). Five commits on top of `e28aedbe` deliver +8 frontend tests (404 → 412 cumulative), no critical or important reviewer issues at close, two new deferred-polish observations filed, three active Phase-5 cross-layer follow-ups still pending (unchanged from Phase 7 close), and zero unresolved blockers.

Phase 9 (Playwright E2E + demo-SLA assertions) is the next phase. Phase 10 closes documentation and CHANGELOG.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (67 commits ahead of `RC5.2` as of Phase 8 close; top commit `cdee530f`)
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results.
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). `vitest` is the test runner. The canonical typecheck is `npx tsc -p tsconfig.app.json --noEmit` — NOT `npx tsc --noEmit` at the root.
- **ESLint is broken** on this worktree due to v9 flat-config migration debt — not Phase 8's job to fix. Don't run `npx eslint src/`; rely on vitest + tsc for the per-task gate.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 9 starts line ~4560)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (Phase 9 relevant: §10 "Ring 3 + demo SLA")
- **Inbound handover:** `docs/superpowers/handovers/2026-05-12-phase-7-complete.md` — the Phase 8 brief; references back to Phase 6 close.

## What's delivered in Phase 8

Five commits on top of Phase 7 closure (`acf712d2` + handover doc `e28aedbe`):

### Per-task commit summary

| Task | Component | Commits | Notes |
|------|-----------|---------|-------|
| 8.1 | `ChatPanel` mode discriminator | `a46f05c9` (initial) + `35850ace` (fix-up) | Three-branch router with completed-first precedence. The plan body had five drift points (`<ExistingChatPanelBody />` pseudocomponent, Rules-of-Hooks violation, wrong prop name on `GuidedHistory`, missing `void` wrapper on `respondGuided`, missing `id="chat-main"` preservation); all overridden in the dispatch brief. Six new tests pin all three branches plus the obs-0a1002de6d regression invariant. Fix-up added discriminator-anchored `chat-panel--guided` class assertion (parallel to test 2's `chat-panel--completed` check) and rewrote the regression-pin comment to describe the correct failure mode. |
| 8.2 | A11y audit pass + `role="log" aria-live="polite"` region | `ff388128` (initial) + `23e7df5c` (cross-ref fix-up) + `cdee530f` (cross-ref completion sweep) | The audit pass surfaced one concrete a11y regression: `InspectAndConfirmTurn.tsx:39-46` documented a parent-ChatPanel `role="log" aria-live="polite"` region as a load-bearing contract for its `<aside>` warnings, but Task 8.1's discriminator had NOT honoured it. Wrapping ONLY `<GuidedTurn>` (not `<GuidedHistory>`, not `<ExitToFreeformButton>`) inside the region closes the contract. Two cross-ref fix-ups symbol-anchored five total stale `Filename.tsx:LL-LL` references across `ChatPanel.tsx`, `ChatPanel.test.tsx`, and `InspectAndConfirmTurn.tsx`. Spec §7.4 items 1-5 verified PASS with evidence; item 6 (focus on first interactive after step advance) deferred as `elspeth-obs-5ea21f94af`. |

### Gate state at close

- **Vitest:** **412 / 412 pass** (39 test files; +8 from Phase 7 baseline of 404)
- **`npx tsc -p tsconfig.app.json --noEmit`:** clean
- **Backend pytest** (narrow guided + sessions slice): unchanged from Phase 7 close (Phase 8 is frontend-only — backend file modifications are zero)
- **Backend mypy** on `src/elspeth/web/composer/` + `src/elspeth/web/sessions/routes.py`: clean (no Phase-8 changes)
- **ESLint:** not run — broken on this worktree (v9 flat-config migration debt); pre-existing infra issue unrelated to Phase 8

### Per-task review iteration count (data point for future planning)

| Task | Implementer dispatches | Why iteration was needed |
|------|------------------------|---------------------------|
| 8.1 | 2 (initial + fix-up) | Code-quality reviewer caught three Important comment-precision items: (a) discriminator block comment should add an explicit `exited_to_freeform` boolean walkthrough; (b) regression-pin test 3 comment misdescribed the failure mode ("three identical-action buttons" — actual: a button with literal label "Exit to freeform"); (c) test 1 should add a `chat-panel--guided` class assertion parallel to test 2's. All three landed mechanically with the recommended phrasing verbatim. |
| 8.2 | 3 (initial + 2 fix-ups) | Code-quality reviewer (Task 8.2) caught two Important stale-line-number cross-references in `InspectAndConfirmTurn.tsx:42` and `ChatPanel.tsx`'s new comment block; `23e7df5c` retargeted them by symbol. Cross-task reviewer (whole-Phase-8 final pass) then caught three more stale `Filename.tsx:LL-LL` references that survived the first sweep, including the very `InspectAndConfirmTurn.tsx:39-46` references the prior fix-up had introduced as outbound pointers; `cdee530f` closed the loop by symbol-anchoring all three. |

Pattern: every Phase 8 task landed with exactly one fix-up cycle per reviewer. The Task 8.2 cross-reference chain (initial → outbound retarget → inbound retarget) is a structural quirk worth flagging: when a fix-up replaces line-number references with symbol references, the *inbound* references to the moved symbol also become candidates for retargeting. Three sequential commits (`ff388128` → `23e7df5c` → `cdee530f`) all touching the same comment chain is unusually iterative but each step closed a discrete reviewer-named issue.

## End-to-end frontend capability (post Phase 8)

Phase 8 wires the Phase 7 widget surface into `ChatPanel.tsx`. After Phase 8:

1. The three-branch mode discriminator is live at the top of `ChatPanel`. When `guidedSession?.terminal?.kind === "completed"`, the chat surface is replaced by `<CompletionSummary terminal={guidedSession.terminal} />` inside `<div id="chat-main" className="chat-panel chat-panel--completed" aria-label="Pipeline summary">`.
2. When `guidedSession && !guidedSession.terminal && guidedNextTurn`, the chat surface is replaced by:
   ```tsx
   <div id="chat-main" className="chat-panel chat-panel--guided" aria-label="Guided composer">
     <GuidedHistory history={guidedSession.history} />
     <div className="chat-panel-guided-log" role="log" aria-label="Guided wizard step" aria-live="polite" aria-relevant="additions">
       <GuidedTurn turn={guidedNextTurn} onSubmit={(body) => void respondGuided(body)} />
     </div>
     <ExitToFreeformButton />
   </div>
   ```
3. Otherwise the existing freeform body renders unchanged (byte-identical to pre-Phase-8 — verified by structural diff).
4. The new `chat-panel--guided`, `chat-panel--completed`, and `chat-panel-guided-log` classes are placeholder CSS hooks — **no CSS rules added in Phase 8**. They exist for future styling and as testable discriminator-anchored signals.
5. The `id="chat-main"` skip-link target is preserved across all three branches.
6. The session-title header (lines 136-140 of pre-Phase-8 `ChatPanel.tsx`) is dropped in the guided/completed branches and retained in the freeform branch (the spec implies guided mode replaces the FULL chat surface; freeform keeps the session-title affordance).
7. `<GuidedTurn>` lives inside the `role="log" aria-live="polite"` region; `<GuidedHistory>` and `<ExitToFreeformButton>` live outside it. Rationale: historical context and persistent action buttons are not announced on turn change; only the live turn surface is.

The chat surface is now fully wired for guided mode. The only remaining frontend work is Phase 9's Playwright E2E coverage and Phase 10's documentation.

## What's pending — Phase 9 onward

Phase 9 lands three Playwright flows + the demo-SLA assertion test:

1. **Recipe-match happy path** (Task 9.1) — CSV → classify-rows-llm-jsonl in ≤7 clicks, <30s. Demo SLA assertion at the bottom of the test.
2. **Hand-built path** (Task 9.2) — LLM-driven Step 3 with stubbed LLM, drive `<ProposeChainTurn>` through Accept. **Blocked by `elspeth-2c08408170`** — the backend Step-3 handler only consumes `chosen: ["accept"]` (success) and `["reject"]` (501 stub); the per-step Edit / Ask-advisor buttons remain absent. A full hand-built path test that exercises Edit/Ask-advisor cannot land until backend completes.
3. **Auto-drop path** (Task 9.3) — Force LLM stub to return invalid chains twice; assert wizard auto-drops to freeform with partial pipeline state intact.

Phase 10 closes docs + CHANGELOG (Tasks 10.1–10.3). Tasks 10.1 and 10.2 are doc-only commits (`--no-verify` permitted per project memory `feedback_doc_only_commits_no_ci`). Task 10.3 runs the full sweep and opens the umbrella PR — see the project memory `project_adr010_umbrella_branch` for the umbrella-branch scope convention.

## Cross-layer follow-ups (active Filigree issues — unchanged from Phase 7 close)

Three durably-tracked issues for Phase 5 backend completion work that surfaced during Phase 7 widget implementation. **All three remain open at Phase 8 close.** Phase 8 did not touch any of them (correct scope discipline).

| Issue | Surface | Phase impact |
|---|---|---|
| **`elspeth-5e905f3c9d`** | `MultiSelectWithCustomTurn` escape button — wire shape (`{edited_values: {schema_mode, required_fields}}` per plan) contradicts backend `_advance_step_2` (reads `outputs[]` shape). Widget defers rendering. | Phase 9 demo path requires Step 2 escape if user wants to skip required-fields collection. Resolve before Phase 9 E2E. |
| **`elspeth-2c08408170`** | `ProposeChainTurn` Step-3 backend handler completion (Reject + per-step Edit + Ask advisor). Backend `routes.py:2030-2137` consumes only `chosen: ["accept"]` (success) and `["reject"]` (501 stub). 3 of 4 planned buttons absent. | Phase 9 E2E cannot exercise Reject/Edit/Ask-advisor flows until backend lands. Task 9.2 hand-built path test will need the Edit button. |
| **`elspeth-611fc01d94`** | `GuidedHistory` rich step summaries — wire `TurnRecordResponse` is hash-only by audit-trail design. Decision needed: wire extension (`summary` field) or payload-fetch endpoint. | Phase 8 ChatPanel integration ships hash-only history (which Phase 7 already delivered); Phase 9 demo can describe but not show "Step 1: CSV with cols [price, qty]" until this lands. |

Same recommendation as Phase 7 close: bundle as a single "Phase 5 backend completion" task with three sub-deliverables.

## New deferred-polish observations from Phase 8 (14-day expiry unless promoted)

Two new observations filed at Phase 8 close. Together with the three Phase-7-era observations still active, the running deferred-polish list is now five items.

| Observation | Description | Priority |
|---|---|---|
| **`elspeth-obs-5ea21f94af`** (NEW) | Spec §7.4 "Maintain focus on the first interactive element of the new turn after step advance" deferred from Task 8.2. When `respondGuided` resolves and a new turn arrives, focus falls to `<body>` and users must Tab through the whole document to reach the new turn's first interactive. Three implementation options documented in the observation (ref-forwarding from `<GuidedTurn>`; `autoFocus` per leaf widget; `useEffect` in ChatPanel keyed on `step_index`). Likely best tackled BEFORE Phase 9 E2E because Tab-cycling between steps will inflate demo-SLA timing measurements. | P2 |
| **`elspeth-obs-a076365f64`** (NEW) | `ChatPanel.test.tsx` grew 85 → 404 lines across Phase 8. Each test reproduces ~10 lines of `useSessionStore.setState({...})` boilerplate. Helper extraction (`setupGuidedActive(turn?)` / `setupCompleted(terminal?)` / `setupFreeform()`) recommended before Phase 9 adds more discriminator-side tests. | P3 |
| **`elspeth-obs-510a4fbdeb`** (carried) | `TurnPayload` discriminated-union refactor (Option B). Eliminates 6 `as` casts in `GuidedTurn.tsx`. Cascade cost: test fixtures in `client.guided.test.ts` and `sessionStore.guided.test.ts` use shorthand `payload: {options: [...]}` literals that need rewriting. Phase 7 deferred; no Phase 8 movement. | (Phase 7 era) |
| **`elspeth-obs-0a1002de6d`** (RESOLVED in Phase 8) | "Phase 8 ChatPanel must suppress persistent `ExitToFreeformButton` when `terminal.kind === "completed"`" — **honoured** by the discriminator design (completed branch renders `CompletionSummary` alone). Regression pin test 3 (`ChatPanel.test.tsx`) catches future re-introduction of the button on the completed surface. The observation can be dismissed; the contract is enforced by test. | (now closed) |
| **`elspeth-obs-f9e991f517`** (carried) | `useNonInitialEffect` hook extraction candidate. `firstRunRef` initial-mount-skip pattern in 7.3/7.4/7.5. Phase 7 deferred past Phase 8; still unaddressed. | (Phase 7 era) |

The first two (Phase 8 era) should remain visible to the Phase 9 implementer. The first one in particular (`elspeth-obs-5ea21f94af`) likely **should be addressed before Phase 9 E2E** because demo-SLA flakiness is directly tied to focus management.

The third Phase-7-era observation (`obs-0a1002de6d`) is now structurally enforced by Phase 8 — it can be dismissed from `list_observations`. Promotion is not needed because the contract is pinned by test.

## Phase 9 starting brief

Read this in conjunction with the plan file's Phase 9 section (line ~4560).

### Convention reminders inherited from Phase 8

All eight conventions from the Phase 7 handover §"Convention reminders inherited from Phase 7" remain in force. Phase 8 adds these:

9. **Three-branch discriminator order is completed-first → active-guided → fall-through.** If Phase 9 adds a fourth branch (unlikely but possible — e.g., for a `kind: "exited_to_freeform"` terminal that needs its own UI affordance), the precedence rule (more-specific before less-specific) must be preserved.
10. **The `chat-panel-guided-log` region wraps `<GuidedTurn>` ONLY.** Future code that introduces new live regions (e.g., per-widget aria-live for warnings) must not nest inside this region (the "don't nest live regions" convention documented at `InspectAndConfirmTurn.tsx:48-50` and `ComposingIndicator.test.tsx` "ComposingIndicator live region scope" describe block).
11. **Symbol-anchored cross-references, never line numbers.** Five line-number references rotted across Phase 8's commit chain and required three follow-up commits to clean up. Any cross-file comment pointing at a referenced symbol should use the class name, function name, or describe-block name — never `Filename.tsx:LL-LL`.
12. **Discriminator-anchored test assertions.** Discriminator tests assert on `chat-panel--guided` / `chat-panel--completed` / `chat-panel-guided-log` class presence, NOT on widget-internal DOM choices. A widget-DOM refactor (e.g., swapping `<fieldset>` to `<div role="radiogroup">`) should not silently fail a discriminator test that was meant to assert routing.

### Plan-vs-reality drift expected in Phase 9

Phase 9 is Playwright E2E. The drift patterns expected:

- The plan's recipe-match happy path (`Task 9.1`) at line 4565 assumes ~7 clicks and <30s. The Phase 8 cross-task reviewer flagged that without `elspeth-obs-5ea21f94af` (focus-on-step-advance), additional Tab cycles will inflate click count and timing. Either resolve the observation before Phase 9, or adjust the SLA budget with explicit rationale.
- The plan's hand-built path (`Task 9.2`) at line 4612 assumes the Phase-5 Step-3 backend handler is complete. As of Phase 8 close, **it is not** — only `chosen: ["accept"]` works. Either land `elspeth-2c08408170` first, or rescope 9.2 to test only the Accept-all path until backend completion.
- The plan's auto-drop path (`Task 9.3`) at line 4622 is the most decoupled — the auto-drop logic was finalized in Phase 5 and the discriminator's "fall through to freeform when guidedSession is null" behavior is verified by `ChatPanel.test.tsx` Test 4. Playwright should be able to verify end-to-end with a stubbed LLM.
- The Playwright LLM-stub fixture mentioned at `Task 9.2 Step 1` should be verified to exist before starting — the plan asserts "existing Playwright LLM-stub fixture" but Phase 7/8 didn't touch it.

### First action when you start Phase 9

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline acf712d2..HEAD                          # expect 26 commits, top = cdee530f (Phase 8 sweep) or a new handover doc commit
cd src/elspeth/web/frontend
npm test -- --run                                          # expect 412 / 412 pass
npx tsc -p tsconfig.app.json --noEmit                      # expect clean
ls tests/playwright/                                       # verify Playwright infra exists; check for `composer-guided.spec.ts` (Phase 9 creates this)
```

If any baseline gate fails, **stop and investigate** before starting Phase 9.

Strongly consider addressing `elspeth-obs-5ea21f94af` (focus-on-step-advance) as the FIRST task of Phase 9 — before any Playwright work. Without it, the demo SLA assertion is unstable.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic; no migration scripts; no `from_dict` backward-compat defaults.
- **Default to worktree.** Stay here; do not work in `/home/john/elspeth` main.
- **No git stash.** Commit work to a branch if preservation is needed.
- **No calendar shipping commitments.** ELSPETH ships work-until-done.
- **Correctness beats performance always.**
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session. (Phase 8 honoured this: the cross-task reviewer's three stale line-number references were fixed in `cdee530f` rather than deferred.)
- **`any` is forbidden in TypeScript.** Use `unknown` for opaque, closed unions or interfaces otherwise.
- **No optimistic updates.** Server is authoritative.
- **snake_case wire field names in interfaces; camelCase store-internal fields.**
- **ESLint is broken on this worktree.** Don't try to fix it inside Phase 9 unless a separate task is opened for the v9 flat-config migration.
- **Symbol-anchored cross-references, never line numbers.** Established Phase 8.
- **No PR-open at the end of Phase 8.** This handover + project memory entry close Phase 8. PR is deferred per project convention; the branch will accumulate Phases 9-10 before PR-open consideration.
