# Phase 3A IA-Cleanup — Review Synthesis

**Branch:** `feat/composer-phase-3-ia-cleanup`
**Commit range:** `9e8c54ed6..c3fc5670e` (7 commits)
**Synthesised:** 2026-05-17
**Sources:** Agent A (solution architect), Agent B (systems thinker), Agent C (code quality), Agent D (quality/test)

---

## Verdict

**APPROVED_WITH_CHANGES_REQUESTED** — no agent flagged a BLOCK; converged on 1 MAJOR (multi-site), several MINORs, and a handful of NITs. Merge bar is "demo succeeds" per memory `project_rc5ux_demo_prep_scope`. The single MAJOR has a visible demo failure mode (advertised Alt-key shortcut does nothing) and must fix before merge. Several MINORs are deferrable to observations or follow-up branches.

**Entry counts:** 1 MAJOR / 7 MINOR / 4 NIT = 12 entries
**Must-fix before merge:** P3A-001, P3A-002 (test guard for P3A-001), P3A-003 (operator policy decision required), P3A-008 (test re-home decision required — operator gate)
**Defer as observation:** P3A-004, P3A-005, P3A-006, P3A-007, P3A-009, P3A-010, P3A-011, P3A-012

---

## Reviewer-Conflict Resolutions

### Conflict 1: Severity of the dead-call-sites cluster (tabMap / ShortcutsHelp / executionStore string / App.css comment / page-object type)

| Agent | Framing |
|---|---|
| A | "Known false-negative" — explicitly deferred per plan 15a2 §footer |
| B | MAJOR — "Eroding Goals" archetype; demo-visible silent failure |
| C | MAJOR — "No Legacy Code Policy / call-sites-in-same-commit" violation |
| D | Not specifically called out (focus was tests, not source consumers) |

**Load-bearing framing: B + C (MAJOR).** Agent A's "known false-negative" reading defers a defect the plan itself acknowledged but did not actually fix in scope; per memory `feedback_default_is_fix_not_ticket` and `feedback_no_scope_dumping`, "the plan said defer" is not a stopping condition when the defect breaks an advertised user-facing affordance. Agents B and C correctly elevate this: Alt+1 silently doing nothing during a demo is a credibility hit, and the CLAUDE.md "change all call sites in the same commit" rule is explicit. Operator's demo-prep program (memory `project_rc5ux_demo_prep_scope`) tips the scale toward MAJOR.

### Conflict 2: Granularity of the dead-call-sites cluster

The brief asks: one entry, two entries (one per deleted tab), or five entries (one per call-site)?

**Resolution: One umbrella entry (P3A-001) with five sub-items.** The fix is a single atomic sweep: grep for `"spec"` and `"runs"` strings across `src/elspeth/web/frontend/src/**` and `tests/e2e/page-objects/**`, edit each call-site, run vitest. That's one commit, one review cycle, one rebase risk. Splitting per-tab (two entries) would require running the same grep twice. Splitting per-call-site (five entries) generates five micro-commits with five rebase exposures and obscures that they share one root cause (the deletion commits did not run a residual-reference grep). The "smallest meaningful atomic unit" for the fix-loop is the umbrella commit; the sub-items live as a checklist inside that commit's message, not as separate punch-list entries. A separate companion entry (P3A-002) captures the test-guard that would have prevented this class of defect from recurring.

### Conflict 3: Severity of migration shims (RETIRED_SIDEBAR_COLLAPSED_KEY, SIDERAIL_WIDTH_KEY preserve, redirect-toast)

| Agent | Framing |
|---|---|
| A | Not flagged (treats as plan-faithful) |
| B | MINOR — staging-use is mitigating context |
| C | MAJOR — strict CLAUDE.md No Legacy Code reading |

**Load-bearing framing: B (MINOR), with operator escalation.** CLAUDE.md is explicit ("WE HAVE NO USERS YET — DELETE THE OLD CODE COMPLETELY"), so C's read of the rule is correct in the abstract. But memory `project_staging_deployment` confirms `elspeth.foundryside.dev` is the operator's actual staging deploy with real bookmarks; the redirect-toast and width-key preservation are operator courtesies, not architectural rot. The RETIRED_SIDEBAR_COLLAPSED_KEY cleanup is the weakest of the three (no real data to migrate). Surface as a MINOR with an explicit ask: operator chooses delete-all-three (strict policy) or keep-the-toast-only (pragmatic). Don't unilaterally rip them out — that risks staging surprise for the operator personally.

### Conflict 4: M2 from Agent A (InspectorPanel.handleValidationComponentClick keeps `setActiveTab("graph")`)

| Agent | Framing |
|---|---|
| A | MINOR — plan-vs-impl deviation; implementation is the better behaviour |
| B/C/D | Not flagged |

**Load-bearing framing: A (MINOR, accept the deviation).** A's analysis is convincing: the test at `InspectorPanel.test.tsx:589-617` locks in the implementation, the alternative (silent click on YAML tab) is worse UX, and the plan-spec'd "future audit-readiness Explain surface" is not yet built. Treat as plan-update needed, not code-change needed. Logged as P3A-009 (defer, update plan in next doc cycle).

### Conflict 5: Test-coverage gaps in `useHashRouter` and deleted-component re-homing (Agent D's MAJOR #3, #5)

| Agent | Framing |
|---|---|
| A | M3 minor — single missing cross-path dismissal test |
| D | MAJOR #3 (four useHashRouter behaviours untested), MAJOR #5 (RunsView capabilities not re-homed) |

**Load-bearing framing: D (operator gate, not auto-MAJOR).** Agent D is correct that the test coverage is thin, but the underlying behaviours either exist and are smoke-tested elsewhere (the useHashRouter session-change pushState is exercised by the existing `smoke.spec.ts`), or describe capabilities that were intentionally deleted (RunsView diagnostics, polling, LLM explanation — see P3A-008 below for the open question). MAJOR severity is appropriate if the operator intends these capabilities to return; MINOR if they are gone for good. Surface as operator decision (P3A-008), not auto-blocker.

---

## Punch List (Severity-Ordered, Dependency-Ordered Within Tier)

### MAJOR

#### P3A-001 — Dead `Spec`/`Runs` references survive across deletion commits (umbrella, 5 sub-items)

- **Severity:** MAJOR
- **Sources:** Agent B (Findings 1, 2, 3), Agent C (M1, M2, M3, M4)
- **Surfaces (sub-items):**
  1. `src/elspeth/web/frontend/src/App.tsx:153-169` — `tabMap` still maps `"1"→"spec"`, `"4"→"runs"`; Alt+1 and Alt+4 fire `SWITCH_TAB_EVENT` that `InspectorPanel` rejects → silent no-op
  2. `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx:15` — keyboard-shortcut help still advertises `"Switch inspector tab (Spec/Graph/YAML/Runs)"` to users
  3. `src/elspeth/web/frontend/src/stores/executionStore.ts:331` — user-facing error message: `"Pipeline execution failed. Check the Runs tab for error details."` (the Runs tab no longer exists); also `:306` comment stale
  4. `src/elspeth/web/frontend/src/App.css:595` — section banner comment references "Spec tab"; the associated rules may also be dead and should be net-out
  5. `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts:1,6` — `InspectorTab` type union still admits `"spec" | "runs"`; comment on line 1 names both
- **What's wrong:** Phase 3A.5 (`66748edb9`) and 3A.6 (`8b04d53ce`) deleted the `runs` and `spec` tabs but did not grep for residual references. Per CLAUDE.md "No Legacy Code Policy" (*"Change all call sites in the same commit"*), all five sites should have moved with the deletion. The behavioural sub-item (1) is the demo-visible regression: a user who presses Alt+1 expecting Spec view (because ShortcutsHelp told them so) gets silence with no error.
- **Smallest remediation (single atomic commit):**
  1. `App.tsx:153-169`: narrow `tabMap` to `{"1": "graph", "2": "yaml"}`; rename keyboard-handler comment from "Alt+1/2/3/4" to "Alt+1/2".
  2. `ShortcutsHelp.tsx:15`: change to `{ keys: "Alt+1-2", action: "Switch inspector tab (Graph/YAML)" }`.
  3. `executionStore.ts:331`: rewrite the user-facing string to reference the surviving affordance (e.g. `"Pipeline execution failed. Check the run results panel for error details."`); update `:306` comment to mention `InlineRunResults`.
  4. `App.css:595`: delete the section comment and any rules whose only consumer was the deleted Spec tab (verify with grep before deleting CSS).
  5. `inspector-page.ts:6`: narrow `InspectorTab = "graph" | "yaml"`; update line 1 comment to name only `graph/yaml`.
  6. Update or add unit tests as in P3A-002.
- **Verification:** `cd src/elspeth/web/frontend && npm run typecheck && npm test -- --run` is green; manual smoke: open dev build, press Alt+1 → Graph tab activates; press Alt+2 → YAML tab activates; open Shortcuts help dialog → reads "Alt+1-2 Graph/YAML"; `git grep -nE '"(spec|runs)"' src/elspeth/web/frontend/src tests/e2e/page-objects` returns only the `useHashRouter.ts` REMOVED_TAB_MESSAGES entries (intentional, retained) and the `REMOVED_TAB_MESSAGES`-referencing tests.
- **Must-fix:** Yes — sub-item 1 is demo-visible.

#### P3A-002 — Add test guard so future tab removals fail loud, not silent

- **Severity:** MAJOR (companion to P3A-001; fixes the *cause* of P3A-001 recurring)
- **Sources:** Agent B (Finding 1, Meadows L6 "Information Flows" intervention)
- **Surface:** New test in `src/elspeth/web/frontend/src/App.test.tsx` (or co-located with `InspectorPanel.test.tsx` where `TABS` lives)
- **What's wrong:** No test asserts that every key in `App.tsx`'s `tabMap` maps to a member of `InspectorPanel.TABS`. The next tab removal will replay this defect class unless a mechanical guard exists. CLAUDE.md "if it's not mechanically enforced (by types, tests, CI, or named constants), assume the next session won't know about it."
- **Smallest remediation:** Export `TABS` (or the validated tab-name union) from `InspectorPanel`, import in test file, write:
  ```typescript
  it("every Alt-key shortcut targets a live tab", () => {
    // Import tabMap from App.tsx (extract to a named const if needed)
    for (const [key, target] of Object.entries(tabMap)) {
      expect(TABS).toContain(target);
    }
  });
  ```
- **Verification:** Delete `"graph"` from `TABS` locally and re-run; the new test must fail. Restore and re-run; it must pass.
- **Must-fix:** Yes — without this, P3A-001 silently regresses next time.

---

### MINOR

#### P3A-003 — Migration shims for "users who don't exist" — operator policy decision required

- **Severity:** MINOR (with operator escalation)
- **Sources:** Agent B (Finding 5), Agent C (M5)
- **Surfaces:**
  - `src/elspeth/web/frontend/src/App.tsx:28,73-75` — `RETIRED_SIDEBAR_COLLAPSED_KEY` useEffect cleanup
  - `src/elspeth/web/frontend/src/components/common/Layout.tsx:14` — `SIDERAIL_WIDTH_KEY = "elspeth_inspector_width"` (rename declined to preserve user-bookmarked width)
  - `src/elspeth/web/frontend/src/hooks/useHashRouter.ts:27-32, 67-79, 86-88, 200-209` and `src/elspeth/web/frontend/src/App.tsx:233-245` — redirect-toast machinery (~30 LOC) for stale `#/runs` / `#/spec` bookmarks
- **What's wrong:** CLAUDE.md "No Legacy Code Policy" is explicit that compat shims for not-yet-real users are forbidden. Counterweight: memory `project_staging_deployment` confirms `elspeth.foundryside.dev` is the operator's own actively-used staging instance with real bookmarks.
- **Smallest remediation (three options, operator picks):**
  - (a) **Strict-policy:** delete all three. The RETIRED key cleanup is ~3 LOC; the width-key rename is ~1 LOC + comment removal; the redirect-toast is ~30 LOC + `useHashRouter.test.ts` redirect tests + `App.tsx:233-245` banner. Cleanest, but the operator's own staging bookmarks (Spec/Runs hashes) 404 silently.
  - (b) **Keep-toast-only:** delete the RETIRED_SIDEBAR_COLLAPSED_KEY cleanup and the SIDERAIL_WIDTH_KEY preservation (these have no observable user value); keep the redirect-toast because it's the only one the operator-as-user notices on staging.
  - (c) **Keep all three:** document in the merge PR description that the shims are intentional staging courtesies; add a comment to each pointing to the rationale.
- **Verification:** Whichever option is chosen, `npm test -- --run` must remain green and existing `useHashRouter.test.ts` tests adjust accordingly.
- **Must-fix:** **Operator gate** — surface to operator before merge; do not unilaterally delete (risks staging surprise) or unilaterally keep (risks policy violation).

#### P3A-004 — Plan 15a1 contradicts 15a2 on `VALID_TABS` preservation

- **Severity:** MINOR
- **Sources:** Agent A (M1)
- **Surface:** `docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md:7`
- **What's wrong:** 15a1 header still says *"`VALID_TABS` and default-tab `\"spec\"` are preserved in this plan and migrated in 15b"*. Implementation followed 15a2's later-revised "narrow in 15a with redirect toast bridge" direction (the correct call). Future reviewers reading 15a1 in isolation will think the implementation overran scope.
- **Smallest remediation:** Update 15a1 line 7 to reference 15a2's narrowing decision and the Section A panel-fix (2026-05-17). No code change.
- **Verification:** `git diff docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md` shows only line 7 updated and reads cleanly against 15a2's operative text.
- **Must-fix:** No — defer as observation; documentary debt, not behavioural defect.

#### P3A-005 — Session rename/delete usability regression post-SessionSidebar removal

- **Severity:** MINOR
- **Sources:** Agent A (M4)
- **Surface:** `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx`, `src/elspeth/web/frontend/src/components/common/CommandPalette.tsx:165-194`
- **What's wrong:** Pre-Phase-3A `SessionSidebar` had inline pencil (rename) and X (archive) icons per session. `HeaderSessionSwitcher` only exposes switch + new-session. `CommandPalette` lists sessions but only registers switch actions. Until 15b adds a session-management surface, users cannot rename or delete sessions through the UI.
- **Smallest remediation:** Operator decision — (a) accept regression, document in merge PR; (b) add `session-rename-{id}` / `session-archive-{id}` CommandPalette commands (~40 LOC + tests); (c) add per-item ⋮ menu to dropdown (~60 LOC + tests).
- **Verification:** If (b) chosen, command-palette test: type session name, see Rename + Archive actions; if (c), keyboard-navigate to menu item and confirm verbs render.
- **Must-fix:** No — defer with operator's decision recorded in PR description. Demo paths do not exercise rename/delete; demo-success bar is met without it.

#### P3A-006 — `RunsHistoryDrawer` ARIA contract not asserted

- **Severity:** MINOR
- **Sources:** Agent D (Finding 1)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- **What's wrong:** Implementation sets `role="dialog"`, `aria-modal="true"`, `aria-label="Past pipeline runs"`. None are asserted in tests, so a future attribute deletion is undetectable.
- **Smallest remediation:** Add three assertions to the existing drawer test:
  ```typescript
  expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  expect(screen.getByRole("dialog")).toHaveAccessibleName("Past pipeline runs");
  ```
- **Verification:** Test passes against current implementation; remove `aria-modal` locally and the test must fail.
- **Must-fix:** No — defer as observation; existing focus-trap + Escape tests provide partial guard.

#### P3A-007 — `InlineRunResults` "Past runs" click does not open drawer in test

- **Severity:** MINOR
- **Sources:** Agent D (Finding 2)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/InlineRunResults.test.tsx`
- **What's wrong:** Tests confirm "Past runs" button renders; no test clicks it and confirms the drawer mounts. Shallow coverage for a user-visible interaction.
- **Smallest remediation:** Add one userEvent test: render with `runs.length > 0`, click "Past runs", assert `getByRole("dialog")` appears, click Close, assert dialog absent.
- **Verification:** Test passes; mutate `setShowHistory(true)` to `setShowHistory(false)` and the test must fail.
- **Must-fix:** No — defer as observation; integration smoke implicitly exercises this via Playwright.

#### P3A-008 — RunsView capability re-homing — operator decision required

- **Severity:** MINOR (with operator escalation; could be MAJOR if intent is to restore)
- **Sources:** Agent D (Finding 5)
- **Surface:** No file; conceptual gap left by `RunsView.test.tsx` (579 lines) deletion in `66748edb9`
- **What's wrong:** Behaviours covered by the deleted tests — diagnostics accordion, polling while active, token states + artifacts on inspect, LLM explanation, failure_detail rendering, fan-out accounting, Inspect-button `aria-expanded`/`aria-controls`, suggestion-banner from SpecView, active-run indicator + inline-rename from SessionSidebar — have no successor tests. If these capabilities are intentionally retired (Phase 3A direction), the gap is acceptable. If they return in 15b/Phase 3B, they return without a test harness.
- **Smallest remediation:** Operator decision —
  - (a) **Capabilities retired:** add a one-line ADR/note in `docs/composer/ux-redesign-2026-05/15a2-phase-3a-removals-part-2.md` recording that diagnostics-accordion / inline-rename / suggestion-banner are intentional removals not coming back in 15b. No test changes.
  - (b) **Capabilities deferred:** the missing-tests list (Agent D §2) becomes a 15b/Phase 3B test-rehoming checklist; track as filigree issue with dependency on whichever phase restores the surfaces.
- **Verification:** PR description references the operator's decision; if (b), filigree issue exists with the rehoming checklist.
- **Must-fix:** **Operator gate** — surface before merge. Demo-prep bar is met regardless (demo path is hello-world tutorial, not RunsView diagnostics), so this is documentary not behavioural.

#### P3A-009 — `InspectorPanel.handleValidationComponentClick` retains plan-spec'd tab switch (accept deviation, update plan)

- **Severity:** MINOR
- **Sources:** Agent A (M2)
- **Surface:** `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx:393-403`
- **What's wrong:** Plan said drop the `setActiveTab("graph")` line; implementation kept it because the alternative (silent click on YAML tab) is worse UX. Test at `InspectorPanel.test.tsx:589-617` locks in the implementation.
- **Smallest remediation:** Update plan (15a2 Task 6 Step 3.7) to record the better behaviour and the locking-in test. No code change.
- **Verification:** Plan text now matches implementation; test continues passing.
- **Must-fix:** No — defer as observation; test prevents drift.

---

### NIT

#### P3A-010 — Redirect-toast uses `role="status"` (correct), but plan and a11y-test pattern said `role="alert"`

- **Severity:** NIT
- **Sources:** Agent A (N1)
- **Surface:** `src/elspeth/web/frontend/src/App.tsx:234`
- **What's wrong:** Plan spec'd `role="alert"` (assertive) for the redirect toast. Implementation used `role="status"` (polite) — the correct ARIA semantic for an informational redirect notice. No test asserts either.
- **Smallest remediation:** Add a single comment at the `<div role="status">` line documenting why `status` not `alert` (informational, not interrupt-worthy). Optionally add `screen.getByRole("status")` assertion to a smoke test.
- **Verification:** `git blame` on the comment shows the rationale next session.
- **Must-fix:** No — defer.

#### P3A-011 — `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup is a `useEffect` not module-scope (functionally inert)

- **Severity:** NIT
- **Sources:** Agent A (N2)
- **Surface:** `src/elspeth/web/frontend/src/App.tsx:73-75`
- **What's wrong:** Plan said module-scope; implementation used `useEffect`. Functionally inert because no other initializer reads the deleted key. Subsumed by P3A-003 if operator chooses option (a) or (b) — the cleanup goes away entirely.
- **Smallest remediation:** Resolve via P3A-003 (delete the cleanup); if P3A-003 chooses (c) keep-all-three, optionally move to module scope per plan.
- **Verification:** Whatever P3A-003 chooses, no test regresses.
- **Must-fix:** No — defer; dependent on P3A-003.

#### P3A-012 — `RunsHistoryDrawer` does not actively restore focus to trigger after close

- **Severity:** NIT
- **Sources:** Agent A (N3)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx`
- **What's wrong:** Focus return relies on React re-rendering the "Past runs" button with default cursor semantics; structural luck, not an enforced contract. No test guards it.
- **Smallest remediation:** Add a userEvent test: open drawer, close drawer, assert `document.activeElement` is the "Past runs" button.
- **Verification:** Test passes against current React behaviour; if a future refactor unmounts the button, the test fails before users notice.
- **Must-fix:** No — defer as observation.

#### P3A-013 — Minor test-hygiene items batched

- **Severity:** NIT (composite of several minor test gaps)
- **Sources:** Agent D (Findings 6, 7, 9, 10, 11), Agent C (N3)
- **Surfaces & items:**
  - `SideRail.test.tsx` — `aria-label="Composer side rail"` not asserted
  - `HeaderSessionSwitcher.test.tsx` — `aria-current="page"` on active session not asserted
  - `AppHeader.test.tsx` — no `localStorage.clear()` in `beforeEach` (vitest isolates by default but hygiene)
  - `subscriptions.test.ts:266-300` — no test for `isExecuting` going true mid-loop (correctness logic is fine; speculative race)
  - `RunsHistoryDrawer.test.tsx:21-22` — `getByText(/r1/)` queries are brittle; prefer scoped `getByRole("listitem")`
  - `stores/subscriptions.ts:181-191` — `FRAGILE` correctness-loop comment deserves a design-note expansion (covered by tests, not broken)
- **What's wrong:** Each is a small belt-and-braces improvement; none is a behavioural defect.
- **Smallest remediation:** Treat as a single observation/follow-up issue rather than 6 separate punch-list entries. One commit that adds the four missing assertions + the `localStorage.clear()` + the design note. ~30 LOC across 5 files.
- **Verification:** All new assertions pass; existing tests still pass.
- **Must-fix:** No — defer as observation; demo-prep bar is met.

---

## Stop-Conditions Summary

### Must fix in this branch before merge

| ID | Title | Why must-fix |
|---|---|---|
| P3A-001 | Dead Spec/Runs references (umbrella, 5 sub-items) | Sub-item 1 is demo-visible silent keyboard failure |
| P3A-002 | Test guard for tab-shortcut wiring | Without this, P3A-001 silently regresses next time |
| P3A-003 | Migration-shim policy decision | **Operator gate** — surface before merge; do not unilaterally decide |
| P3A-008 | RunsView capability re-homing | **Operator gate** — record intent (retired vs deferred) in PR description |

### Defer to observations / follow-up issues

| ID | Title | Deferral rationale |
|---|---|---|
| P3A-004 | Plan 15a1 stale `VALID_TABS` text | Documentary debt, not behavioural |
| P3A-005 | Session rename/delete usability regression | Demo does not exercise; operator-policy adjacent |
| P3A-006 | RunsHistoryDrawer ARIA assertions missing | Existing focus/Escape tests provide partial guard |
| P3A-007 | InlineRunResults "Past runs" click not exercised | Implicit Playwright coverage |
| P3A-009 | InspectorPanel tab-switch deviation | Test locks in correct behaviour; plan needs update only |
| P3A-010 | `role="status"` rationale comment | Implementation is correct; comment is documentary |
| P3A-011 | `useEffect` vs module-scope cleanup | Subsumed by P3A-003 resolution |
| P3A-012 | RunsHistoryDrawer focus-restore test | Structural-luck behaviour works in practice |
| P3A-013 | Minor test-hygiene batch | Composite of belt-and-braces items |

---

## Notes on Out-of-Scope Items the Reviewers Flagged

- **Agent C N2** (Layout `calc(100vh - …)` height-budget regression after AppHeader insertion): not in the punch list because it's a "verify in browser" item, not a static-evidence defect. Add to operator's pre-merge smoke checklist: open dev build at common viewport heights and confirm InspectorPanel does not overflow below the fold.
- **Agent C N4** (`App.tsx:84,109` `console.error` for non-fatal events): pre-existing, explicitly out of scope per Agent C; do not fix in this branch (would expand scope and trigger memory `feedback_no_scope_dumping` adjacency).
- **Agent A "per-commit launchability"** caveat: structural inference only; operator can confirm with a 3-minute per-commit `vitest run` sweep before merge if zero-doubt is required. Not blocking.
- **Agent D Finding 4** (Playwright spec too thin): the spec is structural-smoke, not integration. Expanding it is Phase 3B test scope, not Phase 3A fix scope.
