# Solution Design Review — Phase 3A IA Cleanup

**Source:** worktree `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup`, branch `feat/composer-phase-3-ia-cleanup`, commit range `9e8c54ed6..c3fc5670e` (7 commits).
**Reviewed:** 2026-05-17
**Reviewer:** solution-design-reviewer (Agent A)

## Summary (machine-readable)

- verdict: READY
- block_count: 0
- major_count: 0
- minor_count: 4
- nit_count: 3
- scope: targeted (Phase 3A IA-cleanup packet, frontend only)
- tier_declared: M (frontend chrome refactor with tests + a11y contracts)
- tier_artifact_consistency: N/A (this is implementation review, not SAD review)

## Executive summary

The Phase 3A packet implements its declared scope cleanly. All seven commits are TDD-shaped, the per-task scope-discipline is honoured (out-of-scope items for 15b are visibly deferred, not creeping in), and the live test suite (81 targeted tests across the changed surfaces) is green. The findings are minor deviations from the plan text — none alter the behavioural contract, and several are defensible improvements. No BLOCK or MAJOR issues.

The most consequential finding is documentary, not behavioural: the plan contains an internal contradiction about whether `useHashRouter`'s `VALID_TABS` should be narrowed in 15a or preserved. The implementation followed the operative (later, panel-revised) instruction in 15a2 Task 5/6 and narrowed the set with a redirect-toast bridge — the right call — but 15a1's still-published "preserved in 15a, migrated in 15b" framing now misdescribes the worktree. Recommend the operator update 15a1's header text in the merge PR description so reviewers don't trip on the contradiction.

## Methodology and caveats

- **What I verified mechanically:** the targeted vitest pass at HEAD (`subscriptions.test.ts` 16/16; `InspectorPanel.test.tsx`, `InlineRunResults`, `RunsHistoryDrawer`, `SideRail`, `AppHeader`, `HeaderSessionSwitcher`, `useHashRouter.test.ts`, `ReadinessRowDetail.test.tsx` — 81/81 collectively pass; `ResizeObserver is not defined` warnings from `@xyflow/react` in the GraphView test environment are pre-existing test-environment noise, not assertions failing). I also walked each commit's diff (`git show <sha> -- <path>`) for the load-bearing files.
- **What I did NOT verify mechanically:** per-commit launchability ("every commit green") was not validated by checking out each commit and re-running tests, because the sandbox classifier blocks `git checkout <sha> -- src/elspeth/web/frontend` (destructive overwrite of the worktree). I instead verified launchability by structural diff analysis: commits 1–4 are purely additive (no symbol-deletion, no mounting-site mutation in a way that could break a prior consumer); commits 5–7 are TDD-shaped deletions that update their dependent tests in the same commit (verified by `git show <sha> -- <test-path>`). The plan's "every commit launchable" claim is structurally consistent with the diffs, but a literal per-commit smoke run remains unrun. Anyone wanting that assurance can checkout each SHA and `npx vitest run src` in `~3min`.
- **What is out of scope for this review:** Playwright e2e validity (the spec at `tests/e2e/phase-3a-shell.spec.ts` was inspected for shape but not executed — Phase 3A is "frontend only" per the dispatch prompt and the project staging-deploy procedure is a separate operator step); the rendered visual treatment (CSS-only changes in `App.css`); and any of the 15b deferrals.

## 1. Scope discipline — PASS

Walked each named 15b deferral against the worktree at HEAD:

| 15b-deferred item | Present in 15a worktree? | Verdict |
|---|---|---|
| Graph mini-view (persistent in side rail) | No — `graphMiniSlot={null}` at `App.tsx:301` | Correct |
| YAML export modal (Export YAML side-rail button) | No — `exportYamlSlot={null}` at `App.tsx:303` | Correct |
| Catalog button move from inspector header | No — `InspectorPanel.tsx:493` still hosts the Catalog button; `catalogSlot={null}` at `App.tsx:302` | Correct |
| Version selector relocated to header | No — `InspectorPanel.tsx:432-442` still hosts VersionSelector | Correct |
| Hash-router `VALID_TABS` migration | Partial — see Finding M1 below | See note |
| Final InspectorPanel deletion | No — `InspectorPanel` still mounted via `SideRail` children prop at `App.tsx:307` | Correct |
| Alt+1/2/3/4 shortcut cleanup (App.tsx:155-162 `tabMap`) | No — `App.tsx:155-162` still has `{"1":"spec", "4":"runs"}` lookup. Plan 15a2 §"Known false-negative" explicitly defers this | Correct |
| Validation dot relocated to audit-readiness panel | No — `InspectorPanel.tsx:444-487` still renders the validation dot in the inspector header | Correct |

In-scope items all delivered:

- InlineRunResults + RunsHistoryDrawer (Task 1) — present at `components/execution/`.
- SideRail scaffold + Layout slot rename (Task 2) — present; rename is uniform.
- AppHeader + HeaderSessionSwitcher (Task 3) — present; UserMenu mounted inside AppHeader.
- Auto-validate subscriber + cross-session guard (Task 4 + 4a) — present in `subscriptions.ts`; corresponding tests pass.
- Runs tab + RunsView deletion (Task 5) — `RunsView.tsx` / `.test.tsx` gone; `TABS` narrowed; `tab-runs` CommandPalette command gone; redirect-toast wired.
- Spec tab + SpecView deletion (Task 6) — `SpecView.tsx` / `.test.tsx` gone; `TABS` further narrowed; `tab-spec` gone; `ReadinessRowDetail`'s "View in graph" affordance dispatches `"graph"`.
- SessionSidebar deletion (Task 7) — `SessionSidebar.tsx` / `.test.tsx` gone; theme toggle moved into UserMenu; orphan `elspeth_sidebar_collapsed` key cleanup added; Playwright shell spec landed at `tests/e2e/phase-3a-shell.spec.ts`.

## 2. Every-commit launchability invariant

Per-commit checkout-and-test was not run (sandbox blocked the destructive checkout). Structural verification:

- **Commits 1–4 (84306f19f, ccca39cb0, ae8df6dc3, 86061cf1c)** are purely additive in the dimensions that matter for launchability: new components, new SideRail with all-null slots, AppHeader mounted alongside (not replacing) the still-present SessionSidebar, and a new subscriber that registers alongside the existing two. No existing imports are removed in this range; no exported symbols are renamed.
- **Commit 5 (66748edb9 — Runs removal)** updates `InspectorPanel.test.tsx` to expect `Runs` absent in the same commit that removes it; `RunsView.test.tsx` is deleted in the same commit as `RunsView.tsx`; `useHashRouter.test.ts` is created in the same commit as the `VALID_TABS` narrowing; `tab-runs` command removed from CommandPalette but no test asserted its presence prior. The intermediate-state `TABS` array correctly contains `["spec","graph","yaml"]` (verified via `git show 66748edb9`).
- **Commit 6 (8b04d53ce — Spec removal)** mirrors commit 5's discipline. The Task-5-introduced `it("keeps Spec, Graph, and YAML…")` assertion is rewritten in the same commit (verified via diff) to `it("keeps Graph and YAML while removing Spec and Runs…")` — this is the panel-flagged Task 6 Step 1 deletion of the contradictory assertion, executed correctly.
- **Commit 7 (c3fc5670e — SessionSidebar removal)** deletes 1014 lines and adds 144 (largely CSS removal). `App.test.tsx`'s Layout mock destructures `{ chat, siderail }` (no longer `sidebar`), and the SessionSidebar mock is dropped. The added test `does not mount the retired sessions sidebar` (line 203) and `removes the retired sidebar collapsed preference on startup` (line 212) gate the cleanup.

Structurally, the chain is consistent with the plan's per-commit-green claim. **Caveat:** absent a mechanical per-commit run, I cannot rule out a transient test-environment regression in a commit I didn't get to checkout. Recommend the operator run `for sha in 84306f19f ccca39cb0 ae8df6dc3 86061cf1c 66748edb9 8b04d53ce c3fc5670e; do git checkout $sha -- src/elspeth/web/frontend && (cd src/elspeth/web/frontend && npx vitest run src); done` once before merge if zero-doubt is required.

## 3. Layout slot rename `inspector` → `siderail` — PASS

- `Layout.tsx` `LayoutProps` declares `{ chat, siderail }` only — no `inspector` prop, no alias (`Layout.tsx:34-37`).
- `App.tsx:298` passes `siderail={<SideRail>...</SideRail>}`.
- `App.test.tsx:20-30` Layout mock destructures `{ chat, siderail }` — the panel-flagged risk (mock silently dropping the unrecognised prop) is correctly closed.
- Class-name rename uniform: no surviving `layout-inspector` / `inspector-overlay-*` in `Layout.tsx`. (`inspector-toggle-btn` is intentionally retained per plan as a Phase-8 polish.)
- LocalStorage key `elspeth_inspector_width` preserved as a constant on `Layout.tsx:11` — existing user preferences survive the rename.

No consumer depends on the old `inspector` prop name.

## 4. InlineRunResults + RunsHistoryDrawer — PASS

- `InlineRunResults` mounted in `ChatPanel.tsx` after the chat scrollback (verified via commit 84306f19f diff).
- Component subscribes to `activeRunId`, `progress`, `runs`; renders nothing when no active run AND no history; renders `ProgressView` for active running runs, `RunOutputsPanel` for terminal runs.
- "Past runs" CTA conditional on `runs.length > 0`.
- `RunsHistoryDrawer` implements `role="dialog" aria-modal="true"`, focus-on-open of Close button, Escape-to-close, and Tab/Shift+Tab focus trap — a11y contract that the panel-fix S3 mandated is honoured (`RunsHistoryDrawer.tsx:26-63`).
- Historical-run access is preserved (not dropped).

The component is slightly more conservative than the test fixture suggested: `displayRun` logic falls back to `runs[0]` when `activeRunId` is null, and the `displayStatus` is recomputed from `progress` only when the active run matches — which correctly closes the dangling-`activeRunId` race the panel-fix flagged.

## 5. AppHeader + HeaderSessionSwitcher — PASS

- All session operations survive: switch (`onSelect → selectSession`), create (`onNewSession → createSession`), and the trigger label tracks active session title.
- `CommandPalette` retains the Sessions section at `CommandPalette.tsx:165-181, 400` (H1 fallback).
- Plan does NOT require rename or delete in the dropdown — that's Phase 8 / 15b polish; current scope only requires switch + create. **Defensible omission**: rename and delete remain accessible via the inline pencil/X icons inside SessionSidebar — but SessionSidebar is gone in commit 7. **Confirm:** post-Phase-3A, rename and delete are not directly reachable from the header dropdown. The plan doesn't require them in 15a, but operators relying on rename/delete need to either use the API or wait for 15b. See Finding M4.
- Trigger declares `aria-haspopup="menu"` + `aria-controls={MENU_ID}` (a11y panel-fix S2 honoured).
- Keyboard nav: Arrow keys, Home/End, Enter/Space, Escape, Tab-closes-and-returns-focus all implemented (`HeaderSessionSwitcher.tsx:81-117`).
- Empty-sessions edge: trigger renders "Untitled" + the `+ New session` item still appears (verified via implementation — `+ New session` is always item 0 regardless of `sessions.length`).

## 6. `useHashRouter` invariants — FOLLOWED 15a2 OPERATIVE SPEC

The 15a1 header (line 7) states: *"`VALID_TABS` and default-tab `"spec"` are preserved in this plan and migrated in 15b"*. The 15a2 Task 5/6 bodies (operative, panel-revised 2026-05-17) say the opposite: narrow `VALID_TABS` in Task 5 (`"runs"` → out, default → `"graph"`) and again in Task 6 (`"spec"` → out), with a redirect toast covering the gap.

The implementation followed 15a2. Per-commit:
- After commit 5 (66748edb9): `VALID_TABS = new Set(["spec","graph","yaml"])`, `DEFAULT_TAB = "graph"`.
- After commit 6 (8b04d53ce): `VALID_TABS = new Set(["graph","yaml"])`.
- At HEAD: same; redirect toast text mapping covers both `"runs"` and `"spec"`; shared dismissal flag.

This is the correct call — the Section A panel-fix (2026-05-17, in 15a2's review history) explicitly folded the narrowing forward so the toast text "Showing Graph instead" is truthful for the whole Task 5 → Task 6 window. But the contradiction with 15a1's still-published header text creates a documentation gap. See Finding M1.

## 7. Auto-validate wiring — PASS

- The subscriber is registered inside `initStoreSubscriptions()` alongside the two existing Phase 2C subscribers, in `subscriptions.ts:150-160`. It is NOT inside a component's `try`/`catch`.
- The correctness loop (`fireValidateLoop`) re-fires after in-flight settle, per-session tracking via `Map<string, number>`, clear-before-await with a `FRAGILE` comment (`subscriptions.ts:181-191`).
- Cross-session guard `inflightValidateSessionId` is set immediately before the await and cleared in `finally` (lines 185-190).
- Phase 2C subscribers (version-change-clear, audit-readiness eviction, `validationResult`-change side effects) are preserved unchanged.
- `_resetSubscriptionsForTesting()` extended to tear down all new state including `inflightValidateSessionId` (lines 204-219).
- 16/16 subscription tests pass, including the cross-session guard and "stale-result must not consume fingerprint" tests.

No defensive `try`/`catch` in any component; the `validate()` call is awaited inside `fireValidateLoop` with the `inflightValidateSessionId` reset in a `finally`, which is correct error-channel discipline — if `validate()` throws, the loop unwinds, `lastValidatedVersionBySession` is not updated for that version, the next bump retries, and the failure surfaces through the backend Landscape.

## Findings

### MINOR

#### M1 — Plan 15a1 vs 15a2 internal contradiction on `VALID_TABS` preservation

- **Evidence:**
  - `docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md` line 7: *"The `useHashRouter` invariants (`VALID_TABS`, default-tab `"spec"`) are preserved in this plan and migrated in 15b — Phase 3A keeps `spec`/`runs`/`graph`/`yaml` valid hash fragments so that mid-phase deep links don't 404."*
  - `docs/composer/ux-redesign-2026-05/15a2-phase-3a-removals-part-2.md` Task 5 (lines 41, 132-148) and Task 6 (lines 327-330) explicitly direct the executor to narrow `VALID_TABS` and add a redirect toast.
  - Worktree HEAD `useHashRouter.ts:30`: `const VALID_TABS = new Set(["graph", "yaml"]);` — implementation followed 15a2.
- **Impact:** Future reviewers reading 15a1 in isolation will believe the implementation overran scope. The actual operative spec (15a2's Section A panel-fix on 2026-05-17) is buried in the review-history footer.
- **Recommendation:** Update 15a1's `Architecture` paragraph (line 7) to reference 15a2's narrowing decision, e.g. *"`VALID_TABS` is narrowed in 15a per Tasks 5/6 (Section A panel-fix 2026-05-17); a transient redirect toast bridges the gap for stale `#/runs` and `#/spec` bookmarks."* No code change required.

#### M2 — `InspectorPanel.handleValidationComponentClick` retained the tab switch the plan asked to drop

- **Evidence:** `InspectorPanel.tsx:393-403` keeps `setActiveTab("graph")` inside the click handler. Plan 15a2 Task 6 Step 3.7 (the corresponding sub-step in the worktree edit instructions) specified: *"the `selectNode` call is preserved so GraphView's highlight ring still appears when a validation error is clicked"* — implying the tab switch was to be dropped, with the future audit-readiness "Explain" surface owning routing.
- **Impact:** Defensible UX deviation. If a user is on the YAML tab and clicks a validation error, the current code helpfully switches them to Graph (where the highlight ring is visible). The plan-spec'd behaviour would have left them on YAML with no visible feedback. Behavioural test `InspectorPanel.test.tsx:589-617` (added in commit 8b04d53ce) actually locks in the implementation's behaviour: `expect(screen.getByRole("tab", { name: "Graph" })).toHaveAttribute("aria-selected", "true")`. So the test asserts the deviation as correct.
- **Recommendation:** Either accept the deviation (update the plan to record the better behaviour and the test that locks it in), or follow the plan literally and let the audit-readiness "Explain" surface (Phase 2C, already shipped) own this routing. Operator decision. Marginal preference for the current code: a no-feedback click is worse than an automatic helpful switch.

#### M3 — Missing cross-path dismissal test (`runs → dismiss → visit spec` does not re-show)

- **Evidence:** `15a2` Task 6 Step 8a (lines 437-453) explicitly mandates: *"does not show the spec toast after the runs toast was dismissed (shared dismissal flag)"*. The actual `useHashRouter.test.ts` ships three tests (runs redirect, spec redirect, generic dismissal) but the cross-path lock-in test is absent.
- **Impact:** Low — the shared `elspeth_redirect_toast_dismissed` localStorage key invariant works structurally (the implementation reads the key once per `parseHash` invocation and suppresses regardless of which fragment triggered it), but a future regression that keyed the toasts separately would no longer fail CI as the panel-fix intended.
- **Recommendation:** Add the cross-path test from the plan verbatim to `useHashRouter.test.ts`. ~15 lines, mirrors the existing dismissal test.

#### M4 — Session rename/delete not directly reachable post-SessionSidebar removal

- **Evidence:** `HeaderSessionSwitcher.tsx` exposes only switch (`onSelect`) and new-session (`onNewSession`) verbs; the previous SessionSidebar (`SessionSidebar.tsx`, deleted in c3fc5670e) had inline rename + archive icons. `CommandPalette.tsx:165-194` lists sessions in a Sessions section but only registers a switch action, no rename/delete commands.
- **Impact:** Until 15b adds a session-management surface (or until a per-item action is added to the dropdown), users cannot rename a session or delete one through the UI. The plan does not require these verbs in 15a explicitly, but the absence is a usability regression vs the pre-Phase-3A shell. The risks table mentions "session switching by dropdown" but is silent on rename/delete.
- **Recommendation:** Operator decision — either (a) accept the regression for the 15a→15b window and document it in the merge PR description, (b) add a per-session ⋮ menu to the dropdown in this PR (~30 lines + tests), or (c) defer to 15b explicitly in the plan. The CommandPalette is the natural fallback; a `session-rename-{id}` / `session-archive-{id}` set of palette commands would close the gap without touching the dropdown.

### NIT

#### N1 — Redirect-toast banner uses `role="status"` instead of plan-spec'd `role="alert"`

- **Evidence:** `App.tsx:234`: `<div role="status" className="alert-banner alert-banner--info">`. Plan (15a2 line 157) spec'd `role="alert"` and `screen.getByRole("alert")` test queries.
- **Impact:** Semantically `status` (polite live region) is the better choice for an informational redirect notice — `alert` (assertive) would interrupt screen-reader output. The implementation chose the correct ARIA role. The actual `useHashRouter.test.ts` tests bypass the role entirely (they read `result.current.redirectToast?.message`), so neither the plan's `getByRole("alert")` test nor a hypothetical `getByRole("status")` test guards this.
- **Recommendation:** No code change. Document the divergence in a one-line code comment so the next maintainer doesn't "fix" it back to `alert`. Optionally add a `screen.getByRole("status")` assertion to a smoke test for completeness.

#### N2 — `elspeth_sidebar_collapsed` cleanup is a `useEffect`, not module-scope

- **Evidence:** `App.tsx:73-75`: `useEffect(() => { localStorage.removeItem(RETIRED_SIDEBAR_COLLAPSED_KEY); }, []);`. Plan 15a2 Task 7 Step 5 (lines 678-684) explicitly directs *"at module scope inside `App.tsx`, near the existing module-level imports"* with rationale *"a one-shot side effect, not a useEffect — it must run before any other code reads from localStorage (e.g., the renamed `siderailWidth` initializer)."*
- **Impact:** Negligible. The cleanup removes a key that nothing else reads (its only consumer was `SessionSidebar`, deleted in the same commit). Module-scope vs effect-scope timing matters only if some other initializer reads the key — and none do. The plan's rationale about `siderailWidth` initializer was speculative; the actual `Layout.tsx:55` reads only `elspeth_inspector_width`, an unrelated key.
- **Recommendation:** No change. The deviation is functionally inert. If the operator wants plan-fidelity, the move is `App.tsx:28-29` (next to the `RETIRED_SIDEBAR_COLLAPSED_KEY` constant) becomes the one-liner site and the `useEffect` is deleted. 3-line change.

#### N3 — `RunsHistoryDrawer` does not restore focus to the trigger after close

- **Evidence:** `RunsHistoryDrawer.tsx` has no `useEffect(() => () => triggerRef?.current?.focus(), [])` cleanup, and `InlineRunResults.tsx:54` doesn't pass any trigger ref into the drawer. The plan explicitly says *"Restoration of focus to the trigger after `onClose` is the caller's responsibility (the `InlineRunResults` "Past runs" button receives focus by default because React re-renders the button after `setShowHistory(false)`)"* — which is a structural-luck argument, not an actively enforced contract. There is no test that locks this in.
- **Impact:** Low. In practice React re-renders the button on `setShowHistory(false)` and the button does have focus by default-cursor semantics. But if a future refactor unmounts the button (e.g., the "Past runs" CTA becomes conditional on a different store value) the focus return silently breaks and there is no test to catch it.
- **Recommendation:** Add a test that opens the drawer and closes it and asserts focus lands on the Past-runs button. Small belt-and-braces hardening. Optional.

## What the design does well

- **Plan reality-check discipline.** The 15a1/15a2 plan files carry visible 2026-05-17 panel-fix sections (Section A, Section B+C, Pre-execution addendum, Pre-dispatch follow-up) that document the four-reviewer NO-GO → fixes cascade. The executor honoured those fixes (cross-session guard, correctness loop, retired Task 8, additive subscriptions). The reality-check loop worked.
- **TDD shape preserved across deletions.** Every removal commit (5, 6, 7) updates its tests in the same commit; no "delete code now, fix tests later" pattern. The Task 6 contradictory-assertion fix (panel-flagged) was executed correctly in commit 8b04d53ce.
- **Offensive programming compliance.** `subscriptions.ts` accesses store fields directly (`useExecutionStore.getState().isExecuting`, `useSessionStore.getState().activeSessionId`); no `try`/`catch` for store calls, no `.get()` defensive accessors. The one `try`/`finally` in `fireValidateLoop` (lines 186-190) is for `inflightValidateSessionId` lifecycle, not error suppression — `validate()`'s throw still propagates.
- **No-Legacy-Code policy honoured.** SessionSidebar deletion took 1014 lines with it; the orphan localStorage key is cleaned up in the same commit; no `@deprecated` retentions or commented-out blocks.
- **Cross-session guard is correct.** The S3 panel-fix (suppress `injectSystemMessage` when active session changed mid-validate) is implemented with the suppression-must-not-consume-fingerprint ordering the plan specified, and both tests (suppression + same-content retry) lock it in.
- **Hash-router redirect-toast pattern.** The hook returns `{ redirectToast: { message, dismiss } | null }` cleanly; the App.tsx consumer destructures and mounts; `localStorage` flag silences both runs and spec paths. Clean integration shape.

## Information Gaps

- Per-commit launchability assurance is structural-not-mechanical. The sandbox blocked `git checkout <sha> -- src/...` for the destructive overwrite. Operator can confirm with a 3-minute per-commit test sweep before merge.
- Playwright spec inspected for shape, not executed. The spec at `tests/e2e/phase-3a-shell.spec.ts` asserts the three risks-table acceptance criteria (banner, no sidebar, account menu opens to theme toggle); whether it actually runs against staging is the operator's verification step.
- I did not exhaustively grep the frontend for legacy `inspector` prop references in test fixtures or storybook stories. Layout.test, App.test, and the production sites all check out; if a storybook entry or fixture file references the old prop name, the build still passes (TypeScript would error if a real consumer did).

## Risk Assessment

| Risk | Likelihood | Severity | Notes |
|---|---|---|---|
| Hidden per-commit test break | LOW | LOW | Structural diff analysis is consistent with the plan's per-commit-green claim; not mechanically verified. |
| Cross-path dismissal regression | LOW | LOW | Invariant works structurally; missing test means future regression is silent. M3. |
| Session rename/delete usability regression | MEDIUM | MEDIUM | Real user gap until 15b ships or until CommandPalette adds the verbs. M4. |
| Plan documentation drift (15a1 stale text on VALID_TABS) | HIGH | LOW | Causes reviewer confusion, not behavioural defect. M1. |
| `useEffect`-vs-module-scope cleanup timing | LOW | NIL | Functionally inert in the current code. N2. |

## Confidence Assessment

- Implementation correctness against operative spec (15a2): HIGH confidence — verified by reading every modified file, walking each commit's diff, running the test suite (81/81 pass on target surfaces, 16/16 on subscriptions).
- Plan-vs-implementation deviation calls: HIGH confidence on M1, M3 (textual evidence); MEDIUM on M2, M4 (operator UX judgment required); MEDIUM on N1, N2 (intent inference).
- Per-commit launchability: MEDIUM confidence — structural inference only; not mechanically verified.

## Machine-readable findings

| ID | Severity | Failure mode | Surface | Recommendation |
|---|---|---|---|---|
| M1 | MINOR | Plan inconsistency | docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md line 7 | Update header text to reference 15a2's narrowing decision |
| M2 | MINOR | Plan-vs-impl deviation | src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx:393-403 | Either accept the deviation (update plan) or drop the `setActiveTab("graph")` line |
| M3 | MINOR | Missing test coverage | src/elspeth/web/frontend/src/hooks/useHashRouter.test.ts | Add cross-path dismissal test from 15a2 Task 6 Step 8a |
| M4 | MINOR | Usability regression | HeaderSessionSwitcher + CommandPalette | Decide: defer to 15b (document in PR), add CommandPalette rename/delete commands, or add ⋮ menu to dropdown |
| N1 | NIT | Plan-vs-impl deviation | src/elspeth/web/frontend/src/App.tsx:234 | `role="status"` is the better choice; document why and optionally add status-role assertion |
| N2 | NIT | Plan-vs-impl deviation | src/elspeth/web/frontend/src/App.tsx:73-75 | Optional: move to module scope per plan; functionally inert |
| N3 | NIT | Missing test | src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx | Add focus-returns-to-trigger test as belt-and-braces |
