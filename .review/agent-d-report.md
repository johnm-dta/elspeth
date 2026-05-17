# Phase 3A Test Suite Review — Agent D

**Commit range:** `9e8c54ed6..c3fc5670e`
**Branch:** `feat/composer-phase-3-ia-cleanup`
**Review date:** 2026-05-17

---

## Confidence Assessment

High confidence on structural findings (missing tests, deletion coverage, anti-patterns). Moderate confidence on the severity rating for the two behavioural-coverage gaps — they depend on how much of RunsView's diagnostic/polling logic survives inside InlineRunResults vs. being fully handled by the store layer.

---

## Risk Assessment

The single highest-risk finding is the complete loss of RunsView's diagnostic-expansion, polling, and aria attribute tests with no re-homing. RunsView was deleted alongside its 579-line test file. The replacement components (InlineRunResults, RunsHistoryDrawer) are much thinner than RunsView was, and their tests match their current implementation scope — but the deleted tests covered behaviours (live-poll while active, diagnostics accordion, LLM explanation rendering, aria-expanded/aria-controls contract) that existed in the old UI and whose equivalents have not been confirmed absent from the new UI. If any of those behaviours snuck through in a follow-on PR they will arrive without a test harness.

---

## Information Gaps

- Whether `InlineRunResults` is intended to eventually grow the diagnostics accordion that `RunsView` had, or whether that capability is permanently removed. If it grows back, the missing tests gap becomes blocking.
- Whether the E2E spec is intentionally minimal (a placeholder for a fuller suite once the IA stabilises) or is expected to be the primary integration proof point.

---

## Findings

### 1. New Component Coverage

#### SideRail (`SideRail.test.tsx`)
The eight slot-presence tests are adequate for a scaffold component. One gap: the `aria-label="Composer side rail"` on the `<aside>` element is never asserted. The landmark role is the primary accessibility contract for screen-reader users and is the kind of thing that a later refactor removes by accident. This is low severity because it will eventually break an accessibility audit, but the component is otherwise well-tested.

#### InlineRunResults (`InlineRunResults.test.tsx`)
Good coverage of the display-logic state machine (no runs, active running run, terminal active run, most-recent run, dangling activeRunId). One behavioral gap: clicking "Past runs" should open the `RunsHistoryDrawer`; no test exercises the button click and confirms the drawer mounts. The test confirms the button is present but stops there. This is a shallow test for a user-visible interaction.

#### RunsHistoryDrawer (`RunsHistoryDrawer.test.tsx`)
Focus management, trap, Escape, and empty-state are all covered. Two gaps:
1. The `role="dialog"` / `aria-modal="true"` / `aria-label="Past pipeline runs"` ARIA contract on the root `<div>` is not asserted. The implementation sets all three but no test verifies them, so a renames/attribute deletion would go undetected.
2. No test exercises run-status formatting (the `run.status.replace(/_/g, " ")` path). A status value like `"in_progress"` would render as `"in progress"` — that transformation is implementation logic, not just a slot.

#### HeaderSessionSwitcher (`HeaderSessionSwitcher.test.tsx`)
Thorough coverage of keyboard navigation, Escape, Tab-to-close, ArrowDown/ArrowUp wrap, empty-session edge case, and "New session" CTA. One gap: `aria-current="page"` on the active session menu item is not asserted. The implementation sets it but the test never checks it; this is the ARIA contract that assistive technology uses to identify the current session.

#### AppHeader (`AppHeader.test.tsx`)
Three smoke-render tests only. The component is a thin composition of `HeaderSessionSwitcher` + `UserMenu`, so the lightweight coverage is proportionate. The only missing item is that `onOpenSettings` and `onSignOut` callbacks are never confirmed to be wired through — both are passed to `UserMenu` but no test verifies the callbacks propagate. That wiring is tested in `UserMenu.test.tsx` in isolation, so this is acceptable.

#### useHashRouter (`useHashRouter.test.ts`)
Tests cover only the removed-tab redirect path (Runs hash → Graph, Spec hash → Graph, dismiss persistence). The hook has five additional behaviours that are completely untested at the unit level:

- Back/forward navigation via `popstate` (the `handleHashChange` branch).
- Session changes writing `pushState` (activeSessionId subscription).
- Tab changes writing `replaceState` (TAB_CHANGED_EVENT listener).
- Initial hash-less load writing the current session to the URL.
- Invalid session-in-hash clearing when sessions load (`sessions.length` subscriber).

Some of these are integration-tested in the existing `smoke.spec.ts` E2E suite (the hash-router test), but the unit-level coverage is restricted to the removed-tab redirect slice only. Given that hash-routing is the primary deep-link mechanism, the missing unit coverage for session-change → pushState and back/forward → applyHash is a noticeable gap.

#### subscriptions.ts additions — auto-validate subscriber (`subscriptions.test.ts`)
Coverage is strong: initial version fire, reference-equality guard (version unchanged), isExecuting suppression, null activeSessionId guard, correctness loop (newer version arrives during in-flight), cross-session isolation, cross-session stale injection suppression, and fingerprint-consumption guard for suppressed stale results. The cleanup path is also verified implicitly — `_resetSubscriptionsForTesting` clears `unsubscribeAutoValidate`, and each `beforeEach` calls it and then re-calls `initStoreSubscriptions`.

One gap: there is no test for the case where `isExecuting` becomes `true` between the subscription firing and `fireValidateLoop` running (i.e., `isExecuting` is checked inside the loop iteration, not just at subscription time). The existing test sets `isExecuting: true` before the version changes; the implementation also guards inside `fireValidateLoop` itself. A test verifying that a version increment is silently dropped when `isExecuting` goes true _after_ the subscription fires but _before_ the loop resolves would close this gap. This is a minor gap — the implementation is correct and the existing test covers the common case.

---

### 2. Coverage of Deletions

#### RunsView.test.tsx (579 lines deleted)
RunsView and its test file were both fully deleted. The test file contained assertions that covered behaviours now partially redistributed across new components:

| Old RunsView test | Status |
|---|---|
| Surfaces pending pipeline proposals when no runs exist | Not re-homed. Proposals are handled by GraphView/YamlView/MessageBubble, not InlineRunResults/RunsHistoryDrawer. This is the largest behavioral gap. |
| Renders stored failure reason for failed runs | Not re-homed to InlineRunResults or RunsHistoryDrawer. |
| Never renders negative duration for terminal runs | Not re-homed. |
| Fan-out accounting rendering | Not re-homed. |
| Status badge aria-hidden symbols | Not re-homed. |
| Rows routed to virtual discard sink | Not re-homed. |
| Polls session runs while a run is active | Not re-homed at unit level. |
| Polls expanded diagnostics while an inspected run is active | Not re-homed. |
| Shows token states and artifacts when diagnostics are opened | Not re-homed. |
| LLM explanation for diagnostics | Not re-homed. |
| failure_detail.error_message in diagnostics panel | Not re-homed. |
| RunsView Inspect button aria-expanded/aria-controls | Not re-homed. |
| Cancelling badge class (visual differentiation) | Not re-homed. |

The key mitigation is that `RunsHistoryDrawer` is currently a read-only list component with no diagnostics accordion, no expand-on-click, and no proposal surface. If these features are permanently removed (not deferred), the missing tests are irrelevant. If any of them return, there is no test harness to catch regressions.

The `InspectorPanel.test.tsx` additions for Phase 3A do correctly cover the Runs and Spec tab removal from the InspectorPanel tab strip, and the redirect of validation component clicks to Graph — these are correctly re-homed.

#### SpecView.test.tsx (141 lines deleted)
SpecView and its test file were deleted. The behaviours it covered (error/warning/suggestion banners, proposal rows affecting graph) are now tested in:
- InspectorPanel test: "Runs tab removal" describe block covers the absence of the Spec tab.
- Validation banners: partially covered by InspectorPanel's existing validation feedback tests.
- Proposal rows: GraphView and YamlView tests cover proposal pill rendering for their respective areas.

The SpecView deletion is cleaner than RunsView. The only notable gap is that the "renders suggestion banner" case (SpecView line 94) has no direct equivalent anywhere in the Phase 3A test additions.

#### SessionSidebar.test.tsx (113 lines deleted)
SessionSidebar and its test file were deleted. The behaviours it covered:
- Active-run indicator (non-terminal/terminal status distinction): not re-homed. The `HeaderSessionSwitcher.test.tsx` does not track run state.
- Inline rename (double-click open, save trimmed title, Escape cancel): the rename action is tested at the store level in `sessionStore.test.ts` (`renameSession` describe block, line 377), but the inline-rename UI interaction (double-click to open, Escape to cancel) is not re-homed to `HeaderSessionSwitcher` or any other component test.

The active-run indicator and inline-rename UI flow are behavioural regressions at the component level. If `HeaderSessionSwitcher` does not implement them (and it does not — confirmed by reading the implementation), the gap is intentional. But it should be documented, not silently absent.

---

### 3. E2E Spec (`phase-3a-shell.spec.ts`)

The spec is 22 lines and contains a single test. It verifies:
- The `banner` landmark is visible.
- The session switcher trigger button (labelled "untitled") is visible.
- The account menu button is visible.
- The legacy `Sessions sidebar` label is absent.
- Opening the account menu shows a theme-toggle button.

This is a structural smoke test, not an integration test of the new IA. It does not exercise:
- Session switching via the header dropdown (clicking the session switcher, selecting a session, confirming navigation).
- SideRail presence or any slot content.
- InlineRunResults mounting after a run.
- RunsHistoryDrawer opening from "Past runs".
- Stale hash redirect toast appearing in the rendered UI.

The spec adds value as a negative assertion (no sidebar) and a basic presence check. It does not validate that the new IA actually works end-to-end. For a phase that redesigns the primary navigation surface, a single smoke test is insufficient E2E coverage.

---

### 4. Anti-Patterns

#### Sleepy assertions
None found. No `sleep()`, `setTimeout`, or `Thread.sleep` equivalents appear in any of the new or modified test files. All async assertions use `waitFor`, `act`, or `await userEvent.*`. Clean.

#### Test interdependence
None. Every test that touches store state uses `beforeEach` to reset it via `useXxxStore.setState({...})` and `_resetSubscriptionsForTesting()`. The `let` variables in `subscriptions.test.ts` (`resolveFirst`, `resolveValidate`) are declared inside individual test bodies, not at module scope.

#### Shared mutable module state
None introduced in test files. The `let` declarations cited in the diff are test-local.

#### Missing assertions (assertion-free tests)
None. Every test has at least one `expect` call.

#### Wrong test level
The E2E spec is the right level for what it tests. The concern is that it is too thin, not that it is at the wrong level. Unit tests cover component logic; the E2E spec should cover the integrated IA — but the IA integration coverage is currently insufficient.

#### Brittle queries
Two `getByText(/r1/)` and `getByText(/r2/)` calls in `RunsHistoryDrawer.test.tsx` lines 21–22 query run ID strings rendered verbatim. These are stable (run IDs don't change in the test fixture) and short. If the implementation wraps them in a label or decorates the ID in future, the query would break. A `getByRole("listitem")` query scoped to the list would be more resilient. This is a nit.

The `getByText(/no prior runs/i)` at line 42 is a natural-language string embedded in the implementation (not a translation key). Since the project has no i18n layer, this is acceptable.

#### localStorage isolation
`UserMenu.test.tsx` adds a correct `beforeEach` that calls `localStorage.clear()` and cleans `document.documentElement` attributes. `useHashRouter.test.ts` calls `localStorage.clear()` in `beforeEach`.

`AppHeader.test.tsx` renders `<AppHeader>` which mounts `HeaderSessionSwitcher` and `UserMenu`. `UserMenu` reads `localStorage` for the theme preference. `AppHeader.test.tsx` has no `localStorage.clear()` in its `beforeEach`. If tests in another file (e.g., `UserMenu.test.tsx`) set `localStorage.getItem("elspeth_theme")` and the test runner does not clear storage between files, `AppHeader.test.tsx` could observe stale theme state. In practice vitest's jsdom resets localStorage between test files by default, so this is unlikely to cause flakiness — but it is still a missing hygiene step.

---

### 5. Smoke Render (`App.test.tsx`)

Confirmed passing: `npm test -- --run App.test` reports 13/13 tests passed. The Phase 3A modifications to `App.test.tsx` are correct: the mock is updated to use the new `siderail` prop name, `SessionSidebar` mock is removed, `localStorage.clear()` and `window.history.replaceState` are added to `beforeEach`, and four new assertions cover the redirect toast, absence of the legacy sidebar, and stale-key cleanup.

---

## Summary Table

| # | Severity | File:Line | Description |
|---|---|---|---|
| 1 | MAJOR | `RunsHistoryDrawer.test.tsx` — entire file | `role="dialog"` / `aria-modal` / `aria-label` ARIA contract on drawer root never asserted; attribute deletion is undetectable |
| 2 | MAJOR | `InlineRunResults.test.tsx` — entire file | Clicking "Past runs" button to open the drawer is never exercised; presence check stops short of the interactive behaviour |
| 3 | MAJOR | `useHashRouter.test.ts` — entire file | Four of five routing behaviours untested: session-change pushState, tab-change replaceState, initial hash-less write, invalid-session clearing |
| 4 | MAJOR | `phase-3a-shell.spec.ts` — entire file | Single smoke test; does not exercise session switching, SideRail, InlineRunResults, RunsHistoryDrawer, or stale-hash redirect toast in E2E context |
| 5 | MAJOR | No re-homed file | RunsView diagnostic expansion, polling, aria-expanded/aria-controls, failure_detail, LLM explanation, accounting, and badge tests deleted with no re-homing; gap is only safe if capabilities are permanently absent |
| 6 | MINOR | `SideRail.test.tsx` — entire file | `aria-label="Composer side rail"` on the `<aside>` landmark is never asserted |
| 7 | MINOR | `HeaderSessionSwitcher.test.tsx` — entire file | `aria-current="page"` on the active session menu item is never asserted |
| 8 | MINOR | `SessionSidebar.test.tsx` (deleted) | Active-run indicator and inline-rename UI interaction tests not re-homed to `HeaderSessionSwitcher` or any successor |
| 9 | MINOR | `AppHeader.test.tsx` — entire file | No `localStorage.clear()` in `beforeEach`; stale theme state from other files could leak in |
| 10 | MINOR | `subscriptions.test.ts`:266–300 | No test for `isExecuting` going true between version-increment subscription firing and loop iteration |
| 11 | NIT | `RunsHistoryDrawer.test.tsx`:21–22 | `getByText(/r1/)` / `getByText(/r2/)` — would break if implementation decorates the run ID; prefer scoped `getByRole("listitem")` |

**Totals: 0 BLOCK, 5 MAJOR, 5 MINOR, 1 NIT**

---

## Architecture Assessment

- **Pyramid shape:** Healthy. Unit tests dominate (671 total), E2E is a thin layer. The problem is that E2E is too thin for the scope of the IA change, not that it is over-represented.
- **Isolation:** Good. Store state is reset per-test via `setState`. Subscriptions are reset via `_resetSubscriptionsForTesting`. `userEvent.setup()` is used correctly.
- **Determinism:** Good. No fixed sleeps. `waitFor` and `act` are used correctly for async assertions. The `resolveFirst`/`resolveValidate` promise-resolution pattern in `subscriptions.test.ts` is a clean way to test ordered async sequences without wall-clock timing.
- **Deletion coverage:** Incomplete. The RunsView deletion is the largest gap. SpecView and SessionSidebar deletions are cleaner but leave the inline-rename UI and suggestion-banner cases uncovered.
- **New component coverage:** Adequate at the slot/slot-prop level; thin on ARIA contracts and interactive compositions (drawer-open, session-switch flow).

---

## Caveats

1. This review is based on static diff analysis and a full unit test run. The E2E tests were not executed (they require a live backend). The E2E findings (finding #4) are based on reading the spec file, not on observing its runtime behaviour.
2. The RunsView capability-removal assessment (finding #5) assumes that the features listed (diagnostics accordion, polling, LLM explanation) are not implemented in `InlineRunResults` or `RunsHistoryDrawer`. This was confirmed by reading both implementations in full. If those capabilities are added in a subsequent commit, this finding should be revisited.
3. The `isExecuting` mid-loop gap (finding #10) is speculative — the implementation logic is correct, and the described scenario is an unlikely race. It is listed as MINOR rather than MAJOR because the correctness loop test exercises a similar temporal ordering.
