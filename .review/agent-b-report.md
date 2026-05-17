# Agent B Report — Systems-Archetype Review: Phase 3A IA-Cleanup

**Branch:** `feat/composer-phase-3-ia-cleanup`
**Commit range:** `9e8c54ed6..c3fc5670e`
**Reviewer date:** 2026-05-17
**Scope:** Second-order effects, archetype patterns, historical-pattern repeats

---

## 1. Narrative Analysis

### 1.1 Shifting the Burden — L3 layer lazy-import check

**Finding: NONE**

CLAUDE.md explicitly cites the "Shifting the Burden" archetype by name in the context of cross-layer lazy imports: "NEVER: Add a lazy import with an apologetic comment." All seven commits in this range are entirely frontend (TypeScript); no Python L0–L3 import graph is involved. No apologetic `# TODO: fix the real problem` imports were added, and no new `TYPE_CHECKING` workarounds appear in any diff. The intervention-deference pattern does not manifest here.

**Phase 3A verdict:** Clean.

---

### 1.2 Eroding Goals — "Every commit leaves the app green"

**Finding: MAJOR (two instances)**

The plan's green-bar promise means every cleanup commit must also update all artifacts that depended on the thing it deleted. Two cleanup commits violated this by not updating co-located artifacts:

#### Instance A — `App.tsx:155-160` tabMap not updated across Phase 3A.5 / 3A.6

The tabMap inside `App.tsx`'s keyboard handler still maps `"1"` to `"spec"` and `"4"` to `"runs"`:

```typescript
const tabMap: Record<string, string> = {
  "1": "spec",   // tab deleted in 8b04d53ce (Phase 3A.6)
  "2": "graph",
  "3": "yaml",
  "4": "runs",   // tab deleted in 66748edb9 (Phase 3A.5)
};
```

`App.tsx` was touched in both Phase 3A.5 and Phase 3A.6, but the tabMap was not updated in either commit (confirmed by examining `git show 66748edb9 -- src/App.tsx` and `git show 8b04d53ce -- src/App.tsx`: neither diff modifies the tabMap block). The event fires, but `InspectorPanel.tsx:320` rejects anything that is not `"graph"` or `"yaml"`, so Alt+1 and Alt+4 silently do nothing. This is not caught by tests because the keyboard path goes through a `CustomEvent` that tests do not wire end-to-end.

#### Instance B — `ShortcutsHelp.tsx:15` not updated across Phase 3A.5 / 3A.6

`ShortcutsHelp.tsx` was never modified in the Phase 3A commit range (confirmed by `git log --oneline 9e8c54ed6..c3fc5670e -- src/.../ShortcutsHelp.tsx` returning empty). It still reads:

```typescript
{ keys: "Alt+1-4", action: "Switch inspector tab (Spec/Graph/YAML/Runs)" },
```

This is user-facing text in a live dialog. A user pressing Alt+1 after reading "Spec" in the help dialog gets silence. The help and the behavior are now inconsistent, and the inconsistency is not guarded by any test.

**Why this is Eroding Goals and not just Tragedy of the Commons:** The sequence is instructive. Phase 3A.5 removed the Runs tab; Phase 3A.6 removed the Spec tab. Both commits touched `App.tsx` for unrelated changes but neither updated the tabMap or help text. The green bar was preserved by the absence of a test that would catch this, not by completing the cleanup. The standard ("every commit leaves the app green") eroded under delivery pressure: the commits were "green" in the sense that vitest passed, but the behavioral invariant (advertised keyboard shortcuts work) was silently violated.

**Intervention level:** Meadows Level 6 (Information Flows) — add a test that asserts every key in `tabMap` is a member of the live `TABS` constant in `InspectorPanel`, so future tab removals fail fast rather than silently breaking advertised shortcuts.

---

### 1.3 Fixes that Fail — Phase 3A.4 auto-validate vs Phase 2C handleValidate deletion

**Finding: NONE**

Phase 2C deleted `handleValidate` and its try/catch from `InspectorPanel` and relocated all validation side-effects into `subscriptions.ts` as a Zustand subscriber. The key risk for Phase 3A.4 was that adding auto-validate might resurrect the try/catch pattern.

It did not. The Phase 3A.4 auto-validate code (`subscriptions.ts:146–197`) is a third Zustand subscriber inside `initStoreSubscriptions()`. It calls `validate()` from `executionStore` via the store's own async action, which already owns its own error handling. There is no try/catch wrapping the validate call at the subscriber level, and no `handleValidate` symbol appears anywhere in production code. The FRAGILE comment on line 175 is honest documentation of the `pendingValidateTarget = null`-before-await sequencing; it is not a hidden error-swallow.

The cross-session guard (`inflightValidateSessionId`) correctly suppresses stale validation side-effects without consuming the fingerprint, as confirmed by the last two tests in the `auto-validate` describe block.

**Phase 2C regression verdict:** Not introduced. Phase 3A.4 is additive and clean.

---

### 1.4 Tragedy of the Commons — Orphan artifacts after component deletions

**Finding: MAJOR (one confirmed) + MINOR (two confirmed)**

Three components were deleted across the commit range: `RunsView` (Phase 3A.5), `SpecView` (Phase 3A.6), and `SessionSidebar` (Phase 3A.7). The commits deleted the component files, their test files, the import sites, and the CSS rules. However, three orphan artifacts survived:

#### Major orphan: `inspector-page.ts:6` type union includes deleted tab names

File: `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts`

```typescript
// Line 1 (comment)
// Page object for the right-side inspector panel (tabs: spec/graph/yaml/runs).
// Line 6
export type InspectorTab = "spec" | "graph" | "yaml" | "runs";
```

This page object is the E2E test harness for the inspector panel. The comment on line 1 and the type union on line 6 both name `spec` and `runs` as valid tab identifiers. No current E2E spec calls `.openTab("spec")` or `.openTab("runs")`, but the type admits them without a compile error (TypeScript will not flag the union; only a runtime Playwright call would fail). The concern is that the next author writing an E2E test against the inspector will read this type and believe "spec" and "runs" are valid tabs. The type is the documentation here.

This is Tragedy of the Commons: the page object is shared infrastructure, nobody felt individually responsible for updating it when they deleted the tabs, and the cost of the stale type will be paid by the next person who uses it incorrectly.

#### Minor orphan: `useHashRouter.ts:33–36` has a dead `REMOVED_TAB_MESSAGES` entry for "spec"

The `REMOVED_TAB_MESSAGES` object correctly handles stale URL hashes for both "runs" and "spec" — this is intentional and desirable. However, the redirect toast dismissal key (`elspeth_redirect_toast_dismissed`) is shared across both migration notices: if a user dismisses the "Runs tab was removed" notice, they will never see the "Spec tab was removed" notice either. These two tab removals happened in different commits; the redirect notice was designed for the first removal and the second removal reused it without checking whether a per-removal key was appropriate. This is a minor behavioral gap — the toast mechanism works for the first tab removal a user encounters, then silently suppresses subsequent ones.

#### Minor orphan (policy): Redirect-toast migration machinery and RETIRED_SIDEBAR_COLLAPSED_KEY shim

Two migration artifacts share the same policy problem under CLAUDE.md's No Legacy Code Policy: "WE HAVE NO USERS YET — deferring breaking changes is the opposite of what we want." and "When something is removed or changed, DELETE THE OLD CODE COMPLETELY. No version checks, feature flags for old behaviour, adapter/wrapper/proxy shims."

**RETIRED_SIDEBAR_COLLAPSED_KEY** (`App.tsx:28,74`):

```typescript
const RETIRED_SIDEBAR_COLLAPSED_KEY = "elspeth_sidebar_collapsed";
// ...in a useEffect:
localStorage.removeItem(RETIRED_SIDEBAR_COLLAPSED_KEY);
```

A useEffect cleanup shim for a localStorage key that has no real user data to migrate. The correct action is to simply stop writing the old key — no cleanup is needed.

**Redirect-toast migration machinery** (`useHashRouter.ts:33-36`, `App.tsx:233-245`):

```typescript
const REMOVED_TAB_MESSAGES: Record<string, string> = {
  runs: "The Runs tab was removed in this update. Showing Graph instead.",
  spec: "The Spec tab was removed in this update. Showing Graph instead.",
};
// + redirectToast state + dismissal localStorage key + banner UI in App.tsx
```

This is approximately 30+ LOC of migration machinery for users who have stale URL bookmarks to tabs that were never in production. The policy caveat is that `elspeth.foundryside.dev` (per `project_staging_deployment` memory) is in real active use, so the "no users yet" assertion is strictly true only for end users, not for the team's own staging bookmarks. Both artifacts are in the same category: soft migrations for a transient population. The RETIRED key cleanup is smaller (2 LOC) but the redirect-toast machinery is larger; applying the policy to one and not the other would be inconsistent. The reviewer's position is that both are minor policy gaps in the same category rather than a BLOCK-level concern, given the staging context.

---

### 1.5 Tragedy of the Commons — CSS dead class check

**Finding: NONE**

The App.css changes in Phase 3A.7 were thorough. All old selectors (`layout-sidebar`, `layout-sidebar-toolbar`, `layout-sidebar-content`, `sidebar-toggle`, `inspector-overlay-backdrop`, `inspector-overlay-close`, `layout-inspector`) were removed. New selectors (`layout-siderail`, `layout-siderail--overlay`, `siderail-overlay-backdrop`, `siderail-overlay-close`, `.side-rail`, `.side-rail-slot`, `.side-rail-transitional`) were added. No orphan CSS class names from deleted components were found.

---

### 1.6 Limits to Growth — SideRail slot architecture extensibility

**Finding: NONE**

The SideRail component (`SideRail.tsx:10–52`) uses explicit `ReactNode | null` props for each of the six planned tenants:

```typescript
interface SideRailProps {
  auditReadinessSlot?: ReactNode | null;
  graphMiniSlot?: ReactNode | null;
  catalogSlot?: ReactNode | null;
  exportYamlSlot?: ReactNode | null;
  executeButtonSlot?: ReactNode | null;
  completionBarSlot?: ReactNode | null;
  children?: ReactNode;
}
```

The CSS for `.side-rail-slot` uses `display: flex; flex-direction: column;` with `:empty { display: none }` — vacant slots collapse to zero height rather than occupying fixed space. The `.side-rail-transitional` wrapper that holds the `InspectorPanel` child uses `flex: 1 1 auto; min-height: 0` which allows it to take all remaining space without hard-coding a height budget.

This is not a hard-coded scaffold. Adding a 15b tenant is a two-step operation: add a prop to `SideRailProps` and wire the content at the `App.tsx` call site. No dimensions need re-cutting. The architecture handles extensibility correctly. The one mild observation is that the `children` prop (carrying `InspectorPanel` as the transitional occupant) is structurally separate from the named slots; when `InspectorPanel` is eventually migrated into specific named slots, the `children` prop can be cleanly deleted.

---

### 1.7 Drift from Prior Memory

#### Memory: `project_phase2c_implementation_complete.md`

The Phase 2C memory records: "commit f25a082d4 — Delete Validate button + handleValidate/canValidate/isValidating/injectSystemMessage/sendValidationFeedback from InspectorPanel; add useExecutionStore.validationResult subscriber to subscriptions.ts."

Phase 3A.4 is consistent with this. The auto-validate subscriber is a third subscriber, additive. `injectSystemMessage` and `sendValidationFeedback` remain in subscriptions.ts from Phase 2C; Phase 3A.4 does not remove or duplicate them. Consistent.

The Phase 2C memory also records: "When Phase 3A is next worked on, 15a2 Task 8 must be marked absorbed — that doc edit is a separate cycle." **Verified:** `docs/composer/ux-redesign-2026-05/15a2-phase-3a-removals-part-2.md` line 730 reads "## Task 8: ~~Remove Validate button from inspector header~~ **RETIRED 2026-05-17 — Phase 2C absorbed**". The absorption is documented. No drift.

#### Memory: `project_phase1b_default_mode_frontend_complete.md`

Phase 1B delivered `UserMenu` inside the sidebar toolbar with specific wiring (`onOpenSettings`, `onSignOut`). Phase 3A.3 moved `UserMenu` into a new `AppHeader` component and Phase 3A.7 removed the sidebar entirely. The wiring still passes through but via a different path: `App.tsx → AppHeader → UserMenu`. The `DefaultModeChangedBanner` previously mounted inside `layout-chat`; it still mounts inside the chat column in the new Layout. Phase 1B's "height budget concern" comment is preserved in the CSS structure. No drift.

Phase 1B's three E2E gotchas are still valid (serial mode, `waitForResponse` before click, `getByLabel` for guided body) and nothing in Phase 3A changes the behavioral surface that motivated them.

#### Memory: `feedback_locked_in_buggy_expectations.md`

The tabMap / ShortcutsHelp finding (1.2 above) is the inverse of this pattern: the tests did not catch the inconsistency because no test encodes "Alt+1 should open Spec tab." The test gap is the problem, not locked-in expectations. The policy implication is the same: add a test that would have caught this.

#### Memory: `feedback_no_path_priming_in_dispatch.md`

Not applicable to this commit range. This memory governs persona-subagent dispatch prompts, which are not part of the Phase 3A code changes.

---

## 2. Confidence Assessment

- **Shifting the Burden (NONE):** High confidence. No Python files changed; no import graph affected.
- **Eroding Goals (MAJOR):** High confidence. Evidence is the live file content of `ShortcutsHelp.tsx:15` and `App.tsx:155-160`, confirmed against git log showing neither was touched by the tab-removal commits.
- **Fixes that Fail (NONE):** High confidence. I read the full diff of `subscriptions.ts` and confirmed no try/catch wraps `validate()`.
- **Tragedy of the Commons (MAJOR/MINOR):** High confidence. `inspector-page.ts` type union and the RETIRED_SIDEBAR_COLLAPSED_KEY shim are directly in the diff. The shared dismissal key is a behavioral inference but a sound one.
- **Limits to Growth (NONE):** High confidence. I read `SideRail.tsx` in full and the relevant App.css rules. The slot model is composable.

---

## 3. Risk Assessment

**Highest priority:** The Alt+1 / Alt+4 silent failure (tabMap + ShortcutsHelp.tsx). Users see advertised shortcuts; the shortcuts silently fail. During a demo, a user pressing Alt+1 expecting to open Spec view gets no response with no error. This is the most visible defect in a demo context.

**Second priority:** `InspectorTab` type in `inspector-page.ts`. Not a runtime defect today, but a trap for the next E2E test author. As the Playwright suite grows, the probability of someone calling `.openTab("spec")` and getting a timeout rather than a TypeScript error increases.

**Third priority:** RETIRED_SIDEBAR_COLLAPSED_KEY. Policy violation but zero user impact — confirmed by "WE HAVE NO USERS YET."

**Fourth priority:** Shared redirect toast dismissal key. The second migration notice is silently suppressed if a user dismissed the first. Low-impact given the migration is a one-time event per user.

---

## 4. Information Gaps

- Whether `confirmFanoutExecution`'s removal of the `SWITCH_TAB_EVENT` "runs" dispatch (Phase 3A.7, `App.tsx:89-94`) was also accompanied by an E2E test update. No E2E test was found calling `confirmFanoutExecution` directly, so this is likely clean but not confirmed.

---

## 5. Caveats

- This review covers `9e8c54ed6..c3fc5670e` only. The underlying correctness of the auto-validate subscriber against a real backend (round-trip latency, rate limiting, error handling on network failure) was not assessed and is not assessable from static analysis.
- CSS analysis was performed by grep, not by running a CSS coverage report. Orphan selectors that are conditionally applied and never triggered at runtime could still exist.
- The `compositionState.version` cadence was traced to server-assigned increments (via `sendMessage` / `recompose` API calls), not per-keystroke. This eliminates the auto-validate rate-hammering concern the task brief flagged.

---

## 6. Findings Summary Table

| # | Severity | Archetype | Finding | File(s) | Evidence | Meadows Level |
|---|----------|-----------|---------|---------|----------|---------------|
| 1 | **MAJOR** | Eroding Goals | `App.tsx` tabMap still maps `"1"→"spec"` and `"4"→"runs"` after those tabs were deleted; Alt+1/Alt+4 silently no-op | `src/.../App.tsx:155-160` | tabMap not changed in 66748edb9 or 8b04d53ce diffs | L6 — Information Flows (add test asserting tabMap ⊆ TABS) |
| 2 | **MAJOR** | Eroding Goals | `ShortcutsHelp.tsx` advertises "Spec/Graph/YAML/Runs" tabs in user-facing text; two of four are deleted | `src/.../ShortcutsHelp.tsx:15` | File not touched in 9e8c54ed6..c3fc5670e range | L6 — Information Flows (same test would catch this) |
| 3 | **MAJOR** | Tragedy of the Commons | `inspector-page.ts` `InspectorTab` type union includes `"spec"` and `"runs"`; page comment names both deleted tabs | `tests/e2e/page-objects/inspector-page.ts:1,6` | File not updated by any tab-removal commit | L6 — Information Flows (update type; add guard comment) |
| 4 | MINOR | Tragedy of the Commons | Shared redirect-toast dismissal key suppresses second migration notice if first was dismissed; applies to both tab migrations | `src/.../hooks/useHashRouter.ts:33-36,68-70` | Single key `elspeth_redirect_toast_dismissed` covers both "spec" and "runs" migrations | L5 — Rules (use per-tab dismissal key if notices must be independently dismissible) |
| 5 | MINOR | Policy gap (No Legacy Code) | Redirect-toast migration machinery (~30 LOC) + `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup shim (2 LOC) are soft migrations for a pre-production system; inconsistent with "WE HAVE NO USERS YET / DELETE THE OLD CODE COMPLETELY." Staging use is the mitigating context. | `src/.../hooks/useHashRouter.ts:33-36`, `src/.../App.tsx:28,74,233-245` | CLAUDE.md No Legacy Code Policy; both artifacts are migration paths | L10 — Structure (operator-adjudicate; delete both if staging bookmarks are deemed non-users) |
| 6 | NONE | Fixes that Fail | Phase 3A.4 auto-validate does NOT resurrect Phase 2C's deleted `handleValidate` try/catch; the new subscriber is cleanly additive | `src/.../stores/subscriptions.ts:146-197` | No `handleValidate` in production code; no try/catch wraps `validate()` | — |
| 7 | NONE | Shifting the Burden | No lazy imports or apologetic comments deferring L3 layer structural fixes | All `.ts`/`.tsx` diffs | No Python changed; no `TYPE_CHECKING` blocks added | — |
| 8 | NONE | Limits to Growth | SideRail slot architecture is composable; slots are `ReactNode\|null` props, vacant slots collapse, transitional child cleanly deletable | `src/.../common/SideRail.tsx` | `:empty { display: none }` + `flex: 1 1 auto` CSS; explicit named props | — |
| 9 | NONE | Drift from memory | Phase 2C Task 8 absorption: `15a2-phase-3a-removals-part-2.md:730` confirms "RETIRED 2026-05-17 — Phase 2C absorbed"; no carry-forward gap | `docs/.../15a2-phase-3a-removals-part-2.md:730` | Verified by grep | — |

**Severity counts:** BLOCK: 0, MAJOR: 3, MINOR: 2, NONE confirmed: 4
