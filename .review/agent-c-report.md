# Phase 3A Review — Agent C

**Scope.** `git diff 9e8c54ed6..c3fc5670e` on branch `feat/composer-phase-3-ia-cleanup`.
HEAD: `c3fc5670e7598d2620468b44be90740e82279fa2`.

Commits in scope:

- `84306f19f` feat: InlineRunResults + RunsHistoryDrawer (Phase 3A.1)
- `ccca39cb0` feat: SideRail scaffold + rename Layout slot (Phase 3A.2)
- `ae8df6dc3` feat: AppHeader with session switcher + UserMenu (Phase 3A.3)
- `86061cf1c` feat: auto-validate on compositionState.version increment (Phase 3A.4)
- `66748edb9` feat: remove Runs tab (Phase 3A.5)
- `8b04d53ce` feat: remove Spec tab (Phase 3A.6)
- `c3fc5670e` feat: remove SessionSidebar (Phase 3A.7)

**Build status.**

- `npm run typecheck` — clean.
- `npm test -- --run` — **671 / 671 tests pass** (68 files). React-Flow `ResizeObserver is not defined` warnings appear in the output stream but are unhandled jsdom/@xyflow noise; the suite still reports green.

---

## Findings

### BLOCK — none.

### MAJOR

#### M1. Dead `Alt+1` / `Alt+4` keyboard mappings to removed tabs

**File:** `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup/src/elspeth/web/frontend/src/App.tsx:153-169`

The Alt-key tab-switch handler still dispatches `SWITCH_TAB_EVENT` for `"spec"` and `"runs"`. `InspectorPanel`'s listener (post-removal) only accepts `"graph" | "yaml"`, so these key presses are now silent no-ops — an exact violation of the "No Legacy Code" rule that *all call sites must be changed in the same commit*. The `CommandPalette` was correctly pruned in the same change; this site was missed.

Minimal patch:

```tsx
// Alt+1/2: Switch inspector tabs
if (e.altKey && !e.ctrlKey && !e.metaKey) {
  const tabMap: Record<string, string> = {
    "1": "graph",
    "2": "yaml",
  };
  // ...
}
```

#### M2. `ShortcutsHelp` still advertises the removed `Spec` and `Runs` tabs

**File:** `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup/src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx:15`

```tsx
{ keys: "Alt+1-4", action: "Switch inspector tab (Spec/Graph/YAML/Runs)" },
```

Outside the diff but functionally part of the same removal — the user-visible keyboard-shortcuts overlay still teaches users to press `Alt+1` for Spec and `Alt+4` for Runs. Update to e.g. `{ keys: "Alt+1-2", action: "Switch inspector tab (Graph/YAML)" }` in the same commit that drops the dead Alt mappings (M1).

#### M3. Stale `Runs tab` mentions in `executionStore` user-visible text

**File:** `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup/src/elspeth/web/frontend/src/stores/executionStore.ts:306` (comment) and `:331` (user-facing error message).

Line 331 — `"Pipeline execution failed. Check the Runs tab for error details."` — is a string that surfaces in chat. The Runs tab no longer exists; the message now points the user at a non-existent affordance. Replace with a reference to the new `InlineRunResults` / `Past runs` button (or a more general "Check the run results panel for error details").

Line 306's comment is stale orientation for future readers; update to "Refresh runs list so the new run appears in InlineRunResults immediately" or similar.

#### M4. Stale `Spec tab` reference in `App.css`

**File:** `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup/src/elspeth/web/frontend/src/App.css:595`

Section banner — `Interactive Component Highlight (click-to-highlight in Spec tab)` — references the deleted tab. Either drop the section (if the styles are now unused), or relabel to the surviving affordance. Net out the unused rules in the same pass.

#### M5. Migration shims preserved for users that "don't exist"

CLAUDE.md "No Legacy Code Policy" is explicit: *WE HAVE NO USERS YET — deferring breaking changes is the opposite of what we want.* The diff introduces three preservation shims:

| Shim | File | Line(s) |
|---|---|---|
| `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup useEffect | `App.tsx` | 28, 73-75 |
| `SIDERAIL_WIDTH_KEY = "elspeth_inspector_width"` rename-but-preserve-key | `components/common/Layout.tsx` | 14, 112-114 |
| Removed-tab redirect-toast machinery + dismissal-persistence | `hooks/useHashRouter.ts` | 27-32, 67-79, 86-88, 200-209 |

Each *only* exists to spare a hypothetical user the consequences of a hash they bookmarked or a localStorage key they accumulated. Per project rules the operator surfaces breaking changes by deleting the old key/path outright. The redirect-toast on `elspeth.foundryside.dev` staging is the most defensible (operator courtesy on a live host — see memory `project_staging_deployment.md`), but the other two are pure cruft. Surface this to the operator and default to deletion if they don't actively want them kept.

Concrete deletion patches:

```diff
- const RETIRED_SIDEBAR_COLLAPSED_KEY = "elspeth_sidebar_collapsed";
- useEffect(() => {
-   localStorage.removeItem(RETIRED_SIDEBAR_COLLAPSED_KEY);
- }, []);
```

```diff
- const SIDERAIL_WIDTH_KEY = "elspeth_inspector_width";
+ const SIDERAIL_WIDTH_KEY = "elspeth_siderail_width";
// (and drop the "intentionally preserves its pre-rename value" comment)
```

---

### MINOR

#### N1. `SideRail` accepts `children` as a transitional inspector mount

**Files:** `components/common/SideRail.tsx:17,46-46`, `App.tsx:298-309`

`SideRail` declares six explicit slot props plus a `children` escape hatch the comment labels "transitional." `App.tsx` then passes `<InspectorPanel />` through `children` while every named slot is `null`. The scaffold is forward-looking rather than legacy, but the present-tense shape is a no-op wrapper around the inspector. Once Phase 3B moves Audit Readiness / Graph mini / Catalog / Export YAML / Execute / Completion Bar into their typed slots, the `children` prop should be removed and the `side-rail-transitional` div with it. Track as Phase 3B follow-up; not a BLOCK because the comment is honest about the scaffold's lifespan.

#### N2. `Layout`'s preserved-comment about retired `calc(100vh - …)` budget

**File:** `components/common/Layout.tsx` — original (deleted) comment about UserMenu being threaded through Layout "to avoid a header row that would change the calc(100vh - …) height budget" was removed. The new `AppHeader` *does* introduce a top-level header row above `.app-main`. Verify in browser that the chat column and side rail still fit on viewport heights at the existing breakpoints — if the inspector now overflows below the fold, this is a behavioural regression hidden by the rename. The unit tests stub `Layout`, so this is uncovered.

#### N3. `fireValidateLoop` race annotation

**File:** `stores/subscriptions.ts:181-191`

Comment marks the pattern `FRAGILE`. Logic is correct (`pendingValidateTarget = null` before `await`, the subscription requeues if a newer version arrives during the await, the loop re-checks `lastValidatedVersionBySession` and `activeSessionId` between iterations). Tests at `stores/subscriptions.test.ts:295-320` cover the correctness loop. No fix required; flag because "fragile + module-level mutable state + async loop" is a future-bug hot spot and warrants a brief design note in the file header or a follow-up issue to memoise the invariants tested.

#### N4. `App.tsx:84` and `:109` use `console.error` for non-fatal bootstrap and health-check failures

These pre-date this diff (not introduced by Phase 3A) so they fall outside scope, but worth recording because the file was touched: per CLAUDE.md telemetry-primacy `console.error` is *not* the right channel — preferences-bootstrap failure and health-check failure should be operational telemetry, not browser-console noise. The operator already has a memory note (`feedback_no_slog_recommendations.md`) on the equivalent backend rule. Out of scope for this BLOCK list because the lines are not new; flagging because someone is likely to read this review and copy the pattern.

---

### NIT

#### T1. `HeaderSessionSwitcher` test uses `as never` proliferation for store stubbing

**File:** `components/sessions/HeaderSessionSwitcher.test.tsx:11-15, 34, 64, 70-72, 81`

`as never` is the project's accepted pattern for typed-store stubbing in Vitest (confirmed by surveying other store tests). Not a flag against this PR, just calling out for future cleanup if a typed `mockStore<T>()` helper lands.

#### T2. `AppHeader.test.tsx` only verifies render-presence

**File:** `components/common/AppHeader.test.tsx` — three tests all assert `getByText / getByRole(...).toBeInTheDocument()`. CLAUDE.md test-discipline forbids tests that *only* assert "renders without crashing," but these do check specific roles and text and the underlying behaviour (session-switcher dropdown, user menu actions) is tested in the dedicated `HeaderSessionSwitcher.test.tsx` and `UserMenu.test.tsx`. Acceptable composition-level smoke for a 12-line wrapper.

---

## Confidence-graded summary table

| Sev | ID | File:line | Issue | Confidence |
|---|---|---|---|---|
| MAJOR | M1 | `src/App.tsx:153-169` | Dead `Alt+1`/`Alt+4` mappings to removed `spec`/`runs` tabs | 95 |
| MAJOR | M2 | `src/components/common/ShortcutsHelp.tsx:15` | Shortcut help still advertises `Spec/Graph/YAML/Runs` | 95 |
| MAJOR | M3 | `src/stores/executionStore.ts:306,331` | User-facing error and code comment reference the removed Runs tab | 90 |
| MAJOR | M4 | `src/App.css:595` | CSS comment + likely-unused rules reference the removed Spec tab | 80 |
| MAJOR | M5 | `App.tsx:28,73-75`; `Layout.tsx:14`; `useHashRouter.ts:27-32,200-209` | Migration shims for users that "don't exist" per `No Legacy Code Policy` | 85 |
| MINOR | N1 | `components/common/SideRail.tsx:17,46`; `App.tsx:298-309` | `SideRail.children` transitional escape hatch — retire in Phase 3B | 80 |
| MINOR | N2 | `components/common/Layout.tsx` | New `AppHeader` row may regress the deleted `calc(100vh - …)` height budget; uncovered by stubbed Layout tests | 70 |
| MINOR | N3 | `stores/subscriptions.ts:181-191` | `FRAGILE` correctness loop — covered by tests, deserves a design note | 60 |
| MINOR | N4 | `App.tsx:84,109` | Pre-existing `console.error` for telemetry-worthy events (out of scope but touched file) | 60 |
| NIT | T1 | `HeaderSessionSwitcher.test.tsx` | `as never` stubbing pattern — accepted convention | 30 |
| NIT | T2 | `AppHeader.test.tsx` | Render-only smoke tests, behaviour covered elsewhere | 30 |

---

## Build verification

```
npm run typecheck         # PASS (tsc -p tsconfig.app.json --noEmit, clean)
npm test -- --run         # PASS — 671/671 tests across 68 files
```

(ResizeObserver console warnings during vitest run are jsdom + `@xyflow/react` interaction noise; suite still reports green.)
