# Phase 3B Test Suite Review — Agent D (v2)

**Commit range:** `9e8c54ed6..HEAD` (26 commits, includes Phase 3A + Phase 3B + fix-loop)
**HEAD:** `020db17e3551514bf865e83060f43d803acb921d`
**Branch:** `feat/composer-phase-3-ia-cleanup`
**Review date:** 2026-05-17
**Vitest total:** 686 passed / 686 (75 test files)

---

## Confidence Assessment

High confidence on structural findings (presence/absence of test coverage, ARIA contract
assertions, anti-patterns). High confidence on the P3A-002 re-triage — confirmed by reading
the commit that deleted the test and the replacement tests in App.test.tsx. Moderate
confidence on the P3A-013 hygiene items that reference existing files not reopened in this
session; they are confirmed green (686/686 passes) but not all modified-existing test files
were re-read in full depth.

---

## Risk Assessment

The highest-risk finding in this cycle is that the new YAML export and graph modals lack
focus-restore assertions. Both use `useFocusTrap`, which does restore focus (confirmed by
reading the implementation at `src/hooks/useFocusTrap.ts:57-63`), but no test verifies this.
A future refactor that breaks the focus chain would be invisible until discovered by a
keyboard-only user.

Second risk: no equivalent of the P3A-002 mechanical guard exists for the new modal shortcut
surface. A developer adding a new Ctrl+Shift key in App.tsx without updating ShortcutsHelp
will pass all tests and CI.

---

## Information Gaps

- ChatPanel.test.tsx, CompletionSummary.test.tsx, Layout.test.tsx, ReadinessRowDetail.test.tsx,
  UserMenu.test.tsx, and executionStore.test.ts were modified in this range but not re-read in
  full depth in v2 (they are confirmed green). The modifications were spot-checked for sleep
  anti-patterns (none found) and assertion presence (confirmed). No structural issues identified
  from the diff metadata.
- E2E tests were not executed (require live backend). E2E findings are based on reading the
  spec files only.

---

## Caveats

1. The Phase 3A prior review (agent-d-report.md) was based on range `9e8c54ed6..c3fc5670e`.
   This v2 review covers the full range to HEAD including all Phase 3B additions and fix-loop
   commits. Prior-cycle entries are re-triaged per the synthesis.md identifiers (P3A-006,
   P3A-007, P3A-012, P3A-013).
2. The "prior deferred entries" referenced in the task map to synthesis.md identifiers. Their
   v2 disposition is recorded in the "Prior Cycle Re-triage" section below.

---

## Smoke Render — App.test

`npm test -- --run App.test` (frontend directory): **17/17 passed**.

The Phase 3B modifications to App.test.tsx add two new describe blocks: shortcuts dispatching
Ctrl+Shift+G/Y modal events (lines 297-325) and retired Alt+digit shortcuts no longer firing
(lines 327-344). Both pass correctly.

---

## Full Suite

`npm test -- --run` (frontend directory): **686/686 passed, 75 test files**.

Pre-Phase-3B baseline was 672; the writer's most recent commit reported 686. Count confirmed.
Net addition from Phase 3B: 14 tests (686 − 672). The breakdown is approximately: new files
add ~70 tests; deleted files remove ~56 tests (InspectorPanel.test.tsx 734 lines, but many
describe blocks overlap; net is +14 from the diff).

---

## P3A-002 Guard — Detailed Re-triage

**Verdict: MOOT (guard deleted correctly) — NEW gap exists for the replacement surface.**

Sub-question answers:

- **Is `TAB_SHORTCUT_MAP` still non-empty?** `TAB_SHORTCUT_MAP` no longer exists. It was a
  named constant in `App.tsx` mapping Alt+digit keys to inspector tab IDs. Commit `82957ea14`
  (retarget app shortcuts) deleted the constant and rewrote the keyboard handler to dispatch
  Ctrl+Shift events for modal actions directly, with no runtime map.
- **Is `TABS` still non-empty?** `TABS` no longer exists. It was exported from
  `InspectorPanel.tsx`, which was deleted wholesale in commit `a43594051`. There is no
  inspector tab array anywhere in the current codebase.
- **Does the test still iterate and assert membership?** The test file
  `src/__tests__/tab-shortcut-wiring.test.ts` was deleted in commit `82957ea14` along with
  the constructs it guarded. There is nothing to iterate.
- **Does the test pass vacuously?** It does not pass vacuously — it does not run at all. Under
  CLAUDE.md's No Legacy Code Policy, deleting the test along with the dead constructs is
  correct. A vacuous-pass would require the test file to still exist but with empty iteration.

**New gap (no P3A predecessor):** The replacement shortcut surface (Ctrl+Shift+G → Graph modal,
Ctrl+Shift+Y → YAML modal, Ctrl+Shift+P → Catalog) has behavioral coverage in App.test.tsx
lines 297-325. However, there is no mechanical coherence guard analogous to the old
tab-shortcut-wiring guard: no test asserts that every App.tsx keyboard shortcut has a
corresponding entry in ShortcutsHelp. If a developer adds a new Ctrl+Shift+X shortcut in
App.tsx but forgets to update ShortcutsHelp, no test fails. This is classified as a new
MINOR finding (B3B-002 below), not a P3A regression.

---

## Phase 3B Coverage — New Components

### 1. YAML Export Modal (`ExportYamlModal.test.tsx`)

Coverage: open on event, close on Escape, close on backdrop click, close on button click,
renders YamlView. `getByRole("dialog", { name: /export yaml/i })` confirms both `role` and
accessible name.

Gaps:
- `aria-modal="true"` is never directly asserted. The test uses `getByRole("dialog")` which
  validates `role` but does not verify the `aria-modal` attribute. A future change that removes
  `aria-modal="true"` leaves the dialog non-modal to assistive technology without failing the
  test.
- Focus-trap and focus-restore on close are not asserted. The implementation wires `useFocusTrap`
  (confirmed), which restores focus on deactivation. No test verifies that after Escape, focus
  returns to the trigger element.

### 2. Graph Mini Modal (`GraphModal.test.tsx`)

Same pattern as ExportYamlModal. Identical gaps: `aria-modal` not directly asserted; no
focus-restore test.

One additional item in `useHashRouter.test.ts`: the cold-load test (line 79) renders GraphModal
alongside `useHashRouter` and asserts the dialog appears via `getByRole("dialog")`. This is a
useful integration-level check that confirms the event-bus wiring.

### 3. Header Composition History Selector (`HeaderVersionSelector.test.tsx`)

Coverage: no-active-session → null render; version label displays; "Composition history"
button present; click loads versions; version pick + confirm + revert flow.

Gap: the confirm dialog's cancel path is not tested. If the user opens the revert confirm and
clicks Cancel, `revertToVersion` must not be called. There is no test for this branch.

### 4. Graph Mini View (`GraphMiniView.test.tsx`)

Coverage: aria-labelled button present; empty state when no composition; click dispatches
OPEN_GRAPH_MODAL_EVENT.

Gap: the stub of `GraphView` via `vi.mock` suppresses the actual rendering; no test verifies
that the embedded GraphView renders pipeline nodes. This is proportionate for a thin wrapper
component.

### 5. Catalog Button (`CatalogButton.test.tsx`)

Coverage: button presence; click dispatches OPEN_CATALOG_EVENT. Clean.

### 6. Execute Button (`ExecuteButton.test.tsx`)

Coverage: null render with no session; Run button when valid; button disabled when invalid;
execute called with session id. Clean.

### 7. Export YAML Button (`ExportYamlButton.test.tsx`)

Coverage: null render with no session; button renders with session; click dispatches
OPEN_YAML_MODAL_EVENT. Clean.

### 8. SideRail (`SideRail.test.tsx` — Phase 3B additions)

Phase 3B expanded SideRail from a scaffold to a fully-wired component with slots for:
validation banner, graph mini, catalog, export-yaml, execute-button, completion-bar. Tests
cover all six slot data-testids plus the `executeButtonSlot` prop wire-through and DOM order
(validation above graph).

Gap inherited from Phase 3A: `aria-label="Composer side rail"` on the `<aside>` landmark is
still not asserted (confirmed by re-reading SideRail.test.tsx, same finding as v1).

### 9. SideRail Validation Banner (`SideRailValidationBanner.test.tsx`)

Coverage: execution-store errors as `role="alert"`; validation result errors rendered as
component buttons; click selects node and does not dispatch retired `elspeth-switch-tab` event.
The third test is a direct re-homing of the InspectorPanel behavior for validation-click
handling. This is a correctly re-homed test from the deleted InspectorPanel suite.

### 10. Retargeted App Shortcuts (`App.test.tsx` additions)

Three new tests in the Phase 3B range:
- Ctrl+Shift+G dispatches OPEN_GRAPH_MODAL_EVENT
- Cmd+Shift+Y dispatches OPEN_YAML_MODAL_EVENT
- Alt+1 does not dispatch `elspeth-switch-tab`

These correctly replace the behavioral intent of the deleted `tab-shortcut-wiring.test.ts`.

### 11. Retargeted Command Palette (`CommandPalette.test.tsx` additions)

Phase 3B adds tests confirming: "Open graph view" command dispatches OPEN_GRAPH_MODAL_EVENT;
"Export YAML" command dispatches OPEN_YAML_MODAL_EVENT; old "Switch to Graph Tab" / "Switch to
YAML Tab" / "Switch to Runs Tab" / "Switch to Spec Tab" commands are absent.

Coverage is complete for the retargeted commands.

### 12. ShortcutsHelp (`ShortcutsHelp.test.tsx` additions)

Phase 3B adds three tests: plugin catalog shortcut listed with correct keys; graph and YAML
modal shortcuts listed; retired Alt+1-2 inspector tab shortcuts absent.

Coverage is complete for the new shortcut surface in the help dialog.

### 13. Hash Router (`useHashRouter.test.ts` — Phase 3B rewrite)

The test file has been substantially updated. It now covers Phase 3B fragment migration:
`#/{id}/spec` → `#/{id}`, `#/{id}/runs` → `#/{id}`, `#/{id}/graph` → open graph modal +
strip verb, `#/{id}/yaml` → open yaml modal + strip verb, unrecognized verb stripped. Plus a
cold-load integration test confirming GraphModal appears when the hash contains `/graph` before
mount.

**This is a substantial improvement over Phase 3A coverage.** Five of the six route cases are
now tested at unit level.

Remaining gap (same class as Phase 3A v1 report): session-change `pushState`, popstate
back/forward navigation, initial hash-less write, and invalid-session clearing are still not
covered at unit level. These are the same gaps as Phase 3A; the v2 re-triage below records
their current status.

---

## Phase 3B Coverage — Deletions

### `InspectorPanel.tsx` deleted (`a43594051`)

The deleted InspectorPanel.test.tsx (734 lines) contained tests for: ValidationDot states,
tab navigation, Runs/Spec tab removal guards (added in Phase 3A), validation-component click
routing to graph tab, CompositionSummary rendering, GraphView tab content,
YamlView tab content, history panel load, revert flow, audit-readiness panel load.

**Re-homing status at Phase 3B HEAD:**

| Old InspectorPanel test | Status at HEAD |
|---|---|
| ValidationDot (amber/green/red) | NOT re-homed. No test covers the validation state indicator in the new SideRailValidationBanner or SideRail slot assembly. The dot rendered in the old tab strip; no equivalent exists in Phase 3B. |
| Runs tab removal (Phase 3A) | Superseded by `CommandPalette.test.tsx` and `ShortcutsHelp.test.tsx` negative assertions. ADEQUATELY REPLACED. |
| Spec tab removal (Phase 3A) | Same as Runs tab removal. ADEQUATELY REPLACED. |
| Validation-component click routes to graph tab | RE-HOMED to `SideRailValidationBanner.test.tsx` line 56 ("does not dispatch retired inspector tab events"). Correctly re-homed. |
| GraphView tab content | Covered at unit level by `GraphModal.test.tsx` (stub) and `GraphMiniView.test.tsx`. ADEQUATELY REPLACED. |
| YamlView tab content | Covered at unit level by `ExportYamlModal.test.tsx` (stub). ADEQUATELY REPLACED. |
| History panel load + revert flow | RE-HOMED to `HeaderVersionSelector.test.tsx`. ADEQUATELY REPLACED. |
| Audit-readiness panel | Unchanged (AuditReadinessPanel exists in SideRail slot; existing ReadinessRowDetail tests cover the panel). ADEQUATELY REPLACED. |

The ValidationDot re-homing is the only material gap from the InspectorPanel deletion. Since
Phase 3B removes the inspector panel entirely, the visual validation indicator has no direct
successor component with a test. If a future phase adds a validation status indicator to the
SideRail or AppHeader, it will arrive without a prior test harness.

---

## Phase 3A Regression — E2E Spec

`tests/e2e/phase-3a-shell.spec.ts` tests: banner landmark visible; session switcher button
visible; account menu button visible; no legacy sidebar; account menu theme toggle visible.

All five assertions reference DOM that is still present in Phase 3B. The spec is not stale and
does not need updating for Phase 3B. It tests Phase 3A additions that are still correct.

What the spec does NOT cover (Phase 3B new flows): header composition history selector
interaction, YAML export modal open/close, graph mini modal open/close, Catalog trigger from
SideRail, Execute from SideRail. These are all unit-tested but have no E2E coverage.

---

## Anti-Patterns Check (Full Phase 3B Range)

### Sleepy assertions
None. All new and modified test files use `waitFor`, `act`, `await userEvent.*`, or
`fireEvent` for async operations. No `sleep()`, `setTimeout`, or polling loops found.

### Test interdependence
None. Every test that touches store state uses `beforeEach` with `resetStore()` or
`useXxxStore.setState({...})`. Event listener cleanup (removeEventListener) is performed
after each event-dispatch test.

### Shared mutable state
None introduced. New test files follow the existing pattern of per-test store resets.

### Missing assertions
None. Every new test has at least one `expect` call.

### Wrong test level
No test pyramid inversion. New components are unit-tested; E2E coverage is thin (by design
for the phase). The concern is insufficient E2E coverage, not inverted pyramid.

### Brittle queries (inherited)
`getByText(/r1/)` and `getByText(/r2/)` in RunsHistoryDrawer.test.tsx lines 21-22 (from
Phase 3A v1 NIT, unchanged). Still present; still a NIT.

### localStorage isolation
No new gaps introduced. Event listener cleanup patterns in CatalogButton, ExportYamlButton,
GraphMiniView tests correctly call `removeEventListener` after each test.

---

## Findings

### New Phase 3B Findings

| # | Severity | ID | File:Line | Description |
|---|---|---|---|---|
| 1 | MAJOR | B3B-001 | `ExportYamlModal.test.tsx`, `GraphModal.test.tsx` — entire files | No focus-restore assertion in either modal test; `useFocusTrap` restores focus but the contract is unverified — a future refactor breaking the chain is invisible |
| 2 | MINOR | B3B-002 | No file | No mechanical coherence guard for new modal shortcuts: App.tsx handler ↔ ShortcutsHelp listing. Replaces the P3A-002 guard's invariant for the new surface. |
| 3 | MINOR | B3B-003 | `HeaderVersionSelector.test.tsx:54-88` | Revert-confirm cancel path not tested; `revertToVersion` must not be called when user cancels, but no test covers this branch |
| 4 | MINOR | B3B-004 | No file | ValidationDot (visual validation state indicator in InspectorPanel) has no successor component test; deleted with InspectorPanel; no re-homing in SideRail/AppHeader |
| 5 | MINOR | B3B-005 | `ExportYamlModal.test.tsx`, `GraphModal.test.tsx` | `aria-modal="true"` not directly asserted; `getByRole("dialog")` validates role and name but not `aria-modal` attribute |
| 6 | MINOR | B3B-006 | `phase-3a-shell.spec.ts` — no update needed | E2E spec is correct for Phase 3A but covers zero Phase 3B flows (graph modal, YAML modal, catalog trigger, execute, history selector). Phase 3B E2E coverage is entirely absent. |
| 7 | NIT | B3B-007 | `SideRail.test.tsx` | `aria-label="Composer side rail"` on `<aside>` landmark still not asserted (inherited from Phase 3A, still present) |

### Phase 3A Findings — Re-triage

| Prior ID | Title | Disposition | Notes |
|---|---|---|---|
| P3A-006 | RunsHistoryDrawer ARIA contract not asserted | STILL-APPLIES | `aria-modal="true"` and `aria-label` on drawer root not asserted. Same gap now extends to GraphModal and ExportYamlModal (B3B-005 above). |
| P3A-007 | InlineRunResults "Past runs" click not exercised | STILL-APPLIES | Button presence confirmed; click-to-open-drawer still not tested. |
| P3A-012 | RunsHistoryDrawer focus-restore test | STILL-APPLIES | `useFocusTrap` wired correctly; no test verifies restoration. Same gap exists for new modals (B3B-001 above). |
| P3A-013 | Minor test-hygiene batch | STILL-APPLIES (partial) | Items from synthesis: SideRail aria-label (→ B3B-007), HeaderSessionSwitcher aria-current, AppHeader localStorage.clear, subscriptions isExecuting mid-loop, RunsHistoryDrawer brittle getByText. None have been addressed. |

**P3A-002 verdict:** MOOT. `TAB_SHORTCUT_MAP` does not exist; `TABS` does not exist;
`tab-shortcut-wiring.test.ts` does not exist. All three were deleted correctly when the
InspectorPanel and tab-map were deleted. The behavioral intent of the guard is replaced by
App.test.tsx lines 297-344. The mechanical coherence guarantee (shortcut map ↔ help dialog) is
not replaced; see B3B-002.

**MOOT count: 1** (P3A-002)
**STILL-APPLIES count: 4** (P3A-006, P3A-007, P3A-012, P3A-013)

---

## Architecture Assessment

- **Pyramid shape:** Healthy. 686 unit tests, thin E2E layer. Problem is E2E under-coverage
  for Phase 3B flows, not pyramid inversion.
- **Isolation:** Good. Store reset pattern is consistent across all new test files.
- **Determinism:** Good. No fixed sleeps. Event-dispatch and `fireEvent` patterns are
  synchronous; `act(async () => {})` is used correctly where useEffect async chains need
  flushing.
- **New component coverage:** Adequate for behavioral logic (open/close, dispatch, state-
  machine paths). Thin on ARIA contracts (`aria-modal`) and focus lifecycle (restore on close).
- **Deletion coverage:** Mostly clean. ValidationDot is the only test behavior without a
  successor. InspectorPanel's validation-click, history-revert, and graph/YAML content tests
  are all re-homed.
- **Shortcut surface:** Behavioral coverage is good; mechanical-coherence guard is absent.

---

## Summary Table (Full — Phase 3B Findings + Re-triaged)

| # | Severity | ID | File | Description |
|---|---|---|---|---|
| 1 | MAJOR | B3B-001 | ExportYamlModal.test.tsx, GraphModal.test.tsx | No focus-restore assertion; useFocusTrap contract unverified |
| 2 | MINOR | B3B-002 | (no file) | No mechanical coherence guard: App.tsx shortcuts ↔ ShortcutsHelp |
| 3 | MINOR | B3B-003 | HeaderVersionSelector.test.tsx | Revert-confirm cancel path not tested |
| 4 | MINOR | B3B-004 | (no file) | ValidationDot deleted with InspectorPanel; no successor test |
| 5 | MINOR | B3B-005 | ExportYamlModal.test.tsx, GraphModal.test.tsx | aria-modal="true" not directly asserted |
| 6 | MINOR | B3B-006 | phase-3a-shell.spec.ts (gap) | Zero E2E coverage for any Phase 3B flow |
| 7 | NIT | B3B-007 | SideRail.test.tsx | aria-label on aside landmark not asserted (inherited) |
| P3A-006 | MINOR | — | RunsHistoryDrawer.test.tsx | ARIA contract on drawer root not asserted — STILL-APPLIES |
| P3A-007 | MINOR | — | InlineRunResults.test.tsx | "Past runs" click to open drawer not tested — STILL-APPLIES |
| P3A-012 | NIT | — | RunsHistoryDrawer.test.tsx | Focus-restore on drawer close not asserted — STILL-APPLIES |
| P3A-013 | NIT | — | Multiple | Hygiene batch (aria-current, localStorage.clear, brittle queries) — STILL-APPLIES |

**Totals: 1 MAJOR, 6 MINOR, 3 NIT**
