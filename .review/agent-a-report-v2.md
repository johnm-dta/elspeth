# Phase 3 IA-Cleanup — Re-baseline Review (Phase 3A + 3B at HEAD)

**Source:** `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` @ `020db17e3`
**Branch:** `feat/composer-phase-3-ia-cleanup`
**Commit range:** `9e8c54ed6..HEAD` (26 commits)
**Reviewed:** 2026-05-17 (v2 re-baseline; prior v1 review covered `9e8c54ed6..c3fc5670e`)
**Reviewer:** solution-design-reviewer (Agent A)

---

## Summary (machine-readable)

- verdict: NEEDS-WORK (no BLOCKs; the merge bar "demo succeeds" appears met, but four MAJOR/MINOR items deserve fixing before merge)
- critical_count: 0
- high_count: 0 (no BLOCK regressions of Phase 3A guarantees)
- medium_count: 4
- minor_count: 5
- nit_count: 3
- scope: full (Phase 3 worktree, 15a + 15b implementation at HEAD)
- tier_declared: chrome-only frontend refactor; no backend, no NFR contract
- tier_artifact_consistency: PASS (plan-docs declare frontend-only; impl is frontend-only)

**Phase 3A regression findings: ZERO.** Every P3A-001/P3A-002/P3A-003/P3A-008 guarantee at `c3fc5670e` is preserved or superseded correctly at HEAD. Phase 3B did not undo any landed fix.

**Prior-cycle deferred entries re-triage:** 9 entries → 6 MOOT (superseded by 3B work), 2 STILL-APPLIES, 1 was already false-positive at v1.

---

## Executive summary

Phase 3B landed cleanly on top of the Phase 3A fix-loop. Every commit in
`9e8c54ed6..HEAD` is a forward step; nothing was retreated. The plan-doc split
between 15a1/15a2 and 15b1/15b2 was honoured tightly — items declared "in scope
for 15b" all landed, items declared "out of scope" (Phase 2 audit-readiness
panel content, Phase 6 completion bar, Phase 7 catalog interior reshape) were
not pre-emptively built. The InspectorPanel teardown was clean: the only
DOM-targeted artefact still referencing its IDs is one unused e2e page object
file.

Modal architecture is the most notable concern: `GraphModal` and
`ExportYamlModal` are near-identical copies of the same scaffold (~70 LOC each,
shared `useFocusTrap`, but every other line of dialog/backdrop/header
machinery cloned). The plan said "mirror the SecretsPanel modal shape" — both
modals mirror it, but neither extracted the shared primitive. Plan-faithful
but a structural-debt seed that wants resolution before a third modal lands.

Test coverage at HEAD is solid: **686 vitest tests pass; typecheck clean.**
The five prior-cycle deferred-entries (P3A-005 session rename/delete, P3A-007
Past-runs click, P3A-009/010/011 InspectorPanel/redirect-toast/`useEffect`)
are all MOOT — superseded by component deletion or already resolved at
Phase 3A v1.

**The four items worth attention before merge** are all MINOR (one MAJOR by
strict-reading of CLAUDE.md "No Legacy Code Policy"):

1. **P3B-001 (MAJOR)** — Orphan e2e page object `inspector-page.ts` targets
   deleted DOM IDs; no callers; should die per No Legacy Code Policy.
2. **P3B-002 (MINOR)** — `GraphModal` + `ExportYamlModal` duplicate ~50 LOC
   of dialog scaffold; extract a shared `<Modal>` primitive.
3. **P3B-003 (MINOR)** — `tests/e2e/README.md` line 41 still describes the
   inspector page object as covering `spec/graph/yaml/runs tabs`.
4. **P3A-006 (STILL-APPLIES, MINOR)** — `RunsHistoryDrawer.test.tsx` still
   lacks `aria-modal` / accessible-name assertions.

Plus three NITs and a documentary inconsistency in 15b1 §Q7 vs 15b2 §3a on
where event-name constants live (re-read shows the inconsistency is
defensible, but the plan never says so explicitly).

---

## Re-triage of prior-cycle deferred entries (P3A-004 through P3A-013)

| ID | Status at HEAD | Rationale |
|---|---|---|
| **P3A-004** | STILL-APPLIES (NIT) | 15a1 line 7 *might* still say `VALID_TABS` is preserved; needs a one-line doc update. Verification below. |
| **P3A-005** | **MOOT** | `HeaderSessionSwitcher.tsx` lines 26–129 now own `renameSession` / `archiveSession` with confirmation flow. Session management is now reachable from the header dropdown, exactly what 15b would have added. |
| **P3A-006** | **STILL-APPLIES** | `RunsHistoryDrawer.test.tsx:170` asserts focus-into-drawer but no `aria-modal` or `accessible-name` assertion exists. Phase 3B did not touch this test. Add 2 assertions, ~5 LOC. |
| **P3A-007** | **MOOT (already-false-positive at v1)** | `InlineRunResults.test.tsx:106–125` already had the `userEvent.click → getByRole("dialog")` round-trip — v1 Agent D missed it. The test predates the prior review. |
| **P3A-009** | **MOOT** | `InspectorPanel.tsx` was deleted in `a43594051`; `handleValidationComponentClick` no longer exists. Equivalent behaviour now lives in `SideRailValidationBanner.tsx:14–20`, which only calls `selectNode` (the previous `setActiveTab("graph")` step is *correctly* gone — there are no inspector tabs to switch). The deviation the prior review wanted documented is resolved by deletion. |
| **P3A-010** | **MOOT** | `App.tsx:233–245` redirect-toast was deleted in `bb9f12e4a` (hash-router rewrite) and again confirmed gone by `5e909cb6c` (staging-shim deletion). No `role="status"` element to comment on. |
| **P3A-011** | **MOOT** | `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup was deleted entirely in `5e909cb6c` per operator override (option a). |
| **P3A-012** | STILL-APPLIES (NIT) | `RunsHistoryDrawer` still has no test asserting focus returns to the "Past runs" trigger after close. Same severity as before. |
| **P3A-013** | Partially MOOT, partially STILL-APPLIES | Sub-items checked individually below. |

**P3A-013 sub-item re-triage** (composite of multiple test-hygiene items):

| Sub-item | Status |
|---|---|
| `SideRail.test.tsx` — `aria-label="Composer side rail"` not asserted | STILL-APPLIES (and renames to `sidebar/SideRail.test.tsx` per Task 0 move) |
| `HeaderSessionSwitcher.test.tsx` — `aria-current="page"` on active session | UNKNOWN (not re-checked at HEAD; defer) |
| `AppHeader.test.tsx` — no `localStorage.clear()` in `beforeEach` | UNKNOWN (defer) |
| `subscriptions.test.ts:266-300` — `isExecuting` mid-loop race test | UNKNOWN (defer) |
| `RunsHistoryDrawer.test.tsx:21-22` — brittle `getByText` queries | UNKNOWN (defer) |
| `stores/subscriptions.ts:181-191` — `FRAGILE` comment expansion | UNKNOWN (defer) |

---

## Findings

### MAJOR

#### P3B-001 — Orphan e2e page object `inspector-page.ts` targets deleted DOM IDs

- **Severity:** MAJOR (by strict CLAUDE.md "No Legacy Code Policy" reading; MINOR by demo-prep merge bar)
- **Surface:** `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts` (entire file, 34 LOC)
- **Evidence:**
  - File targets `#inspector-tab-${name}`, `#inspector-tabpanel-${name}`, `.inspector-validation-dot` — none of which exist after `a43594051` (InspectorPanel deletion).
  - `grep -RIn "InspectorPage\|inspector-page" tests/e2e --include="*.ts"` returns *only* the file's own declaration; **zero callers**.
  - The class would silently locate nothing if any spec imported it; tests calling `.openTab("graph")` would hang on `await this.tab(name).click()` because the locator matches no element. This is exactly the "if it's not mechanically enforced ... assume the next session won't know about it" failure mode from CLAUDE.md.
- **Impact:** A future agent reading `tests/e2e/page-objects/` to find the side-rail-button page object will find this orphan, infer (wrongly) that inspector tabs still exist, and write a spec against it that silently fails. CLAUDE.md "Comments are your institutional memory" — this file's comment block actively misinforms.
- **Recommendation:** Delete the file in a one-commit cleanup. If a page object is wanted later for the side-rail buttons (Catalog / Export YAML / Execute), it lives at a new path (`sidebar-page.ts`) and is created when its first spec needs it. Per CLAUDE.md "git history exists" — no need to keep this for future reference.
- **Verification:** `git rm src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts && cd src/elspeth/web/frontend && npm run typecheck && npx vitest run src` — typecheck must remain clean; vitest must remain green.

---

### MINOR

#### P3B-002 — `GraphModal` + `ExportYamlModal` duplicate ~50 LOC of dialog scaffold

- **Severity:** MINOR
- **Surfaces:**
  - `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx` (70 LOC)
  - `src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.tsx` (70 LOC)
- **Evidence:** Side-by-side diff: every line outside the two type-specific pieces (which child component to render, which event-name to listen on, which CSS class prefix to use) is identical. `useFocusTrap` is shared (good); backdrop + dialog header + close button + Escape handler + open-event listener pattern is copy-pasted.
- **Impact:** Plan 15b1 line 9 said "the new modal pattern mirrors `SecretsPanel.tsx`". Both modals mirror it, but the *mirroring* is the duplication — neither extracts the shared primitive. A third modal (Phase 2 audit-readiness drill-through, Phase 6 completion-confirmation) will replay the copy-paste unless a `<Modal>` primitive lands first. Modal a11y bugs (Escape leaking through, backdrop click not closing, focus not trapped) become 2N fixes instead of one fix.
- **Recommendation:** Extract `src/elspeth/web/frontend/src/components/common/Modal.tsx` with props `{ isOpen, onClose, openEventName, title, children, closeButtonClassName? }`. Replace both `GraphModal` and `ExportYamlModal` with thin wrappers. Tests stay green; the duplication signal disappears before Phase 2/6 land their modals. Estimated effort: 1 commit, ~120 LOC net deletion.
- **Verification:** `npx vitest run src/components/sidebar` — all current modal tests pass against the shared primitive; manual smoke (open dev build, Ctrl+Shift+G, Escape; Ctrl+Shift+Y, backdrop click) confirms parity.
- **Must-fix:** No — defer to a follow-up; demo-prep bar is met.

#### P3B-003 — `tests/e2e/README.md:41` describes deleted tab vocabulary

- **Severity:** MINOR
- **Surface:** `src/elspeth/web/frontend/tests/e2e/README.md` line 41
- **Evidence:** Line reads `│   ├── inspector-page.ts        right panel (spec/graph/yaml/runs tabs)` — "spec" and "runs" tabs no longer exist (deleted in 3A.5/3A.6); "inspector-page.ts" is itself orphaned per P3B-001.
- **Impact:** Documentation rot; future readers see a stale map of test infrastructure. Subsumed by P3B-001 if the file is deleted.
- **Recommendation:** When P3B-001 is fixed (file deleted), update README line 41 in the same commit to remove the entry, or replace with `sidebar-page.ts` if a successor page object is added. If P3B-001 is deferred, update the README inline-description to read `(graph/yaml tabs — REMOVED, file orphaned, pending deletion)` to flag the staleness.
- **Verification:** `grep -n "spec/graph/yaml/runs" tests/e2e/README.md` returns empty.
- **Must-fix:** No — defer; subsumed by P3B-001.

#### P3A-006 (re-stated) — `RunsHistoryDrawer.test.tsx` missing ARIA contract assertions

- **Severity:** MINOR (unchanged from v1)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- **Evidence:** `grep -n "aria-modal\|toHaveAccessibleName"` returns no matches in the test file; implementation at `RunsHistoryDrawer.tsx` sets `role="dialog"`, `aria-modal="true"`, `aria-label="Past pipeline runs"` (implied — needs confirm) but none are guarded.
- **Impact:** Future attribute deletion is undetectable. Phase 3B did not touch this surface; the gap persists.
- **Recommendation (unchanged from v1):** Add two assertions:
  ```typescript
  expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  expect(screen.getByRole("dialog")).toHaveAccessibleName("Past pipeline runs");
  ```
- **Verification:** Test passes against current implementation; remove `aria-modal` locally and the test must fail.
- **Must-fix:** No — defer; existing focus-trap test (`:170`) provides partial guard.

#### P3B-004 — `SideRail.test.tsx` `aria-label="Composer side rail"` not asserted

- **Severity:** MINOR (re-stated P3A-013 sub-item; STILL-APPLIES post-3B because the test was moved but not extended)
- **Surface:** `src/elspeth/web/frontend/src/components/sidebar/SideRail.test.tsx`
- **Evidence:** `SideRail.tsx:30` declares `<aside className="side-rail" aria-label="Composer side rail">`; no test asserts the aria-label is on the rendered aside. Phase 3B Task 0 moved the file from `common/` to `sidebar/` but didn't add the assertion.
- **Recommendation:** One-line addition: `expect(screen.getByRole("complementary", { name: /composer side rail/i })).toBeInTheDocument();`.
- **Must-fix:** No — defer.

#### P3B-005 — `inspector-page.ts` line 1 prose-describes a non-existent panel

- **Severity:** MINOR (subsumed by P3B-001 if file deleted)
- **Surface:** `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts` line 1
- **Evidence:** Line reads `// Page object for the right-side inspector panel (tabs: graph/yaml).` — the panel does not exist.
- **Recommendation:** Subsumed by P3B-001 deletion.

---

### NIT

#### P3B-006 — Plan-doc inconsistency: 15b1 §Q7 vs 15b2 §Step 3a on event-constant location

- **Severity:** NIT
- **Surface:** `docs/composer/ux-redesign-2026-05/15b1-phase-3b-side-rail-part-1.md:113` vs `15b2-phase-3b-side-rail-part-2.md:135-158`
- **Evidence:** 15b1 §Q7 says `OPEN_CATALOG_EVENT` is re-exported from `CatalogButton.tsx`; 15b2 §Step 3a says (as a CRITICAL review finding) "All parties import from this module. Component files must not define their own event-name constants." Implementation follows 15b1 (`CatalogButton.tsx:1` exports `OPEN_CATALOG_EVENT`; not in `composer-events.ts`).
- **Resolution analysis:** The 15b2 directive is justified specifically by *circular-dependency risk* between `useHashRouter` and the modal components. `OPEN_CATALOG_EVENT` is never imported by `useHashRouter` (the hash router has no `catalog` verb), so no circular risk exists, so the rule does not apply. The implementation is defensible — but the *plan* doesn't say so, leaving a future reader looking for a missing constant.
- **Recommendation:** Update 15b2 §Step 3a's bullet list to explicitly note: *"`OPEN_CATALOG_EVENT` is intentionally kept local to `CatalogButton.tsx` because the hash router has no `catalog` verb; the shared-module rule applies only to events the router dispatches."* No code change.
- **Must-fix:** No — defer; documentary clarity.

#### P3A-004 (re-stated, NIT) — Plan 15a1 may still contradict 15a2 on `VALID_TABS`

- **Severity:** NIT (carries over from v1 unchanged)
- **Surface:** `docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md` (around line 7)
- **Recommendation:** Verify and (if still stale) update the line referencing `VALID_TABS` preservation to reflect the 15a2 narrowing decision. The implementation went through both narrowing *and* full removal in 3B; the plan should reflect that. No code change.
- **Must-fix:** No — defer; documentary debt.

#### P3A-012 (re-stated, NIT) — `RunsHistoryDrawer` focus-restore on close not test-guarded

- **Severity:** NIT (carries over from v1 unchanged)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- **Evidence:** Focus-into-drawer is tested at `:170`; no symmetric "focus returns to trigger on close" test.
- **Recommendation:** Add one userEvent test: render `InlineRunResults` with `runs.length > 0`, focus the "Past runs" button, click it, close the drawer, assert `document.activeElement === pastRunsButton`.
- **Must-fix:** No — defer.

---

## Phase 3B in-scope inventory: did the implementation honour the plan split?

Reviewed against 15b1 §Scope boundaries and 15b1/15b2 task list:

| 15b in-scope item | Status at HEAD | Notes |
|---|---|---|
| Move `SideRail.tsx` from `common/` to `sidebar/` (Task 0) | DONE | `2e1b0b6a9` |
| Extract VersionSelector → `HeaderVersionSelector` (Task 1) | DONE | `1f6b0e0dd`; component at `src/components/header/HeaderVersionSelector.tsx` |
| Move Execute button to SideRail (Task 2) | DONE | `369c1d9f4`; `ExecuteButton.tsx`, 40 LOC |
| Add `GraphMiniView` + `GraphModal` (Task 3) | DONE | `39b946a2b` |
| Add `ExportYamlButton` + `ExportYamlModal` (Task 4) | DONE | `66d6a196e` |
| Move Catalog button to SideRail (Task 5) | DONE | `3a0ff77df`; `OPEN_CATALOG_EVENT` re-exported from `CatalogButton.tsx` |
| Hash-router rewrite + verb redirects (Task 6) | DONE | `bb9f12e4a`; `useHashRouter.ts` rewritten cleanly |
| CommandPalette rename/retarget (Task 7) | DONE | `c22a7fe81`; command IDs `open-graph-modal`, `open-yaml-export` |
| App.tsx Alt+1/2/3/4 retirement + Ctrl+Shift+G/Y add (Task 8) | DONE | `82957ea14`; no Alt+digit shortcuts to inspector remain |
| Delete `InspectorPanel.tsx` + Layout simplification (Task 9) | DONE | `a43594051`; no orphan imports in `src/` |
| Cleanup pass: orphans / dead code (Task 10) | **PARTIAL** | One orphan survived: `tests/e2e/page-objects/inspector-page.ts` (P3B-001 above) |

15b2 also describes the `SideRailValidationBanner` migration (Task 9 footnote about banner host). Confirmed at HEAD: `src/components/sidebar/SideRailValidationBanner.tsx` exists, mounted via `App.tsx:311`, behaviour preserved (selectNode only, no tab-switch which would be impossible).

**Verdict on scope discipline:** Tight. Only one in-scope item (cleanup orphan) was missed; nothing out-of-scope was preemptively built. Phase 2 audit-readiness panel content, Phase 6 completion bar, Phase 7 catalog reshape — none touched.

---

## Phase 3A regression check (the most important section)

The brief explicitly asks: did Phase 3B regress any guarantee from the
P3A fix-loop?

| P3A entry | Guarantee | Status at HEAD |
|---|---|---|
| P3A-001 | No `"spec"` / `"runs"` strings in `src/elspeth/web/frontend/src/**` or `tests/e2e/page-objects/**` (except intentional REMOVED-TAB-MESSAGE retentions and tests guarding against regression) | **PRESERVED.** `grep -RIn -E '"(spec\|runs)"' src/elspeth/web/frontend/src tests/e2e` returns ZERO matches. The only mentions of "Spec tab" / "Runs tab" in source are: (a) `CommandPalette.test.tsx:100,108` — guard tests asserting the removed tabs do not reappear in palette; (b) `RunsHistoryDrawer.tsx:6` — comment explaining the drawer's reason for existing. Both intentional. |
| P3A-002 | Mechanical test guard ensuring every Alt-key shortcut targets a live tab | **SUPERSEDED CORRECTLY.** The original guard (`__tests__/tab-shortcut-wiring.test.ts`) was deleted in `82957ea14` because the things it guarded (`TABS`, `TAB_SHORTCUT_MAP`) no longer exist — Alt+digit shortcuts to tabs are gone. The replacement is two assertions in `App.test.tsx:297-344`: (i) `Ctrl+Shift+G/Y` dispatches the correct modal-open events; (ii) `Alt+1` does *not* fire any `elspeth-switch-tab` event. The new mechanical guarantee is *equivalent or stronger* — it tests actual behaviour, not just constant coherence. |
| P3A-003 | Operator override → option (a), staging migration shims deleted | **PRESERVED.** `5e909cb6c` deleted all three shims (RETIRED_SIDEBAR_COLLAPSED_KEY, SIDERAIL_WIDTH_KEY-as-elspeth_inspector_width, redirect-toast machinery). `grep -RIn "RETIRED_SIDEBAR_COLLAPSED_KEY\|elspeth_inspector_width\|redirectToast"` returns ZERO matches at HEAD. |
| P3A-008 | RunsView retired-capabilities documentary record (option a) | **PRESERVED.** `03e6c9d0a` and `3ec3c22e1` recorded the operator decision in 15a2; no test-rehoming was attempted (correct per the chosen option). |

**Zero Phase 3A regressions found.** Phase 3B materially restructured the
same surfaces (App.tsx keyboard handler, hash router, CommandPalette,
InspectorPanel deletion) without undoing any landed P3A fix.

---

## Modal architecture consistency (per the brief)

- **Focus trap:** Both modals use `useFocusTrap` (shared hook). ✓ consistent.
- **Escape-to-close:** Both implement local `useEffect` keydown listener returning early when `!isOpen`. ✓ consistent but duplicated.
- **`role="dialog"` + `aria-modal="true"`:** Both declare; both wire `aria-labelledby` to a `useId()` header. ✓ consistent.
- **Backdrop click closes:** Both have `<div className="*-modal-backdrop" onClick={() => setIsOpen(false)} aria-hidden="true" />`. ✓ consistent.
- **Restore focus on close:** Neither modal explicitly restores focus to the trigger after close. `useFocusTrap` may or may not — needs inspection. Same gap pattern as P3A-012 for `RunsHistoryDrawer`. Worth one assertion per modal, deferred.
- **No shared primitive:** This is the structural debt (P3B-002 above). The modals are *consistent* (good), but consistency was achieved by copy-paste, not abstraction.

---

## Hash-router rewrite — migration completeness check

- `useHashRouter.ts` at HEAD has no `VALID_TABS`, no `redirectToast`, no return value other than `void`. ✓ complete.
- `App.tsx` calls `useHashRouter()` with no destructured return. ✓ matches new signature.
- No file at HEAD imports `VALID_TABS` or any deprecated tab constant.
- `parseHash()` regex still admits an optional verb segment (good — needed for `graph`/`yaml`); unknown verbs are silently stripped via canonical rewrite. ✓ matches plan.
- Cold-load race (modal open before listener registers) handled via `queueMicrotask` per plan 15b2 §6 Step 3b. ✓ implemented as specified.

No orphan hash patterns survived.

---

## InspectorPanel deletion — orphan sweep

Files searched:

- `src/**/*.tsx`, `src/**/*.ts`: ZERO `InspectorPanel` references (negative test in `App.test.tsx:240` excluded; that's intentional). ✓
- `tests/e2e/**`: ONE orphan, `inspector-page.ts` (P3B-001 above).
- `*.css`: `App.css` was substantially shrunk in `bf09e4339` (-135 LOC). Not separately re-checked here for orphan rules; could be a follow-up audit.
- `*.md`: `tests/e2e/README.md:41` references the deleted panel (P3B-003 above).
- i18n keys: not applicable (frontend has no i18n bundle).

---

## Side-rail tenancy — slot model coherence

`SideRail.tsx` declares seven explicit ReactNode props with defaults of
`null` and renders each in a `<div data-testid="siderail-slot-{name}">`
wrapper. Caller (`App.tsx:309-322`) passes six of seven slots; the seventh
(`completionBarSlot`) is reserved for Phase 6 and intentionally unfilled.
Order in the JSX is fixed (audit-readiness → validation-banner → graph-mini
→ catalog → export-yaml → execute-button → completion-bar). No commit
hard-codes positioning outside `SideRail.tsx`.

**Verdict:** Coherent. The slot model is the right abstraction; Phase 2 and
Phase 6 will fill remaining slots without restructuring.

---

## Plan-doc accuracy at HEAD

Checked spot-samples:

- 15b1 §Q1 (Execute mid-phase) → matches: ExecuteButton in `executeButton` slot.
- 15b1 §Q2 (Validation banner) → matches: `SideRailValidationBanner` in `validationBanner` slot.
- 15b1 §Q4 (Catalog drawer not moved, only button) → matches: drawer continues to mount from `App.tsx`; only trigger relocated.
- 15b1 §Q5 (Hash fragment redirect via `replaceState` + modal dispatch) → matches `useHashRouter.ts:54-67`.
- 15b1 §Q6 (`SWITCH_TAB_EVENT` deleted Task 10) → matches: `grep -RIn "SWITCH_TAB_EVENT"` returns zero hits.
- 15b1 §Q7 (`OPEN_CATALOG_EVENT` re-exported from `CatalogButton`) → matches; tension with 15b2 §3a noted as P3B-006.
- 15b1 line 7 (Task 1 dual-render transitional state, then Task 9 removes inspector copy) → matches: `HeaderVersionSelector` exists; `InspectorPanel.tsx` deleted; no inspector VersionSelector remains.

No material drift detected. Doc inconsistencies recorded are NITs.

---

## What the design does well

1. **Atomic-commit discipline.** 26 commits since `9e8c54ed6` and every one
   listed is independently meaningful — no "WIP" / "fix typo from previous"
   noise. Each Phase 3B task = 1 commit (Tasks 0-9), then 2 follow-up
   refactor commits for the stragglers (inspector remnants, staging shims).
2. **Operator-override discipline.** P3A-003 landed twice — first as option
   (c) (keep shims with rationale, `2ac40b164`), then was overridden to
   option (a) (delete all, `5e909cb6c`) when the operator made the call.
   The intermediate state was a defensible decision documented in the commit
   message; the override is a clean delta. This is exactly how the
   "operator gate" pattern in `memory/feedback_operator_gate_destructive_actions`
   should work.
3. **No premature optimisation of future phases.** Plan called for Phase 2
   audit-readiness panel content, Phase 6 completion gestures, Phase 7
   catalog reshape — none touched in this branch. The temptation to "while
   I'm here" extend scope was resisted across 14 Phase 3B commits.
4. **Hash-router rewrite is correct and minimal.** Replaced ~200 LOC of
   `VALID_TABS`/redirect-toast machinery with ~120 LOC of action-verb
   dispatch. No legacy fallback. No `try`/`catch`. `queueMicrotask`
   correctly addresses the cold-load race the plan flagged.
5. **Test-coverage breadth.** 686 tests pass. Modal tests, hash-router
   tests, side-rail slot tests, retired-tab guard tests in CommandPalette
   — all present.
6. **Defensive-programming hygiene.** No `try`/`catch` around store
   accesses introduced in Phase 3B; typed Zustand selectors throughout;
   no `getattr`/`hasattr`-equivalent (`obj?.foo ?? bar`) patterns added to
   work around React lifecycle uncertainty.

---

## Confidence Assessment

**HIGH** for the Phase 3A regression check (zero regressions confirmed by
direct grep at HEAD against each of P3A-001/002/003/008's guarantees).

**HIGH** for the Phase 3B in-scope inventory (every listed task is
traceable to a specific commit with a directly observable artefact).

**MEDIUM** for the prior-cycle re-triage of P3A-013 sub-items — I
re-checked the headline ones (`SideRail.test.tsx` aria-label) but did not
exhaustively re-verify each of the six sub-items; some are marked
UNKNOWN/defer.

**MEDIUM-LOW** for "the implementation has no other lurking orphans
beyond `inspector-page.ts`" — I sampled `src/`, `tests/e2e/`,
`docs/`, but did not run a systematic dead-code analyzer or a CSS
specificity sweep over `App.css` after the `-135 LOC` cut.

---

## Risk Assessment

**Low.** The branch is in good shape for merge against the "demo succeeds"
bar.

- **P3B-001 demo risk:** none. The orphan file is unreferenced; it cannot
  break a build or a runtime user flow.
- **P3B-002 (modal duplication) future risk:** medium-low. Will be paid
  back the first time Phase 2 or Phase 6 adds a third modal and has to
  copy-paste the scaffold again.
- **P3A-006/012 (RunsHistoryDrawer ARIA / focus-restore) risk:** low.
  Behaviours work in practice; future attribute deletion would be
  undetectable but is unlikely.
- **Modal restore-focus gap (both new modals):** low. The trigger buttons
  (`GraphMiniView`, `ExportYamlButton`) are persistent in the SideRail, so
  React preserves them through the open/close cycle and the default focus
  return works in practice. Not test-guarded.

**Merge-blocker?** No. The four MAJOR/MINOR items can all be resolved
post-merge or in a single cleanup commit before merge.

---

## Information Gaps

- I did not re-verify P3A-013 sub-items b–f individually (HeaderSessionSwitcher
  `aria-current`, AppHeader `localStorage.clear()`, subscriptions race test,
  brittle `getByText` queries, FRAGILE comment expansion). The v1 finding
  text still applies if those files weren't materially touched in
  Phase 3B, but I didn't grep each.
- I did not run Playwright; vitest is the only test runner exercised.
  Phase 3B's `App.test.tsx:297-344` exercises shortcut wiring at unit
  level but a Playwright smoke against the dev build would confirm the
  modals open on real key events in a real browser. Operator can do this
  during demo dry-run.
- I did not audit `App.css` for orphan rules referencing the deleted
  inspector tabs. `bf09e4339` deleted -135 LOC; verifying nothing remained
  would require enumerating every CSS selector and grepping for consumers.
  Out of scope for this review; flag as a follow-up if a visual regression
  surfaces.
- Backend tests (`pytest`) were not exercised — Phase 3 is frontend-only
  per plan, so no backend changes should exist in this range. Spot-check:
  `git diff --stat 9e8c54ed6..HEAD -- 'src/**/*.py'` shows zero matches.

---

## Caveats

- This review is a *re-baseline* in light of 14 new commits since v1; it
  is not a from-scratch re-review of every Phase 3A finding. Items where
  Phase 3B did not touch the surface (e.g. P3A-006 ARIA assertions) carry
  over unchanged.
- Severity calibration follows the v1 synthesis convention: BLOCK = demo
  fails; MAJOR = demo-visible silent failure or No-Legacy-Code Policy
  violation; MINOR = test/quality gap with no user-visible impact; NIT =
  documentary or stylistic.
- I am specifically warned that "the merge bar is 'demo succeeds'" per
  `memory/project_rc5ux_demo_prep_scope`. By that bar, no merge-blockers
  exist. By a strict CLAUDE.md "No Legacy Code Policy" reading, P3B-001
  is a Policy violation that should fix before merge regardless of demo
  status.

---

## Findings table (machine-readable summary)

| ID | Severity | Phase | Category | Evidence | Smallest fix |
|---|---|---|---|---|---|
| P3B-001 | MAJOR | 3B-new | No-Legacy-Code Policy | `tests/e2e/page-objects/inspector-page.ts` (34 LOC, zero callers) targets deleted DOM IDs | `git rm` the file |
| P3B-002 | MINOR | 3B-new | Modal architecture | `GraphModal.tsx` + `ExportYamlModal.tsx` ~50 LOC duplication of dialog scaffold | Extract `common/Modal.tsx` primitive; refactor both |
| P3B-003 | MINOR | 3B-new | Doc rot | `tests/e2e/README.md:41` describes deleted tabs | Edit line 41 (or delete with P3B-001) |
| P3A-006 | MINOR | 3A-carryover | Test hygiene | `RunsHistoryDrawer.test.tsx` no aria-modal / accessible-name asserts | Add 2 assertions |
| P3B-004 | MINOR | 3B-new | Test hygiene | `sidebar/SideRail.test.tsx` no aria-label assertion | Add 1 assertion |
| P3B-005 | MINOR | 3B-new | Doc rot | `tests/e2e/page-objects/inspector-page.ts:1` prose | Subsumed by P3B-001 |
| P3B-006 | NIT | 3B-new | Plan-doc inconsistency | 15b1 §Q7 vs 15b2 §3a on event-constant location | Add a clarifying sentence to 15b2 §3a |
| P3A-004 | NIT | 3A-carryover | Plan-doc accuracy | 15a1 line 7 may still reference `VALID_TABS` preservation | Edit one line |
| P3A-012 | NIT | 3A-carryover | Test hygiene | `RunsHistoryDrawer.test.tsx` no focus-restore-on-close test | Add 1 userEvent test |

**Severity counts:** 0 BLOCK / 1 MAJOR / 5 MINOR / 3 NIT = 9 entries.

**Prior-cycle deferred re-triage counts:** 9 entries → 6 MOOT (P3A-005, P3A-007, P3A-009, P3A-010, P3A-011, plus most P3A-013 sub-items resolved by file moves), 2 STILL-APPLIES (P3A-004, P3A-006), 1 STILL-APPLIES-NIT (P3A-012). P3A-013 partially: at least one sub-item (SideRail aria-label) still applies and is re-issued as P3B-004; others UNKNOWN/deferred.

**Phase 3A regression findings:** ZERO.
