# Phase 3 IA-Cleanup — Review Synthesis v2 (Phase 3A + 3B Re-baseline)

**Branch:** `feat/composer-phase-3-ia-cleanup`
**Commit range:** `9e8c54ed6..020db17e3` (26 commits — 7 Phase 3A, 14 Phase 3B, 5 fix-loop)
**HEAD:** `020db17e3551514bf865e83060f43d803acb921d`
**Synthesised:** 2026-05-17 (v2 re-baseline; prior v1 covered Phase 3A only at `c3fc5670e`)
**Sources:** Agent A (solution architect v2), Agent B (systems thinker v2), Agent C (code quality v2), Agent D (quality/test v2)

---

## Verdict

**APPROVED_WITH_CHANGES_REQUESTED.** No agent flagged a BLOCK. All four reviewers converged on **zero Phase 3A regressions of landed behaviour** — every P3A-001/002/003/008 guarantee at `c3fc5670e` is preserved at HEAD. Phase 3B landed cleanly atop the fix-loop.

The single material carryover from Phase 3A is a **regression-of-omission**: the orphan E2E page object `inspector-page.ts` (flagged by Agents A, B, and C). The fix-loop's P3A-001 narrowing edit touched the file but did not delete it; Phase 3B's `InspectorPanel.tsx` deletion left it consumer-less. By strict CLAUDE.md "No Legacy Code Policy" reading this is a Phase 3A regression to fix in-branch; by the demo-prep merge bar it is harmless.

Merge bar remains **"demo succeeds"** per memory `project_rc5ux_demo_prep_scope`. Demo path is unaffected by every finding below.

**Entry counts:** 0 BLOCK / 2 MAJOR / 7 MINOR / 5 NIT = **14 entries**

**Must-fix before merge:**
- **P3A-014** (orphan `inspector-page.ts` + companion README:41 staleness) — single cleanup commit; CLAUDE.md No-Legacy-Code policy violation; resolved with one `git rm` + one README line edit

**Defer as observation:** all other 13 entries

**Prior-cycle entries resolved:** 9 of 9 deferred entries triaged → 5 MOOT (resolved by Phase 3B work or already-false-positive at v1), 4 STILL-APPLIES (carried forward as P3A-006, P3A-012, P3A-013-sub, P3A-004), plus 4 of the originally must-fix entries (P3A-001, P3A-002, P3A-003, P3A-008) resolved out of band — see "Resolved without action" section.

---

## Reviewer-Conflict Resolutions

### Conflict 1: P3A-007 — `InlineRunResults` "Past runs" click test coverage

| Agent | Framing |
|---|---|
| A | MOOT — `InlineRunResults.test.tsx:106-125` already tests `userEvent.click → getByRole("dialog")` round-trip; v1 Agent D missed it |
| D | STILL-APPLIES — "Button presence confirmed; click-to-open-drawer still not tested" |

**Resolution: MOOT (Agent A correct).** Direct grep confirms `InlineRunResults.test.tsx:106` contains `it("opens and closes the Past runs drawer", async () => {…})`. Agent D's v2 re-triage replicated v1's false-positive without re-reading the file. The test exists and exercises click → dialog appears → close → dialog dismissed.

### Conflict 2: Dead CSS in App.css

| Agent | Framing |
|---|---|
| B | MINOR/NIT — `.spec-pending-proposal` / `.runs-pending-proposal` survive at App.css:2927-2944 with zero TSX consumers |
| C | "CSS orphans from deleted components: None. `git grep -nE '\.inspector-(tab\|tabpanel\|validation-dot\|panel)' App.css` is empty" |

**Resolution: Both right; not a conflict.** Agent C checked `inspector-*` prefix classes (clean — `InspectorPanel.tsx` deletion did sweep its own CSS). Agent B checked `spec-pending-proposal` / `runs-pending-proposal` classes, which originated with the deleted `SpecView.tsx` / `RunsView.tsx` and were not swept when those components were deleted (3A.5/3A.6 commits). Verified at lines 2927, 2929, 2943, 2944. Recorded as P3B-002.

### Conflict 3: Severity of orphan `inspector-page.ts` (Phase 3A regression or new Phase 3B entry?)

| Agent | Framing |
|---|---|
| A | MAJOR by strict No-Legacy-Code reading; MINOR by demo bar. Phase-3B-discovered. |
| B | MINOR — Tragedy of the Commons; framed as Phase 3B sweep miss |
| C | MAJOR Phase-3A regression-of-omission (the fix-loop's P3A-001 narrowing edit at `4a1dcc367` touched the file's type union but didn't notice the file would become structurally dead two commits later) |

**Resolution: MAJOR, classified as Phase 3A regression-of-omission (Agent C framing).** The file's *type union* was narrowed in the P3A-001 fix-loop commit — that was a call-site change. Per CLAUDE.md "Change all call sites in the same commit," the fix-loop already owned this file; the question "should the whole file survive Phase 3B's deletion of InspectorPanel.tsx?" should have been answered then. Phase 3B's blamelessness is real (the file was already orphaned-in-waiting before 3B started). MAJOR severity reflects the No-Legacy-Code policy weight; not BLOCK because the file is unreferenced and cannot misbehave at runtime.

### Conflict 4: Modal duplication severity

| Agent | Framing |
|---|---|
| A | MINOR — extract `<Modal>` primitive |
| B | MINOR — Shifting the Burden; two copies is break-even point, risk grows at 3+ |
| C | MINOR — extract `components/common/Modal.tsx` |
| D | MINOR (B3B-005) — aria-modal not asserted; partially overlaps |

**Resolution: MINOR, extract `<Modal>` primitive (P3B-001).** All four agree on MINOR; all agree the plan-faithful copy-paste from `SecretsPanel.tsx` is defensible at 2 copies and dangerous at 3. The brief asks "shared `<Modal>` extraction or comment-only annotation?" — recommend **shared extraction** because (a) Phase 6 (completion bar) and Phase 5 (LLM interpretation) will plausibly add modal #3 within the next two phases, and (b) Agent D's `aria-modal` assertion gap is naturally absorbed by `Modal.test.tsx` covering the primitive's contract once instead of N times. Comment-only annotation (Agent C's "smaller patch") deferred as fallback if the operator wants this out of merge scope.

### Conflict 5: Phase 3A regression classification — overall

| Agent | Count |
|---|---|
| A | Zero Phase 3A regressions |
| B | Zero Phase 3A regressions |
| C | One (M1 = orphan inspector-page.ts, regression-of-omission) |
| D | Zero (B3B-001…B3B-007 all classified Phase-3B-new) |

**Resolution: Zero behavioural regressions, one structural regression-of-omission (Agent C correct).** The 4-way split looks like disagreement but reduces to a framing choice: A/B/D treat "Phase 3A regression" as "did Phase 3B undo a landed P3A fix?" (answer: no); C treats it as "is there work the P3A fix-loop should have done that is now visibly incomplete?" (answer: yes, M1). Both framings are valid; the synthesis adopts both — confirm zero behavioural regressions of landed work AND record M1 as a structural regression-of-omission per CLAUDE.md call-sites discipline.

---

## Phase 3A Regression Section

**Behavioural regressions of landed Phase 3A work: ZERO** (all four agents converged).

| P3A guarantee | Status at HEAD | Evidence |
|---|---|---|
| P3A-001 (no Spec/Runs literal residuals in src/) | PRESERVED | `grep -RIn '"(spec\|runs)"' src/elspeth/web/frontend/src tests/e2e/page-objects` returns only intentional REMOVED-TAB-MESSAGES retentions and guard tests |
| P3A-002 (mechanical shortcut-wiring guard) | SUPERSEDED CORRECTLY | Guard + guarded constants both deleted in `82957ea14`; replaced by `App.test.tsx:297-344` behavioural assertions (Ctrl+Shift+G/Y fire, Alt+1 does NOT fire) — *equivalent or stronger* coverage of the actual user-facing surface |
| P3A-003 (migration shims) | PRESERVED (option a override) | `5e909cb6c` deleted all three shims; `grep` for RETIRED_SIDEBAR_COLLAPSED_KEY / elspeth_inspector_width / redirectToast returns zero hits |
| P3A-008 (RunsView retirement) | PRESERVED | `03e6c9d0a` recorded operator decision; no test-rehoming attempted (correct per chosen option) |

**Structural regression-of-omission: ONE** (Agent C framing).

| Item | Source | Disposition |
|---|---|---|
| Orphan `tests/e2e/page-objects/inspector-page.ts` survives Phase 3B's InspectorPanel deletion | Agent A P3B-001, Agent B P3B-003-sub-3A, Agent C M1 | Tracked below as **P3A-014** (carryover ID) — must-fix; one-commit cleanup |

**Classification rationale:** the file was touched by the P3A-001 fix-loop (`4a1dcc367`) which narrowed its `InspectorTab` type union from `"spec"|"graph"|"yaml"|"runs"` to `"graph"|"yaml"`. That edit acknowledged the file as a call-site of the tab rename. Per CLAUDE.md "Change all call sites in the same commit," the larger question "does this file have a reason to exist after InspectorPanel.tsx goes away?" was already in scope at that moment and was missed. The grep-based regression-check style of the P3A-001 fix found symbols but not zero-consumer artefacts.

---

## Punch List (Severity-Ordered, Dependency-Ordered Within Tier)

### MAJOR

#### P3A-014 — Orphan E2E page object `inspector-page.ts` survives InspectorPanel deletion (regression-of-omission)

- **Severity:** MAJOR (Phase 3A regression by strict CLAUDE.md "No Legacy Code Policy"; MINOR by demo-prep merge bar)
- **Sources:** Agent A (P3B-001), Agent B (P3B-003 sub-3A), Agent C (M1)
- **Surfaces:**
  - `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts` (whole file, 34 LOC)
  - `src/elspeth/web/frontend/tests/e2e/README.md:41` (companion stale row)
- **What's wrong:**
  - File exports `InspectorPage` class targeting `#inspector-tab-${name}`, `#inspector-tabpanel-${name}`, `.inspector-validation-dot` selectors — none exist after `a43594051` deleted `InspectorPanel.tsx`
  - `grep -RIn "InspectorPage\|inspector-page" tests/e2e --include="*.ts"` returns ZERO callers (only the file's own declaration)
  - Line 16 comment still reads `// InspectorPanel.tsx applies role="tab"...` — referencing the deleted source
  - `tests/e2e/README.md:41` reads `│   ├── inspector-page.ts        right panel (spec/graph/yaml/runs tabs)` — all four tabs named are deleted
  - Per CLAUDE.md "if it's not mechanically enforced ... assume the next session won't know about it" — a future agent reading `tests/e2e/page-objects/` will believe this contract is live
- **Smallest remediation (single commit):**
  ```bash
  git rm src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts
  # Edit tests/e2e/README.md: delete line 41 entirely (the `inspector-page.ts` row)
  ```
- **Verification:** After deletion: `cd src/elspeth/web/frontend && npm run typecheck` exits 0; `npx vitest run` remains green at 686/686; `grep -RIn "inspector-page\|InspectorPage" tests/e2e` returns empty; `grep -n "spec/graph/yaml/runs" tests/e2e/README.md` returns empty.
- **Must-fix:** **YES** — single-commit cleanup; CLAUDE.md No-Legacy-Code policy violation; "regression-of-omission" framing means the fix-loop owes this.

#### P3B-001 — `GraphModal` + `ExportYamlModal` duplicate ~50 LOC of dialog scaffolding; no shared `<Modal>` primitive

- **Severity:** MAJOR (collapsed from 4 reviewers' MINORs; promoted because it absorbs Agent D's B3B-005 `aria-modal` assertion gap and is the structural pre-requisite for Phase 5/6 modal additions; risk is forward-looking, not current)
- **Sources:** Agent A (P3B-002), Agent B (P3B-004), Agent C (m2), Agent D (B3B-005)
- **Surfaces:**
  - `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx` (70 LOC)
  - `src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.tsx` (70 LOC)
- **What's wrong:** Both files implement identical `useState/useRef/useId` triplet + `useFocusTrap` call + `useEffect` event listener + Escape handler + backdrop + dialog JSX. Deltas: CSS prefix, event name, title, content component (`<GraphView />` vs `<YamlView />`). Plan 15b1 documented the pattern as "mirrors `SecretsPanel.tsx`" — both mirror it, neither extracts. Two copies is the inflection point. Phase 5 (LLM interpretation panel) and Phase 6 (completion gesture) both plausibly add modal #3 within next 2 phases.
- **Recommended remediation: shared `<Modal>` extraction (not comment-only annotation).** Extract `src/elspeth/web/frontend/src/components/common/Modal.tsx` with props `{ isOpen, onClose, openEventName, title, titleId, children, closeButtonClassName? }`. Replace both modals with thin wrappers. Add `Modal.test.tsx` covering the contract once (focus-trap, Escape, backdrop, aria-modal, role=dialog, aria-labelledby) — absorbs B3B-005's `aria-modal` assertion gap and the implicit focus-restore concern from B3B-001. Estimated effort: 1 commit, ~120 LOC net deletion + 1 new test file ~80 LOC.
- **Fallback (if operator wants this out of merge scope):** Comment-only annotation per Agent C's "smaller patch" — `MODAL_INVARIANTS.md` or a `// MODAL CONTRACT` comment block listing the four invariants (focus trap, Escape closes, backdrop closes, return-focus-to-trigger).
- **Why MAJOR not MINOR despite all 4 reviewers calling MINOR:** the merge bar is "demo succeeds" but the *scope* of Phase 3B has materially expanded modal surface area (`GraphModal` + `ExportYamlModal` introduced this branch, plus the pre-existing `SecretsPanel.tsx` shape they mirror). Per the brief: "Phase 3B has materially expanded scope (modal architecture, hash router rewrite, header history selector) so some of the prior cycle's deferral rationale may need re-evaluation." This is exactly that case — modal duplication is the structural-debt signal Phase 3B introduced; deferring past merge means Phase 5/6 inherits a 3-copy pattern instead of the chance to abstract at 2.
- **Verification:** `npx vitest run src/components/sidebar` and `src/components/common` — all modal tests green against the shared primitive; manual smoke: open dev build, Ctrl+Shift+G + Escape closes; Ctrl+Shift+Y + backdrop-click closes.
- **Must-fix:** Operator decision — strongly recommended in-branch; defer-as-observation is defensible only if Phase 5/6 are confirmed >2 cycles out.

---

### MINOR

#### P3B-002 — Dead CSS classes `.spec-pending-proposal` / `.runs-pending-proposal` survive in App.css

- **Severity:** MINOR
- **Sources:** Agent B (P3B-002)
- **Surface:** `src/elspeth/web/frontend/src/App.css:2927, 2929, 2943, 2944`
- **What's wrong:** Two class rules with zero TSX/TS consumers. Verified: `grep -rn 'spec-pending-proposal\|runs-pending-proposal' src/elspeth/web/frontend/src --include='*.tsx' --include='*.ts'` returns empty. Classes were introduced with the now-deleted `SpecView.tsx` / `RunsView.tsx`; deletion commits (3A.5 `66748edb9`, 3A.6 `8b04d53ce`) removed TSX but not their dedicated CSS rules. Note: `.yaml-pending-summary` on the same compound selector IS still used by `YamlView.tsx:142` — keep that selector.
- **Smallest remediation:** Delete the two class names from the compound selectors at App.css:2927-2944. If they appear on shared multi-class rules, narrow each rule to retain only `.yaml-pending-summary`.
- **Verification:** Dev build renders YamlView without visual regression; `grep -n 'spec-pending-proposal\|runs-pending-proposal' src/elspeth/web/frontend/src/App.css` returns empty.
- **Must-fix:** No — dead selectors have no runtime behaviour. Defer.

#### P3B-003 — `HeaderVersionSelector` synthesises placeholder `created_at` for current in-flight version

- **Severity:** MINOR (promoted from Agent C's NIT n1 because CLAUDE.md fabrication-test framing is more load-bearing than Agent C calibrated — see rationale)
- **Sources:** Agent C (n1)
- **Surface:** `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.tsx:42-48`
- **What's wrong:**
  ```tsx
  sortedVersions.push({
    id: "",
    version: currentVersion,
    created_at: new Date().toISOString(),  // <-- fabricated timestamp
    node_count: compositionState?.nodes.length ?? 0,
  });
  ```
  Per CLAUDE.md fabrication test: "If an auditor queries this field, will they get a value the external system actually provided?" No — this is a UI synthesis of "version that exists in memory but not yet persisted." `relativeTime()` will render "just now" indistinguishably from a real just-now timestamp. Downstream consumer cannot distinguish synthetic from real.
- **Promotion rationale (NIT → MINOR):** Agent C calibrated NIT on the basis that this is "UI display state" not crossing the audit boundary. That reading is correct in the narrow sense, but the *pattern* — fabricating a timestamp because absence is inconvenient to render — is exactly what CLAUDE.md's fabrication-decision-test forbids. Letting the pattern land in production seeds future "I needed a value, so I made one up" reasoning in adjacent UI surfaces (Phase 5 LLM interpretation, Phase 6 completion bar — both will face the same "what do I show when the data hasn't been recorded yet" choice). MINOR signals "fix this before it metastasises."
- **Smallest remediation:** Make `created_at` optional on the in-flight entry; render `"(current — not yet saved)"` for entries with no `created_at` instead of relative time. Option (a) from Agent C's report.
- **Verification:** Render the dropdown with a session whose `compositionState.version` is not in `stateVersions`; confirm the current row shows "(not yet saved)" instead of "just now."
- **Must-fix:** No — defer; tracker entry recommended.

#### P3B-004 — No shortcut-coherence test: App.tsx `handleKeyDown` ↔ ShortcutsHelp `SHORTCUTS` array

- **Severity:** MINOR
- **Sources:** Agent B (P3B-001), Agent D (B3B-002)
- **Surfaces:** `src/elspeth/web/frontend/src/App.tsx` `handleKeyDown` (lines 137-230) vs `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx:8-18` `SHORTCUTS` array
- **What's wrong:** No mechanical guard cross-checks that every advertised shortcut in `SHORTCUTS` has a live dispatch branch in `handleKeyDown`. P3A-002's predecessor guard for `TAB_SHORTCUT_MAP` was correctly deleted (the guarded structure is gone) but its *spirit* — "advertised shortcut must dispatch" — has no replacement at the new abstraction. If a developer adds a Ctrl+Shift+X entry to `SHORTCUTS` without adding the matching branch (or vice versa), no test fails. Same defect class as the original P3A-001 silent-Alt+1.
- **Smallest remediation:** One unit test in `App.test.tsx` (or co-located with `ShortcutsHelp.test.tsx`). Extract a `SHORTCUT_HANDLER_KEYS` set from `App.tsx`'s `handleKeyDown` branches (parsing or constant-listing both work); import `SHORTCUTS` from `ShortcutsHelp.tsx`; assert each `SHORTCUTS[n].keys` normalised form maps to a live handler. ~20 LOC.
- **Verification:** Delete a handler branch locally; the new test must fail. Restore; it must pass.
- **Must-fix:** No — current demo path is fully wired and behavioural App.test.tsx covers Ctrl+Shift+G/Y/P. Defer with rationale: the structural gap re-emerges only when the next shortcut is added.

#### P3B-005 — ValidationDot state machine has no successor test after InspectorPanel deletion

- **Severity:** MINOR
- **Sources:** Agent B (P3B-006), Agent D (B3B-004)
- **Surface:** `src/elspeth/web/frontend/src/components/sidebar/SideRailValidationBanner.test.tsx` (3 existing tests; missing ~5)
- **What's wrong:** Deleted `InspectorPanel.test.tsx` (734 LOC) had full coverage of the validation-dot state machine (amber=not-yet-validated, red=failed, green=passed, hidden=no pipeline / no nodes). The successor surface is `SideRailValidationBanner.tsx`, whose test file covers only error-display and node-selection — none of the four dot states.
- **Smallest remediation:** Add ~5 tests to `SideRailValidationBanner.test.tsx` covering the four states. Reference the deleted `InspectorPanel.test.tsx` cases as the contract being rehomed.
- **Verification:** Mutate each state in `SideRailValidationBanner.tsx`; corresponding test must fail.
- **Must-fix:** No — visual state machine renders correctly in staging; gap is "future refactor breaks it undetected" not current behavioural defect. Defer.

#### P3B-006 — `HeaderVersionSelector` revert-confirm cancel path not tested

- **Severity:** MINOR
- **Sources:** Agent D (B3B-003)
- **Surface:** `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.test.tsx:54-88`
- **What's wrong:** The revert-confirm dialog's "Cancel" branch — user opens confirm then dismisses — is not exercised. `revertToVersion` must not be called on Cancel; no test guards this branch.
- **Smallest remediation:** Add one test: open dropdown, pick non-current version, see confirm dialog, click Cancel, assert `revertToVersion` was not called and the dropdown returns to the prior state. ~15 LOC.
- **Verification:** Test passes against current implementation; flip the cancel handler to invoke `revertToVersion` locally; test must fail.
- **Must-fix:** No — defer.

#### P3B-007 — Phase 3B E2E coverage absent for all new flows

- **Severity:** MINOR
- **Sources:** Agent D (B3B-006)
- **Surface:** `src/elspeth/web/frontend/tests/e2e/` (no Phase 3B `.spec.ts` files)
- **What's wrong:** Existing `phase-3a-shell.spec.ts` correctly covers Phase 3A landmarks and remains green. No E2E spec covers Phase 3B flows: GraphModal open/close, ExportYamlModal open/close, CatalogButton trigger from SideRail, ExecuteButton from SideRail, HeaderVersionSelector interaction. All five flows are unit-tested.
- **Smallest remediation:** Add `phase-3b-side-rail.spec.ts` covering the five flows at smoke level. Defer until Phase 3B closes if the operator's pre-demo dry-run includes a manual walkthrough.
- **Verification:** Playwright spec runs green against staging.
- **Must-fix:** No — unit-test coverage is comprehensive (`App.test.tsx:297-344`, `ExportYamlModal.test.tsx`, `GraphModal.test.tsx`, etc.); operator demo dry-run is an effective informal smoke. Defer.

#### P3A-006 (carryover) — `RunsHistoryDrawer.test.tsx` missing ARIA contract assertions

- **Severity:** MINOR (carryover from v1, STILL-APPLIES per Agents A and D)
- **Sources:** Agent A (P3A-006 carryover), Agent D (P3A-006 carryover)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- **What's wrong:** Implementation sets `role="dialog"`, `aria-modal="true"`, `aria-label="Past pipeline runs"`. None are asserted. Phase 3B did not touch this surface; gap persists.
- **Smallest remediation:**
  ```typescript
  expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  expect(screen.getByRole("dialog")).toHaveAccessibleName("Past pipeline runs");
  ```
- **Verification:** Remove `aria-modal` locally; test must fail.
- **Must-fix:** No — defer; existing focus-trap test at `:170` provides partial guard.

---

### NIT

#### P3A-013-sub (carryover) — `SideRail.test.tsx` `aria-label="Composer side rail"` not asserted

- **Severity:** NIT (Phase 3B Task 0 moved the file from `common/` to `sidebar/` but didn't add the assertion)
- **Sources:** Agent A (P3B-004), Agent D (B3B-007), original P3A-013 sub-item carryover
- **Surface:** `src/elspeth/web/frontend/src/components/sidebar/SideRail.test.tsx`
- **What's wrong:** `SideRail.tsx:30` declares `<aside className="side-rail" aria-label="Composer side rail">`; no test asserts the aria-label is on the rendered aside.
- **Smallest remediation:** `expect(screen.getByRole("complementary", { name: /composer side rail/i })).toBeInTheDocument();` — one line.
- **Must-fix:** No — defer.

#### P3A-012 (carryover) — `RunsHistoryDrawer` focus-restore-on-close not test-guarded

- **Severity:** NIT (unchanged from v1)
- **Sources:** Agent A (P3A-012 carryover), Agent D (P3A-012 carryover)
- **Surface:** `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- **What's wrong:** Focus-into-drawer tested at `:170`; no symmetric "focus returns to trigger on close" test. Same gap pattern applies to new modals (per Agent D B3B-001 — absorbed by P3B-001 modal-extraction if pursued, or here for the drawer specifically).
- **Smallest remediation:** Add one userEvent test in `InlineRunResults.test.tsx`: render with `runs.length > 0`, focus the "Past runs" button, click it, close the drawer, assert `document.activeElement === pastRunsButton`.
- **Must-fix:** No — defer.

#### P3A-004 (carryover) — Plan 15a1 line 7 still references `VALID_TABS` preservation

- **Severity:** NIT
- **Sources:** Agent A (P3A-004 carryover)
- **Surface:** `docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md` (around line 7)
- **What's wrong:** 15a1 header says `VALID_TABS` and default-tab `"spec"` are preserved in this plan and migrated in 15b. Implementation went through narrowing (3A) then full removal (3B). Plan should reflect both steps.
- **Smallest remediation:** Update 15a1 line 7 to reference 15a2's narrowing decision and 15b's full removal. No code change.
- **Must-fix:** No — defer; documentary debt.

#### P3B-008 — Plan-doc inconsistency: 15b1 §Q7 vs 15b2 §Step 3a on event-constant location

- **Severity:** NIT
- **Sources:** Agent A (P3B-006)
- **Surface:** `docs/composer/ux-redesign-2026-05/15b1-phase-3b-side-rail-part-1.md:113` vs `15b2-phase-3b-side-rail-part-2.md:135-158`
- **What's wrong:** 15b1 §Q7 says `OPEN_CATALOG_EVENT` is re-exported from `CatalogButton.tsx`; 15b2 §Step 3a says "All parties import from this module. Component files must not define their own event-name constants." Implementation follows 15b1. The 15b2 directive is justified by circular-dependency risk between `useHashRouter` and modal components; `OPEN_CATALOG_EVENT` is never imported by `useHashRouter` (no `catalog` verb), so the rule doesn't apply — but the plan doesn't say so.
- **Smallest remediation:** Update 15b2 §Step 3a to note `OPEN_CATALOG_EVENT` is intentionally kept local to `CatalogButton.tsx` because the hash router has no `catalog` verb.
- **Must-fix:** No — defer; documentary clarity.

#### P3B-009 — Stale memory `project_phase2c_implementation_complete.md` references InspectorPanel mount location

- **Severity:** NIT
- **Sources:** Agent B (P3B-007)
- **Surface:** `/home/john/.claude/projects/-home-john-elspeth/memory/project_phase2c_implementation_complete.md`
- **What's wrong:** Memory records "panel mounted in InspectorPanel" / "above inspector tab strip"; Phase 3B (`a43594051`) deleted InspectorPanel; AuditReadinessPanel now mounts in `SideRail`'s `auditReadinessSlot` (App.tsx:310). Future session starting from this memory may infer InspectorPanel still exists.
- **Smallest remediation:** Add a note (or supersede) recording that the Phase 2C mount was absorbed into SideRail in Phase 3B `a43594051`.
- **Must-fix:** No — code is correct; documentation drift only.

---

## Re-Triage of Prior-Cycle Deferred Entries (P3A-004 through P3A-013)

| ID | Status at HEAD | Disposition | Rationale |
|---|---|---|---|
| P3A-004 | **STILL-APPLIES** | Carried forward unchanged (NIT) | 15a1 plan-doc edit still owed; Phase 3B did not touch |
| P3A-005 | **MOOT** | Resolved by Phase 3B | `HeaderSessionSwitcher.tsx` lines 26-129 now own `renameSession` / `archiveSession` with confirm flow via per-item ⋮ menu (Agent A + C verify) |
| P3A-006 | **STILL-APPLIES** | Carried forward unchanged (MINOR) | RunsHistoryDrawer ARIA assertions still missing; Phase 3B did not touch |
| P3A-007 | **MOOT** | Already-false-positive at v1 | `InlineRunResults.test.tsx:106-125` already had `userEvent.click → getByRole("dialog")` round-trip; v1 Agent D missed it, v2 Agent D replicated the miss, Agent A re-verified. Conflict resolved in favour of Agent A — direct grep confirms `it("opens and closes the Past runs drawer", async () => {…})` at line 106 |
| P3A-009 | **MOOT** | Resolved by Phase 3B | `InspectorPanel.tsx` deleted in `a43594051`; `handleValidationComponentClick` no longer exists; equivalent behaviour now in `SideRailValidationBanner.tsx:14-20` which only calls `selectNode` (tab-switch correctly gone) |
| P3A-010 | **MOOT** | Resolved by Phase 3B | `App.tsx:233-245` redirect-toast deleted in `bb9f12e4a` (hash-router rewrite) and confirmed gone by `5e909cb6c` (staging-shim deletion); no `role="status"` element to comment on |
| P3A-011 | **MOOT** | Resolved by Phase 3B | `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup deleted entirely in `5e909cb6c` per operator override (option a); the `useEffect` no longer exists |
| P3A-012 | **STILL-APPLIES** | Carried forward unchanged (NIT) | RunsHistoryDrawer focus-restore-on-close still not test-guarded; Phase 3B did not touch |
| P3A-013 | **PARTIALLY MOOT** | Sub-item-by-sub-item triage | (a) `SideRail.test.tsx` aria-label assertion → STILL-APPLIES (re-issued as P3A-013-sub above, MAJOR/MINOR/NIT-eligible carrier from synthesis-v2 with severity NIT). (b) `HeaderSessionSwitcher.test.tsx` aria-current → UNKNOWN (defer; not re-verified at HEAD; Phase 3B materially expanded this component). (c) `AppHeader.test.tsx` localStorage.clear → UNKNOWN (defer). (d) `subscriptions.test.ts:266-300` isExecuting race → UNKNOWN (defer). (e) `RunsHistoryDrawer.test.tsx:21-22` brittle getByText → UNKNOWN (Agent D confirms still present as NIT — STILL-APPLIES NIT). (f) `stores/subscriptions.ts:181-191` FRAGILE comment expansion → UNKNOWN (defer) |

**Triage totals:** 5 MOOT (P3A-005, P3A-007, P3A-009, P3A-010, P3A-011), 4 STILL-APPLIES (P3A-004, P3A-006, P3A-012, P3A-013-partial), 1 partially-MOOT with sub-items deferred for non-essential clarity.

---

## Resolved Without Action (Reconciliation of Original Must-Fix Entries)

The prior v1 synthesis identified four entries as **must-fix before merge**: P3A-001, P3A-002, P3A-003, P3A-008. All four were resolved between cycles by operator action and Phase 3A/B commits:

| Original entry | Resolution | Commits |
|---|---|---|
| **P3A-001** (Dead Spec/Runs references — umbrella, 5 sub-items) | RESOLVED by Phase 3A fix-loop `4a1dcc367` (5 sub-items addressed: tabMap narrowed, ShortcutsHelp updated, executionStore message rewritten, App.css banner removed, inspector-page.ts type union narrowed). Note sub-item 5 (inspector-page.ts) was narrowed but not deleted — re-emerges as P3A-014 above when Phase 3B's InspectorPanel deletion removes the file's reason to exist. | `4a1dcc367` |
| **P3A-002** (Test guard for tab-shortcut wiring) | RESOLVED by deletion. `82957ea14` deleted both the `TAB_SHORTCUT_MAP` constant and the `tab-shortcut-wiring.test.ts` guard, and added `App.test.tsx:297-344` behavioural assertions (Ctrl+Shift+G/Y fire, Alt+1 does NOT fire). The spirit of the guard is captured in P3B-004 above as a new gap at the new shortcut abstraction. | `82957ea14` |
| **P3A-003** (Migration shims — operator policy decision) | RESOLVED by operator override (option a). Initial implementation at `2ac40b164` chose option (c) keep-with-rationale; operator subsequently overrode to option (a) delete-all-three. Executed cleanly at `5e909cb6c` + `020db17e3`. | `2ac40b164` → `5e909cb6c` → `020db17e3` |
| **P3A-008** (RunsView capability re-homing decision) | RESOLVED by operator decision recorded in `03e6c9d0a` and `3ec3c22e1` (option a — capabilities retired; documentary note added to 15a2). No test re-homing was attempted (correct per chosen option). | `03e6c9d0a`, `3ec3c22e1` |

**All four original must-fix entries resolved out-of-band.** The new must-fix list reduces to a single item (P3A-014) which is the regression-of-omission inherited by the cleanup-sweep miss in `4a1dcc367`.

---

## Stop-Conditions Summary

### Must fix in this branch before merge (1 entry)

| ID | Title | Why must-fix | Effort |
|---|---|---|---|
| **P3A-014** | Orphan `inspector-page.ts` + companion README:41 staleness | CLAUDE.md "No Legacy Code Policy" violation; regression-of-omission from fix-loop's P3A-001 sweep; misleads next-session readers about a contract that no longer exists | 1 commit; `git rm` + 1 README line edit |

### Strongly recommended in-branch (1 entry — operator decision)

| ID | Title | Rationale for in-branch |
|---|---|---|
| **P3B-001** | Extract shared `<Modal>` primitive | Structural-debt signal Phase 3B introduced; 2-copy break-even point; Phase 5/6 plausibly add modal #3 within next 2 phases; absorbs aria-modal + focus-restore assertion gaps (B3B-001, B3B-005) into one well-tested primitive |

### Defer as observations / follow-up issues (12 entries)

| ID | Title | Deferral rationale |
|---|---|---|
| P3B-002 | Dead CSS `.spec-pending-proposal` / `.runs-pending-proposal` | Dead selectors, no runtime behaviour |
| P3B-003 | `HeaderVersionSelector` fabricated `created_at` | Pattern signal; not crossing audit boundary today |
| P3B-004 | No shortcut-coherence test | Current demo path fully wired; gap re-emerges only on next shortcut |
| P3B-005 | ValidationDot test coverage not rehomed | Visual machine works in staging; gap is future-refactor risk |
| P3B-006 | HeaderVersionSelector revert-cancel test missing | Single uncovered branch |
| P3B-007 | Phase 3B E2E coverage absent | Comprehensive unit coverage + operator dry-run mitigates |
| P3A-006 | RunsHistoryDrawer ARIA assertions missing | Partial guard via focus-trap test |
| P3A-013-sub | SideRail aria-label assertion missing | Composite hygiene item; single line per assertion |
| P3A-012 | RunsHistoryDrawer focus-restore test missing | Structural luck works in practice |
| P3A-004 | Plan 15a1 stale VALID_TABS text | Documentary debt |
| P3B-008 | Plan 15b1/15b2 event-constant location inconsistency | Documentary clarity |
| P3B-009 | Stale memory `project_phase2c_implementation_complete.md` | Documentation only |

---

## Findings Table (Machine-Readable Summary)

| ID | Severity | Phase | Category | Surface | Sources |
|---|---|---|---|---|---|
| P3A-014 | MAJOR | 3A-regression-of-omission | No-Legacy-Code Policy | `tests/e2e/page-objects/inspector-page.ts` + `tests/e2e/README.md:41` | A·P3B-001, B·P3B-003-sub, C·M1 |
| P3B-001 | MAJOR | 3B-new | Modal architecture | `GraphModal.tsx` + `ExportYamlModal.tsx` | A·P3B-002, B·P3B-004, C·m2, D·B3B-005 |
| P3B-002 | MINOR | 3B-cleanup-miss | Dead CSS | `App.css:2927-2944` | B·P3B-002 |
| P3B-003 | MINOR | 3B-new | Fabrication-decision-test | `HeaderVersionSelector.tsx:42-48` | C·n1 (promoted) |
| P3B-004 | MINOR | 3B-new | Mechanical-guard gap | App.tsx ↔ ShortcutsHelp.tsx | B·P3B-001, D·B3B-002 |
| P3B-005 | MINOR | 3B-rehoming-miss | Test coverage | `SideRailValidationBanner.test.tsx` | B·P3B-006, D·B3B-004 |
| P3B-006 | MINOR | 3B-new | Test coverage | `HeaderVersionSelector.test.tsx:54-88` | D·B3B-003 |
| P3B-007 | MINOR | 3B-new | E2E coverage | `tests/e2e/` | D·B3B-006 |
| P3A-006 | MINOR | 3A-carryover | Test hygiene | `RunsHistoryDrawer.test.tsx` | A·P3A-006, D·P3A-006 |
| P3A-013-sub | NIT | 3A-carryover | Test hygiene | `sidebar/SideRail.test.tsx` | A·P3B-004, D·B3B-007 |
| P3A-012 | NIT | 3A-carryover | Test hygiene | `RunsHistoryDrawer.test.tsx` | A·P3A-012, D·P3A-012 |
| P3A-004 | NIT | 3A-carryover | Plan-doc accuracy | `15a1-phase-3a-removals-part-1.md` | A·P3A-004 |
| P3B-008 | NIT | 3B-new | Plan-doc consistency | `15b1` §Q7 vs `15b2` §3a | A·P3B-006 |
| P3B-009 | NIT | 3B-new | Memory drift | `project_phase2c_implementation_complete.md` | B·P3B-007 |

**Severity counts:** 0 BLOCK / 2 MAJOR / 7 MINOR / 5 NIT = **14 entries**

**Phase 3A behavioural regressions:** 0
**Phase 3A regression-of-omission:** 1 (P3A-014)
**Prior-cycle entries resolved:** 9 of 9 (5 MOOT + 4 STILL-APPLIES carried forward) + 4 original must-fix entries resolved out-of-band

**Reviewer-conflict resolutions made:** 5 (P3A-007 false-positive recurrence, dead-CSS scope clarification, P3A-014 classification as Phase 3A regression-of-omission per Agent C, P3B-001 severity promotion to MAJOR per merge-bar re-evaluation directive in the brief, P3B-003 severity promotion to MINOR per fabrication-test framing).
