# Pattern Recognition Review — Phase 3A + Phase 3B (Full Post-3A Delivery)

**Scope:** Commit range `9e8c54ed6..020db17e3` (26 commits)
**Branch:** `feat/composer-phase-3-ia-cleanup`
**Specs reviewed:** `15a1`, `15a2`, `15b1`, `15b2`, `03-target-information-architecture.md`
**Prior review for context:** `.review/synthesis.md` (Phase 3A only, range `9e8c54ed6..c3fc5670e`)
**Reviewer:** Systems Pattern Recognition Agent
**Date:** 2026-05-17

---

## Confidence Assessment

- **High confidence** on Fixes that Fail (P3B-001): direct git log evidence, deleted file confirmed
- **High confidence** on Tragedy of the Commons (P3B-002, P3B-003): static grep confirms no consumers
- **High confidence** on Shifting the Burden (P3B-004): direct source read, both modal files identical in structure
- **High confidence** on Limits to Growth (P3B-005): SideRail.tsx slot model directly read
- **Medium confidence** on Eroding Goals (P3B-006): no test run executed; inference from deleted/added test counts and git diffs
- **Low confidence** on uncovered edge cases in useHashRouter — the test suite passes per commit messages; no independent execution

## Risk Assessment

- **P3B-001** carries demo risk if a future shortcut is added to ShortcutsHelp without a corresponding handler branch — the same class of silent failure as P3A-001's Alt+1 no-op
- **P3B-002** (dead CSS) is low risk — dead selectors have no runtime behavior; risk is maintenance noise only
- **P3B-003** (E2E page-object / README stale) is low risk — the `InspectorPage` class is not imported by any E2E spec; the README stale text is documentation only
- **P3B-004** (modal duplication) becomes higher risk as the system grows — each new modal copies the pattern
- **Phase 3A regression count: 0** — P3A-001, P3A-002, P3A-003 were all correctly handled (P3A-002 guard was correctly deleted because its guarded structure was deleted; a replacement gap exists but it is not a regression of landed work)

## Information Gaps

- Test suite was not independently executed; green-at-HEAD status is inferred from commit messages
- App.css has thousands of lines; the orphan scan was targeted, not exhaustive
- No Storybook, no i18n system — those surfaces do not apply

## Caveats

The prior Phase 3A synthesis (`.review/synthesis.md`) framed P3A-002 as a "must-fix before merge." Phase 3B correctly deleted P3A-002's test because the guarded data structure (`TAB_SHORTCUT_MAP`) was deleted — that is not a regression. However, the *spirit* of P3A-002 (a cross-check that advertised shortcuts are live) has no replacement at the new abstraction. This report treats that as a new finding (P3B-001) rather than a Phase 3A regression.

---

## Commit Sequence (26 commits)

```
9e8c54ed6  (base — 03 doc fix, pre-3A)
84306f19f  Phase 3A.1 — InlineRunResults + RunsHistoryDrawer
ccca39cb0  Phase 3A.2 — SideRail scaffold + rename Layout slot
ae8df6dc3  Phase 3A.3 — AppHeader with session switcher + UserMenu
86061cf1c  Phase 3A.4 — auto-validate on compositionState.version
66748edb9  Phase 3A.5 — remove Runs tab
8b04d53ce  Phase 3A.6 — remove Spec tab
c3fc5670e  Phase 3A.7 — remove SessionSidebar
4a1dcc367  fix — scrub dead Spec/Runs tab references (P3A-001)
7cbd1d666  test — guard tab-shortcut wiring (P3A-002)
2ac40b164  docs — annotate migration shims (P3A-003 option c, later overridden)
03e6c9d0a  docs — record retired RunsView capabilities (P3A-008)
3ec3c22e1  docs — fix P3A-008 section placement
a30e07f20  chore — checkpoint pre-phase3b worktree
2e1b0b6a9  refactor — move SideRail into sidebar/ dir (Phase 3B Task 0)
1f6b0e0dd  feat — header composition history selector (Phase 3B Task 4)
369c1d9f4  feat — move Execute button to SideRail (Phase 3B Task 2)
39b946a2b  feat — add graph mini modal (Phase 3B Task 1a)
66d6a196e  feat — add YAML export modal (Phase 3B Task 1b)
bb9f12e4a  feat — rewrite hash router for composer actions (Phase 3B Task 6)
c22a7fe81  feat — retarget palette commands to composer modals (Phase 3B Task 7)
82957ea14  feat — retarget app shortcuts to composer modals (Phase 3B Task 8)
a43594051  feat — remove composer inspector panel (Phase 3B Task 9)
bf09e4339  refactor — remove composer inspector tab remnants (Phase 3B cleanup)
5e909cb6c  refactor — delete staging migration shims (P3A-003 option a override)
020db17e3  docs — fill in P3A-003 commit SHA in 15a2 decision log
```

---

## Variables Identified

- **ShortcutsHelp SHORTCUTS array** — 10 entries; can increase or decrease
- **App.tsx handleKeyDown branches** — 8 active dispatch branches; can increase or decrease
- **Modal count** — 2 (GraphModal, ExportYamlModal); will grow as surfaces are added
- **SideRail tenant count** — 7 slots (auditReadiness, validationBanner, graphMini, catalog, exportYaml, executeButton, completionBar); completionBar=null currently; will populate
- **Dead CSS selector count** — 2 confirmed (spec-pending-proposal, runs-pending-proposal)
- **InspectorPanel.test.tsx coverage** — 734 lines deleted; partial rehoming confirmed (version selector: 5 tests, validation banner: 3 tests, validation-dot states: 0 tests)

---

## Archetype Findings

### P3B-001 — MAJOR | Fixes that Fail | "No replacement guard for shortcut-handler wiring"

**Archetype:** Fixes that Fail

**Feedback loop:**
```
R1: Advertised-shortcut silent-failure loop
ShortcutsHelp SHORTCUTS entries → users attempt shortcut → if no handler → silent no-op → trust lost
Fix (specific behavioral test) → removes immediate failure → reduces pressure for structural guard → next shortcut added without guard → pattern recurs
```

**Evidence:**

Phase 3A introduced P3A-001 (dead Alt+1/Alt+4 shortcuts, demo-visible) and P3A-002 (mechanical guard: test cross-checking `TAB_SHORTCUT_MAP` keys against `TABS`). Phase 3B:

1. Deleted `TAB_SHORTCUT_MAP` in commit `82957ea14` — the constant that P3A-002 guarded no longer exists
2. Deleted `src/elspeth/web/frontend/src/__tests__/tab-shortcut-wiring.test.ts` in the same commit (`82957ea14`, confirmed via `git log --diff-filter=D -- "*/tab-shortcut-wiring.test.ts"`)
3. Replaced the Alt+digit tab shortcuts with Ctrl+Shift+G/Y modal events
4. Added `App.test.tsx:297` testing specific modal dispatch — behavioral coverage, not structural guard

**What is missing:** No test cross-checks that every entry in `ShortcutsHelp.tsx`'s `SHORTCUTS` array has a matching dispatch branch in `App.tsx`'s `handleKeyDown`. The `SHORTCUTS` array has 10 entries. `handleKeyDown` has 8 active dispatch branches. `"Escape"` and `"?"` are handled structurally (component-level), not in `handleKeyDown`. This matches — but it's not mechanically verified.

**The P3A-001 pattern replayable?** Yes. If a new shortcut is added to `SHORTCUTS` (the advertising surface) without a corresponding branch in `handleKeyDown` (the dispatch surface), the defect class silently recurs. CLAUDE.md: *"if it's not mechanically enforced (by types, tests, CI, or named constants), assume the next session won't know about it."*

**Phase 3A regression?** No — the deletion was mechanically correct (the guarded structure was deleted). This is a new gap at the new abstraction level.

**Diagnostic questions:**
- Does the fix work initially? Yes — the individual App.test.tsx:297 test covers Ctrl+Shift+G/Y dispatch
- Does applying more fix reduce pressure for structural guard? Yes — specific behavioral tests satisfy code review without exposing the structural gap
- Side effects making original problem worse? Not yet — but structural risk grows with each new shortcut added

**Smallest remediation:** Add a constant-coherence test (no DOM rendering) analogous to P3A-002's shape. Extract a `SHORTCUT_HANDLER_KEYS` set from `App.tsx`'s `handleKeyDown` branches, import `SHORTCUTS` from `ShortcutsHelp.tsx`, assert every `SHORTCUTS[n].keys` normalized form maps to a live handler. One test, ~20 LOC.

**Meadows intervention level:** Level 6 (Information Flows) — the structural relationship between advertising and dispatch is not visible to the system; a test makes it self-announcing.

**Must-fix before merge:** Operator decision. Demo path does not exercise the structural gap. Framed as MAJOR because P3A-001's root cause was exactly this structural-gap class.

---

### P3B-002 — MINOR | Tragedy of the Commons | "Dead CSS from deleted SpecView and RunsView"

**Archetype:** Tragedy of the Commons

**Evidence:**

`App.css` contains two class rules with no consumers in any `.tsx` or `.ts` file:

- `.spec-pending-proposal` — lines 2927, 2943 (verified: `grep -rn "spec-pending-proposal" src/ --include="*.tsx" --include="*.ts"` returns zero hits)
- `.runs-pending-proposal` — lines 2929, 2944 (same: zero hits)

Note: `.yaml-pending-summary` on the same block IS used by `YamlView.tsx:142`.

The `InspectorPanel.test.tsx` deletion commit (`a43594051`) deleted 134 lines from App.css but did not sweep the SpecView/RunsView CSS that SpecView and RunsView were the sole consumers of. These classes were introduced with the now-deleted `SpecView.tsx` and `RunsView.tsx`.

**Why it happens:** Each deletion commit (3A.5 — Runs tab, 3A.6 — Spec tab) removed the TSX files but did not run a "does any live code use this CSS class?" sweep before closing. With no enforced CSS-consumer registry, the commons (App.css) degraded silently.

**Smallest remediation:** Delete both class rules plus their shared dashed-border `color-mix` block (lines 2927–2950 in App.css, verified that `yaml-pending-summary` uses only the parent block, not the `.spec-`/`.runs-` specific rules). Then confirm `css-modules` or a CSS-consumer grep in CI for future deletions.

**Meadows intervention level:** Level 6 (Information Flows) — the information that these classes have no consumers is not visible to the committer.

**Must-fix:** No — dead CSS has no runtime behavior. Defer as observation.

---

### P3B-003 — MINOR | Tragedy of the Commons | "Orphan E2E page-object and README stale reference"

**Archetype:** Tragedy of the Commons

**Evidence — two sub-items sharing one root cause (deletions without sweeping all referencing documents):**

**Sub-item 3A:** `src/elspeth/web/frontend/tests/e2e/page-objects/inspector-page.ts`

The file itself was correctly cleaned in commit `4a1dcc367` (P3A-001): `InspectorTab` type is now `"graph" | "yaml"` (line 6); line 1 comment says "(tabs: graph/yaml)". However:

- The class `InspectorPage` is NOT imported or used by any E2E spec file (confirmed: `grep -rn "InspectorPage\|inspector-page" tests/e2e/*.spec.ts` returns zero hits)
- The file thus exists as a dead page-object class with no spec consumers

This is a partial Tragedy of the Commons: the page object was updated but not assessed for whether it is still needed.

**Sub-item 3B:** `src/elspeth/web/frontend/tests/e2e/README.md:41`

```text
│   ├── inspector-page.ts        right panel (spec/graph/yaml/runs tabs)
```

Still says "(spec/graph/yaml/runs tabs)" — all four historical tab names. The correct text after Phase 3B would be "(graph/yaml modals; page-object not yet exercised by specs)".

**Smallest remediation:** Determine whether any E2E spec needs `InspectorPage` in its current form. If yes, write the test. If no, delete `inspector-page.ts` and update README.md:41. One commit.

**Meadows intervention level:** Level 6 (Information Flows).

**Must-fix:** No — the page object file has no runtime impact. Defer as observation.

---

### P3B-004 — MINOR | Shifting the Burden | "No shared modal primitive; copy-paste pattern accumulates"

**Archetype:** Shifting the Burden

**Evidence:**

Commits `39b946a2b` and `66d6a196e` introduced two new modal components:

- `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx` (70 lines)
- `src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.tsx` (70 lines)

Both files are structurally identical — the only differences are the CSS prefix (`graph-modal-` vs `yaml-modal-`), the modal title, and the content component (`<GraphView />` vs `<YamlView />`). Both implement:

- `useState(false)` open/close
- `useRef<HTMLDivElement>` dialogRef
- `useId()` titleId
- `useFocusTrap(dialogRef, isOpen, ".xxx-modal-close")`
- `useEffect(() => window.addEventListener(OPEN_XXX_EVENT, ...))` listener
- `useEffect(() => document.addEventListener("keydown", ...)` Escape handler
- `role="dialog"` / `aria-modal="true"` / `aria-labelledby={titleId}`
- backdrop `<div>` with `onClick={() => setIsOpen(false)}`

Plan 15b1 explicitly documented this pattern choice: *"mirrors the SecretsPanel.tsx modal shape."* The duplication is plan-faithful, not unplanned.

**Why it is Shifting the Burden:** The plan noted the shape was borrowed from `SecretsPanel.tsx`; it did not extract a shared primitive. The pressure to deliver each new surface quickly ("just copy SecretsPanel") deferred the extraction. A third modal (`ConfirmDialog` already exists but has a different shape) will face the same choice and will likely copy again.

**Loop structure:**
```
R1 (growth): New modal needed → copy nearest modal → works immediately → reduces pressure to abstract → next modal copies again
B1 (balancing): Technical debt accumulates → ARIA contract diverges across copies → maintenance friction increases → pressure to abstract eventually wins
```

The current system is in the early R1 phase (2 copies). Bug risk is low now; it grows quadratically with each additional copy (each copy can independently diverge on ARIA, focus trap, Escape behavior).

**Smallest remediation:** No action required now. If a third modal surface is added (Phase 6 completion bar, Phase 5 LLM interpretation panel), extract a `<ComposerModal>` primitive at that point. Track as observation for Phase 6.

**Meadows intervention level:** Level 10 (System Structure) — requires creating the shared abstraction rather than patching individual copies.

**Must-fix:** No — 2 copies is the break-even point; risk is theoretical until copy 3.

---

### P3B-005 — NIT | Limits to Growth | "SideRail slot order is hard-coded in JSX"

**Archetype:** Limits to Growth (mild)

**Evidence:**

`SideRail.tsx` receives 7 named props as `ReactNode` slots and renders them in fixed order:

```
auditReadinessSlot → validationBannerSlot → graphMiniSlot → catalogSlot → exportYamlSlot → executeButtonSlot → completionBarSlot
```

The slot model correctly isolates tenants (each prop is independently nullable; empty slots are hidden via `.side-rail-slot:empty { display: none }`). Any tenant can be added or removed by changing the App.tsx prop assignment without editing SideRail.tsx.

**However:** the render order is structurally fixed in SideRail.tsx's JSX. A Phase 6 or Phase 7 change that needs to reorder tenants (e.g., Execute below completionBar, or Catalog above audit-readiness) requires editing SideRail.tsx itself, not just App.tsx. This is a mild structural constraint, not a current defect.

**The slot model IS coherent for the "addable/removable" test:** each slot can be nulled or populated independently without breaking others. The constraint is specifically on *ordering*.

**Smallest remediation:** Document the slot order as intentional in SideRail.tsx comments. If reordering becomes needed, the fix is to move the slot `<div>` in SideRail.tsx — one-line change, no prop API change. Track as observation for Phase 6 when completionBar slot gets populated.

**Meadows intervention level:** Level 10 (Structure) — the constraint is in the component structure.

**Must-fix:** No.

---

### P3B-006 — MINOR | Eroding Goals | "InspectorPanel.test.tsx coverage not fully rehomed after Phase 3B deletion"

**Archetype:** Eroding Goals (pressure → lower standard)

**Evidence:**

The pressure to delete InspectorPanel (`a43594051`) was legitimate — the component had no remaining content. However, 734 lines of tests were deleted and only partially rehomed:

| Behavior class | InspectorPanel tests | Successor tests | Gap |
|---|---|---|---|
| Validation dot (amber/red/green/hidden) | ~5 tests | 0 | Complete gap |
| Version selector | ~8 tests | 5 (HeaderVersionSelector.test.tsx) | Partial |
| Tab default (graph on startup) | ~2 tests | 0 | Complete gap |
| ARIA (no aria-live on tabpanel) | ~2 tests | 0 | Complete gap |
| Render-without-crash (null session/state) | ~1 test | 0 (App.test.tsx has smoke but not null-null) | Partial |
| Audit readiness panel mounting | ~4 tests | App.test.tsx:236 (smoke) | Partial |
| Catalog/Execute absent from inspector | ~2 tests | App.test.tsx:240 (negative assertion) | Partial |

The validation-dot state machine (amber for not-yet-validated, red for failed, green for passed, hidden for no pipeline / no nodes) had full coverage in the old tests and has zero coverage now. This machine is now realized by `SideRailValidationBanner.tsx` — but `SideRailValidationBanner.test.tsx` has only 3 tests covering error display and node-selection, not the validation-dot states.

**Diagnostic questions:**
- Are standards slipping from pressure? Yes — the pressure here was correct architectural cleanup (delete the dead component), but the test sweep did not follow
- Could we hit the original standard given more time? Yes — the validation-dot states are straightforward unit tests on `SideRailValidationBanner`

**Smallest remediation:** Add ~5 validation-dot state tests to `SideRailValidationBanner.test.tsx` covering the four states (not-validated, passed, failed-with-warnings, hidden). Reference the deleted `InspectorPanel.test.tsx` cases "shows amber dot when not validated", "shows green dot when validation passed", "shows red dot when validation failed", "hides dot when no pipeline", "hides dot when pipeline has no nodes".

**Meadows intervention level:** Level 3 (Goals) — the standard for test coverage on moved behaviors needs to be reinstated.

**Must-fix:** No — no behavioral regression is asserted; the validationBanner renders correctly in staging. Defer to a follow-up commit on this branch or as a companion to Phase 3B closure.

---

### P3B-007 — NIT | Tragedy of the Commons | "project_phase2c_implementation_complete memory stale — AuditReadinessPanel mount location"

**Archetype:** Tragedy of the Commons (shared documentation commons degrading)

**Evidence:**

Memory `project_phase2c_implementation_complete.md` records:

> "Mount `<AuditReadinessPanel />` above inspector tab strip + 4 mount tests"
> (commit 5d689b9d2)

Phase 3B (`a43594051`) deleted InspectorPanel entirely. AuditReadinessPanel is now mounted in `SideRail`'s `auditReadinessSlot` in App.tsx (line 310). The memory still describes the Phase 2C mount location as "above inspector tab strip" — which no longer exists.

**Impact:** A future session starting from this memory may incorrectly infer that InspectorPanel still exists as a mounting surface. The memory `project_phase2c_implementation_complete.md` also describes a "Coverage gap acknowledged" note about the Phase 3 remount, which is now resolved but the resolution is not recorded in the memory.

**Smallest remediation:** Add a note to the memory (or supersede it) stating that the InspectorPanel mount from Phase 2C was absorbed into SideRail in Phase 3B commit `a43594051`.

**Meadows intervention level:** Level 6 (Information Flows).

**Must-fix:** No — the code is correct; the memory is documentation.

---

## Phase 3A Regression Check

The brief requires explicit confirmation of whether Phase 3B **undid** any Phase 3A landed work.

| P3A item | Phase 3B treatment | Regression? |
|---|---|---|
| P3A-001 — Dead Spec/Runs references fixed | All 5 sub-items confirmed clean at HEAD. No "spec" or "runs" literals in production code outside intentional hash-router strip-path. | None |
| P3A-002 — Tab-shortcut wiring guard test | Deleted by `82957ea14`. Guard was correct to delete: `TAB_SHORTCUT_MAP` (the guarded constant) was also deleted in the same commit. The spirit of the guard has no replacement (P3B-001 above). | Not a regression — guarded structure is gone. New gap exists. |
| P3A-003 option (c) → overridden to option (a) | Correctly executed by `5e909cb6c`. Migration shims deleted. Operator override recorded in `15a2` doc. | None |
| P3A-008 — RunsView capabilities retired | `03e6c9d0a` records the retirement decision. InspectorPanel deletion is consistent with the retirement. | None |

**Phase 3A regression count: 0**

---

## Fixes that Fail — Pattern Replay Assessment (Priority Question from Brief)

**Did Phase 3B replay the P3A-001 dead-call-site defect class?**

Specific check results:

1. `"spec"` or `"runs"` literal references in `src/elspeth/web/frontend/src/` (excluding intentional hash-router paths): **None found.** The hash-router tests at lines 29/37 assert `#/spec` and `#/runs` fragments are silently stripped — correct behavior, not a defect.

2. `TAB_SHORTCUT_MAP` after retarget commit: **Does not exist.** Completely removed in `82957ea14`. The structural guard (P3A-002) was correctly deleted with it. A replacement structural guard for the new shortcut architecture is absent (P3B-001).

3. `InspectorPanel` orphan imports or dead test files: **None.** The component directory (`components/inspector/`) contains only `GraphView`, `YamlView`, `RunOutputsPanel` — all three are live. No orphan test file for InspectorPanel.

4. `InspectorPanel` orphan CSS: **Partially cleaned.** No `.inspector-panel`, `.inspector-tab`, `.inspector-content` CSS found. The `.spec-pending-proposal` and `.runs-pending-proposal` classes (from SpecView and RunsView, not InspectorPanel itself) survived — see P3B-002.

**Verdict:** The Phase 3B commits did NOT replay the P3A-001 dead-call-site defect for InspectorPanel removal. The cleanup was more thorough. The surviving gap (P3B-001) is a structural-guard gap, not a dead-call-site gap.

---

## Shifting the Burden Check (Brief Question 2)

**Is there one shared `<Modal>` component, or did each commit invent its own?**

Two modals exist: `GraphModal.tsx` and `ExportYamlModal.tsx`. Both are structurally identical (70 lines each). No shared `<Modal>` primitive exists.

**Mitigating factor:** Plan 15b1 explicitly documented the pattern as borrowing from `SecretsPanel.tsx` shape — this was a documented architectural choice, not an accidental copy. The duplication is plan-faithful.

**Risk framing:** With 2 copies, the risk is low. With 3+ copies (Phase 6 will likely add a completion-gesture modal), the probability of ARIA or focus-trap divergence becomes non-trivial. Treat as P3B-004 (MINOR) and track for Phase 6.

---

## Eroding Goals Check (Brief Question 3)

**Did any commit relax a test, skip one, or weaken an assertion?**

Deleted test files across 26 commits:

| Commit | File deleted | LOC | Justification |
|---|---|---|---|
| `66748edb9` | `RunsView.test.tsx` | ~579 | Component deleted |
| `8b04d53ce` | `SpecView.test.tsx` | unknown | Component deleted |
| `c3fc5670e` | `SessionSidebar.test.tsx` | unknown | Component deleted |
| `a43594051` | `InspectorPanel.test.tsx` | 734 | Component deleted |
| `82957ea14` | `tab-shortcut-wiring.test.ts` | 30 | Guarded structure deleted |

All deletions are correctly motivated by component/structure removal. The only concern is the coverage gap from `InspectorPanel.test.tsx` — specifically validation-dot state transitions — which have no successor tests (P3B-006).

Test count trajectory inferred from commit messages: started at ~672 (7cbd1d666 commit message); Phase 3B additions include SideRailValidationBanner (3 tests), HeaderVersionSelector (5 tests), ExportYamlModal tests, GraphModal tests, CommandPalette tests (+26 inferred), ShortcutsHelp tests, useHashRouter tests (6). No commit message claims a test count reduction.

**No test was weakened to land.** The test deletions matched component deletions. One coverage gap (P3B-006) exists from incomplete rehoming.

---

## Tragedy of the Commons Check (Brief Question 4)

**Orphan scan results across SpecView, RunsView, SessionSidebar, InspectorPanel deletions:**

| Resource type | Orphan found | Evidence |
|---|---|---|
| Import statements | None | grep confirmed |
| Dead test files | None (all deleted with components) | ls confirmed |
| Dead i18n keys | N/A — no i18n system | |
| Dead Storybook entries | N/A — no Storybook | |
| Dead route registrations | N/A — single-page app, no router | |
| Dead CSS | YES: `.spec-pending-proposal`, `.runs-pending-proposal` | No tsx consumers (P3B-002) |
| E2E page-object unused | YES: `InspectorPage` class not imported by any spec | grep confirmed (P3B-003) |
| README stale text | YES: `inspector-page.ts` description says "spec/graph/yaml/runs tabs" | README.md:41 (P3B-003) |
| Memory stale | YES: `project_phase2c_implementation_complete.md` says "above inspector tab strip" | (P3B-007) |

---

## Limits to Growth Check (Brief Question 5)

**SideRail slot coherence:**

`SideRail.tsx` implements a slot-injection model via 7 named `ReactNode` props. The mount in App.tsx:

```
auditReadinessSlot={<AuditReadinessPanel />}
validationBannerSlot={<SideRailValidationBanner />}
graphMiniSlot={<GraphMiniView />}
catalogSlot={<CatalogButton />}
exportYamlSlot={<ExportYamlButton />}
executeButtonSlot={<ExecuteButton />}
completionBarSlot={null}   ← reserved for Phase 6
```

Assessment: The slot model is coherent for addability and removability. Any tenant can be nulled or set independently. The only constraint is render order (hard-coded in SideRail.tsx JSX) — reordering requires editing SideRail.tsx. See P3B-005 (NIT).

---

## Memory Drift Check (Brief Question 6)

| Memory | Relevant content | Status after Phase 3B |
|---|---|---|
| `project_phase2c_implementation_complete.md` | "panel mounted in InspectorPanel" | STALE — InspectorPanel deleted; panel now in SideRail (P3B-007) |
| `project_phase1b_default_mode_frontend_complete.md` | Frontend UX for default-mode | Not impacted by Phase 3 |
| `feedback_locked_in_buggy_expectations.md` | Tests locked in the bug; update tests, not revert | Followed: CompletionSummary.test.tsx was correctly updated when dispatch target changed |
| `feedback_no_path_priming_in_dispatch.md` | Don't seed paths into dispatch prompts | Not applicable to this delivery (no subagent dispatch in these commits) |
| `feedback_subagents_cant_use_worktrees.md` | Subagents inherit parent CWD | Not applicable — no subagent dispatch in these commits |
| `feedback_default_is_fix_not_ticket.md` | Investigation that surfaces a fixable defect MUST fix in session | Followed: P3A-001 was fixed in `4a1dcc367`, not deferred to a ticket |

---

## Stop-Conditions Summary

### MAJOR (1)

| ID | Title | Phase 3A regression? | Must-fix |
|---|---|---|---|
| P3B-001 | No replacement structural guard for shortcut-handler wiring after TAB_SHORTCUT_MAP deletion | No | Operator decision — demo path covered; structural risk grows on next shortcut addition |

### MINOR (3)

| ID | Title | Phase 3A regression? | Must-fix |
|---|---|---|---|
| P3B-004 | Two structurally identical modals — no shared primitive; Shifting the Burden accumulates | No | No — defer until third modal surfaces |
| P3B-006 | InspectorPanel.test.tsx coverage not fully rehomed — validation-dot states unguarded | No | No — defer; add to SideRailValidationBanner.test.tsx in cleanup pass |
| P3B-003 | Orphan E2E page-object (unused) + README stale tab description | No | No — defer as observation |

### NIT (2)

| ID | Title | Phase 3A regression? | Must-fix |
|---|---|---|---|
| P3B-002 | Dead `.spec-pending-proposal` / `.runs-pending-proposal` CSS in App.css | No | No — dead selectors, delete in cleanup |
| P3B-005 | SideRail slot render order hard-coded in JSX | No | No — document as intentional; revisit if reorder is needed in Phase 6 |
| P3B-007 | Memory `project_phase2c_implementation_complete.md` says "above inspector tab strip" (stale) | No | No — documentation only |

---

## Archetype Summary Table

| ID | Severity | Archetype | Meadows Level | Evidence |
|---|---|---|---|---|
| P3B-001 | MAJOR | Fixes that Fail | L6 (Info) | `82957ea14` deletes guard; no replacement; `ShortcutsHelp.tsx:8-18` vs `App.tsx:137-230` |
| P3B-002 | NIT | Tragedy of the Commons | L6 (Info) | `App.css:2927,2929,2943,2944`; zero tsx consumers |
| P3B-003 | MINOR | Tragedy of the Commons | L6 (Info) | `tests/e2e/page-objects/inspector-page.ts` unused; `README.md:41` stale |
| P3B-004 | MINOR | Shifting the Burden | L10 (Structure) | `GraphModal.tsx` and `ExportYamlModal.tsx` 70-line identical structure |
| P3B-005 | NIT | Limits to Growth | L10 (Structure) | `SideRail.tsx` fixed slot order |
| P3B-006 | MINOR | Eroding Goals | L3 (Goals) | `InspectorPanel.test.tsx` 734 lines deleted; validation-dot states have no successor |
| P3B-007 | NIT | Tragedy of the Commons | L6 (Info) | `project_phase2c_implementation_complete.md` stale mount description |
