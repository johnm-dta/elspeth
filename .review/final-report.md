# Phase 3A IA-Cleanup — Final Orchestration Report

**Branch:** `feat/composer-phase-3-ia-cleanup`
**Base commit (pre-fix-loop):** `c3fc5670e` (last commit of original Phase 3A delivery)
**Final tip:** `3ec3c22e104ee3851bd1f365fb82b9462c460bc3`
**Date:** 2026-05-17
**Orchestrator pattern:** 4 parallel reviewers → 1 synthesizer → per-defect writer + reviewer fix loop

---

## TL;DR

The Phase 3A delivery shipped clean **structurally** (no BLOCK findings from any of four independent reviewers) but with a cluster of dead Spec/Runs references that the deletion commits forgot to grep for. The fix loop closed every must-fix entry, added a mechanical guard so the defect class can't silently recur, and deferred 9 MINOR/NIT items to observations. **Recommendation: ready to merge to `RC5.2` pending operator confirmation of two reasonable-call defaults + a 60-second runtime Alt-key smoke check** (details in §6, §7).

---

## 1. Commits added by the fix loop

Five commits land on top of `c3fc5670e`:

```
3ec3c22e1 docs(composer): fix P3A-008 section placement (useHashRouter handoff stays in pre-dispatch)
03e6c9d0a docs(composer): record retired RunsView capabilities (P3A-008 option a)
2ac40b164 docs(web/frontend): annotate staging migration shims with retention rationale (P3A-003 option c)
7cbd1d666 test(web/frontend): guard tab-shortcut wiring against silent dead-key regression (P3A-002)
4a1dcc367 fix(web/frontend): scrub dead Spec/Runs tab references (P3A-001)
```

Two-commit P3A-008 (initial + placement fix) is transparent in `git log`. Operator may choose to squash at merge-review time; the net diff is correct either way.

---

## 2. Punch-list outcomes

### Must-fix (all closed)

| ID | Title | Verdict | Commit |
|---|---|---|---|
| P3A-001 | Dead Spec/Runs references (5 sub-items in one atomic commit) | PASS | `4a1dcc367` |
| P3A-002 | Mechanical guard: every Alt-key shortcut targets a live tab | PASS | `7cbd1d666` |
| P3A-003 | Migration-shim rationale comments (operator-gate, default opt c) | PASS | `2ac40b164` |
| P3A-008 | Record RunsView retired intent (operator-gate, default opt a) | PASS | `03e6c9d0a` + `3ec3c22e1` |

All four verifications were performed by an independent reviewer agent that re-ran the prescribed verification check (including a mutation sanity-check for P3A-002 that produced the diagnostic `"Alt+9 targets \"phantom\" which is not a live tab — update TAB_SHORTCUT_MAP or restore the tab"`).

### Deferred to observations (9 entries — operator promotes selectively)

These are listed here with file:line precision so the operator can decide what to file as a `filigree` observation versus what to absorb into a follow-up branch. None block merge.

| ID | Severity | File:Line | One-line description |
|---|---|---|---|
| P3A-004 | MINOR | `docs/composer/ux-redesign-2026-05/15a1-phase-3a-removals-part-1.md:7` | Plan 15a1 header text contradicts 15a2's revised `VALID_TABS` decision; doc-only update |
| P3A-005 | MINOR | `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx` + `components/common/CommandPalette.tsx:165-194` | Session rename/delete reachable only via 15b surface; not on demo paths |
| P3A-006 | MINOR | `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx` | `role="dialog"` / `aria-modal` / `aria-label` set in impl but not asserted in tests |
| P3A-007 | MINOR | `src/elspeth/web/frontend/src/components/execution/InlineRunResults.test.tsx` | "Past runs" button is asserted present; click-opens-drawer interaction not tested |
| P3A-009 | MINOR | `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx:393-403` | `handleValidationComponentClick` retains `setActiveTab("graph")` against plan; impl is the better UX, plan needs updating |
| P3A-010 | NIT | `src/elspeth/web/frontend/src/App.tsx:234` | Redirect-toast uses correct `role="status"`; plan asked for `role="alert"`; add an explanatory comment |
| P3A-011 | NIT | `src/elspeth/web/frontend/src/App.tsx:73-75` | `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup uses `useEffect` not module-scope; subsumed by P3A-003 if shims later removed |
| P3A-012 | NIT | `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx` | Focus-return-to-trigger relies on React re-render luck; add a test guard |
| P3A-013 | NIT (composite) | Multiple — see synthesis §P3A-013 | Six belt-and-braces test improvements batched: `SideRail.test.tsx` aria-label, `HeaderSessionSwitcher.test.tsx` aria-current, `AppHeader.test.tsx` localStorage isolation, `subscriptions.test.ts:266-300` isExecuting-mid-loop, `RunsHistoryDrawer.test.tsx:21-22` brittle queries, `stores/subscriptions.ts:181-191` FRAGILE comment expansion |

Full narrative and dependency rationale for each deferred entry: `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup/.review/synthesis.md`.

---

## 3. Final exit-condition gates

| Gate | Status | Evidence |
|---|---|---|
| Every must-fix entry has a PASS verdict | OK | 4 of 4 |
| `npm test -- --run` in `src/elspeth/web/frontend` | OK | 672 tests pass across 69 files |
| `npm run typecheck` in `src/elspeth/web/frontend` | OK | `tsc -p tsconfig.app.json --noEmit` exit 0 |
| No backend changes in `9e8c54ed6..HEAD` | OK | Only non-frontend non-docs deltas are `.gitignore` (`+*.tsbuildinfo`) and `tsconfig.tsbuildinfo` deletion, both from the original Phase 3A.7 commit `c3fc5670e` (legitimate adjacent hygiene, not fix-loop scope creep) |

Pre-existing `ResizeObserver is not defined` lines in test stderr are jsdom/`@xyflow/react` test-environment incompatibilities; they appear before this loop and after it, do not affect any test outcome, and are unrelated to any commit in the loop.

---

## 4. Operator-gate defaults (override at merge review)

Two punch-list entries required operator policy decisions, not code fixes. The orchestrator selected reasonable-call defaults consistent with the session instruction ("make the reasonable call and continue") and the active program context (memory `project_rc5ux_demo_prep_scope`). Override at merge review if either default is wrong.

### P3A-003 → option (c) chosen: keep all three migration shims, add rationale comments

- **Shims retained:** `RETIRED_SIDEBAR_COLLAPSED_KEY` cleanup in `App.tsx:28-32`; `SIDERAIL_WIDTH_KEY = "elspeth_inspector_width"` legacy prefix in `Layout.tsx:11-13`; redirect-toast machinery in `useHashRouter.ts:33-38` + `App.tsx:233-245`.
- **Why:** memory `project_staging_deployment` confirms `elspeth.foundryside.dev` is the operator's actively-used staging instance with real bookmarks; deletion of any of these three shims would silently break operator-as-user state. Option (c) satisfies CLAUDE.md's discoverability mandate by making the retention reason explicit in-source, citing the synthesis ID `P3A-003` and the operator-gated removal condition.
- **Override:** if strict CLAUDE.md "No Legacy Code" reading should win, delete all three shims in a follow-up branch; `git grep "P3A-003"` finds the three call sites instantly.

### P3A-008 → option (a) chosen: RunsView capabilities recorded as retired (not returning in 15b/3B)

- **Decision recorded in:** `docs/composer/ux-redesign-2026-05/15a2-phase-3a-removals-part-2.md:834-860` (new `###` Review-history entry).
- **Capabilities recorded as retired:** diagnostics accordion, polling-while-active, token states + artifacts on inspect, LLM explanation, failure_detail rendering, fan-out accounting, Inspect-button `aria-expanded`/`aria-controls`, suggestion-banner from SpecView, active-run indicator + inline-rename from SessionSidebar.
- **Why:** Phase 3A's stated direction is IA cleanup; the deleted capabilities were intentionally removed; the lighter `RunsHistoryDrawer` is the successor surface. The default for "we deleted this on purpose" is that it stays deleted unless an explicit need re-emerges.
- **Override:** if any capability is intended to return in 15b/3B, that phase plan must absorb it AND add a tests-rehoming sub-task (the synthesis preserves the missing-test list from Agent D Finding 5 for reuse).

---

## 5. Required operator action before merge

| Item | Action | Why |
|---|---|---|
| Runtime Alt-key smoke check | Open dev build (`npm run dev` in `src/elspeth/web/frontend`). With composer open: press Alt+1 → Graph tab activates; press Alt+2 → YAML tab activates. Optionally also press Alt+3 and Alt+4 → nothing happens (no silent error, no visible glitch). | Vitest covers the constant-coherence guard but not real keyboard behaviour. Per P3A-001 reviewer's runtime caveat. |
| Confirm or override P3A-003 default (option c) | Read §4 above; if the strict CLAUDE.md reading should win, file a follow-up issue to delete all three shims. | Operator-gate decision. |
| Confirm or override P3A-008 default (option a) | Read §4 above; if any RunsView capability is intended to return, name it in the 15b/3B plan and absorb the tests-rehoming sub-task. | Operator-gate decision. |
| Promote/dismiss deferred MINOR/NIT entries | Read §2 deferred-table; promote any worth filing as `filigree` issues, dismiss the rest. | Per CLAUDE.md "you do NOT use observations to finish work prematurely" — these are genuinely deferral-worthy, not hidden debt. |

---

## 6. Out-of-scope findings (surfaced for operator awareness)

These were detected during the review but predate or fall outside this fix loop. The orchestrator did NOT fix them.

### Dead `.spec-pending-proposal` / `.runs-pending-proposal` CSS

- **Location:** `src/elspeth/web/frontend/src/App.css:2881-2898`
- **Status:** Zero TSX consumers (`git grep` confirms); predates this branch (was not introduced by Phase 3A.5/3A.6 or this fix loop).
- **Source:** P3A-001 reviewer (out-of-scope observation, file:line cited explicitly).
- **Recommended disposition:** file as a separate observation/issue for a focused CSS cleanup pass; do NOT absorb into P3A-001 (it wasn't part of P3A-001's defect class and would obscure that commit's intent).

### Unstaged 15b doc edits in the worktree

- **Files:** `docs/composer/ux-redesign-2026-05/15b1-phase-3b-side-rail-part-1.md` (+51 lines), `15b2-phase-3b-side-rail-part-2.md` (+120 insertions / −50 deletions).
- **Status:** Predate this orchestration session — they were already present as unstaged working-tree modifications before P3A-001 ran. None of the five fix-loop commits touched them. Every writer dispatch was told to leave them alone.
- **Source:** Detected during P3A-002 writer's status check.
- **Recommended disposition:** operator handles in a separate workstream; the fix-loop history is clean of any cross-contamination.

---

## 7. Merge recommendation

**READY TO MERGE TO `RC5.2`** conditional on the four items in §5.

The branch:
- Closes the demo-visible behavioural regression (Alt+1/Alt+4 silent no-op).
- Adds a mechanical test guard so the defect class can't silently recur (this is a Meadows Level-6 "Information Flows" intervention — much higher leverage than a one-off fix).
- Documents two operator-gated retention decisions in-source with greppable IDs (`P3A-003`, `P3A-008`).
- Preserves staging-user state (operator's bookmarks survive).
- Passes all 672 vitest tests + typecheck.
- Leaves no scope creep beyond the original Phase 3A.7 hygiene already on `c3fc5670e`.

The 9 deferred entries are genuine deferrals (test-coverage belt-and-braces + doc-internal-contradiction + one usability regression on a non-demo path), not hidden debt — each is listed with file:line precision so the operator can promote what's worth tracking.

---

## 8. Orchestration trail (for audit)

| Artifact | Path |
|---|---|
| Solution-architect review (Agent A) | `.review/agent-a-report.md` |
| Systems-thinker review (Agent B) | `.review/agent-b-report.md` |
| Code-quality review (Agent C) | `.review/agent-c-report.md` |
| Quality/test review (Agent D) | `.review/agent-d-report.md` |
| Synthesis with reviewer-conflict resolutions | `.review/synthesis.md` |
| This final report | `.review/final-report.md` |

The `.review/` directory is currently untracked. The operator may commit it for audit traceability or leave it as a working-tree artifact; the orchestrator chose not to commit it unilaterally because adding tracked review artifacts to a branch is a policy decision.
