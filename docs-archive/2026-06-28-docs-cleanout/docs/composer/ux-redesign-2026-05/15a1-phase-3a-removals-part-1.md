# Phase 3A — IA Cleanup: side-rail scaffold, header session switcher, Spec/Runs removal, run-results inline (Part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the first half of Phase 3 — introduce the side-rail scaffold, header session switcher dropdown, inline run-results component, and remove the Spec tab + Runs tab + always-on session sidebar. Leaves the app launchable and green at every commit. Defers the graph mini-view, YAML export modal, Catalog button move, version-selector relocation, hash-router migration, and the final InspectorPanel deletion to **15b**.

**Architecture:** Frontend chrome refactor. No new trust boundary; no backend changes. The `useHashRouter` invariants (`VALID_TABS`, default-tab `"spec"`) are *preserved* in this plan and migrated in 15b — Phase 3A keeps `spec`/`runs`/`graph`/`yaml` valid hash fragments so that mid-phase deep links don't 404, and the InspectorPanel keeps rendering Graph + YAML inside the inspector until 15b extracts them.

**Tech Stack:** React + Zustand + Vitest + testing-library. No new dependencies.

> **Phase 3 block notice (added 2026-05-17; target corrected 2026-05-17):** This plan is one of four (15a1, 15a2, 15b1, 15b2) that together comprise the Phase 3 IA-cleanup work. **All four land as a single block on the dedicated Phase 3 worktree/branch for this IA-cleanup block** and **merge as one PR**. The canonical target for this packet is worktree `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on branch `feat/composer-phase-3-ia-cleanup`, created from `RC5.2` with `git worktree add .worktrees/composer-phase-3-ia-cleanup -b feat/composer-phase-3-ia-cleanup RC5.2` if it does not already exist. Do **not** use the old Phase 2A/2B/2C worktree or branch (`.worktrees/phase-2a-backend`, `feat/composer-phase-2a-backend`); those references are stale. Phrases below like "deferred to 15b" or "Phase 3B" mean "later tasks in the same Phase 3 branch," not a separate cycle. The 15a1→15a2→15b1→15b2 split is task sequencing and document organisation, not delivery sequencing — sequencing within the block still matters per task ordering.
>
> **Subagent dispatch discipline.** Every subagent prompt for this packet MUST start with this CWD-discipline preamble as its first Bash call: `cd /home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup && pwd && git rev-parse --abbrev-ref HEAD`; expected branch: `feat/composer-phase-3-ia-cleanup`. If the operator explicitly chooses a different Phase 3 worktree/branch, update this notice in **all four** 15a1/15a2/15b1/15b2 files before dispatch and use the chosen concrete values in every subagent prompt. The prompt must also state that `.worktrees/phase-2a-backend` and `feat/composer-phase-2a-backend` are stale Phase 2 targets and forbidden for Phase 3 work. Use absolute paths only thereafter for every Read/Bash/Grep. Bash `cd` does NOT persist between tool calls — relative paths can silently read the wrong branch.

**Sibling plans:**
- Predecessor: [13-phase-1b-frontend.md](13-phase-1b-frontend.md) — Phase 3A **assumes Phase 1B has shipped**, because Phase 1B adds the `UserMenu` to the header. If 1B has not shipped, do not start this plan — fix 1B first.
- Successor: [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md) / [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md) — Graph mini, YAML modal, Catalog button move, version selector relocation, hash redirects, InspectorPanel deletion.

**Part split:** This file covers Tasks 1–4 (additive work). Tasks 5–8 (the removal tasks) plus Risks, Memory references, and Review history are in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3) and §A (H1, H2 calls).

**Design spec:** [03-target-information-architecture.md](03-target-information-architecture.md).

---

## Scope boundaries

**In scope (this plan, 15a):**
- New `SideRail.tsx` scaffold mounted to the right of the chat column. Slots for: audit-readiness placeholder (**Phase 2C has already mounted `AuditReadinessPanel` inside `InspectorPanel` at line 534**, so this slot stays `null` in 15a; 15b2 Task 9 Step 4a migrates the mount from InspectorPanel into the slot before `InspectorPanel.tsx` is deleted), graph mini placeholder (filled by 15b), Catalog button placeholder (filled by 15b), Export-YAML button placeholder (filled by 15b), completion bar placeholder (filled by Phase 6). The side rail renders even when all real content is deferred — it carries the layout shape.
- New `HeaderSessionSwitcher.tsx` — header dropdown that lists sessions (`session-switcher` per design doc 03). Replaces the always-on `SessionSidebar`.
- New `InlineRunResults.tsx` — mounts in the chat column below `ChatPanel`; subscribes to `executionStore.activeRunId` + `progress`; renders `ProgressView` and `RunOutputsPanel` for the active/most-recent run. Carries the previous Runs-tab functionality minus the historical list.
- New `RunsHistoryDrawer.tsx` — preserves access to historical runs via a "Past runs" affordance in `InlineRunResults`. Required because the design doc is silent on the runs list; this plan resolves that as "keep, demote to a drawer" (see §"Open scope questions resolved" below).
- Removal of `SessionSidebar.tsx` import + render from `App.tsx` / `Layout.tsx`. The file itself is deleted.
- Removal of `SpecView.tsx` import + render from `InspectorPanel.tsx`. The file itself is deleted. `Spec` tab button is dropped from the tab strip.
- Removal of `RunsView.tsx` import + render from `InspectorPanel.tsx`. The file itself is deleted. `Runs` tab button is dropped from the tab strip.
- ~~Removal of the `Validate` button from the inspector header.~~ **Done by Phase 2C** (commits `d218417c1..2f2ba300e`, 2026-05-17). The button is gone; the `validationResult`→`injectSystemMessage`+`sendValidationFeedback` side effects are now owned by an `executionStore` subscriber inside `subscriptions.ts` (Phase 2C). `Ctrl+Shift+V` continues to fire `validate(activeSessionId)` directly from `App.tsx`, and the side-effect subscriber publishes the system message regardless of which trigger called validate. The validation banner is replaced in-place by the `AuditReadinessPanel` (also Phase 2C).
- Layout grid renamed: `sidebar / chat / inspector` slots become `chat / siderail` (the old `sidebar` slot is gone; the inspector slot is renamed to `siderail` but, importantly, the inspector continues to occupy that slot until 15b deletes it. We rename the prop in this plan so that the slot is *semantically* the side rail; the inspector lives in the side-rail slot temporarily).
- `App.test.tsx` smoke-render assertion runs at the end of every removal task to confirm the app still launches.
- `CommandPalette` retains its existing Sessions section (already covers H1 fallback per CommandPalette.tsx:182–196). No new commands added here.

**Out of scope (deferred to 15b):**
- Graph mini-view (persistent in side rail) and its full-view modal.
- YAML view replaced by "Export YAML" side-rail button + modal.
- Catalog button moved from inspector header to side rail (chrome change).
- Version selector relocated to header (next to session name).
- Validation dot relocated from inspector to audit-readiness panel (this is Phase 2's responsibility; 15b only deletes the inspector mounting point once Phase 2 has provided the new home).
- Hash-router `VALID_TABS` migration and redirect of stale `#/{id}/spec`/`runs` links.
- Deletion of `InspectorPanel.tsx` itself.
- `Alt+1/2/3/4` shortcut cleanup. Until 15b ships, these still navigate to the surviving graph/yaml tabs.
- Repurposing of `Ctrl+E` (execute) and `SWITCH_TAB_EVENT` event semantics. Both survive 15a unchanged.

**Out of scope (other phases):**
- Audit-readiness panel content (Phase 2 fills the side-rail slot).
- Completion bar verbs Save-for-review / Run pipeline (Phase 6 fills the side-rail slot).
- Catalog drawer reshape (Phase 7).
- First-run tutorial (Phase 4).
- Mode-related layout changes beyond what Phase 1B provided.

## Trust tier check

Phase 3A is **frontend chrome only**. There are no new trust boundaries:

- No new external-data ingestion. The existing `executionStore.activeRunId` / `progress` / diagnostics maps already pass through Tier 3 validation at the backend boundary; this plan only relocates where they render.
- No new audit-recorder events. The validation-on-change effect (Task 4) uses the existing `executionStore.validate(sessionId)` action, which already routes through the audit boundary at the backend.
- No new persistent state. The component re-organization changes nothing on disk.

Per [CLAUDE.md](../../../CLAUDE.md) "Defensive Programming: Forbidden", this plan does **not** introduce `try`/`catch` around store calls "to be safe." The existing patterns (e.g., Phase 2C deleted the historical `handleValidate` try/catch when it moved the side-effect orchestration into `subscriptions.ts`; the surviving subscriber accesses store fields directly per CLAUDE.md offensive-programming) are preserved as-is. New components access typed store fields directly.

## Sequencing and dependencies

15a tasks are ordered so that **every commit leaves the app in a state where**:

1. `npm test` passes (vitest + the existing testing-library suite).
2. `App.test.tsx`'s smoke render succeeds.
3. A human opening `elspeth.foundryside.dev` (`project_staging_deployment`) can compose a pipeline using whichever surfaces survive at that point.

The order is:

```
Task 1 — Add InlineRunResults + RunsHistoryDrawer  (additions; nothing removed)
Task 2 — Add SideRail scaffold + rename Layout slot (additions; nothing removed)
Task 3 — Add HeaderSessionSwitcher                  (additions; nothing removed)
Task 4 — Add auto-validate-on-change subscriber     (additive change to subscriptions.ts; Phase 2C subscribers untouched)
Task 5 — Remove Runs tab + RunsView + delete file   (first removal; runs handled by Task 1)
Task 6 — Remove Spec tab + SpecView + delete file   (second removal; the Spec→banner-click flow is dropped — see risks)
Task 7 — Remove SessionSidebar mount + delete file  (third removal; session switching handled by Task 3 + palette)
                                                    (Task 8 retired — Phase 2C already removed the Validate button and
                                                     wired the validationResult→system-message side effects in
                                                     subscriptions.ts. See 15a2 panel note.)
```

Tasks 1–4 are in this file. Tasks 5–7 are in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).

Each task is TDD-shaped: failing test, implementation, passing test, smoke render, commit.

## Open scope questions resolved by this plan

1. **Auto-validate on composition change vs explicit Validate button.** Resolution: **auto-validate**. The audit-readiness panel from Phase 2 will be the indicator surface, but Phase 2 has no Validate button either by design ("the indicator already does the work" — design doc 03 table row "Validate button"). To bridge Phase 3 → Phase 2 without losing validation entirely, Phase 3A adds an `executionStore` cross-store subscription that fires `validate(sessionId)` whenever `compositionState.version` increments (debounced, deduplicated). `Ctrl+Shift+V` survives as a manual re-trigger that doesn't depend on a button. See Task 4.
2. **Where Execute lives between Phase 3 and Phase 6.** Resolution: **inspector header Execute button stays through 15a and 15b**. The "Execute moves to completion bar" is **Phase 6**; Phase 3 does *not* delete it. The design doc lists Execute → completion bar as a Phase 6 row, not Phase 3. The side-rail scaffold in Task 2 reserves a `completion-bar` slot but renders nothing there until Phase 6 fills it. This is explicitly called out so executors don't preemptively remove Execute.
3. **Historical runs list.** Resolution: **keep, demoted to a drawer** (`RunsHistoryDrawer`). Decision recorded 2026-05-17 (Section A pass): the audit-focused persona Linda (see `project_composer_personas` memory) benefits from past-runs access during audit review; the drawer is self-contained (one component + test) so deletion is cheap in Phase 8 if no production demand emerges. The drawer is opened by a "Past runs" button in `InlineRunResults`. The design doc is silent on the runs list; this plan resolves that silence as "preserve via drawer." If, after Phase 8, telemetry or operator feedback shows no use, delete the drawer in a one-commit follow-up.
4. **`SWITCH_TAB_EVENT` / `TAB_CHANGED_EVENT` semantics.** Resolution: **survive 15a unchanged**. Both events still fire for `spec`/`graph`/`yaml`/`runs` tab navigation against the inspector. 15b extracts Graph and YAML into modal-open events and decides whether the constants live on as `OPEN_GRAPH_MODAL_EVENT` / `OPEN_YAML_MODAL_EVENT` or get deleted.
5. **CommandPalette Sessions section as H1 fallback.** Resolution: **already covered**. CommandPalette.tsx:182–196 lists recent sessions (up to 10) under a "Sessions" group. The H1 recommended call (header dropdown + Cmd palette) is structurally complete the moment Task 3 lands; no palette code changes in 15a. Tab-switch palette commands (`tab-spec`, `tab-graph`, etc.) survive 15a; 15b prunes them.
6. **SpecView click-navigation from validation banner.** Resolution: **drop the navigation; banner becomes click-through-noop temporarily**. InspectorPanel.tsx:448–458 selects a node and switches to Spec tab on validation-banner click. With Spec gone (Task 6), the navigation has no destination. Phase 2's audit-readiness "Explain" surface is the future home of this routing. Task 6 changes `handleValidationComponentClick` to a noop (`selectNode` still fires for the GraphView highlight, but no tab switch). Documented as a Phase-3-to-Phase-2 handoff in §Risks.
7. **Header layout where session switcher and UserMenu coexist.** Resolution: **Phase 1B's UserMenu is top-right; HeaderSessionSwitcher is top-left next to the ELSPETH brand**. Both live in a new top-level `AppHeader.tsx` introduced by Task 3, which Phase 1B's `UserMenu` is *moved into* (it's currently floating with no proper header — Phase 1B Task 6 noted that the header structure was soft). Task 3 includes the small refactor that gives `UserMenu` a permanent home.

---

## Task 1: Add `InlineRunResults` + `RunsHistoryDrawer` (additive)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/execution/InlineRunResults.tsx`.
- Create: `src/elspeth/web/frontend/src/components/execution/InlineRunResults.test.tsx`.
- Create: `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx`.
- Create: `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` — mount `<InlineRunResults />` after the chat scrollback (and before the chat input). Do not inline the component logic into ChatPanel (622-line component, already complex).

This task is **purely additive**. After Task 1, run-result inline rendering co-exists with the still-present Runs tab. Task 5 removes the tab once the inline path is verified.

- [ ] **Step 1: Inspect the run-result render shape**

Open `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` and identify the JSX that renders the active run (lines 280–530-ish). The relevant subset is:
- The status badge for the active run.
- The `<ProgressView />` for the active run.
- The `<RunOutputsPanel runId={...} />` for completed runs.
- The accounting summary (source counts, token counts, durations).

`InlineRunResults` shows only **the active or most recent run** — not the full list. The full list moves to `RunsHistoryDrawer`.

- [ ] **Step 2: Write the failing test for `InlineRunResults`**

Create `src/elspeth/web/frontend/src/components/execution/InlineRunResults.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { InlineRunResults } from "./InlineRunResults";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

// ProgressView and RunOutputsPanel have heavy dependencies (websocket / blob
// fetches). Stub them so InlineRunResults can be unit-tested in isolation.
vi.mock("@/components/execution/ProgressView", () => ({
  ProgressView: () => <div data-testid="progress-view-stub" />,
}));
vi.mock("@/components/inspector/RunOutputsPanel", () => ({
  RunOutputsPanel: ({ runId }: { runId: string }) => (
    <div data-testid="run-outputs-stub" data-run-id={runId} />
  ),
}));

describe("InlineRunResults", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      runs: [],
      activeRunId: null,
      progress: null,
      diagnosticsByRunId: {},
      diagnosticsLoadingByRunId: {},
      diagnosticsEvaluatingByRunId: {},
      diagnosticsErrorByRunId: {},
      diagnosticsExplanationByRunId: {},
      diagnosticsWorkingViewByRunId: {},
      validationResult: null,
      pendingFanoutGuard: null,
      pendingFanoutSessionId: null,
      isValidating: false,
      isExecuting: false,
      wsDisconnected: false,
      error: null,
    } as never);
    useSessionStore.setState({
      activeSessionId: "sess-1",
    } as never);
  });

  it("renders nothing when there are no runs", () => {
    const { container } = render(<InlineRunResults />);
    // Empty result is rendered as a single empty container or null; either
    // way nothing run-shaped appears.
    expect(container.querySelector("[data-testid='progress-view-stub']")).toBeNull();
    expect(container.querySelector("[data-testid='run-outputs-stub']")).toBeNull();
  });

  it("renders ProgressView for an active running run", () => {
    useExecutionStore.setState({
      activeRunId: "run-A",
      progress: {
        run_id: "run-A",
        status: "running",
        // ...minimum RunProgress fields the badge logic reads
      } as never,
      runs: [{ id: "run-A", status: "running" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("progress-view-stub")).toBeInTheDocument();
  });

  it("renders RunOutputsPanel for a terminal active run", () => {
    useExecutionStore.setState({
      activeRunId: "run-B",
      progress: {
        run_id: "run-B",
        status: "completed",
      } as never,
      runs: [{ id: "run-B", status: "completed" } as never],
    } as never);
    render(<InlineRunResults />);
    expect(screen.getByTestId("run-outputs-stub")).toHaveAttribute(
      "data-run-id",
      "run-B",
    );
  });

  it("exposes a 'Past runs' button when historical runs exist", () => {
    useExecutionStore.setState({
      activeRunId: null,
      runs: [
        { id: "run-old-1", status: "completed" } as never,
        { id: "run-old-2", status: "completed" } as never,
      ],
    } as never);
    render(<InlineRunResults />);
    expect(
      screen.getByRole("button", { name: /past runs/i }),
    ).toBeInTheDocument();
  });

  it("hides the 'Past runs' button when no historical runs exist", () => {
    useExecutionStore.setState({
      activeRunId: null,
      runs: [],
    } as never);
    render(<InlineRunResults />);
    expect(
      screen.queryByRole("button", { name: /past runs/i }),
    ).not.toBeInTheDocument();
  });

  it("handles a dangling activeRunId (race during reconnect / session switch)", () => {
    // Contract: activeRunId can briefly point to a run that is not yet (or
    // no longer) in `runs[]` during WebSocket reconnect or session switch.
    // The component must not crash. The intended shape: progress is null in
    // this state, so hasActive is false and nothing run-shaped renders.
    useExecutionStore.setState({
      activeRunId: "run-ghost",
      progress: null,
      runs: [],
    } as never);
    const { container } = render(<InlineRunResults />);
    expect(container.querySelector("[data-testid='progress-view-stub']")).toBeNull();
    expect(container.querySelector("[data-testid='run-outputs-stub']")).toBeNull();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/execution/InlineRunResults.test.tsx
```

Expected: FAIL — module not found.

- [ ] **Step 4: Implement `InlineRunResults`**

Create `src/elspeth/web/frontend/src/components/execution/InlineRunResults.tsx`:

```typescript
// ============================================================================
// InlineRunResults
//
// Mounts in the chat column (after the chat scrollback, before the chat input)
// and renders the active/most-recent run's progress + outputs. Carries the
// active-run subset of the old RunsView; the historical list lives in
// RunsHistoryDrawer reachable via the "Past runs" button.
//
// Subscribes to executionStore: activeRunId, progress, runs (for the
// past-runs button's existence check).
// ============================================================================

import { useState } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { ProgressView } from "@/components/execution/ProgressView";
import { RunOutputsPanel } from "@/components/inspector/RunOutputsPanel";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";
import { isTerminalRunStatus } from "@/types/index";

export function InlineRunResults(): JSX.Element | null {
  const activeRunId = useExecutionStore((s) => s.activeRunId);
  const progress = useExecutionStore((s) => s.progress);
  const runs = useExecutionStore((s) => s.runs);

  const [showHistory, setShowHistory] = useState(false);

  const hasActive = activeRunId !== null && progress !== null;
  const isTerminal = hasActive && isTerminalRunStatus(progress.status);
  const hasHistory = runs.length > 0 || hasActive;

  if (!hasActive && !hasHistory) {
    return null;
  }

  return (
    <section
      className="inline-run-results"
      aria-label="Pipeline run results"
    >
      {/* Live or just-finished run. The two distinct sub-components mirror
          the RunsView split: ProgressView for active runs, RunOutputsPanel
          for the immutable manifest of a finished run. */}
      {hasActive && !isTerminal && <ProgressView />}
      {hasActive && isTerminal && activeRunId && (
        <RunOutputsPanel runId={activeRunId} />
      )}

      {/* "Past runs" affordance — opens a drawer with the historical list.
          Per design-spec resolution: historical access is preserved, not
          inlined into the main column. */}
      {hasHistory && (
        <div className="inline-run-results-history-cta">
          <button
            type="button"
            onClick={() => setShowHistory(true)}
            className="btn"
          >
            Past runs ({runs.length})
          </button>
        </div>
      )}

      {showHistory && (
        <RunsHistoryDrawer onClose={() => setShowHistory(false)} />
      )}
    </section>
  );
}
```

- [ ] **Step 5: Write the failing test for `RunsHistoryDrawer`**

Create `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunsHistoryDrawer } from "./RunsHistoryDrawer";
import { useExecutionStore } from "@/stores/executionStore";

describe("RunsHistoryDrawer", () => {
  beforeEach(() => {
    useExecutionStore.setState({
      runs: [
        { id: "r1", status: "completed" } as never,
        { id: "r2", status: "failed" } as never,
      ],
      activeRunId: null,
      progress: null,
    } as never);
  });

  it("lists every run from the store", () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/r1/)).toBeInTheDocument();
    expect(screen.getByText(/r2/)).toBeInTheDocument();
  });

  it("calls onClose when the Close button is clicked", async () => {
    const onClose = vi.fn();
    render(<RunsHistoryDrawer onClose={onClose} />);
    await userEvent.click(screen.getByRole("button", { name: /close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("calls onClose when Escape is pressed", async () => {
    const onClose = vi.fn();
    render(<RunsHistoryDrawer onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("renders 'No prior runs' when the runs list is empty", () => {
    useExecutionStore.setState({ runs: [] } as never);
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByText(/no prior runs/i)).toBeInTheDocument();
  });

  it("moves focus into the drawer on open (Close button receives focus)", () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    expect(screen.getByRole("button", { name: /close/i })).toHaveFocus();
  });

  it("traps Tab and Shift+Tab inside the drawer", async () => {
    render(<RunsHistoryDrawer onClose={vi.fn()} />);
    const closeBtn = screen.getByRole("button", { name: /close/i });
    closeBtn.focus();
    // Forward Tab from the last focusable cycles back to the first.
    // The drawer's only focusable surface in this minimal shape is the
    // Close button; Tab should re-focus it rather than leaving the drawer.
    await userEvent.tab();
    expect(closeBtn).toHaveFocus();
    // Shift+Tab from the first focusable cycles to the last (same button).
    await userEvent.tab({ shift: true });
    expect(closeBtn).toHaveFocus();
  });
});
```

- [ ] **Step 6: Implement `RunsHistoryDrawer`**

**Focus-trap mechanism.** `role="dialog" aria-modal="true"` obligates a focus trap. This drawer implements one manually (no `inert` polyfill): on mount, focus the Close button; on Tab / Shift+Tab, cycle focus among the drawer's focusable descendants; Escape calls `onClose`. Restoration of focus to the trigger after `onClose` is the caller's responsibility (the `InlineRunResults` "Past runs" button receives focus by default because React re-renders the button after `setShowHistory(false)`).

Create `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx`:

```typescript
// ============================================================================
// RunsHistoryDrawer
//
// Slide-over drawer listing every run for the current session. Opened from
// InlineRunResults' "Past runs" button. Preserves audit-trail access to old
// runs after the inspector Runs tab is removed (Phase 3A Task 5).
//
// A11y contract: role="dialog" aria-modal="true" obligates a focus trap.
// We implement it manually (no inert polyfill): on mount, focus the Close
// button; on Tab / Shift+Tab, cycle focus within the drawer's focusable
// descendants. Escape calls onClose. Caller is responsible for restoring
// focus to the trigger after onClose.
//
// Minimal shape in this plan — list of run-id + status badge + duration.
// Phase 8 polish may expand to include accounting summaries.
// ============================================================================

import { useEffect, useRef } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

interface RunsHistoryDrawerProps {
  onClose: () => void;
}

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function RunsHistoryDrawer({ onClose }: RunsHistoryDrawerProps): JSX.Element {
  const runs = useExecutionStore((s) => s.runs);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const drawerRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);

  // Move focus into the drawer on open. Per WAI-ARIA Authoring Practices,
  // a role="dialog" aria-modal="true" element must receive focus when shown.
  useEffect(() => {
    closeBtnRef.current?.focus();
  }, []);

  // Focus trap: cycle Tab / Shift+Tab among focusable descendants.
  useEffect(() => {
    function handle(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const drawer = drawerRef.current;
      if (!drawer) return;
      const focusables = drawer.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && (active === first || !drawer.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !drawer.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [onClose]);

  return (
    <div
      ref={drawerRef}
      role="dialog"
      aria-modal="true"
      aria-label="Past pipeline runs"
      className="runs-history-drawer"
    >
      <header className="runs-history-drawer-header">
        <h2>Past runs</h2>
        <button
          ref={closeBtnRef}
          type="button"
          aria-label="Close past runs"
          onClick={onClose}
          className="btn"
        >
          Close
        </button>
      </header>
      <div className="runs-history-drawer-body">
        {runs.length === 0 ? (
          <p>No prior runs for session {activeSessionId ?? "(none)"}.</p>
        ) : (
          <ul className="runs-history-list">
            {runs.map((run) => (
              <li key={run.id} className="runs-history-item">
                <span className="runs-history-item-id">{run.id}</span>
                <span className="runs-history-item-status">{run.status}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Mount `<InlineRunResults />` in `ChatPanel.tsx`**

Read `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` and identify the JSX region between the chat scrollback and the chat input (the `data-chat-input` textarea). Add:

```tsx
import { InlineRunResults } from "@/components/execution/InlineRunResults";

// ...somewhere between scrollback and input:
<InlineRunResults />
```

The component returns `null` when there are no runs, so it's safe to mount unconditionally.

- [ ] **Step 8: Run all tests and the smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/execution src/components/chat src/App.test.tsx
```

Expected: PASS. The pre-existing Runs tab inside the inspector still works; `InlineRunResults` runs in parallel.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/web/frontend/src/components/execution/InlineRunResults.tsx \
        src/elspeth/web/frontend/src/components/execution/InlineRunResults.test.tsx \
        src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx \
        src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx
git commit -m "feat(web/frontend): InlineRunResults + RunsHistoryDrawer (Phase 3A.1)"
```

---

## Task 2: Add `SideRail` scaffold + rename `Layout` slot

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/SideRail.tsx`.
- Create: `src/elspeth/web/frontend/src/components/common/SideRail.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx` — rename the `inspector` prop to `siderail` in the same commit (per CLAUDE.md "No Legacy Code Policy" — no deprecated alias; the only caller is `App.tsx`, also modified in this task). Adjust internal class names from `layout-inspector` → `layout-siderail` and `inspector-overlay-*` → `siderail-overlay-*`.
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.test.tsx` — update slot-name assertions.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — pass `siderail` prop (carrying the InspectorPanel inside the new SideRail for now, so nothing visually moves).
- Modify: `src/elspeth/web/frontend/src/App.test.tsx` — the Layout mock at lines 19–35 destructures `{ sidebar, chat, inspector }`. **Rename `inspector` → `siderail` in the destructure AND in the render body.** Without this, React will silently drop the unrecognised `siderail` prop and the mocked Layout will render no side-rail content — App.test.tsx will stay green while `App.tsx` is broken (Quality panel finding 2026-05-17).

The side rail in 15a is **a component that accepts named slot props** (render-props). Slots are declared as `ReactNode | null` props (e.g., `auditReadinessSlot`, `executeButtonSlot`, `graphMiniSlot`, etc.). The scaffold renders each slot in a wrapper `<div>` with a `data-testid`. Each downstream phase passes its component **as a prop from the mounting site** — currently `App.tsx`. Components are NEVER mounted by event or by SideRail's own internal logic; the composition contract is: **caller passes content, SideRail places it**.

Empty slots (props left `null`) render as zero-height invisible markers. The current `InspectorPanel` becomes the value of the `children` prop temporarily — visually nothing moves except that the prop name and class names rename.

> **Review finding (IMPORTANT):** Slots are render-props, not empty divs with internal mounts. The 15a scaffold adds the `executeButton` slot (required by 15b Task 2 before the inspector is deleted). The 15b1 Task 2 amendment below fills that slot.

- [ ] **Step 1: Write the failing test for `SideRail`**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SideRail } from "./SideRail";

describe("SideRail", () => {
  it("renders the audit-readiness slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-audit-readiness"),
    ).toBeInTheDocument();
  });

  it("renders the graph mini slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-graph-mini"),
    ).toBeInTheDocument();
  });

  it("renders the catalog slot region", () => {
    render(<SideRail />);
    expect(screen.getByTestId("siderail-slot-catalog")).toBeInTheDocument();
  });

  it("renders the export-yaml slot region", () => {
    render(<SideRail />);
    expect(screen.getByTestId("siderail-slot-export-yaml")).toBeInTheDocument();
  });

  it("renders the execute-button slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-execute-button"),
    ).toBeInTheDocument();
  });

  it("renders the completion-bar slot region", () => {
    render(<SideRail />);
    expect(
      screen.getByTestId("siderail-slot-completion-bar"),
    ).toBeInTheDocument();
  });

  it("renders content passed via the executeButton slot prop", () => {
    render(
      <SideRail executeButtonSlot={<button>Run</button>} />,
    );
    expect(screen.getByRole("button", { name: /run/i })).toBeInTheDocument();
  });

  it("renders children passed through (the transitional inspector mount)", () => {
    render(
      <SideRail>
        <div data-testid="children-marker" />
      </SideRail>,
    );
    expect(screen.getByTestId("children-marker")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/SideRail.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Implement `SideRail`**

```typescript
// ============================================================================
// SideRail
//
// Right-column scaffold that hosts (top-to-bottom):
//   1. AUDIT-READINESS slot (Phase 2 fills via auditReadinessSlot prop)
//   2. GRAPH MINI slot (15b fills via graphMiniSlot prop)
//   3. CATALOG button slot (15b fills via catalogSlot prop)
//   4. EXPORT-YAML button slot (15b fills via exportYamlSlot prop)
//   5. EXECUTE-BUTTON slot (15b Task 2 fills via executeButtonSlot prop;
//      Phase 6 replaces with the completion bar)
//   6. (transitional) children — currently <InspectorPanel /> until 15b
//      deletes the inspector.
//   7. COMPLETION-BAR slot (Phase 6 fills via completionBarSlot prop)
//
// Composition contract: slots are render-props (ReactNode | null).  The
// caller (App.tsx) passes components as props; SideRail places them.
// Components are NEVER mounted by SideRail's own internal logic or by
// event — always via props from the mount site.  Empty slots render as
// zero-height invisible markers (visually no gap).
// ============================================================================

import { type ReactNode } from "react";

interface SideRailProps {
  auditReadinessSlot?: ReactNode | null;
  graphMiniSlot?: ReactNode | null;
  catalogSlot?: ReactNode | null;
  exportYamlSlot?: ReactNode | null;
  executeButtonSlot?: ReactNode | null;
  completionBarSlot?: ReactNode | null;
  children?: ReactNode;
}

export function SideRail({
  auditReadinessSlot = null,
  graphMiniSlot = null,
  catalogSlot = null,
  exportYamlSlot = null,
  executeButtonSlot = null,
  completionBarSlot = null,
  children,
}: SideRailProps): JSX.Element {
  return (
    <aside className="side-rail" aria-label="Composer side rail">
      <div data-testid="siderail-slot-audit-readiness" className="side-rail-slot">
        {auditReadinessSlot}
      </div>
      <div data-testid="siderail-slot-graph-mini" className="side-rail-slot">
        {graphMiniSlot}
      </div>
      <div data-testid="siderail-slot-catalog" className="side-rail-slot">
        {catalogSlot}
      </div>
      <div data-testid="siderail-slot-export-yaml" className="side-rail-slot">
        {exportYamlSlot}
      </div>
      <div data-testid="siderail-slot-execute-button" className="side-rail-slot">
        {executeButtonSlot}
      </div>
      {/* Transitional: current InspectorPanel is passed as children while
          15b is extracting graph + yaml into modals. Once 15b deletes
          InspectorPanel, `children` will be removed from this prop. */}
      {children && <div className="side-rail-transitional">{children}</div>}
      <div data-testid="siderail-slot-completion-bar" className="side-rail-slot">
        {completionBarSlot}
      </div>
    </aside>
  );
}
```

- [ ] **Step 4: Rename `Layout` prop and class names**

Open `src/elspeth/web/frontend/src/components/common/Layout.tsx`. Change `LayoutProps`:

```typescript
interface LayoutProps {
  sidebar: ReactNode;
  chat: ReactNode;
  // Renamed: inspector → siderail. This slot is now the side-rail column;
  // the inspector lives temporarily inside it (15b removes the inspector).
  siderail: ReactNode;
}
```

Update the destructure and references:

```typescript
export function Layout({ sidebar, chat, siderail }: LayoutProps) {
  // ...

  // Rename class names: layout-inspector → layout-siderail, inspector-overlay-* →
  // siderail-overlay-*. Update App.css accordingly (search for "layout-inspector"
  // / "inspector-overlay" / "inspector-toggle" and apply the rename).
```

A search-and-replace inside `Layout.tsx`:
- `layout-inspector` → `layout-siderail`
- `inspector-overlay-backdrop` → `siderail-overlay-backdrop`
- `inspector-overlay-close` → `siderail-overlay-close`
- `inspectorWidth` (state and constants) → `siderailWidth` (refactor; the localStorage key `elspeth_inspector_width` is **kept** so existing user width preferences survive; only the in-memory identifier renames).
- `INSPECTOR_WIDTH_KEY` constant name → leave the key string `"elspeth_inspector_width"` unchanged for storage compatibility, but rename the JS identifier to `SIDERAIL_WIDTH_KEY` if preferred. Optional.
- `setInspectorVisible` / `inspectorVisible` → `setSideRailVisible` / `sideRailVisible`.
- The `inspector-toggle-btn` className **stays** for now (Phase 8 polish can rename it). Just update the prop names.

Also update `Layout.test.tsx`:

```typescript
// Before:
render(<Layout sidebar={...} chat={...} inspector={...} />);
// After:
render(<Layout sidebar={...} chat={...} siderail={...} />);
```

Update any test ids it relied on: search the test file for `inspector` and update to `siderail` consistently.

Run the Layout test suite:

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/Layout.test.tsx
```

Expected: PASS.

Also append the following `it(...)` block to `Layout.test.tsx` to lock in the localStorage key-string preservation:

```typescript
it("reads side-rail width from the pre-rename elspeth_inspector_width localStorage key", () => {
  // The key string is intentionally preserved across the inspector→siderail
  // rename so existing users' width settings survive. This test locks that
  // in: if a future search-replace accidentally renames the key string, the
  // assertion fails.
  localStorage.setItem("elspeth_inspector_width", "420");
  const { container } = render(
    <Layout chat={<div data-testid="chat" />} siderail={<div data-testid="siderail" />} />,
  );
  const layoutNode = container.querySelector(".app-layout") as HTMLElement;
  expect(layoutNode.style.gridTemplateColumns).toContain("420px");
  localStorage.removeItem("elspeth_inspector_width");
});
```

- [ ] **Step 5: Wire `App.tsx` to pass `siderail={<SideRail>…</SideRail>}`**

Replace the Layout render in `App.tsx`. All slots start as `null`; 15b fills them via their named props:

```tsx
import { SideRail } from "./components/common/SideRail";

// ...inside the return:
<Layout
  sidebar={<SessionSidebar />}
  chat={<ChatPanel onOpenSecrets={openSecrets} />}
  siderail={
    <SideRail
      auditReadinessSlot={null}   // Phase 2 fills
      graphMiniSlot={null}        // 15b Task 3 fills via <GraphMiniView />
      catalogSlot={null}          // 15b Task 5 fills via <CatalogButton />
      exportYamlSlot={null}       // 15b Task 4 fills via <ExportYamlButton />
      executeButtonSlot={null}    // 15b Task 2 fills via <ExecuteButton />
      completionBarSlot={null}    // Phase 6 fills
    >
      <InspectorPanel />
    </SideRail>
  }
/>
```

The `InspectorPanel` becomes a child of `SideRail` for now; all slot props are `null`, so the named slot divs render as zero-height invisible markers.

- [ ] **Step 6: Add a CSS class for `.side-rail` (and stub the slot styling)**

Append to `src/elspeth/web/frontend/src/App.css` (or wherever existing component styles live):

```css
.side-rail {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.side-rail-slot:empty {
  display: none;
}

.side-rail-transitional {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
}
```

(The empty-collapse rule ensures slots that Phases 2/6/15b haven't filled yet don't claim layout space.)

- [ ] **Step 7: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS — including `App.test.tsx`'s smoke render.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/SideRail.tsx \
        src/elspeth/web/frontend/src/components/common/SideRail.test.tsx \
        src/elspeth/web/frontend/src/components/common/Layout.tsx \
        src/elspeth/web/frontend/src/components/common/Layout.test.tsx \
        src/elspeth/web/frontend/src/App.tsx \
        src/elspeth/web/frontend/src/App.css
git commit -m "feat(web/frontend): SideRail scaffold + rename Layout slot (Phase 3A.2)"
```

---

## Task 3: Add `HeaderSessionSwitcher` + new `AppHeader`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/AppHeader.tsx`.
- Create: `src/elspeth/web/frontend/src/components/common/AppHeader.test.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — mount `<AppHeader />` above `<Layout />`.
- Modify: `src/elspeth/web/frontend/src/components/common/UserMenu.tsx` — **no internal change**, just confirm it accepts being mounted in `AppHeader` (it should — it's already a self-contained component from Phase 1B).

The session-switcher dropdown lists every session and lets the user switch. It also exposes the "New session" verb. Together with `UserMenu` (top-right) and the `ELSPETH` brand (top-left), it forms the new `AppHeader`.

Phase 1B's existing `UserMenu` is moved from its current `Layout` mounting point into `AppHeader`. Where Layout currently renders it (top-right of the sidebar toolbar — Layout.tsx:248–263 has the theme toggle, not UserMenu; UserMenu's actual location depends on what Phase 1B Task 6 shipped) — if Phase 1B mounted `UserMenu` in `Layout`, move it out to `AppHeader`.

- [ ] **Step 1: Inspect Phase 1B's UserMenu mount point**

```bash
grep -n "UserMenu" src/elspeth/web/frontend/src/components/common/Layout.tsx \
                    src/elspeth/web/frontend/src/App.tsx 2>/dev/null
```

Determine: did Phase 1B mount `UserMenu` in `Layout` or directly in `App.tsx`? Either way, Task 3 *moves* it (or its callsite) into `AppHeader`.

- [ ] **Step 2: Write the failing test for `HeaderSessionSwitcher`**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HeaderSessionSwitcher } from "./HeaderSessionSwitcher";
import { useSessionStore } from "@/stores/sessionStore";

describe("HeaderSessionSwitcher", () => {
  beforeEach(() => {
    useSessionStore.setState({
      sessions: [
        { id: "s1", title: "First", updated_at: "2026-05-15T00:00:00Z" } as never,
        { id: "s2", title: "Second", updated_at: "2026-05-14T00:00:00Z" } as never,
      ],
      activeSessionId: "s1",
    } as never);
  });

  it("shows the active session title as the trigger label", () => {
    render(<HeaderSessionSwitcher />);
    expect(
      screen.getByRole("button", { name: /first/i }),
    ).toBeInTheDocument();
  });

  it("opens a menu of all sessions when clicked", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    expect(screen.getByRole("menuitem", { name: /first/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /second/i })).toBeInTheDocument();
  });

  it("calls selectSession when a menu item is clicked", async () => {
    const selectSession = vi.fn();
    useSessionStore.setState({ selectSession } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /second/i }));
    expect(selectSession).toHaveBeenCalledWith("s2");
  });

  it("offers a 'New session' verb at the top of the dropdown", async () => {
    const createSession = vi.fn();
    useSessionStore.setState({ createSession } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /new session/i }));
    expect(createSession).toHaveBeenCalled();
  });

  it("closes on Escape", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    expect(
      screen.getByRole("menuitem", { name: /second/i }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(
      screen.queryByRole("menuitem", { name: /second/i }),
    ).not.toBeInTheDocument();
  });

  it("renders 'untitled' fallback when no session is active", () => {
    useSessionStore.setState({ activeSessionId: null } as never);
    render(<HeaderSessionSwitcher />);
    expect(screen.getByRole("button")).toHaveTextContent(/untitled/i);
  });

  it("shows 'New session' even when the sessions list is empty (just-archived edge)", async () => {
    // Contract: the dropdown must always offer the New-session verb, even
    // when activeSessionId points to a session that has been removed from
    // the sessions array (e.g. just archived) and the list is empty.
    useSessionStore.setState({
      sessions: [],
      activeSessionId: "sess-orphaned",
    } as never);
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("menuitem", { name: /new session/i })).toBeInTheDocument();
  });

  it("ArrowDown moves focus through menu items", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    const items = screen.getAllByRole("menuitem");
    // After open, focus should be on the first menuitem ("New session").
    expect(items[0]).toHaveFocus();
    await userEvent.keyboard("{ArrowDown}");
    expect(items[1]).toHaveFocus();
    await userEvent.keyboard("{ArrowDown}");
    expect(items[2]).toHaveFocus();
  });

  it("ArrowUp from the first item wraps to the last", async () => {
    render(<HeaderSessionSwitcher />);
    await userEvent.click(screen.getByRole("button", { name: /first/i }));
    const items = screen.getAllByRole("menuitem");
    expect(items[0]).toHaveFocus();
    await userEvent.keyboard("{ArrowUp}");
    expect(items[items.length - 1]).toHaveFocus();
  });

  it("Tab from inside the menu closes it and returns focus to the trigger", async () => {
    render(<HeaderSessionSwitcher />);
    const trigger = screen.getByRole("button", { name: /first/i });
    await userEvent.click(trigger);
    expect(screen.getByRole("menu")).toBeInTheDocument();
    await userEvent.tab();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/sessions/HeaderSessionSwitcher.test.tsx
```

Expected: FAIL — module not found.

- [ ] **Step 4: Implement `HeaderSessionSwitcher`**

```typescript
// ============================================================================
// HeaderSessionSwitcher
//
// Top-left header dropdown listing every session, plus a "New session" verb.
// Replaces the always-on SessionSidebar. Per design doc 03 §"Layout sketch"
// (▾ Session: cool-government-pages-1) — the active session title is the
// trigger label; the dropdown shows the full list.
//
// A11y contract: aria-haspopup="menu" + aria-controls links the trigger to
// the menu by id. Menu items use roving tabindex (one item tabbable at a
// time) so ArrowUp/ArrowDown move focus per WAI-ARIA Authoring Practices
// for menu widgets. Tab closes the menu and returns focus to the trigger
// (the menu does NOT participate in the page tab order).
//
// H1 resolution: this + CommandPalette Sessions section together cover the
// header-dropdown + Cmd-palette path described in 00-implementation-roadmap.md
// §A row H1.
// ============================================================================

import { useState, useRef, useEffect, useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";

const MENU_ID = "header-session-switcher-menu";

export function HeaderSessionSwitcher(): JSX.Element {
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const selectSession = useSessionStore((s) => s.selectSession);
  const createSession = useSessionStore((s) => s.createSession);

  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLLIElement | null)[]>([]);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const triggerLabel = activeSession?.title ?? "Untitled";

  // The menu always offers "New session" as item 0, followed by the session list.
  const itemCount = 1 + sessions.length;

  const closeAndReturnFocus = useCallback(() => {
    setOpen(false);
    triggerRef.current?.focus();
  }, []);

  // Reset focus index whenever the menu opens.
  useEffect(() => {
    if (open) {
      setFocusIndex(0);
    }
  }, [open]);

  // Move physical focus to the item indexed by focusIndex.
  useEffect(() => {
    if (!open) return;
    itemRefs.current[focusIndex]?.focus();
  }, [open, focusIndex]);

  // Close on click-outside.
  useEffect(() => {
    if (!open) return;
    function handle(e: MouseEvent) {
      if (
        wrapperRef.current &&
        !wrapperRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  const onNewSession = useCallback(() => {
    closeAndReturnFocus();
    void createSession();
  }, [closeAndReturnFocus, createSession]);

  const onSelect = useCallback(
    (id: string) => {
      closeAndReturnFocus();
      void selectSession(id);
    },
    [closeAndReturnFocus, selectSession],
  );

  const onMenuKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLUListElement>) => {
      switch (e.key) {
        case "Escape":
          e.preventDefault();
          closeAndReturnFocus();
          break;
        case "Tab":
          // Tab leaves the menu; close and return focus to the trigger.
          // Default Tab behavior is preserved (focus moves to the next
          // focusable in the page), but the menu must be closed first.
          setOpen(false);
          break;
        case "ArrowDown":
          e.preventDefault();
          setFocusIndex((i) => (i + 1) % itemCount);
          break;
        case "ArrowUp":
          e.preventDefault();
          setFocusIndex((i) => (i - 1 + itemCount) % itemCount);
          break;
        case "Home":
          e.preventDefault();
          setFocusIndex(0);
          break;
        case "End":
          e.preventDefault();
          setFocusIndex(itemCount - 1);
          break;
        case "Enter":
        case " ":
          e.preventDefault();
          if (focusIndex === 0) {
            onNewSession();
          } else {
            onSelect(sessions[focusIndex - 1].id);
          }
          break;
      }
    },
    [itemCount, focusIndex, sessions, onNewSession, onSelect, closeAndReturnFocus],
  );

  return (
    <div ref={wrapperRef} className="header-session-switcher">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={MENU_ID}
        onClick={() => setOpen((v) => !v)}
        className="header-session-switcher-trigger"
      >
        <span aria-hidden="true">Session:</span>{" "}
        <strong>{triggerLabel}</strong>
        <span aria-hidden="true"> ▾</span>
      </button>
      {open && (
        <ul
          id={MENU_ID}
          role="menu"
          aria-label="Sessions"
          className="header-session-switcher-menu"
          onKeyDown={onMenuKeyDown}
        >
          <li
            ref={(el) => {
              itemRefs.current[0] = el;
            }}
            role="menuitem"
            tabIndex={focusIndex === 0 ? 0 : -1}
            onClick={onNewSession}
            className="header-session-switcher-item header-session-switcher-item-new"
          >
            + New session
          </li>
          {sessions.map((session, idx) => {
            const itemIndex = idx + 1;
            return (
              <li
                key={session.id}
                ref={(el) => {
                  itemRefs.current[itemIndex] = el;
                }}
                role="menuitem"
                tabIndex={focusIndex === itemIndex ? 0 : -1}
                aria-current={session.id === activeSessionId ? "page" : undefined}
                onClick={() => onSelect(session.id)}
                className="header-session-switcher-item"
              >
                {session.title || `Session ${session.id.slice(0, 8)}`}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Write the failing test for `AppHeader`**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AppHeader } from "./AppHeader";

describe("AppHeader", () => {
  it("renders the ELSPETH brand", () => {
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    expect(screen.getByText(/ELSPETH/i)).toBeInTheDocument();
  });

  it("renders the session switcher", () => {
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    // The visible "Session:" prefix is aria-hidden; the accessible name is
    // the active title or fallback title.
    expect(screen.getByRole("button", { name: /untitled/i })).toBeInTheDocument();
  });

  it("renders the user menu", () => {
    render(<AppHeader onOpenSettings={() => {}} onSignOut={() => {}} />);
    expect(screen.getByRole("button", { name: /account/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Implement `AppHeader`**

```typescript
// ============================================================================
// AppHeader
//
// Thin top-level header. Three regions left-to-right:
//   - ELSPETH brand
//   - HeaderSessionSwitcher (replaces the SessionSidebar)
//   - UserMenu (settings, sign out — from Phase 1B)
// ============================================================================

import { HeaderSessionSwitcher } from "@/components/sessions/HeaderSessionSwitcher";
import { UserMenu } from "@/components/common/UserMenu";

interface AppHeaderProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

export function AppHeader({
  onOpenSettings,
  onSignOut,
}: AppHeaderProps): JSX.Element {
  return (
    <header className="app-header" role="banner">
      <div className="app-header-left">
        <span className="app-header-brand">ELSPETH</span>
        <HeaderSessionSwitcher />
      </div>
      <div className="app-header-right">
        <UserMenu onOpenSettings={onOpenSettings} onSignOut={onSignOut} />
      </div>
    </header>
  );
}
```

- [ ] **Step 7: Mount `AppHeader` in `App.tsx`**

```tsx
import { AppHeader } from "./components/common/AppHeader";

// ...inside the render, between the alert banners and `<div className="app-main">`:
<AppHeader
  onOpenSettings={() => setShowComposerSettings(true)}
  onSignOut={() => useAuthStore.getState().logout()}
/>
```

If Phase 1B already wired `onOpenSettings` and `onSignOut` through Layout, **remove those from Layout's prop list** — Layout no longer needs them; the header now owns them.

- [ ] **Step 8: CSS scaffolding for the header**

Append to `App.css`:

```css
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 40px;
  padding: 0 12px;
  border-bottom: 1px solid var(--color-border);
  background: var(--color-bg);
}

.app-header-left,
.app-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-header-brand {
  font-weight: 600;
  letter-spacing: 0.05em;
}

.header-session-switcher {
  position: relative;
}

.header-session-switcher-trigger {
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 4px 8px;
  cursor: pointer;
}

.header-session-switcher-menu {
  position: absolute;
  top: 100%;
  left: 0;
  margin: 0;
  padding: 4px 0;
  list-style: none;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  min-width: 240px;
  z-index: 100;
  max-height: 60vh;
  overflow-y: auto;
}

.header-session-switcher-item {
  padding: 6px 12px;
  cursor: pointer;
}

.header-session-switcher-item:hover,
.header-session-switcher-item:focus {
  /* --color-bg-hover is not currently defined in App.css; fallback to a
     translucent border tint matches the existing dropdown hover treatment
     (see CommandPalette / VersionSelector). When/if --color-bg-hover lands
     as a real token, the fallback is harmless. */
  background: var(--color-bg-hover, rgba(143, 200, 200, 0.08));
}

.header-session-switcher-item[aria-current="page"] {
  font-weight: 600;
}

.header-session-switcher-item-new {
  border-bottom: 1px solid var(--color-border);
}
```

- [ ] **Step 9: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/AppHeader.tsx \
        src/elspeth/web/frontend/src/components/common/AppHeader.test.tsx \
        src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.tsx \
        src/elspeth/web/frontend/src/components/sessions/HeaderSessionSwitcher.test.tsx \
        src/elspeth/web/frontend/src/App.tsx \
        src/elspeth/web/frontend/src/App.css
git commit -m "feat(web/frontend): AppHeader with session switcher + UserMenu (Phase 3A.3)"
```

---

## Task 4: Add auto-validate-on-composition-change subscription

> **2026-05-17 panel reality-check fix:** This task was originally specified as a **full rewrite** of `subscriptions.ts` (closure migration, `initialized` → `teardown` handle, rename `_resetSubscriptionsForTesting` → `_resetForTests`). That spec was written against the pre-Phase-2C version of the file. **Phase 2C already restructured `subscriptions.ts`** to own three subscribers (version-change clears `validationResult`; session-removal evicts the `auditReadinessStore` cache; `validationResult`-change fires `injectSystemMessage` + `sendValidationFeedback` — the side effects formerly owned by `InspectorPanel.handleValidate`, which Phase 2C also deleted). Executing the original Task 4 spec would have silently regressed that work. **The closure migration is no longer needed and is explicitly NOT done in this task.** Task 4 is now a strictly additive change: one new subscriber, alongside the three already in place, sharing the same module-level isolation pattern. Audit-readiness eviction, the `validationResult` side-effect subscriber, the existing test export name `_resetSubscriptionsForTesting`, and the existing test file's import all stay exactly as they are.

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/subscriptions.ts` — add a new `auto-validate` subscriber alongside the three already there. Do NOT touch the existing `useSessionStore` subscriber (version-change clears + session-removal cache eviction) or the existing `useExecutionStore` subscriber (`validationResult`-change side effects). Extend `_resetSubscriptionsForTesting()` so it tears down the new subscriber's state too.
- Modify: `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — append the new tests; keep all existing tests and the existing `_resetSubscriptionsForTesting` import.

**Trigger semantics.** The new subscriber fires `validate(sessionId)` when:

- `activeSessionId` is set, AND
- `compositionState.version` strictly increased since the last *successfully completed* validate for this session, AND
- The store is not currently executing.

**Correctness requirement (Systems-panel finding 2026-05-17).** A naive "skip if `isValidating`" guard creates a correctness gap: during rapid composition (LLM tool calls bumping version N, N+1, N+2 in quick succession), the in-flight `validate(N)` would cause the subscriber to discard versions N+1 and N+2, and the resulting `validationResult` would correspond to N while the composition is at N+2. The user would then see a confident "✓ Validation passed" badge for a stale snapshot — a CLAUDE.md `feedback_correctness_beats_performance` violation.

The fix: track the **latest observed version separately from the last validated version**, and after every in-flight `validate()` settles, re-fire if a newer version arrived in the meantime. `executionStore.validate(sessionId): Promise<void>` does not accept an `AbortSignal` today (executionStore.ts:58), so abort-and-retry is not the chosen shape; the loop pattern is.

Per-session tracking matters: `lastValidatedVersion` is a `Map<string, number>` keyed by session id, not a global. This way switching sessions does not falsely satisfy the "version unchanged" check, and switching back to a previously validated session does not re-fire unnecessarily.

Debounce is **explicitly not added.** Per the panel finding, deferring to "Phase 8 if telemetry shows pain" is non-falsifiable (there is no frontend telemetry). The correctness loop is the real fix; load-shaping is a separate decision that can be re-opened if a concrete trigger arises (e.g. backend returning 429, or operator-observed lag).

- [ ] **Step 1: Read the current `subscriptions.ts` end-to-end**

```bash
cat src/elspeth/web/frontend/src/stores/subscriptions.ts
```

Confirm **two** `subscribe()` calls are present implementing **three** logical behaviors: `useSessionStore.subscribe(...)` (one call) handles version-change clear AND session-removal audit-readiness cache eviction; `useExecutionStore.subscribe(...)` (one call) handles `validationResult`-change side effects (`injectSystemMessage` + `sendValidationFeedback`). The module-level trackers Phase 2C established are `previousVersion`, `previousSessionIds`, and `previousValidationFingerprint` (a stringified content guard — **not** named `previousValidationResult` in earlier drafts of this plan; the actual export uses `Fingerprint`). `_resetSubscriptionsForTesting` is exported at file end. If the file does not match this shape — particularly if there are fewer than two subscribe calls, or three subscribe calls, or the fingerprint tracker has been renamed — stop and reconcile against the Phase 2C state.

- [ ] **Step 2: Write the failing test (append to existing `subscriptions.test.ts`)**

The existing test file already imports `_resetSubscriptionsForTesting`. **Do not change that import.** Append a new `describe` block:

```typescript
import { act, waitFor } from "@testing-library/react";

describe("auto-validate on composition-state version change", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
      sessions: [{ id: "sess-1", title: "x" } as never],
    } as never);
    useExecutionStore.setState({
      isExecuting: false,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("fires validate when compositionState.version increments", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-1"));

    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
  });

  it("does not fire when version is unchanged (reference change only)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
    } as never);
    // Give the loop a chance to re-fire if it were going to.
    // Flush microtasks deterministically. If the subscriber were going to
    // fire validate(), it would have done so by now.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("does NOT fire while executing", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate, isExecuting: true } as never);

    useSessionStore.setState({
      compositionState: { version: 9, source: null, nodes: [], outputs: [] } as never,
    } as never);

    // Flush microtasks deterministically. If the subscriber were going to
    // fire validate(), it would have done so by now.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("does not fire when activeSessionId is null", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);
    useSessionStore.setState({ activeSessionId: null } as never);

    useSessionStore.setState({
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);

    // Flush microtasks deterministically. If the subscriber were going to
    // fire validate(), it would have done so by now.
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(validate).not.toHaveBeenCalled();
  });

  it("re-fires after the in-flight validate settles if a newer version arrived (correctness loop)", async () => {
    // This is the correctness guarantee: the user must not see a stale
    // validationResult badge for a snapshot that no longer matches the
    // composition. We simulate validate() that takes a microtask to resolve;
    // between the call and the resolve, version increments. The subscriber
    // MUST fire a second validate() once the first settles.
    let resolveFirst: (() => void) | null = null;
    const firstCallPromise = new Promise<void>((r) => {
      resolveFirst = r;
    });
    const validate = vi
      .fn()
      .mockImplementationOnce(() => firstCallPromise)
      .mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(1));

    // Version increments while validate(v1) is still in flight.
    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    // Loop has not re-fired yet — the in-flight call is still pending.
    expect(validate).toHaveBeenCalledTimes(1);

    // Settle the in-flight call. The loop must observe pending=v2 and re-fire.
    resolveFirst!();
    await waitFor(() => expect(validate).toHaveBeenCalledTimes(2));
    expect(validate).toHaveBeenLastCalledWith("sess-1");
  });

  it("resets per-session tracking when activeSessionId changes (cross-session isolation)", async () => {
    const validate = vi.fn().mockResolvedValue(undefined);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-A",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));

    // Switch to a fresh session whose first observed version happens to also be 1.
    // Per-session tracking means this must fire — global tracking would falsely
    // satisfy "version unchanged".
    useSessionStore.setState({
      activeSessionId: "sess-B",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-B"));
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/subscriptions.test.ts -t "auto-validate"
```

Expected: FAIL — the new subscriber does not exist yet.

- [ ] **Step 4: Implement the additive subscriber**

Edit `src/elspeth/web/frontend/src/stores/subscriptions.ts`. **Do not delete or rewrite anything that already exists.** Add module-level state for the new subscriber alongside the existing trackers, add a third subscribe call inside `initStoreSubscriptions()` after the two existing ones, and extend `_resetSubscriptionsForTesting()` to tear it down.

The additive sketch (insert into the existing file, do not replace it):

```typescript
// At module scope, alongside the existing previousVersion / previousSessionIds /
// previousValidationFingerprint trackers:

// Per-session tracking for the auto-validate subscriber. lastValidated maps
// session-id → highest version successfully completed by validate(). pending
// holds the most recently observed (session-id, version) that still needs
// validating; the loop reads it on every settle.
const lastValidatedVersionBySession = new Map<string, number>();
let pendingValidateTarget: { sessionId: string; version: number } | null = null;
let validateInflight = false;
let unsubscribeAutoValidate: (() => void) | null = null;

// Inside initStoreSubscriptions(), after the existing two subscribe calls:

unsubscribeAutoValidate = useSessionStore.subscribe((state) => {
  const sessionId = state.activeSessionId;
  const version = state.compositionState?.version ?? null;
  if (!sessionId || version === null) return;
  if (lastValidatedVersionBySession.get(sessionId) === version) return;
  if (useExecutionStore.getState().isExecuting) return;

  pendingValidateTarget = { sessionId, version };
  if (validateInflight) return;  // settle-handler in fireValidateLoop catches it
  void fireValidateLoop();
});

// Helper — runs outside any subscribe callback so the loop can re-read store
// state between iterations. Re-reads handle: (a) session switches, (b) version
// increments during in-flight call, (c) execute() starting mid-loop.
async function fireValidateLoop(): Promise<void> {
  validateInflight = true;
  try {
    while (pendingValidateTarget !== null) {
      const target = pendingValidateTarget;
      // Bail out if execute started, session cleared, or version already valid.
      if (useExecutionStore.getState().isExecuting) {
        pendingValidateTarget = null;
        break;
      }
      if (target.sessionId !== useSessionStore.getState().activeSessionId) {
        // Active session changed while we were preparing to fire. Drop this
        // target; the subscribe handler will re-queue the new session if needed.
        pendingValidateTarget = null;
        break;
      }
      if (lastValidatedVersionBySession.get(target.sessionId) === target.version) {
        pendingValidateTarget = null;
        break;
      }

      // FRAGILE — see test "re-fires after the in-flight validate settles if a newer version arrived (correctness loop)".
      // Do not add code between this clear and the await; the correctness
      // invariant depends on a newer version setting pendingValidateTarget
      // via the subscribe handler during the await window.
      // Clear pending BEFORE awaiting, so a newer version arriving during the
      // await is captured by the subscribe handler into pendingValidateTarget.
      pendingValidateTarget = null;
      await useExecutionStore.getState().validate(target.sessionId);
      // Successful-validation tracking is intentionally post-await. If validate()
      // throws, do not mark this version validated; the next version bump retries.
      lastValidatedVersionBySession.set(target.sessionId, target.version);
      // Loop re-checks pendingValidateTarget. If a newer version arrived during
      // the await, we run again. Otherwise we exit.
    }
  } finally {
    validateInflight = false;
  }
}
```

Extend `_resetSubscriptionsForTesting()`:

```typescript
export function _resetSubscriptionsForTesting(): void {
  unsubscribe?.();
  unsubscribe = null;
  unsubscribeExecution?.();
  unsubscribeExecution = null;
  unsubscribeAutoValidate?.();
  unsubscribeAutoValidate = null;
  previousVersion = null;
  previousValidationFingerprint = null;
  previousSessionIds = new Set();
  lastValidatedVersionBySession.clear();
  pendingValidateTarget = null;
  validateInflight = false;
  initialized = false;
}
```

Update the JSDoc on `initStoreSubscriptions` to list the new subscriber alongside the three already there. **Do not change the function signature; do not rename the export.**

Notes:

- The loop re-reads store state every iteration. This handles session switches and `isExecuting` transitions correctly — bail out and re-queue from a fresh subscribe firing.
- `pendingValidateTarget` is cleared **before** awaiting `validate()`. A subsequent version increment arriving during the await sets it back to the new `(sessionId, version)`, which the loop body picks up on its next iteration.
- `lastValidatedVersionBySession.set(target.sessionId, target.version)` intentionally stays **after** the awaited `validate()` call. If `validate()` throws, the loop unwinds before recording that version as successfully validated; the next version bump retries instead of silently marking a stale snapshot complete. Do not "simplify" this by moving the set above the await.
- The `validationResult` consumed by `injectSystemMessage` / `sendValidationFeedback` is published by the existing executionStore subscriber (Phase 2C, untouched). The auto-validate subscriber's only side effect is the `validate()` call itself; the result-driven side effects flow through the existing path.
- No frontend telemetry / no logger call. Failures inside `validate()` are recorded in the backend audit Landscape per CLAUDE.md primacy.

- [ ] **Step 4a: Add cross-session guard to the Phase 2C `validationResult` subscriber (Section A panel-fix S3)**

> **Review finding (IMPORTANT — Section A panel fix S3):** Auto-validate widens the timing window between `validate(sessionId)` firing and the `validationResult`-change subscriber executing `injectSystemMessage`. If the user switches sessions during that window, the system message is injected into the **new** session's chat (because `injectSystemMessage` reads `useSessionStore.getState().activeSessionId` at call time). The fix is to gate the Phase 2C subscriber on a session-id match.

First, inspect the `ValidationResult` shape and confirm the current type:

```bash
grep -n "ValidationResult\|validationResult" src/elspeth/web/frontend/src/types/index.ts \
                                                  src/elspeth/web/frontend/src/stores/executionStore.ts \
                                                  src/elspeth/web/frontend/src/api/*.ts 2>/dev/null \
  | head -40
```

As of the Phase 3 review, live `ValidationResult` has `is_valid`, `checks`, `errors`, and optional `warnings`; it does **not** carry `session_id`. Use the transient tracker below unless the type is deliberately changed before implementation and every caller is updated in the same commit.

Add a transient tracker in `subscriptions.ts` that captures the session id at the moment Task 4's auto-validate fires, and exposes it for the Phase 2C subscriber. Add at module scope alongside the other trackers:

```typescript
// Tracks the session id passed to the most recent validate() call. Used by
// the Phase 2C validationResult subscriber to guard against cross-session
// races widened by auto-validate's increased timing window (Section A S3).
let inflightValidateSessionId: string | null = null;
```

In `fireValidateLoop`, set `inflightValidateSessionId = target.sessionId` immediately BEFORE the `await useExecutionStore.getState().validate(target.sessionId)` line, and clear it (`inflightValidateSessionId = null`) immediately after.

In the Phase 2C `validationResult`-change subscriber, add the guard before any non-null `validationResult` fingerprint computation or assignment:

```typescript
const result = state.validationResult;
if (!result) {
  previousValidationFingerprint = null;
  return;
}
const currentSessionId = useSessionStore.getState().activeSessionId;
if (inflightValidateSessionId !== null && inflightValidateSessionId !== currentSessionId) {
  // Auto-validate fired for a session the user has since left. Suppress
  // the UI side effect; the audit Landscape still records the outcome.
  return;
}

const fingerprint = validationFingerprint(result);
```

That ordering is load-bearing: a suppressed stale result must **not** mutate `previousValidationFingerprint`, or a later same-content result for the current session would be incorrectly suppressed.

Manual `Ctrl+Shift+V` validation in App.tsx does not flow through `fireValidateLoop`, so for that path `inflightValidateSessionId` stays `null` and the guard is a no-op (preserves existing manual-validate behavior). Update `_resetSubscriptionsForTesting` to clear `inflightValidateSessionId = null` if Path B is chosen.

Add this failing test BEFORE the implementation (append to the auto-validate `describe` block in `subscriptions.test.ts`):

```typescript
  it("does not inject system message when the user switched sessions mid-validate (cross-session guard)", async () => {
    const injectSystemMessageSpy = vi.fn();
    const staleValidationResult = {
      is_valid: false,
      checks: [],
      errors: [
        {
          component_type: "source",
          component_id: "csv_source",
          message: "Missing path",
        } as never,
      ],
      warnings: [],
    };
    useSessionStore.setState({
      activeSessionId: "sess-A",
      sessions: [{ id: "sess-A" } as never, { id: "sess-B" } as never],
      injectSystemMessage: injectSystemMessageSpy,
    } as never);

    let resolveValidate: (() => void) | null = null;
    const validatePromise = new Promise<void>((r) => {
      resolveValidate = r;
    });
    const validate = vi.fn().mockImplementation(() => validatePromise);
    useExecutionStore.setState({ validate } as never);

    // Trigger auto-validate for sess-A.
    useSessionStore.setState({
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));

    // User switches to sess-B while validate(sess-A) is in flight.
    useSessionStore.setState({ activeSessionId: "sess-B" } as never);

    // Validate(sess-A) settles. The Phase 2C subscriber must NOT inject
    // a message into sess-B's chat.
    useExecutionStore.setState({
      validationResult: staleValidationResult as never,
    } as never);
    resolveValidate!();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(injectSystemMessageSpy).not.toHaveBeenCalled();
  });

  it("does not let a suppressed stale result consume the validation fingerprint", async () => {
    const injectSystemMessageSpy = vi.fn();
    const sameContentResult = {
      is_valid: false,
      checks: [],
      errors: [
        {
          component_type: "source",
          component_id: "csv_source",
          message: "Missing path",
        } as never,
      ],
      warnings: [],
    };
    useSessionStore.setState({
      activeSessionId: "sess-A",
      sessions: [{ id: "sess-A" } as never, { id: "sess-B" } as never],
      injectSystemMessage: injectSystemMessageSpy,
    } as never);

    let resolveValidate: (() => void) | null = null;
    const validatePromise = new Promise<void>((r) => {
      resolveValidate = r;
    });
    const validate = vi.fn().mockImplementation(() => validatePromise);
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    await waitFor(() => expect(validate).toHaveBeenCalledWith("sess-A"));
    useSessionStore.setState({ activeSessionId: "sess-B" } as never);

    // This stale result is suppressed and must not update previousValidationFingerprint.
    useExecutionStore.setState({ validationResult: sameContentResult as never } as never);
    resolveValidate!();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(injectSystemMessageSpy).not.toHaveBeenCalled();

    // A later same-content result for the current session must still inject once.
    useExecutionStore.setState({
      validationResult: { ...sameContentResult } as never,
    } as never);
    await waitFor(() => expect(injectSystemMessageSpy).toHaveBeenCalledTimes(1));
  });
```

Run the test (expect FAIL — no guard yet), implement the chosen Path, re-run (expect PASS). Commit as `feat(web/frontend): cross-session guard on validationResult subscriber (Phase 3A.4a)` with body explaining the chosen path and rationale.

- [ ] **Step 5: Run all subscription tests**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores
```

Expected: PASS — including the existing tests for version-change, audit-readiness eviction, and `validationResult` side-effects (all untouched), plus the new auto-validate suite.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/subscriptions.ts \
        src/elspeth/web/frontend/src/stores/subscriptions.test.ts
git commit -m "feat(web/frontend): auto-validate on compositionState.version increment (Phase 3A.4)

Additive third subscriber inside initStoreSubscriptions(); existing
version-clear, audit-readiness-evict, and validationResult-driven
injectSystemMessage subscribers (all Phase 2C) untouched. Correctness
loop re-fires validate() after in-flight settle when a newer version
arrived, so the user never sees a stale validation badge."
```

---

**Tasks 5–7 (the removal tasks; Task 8 retired — see panel note in 15a2) continue in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).**

---

## Review history

### 2026-05-15 — Review panel applied

**BLOCKER (Quality+Systems):** `lastValidatedVersion`, the pre-existing `previousVersion`, and a new `previousSessionId` tracker all moved from module scope into `initStoreSubscriptions()`'s closure. The module-level `initialized` boolean is replaced by a module-level `teardown` handle; a test-only `_resetForTests()` tears down the active subscription and rebuilds the closure between tests. Cross-test isolation test in Task 4 Step 2 calls `_resetForTests()` in `beforeEach`. Same shape as Phase 1A finding 7.

**CRITICAL (Reality+Systems):** Operator adjudication 2026-05-15 — Task 4 Step 4's `telemetry.record(...)` call is **deleted**, not retained with a different channel. No frontend telemetry module exists; per CLAUDE.md audit-primacy ("logger is NOT for pipeline activity"; primacy order audit → telemetry → log), validate-failure breadcrumbs in the frontend would duplicate the backend audit Landscape without adding probative value. Phase 8 (polish + telemetry) is the right owner if a frontend operational signal proves useful later. `void exec.validate(sessionId)` is the fire-and-forget shape.

**CRITICAL (Reality):** Operator adjudication 2026-05-15 — the previous draft used `useSessionStore.subscribe((state, prevState) => ...)`. Zustand's default `subscribe` is single-arg; the two-arg overload requires `subscribeWithSelector` middleware, which is not wired in the stores. Replaced with a closure-internal `previousSessionId` updated at the tail of the subscription. Matches the existing `previousVersion` precedent.

**IMPORTANT (Architecture):** Task 2 scaffold updated to use render-props (`auditReadinessSlot`, `executeButtonSlot`, etc.) instead of empty-div placeholders. `SideRailProps` now explicitly declares all named slots as `ReactNode | null`. `executeButtonSlot` is added here (required by 15b Task 2 before the inspector deletion). The composition contract is explicit: callers pass content via props from `App.tsx`; `SideRail` does not mount components by event or by its own internal logic.

**Cross-file decision:** `SWITCH_TAB_EVENT` — IMPORTANT (15b2 §4) overrides SUGGESTION (15b1 §3); the constant is deleted in Phase 3B, not deferred to Phase 6. See 15b1 and 15b2 review history entries.

### 2026-05-17 — Reality-check panel applied (NO-GO → fixes landed)

Four reviewers (Reality / Architecture / Quality / Systems) ran against the worktree after Phase 2C landed. Reality returned NO-GO; the other three returned CONDITIONAL GO. The convergent finding: the plan was authored against the pre-Phase-2C state and would have regressed Phase 2C work if executed. Fixes landed in this revision:

**CRITICAL (Reality+Architecture+Quality, convergent) — Task 4 reframed from "full rewrite" to additive change.** The original Task 4 spec rewrote `subscriptions.ts` in full and renamed `_resetSubscriptionsForTesting` → `_resetForTests`. Phase 2C already restructured `subscriptions.ts` to own three subscribers (version-change clears `validationResult`; session-removal evicts `auditReadinessStore`; `validationResult`-change fires `injectSystemMessage` + `sendValidationFeedback`). The original rewrite would have silently dropped the audit-readiness eviction and the `validationResult` side-effect subscriber. The new Task 4 spec adds a **fourth** subscriber alongside the three already there, preserves the existing export name `_resetSubscriptionsForTesting`, and extends `_resetSubscriptionsForTesting()` to tear down the new subscriber. The closure-migration framing from the 2026-05-15 panel is moot now — module-scope state is the established Phase 2C pattern.

**CRITICAL (Reality) — Task 8 retired.** Phase 2C already deleted the Validate button (`InspectorPanel.tsx:557-572`), the `handleValidate` callback (`InspectorPanel.tsx:387-424`), and moved the side-effect orchestration into `subscriptions.ts`. The original Task 8 spec referenced lines that no longer exist. Sequencing diagram updated. 15a2 carries the retirement note.

**CRITICAL (Reality) — Audit-readiness slot is RESERVED, not "filled by Phase 2".** Scope-boundaries text updated to acknowledge that Phase 2C mounted `AuditReadinessPanel` inside `InspectorPanel.tsx:534`. The SideRail's `auditReadinessSlot` stays `null` through 15a; 15b2 Task 9 Step 4a migrates the mount before `InspectorPanel.tsx` is deleted.

**IMPORTANT (Systems) — Auto-validate correctness loop.** Task 4's original `isValidating` skip-guard would silently discard `compositionState.version` increments that arrived during in-flight validate calls, leaving the user with a stale validation badge during rapid composition flows (LLM tool calls bump version N→N+1→N+2). New spec uses per-session `lastValidatedVersionBySession` + `pendingValidateTarget` + a `fireValidateLoop()` that re-checks store state on every iteration. `executionStore.validate(sessionId)` does not accept an `AbortSignal` (`executionStore.ts:58`), so the cancel-and-retry variant from the Systems suggestion was reshaped as a track-and-re-fire loop. Debounce deferral retired (the "if telemetry shows pain" condition was non-falsifiable — no frontend telemetry exists).

**IMPORTANT (Quality) — `App.test.tsx` Layout mock prop rename.** Task 2 now explicitly instructs updating the `App.test.tsx` Layout mock destructure from `{ sidebar, chat, inspector }` → `{ sidebar, chat, siderail }`. Without this, React silently drops the unrecognised prop and the mocked test stays green while real `App.tsx` is broken.

**IMPORTANT (Quality) — Dangling `activeRunId` test added** to InlineRunResults' test suite (Task 1 Step 2). Covers the WebSocket-reconnect / session-switch race where `activeRunId` briefly points to a run not in `runs[]`.

**IMPORTANT (Quality) — Empty-sessions test added** to HeaderSessionSwitcher's test suite (Task 3 Step 2). Verifies the "New session" verb stays visible when `sessions: []` and `activeSessionId` is non-null (just-archived edge).

**SUGGESTION (Reality) — `--color-bg-hover` fallback.** The CSS variable used in Task 3 Step 8 is undefined in `App.css`. Replaced `var(--color-bg-hover)` with `var(--color-bg-hover, rgba(143, 200, 200, 0.08))` so the hover state is visible regardless of whether the token is later defined.

**SUGGESTION (Reality) — ChatPanel line count refresh** from 573 → 622 (drift from Phase 2C additions). Cosmetic but indicative.

### 2026-05-17 — Section A panel-fix pass (Q1 / Q2)

**IMPORTANT (Quality) — RunsHistoryDrawer focus trap landed.** `role="dialog" aria-modal="true"` was committed without a focus trap; the contract is now satisfied by a manual Tab / Shift+Tab cycle plus focus-on-open lands the Close button. Two tests added in Task 1 Step 5 (`moves focus into the drawer on open`, `traps Tab and Shift+Tab inside the drawer`). No `inert` polyfill — keeps the project's vanilla-React idiom.

**IMPORTANT (Quality) — HeaderSessionSwitcher keyboard nav + aria-controls.** Trigger now declares `aria-controls={MENU_ID}` linking it to the menu's `id`. Menu uses roving tabindex; ArrowUp / ArrowDown wrap, Home / End jump to ends, Enter / Space activate, Tab closes the menu and returns focus to the trigger. Three tests added in Task 3 Step 2 covering ArrowDown traversal, ArrowUp wrap, and Tab-exit-to-trigger.

### 2026-05-17 — Section B+C panel-fix pass (B9 / B11 / C14 / C15 / C16 / A3 / S3)

**IMPORTANT (Quality, B9) — Layout localStorage round-trip test added.** Task 2 Step 4 now includes a test that sets `elspeth_inspector_width` to a known value and asserts the resulting grid column. Defends against accidental key-string rename during the `inspector → siderail` search-replace.

**IMPORTANT (Quality, B11) — Deterministic microtask flush replaces `setTimeout(r, 0)`.** Task 4's three negative-path tests now use `await act(async () => { await Promise.resolve(); await Promise.resolve(); })` instead of a one-tick timeout. Symmetric with the positive-path `waitFor` discipline.

**MINOR (Reality, C14) — Stale try/catch prose struck.** Trust-tier-check paragraph no longer claims `InspectorPanel.tsx:410–415` wraps `sendValidationFeedback` — Phase 2C deleted that code; the surviving subscriber accesses store fields directly.

**MINOR (Systems, C15) — Task 4 Step 1 reconciliation gate clarified.** Now explicitly counts **two** `subscribe()` calls implementing **three** logical behaviors. Executor counting `subscribe()` calls and expecting `3` would false-trigger the gate.

**MINOR (Architecture, C16) — `FRAGILE` comment added to `fireValidateLoop`.** Above the `pendingValidateTarget = null` clear-before-await, a comment cites the test that locks the invariant. A future maintainer adding code between the clear and the await must read the cited test first.

**OPERATOR DECISION (A3) — `RunsHistoryDrawer` preserved.** Decision recorded in the Open-Scope-Questions item 3: the audit-focused persona Linda (`project_composer_personas` memory) benefits from past-runs access. Drawer is self-contained; deletion in Phase 8 is cheap if no demand emerges.

**OPERATOR DECISION (S3) — Cross-session guard on Phase 2C subscriber.** New Task 4 Step 4a adds a guard to the existing `validationResult`-change subscriber. The auto-validate path widens the cross-session timing window; the guard suppresses `injectSystemMessage` if `activeSessionId` no longer matches the session that produced the `validationResult`. The live `ValidationResult` type does not carry `session_id`, so the plan now uses the transient `inflightValidateSessionId` tracker path only. The guard is ordered before non-null fingerprint mutation, and tests use valid `ValidationResult` objects plus a same-content retry case to prove suppressed stale results do not consume `previousValidationFingerprint`.

### 2026-05-17 — Pre-execution reality-check addendum (tracker-name + executor-clarity)

Final pre-execution sweep against the then-current Phase 3 plan branch turned up three small things worth resolving before the executor picks up the plan; all doc-only, no behavioural change.

**MINOR (Reality) — Fingerprint tracker name corrected.** Task 4 Step 1's reconciliation gate now names `previousValidationFingerprint` (the stringified content-guard actually exported by `subscriptions.ts:19`) rather than the earlier draft's `previousValidationResult`. Two downstream code-comment / reset-helper references in Step 4 updated to match. The gate's behavioral check ("two subscribe calls, three logical behaviors") is unchanged.

**MINOR (Architecture) — Alt-key tabMap dead-dispatch documented as deliberate scope-deferral.** 15a2 Task 6 Step 6a's smoke-grep `grep 'detail:.*"spec"'` cannot catch `App.tsx:155-162`'s keyboard tabMap (which constructs `detail: tab` from a lookup table). A "known false-negative" note now flags this site as intentionally deferred to 15b per the §"Out of scope" list, with the rationale (Alt+2 / Alt+3 must keep working untouched until the keyboard handler is refactored). Cross-referenced from 15a2.

**MINOR (Quality) — Redirect-toast mount-point made explicit in 15a2.** Tasks 5 Step 6 / 6a and Task 6 Step 8a now name the integration shape: `useHashRouter` returns a `redirectToast: { message; dismiss } | null` field; `App.tsx` mounts an `<div role="alert" className="alert-banner alert-banner--info">…</div>` immediately above the existing `App.tsx:235` alert-banner; the dismissal localStorage key works across both `runs` and `spec` paths because the toast field becomes `null` whenever the dismissal flag is set. This unblocks the `screen.getByRole("alert")` test assertions added by S4 / S2.

### 2026-05-17 — Pre-dispatch NO-GO follow-up

**BLOCKER (Execution target) — Phase 3 worktree/branch made concrete.** Shared header now names `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on `feat/composer-phase-3-ia-cleanup` from `RC5.2`; the old Phase 2A worktree/branch are explicitly forbidden.

**BLOCKER (ValidationResult shape) — Cross-session guard test corrected.** Task 4 Step 4a now uses the live `ValidationResult` shape (`is_valid`, `checks`, `errors`, `warnings`) instead of `{ ok: true }`, orders the guard before fingerprint mutation, and adds a same-content current-session retry test.

**MINOR (Quality) — AppHeader accessible-name assertion corrected.** The `Session:` prefix remains `aria-hidden`; the AppHeader test now queries the active/fallback title rather than expecting `Session:` in the accessible name.
