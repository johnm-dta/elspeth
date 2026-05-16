# Phase 3A — IA Cleanup: side-rail scaffold, header session switcher, Spec/Runs removal, run-results inline (Part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the first half of Phase 3 — introduce the side-rail scaffold, header session switcher dropdown, inline run-results component, and remove the Spec tab + Runs tab + always-on session sidebar. Leaves the app launchable and green at every commit. Defers the graph mini-view, YAML export modal, Catalog button move, version-selector relocation, hash-router migration, and the final InspectorPanel deletion to **15b**.

**Architecture:** Frontend chrome refactor. No new trust boundary; no backend changes. The `useHashRouter` invariants (`VALID_TABS`, default-tab `"spec"`) are *preserved* in this plan and migrated in 15b — Phase 3A keeps `spec`/`runs`/`graph`/`yaml` valid hash fragments so that mid-phase deep links don't 404, and the InspectorPanel keeps rendering Graph + YAML inside the inspector until 15b extracts them.

**Tech Stack:** React + Zustand + Vitest + testing-library. No new dependencies.

**Sibling plans:**
- Predecessor: [13-phase-1b-frontend.md](13-phase-1b-frontend.md) — Phase 3A **assumes Phase 1B has shipped**, because Phase 1B adds the `UserMenu` to the header. If 1B has not shipped, do not start this plan — fix 1B first.
- Successor: [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md) / [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md) — Graph mini, YAML modal, Catalog button move, version selector relocation, hash redirects, InspectorPanel deletion.

**Part split:** This file covers Tasks 1–4 (additive work). Tasks 5–8 (the removal tasks) plus Risks, Memory references, and Review history are in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3) and §A (H1, H2 calls).

**Design spec:** [03-target-information-architecture.md](03-target-information-architecture.md).

---

## Scope boundaries

**In scope (this plan, 15a):**
- New `SideRail.tsx` scaffold mounted to the right of the chat column. Slots for: audit-readiness placeholder (filled by Phase 2), graph mini placeholder (filled by 15b), Catalog button placeholder (filled by 15b), Export-YAML button placeholder (filled by 15b), completion bar placeholder (filled by Phase 6). The side rail renders even when all real content is deferred — it carries the layout shape.
- New `HeaderSessionSwitcher.tsx` — header dropdown that lists sessions (`session-switcher` per design doc 03). Replaces the always-on `SessionSidebar`.
- New `InlineRunResults.tsx` — mounts in the chat column below `ChatPanel`; subscribes to `executionStore.activeRunId` + `progress`; renders `ProgressView` and `RunOutputsPanel` for the active/most-recent run. Carries the previous Runs-tab functionality minus the historical list.
- New `RunsHistoryDrawer.tsx` — preserves access to historical runs via a "Past runs" affordance in `InlineRunResults`. Required because the design doc is silent on the runs list; this plan resolves that as "keep, demote to a drawer" (see §"Open scope questions resolved" below).
- Removal of `SessionSidebar.tsx` import + render from `App.tsx` / `Layout.tsx`. The file itself is deleted.
- Removal of `SpecView.tsx` import + render from `InspectorPanel.tsx`. The file itself is deleted. `Spec` tab button is dropped from the tab strip.
- Removal of `RunsView.tsx` import + render from `InspectorPanel.tsx`. The file itself is deleted. `Runs` tab button is dropped from the tab strip.
- Removal of the `Validate` button from the inspector header. Validation is now driven by an auto-validate-on-composition-change effect in `executionStore` (see Task 4 below) plus the existing `Ctrl+Shift+V` shortcut, which is retained as a manual re-trigger. The validation banner stays where it is in the inspector header for this slice; 15b moves it.
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

Per [CLAUDE.md](../../../CLAUDE.md) "Defensive Programming: Forbidden", this plan does **not** introduce `try`/`catch` around store calls "to be safe." The existing patterns (e.g., InspectorPanel.tsx:410–415 wraps `sendValidationFeedback` in a try/catch but only because the user-visible system message has already been injected; the wrap is purposeful) are preserved as-is. New components access typed store fields directly.

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
Task 4 — Add auto-validate-on-change effect         (behaviour addition; nothing removed)
Task 5 — Remove Runs tab + RunsView + delete file   (first removal; runs handled by Task 1)
Task 6 — Remove Spec tab + SpecView + delete file   (second removal; the Spec→banner-click flow is dropped — see risks)
Task 7 — Remove SessionSidebar mount + delete file  (third removal; session switching handled by Task 3 + palette)
Task 8 — Remove Validate button from inspector hdr  (last removal; auto-validate from Task 4 + Ctrl+Shift+V cover it)
```

Tasks 1–4 are in this file. Tasks 5–8 are in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).

Each task is TDD-shaped: failing test, implementation, passing test, smoke render, commit.

## Open scope questions resolved by this plan

1. **Auto-validate on composition change vs explicit Validate button.** Resolution: **auto-validate**. The audit-readiness panel from Phase 2 will be the indicator surface, but Phase 2 has no Validate button either by design ("the indicator already does the work" — design doc 03 table row "Validate button"). To bridge Phase 3 → Phase 2 without losing validation entirely, Phase 3A adds an `executionStore` cross-store subscription that fires `validate(sessionId)` whenever `compositionState.version` increments (debounced, deduplicated). `Ctrl+Shift+V` survives as a manual re-trigger that doesn't depend on a button. See Task 4.
2. **Where Execute lives between Phase 3 and Phase 6.** Resolution: **inspector header Execute button stays through 15a and 15b**. The "Execute moves to completion bar" is **Phase 6**; Phase 3 does *not* delete it. The design doc lists Execute → completion bar as a Phase 6 row, not Phase 3. The side-rail scaffold in Task 2 reserves a `completion-bar` slot but renders nothing there until Phase 6 fills it. This is explicitly called out so executors don't preemptively remove Execute.
3. **Historical runs list.** Resolution: **keep, demoted to a drawer** (`RunsHistoryDrawer`). The design doc says "run results appear inline after Execute fires" but does not say the historical list disappears. Linda (audit-focused persona) needs access to past runs for audit review. The drawer is opened by a "Past runs" button in `InlineRunResults`. Phase 8 polish may revisit shape; structure preserved here so we don't destroy access. *Assumption flagged.*
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
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` — mount `<InlineRunResults />` after the chat scrollback (and before the chat input). Do not inline the component logic into ChatPanel (573-line component, already complex).

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
});
```

- [ ] **Step 6: Implement `RunsHistoryDrawer`**

Create `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx`:

```typescript
// ============================================================================
// RunsHistoryDrawer
//
// Slide-over drawer listing every run for the current session. Opened from
// InlineRunResults' "Past runs" button. Preserves audit-trail access to old
// runs after the inspector Runs tab is removed (Phase 3A Task 5).
//
// Minimal shape in this plan — list of run-id + status badge + duration.
// Phase 8 polish may expand to include accounting summaries.
// ============================================================================

import { useEffect } from "react";
import { useExecutionStore } from "@/stores/executionStore";
import { useSessionStore } from "@/stores/sessionStore";

interface RunsHistoryDrawerProps {
  onClose: () => void;
}

export function RunsHistoryDrawer({ onClose }: RunsHistoryDrawerProps): JSX.Element {
  const runs = useExecutionStore((s) => s.runs);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  // Close on Escape.
  useEffect(() => {
    function handle(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Past pipeline runs"
      className="runs-history-drawer"
    >
      <header className="runs-history-drawer-header">
        <h2>Past runs</h2>
        <button
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
// H1 resolution: this + CommandPalette Sessions section together cover the
// header-dropdown + Cmd-palette path described in 00-implementation-roadmap.md
// §A row H1.
// ============================================================================

import { useState, useRef, useEffect, useCallback } from "react";
import { useSessionStore } from "@/stores/sessionStore";

export function HeaderSessionSwitcher(): JSX.Element {
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const selectSession = useSessionStore((s) => s.selectSession);
  const createSession = useSessionStore((s) => s.createSession);

  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const triggerLabel = activeSession?.title ?? "Untitled";

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

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    function handle(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [open]);

  const onNewSession = useCallback(() => {
    setOpen(false);
    void createSession();
  }, [createSession]);

  const onSelect = useCallback(
    (id: string) => {
      setOpen(false);
      void selectSession(id);
    },
    [selectSession],
  );

  return (
    <div ref={wrapperRef} className="header-session-switcher">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="header-session-switcher-trigger"
      >
        <span aria-hidden="true">Session:</span>{" "}
        <strong>{triggerLabel}</strong>
        <span aria-hidden="true"> ▾</span>
      </button>
      {open && (
        <ul role="menu" aria-label="Sessions" className="header-session-switcher-menu">
          <li
            role="menuitem"
            tabIndex={0}
            onClick={onNewSession}
            onKeyDown={(e) => e.key === "Enter" && onNewSession()}
            className="header-session-switcher-item header-session-switcher-item-new"
          >
            + New session
          </li>
          {sessions.map((session) => (
            <li
              key={session.id}
              role="menuitem"
              tabIndex={0}
              aria-current={session.id === activeSessionId ? "page" : undefined}
              onClick={() => onSelect(session.id)}
              onKeyDown={(e) => e.key === "Enter" && onSelect(session.id)}
              className="header-session-switcher-item"
            >
              {session.title || `Session ${session.id.slice(0, 8)}`}
            </li>
          ))}
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
    // The switcher renders a button labelled "Session: <title>".
    expect(screen.getByRole("button", { name: /session:/i })).toBeInTheDocument();
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
  background: var(--color-bg-hover);
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

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/subscriptions.ts` — add a cross-store subscription that fires `validate(sessionId)` when `compositionState.version` increments.
- Modify: `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — add the subscription test.

The current `subscriptions.ts` already wires session→execution auto-clear (per executionStore.ts:4–10 comment). Task 4 adds a complementary "auto-validate" subscription that fires when:

- `activeSessionId` is set, AND
- `compositionState.version` strictly increased since last fire, AND
- The store is not currently validating or executing.

Debounce is **not** added in v1: composition-state version increments are themselves coarse (per-edit, not per-keystroke). If telemetry shows excessive validate calls, Phase 8 can add a debounce.

> **Review finding (BLOCKER, operator-adjudicated 2026-05-15):** `lastValidatedVersion`, the pre-existing `previousVersion`, and the new `previousSessionId` tracker must all live inside `initStoreSubscriptions()`'s closure — not at module scope. The current module-level `initialized` boolean is replaced by a module-level `teardown` handle that `_resetForTests()` invokes between tests to rebuild the closure cleanly. The implementation in Step 4 and the isolation test in Step 2 reflect this requirement. **Operator adjudication:** previous-session tracking uses a closure variable updated at the tail of the subscription rather than the zustand `subscribe((state, prevState) => ...)` overload, which would require `subscribeWithSelector` middleware that the project does not wire.

- [ ] **Step 1: Read `subscriptions.ts` end-to-end**

```bash
grep -n "" src/elspeth/web/frontend/src/stores/subscriptions.ts | head -80
```

Identify the existing subscription pattern. The new subscription mirrors it.

- [ ] **Step 2: Write the failing test**

Append to `src/elspeth/web/frontend/src/stores/subscriptions.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { initStoreSubscriptions, _resetForTests } from "./subscriptions";

describe("auto-validate on composition-state version change", () => {
  beforeEach(() => {
    // _resetForTests() tears down the previous test's subscription and
    // clears all closure trackers, so the subsequent initStoreSubscriptions()
    // call rebuilds a fresh closure.  Without this, the module-level
    // teardown handle would short-circuit and the test would see stale
    // lastValidatedVersion / previousSessionId values.
    _resetForTests();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: null,
    } as never);
    useExecutionStore.setState({
      isValidating: false,
      isExecuting: false,
      validationResult: null,
    } as never);
    initStoreSubscriptions();
  });

  it("fires validate when compositionState.version increments", () => {
    const validate = vi.fn();
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).toHaveBeenCalledWith("sess-1");

    // Version increments again -- fires again.
    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).toHaveBeenCalledTimes(2);
  });

  it("does not fire when version is unchanged", () => {
    const validate = vi.fn();
    useExecutionStore.setState({ validate } as never);

    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
    } as never);
    expect(validate).toHaveBeenCalledTimes(1);

    // Same version, different reference -- should not refire.
    useSessionStore.setState({
      compositionState: { version: 5, source: null, nodes: [], outputs: [] } as never,
    } as never);
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("does not fire while a validation is already in-flight", () => {
    const validate = vi.fn();
    useExecutionStore.setState({ validate, isValidating: true } as never);

    useSessionStore.setState({
      compositionState: { version: 7, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).not.toHaveBeenCalled();
  });

  it("does not fire while an execution is in-flight", () => {
    const validate = vi.fn();
    useExecutionStore.setState({ validate, isExecuting: true } as never);

    useSessionStore.setState({
      compositionState: { version: 9, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).not.toHaveBeenCalled();
  });

  it("does not fire when activeSessionId is null", () => {
    const validate = vi.fn();
    useExecutionStore.setState({ validate } as never);
    useSessionStore.setState({ activeSessionId: null } as never);

    useSessionStore.setState({
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);

    expect(validate).not.toHaveBeenCalled();
  });

  it("resets lastValidatedVersion between test runs (cross-test isolation)", () => {
    // beforeEach calls _resetForTests() then initStoreSubscriptions().  A
    // fresh closure must start with lastValidatedVersion=null so version=1
    // fires again, even though a previous test in this describe block
    // already saw version=1.  If lastValidatedVersion were module-scoped
    // (or if _resetForTests did not actually tear down the closure), this
    // test would silently regress.
    const validate = vi.fn();
    useExecutionStore.setState({ validate } as never);
    useSessionStore.setState({
      activeSessionId: "sess-fresh",
      compositionState: { version: 1, source: null, nodes: [], outputs: [] } as never,
    } as never);
    expect(validate).toHaveBeenCalledWith("sess-fresh");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/subscriptions.test.ts -t "auto-validate"
```

Expected: FAIL.

- [ ] **Step 4: Implement the subscription**

`src/elspeth/web/frontend/src/stores/subscriptions.ts` currently keeps both `previousVersion` and `initialized` at module scope. That shape is the same defect as Phase 1A finding 7 (cross-phase finding §H7). Task 4 **rewrites the file** so that:

1. All previously module-scoped trackers (`previousVersion`, `initialized`, plus the new `lastValidatedVersion` and `previousSessionId`) live inside `initStoreSubscriptions()`'s closure.
2. The existing module-level `initialized` boolean is replaced by a closure-internal `started` flag.
3. The subscription's unsubscribe handle is held in the closure so `_resetForTests()` can tear it down.
4. A test-only `_resetForTests()` function is exported. The underscore prefix is the project's "test-only API" convention; no `@deprecated` annotation per "No Legacy Code Policy."
5. Previous-session tracking uses a **closure variable** (`previousSessionId`) updated at the tail of the subscription, **not** zustand's `subscribe((state, prevState) => ...)` overload — that overload requires the `subscribeWithSelector` middleware, which is not wired in the stores. The closure pattern matches the existing `previousVersion` precedent in this file.

The full rewrite of `subscriptions.ts`:

```typescript
// stores/subscriptions.ts
//
// Cross-store subscriptions extracted from executionStore to break the
// circular import between sessionStore and executionStore. Call
// initStoreSubscriptions() once at app startup (e.g. in App.tsx).
//
// All mutable state — start flag, previous-version tracker, previous
// session-id tracker, last-validated-version tracker, and the active
// subscription teardown — lives in the closure created by
// initStoreSubscriptions().  Module scope is intentionally bare so that
// _resetForTests() can rebuild the closure on every beforeEach.

import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";

let teardown: (() => void) | null = null;

export function initStoreSubscriptions(): void {
  if (teardown !== null) return; // already started

  // Closure-scoped trackers.  Each initStoreSubscriptions() call (after a
  // _resetForTests()) starts these afresh.
  let previousVersion: number | null = null;
  let previousSessionId: string | null = null;
  let lastValidatedVersion: number | null = null;

  const unsubscribe = useSessionStore.subscribe((state) => {
    const sessionId = state.activeSessionId;
    const currentVersion = state.compositionState?.version ?? null;

    // (1) Pre-existing behaviour: auto-clear validation when composition
    // version changes.  Moved from module scope into the closure.
    if (previousVersion !== null && currentVersion !== previousVersion) {
      useExecutionStore.getState().clearValidation();
    }
    previousVersion = currentVersion;

    // (2) New auto-validate behaviour.
    if (!sessionId) {
      lastValidatedVersion = null;
      previousSessionId = sessionId;
      return;
    }

    // Active session changed — reset the last-validated tracker so the
    // first version observed for the new session fires a validate().
    if (sessionId !== previousSessionId) {
      lastValidatedVersion = null;
      previousSessionId = sessionId;
    }

    if (currentVersion === null) return;
    if (currentVersion === lastValidatedVersion) return;

    const exec = useExecutionStore.getState();
    if (exec.isValidating || exec.isExecuting) return;

    lastValidatedVersion = currentVersion;
    // Fire-and-forget; rejection is intentionally not surfaced to a
    // diagnostic channel in this phase.  Per CLAUDE.md audit-primacy
    // policy, failures from validate() are already recorded in the audit
    // Landscape at the backend boundary; adding a frontend log/telemetry
    // breadcrumb here would duplicate the audit trail without adding
    // probative value.  Phase 8 (polish + telemetry) is the right owner
    // if a frontend operational signal proves useful.
    void exec.validate(sessionId);
  });

  teardown = () => {
    unsubscribe();
    previousVersion = null;
    previousSessionId = null;
    lastValidatedVersion = null;
  };
}

/**
 * Test-only: tear down the active subscription and clear all closure
 * trackers so the next initStoreSubscriptions() call rebuilds a fresh
 * closure.  Underscore prefix marks this as not-for-runtime-use.
 */
export function _resetForTests(): void {
  if (teardown) {
    teardown();
    teardown = null;
  }
}
```

Notes:

- `previousSessionId` is a **closure variable updated at the tail of the subscription**. This is the operator-adjudicated alternative to `useSessionStore.subscribe((state, prevState) => ...)`, which would require `subscribeWithSelector` middleware that the project does not currently wire.
- The rejection from `validate()` is **not** surfaced to a frontend telemetry channel. No `telemetry.record(...)` call is made. The backend's audit Landscape already records validate failures; the frontend would only duplicate that record. Operator adjudication 2026-05-15.
- The existing single-arg `useSessionStore.subscribe((state) => ...)` precedent at the top of the previous version of this file confirms the signature.

- [ ] **Step 5: Run all tests**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/subscriptions.ts \
        src/elspeth/web/frontend/src/stores/subscriptions.test.ts
git commit -m "feat(web/frontend): auto-validate on composition-state version increment (Phase 3A.4)"
```

---

**Tasks 5–8 (the removal tasks) continue in [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md).**

---

## Review history

### 2026-05-15 — Review panel applied

**BLOCKER (Quality+Systems):** `lastValidatedVersion`, the pre-existing `previousVersion`, and a new `previousSessionId` tracker all moved from module scope into `initStoreSubscriptions()`'s closure. The module-level `initialized` boolean is replaced by a module-level `teardown` handle; a test-only `_resetForTests()` tears down the active subscription and rebuilds the closure between tests. Cross-test isolation test in Task 4 Step 2 calls `_resetForTests()` in `beforeEach`. Same shape as Phase 1A finding 7.

**CRITICAL (Reality+Systems):** Operator adjudication 2026-05-15 — Task 4 Step 4's `telemetry.record(...)` call is **deleted**, not retained with a different channel. No frontend telemetry module exists; per CLAUDE.md audit-primacy ("logger is NOT for pipeline activity"; primacy order audit → telemetry → log), validate-failure breadcrumbs in the frontend would duplicate the backend audit Landscape without adding probative value. Phase 8 (polish + telemetry) is the right owner if a frontend operational signal proves useful later. `void exec.validate(sessionId)` is the fire-and-forget shape.

**CRITICAL (Reality):** Operator adjudication 2026-05-15 — the previous draft used `useSessionStore.subscribe((state, prevState) => ...)`. Zustand's default `subscribe` is single-arg; the two-arg overload requires `subscribeWithSelector` middleware, which is not wired in the stores. Replaced with a closure-internal `previousSessionId` updated at the tail of the subscription. Matches the existing `previousVersion` precedent.

**IMPORTANT (Architecture):** Task 2 scaffold updated to use render-props (`auditReadinessSlot`, `executeButtonSlot`, etc.) instead of empty-div placeholders. `SideRailProps` now explicitly declares all named slots as `ReactNode | null`. `executeButtonSlot` is added here (required by 15b Task 2 before the inspector deletion). The composition contract is explicit: callers pass content via props from `App.tsx`; `SideRail` does not mount components by event or by its own internal logic.

**Cross-file decision:** `SWITCH_TAB_EVENT` — IMPORTANT (15b2 §4) overrides SUGGESTION (15b1 §3); the constant is deleted in Phase 3B, not deferred to Phase 6. See 15b1 and 15b2 review history entries.
