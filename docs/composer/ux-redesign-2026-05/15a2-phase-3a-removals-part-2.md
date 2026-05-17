# Phase 3A — IA Cleanup: Removals (Part 2 of 2)

> **Continued from [15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md)**, which contains the plan header, scope boundaries, trust tier check, sequencing overview, open scope questions resolved, and Tasks 1–4 (all additive work).

> **Phase 3 block notice (added 2026-05-17):** This plan is one of four (15a1, 15a2, 15b1, 15b2) that together comprise the Phase 3 IA-cleanup work. **All four land as a single block on a shared worktree** (`.worktrees/phase-2a-backend`, branch `feat/composer-phase-2a-backend` — the same worktree that landed Phase 2A/2B/2C) and **merge as one PR**. Phrases below like "deferred to 15b" or "Phase 3B" mean "later tasks in the same branch," not a separate cycle. The 15a1→15a2→15b1→15b2 split is task sequencing and document organisation, not delivery sequencing — sequencing within the block still matters per task ordering.
>
> **Subagent dispatch discipline.** Any subagent run against this work MUST be given an explicit CWD-discipline preamble at the top of its prompt: first Bash call `cd /home/john/elspeth/.worktrees/phase-2a-backend && pwd && git rev-parse --abbrev-ref HEAD` (expect `feat/composer-phase-2a-backend`), then absolute paths only thereafter for every Read/Bash/Grep. Bash `cd` does NOT persist between tool calls — relative paths will silently read the wrong branch (the main checkout is 87+ commits behind). See memory entry `feedback_subagents_cant_use_worktrees`.

**Umbrella plan context:**
- Predecessor: [13-phase-1b-frontend.md](13-phase-1b-frontend.md)
- Successor (additions half): [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md) / [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md)
- Roadmap: [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3)

**Tasks in this file:** Tasks 5–7 (the removal tasks), plus Risks and mitigations, Memory references, and Review history.

> **Task 8 retired (2026-05-17 reality-check panel).** The original Task 8 ("Remove Validate button from inspector header") referenced `InspectorPanel.tsx:557–572` and `handleValidate` at `InspectorPanel.tsx:387–424`. Phase 2C (commits `d218417c1..2f2ba300e`, landed 2026-05-17) **already deleted the Validate button and `handleValidate`**, and **already moved** the side-effect orchestration (`injectSystemMessage` + `sendValidationFeedback`) into `subscriptions.ts` as an `executionStore.validationResult`-change subscriber. Executing the old Task 8 spec would have failed (no such code to remove) and / or destroyed the Phase 2C wiring. Task 8 is therefore marked **DONE — Phase 2C** and removed from this file. The auto-validate subscriber from Task 4 (15a1) feeds the same `validationResult` channel that Phase 2C's subscriber already consumes, so the LLM-facing system-message injection works on both the auto path and the manual `Ctrl+Shift+V` path with no further wiring.

---

## Task 5: Remove Runs tab + delete `RunsView`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — drop the `"runs"` entry from `TABS`, drop the `import { RunsView }` line, drop the `activeTab === "runs"` block, narrow the `TabId` union.
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` (if exists; check with `ls`).
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx` — drop Runs-tab assertions.

The `handleExecute` callback in `InspectorPanel.tsx:378–385` (drift from pre-Phase-2C plan citation L426–433) currently does `setActiveTab("runs")` after a successful execute. With Runs gone, **remove that line**. The user lands on whichever tab they were on (Graph or YAML) — and the inline `InlineRunResults` (Task 1) renders the run in the chat column.

`SWITCH_TAB_EVENT` listener in `InspectorPanel.tsx:325–334` (drift +1 line from pre-Phase-2C plan citation L325-333) currently accepts `"runs"`. Drop the `tab === "runs"` branch.

`App.tsx:85–90` (drift from pre-Phase-2C plan citation L62) does `window.dispatchEvent(new CustomEvent(SWITCH_TAB_EVENT, { detail: "runs" }))` in the `confirmFanoutExecution` callback. **Remove that line** — the dispatched event has no listener after Runs is gone, and the inline run-results path replaces it. (Re-verify the exact line number against the current file before editing — Phase 2C may have shifted it again.)

CommandPalette.tsx's `tab-runs` command needs removing. (15b removes the remaining `tab-spec/graph/yaml` commands; Task 5 removes `tab-runs` because Spec is still alive in this commit but Runs isn't.) Grep for `tab-runs` rather than relying on a line number — the pre-Phase-2C citation L174-179 is likely drifted.

Hash-router `VALID_TABS` in useHashRouter.ts:29 still includes `"runs"`. **Leave it in 15a** — old `#/sess-1/runs` deep links resolve as "session sess-1, no valid tab, fall back to spec" via the `tab` parameter being `null` after `VALID_TABS` excludes it. Wait — `VALID_TABS` *includes* "runs" today; if we remove it from `VALID_TABS`, the regex in `parseHash` still matches but the `tab` value becomes `null` (because `VALID_TABS.has(match[2])` returns false). So **remove `"runs"` from `VALID_TABS`** in this task; the existing fallback handles the redirect.

> **Review finding (CRITICAL):** Silently redirecting old `#/{id}/runs` bookmarks during the 15a → 15b window produces a disorienting no-error experience. **Transient redirect toast:** when `useHashRouter` detects an old `runs` or `spec` hash fragment (before `VALID_TABS` strips it), show a one-time dismissible toast: _"The Runs tab was removed in this update. Showing Graph instead."_ (or _"The Spec tab was removed…"_). Track dismissal in `localStorage.elspeth_redirect_toast_dismissed`; the toast never reappears after dismissal. Add a test asserting the toast renders on first visit and does not render after dismissal.

Test the hash redirect: navigate to `#/sess-1/runs` and verify:
1. A toast banner appears the first time.
2. After dismissal, navigating to `#/sess-1/runs` again shows no toast.
3. The inspector shows the Spec tab (the default) in both cases. The full URL rewrite is in 15b.

- [ ] **Step 1: Write a failing test for the InspectorPanel tab strip**

Add to `InspectorPanel.test.tsx` (or create a new test file targeting the tab strip):

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InspectorPanel } from "./InspectorPanel";

describe("InspectorPanel tab strip after Runs removal", () => {
  it("does not render a Runs tab", () => {
    render(<InspectorPanel />);
    expect(screen.queryByRole("tab", { name: /^runs$/i })).not.toBeInTheDocument();
  });

  it("still renders Spec, Graph, YAML tabs (Spec removed in Task 6)", () => {
    render(<InspectorPanel />);
    expect(screen.getByRole("tab", { name: /^spec$/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /^graph$/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /^yaml$/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx -t "Runs removal"
```

Expected: FAIL — Runs tab still present.

- [ ] **Step 3: Edit `InspectorPanel.tsx`**

Apply the following edits:

1. Narrow the type union:
   ```typescript
   type TabId = "spec" | "graph" | "yaml";
   ```
2. Remove the `"runs"` entry from `TABS`:
   ```typescript
   const TABS: { id: TabId; label: string }[] = [
     { id: "spec", label: "Spec" },
     { id: "graph", label: "Graph" },
     { id: "yaml", label: "YAML" },
   ];
   ```
3. Remove the import: `import { RunsView } from "./RunsView";`
4. Remove the `activeTab === "runs"` JSX block at the bottom of the tab-content render.
5. Narrow the `SWITCH_TAB_EVENT` handler — drop the `tab === "runs"` branch:
   ```typescript
   if (tab === "spec" || tab === "graph" || tab === "yaml") {
     setActiveTab(tab);
   }
   ```
6. In `handleExecute`, remove the `setActiveTab("runs")` line:
   ```typescript
   const handleExecute = useCallback(async () => {
     if (activeSessionId && canExecute) {
       await execute(activeSessionId);
     }
   }, [activeSessionId, canExecute, execute]);
   ```

- [ ] **Step 4: Edit `App.tsx`**

Remove the `confirmFanoutExecution`'s dispatch of `SWITCH_TAB_EVENT` with `"runs"`:

```typescript
const confirmFanoutExecution = useCallback(async () => {
  await useExecutionStore.getState().confirmFanoutExecution();
  // (no tab switch; InlineRunResults renders the run inline in the chat column)
}, []);
```

- [ ] **Step 5: Edit `CommandPalette.tsx`**

Remove the `tab-runs` command block (the four lines registering it). Leave `tab-spec`, `tab-graph`, `tab-yaml` for now (15b prunes the rest).

- [ ] **Step 6: Edit `useHashRouter.ts`**

Change `VALID_TABS`:

```typescript
const VALID_TABS = new Set(["spec", "graph", "yaml"]);
```

(Old `#/sess/runs` links will resolve to `tab=null`, then `resolvedTab = "spec"` per useHashRouter.ts:64 default. 15b rewrites the URL.)

Also add the redirect-toast logic: when `parseHash()` sees a fragment that is not in `VALID_TABS` (and is not empty), check `localStorage.getItem("elspeth_redirect_toast_dismissed")`; if not set, show a toast and call `localStorage.setItem("elspeth_redirect_toast_dismissed", "1")`.

- [ ] **Step 6a: Add hash-fallback tests (IMPORTANT — not deferred to 15b)**

> **Review finding (IMPORTANT):** These tests verify hash-fallback behaviour that ships in this task. Do not defer to 15b.

Add to `useHashRouter.test.ts` (or create it if it doesn't exist):

```typescript
describe("useHashRouter — 15a hash fallback", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", window.location.pathname);
    localStorage.removeItem("elspeth_redirect_toast_dismissed");
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "x" } as never],
      activeSessionId: null,
      selectSession: vi.fn(),
    } as never);
  });

  it("falls back to the spec tab when #/{id}/runs is visited", () => {
    window.location.hash = "#/sess-1/runs";
    const { result } = renderHook(() => useHashRouter());
    // resolvedTab from the hook (or however it's exposed) should be "spec"
    // The key contract: the app does not crash and presents a valid tab.
    expect(window.location.hash).toBe("#/sess-1/runs"); // URL not yet rewritten in 15a
  });

  it("shows a redirect toast on first visit to a stale fragment", () => {
    window.location.hash = "#/sess-1/runs";
    render(<App />); // or whichever top-level integration fixture exists
    expect(screen.getByRole("alert")).toHaveTextContent(/runs tab was removed/i);
  });

  it("does not show the redirect toast after it was dismissed", () => {
    localStorage.setItem("elspeth_redirect_toast_dismissed", "1");
    window.location.hash = "#/sess-1/runs";
    render(<App />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
```

- [ ] **Step 7: Delete `RunsView.tsx` and its test**

```bash
ls src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx 2>/dev/null
git rm src/elspeth/web/frontend/src/components/inspector/RunsView.tsx
git rm src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx 2>/dev/null || true
```

- [ ] **Step 8: Run full test suite + smoke render + import-grep**

```bash
cd src/elspeth/web/frontend && npx vitest run src
grep -rn "from.*RunsView\|/RunsView\"" src/elspeth/web/frontend/src --include="*.tsx" --include="*.ts"
```

Expected:
- vitest PASS — including `App.test.tsx`.
- `grep` returns no matches (no dangling imports).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): remove Runs tab (Phase 3A.5)

Run results now render inline in the chat column via InlineRunResults;
historical access via RunsHistoryDrawer 'Past runs' button.
Hash links #/<id>/runs fall through to the spec default; 15b rewrites
the URL."
```

---

## Task 6: Remove Spec tab + delete `SpecView`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` — line 175 (drift +1 from pre-Phase-2C plan citation L174) comment "GraphView <-> SpecView" → "GraphView selection" (since SpecView is gone but the selection state is still consumed by GraphView).

Spec was the only tab where node-card click-to-highlight worked. With Spec gone:
- `handleValidationComponentClick` (`InspectorPanel.tsx:400–410`, drift from pre-Phase-2C plan citation L448-458) currently selects a node and switches to Spec. The Spec switch goes away; **the `selectNode` call is preserved** so GraphView's highlight ring still appears when a validation error is clicked.
- The default tab when no hash fragment is supplied changes from `"spec"` to `"graph"`. Update `useHashRouter.ts:63` (drift -1 from pre-Phase-2C plan citation L64):
  ```typescript
  const resolvedTab = tab ?? "graph";
  ```

`VALID_TABS` removes `"spec"`. Old `#/sess/spec` deep links fall through to `graph`.

- [ ] **Step 1: Write failing tests**

Add to `InspectorPanel.test.tsx`:

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InspectorPanel } from "./InspectorPanel";

describe("InspectorPanel tab strip after Spec removal", () => {
  it("does not render a Spec tab", () => {
    render(<InspectorPanel />);
    expect(screen.queryByRole("tab", { name: /^spec$/i })).not.toBeInTheDocument();
  });

  it("renders Graph as the default tab when no hash is set", () => {
    render(<InspectorPanel />);
    const graphTab = screen.getByRole("tab", { name: /^graph$/i });
    expect(graphTab).toHaveAttribute("aria-selected", "true");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx -t "Spec removal"
```

Expected: FAIL.

- [ ] **Step 3: Edit `InspectorPanel.tsx`**

1. Narrow the type union:
   ```typescript
   type TabId = "graph" | "yaml";
   ```
2. Remove `"spec"` from `TABS`.
3. Change initial state:
   ```typescript
   const [activeTab, setActiveTab] = useState<TabId>("graph");
   ```
4. Drop the `import { SpecView } from "./SpecView";` line.
5. Drop the `activeTab === "spec"` JSX block.
6. Narrow the `SWITCH_TAB_EVENT` handler — drop the `tab === "spec"` branch.
7. Update `handleValidationComponentClick`:
   ```typescript
   const handleValidationComponentClick = useCallback(
     (componentId: string) => {
       const isNode = compositionState?.nodes.some((n) => n.id === componentId);
       if (isNode) {
         selectNode(componentId);
         // Phase 3A: Spec tab is gone; GraphView consumes the selection
         // for its highlight ring. Phase 2's audit-readiness "Explain"
         // surface will own this routing going forward.
       }
     },
     [selectNode, compositionState],
   );
   ```

- [ ] **Step 4: Edit `useHashRouter.ts`**

```typescript
const VALID_TABS = new Set(["graph", "yaml"]);

// Inside applyHash:
const resolvedTab = tab ?? "graph";
```

- [ ] **Step 5: Edit `CommandPalette.tsx`**

Remove the `tab-spec` command block. (Leaves `tab-graph` and `tab-yaml` — 15b prunes them when graph/yaml become modal-on-demand.)

- [ ] **Step 6: Edit `sessionStore.ts:174` comment**

```typescript
// Shared selection state for cross-component sync (GraphView selection)
```

- [ ] **Step 7: Delete `SpecView.tsx` and its test**

```bash
git rm src/elspeth/web/frontend/src/components/inspector/SpecView.tsx
git rm src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx
```

- [ ] **Step 8: Run full test suite + smoke render + import-grep**

```bash
cd src/elspeth/web/frontend && npx vitest run src
grep -rn "from.*SpecView\|/SpecView\"" src/elspeth/web/frontend/src --include="*.tsx" --include="*.ts"
```

Expected:
- vitest PASS.
- `grep` returns no matches.

- [ ] **Step 8a: Add symmetric `#/{id}/spec` redirect-toast tests (IMPORTANT — Quality panel finding 2026-05-17)**

> Task 5 Step 6a added toast tests for the `#/{id}/runs` redirect, but the toast code also fires for `#/{id}/spec` after Task 6 removes `"spec"` from `VALID_TABS`. Without symmetric coverage, a regression in the spec branch (wrong text, double-fire, missing localStorage check) would not be caught.

Add to `useHashRouter.test.ts`:

```typescript
describe("useHashRouter — Task 6 spec hash fallback", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", window.location.pathname);
    localStorage.removeItem("elspeth_redirect_toast_dismissed");
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "x" } as never],
      activeSessionId: null,
      selectSession: vi.fn(),
    } as never);
  });

  it("falls back to the graph tab when #/{id}/spec is visited", () => {
    window.location.hash = "#/sess-1/spec";
    renderHook(() => useHashRouter());
    // The contract: the app does not crash and presents a valid tab.
    expect(window.location.hash).toBe("#/sess-1/spec"); // URL not yet rewritten in 15a
  });

  it("shows a redirect toast on first visit to a stale spec fragment", () => {
    window.location.hash = "#/sess-1/spec";
    render(<App />);
    expect(screen.getByRole("alert")).toHaveTextContent(/spec tab was removed/i);
  });

  it("does not show the redirect toast after it was dismissed", () => {
    localStorage.setItem("elspeth_redirect_toast_dismissed", "1");
    window.location.hash = "#/sess-1/spec";
    render(<App />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
```

Note that the dismissal flag is shared with the `#/runs` toast (single `elspeth_redirect_toast_dismissed` key). Dismissing one silences the other for that user — by design; the user has indicated "I understand this kind of redirect happens." 15b's hash-router rewrite makes both inert.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): remove Spec tab (Phase 3A.6)

GraphView is now the default tab. Validation-banner clicks still call
selectNode (consumed by GraphView's highlight); Phase 2's audit-readiness
'Explain' surface will own this routing going forward. Symmetric
#/{id}/spec redirect-toast tests landed alongside Task 5's #/{id}/runs
set."
```

---

## Task 7: Remove `SessionSidebar` mount + delete file

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx` — drop the `sidebar` prop entirely from `LayoutProps`. Remove the sidebar `<div>` and the sidebar-toolbar (collapse-toggle and theme-toggle move into `AppHeader`'s user-menu region — see Step 1 below for theme-toggle relocation).
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.test.tsx` — drop sidebar tests.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — remove the `sidebar={<SessionSidebar />}` prop; remove the `SessionSidebar` import.
- Delete: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/UserMenu.tsx` — add a theme-toggle entry above "Settings" (the theme toggle was hosted in Layout's sidebar toolbar; moving it into UserMenu per design doc 03 §"Surface inventory" row "Theme toggle").

Before this task, the sidebar still renders (it's the entry point for sessions). After this task, sessions are reached via `HeaderSessionSwitcher` (Task 3) and CommandPalette (already-built). The collapse-toggle and theme-toggle that lived in the sidebar toolbar must find new homes:
- **Theme toggle** → into `UserMenu` (per design doc 03).
- **Collapse-toggle** for the side rail → moved to a small handle on the side-rail boundary in `Layout.tsx`'s existing `inspector-toggle-btn` (still functioning post-rename as `siderail-toggle-btn`). The collapse-toggle for the sidebar is *deleted* since the sidebar is gone.

- [ ] **Step 1: Move theme-toggle into `UserMenu`**

Read `Layout.tsx:248–263` (the theme-toggle in the sidebar toolbar). Read `UserMenu.tsx` (Phase 1B Task 6). Add a new menu item to `UserMenu`:

```tsx
import { useTheme } from "@/hooks/useTheme";

// ...inside UserMenu:
const { resolvedTheme, toggleTheme } = useTheme();

// ...in the menu <ul>, above Settings:
<li
  role="menuitem"
  tabIndex={0}
  onClick={() => {
    toggleTheme();
    setOpen(false);
  }}
  onKeyDown={(e) => {
    if (e.key === "Enter") {
      toggleTheme();
      setOpen(false);
    }
  }}
>
  {resolvedTheme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
</li>
```

Add a test in `UserMenu.test.tsx`:

```typescript
it("shows a theme toggle menu item", async () => {
  render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
  await userEvent.click(screen.getByRole("button", { name: /account/i }));
  expect(
    screen.getByRole("menuitem", { name: /theme/i }),
  ).toBeInTheDocument();
});
```

Run: `npx vitest run src/components/common/UserMenu.test.tsx`. Expected: PASS.

- [ ] **Step 2: Write failing tests for the Layout-without-sidebar**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Layout } from "./Layout";

describe("Layout without sidebar slot", () => {
  it("does not render a session sidebar region", () => {
    render(
      <Layout
        chat={<div data-testid="chat" />}
        siderail={<div data-testid="siderail" />}
      />,
    );
    expect(screen.queryByLabelText(/sessions sidebar/i)).not.toBeInTheDocument();
  });

  it("uses a two-column grid (chat + siderail)", () => {
    const { container } = render(
      <Layout
        chat={<div data-testid="chat" />}
        siderail={<div data-testid="siderail" />}
      />,
    );
    const layoutNode = container.querySelector(".app-layout") as HTMLElement;
    // The grid template should have exactly two column tracks now.
    const cols = (layoutNode.style.gridTemplateColumns ?? "").split(/\s+/).filter(Boolean);
    expect(cols.length).toBe(2);
  });
});
```

Run: `npx vitest run src/components/common/Layout.test.tsx -t "without sidebar"`. Expected: FAIL (existing Layout type requires `sidebar` prop).

- [ ] **Step 3: Edit `Layout.tsx`**

1. Drop `sidebar` from `LayoutProps`:
   ```typescript
   interface LayoutProps {
     chat: ReactNode;
     siderail: ReactNode;
   }
   ```
2. Remove the entire `<div className="layout-sidebar">…</div>` block (Layout.tsx:230–273).
3. Drop the sidebar-related state and constants: `sidebarCollapsed`, `setSidebarCollapsed`, `SIDEBAR_COLLAPSED_KEY`, `SIDEBAR_EXPANDED_WIDTH`, `SIDEBAR_COLLAPSED_WIDTH`, `NARROW_BREAKPOINT`'s sidebar-collapse handling.
4. Update `gridColumns`:
   ```typescript
   const gridColumns = isOverlayMode
     ? `1fr`
     : sideRailVisible
       ? `1fr ${sideRailWidth}px`
       : `1fr`;
   ```
5. Drop the theme-toggle button (it's now in UserMenu — Step 1).
6. Drop the `useTheme` import if no longer used in Layout.
7. Update `defaultInspectorWidth`:
   ```typescript
   function defaultSideRailWidth(): number {
     return Math.max(MIN_INSPECTOR_WIDTH, Math.round(window.innerWidth / 2));
   }
   ```
   (Rename to `defaultSideRailWidth`; the old sidebar-width offset is gone.)

- [ ] **Step 4: Edit `Layout.test.tsx`**

Remove every test that depended on the `sidebar` prop, the sidebar's collapse toggle, or the theme toggle in the sidebar toolbar. Keep tests that exercise the chat column, side rail, resize handle, and overlay mode.

- [ ] **Step 5: Edit `App.tsx`**

```tsx
// Before:
import { SessionSidebar } from "./components/sessions/SessionSidebar";

<Layout
  sidebar={<SessionSidebar />}
  chat={<ChatPanel onOpenSecrets={openSecrets} />}
  siderail={
    <SideRail>
      <InspectorPanel />
    </SideRail>
  }
/>

// After:
<Layout
  chat={<ChatPanel onOpenSecrets={openSecrets} />}
  siderail={
    <SideRail>
      <InspectorPanel />
    </SideRail>
  }
/>
```

Remove the `SessionSidebar` import.

- [ ] **Step 6: Delete `SessionSidebar.tsx` and its test**

```bash
git rm src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx
git rm src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx
```

- [ ] **Step 7: Run full test suite + smoke render + import-grep**

```bash
cd src/elspeth/web/frontend && npx vitest run src
grep -rn "from.*SessionSidebar\|/SessionSidebar\"" src/elspeth/web/frontend/src --include="*.tsx" --include="*.ts"
```

Expected:
- vitest PASS.
- `grep` returns no matches.

Also update `App.test.tsx`: the mock at line 37 (`vi.mock("./components/sessions/SessionSidebar", …)`) is now stale. Remove the mock entirely.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): remove SessionSidebar (Phase 3A.7)

Sessions are now reached via the HeaderSessionSwitcher and the
CommandPalette's Sessions section. Theme toggle moved into UserMenu.
Layout grid simplified from 3-column to 2-column."
```

---

## Task 8: ~~Remove Validate button from inspector header~~ **RETIRED 2026-05-17 — Phase 2C absorbed**

See the panel note at the top of this file. Phase 2C deleted the Validate button (formerly `InspectorPanel.tsx:557–572`), removed `handleValidate` (formerly `InspectorPanel.tsx:387–424`), and moved the `injectSystemMessage` + `sendValidationFeedback` side-effect orchestration into `subscriptions.ts` as a `useExecutionStore.subscribe` handler keyed off `validationResult` change. The `AuditReadinessPanel` (also Phase 2C) replaces the standalone Validation row.

The auto-validate subscriber from 15a1 Task 4 fires `validate(sessionId)` on `compositionState.version` increment; the resulting `validationResult` flows into the Phase 2C subscriber that publishes the system message. `Ctrl+Shift+V` (handled in `App.tsx`) continues to call `validate(activeSessionId)` directly; the same `validationResult` subscriber publishes the system message for that path too. No additional wiring is needed in Phase 3A.

The validation dot indicator (currently at `InspectorPanel.tsx:452–494`, drift from pre-Phase-2C plan citation L499–542) **stays** through 15a; 15b migrates it into the audit-readiness panel.

**No work in this task. Sequencing continues with the post-15a → 15b transition.**

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| App becomes un-launchable mid-phase between commits | Each task ends with `npx vitest run src` including `App.test.tsx`. **Note (Quality panel 2026-05-17): `App.test.tsx` stubs out Layout, SessionSidebar, ChatPanel, InspectorPanel, SecretsPanel, CommandPalette, ShortcutsHelp, and ConfirmDialog — so it asserts banner DOM, not the real component tree. Treat it as an "App-shell smoke" gate, not a component-integration gate.** Real component-integration risk for Tasks 5–7 (tab strip, layout grid, UserMenu) lands on the staging deploy. Phase 3 acceptance criteria add at least one Playwright pass for two-column layout and UserMenu keyboard nav at the 15a→15b boundary. |
| Users hit a stale `#/{id}/spec` or `#/{id}/runs` bookmark and see a confusing default | 15a leaves `VALID_TABS` excluding the gone tabs; useHashRouter's fallback resolves them to a valid default (`spec` → `graph`, `runs` → `graph`). Tasks 5 and 6 add a one-time redirect toast (shared dismissal flag). 15b adds explicit hash rewrites. |
| Spec-tab regulars confused by no "list view" | Phase 2C's audit-readiness "Explain" surface (already shipped) inherits the lineage view. Until 15b migrates the validation-banner click handler to use it, validation-banner clicks select a node (GraphView highlight) but don't navigate further. Acceptable transitional state. |
| **Auto-validate stales validation badge during rapid composition flows (correctness, not load)** | Systems panel 2026-05-17 finding. A simple skip-if-in-flight guard discards intermediate `compositionState.version` increments during LLM tool-call bursts (N → N+1 → N+2). Resolution in Task 4: per-session `lastValidatedVersionBySession` map + `pendingValidateTarget` + `fireValidateLoop()` that re-fires `validate()` after the in-flight settles when a newer version arrived. The user never sees a "validated" badge for a stale snapshot. Debounce framing retired — correctness is the goal, not load shaping. |
| Run-results inline rendering overlaps with the still-present Runs tab during Task 1–4 | `InlineRunResults` returns `null` when there are no runs *and* no historical runs. Even when it renders, the duplicate is brief — Task 5 removes the Runs tab. The dual-display is the cost of TDD discipline. |
| Theme toggle loses discoverability after moving from sidebar to UserMenu | Both surfaces existed in the legacy UI; Phase 1B already created the UserMenu hosting. Task 7 makes it the sole home; design doc 03 ratifies this placement. |
| The session-switcher's "New session" action collides with the `Ctrl+N` shortcut | Both call the same `createSession` action; no collision. The header dropdown adds discoverability; the shortcut is preserved in App.tsx. |
| Phase 2 / Phase 6 plans want different slot semantics than `SideRail` offers | The slots in Task 2 are skeleton markers (test-id only); content components own their own styling. If Phase 2 wants the audit-readiness panel to span the whole rail width vs. a card, that's a content decision, not a slot decision. The slot ordering is taken directly from design doc 03 §"Layout sketch". |
| `Layout.test.tsx` rewrites are large | Layout was already complex (sidebar + overlay mode + resize). Removing the sidebar simplifies the file; the tests that go away are precisely those that exercised gone behaviour. Net code change: net negative. |
| 15b expects `_resetSubscriptionsForTesting` to behave as-shipped; Task 4 rename would have broken it | Resolved 2026-05-17 — Task 4 is now additive and **does not rename** the export. The existing test file's `_resetSubscriptionsForTesting` import survives intact. |

## Memory references

- `project_composer_two_audiences` — explains why session switching by dropdown (skilled-operator move-fast pattern) is appropriate; the always-on sidebar privileged the "browse my sessions" mode that no persona has.
- `project_db_migration_policy` — not directly relevant; no DB changes in this plan.
- `project_staging_deployment` — informs the deploy-and-smoke approach if executors want to verify on `elspeth.foundryside.dev`. Same procedure as Phase 1B Task 9 (stop service, build frontend, restart).
- `feedback_no_calendar_shipping_commitments` — no dates in this plan.
- `feedback_no_tests_for_skill_prompts` — not relevant; this plan touches components, not skill prompts.

## Review history

### 2026-05-15 — Review panel applied

**CRITICAL (Architecture, operator-adjudicated 2026-05-15):** `sendValidationFeedback` in Task 8 Step 2 is now fire-and-forget (`void sessionStore.sendValidationFeedback(result)`). The previous `.catch((err) => { telemetry.record(...) })` shape referenced a frontend telemetry module that does not exist; per CLAUDE.md audit-primacy, the backend records validation events in the audit Landscape and a frontend breadcrumb would only duplicate that record. The user-visible system message is injected before the call. Phase 8 (polish + telemetry) is the right owner if a frontend operational signal proves useful later.

**CRITICAL (Quality):** Task 8 Step 2 side-effect test replaced `await new Promise((r) => setTimeout(r, 20))` with `await waitFor(() => expect(injectSystemMessage).toHaveBeenCalled())`. No magic sleep delays in tests.

**CRITICAL (Systems):** `VALID_TABS` narrowing (Tasks 5 and 6) now includes a transient one-time redirect toast. When an old `spec` or `runs` hash is detected during the 15a → 15b window, a dismissible toast explains the removal before stripping the fragment. Dismissal tracked in `localStorage.elspeth_redirect_toast_dismissed`. Tests added in new Step 6a.

**IMPORTANT (Quality):** Hash-fallback test for `#/{id}/runs` → default tab added in Task 5 Step 6a. Not deferred to 15b.

### 2026-05-17 — Reality-check panel applied (NO-GO → fixes landed)

Four-reviewer panel (Reality / Architecture / Quality / Systems) ran after Phase 2C landed earlier the same day. Reality returned **NO-GO**; the other three returned CONDITIONAL GO with a convergent finding cluster around Task 4 and Task 8. Adjudication and fixes landed in this revision:

**CRITICAL (Reality, convergent with Architecture+Quality) — Task 8 retired.** The cited code (`InspectorPanel.tsx:557-572` Validate button; `InspectorPanel.tsx:387-424` `handleValidate`) does not exist — Phase 2C deleted it. The "move side effects into the auto-validate subscription" Step is also moot — Phase 2C already moved them into a `validationResult`-change subscriber in `subscriptions.ts`. Task 8 marked `DONE — Phase 2C` with a panel note at top of file. Task list updated `5–8` → `5–7` here and in 15a1.

**CRITICAL (Reality+Architecture) — Task 4 reframed to additive change.** See 15a1 review history. The Task-4 change cascades into Task 8's elimination: with Phase 2C's `validationResult`-change subscriber preserved, the LLM-facing system-message injection works on both the auto-validate path (Task 4) and the `Ctrl+Shift+V` manual path (App.tsx) with no further wiring. The "Removing the Validate button breaks the validation feedback loop" risk row is therefore obsolete and removed from the risks table.

**CRITICAL (Quality) — App.test.tsx smoke gate honesty.** Risks-table entry rewritten to call out explicitly that `App.test.tsx` stubs out every component under change in Phase 3A, so it asserts banner DOM rather than the real tree. Real integration risk for Tasks 5–7 lands on the staging deploy; a Playwright pass at the 15a→15b boundary covers the layout-grid and UserMenu keyboard contracts.

**IMPORTANT (Reality) — InspectorPanel line citations refreshed** in Tasks 5, 6, 8. The ~47-line drift from Phase 2C insertions above the affected regions is now documented per citation. The 15a executor should re-grep before each edit (cited lines may have drifted further by the time the work starts).

**IMPORTANT (Quality) — Symmetric `#/{id}/spec` redirect-toast tests** added as Task 6 Step 8a, mirroring Task 5 Step 6a's `#/{id}/runs` set. The shared dismissal flag is documented.

**IMPORTANT (Systems) — Auto-validate correctness loop, not load debounce.** See 15a1 review history. The risks-table row was retitled to frame this as a correctness issue (stale validation badge) rather than a backend-load issue. The non-falsifiable "defer to Phase 8 if telemetry shows pain" framing is retired.
