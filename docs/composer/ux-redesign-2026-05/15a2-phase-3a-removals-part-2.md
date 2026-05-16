# Phase 3A — IA Cleanup: Removals (Part 2 of 2)

> **Continued from [15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md)**, which contains the plan header, scope boundaries, trust tier check, sequencing overview, open scope questions resolved, and Tasks 1–4 (all additive work).

**Umbrella plan context:**
- Predecessor: [13-phase-1b-frontend.md](13-phase-1b-frontend.md)
- Successor (additions half): [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md) / [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md)
- Roadmap: [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3)

**Tasks in this file:** Tasks 5–8 (the removal tasks), plus Risks and mitigations, Memory references, and Review history.

---

## Task 5: Remove Runs tab + delete `RunsView`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — drop the `"runs"` entry from `TABS`, drop the `import { RunsView }` line, drop the `activeTab === "runs"` block, narrow the `TabId` union.
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` (if exists; check with `ls`).
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx` — drop Runs-tab assertions.

The `handleExecute` callback in InspectorPanel.tsx:426–433 currently does `setActiveTab("runs")` after a successful execute. With Runs gone, **remove that line**. The user lands on whichever tab they were on (Graph or YAML) — and the inline `InlineRunResults` (Task 1) renders the run in the chat column.

`SWITCH_TAB_EVENT` listener in InspectorPanel.tsx:325–333 currently accepts `"runs"`. Drop the `tab === "runs"` branch.

`App.tsx:62` does `window.dispatchEvent(new CustomEvent(SWITCH_TAB_EVENT, { detail: "runs" }))` in the `confirmFanoutExecution` callback. **Remove that line** — the dispatched event has no listener after Runs is gone, and the inline run-results path replaces it.

CommandPalette.tsx:174–179's `tab-runs` command needs removing. (15b removes the remaining `tab-spec/graph/yaml` commands; Task 5 removes `tab-runs` because Spec is still alive in this commit but Runs isn't.)

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
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` — line 174 comment "GraphView <-> SpecView" → "GraphView selection" (since SpecView is gone but the selection state is still consumed by GraphView).

Spec was the only tab where node-card click-to-highlight worked. With Spec gone:
- `handleValidationComponentClick` (InspectorPanel.tsx:448–458) currently selects a node and switches to Spec. The Spec switch goes away; **the `selectNode` call is preserved** so GraphView's highlight ring still appears when a validation error is clicked.
- The default tab when no hash fragment is supplied changes from `"spec"` to `"graph"`. Update `useHashRouter.ts:64`:
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

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): remove Spec tab (Phase 3A.6)

GraphView is now the default tab. Validation-banner clicks still call
selectNode (consumed by GraphView's highlight); Phase 2's audit-readiness
'Explain' surface will own this routing going forward."
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

## Task 8: Remove Validate button from inspector header

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx`.

The Validate button (InspectorPanel.tsx:557–572) is theater (design doc 03). Task 4 added auto-validate-on-change; `Ctrl+Shift+V` still works as a manual re-trigger. Remove the button and its `handleValidate` callback.

The validation banner (`<ValidationResultBanner>`) **stays** at its current location (between header and tab content). 15b moves it; Phase 2 owns the long-term destination.

The validation dot (InspectorPanel.tsx:499–542) **stays**. Phase 2 moves it into the audit-readiness panel.

- [ ] **Step 1: Write failing test**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InspectorPanel } from "./InspectorPanel";

describe("InspectorPanel without Validate button", () => {
  it("does not render a Validate button", () => {
    render(<InspectorPanel />);
    expect(
      screen.queryByRole("button", { name: /^validate$/i }),
    ).not.toBeInTheDocument();
  });

  it("still renders the validation dot indicator", () => {
    render(<InspectorPanel />);
    // dot has aria-label one of "Not validated"/"Validation passed"/...
    expect(screen.getByLabelText(/validation|not validated/i)).toBeInTheDocument();
  });
});
```

Run: `npx vitest run src/components/inspector/InspectorPanel.test.tsx -t "without Validate"`. Expected: FAIL.

- [ ] **Step 2: Edit `InspectorPanel.tsx`**

1. Remove the `<button onClick={handleValidate} …>Validate</button>` block (InspectorPanel.tsx:557–572).
2. Remove `handleValidate` (the entire `useCallback` at lines 387–424). The auto-validate subscription does the work; `Ctrl+Shift+V` calls `validate(activeSessionId)` directly without the side effects.

   But wait — `handleValidate` did real side effects: injecting a system message and `sendValidationFeedback(result)`. Those side effects are valuable. **Move them into the auto-validate subscription** so they fire on auto-validate too:

   In `subscriptions.ts`, after the `validate(sessionId)` call, await it and then orchestrate the same side effects:

   ```typescript
   lastValidatedVersion = nextVersion;
   void (async () => {
     await exec.validate(sessionId);
     const result = useExecutionStore.getState().validationResult;
     if (!result) return;

     const sessionStore = useSessionStore.getState();
     const VALIDATION_MSG_ID = "system-validation-current";

     if (!result.is_valid && result.errors.length > 0) {
       const lines = ["**Validation failed** — the following errors were sent to the agent:"];
       for (const err of result.errors) {
         lines.push(
           `- **[${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}:** ${err.message}`,
         );
       }
       sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
       // sendValidationFeedback is fire-and-forget here.  Per CLAUDE.md
       // audit-primacy, the backend records the validation event in the
       // audit Landscape; a frontend telemetry breadcrumb would duplicate
       // that record without adding probative value.  The user-visible
       // system message is already injected above.  Phase 8 (polish +
       // telemetry) is the right owner if a frontend operational signal
       // proves useful.  Operator adjudication 2026-05-15.
       void sessionStore.sendValidationFeedback(result);
     } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
       const lines = ["**Validation passed with warnings:**"];
       for (const warn of result.warnings) {
         lines.push(
           `- **[${warn.component_type ?? "unknown"}] ${warn.component_id ?? "unknown"}:** ${warn.message}`,
         );
       }
       sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
     }
   })();
   ```

   Update Task 4's test set to include side-effect assertions if not already covered. Add:

   ```typescript
   it("injects a system message when validation fails", async () => {
     const injectSystemMessage = vi.fn();
     const sendValidationFeedback = vi.fn().mockResolvedValue(undefined);
     const validate = vi.fn().mockImplementation(async () => {
       useExecutionStore.setState({
         validationResult: {
           is_valid: false,
           errors: [{ component_type: "source", component_id: "s1", message: "boom" }],
           warnings: [],
         } as never,
       } as never);
     });
     useExecutionStore.setState({ validate, isValidating: false, isExecuting: false } as never);
     useSessionStore.setState({
       activeSessionId: "sess-1",
       injectSystemMessage,
       sendValidationFeedback,
     } as never);

     useSessionStore.setState({
       compositionState: { version: 11, source: null, nodes: [], outputs: [] } as never,
     } as never);

     // CRITICAL: use waitFor instead of setTimeout so the test waits on
     // actual async completion rather than a magic delay.
     await waitFor(() => expect(injectSystemMessage).toHaveBeenCalled());
     expect(sendValidationFeedback).toHaveBeenCalled();
   });
   ```

3. Remove `isValidating`, `validate`, `injectSystemMessage`, `sendValidationFeedback` from the InspectorPanel's destructure if they were only used inside `handleValidate`. **Keep** them if they're used elsewhere (re-grep the file).

4. Remove `canValidate` if no longer referenced. (`hasCompositionContent` may still be used by the validation dot — keep that.)

- [ ] **Step 3: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS, including the new auto-validate side-effect test.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): remove Validate button (Phase 3A.8)

Validation now fires automatically on compositionState version
increment; Ctrl+Shift+V remains as a manual re-trigger.  System-message
injection and sendValidationFeedback are now driven by the auto-validate
subscription so the LLM continues to receive validation failures."
```

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| App becomes un-launchable mid-phase between commits | Each task ends with `npx vitest run src` including `App.test.tsx`'s smoke render. Tasks are ordered so additions precede removals. |
| Users hit a stale `#/{id}/spec` or `#/{id}/runs` bookmark and see a confusing default | 15a leaves `VALID_TABS` excluding the gone tabs; useHashRouter's fallback resolves them to a valid default (`spec` → `graph`, `runs` → `graph`). 15b adds explicit hash rewrites. Both behaviours produce a working app; 15b is a UX polish. |
| Spec-tab regulars confused by no "list view" | Phase 2's audit-readiness "Explain" surface inherits the lineage view. Phase 3A documents this in the commit message; until Phase 2 lands, validation-banner clicks select a node (GraphView highlight) but don't navigate further. Acceptable transitional state. |
| Auto-validate causes excessive backend calls | Subscription is gated on `compositionState.version` increment (not on every keystroke). Backend validate is idempotent. If telemetry shows pain, Phase 8 adds a debounce. Documented in Task 4. |
| Run-results inline rendering overlaps with the still-present Runs tab during Task 1–4 | `InlineRunResults` returns `null` when there are no runs *and* no historical runs. Even when it renders, the duplicate is brief — Task 5 removes the Runs tab. The dual-display is the cost of TDD discipline. |
| Removing the Validate button while Phase 2 hasn't shipped breaks the validation feedback loop | Task 8 explicitly moves the side-effects (system message injection + `sendValidationFeedback`) into the auto-validate subscription so the LLM still receives validation failures. The button removal is purely cosmetic now. |
| Theme toggle loses discoverability after moving from sidebar to UserMenu | Both surfaces existed in the legacy UI; Phase 1B already created the UserMenu hosting. Task 7 makes it the sole home; design doc 03 ratifies this placement. |
| The session-switcher's "New session" action collides with the `Ctrl+N` shortcut | Both call the same `createSession` action; no collision. The header dropdown adds discoverability; the shortcut is preserved in App.tsx. |
| Phase 2 / Phase 6 plans want different slot semantics than `SideRail` offers | The slots in Task 2 are skeleton markers (test-id only); content components own their own styling. If Phase 2 wants the audit-readiness panel to span the whole rail width vs. a card, that's a content decision, not a slot decision. The slot ordering is taken directly from design doc 03 §"Layout sketch". |
| `Layout.test.tsx` rewrites are large | Layout was already complex (sidebar + overlay mode + resize). Removing the sidebar simplifies the file; the tests that go away are precisely those that exercised gone behaviour. Net code change: net negative. |

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
