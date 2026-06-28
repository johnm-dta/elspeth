# Phase 3A — IA Cleanup: Removals (Part 2 of 2)

> **Continued from [15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md)**, which contains the plan header, scope boundaries, trust tier check, sequencing overview, open scope questions resolved, and Tasks 1–4 (all additive work).

> **Phase 3 block notice (added 2026-05-17; target corrected 2026-05-17):** This plan is one of four (15a1, 15a2, 15b1, 15b2) that together comprise the Phase 3 IA-cleanup work. **All four land as a single block on the dedicated Phase 3 worktree/branch for this IA-cleanup block** and **merge as one PR**. The canonical target for this packet is worktree `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on branch `feat/composer-phase-3-ia-cleanup`, created from `RC5.2` with `git worktree add .worktrees/composer-phase-3-ia-cleanup -b feat/composer-phase-3-ia-cleanup RC5.2` if it does not already exist. Do **not** use the old Phase 2A/2B/2C worktree or branch (`.worktrees/phase-2a-backend`, `feat/composer-phase-2a-backend`); those references are stale. Phrases below like "deferred to 15b" or "Phase 3B" mean "later tasks in the same Phase 3 branch," not a separate cycle. The 15a1→15a2→15b1→15b2 split is task sequencing and document organisation, not delivery sequencing — sequencing within the block still matters per task ordering.
>
> **Subagent dispatch discipline.** Every subagent prompt for this packet MUST start with this CWD-discipline preamble as its first Bash call: `cd /home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup && pwd && git rev-parse --abbrev-ref HEAD`; expected branch: `feat/composer-phase-3-ia-cleanup`. If the operator explicitly chooses a different Phase 3 worktree/branch, update this notice in **all four** 15a1/15a2/15b1/15b2 files before dispatch and use the chosen concrete values in every subagent prompt. The prompt must also state that `.worktrees/phase-2a-backend` and `feat/composer-phase-2a-backend` are stale Phase 2 targets and forbidden for Phase 3 work. Use absolute paths only thereafter for every Read/Bash/Grep. Bash `cd` does NOT persist between tool calls — relative paths can silently read the wrong branch.

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
- Modify: `src/elspeth/web/frontend/src/App.tsx` — remove the fanout-execute `"runs"` tab dispatch; consume `useHashRouter`'s widened return object via `const { redirectToast } = useHashRouter();`; render the redirect-toast banner.
- Modify: `src/elspeth/web/frontend/src/components/common/CommandPalette.tsx` — remove the `tab-runs` command.
- Modify: `src/elspeth/web/frontend/src/hooks/useHashRouter.ts` — remove `"runs"` from `VALID_TABS`, change the default to `"graph"`, add redirect-toast state, and widen the hook signature from `void` to `{ redirectToast: { message: string; dismiss: () => void } | null }`.
- Modify: `src/elspeth/web/frontend/src/hooks/useHashRouter.test.ts` — add the `#/{id}/runs` hash-fallback and redirect-toast tests.
- Modify: `src/elspeth/web/frontend/src/App.css` — add the info alert-banner modifier used by the redirect toast if no existing info style is present.

The `handleExecute` callback in `InspectorPanel.tsx:378–385` (drift from pre-Phase-2C plan citation L426–433) currently does `setActiveTab("runs")` after a successful execute. With Runs gone, **remove that line**. The user lands on whichever tab they were on (Graph or YAML) — and the inline `InlineRunResults` (Task 1) renders the run in the chat column.

`SWITCH_TAB_EVENT` listener in `InspectorPanel.tsx:325–334` (drift +1 line from pre-Phase-2C plan citation L325-333) currently accepts `"runs"`. Drop the `tab === "runs"` branch.

`App.tsx:85–90` (drift from pre-Phase-2C plan citation L62) does `window.dispatchEvent(new CustomEvent(SWITCH_TAB_EVENT, { detail: "runs" }))` in the `confirmFanoutExecution` callback. **Remove that line** — the dispatched event has no listener after Runs is gone, and the inline run-results path replaces it. (Re-verify the exact line number against the current file before editing — Phase 2C may have shifted it again.)

CommandPalette.tsx's `tab-runs` command needs removing. (15b removes the remaining `tab-spec/graph/yaml` commands; Task 5 removes `tab-runs` because Spec is still alive in this commit but Runs isn't.) Grep for `tab-runs` rather than relying on a line number — the pre-Phase-2C citation L174-179 is likely drifted.

Hash-router `VALID_TABS` in useHashRouter.ts:29 still includes `"runs"` today. **Remove `"runs"` from `VALID_TABS` in this task.** The regex in `parseHash` still matches old `#/sess-1/runs` links, but the `tab` value becomes `null` because `VALID_TABS.has(match[2])` returns false. **Default-tab change folded into this task (Section A panel fix).** The `resolvedTab = tab ?? "graph"` default change — originally specified for Task 6 — moves up to Task 5 so old Runs links fall back to Graph and the redirect toast text ("Showing Graph instead") is truthful for the entire Task 5 → Task 6 window. Task 6 leaves the default at `"graph"`.

> **Review finding (CRITICAL):** Silently redirecting old `#/{id}/runs` bookmarks during the 15a → 15b window produces a disorienting no-error experience. **Transient redirect toast:** when `useHashRouter` detects an old `runs` or `spec` hash fragment (before `VALID_TABS` strips it), show a one-time dismissible toast: _"The Runs tab was removed in this update. Showing Graph instead."_ (or _"The Spec tab was removed…"_). Track dismissal in `localStorage.elspeth_redirect_toast_dismissed`; the toast never reappears after dismissal. Add a test asserting the toast renders on first visit and does not render after dismissal.

Test the hash redirect: navigate to `#/sess-1/runs` and verify:
1. A toast banner appears the first time.
2. After dismissal, navigating to `#/sess-1/runs` again shows no toast.
3. The inspector shows the Graph tab (the default, per the S4 fold-forward in Step 6) in both cases. The full URL rewrite is in 15b.

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

// Inside applyHash / resolvedTab computation: default tab changes from
// "spec" to "graph" in the SAME edit as the VALID_TABS narrowing, so the
// redirect toast text ("Showing Graph instead") matches the actual
// destination during the Task 5 → Task 6 window. Task 6 keeps this value
// (no further change to the default).
const resolvedTab = tab ?? "graph";
```

(Old `#/sess/runs` links now resolve to `tab=null`, then `resolvedTab = "graph"` — the toast text matches what the user actually sees. 15b rewrites the URL so the address bar also reflects this.)

Also add the redirect-toast logic: when `parseHash()` sees a fragment that is not in `VALID_TABS` (and is not empty), check `localStorage.getItem("elspeth_redirect_toast_dismissed")`; if not set, surface the toast and call `localStorage.setItem("elspeth_redirect_toast_dismissed", "1")` on dismiss.

**Toast mount point (IMPORTANT — pre-review-finding 2026-05-17 follow-up).** `useHashRouter` is a hook, not a UI mounter. The redirect message DOM has to render *somewhere*. Specification:

- `useHashRouter` returns a new field on its return value, `redirectToast: { message: string; dismiss: () => void } | null`. The hook owns the localStorage read on first invocation and the message-text mapping (`"runs"` → _"The Runs tab was removed in this update. Showing Graph instead."_; `"spec"` → _"The Spec tab was removed in this update. Showing Graph instead."_). `dismiss` calls `localStorage.setItem("elspeth_redirect_toast_dismissed", "1")` and clears the in-state message.
- This widens the hook signature from `void` to `{ redirectToast: { message: string; dismiss: () => void } | null }`. Update `App.tsx`'s existing bare call (`useHashRouter();`) to destructure it: `const { redirectToast } = useHashRouter();`. Existing hook tests that call `renderHook(() => useHashRouter())` should still compile, but any typed caller must accept the new return object.
- `App.tsx` consumes the hook (search the file for `useHashRouter(` to find the call site). Render a banner immediately above the existing alert-banner region at `App.tsx:235`:

  ```tsx
  {redirectToast && (
    <div role="alert" className="alert-banner alert-banner--info">
      <span>{redirectToast.message}</span>
      <button type="button" onClick={redirectToast.dismiss} aria-label="Dismiss">
        Dismiss
      </button>
    </div>
  )}
  ```

  The `role="alert"` selector is what the Step 6a / Step 8a tests target via `screen.getByRole("alert")`. Reusing the existing `alert-banner` class keeps the visual treatment consistent; the `--info` modifier (add to `App.css` as a one-line `background: var(--color-info, #2b3a4a);` rule, or whatever the existing info-banner colour is) distinguishes the redirect notice from the error banner already at that location.
- The dismissal-flag tests (Step 6a and Step 8a, including the cross-path "shared dismissal flag" test) work against this shape because `redirectToast` becomes `null` whenever `elspeth_redirect_toast_dismissed === "1"`, regardless of which fragment triggered the read.

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

  it("falls back to the graph tab when #/{id}/runs is visited", () => {
    window.location.hash = "#/sess-1/runs";
    const { result } = renderHook(() => useHashRouter());
    // resolvedTab from the hook (or however it's exposed) should be "graph"
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
- Modify: `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx` — Phase 2C added a "View in Spec" affordance that dispatches `SWITCH_TAB_EVENT` with `detail: "spec"`. With Spec gone, change the dispatch to `detail: "graph"` and rename the button label to "View in graph". The GraphView's highlight ring (driven by `selectNode`) is the surviving destination for the audit-readiness → composition surface link.
- Modify: `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx` — update the assertion that exercises the View-in-Spec button to expect the new label and the new dispatch payload.

Spec was the only tab where node-card click-to-highlight worked. With Spec gone:
- `handleValidationComponentClick` (`InspectorPanel.tsx:400–410`, drift from pre-Phase-2C plan citation L448-458) currently selects a node and switches to Spec. The Spec switch goes away; **the `selectNode` call is preserved** so GraphView's highlight ring still appears when a validation error is clicked.
- The default tab when no hash fragment is supplied changes from `"spec"` to `"graph"`. Update `useHashRouter.ts:63` (drift -1 from pre-Phase-2C plan citation L64):
  ```typescript
  const resolvedTab = tab ?? "graph";
  ```

`VALID_TABS` removes `"spec"`. Old `#/sess/spec` deep links fall through to `graph`.

- [ ] **Step 1: Write failing tests**

> **Delete the contradictory Task-5 assertion first.** Task 5 Step 1 added an `it("still renders Spec, Graph, YAML tabs (Spec removed in Task 6)", ...)` block to `InspectorPanel.test.tsx`. That block now asserts the wrong thing (it expects Spec to be present, and this task removes Spec). Open `InspectorPanel.test.tsx` and delete that entire `it(...)` block from inside the `describe("InspectorPanel tab strip after Runs removal", ...)` group before adding the new failing tests below. Without this deletion, the suite breaks on the Task 6 commit and "every commit green" is violated.

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

- [ ] **Step 6a: Re-target `ReadinessRowDetail`'s "View in Spec" affordance**

> **Review finding (CRITICAL — Section A panel fix S2):** `ReadinessRowDetail.tsx` (Phase 2C) dispatches `SWITCH_TAB_EVENT` with `detail: "spec"` when the user clicks the "View in Spec" button. After this task narrows the `InspectorPanel` listener to `graph | yaml`, that dispatch becomes a silent no-op — no error, no feedback, dead affordance. The fix is to retarget the dispatch to `"graph"` and rename the button so the user sees a truthful label.

First, write the failing test. Open `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx` and replace the assertion that currently exercises the "View in Spec" button. The new assertion should read:

```typescript
it("dispatches SWITCH_TAB_EVENT with 'graph' when the View-in-graph button is clicked", async () => {
  const listener = vi.fn();
  window.addEventListener(SWITCH_TAB_EVENT, listener);
  render(<ReadinessRowDetail {/* ...props with a node-anchored row... */} />);
  await userEvent.click(screen.getByRole("button", { name: /view in graph/i }));
  expect(listener).toHaveBeenCalledTimes(1);
  expect((listener.mock.calls[0][0] as CustomEvent).detail).toBe("graph");
  window.removeEventListener(SWITCH_TAB_EVENT, listener);
});
```

Run: `npx vitest run src/components/audit/ReadinessRowDetail.test.tsx -t "View-in-graph"`. Expected: FAIL — the button is still labelled "View in Spec" and dispatches `"spec"`.

Now edit `ReadinessRowDetail.tsx`. Two edits:

1. Change the button label text from "View in Spec" (or whatever Phase 2C wrote) to "View in graph". Re-grep first because Phase 2C's exact wording may have changed; the goal is the user-visible label that previously routed to Spec.
2. Change the `window.dispatchEvent(new CustomEvent(SWITCH_TAB_EVENT, { detail: "spec" }))` call to `detail: "graph"`. Re-grep to confirm `"spec"` does not appear in any other dispatch in this file (sanity check; ReadinessRowDetail has one dispatch).

Re-run the test. Expected: PASS.

Smoke-grep for any other surviving `"spec"` dispatches across the frontend in case Phase 2C added a second one I have not anticipated:

```bash
grep -rn 'detail:.*"spec"' src/elspeth/web/frontend/src --include="*.tsx" --include="*.ts"
```

Expected: zero matches. If there is a hit, surface it before continuing — it is another dead-dispatch site this task must also retarget.

> **Known false-negative (deliberate scope):** This grep does NOT catch `App.tsx:155-162` (line numbers approximate; re-grep before relying), which builds a `tabMap = { "1": "spec", "2": "graph", "3": "yaml", "4": "runs" }` and dispatches `new CustomEvent(SWITCH_TAB_EVENT, { detail: tab })` — the literal `"spec"` never appears at the dispatch site, only in the lookup table. After Tasks 5 and 6 narrow the `InspectorPanel` listener to `"graph" | "yaml"`, Alt+1 (→`"spec"`) and Alt+4 (→`"runs"`) become silent no-ops. This is **explicitly deferred to 15b** per 15a1 §"Out of scope (deferred to 15b)" (`Alt+1/2/3/4 shortcut cleanup`); the keyboard handler stays untouched in 15a so that the surviving Alt+2 (Graph) and Alt+3 (YAML) keep working without an interleaved refactor. Do not "fix" this in 15a — the deferral is intentional and the dead-dispatch is harmless (no listener = silent return).

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

  it("does not show the spec toast after the runs toast was dismissed (shared dismissal flag)", () => {
    // Cross-path verification: the runs toast and the spec toast share a
    // single localStorage key. Dismissing either silences both. Without
    // this test, a regression that keyed the spec branch separately would
    // leave dismissed-runs users seeing the spec toast on first visit.
    window.location.hash = "#/sess-1/runs";
    const first = render(<App />);
    // Simulate the user dismissing the runs toast.
    userEvent.click(screen.getByRole("button", { name: /dismiss|close/i }));
    expect(localStorage.getItem("elspeth_redirect_toast_dismissed")).toBe("1");
    first.unmount();

    // Same user then visits a stale spec link. The spec toast must NOT appear.
    window.location.hash = "#/sess-1/spec";
    render(<App />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
```

Note that the dismissal flag is shared with the `#/runs` toast (single `elspeth_redirect_toast_dismissed` key). Dismissing one silences the other for that user — by design; the user has indicated "I understand this kind of redirect happens." 15b's hash-router rewrite makes both inert. The new cross-path test (`does not show the spec toast after the runs toast was dismissed`) locks in the shared-flag invariant explicitly, so a future regression that keys the two toasts separately fails CI rather than silently re-introducing two announcements per dismissal cycle.

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
- Create: `src/elspeth/web/frontend/tests/e2e/phase-3a-shell.spec.ts` — owns the Phase 3A shell Playwright pass named in the risks table (InlineRunResults real wiring, two-column layout, UserMenu keyboard nav).

Before this task, the sidebar still renders (it's the entry point for sessions). After this task, sessions are reached via `HeaderSessionSwitcher` (Task 3) and CommandPalette (already-built). The collapse-toggle and theme-toggle that lived in the sidebar toolbar must find new homes:
- **Theme toggle** → into `UserMenu` (per design doc 03).
- **Collapse-toggle** for the side rail → moved to a small handle on the side-rail boundary in `Layout.tsx`'s existing `inspector-toggle-btn` (still functioning post-rename as `siderail-toggle-btn`). The collapse-toggle for the sidebar is *deleted* since the sidebar is gone.

- [ ] **Step 1: Move theme-toggle into `UserMenu`**

Read `Layout.tsx:248–263` (the theme-toggle in the sidebar toolbar). Read `UserMenu.tsx` (Phase 1B Task 6). Preserve the existing disclosure/popover accessibility contract in `UserMenu`: it intentionally uses plain buttons inside list items, not `role="menu"` / `role="menuitem"`, because it does not implement the full ARIA menu keyboard contract. Add a new button-in-list item to `UserMenu`:

```tsx
import { useTheme } from "@/hooks/useTheme";

// ...inside UserMenu:
const { resolvedTheme, toggleTheme } = useTheme();

// ...in the menu <ul>, above Settings:
<li>
  <button
    type="button"
    onClick={() => {
      toggleTheme();
      setOpen(false);
    }}
  >
    {resolvedTheme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  </button>
</li>
```

Match the existing `UserMenu` list-item/button styling and focus handling rather than introducing ARIA menu roles. Add three tests in `UserMenu.test.tsx` covering label-reflects-state, label-inversion, and click-fires-and-closes contracts. Presence-only assertions are too weak — a stub `useTheme` returning the wrong direction or a click handler that doesn't dismiss the menu would slip past.

**Mocking discipline.** `useTheme` is a hook the component reads at render time; the tests need control over what it returns. Two pitfalls to avoid:

1. **Do not set `document.documentElement.dataset.theme` and assume `useTheme` reads it.** Whatever `useTheme`'s internal implementation is (CSS media query, React state, localStorage), setting a DOM attribute before `render()` does not synchronously force the hook's return value. The test must mock `useTheme` directly.
2. **Do not call `vi.mock(...)` inside an `it()` body.** Vitest hoists `vi.mock` calls to the top of the module ONLY when they appear at module/describe scope. A `vi.mock` inside `it()` runs after module evaluation and the mock will not be applied at the next `render()`. Tests that need per-test control over a mock's return value use `vi.hoisted` to capture a mutable state object the module-level `vi.mock` factory closes over.

At the top of `UserMenu.test.tsx` (or merging into Phase 1B Task 6's existing mock block if one is there), add a hoisted mock setup:

```typescript
const { mockUseThemeState } = vi.hoisted(() => ({
  mockUseThemeState: {
    resolvedTheme: "light" as "light" | "dark",
    toggleTheme: vi.fn(),
  },
}));
vi.mock("@/hooks/useTheme", () => ({
  useTheme: () => mockUseThemeState,
}));
```

Then add the three tests inside the existing `describe("UserMenu", ...)` block:

```typescript
describe("UserMenu — theme toggle (Phase 3A Task 7)", () => {
  beforeEach(() => {
    mockUseThemeState.resolvedTheme = "light";
    mockUseThemeState.toggleTheme = vi.fn();
  });

  it("shows a theme toggle button with a label reflecting current theme", async () => {
    // resolvedTheme = "light" from beforeEach
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(
      screen.getByRole("button", { name: /switch to dark theme/i }),
    ).toBeInTheDocument();
  });

  it("inverts the label when current theme is dark", async () => {
    mockUseThemeState.resolvedTheme = "dark";
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(
      screen.getByRole("button", { name: /switch to light theme/i }),
    ).toBeInTheDocument();
  });

  it("calls toggleTheme and closes the menu when the theme item is clicked", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("button", { name: /switch to dark theme/i }));
    expect(mockUseThemeState.toggleTheme).toHaveBeenCalledTimes(1);
    // Menu collapses after toggle.
    expect(
      screen.queryByRole("button", { name: /switch to dark theme/i }),
    ).not.toBeInTheDocument();
  });
});
```

**Also explicitly delete the orphan Layout theme-toggle test.** Search `src/elspeth/web/frontend/src/components/common/Layout.test.tsx` for any test named "keeps the theme toggle available" or that asserts the theme toggle inside the sidebar toolbar — delete those `it(...)` blocks in the same commit. The new UserMenu tests own this contract.

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
2. Remove the entire `<div className="layout-sidebar">…</div>` block. Search for `layout-sidebar` in `Layout.tsx` and delete the enclosing `<div>` from its opening tag to its matching closing tag (this region was approximately Layout.tsx:243–292 at the time this plan was written; re-grep before editing because Phase 2C and the SideRail rename may have shifted the range).
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

Add a one-shot localStorage cleanup for the orphaned `elspeth_sidebar_collapsed` key. The sidebar's collapse state is no longer read by any code after this task, but the key persists in every existing user's `localStorage`. Per CLAUDE.md No-Legacy Policy ("we have no users yet — deferring breaking changes is the opposite of what we want"), delete it in the same commit. Add this at module scope inside `App.tsx`, near the existing module-level imports:

```typescript
// Phase 3A Task 7 — sidebar removed; clean up orphaned preference key.
// Runs once on app boot. Safe to delete in Phase 8 polish.
if (typeof localStorage !== "undefined") {
  localStorage.removeItem("elspeth_sidebar_collapsed");
}
```

This is a one-shot side effect, not a useEffect — it must run before any other code reads from localStorage (e.g., the renamed `siderailWidth` initializer). Running it at module scope guarantees that ordering.

- [ ] **Step 6: Delete `SessionSidebar.tsx` and its test**

```bash
git rm src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx
git rm src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx
```

- [ ] **Step 6a: Add the Phase 3A shell Playwright spec**

Create `src/elspeth/web/frontend/tests/e2e/phase-3a-shell.spec.ts` in this Task 7 commit. It owns the three risks-table acceptance checks:

1. **InlineRunResults real wiring** — start a pipeline run, assert a `[aria-label="Pipeline run results"]` region appears in the chat column and contains non-stub `ProgressView` content (live status + token counters).
2. **Two-column layout** — assert the `.app-layout` element resolves to two grid-column tracks (chat + side rail), with no surviving session-sidebar region.
3. **UserMenu keyboard nav** — open the user menu, ArrowDown to the theme-toggle button, Enter, assert the theme toggle fires and the menu closes.

Run the spec with the repo's normal Playwright command (for example, `npm run test:e2e -- tests/e2e/phase-3a-shell.spec.ts` if that script is present). If Playwright cannot run locally, keep the spec in the commit and record the exact blocker in the handoff.

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
| App becomes un-launchable mid-phase between commits | Each task ends with `npx vitest run src` including `App.test.tsx`. **Note (Quality panel 2026-05-17): `App.test.tsx` stubs out Layout, SessionSidebar, ChatPanel, InspectorPanel, SecretsPanel, CommandPalette, ShortcutsHelp, and ConfirmDialog — so it asserts banner DOM, not the real component tree. Treat it as an "App-shell smoke" gate, not a component-integration gate.** Real component-integration risk for Tasks 5–7 (tab strip, layout grid, UserMenu) lands on the staging deploy. Phase 3 acceptance criteria add three named Playwright passes at the 15a→15b boundary: (1) **InlineRunResults real wiring** — start a pipeline run, assert a `[aria-label="Pipeline run results"]` region appears in the chat column and contains non-stub `ProgressView` content (live status + token counters); (2) **Two-column layout** — assert the `.app-layout` element resolves to two grid-column tracks (chat + side rail), with no surviving session-sidebar region; (3) **UserMenu keyboard nav** — open the user menu, ArrowDown to the theme-toggle button, Enter, assert the theme toggle fires and the menu closes. The Playwright spec lives at `src/elspeth/web/frontend/tests/e2e/phase-3a-shell.spec.ts` (new file in this PR and owned by Task 7 Step 6a). |
| Users hit a stale `#/{id}/spec` or `#/{id}/runs` bookmark and see a confusing default | 15a leaves `VALID_TABS` excluding the gone tabs; useHashRouter's fallback resolves them to a valid default (`spec` → `graph`, `runs` → `graph`). Tasks 5 and 6 add a one-time redirect toast (shared dismissal flag). 15b adds explicit hash rewrites. |
| Spec-tab regulars confused by no "list view" | Phase 2C's audit-readiness "Explain" surface (already shipped) inherits the lineage view. Until 15b migrates the validation-banner click handler to use it, validation-banner clicks select a node (GraphView highlight) but don't navigate further. Acceptable transitional state. |
| **Auto-validate stales validation badge during rapid composition flows (correctness, not load)** | Systems panel 2026-05-17 finding. A simple skip-if-in-flight guard discards intermediate `compositionState.version` increments during LLM tool-call bursts (N → N+1 → N+2). Resolution in Task 4: per-session `lastValidatedVersionBySession` map + `pendingValidateTarget` + `fireValidateLoop()` that re-fires `validate()` after the in-flight settles when a newer version arrived. The user never sees a "validated" badge for a stale snapshot. Debounce framing retired — correctness is the goal, not load shaping. |
| Manual `Ctrl+Shift+V` validation races across sessions | Deliberate trade-off: the Path B guard in 15a1 only protects auto-validate results because manual validation does not flow through `fireValidateLoop`, so `inflightValidateSessionId === null` and the guard is a no-op. Manual validation is rare and user-initiated; fixing it fully requires adding session identity to `ValidationResult` or routing manual validation through the same tracker. Do not mistake the no-op for an accidental omission. |
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

### 2026-05-17 — Section A panel-fix pass (S2 / S11 / S4 / A1 / Q4 / Q3)

**CRITICAL (Systems) — `ReadinessRowDetail` dead-dispatch fix folded into Task 6.** Phase 2C's "View in Spec" affordance dispatches `SWITCH_TAB_EVENT("spec")`, which goes silently dead after this task narrows the listener. New Step 6a retargets the dispatch to `"graph"` and renames the button to "View in graph"; the `ReadinessRowDetail.test.tsx` assertion is updated to lock in the new payload. The Task 6 **Files:** list now includes `ReadinessRowDetail.tsx` + its test.

**CRITICAL (Systems) — Pipeline-self-break removed.** Task 6 Step 1 now explicitly deletes the `it("still renders Spec, Graph, YAML tabs ...")` assertion that Task 5 added; without this, the suite breaks at Task 6 commit and the block's "every commit green" claim is violated.

**IMPORTANT (Systems) — Hash-fallback default folded forward.** `resolvedTab = tab ?? "graph"` moves into Task 5 alongside the `VALID_TABS` narrowing, so the redirect toast text ("Showing Graph instead") is truthful for the entire Task 5 → Task 6 window. Task 6 leaves the default at `"graph"`.

**IMPORTANT (Architecture) — Orphan localStorage key cleaned up.** Task 7 Step 5 adds a one-shot `localStorage.removeItem("elspeth_sidebar_collapsed")` at `App.tsx` module scope, deleting the key in the same commit as the code that wrote it. Per CLAUDE.md No-Legacy.

**IMPORTANT (Quality) — Cross-toast-flag test added.** Task 6 Step 8a now includes a test that dismisses the `#/runs` toast and verifies the `#/spec` toast does not appear (shared dismissal flag). Locks in the design choice that was previously documented but untested.

**IMPORTANT (Quality) — Risks-table Playwright scenarios named.** The smoke-gate-honesty row now names three specific Playwright passes (InlineRunResults real wiring, two-column layout, UserMenu keyboard nav) instead of the vague "at least one Playwright pass" wording.

### 2026-05-17 — Section B+C panel-fix pass (B10 / C13)

**IMPORTANT (Quality, B10) — Theme-toggle test strengthened.** Task 7 Step 1 now spec's three tests covering label-reflects-state, label-inversion (dark→light wording), and click-fires-and-closes contracts. The previous presence-only assertion would have passed against a stub `useTheme` returning the wrong direction. The orphan Layout theme-toggle test is explicitly marked for deletion in the same commit.

**IMPORTANT (Quality, B10 follow-up) — Vitest mocking pattern corrected.** Initial B10 spec set `document.documentElement.dataset.theme` and used `vi.mock(...)` inside the `it()` body — both wrong. `dataset.theme` does not influence what `useTheme` returns (the hook has its own internal state); `vi.mock` inside `it()` is not hoisted by Vitest and runs too late to apply at render time. Replaced with the canonical pattern: `vi.hoisted` captures a mutable state object, `vi.mock` at module scope returns it from `useTheme`, `beforeEach` resets the state. Mocking-discipline prose added above the test fence so the executor doesn't repeat either mistake.

**MINOR (Reality, C13) — Layout sidebar line range corrected.** Task 7 Step 3's "Layout.tsx:230–273" range was misleading (the actual sidebar `<div>` is at ~243–292). The instruction now tells the executor to grep for `layout-sidebar` and delete the enclosing div, with the line range noted as approximate. Defends against an executor literally cutting line 230 (the `gridColumns` block).

### 2026-05-17 — Pre-execution reality-check addendum (cross-references 15a1)

Final pre-execution sweep landed three small doc-only clarifications; this file carries two of them (the third lives in 15a1). No behavioural change.

**MINOR (Quality) — Redirect-toast mount-point spec'd.** Task 5 Step 6 now defines the precise integration shape between `useHashRouter` (which owns the localStorage read and the message-text mapping) and `App.tsx` (which mounts a `role="alert"` banner immediately above the existing alert-banner region at `App.tsx:235`). The `useHashRouter` hook gains a `redirectToast: { message; dismiss } | null` return field. The shared dismissal-flag invariant (a single `elspeth_redirect_toast_dismissed` key silences both the runs- and spec-path toasts) is preserved structurally because the field becomes `null` whenever the flag is set, regardless of which fragment triggered the read. The `screen.getByRole("alert")` test assertions in Steps 6a and 8a now have a concrete mount point to verify.

**MINOR (Architecture) — Alt-key tabMap dead-dispatch documented as deliberate scope-deferral.** Task 6 Step 6a's smoke-grep `grep 'detail:.*"spec"'` cannot catch `App.tsx:155-162`'s keyboard tabMap (which constructs `detail: tab` from a lookup table rather than dispatching a literal). A "known false-negative" note now flags this site as intentionally deferred to 15b per 15a1 §"Out of scope (deferred to 15b)" (`Alt+1/2/3/4 shortcut cleanup`). Rationale: removing the table entries for `"spec"` and `"runs"` in 15a would interleave a keyboard-handler refactor into an IA-cleanup pass; the dead dispatch is a silent no-op (no listener, no crash) so deferral is safe.

### 2026-05-17 — Pre-dispatch NO-GO follow-up

**BLOCKER (Execution target) — Phase 3 worktree/branch made concrete.** Shared header now names `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on `feat/composer-phase-3-ia-cleanup` from `RC5.2`; the old Phase 2A worktree/branch are explicitly forbidden.

**BLOCKER (Accessibility) — UserMenu theme toggle preserves disclosure semantics.** Task 7 no longer adds `role="menuitem"` or tests with `getByRole("menuitem")`; it preserves the existing button-in-list disclosure/popover pattern.

**IMPORTANT (Handoff) — Task 7 owns the Playwright shell spec.** `phase-3a-shell.spec.ts` is now in the Task 7 file list and Step 6a, matching the risks-table acceptance criteria.

**IMPORTANT (Handoff) — `useHashRouter` return widening called out.** Task 5 Step 6 now tells implementers to change `App.tsx` from bare `useHashRouter();` to `const { redirectToast } = useHashRouter();`.

### 2026-05-17 — P3A-008 — Retired RunsView capabilities (decision: most preserved in successors; two genuinely retired)

The deletion of `RunsView.test.tsx` (579 lines, commit `66748edb9`) removed test coverage for a set of behaviours. A subsequent post-merge audit found that most of the capabilities Agent D Finding 5 listed as "deleted" were in fact preserved by the successor surfaces (`RunsHistoryDrawer` + `RunDiagnosticsPanel` + `RunOutputsPanel`); the original entry mislabelled them. This entry records the corrected disposition.

**Capability inventory, corrected (Agent D Finding 5 list, re-audited 2026-05-17 against `RunsHistoryDrawer.tsx`):**

- Diagnostics accordion — **PRESERVED** in `RunsHistoryDrawer.tsx:170-175` (expandedRunId / hidden + conditional render).
- Polling while active — **RETIRED.** No interval-driven refresh; user clicks Refresh.
- Token states + artifacts on inspect — **PRESERVED** in `RunsHistoryDrawer.tsx:273-291` (per-token state rendering) + `RunOutputsPanel` mounted at `RunsHistoryDrawer.tsx:187` (artifacts).
- LLM explanation rendering — **PRESERVED** in `RunDiagnosticsPanel` via the `explanation` prop and Explain button (`RunsHistoryDrawer.tsx:180,244-246`).
- `failure_detail` rendering — **PRESERVED** in `RunDiagnosticsPanel` (`RunsHistoryDrawer.tsx:252-259`: operation_type, node_id, error_message).
- Fan-out accounting — **RETIRED.** No multi-path token count display.
- Inspect-button `aria-expanded` / `aria-controls` — **PRESERVED** at `RunsHistoryDrawer.tsx:151-152` (disclosure semantics maintained).
- Suggestion-banner from SpecView — **RESTORED** in `SideRailValidationBanner.tsx` (commit `a58479c19`, Phase 3A/3B Batch 5). The data source (`compositionState.validation_suggestions`) is now consumed by the new SuggestionList sub-component; the prior Runs-surface placement is gone but the field is no longer dropped on the floor.
- Active-run indicator + inline-rename from SessionSidebar — **RETIRED.** SessionSidebar is gone; HeaderSessionSwitcher does not replicate the active-run badge or rename-in-place affordance.

**Decision: keep retired what is retired; do not re-add the two genuine drops.** Polling and fan-out accounting are deliberately out of scope for the post-cleanup steady state — operators refresh on demand, and DAG-level token accounting is owned by the audit Landscape, not by the Runs surface. The active-run indicator and inline-rename can be re-added if the header session switcher acquires a richer affordance, but neither is required for the demo merge bar.

**Rationale:**

- Phase 3A's stated direction is IA cleanup, not feature deletion. `RunsView` was the unit being deleted; its **container** went away but most of its rendering responsibilities migrated to `RunsHistoryDrawer` / `RunDiagnosticsPanel` / `RunOutputsPanel` rather than vanishing. The original entry conflated "test file deleted" with "capability deleted"; only two of the nine items were genuinely retired.
- Test rehoming for the preserved capabilities is owed: the deleted `RunsView.test.tsx` covered behaviour that the new surfaces implement but do not yet test at the same granularity. Filing as a follow-up rather than amending 3A retroactively.
- The default for "deleted on purpose" is that it stays deleted unless an explicit product need re-emerges. The two genuine retirements (polling, fan-out accounting) follow this rule.

**Override path:** If polling-while-active or fan-out accounting is intended to return, the plan for that phase must (1) include the capability explicitly in scope and (2) add the corresponding tests. The six preserved capabilities are already live; their test coverage is the rehoming target, tracked separately. This section is now the visible historical inventory; the hidden reviewer packet was removed during the repository cleanout.

### 2026-05-17 — P3A-003 — Migration shim policy decision (operator override)

**Background.** Commit `2ac40b164` (orchestrator initial decision, option c) annotated three staging migration shims with retention rationale comments rather than deleting them, on the grounds that the operator's own browser state on `elspeth.foundryside.dev` contained localStorage entries from the pre-Phase-3A UI that a forced deletion would silently erase.

**Operator override (option a).** At merge review the operator overrode option (c) and directed option (a): delete all three shims per CLAUDE.md "No Legacy Code Policy". The operator accepted the one-time UX cost explicitly.

**Actual execution state at override time.** When the override was applied against the current HEAD (`bf09e4339`), two of the three shims had already been removed by intervening Phase 3A commits:

- **Shim 2 — `SIDERAIL_WIDTH_KEY` / `"elspeth_inspector_width"` (Layout.tsx):** removed in commit `a43594051` (`feat(web/frontend): remove composer inspector panel`). The entire resizable side-rail panel was replaced with a fixed-width layout; the localStorage-persisted width concept was abandoned. No rename to `"elspeth_siderail_width"` was necessary — the key ceased to exist. Operator-visible impact: saved sidebar width already reset to the fixed `320px` default for any browser that loaded the updated code.
- **Shim 3 — redirect-toast machinery (useHashRouter.ts / App.tsx):** removed in commit `bb9f12e4a` (`feat(web/frontend): rewrite hash router for composer actions`). The entire tab-based + redirect-toast system (`REDIRECT_TOAST_DISMISSED_KEY`, `REMOVED_TAB_MESSAGES`, `RedirectToast` interface, `maybeShowRedirectToast`, `dismissRedirectToast`, and the `redirectToast` return field) was replaced with the Phase 3B action-verb router that returns `void`. `useHashRouter.test.ts` was also rewritten — the 6 tests it now contains all exercise non-toast Phase 3B fragment behavior and are load-bearing; the file was preserved.

Only **Shim 1** required a deletion commit:

- **Shim 1 — `RETIRED_SIDEBAR_COLLAPSED_KEY` / `"elspeth_sidebar_collapsed"` (App.tsx):** deleted in this commit (see below). Both the NOTE comment block + const declaration and the `useEffect(() => { localStorage.removeItem(RETIRED_SIDEBAR_COLLAPSED_KEY); }, [])` call were removed. The colocated test `"removes the retired sidebar collapsed preference on startup"` in `App.test.tsx` was also deleted (it existed solely to verify the shim's cleanup behavior).

**Operator-visible impact of the full option-(a) outcome:**

1. `"elspeth_sidebar_collapsed"` localStorage entry left orphaned in operator's browser (~30 bytes, no functional effect — the SessionSidebar that read it was already deleted in Phase 3A.7).
2. Saved sidebar width already reset to `320px` at the `a43594051` deploy; no additional impact.
3. Bookmarks to `#/<sessionId>/spec` or `#/<sessionId>/runs` silently land on Graph (default) without an explanation banner — the redirect-toast was never deployed to staging, so no operator-visible regression.

**Commit:** `5e909cb6c` (`refactor(web/frontend): delete staging migration shims per operator override (P3A-003 option a)`).

**Operator note:** This entry records the orchestrator's reasonable-call default. Operator may override at merge review by selecting option (b) — deferred to 15b/3B — in which case the missing-tests list above becomes a mandatory 15b/3B sub-task checklist item.
