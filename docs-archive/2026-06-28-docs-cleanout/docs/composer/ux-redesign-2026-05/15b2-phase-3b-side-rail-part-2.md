# Phase 3B — IA Cleanup: hash-router migration, InspectorPanel teardown, cleanup (Part 2 of 2)

> **Continued from [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md)**, which contains the plan header, scope boundaries, trust tier check, sequencing overview, open scope questions resolved, and Tasks 1–5 (additive work and the Catalog button move).

> **Phase 3 block notice (added 2026-05-17; target corrected 2026-05-17):** This plan is one of four (15a1, 15a2, 15b1, 15b2) that together comprise the Phase 3 IA-cleanup work. **All four land as a single block on the dedicated Phase 3 worktree/branch for this IA-cleanup block** and **merge as one PR**. The canonical target for this packet is worktree `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on branch `feat/composer-phase-3-ia-cleanup`, created from `RC5.2` with `git worktree add .worktrees/composer-phase-3-ia-cleanup -b feat/composer-phase-3-ia-cleanup RC5.2` if it does not already exist. Do **not** use the old Phase 2A/2B/2C worktree or branch (`.worktrees/phase-2a-backend`, `feat/composer-phase-2a-backend`); those references are stale. Phrases below like "Phase 3A" / "15a" mean "earlier tasks in the same Phase 3 branch," not a prior cycle. The 15a1→15a2→15b1→15b2 split is task sequencing and document organisation, not delivery sequencing — sequencing within the block still matters per task ordering.
>
> **Subagent dispatch discipline.** Every subagent prompt for this packet MUST start with this CWD-discipline preamble as its first Bash call: `cd /home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup && pwd && git rev-parse --abbrev-ref HEAD`; expected branch: `feat/composer-phase-3-ia-cleanup`. If the operator explicitly chooses a different Phase 3 worktree/branch, update this notice in **all four** 15a1/15a2/15b1/15b2 files before dispatch and use the chosen concrete values in every subagent prompt. The prompt must also state that `.worktrees/phase-2a-backend` and `feat/composer-phase-2a-backend` are stale Phase 2 targets and forbidden for Phase 3 work. Use absolute paths only thereafter for every Read/Bash/Grep. Bash `cd` does NOT persist between tool calls — relative paths can silently read the wrong branch.

**Umbrella plan context:**
- Predecessor: [15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md) / [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md)
- Successor: Phase 2 (audit-readiness panel), Phase 6 (completion gestures), Phase 7 (catalog reshape)
- Roadmap: [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3)

**Tasks in this file:** Tasks 6–10 (hash-router rewrite, CommandPalette rename/retarget, keyboard shortcut audit, InspectorPanel deletion, cleanup pass), plus Risks and mitigations, Memory references, and Review history.

---

## Task 6: `useHashRouter` rewrite + redirect stale fragments

**Files:**
- Modify: `src/elspeth/web/frontend/src/hooks/useHashRouter.ts`.
- Modify: `src/elspeth/web/frontend/src/hooks/useHashRouter.test.ts` (or whichever existing test file covers it; create one if it doesn't exist).

The new vocabulary:
- `#/{sessionId}` — canonical.
- `#/{sessionId}/graph` — opens the graph modal on mount, then rewrites the URL to `#/{sessionId}`.
- `#/{sessionId}/yaml` — opens the YAML export modal on mount, then rewrites the URL to `#/{sessionId}`.
- `#/{sessionId}/spec`, `#/{sessionId}/runs`, or any unrecognised verb — silently rewrites to `#/{sessionId}`. No modal.

- [ ] **Step 1: Failing test**

Before adding the tests, create the constants-only `src/lib/composer-events.ts` module described in Step 3a if it does not already exist. This prevents the red test from failing on a missing import instead of the router behaviour under test; the module itself is just shared string constants and adds no runtime behaviour.

Create or extend `src/elspeth/web/frontend/src/hooks/useHashRouter.test.ts`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook } from "@testing-library/react";
import { useHashRouter } from "./useHashRouter";
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT, OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

describe("useHashRouter — Phase 3B fragment migration", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", window.location.pathname);
    useSessionStore.setState({
      sessions: [{ id: "sess-1", title: "x" } as never],
      activeSessionId: null,
      selectSession: vi.fn(),
    } as never);
  });

  it("rewrites #/{id}/spec to #/{id}", () => {
    window.location.hash = "#/sess-1/spec";
    renderHook(() => useHashRouter());
    // mount applies the hash, then the rewrite clears the verb
    expect(window.location.hash).toBe("#/sess-1");
  });

  it("rewrites #/{id}/runs to #/{id}", () => {
    window.location.hash = "#/sess-1/runs";
    renderHook(() => useHashRouter());
    expect(window.location.hash).toBe("#/sess-1");
  });

  it("opens the graph modal and rewrites for #/{id}/graph", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    window.location.hash = "#/sess-1/graph";
    renderHook(() => useHashRouter());
    await act(async () => {});
    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });

  it("opens the yaml modal and rewrites for #/{id}/yaml", async () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    window.location.hash = "#/sess-1/yaml";
    renderHook(() => useHashRouter());
    await act(async () => {});
    expect(handler).toHaveBeenCalled();
    expect(window.location.hash).toBe("#/sess-1");
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });

  it("strips any unrecognised verb", () => {
    window.location.hash = "#/sess-1/nonsense";
    renderHook(() => useHashRouter());
    expect(window.location.hash).toBe("#/sess-1");
  });

  it("cold-load race: modal renders when App mounts with a graph hash", async () => {
    // CRITICAL: mount a minimal integration tree that includes BOTH
    // useHashRouter and <GraphModal /> with a #/sess-1/graph hash set before
    // mount.  The queueMicrotask deferral ensures the modal listener has
    // registered before the dispatch fires.
    //
    // No app-wide fixture is assumed — we compose the minimum tree inline.
    // (`render`, `screen`, `act`, and `GraphModal` are imported at the top
    // of the file with the other imports already in scope.)
    window.location.hash = "#/sess-1/graph";
    function HarnessTree(): JSX.Element {
      useHashRouter();
      return <GraphModal />;
    }
    render(<HarnessTree />);
    // queueMicrotask fires after the current microtask queue drains.
    await act(async () => {});
    expect(screen.getByRole("dialog", { name: /pipeline graph/i })).toBeInTheDocument();
  });
});
```

The harness imports added at the top of the test file:

```typescript
import { render, screen, act } from "@testing-library/react";
import { GraphModal } from "@/components/sidebar/GraphModal";
```

(`GraphView` is mocked at the suite level via `vi.mock("@/components/inspector/GraphView", …)` — the existing mock in `GraphModal.test.tsx` is duplicated here, or the `useHashRouter` test extends the same mock registry. Either approach is acceptable; the contract is "GraphView never executes its real body in this test.")

- [ ] **Step 2: Run test to verify failure**

```bash
cd src/elspeth/web/frontend && npx vitest run src/hooks/useHashRouter.test.ts
```

Expected: FAIL — current router has `VALID_TABS` and routes `spec`/`runs` to themselves.

- [ ] **Step 3a: Create or verify `src/lib/composer-events.ts`**

> **Review finding (IMPORTANT):** `OPEN_GRAPH_MODAL_EVENT` and `OPEN_YAML_MODAL_EVENT` must live in a shared constants module. `useHashRouter` importing from component files creates a circular dependency risk (hook → component → hook). Task 3 creates this module before GraphMiniView's first test; this step verifies it still exists and updates any missed importers.

Create `src/elspeth/web/frontend/src/lib/composer-events.ts`:

```typescript
// ============================================================================
// Composer event constants — shared across useHashRouter, modal components,
// and any consumer that needs to open a modal programmatically.
//
// All parties import from this module. Component files must not define their
// own event-name constants.
// ============================================================================

export const OPEN_GRAPH_MODAL_EVENT = "elspeth-open-graph-modal";
export const OPEN_YAML_MODAL_EVENT = "elspeth-open-yaml-modal";
```

Update importers:
- `GraphMiniView.tsx` — import `OPEN_GRAPH_MODAL_EVENT` from `@/lib/composer-events`; if a local export slipped in during Task 3, delete it.
- `ExportYamlButton.tsx` — import `OPEN_YAML_MODAL_EVENT` from `@/lib/composer-events`; if a local export slipped in during Task 4, delete it.
- `GraphModal.tsx` — update import.
- `ExportYamlModal.tsx` — update import.
- `useHashRouter.ts` — update import (see Step 3b below).
- `CommandPalette.tsx`, `CompletionSummary.tsx`, and `ReadinessRowDetail.tsx` — update modal-open dispatches to import the named constants from this module. Do not keep hard-coded modal event strings in these files.

- [ ] **Step 3b: Rewrite `useHashRouter.ts`**

Replace the contents of `useHashRouter.ts`:

```typescript
/**
 * Hash-based router for session deep linking.
 *
 * Format: #/{sessionId}                      → canonical
 *         #/{sessionId}/graph                → open the graph modal, rewrite
 *         #/{sessionId}/yaml                 → open the YAML export modal,
 *                                             rewrite
 *         #/{sessionId}/{anything-else}      → silently strip the verb
 *
 * Phase 3B replaced the previous inspector-tab vocabulary (spec / graph /
 * yaml / runs).  Stale bookmarks for spec/runs land on the canonical session
 * URL; bookmarks for graph/yaml additionally open the corresponding modal.
 */

import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT, OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

interface HashState {
  sessionId: string | null;
  verb: string | null;
}

interface HashRouterResult {
  redirectToast: null;
}

const ACTION_VERBS: Record<string, string> = {
  graph: OPEN_GRAPH_MODAL_EVENT,
  yaml: OPEN_YAML_MODAL_EVENT,
};

function parseHash(): HashState {
  const hash = window.location.hash;
  const match = hash.match(/^#\/([^/]+?)(?:\/([a-z]+))?$/);
  if (!match) return { sessionId: null, verb: null };
  return { sessionId: match[1], verb: match[2] ?? null };
}

function buildCanonicalHash(sessionId: string | null): string {
  return sessionId ? `#/${sessionId}` : "";
}

export function useHashRouter(): HashRouterResult {
  const lastWrittenHash = useRef<string>("");
  const applying = useRef(false);

  const applyHash = (state: HashState) => {
    applying.current = true;
    const { sessionId, verb } = state;
    const store = useSessionStore.getState();

    if (sessionId && sessionId !== store.activeSessionId) {
      store.selectSession(sessionId);
    }

    if (verb && verb in ACTION_VERBS) {
      // CRITICAL: dispatch via queueMicrotask so modal listeners have time
      // to register before the event fires.  On cold load, App mounts
      // App-root modals in the same React commit as useHashRouter; without
      // the microtask deferral the event fires before addEventListener runs.
      const eventName = ACTION_VERBS[verb];
      queueMicrotask(() => window.dispatchEvent(new CustomEvent(eventName)));
    }

    // Always rewrite to canonical — the verb is never part of the steady
    // state, only an action to fire on landing.
    const canonical = buildCanonicalHash(sessionId);
    if (canonical !== window.location.hash) {
      lastWrittenHash.current = canonical;
      window.history.replaceState(
        null,
        "",
        canonical || window.location.pathname,
      );
    }

    applying.current = false;
  };

  useEffect(() => {
    const initial = parseHash();
    if (initial.sessionId) {
      lastWrittenHash.current = window.location.hash;
      applyHash(initial);
    } else {
      const { activeSessionId } = useSessionStore.getState();
      if (activeSessionId) {
        const hash = buildCanonicalHash(activeSessionId);
        lastWrittenHash.current = hash;
        window.history.replaceState(
          null,
          "",
          hash || window.location.pathname,
        );
      }
    }
  }, []);

  useEffect(() => {
    function handleHashChange() {
      const newHash = window.location.hash;
      if (newHash === lastWrittenHash.current) return;
      lastWrittenHash.current = newHash;
      applyHash(parseHash());
    }
    window.addEventListener("popstate", handleHashChange);
    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("popstate", handleHashChange);
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (applying.current) return;
      if (state.activeSessionId === prevState.activeSessionId) return;
      const hash = buildCanonicalHash(state.activeSessionId);
      if (hash === lastWrittenHash.current) return;
      lastWrittenHash.current = hash;
      if (hash) {
        window.history.pushState(null, "", hash);
      } else {
        window.history.replaceState(null, "", window.location.pathname);
      }
    });
    return unsub;
  }, []);

  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (prevState.sessions.length > 0 || state.sessions.length === 0) return;
      const { sessionId } = parseHash();
      if (!sessionId) return;
      const exists = state.sessions.some((s) => s.id === sessionId);
      if (!exists && state.activeSessionId === sessionId) {
        lastWrittenHash.current = "";
        window.history.replaceState(null, "", window.location.pathname);
        useSessionStore.setState({ activeSessionId: null });
      }
    });
    return unsub;
  }, []);

  // Preserve the widened return shape introduced in 15a so App.tsx can keep
  // destructuring `{ redirectToast }`.  Phase 3B rewrites stale fragments
  // immediately, so there is no redirect toast to show.
  return { redirectToast: null };
}
```

Notes on the rewrite:
- The previous `TAB_CHANGED_EVENT` listener is **deleted**. No surface dispatches it; 15a's auto-validate effect doesn't need it; nothing else listens. The export from this file is removed.
- The default-tab `"spec"` is gone. The dispatch on `SWITCH_TAB_EVENT` is gone. The previous "dispatch the resolved tab on apply" is replaced by "dispatch the action verb if any."
- The current-tab `useRef<string | null>` is gone — there is no tab state anymore.
- The return type remains an object for compatibility with 15a's redirect-toast mount point. `redirectToast` is always `null` in 15b because stale fragments are canonicalized immediately.

- [ ] **Step 4: Remove `TAB_CHANGED_EVENT` import in `InspectorPanel.tsx`**

```bash
cd src/elspeth/web/frontend && grep -RIn "TAB_CHANGED_EVENT" src
```

For every match, delete. Currently `InspectorPanel.tsx:18,343–348` dispatches it; that block is deleted. Task 9 deletes the file entirely; this is the bridge state.

- [ ] **Step 5: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS. Some inspector-panel-level tests for the tab-change behaviour will fail — update them to match the new contract (no tab change emission). If they were asserting hash-change-on-tab-switch, delete those assertions; Task 9 deletes the entire `InspectorPanel.test.tsx`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): rewrite useHashRouter for Phase 3B vocabulary

#/{id}/spec and #/{id}/runs are silently rewritten to #/{id}.  graph
and yaml fragments open the corresponding modals before rewriting.
TAB_CHANGED_EVENT is removed; it has no listeners after the inspector
panel is gone (Task 9 deletes the inspector entirely)."
```

---

## Task 7: `CommandPalette` rename / retarget

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/common/CommandPalette.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/CommandPalette.test.tsx`.

Changes:
- Delete `tab-spec` and `tab-runs` commands.
- Rename `tab-graph` → `open-graph-modal`, retarget to dispatch `OPEN_GRAPH_MODAL_EVENT`. Title: "Open graph view". Shortcut: `Ctrl+Shift+G`.
- Rename `tab-yaml` → `open-yaml-export`, retarget to dispatch `OPEN_YAML_MODAL_EVENT`. Title: "Export YAML". Shortcut: `Ctrl+Shift+Y`.
- The internal `switchTab` helper is gone; commands call `window.dispatchEvent` directly.
- The import/use of `SWITCH_TAB_EVENT` is removed from the palette commands. The constant export may survive only until Task 10, while `CompletionSummary.tsx` and `ReadinessRowDetail.tsx` are retargeted; after Task 10 it must be gone.

- [ ] **Step 1: Failing test**

Add to `CommandPalette.test.tsx`:

```typescript
import { OPEN_GRAPH_MODAL_EVENT, OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

it("opens the graph modal via the command 'Open graph view'", async () => {
  const handler = vi.fn();
  window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  const { getByRole, getByText } = render(
    <CommandPalette isOpen onClose={() => {}} />,
  );
  fireEvent.click(getByText(/open graph view/i));
  expect(handler).toHaveBeenCalled();
  window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
});

it("opens the yaml export modal via the command 'Export YAML'", () => {
  const handler = vi.fn();
  window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
  const { getByText } = render(
    <CommandPalette isOpen onClose={() => {}} />,
  );
  fireEvent.click(getByText(/export yaml/i));
  expect(handler).toHaveBeenCalled();
  window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
});

it("does not list Spec or Runs as navigation commands", () => {
  const { queryByText } = render(
    <CommandPalette isOpen onClose={() => {}} />,
  );
  expect(queryByText(/switch to spec tab/i)).toBeNull();
  expect(queryByText(/switch to runs tab/i)).toBeNull();
});
```

- [ ] **Step 2: Apply the changes in `CommandPalette.tsx`**

At the top of `CommandPalette.tsx`, import the shared event constants:

```typescript
import { OPEN_GRAPH_MODAL_EVENT, OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
```

Then replace lines 140–180 (the `switchTab` helper and the four `tab-*` command pushes) with two direct commands:

```typescript
cmds.push({
  id: "open-graph-modal",
  title: "Open graph view",
  category: "navigation",
  shortcut: "Ctrl+Shift+G",
  action: () => {
    window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    onClose();
  },
});

cmds.push({
  id: "open-yaml-export",
  title: "Export YAML",
  category: "navigation",
  shortcut: "Ctrl+Shift+Y",
  action: () => {
    window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
    onClose();
  },
});
```

Note: import event constants from `@/lib/composer-events`, not from `GraphMiniView` / `ExportYamlButton`. The shared module exists specifically to avoid a component import cycle while keeping one canonical source for event names.

- [ ] **Step 3: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): rename inspector-tab palette commands (Phase 3B.7)

Switch to Spec/Runs commands deleted.  Switch to Graph → 'Open graph
view' (Ctrl+Shift+G).  Switch to YAML → 'Export YAML' (Ctrl+Shift+Y).
Both dispatch the corresponding modal-open event.  SWITCH_TAB_EVENT
is deleted in Task 10 cleanup; no deferral to Phase 6."
```

---

## Task 8: `App.tsx` keyboard shortcuts: retire Alt+1/2/3/4, add Ctrl+Shift+G / Y, update `ShortcutsHelp`

**Files:**
- Modify: `src/elspeth/web/frontend/src/App.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`.

- [ ] **Step 1: Failing test**

Open `ShortcutsHelp.test.tsx` and add:

```typescript
it("documents Ctrl+Shift+G as Open graph view", () => {
  render(<ShortcutsHelp onClose={() => {}} />);
  expect(screen.getByText(/ctrl\+shift\+g/i)).toBeInTheDocument();
  expect(screen.getByText(/open graph view/i)).toBeInTheDocument();
});

it("documents Ctrl+Shift+Y as Export YAML", () => {
  render(<ShortcutsHelp onClose={() => {}} />);
  expect(screen.getByText(/ctrl\+shift\+y/i)).toBeInTheDocument();
  expect(screen.getByText(/export yaml/i)).toBeInTheDocument();
});

it("no longer documents Alt+1/2/3/4 tab shortcuts", () => {
  render(<ShortcutsHelp onClose={() => {}} />);
  expect(screen.queryByText(/alt\+1/i)).toBeNull();
  expect(screen.queryByText(/alt\+2/i)).toBeNull();
  expect(screen.queryByText(/alt\+3/i)).toBeNull();
  expect(screen.queryByText(/alt\+4/i)).toBeNull();
});
```

- [ ] **Step 2: Update `App.tsx`**

At the top of `App.tsx`, import the shared event constants:

```typescript
import { OPEN_GRAPH_MODAL_EVENT, OPEN_YAML_MODAL_EVENT } from "./lib/composer-events";
```

Then replace the `Alt+1/2/3/4` block in `App.tsx:123–139` with handlers for the two new shortcuts:

```typescript
// Ctrl+Shift+G / Cmd+Shift+G: Open graph view modal
if (
  e.key.toLowerCase() === "g" &&
  e.shiftKey &&
  (e.ctrlKey || e.metaKey)
) {
  e.preventDefault();
  window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
  return;
}

// Ctrl+Shift+Y / Cmd+Shift+Y: Open YAML export modal
if (
  e.key.toLowerCase() === "y" &&
  e.shiftKey &&
  (e.ctrlKey || e.metaKey)
) {
  e.preventDefault();
  window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
  return;
}
```

Also: if any stale dispatch remains in `confirmFanoutExecution` (`new CustomEvent(SWITCH_TAB_EVENT, { detail: "runs" })`), remove it — there is no Runs tab. In the current Phase 3A worktree this should already be absent. The post-fan-out behaviour is to do nothing special; `InlineRunResults` will pick up the activeRunId via its store subscription as the run starts.

```typescript
const confirmFanoutExecution = useCallback(async () => {
  await useExecutionStore.getState().confirmFanoutExecution();
}, []);
```

The `import { SWITCH_TAB_EVENT } from "./components/common/CommandPalette"` import is removed from `App.tsx` (the constant still exists but App.tsx no longer needs it).

- [ ] **Step 3: Update `ShortcutsHelp.tsx`**

In `ShortcutsHelp.tsx`, find the section enumerating tab-switch shortcuts (currently listing Alt+1 Spec, Alt+2 Graph, Alt+3 YAML, Alt+4 Runs). Replace with the two new entries:

```tsx
<dt><kbd>Ctrl+Shift+G</kbd></dt>
<dd>Open graph view</dd>
<dt><kbd>Ctrl+Shift+Y</kbd></dt>
<dd>Export YAML</dd>
```

Note: `Ctrl+Shift+P` for "Open plugin catalog (reference)" stays as-is.

- [ ] **Step 4: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): retire Alt+1/2/3/4 shortcuts (Phase 3B.8)

Ctrl+Shift+G opens the graph modal.  Ctrl+Shift+Y opens the YAML
export modal.  The Alt+ shortcuts pointed at inspector tabs that are
gone (Spec, Runs) or that are now reached via a different mechanism
(Graph, YAML).  ShortcutsHelp regenerated.

confirmFanoutExecution no longer dispatches SWITCH_TAB_EVENT; inline
run results pick up the new run from the store directly."
```

---

## Task 9: Delete `InspectorPanel.tsx` + simplify `Layout` grid

**Files:**
- Delete: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx`.
- Delete: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` (already done in 15a Task 6 if 15a executed cleanly — re-verify with `git ls-files`).
- Delete: `src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx` (ditto).
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` (ditto).
- Delete: `src/elspeth/web/frontend/src/components/inspector/RunsView.test.tsx` (ditto).
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx` — remove the `sidebar` slot prop entirely (already unused after 15a); rename the `inspector` slot prop to `siderail`; remove the `inspector-overlay` mode (the side rail is fixed-width and not collapsible in 15b — Phase 6's completion bar might re-introduce this, but Phase 3 deletes it); delete the transitional `elspeth_inspector_width` reader/writer from 15a and do not replace it with a new width-persistence key; preserve the existing `<DefaultModeChangedBanner />` mount inside the chat column.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — replace `<InspectorPanel />` with `<SideRail />`; delete the `import { InspectorPanel }` line; delete the `<SessionSidebar />` mount that 15a left as a no-op-stripped pass-through (if any survived).
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.test.tsx` — drop tests of overlay mode, sidebar-collapse, and inspector-resize. Add a test that asserts the two-column grid: `chat`, `siderail`.

Critically: **the `ValidationResultBanner` lives in `SideRail` after this task**. The banner's previous home (between the inspector tab strip and tab content) is gone.

Per the slot composition contract established in 15a1 ("caller passes content, SideRail places it"), the banner is **NOT** inlined inside `SideRail`'s body. Instead:

1. Extend `SideRail`'s prop type with a new `validationBannerSlot?: ReactNode | null` (added in this task), rendered in a wrapper div above the `graph-mini` slot with `data-testid="siderail-slot-validation-banner"`.
2. Extract the banner + error rendering into a new sidebar-local component `SideRailValidationBanner.tsx` (file: `src/elspeth/web/frontend/src/components/sidebar/SideRailValidationBanner.tsx`) that owns its own `useExecutionStore.validationResult` / `useExecutionStore.error` / `useSessionStore.compositionState.nodes` subscriptions. It calls `useSessionStore.getState().selectNode(nodeId)` on `onComponentClick` (no tab switch — 15a Task 6 already nooped that).
3. `App.tsx` passes `validationBannerSlot={<SideRailValidationBanner />}` alongside the other slots.

Wire shape:

```tsx
// SideRail.tsx — extended prop list
interface SideRailProps {
  validationBannerSlot?: ReactNode | null;  // NEW in Task 9
  auditReadinessSlot?: ReactNode | null;
  graphMiniSlot?: ReactNode | null;
  catalogSlot?: ReactNode | null;
  exportYamlSlot?: ReactNode | null;
  executeButtonSlot?: ReactNode | null;
  completionBarSlot?: ReactNode | null;
}

// Rendered slot order, top to bottom:
//   validation-banner → audit-readiness → graph-mini → catalog → export-yaml → execute-button → completion-bar
```

```tsx
// App.tsx — cumulative wire after Task 9
<SideRail
  validationBannerSlot={<SideRailValidationBanner />}
  auditReadinessSlot={<AuditReadinessPanel />}
  graphMiniSlot={<GraphMiniView />}
  catalogSlot={<CatalogButton />}
  exportYamlSlot={<ExportYamlButton />}
  executeButtonSlot={<ExecuteButton />}
  /* completionBarSlot remains null for Phase 6 */
/>
```

The error banner (from `executionStore.error`) is rendered inside `SideRailValidationBanner` alongside the validation banner.

- [ ] **Step 1: Failing test**

Add to `src/elspeth/web/frontend/src/components/common/Layout.test.tsx`:

```typescript
it("renders a two-column grid: chat and siderail", () => {
  render(<Layout chat={<div data-testid="chat" />} siderail={<div data-testid="rail" />} />);
  expect(screen.getByTestId("chat")).toBeInTheDocument();
  expect(screen.getByTestId("rail")).toBeInTheDocument();
});

it("does not render a sidebar slot", () => {
  render(<Layout chat={<div />} siderail={<div />} />);
  expect(document.querySelector(".layout-sidebar")).toBeNull();
});

it("does not render an inspector resize handle", () => {
  render(<Layout chat={<div />} siderail={<div />} />);
  expect(document.querySelector(".resize-handle")).toBeNull();
});
```

And in `SideRailValidationBanner.test.tsx` (new file alongside the new component — `src/elspeth/web/frontend/src/components/sidebar/SideRailValidationBanner.test.tsx`):

```typescript
it("renders the validation banner when validationResult is present", () => {
  useExecutionStore.setState({
    validationResult: {
      is_valid: false,
      errors: [{ component_type: "source", component_id: "s1", message: "x" } as never],
      warnings: [],
    } as never,
  } as never);
  render(<SideRailValidationBanner />);
  expect(
    screen.getByText(/validation/i),
  ).toBeInTheDocument();
});
```

The slot-presence assertion (that `<SideRail validationBannerSlot={…} />` places the prop content under `data-testid="siderail-slot-validation-banner"`) lives in `SideRail.test.tsx` — see Step 6 below.

Also preserve Phase 1B's default-mode banner host in `Layout.test.tsx`. Mock the component at the top of the test file if necessary:

```typescript
vi.mock("./DefaultModeChangedBanner", () => ({
  DefaultModeChangedBanner: () => <div data-testid="default-mode-changed-banner" />,
}));

it("keeps the default-mode changed banner in the chat column", () => {
  render(<Layout chat={<div data-testid="chat" />} siderail={<div data-testid="rail" />} />);
  const chatColumn = screen.getByTestId("chat").closest(".layout-chat");
  expect(chatColumn).toContainElement(
    screen.getByTestId("default-mode-changed-banner"),
  );
});
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/Layout.test.tsx src/components/sidebar/SideRail.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Rewrite `Layout.tsx`**

Replace the existing Layout. Strip out: `sidebar` prop, `sidebarCollapsed` state, `INSPECTOR_WIDTH_KEY` / `elspeth_inspector_width` persistence, the overlay-mode branch, the resize handle. Preserve the Phase 1B `<DefaultModeChangedBanner />` mount in the chat column. Keep only:

```typescript
import { type ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";

interface LayoutProps {
  chat: ReactNode;
  siderail: ReactNode;
}

const SIDERAIL_WIDTH = 320; // fixed Phase 3B width; no localStorage width persistence

export function Layout({ chat, siderail }: LayoutProps): JSX.Element {
  return (
    <div
      className="app-layout"
      style={{ gridTemplateColumns: `1fr ${SIDERAIL_WIDTH}px` }}
    >
      <div className="layout-chat">
        <DefaultModeChangedBanner />
        <ErrorBoundary label="Chat panel">{chat}</ErrorBoundary>
      </div>
      <div className="layout-siderail">
        <ErrorBoundary label="Side rail">{siderail}</ErrorBoundary>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Update `App.tsx`**

Find the `<Layout>` mount and replace:

```tsx
<Layout
  chat={
    <ChatPanel
      onOpenSecrets={openSecrets}
      onOpenComposerPreferences={openComposerSettings}
    />
  }
  siderail={<SideRail />}
/>
```

Delete the import of `InspectorPanel` and `OPEN_CATALOG_EVENT` from `InspectorPanel` (the latter was already repointed in Task 5; double-check no orphan import remains). Delete the import of `SessionSidebar` if any survived 15a's removal.

- [ ] **Step 4a: Relocate `<AuditReadinessPanel />` BEFORE deleting the inspector (CRITICAL)**

> **Review finding (CRITICAL):** This sub-step must execute before `git rm InspectorPanel.tsx`. Phase 2 (14c) placed `<AuditReadinessPanel />` inside the inspector; deleting the inspector without relocating it removes the panel from the UI.

**Actions:**
1. Search for the `<AuditReadinessPanel />` mount in `InspectorPanel.tsx`:
   ```bash
   grep -n "AuditReadinessPanel" src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx
   ```
2. Cut the mount AND the import line (`import { AuditReadinessPanel } from "../audit/AuditReadinessPanel";`) from `InspectorPanel.tsx`. Add the import to `App.tsx`:
   ```tsx
   import { AuditReadinessPanel } from "./components/audit/AuditReadinessPanel";
   ```
   Then pass it into `SideRail` via the `auditReadinessSlot` prop:
   ```tsx
   <SideRail
     auditReadinessSlot={<AuditReadinessPanel />}
     graphMiniSlot={<GraphMiniView />}
     catalogSlot={<CatalogButton />}
     exportYamlSlot={<ExportYamlButton />}
     executeButtonSlot={<ExecuteButton />}
     {/* completionBarSlot remains null for Phase 6 */}
   />
   ```
3. Add a **failing test** (write it first, then wire):
   ```typescript
   it("renders AuditReadinessPanel inside the side rail after Task 9", () => {
     // This test fails until the panel is wired via auditReadinessSlot.
     render(
       <SideRail
         auditReadinessSlot={<div data-testid="audit-readiness-panel" />}
       />
     );
     const slot = screen.getByTestId("siderail-slot-audit-readiness");
     expect(slot).toContainElement(screen.getByTestId("audit-readiness-panel"));
   });
   ```
4. Verify the test passes after the App.tsx wiring.

- [ ] **Step 5: Delete the inspector files**

```bash
cd src/elspeth/web/frontend
git rm src/components/inspector/InspectorPanel.tsx
git rm src/components/inspector/InspectorPanel.test.tsx
```

`GraphView.tsx`, `YamlView.tsx`, and `RunOutputsPanel.tsx` are **kept** — they're hosted by `GraphModal`, `ExportYamlModal`, and `InlineRunResults` respectively. `SpecView.tsx` and `RunsView.tsx` should already be gone from 15a; verify:

```bash
git ls-files src/components/inspector | grep -E "(SpecView|RunsView)"
```

If any survive, `git rm` them.

- [ ] **Step 6: Add the validation banner to `SideRail` via the slot-prop contract**

Three sub-steps:

1. Extend `SideRail`'s `SideRailProps` with `validationBannerSlot?: ReactNode | null` and render it in a wrapper `<div data-testid="siderail-slot-validation-banner">` above the audit-readiness slot.
2. Create `src/elspeth/web/frontend/src/components/sidebar/SideRailValidationBanner.tsx`. It subscribes to `useExecutionStore.validationResult`, `useExecutionStore.error`, and `useSessionStore.compositionState.nodes`. Renders the existing `<ValidationResultBanner />` when `validationResult` is non-null and the error banner when `executionStore.error` is non-null. `onComponentClick={(nodeId) => useSessionStore.getState().selectNode(nodeId)}` (no tab switch — 15a Task 6 already nooped that).
3. In `App.tsx`, pass `validationBannerSlot={<SideRailValidationBanner />}` alongside the other slots.

Add a failing slot-presence test in `SideRail.test.tsx`:

```typescript
it("renders content passed via the validationBanner slot prop", () => {
  render(
    <SideRail
      validationBannerSlot={<div data-testid="banner-content" />}
    />,
  );
  expect(screen.getByTestId("siderail-slot-validation-banner")).toContainElement(
    screen.getByTestId("banner-content"),
  );
});
```

The earlier-in-this-task `validationResult is present` test (which renders `<SideRail />` and expects the banner text) becomes an integration test against the wired `App.tsx` layout (or is updated to pass `<SideRailValidationBanner />` explicitly).

- [ ] **Step 7: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS. A non-trivial number of tests in `InspectorPanel.test.tsx` are deleted (along with the source file); the surviving tests in `GraphView.test.tsx`, `YamlView.test.tsx`, `RunOutputsPanel.test.tsx` still pass because those components are now mounted under their respective modals / `InlineRunResults`.

The deletion of `Layout.test.tsx`'s overlay/sidebar/resize tests removes a substantial chunk — that's expected; those tests exercised behaviour that is structurally gone.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): delete InspectorPanel + simplify Layout (Phase 3B.9)

InspectorPanel.tsx is gone; its remaining responsibilities (validation
banner display, Graph and YAML rendering) are absorbed by SideRail,
GraphModal, and ExportYamlModal.  Layout is now a two-column grid:
chat + siderail.  The resize handle, sidebar overlay mode, and
collapsed-sidebar branches are deleted — design doc 03 specifies a
fixed-width side rail.  GraphView, YamlView, RunOutputsPanel are
preserved as the modal / inline components."
```

---

## Task 10: Cleanup pass

**Goal:** remove now-orphaned imports, dead constants, dead CSS, and dead test fixtures.

- [ ] **Step 1: Find orphans**

```bash
cd src/elspeth/web/frontend
grep -RIn "InspectorPanel" src
grep -RIn "TAB_CHANGED_EVENT" src
grep -RIn "VALID_TABS" src
grep -RIn "SpecView" src
grep -RIn "RunsView" src
grep -RIn "SessionSidebar" src
```

Every match is either:
1. A surviving file that imports a now-deleted symbol — fix it (usually delete the import line).
2. A CSS class name in `App.css` or a module CSS file — delete the rule if it's no longer used. Use `grep` to confirm the class isn't referenced elsewhere first.
3. A test fixture / mock — delete.

- [ ] **Step 2: Delete `SWITCH_TAB_EVENT` and all dispatch/listener sites (IMPORTANT)**

> **Review finding (IMPORTANT):** `SWITCH_TAB_EVENT` is dead code after Phase 3B. Delete it now — do not defer to Phase 6. Per §"Open scope questions resolved" 6 (updated in 15b1): deletion in-phase, not deferred. This cleanup must retarget the two non-inspector user-facing callers that survived Phase 3A: `CompletionSummary.tsx` and `ReadinessRowDetail.tsx`.

Actions:
1. Find all references:
   ```bash
   grep -RIn "SWITCH_TAB_EVENT" src/elspeth/web/frontend/src
   ```
2. Retarget `src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.tsx`: import `OPEN_YAML_MODAL_EVENT` from `@/lib/composer-events`; the "Review YAML" button dispatches `new CustomEvent(OPEN_YAML_MODAL_EVENT)`. Update `CompletionSummary.test.tsx` so the button test listens for `OPEN_YAML_MODAL_EVENT` and says "opens the YAML export modal" rather than "dispatches a YAML inspector-tab switch".
3. Retarget `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx`: import `OPEN_GRAPH_MODAL_EVENT` from `@/lib/composer-events`; `handleJump` still calls `selectNode(componentId)`, then dispatches `new CustomEvent(OPEN_GRAPH_MODAL_EVENT)`, then closes the detail. Update `ReadinessRowDetail.test.tsx` to listen for `OPEN_GRAPH_MODAL_EVENT` and assert the graph modal request fired. Do not reintroduce tab state.
4. Verify `CommandPalette.tsx` already dispatches `OPEN_GRAPH_MODAL_EVENT` / `OPEN_YAML_MODAL_EVENT` directly from Task 7; fix it if a stale `SWITCH_TAB_EVENT` use remains.
5. Delete the `SWITCH_TAB_EVENT` export from `CommandPalette.tsx`.
6. Delete any remaining `dispatchEvent(SWITCH_TAB_EVENT, ...)` calls.
7. Delete any remaining `addEventListener(SWITCH_TAB_EVENT, ...)` listeners.
8. Re-run the grep across both source and tests:
   ```bash
   grep -RIn "SWITCH_TAB_EVENT" src/elspeth/web/frontend/src src/elspeth/web/frontend/tests
   ```
   Expected: zero matches.

> **No filigree observation is filed.** The work is done here.

- [ ] **Step 3: Delete dead CSS**

Inspect `src/styles/*.css` and `src/App.css` for class names that no longer have a host (`.inspector-`, `.tab-strip-`, `.layout-sidebar`, `.inspector-overlay-*`, `.resize-handle-*`). Delete every rule whose selectors are all dead. Run the build to confirm nothing produces a TypeScript compile error:

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 4: Final run-all + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS — *all* tests, every removal accounted for.

Also build:

```bash
cd src/elspeth/web/frontend && npm run build
```

Expected: build succeeds. No TypeScript errors, no unused-import warnings (if `tsconfig.json` has `noUnusedLocals: true`).

- [ ] **Step 5 (was 6): Commit**

```bash
git add -A
git commit -m "chore(web/frontend): Phase 3B cleanup — orphaned imports, dead CSS, SWITCH_TAB_EVENT

Removes lingering imports of deleted InspectorPanel symbols and CSS
rules whose hosts were deleted in Task 9.  SWITCH_TAB_EVENT is fully
deleted: export, all dispatch sites, and all listener sites removed.
Not deferred to Phase 6."
```

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Modal-open events fire before their listeners are mounted (e.g., a hash arrives before `<GraphModal />` has mounted) | Task 6 uses `queueMicrotask(() => dispatchEvent(...))` so the dispatch is deferred until after the current synchronous mount cycle completes. A cold-load race test is added to Task 6's test suite. |
| Validation banner becomes invisible after the inspector is deleted | Task 9 explicitly mounts the banner in `SideRail`'s `validation-banner` slot. The banner subscribes to the same `executionStore.validationResult` it always did. Phase 2 (audit-readiness panel) absorbs this slot later; until then the banner is the indicator. |
| Catalog drawer hosts in two places mid-phase (inspector mounts its own, side rail mounts another) | Task 5 cuts the inspector's `<CatalogDrawer>` mount in the same commit that mounts the side-rail one. There is never a state where both render. |
| `OPEN_CATALOG_EVENT` import path break | Task 5 moves the constant declaration to `CatalogButton.tsx` and uses `grep` to find every importer (`App.tsx`). Build will fail loudly if any importer is missed; mitigation is to run `npm run build` after Task 5. |
| Hash-router race: a graph/yaml fragment dispatches the modal event before `<GraphModal>` is mounted | Task 6 uses `queueMicrotask(() => dispatchEvent(...))` and includes a cold-load integration test that mounts `useHashRouter()` and `<GraphModal />` in the same tree. Do not use synchronous dispatch; if the cold-load test fails in implementation, fix the scheduling and test together. |
| Users hit Ctrl+Shift+G or Ctrl+Shift+Y in a browser that already binds them (e.g., Ctrl+Shift+Y is "history" in some Firefox versions) | The handlers call `e.preventDefault()` first; that's sufficient for app-focused key events. If a user reports a clash, the shortcut can be remapped without an architectural change. Documented in `ShortcutsHelp` and the design-doc-aligned vocabulary leaves room for rebinding. |
| Stale `#/{id}/spec` bookmarks open the page in a strange-looking state (no modal, no error) | The router's silent rewrite produces a canonical URL and the standard composition surface. There is no error. This is the design-intended behaviour — bookmark continues to work, just lands on the canonical home rather than the (now-deleted) spec listing. |
| The graph mini doesn't render anything useful for very large pipelines (>20 transforms) | The mini collapses to bucket counts ("src • 17 tx • sink"), not per-node rendering. The full graph modal is the right tool for large pipelines. Phase 8 polish may revisit if Marcus or Linda find the bucket compression unhelpful. |
| `RunsHistoryDrawer` (added in 15a) loses its mount when the inspector is deleted | 15a mounted `RunsHistoryDrawer` inside `InlineRunResults`, which lives in `ChatPanel`. The inspector deletion in Task 9 doesn't touch the chat column. `RunsHistoryDrawer` continues to render via `InlineRunResults`. |
| `validationResult.onComponentClick` callback dispatches a tab switch that no longer goes anywhere | 15a Task 6 already nooped the tab-switch portion. The remaining `selectNode` call fires; the next time the user opens the graph modal, the selected node will be highlighted. Acceptable as the bridge to Phase 2's "Explain" surface. |
| `Layout.test.tsx` lost a lot of tests | Net negative test count is expected — we removed a lot of structural behaviour (resize, overlay, sidebar collapse). The surviving tests cover the simpler two-column grid faithfully. Treat as feature, not regression. |
| Side rail width is hardcoded in `Layout.tsx` (no resize) | Intentional per design doc 03 ("desktop authoring surface >1024px," "audit-readiness panel and graph mini-view should remain visible at >=1280px"). A future Phase 6 or 8 may add a single user-controlled "rail width" persisted setting; not in scope here. |
| `npm run build` fails because of an orphan `TAB_CHANGED_EVENT` reference | Task 6 deletes the export; Task 9 deletes the file holding the orphan import (the inspector). Task 10 confirms with a `grep` sweep. Build is the final gate. |

## Memory references

- `project_composer_two_audiences` — confirms why the graph mini is a continuous (skilled-operator verification) surface rather than an on-demand (one-off) view. The same persona that wants a thin chat history wants a continuous graph.
- `project_db_migration_policy` — not relevant.
- `project_staging_deployment` — informs the deploy-and-smoke procedure for `elspeth.foundryside.dev`. After Task 9 lands, a staging smoke is recommended to verify hash-fragment redirects from the wild (real bookmarks the operator has lying around).
- `feedback_no_calendar_shipping_commitments` — no dates in this plan.
- `feedback_no_tests_for_skill_prompts` — not relevant; this plan touches components and routing, not skill prompts.
- `feedback_fix_errors_you_encounter` — applies to the cleanup pass in Task 10. If `grep` turns up an orphan reference in a file we didn't expect to touch, fix it; don't defer.

## Review history

### 2026-05-15 — Review panel applied

**CRITICAL (Systems):** `useHashRouter` now uses `queueMicrotask(() => dispatchEvent(...))` for modal-open dispatches, preventing the cold-load race where events fire before modal listeners have registered. A cold-load integration test added to Task 6 Step 1.

**IMPORTANT (Architecture):** `OPEN_GRAPH_MODAL_EVENT` and `OPEN_YAML_MODAL_EVENT` are standardized in the shared `src/lib/composer-events.ts` module. `useHashRouter`, modal components, and all consumers import from there; if earlier task execution introduced component-local exports, delete them during Task 6 Step 3a. Eliminates the hook -> component circular dependency risk.

**CRITICAL (Systems):** Task 9 has a new explicit sub-step 4a: BEFORE deleting `InspectorPanel.tsx`, relocate `<AuditReadinessPanel />` from inside the inspector into `SideRail.auditReadinessSlot` via `App.tsx`. Failing test added for the post-Task-9 state. Prevents silent removal of the panel from the UI.

**IMPORTANT (Systems):** `SWITCH_TAB_EVENT` is deleted in Task 10 Step 2, not deferred to Phase 6. The export, all dispatch sites, and all listener sites are removed in the cleanup pass. `CommandPalette.tsx` dispatches `OPEN_GRAPH_MODAL_EVENT` / `OPEN_YAML_MODAL_EVENT` directly. No filigree observation filed. §"Open scope questions resolved" in 15b1 updated to match.

**Cross-file decision (SWITCH_TAB_EVENT):** IMPORTANT (15b2 §4) supersedes SUGGESTION (15b1 §3). Deletion is in-phase.

### 2026-05-16 — Cross-phase coherence review applied

**CRITICAL (Architecture / Coherence #6):** Task 9 originally inlined the validation banner inside `SideRail`'s internal markup, violating the slot composition contract established in 15a1 ("caller passes content, SideRail places it"). Rewritten: `SideRail` gains a new `validationBannerSlot?: ReactNode | null` prop (this task is the first time the slot is reserved AND filled); a new `SideRailValidationBanner.tsx` under `components/sidebar/` owns the store subscriptions; `App.tsx` passes it as the slot prop. Slot-presence test added to `SideRail.test.tsx`; behaviour test moved to a new `SideRailValidationBanner.test.tsx`.

**CRITICAL (Quality / Reality):** Task 6 Step 1's cold-load race test referenced an undefined `<AppWithModals />` fixture. Replaced with an inline `HarnessTree` component that calls `useHashRouter()` and renders `<GraphModal />` — minimal integration tree, no project-wide fixture assumed. Imports for `render`, `screen`, `act`, and `GraphModal` documented.

### 2026-05-17 — Pre-dispatch NO-GO follow-up

**BLOCKER (Execution target) — Phase 3 worktree/branch made concrete.** Shared header now names `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on `feat/composer-phase-3-ia-cleanup` from `RC5.2`; the old Phase 2A worktree/branch are explicitly forbidden.

**IMPORTANT (Layout consistency) — Width persistence decision aligned with 15b1.** Task 9 now explicitly deletes the transitional `elspeth_inspector_width` reader/writer and keeps `SIDERAIL_WIDTH = 320` as a fixed Phase 3B width with no replacement persistence key.

**IMPORTANT (Hook compatibility) — 15b router preserves 15a return shape.** The 15b `useHashRouter` rewrite returns `{ redirectToast: null }` so `App.tsx` can keep the 15a destructuring while stale fragments are canonicalized immediately.

### 2026-05-17 — Authoritative-plan no-go blockers absorbed

**BLOCKER (Layout behaviour preservation):** Task 9 now preserves `DefaultModeChangedBanner` inside the chat column when simplifying `Layout.tsx`; the replacement code and tests were updated so Phase 1B's default-mode banner host is not silently dropped.

**BLOCKER (App wiring preservation):** Task 9's `App.tsx` snippet now keeps the existing `ChatPanel` callbacks (`onOpenSecrets`, `onOpenComposerPreferences`) instead of replacing the mount with bare `<ChatPanel />`.

**BLOCKER (Router test timing):** Task 6's graph/yaml fragment tests now await a microtask after `renderHook()` because the planned implementation deliberately dispatches modal-open events via `queueMicrotask`.

**BLOCKER (SWITCH_TAB_EVENT callers):** Task 10 now explicitly retargets `CompletionSummary.tsx` and `ReadinessRowDetail.tsx` to `OPEN_YAML_MODAL_EVENT` / `OPEN_GRAPH_MODAL_EVENT` before deleting `SWITCH_TAB_EVENT`.
