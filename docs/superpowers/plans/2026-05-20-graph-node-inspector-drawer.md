# Graph Node Inspector Drawer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a right-edge side drawer to the ELSPETH composer's GraphView that opens when an operator clicks a node, displaying that node's full resolved configuration (read-only) without losing graph context.

**Filigree:** [elspeth-11b73d5eea](https://filigree.local/elspeth-11b73d5eea) — "Graph node click in orchestrator — show node config in popup"

**Architecture:** New presentational component tree (`NodeInspectorDrawer` → `NodeInspectorBody` → `OptionsRenderer`) mounted inside `GraphModal.tsx` (the existing parent of `GraphView`). Visibility is driven entirely by `useSessionStore.selectedNodeId !== null` (already wired by the existing `onNodeClick` toggle in `GraphView.tsx:159-163`). The drawer's body is a discriminated renderer keyed on `node_type` (with synthetic discriminators for source/sink). No new backend endpoints; reads from the already-loaded `compositionState`.

**Tech Stack:** React 18 + TypeScript, Zustand store (`useSessionStore`), existing `useFocusTrap` hook, existing design tokens (`BADGE_COLORS`, `BADGE_BACKGROUNDS`, `VALIDATION_COLORS`), CSS in `App.css` following `.catalog-drawer` / `.runs-history-drawer` precedent.

**Adaptations from the input design spec** (verified against current codebase):

1. **"Open in YAML view" link is deferred to a follow-up.** `GraphView` and `YamlView` are not side-by-side tabs in this codebase — each lives in its own sidebar modal (`GraphModal.tsx`, `ExportYamlModal.tsx`). Cross-modal navigation would close the surface the operator came from (losing graph context) and land in a YAML view that has no node-anchor scroll support. The sidebar already provides YAML as an organic escalation path. Track as out-of-scope follow-up — pre-requisite is either a tabbed inspector layout or anchor support in `YamlDisplay.tsx`.
2. **The "node no longer in pipeline" error state is dropped.** `sessionStore.ts:589, 1002, 1133` auto-clear `selectedNodeId` whenever the selected node disappears from `compositionState`. The drawer simply closes when topology mutates the node out — the error state in the spec is unreachable in normal flow.
3. **The "Audit context" section (input/output schema digests, trust tier) is omitted from v1.** None of `SourceSpec`, `NodeSpec`, or `OutputSpec` in `src/elspeth/web/frontend/src/types/index.ts:90-131` carry these fields. Adding them would require backend extensions. Track as out-of-scope follow-up dependent on backend schema-digest exposure.
4. **The drawer mounts inside `GraphModal.tsx`** (not at App level or in a hypothetical `Layout.tsx`), because that is the actual parent of `GraphView` in this codebase.
5. **ESC handling uses capture-phase + `stopImmediatePropagation()`** so the drawer's close fires before `GraphModal`'s ESC-to-close, without coupling the two components.

**Out-of-scope follow-ups** (not v1, surface as separate filigree issues):

- "Open in YAML view" link (depends on tab layout or YAML anchor scrolling).
- "Audit context" section (depends on backend exposing schema digests and trust tier per node).
- Multi-select compare across two drawers (RC5-UX track, `elspeth-de91358c30`).
- Per-node runtime state surface (waits for execution-state design).
- Edit-from-drawer affordances (waits for editor-vs-viewer UX decision).

---

## File Structure

Files this plan creates or modifies:

- **Create** `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx` — drawer shell, visibility wiring, ESC + responsive layering, focus trap on mobile, live-region rebind announcement.
- **Create** `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx` — drawer-level integration tests.
- **Create** `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx` — discriminated renderer keyed on node shape (source / transform / gate / aggregation / coalesce / sink). Renders Identity + Options + Routing sections.
- **Create** `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.test.tsx` — per-shape rendering tests.
- **Create** `src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.tsx` — recursive renderer for option values: scalars (with truncation), nested objects (collapsible), arrays (counted + expandable), `null` (muted), blob_ref pills.
- **Create** `src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.test.tsx` — value-shape tests including blob_ref detection.
- **Modify** `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx` — mount `<NodeInspectorDrawer />` after `<GraphView />`.
- **Modify** `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` — add `tabIndex={0}` + `onKeyDown` (Enter/Space) on rendered ReactFlow nodes for keyboard reachability.
- **Modify** `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx` — assert keyboard reachability behavior.
- **Modify** `src/elspeth/web/frontend/src/App.css` — add `.node-inspector-drawer*` styles (responsive variants, animations, prefers-reduced-motion branch).

Splitting decisions: `NodeInspectorDrawer` owns lifecycle (visibility, ESC, focus, announcements). `NodeInspectorBody` owns per-shape discrimination — keeping it separate lets the shape-discrimination tests run without standing up store, focus-trap, and ESC machinery. `OptionsRenderer` owns recursive value rendering, including the blob_ref pill — separate because it's the part most likely to grow as new value shapes appear in plugin options.

---

## Task 1: Drawer shell + visibility wiring

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx`

- [ ] **Step 1: Write the failing visibility test**

```tsx
// NodeInspectorDrawer.test.tsx
import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, beforeEach } from "vitest";
import { useSessionStore } from "@/stores/sessionStore";
import { NodeInspectorDrawer } from "./NodeInspectorDrawer";
import type { CompositionState } from "@/types/index";

const baseComposition: CompositionState = {
  id: "comp-1",
  version: 1,
  source: { plugin: "csv", options: { path: "data.csv" }, on_success: "classify" },
  nodes: [
    {
      id: "classify",
      node_type: "transform",
      plugin: "llm.openai",
      input: "source",
      on_success: "approved",
      on_error: null,
      options: { model: "gpt-4" },
    },
  ],
  edges: [],
  outputs: [{ name: "approved", plugin: "csv", options: { path: "out.csv" } }],
  metadata: { name: null, description: null },
};

describe("NodeInspectorDrawer visibility", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: null });
    });
  });

  it("renders nothing when selectedNodeId is null", () => {
    render(<NodeInspectorDrawer />);
    expect(screen.queryByRole("dialog", { name: /node inspector/i })).toBeNull();
  });

  it("renders the drawer when selectedNodeId is set", () => {
    act(() => {
      useSessionStore.setState({ selectedNodeId: "classify" });
    });
    render(<NodeInspectorDrawer />);
    expect(screen.getByRole("dialog", { name: /classify/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: FAIL — module not found / `NodeInspectorDrawer` is not defined.

- [ ] **Step 3: Implement the minimal shell**

```tsx
// NodeInspectorDrawer.tsx
import { useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";

export function NodeInspectorDrawer(): JSX.Element | null {
  const selectedNodeId = useSessionStore((s) => s.selectedNodeId);
  const compositionState = useSessionStore((s) => s.compositionState);
  const titleRef = useRef<HTMLHeadingElement>(null);

  if (selectedNodeId === null || compositionState === null) {
    return null;
  }

  // Resolve the displayed node identity. The source node is rendered in
  // GraphView with the synthetic id "source" (see GraphView.tsx:350) — recognise
  // that here so the drawer opens for source clicks too.
  const displayName =
    selectedNodeId === "source"
      ? "source"
      : compositionState.nodes.find((n) => n.id === selectedNodeId)?.id ??
        compositionState.outputs.find((o) => o.name === selectedNodeId)?.name ??
        selectedNodeId;

  return (
    <aside
      className="node-inspector-drawer"
      role="dialog"
      aria-modal="false"
      aria-labelledby="node-inspector-title"
    >
      <header className="node-inspector-drawer-header">
        <h2 id="node-inspector-title" ref={titleRef}>
          {displayName}
        </h2>
      </header>
      <div className="node-inspector-drawer-body" />
    </aside>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: PASS — both tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx
git commit -m "feat(inspector): scaffold NodeInspectorDrawer shell wired to selectedNodeId"
```

---

## Task 2: ESC handler with capture-phase layering vs GraphModal

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx`

The drawer must close on ESC. The drawer is mounted inside `GraphModal`, which itself listens for ESC at the document level (`GraphModal.tsx:25-29`). Without capture-phase + `stopImmediatePropagation`, hitting ESC would close BOTH the drawer and the surrounding graph modal in one keypress, which is wrong: the operator only wants to dismiss the most-recently-opened layer.

- [ ] **Step 1: Write the failing ESC test**

```tsx
// In NodeInspectorDrawer.test.tsx, append:
import { fireEvent } from "@testing-library/react";

describe("NodeInspectorDrawer ESC handling", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: "classify" });
    });
  });

  it("calls selectNode(null) on ESC", () => {
    render(<NodeInspectorDrawer />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(useSessionStore.getState().selectedNodeId).toBeNull();
  });

  it("stops ESC propagation so an outer modal handler does not also fire", () => {
    render(<NodeInspectorDrawer />);
    let outerFired = false;
    const outerHandler = () => {
      outerFired = true;
    };
    document.addEventListener("keydown", outerHandler);
    try {
      fireEvent.keyDown(document, { key: "Escape" });
      expect(outerFired).toBe(false);
    } finally {
      document.removeEventListener("keydown", outerHandler);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: FAIL — drawer does not yet handle ESC.

- [ ] **Step 3: Implement capture-phase ESC handler**

In `NodeInspectorDrawer.tsx`, add the effect (just inside the component, after the early return):

```tsx
import { useEffect, useRef } from "react";
// ...
const selectNode = useSessionStore((s) => s.selectNode);

useEffect(() => {
  if (selectedNodeId === null) return;
  function onKeyDown(e: KeyboardEvent) {
    if (e.key !== "Escape") return;
    e.preventDefault();
    e.stopImmediatePropagation();
    selectNode(null);
  }
  // Capture phase so we run before GraphModal's bubble-phase listener.
  document.addEventListener("keydown", onKeyDown, { capture: true });
  return () => document.removeEventListener("keydown", onKeyDown, { capture: true });
}, [selectedNodeId, selectNode]);
```

Move the early return below this effect (effects must be unconditional in the component body — restructure so the effect is registered regardless, but it self-exits when `selectedNodeId === null`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: PASS — all four tests green.

- [ ] **Step 5: Add an explanatory comment and commit**

Inside the effect, above `addEventListener`, add:

```tsx
// Capture phase + stopImmediatePropagation: GraphModal (our mount parent) also
// listens for Escape on document — without capture phase + propagation halt, a
// single ESC press would close both the drawer AND the surrounding graph modal.
// Operators expect ESC to peel off layers one at a time.
```

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx
git commit -m "feat(inspector): drawer ESC closes drawer without escaping outer graph modal"
```

---

## Task 3: Header — type badge, plugin id, close button, validation indicator

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx`

The header shows: type badge (using `BADGE_COLORS` / `BADGE_BACKGROUNDS` from `@/styles/tokens`), the node id as `<h2>`, the plugin identifier as a subhead, a close button, and the validation status indicator (using `VALIDATION_COLORS`). Validation status is derived from the same `validation_errors` / `validation_warnings` / `validation_suggestions` arrays on `CompositionState` that `GraphView` already reads at `GraphView.tsx:557`.

- [ ] **Step 1: Write the failing header tests**

```tsx
// Append to NodeInspectorDrawer.test.tsx
describe("NodeInspectorDrawer header", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: "classify" });
    });
  });

  it("renders a transform type badge", () => {
    render(<NodeInspectorDrawer />);
    expect(screen.getByLabelText(/node type: transform/i)).toBeInTheDocument();
  });

  it("renders the plugin identifier", () => {
    render(<NodeInspectorDrawer />);
    expect(screen.getByText("llm.openai")).toBeInTheDocument();
  });

  it("renders a close button that clears selection", () => {
    render(<NodeInspectorDrawer />);
    const closeBtn = screen.getByRole("button", { name: /close node inspector/i });
    closeBtn.click();
    expect(useSessionStore.getState().selectedNodeId).toBeNull();
  });

  it("renders a valid status badge when no validation entries reference the node", () => {
    render(<NodeInspectorDrawer />);
    expect(screen.getByLabelText(/validation: ok/i)).toBeInTheDocument();
  });

  it("renders an error status badge when validation_errors mentions the node id", () => {
    act(() => {
      useSessionStore.setState({
        compositionState: { ...baseComposition, validation_errors: ["classify: missing option model"] },
        selectedNodeId: "classify",
      });
    });
    render(<NodeInspectorDrawer />);
    expect(screen.getByLabelText(/validation: error/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: FAIL — header content is empty.

- [ ] **Step 3: Implement the header**

In `NodeInspectorDrawer.tsx`, replace the existing `<header>` block and add a node-shape derivation. The shape derivation table:

| Selector matches | Node shape | Type label | Plugin source |
|---|---|---|---|
| `selectedNodeId === "source"` and `compositionState.source !== null` | `"source"` | `"source"` | `compositionState.source.plugin` |
| `compositionState.nodes.find((n) => n.id === selectedNodeId)` | `node.node_type` | same | `node.plugin ?? "(no plugin)"` |
| `compositionState.outputs.find((o) => o.name === selectedNodeId)` | `"sink"` | `"sink"` | `output.plugin` |
| otherwise | `null` (drawer renders an "unknown node" footer-only state) | — | — |

```tsx
import { BADGE_COLORS, BADGE_BACKGROUNDS, VALIDATION_COLORS } from "@/styles/tokens";
import type { SourceSpec, NodeSpec, OutputSpec, CompositionState } from "@/types/index";

type ResolvedNode =
  | { kind: "source"; spec: SourceSpec }
  | { kind: "transform" | "gate" | "aggregation" | "coalesce"; spec: NodeSpec }
  | { kind: "sink"; spec: OutputSpec }
  | { kind: "unknown"; id: string };

function resolveNode(state: CompositionState, id: string): ResolvedNode {
  if (id === "source" && state.source) return { kind: "source", spec: state.source };
  const node = state.nodes.find((n) => n.id === id);
  if (node) return { kind: node.node_type, spec: node };
  const output = state.outputs.find((o) => o.name === id);
  if (output) return { kind: "sink", spec: output };
  return { kind: "unknown", id };
}

type ValidationLevel = "valid" | "warning" | "error";

function validationLevelForNode(state: CompositionState, id: string): ValidationLevel {
  // Errors are flat strings; we conservatively flag the node as error if its id
  // appears anywhere in any error string. Warnings/suggestions are ValidationEntryDTO
  // shapes with a structured `component` field — match exactly on that.
  if (state.validation_errors?.some((msg) => msg.includes(id))) return "error";
  if (state.validation_warnings?.some((entry) => entry.component === id)) return "warning";
  return "valid";
}
```

Replace the header JSX:

```tsx
const resolved = resolveNode(compositionState, selectedNodeId);
const level = validationLevelForNode(compositionState, selectedNodeId);
const typeLabel =
  resolved.kind === "unknown" ? "unknown" : resolved.kind;
const pluginLabel =
  resolved.kind === "source"
    ? resolved.spec.plugin
    : resolved.kind === "sink"
      ? resolved.spec.plugin
      : resolved.kind === "unknown"
        ? null
        : resolved.spec.plugin ?? "(no plugin)";

return (
  <aside
    className="node-inspector-drawer"
    role="dialog"
    aria-modal="false"
    aria-labelledby="node-inspector-title"
  >
    <header className="node-inspector-drawer-header">
      <span
        className="node-inspector-type-badge"
        aria-label={`node type: ${typeLabel}`}
        style={{
          background: BADGE_BACKGROUNDS[typeLabel as keyof typeof BADGE_BACKGROUNDS] ?? "transparent",
          color: BADGE_COLORS[typeLabel as keyof typeof BADGE_COLORS] ?? "inherit",
        }}
      >
        {typeLabel}
      </span>
      <h2 id="node-inspector-title">{selectedNodeId}</h2>
      {pluginLabel !== null && (
        <p className="node-inspector-plugin">{pluginLabel}</p>
      )}
      <span
        className="node-inspector-validation"
        aria-label={`validation: ${level === "valid" ? "ok" : level}`}
        style={{ background: VALIDATION_COLORS[level] }}
      />
      <button
        type="button"
        className="node-inspector-close"
        aria-label="Close node inspector"
        onClick={() => selectNode(null)}
      >
        ×
      </button>
    </header>
    <div className="node-inspector-drawer-body" />
  </aside>
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: PASS — all header tests green; earlier tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx
git commit -m "feat(inspector): header with type badge, plugin id, validation status, close button"
```

---

## Task 4: OptionsRenderer — scalars, nested objects, arrays, null, blob_ref pills

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.test.tsx`

This is the recursive value renderer. Rules:

| Value shape | Rendering |
|---|---|
| `null` | `<span className="opt-null">null</span>` (muted — Tier 1 absence is meaningful, never omit). |
| `string`, `number`, `boolean` | Inline, monospaced; if string longer than 80 chars, single-line ellipsis + `title` attribute = full value. |
| `Array` | `Array · N items` with a `▶` expand button that reveals indexed children. |
| Plain object NOT matching blob_ref shape | `{ ... }` with `▶` expand revealing key/value rows, collapsed by default at the top level. |
| `{ kind: "blob_ref", sha256: string, size: number, ... }` | Pill: `blob_ref · {formatted size} · sha256 {sha256.slice(0,8)}…`. **Never inline the resolved content.** |

- [ ] **Step 1: Write the failing renderer tests**

```tsx
// OptionsRenderer.test.tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OptionsRenderer } from "./OptionsRenderer";

describe("OptionsRenderer", () => {
  it("renders scalars inline", () => {
    render(<OptionsRenderer value={{ a: "hello", b: 42, c: true }} />);
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("true")).toBeInTheDocument();
  });

  it("renders null as a muted literal, not as missing", () => {
    render(<OptionsRenderer value={{ a: null }} />);
    expect(screen.getByText("null")).toHaveClass("opt-null");
  });

  it("renders arrays with a count and reveals items when expanded", () => {
    render(<OptionsRenderer value={{ items: [1, 2, 3] }} />);
    expect(screen.getByText(/array · 3 items/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /expand items/i }));
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("collapses nested objects by default and expands on click", () => {
    render(<OptionsRenderer value={{ nested: { inner: "x" } }} />);
    expect(screen.queryByText("inner")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /expand nested/i }));
    expect(screen.getByText("inner")).toBeInTheDocument();
  });

  it("renders blob_ref-shaped values as a pill, never inline", () => {
    render(
      <OptionsRenderer
        value={{
          payload: {
            kind: "blob_ref",
            sha256: "abcdef0123456789",
            size: 3_456_789,
          },
        }}
      />,
    );
    const pill = screen.getByText(/blob_ref/);
    expect(pill).toHaveTextContent(/3\.3 MB/);
    expect(pill).toHaveTextContent(/abcdef01/);
    // No nested keys revealed — pill is opaque
    expect(screen.queryByText("sha256")).toBeNull();
  });

  it("truncates long strings with a title tooltip carrying the full value", () => {
    const long = "x".repeat(120);
    render(<OptionsRenderer value={{ blob: long }} />);
    const el = screen.getByTitle(long);
    expect(el.textContent?.length).toBeLessThan(100);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/OptionsRenderer.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement OptionsRenderer**

```tsx
// OptionsRenderer.tsx
import { useState } from "react";

const TRUNCATE_AT = 80;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`;
}

function isBlobRef(v: unknown): v is { kind: "blob_ref"; sha256: string; size: number } {
  return (
    typeof v === "object" &&
    v !== null &&
    (v as { kind?: unknown }).kind === "blob_ref" &&
    typeof (v as { sha256?: unknown }).sha256 === "string" &&
    typeof (v as { size?: unknown }).size === "number"
  );
}

function ValueCell({ value }: { value: unknown }): JSX.Element {
  if (value === null) return <span className="opt-null">null</span>;
  if (isBlobRef(value)) {
    return (
      <span className="opt-blob-ref" data-testid="blob-ref-pill">
        blob_ref · {formatBytes(value.size)} · sha256 {value.sha256.slice(0, 8)}…
      </span>
    );
  }
  if (Array.isArray(value)) return <ArrayCell items={value} />;
  if (typeof value === "object") return <ObjectCell value={value as Record<string, unknown>} />;
  const str = String(value);
  if (typeof value === "string" && str.length > TRUNCATE_AT) {
    return (
      <span className="opt-scalar opt-truncated" title={str}>
        {str.slice(0, TRUNCATE_AT)}…
      </span>
    );
  }
  return <span className="opt-scalar">{str}</span>;
}

function ArrayCell({ items, label = "items" }: { items: unknown[]; label?: string }): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <div className="opt-array">
      <button
        type="button"
        className="opt-toggle"
        aria-expanded={open}
        aria-label={`Expand ${label}`}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "▼" : "▶"} Array · {items.length} items
      </button>
      {open && (
        <ol className="opt-array-items">
          {items.map((item, i) => (
            <li key={i}>
              <ValueCell value={item} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function ObjectCell({ value }: { value: Record<string, unknown> }): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <div className="opt-object">
      <button
        type="button"
        className="opt-toggle"
        aria-expanded={open}
        aria-label="Expand nested"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "▼" : "▶"} {`{ ${Object.keys(value).length} keys }`}
      </button>
      {open && (
        <dl className="opt-rows">
          {Object.entries(value).map(([k, v]) => (
            <div key={k} className="opt-row">
              <dt>{k}</dt>
              <dd>
                <ValueCell value={v} />
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

export function OptionsRenderer({ value }: { value: Record<string, unknown> }): JSX.Element {
  return (
    <dl className="opt-rows opt-rows-top">
      {Object.entries(value).map(([k, v]) => (
        <div key={k} className="opt-row">
          <dt>{k}</dt>
          <dd>
            <ValueCell value={v} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
```

Note: top-level entries render their values directly (so scalars don't sit behind a collapse). Nested objects start collapsed. `ArrayCell`'s label defaults to `"items"` but the test uses a per-key label — the test calls `Expand items`, which matches because the array sits at the `items` key with the default label. If the discriminated body later passes a per-key label, override the prop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/OptionsRenderer.test.tsx`
Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.tsx \
        src/elspeth/web/frontend/src/components/inspector/OptionsRenderer.test.tsx
git commit -m "feat(inspector): OptionsRenderer with blob_ref pill, null-as-null, truncated scalars"
```

---

## Task 5: NodeInspectorBody — discriminated rendering of Identity + Options

**Files:**
- Create: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx`
- Create: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx` (mount body)

`NodeInspectorBody` accepts a `ResolvedNode` (the type from Task 3) and renders the Identity and Options sections. The Routing section comes in Task 6. The body must NOT render rows for fields that don't apply to a given shape — that's the "honest absence" rule from the spec.

- [ ] **Step 1: Write the failing per-shape tests**

```tsx
// NodeInspectorBody.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { NodeInspectorBody } from "./NodeInspectorBody";

describe("NodeInspectorBody Identity + Options", () => {
  it("renders Identity for a source", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "source",
          spec: { plugin: "csv", options: { path: "data.csv" }, on_success: "next" },
        }}
        selectedId="source"
      />,
    );
    expect(screen.getByText(/identity/i)).toBeInTheDocument();
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.getByText("data.csv")).toBeInTheDocument();
  });

  it("renders Identity for a transform with id and node_type", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "transform",
          spec: {
            id: "classify",
            node_type: "transform",
            plugin: "llm.openai",
            input: "source",
            on_success: "approved",
            on_error: null,
            options: { model: "gpt-4" },
          },
        }}
        selectedId="classify"
      />,
    );
    expect(screen.getByText(/transform/i)).toBeInTheDocument();
    expect(screen.getByText("classify")).toBeInTheDocument();
    expect(screen.getByText("llm.openai")).toBeInTheDocument();
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
  });

  it("renders Identity for a sink keyed by name", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "sink",
          spec: { name: "approved", plugin: "csv", options: { path: "out.csv" } },
        }}
        selectedId="approved"
      />,
    );
    expect(screen.getByText(/identity/i)).toBeInTheDocument();
    expect(screen.getByText("approved")).toBeInTheDocument();
    expect(screen.getByText("out.csv")).toBeInTheDocument();
  });

  it("does NOT render an Options section when options is empty", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "source",
          spec: { plugin: "csv", options: {}, on_success: "next" },
        }}
        selectedId="source"
      />,
    );
    expect(screen.queryByText(/^options$/i)).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorBody.test.tsx`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement NodeInspectorBody (Identity + Options only — Routing comes in Task 6)**

```tsx
// NodeInspectorBody.tsx
import type { SourceSpec, NodeSpec, OutputSpec } from "@/types/index";
import { OptionsRenderer } from "./OptionsRenderer";

export type ResolvedNode =
  | { kind: "source"; spec: SourceSpec }
  | { kind: "transform" | "gate" | "aggregation" | "coalesce"; spec: NodeSpec }
  | { kind: "sink"; spec: OutputSpec }
  | { kind: "unknown"; id: string };

function IdentitySection({
  resolved,
  selectedId,
}: {
  resolved: ResolvedNode;
  selectedId: string;
}): JSX.Element {
  const rows: Array<[string, string]> = [];
  if (resolved.kind === "source") {
    rows.push(["id", "source"], ["plugin", resolved.spec.plugin]);
  } else if (resolved.kind === "sink") {
    rows.push(["name", resolved.spec.name], ["plugin", resolved.spec.plugin]);
  } else if (resolved.kind !== "unknown") {
    rows.push(
      ["id", resolved.spec.id],
      ["node_type", resolved.spec.node_type],
      ["plugin", resolved.spec.plugin ?? "(no plugin)"],
    );
  } else {
    rows.push(["id", selectedId], ["note", "Node not found in current composition"]);
  }
  return (
    <section className="node-inspector-section" aria-labelledby="identity-heading">
      <h3 id="identity-heading">Identity</h3>
      <dl className="opt-rows">
        {rows.map(([k, v]) => (
          <div key={k} className="opt-row">
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function OptionsSection({ resolved }: { resolved: ResolvedNode }): JSX.Element | null {
  if (resolved.kind === "unknown") return null;
  const options = resolved.spec.options;
  if (!options || Object.keys(options).length === 0) return null;
  return (
    <section className="node-inspector-section" aria-labelledby="options-heading">
      <h3 id="options-heading">Options</h3>
      <OptionsRenderer value={options} />
    </section>
  );
}

export function NodeInspectorBody({
  resolved,
  selectedId,
}: {
  resolved: ResolvedNode;
  selectedId: string;
}): JSX.Element {
  return (
    <div className="node-inspector-body-inner">
      <IdentitySection resolved={resolved} selectedId={selectedId} />
      <OptionsSection resolved={resolved} />
    </div>
  );
}
```

- [ ] **Step 4: Wire body into the drawer**

In `NodeInspectorDrawer.tsx`:

```tsx
import { NodeInspectorBody, type ResolvedNode } from "./NodeInspectorBody";
```

Replace the empty `<div className="node-inspector-drawer-body" />` with:

```tsx
<div className="node-inspector-drawer-body">
  <NodeInspectorBody resolved={resolved} selectedId={selectedId} />
</div>
```

Move the `ResolvedNode` type and `resolveNode()` helper from `NodeInspectorDrawer.tsx` into `NodeInspectorBody.tsx` (where the type is now exported); import them back into the drawer. This co-locates the discrimination logic with its consumer.

- [ ] **Step 5: Run all inspector tests**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/`
Expected: PASS — body tests + drawer tests + options tests all green.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.test.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx
git commit -m "feat(inspector): discriminated Identity + Options body for all node shapes"
```

---

## Task 6: Routing section — per-shape field gating + reverse-edge lookup for sinks

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.test.tsx`

Per-shape routing table (from spec, distilled against the actual `NodeSpec` type at `types/index.ts:101`):

| Shape | Fields rendered |
|---|---|
| `source` | `on_success`, `on_validation_failure` (both optional). |
| `transform` | `input`, `on_success`, `on_error`. |
| `gate` | `input`, `on_success`, `on_error`, `routes` (object: condition value → target id), `condition`. |
| `aggregation` | `input`, `on_success`, `on_error`, `fork_to`, `branches`, `merge`, `policy`. |
| `coalesce` | `input`, `on_success`, `on_error`, `branches`, `policy`, `merge`. |
| `sink` | `inbound from` — derived by scanning `compositionState.edges` for `edge.to_node === selectedId` and listing each `from_node`. |

Hide fields that don't apply to the shape. Render `null` values (not omit), per the "honest absence" rule.

- [ ] **Step 1: Write the failing routing tests**

```tsx
// Append to NodeInspectorBody.test.tsx
describe("NodeInspectorBody Routing", () => {
  it("source shows on_success and on_validation_failure but never input/routes/fork_to", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "source",
          spec: { plugin: "csv", options: {}, on_success: "classify", on_validation_failure: "quarantine" },
        }}
        selectedId="source"
        edges={[]}
      />,
    );
    expect(screen.getByText(/on_success/i)).toBeInTheDocument();
    expect(screen.getByText("classify")).toBeInTheDocument();
    expect(screen.getByText(/on_validation_failure/i)).toBeInTheDocument();
    expect(screen.queryByText(/^input$/i)).toBeNull();
    expect(screen.queryByText(/^routes$/i)).toBeNull();
  });

  it("transform shows input/on_success/on_error and hides routes/fork_to/branches/policy/merge", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "transform",
          spec: {
            id: "n", node_type: "transform", plugin: "p", input: "source",
            on_success: "next", on_error: null, options: {},
          },
        }}
        selectedId="n"
        edges={[]}
      />,
    );
    expect(screen.getByText(/^input$/i)).toBeInTheDocument();
    expect(screen.getByText(/on_error/i)).toBeInTheDocument();
    expect(screen.queryByText(/^routes$/i)).toBeNull();
    expect(screen.queryByText(/^fork_to$/i)).toBeNull();
    expect(screen.queryByText(/^policy$/i)).toBeNull();
  });

  it("transform on_error: null renders as the literal null, not omitted", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "transform",
          spec: {
            id: "n", node_type: "transform", plugin: "p", input: "source",
            on_success: "next", on_error: null, options: {},
          },
        }}
        selectedId="n"
        edges={[]}
      />,
    );
    const onError = screen.getByText(/on_error/i).closest(".opt-row")!;
    expect(onError).toHaveTextContent("null");
  });

  it("gate renders routes and condition", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "gate",
          spec: {
            id: "g", node_type: "gate", plugin: "rule.threshold", input: "n",
            on_success: null, on_error: null, options: {},
            condition: "score > 0.8",
            routes: { high: "approved", low: "rejected" },
          },
        }}
        selectedId="g"
        edges={[]}
      />,
    );
    expect(screen.getByText(/^condition$/i)).toBeInTheDocument();
    expect(screen.getByText("score > 0.8")).toBeInTheDocument();
    expect(screen.getByText(/^routes$/i)).toBeInTheDocument();
    expect(screen.getByText("approved")).toBeInTheDocument();
    expect(screen.getByText("rejected")).toBeInTheDocument();
  });

  it("aggregation renders fork_to/branches/policy/merge and hides routes/condition", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "aggregation",
          spec: {
            id: "a", node_type: "aggregation", plugin: "agg.stats", input: "n",
            on_success: "next", on_error: null, options: {},
            fork_to: ["a-mean", "a-stddev"], branches: ["mean", "stddev"],
            policy: "wait_all", merge: "concat",
          },
        }}
        selectedId="a"
        edges={[]}
      />,
    );
    expect(screen.getByText(/fork_to/i)).toBeInTheDocument();
    expect(screen.getByText(/branches/i)).toBeInTheDocument();
    expect(screen.getByText(/policy/i)).toBeInTheDocument();
    expect(screen.getByText(/merge/i)).toBeInTheDocument();
    expect(screen.queryByText(/^routes$/i)).toBeNull();
    expect(screen.queryByText(/^condition$/i)).toBeNull();
  });

  it("sink renders 'inbound from' derived from edges, not its own fields", () => {
    render(
      <NodeInspectorBody
        resolved={{
          kind: "sink",
          spec: { name: "approved", plugin: "csv", options: {} },
        }}
        selectedId="approved"
        edges={[
          { id: "e1", from_node: "classify", to_node: "approved", edge_type: "on_success", label: null },
          { id: "e2", from_node: "review", to_node: "approved", edge_type: "on_success", label: null },
        ]}
      />,
    );
    expect(screen.getByText(/inbound from/i)).toBeInTheDocument();
    expect(screen.getByText("classify")).toBeInTheDocument();
    expect(screen.getByText("review")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorBody.test.tsx`
Expected: FAIL — Routing section not implemented; `edges` prop unknown.

- [ ] **Step 3: Implement RoutingSection**

In `NodeInspectorBody.tsx`, extend the component signature to accept `edges`:

```tsx
import type { EdgeSpec } from "@/types/index";

const ROUTING_FIELDS_BY_KIND: Record<
  Exclude<ResolvedNode["kind"], "unknown" | "source" | "sink">,
  Array<keyof NodeSpec>
> = {
  transform: ["input", "on_success", "on_error"],
  gate: ["input", "on_success", "on_error", "routes", "condition"],
  aggregation: ["input", "on_success", "on_error", "fork_to", "branches", "policy", "merge"],
  coalesce: ["input", "on_success", "on_error", "branches", "policy", "merge"],
};

function RoutingSection({
  resolved,
  edges,
  selectedId,
}: {
  resolved: ResolvedNode;
  edges: EdgeSpec[];
  selectedId: string;
}): JSX.Element | null {
  if (resolved.kind === "unknown") return null;

  if (resolved.kind === "source") {
    return (
      <section className="node-inspector-section" aria-labelledby="routing-heading">
        <h3 id="routing-heading">Routing</h3>
        <dl className="opt-rows">
          <Row label="on_success" value={resolved.spec.on_success ?? null} />
          <Row label="on_validation_failure" value={resolved.spec.on_validation_failure ?? null} />
        </dl>
      </section>
    );
  }

  if (resolved.kind === "sink") {
    const inbound = edges.filter((e) => e.to_node === selectedId).map((e) => e.from_node);
    return (
      <section className="node-inspector-section" aria-labelledby="routing-heading">
        <h3 id="routing-heading">Routing</h3>
        <dl className="opt-rows">
          <div className="opt-row">
            <dt>inbound from</dt>
            <dd>
              {inbound.length === 0 ? (
                <span className="opt-null">null</span>
              ) : (
                <ul className="opt-inbound">
                  {inbound.map((id) => (
                    <li key={id}>{id}</li>
                  ))}
                </ul>
              )}
            </dd>
          </div>
        </dl>
      </section>
    );
  }

  const fields = ROUTING_FIELDS_BY_KIND[resolved.kind];
  return (
    <section className="node-inspector-section" aria-labelledby="routing-heading">
      <h3 id="routing-heading">Routing</h3>
      <dl className="opt-rows">
        {fields.map((field) => (
          <Row key={field} label={field} value={resolved.spec[field] ?? null} />
        ))}
      </dl>
    </section>
  );
}

function Row({ label, value }: { label: string; value: unknown }): JSX.Element {
  return (
    <div className="opt-row">
      <dt>{label}</dt>
      <dd>
        {value === null ? (
          <span className="opt-null">null</span>
        ) : typeof value === "object" ? (
          <OptionsRenderer value={value as Record<string, unknown>} />
        ) : Array.isArray(value) ? (
          <ul className="opt-list">
            {(value as unknown[]).map((v, i) => (
              <li key={i}>{String(v)}</li>
            ))}
          </ul>
        ) : (
          <span className="opt-scalar">{String(value)}</span>
        )}
      </dd>
    </div>
  );
}
```

Update `NodeInspectorBody` and its export to thread `edges`:

```tsx
export function NodeInspectorBody({
  resolved,
  selectedId,
  edges,
}: {
  resolved: ResolvedNode;
  selectedId: string;
  edges: EdgeSpec[];
}): JSX.Element {
  return (
    <div className="node-inspector-body-inner">
      <IdentitySection resolved={resolved} selectedId={selectedId} />
      <OptionsSection resolved={resolved} />
      <RoutingSection resolved={resolved} edges={edges} selectedId={selectedId} />
    </div>
  );
}
```

- [ ] **Step 4: Pass `edges` through from the drawer**

In `NodeInspectorDrawer.tsx`:

```tsx
<NodeInspectorBody
  resolved={resolved}
  selectedId={selectedNodeId}
  edges={compositionState.edges}
/>
```

- [ ] **Step 5: Run all inspector tests**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/`
Expected: PASS — every test green; routing per-shape gates correctly; sink inbound list derived from edges.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.test.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx
git commit -m "feat(inspector): per-shape Routing section with edge-derived sink inbound list"
```

---

## Task 7: Responsive variant — modal on mobile, non-modal on tablet/desktop, with focus trap

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx`

Spec rules:

| Viewport | Drawer width | `aria-modal` | Focus trap | Backdrop scrim |
|---|---|---|---|---|
| `<768px` (mobile) | `100vw` | `"true"` | Active | Present |
| `768–1024px` (tablet) | `360px` | `"false"` | Inactive | None |
| `>1024px` (desktop) | `clamp(360px, 33vw, 480px)` | `"false"` | Inactive | None |

Detect via `window.matchMedia("(max-width: 767.98px)")`. Use a `useState` + `addEventListener("change")` pattern so updates fire on viewport resize.

Nested-trap behavior: `GraphModal` registers `useFocusTrap` on the whole modal (`GraphModal.tsx:11`). On mobile, the drawer's own focus trap takes over because the drawer fully overlays the graph — nothing meaningful is reachable behind it. On desktop the drawer is non-modal, so the outer trap correctly clamps focus inside GraphModal (which includes the drawer).

- [ ] **Step 1: Write the failing responsive tests**

```tsx
// Append to NodeInspectorDrawer.test.tsx
function mockMatchMedia(matches: boolean) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches,
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  }));
}

describe("NodeInspectorDrawer responsive variant", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: "classify" });
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses aria-modal=true on mobile", () => {
    mockMatchMedia(true);
    render(<NodeInspectorDrawer />);
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "true");
  });

  it("uses aria-modal=false on tablet/desktop", () => {
    mockMatchMedia(false);
    render(<NodeInspectorDrawer />);
    expect(screen.getByRole("dialog")).toHaveAttribute("aria-modal", "false");
  });

  it("renders the mobile backdrop only on mobile", () => {
    mockMatchMedia(true);
    const { rerender } = render(<NodeInspectorDrawer />);
    expect(screen.getByTestId("node-inspector-backdrop")).toBeInTheDocument();

    mockMatchMedia(false);
    rerender(<NodeInspectorDrawer />);
    expect(screen.queryByTestId("node-inspector-backdrop")).toBeNull();
  });

  it("backdrop click on mobile clears selection", () => {
    mockMatchMedia(true);
    render(<NodeInspectorDrawer />);
    fireEvent.click(screen.getByTestId("node-inspector-backdrop"));
    expect(useSessionStore.getState().selectedNodeId).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: FAIL — drawer doesn't yet branch on viewport.

- [ ] **Step 3: Implement the responsive hook + variants**

In `NodeInspectorDrawer.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

function useIsMobileViewport(): boolean {
  const [isMobile, setIsMobile] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(max-width: 767.98px)").matches;
  });
  useEffect(() => {
    const mql = window.matchMedia("(max-width: 767.98px)");
    const onChange = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return isMobile;
}
```

Inside the component:

```tsx
const isMobile = useIsMobileViewport();
const drawerRef = useRef<HTMLDivElement>(null);
useFocusTrap(drawerRef, isMobile && selectedNodeId !== null, ".node-inspector-close");
```

Update the JSX to conditionally render the backdrop and switch `aria-modal`:

```tsx
return (
  <>
    {isMobile && (
      <div
        className="node-inspector-backdrop"
        data-testid="node-inspector-backdrop"
        aria-hidden="true"
        onClick={() => selectNode(null)}
      />
    )}
    <aside
      ref={drawerRef}
      className={`node-inspector-drawer${isMobile ? " node-inspector-drawer--mobile" : ""}`}
      role="dialog"
      aria-modal={isMobile ? "true" : "false"}
      aria-labelledby="node-inspector-title"
    >
      {/* header + body unchanged */}
    </aside>
  </>
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: PASS — responsive tests green; prior tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx
git commit -m "feat(inspector): mobile-modal variant with backdrop + focus trap; desktop non-modal"
```

---

## Task 8: Conditional live-region rebind announcement

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx`

Spec rule: when the operator clicks a *different* graph node while the drawer is already open, the drawer rebinds (no remount). Screen readers will not announce the change because the dialog is already mounted. Add a `role="status"` live region that announces `Inspecting {new-node-id}` on `selectedNodeId` *change while the drawer is mounted* — NOT on initial mount (the dialog open is the announcement) and NOT on close.

Implementation: store the previous `selectedNodeId` in a ref; fire the announcement only when prev was non-null and new is non-null and different.

- [ ] **Step 1: Write the failing live-region tests**

```tsx
// Append to NodeInspectorDrawer.test.tsx
describe("NodeInspectorDrawer live-region rebind", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: "classify" });
    });
  });

  it("does NOT populate the live region on initial mount", () => {
    render(<NodeInspectorDrawer />);
    const status = screen.getByRole("status");
    expect(status.textContent).toBe("");
  });

  it("populates the live region when rebinding from one node to another", () => {
    render(<NodeInspectorDrawer />);
    act(() => {
      useSessionStore.setState({ selectedNodeId: "approved" });
    });
    expect(screen.getByRole("status")).toHaveTextContent(/inspecting approved/i);
  });

  it("clears the live region on close (selectedNodeId -> null) — drawer unmounts so this is implicit", () => {
    const { container } = render(<NodeInspectorDrawer />);
    act(() => {
      useSessionStore.setState({ selectedNodeId: null });
    });
    expect(container.querySelector('[role="status"]')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: FAIL — no live region present.

- [ ] **Step 3: Implement the live region**

In `NodeInspectorDrawer.tsx`:

```tsx
const [rebindAnnouncement, setRebindAnnouncement] = useState<string>("");
const previousSelectedId = useRef<string | null>(null);

useEffect(() => {
  const prev = previousSelectedId.current;
  // Only fire on prev non-null + new non-null + different.
  if (prev !== null && selectedNodeId !== null && prev !== selectedNodeId) {
    setRebindAnnouncement(`Inspecting ${selectedNodeId}`);
  }
  previousSelectedId.current = selectedNodeId;
}, [selectedNodeId]);
```

Inside the drawer JSX, just before the closing `</aside>`:

```tsx
<div role="status" aria-live="polite" className="node-inspector-live">
  {rebindAnnouncement}
</div>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/NodeInspectorDrawer.test.tsx`
Expected: PASS — initial mount has empty status; rebind sets text; close unmounts drawer (live region gone).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx \
        src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.test.tsx
git commit -m "feat(inspector): live-region rebind announcement on selectedNodeId change while open"
```

---

## Task 9: Mount drawer in GraphModal + smoke integration test

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx`
- Modify: `src/elspeth/web/frontend/src/components/sidebar/GraphModal.test.tsx` (or create if absent)

The drawer mounts after `<GraphView />` so it sits in the same modal-content stacking context. Visibility is driven by `selectedNodeId`; the drawer takes care of returning `null` when no node is selected.

- [ ] **Step 1: Check if GraphModal.test.tsx exists**

Run: `ls src/elspeth/web/frontend/src/components/sidebar/GraphModal.test.tsx 2>&1`

If absent, create it in the next step. If present, append.

- [ ] **Step 2: Write the failing integration test**

```tsx
// GraphModal.test.tsx (create or extend)
import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, beforeEach } from "vitest";
import { GraphModal } from "./GraphModal";
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

const baseComposition = {
  id: "comp-1", version: 1,
  source: { plugin: "csv", options: {}, on_success: "classify" },
  nodes: [{
    id: "classify", node_type: "transform" as const, plugin: "llm.openai",
    input: "source", on_success: null, on_error: null, options: {},
  }],
  edges: [], outputs: [], metadata: { name: null, description: null },
};

describe("GraphModal + NodeInspectorDrawer integration", () => {
  beforeEach(() => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: null });
    });
  });

  it("does not show drawer when modal is open but no node selected", () => {
    render(<GraphModal />);
    act(() => {
      window.dispatchEvent(new Event(OPEN_GRAPH_MODAL_EVENT));
    });
    expect(screen.queryByRole("dialog", { name: /classify/i })).toBeNull();
  });

  it("shows drawer when selectedNodeId is set while modal is open", () => {
    render(<GraphModal />);
    act(() => {
      window.dispatchEvent(new Event(OPEN_GRAPH_MODAL_EVENT));
      useSessionStore.setState({ selectedNodeId: "classify" });
    });
    expect(screen.getByRole("dialog", { name: /classify/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/sidebar/GraphModal.test.tsx`
Expected: FAIL — drawer not yet mounted.

- [ ] **Step 4: Mount the drawer in GraphModal**

In `GraphModal.tsx`:

```tsx
import { NodeInspectorDrawer } from "@/components/inspector/NodeInspectorDrawer";
```

Inside the modal body, after `<GraphView />`:

```tsx
<div className="graph-modal-body">
  <GraphView />
  <NodeInspectorDrawer />
</div>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/sidebar/GraphModal.test.tsx`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx \
        src/elspeth/web/frontend/src/components/sidebar/GraphModal.test.tsx
git commit -m "feat(inspector): mount NodeInspectorDrawer inside GraphModal"
```

---

## Task 10: Graph node keyboard reachability — Enter/Space opens drawer

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`
- Modify: `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`

GraphView currently wires `onNodeClick` (`GraphView.tsx:159-163`) but the rendered ReactFlow nodes are not keyboard-focusable. The spec requires Enter/Space to open the drawer for accessibility. Use the ReactFlow `onNodeKeyDown` prop (if available in the installed version) or, failing that, render keyboard-focusable wrappers inside `makeRfNode`.

- [ ] **Step 1: Check the installed ReactFlow API surface**

Run: `cd src/elspeth/web/frontend && node -e "const rf = require('@xyflow/react/package.json'); console.log(rf.version)"`

Then grep for whether `onNodeKeyDown` or `nodesFocusable` is exposed:

```bash
grep -rn "onNodeKeyDown\|nodesFocusable\|tabIndex" node_modules/@xyflow/react/dist/index.d.ts 2>/dev/null | head -10
```

The modern `@xyflow/react` exposes `nodesFocusable` on `<ReactFlow>` (makes nodes tab-focusable) and supports `onNodeKeyDown`. If the installed version lacks `onNodeKeyDown`, fall back to attaching a `data` field interpreted by the custom node renderer.

- [ ] **Step 2: Write the failing keyboard test**

```tsx
// Append to GraphView.test.tsx
import { fireEvent } from "@testing-library/react";

describe("GraphView keyboard reachability", () => {
  it("calls selectNode(nodeId) when Enter is pressed on a focused graph node", () => {
    // Set up store with a single node
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: null });
    });
    render(<GraphView />);
    // ReactFlow renders nodes with role="button" when nodesFocusable is enabled.
    const node = screen.getByRole("button", { name: /classify/i });
    node.focus();
    fireEvent.keyDown(node, { key: "Enter" });
    expect(useSessionStore.getState().selectedNodeId).toBe("classify");
  });

  it("calls selectNode(nodeId) when Space is pressed on a focused graph node", () => {
    act(() => {
      useSessionStore.setState({ compositionState: baseComposition, selectedNodeId: null });
    });
    render(<GraphView />);
    const node = screen.getByRole("button", { name: /classify/i });
    node.focus();
    fireEvent.keyDown(node, { key: " " });
    expect(useSessionStore.getState().selectedNodeId).toBe("classify");
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/GraphView.test.tsx -t "keyboard reachability"`
Expected: FAIL — nodes are not focusable.

- [ ] **Step 4: Enable focusable nodes + key handler**

In `GraphView.tsx`, find the `<ReactFlow … />` element (around line 596) and add:

```tsx
nodesFocusable
onNodeKeyDown={(event, node) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    selectNode(selectedNodeId === node.id ? null : node.id);
  }
}}
```

If `onNodeKeyDown` is not available in the installed `@xyflow/react`, attach the handler inside the custom node component's `tabIndex`/`onKeyDown` instead — locate `makeRfNode` (around `GraphView.tsx:340-385`) and add `tabIndex: 0` + an `onKeyDown` in the node `data` consumed by the custom node renderer. Note: this codebase may not have a custom node renderer; the default ReactFlow node accepts ARIA props via `nodesFocusable`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/inspector/GraphView.test.tsx`
Expected: PASS — Enter and Space both select the focused node.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/GraphView.tsx \
        src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx
git commit -m "feat(inspector): graph nodes are keyboard-focusable; Enter/Space opens inspector"
```

---

## Task 11: CSS — drawer styles, prefers-reduced-motion, design tokens

**Files:**
- Modify: `src/elspeth/web/frontend/src/App.css`

Locate the existing `.catalog-drawer` / `.runs-history-drawer` blocks (around `App.css:5327` and elsewhere) and add a `.node-inspector-drawer*` block matching the visual spec from the design input.

- [ ] **Step 1: Add the drawer styles**

Append to `App.css`:

```css
/* ============================================================================
   NodeInspectorDrawer
   Right-edge slide-over inside GraphModal. Non-modal on tablet/desktop, modal
   on mobile (full-screen + backdrop). prefers-reduced-motion drops animation.
   ============================================================================ */

.node-inspector-backdrop {
  position: absolute;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.4);
  z-index: 1;
}

.node-inspector-drawer {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: clamp(360px, 33vw, 480px);
  background-color: var(--color-surface-elevated);
  border-left: 1px solid var(--color-border-strong);
  box-shadow: -4px 0 24px rgba(0, 0, 0, 0.12);
  display: flex;
  flex-direction: column;
  z-index: 2;
  transform: translateX(0);
  transition: transform 200ms cubic-bezier(0.32, 0.72, 0, 1);
}

.node-inspector-drawer--mobile {
  width: 100vw;
  inset: 0;
}

@media (max-width: 1023.98px) and (min-width: 768px) {
  .node-inspector-drawer {
    width: 360px;
  }
}

.node-inspector-drawer-header {
  display: grid;
  grid-template-columns: auto 1fr auto auto;
  gap: 8px;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid var(--color-border);
  position: sticky;
  top: 0;
  background-color: var(--color-surface-elevated);
  z-index: 1;
}

.node-inspector-type-badge {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 2px 8px;
  border-radius: 4px;
}

.node-inspector-drawer-header h2 {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.node-inspector-plugin {
  grid-column: 1 / -1;
  margin: 0;
  font-size: 13px;
  color: var(--color-text-muted);
  font-family: var(--font-mono);
}

.node-inspector-validation {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.node-inspector-close {
  width: 40px;
  height: 40px;
  background: transparent;
  border: none;
  color: var(--color-text-muted);
  font-size: 20px;
  cursor: pointer;
  border-radius: 4px;
}

.node-inspector-close:hover {
  background-color: var(--color-surface-hover);
  color: var(--color-text);
}

.node-inspector-close:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
  color: var(--color-text);
}

@media (max-width: 767.98px) {
  .node-inspector-close {
    width: 48px;
    height: 48px;
  }
}

.node-inspector-drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.node-inspector-section + .node-inspector-section {
  margin-top: 20px;
}

.node-inspector-section h3 {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
  margin: 0 0 8px 0;
}

.opt-rows {
  margin: 0;
  display: grid;
  gap: 8px;
}

.opt-row {
  display: grid;
  grid-template-columns: 140px 1fr;
  gap: 12px;
  align-items: baseline;
}

.opt-row dt {
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-muted);
}

.opt-row dd {
  margin: 0;
  font-size: 14px;
  color: var(--color-text);
}

.opt-null {
  font-family: var(--font-mono);
  color: var(--color-text-muted);
  font-style: italic;
}

.opt-blob-ref {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  background-color: var(--color-surface-pressed);
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--color-text-muted);
}

.opt-scalar {
  font-family: var(--font-mono);
  font-size: 13px;
}

.opt-truncated {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.opt-toggle {
  background: transparent;
  border: none;
  color: var(--color-text);
  cursor: pointer;
  font-size: 13px;
  padding: 0;
  text-align: left;
}

.opt-toggle:focus-visible {
  outline: 2px solid var(--color-focus-ring);
  outline-offset: 2px;
}

.opt-array-items,
.opt-list,
.opt-inbound {
  padding-left: 16px;
  margin: 4px 0 0 0;
}

.node-inspector-live {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (prefers-reduced-motion: reduce) {
  .node-inspector-drawer {
    transition-duration: 0ms;
  }
}
```

- [ ] **Step 2: Visually verify in the dev server**

Run the staging deploy locally (per memory: source-checkout systemd/Caddy at elspeth.foundryside.dev) OR run the frontend dev server:

```bash
cd src/elspeth/web/frontend && npm run dev
```

Open the composer, open the Graph modal (from the sidebar), click a node. Verify:
- Drawer slides in from the right.
- Header shows type badge + node id + plugin + validation indicator + close button.
- Identity, Options, and Routing sections render correctly per node shape.
- Clicking pane closes drawer; clicking another node rebinds.
- ESC closes drawer but leaves the graph modal open. Pressing ESC again closes the graph modal.
- Resize the window to mobile width — drawer becomes full-screen with backdrop.
- Set `prefers-reduced-motion: reduce` in DevTools rendering panel — drawer appears/disappears without slide animation.

If you cannot test the UI (e.g. no display available), state this explicitly in the commit message rather than claiming visual verification.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/App.css
git commit -m "feat(inspector): NodeInspectorDrawer styles with responsive + reduced-motion variants"
```

---

## Verification & finishing

After all 11 tasks land, run the full frontend check:

```bash
cd src/elspeth/web/frontend
npx vitest run                                 # all unit + integration tests
npx tsc --noEmit                               # type-check
npm run lint                                   # if a lint script exists
npx playwright test --grep "inspector"         # if Playwright e2e suite covers this surface
```

Open three filigree observations for the deferred follow-ups (not while still completing the task — only at the end, since these are *legitimately out of scope* for this issue per the spec's own statements):

```bash
filigree observe \
  --message "Open in YAML view from NodeInspectorDrawer — depends on tabbed inspector layout or YAML anchor support" \
  --file-path src/elspeth/web/frontend/src/components/inspector/NodeInspectorDrawer.tsx

filigree observe \
  --message "NodeInspectorDrawer 'Audit context' section — needs backend exposing schema digests + trust_tier per node" \
  --file-path src/elspeth/web/frontend/src/components/inspector/NodeInspectorBody.tsx

filigree observe \
  --message "Multi-select compare across two NodeInspectorDrawers — RC5-UX follow-up under elspeth-de91358c30"
```

Close the original issue:

```bash
filigree close elspeth-11b73d5eea
```

---

## Plan Self-Review Notes

- **Spec coverage:** Every spec requirement that is reachable in this codebase has a task. The three adaptations (no YAML link, no error state, no audit-context section) are explicit and justified against verified code paths. Mobile/desktop responsive split is covered in Task 7. Reduced-motion in Task 11. Live-region rebind in Task 8. Keyboard reachability in Task 10. Discriminated body in Tasks 5–6. Blob-ref pill in Task 4. Tier-1 "render null, never omit" in Task 4 + Task 6.

- **Type consistency:** `ResolvedNode` is defined once in `NodeInspectorBody.tsx` and imported by `NodeInspectorDrawer.tsx`. The renderer keys off this single discriminated union. `EdgeSpec` is imported from `@/types/index`. `ROUTING_FIELDS_BY_KIND` only references actual `NodeSpec` keys.

- **No placeholders:** Every code step contains the actual content. No "TBD" / "add validation here" / "similar to Task N" stubs.

- **Known risk:** Task 10's `onNodeKeyDown` prop depends on the installed `@xyflow/react` version. If the API isn't there, the fallback (custom-node `tabIndex` + `onKeyDown`) requires touching `makeRfNode`'s rendered node data — defer this judgment to the implementer when they verify the installed API surface in Task 10 Step 1.
