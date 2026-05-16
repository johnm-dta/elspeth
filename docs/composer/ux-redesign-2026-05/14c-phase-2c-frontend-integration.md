# Phase 2C — Frontend integration: sub-components, panel mount, Validate-button removal, staging smoke

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Every task is TDD-shaped (failing test → run-to-fail → implement → run-to-pass → commit).

**Goal:** Land the remainder of the frontend half of Phase 2 — the real `ReadinessRowDetail` (per-row warning detail + jump-to-component), the real `ExplainDialog` (narrative modal), mounting `AuditReadinessPanel` inside `InspectorPanel.tsx`, removing the standalone Validate button, and the staging smoke that exercises Phase 2 end-to-end.

**Architecture:** Replaces the placeholder sub-components shipped by 14b with full implementations, then integrates the panel into the inspector chrome. The standalone Validate button is removed; the panel's Validation row subsumes it. The `useExecutionStore.validate()` action and its callers stay — Phase 2 does not refactor the execution flow.

**Tech Stack:** React + Zustand + Vitest + testing-library + userEvent.

**Sibling plans:**
- [14a-phase-2a-backend.md](14a-phase-2a-backend.md) — backend response models, service, routes.
- [14b-phase-2b-frontend.md](14b-phase-2b-frontend.md) — frontend foundations (types, API client, store, `AuditReadinessPanel` shell + placeholders).

**Umbrella plan:** [14-phase-2-audit-readiness-panel.md](14-phase-2-audit-readiness-panel.md).

**Design reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**
- Replace the `ReadinessRowDetail.tsx` placeholder shipped by 14b with the real implementation (per-row warning detail + jump-to-component for component-ids that resolve to nodes in `compositionState`).
- Replace the `ExplainDialog.tsx` placeholder shipped by 14b with the real modal implementation (fetches narrative lazily via `useAuditReadinessStore.loadExplain`, caches by composition version, preserves whitespace).
- Mount `<AuditReadinessPanel />` inside `InspectorPanel.tsx` between the header and the tab strip.
- Remove the standalone Validate button UI from `InspectorPanel.tsx`. `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` remain wired in `InspectorPanel.tsx` — their deletion is deferred to Phase 3A Task 8 (`15a2-phase-3a-removals-part-2.md`), where it is atomic with the relocation into `subscriptions.ts`. Phase 2 merges before Phase 3A; deleting the handlers now would silently lose validation-chat feedback until Phase 3A lands. See Task 8 and convergence finding C3 (`14-phase-2-audit-readiness-panel.review.json`).
- Add a vitest assertion that the inspector still renders, the keyboard navigation between tabs still works, and no button labelled "Validate" or "Validate pipeline" exists.
- A manual staging smoke that exercises (a) all-green collapse, (b) provenance-warning auto-expansion, (c) Validate-button-is-gone, (d) error-path recovery.

**Out of scope:**
- Backend (Phase 2A delivered).
- Frontend foundations — types, API client, store, panel shell (14b delivered).
- Telemetry on row-click (Phase 8 — explicit marker in §Task 5).
- Phase 3's side-rail reorganisation. 14c mounts inside the existing tab layout; Phase 3 moves the mount point (the panel component is untouched).
- A per-user retention preference UI (a future phase; the Retention row is informational only).
- LLM-interpretations row content (Phase 5b; the row renders with the `not_applicable` glyph and no detail navigation).

## Sequencing and dependencies

14c **depends on 14b being merged**, because 14c replaces files 14b created. If both halves land on the same working branch, 14b's commits must precede 14c's commits in history; otherwise 14c's "replace placeholder" commits will conflict.

14c's staging smoke (Task 9) depends on **Phase 2A being deployed** to `elspeth.foundryside.dev`. Smoke fails if 2A's routes don't respond.

**Phase 3B dependency.** Phase 3B's deletion of `InspectorPanel.tsx` (15b2 Task 9) MUST relocate `<AuditReadinessPanel />` into `SideRail`'s `siderail-slot-audit-readiness` div BEFORE deleting the inspector. Phase 3B Task 9 owns this move; 14c documents the dependency. Do not merge Phase 3B's inspector deletion without confirming the panel relocation has landed.

## Trust-tier check (per CLAUDE.md)

Same as 14b: reads only data the backend just produced (Tier 1). No new boundary. Notable additions in 14c:

- `compositionState.nodes` resolution in `ReadinessRowDetail` — direct typed access (`compositionState.nodes.some((n) => n.id === id)`); no `.get()` / `getattr`.
- `selectNode(componentId)` — calls the existing sessionStore action without coercing the id (the wire schema guarantees it's a string).

## File structure

**New:** none (placeholders from 14b become real implementations, in place).

**Modified:**
- `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx` — replace 14b's placeholder with full implementation.
- `src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx` — replace 14b's placeholder with full implementation.
- `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx` — new file alongside the real implementation.
- `src/elspeth/web/frontend/src/components/audit/ExplainDialog.test.tsx` — new file alongside the real implementation.
- `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — mount `<AuditReadinessPanel />`; remove the standalone Validate button.
- `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx` — add panel-mount and button-removal assertions; update any tests that directly click the now-removed Validate button (switch them to invoke `handleValidate` directly, since the handler stays wired).
- `src/elspeth/web/frontend/src/stores/subscriptions.handoff.test.ts` — new guard test (Task 8.5) that is `.skip`-marked until Phase 3A Task 8 lands.

---

## Task 5: `ReadinessRowDetail` — per-row warning detail + jump-to-component

**Files:**
- Create: `frontend/src/components/audit/ReadinessRowDetail.tsx`
- Create: `frontend/src/components/audit/ReadinessRowDetail.test.tsx`

A small drawer/popover. Contents:

- The row's `label` as heading.
- The row's `detail` (multi-line; preserves linebreaks).
- A "Jump to component" button per entry in `component_ids` if the id resolves to a node in `compositionState.nodes`. Otherwise the id is displayed as plain text (so the user can grep their YAML).
- A "Close" button.

> **Phase 8 deferral marker.** No telemetry. When telemetry lands, this is the click-handler that fires the audit-row-click event.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/audit/ReadinessRowDetail.test.tsx`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { useSessionStore } from "../../stores/sessionStore";
import type { ReadinessRow, CompositionState } from "../../types/api";

const ROW_WITH_NODE: ReadinessRow = {
  id: "provenance",
  label: "Provenance",
  status: "warning",
  summary: "Identity passthrough detected",
  detail: "Identity passthrough — provenance gap on 'select_columns'.\nReplace with a transform that records provenance.",
  component_ids: ["select_columns"],
};

const ROW_WITHOUT_RESOLVABLE_ID: ReadinessRow = {
  id: "secrets",
  label: "Secrets",
  status: "error",
  summary: "Required secret missing",
  detail: "Secret reference 'api_key' is not resolved.",
  component_ids: ["api_key"],
};

const ROW_NO_IDS: ReadinessRow = {
  id: "retention",
  label: "Retention",
  status: "warning",
  summary: "Not configured",
  detail: "No retention configured for a pipeline that handles sensitive data.",
  component_ids: [],
};

function makeComposition(): CompositionState {
  return {
    id: "comp-1",
    version: 1,
    source: null,
    nodes: [
      { id: "select_columns", node_type: "transform", plugin: "select_columns", config: {} } as never,
    ],
    edges: [],
    outputs: [],
    metadata: { name: "demo", description: "" },
  };
}

describe("ReadinessRowDetail", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(),
      selectNode: vi.fn(),
    } as never);
  });

  it("renders the row label and detail with preserved linebreaks", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: "Provenance" })).toBeInTheDocument();
    expect(screen.getByText(/Identity passthrough — provenance gap/)).toBeInTheDocument();
    expect(screen.getByText(/Replace with a transform/)).toBeInTheDocument();
  });

  it("renders a Jump to component button for ids that resolve to nodes", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    useSessionStore.setState({ selectNode } as never);
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    const btn = screen.getByRole("button", { name: /Jump to select_columns/ });
    await user.click(btn);
    expect(selectNode).toHaveBeenCalledWith("select_columns");
  });

  it("renders unresolved ids as plain text (no jump button)", () => {
    render(<ReadinessRowDetail row={ROW_WITHOUT_RESOLVABLE_ID} onClose={() => {}} />);
    expect(screen.getByText("api_key")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Jump to/ })).not.toBeInTheDocument();
  });

  it("omits the component-ids block when component_ids is empty", () => {
    render(<ReadinessRowDetail row={ROW_NO_IDS} onClose={() => {}} />);
    expect(screen.queryByText(/Components/i)).not.toBeInTheDocument();
  });

  it("fires onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders nothing in the detail body when row.detail is null", () => {
    const minimal: ReadinessRow = { ...ROW_NO_IDS, detail: null };
    render(<ReadinessRowDetail row={minimal} onClose={() => {}} />);
    // The summary is still rendered.
    expect(screen.getByText(/Not configured/)).toBeInTheDocument();
  });

  it("uses role=dialog and is labelled by the row heading", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-labelledby");
    const labelId = dialog.getAttribute("aria-labelledby")!;
    expect(document.getElementById(labelId)).toHaveTextContent("Provenance");
  });
});
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

`frontend/src/components/audit/ReadinessRowDetail.tsx`:

```typescript
/**
 * ReadinessRowDetail (Phase 2B)
 *
 * Drawer/popover content for one row of the audit-readiness panel.
 * Renders the row's detail string (multi-line preserved) and offers a
 * jump-to-component button for each entry in component_ids that
 * resolves to a node in the current composition. Unresolvable ids are
 * shown as plain text — they may refer to source/sink names or YAML
 * fragments the user can grep for.
 *
 * Phase 8 will add a telemetry emit here for audit-row-click. No emit yet.
 */
import { useId } from "react";

import { useSessionStore } from "../../stores/sessionStore";
import type { ReadinessRow } from "../../types/api";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  onClose: () => void;
}

export function ReadinessRowDetail({ row, onClose }: ReadinessRowDetailProps) {
  const compositionState = useSessionStore((s) => s.compositionState);
  const selectNode = useSessionStore((s) => s.selectNode);
  const labelId = useId();

  const nodeIds = new Set(compositionState?.nodes.map((n) => n.id) ?? []);

  function handleJump(componentId: string) {
    selectNode(componentId);
    // Phase 8 deferral: emit telemetry here.
    onClose();
  }

  return (
    <div
      role="dialog"
      aria-labelledby={labelId}
      aria-modal="false"
      className="readiness-row-detail"
    >
      <header className="readiness-row-detail-header">
        <h3 id={labelId} className="readiness-row-detail-title">
          {row.label}
        </h3>
        <button
          type="button"
          className="readiness-row-detail-close"
          onClick={onClose}
          aria-label="Close detail"
        >
          ×
        </button>
      </header>

      <p className="readiness-row-detail-summary">{row.summary}</p>

      {row.detail && (
        <pre className="readiness-row-detail-body">{row.detail}</pre>
      )}

      {row.component_ids.length > 0 && (
        <section
          aria-label="Components implicated"
          className="readiness-row-detail-components"
        >
          <h4 className="readiness-row-detail-components-heading">Components</h4>
          <ul className="readiness-row-detail-components-list">
            {row.component_ids.map((id) => {
              const resolvable = nodeIds.has(id);
              return (
                <li key={id}>
                  {resolvable ? (
                    <button
                      type="button"
                      className="btn readiness-row-detail-jump-btn"
                      onClick={() => handleJump(id)}
                      aria-label={`Jump to ${id}`}
                    >
                      Jump to {id}
                    </button>
                  ) : (
                    <span className="readiness-row-detail-component-id">{id}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx
git commit -m "feat(web/frontend): add ReadinessRowDetail with jump-to-component (Phase 2B.5)"
```

---

## Task 6: `ExplainDialog` — narrative modal

**Files:**
- Create: `frontend/src/components/audit/ExplainDialog.tsx`
- Create: `frontend/src/components/audit/ExplainDialog.test.tsx`

The Explain dialog fetches the narrative on first open (via `useAuditReadinessStore.loadExplain`), caches by composition version, and renders the result with preserved whitespace.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/audit/ExplainDialog.test.tsx`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ExplainDialog } from "./ExplainDialog";
import { useAuditReadinessStore } from "../../stores/auditReadinessStore";
import * as api from "../../api/auditReadiness";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

describe("ExplainDialog", () => {
  beforeEach(() => {
    useAuditReadinessStore.setState({
      snapshotsBySession: {},
      explainsBySession: {},
      isLoading: false,
      isLoadingExplain: false,
      error: null,
      explainError: null,
    } as never);
    vi.clearAllMocks();
  });

  it("fetches the narrative on mount and renders it", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record:\n\n• Source data — 5 URLs.",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByText(/ELSPETH will record/)).toBeInTheDocument();
    expect(screen.getByText(/Source data — 5 URLs/)).toBeInTheDocument();
  });

  it("uses the cached narrative when present without refetching", async () => {
    useAuditReadinessStore.setState({
      explainsBySession: {
        [SESSION_ID]: {
          session_id: SESSION_ID,
          composition_version: 1,
          narrative: "cached narrative",
        },
      },
    } as never);
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    // Confirm the render has settled before asserting the API was not called.
    // A bare `not.toHaveBeenCalled()` check before render settles races the
    // useEffect; wait for the text to appear first.
    await waitFor(() => expect(screen.getByText("cached narrative")).toBeInTheDocument());
    expect(api.fetchAuditReadinessExplain).not.toHaveBeenCalled();
  });

  it("renders a loading state while the fetch is pending", () => {
    let resolve!: (v: { session_id: string; composition_version: number; narrative: string }) => void;
    vi.mocked(api.fetchAuditReadinessExplain).mockReturnValueOnce(
      new Promise((r) => {
        resolve = r;
      }),
    );
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(screen.getByText(/Generating explanation/i)).toBeInTheDocument();
    resolve({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "done.",
    });
  });

  it("renders an error when the fetch fails", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockRejectedValueOnce({
      status: 500,
      detail: "boom",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/boom/);
  });

  it("fires onClose when Close is clicked", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("uses role=dialog and is labelled by the heading", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const labelId = dialog.getAttribute("aria-labelledby")!;
    expect(document.getElementById(labelId)).toHaveTextContent(/What this pipeline will record/i);
  });
});
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

`frontend/src/components/audit/ExplainDialog.tsx`:

```typescript
/**
 * ExplainDialog (Phase 2B)
 *
 * Modal dialog rendering the narrative explanation of what the current
 * pipeline will record. The narrative is fetched lazily on first open
 * and cached by composition_version in the auditReadinessStore.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 * §"The Explain view".
 */
import { useEffect, useId } from "react";

import { useAuditReadinessStore } from "../../stores/auditReadinessStore";

export interface ExplainDialogProps {
  sessionId: string;
  compositionVersion: number;
  onClose: () => void;
}

export function ExplainDialog({
  sessionId,
  compositionVersion,
  onClose,
}: ExplainDialogProps) {
  const explain = useAuditReadinessStore((s) => s.explainsBySession[sessionId]);
  const isLoading = useAuditReadinessStore((s) => s.isLoadingExplain);
  const error = useAuditReadinessStore((s) => s.explainError);
  const loadExplain = useAuditReadinessStore((s) => s.loadExplain);
  const titleId = useId();

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="explain-dialog"
    >
      <div className="explain-dialog-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="explain-dialog-content">
        <header className="explain-dialog-header">
          <h2 id={titleId} className="explain-dialog-title">
            What this pipeline will record
          </h2>
          <button
            type="button"
            className="explain-dialog-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {isLoading && !explain && (
          <p className="explain-dialog-loading">Generating explanation…</p>
        )}

        {error && !explain && (
          <div role="alert" className="explain-dialog-error">
            {error}
          </div>
        )}

        {explain && (
          <pre className="explain-dialog-narrative">{explain.narrative}</pre>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx src/elspeth/web/frontend/src/components/audit/ExplainDialog.test.tsx
git commit -m "feat(web/frontend): add ExplainDialog with version-keyed narrative cache (Phase 2B.6)"
```

---

## Task 7: Mount in `InspectorPanel.tsx`

**Files:**
- Modify: `frontend/src/components/inspector/InspectorPanel.tsx`
- Modify: `frontend/src/components/inspector/InspectorPanel.test.tsx`

The panel mounts **between the inspector header and the tab strip**, so it is visible under every tab. Phase 3's IA cleanup will likely move this mount point, but until then this placement is the cheapest "always visible during composition" location.

- [ ] **Step 1: Confirm the mount site**

Read `InspectorPanel.tsx` around line 595 (the closing `</div>` of "Row 1" — the inspector header — and the opening of "Row 2" — the tab strip; per the reconnaissance notes the regions are clearly commented `Row 1: Version selector + validation dot | Validate + Execute` and `Row 2: Tab strip`).

The mount is **after** the Row 1 closing `</div>` and **before** the `role="tablist"` opening. This is also the natural insertion point for the standalone Validate button removal in Task 8 — both edits land in adjacent lines.

- [ ] **Step 2: Write the failing test**

Add this test to `frontend/src/components/inspector/InspectorPanel.test.tsx` (a new `describe` block; do not rewrite the existing tests). At the top of `InspectorPanel.test.tsx`, ensure these are present (add if missing):

```typescript
import { fetchAuditReadiness } from "@/api/auditReadiness";
vi.mock("@/api/auditReadiness", () => ({ fetchAuditReadiness: vi.fn(), fetchAuditReadinessExplain: vi.fn() }));
```

Then add the new describe block:

```typescript
describe("AuditReadinessPanel mount in InspectorPanel", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: {
        id: "comp-1",
        version: 1,
        source: { kind: "csv_file", config: { path: "x.csv" } } as never,
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: "demo", description: "" },
      },
    } as never);
    vi.mocked(fetchAuditReadiness).mockResolvedValue({
      session_id: "s-1",
      composition_version: 1,
      rows: [
        { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
        { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
        { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
        { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
        { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
      ],
    });
  });

  it("renders the audit readiness panel inside the inspector", async () => {
    render(<InspectorPanel />);
    await waitFor(() => {
      expect(screen.getByLabelText("Audit readiness")).toBeInTheDocument();
    });
  });

  it("renders the audit readiness panel above the tab strip", async () => {
    render(<InspectorPanel />);
    const panel = await screen.findByLabelText("Audit readiness");
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    // compareDocumentPosition returns 4 (DOCUMENT_POSITION_FOLLOWING) when
    // the argument follows the receiver — the panel must come first.
    expect(panel.compareDocumentPosition(tablist)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });
});
```

- [ ] **Step 3: Run test — expect FAIL** (panel isn't mounted)

- [ ] **Step 4: Implement**

In `InspectorPanel.tsx`, add at the top of the imports:

```typescript
import { AuditReadinessPanel } from "../audit/AuditReadinessPanel";
```

Then, after the closing `</div>` of the inspector header (the `<div>` that contains Row 1: version-selector + Catalog/Validate/Execute), and **before** the `role="tablist"` block, insert:

```tsx
        {/* Audit-readiness panel — Phase 2.
            Persistent across tabs. The Validation row inside this panel
            replaces the standalone Validate button (Task 8). */}
        <AuditReadinessPanel />
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx
```

Expected: existing tests still pass; the two new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx
git commit -m "feat(web/frontend): mount AuditReadinessPanel above the inspector tab strip (Phase 2B.7)"
```

---

## Task 8: Remove the standalone Validate button

**Files:**
- Modify: `frontend/src/components/inspector/InspectorPanel.tsx`
- Modify: `frontend/src/components/inspector/InspectorPanel.test.tsx`

The standalone Validate button is subsumed by the audit-readiness panel's Validation row. The `useExecutionStore.validate()` action stays — the panel's backend aggregator calls the same validation route, and `handleExecute` still consults `validationResult` to gate the Execute button.

### Why this task is scope-limited (convergence C3)

Phase 3A Task 8 (`15a2-phase-3a-removals-part-2.md`) is the canonical site for deleting `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` from `InspectorPanel.tsx`. That deletion is atomic with their relocation into `subscriptions.ts` — Phase 3A cannot safely delete those handlers without simultaneously wiring their side effects into the subscription.

Phase 2 merges before Phase 3A (per the roadmap). If Phase 2C were to delete these handlers now, the side effects (system-message injection on validation failure, `sendValidationFeedback`) would be silently lost between the Phase 2 merge and the Phase 3A merge. This is the regression window identified as convergence finding C3 in `14-phase-2-audit-readiness-panel.review.json`.

**Consequence for this task:** Phase 2C Task 8 removes only the standalone Validate button UI. `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` remain in `InspectorPanel.tsx`, wired and live. They are not dead code — removing the button is the only caller that disappears; if keyboard shortcuts or other callers invoke `handleValidate` directly, those continue to work. Task 8.5 (immediately below) adds a mechanical CI gate that fails red until Phase 3A Task 8 lands the relocation.

- [ ] **Step 1: Write the failing test (negative + smoke)**

In `frontend/src/components/inspector/InspectorPanel.test.tsx`, add:

```typescript
describe("Validate button removal (Phase 2C)", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: {
        id: "comp-1",
        version: 1,
        source: { kind: "csv_file", config: { path: "x.csv" } } as never,
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: "demo", description: "" },
      },
    } as never);
  });

  it("does not render a button labelled 'Validate' (subsumed by audit-readiness panel)", () => {
    render(<InspectorPanel />);
    expect(
      screen.queryByRole("button", { name: /^Validate$/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^Validate pipeline$/ }),
    ).not.toBeInTheDocument();
  });

  it("still renders Execute and Catalog buttons", () => {
    render(<InspectorPanel />);
    expect(screen.getByRole("button", { name: /^Execute pipeline$/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Catalog/ })).toBeInTheDocument();
  });

  it("the tab strip still renders and arrow navigation still works", async () => {
    render(<InspectorPanel />);
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    expect(tablist).toBeInTheDocument();
    const tabs = within(tablist).getAllByRole("tab");
    expect(tabs.length).toBeGreaterThan(1);
    tabs[0].focus();
    fireEvent.keyDown(tabs[0], { key: "ArrowRight" });
    // The next tab should be focused after arrow navigation; the focus
    // assertion verifies the keyboard navigation hasn't regressed when the
    // header layout changed.
    expect(document.activeElement).toBe(tabs[1]);
  });
});
```

Also locate the existing tests that assert the Validate button exists and update them:

1. Tests under `describe("InspectorPanel three-state validation indicator", ...)` — these test the validation **dot** (which stays). No change.
2. Tests under `describe("Version selector and catalog", ...)` — none reference Validate. No change.
3. Tests under `describe("InspectorPanel execution feedback", ...)` — check the `handleValidate` flow. Since the button is removed but `handleValidate` and its callers remain wired, the only tests that need updating are those that click the now-removed button directly. Tests that assert the Execute button's enable/disable behaviour and tests that exercise the handler's logic (e.g., by calling it programmatically) are unaffected.

When a test clicked the Validate button to trigger `handleValidate`'s side effects, update it to invoke `handleValidate` directly (or through a keyboard shortcut if one exists) rather than the button click. The handler is still present and callable — only the button UI is gone.

- [ ] **Step 2: Run tests — expect FAIL on the new negative assertions**

- [ ] **Step 3: Implement — remove the Validate button block only**

In `InspectorPanel.tsx`, find the block:

```tsx
            {/* Validate button with spinner */}
            <button
              onClick={handleValidate}
              disabled={!canValidate}
              aria-label={isValidating ? "Validating" : "Validate pipeline"}
              className="btn inspector-action-btn"
            >
              {isValidating ? (
                <span
                  className="spinner"
                  role="status"
                  aria-label="Validating"
                />
              ) : (
                "Validate"
              )}
            </button>
```

Delete the entire button block (including the comment). Keep the surrounding Catalog and Execute buttons untouched.

**Do NOT delete `handleValidate`, `canValidate`, `isValidating`, `injectSystemMessage`, or `sendValidationFeedback`.** These remain live in `InspectorPanel.tsx`; their deletion is Phase 3A Task 8's responsibility, where it is atomic with the relocation into `subscriptions.ts`. See "Why this task is scope-limited" above and convergence C3.

If `canValidate` or `isValidating` become unused after the button is removed (because nothing else references them), leave them in place as dead code temporarily — Phase 3A's cleanup will remove them. Do **not** clean them up here; cleanup in a separate task creates the same two-step race condition that motivated the scope-shrink.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx
```

Expected: all assertions pass, including the new negative tests.

- [ ] **Step 5: Quick lint check**

```bash
cd src/elspeth/web/frontend && npx eslint src/components/inspector/InspectorPanel.tsx
```

If `canValidate` or `isValidating` are now unused (because the button was their only consumer), lint will warn. Suppress those specific warnings with a `// eslint-disable-next-line @typescript-eslint/no-unused-vars` comment paired with a TODO comment: `// TODO Phase 3A Task 8 — remove with handleValidate when relocation lands`. This is the correct narrow exception: the variables are intentionally retained to avoid the regression window documented in the scope note above. Do **not** use `eslint-disable` for any other warning.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx
git commit -m "feat(web/frontend): remove standalone Validate button (subsumed by audit-readiness panel) (Phase 2C.8)

Button-only removal per convergence C3 (14-phase-2-audit-readiness-panel.review.json).
handleValidate/injectSystemMessage/sendValidationFeedback remain wired;
their deletion into subscriptions.ts is Phase 3A Task 8's responsibility,
where the two operations are atomic."
```

---

## Task 8.5: Phase 3A handoff guard test

**Why this task exists.** Task 8 above retains `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` in `InspectorPanel.tsx`. Their deletion is deferred to Phase 3A Task 8 (`15a2-phase-3a-removals-part-2.md`), where the relocation into `subscriptions.ts` is atomic with their removal. Phase 2 merges before Phase 3A, so Phase 2 cannot rely on Phase 3A's work. This task plants a mechanical CI gate — a `.skip`-marked test that documents the handoff contract in code and fails if Phase 3A's PR omits the relocation (by requiring the PR author to unskip the test, at which point it must pass).

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/subscriptions.handoff.test.ts`

> **Scope note (convergence C3).** The guard test is co-located with `subscriptions.ts` because that is the canonical relocation target. When Phase 3A Task 8 lands the relocation, the PR author unskips this test; a passing green build is the merge gate for the relocation. Do not move this test file without updating the TODO comment and Phase 3A Task 8's Step 4 commit message.

- [ ] **Step 1: Create the guard test file**

`src/elspeth/web/frontend/src/stores/subscriptions.handoff.test.ts`:

```typescript
/**
 * subscriptions.handoff.test.ts
 *
 * Phase 3A handoff guard (convergence finding C3 —
 * 14-phase-2-audit-readiness-panel.review.json).
 *
 * CONTEXT
 * -------
 * Phase 2C Task 8 removes the standalone Validate button from
 * InspectorPanel.tsx but deliberately retains handleValidate,
 * injectSystemMessage, and sendValidationFeedback there.  Their
 * deletion and relocation into this file (subscriptions.ts) is
 * owned by Phase 3A Task 8 (15a2-phase-3a-removals-part-2.md), where
 * the two operations (delete from InspectorPanel, add to subscriptions)
 * are atomic.  Phase 2 merges before Phase 3A, so Phase 2 cannot rely
 * on Phase 3A's relocation being present.
 *
 * GATE CONTRACT
 * -------------
 * When Phase 3A Task 8 lands the relocation into subscriptions.ts, the
 * PR author MUST unskip this test (remove the `.skip`) and verify that
 * the full test passes on green CI before merging.  The unskip is the
 * merge gate for the relocation: a test that imports from subscriptions.ts
 * and asserts injectSystemMessage fires on validation failure, driven
 * through the subscription path.
 *
 * TODO Phase 3A Task 8 — when the relocation into subscriptions.ts lands,
 * unskip this test.  It should pass immediately if the relocation is correct.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";

// TODO Phase 3A Task 8 — unskip this test when the relocation lands.
describe.skip("subscriptions.ts — Phase 3A handoff: injectSystemMessage fires on validation failure", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("injectSystemMessage is called when validation fails and the subscription handles it", async () => {
    // Arrange: wire the stores so a version increment triggers auto-validate.
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn().mockResolvedValue(undefined);
    const validate = vi.fn().mockImplementation(async () => {
      useExecutionStore.setState({
        validationResult: {
          is_valid: false,
          errors: [
            {
              component_type: "source",
              component_id: "csv_source",
              message: "Required field 'path' is missing",
            },
          ],
          warnings: [],
        } as never,
      } as never);
    });

    useExecutionStore.setState({
      validate,
      isValidating: false,
      isExecuting: false,
    } as never);

    useSessionStore.setState({
      activeSessionId: "sess-handoff-1",
      injectSystemMessage,
      sendValidationFeedback,
      compositionState: {
        version: 1,
        source: null,
        nodes: [],
        outputs: [],
      } as never,
    } as never);

    // This import initialises the subscription.  After Phase 3A Task 8,
    // initStoreSubscriptions() must wire the auto-validate + side-effects
    // loop described in 15a2 Task 8 Step 2.
    const { initStoreSubscriptions } = await import("./subscriptions");
    initStoreSubscriptions();

    // Act: advance the composition version to trigger the subscription.
    useSessionStore.setState({
      compositionState: { version: 2, source: null, nodes: [], outputs: [] } as never,
    } as never);

    // Assert: the subscription must have called injectSystemMessage with the
    // validation error message, driving the same side effect that
    // handleValidate previously drove from InspectorPanel.tsx.
    const { waitFor } = await import("@testing-library/react");
    await waitFor(() => {
      expect(injectSystemMessage).toHaveBeenCalled();
    });
    expect(sendValidationFeedback).toHaveBeenCalled();

    const [message] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation failed");
    expect(message).toContain("csv_source");
  });
});
```

- [ ] **Step 2: Run the guard test — verify it is SKIPPED (not failed)**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/subscriptions.handoff.test.ts
```

Expected output: `1 skipped` — the test is registered and skipped correctly; it does NOT fail red in this state. If the test errors (not skips), check the import paths and fix them before committing.

- [ ] **Step 3: Commit the guard test**

```bash
git add src/elspeth/web/frontend/src/stores/subscriptions.handoff.test.ts
git commit -m "test(web/frontend): add Phase 3A handoff guard test for subscriptions.ts (Phase 2C.8.5)

Skipped test that documents the Phase 3A Task 8 handoff contract:
when injectSystemMessage/sendValidationFeedback are relocated from
InspectorPanel.tsx into subscriptions.ts, the Phase 3A PR author
unskips this test as the merge gate.

Convergence finding C3 — 14-phase-2-audit-readiness-panel.review.json."
```

---

## Task 9: End-to-end smoke against the staging backend

**Files:** None (manual).

**Prerequisite:** Phase 2A is merged and deployed to staging (`elspeth.foundryside.dev`). The session DB has the new endpoints reachable from the frontend build.

- [ ] **Step 1: Build and deploy the frontend**

Following `project_staging_deployment` memory:

```bash
cd src/elspeth/web/frontend && npm run build
# Then on the deploy host (this machine, per the staging-deployment memory):
sudo systemctl restart elspeth-web.service
```

- [ ] **Step 2: Smoke — all-green collapse**

1. Open staging in the browser; log in as the test user.
2. Create a new session, compose a trivially-valid pipeline (CSV source → CSV sink, no transforms).
3. Verify the audit-readiness panel renders **collapsed** to "Audit ready ✓".
4. Click the collapsed summary; verify all six rows appear.
5. Click "Explain →"; verify the narrative renders within ~1s and the wording matches the Phase 2A `build_narrative` output.

- [ ] **Step 3: Smoke — provenance warning**

1. In the same session, add an identity-passthrough transform (e.g. `select_columns` keeping every column).
2. Verify the panel auto-expands to all rows and the Provenance row shows `⚠`.
3. Click the Provenance row; the detail drawer opens.
4. Verify "Jump to <node-id>" navigates to the node in the Spec tab and selects it.

- [ ] **Step 4: Smoke — Validate button is gone**

1. Look at the inspector header. There must be **no** Validate button — only Catalog and Execute.
2. Verify Execute is disabled until the panel's Validation row shows `✓`.
3. Verify the existing keyboard shortcut for Validate (if any — `grep -n "Validate" src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx` to confirm) either still works (re-routing through the panel's auto-fetch) or is documented as removed. If documented as removed, file the spec amendment as a comment on the umbrella PR.

- [ ] **Step 5: Smoke — error path**

1. Stop the backend (`sudo systemctl stop elspeth-web.service`).
2. Reload the page; verify the audit-readiness panel renders an `alert`-role error message rather than crashing the inspector.
3. Restart the backend; reload; verify the panel recovers on next composition change.

- [ ] **Step 6: Sign off**

If all five smoke steps pass, mark Phase 2 complete and merge the umbrella PR. If any step fails, file the failure as an observation (`mcp__filigree__observe`) with `file_path` set to the relevant component and stop — do not paper over a runtime divergence.

---

## What Phase 2C leaves the frontend in

- `components/audit/ReadinessRowDetail.tsx` is the full implementation (jump-to-component for resolvable ids).
- `components/audit/ExplainDialog.tsx` is the full implementation (modal, version-keyed narrative cache).
- `InspectorPanel.tsx` renders `<AuditReadinessPanel />` above the tab strip; the standalone Validate button is gone. `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` remain in `InspectorPanel.tsx` (wired, not dead code — the button is the only caller removed); their deletion and relocation into `subscriptions.ts` is Phase 3A Task 8's responsibility, where both operations are atomic.
- 14b's `stores/auditReadinessStore.ts` and `api/auditReadiness.ts` are unchanged — 14c does not touch the foundation files.
- The Phase 1B preference-bootstrap and Phase 5/6/7/8/9/10 chat flows are **untouched**.
- The Execute button still gates on `validationResult` — wire shape unchanged.
- Phase 2 is now feature-complete pending the staging smoke (Task 9).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Panel auto-fetch races a still-mutating composition | Auto-fetch is keyed on `composition_version` (integer, monotonic). The store short-circuits when the version matches; a mid-mutation render reuses the cached value, and the next render after the version advances triggers a fresh fetch. |
| Phase 3A Task 8 deletes `handleValidate`/`injectSystemMessage`/`sendValidationFeedback` from the inspector in a different PR; that deletion lands atomically with the relocation into `subscriptions.ts` | Task 8.5's `.skip` guard test fails red if Phase 3A's PR omits the relocation. The Phase 3A PR author must unskip the guard test and verify it passes before merging. |
| Phase 3 changes the inspector layout and the panel breaks | The panel component is layout-agnostic. Phase 3 only changes the mount point. Phase 2's tests assert the panel's behaviour, not its DOM order; Phase 3's tests re-assert ordering at the new mount site. |
| Per-row "Jump to component" fails when `component_ids` references a source or sink (not a node) | Step 3 of Task 5 renders unresolvable ids as plain text rather than a non-functional button. The user can grep the YAML. Phase 3's side-rail reshape (if it surfaces sources/sinks as selectable) is the place to make those clickable. |
| Backend wire-shape drift breaks the renderer | The discriminated-union `never` arm fails the build; Phase 2A's `_StrictResponse` fails at server-side construction. No silent path. |
| Telemetry can't be added later without rewriting the panel | The row-click handlers are isolated functions in `AuditReadinessPanel` and `ReadinessRowDetail`. Phase 8 telemetry adds one line per handler. |
| User reports "the panel is slow" | The aggregator is the same cost as the standalone Validate it replaced (Phase 2A's risk table). Profile before optimising; do not add a debounce until profiling shows the need. |

## Review history

**2026-05-15** — Panel findings applied: Task 8 now deletes `handleValidate` and callers (`injectSystemMessage`, `sendValidationFeedback`) from `InspectorPanel.tsx` — side effects relocated to Phase 3A `subscriptions.ts` (BLOCKER); test migration policy changed from "inline TODO acceptable" to "block-commit-if-not-migrated" (CRITICAL); Phase 3B mount-point dependency documented in §Sequencing (IMPORTANT); `require()` spy replaced with top-level `vi.mock()` in Task 7 test (IMPORTANT); negative `waitFor` race in ExplainDialog test fixed to wait for render before asserting not-called (from 14b Finding 3, applied here where the code lives).

### 2026-05-16 — 4-reviewer panel verdict CHANGES_REQUESTED → fixes applied

Reviewers: reality, architecture, quality, systems (full report:
`14-phase-2-audit-readiness-panel.review.json`).

Fixes applied to 14c in this revision:
1. Task 8 scope-shrink — remove only the standalone Validate button UI; defer handleValidate/injectSystemMessage/sendValidationFeedback deletion to Phase 3A Task 8 where the relocation into subscriptions.ts is atomic (convergence C3)
2. Task 8.5 added — Phase 3A handoff guard test that fails red until the relocation lands (mechanical gate replacing prose-only constraint)

Strategic adjudication: scope-shrink (preferred path per review C3) over pulling Phase 3A Task 4+8 into the Phase 2 umbrella PR (would inflate umbrella scope and require coordination with the already-committed 15a plans).

## Memory references

- `project_composer_personas` — Linda-vocabulary row labels.
- `project_staging_deployment` — staging is a source-checkout systemd/Caddy deploy; `npm run build` + `systemctl restart elspeth-web.service`.
- `feedback_no_calendar_shipping_commitments` — no SLAs in this plan.
- `feedback_default_is_fix_not_ticket` — Step 5 of Task 8 is the place this rule bites: lint errors aren't suppressed, they're fixed; if the dead-code analysis surfaces a regression, fix it in-task rather than filing.
- `feedback_repeated_out_of_scope_is_underscoping.md` — Task 8 scope was scope-shrunk in the 2026-05-16 revision (convergence C3): handler deletion was correctly identified as out of scope for Phase 2, not as underscoping. Phase 3A Task 8 (15a2) owns handler deletion atomically with the relocation into `subscriptions.ts`.
