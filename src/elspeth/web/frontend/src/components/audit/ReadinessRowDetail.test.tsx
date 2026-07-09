import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "../../stores/sessionStore";
import { makeComposition } from "@/test/composerFixtures";
import type { ReadinessRow } from "../../types/api";

// CANONICAL FIXTURE — `makeComposition` lives at
// `src/elspeth/web/frontend/src/test/composerFixtures.ts` (alias
// `@/test/composerFixtures`). It already returns the correct `NodeSpec` shape
// (7 fields: id, node_type, plugin, input, on_success, on_error, options) and
// `SourceSpec` shape (plugin, options) — do NOT inline a literal here with
// `as never` casts; that's how drift gets introduced.

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

describe("ReadinessRowDetail", () => {
  beforeEach(() => {
    // makeComposition(1) returns a composition with one node id="select_columns",
    // which the ROW_WITH_NODE fixture above expects to be jumpable. Confirm by
    // reading composerFixtures.ts before changing the node-id assumption.
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1),
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
    const btn = screen.getByRole("button", { name: /Jump to the "pick the columns you need" step/ });
    await user.click(btn);
    expect(selectNode).toHaveBeenCalledWith("select_columns");
  });

  it("opens the Graph modal when jumping to a component", async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    try {
      render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
      await user.click(screen.getByRole("button", { name: /Jump to the "pick the columns you need" step/ }));
      expect(handler).toHaveBeenCalledTimes(1);
    } finally {
      window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    }
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

  // W6 — added 2026-05-17

  it("renders both resolvable and unresolvable ids in the same row (mixed)", () => {
    // One id resolves to a node (select_columns is in makeComposition(1));
    // one id does not resolve (api_key is unknown). Both branches of the
    // resolvable/unresolvable split are exercised within a single render.
    const ROW_MIXED: ReadinessRow = {
      id: "secrets",
      label: "Secrets",
      status: "error",
      summary: "Mixed component refs",
      detail: "One resolvable, one not.",
      component_ids: ["select_columns", "api_key"],
    };
    render(<ReadinessRowDetail row={ROW_MIXED} onClose={() => {}} />);
    // select_columns resolves → Jump button.
    expect(screen.getByRole("button", { name: /Jump to the "pick the columns you need" step/ })).toBeInTheDocument();
    // api_key does not resolve → plain text, no button.
    expect(screen.getByText("api_key")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Jump to api_key/ })).not.toBeInTheDocument();
  });

  it("fires onClose when Escape is pressed (post-P0.4: relies on mount-time focus)", async () => {
    // Drawer keyboard-dismiss contract: pressing Escape while the
    // drawer has focus should fire onClose. The earlier version of
    // this test manually `.focus()`'d the Close button to ensure
    // Escape dispatched to the dialog's onKeyDown handler — that
    // pre-focus masked P0.4(b) (no mount-time focus). The
    // implementation now auto-focuses the Close button on mount, so
    // this test asserts the contract WITHOUT pre-focusing: Escape
    // must work straight out of render.
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={onClose} />);
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  // P0.4(b) — mount-time focus moves to Close.
  it("focuses the Close button on mount so Escape and keyboard navigation work without a click first", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /Close/i }),
    );
  });

  // P0.4(a) — tabIndex={-1} on the dialog root.
  it("the dialog root carries tabIndex=-1 so keyboard focus can land on it programmatically", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    expect(screen.getByRole("dialog")).toHaveAttribute("tabindex", "-1");
  });

  // P0.4(c) — detail prose is announced as paragraph, not preformatted/code.
  it("renders the detail body as a <p> with whiteSpace: pre-line, not <pre>", () => {
    const { container } = render(
      <ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />,
    );
    // No <pre> element anywhere in the drawer.
    expect(container.querySelector("pre")).toBeNull();
    const body = container.querySelector("p.readiness-row-detail-body");
    expect(body).not.toBeNull();
    expect(body).toHaveStyle({ whiteSpace: "pre-line" });
    // Linebreak preservation contract is verified by the first test
    // ("renders the row label and detail with preserved linebreaks"),
    // which still passes against pre-line styling.
  });

  // P0.2 — Jump must NOT close the drawer (user needs to see the
  // highlighted component in context).
  it("Jump leaves the drawer open so the user can confirm the highlighted target", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={onClose} />);
    await user.click(
      screen.getByRole("button", { name: /Jump to the "pick the columns you need" step/ }),
    );
    expect(onClose).not.toHaveBeenCalled();
    // Drawer still present.
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  // ── elspeth-901a404926: the validation row re-humanises its findings ──────
  const VALIDATION_ROW: ReadinessRow = {
    id: "validation",
    label: "Validation",
    status: "error",
    summary: "1 problem to fix — see details",
    detail: "Add an output step so your pipeline has somewhere to send its results.",
    component_ids: [],
  };

  it("renders a reframed settings finding as a plain line — no pydantic leak, no Technical-details disclosure", () => {
    render(
      <ReadinessRowDetail
        row={VALIDATION_ROW}
        validationErrors={[
          {
            component_id: null,
            component_type: null,
            message: "Add an output step so your pipeline has somewhere to send its results.",
            suggestion: null,
            error_code: "missing_sink",
          },
        ]}
        onClose={() => {}}
      />,
    );
    expect(screen.getByText(/Add an output step/)).toBeInTheDocument();
    // The clean finding carries no raw dump, so no disclosure appears.
    expect(screen.queryByText("Technical details")).toBeNull();
    expect(screen.queryByText(/Field required/)).toBeNull();
    expect(screen.queryByText(/errors\.pydantic\.dev/)).toBeNull();
  });

  it("humanises an engine dump in the validation row and tucks the raw text behind Technical details", () => {
    const rawDump =
      "Schema contract violation: 'select_columns' -> 'out'. " +
      "Consumer (csv) requires fields: [score]. Producer (llm) guarantees: [body].";
    const { container } = render(
      <ReadinessRowDetail
        row={{ ...VALIDATION_ROW, detail: rawDump }}
        validationErrors={[
          {
            component_id: "select_columns",
            component_type: "transform",
            message: rawDump,
            suggestion: null,
            error_code: null,
          },
        ]}
        onClose={() => {}}
      />,
    );
    // Humanised headline shows, mapped through the gloss.
    expect(screen.getByText(/connected correctly/i)).toBeInTheDocument();
    // The raw engine dump is NOT a top-level body line — it lives behind the
    // Technical-details disclosure.
    expect(screen.getByText("Technical details")).toBeInTheDocument();
    const pre = container.querySelector("pre.readiness-row-detail-raw-text");
    expect(pre?.textContent).toContain("Schema contract violation");
  });

  // Test 5.C SKIPPED — documented skip.
  // `ReadinessRow.component_ids` is typed `readonly string[]` (types/api.ts),
  // which is non-nullable. The implementation's `row.component_ids.length > 0`
  // cannot receive `null` at the TypeScript layer. A null-guard test would
  // only be valid if the backend could send `null` in place of the array,
  // but the type does not allow it and there is no `as never` bypass in the
  // production code path. Adding a `null as never` test would be defensive
  // programming against an impossible state — forbidden by project policy.
});
