// src/components/chat/guided/ProposeChainTurn.test.tsx
//
// Regression suite for ProposeChainTurn (Task 7.6).
//
// Pinned contracts:
//   1. Card-list rendering of payload.steps — each step becomes a card showing
//      plugin name, rationale, and a key-value options list.
//   2. Step-3 wire-shape contract — Accept all, per-step Edit, Reject, and
//      Ask advisor each emit the explicit GuidedRespondRequest shape that the
//      backend consumes.
//   3. payload.blockers show/hide based on emptiness — blockers list renders when
//      non-empty; absent from DOM when blockers is [].
//   4. DOM-ID distinctness pin — two simultaneous instances produce element IDs
//      that are NOT the same node (per-instance useId() scoping).
//
// Wire-shape verification:
//   chosen: ["accept"]                  -> pipeline committed
//   edit_step_index: n                   -> edit the nth proposed transform
//   control_signal: "reject"             -> regenerate proposal
//   control_signal: "request_advisor"    -> advisor-guided regeneration

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ProposeChainTurn } from "./ProposeChainTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type { ProposeChainPayload } from "@/types/guided";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const STEP_A: ProposeChainPayload["steps"][number] = {
  plugin: "csv_transform",
  options: { delimiter: ",", skip_header: true },
  rationale: "Parses CSV rows into typed records for downstream enrichment.",
};

const STEP_B: ProposeChainPayload["steps"][number] = {
  plugin: "geo_enrich",
  options: { api_key: "***", timeout: 5 },
  rationale: "Adds lat/lon to each record using the geocoding API.",
};

const TWO_STEP_PAYLOAD: ProposeChainPayload = {
  steps: [STEP_A, STEP_B],
  why: "CSV source needs type coercion then geo enrichment before sink.",
  blockers: [],
};

const WITH_BLOCKERS_PAYLOAD: ProposeChainPayload = {
  steps: [STEP_A],
  why: "Single enrichment step proposed.",
  blockers: ["geocoding API key not yet configured", "output schema ambiguous"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderWidget(
  payload: ProposeChainPayload,
  onSubmit = vi.fn(),
) {
  return render(<ProposeChainTurn payload={payload} onSubmit={onSubmit} />);
}

// ---------------------------------------------------------------------------
// 1. Card-list rendering
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — card-list rendering", () => {
  it("renders payload.why as a context paragraph", () => {
    renderWidget(TWO_STEP_PAYLOAD);
    expect(
      screen.getByText(
        "CSV source needs type coercion then geo enrichment before sink.",
      ),
    ).toBeInTheDocument();
  });

  it("renders one card per step", () => {
    renderWidget(TWO_STEP_PAYLOAD);
    // Each step card has a heading with the plugin name.
    expect(screen.getByText("csv_transform")).toBeInTheDocument();
    expect(screen.getByText("geo_enrich")).toBeInTheDocument();
  });

  it("each step's plugin name is a level-3 heading for screen-reader navigation", () => {
    // Pins the <h3> semantics so a future "simplify back to <span>" regresses
    // the test instead of silently breaking landmark navigation. Same convention
    // as Task 7.3 M10 (edit-mode <h3>).
    renderWidget(TWO_STEP_PAYLOAD);
    const headings = screen.getAllByRole("heading", { level: 3 });
    expect(headings).toHaveLength(TWO_STEP_PAYLOAD.steps.length);
    expect(headings[0]).toHaveTextContent(TWO_STEP_PAYLOAD.steps[0].plugin);
    expect(headings[1]).toHaveTextContent(TWO_STEP_PAYLOAD.steps[1].plugin);
  });

  it("renders the rationale for each step", () => {
    renderWidget(TWO_STEP_PAYLOAD);
    expect(
      screen.getByText(
        "Parses CSV rows into typed records for downstream enrichment.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Adds lat/lon to each record using the geocoding API.",
      ),
    ).toBeInTheDocument();
  });

  it("renders option keys and values inside each step card", () => {
    renderWidget(TWO_STEP_PAYLOAD);
    // STEP_A options: delimiter, skip_header
    expect(screen.getByText("delimiter")).toBeInTheDocument();
    // STEP_B options: api_key, timeout
    expect(screen.getByText("api_key")).toBeInTheDocument();
  });

  it("renders a single step correctly (edge case: 1-step chain)", () => {
    const single: ProposeChainPayload = {
      steps: [STEP_A],
      why: "Only one transform needed.",
      blockers: [],
    };
    renderWidget(single);
    expect(screen.getByText("csv_transform")).toBeInTheDocument();
    expect(screen.queryByText("geo_enrich")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// 2. Accept-all wire-shape contract
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — accept-all submit", () => {
  it("clicking Accept all steps fires onSubmit with chosen: ['accept'] and other fields null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(TWO_STEP_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /accept all steps/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        ...nullResponse(),          // spread FIRST; overridden by explicit fields below
        chosen: ["accept"],
        custom_inputs: null,
      }),
    );
    // Verify the full body — all 6 GuidedRespondRequest fields explicit.
    const body = onSubmit.mock.calls[0][0];
    expect(body).toEqual({
      ...nullResponse(),
      chosen: ["accept"],
      custom_inputs: null,
    });
  });

  it("clicking Accept works when there is only one step", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const single: ProposeChainPayload = {
      steps: [STEP_A],
      why: "One step.",
      blockers: [],
    };
    renderWidget(single, onSubmit);

    await user.click(screen.getByRole("button", { name: /accept all steps/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: ["accept"],
      custom_inputs: null,
    });
  });

  it("clicking Accept works when blockers are present", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(WITH_BLOCKERS_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /accept all steps/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: ["accept"],
      custom_inputs: null,
    });
  });

  it("no widget-side state leaks between renders — second click still emits clean body", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(TWO_STEP_PAYLOAD, onSubmit);

    // Click once, then click again — each call must be independent.
    await user.click(screen.getByRole("button", { name: /accept all steps/i }));
    await user.click(screen.getByRole("button", { name: /accept all steps/i }));

    expect(onSubmit).toHaveBeenCalledTimes(2);
    expect(onSubmit.mock.calls[1][0]).toEqual({
      ...nullResponse(),
      chosen: ["accept"],
      custom_inputs: null,
    });
  });
});

describe("ProposeChainTurn — remediation submit paths", () => {
  it("renders an Edit button for each proposed step", () => {
    renderWidget(TWO_STEP_PAYLOAD);

    expect(screen.getAllByRole("button", { name: /edit step/i })).toHaveLength(
      TWO_STEP_PAYLOAD.steps.length,
    );
  });

  it("clicking a per-step Edit button submits edit_step_index with other fields null", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(TWO_STEP_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /edit step 2/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      edit_step_index: 1,
    });
  });

  it("clicking Reject opens a confirm dialog before submitting control_signal='reject'", async () => {
    // S3.5 (button-audit): Reject is destructive — discards a multi-step plan —
    // so it now opens a ConfirmDialog instead of submitting immediately. Clicking
    // the dialog's primary action ("Reject plan") submits the reject signal.
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(TWO_STEP_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /^reject$/i }));

    // Submit must not be called by the dialog-opening click alone.
    expect(onSubmit).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /reject plan/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      control_signal: "reject",
    });
  });

  it("clicking Ask advisor submits control_signal='request_advisor'", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    renderWidget(TWO_STEP_PAYLOAD, onSubmit);

    await user.click(screen.getByRole("button", { name: /ask advisor/i }));

    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: null,
      custom_inputs: null,
      control_signal: "request_advisor",
    });
  });
});

// ---------------------------------------------------------------------------
// 3. payload.blockers show/hide
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — blockers section", () => {
  it("renders a blockers section when blockers is non-empty", () => {
    renderWidget(WITH_BLOCKERS_PAYLOAD);
    expect(
      screen.getByText("geocoding API key not yet configured"),
    ).toBeInTheDocument();
    expect(screen.getByText("output schema ambiguous")).toBeInTheDocument();
  });

  it("does NOT render a blockers section when blockers is empty (negative-space pin)", () => {
    renderWidget(TWO_STEP_PAYLOAD); // blockers: []
    // Neither the section heading nor any blocker text should appear.
    expect(screen.queryByText(/blocker/i)).not.toBeInTheDocument();
  });

  it("renders all individual blocker strings as list items", () => {
    renderWidget(WITH_BLOCKERS_PAYLOAD);
    const items = screen.getAllByRole("listitem");
    const blockerItems = items.filter(
      (el) =>
        el.textContent?.includes("geocoding API key") ||
        el.textContent?.includes("output schema ambiguous"),
    );
    expect(blockerItems).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// 4. DOM-ID distinctness pin (Task 7.4 I4 convention)
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — DOM-ID distinctness pin", () => {
  it("two simultaneous instances produce distinct element nodes for the accept button", () => {
    // Render two instances into the same document.
    const { container: c1 } = render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={vi.fn()} />,
    );
    const { container: c2 } = render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={vi.fn()} />,
    );

    // Each instance's per-card IDs must be scoped by useId().
    // Grab the first scoped element from each container.
    const card0_id = c1.querySelector("[id]")?.id;
    const card1_id = c2.querySelector("[id]")?.id;

    // Both must exist as real DOM nodes.
    expect(card0_id).toBeDefined();
    expect(card1_id).toBeDefined();

    // The IDs must be different strings (useId() scoping per instance).
    expect(card0_id).not.toBe(card1_id);

    // Identity check: the same string ID must not resolve to the same node.
    // (If IDs were identical, document.getElementById would return the first one
    // for both — the .not.toBe check would still pass; we assert both resolve.)
    const node0 = document.getElementById(card0_id!);
    const node1 = document.getElementById(card1_id!);
    expect(node0).not.toBeNull();
    expect(node1).not.toBeNull();
    // Different IDs -> definitely different nodes.
    expect(node0).not.toBe(node1);
  });
});

// ---------------------------------------------------------------------------
// 5. Focus / auto-focus (negative-space pin)
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — focus management", () => {
  it("does NOT auto-focus any element on initial render", () => {
    // ProposeChainTurn has no view transitions or collapsible regions that
    // would trigger focus management. The component renders fully on mount
    // with no focus side-effects (per convention from Task 7.2 template).
    renderWidget(TWO_STEP_PAYLOAD);
    // document.activeElement should be body (no programmatic focus).
    expect(document.activeElement).toBe(document.body);
  });
});

// ---------------------------------------------------------------------------
// 6. Tutorial passive mode (isTutorial)
// ---------------------------------------------------------------------------

describe("ProposeChainTurn — tutorial passive mode", () => {
  it("hides Edit / Reject / Ask advisor and keeps only Accept all steps", () => {
    render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={vi.fn()} isTutorial />,
    );
    expect(screen.queryByRole("button", { name: /edit step/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^reject$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /ask advisor/i })).toBeNull();
    expect(
      screen.getByRole("button", { name: /accept all steps/i }),
    ).toBeInTheDocument();
  });

  it("still emits the accept wire-shape from the passive Accept button", async () => {
    const onSubmit = vi.fn();
    render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={onSubmit} isTutorial />,
    );
    await userEvent.click(
      screen.getByRole("button", { name: /accept all steps/i }),
    );
    expect(onSubmit).toHaveBeenCalledWith({
      chosen: ["accept"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  });

  it("collapses each step's raw config behind a closed <details> (no monospace wall)", () => {
    const { container } = render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={vi.fn()} isTutorial />,
    );
    // One <details> per step that has options (both steps here).
    const details = container.querySelectorAll(
      "details.guided-propose-options-details",
    );
    expect(details).toHaveLength(2);
    // Collapsed by default — the config wall is not expanded on mount.
    details.forEach((d) => expect(d.hasAttribute("open")).toBe(false));
    // Transparency preserved: a friendly summary label is shown, not the dump.
    expect(screen.getAllByText(/configuration \(\d+\)/i).length).toBe(2);
  });

  it("does NOT collapse config in normal (non-tutorial) mode", () => {
    const { container } = render(
      <ProposeChainTurn payload={TWO_STEP_PAYLOAD} onSubmit={vi.fn()} />,
    );
    expect(
      container.querySelector("details.guided-propose-options-details"),
    ).toBeNull();
    // The inline option list is rendered directly.
    expect(screen.getByText("delimiter")).toBeInTheDocument();
  });
});
