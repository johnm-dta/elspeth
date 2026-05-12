// ============================================================================
// GuidedHistory -- collapsible read-only step-history list regression coverage.
//
// Pins SEVEN contracts:
//   1. Toggle button renders "Show steps (N)" / "Hide steps (N)" per state.
//   2. Initial state is collapsed: history items region has `hidden` attribute.
//   3. aria-controls cross-state resolution: toggle's aria-controls points to an
//      existing DOM element in BOTH collapsed AND expanded states.  This is the
//      regression pin for the Task 7.5 I2 fix — the expanded-only conditional
//      render pattern ({expanded && <div id={...}>}) is BROKEN because the id
//      becomes a dangling reference while collapsed.  The `hidden` attribute
//      pattern (div always present, hidden toggles) is the required approach.
//   4. aria-expanded toggles correctly on click.
//   5. Per-entry rendering: one <li> per TurnRecord when expanded; step label +
//      turn type + emitter visible.
//   6. Empty-history negative space: returns nothing when history is empty.
//   7. Distinctness pin (useId, Task 7.4 I4 inheritance): two simultaneous
//      GuidedHistory instances have distinct aria-controls IDs, proving useId()
//      scoping prevents cross-instance collision.
//   8. Focus management: focus stays on the toggle button after expand/collapse
//      (history items are non-interactive read-only <li>s; moving focus into
//      the list would require non-standard tabindex=-1 ref dance).
//   9. Initial-mount no-auto-focus: rendering the widget does not steal focus.
//  10. Scope reduction documented: wire carries only step/turn_type/emitter,
//      not rich summary data.  See Tracker: elspeth-obs-cc8fa78524.
//
// Source of truth:
//   - types/guided.ts:44-51 (TurnRecord wire shape — hashes only, no summary)
//   - schemas.py:213-220 (TurnRecordResponse — confirms scope reduction)
//   - SchemaFormTurn.test.tsx:659-673 (aria-controls cross-state regression pin template)
// ============================================================================

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GuidedHistory } from "./GuidedHistory";
import type { TurnRecord } from "@/types/guided";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const TURN_1: TurnRecord = {
  step: "step_1_source",
  turn_type: "single_select",
  payload_hash: "aabbcc001122",
  response_hash: "ddeeff334455",
  emitter: "server",
};

const TURN_2: TurnRecord = {
  step: "step_2_sink",
  turn_type: "schema_form",
  payload_hash: "112233aabbcc",
  response_hash: null,
  emitter: "llm",
};

const TWO_TURNS: TurnRecord[] = [TURN_1, TURN_2];

// ── 1. Toggle button label ────────────────────────────────────────────────────

describe("toggle button label", () => {
  it("shows 'Show steps (N)' when collapsed (initial state)", () => {
    render(<GuidedHistory history={TWO_TURNS} />);
    expect(
      screen.getByRole("button", { name: /show steps \(2\)/i }),
    ).toBeInTheDocument();
  });

  it("shows 'Hide steps (N)' when expanded", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps \(2\)/i }));
    expect(
      screen.getByRole("button", { name: /hide steps \(2\)/i }),
    ).toBeInTheDocument();
  });

  it("shows correct count for single entry", () => {
    render(<GuidedHistory history={[TURN_1]} />);
    expect(
      screen.getByRole("button", { name: /show steps \(1\)/i }),
    ).toBeInTheDocument();
  });
});

// ── 2. Initial state collapsed ────────────────────────────────────────────────

describe("initial collapsed state", () => {
  it("history region has the hidden attribute on initial render", () => {
    render(<GuidedHistory history={TWO_TURNS} />);
    const toggle = screen.getByRole("button", { name: /show steps/i });
    const controlsId = toggle.getAttribute("aria-controls");
    expect(controlsId).toBeTruthy();
    const region = document.getElementById(controlsId!);
    expect(region).not.toBeNull();
    // The `hidden` attribute collapses the region and removes it from the AT
    // tree, but the element stays in the DOM so aria-controls resolves.
    expect(region!.hasAttribute("hidden")).toBe(true);
  });

  it("list items are not visible (region hidden) on initial render", () => {
    render(<GuidedHistory history={TWO_TURNS} />);
    // Items inside a `hidden` container are excluded from the accessible role
    // tree; getByRole would not find them.
    expect(screen.queryByRole("list")).toBeNull();
  });

  it("respects initiallyExpanded=true prop", () => {
    render(<GuidedHistory history={TWO_TURNS} initiallyExpanded={true} />);
    const toggle = screen.getByRole("button", { name: /hide steps/i });
    const controlsId = toggle.getAttribute("aria-controls");
    const region = document.getElementById(controlsId!);
    expect(region!.hasAttribute("hidden")).toBe(false);
  });
});

// ── 3. aria-controls cross-state resolution (Task 7.5 I2 contract) ───────────

describe("aria-controls / aria-expanded contract", () => {
  it(
    "aria-controls id resolves to a DOM element even when collapsed (initial render)",
    () => {
      // Regression pin: mirrors SchemaFormTurn.test.tsx:660-673.
      // If GuidedHistory were to conditionally render the region only when
      // expanded ({expanded && <div id={controlsId}>...}), the id would be a
      // dangling reference while collapsed. The `hidden` attribute pattern
      // (always rendered, hidden toggles) is the required approach.
      render(<GuidedHistory history={TWO_TURNS} />);
      const toggle = screen.getByRole("button", { name: /show steps/i });
      const controlsId = toggle.getAttribute("aria-controls");
      expect(controlsId).toBeTruthy();
      // MUST resolve — not null — in the collapsed state.
      expect(document.getElementById(controlsId!)).not.toBeNull();
    },
  );

  it("aria-controls id resolves after expand", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    const toggle = screen.getByRole("button", { name: /show steps/i });
    const controlsId = toggle.getAttribute("aria-controls");
    await user.click(toggle);
    // Must still resolve in expanded state (same element, hidden removed).
    expect(document.getElementById(controlsId!)).not.toBeNull();
  });

  it("aria-controls id resolves after collapse (re-hidden)", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    const toggle = screen.getByRole("button", { name: /hide steps/i });
    const controlsId = toggle.getAttribute("aria-controls");
    await user.click(toggle);
    // Must still resolve after re-collapsing.
    const showBtn = screen.getByRole("button", { name: /show steps/i });
    expect(document.getElementById(showBtn.getAttribute("aria-controls")!)).not.toBeNull();
    expect(controlsId).toBeTruthy();
  });

  it("aria-expanded starts false and flips to true on expand", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    const toggle = screen.getByRole("button", { name: /show steps/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);
    expect(
      screen.getByRole("button", { name: /hide steps/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("history region announces itself as a labelled region when expanded", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    // role=region + aria-label gives screen readers a discoverable landmark.
    expect(
      screen.getByRole("region", { name: /wizard step history/i }),
    ).toBeInTheDocument();
  });
});

// ── 4. Per-entry rendering when expanded ─────────────────────────────────────

describe("per-entry rendering", () => {
  it("renders one list item per TurnRecord when expanded", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
  });

  it("shows step label for step_1_source entry", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={[TURN_1]} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    // Should render a human-readable step label, not the raw enum value.
    expect(screen.getByText(/source/i)).toBeInTheDocument();
  });

  it("shows turn_type for each entry", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={[TURN_1]} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    expect(screen.getByText(/single_select/i)).toBeInTheDocument();
  });

  it("shows emitter for each entry", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={[TURN_1]} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    expect(screen.getByText(/server/i)).toBeInTheDocument();
  });

  it("renders entries for both turns with distinct step labels", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    // Both step labels should appear (Source and Sink).
    expect(screen.getByText(/source/i)).toBeInTheDocument();
    expect(screen.getByText(/sink/i)).toBeInTheDocument();
  });

  it("shows ordinal step numbers (1-based index)", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    // Step 1 and Step 2 ordinals should appear.
    expect(screen.getByText(/step 1/i)).toBeInTheDocument();
    expect(screen.getByText(/step 2/i)).toBeInTheDocument();
  });
});

// ── 5. Empty-history negative space ──────────────────────────────────────────

describe("empty history", () => {
  it("renders nothing when history is empty", () => {
    const { container } = render(<GuidedHistory history={[]} />);
    expect(container.firstChild).toBeNull();
  });
});

// ── 6. Distinctness pin (useId scoping — Task 7.4 I4 inheritance) ─────────────

describe("useId distinctness", () => {
  it(
    "two simultaneous GuidedHistory instances have distinct aria-controls IDs",
    () => {
      // Regression pin: mirrors SchemaFormTurn.test.tsx:726-755.
      // Each GuidedHistory instance must get its own useId() prefix so the
      // toggle↔region aria-controls pair is unique per instance.
      render(
        <>
          <GuidedHistory history={TWO_TURNS} />
          <GuidedHistory history={TWO_TURNS} />
        </>,
      );
      const toggles = screen.getAllByRole("button", {
        name: /show steps \(2\)/i,
      });
      expect(toggles).toHaveLength(2);

      const id1 = toggles[0].getAttribute("aria-controls");
      const id2 = toggles[1].getAttribute("aria-controls");
      expect(id1).toBeTruthy();
      expect(id2).toBeTruthy();
      // IDs must be distinct strings (different useId() prefixes).
      expect(id1).not.toBe(id2);
      // The DOM elements they reference must be distinct nodes.
      const region1 = document.getElementById(id1!);
      const region2 = document.getElementById(id2!);
      expect(region1).not.toBeNull();
      expect(region2).not.toBeNull();
      expect(region1).not.toBe(region2);
    },
  );
});

// ── 7. Focus management ───────────────────────────────────────────────────────

describe("focus management", () => {
  it("focus stays on toggle button after expand", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    const showBtn = screen.getByRole("button", { name: /show steps/i });
    await user.click(showBtn);
    // After expand, the button (now "Hide steps") retains focus.
    // History items are non-interactive read-only <li>s; focusing them would
    // require non-standard tabindex=-1 ref dance and isn't idiomatic for
    // a read-only list.
    const hideBtn = screen.getByRole("button", { name: /hide steps/i });
    expect(document.activeElement).toBe(hideBtn);
  });

  it("focus stays on toggle button after collapse", async () => {
    const user = userEvent.setup();
    render(<GuidedHistory history={TWO_TURNS} />);
    await user.click(screen.getByRole("button", { name: /show steps/i }));
    await user.click(screen.getByRole("button", { name: /hide steps/i }));
    // After collapse, the button (now "Show steps") retains focus.
    const showBtn = screen.getByRole("button", { name: /show steps/i });
    expect(document.activeElement).toBe(showBtn);
  });
});

// ── 8. Initial-render no-auto-focus ──────────────────────────────────────────

describe("initial-render no-auto-focus", () => {
  it("does not steal focus from the document on mount", () => {
    // Create a button to hold focus before rendering GuidedHistory.
    const outside = document.createElement("button");
    outside.textContent = "Outside";
    document.body.appendChild(outside);
    outside.focus();

    render(<GuidedHistory history={TWO_TURNS} />);

    // GuidedHistory must not have moved focus away from the outside element.
    expect(document.activeElement).toBe(outside);

    document.body.removeChild(outside);
  });
});
